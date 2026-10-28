# src/trainers/purejax_trainer.py

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import time
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict, Callable
from flax.training.train_state import TrainState
import distrax
import gymnax
import functools
from gymnax.environments import spaces
from src.envs.wrappers import BatchEnvWrapper, OptimisticResetVecEnvWrapper


def flush_carry_state(hstate, last_done):
    """
    Symmetrically flush the heterogeneous PyTree carry state using the last_done mask
    without triggering JAX shape-mismatch compilation errors.
    """

    def flush_leaf(x):
        dims_to_add = x.ndim - last_done.ndim
        broadcast_done = jnp.expand_dims(last_done, axis=tuple(range(1, 1 + dims_to_add)))
        return x * (1.0 - broadcast_done)

    return jax.tree_util.tree_map(flush_leaf, hstate)


class ActorCriticRNN(nn.Module):
    rnn_module: Callable
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.config["HIDDEN"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = self.rnn_module()(hidden, rnn_in)

        actor_mean = nn.Dense(self.config["HIDDEN"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        raw_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        action_logits = 5.0 * jnp.tanh(raw_logits / 5.0)
        action_logits = jnp.nan_to_num(action_logits, nan=0.0, posinf=5.0, neginf=-5.0)
        pi = distrax.Categorical(logits=action_logits)

        critic = nn.Dense(self.config["HIDDEN"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        critic = jnp.nan_to_num(jnp.squeeze(critic, axis=-1), nan=0.0, posinf=10.0, neginf=-10.0)

        return hidden, pi, critic


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_update(rng, config, env, env_params, rnn_module: Callable = None, rnn_carry_init: Callable = None):
    config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
            config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    if config['OPTIMISTIC_RESETS']:
        default_ratio = min(64, config["NUM_ENVS"])
        reset_ratio = config.get("RESET_RATIO", default_ratio)
        if config["NUM_ENVS"] % reset_ratio != 0:
            reset_ratio = min(16, config["NUM_ENVS"])
        env = OptimisticResetVecEnvWrapper(
            env, config["NUM_ENVS"], reset_ratio=reset_ratio)
    else:
        env = BatchEnvWrapper(env, config["NUM_ENVS"])

    def linear_schedule(count):
        frac = (
                1.0
                - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
                / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    # INIT NETWORK
    if config["RNN"]["RNN_TYPE"] == "sdlg_hybrid":
        from src.models.purejaxrl.wrappers import ActorCriticSdLGHybrid
        network = ActorCriticSdLGHybrid(
            action_dim=env.action_space(env_params).n, config=config)
    elif config["RNN"]["RNN_TYPE"] == "ablation":
        network = ActorCriticRNN(
            rnn_module, env.action_space(env_params).n, config=config)
    else:
        network = ActorCriticRNN(
            rnn_module, env.action_space(env_params).n, config=config)


    rng, _rng = jax.random.split(rng)
    init_x = (
        jnp.zeros(
            (1, config["NUM_ENVS"], *env.observation_space(env_params).shape)
        ),
        jnp.zeros((1, config["NUM_ENVS"])),
    )
    init_hstate = rnn_carry_init(config["NUM_ENVS"])
    network_params = network.init(_rng, init_hstate, init_x)

    if config["ANNEAL_LR"]:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    obsv, env_state = env.reset(_rng, env_params)
    init_hstate = rnn_carry_init(config["NUM_ENVS"])

    rng, _rng = jax.random.split(rng)
    runner_state = (
        train_state,
        env_state,
        obsv,
        jnp.zeros((config["NUM_ENVS"]), dtype=bool),
        init_hstate,
        _rng,
    )

    def _update_step(runner_state, unused):
        def _env_step(runner_state, unused):
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            rng, _rng = jax.random.split(rng)

            # Masked Hidden-State Flushing
            if config["RNN"]["RNN_TYPE"] in ["sdlg_hybrid", "exact_sdlg", "ablation"]:
                hstate = flush_carry_state(hstate, last_done)

            # SELECT ACTION
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            intrinsic_bonus = 0.0
            if config["RNN"]["RNN_TYPE"] in ["sdlg_hybrid", "ablation"]:
                action_logits, value, bonus, hstate = network.apply(
                    train_state.params, hstate, ac_in)
                pi = distrax.Categorical(logits=action_logits)
                intrinsic_bonus = bonus.squeeze(0)  # (B,)
            else:
                hstate, pi, value = network.apply(
                    train_state.params, hstate, ac_in)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            value, action, log_prob = (
                value.squeeze(0),
                action.squeeze(0),
                log_prob.squeeze(0),
            )

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            obsv, env_state, reward, done, info = env.step(
                _rng, env_state, action, env_params)

            # Combine external and intrinsic rewards
            total_reward = reward + intrinsic_bonus

            transition = Transition(
                last_done, action, value, total_reward, log_prob, last_obs, info
            )
            runner_state = (train_state, env_state, obsv, done, hstate, rng)
            return runner_state, transition

        initial_hstate = runner_state[4]
        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, config["NUM_STEPS"]
        )

        # CALCULATE ADVANTAGE
        train_state, env_state, last_obs, last_done, hstate, rng = runner_state
        ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
        if config["RNN"]["RNN_TYPE"] in ["sdlg_hybrid", "ablation"]:
            _, last_val, _, _ = network.apply(train_state.params, hstate, ac_in)
        else:
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
        last_val = last_val.squeeze(0)

        def _calculate_gae(traj_batch, last_val, last_done):
            def _get_advantages(carry, transition):
                gae, next_value, next_done = carry
                done, value, reward = transition.done, transition.value, transition.reward
                delta = reward + config["GAMMA"] * \
                        next_value * (1 - next_done) - value
                gae = delta + config["GAMMA"] * \
                      config["GAE_LAMBDA"] * (1 - next_done) * gae
                return (gae, value, done), gae

            _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(
                last_val), last_val, last_done), traj_batch, reverse=True, unroll=16)
            return advantages, advantages + traj_batch.value

        advantages, targets = _calculate_gae(
            traj_batch, last_val, last_done)

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):
            def _update_minbatch(train_state, batch_info):
                init_hstate, traj_batch, advantages, targets = batch_info

                def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                    # RERUN NETWORK
                    first_state = init_hstate
                    if config["RNN"]["RNN_TYPE"] in ["sdlg_hybrid", "ablation"]:
                        action_logits, value, _, _ = network.apply(
                            params, first_state, (traj_batch.obs, traj_batch.done)
                        )
                        pi = distrax.Categorical(logits=action_logits)

                    else:
                        _, pi, value = network.apply(
                            params, first_state, (traj_batch.obs,
                                                  traj_batch.done)
                        )
                    log_prob = pi.log_prob(traj_batch.action)

                    # CALCULATE VALUE LOSS (HUBER LOSS FOR NUMERICAL STABILITY)
                    value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                    ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                    
                    def huber_loss(x, delta=10.0):
                        return jnp.where(jnp.abs(x) < delta, 0.5 * x**2, delta * (jnp.abs(x) - 0.5 * delta))
                        
                    value_losses = huber_loss(value - targets)
                    value_losses_clipped = huber_loss(value_pred_clipped - targets)
                    value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()

                    # CALCULATE ACTOR LOSS
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    # Stop gradient on GAE normalization to prevent NaN if std == 0
                    gae = jax.lax.stop_gradient((gae - gae.mean()) / (gae.std() + 1e-8))
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()
                    entropy = pi.entropy().mean()

                    total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                    )
                    return total_loss, (value_loss, loss_actor, entropy)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(
                    train_state.params, init_hstate, traj_batch, advantages, targets
                )

                # Compute unclipped global gradient norm of grads PyTree
                flat_grads, _ = jax.tree_util.tree_flatten(grads)
                grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in flat_grads) + 1e-8)

                train_state = train_state.apply_gradients(grads=grads)
                # Return gradient norm as part of the minbatch metrics
                return train_state, (total_loss, grad_norm)

            (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            ) = update_state

            rng, _rng = jax.random.split(rng)
            permutation = jax.random.permutation(_rng, config["NUM_ENVS"])

            # Correctly handle init_hstate (axis 0 = NUM_ENVS) vs traj_batch/advantages/targets (axis 1 = NUM_ENVS)
            shuffled_init_hstate = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), init_hstate
            )
            shuffled_traj_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=1), traj_batch
            )
            shuffled_advantages = jnp.take(advantages, permutation, axis=1)
            shuffled_targets = jnp.take(targets, permutation, axis=1)

            minibatch_init_hstate = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                ),
                shuffled_init_hstate,
            )

            minibatch_traj_batch = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(
                    jnp.reshape(
                        x,
                        [x.shape[0], config["NUM_MINIBATCHES"], -1]
                        + list(x.shape[2:]),
                    ),
                    1,
                    0,
                ),
                shuffled_traj_batch,
            )

            minibatch_advantages = jnp.swapaxes(
                jnp.reshape(
                    shuffled_advantages,
                    [advantages.shape[0], config["NUM_MINIBATCHES"], -1],
                ),
                1,
                0,
            )

            minibatch_targets = jnp.swapaxes(
                jnp.reshape(
                    shuffled_targets,
                    [targets.shape[0], config["NUM_MINIBATCHES"], -1],
                ),
                1,
                0,
            )

            minibatches = (
                minibatch_init_hstate,
                minibatch_traj_batch,
                minibatch_advantages,
                minibatch_targets,
            )

            train_state, (total_loss, grad_norm_history) = jax.lax.scan(
                _update_minbatch, train_state, minibatches
            )
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            return update_state, (total_loss, grad_norm_history)

        init_hstate = initial_hstate

        update_state = (
            train_state,
            init_hstate,
            traj_batch,
            advantages,
            targets,
            rng,
        )
        update_state, (loss_info, grad_norm_epoch) = jax.lax.scan(
            _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
        )
        train_state = update_state[0]

        # Extract maximum unclipped gradient norm during the entire epoch
        max_grad_norm_val = jnp.max(grad_norm_epoch)
        metric = traj_batch.info
        rng = update_state[-1]

        # --- PARAMETER FINITENESS PROBE ---
        # Recursively reduce all parameter arrays to check for any NaN/Inf occurrences
        all_finite = jax.tree_util.tree_reduce(
            lambda acc, x: acc & jnp.all(jnp.isfinite(x)),
            train_state.params,
            initializer=jnp.array(True)
        )
        nan_detected = ~all_finite

        metric = dict(metric)
        metric["parameter_nan_detected"] = nan_detected
        metric["max_gradient_norm"] = max_grad_norm_val
        metric["entropy"] = loss_info[1][2]
        metric["value_loss"] = loss_info[1][0]

        runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
        return runner_state, metric

    @jax.jit
    def update_fn(runner_state):
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["LOG_INTERVAL"]
        )
        return {"runner_state": runner_state, "metric": metric}

    return update_fn, runner_state