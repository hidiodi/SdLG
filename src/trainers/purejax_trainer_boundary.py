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
        embedding = nn.Dense(self.config["HIDDEN"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        embedding = nn.relu(embedding)
        rnn_in = (embedding, dones)
        hidden, embedding = self.rnn_module()(hidden, rnn_in)
        actor_mean = nn.Dense(self.config["HIDDEN"], kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        actor_mean = nn.relu(actor_mean)
        raw_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        action_logits = 5.0 * jnp.tanh(raw_logits / 5.0)
        action_logits = jnp.nan_to_num(action_logits, nan=0.0, posinf=5.0, neginf=-5.0)
        pi = distrax.Categorical(logits=action_logits)
        critic = nn.Dense(self.config["HIDDEN"], kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
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
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]

    if config['OPTIMISTIC_RESETS']:
        default_ratio = min(64, config["NUM_ENVS"])
        reset_ratio = config.get("RESET_RATIO", default_ratio)
        if config["NUM_ENVS"] % reset_ratio != 0:
            reset_ratio = min(16, config["NUM_ENVS"])
        env = OptimisticResetVecEnvWrapper(env, config["NUM_ENVS"], reset_ratio=reset_ratio)
    else:
        env = BatchEnvWrapper(env, config["NUM_ENVS"])

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    if config["RNN"]["RNN_TYPE"] == "sdlg_hybrid":
        from src.models.purejaxrl.wrappers import ActorCriticSdLGHybrid
        network = ActorCriticSdLGHybrid(action_dim=env.action_space(env_params).n, config=config)
    elif config["RNN"]["RNN_TYPE"] == "ablation":
        network = ActorCriticRNN(rnn_module, env.action_space(env_params).n, config=config)
    else:
        network = ActorCriticRNN(rnn_module, env.action_space(env_params).n, config=config)

    rng, _rng = jax.random.split(rng)
    init_x = (jnp.zeros((1, config["NUM_ENVS"], *env.observation_space(env_params).shape)), jnp.zeros((1, config["NUM_ENVS"])))
    init_hstate = rnn_carry_init(config["NUM_ENVS"])
    network_params = network.init(_rng, init_hstate, init_x)

    if config["ANNEAL_LR"]:
        tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(learning_rate=linear_schedule, eps=1e-5))
    else:
        tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
    
    train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

    rng, _rng = jax.random.split(rng)
    obsv, env_state = env.reset(_rng, env_params)
    init_hstate = rnn_carry_init(config["NUM_ENVS"])

    rng, _rng = jax.random.split(rng)
    runner_state = (train_state, env_state, obsv, jnp.zeros((config["NUM_ENVS"]), dtype=bool), init_hstate, _rng)

    def _update_step(runner_state, unused):
        def _env_step(runner_state, unused):
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            rng, _rng = jax.random.split(rng)

            if config["RNN"]["RNN_TYPE"] in ["sdlg_hybrid", "exact_sdlg", "ablation"]:
                hstate = flush_carry_state(hstate, last_done)

            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            intrinsic_bonus = 0.0
            if config["RNN"]["RNN_TYPE"] in ["sdlg_hybrid", "ablation"]:
                action_logits, value, bonus, hstate = network.apply(train_state.params, hstate, ac_in)
                pi = distrax.Categorical(logits=action_logits)
                intrinsic_bonus = bonus.squeeze(0)
            else:
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
            
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            value, action, log_prob = value.squeeze(0), action.squeeze(0), log_prob.squeeze(0)

            rng, _rng = jax.random.split(rng)
            obsv, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)
            total_reward = reward + intrinsic_bonus

            transition = Transition(last_done, action, value, total_reward, log_prob, last_obs, info)
            runner_state = (train_state, env_state, obsv, done, hstate, rng)
            return runner_state, transition

        initial_hstate = runner_state[4]
        runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])

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
                delta = reward + config["GAMMA"] * next_value * (1 - next_done) - value
                gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - next_done) * gae
                return (gae, value, done), gae
            _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val), last_val, last_done), traj_batch, reverse=True, unroll=16)
            return advantages, advantages + traj_batch.value

        advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

        def _update_epoch(update_state, unused):
            def _update_minbatch_fixed(train_state, batch_info):
                init_hstate, traj_batch, advantages, targets = batch_info
                
                def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                    first_state = init_hstate
                    if config["RNN"]["RNN_TYPE"] in ["sdlg_hybrid", "ablation"]:
                        action_logits, value, _, _ = network.apply(params, first_state, (traj_batch.obs, traj_batch.done))
                        pi = distrax.Categorical(logits=action_logits)
                    else:
                        _, pi, value = network.apply(params, first_state, (traj_batch.obs, traj_batch.done))
                    log_prob = pi.log_prob(traj_batch.action)
                    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                    def huber_loss(x, delta=10.0):
                        return jnp.where(jnp.abs(x) < delta, 0.5 * x**2, delta * (jnp.abs(x) - 0.5 * delta))
                    value_losses = huber_loss(value - targets)
                    value_losses_clipped = huber_loss(value_pred_clipped - targets)
                    value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()
                    
                    td_error_sq = jnp.square(value - targets)
                    # td_error_sq shape is (T, batch_size) where T is config["NUM_STEPS"]
                    # chunk-boundary is T-1
                    boundary_td = jnp.mean(td_error_sq[-1])
                    interior_td = jnp.mean(td_error_sq[:-1])
                    boundary_bias = boundary_td - interior_td

                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = jax.lax.stop_gradient((gae - gae.mean()) / (gae.std() + 1e-8))
                    loss_actor1 = ratio * gae
                    loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                    entropy = pi.entropy().mean()

                    total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                    return total_loss, (value_loss, loss_actor, entropy, boundary_bias)
                
                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                (total_loss, aux), grads = grad_fn(train_state.params, init_hstate, traj_batch, advantages, targets)
                value_loss, loss_actor, entropy, boundary_bias = aux
                
                flat_grads, _ = jax.tree_util.tree_flatten(grads)
                grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in flat_grads) + 1e-8)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, (total_loss, grad_norm, entropy, value_loss, boundary_bias)

            (train_state, init_hstate, traj_batch, advantages, targets, rng) = update_state
            rng, _rng = jax.random.split(rng)
            permutation = jax.random.permutation(_rng, config["NUM_ENVS"])

            shuffled_init_hstate = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), init_hstate)
            shuffled_traj_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=1), traj_batch)
            shuffled_advantages = jnp.take(advantages, permutation, axis=1)
            shuffled_targets = jnp.take(targets, permutation, axis=1)

            minibatch_init_hstate = jax.tree_util.tree_map(lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])), shuffled_init_hstate)
            minibatch_traj_batch = jax.tree_util.tree_map(lambda x: jnp.swapaxes(jnp.reshape(x, [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:])), 1, 0), shuffled_traj_batch)
            minibatch_advantages = jnp.swapaxes(jnp.reshape(shuffled_advantages, [advantages.shape[0], config["NUM_MINIBATCHES"], -1]), 1, 0)
            minibatch_targets = jnp.swapaxes(jnp.reshape(shuffled_targets, [targets.shape[0], config["NUM_MINIBATCHES"], -1]), 1, 0)

            minibatches = (minibatch_init_hstate, minibatch_traj_batch, minibatch_advantages, minibatch_targets)

            train_state, metrics = jax.lax.scan(_update_minbatch_fixed, train_state, minibatches)
            update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
            return update_state, metrics

        update_state = (train_state, initial_hstate, traj_batch, advantages, targets, rng)
        update_state, metrics_epoch = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
        train_state = update_state[0]
        
        grad_norm_epoch = metrics_epoch[1]
        entropy_epoch = metrics_epoch[2]
        value_loss_epoch = metrics_epoch[3]
        boundary_bias_epoch = metrics_epoch[4]

        max_grad_norm_val = jnp.max(grad_norm_epoch)
        metric = traj_batch.info
        rng = update_state[-1]

        metric = dict(metric)
        metric["max_gradient_norm"] = max_grad_norm_val
        metric["entropy"] = jnp.mean(entropy_epoch)
        metric["value_loss"] = jnp.mean(value_loss_epoch)
        metric["boundary_bias"] = jnp.mean(boundary_bias_epoch)

        runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
        return runner_state, metric

    @jax.jit
    def update_fn(runner_state):
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, config["LOG_INTERVAL"])
        return {"runner_state": runner_state, "metric": metric}

    return update_fn, runner_state
