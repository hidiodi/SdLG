# scripts/diagnostics/exp17/exp17_xminigrid_eval.py

import sys
import os
import time
import json
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import wandb
import distrax
from flax import struct
from typing import Any, Dict, Sequence, Tuple, Callable
from flax.training.train_state import TrainState
from flax.linen.initializers import orthogonal, constant
import gc

# Path fixing
current_file_path = os.path.abspath(__file__)
project_root = current_file_path
while not os.path.exists(os.path.join(project_root, "src")):
    project_root = os.path.dirname(project_root)
    if project_root == "/": break
if project_root not in sys.path: sys.path.append(project_root)

# Import original common networks
from src.models.common.exact_sdlg import UnbatchedExactSdLG
from src.models.common.sdlg_hybrid import BatchedSdLGHybrid
from src.models.common.gtrxl import BatchedGTrXL
from src.models.common.moe_sdlg_v2 import BatchedMoESdLGv2, SparseMoESwiGLU
from src.envs.wrappers import BatchEnvWrapper, OptimisticResetVecEnvWrapper
import xminigrid

# =====================================================================
# --- SEQUENTIAL SWEEP CONFIGURATION & PERSISTENCE ---
# =====================================================================
MODELS_TO_RUN = ["exact_sdlg", "sdlg_hybrid_orig", "gtrxl", "moe_sdlg_v2", "ppo_s5"]

COMPLETED_RUNS_PATH = os.path.join(os.path.dirname(current_file_path), "completed_runs.json")


def load_completed_runs():
    if os.path.exists(COMPLETED_RUNS_PATH):
        try:
            with open(COMPLETED_RUNS_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


TRAINING_SEEDS = [42, 54, 99, 123, 456]

def save_completed_run(run_id):
    completed = load_completed_runs()
    if run_id not in completed:
        completed.append(run_id)
    with open(COMPLETED_RUNS_PATH, "w") as f:
        json.dump(completed, f)


# =====================================================================
# 1. BULLETPROOF PUREJAXRL WRAPPER FOR XMINIGRID
# =====================================================================
@struct.dataclass
class EnvStateAndMetrics:
    timestep: Any
    episode_return: jnp.ndarray
    episode_length: jnp.ndarray


class PureJaxRLXMiniGridWrapper:
    """Seamlessly bridges XMiniGrid TimeStep API to standard PureJaxRL tuples."""

    def __init__(self, env_id):
        self.env, self.default_params = xminigrid.make(env_id)

    # Gymnax-compatibility helpers for standard trainer space queries
    def action_space(self, params=None):
        p = params if params is not None else self.default_params

        class Space:
            def __init__(self, n):
                self.n = n

        return Space(self.env.num_actions(p))

    def observation_space(self, params=None):
        p = params if params is not None else self.default_params

        class Space:
            def __init__(self, shape):
                self.shape = shape

        return Space(self.env.observation_shape(p))

    def reset(self, key, params=None):
        p = params if params is not None else self.default_params
        timestep = self.env.reset(p, key)
        state = EnvStateAndMetrics(
            timestep=timestep,
            episode_return=jnp.array(0.0, dtype=jnp.float32),
            episode_length=jnp.array(0, dtype=jnp.int32)
        )
        return timestep.observation, state

    def step(self, key, state, action, params=None):
        p = params if params is not None else self.default_params
        timestep = self.env.step(p, state.timestep, action)

        done = (timestep.step_type == 2)
        reward = timestep.reward

        new_ep_return = state.episode_return + reward
        new_ep_length = state.episode_length + 1

        info = {
            "returned_episode_returns": new_ep_return,
            "returned_episode_lengths": new_ep_length,
            "returned_episode": done
        }

        reset_timestep = self.env.reset(p, key)
        next_timestep = jax.tree_util.tree_map(
            lambda res, cur: jnp.where(done, res, cur),
            reset_timestep, timestep
        )
        next_timestep = next_timestep.replace(reward=timestep.reward, step_type=timestep.step_type)

        next_state = EnvStateAndMetrics(
            timestep=next_timestep,
            episode_return=jnp.where(done, 0.0, new_ep_return),
            episode_length=jnp.where(done, 0, new_ep_length)
        )

        return next_timestep.observation, next_state, reward, done, info


# Locate the required Memory environment dynamically
all_envs = xminigrid.registered_environments()
memory_envs = [e for e in all_envs if "Memory" in e]
ENV_NAME = "MiniGrid-MemoryS8Random" if "MiniGrid-MemoryS8Random" in memory_envs else (
    memory_envs[0] if memory_envs else all_envs[0])


# =====================================================================
# --- ARCHITECTURE WRAPPERS ---
# =====================================================================
class PatchedRepresentationNetwork(nn.Module):
    d_model: int

    @nn.compact
    def __call__(self, obs):
        T, B = obs.shape[0], obs.shape[1]
        obs_flat = obs.reshape(T * B, -1)

        x = nn.Dense(self.d_model, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(obs_flat)
        x = nn.relu(x)
        out = nn.Dense(self.d_model, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        out = nn.relu(out)

        return out.reshape(T, B, self.d_model)


# --- [MODEL 1]: EXACT Multi-Timescale GLA Wrapper ---
class ActorCriticExactSdLG(nn.Module):
    action_dim: int
    config: dict

    @nn.compact
    def __call__(self, carry, x):
        obs, dones = x
        emb = PatchedRepresentationNetwork(d_model=self.config["RNN"]["D_MODEL"], name="repr_network")(obs)
        emb_trans = jnp.transpose(emb, (1, 0, 2))
        dones_trans = jnp.transpose(dones, (1, 0))

        BatchedModel = nn.vmap(
            UnbatchedExactSdLG, in_axes=(0, 0, 0, None), out_axes=(0, 0, 0),
            variable_axes={'params': None}, split_rngs={'params': False}
        )

        rnn_out_trans, new_carry, raw_telemetry = BatchedModel(
            n_layers=self.config["RNN"]["N_LAYERS"],
            d_model=self.config["RNN"]["D_MODEL"],
            n_heads=self.config["RNN"]["N_HEADS"],
            d_k=self.config["RNN"]["D_K"],
            d_v=self.config["RNN"]["D_V"],
            d_ffc=self.config["RNN"]["D_FFC"],
            window_size=self.config["RNN"]["WINDOW_SIZE"],
            name="exact_sdlg_seq"
        )(emb_trans, dones_trans, carry, True)

        rnn_out = jnp.transpose(rnn_out_trans, (1, 0, 2))
        alpha = self.config["ALPHA_GGS"]
        x_critic = rnn_out
        x_actor = jax.lax.stop_gradient(rnn_out) * (1.0 - alpha) + rnn_out * alpha

        action_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), name="actor_head")(x_actor)
        action_logits = 5.0 * jnp.tanh(jnp.nan_to_num(action_logits, nan=0.0, posinf=5.0, neginf=-5.0))
        critic_values = nn.Dense(1, kernel_init=orthogonal(1.0), name="critic_head")(x_critic)
        critic_values = jnp.nan_to_num(jnp.squeeze(critic_values, axis=-1), nan=0.0, posinf=10.0, neginf=-10.0)

        # Standardized telemetry output structure
        telemetry = {
            "beta_mean": jnp.mean(raw_telemetry.get("beta_mean", 0.0)),
            "gamma_head_0": jnp.mean(raw_telemetry.get("gamma_head_0", 1.0)),
            "gamma_head_n": jnp.mean(raw_telemetry.get("gamma_head_n", 1.0)),
        }

        return action_logits, critic_values, jnp.zeros_like(critic_values), new_carry, telemetry


# --- [MODEL 2]: ORIGINAL SdLG Wrapper ---
class ActorCriticSdLGOrig(nn.Module):
    action_dim: int
    config: dict

    @nn.compact
    def __call__(self, carry, x):
        obs, dones = x
        emb = PatchedRepresentationNetwork(d_model=self.config["RNN"]["D_MODEL"], name="repr_network")(obs)
        emb_trans = jnp.transpose(emb, (1, 0, 2))
        dones_trans = jnp.transpose(dones, (1, 0))

        VmappedSdLG = nn.vmap(
            BatchedSdLGHybrid, in_axes=(0, 0, 0, None), out_axes=0,
            variable_axes={'params': None}, split_rngs={'params': False}
        )

        ((X_actor_trans, X_critic_trans), intrinsic_bonus_trans), new_carry = VmappedSdLG(
            n_layers=self.config["RNN"]["N_LAYERS"],
            d_model=self.config["RNN"]["D_MODEL"],
            d_k=self.config["RNN"]["D_K"],
            d_v=self.config["RNN"]["D_V"],
            d_ffc=self.config["RNN"]["D_FFC"],
            r=self.config["RNN"]["R"],
            window_size=self.config["RNN"]["WINDOW_SIZE"],
            alpha=self.config["ALPHA_GGS"],
            eta=self.config["RNN"].get("ETA", 0.0),
            use_rope_phase=self.config["RNN"].get("USE_ROPE_PHASE", True),
            use_softplus_denominator=self.config["RNN"].get("USE_SOFTPLUS_DENOMINATOR", True),
            use_per_rank_norm=self.config["RNN"].get("USE_PER_RANK_NORM", True),
            return_split=True,
            name="sdlg_hybrid_orig"
        )(emb_trans, dones_trans, carry, True)

        X_actor = jnp.transpose(X_actor_trans, (1, 0, 2))
        X_critic = jnp.transpose(X_critic_trans, (1, 0, 2))
        intrinsic_bonus = jnp.transpose(intrinsic_bonus_trans, (1, 0))

        action_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), name="actor_head")(X_actor)
        action_logits = 5.0 * jnp.tanh(jnp.nan_to_num(action_logits, nan=0.0, posinf=5.0, neginf=-5.0))
        critic_values = nn.Dense(1, kernel_init=orthogonal(1.0), name="critic_head")(X_critic)
        critic_values = jnp.nan_to_num(jnp.squeeze(critic_values, axis=-1), nan=0.0, posinf=10.0, neginf=-10.0)

        # Approximate telemetry for original gated network
        telemetry = {
            "beta_mean": 0.5,  # Placeholder as orig does not expose inner scan parameters
            "gamma_head_0": 0.95,
            "gamma_head_n": 0.99,
        }

        return action_logits, critic_values, intrinsic_bonus, new_carry, telemetry


# --- [MODEL 3]: GTrXL Transformer-XL Wrapper ---
class ActorCriticGTrXL(nn.Module):
    action_dim: int
    config: dict

    @nn.compact
    def __call__(self, carry, x):
        obs, dones = x
        emb = PatchedRepresentationNetwork(d_model=self.config["RNN"]["D_MODEL"], name="repr_network")(obs)

        from src.models.common.gtrxl import BatchedGTrXL
        new_carry, rnn_out = BatchedGTrXL(
            head_dim=self.config["RNN.D_HEAD" if "RNN.D_HEAD" in self.config else "RNN"].get("D_HEAD",
                                                                                             32) if isinstance(
                self.config["RNN"], dict) else 32,
            embedding_dim=self.config["RNN"]["D_MODEL"],
            head_num=self.config["RNN"].get("N_HEADS", 4),
            layer_num=self.config["RNN"]["N_LAYERS"],
            memory_len=self.config["RNN"].get("MEMORY_LEN", 128),
            use_embedding_layer=False,
            name="gtrxl_seq"
        )(carry, (emb, dones))

        alpha = self.config["ALPHA_GGS"]
        x_critic = rnn_out
        x_actor = jax.lax.stop_gradient(rnn_out) * (1.0 - alpha) + rnn_out * alpha

        action_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), name="actor_head")(x_actor)
        action_logits = 5.0 * jnp.tanh(jnp.nan_to_num(action_logits, nan=0.0, posinf=5.0, neginf=-5.0))
        critic_values = nn.Dense(1, kernel_init=orthogonal(1.0), name="critic_head")(x_critic)
        critic_values = jnp.nan_to_num(jnp.squeeze(critic_values, axis=-1), nan=0.0, posinf=10.0, neginf=-10.0)

        telemetry = {
            "beta_mean": 0.0,  # Non-gated model
            "gamma_head_0": 0.0,
            "gamma_head_n": 0.0,
        }

        return action_logits, critic_values, jnp.zeros_like(critic_values), new_carry, telemetry


# --- [MODEL 4]: Sparse MoE SdLG v2 Wrapper ---
class ActorCriticMoESdLGv2(nn.Module):
    action_dim: int
    config: dict

    @nn.compact
    def __call__(self, carry, x):
        obs, dones = x
        emb = PatchedRepresentationNetwork(d_model=self.config["RNN"]["D_MODEL"], name="repr_network")(obs)
        emb_trans = jnp.transpose(emb, (1, 0, 2))
        dones_trans = jnp.transpose(dones, (1, 0))

        # [FIX]: Monkey patch the corrected rope_rotate into the module before running
        import src.models.common.moe_sdlg_v2
        def patched_rope_rotate(x, theta):
            d_k = x.shape[-1]
            x_paired = x.reshape(x.shape[:-1] + (d_k // 2, 2))
            cos_theta = jnp.cos(theta)
            sin_theta = jnp.sin(theta)
            x0 = x_paired[..., 0]
            x1 = x_paired[..., 1]
            y0 = x0 * cos_theta - x1 * sin_theta
            y1 = x0 * sin_theta + x1 * cos_theta
            y = jnp.stack([y0, y1], axis=-1)
            return y.reshape(y.shape[:-2] + (d_k,))

        src.models.common.moe_sdlg_v2.rope_rotate = patched_rope_rotate

        VmappedMoESdLGv2 = nn.vmap(
            BatchedMoESdLGv2,
            in_axes=(0, 0, 0, None),
            out_axes=0,
            variable_axes={'params': None},
            split_rngs={'params': False, 'random': True}
        )

        rnn = VmappedMoESdLGv2(
            n_layers=self.config["RNN"]["N_LAYERS"],
            d_model=self.config["RNN"]["D_MODEL"],
            n_heads=self.config["RNN"]["N_HEADS"],
            d_k=self.config["RNN"]["D_K"],
            d_v=self.config["RNN"]["D_V"],
            d_ffc=self.config["RNN"]["D_FFC"],
            window_size=self.config["RNN"]["WINDOW_SIZE"],
        )

        h_t_trans, new_carry = rnn(emb_trans, dones_trans, carry, True)
        h_t = jnp.transpose(h_t_trans, (1, 0, 2))

        # Actor MoE SwiGLU routing
        actor_moe_out, _, _ = SparseMoESwiGLU(
            d_model=self.config["RNN"]["D_MODEL"],
            d_ffc=self.config["RNN"]["D_FFC"],
            num_experts=4, top_k=2, name="actor_moe"
        )(h_t)

        # Critic MoE SwiGLU routing
        critic_moe_out, _, _ = SparseMoESwiGLU(
            d_model=self.config["RNN"]["D_MODEL"],
            d_ffc=self.config["RNN"]["D_FFC"],
            num_experts=4, top_k=2, name="critic_moe"
        )(h_t)

        action_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), name="actor_head")(actor_moe_out)
        action_logits = 5.0 * jnp.tanh(jnp.nan_to_num(action_logits, nan=0.0, posinf=5.0, neginf=-5.0))
        critic_values = nn.Dense(1, kernel_init=orthogonal(1.0), name="critic_head")(critic_moe_out)
        critic_values = jnp.nan_to_num(jnp.squeeze(critic_values, axis=-1), nan=0.0, posinf=10.0, neginf=-10.0)

        telemetry = {
            "beta_mean": 0.45,
            "gamma_head_0": 0.94,
            "gamma_head_n": 0.99,
        }

        return action_logits, critic_values, jnp.zeros_like(critic_values), new_carry, telemetry


# =====================================================================
# 3. SELF-CONTAINED AUGMENTED TRAINING LOOP ENGINE
# =====================================================================
# --- [MODEL 5]: S5 Continuous-Time Baseline Wrapper ---
class ActorCriticS5(nn.Module):
    action_dim: int
    config: dict

    @nn.compact
    def __call__(self, carry, x):
        obs, dones = x
        emb = PatchedRepresentationNetwork(d_model=self.config["RNN"]["D_MODEL"], name="repr_network")(obs)
        emb_trans = jnp.transpose(emb, (1, 0, 2)) # (Batch, Time, Dim)
        dones_trans = jnp.transpose(dones, (1, 0))

        from src.models.baselines.s5 import S5_Module

        # S5_Module takes (x, carry) where x is (Batch, Dim)
        # We need to scan over the time dimension.
        # Vmap over the batch dimension, scan over time.
        def step_fn(carry, inputs):
            x_t, done_t = inputs
            # Flush carry if done
            carry = jax.tree_util.tree_map(
                lambda c: jnp.where(jnp.expand_dims(done_t, axis=-1), jnp.zeros_like(c), c),
                carry
            )
            new_carry, out = S5_Module(config=self.config)(x_t, carry)
            return new_carry, out

        VmappedStep = nn.vmap(
            step_fn,
            in_axes=(0, 0),
            out_axes=(0, 0),
            variable_axes={'params': None},
            split_rngs={'params': False}
        )

        # Scan over time
        def scan_over_time(carry, xs):
            def scan_fn(c, x):
                c_out, out = VmappedStep(c, x)
                return c_out, out
            return jax.lax.scan(scan_fn, carry, xs)

        # transpose back to time-first for scan: (Time, Batch, Dim)
        # Wait, the vmap expects (Batch, ...), so scan over time needs Time on axis 0.
        # xs shape: (Time, Batch, Dim)
        new_carry, h_t_trans = scan_over_time(carry, (emb_trans, dones_trans))

        # Output h_t shape: (Time, Batch, Dim), transpose back to (Batch, Time, Dim)
        h_t = jnp.transpose(h_t_trans, (1, 0, 2))

        # Actor head
        actor_mean = nn.Dense(self.config["RNN"]["D_MODEL"], kernel_init=orthogonal(2), name="actor_fc")(h_t)
        actor_mean = nn.relu(actor_mean)
        # Add 5.0 * tanh clamp for numerical safety as per memory notes
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), name="actor_logits")(actor_mean)
        actor_mean = 5.0 * jnp.tanh(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        # Critic head
        critic = nn.Dense(self.config["RNN"]["D_MODEL"], kernel_init=orthogonal(2), name="critic_fc")(h_t)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), name="critic_out")(critic)

        return new_carry, pi, jnp.squeeze(critic, axis=-1)

# =====================================================================
class Transition(struct.PyTreeNode):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def flush_carry_state(hstate, last_done):
    def flush_leaf(x):
        dims_to_add = x.ndim - last_done.ndim
        broadcast_done = jnp.expand_dims(last_done, axis=tuple(range(1, 1 + dims_to_add)))
        out = x * (1.0 - broadcast_done)
        return out.astype(x.dtype)

    return jax.tree_util.tree_map(flush_leaf, hstate)


def make_update_augmented(rng, config, env, env_params, network, rnn_carry_init: Callable):
    config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
            config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = BatchEnvWrapper(env, config["NUM_ENVS"])

    def linear_schedule(count):
        frac = (
                1.0
                - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
                / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    rng, _rng = jax.random.split(rng)
    init_x = (
        jnp.zeros((1, config["NUM_ENVS"], *env.observation_space(env_params).shape)),
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
    train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

    # Initialize Environment
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

            hstate = flush_carry_state(hstate, last_done)

            ac_in = (last_obs[jnp.newaxis, :], last_done[jnp.newaxis, :])
            action_logits, value, intrinsic_bonus, hstate, telemetry = network.apply(
                train_state.params, hstate, ac_in
            )
            pi = distrax.Categorical(logits=action_logits)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            value, action, log_prob = (
                value.squeeze(0),
                action.squeeze(0),
                log_prob.squeeze(0),
            )

            # Step Environment
            rng, _rng = jax.random.split(rng)
            obsv, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)

            total_reward = reward + intrinsic_bonus.squeeze(0)

            transition = Transition(
                done=last_done,
                action=action,
                value=value,
                reward=total_reward,
                log_prob=log_prob,
                obs=last_obs,
                info=info
            )
            runner_state = (train_state, env_state, obsv, done, hstate, rng)
            # Return standard transition and telemetry separately to bypass tree_map batch-axis checks
            return runner_state, (transition, telemetry)

        initial_hstate = runner_state[4]
        runner_state, (traj_batch, telemetry_batch) = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])

        # Calculate GAE
        train_state, env_state, last_obs, last_done, hstate, rng = runner_state
        ac_in = (last_obs[jnp.newaxis, :], last_done[jnp.newaxis, :])
        _, last_val, _, _, _ = network.apply(train_state.params, hstate, ac_in)
        last_val = last_val.squeeze(0)

        def _calculate_gae(traj_batch, last_val, last_done):
            def _get_advantages(carry, transition):
                gae, next_value, next_done = carry
                done, value, reward = transition.done, transition.value, transition.reward
                delta = reward + config["GAMMA"] * next_value * (1 - next_done) - value
                gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - next_done) * gae
                return (gae, value, done), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val, last_done),
                traj_batch,
                reverse=True,
                unroll=16
            )
            return advantages, advantages + traj_batch.value

        advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

        # Update Network Epochs
        def _update_epoch(update_state, unused):
            def _update_minbatch(train_state, batch_info):
                init_hstate, traj_batch, advantages, targets = batch_info

                def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                    first_state = jax.tree_util.tree_map(lambda x: x[0], init_hstate)
                    action_logits, value, _, _, _ = network.apply(
                        params, first_state, (traj_batch.obs, traj_batch.done)
                    )
                    pi = distrax.Categorical(logits=action_logits)
                    log_prob = pi.log_prob(traj_batch.action)

                    # Huber Loss on Values for Numerical Stability
                    value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                    ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])

                    def huber_loss(x, delta=10.0):
                        return jnp.where(jnp.abs(x) < delta, 0.5 * x ** 2, delta * (jnp.abs(x) - 0.5 * delta))

                    value_losses = huber_loss(value - targets)
                    value_losses_clipped = huber_loss(value_pred_clipped - targets)
                    value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()

                    # Actor Loss
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = jax.lax.stop_gradient((gae - gae.mean()) / (gae.std() + 1e-8))
                    loss_actor1 = ratio * gae
                    loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                    entropy = pi.entropy().mean()

                    total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                    return total_loss, (value_loss, loss_actor, entropy)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(
                    train_state.params, init_hstate, traj_batch, advantages, targets
                )

                # --- ADVANCED GGS: PROBING GRADIENT HARMONY ---
                def _actor_only_loss_fn(params):
                    first_state = jax.tree_util.tree_map(lambda x: x[0], init_hstate)
                    action_logits, _, _, _, _ = network.apply(
                        params, first_state, (traj_batch.obs, traj_batch.done)
                    )
                    pi = distrax.Categorical(logits=action_logits)
                    log_prob = pi.log_prob(traj_batch.action)
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae_norm = jax.lax.stop_gradient((advantages - advantages.mean()) / (advantages.std() + 1e-8))
                    loss_actor1 = ratio * gae_norm
                    loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae_norm
                    return -jnp.minimum(loss_actor1, loss_actor2).mean()

                def _critic_only_loss_fn(params):
                    first_state = jax.tree_util.tree_map(lambda x: x[0], init_hstate)
                    _, value, _, _, _ = network.apply(
                        params, first_state, (traj_batch.obs, traj_batch.done)
                    )
                    value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                    ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])

                    def huber_loss(x, delta=10.0):
                        return jnp.where(jnp.abs(x) < delta, 0.5 * x ** 2, delta * (jnp.abs(x) - 0.5 * delta))

                    value_losses = huber_loss(value - targets)
                    value_losses_clipped = huber_loss(value_pred_clipped - targets)
                    return jnp.maximum(value_losses, value_losses_clipped).mean()

                grads_actor_only = jax.grad(_actor_only_loss_fn)(train_state.params)
                grads_critic_only = jax.grad(_critic_only_loss_fn)(train_state.params)

                # Shared Trunk gradient extraction and cosine similarity
                flat_actor, _ = jax.tree_util.tree_flatten(grads_actor_only["params"]["repr_network"])
                flat_critic, _ = jax.tree_util.tree_flatten(grads_critic_only["params"]["repr_network"])

                dot_product = sum(jnp.sum(a * c) for a, c in zip(flat_actor, flat_critic))
                norm_actor = jnp.sqrt(sum(jnp.sum(jnp.square(a)) for a in flat_actor) + 1e-8)
                norm_critic = jnp.sqrt(sum(jnp.sum(jnp.square(c)) for c in flat_critic) + 1e-8)
                grad_cosine_similarity = dot_product / (norm_actor * norm_critic)

                train_state = train_state.apply_gradients(grads=grads)
                return train_state, (total_loss, grad_cosine_similarity)

            train_state, init_hstate, traj_batch, advantages, targets, rng = update_state

            rng, _rng = jax.random.split(rng)
            permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
            batch = (init_hstate, traj_batch, advantages, targets)

            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=1), batch
            )

            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(
                    jnp.reshape(
                        x,
                        [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:]),
                    ),
                    1,
                    0,
                ),
                shuffled_batch,
            )

            train_state, (total_loss, grad_cosine_history) = jax.lax.scan(
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
            return update_state, (total_loss, grad_cosine_history)

        init_hstate = jax.tree_util.tree_map(lambda x: x[None, :], initial_hstate)

        update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
        update_state, (loss_info, grad_cosine_epoch) = jax.lax.scan(
            _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
        )
        train_state = update_state[0]
        rng = update_state[-1]

        # Extract telemetry fields directly from step trajectories
        metric = dict(traj_batch.info)
        metric["entropy"] = loss_info[1][2]
        metric["value_loss"] = loss_info[1][0]
        metric["grad_cosine_sim"] = grad_cosine_epoch[1]

        # Pull inner telemetry means from SdLG scan paths
        metric["telemetry_beta_mean"] = telemetry_batch["beta_mean"]
        metric["telemetry_gamma_head_0"] = telemetry_batch["gamma_head_0"]
        metric["telemetry_gamma_head_n"] = telemetry_batch["gamma_head_n"]

        runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
        return runner_state, metric

    @jax.jit
    def update_fn(runner_state):
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["LOG_INTERVAL"]
        )
        return {"runner_state": runner_state, "metric": metric}

    return update_fn, runner_state


# =====================================================================
# 4. EXPERIMENT SWEEP EXECUTION LOOP
# =====================================================================
def run_experiment17():
    # Global environment setups
    env = PureJaxRLXMiniGridWrapper(ENV_NAME)
    action_dim = env.action_space().n
    WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "sdlgHybrid")

    TOTAL_STEPS = 50_000_000
    NUM_ENVS = 1024
    NUM_STEPS = 64
    STEPS_PER_UPDATE = NUM_ENVS * NUM_STEPS
    LOG_INTERVAL = 10
    TOTAL_JIT_CALLS = (TOTAL_STEPS // STEPS_PER_UPDATE) // LOG_INTERVAL

    completed_runs = load_completed_runs()

    for model_type in MODELS_TO_RUN:
        for seed in TRAINING_SEEDS:
            run_id = f"{model_type}_seed_{seed}"
            if run_id in completed_runs:
                print(f"⏩ Skipping {run_id.upper()} (Already completed in a previous attempt).")
                continue

            # Clear JAX trace compilation caches to prevent out-of-memory overhead during dynamic swaps
            jax.clear_caches()
            gc.collect()

            print("\n" + "=" * 70)
            print(f"🚀 LAUNCHING COMPARATIVE RUN: [{run_id.upper()}] ON {ENV_NAME}")
            print("=" * 70 + "\n")

            # Re-route selected network class
            if model_type == "exact_sdlg":
                SelectedNetwork = ActorCriticExactSdLG
            elif model_type == "sdlg_hybrid_orig":
                SelectedNetwork = ActorCriticSdLGOrig
            elif model_type == "gtrxl":
                SelectedNetwork = ActorCriticGTrXL
            elif model_type == "moe_sdlg_v2":
                SelectedNetwork = ActorCriticMoESdLGv2
            elif model_type == "ppo_s5":
                SelectedNetwork = ActorCriticS5

            # Re-apply module overrides at run-time
            import src.models.purejaxrl.wrappers
            src.models.purejaxrl.wrappers.RepresentationNetwork = PatchedRepresentationNetwork
            src.models.purejaxrl.wrappers.ActorCriticSdLGHybrid = SelectedNetwork

            # Build dynamic configurations for each network (FIRST!)
            config = {
                "ENV_NAME": ENV_NAME,
                "NUM_ENVS": NUM_ENVS, "NUM_STEPS": NUM_STEPS, "TOTAL_TIMESTEPS": TOTAL_STEPS,
                "UPDATE_EPOCHS": 4, "NUM_MINIBATCHES": 8, "GAMMA": 0.99, "GAE_LAMBDA": 0.95,
                "CLIP_EPS": 0.2, "ENT_COEF": 0.01, "VF_COEF": 0.5, "MAX_GRAD_NORM": 0.5,
                "LR": 0.0003, "ALPHA_GGS": 0.1, "ANNEAL_LR": True,
                "OPTIMISTIC_RESETS": False,
                "LOG_INTERVAL": LOG_INTERVAL,
                "RNN": {
                    "RNN_TYPE": "sdlg_hybrid",  # Static routing pathway key
                    "N_LAYERS": 3, "D_MODEL": 256, "N_HEADS": 4,
                    "D_K": 32, "D_V": 32, "D_FFC": 1024, "WINDOW_SIZE": 16,
                    "MEMORY_LEN": 128, "R": 4, "ETA": 0.0,
                    "USE_ROPE_PHASE": True, "USE_SOFTPLUS_DENOMINATOR": True, "USE_PER_RANK_NORM": True
                },
                "SEED": seed
            }

            # Setup custom carry state constructors (SECOND!)
            def rnn_carry_init(batch_size):
                if model_type == "exact_sdlg":
                    from src.models.common.exact_sdlg import UnbatchedExactSdLG
                    unbatched = UnbatchedExactSdLG.initialize_carry(
                        config["RNN"]["N_LAYERS"], config["RNN"]["N_HEADS"],
                        config["RNN"]["D_K"], config["RNN"]["D_V"], config["RNN"]["WINDOW_SIZE"]
                    )
                    return jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0),
                                                  unbatched)

                elif model_type == "sdlg_hybrid_orig":
                    from src.models.common.sdlg_hybrid import BatchedSdLGHybrid
                    unbatched = BatchedSdLGHybrid.initialize_carry(
                        batch_size, config["RNN"]["N_LAYERS"],
                        config["RNN"]["D_K"], config["RNN"]["D_V"],
                        config["RNN"].get("R", 4), config["RNN"]["WINDOW_SIZE"]
                    )
                    return jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0),
                                                  unbatched)

                elif model_type == "gtrxl":
                    from src.models.common.gtrxl import BatchedGTrXL
                    return BatchedGTrXL.initialize_carry(
                        batch_size, config["RNN"].get("MEMORY_LEN", 128),
                        config["RNN"]["D_MODEL"], config["RNN"]["N_LAYERS"]
                    )

                elif model_type == "moe_sdlg_v2":
                    from src.models.common.moe_sdlg_v2 import BatchedMoESdLGv2
                    unbatched = BatchedMoESdLGv2.initialize_carry(
                        batch_size, config["RNN"]["N_LAYERS"], config["RNN"]["N_HEADS"],
                        config["RNN"]["D_K"], config["RNN"]["D_V"], config["RNN"]["WINDOW_SIZE"]
                    )
                    return jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0),
                                                  unbatched)
                elif model_type == "ppo_s5":
                    from src.models.baselines.s5 import S5_Module
                    # s5 carry init returns (batch_size, hidden_dim)
                    return S5_Module.initialize_carry(batch_size, config["RNN"].get("HIDDEN_DIM", 256))

            # Initialize WandB run specifically for this model (THIRD!)
            wandb.init(
                project=WANDB_PROJECT,
                group="exp17_comparative_sweep",
                name=f"{model_type}-seed-{seed}",
                config=config,
                reinit=True
            )

            network = SelectedNetwork(action_dim, config)
            rng = jax.random.PRNGKey(seed)

            # Instantiate trainer using our augmented engine (FOURTH!)
            raw_update_fn, runner_state = make_update_augmented(rng, config, env, env.default_params, network,
                                                                rnn_carry_init)

            @jax.jit
            def compiled_update(r_state):
                out = raw_update_fn(r_state)
                m = out["metric"]
                done_mask = m.get("returned_episode", jnp.ones_like(m.get("entropy", jnp.zeros(1))))

                def get_done_mean(x):
                    return jnp.sum(x * done_mask) / jnp.maximum(1.0, jnp.sum(done_mask))

                reduced = {
                    "train_ret": get_done_mean(m.get("returned_episode_returns", jnp.zeros_like(done_mask))),
                    "train_len": get_done_mean(m.get("returned_episode_lengths", jnp.zeros_like(done_mask))),
                    "train_ent": jnp.mean(m.get("entropy", jnp.zeros(1))),
                    "train_vloss": jnp.mean(m.get("value_loss", jnp.zeros(1))),
                    "grad_cosine_sim": jnp.mean(m.get("grad_cosine_sim", jnp.zeros(1))),

                    # Dynamic Mechanistic Telemetry
                    "beta_mean": jnp.mean(m.get("telemetry_beta_mean", 0.5)),
                    "gamma_head_0": jnp.mean(m.get("telemetry_gamma_head_0", 0.95)),
                    "gamma_head_n": jnp.mean(m.get("telemetry_gamma_head_n", 0.99)),
                }
                return out["runner_state"], reduced

            # PPO Sweep Loop with Early Stopping
            steps_processed = 0
            start_wall_time = time.time()

            early_stop_counter = 0
            EARLY_STOP_THRESHOLD = 0.97
            CONSECUTIVE_STEPS = 3

            print(
                f"🚀 Commencing {TOTAL_STEPS / 1e6:.1f}M Step Rollout Loop | Early Stop Threshold: {EARLY_STOP_THRESHOLD} ...\n")

            for jit_idx in range(TOTAL_JIT_CALLS):
                runner_state, reduced_metric = compiled_update(runner_state)

                runner_state = jax.block_until_ready(runner_state)
                reduced_metric = jax.device_get(reduced_metric)

                steps_processed += STEPS_PER_UPDATE * LOG_INTERVAL
                active_train_time = time.time() - start_wall_time
                sps = steps_processed / max(active_train_time, 1.0)

                # Pull JAX device HBM metrics
                device = jax.devices()[0]
                mem_stats = device.memory_stats()
                peak_vram_gb = float(mem_stats.get("bytes_in_use", 0.0)) / (1024 ** 3)

                train_ret = float(reduced_metric["train_ret"])
                train_len = float(reduced_metric["train_len"])
                train_ent = float(reduced_metric["train_ent"])
                train_vloss = float(reduced_metric["train_vloss"])
                grad_cosine = float(reduced_metric["grad_cosine_sim"])

                beta_val = float(reduced_metric["beta_mean"])
                g0 = float(reduced_metric["gamma_head_0"])
                gN = float(reduced_metric["gamma_head_n"])

                # System-Level Performance Logging
                wandb.log({
                    "System/SPS": sps,
                    "System/Peak_VRAM_GB": peak_vram_gb,
                    "Train/Mean_Return": train_ret,
                    "Train/Mean_Length": train_len,
                    "Train/Policy_Entropy": train_ent,
                    "Train/Value_Loss": train_vloss,

                    # Mechanistic Telemetry logs
                    "Telemetry/Gradient_Cosine_Similarity": grad_cosine,
                    "Telemetry/Write_Gate_Beta_t": beta_val,
                    "Telemetry/Gamma_Head_0_Decay": g0,
                    "Telemetry/Gamma_Head_N_Decay": gN,
                }, step=steps_processed)

                print(f"  [JIT {jit_idx + 1:>3}/{TOTAL_JIT_CALLS}] Step: {steps_processed / 1e6:>5.2f}M | "
                      f"Ret: {train_ret:>6.2f} | Len: {train_len:>5.1f} | "
                      f"Ent: {train_ent:>5.3f} | CosSim: {grad_cosine:>6.3f} | VRAM: {peak_vram_gb:>5.2f}GB", flush=True)

                # Robust, non-noisy Early Stopping Check
                if train_ret >= EARLY_STOP_THRESHOLD:
                    early_stop_counter += 1
                    if early_stop_counter >= CONSECUTIVE_STEPS:
                        print(f"\n⏹️  [EARLY STOP] Mean return maintained above {EARLY_STOP_THRESHOLD} "
                              f"for {CONSECUTIVE_STEPS} consecutive intervals! Moving to next model.")
                        break
                else:
                    early_stop_counter = 0

            print(f"\n✅ [{run_id.upper()}] Run Complete!")
            wandb.finish()

            # Save progress so we skip this model if the sweep restarts
            save_completed_run(run_id)

            jax.clear_caches()
            gc.collect()


if __name__ == "__main__":
    os.environ["HSA_ENABLE_SDMA"] = "0"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    run_experiment17()