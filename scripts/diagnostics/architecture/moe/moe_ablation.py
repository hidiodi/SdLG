# moe_ablation.py

from flax.training.train_state import TrainState
from typing import NamedTuple, Callable
import optax
import sys
import os
import jax
import jax.numpy as jnp
import flax.linen as nn
import flax
import numpy as np
import pandas as pd
import csv
import time
import functools
import matplotlib.pyplot as plt
import wandb
import signal

import gc
import orbax.checkpoint as ocp
from pathlib import Path
from typing import Any

# Ensure path includes the project root
os.environ["HSA_ENABLE_SDMA"] = "0"

current_file_path = os.path.abspath(__file__)
project_root = current_file_path
while not os.path.exists(os.path.join(project_root, "src")):
    project_root = os.path.dirname(project_root)
    if project_root == "/":
        break
if project_root not in sys.path:
    sys.path.append(project_root)
from scripts.diagnostics.core.utils import init_wandb
from src.envs.builder import build_purejaxrl_env
from src.envs.wrappers import BatchEnvWrapper, OptimisticResetVecEnvWrapper
from src.utils.config_schema import StaticEnvConfig, DynamicTrainingConfig
import distrax

from src.models.purejaxrl.wrappers import RepresentationNetwork, ActorCriticSdLGHybrid
from src.models.common.gtrxl import BatchedGTrXL, GatedTransformerXLLayer
from src.models.common.moe_sdlg import ActorCriticMoESdLG, BatchedMoESdLG
from src.models.common.moe_sdlg_v2 import ActorCriticMoESdLGv2, BatchedMoESdLGv2
from src.models.common.sdlg_hybrid import BatchedSdLGHybrid
from src.trainers.purejax_trainer import flush_carry_state
import src.trainers.purejax_trainer as trainer_module

from craftax.craftax.constants import Achievement

ACH_KEYS = [ach.name for ach in Achievement]


def plot_moe_routing_heatmap(routing_history, save_path="plots/moe_routing_heatmap.pdf", layer_idx=0):
    """
    Generates a 2D heatmap of expert routing probabilities over a single rollout.

    Args:
        routing_history (np.ndarray): Array of shape (timesteps, num_experts)
                                      or (num_layers, timesteps, num_experts).
        save_path (str): Filepath to save the resulting plot.
        layer_idx (int): Which layer to plot if routing_history contains multiple layers.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Handle both 2D (single layer) and 3D (multi-layer) inputs
    if routing_history.ndim == 3:
        data = routing_history[layer_idx]
    else:
        data = routing_history

    # data is (timesteps, num_experts)
    # Transpose to (num_experts, timesteps) so Y-axis = Experts, X-axis = Timesteps
    heatmap_data = data.T
    num_experts, timesteps = heatmap_data.shape

    fig, ax = plt.subplots(figsize=(12, 4))

    # aspect='auto' is crucial here since timesteps (e.g. 10000) >>> num_experts (e.g. 4)
    cax = ax.imshow(heatmap_data, aspect='auto', cmap='viridis', interpolation='nearest')

    ax.set_title(f"Expert Routing Probabilities Over Time (Layer {layer_idx})")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Expert ID")

    # Format Y-axis to show discrete expert IDs
    ax.set_yticks(np.arange(num_experts))
    ax.set_yticklabels([f"Expert {i}" for i in range(num_experts)])

    # Add a colorbar to represent probability magnitudes
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Routing Probability / Usage")

    plt.tight_layout()
    plt.savefig(save_path, format="pdf", dpi=300)
    plt.close(fig)

    print(f"  [PLOT] Saved MoE routing heatmap to {save_path}", flush=True)


# =====================================================================
# --- CRITICAL BUG FIXES & MONKEY PATCHES ---
# =====================================================================
def patched_gtrxl_call(self, inputs, pos_embedding, u, v, memory, mask=None):
    full_input = jnp.concatenate([memory, inputs], axis=0)
    x1 = self.layernorm1(full_input)
    query_input = x1[-inputs.shape[0]:]
    a1 = self.dropout_fn(self.attention(query_input, pos_embedding, x1, u, v, mask=mask))
    a1 = self.activation(a1)
    o1 = self.gate1(inputs, a1) if self.gating else inputs + a1
    x2 = self.layernorm2(o1)

    m2 = self.dropout_fn(nn.Dense(self.mlp.features)(jax.nn.gelu(nn.Dense(self.mlp.features)(x2)))) if not hasattr(
        self.mlp, 'layers') else self.dropout_fn(self.mlp(x2))

    o2 = self.gate2(o1, m2) if self.gating else o1 + m2
    return o2


GatedTransformerXLLayer.__call__ = patched_gtrxl_call


def patched_flush_carry_state(hstate, last_done):
    def flush_leaf(x):
        ld = jnp.asarray(last_done)

        # If ld has shape (T, B), take the first step's dones (representing the state transition)
        if ld.ndim == 2:
            ld = ld[0]

        ld = jnp.squeeze(ld)

        # Scalar fallback
        if ld.ndim == 0:
            return (x * (1.0 - ld)).astype(x.dtype)

        dims_to_add = max(0, x.ndim - ld.ndim)
        if dims_to_add > 0:
            broadcast_done = jnp.expand_dims(ld, axis=tuple(range(1, 1 + dims_to_add)))
        else:
            broadcast_done = ld

        return (x * (1.0 - broadcast_done)).astype(x.dtype)

    return jax.tree_util.tree_map(flush_leaf, hstate)


trainer_module.flush_carry_state = patched_flush_carry_state
flush_carry_state = patched_flush_carry_state


# =====================================================================
# --- DEADMAN'S SWITCH (ROCm GPU HANG DETECTOR) ---
# =====================================================================
def gpu_hang_handler(signum, frame):
    print("\n[FATAL ERROR] JAX/ROCm GPU Hang Detected (Timeout).", flush=True)
    print("Forcing Python exit to trigger bash watchdog...", flush=True)
    os._exit(43)


signal.signal(signal.SIGALRM, gpu_hang_handler)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    achievements: jnp.ndarray


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
    network = rnn_module()

    rng, _rng = jax.random.split(rng)
    _obs = jnp.zeros((1, config["NUM_ENVS"], *env.observation_space(env_params).shape))
    _dones = jnp.zeros((1, config["NUM_ENVS"]))
    init_x = (_obs, _dones)

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

            if config["RNN"]["RNN_TYPE"] in ["sdlg_hybrid", "moe_sdlg", "moe_sdlg_v2"]:
                hstate = flush_carry_state(hstate, last_done)

            # SELECT ACTION
            current_achievements = env_state.env_state.achievements  # shape: (B, K)

            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            if config["RNN"]["RNN_TYPE"] == "moe_sdlg_v2":
                action_logits, value, _, _, hstate = network.apply(train_state.params, hstate, ac_in)
                pi = distrax.Categorical(logits=action_logits)
                intrinsic_bonus = 0.0
            elif config["RNN"]["RNN_TYPE"] in ["sdlg_hybrid", "moe_sdlg"]:
                action_logits, value, bonus, hstate = network.apply(train_state.params, hstate, ac_in)
                pi = distrax.Categorical(logits=action_logits)
                intrinsic_bonus = bonus.squeeze(0)
            else:
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                intrinsic_bonus = 0.0

            action = pi.sample(seed=_rng)
            action = jnp.clip(action, 0, pi.logits.shape[-1] - 1)
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

            total_reward = reward + intrinsic_bonus

            transition = Transition(
                last_done, action, value, total_reward, log_prob, last_obs, info, current_achievements
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
        if config["RNN"]["RNN_TYPE"] == "moe_sdlg_v2":
            _, last_val, _, _, _ = network.apply(train_state.params, hstate, ac_in)
        elif config["RNN"]["RNN_TYPE"] in ["sdlg_hybrid", "moe_sdlg"]:
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
                    first_state = jax.tree_util.tree_map(lambda x: x[0], init_hstate)

                    moe_loss = 0.0
                    if config["RNN"]["RNN_TYPE"] == "moe_sdlg_v2":
                        action_logits, value, (actor_l_bal, critic_l_bal), _, _ = network.apply(
                            params, first_state, (traj_batch.obs, traj_batch.done)
                        )
                        moe_loss = 0.01 * (actor_l_bal + critic_l_bal)  # Load balancing coef
                    elif config["RNN"]["RNN_TYPE"] in ["sdlg_hybrid", "moe_sdlg"]:
                        action_logits, value, _, _ = network.apply(
                            params, first_state, (traj_batch.obs, traj_batch.done)
                        )
                    else:
                        _, pi, value = network.apply(
                            params, first_state, (traj_batch.obs, traj_batch.done)
                        )
                        action_logits = pi.logits

                    pi = distrax.Categorical(logits=action_logits)
                    log_prob = pi.log_prob(traj_batch.action)

                    value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                    ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])

                    def huber_loss(x, delta=10.0):
                        return jnp.where(jnp.abs(x) < delta, 0.5 * x ** 2, delta * (jnp.abs(x) - 0.5 * delta))

                    value_losses = huber_loss(value - targets)
                    value_losses_clipped = huber_loss(value_pred_clipped - targets)
                    value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()

                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
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
                            + moe_loss
                    )
                    return total_loss, (value_loss, loss_actor, entropy, moe_loss)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(
                    train_state.params, init_hstate, traj_batch, advantages, targets
                )

                flat_grads, _ = jax.tree_util.tree_flatten(grads)
                grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in flat_grads) + 1e-8)

                train_state = train_state.apply_gradients(grads=grads)
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
            batch = (init_hstate, traj_batch, advantages, targets)

            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=1), batch
            )

            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(
                    jnp.reshape(
                        x,
                        [x.shape[0], config["NUM_MINIBATCHES"], -1]
                        + list(x.shape[2:]),
                    ),
                    1,
                    0,
                ),
                shuffled_batch,
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

        init_hstate = jax.tree_util.tree_map(
            lambda x: x[None, :], initial_hstate)

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

        max_grad_norm_val = jnp.max(grad_norm_epoch)
        metric = traj_batch.info
        rng = update_state[-1]

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
        metric["moe_loss"] = loss_info[1][3]

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
# --- EXPERIMENT CONFIGURATION ---
# =====================================================================
TOTAL_STEPS = 50_000_000
TRAIN_LOG_INTERVAL_STEPS = 1_000_000
SESSION_STEP_LIMIT = 50_000_000

MODELS_TO_RUN = ["MoE-SdLG", "MoE-SdLGv2"]
TRAINING_SEEDS = [42]

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "MoE-SdLG-Classic-50M")
CHECKPOINT_BASE_DIR = os.path.abspath("scripts/diagnostics/exp16/checkpoints/exp16_moe_50M")
CSV_PATH = "scripts/diagnostics/exp16/exp16_moe_50M.csv"


# =====================================================================
# 1. BASELINE ARCHITECTURE INJECTION (GTrXL)
# =====================================================================
class ActorCriticGTrXL(nn.Module):
    rnn_module: Any
    action_dim: int
    config: dict

    @nn.compact
    def __call__(self, carry, x):
        obs, dones = x
        emb = RepresentationNetwork(
            d_model=self.config["RNN"]["D_MODEL"], name="repr_network"
        )(obs)

        gtrxl = BatchedGTrXL(
            head_dim=self.config["RNN"].get("D_HEAD", 32),
            embedding_dim=self.config["RNN"]["D_MODEL"],
            head_num=self.config["RNN"].get("N_HEADS", 8),
            mlp_num=2,
            layer_num=self.config["RNN"].get("N_LAYERS", 4),
            memory_len=self.config["RNN"].get("MEMORY_LEN", 128),
            dropout_ratio=0.0,
            gru_gating=True,
            use_embedding_layer=False,
        )

        new_carry, outs = gtrxl(carry, (emb, dones))

        action_logits = nn.Dense(
            self.action_dim, kernel_init=flax.linen.initializers.orthogonal(0.01), name="actor_head"
        )(outs)

        action_logits = 5.0 * jnp.tanh(jnp.nan_to_num(action_logits, nan=0.0, posinf=5.0, neginf=-5.0))

        critic_values = nn.Dense(
            1, kernel_init=flax.linen.initializers.orthogonal(1.0), name="critic_head"
        )(outs)
        critic_values = jnp.squeeze(critic_values, axis=-1)
        critic_values = jnp.nan_to_num(critic_values, nan=0.0, posinf=10.0, neginf=-10.0)

        pi = distrax.Categorical(logits=action_logits)
        return new_carry, pi, critic_values


# =====================================================================
# 2. VRAM-SAFE CHUNKED STOCHASTIC EVALUATOR (WITH SOW EXTRACTION)
# =====================================================================
def build_chunked_evaluator(env, env_params, network, config):
    eval_steps = 10000
    is_moe = config["RNN"]["RNN_TYPE"] == "moe_sdlg"

    num_moe_layers = config["RNN"].get("NUM_MOE_LAYERS", config["RNN"]["N_LAYERS"])
    num_experts = config["RNN"].get("NUM_EXPERTS", 4)

    @functools.partial(jax.jit, static_argnums=(3,))
    def eval_chunk(params, start_seed, base_rng, chunk_size):
        keys = jax.random.split(jax.random.PRNGKey(start_seed), chunk_size)
        reset_fn = jax.vmap(env.reset, in_axes=(0, None))
        step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, None))

        obs, env_state = reset_fn(keys, env_params)

        if config["RNN"]["RNN_TYPE"] in ["sdlg_hybrid", "moe_sdlg", "moe_sdlg_v2"]:
            if config["RNN"]["RNN_TYPE"] == "moe_sdlg_v2":
                unbatched = BatchedMoESdLGv2.initialize_carry(
                    batch_size=chunk_size, n_layers=config["RNN"]["N_LAYERS"],
                    n_heads=config["RNN"]["N_HEADS"],
                    d_k=config["RNN"]["D_K"], d_v=config["RNN"]["D_V"],
                    window_size=config["RNN"]["WINDOW_SIZE"]
                )
            elif config["RNN"]["RNN_TYPE"] == "moe_sdlg":
                unbatched = BatchedMoESdLG.initialize_carry(
                    batch_size=chunk_size, n_layers=config["RNN"]["N_LAYERS"],
                    d_k=config["RNN"]["D_K"], d_v=config["RNN"]["D_V"],
                    r=config["RNN"]["R"], window_size=config["RNN"]["WINDOW_SIZE"]
                )
            else:
                unbatched = BatchedSdLGHybrid.initialize_carry(
                    batch_size=chunk_size, n_layers=config["RNN"]["N_LAYERS"],
                    d_k=config["RNN"]["D_K"], d_v=config["RNN"]["D_V"],
                    r=config["RNN"]["R"], window_size=config["RNN"]["WINDOW_SIZE"]
                )
            hstate = jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), chunk_size, axis=0), unbatched)
        else:
            hstate = BatchedGTrXL.initialize_carry(
                chunk_size, memory_len=config["RNN"]["MEMORY_LEN"],
                embedding_dim=config["RNN"]["D_MODEL"], layer_num=config["RNN"]["N_LAYERS"]
            )

        dones = jnp.zeros((chunk_size,), dtype=bool)
        init_max_ach = {k: jnp.zeros((chunk_size,), dtype=jnp.int32) for k in ACH_KEYS}
        init_max_ret = jnp.full((chunk_size,), -jnp.inf)

        def step_fn_scan(carry, _):
            obs, env_state, hstate, dones, max_ach, max_ret, step_rng = carry
            ac_in = (obs[np.newaxis, :], dones[np.newaxis, :])

            step_rng, action_rng = jax.random.split(step_rng)

            if config["RNN"]["RNN_TYPE"] == "moe_sdlg_v2":
                # v2 naturally outputs usages as explicit returns, no sow needed!
                action_logits, _, _, (actor_exp_w, critic_exp_w), next_hstate = network.apply(params, hstate, ac_in)
                pi = distrax.Categorical(logits=action_logits.squeeze(0))
                action = pi.sample(seed=action_rng)
                step_usage = jnp.stack([actor_exp_w, critic_exp_w])

            elif config["RNN"]["RNN_TYPE"] == "moe_sdlg":
                out, mutables = network.apply(params, hstate, ac_in, mutable=['intermediates'])
                action_logits, _, _, next_hstate = out
                pi = distrax.Categorical(logits=action_logits.squeeze(0))
                action = pi.sample(seed=action_rng)

                if 'intermediates' in mutables:
                    leaves = jax.tree_util.tree_leaves(mutables['intermediates'])
                    step_usage_list = []
                    for leaf in leaves:
                        arr = leaf[0]
                        if arr.ndim == 3:
                            arr = arr.squeeze(0)
                        step_usage_list.append(arr.mean(axis=0))

                    try:
                        step_usage = jnp.stack(step_usage_list)  # (num_moe_layers, num_experts)
                    except ValueError:
                        step_usage = jnp.zeros((num_moe_layers, num_experts))
                else:
                    step_usage = jnp.zeros((num_moe_layers, num_experts))

            elif config["RNN"]["RNN_TYPE"] == "sdlg_hybrid":
                action_logits, _, _, next_hstate = network.apply(params, hstate, ac_in)
                pi = distrax.Categorical(logits=action_logits.squeeze(0))
                action = pi.sample(seed=action_rng)
                step_usage = jnp.zeros((1,))
            else:
                next_hstate, pi, _ = network.apply(params, hstate, ac_in)
                action = pi.sample(seed=action_rng).squeeze(0)
                step_usage = jnp.zeros((1,))

            action = jnp.clip(action, 0, pi.logits.shape[-1] - 1)

            rngs = jax.random.split(jax.random.PRNGKey(0), chunk_size)
            next_obs, next_env_state, reward, next_dones, info = step_fn(rngs, env_state, action, env_params)

            if config["RNN"]["RNN_TYPE"] in ["sdlg_hybrid", "moe_sdlg"]:
                next_hstate = flush_carry_state(next_hstate, next_dones)

            curr_ach = next_env_state.env_state.achievements
            new_max_ach = {
                name: jnp.maximum(max_ach[name], curr_ach[:, i].astype(jnp.int32))
                for i, name in enumerate(ACH_KEYS)
            }

            current_returns = jnp.where(next_dones,
                                        info.get("returned_episode_returns", next_env_state.episode_returns),
                                        next_env_state.episode_returns)
            new_max_ret = jnp.maximum(max_ret, current_returns)

            next_carry = (next_obs, next_env_state, next_hstate, next_dones, new_max_ach, new_max_ret, step_rng)

            return next_carry, step_usage

        rng_scan = jax.random.PRNGKey(start_seed)
        init_carry = (obs, env_state, hstate, dones, init_max_ach, init_max_ret, rng_scan)

        final_carry, exp_usage_history = jax.lax.stop_gradient(
            jax.lax.scan(step_fn_scan, init_carry, None, length=eval_steps)
        )
        _, _, _, _, final_max_ach, final_max_ret, _ = final_carry

        return final_max_ret, final_max_ach, exp_usage_history

    return eval_chunk


# =====================================================================
# 3. BENCHMARKING RUNNER
# =====================================================================
def run_experiment16():
    print("====================================================================")
    print("=== EXPERIMENT 16: MoE-SdLG LLM Modernization Benchmark ===")
    print("====================================================================")

    ENV_NAME = "Craftax-Classic-Symbolic-v1"
    NUM_ENVS = 256
    NUM_STEPS = 64

    STEPS_PER_UPDATE = NUM_ENVS * NUM_STEPS
    TOTAL_UPDATES = TOTAL_STEPS // STEPS_PER_UPDATE
    LOG_INTERVAL_UPDATES = TRAIN_LOG_INTERVAL_STEPS // STEPS_PER_UPDATE

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    os.makedirs(CHECKPOINT_BASE_DIR, exist_ok=True)

    any_resuming = False
    for m in MODELS_TO_RUN:
        for s in TRAINING_SEEDS:
            if os.path.exists(os.path.join(CHECKPOINT_BASE_DIR, f"{m}_seed_{s}")):
                any_resuming = True

    file_mode = "a" if any_resuming else "w"
    csv_file = open(CSV_PATH, file_mode, newline="")
    csv_writer = csv.writer(csv_file)

    if not any_resuming or os.path.getsize(CSV_PATH) == 0:
        header = [
                     "Architecture", "Seed", "Total_Steps", "Wall_Clock_Time", "SPS",
                     "Train_Return", "Train_Entropy", "Train_VLoss", "Eval_Return"
                 ] + [f"Train_{k}" for k in ACH_KEYS] + [f"Eval_{k}" for k in ACH_KEYS]
        csv_writer.writerow(header)
        csv_file.flush()

    for model_name in MODELS_TO_RUN:
        is_hybrid = (model_name == "SdLGHybrid")
        is_moe_v1 = (model_name == "MoE-SdLG")
        is_moe_v2 = (model_name == "MoE-SdLGv2")
        is_moe = is_moe_v1 or is_moe_v2

        alpha = 0.1
        window_size = 8
        ent_coef = 0.01

        if model_name == "PPO-GTrXL":
            trainer_module.ActorCriticRNN = ActorCriticGTrXL
        elif model_name == "SdLGHybrid":
            trainer_module.ActorCriticRNN = ActorCriticSdLGHybrid
        elif model_name == "MoE-SdLG":
            trainer_module.ActorCriticRNN = ActorCriticMoESdLG
        elif model_name == "MoE-SdLGv2":
            trainer_module.ActorCriticRNN = ActorCriticMoESdLGv2

        rnn_type = "moe_sdlg_v2" if is_moe_v2 else (
            "moe_sdlg" if is_moe_v1 else ("sdlg_hybrid" if is_hybrid else "gtrxl"))

        config = {
            "NUM_ENVS": NUM_ENVS, "NUM_STEPS": NUM_STEPS, "TOTAL_TIMESTEPS": TOTAL_STEPS,
            "UPDATE_EPOCHS": 4, "NUM_MINIBATCHES": 8, "GAMMA": 0.99, "GAE_LAMBDA": 0.95,
            "CLIP_EPS": 0.2, "ENT_COEF": ent_coef, "VF_COEF": 0.5, "MAX_GRAD_NORM": 0.5,
            "LR": 0.0002, "OPTIMISTIC_RESETS": False, "ANNEAL_LR": True,
            "LOG_INTERVAL": LOG_INTERVAL_UPDATES,
            "RNN": {
                "RNN_TYPE": rnn_type,
                "N_LAYERS": 4, "D_MODEL": 256, "D_HEAD": 32, "N_HEADS": 8,
                "MEMORY_LEN": 128, "D_K": 32, "D_V": 32, "D_FFC": 1024,
                "R": 4, "WINDOW_SIZE": window_size, "ALPHA": alpha, "ETA": 0.0,
                "USE_ROPE_PHASE": True, "USE_SOFTPLUS_DENOMINATOR": True, "USE_PER_RANK_NORM": True,
                # Explicit MoE specs for Evaluator Chunking
                "NUM_EXPERTS": 4,
                "NUM_MOE_LAYERS": 4
            }
        }

        static_env = StaticEnvConfig(env_name=ENV_NAME, num_envs=NUM_ENVS, num_steps=NUM_STEPS, max_episode_steps=10000,
                                     optimistic_resets=False)
        dynamic_train = DynamicTrainingConfig(
            total_timesteps=TOTAL_STEPS, learning_rate=0.0002, gamma=0.99, gae_lambda=0.95,
            clip_eps=0.2, ent_coef=ent_coef, vf_coef=0.5, max_grad_norm=0.5, update_epochs=4,
            minibatch_size=2048, log_interval=LOG_INTERVAL_UPDATES
        )

        env, env_params = build_purejaxrl_env(static_env, dynamic_train)
        from src.envs.wrappers import LogWrapper
        if not isinstance(env, LogWrapper):
            env = LogWrapper(env)
        action_dim = env.action_space(env_params).n

        if is_moe_v2:
            network = ActorCriticMoESdLGv2(action_dim=action_dim, config=config)

            def rnn_carry_init(batch_size):
                unbatched = BatchedMoESdLGv2.initialize_carry(
                    batch_size=batch_size, n_layers=4, n_heads=8, d_k=32, d_v=32, window_size=8)
                return jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0),
                                              unbatched)
        elif is_moe_v1:
            network = ActorCriticMoESdLG(action_dim=action_dim, config=config)

            def rnn_carry_init(batch_size):
                unbatched = BatchedMoESdLG.initialize_carry(
                    batch_size=batch_size, n_layers=4, d_k=32, d_v=32, r=4, window_size=8)
                return jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0),
                                              unbatched)
        elif is_hybrid:
            network = ActorCriticSdLGHybrid(action_dim=action_dim, config=config)

            def rnn_carry_init(batch_size):
                unbatched = BatchedSdLGHybrid.initialize_carry(
                    batch_size=batch_size, n_layers=4, d_k=32, d_v=32, r=4, window_size=8)
                return jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0),
                                              unbatched)
        else:
            network = ActorCriticGTrXL(rnn_module=None, action_dim=action_dim, config=config)

            def rnn_carry_init(batch_size):
                return BatchedGTrXL.initialize_carry(batch_size, memory_len=128, embedding_dim=256, layer_num=4)

        chunked_eval_fn = build_chunked_evaluator(env, env_params, network, config)

        for seed in TRAINING_SEEDS:
            jax.clear_caches()
            gc.collect()
            print(f"\n[{model_name}] Initializing Compilation & Training | SEED = {seed}", flush=True)

            ckpt_dir = Path(CHECKPOINT_BASE_DIR) / f"{model_name}_seed_{seed}"
            options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
            checkpointer = ocp.StandardCheckpointer()
            ckpt_manager = ocp.CheckpointManager(ckpt_dir.resolve(), checkpointer, options=options)

            rng = jax.random.PRNGKey(seed)
            raw_update_fn, runner_state = make_update(rng, config, env, env_params, lambda: network, rnn_carry_init)

            start_loop_idx = 0
            latest_step = ckpt_manager.latest_step()

            if latest_step is not None:
                print(f"  [CHECKPOINT] Found existing checkpoint! Resuming from Update Step {latest_step}...",
                      flush=True)
                restored = ckpt_manager.restore(latest_step,
                                                args=ocp.args.StandardRestore({"runner_state": runner_state}))
                runner_state = restored["runner_state"]
                start_loop_idx = latest_step // LOG_INTERVAL_UPDATES

                if start_loop_idx >= (TOTAL_UPDATES // LOG_INTERVAL_UPDATES):
                    print(f"  [CHECKPOINT] Run already completed. Skipping.")
                    continue
            else:
                print("  [CHECKPOINT] No existing checkpoint found. Starting from scratch.", flush=True)

            base_tags = os.environ.get("WANDB_TAGS", "").split(",") if os.environ.get("WANDB_TAGS") else []
            wandb_tags = base_tags + [model_name]

            run_id = f"exp16_{model_name}_seed_{seed}".replace(".", "_").lower()
            os.environ["WANDB_RUN_ID"] = run_id

            init_wandb(
                project_override=WANDB_PROJECT,
                group="exp16_moe_ablation",
                name=f"{model_name}-{seed}",
                config=config,
                model_name=model_name,
                resume="allow"
            )

            @functools.partial(jax.jit, donate_argnums=(0,))
            def compiled_update_and_reduce(r_state):
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
                    "max_grad_norm": jnp.max(m.get("max_gradient_norm", jnp.zeros(1))),
                    "nan_detected": jnp.any(m.get("parameter_nan_detected", jnp.zeros(1, dtype=bool))),
                }

                native_achievements = out["runner_state"][1].env_state.achievements
                for i, name in enumerate(ACH_KEYS):
                    reduced[f"train_{name}"] = jnp.mean(native_achievements[:, i].astype(jnp.float32))

                return out["runner_state"], reduced

            NUM_LOG_LOOPS = TOTAL_UPDATES // LOG_INTERVAL_UPDATES
            steps_processed = start_loop_idx * LOG_INTERVAL_UPDATES * STEPS_PER_UPDATE
            start_wall_time = time.time()
            cumulative_eval_time = 0.0

            for loop_idx in range(start_loop_idx, NUM_LOG_LOOPS):
                timeout = 1800 if loop_idx == start_loop_idx else 300
                signal.alarm(timeout)

                try:
                    runner_state, reduced_metric = compiled_update_and_reduce(runner_state)
                    runner_state = jax.block_until_ready(runner_state)
                    reduced_metric = jax.device_get(reduced_metric)
                except Exception as e:
                    print(f"\n[FATAL ERROR] JAX Execution Failed during training loop: {e}", flush=True)
                    os._exit(43)
                finally:
                    signal.alarm(0)

                if bool(reduced_metric["nan_detected"]):
                    print(f"\n[FATAL ERROR] NaN detected in model weights at step {steps_processed / 1e6:.1f}M!")
                    print(f"Last recorded Max Grad Norm: {float(reduced_metric['max_grad_norm']):.2f}")
                    print("Halting gracefully and sending Exit Code 42 to the Immortal Watchdog.")
                    wandb.log({"System/Crash": 1}, step=steps_processed)
                    wandb.finish()
                    csv_file.close()
                    sys.exit(42)

                steps_processed += LOG_INTERVAL_UPDATES * STEPS_PER_UPDATE
                current_update_idx = (loop_idx + 1) * LOG_INTERVAL_UPDATES

                train_ret = float(reduced_metric["train_ret"])
                train_len = float(reduced_metric["train_len"])
                train_ent = float(reduced_metric["train_ent"])
                train_vloss = float(reduced_metric["train_vloss"])
                max_grad = float(reduced_metric["max_grad_norm"])

                train_total_achv = sum(float(reduced_metric[f"train_{k}"]) for k in ACH_KEYS)

                active_train_time = time.time() - start_wall_time - cumulative_eval_time

                session_steps_processed = (loop_idx - start_loop_idx + 1) * LOG_INTERVAL_UPDATES * STEPS_PER_UPDATE
                sps = session_steps_processed / max(active_train_time, 1.0)

                wandb_log_dict = {
                    "System/SPS": sps,
                    "System/Wall_Clock_Time": active_train_time,
                    "Train/Mean_Return": train_ret,
                    "Train/Normalized_Return_Pct": (train_ret / 226.0) * 100.0,
                    "Train/Mean_Episode_Length": train_len,
                    "Train/Policy_Entropy": train_ent,
                    "Train/Value_Loss": train_vloss,
                    "Train/MoE_Loss": float(reduced_metric["train_moeloss"]),
                    "Train/Max_Grad_Norm": max_grad,
                    "Train/Total_Achievements": train_total_achv,
                }

                for k in ACH_KEYS:
                    wandb_log_dict[f"Train_Achv/{k}"] = float(reduced_metric[f"train_{k}"])

                is_final_global_loop = (loop_idx == NUM_LOG_LOOPS - 1)
                is_session_limit_reached = (session_steps_processed >= SESSION_STEP_LIMIT)

                if is_final_global_loop or is_session_limit_reached:
                    eval_start = time.time()

                    ckpt_manager.save(current_update_idx, args=ocp.args.StandardSave({"runner_state": runner_state}))
                    ckpt_manager.wait_until_finished()
                    print(f"  [CHECKPOINT] Saved state at Step {steps_processed / 1e6:.1f}M", flush=True)

                    eval_returns = []
                    eval_ach_chunks = {k: [] for k in ACH_KEYS}
                    eval_usages = []

                    print(f"  [EVAL] Triggering XLA Compilation for 10k-step horizon (this may take 5-15 mins)...",
                          flush=True)

                    for chunk in range(8):
                        print(f"    -> Executing Eval Chunk {chunk + 1}/8...", flush=True)
                        start_test_seed = 10000 + (chunk * 128)

                        ret_c, max_ach_c, usage_c = jax.device_get(
                            chunked_eval_fn(runner_state[0].params, start_test_seed, rng, 128)
                        )

                        eval_returns.append(ret_c)
                        for k in ACH_KEYS:
                            eval_ach_chunks[k].append(max_ach_c[k])
                        eval_usages.append(usage_c)

                    mean_test_ret = float(np.mean(np.concatenate(eval_returns)))
                    mean_eval_ach = {k: float(np.mean(np.concatenate(eval_ach_chunks[k]))) for k in ACH_KEYS}

                    # eval_usages is a list of 8 arrays of shape (10000, num_moe_layers, num_experts)
                    # Stacking makes it (8, 10000, num_layers, num_experts)
                    # We average over chunks (axis 0) and timesteps (axis 1)
                    mean_eval_usage = np.mean(np.stack(eval_usages), axis=(0, 1))

                    eval_end = time.time()
                    cumulative_eval_time += (eval_end - eval_start)

                    wandb_log_dict["Eval/Mean_Return"] = mean_test_ret
                    for k in ACH_KEYS:
                        wandb_log_dict[f"Eval_Achv/{k}"] = mean_eval_ach[k]

                    if is_moe:
                        for l in range(mean_eval_usage.shape[0]):
                            for e in range(mean_eval_usage.shape[1]):
                                layer_name = ["Actor", "Critic"][l] if is_moe_v2 else f"Layer_{l}"
                                wandb_log_dict[f"Eval_ExpertUsage/{layer_name}_Expert_{e}"] = float(
                                    mean_eval_usage[l, e])

                        history_for_plot = np.transpose(eval_usages[0], (1, 0, 2))

                        plot_moe_routing_heatmap(
                            history_for_plot,
                            save_path=f"plots/{model_name}_heatmap_seed_{seed}_step_{steps_processed}.pdf",
                            layer_idx=0
                        )

                    total_eval_achv = sum(mean_eval_ach.values())
                    print(
                        f"  [EVAL DONE] Step: {steps_processed / 1e6:>5.1f}M | Ret: {mean_test_ret:>5.1f} | Achv: {total_eval_achv:>4.1f} | SPS: {int(sps):>5}",
                        flush=True)

                    row = [
                        model_name, seed, steps_processed, active_train_time, sps,
                        train_ret, train_ent, train_vloss, mean_test_ret
                    ]
                    row.extend([float(reduced_metric[f"train_{k}"]) for k in ACH_KEYS])
                    row.extend([mean_eval_ach[k] for k in ACH_KEYS])

                    csv_writer.writerow(row)
                    csv_file.flush()

                    del eval_returns, eval_ach_chunks, eval_usages, ret_c, max_ach_c, usage_c

                    wandb.log(wandb_log_dict, step=steps_processed)
                    del wandb_log_dict, reduced_metric

                    print("  [CLEANUP] Syncing WandB to cloud...", flush=True)
                    wandb.finish(quiet=True)
                    print("  [CLEANUP] WandB synced.", flush=True)

                    if is_final_global_loop:
                        print(f"✅ {model_name} (Seed {seed}) fully completed 50M steps!")
                        print("\n[WATCHDOG] Run completed. Executing RAM Flush (Exit 43)...")
                    else:
                        print(f"\n[WATCHDOG] 50M Session Limit Reached. Executing Radical RAM Flush (Exit 43)...")

                    csv_file.close()

                    os._exit(43)

                else:
                    print(
                        f"  [TRAIN] Step: {steps_processed / 1e6:>5.1f}M | Ret: {train_ret:>5.1f} | Achv: {train_total_achv:>4.1f} | Len: {train_len:>4.0f} | VLoss: {train_vloss:>5.3f} | MaxGrad: {max_grad:>5.2f} | SPS: {int(sps):>5}",
                        flush=True)

                    if loop_idx > start_loop_idx and loop_idx % 100 == 0:
                        ckpt_manager.save(current_update_idx,
                                          args=ocp.args.StandardSave({"runner_state": runner_state}))
                        print(f"  [CHECKPOINT] Saved intermediate state at Step {steps_processed / 1e6:.1f}M",
                              flush=True)

                    wandb.log(wandb_log_dict, step=steps_processed)
                    del wandb_log_dict, reduced_metric

            try:
                ckpt_manager.wait_until_finished()
                wandb.finish()
                del runner_state, compiled_update_and_reduce, raw_update_fn
                jax.clear_caches()
                gc.collect()
            except Exception:
                pass
    csv_file.close()


if __name__ == "__main__":
    run_experiment16()