# scripts/diagnostics/architecture/swa_ablation/swa_ablation.py

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
import gc
import argparse
import hashlib
import json
import subprocess
from typing import Any, NamedTuple
from flax.linen.initializers import constant, orthogonal
from flax import struct

current_file_path = os.path.abspath(__file__)
project_root = current_file_path
while not os.path.exists(os.path.join(project_root, "src")):
    project_root = os.path.dirname(project_root)
    if project_root == "/":
        break
if project_root not in sys.path:
    sys.path.append(project_root)

import wandb
from scripts.diagnostics.core.utils import init_wandb
import distrax
import xminigrid

import src.models.purejaxrl.wrappers as wrappers
from src.models.common.sdlg_hybrid import (
    BatchedSdLGHybrid,
    GatedSlidingWindowAttention,
    GTrXLGRUGate,
    GatedAttentionGLALayer
)
from src.trainers.purejax_trainer import make_update

# =====================================================================
# --- EXPERIMENT 10 CONFIGURATION ---
# =====================================================================
TOTAL_STEPS = 10_000_000
TRAIN_LOG_INTERVAL_STEPS = 250_000

EARLY_STOP_THRESHOLD = 0.98
CONSECUTIVE_STEPS = 3

MODELS_TO_RUN = ["SdLG-SWA", "SdLG-None"]
TRAINING_SEEDS = [42, 54, 99, 123, 456]  # Back to 5 seeds for statistical rigor

# Clean, targeted environment where Horizon >= SWA Window (16)
TARGET_ENVS = ["MiniGrid-MemoryS16"]


# =====================================================================
# 1. XMINIGRID WRAPPER
# =====================================================================
@struct.dataclass
class EnvStateAndMetrics:
    timestep: Any
    episode_return: jnp.ndarray
    episode_length: jnp.ndarray


class PureJaxRLXMiniGridWrapper:
    def __init__(self, env_id):
        self.env, self.default_params = xminigrid.make(env_id)

    def action_space(self, params=None):
        p = params if params is not None else self.default_params

        class Space:
            def __init__(self, n): self.n = n

        return Space(self.env.num_actions(p))

    def observation_space(self, params=None):
        p = params if params is not None else self.default_params

        class Space:
            def __init__(self, shape): self.shape = shape

        return Space(self.env.observation_shape(p))

    def reset(self, key, params=None):
        p = params if params is not None else self.default_params
        timestep = self.env.reset(p, key)
        state = EnvStateAndMetrics(
            timestep=timestep, episode_return=jnp.array(0.0, dtype=jnp.float32),
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
        info = {"returned_episode_returns": new_ep_return, "returned_episode_lengths": new_ep_length,
                "returned_episode": done}
        reset_timestep = self.env.reset(p, key)
        next_timestep = jax.tree_util.tree_map(lambda res, cur: jnp.where(done, res, cur), reset_timestep, timestep)
        next_timestep = next_timestep.replace(reward=timestep.reward, step_type=timestep.step_type)
        next_state = EnvStateAndMetrics(
            timestep=next_timestep, episode_return=jnp.where(done, 0.0, new_ep_return),
            episode_length=jnp.where(done, 0, new_ep_length)
        )
        return next_timestep.observation, next_state, reward, done, info


# =====================================================================
# 2. EXTENDED PYTREE CARRY STATE
# =====================================================================
class SdLGHybridCarryAblation(NamedTuple):
    tilde_k: jnp.ndarray
    tilde_v: jnp.ndarray
    theta: jnp.ndarray
    swa_k: jnp.ndarray
    swa_v: jnp.ndarray
    step_count: jnp.ndarray


# =====================================================================
# 3. DYNAMIC ROUTING BLOCKS
# =====================================================================
class SdLGHybridBlockAblation(nn.Module):
    d_model: int;
    d_k: int;
    d_v: int;
    d_ffc: int;
    r: int;
    window_size: int;
    layer_idx: int
    eta: float = 0.05;
    gru_bias: float = 2.0;
    use_rope_phase: bool = True
    use_softplus_denominator: bool = True;
    use_per_rank_norm: bool = True
    local_mode: str = "swa"

    @nn.compact
    def __call__(self, inputs, terminations, carry, deterministic=True):
        T, _ = inputs.shape
        norm_inputs = nn.LayerNorm(name='ln1')(inputs)
        intrinsic_bonus = jnp.zeros_like(inputs[..., 0])

        if self.layer_idx % 2 == 0:
            if self.local_mode == "swa":
                attn_out, new_swa_k, new_swa_v = GatedSlidingWindowAttention(
                    d_model=self.d_model, d_k=self.d_k, d_v=self.d_v, window_size=self.window_size, name="swa"
                )(norm_inputs, terminations, carry.swa_k, carry.swa_v, carry.step_count, deterministic=deterministic)
            else:
                attn_out = jnp.zeros_like(inputs)
                new_swa_k, new_swa_v = carry.swa_k, carry.swa_v

            if T > 1:
                _, step_counts = jax.lax.scan(lambda c, t: ((c + 1.0) * (1.0 - t),) * 2, carry.step_count, terminations)
                new_step_count = step_counts[-1]
            else:
                new_step_count = (carry.step_count + 1.0) * (1.0 - terminations[0])

            new_carry = SdLGHybridCarryAblation(
                tilde_k=carry.tilde_k, tilde_v=carry.tilde_v, theta=carry.theta,
                swa_k=new_swa_k, swa_v=new_swa_v, step_count=new_step_count
            )
        else:
            attn_out, intrinsic_bonus, (new_tilde_k, new_tilde_v, new_theta), new_step_count = GatedAttentionGLALayer(
                d_model=self.d_model, d_k=self.d_k, d_v=self.d_v, r=self.r, eta=self.eta,
                use_rope_phase=self.use_rope_phase, use_softplus_denominator=self.use_softplus_denominator,
                use_per_rank_norm=self.use_per_rank_norm, name="gla"
            )(norm_inputs, terminations, (carry.tilde_k, carry.tilde_v, carry.theta), carry.step_count)
            new_carry = SdLGHybridCarryAblation(
                tilde_k=new_tilde_k, tilde_v=new_tilde_v, theta=new_theta,
                swa_k=carry.swa_k, swa_v=carry.swa_v, step_count=new_step_count
            )

        gating_out = GTrXLGRUGate(self.d_model, self.gru_bias, name='gru1')(inputs, attn_out)
        norm_gating_out = nn.LayerNorm(name='ln2')(gating_out)
        ffn_out = nn.Sequential([
            nn.Dense(self.d_ffc, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)), nn.gelu,
            nn.Dense(self.d_model, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
        ], name='ffn')(norm_gating_out)
        out = GTrXLGRUGate(self.d_model, self.gru_bias, name='gru2')(gating_out, ffn_out)
        return out, intrinsic_bonus, new_carry


class BatchedSdLGHybridAblation(nn.Module):
    n_layers: int;
    d_model: int;
    d_k: int;
    d_v: int;
    d_ffc: int;
    r: int;
    window_size: int
    alpha: float = 0.1;
    eta: float = 0.05;
    return_split: bool = True;
    use_rope_phase: bool = True
    use_softplus_denominator: bool = True;
    use_per_rank_norm: bool = True;
    local_mode: str = "swa"

    @nn.compact
    def __call__(self, inputs, terminations, carry, deterministic=True):
        if inputs.shape[0] == 1:
            carry = jax.tree_util.tree_map(lambda x: x * (1.0 - terminations[0]), carry)

        x = nn.relu(nn.Dense(self.d_model, kernel_init=orthogonal(jnp.sqrt(2)))(inputs)) if inputs.shape[
                                                                                                -1] != self.d_model else inputs

        new_carry = []
        bonuses = []
        for layer_idx in range(self.n_layers):
            x, intrinsic_bonus, updated_layer_carry = SdLGHybridBlockAblation(
                d_model=self.d_model, d_k=self.d_k, d_v=self.d_v, d_ffc=self.d_ffc, r=self.r,
                window_size=self.window_size,
                layer_idx=layer_idx, eta=self.eta, use_rope_phase=self.use_rope_phase,
                use_softplus_denominator=self.use_softplus_denominator, use_per_rank_norm=self.use_per_rank_norm,
                local_mode=self.local_mode, name=f'layer_{layer_idx}'
            )(x, terminations, carry[layer_idx], deterministic=deterministic)
            new_carry.append(updated_layer_carry)
            bonuses.append(intrinsic_bonus)

        total_intrinsic_bonus = jnp.mean(jnp.stack(bonuses, axis=0), axis=0)
        if self.return_split:
            X_actor = jax.lax.stop_gradient(x) * (1.0 - self.alpha) + x * self.alpha
            return ((X_actor, x), total_intrinsic_bonus), tuple(new_carry)
        return (x, total_intrinsic_bonus), tuple(new_carry)


class ActorCriticSdLGHybridAblation(nn.Module):
    action_dim: int;
    config: dict;
    local_mode: str = "swa"

    @nn.compact
    def __call__(self, carry, x):
        obs, dones = x

        # Normalize & stabilize representations to prevent VLoss explosion
        obs_float = obs.astype(jnp.float32) / 10.0
        x_flat = obs_float.reshape(obs_float.shape[0], obs_float.shape[1], -1)

        emb = nn.relu(nn.Dense(self.config["RNN"]["D_MODEL"], kernel_init=orthogonal(jnp.sqrt(2)))(x_flat))
        emb = nn.relu(nn.Dense(self.config["RNN"]["D_MODEL"], kernel_init=orthogonal(jnp.sqrt(2)))(emb))
        emb = nn.LayerNorm()(emb)  # Hard bound representation drift

        emb_trans, dones_trans = jnp.transpose(emb, (1, 0, 2)), jnp.transpose(dones, (1, 0))

        VmappedSdLG = nn.vmap(BatchedSdLGHybridAblation, in_axes=(0, 0, 0, None), out_axes=0,
                              variable_axes={'params': None}, split_rngs={'params': False})
        ((X_actor_trans, X_critic_trans), intrinsic_bonus_trans), new_carry = VmappedSdLG(
            n_layers=self.config["RNN"]["N_LAYERS"], d_model=self.config["RNN"]["D_MODEL"],
            d_k=self.config["RNN"]["D_K"], d_v=self.config["RNN"]["D_V"], d_ffc=self.config["RNN"]["D_FFC"],
            r=self.config["RNN"]["R"], window_size=self.config["RNN"]["WINDOW_SIZE"], alpha=self.config["RNN"]["ALPHA"],
            return_split=True, local_mode=self.local_mode, name="sdlg_hybrid"
        )(emb_trans, dones_trans, carry, True)

        X_actor, X_critic = jnp.transpose(X_actor_trans, (1, 0, 2)), jnp.transpose(X_critic_trans, (1, 0, 2))

        action_logits = 5.0 * jnp.tanh(
            jnp.nan_to_num(nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(X_actor), nan=0.0, posinf=5.0,
                           neginf=-5.0))
        critic_values = jnp.clip(jnp.squeeze(nn.Dense(1, kernel_init=orthogonal(1.0))(X_critic), axis=-1), -100.0,
                                 100.0)

        return action_logits, critic_values, jnp.transpose(intrinsic_bonus_trans, (1, 0)), new_carry


# =====================================================================
# 4. CHUNKED EVALUATOR
# =====================================================================
def build_chunked_evaluator(env, env_params, network, config):
    eval_steps = 1000

    @functools.partial(jax.jit, static_argnums=(3,))
    def eval_chunk(params, start_seed, base_rng, chunk_size):
        keys = jax.random.split(jax.random.PRNGKey(start_seed), chunk_size)
        reset_fn = jax.vmap(env.reset, in_axes=(0, None))
        step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, None))

        obs, env_state = reset_fn(keys, env_params)
        unbatched = BatchedSdLGHybrid.initialize_carry(batch_size=chunk_size, n_layers=config["RNN"]["N_LAYERS"],
                                                       d_k=config["RNN"]["D_K"], d_v=config["RNN"]["D_V"],
                                                       r=config["RNN"]["R"], window_size=config["RNN"]["WINDOW_SIZE"])

        def convert_carry(c):
            return SdLGHybridCarryAblation(
                tilde_k=c.tilde_k, tilde_v=c.tilde_v, theta=c.theta, swa_k=c.swa_k, swa_v=c.swa_v,
                step_count=c.step_count
            )

        new_unbatched = tuple(convert_carry(c) for c in unbatched)
        hstate = jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), chunk_size, axis=0), new_unbatched)
        dones, init_max_ret = jnp.zeros((chunk_size,), dtype=bool), jnp.full((chunk_size,), -jnp.inf)

        def step_fn_scan(carry, _):
            obs, env_state, hstate, dones, max_ret, step_rng = carry
            step_rng, action_rng = jax.random.split(step_rng)
            action_logits, _, _, next_hstate = network.apply(params, hstate, (obs[np.newaxis, :], dones[np.newaxis, :]))
            action = distrax.Categorical(logits=action_logits.squeeze(0)).sample(seed=action_rng)
            next_obs, next_env_state, reward, next_dones, info = step_fn(
                jax.random.split(jax.random.PRNGKey(0), chunk_size), env_state, action, env_params)

            def flush_leaf(x):
                dims = x.ndim - next_dones.ndim
                return (x * (1.0 - jnp.expand_dims(next_dones, axis=tuple(range(1, 1 + dims))))).astype(x.dtype)

            next_hstate = jax.tree_util.tree_map(flush_leaf, next_hstate)

            current_returns = jnp.where(next_dones, info.get("returned_episode_returns", reward), 0.0)
            new_max_ret = jnp.maximum(max_ret, current_returns)

            return (next_obs, next_env_state, next_hstate, next_dones, new_max_ret, step_rng), None

        final_carry = jax.lax.stop_gradient(
            jax.lax.scan(step_fn_scan, (obs, env_state, hstate, dones, init_max_ret, jax.random.PRNGKey(start_seed)),
                         None, length=eval_steps)[0])
        return final_carry[4]

    return eval_chunk


# =====================================================================
# 5. BENCHMARK RUNNER
# =====================================================================
def _git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()


def _source_sha256():
    with open(__file__, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_experiment10(seeds=None, model_names=None, csv_path=None, overwrite=False, fixed_endpoint=False,
                     manifest_path=None):
    seeds = list(TRAINING_SEEDS if seeds is None else seeds)
    model_names = list(MODELS_TO_RUN if model_names is None else model_names)
    csv_path = csv_path or "scripts/diagnostics/architecture/swa_ablation/swa_ablation.csv"
    print("====================================================================")
    print("=== EXP 10: SWA SPATIAL FILTER SHOWDOWN (XMINIGRID) ===")
    print(f"=== Targets: {TARGET_ENVS}")
    print(f"=== Seeds: {seeds} | Conditions: {model_names} | Fixed endpoint: {fixed_endpoint}")
    print("====================================================================")

    NUM_ENVS = 256
    NUM_STEPS = 64
    STEPS_PER_UPDATE = NUM_ENVS * NUM_STEPS
    TOTAL_UPDATES = TOTAL_STEPS // STEPS_PER_UPDATE
    LOG_INTERVAL_UPDATES = TRAIN_LOG_INTERVAL_STEPS // STEPS_PER_UPDATE

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if manifest_path:
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump({
                "study_id": "MINIGRID-LOCALITY-EXPANSION-2026",
                "git_commit": _git_commit(),
                "git_worktree_dirty": bool(subprocess.check_output(
                    ["git", "status", "--porcelain", "--untracked-files=no"], cwd=project_root, text=True).strip()),
                "runner_sha256": _source_sha256(),
                "argv": sys.argv,
                "runner": os.path.relpath(__file__, project_root),
                "output_csv": os.path.relpath(os.path.abspath(csv_path), project_root),
                "seeds": seeds,
                "conditions": model_names,
                "fixed_endpoint": fixed_endpoint,
                "total_timesteps": TOTAL_STEPS,
                "num_envs": NUM_ENVS,
                "num_steps": NUM_STEPS,
                "endpoint_semantics": "fixed" if fixed_endpoint else "legacy early-stop enabled",
            }, f, indent=2)

    write_header = overwrite or not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "w" if overwrite else "a", newline="") as f:
        if write_header:
            csv.writer(f).writerow(
                ["Environment", "Condition", "Seed", "Total_Steps", "Wall_Clock_Time", "SPS", "Train_Return",
                 "Train_Entropy", "Train_VLoss_Var", "Eval_Return"])

    import src.trainers.purejax_trainer
    def patched_flush_carry_state(hstate, last_done):
        def flush_leaf(x): return (
                    x * (1.0 - jnp.expand_dims(last_done, axis=tuple(range(1, 1 + x.ndim - last_done.ndim))))).astype(
            x.dtype)

        return jax.tree_util.tree_map(flush_leaf, hstate)

    src.trainers.purejax_trainer.flush_carry_state = patched_flush_carry_state

    for env_name in TARGET_ENVS:
        for model_name in model_names:
            m_mode = "swa" if "SWA" in model_name else "none"
            wrappers.ActorCriticSdLGHybrid = functools.partial(ActorCriticSdLGHybridAblation, local_mode=m_mode)

            config = {
                "NUM_ENVS": NUM_ENVS, "NUM_STEPS": NUM_STEPS, "TOTAL_TIMESTEPS": TOTAL_STEPS,
                "UPDATE_EPOCHS": 4, "NUM_MINIBATCHES": 8, "GAMMA": 0.99, "GAE_LAMBDA": 0.95,
                "CLIP_EPS": 0.2, "ENT_COEF": 0.01, "VF_COEF": 0.5, "MAX_GRAD_NORM": 0.5, "LR": 0.0003,
                "OPTIMISTIC_RESETS": False, "ANNEAL_LR": True, "LOG_INTERVAL": LOG_INTERVAL_UPDATES,
                "RNN": {"RNN_TYPE": "sdlg_hybrid", "N_LAYERS": 3, "D_MODEL": 128, "N_HEADS": 4, "D_K": 32, "D_V": 32,
                        "D_FFC": 512, "R": 4, "WINDOW_SIZE": 16, "ALPHA": 0.1, "USE_ROPE_PHASE": True,
                        "USE_SOFTPLUS_DENOMINATOR": True, "USE_PER_RANK_NORM": True}
            }

            env = PureJaxRLXMiniGridWrapper(env_name)
            network = wrappers.ActorCriticSdLGHybrid(action_dim=env.action_space().n, config=config)

            def rnn_carry_init(batch_size):
                unbatched = BatchedSdLGHybrid.initialize_carry(batch_size=batch_size, n_layers=3, d_k=32, d_v=32, r=4,
                                                               window_size=16)

                def convert_carry(c): return SdLGHybridCarryAblation(tilde_k=c.tilde_k, tilde_v=c.tilde_v,
                                                                     theta=c.theta, swa_k=c.swa_k, swa_v=c.swa_v,
                                                                     step_count=c.step_count)

                return jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0),
                                              tuple(convert_carry(c) for c in unbatched))

            chunked_eval_fn = build_chunked_evaluator(env, env.default_params, network, config)

            for seed in seeds:
                jax.clear_caches()
                gc.collect()
                print(f"\n[{model_name} | {env_name}] SEED = {seed}", flush=True)

                run_config = dict(config)
                run_config.update({"RUN_SEED": seed, "STUDY_ID": "MINIGRID-LOCALITY-EXPANSION-2026",
                                   "CONDITION": model_name, "FIXED_ENDPOINT": fixed_endpoint})
                init_wandb(project=os.environ.get("WANDB_PROJECT", "sdlgHybrid"),
                           tags=["swa_ablation", env_name, "empirical_strengthening_2026"],
                           group=f"revision-2026-{model_name}-{env_name}",
                           name=f"revision-2026-{model_name}-{env_name}-Seed-{seed}", config=run_config)

                rng = jax.random.PRNGKey(seed)
                raw_update_fn, runner_state = make_update(rng, config, env, env.default_params, lambda: network,
                                                          rnn_carry_init)

                @functools.partial(jax.jit, donate_argnums=(0,))
                def compiled_update_and_reduce(r_state):
                    out = raw_update_fn(r_state)
                    m = out["metric"]
                    d_mask = m.get("returned_episode", jnp.ones_like(m.get("entropy", jnp.zeros(1))))

                    def mean_d(x): return jnp.sum(x * d_mask) / jnp.maximum(1.0, jnp.sum(d_mask))

                    return out["runner_state"], {
                        "train_ret": mean_d(m.get("returned_episode_returns", jnp.zeros_like(d_mask))),
                        "train_ent": jnp.mean(m.get("entropy", jnp.zeros(1))),
                        "train_vloss_var": jnp.var(m.get("value_loss", jnp.zeros(1)))}

                steps_processed = 0
                start_wall_time = time.time()
                cumulative_eval_time = 0.0
                early_stop_counter = 0

                for loop_idx in range(TOTAL_UPDATES // LOG_INTERVAL_UPDATES):
                    runner_state, metrics = compiled_update_and_reduce(runner_state)
                    runner_state = jax.block_until_ready(runner_state)
                    metrics = jax.device_get(metrics)

                    steps_processed += LOG_INTERVAL_UPDATES * STEPS_PER_UPDATE
                    active_train_time = time.time() - start_wall_time - cumulative_eval_time
                    sps = steps_processed / max(active_train_time, 1.0)
                    train_ret, train_ent, train_vloss_var = float(metrics["train_ret"]), float(
                        metrics["train_ent"]), float(metrics["train_vloss_var"])

                    log_dict = {"System/SPS": sps, "Train/Mean_Return": train_ret, "Train/Policy_Entropy": train_ent,
                                "Train/Value_Loss_Var": train_vloss_var}

                    if train_ret >= EARLY_STOP_THRESHOLD:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0

                    is_final_loop = (loop_idx == (TOTAL_UPDATES // LOG_INTERVAL_UPDATES) - 1)
                    is_early_stop = (not fixed_endpoint) and (early_stop_counter >= CONSECUTIVE_STEPS)

                    if is_final_loop or is_early_stop:
                        eval_start = time.time()
                        eval_returns = []
                        for chunk in range(4):
                            ret_c = jax.device_get(
                                chunked_eval_fn(runner_state[0].params, 10000 + (chunk * 256), rng, 256))
                            eval_returns.append(ret_c)
                        mean_test_ret = float(np.mean(np.concatenate(eval_returns)))
                        cumulative_eval_time += (time.time() - eval_start)

                        log_dict["Eval/Mean_Return"] = mean_test_ret

                        if is_early_stop:
                            print(
                                f"  [EARLY STOP] Solved {env_name}! Step: {steps_processed / 1e6:>5.1f}M | Eval Ret: {mean_test_ret:>5.1f}",
                                flush=True)
                        else:
                            print(
                                f"  [FINAL EVAL] Step: {steps_processed / 1e6:>5.1f}M | Eval Ret: {mean_test_ret:>5.1f}",
                                flush=True)

                        with open(csv_path, "a", newline="") as f:
                            csv.writer(f).writerow(
                                [env_name, model_name, seed, steps_processed, active_train_time, sps, train_ret,
                                 train_ent, train_vloss_var, mean_test_ret])

                        wandb.log(log_dict, step=steps_processed)

                        if is_early_stop:
                            break
                    else:
                        print(
                            f"  [TRAIN] Step: {steps_processed / 1e6:>5.1f}M | Ret: {train_ret:>5.1f} | VLossVar: {train_vloss_var:>6.2f} | SPS: {int(sps):>5}",
                            flush=True)
                        wandb.log(log_dict, step=steps_processed)

                wandb.finish()


# =====================================================================
# 6. DATA VISUALIZATION
# =====================================================================
def plot_experiment10():
    csv_path = "scripts/diagnostics/architecture/swa_ablation/swa_ablation.csv"
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)

    for env_name in df["Environment"].unique():
        env_df = df[df["Environment"] == env_name]
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        color_map = plt.get_cmap("tab10")

        for idx, cond in enumerate(env_df["Condition"].unique()):
            cond_df = env_df[env_df["Condition"] == cond].groupby("Total_Steps").mean(numeric_only=True)
            steps = cond_df.index.values / 1e6
            c = color_map(idx)

            axes[0, 0].plot(steps, cond_df["Train_Return"], label=f"{cond} (Train)", color=c, linewidth=2.0)
            eval_ret = cond_df["Eval_Return"].dropna()
            axes[0, 0].plot(eval_ret.index.values / 1e6, eval_ret.values, label=f"{cond} (Eval)", color=c, marker="o",
                            markersize=8, linestyle="")

            axes[0, 1].plot(steps, cond_df["Train_Entropy"], label=cond, color=c, linewidth=2.0)
            axes[1, 0].plot(steps, cond_df["Train_VLoss_Var"], label=cond, color=c, linewidth=2.0)
            axes[1, 1].plot(steps, cond_df["SPS"], label=cond, color=c, linewidth=2.0)

        axes[0, 0].set_title(f"Generalization: Train vs Eval ({env_name})", fontweight='bold')
        axes[0, 0].set_ylabel("Mean Return")
        axes[0, 1].set_title("Exploration: Policy Entropy", fontweight='bold')
        axes[1, 0].set_title("Optimization Stability: Value Loss Variance", fontweight='bold')
        axes[1, 0].set_yscale('log')
        axes[1, 1].set_title("Hardware Efficiency: Throughput (SPS)", fontweight='bold')

        for ax in axes.flatten():
            ax.set_xlabel("Steps (Millions)")
            ax.legend()

        plt.tight_layout()
        plt.savefig(f"scripts/diagnostics/architecture/swa_ablation/swa_ablation_{env_name}.png", dpi=300)
        plt.close()
        print(f"Saved plot for {env_name}")


def _parse_int_list(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniGrid local-pathway intervention runner")
    parser.add_argument("--seeds", default=",".join(map(str, TRAINING_SEEDS)))
    parser.add_argument("--conditions", default=",".join(MODELS_TO_RUN),
                        help="Comma-separated subset of SdLG-SWA,SdLG-None")
    parser.add_argument("--output", default="scripts/diagnostics/architecture/swa_ablation/swa_ablation.csv")
    parser.add_argument("--manifest")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fixed-endpoint", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    os.environ["HSA_ENABLE_SDMA"] = "0"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    run_experiment10(seeds=_parse_int_list(args.seeds), model_names=args.conditions.split(","),
                     csv_path=args.output, overwrite=args.overwrite, fixed_endpoint=args.fixed_endpoint,
                     manifest_path=args.manifest)
    if args.plot and args.output == "scripts/diagnostics/architecture/swa_ablation/swa_ablation.csv":
        plot_experiment10()
