# scripts/diagnostics/architecture/gradient_shielding/gradient_shielding.py

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
import platform
import json
import subprocess
from typing import Any

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
from src.envs.builder import build_purejaxrl_env
from src.utils.config_schema import StaticEnvConfig, DynamicTrainingConfig
import distrax

from src.models.purejaxrl.wrappers import RepresentationNetwork, ActorCriticSdLGHybrid
from src.models.common.sdlg_hybrid import BatchedSdLGHybrid
from src.trainers.purejax_trainer import make_update, flush_carry_state

# Dynamically load the correct Achievement constants for Craftax-Classic
try:
    from craftax.craftax_classic.constants import Achievement
except ImportError:
    from craftax.craftax.constants import Achievement

ACH_KEYS = [ach.name for ach in Achievement]

# =====================================================================
# --- EXPERIMENT 13 CONFIGURATION ---
# =====================================================================
TOTAL_STEPS = 20_000_000  # 20M is perfect to catch the entropy cliff
TRAIN_LOG_INTERVAL_STEPS = 500_000
EVAL_INTERVAL_STEPS = 5_000_000

MODELS_TO_RUN = ["SdLGHybrid-GGS-0.1", "SdLGHybrid-GGS-1.0"]
#MODELS_TO_RUN = ["SdLGHybrid-GGS-1.0"]
TRAINING_SEEDS = [42, 54, 99, 123, 456]

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "sdlgHybrid")


# =====================================================================
# 1. DIAGNOSTIC PROBE: ACTOR/CRITIC OUTPUT-GRADIENT COSINE
# =====================================================================
@functools.partial(jax.jit, static_argnums=(4,))
def compute_gradient_cosine_similarity(params, carry, obs_batch, dones_batch, apply_fn):
    """
    Computes a diagnostic cosine between gradients of actor-output and
    critic-output probe objectives. These are not the PPO policy/value losses,
    so the statistic does not by itself identify destructive loss conflict.
    """

    def actor_objective(p):
        logits, _, _, _ = apply_fn(p, carry, (obs_batch, dones_batch))
        return jnp.mean(jnp.square(logits))

    def critic_objective(p):
        _, values, _, _ = apply_fn(p, carry, (obs_batch, dones_batch))
        return jnp.mean(jnp.square(values))

    # Compute full gradients
    grad_actor = jax.grad(actor_objective)(params)
    grad_critic = jax.grad(critic_objective)(params)

    # Flatten PyTrees into 1D vectors
    from jax.flatten_util import ravel_pytree
    flat_g_a, _ = ravel_pytree(grad_actor)
    flat_g_c, _ = ravel_pytree(grad_critic)

    # Cosine Similarity: (A dot B) / (||A|| * ||B||)
    cos_sim = jnp.dot(flat_g_a, flat_g_c) / (jnp.linalg.norm(flat_g_a) * jnp.linalg.norm(flat_g_c) + 1e-8)
    return cos_sim


# =====================================================================
# 2. VRAM-SAFE CHUNKED DETERMINISTIC EVALUATOR
# =====================================================================
def build_chunked_evaluator(env, env_params, network, config):
    eval_steps = 10000

    @functools.partial(jax.jit, static_argnums=(3,))
    def eval_chunk(params, start_seed, base_rng, chunk_size):
        keys = jax.random.split(jax.random.PRNGKey(start_seed), chunk_size)
        reset_fn = jax.vmap(env.reset, in_axes=(0, None))
        step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, None))

        obs, env_state = reset_fn(keys, env_params)

        unbatched = BatchedSdLGHybrid.initialize_carry(
            batch_size=chunk_size, n_layers=config["RNN"]["N_LAYERS"],
            d_k=config["RNN"]["D_K"], d_v=config["RNN"]["D_V"],
            r=config["RNN"]["R"], window_size=config["RNN"]["WINDOW_SIZE"]
        )
        hstate = jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), chunk_size, axis=0), unbatched)

        dones = jnp.zeros((chunk_size,), dtype=bool)
        init_max_ach = {k: jnp.zeros((chunk_size,), dtype=jnp.int32) for k in ACH_KEYS}
        init_max_ret = jnp.full((chunk_size,), -jnp.inf)

        def step_fn_scan(carry, _):
            obs, env_state, hstate, dones, max_ach, max_ret, step_rng = carry
            ac_in = (obs[np.newaxis, :], dones[np.newaxis, :])
            step_rng, action_rng = jax.random.split(step_rng)

            action_logits, _, _, next_hstate = network.apply(params, hstate, ac_in)
            pi = distrax.Categorical(logits=action_logits.squeeze(0))
            action = pi.sample(seed=action_rng)

            rngs = jax.random.split(jax.random.PRNGKey(0), chunk_size)
            next_obs, next_env_state, reward, next_dones, info = step_fn(rngs, env_state, action, env_params)

            next_hstate = flush_carry_state(next_hstate, next_dones)

            curr_ach = next_env_state.env_state.achievements
            new_max_ach = {name: jnp.maximum(max_ach[name], curr_ach[:, i].astype(jnp.int32)) for i, name in
                           enumerate(ACH_KEYS)}
            current_returns = jnp.where(next_dones,
                                        info.get("returned_episode_returns", next_env_state.episode_returns),
                                        next_env_state.episode_returns)
            new_max_ret = jnp.maximum(max_ret, current_returns)

            return (next_obs, next_env_state, next_hstate, next_dones, new_max_ach, new_max_ret, step_rng), None

        init_carry = (obs, env_state, hstate, dones, init_max_ach, init_max_ret, jax.random.PRNGKey(start_seed))
        final_carry = jax.lax.stop_gradient(jax.lax.scan(step_fn_scan, init_carry, None, length=eval_steps)[0])
        return final_carry[5], final_carry[4]

    return eval_chunk


# =====================================================================
# 3. BENCHMARKING RUNNER
# =====================================================================
def _git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()


def _source_sha256():
    with open(__file__, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _alpha_label(alpha):
    return f"SdLGHybrid-GGS-{alpha:g}"


def run_experiment13(alphas=None, seeds=None, csv_path=None, overwrite=False, manifest_path=None):
    alphas = [0.1, 1.0] if alphas is None else list(alphas)
    seeds = list(TRAINING_SEEDS if seeds is None else seeds)
    csv_path = csv_path or "scripts/diagnostics/architecture/gradient_shielding/gradient_shielding.csv"
    print("====================================================================")
    print("=== EXPERIMENT 13: ACTOR-GRADIENT SCALE SWEEP ===")
    print(f"=== Alphas: {alphas} | Seeds: {seeds}")
    print("====================================================================")

    ENV_NAME = "Craftax-Classic-Symbolic-v1"
    NUM_ENVS = 256
    NUM_STEPS = 64

    STEPS_PER_UPDATE = NUM_ENVS * NUM_STEPS
    TOTAL_UPDATES = TOTAL_STEPS // STEPS_PER_UPDATE
    LOG_INTERVAL_UPDATES = TRAIN_LOG_INTERVAL_STEPS // STEPS_PER_UPDATE
    EVAL_EVERY_N_LOGS = max(1, EVAL_INTERVAL_STEPS // TRAIN_LOG_INTERVAL_STEPS)

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if manifest_path:
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump({
                "study_id": "CRAFTAX-ACTOR-GRADIENT-SWEEP-2026",
                "git_commit": _git_commit(),
                "git_worktree_dirty": bool(subprocess.check_output(
                    ["git", "status", "--porcelain", "--untracked-files=no"], cwd=project_root, text=True).strip()),
                "runner_sha256": _source_sha256(),
                "argv": sys.argv,
                "runner": os.path.relpath(__file__, project_root),
                "output_csv": os.path.relpath(os.path.abspath(csv_path), project_root),
                "alphas": alphas,
                "seeds": seeds,
                "total_timesteps": TOTAL_STEPS,
                "num_envs": NUM_ENVS,
                "num_steps": NUM_STEPS,
                "endpoint_semantics": "fixed",
                "telemetry": "probe-gradient cosine at every 0.5M training steps; evaluation every 5M steps",
                "hardware": {
                    "host_platform": platform.platform(),
                    "jax_backend": jax.default_backend(),
                    "jax_devices": [str(device) for device in jax.devices()],
                },
            }, f, indent=2)

    write_header = overwrite or not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    csv_file = open(csv_path, "w" if overwrite else "a", newline="")
    csv_writer = csv.writer(csv_file)
    header = ["Architecture", "Seed", "Total_Steps", "Wall_Clock_Time", "SPS", "Train_Return", "Train_Entropy",
              "Train_VLoss_Var", "Grad_Cosine_Sim", "Eval_Return"] + [f"Train_{k}" for k in ACH_KEYS] + [f"Eval_{k}" for
                                                                                                         k in ACH_KEYS]
    if write_header:
        csv_writer.writerow(header)
    csv_file.flush()

    for alpha_val in alphas:
        model_label = _alpha_label(alpha_val)

        config = {
            "NUM_ENVS": NUM_ENVS, "NUM_STEPS": NUM_STEPS, "TOTAL_TIMESTEPS": TOTAL_STEPS,
            "UPDATE_EPOCHS": 4, "NUM_MINIBATCHES": 8, "GAMMA": 0.99, "GAE_LAMBDA": 0.95,
            "CLIP_EPS": 0.2, "ENT_COEF": 0.01, "VF_COEF": 0.5, "MAX_GRAD_NORM": 0.5,
            "LR": 0.0003, "OPTIMISTIC_RESETS": False, "ANNEAL_LR": True, "LOG_INTERVAL": LOG_INTERVAL_UPDATES,
            "RNN": {"RNN_TYPE": "sdlg_hybrid", "N_LAYERS": 4, "D_MODEL": 256, "D_HEAD": 32, "N_HEADS": 8,
                    "MEMORY_LEN": 128, "D_K": 32, "D_V": 32, "D_FFC": 1024, "R": 4, "WINDOW_SIZE": 8,
                    "ALPHA": alpha_val, "ETA": 0.0, "USE_ROPE_PHASE": True, "USE_SOFTPLUS_DENOMINATOR": True,
                    "USE_PER_RANK_NORM": True}
        }

        static_env = StaticEnvConfig(env_name=ENV_NAME, num_envs=NUM_ENVS, num_steps=NUM_STEPS, max_episode_steps=10000,
                                     optimistic_resets=False)
        dynamic_train = DynamicTrainingConfig(total_timesteps=TOTAL_STEPS, learning_rate=0.0003, gamma=0.99,
                                              gae_lambda=0.95, clip_eps=0.2, ent_coef=0.01, vf_coef=0.5,
                                              max_grad_norm=0.5, update_epochs=4, minibatch_size=2048,
                                              log_interval=LOG_INTERVAL_UPDATES)

        env, env_params = build_purejaxrl_env(static_env, dynamic_train)

        # Make sure the log wrapper handles achievements gracefully
        try:
            from src.envs.wrappers import LogWrapper
        except ImportError:
            from gymnax.wrappers.purerl import LogWrapper

        if not isinstance(env, LogWrapper): env = LogWrapper(env)
        action_dim = env.action_space(env_params).n
        network = ActorCriticSdLGHybrid(action_dim=action_dim, config=config)

        def rnn_carry_init(batch_size):
            unbatched = BatchedSdLGHybrid.initialize_carry(batch_size=batch_size, n_layers=4, d_k=32, d_v=32, r=4,
                                                           window_size=8)
            return jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0), unbatched)

        chunked_eval_fn = build_chunked_evaluator(env, env_params, network, config)

        for seed in seeds:
            jax.clear_caches()
            gc.collect()
            print(f"\n[{model_label}] Initializing Compilation & Training | SEED = {seed}", flush=True)

            run_config = dict(config)
            run_config.update({"RUN_SEED": seed, "STUDY_ID": "CRAFTAX-ACTOR-GRADIENT-SWEEP-2026",
                               "GGS_ALPHA": alpha_val})
            extra_tags = os.environ.get("WANDB_TAGS", "").split(",") if os.environ.get("WANDB_TAGS") else []
            init_wandb(project=WANDB_PROJECT, group="revision-2026-ggs-alpha-sweep",
                       name=f"revision-2026-{model_label}-Seed-{seed}", config=run_config,
                       tags=["empirical_strengthening_2026", "ggs_alpha_sweep"] + extra_tags)
            rng = jax.random.PRNGKey(seed)
            raw_update_fn, runner_state = make_update(rng, config, env, env_params, lambda: network, rnn_carry_init)

            @functools.partial(jax.jit, donate_argnums=(0,))
            def compiled_update_and_reduce(r_state):
                out = raw_update_fn(r_state)
                m = out["metric"]
                done_mask = m.get("returned_episode", jnp.ones_like(m.get("entropy", jnp.zeros(1))))

                def get_done_mean(x): return jnp.sum(x * done_mask) / jnp.maximum(1.0, jnp.sum(done_mask))

                reduced = {
                    "train_ret": get_done_mean(m.get("returned_episode_returns", jnp.zeros_like(done_mask))),
                    "train_ent": jnp.mean(m.get("entropy", jnp.zeros(1))),
                    "train_vloss_var": jnp.var(m.get("value_loss", jnp.zeros(1))),
                }
                native_achievements = out["runner_state"][1].env_state.achievements
                for i, name in enumerate(ACH_KEYS): reduced[f"train_{name}"] = jnp.mean(
                    native_achievements[:, i].astype(jnp.float32))
                return out["runner_state"], reduced

            steps_processed = 0
            log_counter = 0
            start_wall_time = time.time()
            cumulative_eval_time = 0.0

            for update_idx in range(0, TOTAL_UPDATES, LOG_INTERVAL_UPDATES):
                runner_state, reduced_metric = compiled_update_and_reduce(runner_state)
                runner_state = jax.block_until_ready(runner_state)

                # --- DIAGNOSTIC PROBE EXECUTION ---
                train_state = runner_state[0]
                last_obs = runner_state[2]

                # Take a small batch slice to compute local similarity efficiently
                probe_obs = last_obs[np.newaxis, :32]
                probe_dones = jnp.zeros((1, 32), dtype=bool)
                probe_carry = rnn_carry_init(32)

                cos_sim = float(jax.device_get(
                    compute_gradient_cosine_similarity(train_state.params, probe_carry, probe_obs, probe_dones,
                                                       network.apply)
                ))
                # -----------------------------------

                reduced_metric = jax.device_get(reduced_metric)
                steps_processed += LOG_INTERVAL_UPDATES * STEPS_PER_UPDATE
                log_counter += 1

                train_ret, train_ent, train_vloss_var = float(reduced_metric["train_ret"]), float(
                    reduced_metric["train_ent"]), float(reduced_metric["train_vloss_var"])
                active_train_time = time.time() - start_wall_time - cumulative_eval_time
                sps = steps_processed / max(active_train_time, 1.0)

                wandb_log_dict = {
                    "System/SPS": sps,
                    "System/Wall_Clock_Time": active_train_time,
                    "Train/Mean_Return": train_ret,
                    "Train/Policy_Entropy": train_ent,
                    "Train/Value_Loss_Var": train_vloss_var,
                    "Diagnostics/Gradient_Cosine_Similarity": cos_sim
                }
                for k in ACH_KEYS: wandb_log_dict[f"Train_Achv/{k}"] = float(reduced_metric[f"train_{k}"])

                is_eval_checkpoint = log_counter % EVAL_EVERY_N_LOGS == 0 or steps_processed >= TOTAL_STEPS
                mean_test_ret = ""
                mean_eval_ach = {k: "" for k in ACH_KEYS}
                if is_eval_checkpoint:
                    eval_start = time.time()
                    eval_returns, eval_ach_chunks = [], {k: [] for k in ACH_KEYS}
                    for chunk in range(8):
                        ret_c, max_ach_c = jax.device_get(
                            chunked_eval_fn(runner_state[0].params, 10000 + (chunk * 128), rng, 128))
                        eval_returns.append(ret_c)
                        for k in ACH_KEYS: eval_ach_chunks[k].append(max_ach_c[k])

                    mean_test_ret = float(np.mean(np.concatenate(eval_returns)))
                    mean_eval_ach = {k: float(np.mean(np.concatenate(eval_ach_chunks[k]))) for k in ACH_KEYS}
                    cumulative_eval_time += (time.time() - eval_start)

                    wandb_log_dict["Eval/Mean_Return"] = mean_test_ret
                    total_achv = sum(mean_eval_ach.values())
                    print(
                        f"  [EVAL] Step: {steps_processed / 1e6:>5.1f}M | Ret: {mean_test_ret:>5.1f} | CosSim: {cos_sim:>5.2f} | Total Achv: {total_achv:>4.1f} | SPS: {int(sps):>5}",
                        flush=True)

                else:
                    print(
                        f"  [TRAIN] Step: {steps_processed / 1e6:>5.1f}M | Ret: {train_ret:>5.1f} | Var(VF): {train_vloss_var:>6.2f} | CosSim: {cos_sim:>5.2f} | SPS: {int(sps):>5}",
                        flush=True)

                # Persist every probe checkpoint. Evaluation fields remain empty between
                # the prespecified 5M-step evaluations rather than being carried forward.
                row = [model_label, seed, steps_processed, active_train_time, sps, train_ret, train_ent,
                       train_vloss_var, cos_sim, mean_test_ret]
                row.extend([float(reduced_metric[f"train_{k}"]) for k in ACH_KEYS])
                row.extend([mean_eval_ach[k] for k in ACH_KEYS])
                csv_writer.writerow(row)
                csv_file.flush()

                wandb.log(wandb_log_dict, step=steps_processed)
            wandb.finish()
    csv_file.close()


# =====================================================================
# 4. 5-PANEL DATA VISUALIZATION (Including Cosine Similarity)
# =====================================================================
def plot_experiment13():
    csv_path = "scripts/diagnostics/architecture/gradient_shielding/gradient_shielding.csv"
    if not os.path.exists(csv_path): return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman']})
    # Expanded to 2x3 grid to fit the new metric beautifully
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    axes = axes.flatten()

    colors = {"SdLGHybrid-GGS-1.0": "#d7191c", "SdLGHybrid-GGS-0.1": "#2c7bb6"}
    styles = {"SdLGHybrid-GGS-1.0": "--", "SdLGHybrid-GGS-0.1": "-"}

    for model_label in MODELS_TO_RUN:
        model_data = df[df["Architecture"] == model_label]
        if model_data.empty: continue

        grp = model_data.groupby("Total_Steps")
        mean_df = grp.mean(numeric_only=True)
        std_df = grp.std(numeric_only=True, ddof=1)
        steps = mean_df.index.values / 1e6

                # Panel 1: Return
        mean_ret = mean_df["Eval_Return"].values
        sem_ret = std_df["Eval_Return"].values / np.sqrt(len(TRAINING_SEEDS))
        axes[0].plot(steps, mean_ret, label=model_label, color=colors[model_label], linestyle=styles[model_label], linewidth=2.5)
        if len(TRAINING_SEEDS) > 1:
            axes[0].fill_between(steps, mean_ret - sem_ret, mean_ret + sem_ret, color=colors[model_label], alpha=0.15)

        # Panel 2: Policy Entropy
        mean_ent = mean_df["Train_Entropy"].values
        axes[1].plot(steps, mean_ent, label=model_label, color=colors[model_label], linestyle=styles[model_label], linewidth=2.5)

        # Panel 3: Value Loss Volatility
        vloss_var = mean_df["Train_VLoss_Var"].values
        axes[2].plot(steps, vloss_var, label=model_label, color=colors[model_label], linestyle=styles[model_label], linewidth=1.5, alpha=0.8)

        # Panel 4: diagnostic actor/critic probe-gradient cosine
        if "Grad_Cosine_Sim" in mean_df.columns:
            cos_sim = mean_df["Grad_Cosine_Sim"].values
            sem_cos_sim = std_df["Grad_Cosine_Sim"].values / np.sqrt(len(TRAINING_SEEDS))
            axes[3].plot(steps, cos_sim, label=model_label, color=colors[model_label], linestyle=styles[model_label], linewidth=2.5)
            if len(TRAINING_SEEDS) > 1:
                axes[3].fill_between(steps, cos_sim - sem_cos_sim, cos_sim + sem_cos_sim, color=colors[model_label], alpha=0.15)

        # Panel 5: SPS
        sps = mean_df["SPS"].values
        axes[4].plot(steps, sps, label=model_label, color=colors[model_label], linestyle=styles[model_label], linewidth=2.5)

    # Hide the empty 6th panel
    fig.delaxes(axes[5])

    axes[0].set_title("Generalization: Mean Return", fontweight='bold')
    axes[1].set_title("Exploration: Policy Entropy", fontweight='bold')
    axes[2].set_title(r"Optimization: Value Loss Variance", fontweight='bold')
    axes[2].set_yscale('log')

    axes[3].set_title("Actor/Critic Probe-Gradient Cosine", fontweight='bold')
    axes[3].set_ylabel("Cosine Similarity (Shared Trunk)")
    # Draw a bold line at 0.0 to clearly show negative interference
    axes[3].axhline(0, color='black', linewidth=1.5, linestyle='--')

    axes[4].set_title(r"Efficiency: Real-World Throughput ($\eta_{SPS}$)", fontweight='bold')

    for ax in axes[:5]:
        ax.set_xlabel("Environment Steps (Millions)")
        ax.legend(frameon=True)

    plt.tight_layout()
    plot_path = "papers/latex/thesis_style/plots/gradient_shielding_with_probe.pdf"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"\n✅ Rendered 5-Panel Scientific Matrix to {plot_path}")

    import shutil
    os.makedirs("scripts/diagnostics/exp13", exist_ok=True)
    shutil.copy(plot_path, "scripts/diagnostics/exp13/exp13_no_ggs_ablation.png")
    print("Saved Copy to scripts/diagnostics/exp13/exp13_no_ggs_ablation.png")


def _parse_float_list(value):
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _parse_int_list(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Craftax actor-gradient scale sweep")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--alphas", default="0.1,1.0")
    parser.add_argument("--seeds", default=",".join(map(str, TRAINING_SEEDS)))
    parser.add_argument("--output", default="scripts/diagnostics/architecture/gradient_shielding/gradient_shielding.csv")
    parser.add_argument("--manifest")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.run:
        run_experiment13(alphas=_parse_float_list(args.alphas), seeds=_parse_int_list(args.seeds),
                         csv_path=args.output, overwrite=args.overwrite, manifest_path=args.manifest)
    if args.plot:
        plot_experiment13()
