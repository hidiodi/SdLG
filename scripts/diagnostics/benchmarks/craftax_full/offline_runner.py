# scripts/diagnostics/exp8/exp8_1b_benchmark.py

import os
import sys

# =====================================================================
# --- HARDWARE TUNING (STANDALONE OFFLINE EXECUTION) ---
# =====================================================================
# Inject ROCm environment variables directly to bypass run_all.py
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["MIOPEN_FIND_MODE"] = "2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.60"

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
import orbax.checkpoint as ocp
from pathlib import Path
from typing import Any

# Ensure path includes the project root


import sys
import os
current_file_path = os.path.abspath(__file__)
project_root = current_file_path
while not os.path.exists(os.path.join(project_root, "src")):
    project_root = os.path.dirname(project_root)
    if project_root == "/":
        break
if project_root not in sys.path:
    sys.path.append(project_root)
from src.envs.builder import build_purejaxrl_env
from src.utils.config_schema import StaticEnvConfig, DynamicTrainingConfig
import distrax

from src.models.purejaxrl.wrappers import RepresentationNetwork, ActorCriticSdLGHybrid
from src.models.common.gtrxl import BatchedGTrXL, GatedTransformerXLLayer
from src.models.common.sdlg_hybrid import BatchedSdLGHybrid
from src.trainers.purejax_trainer import make_update, flush_carry_state
import src.trainers.purejax_trainer as trainer_module

from craftax.craftax.constants import Achievement

ACH_KEYS = [ach.name for ach in Achievement]


# =====================================================================
# --- CRITICAL BUG FIX FOR BASELINE GTrXL ---
# =====================================================================
def patched_gtrxl_call(self, inputs, pos_embedding, u, v, memory, mask=None):
    full_input = jnp.concatenate([memory, inputs], axis=0)
    x1 = self.layernorm1(full_input)
    query_input = x1[-inputs.shape[0]:]
    a1 = self.dropout_fn(self.attention(query_input, pos_embedding, x1, u, v, mask=mask))
    a1 = self.activation(a1)
    o1 = self.gate1(inputs, a1) if self.gating else inputs + a1
    x2 = self.layernorm2(o1)
    m2 = self.dropout_fn(self.mlp(x2))
    o2 = self.gate2(o1, m2) if self.gating else o1 + m2
    return o2


GatedTransformerXLLayer.__call__ = patched_gtrxl_call

# =====================================================================
# --- EXPERIMENT CONFIGURATION ---
# =====================================================================
TOTAL_STEPS = 300_000_000
TRAIN_LOG_INTERVAL_STEPS = 500_000

MODELS_TO_RUN = ["SdLGHybrid","PPO-GTrXL", "PPO-S5"]
TRAINING_SEEDS = [42, 43, 44]

CHECKPOINT_BASE_DIR = os.path.abspath("checkpoints/exp8_benchmark")
CSV_PATH = "scripts/diagnostics/benchmarks/craftax_full/craftax_full.csv"


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

        # [PILLAR 1 FIX] Prevent NaN Entropy over 250M Steps
        action_logits = 20.0 * jnp.tanh(action_logits)

        critic_values = nn.Dense(
            1, kernel_init=flax.linen.initializers.orthogonal(1.0), name="critic_head"
        )(outs)
        critic_values = jnp.squeeze(critic_values, axis=-1)

        pi = distrax.Categorical(logits=action_logits)
        return new_carry, pi, critic_values


# =====================================================================
# 2. VRAM-SAFE CHUNKED STOCHASTIC EVALUATOR
# =====================================================================
def build_chunked_evaluator(env, env_params, network, config):
    eval_steps = 10000

    @functools.partial(jax.jit, static_argnums=(3,))
    def eval_chunk(params, start_seed, base_rng, chunk_size):
        keys = jax.random.split(jax.random.PRNGKey(start_seed), chunk_size)
        reset_fn = jax.vmap(env.reset, in_axes=(0, None))
        step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, None))

        obs, env_state = reset_fn(keys, env_params)

        if config["RNN"]["RNN_TYPE"] == "sdlg_hybrid":
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

            if config["RNN"]["RNN_TYPE"] == "sdlg_hybrid":
                action_logits, _, _, next_hstate = network.apply(params, hstate, ac_in)
                pi = distrax.Categorical(logits=action_logits.squeeze(0))
                action = pi.sample(seed=action_rng)
            else:
                next_hstate, pi, _ = network.apply(params, hstate, ac_in)
                # [PILLAR 5 FIX] Squeeze the time-dimension (T=1) to prevent step_fn shape crashes
                action = pi.sample(seed=action_rng).squeeze(0)

            rngs = jax.random.split(jax.random.PRNGKey(0), chunk_size)
            next_obs, next_env_state, reward, next_dones, info = step_fn(rngs, env_state, action, env_params)

            if config["RNN"]["RNN_TYPE"] == "sdlg_hybrid":
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
            return next_carry, None

        rng_scan = jax.random.PRNGKey(start_seed)
        init_carry = (obs, env_state, hstate, dones, init_max_ach, init_max_ret, rng_scan)

        final_carry = jax.lax.stop_gradient(jax.lax.scan(step_fn_scan, init_carry, None, length=eval_steps)[0])
        _, _, _, _, final_max_ach, final_max_ret, _ = final_carry

        return final_max_ret, final_max_ach

    return eval_chunk


# =====================================================================
# 3. OFFLINE BENCHMARKING RUNNER
# =====================================================================
def run_experiment8():
    print("====================================================================")
    print("=== EXPERIMENT 8: 1-BILLION-STEP OFFLINE BENCHMARK               ===")
    print("====================================================================")

    ENV_NAME = "Craftax-Symbolic-v1"
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
                     "Train_Return", "Train_Entropy", "Train_VLoss"
                 ] + [f"Train_{k}" for k in ACH_KEYS] + ["Eval_Return"] + [f"Eval_{k}" for k in ACH_KEYS]
        csv_writer.writerow(header)
        csv_file.flush()

    for model_name in MODELS_TO_RUN:
        is_hybrid = (model_name == "SdLGHybrid")

        if model_name == "PPO-GTrXL":
            trainer_module.ActorCriticRNN = ActorCriticGTrXL
        elif model_name == "SdLGHybrid":
            trainer_module.ActorCriticRNN = ActorCriticSdLGHybrid

        config = {
            "NUM_ENVS": NUM_ENVS, "NUM_STEPS": NUM_STEPS, "TOTAL_TIMESTEPS": TOTAL_STEPS,
            "UPDATE_EPOCHS": 4, "NUM_MINIBATCHES": 8, "GAMMA": 0.99, "GAE_LAMBDA": 0.95,
            "CLIP_EPS": 0.2, "ENT_COEF": 0.01, "VF_COEF": 0.5, "MAX_GRAD_NORM": 0.5,
            "LR": 0.0002, "OPTIMISTIC_RESETS": False, "ANNEAL_LR": True,
            "LOG_INTERVAL": LOG_INTERVAL_UPDATES,
            "RNN": {
                "RNN_TYPE": "sdlg_hybrid" if is_hybrid else "gtrxl",
                "N_LAYERS": 4, "D_MODEL": 256, "D_HEAD": 32, "N_HEADS": 8,
                "MEMORY_LEN": 128, "D_K": 32, "D_V": 32, "D_FFC": 1024,
                "R": 4, "WINDOW_SIZE": 8, "ALPHA": 0.1, "ETA": 0.0,
                "USE_ROPE_PHASE": True, "USE_SOFTPLUS_DENOMINATOR": True, "USE_PER_RANK_NORM": True
            }
        }

        static_env = StaticEnvConfig(env_name=ENV_NAME, num_envs=NUM_ENVS, num_steps=NUM_STEPS, max_episode_steps=10000,
                                     optimistic_resets=False)
        dynamic_train = DynamicTrainingConfig(
            total_timesteps=TOTAL_STEPS, learning_rate=0.0002, gamma=0.99, gae_lambda=0.95,
            clip_eps=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, update_epochs=4,
            minibatch_size=2048, log_interval=LOG_INTERVAL_UPDATES
        )

        env, env_params = build_purejaxrl_env(static_env, dynamic_train)
        action_dim = env.action_space(env_params).n

        if is_hybrid:
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
            print(f"\n[{model_name}] Initializing Compilation & Training | SEED = {seed}", flush=True)

            ckpt_dir = Path(CHECKPOINT_BASE_DIR) / f"{model_name}_seed_{seed}"
            options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
            ckpt_manager = ocp.CheckpointManager(ckpt_dir.resolve(), options=options)

            rng = jax.random.PRNGKey(seed)
            raw_update_fn, runner_state = make_update(rng, config, env, env_params, lambda: network, rnn_carry_init)

            start_loop_idx = 0
            latest_step = ckpt_manager.latest_step()

            if latest_step is not None:
                print(f"  [CHECKPOINT] Found existing checkpoint! Resuming from Update Step {latest_step}...",
                      flush=True)
                restored = ckpt_manager.restore(latest_step, args=ocp.args.StandardRestore({"runner_state": runner_state}))
                runner_state = restored["runner_state"]
                start_loop_idx = latest_step // LOG_INTERVAL_UPDATES

                if start_loop_idx >= (TOTAL_UPDATES // LOG_INTERVAL_UPDATES):
                    print(f"  [CHECKPOINT] Run already completed. Skipping.")
                    continue
            else:
                print("  [CHECKPOINT] No existing checkpoint found. Starting from scratch.", flush=True)

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
                    "nan_detected": jnp.any(m.get("parameter_nan_detected", jnp.zeros(1, dtype=bool)))
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
                runner_state, reduced_metric = compiled_update_and_reduce(runner_state)
                runner_state = jax.block_until_ready(runner_state)
                reduced_metric = jax.device_get(reduced_metric)

                if bool(reduced_metric["nan_detected"]):
                    print(f"\n[FATAL ERROR] NaN detected in model weights at step {steps_processed / 1e6:.1f}M!")
                    print(f"Last recorded Max Grad Norm: {float(reduced_metric['max_grad_norm']):.2f}")
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

                is_final_loop = (loop_idx == NUM_LOG_LOOPS - 1)

                if is_final_loop:
                    eval_start = time.time()
                    ckpt_manager.save(current_update_idx, items={"runner_state": runner_state})
                    print(f"  [CHECKPOINT] Saved state at Final Step {steps_processed / 1e6:.1f}M", flush=True)

                    eval_returns = []
                    eval_ach_chunks = {k: [] for k in ACH_KEYS}

                    for chunk in range(8):
                        start_test_seed = 10000 + (chunk * 128)
                        ret_c, max_ach_c = jax.device_get(
                            chunked_eval_fn(runner_state[0].params, start_test_seed, rng, 128)
                        )
                        eval_returns.append(ret_c)
                        for k in ACH_KEYS:
                            eval_ach_chunks[k].append(max_ach_c[k])

                    mean_test_ret = float(np.mean(np.concatenate(eval_returns)))
                    mean_eval_ach = {k: float(np.mean(np.concatenate(eval_ach_chunks[k]))) for k in ACH_KEYS}

                    eval_end = time.time()
                    cumulative_eval_time += (eval_end - eval_start)

                    total_eval_achv = sum(mean_eval_ach.values())
                    print(
                        f"  [FINAL EVAL] Step: {steps_processed / 1e6:>5.1f}M | Ret: {mean_test_ret:>5.1f} | Achv: {total_eval_achv:>4.1f} | SPS: {int(sps):>5}",
                        flush=True)

                    # Write full row (Train + Eval)
                    row = [
                        model_name, seed, steps_processed, active_train_time, sps,
                        train_ret, train_ent, train_vloss
                    ]
                    row.extend([float(reduced_metric[f"train_{k}"]) for k in ACH_KEYS])
                    row.append(mean_test_ret)
                    row.extend([mean_eval_ach[k] for k in ACH_KEYS])
                    csv_writer.writerow(row)
                    csv_file.flush()

                    del eval_returns, eval_ach_chunks, ret_c, max_ach_c

                else:
                    print(
                        f"  [TRAIN] Step: {steps_processed / 1e6:>5.1f}M | Ret: {train_ret:>5.1f} | Achv: {train_total_achv:>4.1f} | Len: {train_len:>4.0f} | VLoss: {train_vloss:>5.3f} | MaxGrad: {max_grad:>5.2f} | SPS: {int(sps):>5}",
                        flush=True)

                    # Write incremental row (Train only, Eval is empty)
                    row = [
                        model_name, seed, steps_processed, active_train_time, sps,
                        train_ret, train_ent, train_vloss
                    ]
                    row.extend([float(reduced_metric[f"train_{k}"]) for k in ACH_KEYS])
                    row.append("")  # Empty Eval_Return
                    row.extend(["" for _ in ACH_KEYS])  # Empty Eval_Ach
                    csv_writer.writerow(row)
                    csv_file.flush()

                    if loop_idx > start_loop_idx and loop_idx % 100 == 0:
                        ckpt_manager.save(current_update_idx, items={"runner_state": runner_state})
                        print(f"  [CHECKPOINT] Saved intermediate state at Step {steps_processed / 1e6:.1f}M",
                              flush=True)

                # [PILLAR 5 SYSTEM FIX] Destroy ALL host variables before next loop
                del reduced_metric
                gc.collect()

            ckpt_manager.wait_until_finished()

            # Wipe state and force VRAM clear before next seed/model
            del runner_state, compiled_update_and_reduce, raw_update_fn
            jax.clear_caches()
            gc.collect()

    csv_file.close()


# =====================================================================
# 4. OFFLINE DATA VISUALIZATION
# =====================================================================
def plot_experiment8():
    csv_path = CSV_PATH
    if not os.path.exists(csv_path):
        return

    plt.close('all')
    plt.clf()
    plt.cla()

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    train_ach_cols = [c for c in df.columns if
                      c.startswith("Train_") and c not in ["Train_Return", "Train_Entropy", "Train_VLoss", "Train_Len"]]

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    colors = {"PPO-GTrXL": "#d7191c", "SdLGHybrid": "#2c7bb6"}
    styles = {"PPO-GTrXL": "--", "SdLGHybrid": "-"}

    for model_name in MODELS_TO_RUN:
        model_data = df[df["Architecture"] == model_name]
        if model_data.empty: continue

        grp = model_data.groupby("Total_Steps")
        mean_df = grp.mean(numeric_only=True)
        std_df = grp.std(numeric_only=True, ddof=1)
        steps = mean_df.index.values / 1e6

        # Panel A: Train Return
        mean_ret = mean_df["Train_Return"].values
        sem_ret = std_df["Train_Return"].values / np.sqrt(len(TRAINING_SEEDS))
        axes[0, 0].plot(steps, mean_ret, label=model_name, color=colors[model_name], linestyle=styles[model_name],
                        linewidth=2.5)
        if len(TRAINING_SEEDS) > 1:
            axes[0, 0].fill_between(steps, mean_ret - sem_ret, mean_ret + sem_ret, color=colors[model_name], alpha=0.15)

        # Panel B: Total Capabilities (Training)
        mean_df["Train_Total_Achv"] = mean_df[train_ach_cols].sum(axis=1)
        std_df["Train_Total_Achv"] = std_df[train_ach_cols].sum(axis=1)

        clean_strat = mean_df["Train_Total_Achv"]
        clean_strat_sem = std_df["Train_Total_Achv"] / np.sqrt(len(TRAINING_SEEDS))

        axes[0, 1].plot(steps, clean_strat.values, label=model_name, color=colors[model_name],
                        linestyle=styles[model_name], linewidth=2.5)
        if len(TRAINING_SEEDS) > 1:
            axes[0, 1].fill_between(steps, clean_strat.values - clean_strat_sem.values,
                                    clean_strat.values + clean_strat_sem.values, color=colors[model_name], alpha=0.15)

        # Panel C: Value Loss Volatility
        vloss = mean_df["Train_VLoss"].values
        axes[1, 0].plot(steps, vloss, label=model_name, color=colors[model_name], linestyle=styles[model_name],
                        linewidth=1.5, alpha=0.8)

        # Panel D: SPS
        sps = mean_df["SPS"].values
        axes[1, 1].plot(steps, sps, label=model_name, color=colors[model_name], linestyle=styles[model_name],
                        linewidth=2.5)

    axes[0, 0].set_title("Learning Curve: Mean Return", fontweight='bold')
    axes[0, 0].set_ylabel("Training Episode Return")

    axes[0, 1].set_title("Capabilities: Total Achievements Unlocked", fontweight='bold')
    axes[0, 1].set_ylabel("Achievements Unlocked")

    axes[1, 0].set_title("Optimization: Mean Value Loss", fontweight='bold')
    axes[1, 0].set_ylabel("Value Loss")
    axes[1, 0].set_yscale('log')

    axes[1, 1].set_title(r"Efficiency: Real-World Throughput ($\eta_{SPS}$)", fontweight='bold')
    axes[1, 1].set_ylabel("Steps Per Second (SPS)")

    for ax in axes.flatten():
        ax.set_xlabel("Environment Steps (Millions)")
        ax.legend(frameon=True)

    plt.tight_layout()
    plot_path = "scripts/diagnostics/benchmarks/craftax_full/craftax_full.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"\nRendered 4-Panel Scientific Matrix to {plot_path}")


if __name__ == "__main__":
    run_experiment8()
    plot_experiment8()
















