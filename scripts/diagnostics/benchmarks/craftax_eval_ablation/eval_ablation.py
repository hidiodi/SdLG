import sys
import os
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import pandas as pd
import csv
import time
import functools
import matplotlib.pyplot as plt

import gc
from typing import Any



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

import wandb
from scripts.diagnostics.core.utils import init_wandb
from src.envs.builder import build_purejaxrl_env
from src.utils.config_schema import StaticEnvConfig, DynamicTrainingConfig
import distrax

from src.models.purejaxrl.wrappers import RepresentationNetwork, ActorCriticSdLGHybrid
from src.models.common.sdlg_hybrid import BatchedSdLGHybrid
from src.trainers.purejax_trainer import make_update
import src.trainers.purejax_trainer as trainer_module

from craftax.craftax.constants import Achievement

ACH_KEYS = [ach.name for ach in Achievement]

# =====================================================================
# --- EXPERIMENT 9 CONFIGURATION ---
# =====================================================================
TOTAL_STEPS = 20_000_000  # Small ablation run (20M steps)
EVAL_INTERVAL_STEPS = 5_000_000  # Evaluate every 5M steps
TRAIN_LOG_INTERVAL_STEPS = 500_000

TRAINING_SEEDS = [42, 54, 99, 123, 456]  # Multi-seed evaluation sweep
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "sdlgHybrid")

# =====================================================================

# =====================================================================
# 1. VRAM-SAFE CHUNKED EVALUATOR (WITH DETERMINISTIC TOGGLE)
# =====================================================================
def build_chunked_evaluator(env, env_params, network, config, is_deterministic: bool):
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

            if is_deterministic:
                action = jnp.argmax(action_logits.squeeze(0), axis=-1)
            else:
                action = pi.sample(seed=action_rng)

            rngs = jax.random.split(jax.random.PRNGKey(0), chunk_size)
            next_obs, next_env_state, reward, next_dones, info = step_fn(rngs, env_state, action, env_params)

            from src.trainers.purejax_trainer import flush_carry_state
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
# 2. ABLATION RUNNER
# =====================================================================
def run_experiment9():
    print("====================================================================")
    print("=== EXPERIMENT 9: EVALUATION PROTOCOL ABLATION (STOCH. vs DET.)  ===")
    print("====================================================================")

    ENV_NAME = "Craftax-Symbolic-v1"
    NUM_ENVS = 256
    NUM_STEPS = 64

    STEPS_PER_UPDATE = NUM_ENVS * NUM_STEPS
    TOTAL_UPDATES = TOTAL_STEPS // STEPS_PER_UPDATE
    LOG_INTERVAL_UPDATES = TRAIN_LOG_INTERVAL_STEPS // STEPS_PER_UPDATE
    EVAL_EVERY_N_LOGS = max(1, EVAL_INTERVAL_STEPS // TRAIN_LOG_INTERVAL_STEPS)



    csv_path = "scripts/diagnostics/benchmarks/craftax_eval_ablation/eval_ablation.csv"

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "Seed", "Total_Steps", "Stochastic_Return", "Stochastic_Achv", "Deterministic_Return", "Deterministic_Achv"
    ])
    csv_file.flush()

    trainer_module.ActorCriticRNN = ActorCriticSdLGHybrid

    config = {
        "NUM_ENVS": NUM_ENVS, "NUM_STEPS": NUM_STEPS, "TOTAL_TIMESTEPS": TOTAL_STEPS,
        "UPDATE_EPOCHS": 4, "NUM_MINIBATCHES": 8, "GAMMA": 0.99, "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2, "ENT_COEF": 0.01, "VF_COEF": 0.5, "MAX_GRAD_NORM": 0.5,
        "LR": 0.0002, "OPTIMISTIC_RESETS": False, "ANNEAL_LR": True,
        "LOG_INTERVAL": LOG_INTERVAL_UPDATES,
        "RNN": {
            "RNN_TYPE": "sdlg_hybrid",
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
    from src.envs.wrappers import LogWrapper
    if not isinstance(env, LogWrapper):
        env = LogWrapper(env)
    action_dim = env.action_space(env_params).n

    network = ActorCriticSdLGHybrid(action_dim=action_dim, config=config)

    def rnn_carry_init(batch_size):
        unbatched = BatchedSdLGHybrid.initialize_carry(
            batch_size=batch_size, n_layers=4, d_k=32, d_v=32, r=4, window_size=8)
        return jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0), unbatched)

    stochastic_eval_fn = build_chunked_evaluator(env, env_params, network, config, is_deterministic=False)
    deterministic_eval_fn = build_chunked_evaluator(env, env_params, network, config, is_deterministic=True)

    for seed in TRAINING_SEEDS:
        jax.clear_caches()
        gc.collect()
        print(f"\n[SdLGHybrid] Initializing Training | SEED = {seed}", flush=True)

        # WANDB INITIALIZATION (FIXED)
        init_wandb(project_override=WANDB_PROJECT, group="exp9_eval_ablation", name=f"EvalAblation-{seed}", config=config, tags=os.environ.get("WANDB_TAGS", "").split(",") if os.environ.get("WANDB_TAGS") else None, notes=os.environ.get("WANDB_NOTES", ""))

        rng = jax.random.PRNGKey(seed)
        raw_update_fn, runner_state = make_update(rng, config, env, env_params, lambda: network, rnn_carry_init)

        @functools.partial(jax.jit, donate_argnums=(0,))
        def compiled_update(r_state):
            out = raw_update_fn(r_state)

            done_mask = out["metric"].get("returned_episode", jnp.ones(1))

            def get_done_mean(x):
                return jnp.sum(x * done_mask) / jnp.maximum(1.0, jnp.sum(done_mask))

            reduced = {
                "train_ret": get_done_mean(out["metric"].get("returned_episode_returns", jnp.zeros(1))),
                "train_ent": jnp.mean(out["metric"].get("entropy", jnp.zeros(1))),
            }
            return out["runner_state"], reduced

        steps_processed = 0
        log_counter = 0

        csv_writer.writerow([seed, 0, 0.0, 0.0, 0.0, 0.0])
        csv_file.flush()

        for update_idx in range(0, TOTAL_UPDATES, LOG_INTERVAL_UPDATES):
            runner_state, reduced_metric = compiled_update(runner_state)
            runner_state = jax.block_until_ready(runner_state)
            reduced_metric = jax.device_get(reduced_metric)

            steps_processed += LOG_INTERVAL_UPDATES * STEPS_PER_UPDATE
            log_counter += 1

            train_ret = float(reduced_metric["train_ret"])
            train_ent = float(reduced_metric["train_ent"])

            wandb_log_dict = {
                "Train/Mean_Return": train_ret,
                "Train/Policy_Entropy": train_ent,
            }

            if log_counter % EVAL_EVERY_N_LOGS == 0 or steps_processed >= TOTAL_STEPS:
                print(f"\n  [EVAL] Step: {steps_processed / 1e6:>5.1f}M -> Running dual ablation...", flush=True)

                # --- RUN STOCHASTIC EVAL ---
                stoch_ret, stoch_ach = [], []
                for chunk in range(8):
                    start_test_seed = 10000 + (chunk * 128)
                    ret_c, ach_dict_c = jax.device_get(
                        stochastic_eval_fn(runner_state[0].params, start_test_seed, rng, 128)
                    )
                    stoch_ret.append(ret_c)
                    stoch_ach.append(sum(ach_dict_c.values()))

                mean_stoch_ret = float(np.mean(np.concatenate(stoch_ret)))
                mean_stoch_ach = float(np.mean(np.concatenate(stoch_ach)))
                print(
                    f"         > Stochastic Protocol:    Ret = {mean_stoch_ret:>5.2f} | Achv = {mean_stoch_ach:>4.1f}")

                # --- RUN DETERMINISTIC EVAL ---
                det_ret, d_ach = [], []
                for chunk in range(8):
                    start_test_seed = 10000 + (chunk * 128)
                    ret_c, ach_dict_c = jax.device_get(
                        deterministic_eval_fn(runner_state[0].params, start_test_seed, rng, 128)
                    )
                    det_ret.append(ret_c)
                    d_ach.append(sum(ach_dict_c.values()))

                mean_det_ret = float(np.mean(np.concatenate(det_ret)))
                mean_det_ach = float(np.mean(np.concatenate(d_ach)))
                print(f"         > Deterministic Protocol: Ret = {mean_det_ret:>5.2f} | Achv = {mean_det_ach:>4.1f}")

                # Log results to CSV
                csv_writer.writerow([
                    seed, steps_processed, mean_stoch_ret, mean_stoch_ach, mean_det_ret, mean_det_ach
                ])
                csv_file.flush()

                # Log results to WandB (FIXED)
                wandb_log_dict.update({
                    "Eval/Stochastic_Return": mean_stoch_ret,
                    "Eval/Stochastic_Achievements": mean_stoch_ach,
                    "Eval/Deterministic_Return": mean_det_ret,
                    "Eval/Deterministic_Achievements": mean_det_ach,
                })

                del stoch_ret, stoch_ach, det_ret, d_ach, ret_c, ach_dict_c
                gc.collect()

            wandb.log(wandb_log_dict, step=steps_processed)
            del wandb_log_dict
            gc.collect()

        wandb.finish()
        jax.clear_caches()
        gc.collect()

    csv_file.close()


# =====================================================================
# 3. VISUALIZATION
# =====================================================================
def plot_experiment9():
    csv_path = "scripts/diagnostics/benchmarks/craftax_eval_ablation/eval_ablation.csv"
    if not os.path.exists(csv_path):
        return

    plt.close('all')
    plt.clf()
    plt.cla()

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    steps = df["Total_Steps"].values / 1e6

    # Panel A: Return Ablation
    axes[0].plot(steps, df["Stochastic_Return"], label="Stochastic Selection (pi.sample)", color="#2c7bb6", linewidth=3)
    axes[0].plot(steps, df["Deterministic_Return"], label="Deterministic Selection (argmax)", color="#d7191c",
                 linewidth=3, linestyle="--")

    # Panel B: Capabilities Ablation
    axes[1].plot(steps, df["Stochastic_Achv"], label="Stochastic Selection (pi.sample)", color="#2c7bb6", linewidth=3)
    axes[1].plot(steps, df["Deterministic_Achv"], label="Deterministic Selection (argmax)", color="#d7191c",
                 linewidth=3, linestyle="--")

    axes[0].set_title("Evaluation Collapse: Episode Returns", fontweight='bold')
    axes[0].set_xlabel("Environment Steps (Millions)")
    axes[0].set_ylabel("Mean Episode Return")
    axes[0].legend(frameon=True)

    axes[1].set_title("Evaluation Collapse: Environmental Milestones", fontweight='bold')
    axes[1].set_xlabel("Environment Steps (Millions)")
    axes[1].set_ylabel("Total Achievements Unlocked")
    axes[1].legend(frameon=True)

    plt.tight_layout()
    plot_path = "scripts/diagnostics/benchmarks/craftax_eval_ablation/eval_ablation.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"\nRendered Ablation Plot to {plot_path}")

    import shutil
    os.makedirs("plots", exist_ok=True)
    shutil.copy(plot_path, "plots/exp9_eval_ablation.png")
    shutil.copy(plot_path, "exp9_eval_ablation.png")
    print("Saved Copies to plots/exp9_eval_ablation.png and exp9_eval_ablation.png")


if __name__ == "__main__":
    run_experiment9()
    plot_experiment9()