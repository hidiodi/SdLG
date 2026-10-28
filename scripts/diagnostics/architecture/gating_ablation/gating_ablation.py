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
from typing import Any
from flax.linen.initializers import constant, orthogonal

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
from src.models.common.sdlg_hybrid import (
    BatchedSdLGHybrid,
    rope_rotate,
    parallel_recurrent_scan
)
import src.models.common.sdlg_hybrid as sdlg_hybrid_module  # Import module namespace for dynamic class swapping
from src.trainers.purejax_trainer import make_update, flush_carry_state
import src.trainers.purejax_trainer as trainer_module

from craftax.craftax.constants import Achievement

ACH_KEYS = [ach.name for ach in Achievement]

# =====================================================================
# --- CENTRALIZED WANDB MANAGEMENT ---
# =====================================================================
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "SdLGHybrid-Exp11")
WANDB_TAGS = os.environ.get("WANDB_TAGS", "").split(",") if os.environ.get("WANDB_TAGS") else []
WANDB_NOTES = os.environ.get("WANDB_NOTES", "")

# =====================================================================
# --- DYNAMIC MONKEY-PATCH: THE DECAY MECHANISM SHOWDOWN ---
# =====================================================================
CURRENT_DECAY_TYPE = "sigmoid"  # Modulated dynamically in the run loop


class PatchedGatedAttentionGLALayer(nn.Module):
    """
    First-class Flax Module defined at class-compilation time.
    Safely compiles under Flax's metaclass to allow dynamic self.param allocations.
    """
    d_model: int
    d_k: int
    d_v: int
    r: int
    eta: float = 0.05
    eps: float = 1e-5
    use_rope_phase: bool = True
    use_softplus_denominator: bool = True
    use_per_rank_norm: bool = True

    @nn.compact
    def __call__(self, inputs, terminations, last_memory, carry_step_count):
        T, _ = inputs.shape
        tilde_k_prev, tilde_v_prev, theta_prev = last_memory

        omegas = jnp.linspace(-jnp.pi, jnp.pi, self.r)

        def compute_step_count(carry, term):
            next_step = (carry + 1.0) * (1.0 - term)
            return next_step, next_step

        def compute_phases(carry_theta, term):
            resetted_theta = carry_theta * (1.0 - term)
            next_theta = (resetted_theta + omegas) % (2 * jnp.pi)
            return next_theta, next_theta

        if T > 1:
            _, step_counts = jax.lax.scan(compute_step_count, carry_step_count, terminations)
            _, theta = jax.lax.scan(compute_phases, theta_prev, terminations)
        else:
            step_counts = jnp.expand_dims((carry_step_count + 1.0) * (1.0 - terminations[0]), axis=0)
            resetted_theta = theta_prev * (1.0 - terminations[0])
            theta = jnp.expand_dims((resetted_theta + omegas) % (2 * jnp.pi), axis=0)

        total_gla_dims = 3 * self.d_k + 2 * self.d_v
        joint_gla = nn.Dense(total_gla_dims, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0),
                             name="joint_gla_proj")(inputs)
        split_idx = [self.d_k, 2 * self.d_k, 2 * self.d_k + self.d_v, 2 * self.d_k + 2 * self.d_v]
        q, k, v, beta_raw, gamma_raw = jnp.split(joint_gla, split_idx, axis=-1)

        # --- THE ABLATION SHOWDOWN ROUTING ---
        global CURRENT_DECAY_TYPE
        if CURRENT_DECAY_TYPE == "sigmoid":
            beta = jax.nn.sigmoid(beta_raw)
            gamma = jax.nn.sigmoid(gamma_raw)
            g_v = 1.0 - beta
            g_k = 1.0 - gamma

        elif CURRENT_DECAY_TYPE == "ssm":
            # Native Parameter Creation (Now 100% compiled-safe inside @nn.compact class)
            A_k_param = self.param('A_k', constant(0.5413), (self.d_k,))
            A_v_param = self.param('A_v', constant(0.5413), (self.d_v,))
            A_k = jax.nn.softplus(A_k_param)
            A_v = jax.nn.softplus(A_v_param)

            delta_k = jnp.clip(jax.nn.softplus(gamma_raw), a_min=1e-4, a_max=10.0)
            delta_v = jnp.clip(jax.nn.softplus(beta_raw), a_min=1e-4, a_max=10.0)

            g_k = jnp.exp(-delta_k * A_k)
            g_v = jnp.exp(-delta_v * A_v)

            beta = 1.0 - g_v
            gamma = 1.0 - g_k

        elif CURRENT_DECAY_TYPE == "static":
            g_v = jnp.full_like(beta_raw, 0.99)
            g_k = jnp.full_like(gamma_raw, 0.99)
            beta = jnp.full_like(beta_raw, 0.01)
            gamma = jnp.full_like(gamma_raw, 0.01)

        else:
            raise ValueError(f"Unknown decay type: {CURRENT_DECAY_TYPE}")

        if self.use_rope_phase:
            q_rot = rope_rotate(q[..., None, :], theta)
            k_rot = rope_rotate(k[..., None, :], theta)
            write_v = jnp.cos(theta)[..., None] * (beta * v)[..., None, :]
            write_k = gamma[..., None, :] * k_rot
        else:
            abs_theta = step_counts[..., None] * omegas
            q_rot = jnp.cos(abs_theta)[..., None] * q[..., None, :]
            k_rot = jnp.cos(abs_theta)[..., None] * k[..., None, :]
            write_v = jnp.cos(abs_theta)[..., None] * (beta * v)[..., None, :]
            write_k = gamma[..., None, :] * k_rot

        discount_v = g_v[..., None, :] * (1.0 - terminations[..., None, None])
        discount_k = g_k[..., None, :] * (1.0 - terminations[..., None, None])

        if T > 1:
            final_tilde_k = parallel_recurrent_scan(tilde_k_prev, write_k, discount_k)
            final_tilde_v = parallel_recurrent_scan(tilde_v_prev, write_v, discount_v)
        else:
            final_tilde_k = jnp.expand_dims(discount_k[0] * tilde_k_prev + write_k[0], axis=0)
            final_tilde_v = jnp.expand_dims(discount_v[0] * tilde_v_prev + write_v[0], axis=0)

        dot_prod = jnp.sum(q_rot * final_tilde_k, axis=-1)
        num = jnp.sum(final_tilde_v * dot_prod[..., None], axis=-2)

        if self.use_per_rank_norm:
            den_sum = jnp.sum(final_tilde_k * q_rot, axis=(-2, -1))
        else:
            den_sum = jnp.sum(jnp.mean(final_tilde_k, axis=-2) * jnp.mean(q_rot, axis=-2), axis=-1)

        den = jax.nn.softplus(den_sum) + self.eps if self.use_softplus_denominator else jnp.maximum(den_sum, self.eps)
        attn_out = num / den[..., None]

        step_weight = jnp.clip((step_counts - 32) / 32, 0.0, 1.0)
        intrinsic_bonus = step_weight * (self.eta / jax.lax.stop_gradient(den))

        attn_out = nn.Dense(self.d_model, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0),
                            name="out_proj")(attn_out)
        g_out = jax.nn.sigmoid(
            nn.Dense(self.d_model, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="g_proj")(inputs))
        out = g_out * attn_out

        return out, intrinsic_bonus, (final_tilde_k[-1], final_tilde_v[-1], theta[-1]), step_counts[-1]


# Mutate the global class reference inside the imported module's namespace
sdlg_hybrid_module.GatedAttentionGLALayer = PatchedGatedAttentionGLALayer

# =====================================================================
# --- EXPERIMENT CONFIGURATION ---
# =====================================================================
TOTAL_STEPS = 15_000_000  # 15M Steps per model to capture the shift
TRAIN_LOG_INTERVAL_STEPS = 500_000  # Push to CSV every 500k steps
EVAL_INTERVAL_STEPS = TOTAL_STEPS  # Only evaluate at the absolute final step

MODELS_TO_RUN = [ "SdLG-MambaSSM", "SdLG-Static","SdLG-Sigmoid"]
TRAINING_SEEDS = [42, 54, 99, 123, 456]


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
# 3. EXPERIMENT RUNNER
# =====================================================================
def run_experiment11():
    print("====================================================================")
    print("=== EXPERIMENT 11: THE DECAY MECHANISM SHOWDOWN                  ===")
    print("====================================================================")

    ENV_NAME = "Craftax-Symbolic-v1"
    NUM_ENVS = 256
    NUM_STEPS = 64

    STEPS_PER_UPDATE = NUM_ENVS * NUM_STEPS
    TOTAL_UPDATES = TOTAL_STEPS // STEPS_PER_UPDATE
    LOG_INTERVAL_UPDATES = TRAIN_LOG_INTERVAL_STEPS // STEPS_PER_UPDATE
    EVAL_EVERY_N_LOGS = max(1, EVAL_INTERVAL_STEPS // TRAIN_LOG_INTERVAL_STEPS)

    out_dir = "results/diagnostics/exp11"
    plot_dir = "plots/exp11"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    csv_path = "scripts/diagnostics/architecture/gating_ablation/gating_ablation.csv"

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    header = [
                 "Architecture", "Seed", "Total_Steps", "Wall_Clock_Time", "SPS",
                 "Train_Return", "Train_Entropy", "Train_VLoss_Var", "Eval_Return"
             ] + [f"Train_{k}" for k in ACH_KEYS] + [f"Eval_{k}" for k in ACH_KEYS]

    csv_writer.writerow(header)
    csv_file.flush()

    trainer_module.ActorCriticRNN = ActorCriticSdLGHybrid

    global CURRENT_DECAY_TYPE

    for model_name in MODELS_TO_RUN:

        # Route the architectural math dynamically BEFORE init
        if "Sigmoid" in model_name:
            CURRENT_DECAY_TYPE = "sigmoid"
        elif "Mamba" in model_name:
            CURRENT_DECAY_TYPE = "ssm"
        else:
            CURRENT_DECAY_TYPE = "static"

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

        chunked_eval_fn = build_chunked_evaluator(env, env_params, network, config)

        for seed in TRAINING_SEEDS:
            jax.clear_caches()
            gc.collect()
            print(f"\n[{model_name}] Initializing Compilation & Training | SEED = {seed}", flush=True)

            wandb_tags = WANDB_TAGS + [model_name]
            init_wandb(
                project=WANDB_PROJECT,
                group=model_name,
                name=f"{model_name}-Seed-{seed}",
                config=config,
                tags=wandb_tags,
                notes=WANDB_NOTES if WANDB_NOTES else None
            )

            rng = jax.random.PRNGKey(seed)
            raw_update_fn, runner_state = make_update(rng, config, env, env_params, lambda: network, rnn_carry_init)

            @functools.partial(jax.jit, donate_argnums=(0,))
            def compiled_update_and_reduce(r_state):
                out = raw_update_fn(r_state)
                m = out["metric"]

                done_mask = m.get("returned_episode", jnp.ones_like(m.get("entropy", jnp.zeros(1))))

                def get_done_mean(x):
                    return jnp.sum(x * done_mask) / jnp.maximum(1.0, jnp.sum(done_mask))

                reduced = {
                    "train_ret": get_done_mean(m.get("returned_episode_returns", jnp.zeros_like(done_mask))),
                    "train_ent": jnp.mean(m.get("entropy", jnp.zeros(1))),
                    "train_vloss_var": jnp.var(m.get("value_loss", jnp.zeros(1))),
                }

                native_achievements = out["runner_state"][1].env_state.achievements
                for i, name in enumerate(ACH_KEYS):
                    reduced[f"train_{name}"] = jnp.mean(native_achievements[:, i].astype(jnp.float32))

                return out["runner_state"], reduced

            # --- SAFE LOOP BOUNDING FIX ---
            NUM_LOG_LOOPS = TOTAL_UPDATES // LOG_INTERVAL_UPDATES

            start_wall_time = time.time()
            cumulative_eval_time = 0.0

            row_0 = [model_name, seed, 0, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan] + [0.0] * len(ACH_KEYS) + [np.nan] * len(
                ACH_KEYS)
            csv_writer.writerow(row_0)
            csv_file.flush()

            for loop_idx in range(NUM_LOG_LOOPS):
                runner_state, reduced_metric = compiled_update_and_reduce(runner_state)
                runner_state = jax.block_until_ready(runner_state)
                reduced_metric = jax.device_get(reduced_metric)

                steps_processed = (loop_idx + 1) * LOG_INTERVAL_UPDATES * STEPS_PER_UPDATE
                log_counter = loop_idx + 1

                train_ret = float(reduced_metric["train_ret"])
                train_ent = float(reduced_metric["train_ent"])
                train_vloss_var = float(reduced_metric["train_vloss_var"])

                active_train_time = time.time() - start_wall_time - cumulative_eval_time
                sps = steps_processed / max(active_train_time, 1.0)

                wandb_log_dict = {
                    "System/SPS": sps,
                    "System/Wall_Clock_Time": active_train_time,
                    "Train/Mean_Return": train_ret,
                    "Train/Policy_Entropy": train_ent,
                    "Train/Value_Loss_Var": train_vloss_var,
                }
                for k in ACH_KEYS:
                    wandb_log_dict[f"Train_Achv/{k}"] = float(reduced_metric[f"train_{k}"])

                # Check for eval phase
                if log_counter % EVAL_EVERY_N_LOGS == 0 or steps_processed >= TOTAL_STEPS:
                    print(f"\n  [EVAL] Step: {steps_processed / 1e6:>5.1f}M -> Evaluating {model_name}...", flush=True)
                    eval_start = time.time()

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

                    wandb_log_dict["Eval/Mean_Return"] = mean_test_ret
                    for k in ACH_KEYS:
                        wandb_log_dict[f"Eval_Achv/{k}"] = mean_eval_ach[k]

                    total_achv = sum(mean_eval_ach.values())
                    print(
                        f"         > Ret: {mean_test_ret:>5.2f} | Total Achv: {total_achv:>4.1f} | Var(VF): {train_vloss_var:>6.2f}")

                    row = [
                        model_name, seed, steps_processed, active_train_time, sps,
                        train_ret, train_ent, train_vloss_var, mean_test_ret
                    ]
                    row.extend([float(reduced_metric[f"train_{k}"]) for k in ACH_KEYS])
                    row.extend([mean_eval_ach[k] for k in ACH_KEYS])

                    csv_writer.writerow(row)
                    csv_file.flush()

                    del eval_returns, eval_ach_chunks, ret_c, max_ach_c
                    gc.collect()

                else:
                    print(
                        f"  [TRAIN] Step: {steps_processed / 1e6:>5.1f}M | Ret: {train_ret:>5.1f} | Var(VF): {train_vloss_var:>6.2f} | SPS: {int(sps):>5}",
                        flush=True)

                    row = [
                        model_name, seed, steps_processed, active_train_time, sps,
                        train_ret, train_ent, train_vloss_var, np.nan
                    ]
                    row.extend([float(reduced_metric[f"train_{k}"]) for k in ACH_KEYS])
                    row.extend([np.nan for k in ACH_KEYS])

                    csv_writer.writerow(row)
                    csv_file.flush()

                wandb.log(wandb_log_dict, step=steps_processed)
                del wandb_log_dict
                gc.collect()

            wandb.finish()

            # AGGRESSIVE MEMORY PURGE
            print(f"\n[SYSTEM] Purging memory buffers for {model_name}...")
            del runner_state, compiled_update_and_reduce, raw_update_fn, network, env, chunked_eval_fn
            jax.clear_caches()
            gc.collect()
            time.sleep(3)

    csv_file.close()


# =====================================================================
# 4. VISUALIZATION (BAR CHART REFACTOR)
# =====================================================================
def plot_experiment11():
    csv_path = "scripts/diagnostics/architecture/gating_ablation/gating_ablation.csv"
    if not os.path.exists(csv_path):
        return

    plt.close('all')
    plt.clf()
    plt.cla()

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"SdLG-Sigmoid": "#2c7bb6", "SdLG-MambaSSM": "#d7191c", "SdLG-Static": "#fdae61"}
    styles = {"SdLG-Sigmoid": "-", "SdLG-MambaSSM": "--", "SdLG-Static": ":"}

    for model_name in MODELS_TO_RUN:
        model_data = df[df["Architecture"] == model_name]
        if model_data.empty: continue

        grp = model_data.groupby("Total_Steps")
        mean_df = grp.mean(numeric_only=True)
        steps = mean_df.index.values / 1e6

        # Panel A: Return (Using dropna to safely plot the 5M step intervals)
        eval_rows = model_data.dropna(subset=["Eval_Return"])
        if not eval_rows.empty:
            final_ret = eval_rows["Eval_Return"].iloc[-1]
            axes[0].bar(model_name, final_ret, color=colors[model_name], edgecolor='black', linewidth=1.5)
            axes[0].text(model_name, final_ret + 0.1, f"{final_ret:.2f}", ha='center', va='bottom', fontweight='bold',
                         fontsize=11)

        # Panel B: Value Loss Volatility (Log Scale Line Chart)
        safe_vloss = np.maximum(mean_df["Train_VLoss_Var"].values, 1e-8)
        axes[1].plot(steps, safe_vloss, label=model_name, color=colors[model_name], linestyle=styles[model_name],
                     linewidth=2.5, alpha=0.85)

    axes[0].set_title("Decay Ablation: Final Generalization Return", fontweight='bold')
    axes[0].set_ylabel("Mean Episode Return (1,024 Worlds)")
    axes[0].tick_params(axis='x', labelsize=10)

    axes[1].set_title(r"Optimization Stability: Value Loss Variance ($\sigma^2_{VF}$)", fontweight='bold')
    axes[1].set_xlabel("Environment Steps (Millions)")
    axes[1].set_ylabel("Variance (Log Scale)")
    axes[1].set_yscale('log')
    axes[1].legend(frameon=True)

    plt.tight_layout()
    plot_path = "scripts/diagnostics/architecture/gating_ablation/gating_ablation.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"\nRendered Decay Ablation Plot to {plot_path}")

    # ==========================================
    # LOG FINAL PLOT TO WANDB SUMMARY RUN
    # ==========================================
    print("Uploading final summary plot to W&B...")
    init_wandb(
        project=WANDB_PROJECT,
        group="Summary",
        name="Exp11-Final-Plot",
        tags=WANDB_TAGS + ["summary", "plot"],
        notes="Final comparative plot across all ablation conditions."
    )
    wandb.log({"Ablation_Results/Final_Comparison_Plot": wandb.Image(plot_path)})
    wandb.finish()


if __name__ == "__main__":
    run_experiment11()
    plot_experiment11()