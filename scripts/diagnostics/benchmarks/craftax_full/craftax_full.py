# scripts/diagnostics/benchmarks/craftax_full/craftax_watchdog_benchmark.py

import sys
import os

# Ensure path includes the project root
current_file_path = os.path.abspath(__file__)
project_root = current_file_path
while not os.path.exists(os.path.join(project_root, "src")):
    project_root = os.path.dirname(project_root)
    if project_root == "/":
        break
if project_root not in sys.path:
    sys.path.append(project_root)

os.environ["HSA_ENABLE_SDMA"] = "0"

import jax
import jax.numpy as jnp
import flax.linen as nn
import flax
import numpy as np
import pandas as pd
import csv
import time
import functools
import wandb
import signal
import gc
import orbax.checkpoint as ocp
from pathlib import Path
from typing import Any
from src.models.common.gru import BatchedGRU
from src.models.common.agalite import BatchedAGaLiTe

from scripts.diagnostics.core.utils import init_wandb
from src.envs.builder import build_purejaxrl_env
from src.utils.config_schema import StaticEnvConfig, DynamicTrainingConfig
import distrax

from src.models.purejaxrl.wrappers import RepresentationNetwork, ActorCriticSdLGHybrid
from src.models.common.gtrxl import BatchedGTrXL, GatedTransformerXLLayer
from src.models.common.sdlg_hybrid import BatchedSdLGHybrid
from src.trainers.purejax_trainer import make_update
import src.trainers.purejax_trainer as trainer_module

from craftax.craftax.constants import Achievement

ACH_KEYS = [ach.name for ach in Achievement]


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

    # [FIX] Force GELU activation for the GTrXL MLP to match baselines exactly
    m2 = self.dropout_fn(nn.Dense(self.mlp.features)(jax.nn.gelu(nn.Dense(self.mlp.features)(x2)))) if not hasattr(
        self.mlp, 'layers') else self.dropout_fn(self.mlp(x2))

    o2 = self.gate2(o1, m2) if self.gating else o1 + m2
    return o2


GatedTransformerXLLayer.__call__ = patched_gtrxl_call


def patched_flush_carry_state(hstate, last_done):
    def flush_leaf(x):
        dims_to_add = x.ndim - last_done.ndim
        broadcast_done = jnp.expand_dims(last_done, axis=tuple(range(1, 1 + dims_to_add)))
        out = x * (1.0 - broadcast_done)
        return out.astype(x.dtype)

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

# =====================================================================
# --- EXPERIMENT CONFIGURATION: 1 BILLION STEPS (50M PER WATCHDOG SESSION) ---
# =====================================================================
TOTAL_STEPS = 1_000_000_000
TRAIN_LOG_INTERVAL_STEPS = 1_000_000
SESSION_STEP_LIMIT = 50_000_000

NUM_ENVS = 256
TARGET_MINIBATCH_SIZE = 4096
TRAINING_SEEDS = [42]

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "Craftax-Full-Benchmark")

RUN_CONFIGS = [
    ("SdLGHybrid", 16),
]

class ActorCriticGTrXL(nn.Module):
    rnn_module: Any
    action_dim: int
    config: dict

    @nn.compact
    def __call__(self, carry, x):
        obs, dones = x
        emb = RepresentationNetwork(d_model=self.config["RNN"]["D_MODEL"], name="repr_network")(obs)
        gtrxl = BatchedGTrXL(
            head_dim=self.config["RNN"].get("D_HEAD", 32),
            embedding_dim=self.config["RNN"]["D_MODEL"],
            head_num=self.config["RNN"].get("N_HEADS", 8),
            mlp_num=2, layer_num=self.config["RNN"].get("N_LAYERS", 4),
            memory_len=self.config["RNN"].get("MEMORY_LEN", 128),
            dropout_ratio=0.0, gru_gating=True, use_embedding_layer=False,
        )
        new_carry, outs = gtrxl(carry, (emb, dones))

        action_logits = nn.Dense(self.action_dim, kernel_init=flax.linen.initializers.orthogonal(0.01))(outs)
        # Unbounded logits to match the 1B baseline
        action_logits = jnp.nan_to_num(action_logits, nan=0.0, posinf=10.0, neginf=-10.0)

        critic_raw = nn.Dense(1, kernel_init=flax.linen.initializers.orthogonal(1.0))(outs)
        critic_values = jnp.nan_to_num(jnp.squeeze(critic_raw, axis=-1), nan=0.0, posinf=10.0, neginf=-10.0)

        pi = distrax.Categorical(logits=action_logits)
        return new_carry, pi, critic_values

class ActorCriticDense(nn.Module):
    rnn_module: Any
    action_dim: int
    config: dict

    @nn.compact
    def __call__(self, carry, x):
        obs, dones = x
        emb = RepresentationNetwork(d_model=self.config["RNN"]["D_MODEL"], name="repr_network")(obs)

        # No recurrence, the embedding goes straight to the heads
        action_logits = nn.Dense(self.action_dim, kernel_init=flax.linen.initializers.orthogonal(0.01))(emb)
        action_logits = 5.0 * jnp.tanh(jnp.nan_to_num(action_logits, nan=0.0, posinf=5.0, neginf=-5.0))

        critic_raw = nn.Dense(1, kernel_init=flax.linen.initializers.orthogonal(1.0))(emb)
        critic_values = jnp.nan_to_num(jnp.squeeze(critic_raw, axis=-1), nan=0.0, posinf=10.0, neginf=-10.0)

        pi = distrax.Categorical(logits=action_logits)
        return carry, pi, critic_values

class ActorCriticGRU(nn.Module):
    rnn_module: Any
    action_dim: int
    config: dict

    @nn.compact
    def __call__(self, carry, x):
        obs, dones = x
        emb = RepresentationNetwork(d_model=self.config["RNN"]["D_MODEL"], name="repr_network")(obs)

        # Standard GRU Unroll
        new_carry, outs = BatchedGRU()(carry, (emb, dones))

        action_logits = nn.Dense(self.action_dim, kernel_init=flax.linen.initializers.orthogonal(0.01))(outs)
        action_logits = 5.0 * jnp.tanh(jnp.nan_to_num(action_logits, nan=0.0, posinf=5.0, neginf=-5.0))

        critic_raw = nn.Dense(1, kernel_init=flax.linen.initializers.orthogonal(1.0))(outs)
        critic_values = jnp.nan_to_num(jnp.squeeze(critic_raw, axis=-1), nan=0.0, posinf=10.0, neginf=-10.0)

        pi = distrax.Categorical(logits=action_logits)
        return new_carry, pi, critic_values

class ActorCriticAGaLiTe(nn.Module):
    rnn_module: Any
    action_dim: int
    config: dict

    @nn.compact
    def __call__(self, carry, x):
        obs, dones = x
        emb = RepresentationNetwork(d_model=self.config["RNN"]["D_MODEL"], name="repr_network")(obs)

        agalite = BatchedAGaLiTe(
            n_layers=self.config["RNN"].get("N_LAYERS", 4),
            d_model=self.config["RNN"]["D_MODEL"],
            d_head=self.config["RNN"].get("D_HEAD", 32),
            d_ffc=self.config["RNN"].get("D_FFC", 1024),
            n_heads=self.config["RNN"].get("N_HEADS", 8),
            eta=int(self.config["RNN"].get("ETA", 1)),  # [FIX] Force integer for shape calculation
            r=int(self.config["RNN"].get("R", 4)),      # [FIX] Force integer for shape calculation
            reset_on_terminate=True
        )
        new_carry, outs = agalite(carry, (emb, dones))

        action_logits = nn.Dense(self.action_dim, kernel_init=flax.linen.initializers.orthogonal(0.01))(outs)
        action_logits = 5.0 * jnp.tanh(jnp.nan_to_num(action_logits, nan=0.0, posinf=5.0, neginf=-5.0))

        critic_raw = nn.Dense(1, kernel_init=flax.linen.initializers.orthogonal(1.0))(outs)
        critic_values = jnp.nan_to_num(jnp.squeeze(critic_raw, axis=-1), nan=0.0, posinf=10.0, neginf=-10.0)

        pi = distrax.Categorical(logits=action_logits)
        return new_carry, pi, critic_values

def run_benchmark():
    print("====================================================================")
    print("=== CRAFTAX-1B WATCHDOG BENCHMARK (1 BILLION STEPS | T=128) ===")
    print("====================================================================")

    for model_name, NUM_STEPS in RUN_CONFIGS:
        print(f"\n=======================================================")
        print(f"=== INITIALIZING RUN: {model_name} at T={NUM_STEPS} ===")
        print(f"=======================================================")

        seed = TRAINING_SEEDS[0]

        NUM_MINIBATCHES = (NUM_ENVS * NUM_STEPS) // TARGET_MINIBATCH_SIZE
        STEPS_PER_UPDATE = NUM_ENVS * NUM_STEPS
        TOTAL_UPDATES = TOTAL_STEPS // STEPS_PER_UPDATE
        LOG_INTERVAL_UPDATES = TRAIN_LOG_INTERVAL_STEPS // STEPS_PER_UPDATE

        CHECKPOINT_BASE_DIR = os.path.abspath(f"checkpoints/craftax_1b_watchdog_T{NUM_STEPS}_1B")
        CSV_PATH = os.path.join(CHECKPOINT_BASE_DIR, f"{model_name}_T{NUM_STEPS}_log.csv")
        wandb_group = f"1B_Benchmark_T{NUM_STEPS}_1B"
        wandb_name = f"{model_name}-{seed}-T{NUM_STEPS}"
        wandb_run_id = f"craftax1b_{model_name}_T{NUM_STEPS}_1B_seed_{seed}".replace(".", "_").lower()

        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        os.makedirs(CHECKPOINT_BASE_DIR, exist_ok=True)

        ckpt_dir = Path(CHECKPOINT_BASE_DIR) / f"{model_name}_seed_{seed}"

        options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
        checkpointer = ocp.StandardCheckpointer()
        ckpt_manager = ocp.CheckpointManager(ckpt_dir.resolve(), checkpointer, options=options)

        latest_step = ckpt_manager.latest_step()
        if latest_step is not None:
            start_loop_idx = latest_step // LOG_INTERVAL_UPDATES
            if start_loop_idx >= (TOTAL_UPDATES // LOG_INTERVAL_UPDATES):
                print(f"  -> Run {model_name} (T={NUM_STEPS}) is completely finished! Skipping.")
                continue

        file_mode = "a" if (latest_step is not None) else "w"
        csv_file = open(CSV_PATH, file_mode, newline="")
        csv_writer = csv.writer(csv_file)

        if not latest_step or os.path.getsize(CSV_PATH) == 0:
            header = ["Architecture", "Seed", "Total_Steps", "Wall_Clock_Time", "SPS", "Train_Return", "Train_Entropy",
                      "Train_VLoss"] + [f"Train_{k}" for k in ACH_KEYS]
            csv_writer.writerow(header)
            csv_file.flush()

        if model_name == "PPO-Dense":
            trainer_module.ActorCriticRNN = ActorCriticDense
            rnn_type = "dense"

            def rnn_carry_init(bs):
                return jnp.zeros((bs, 1))  # Dummy carry

        elif model_name == "PPO-GRU":
            trainer_module.ActorCriticRNN = ActorCriticGRU
            rnn_type = "gru"

            def rnn_carry_init(bs):
                return BatchedGRU.initialize_carry(bs, config["RNN"]["D_MODEL"])

        elif model_name == "AGaLiTe":
                trainer_module.ActorCriticRNN = ActorCriticAGaLiTe
                config["RNN"]["RNN_TYPE"] = "agalite"
                config["RNN"]["ETA"] = 1
                config["RNN"]["R"] = int(config["RNN"].get("R", 4))
                def rnn_carry_init(bs):
                    return BatchedAGaLiTe.initialize_carry(
                        bs, config["RNN"]["N_LAYERS"], config["RNN"]["N_HEADS"],
                        config["RNN"]["D_HEAD"], config["RNN"]["ETA"], config["RNN"]["R"]
                    )

        elif model_name == "PPO-GTrXLv8":

            trainer_module.ActorCriticRNN = ActorCriticGTrXL
            rnn_type = "gtrxl"
        elif "SdLGHybrid" in model_name:
            trainer_module.ActorCriticRNN = ActorCriticSdLGHybrid
            rnn_type = "sdlg_hybrid"

        config = {
            "NUM_ENVS": NUM_ENVS, "NUM_STEPS": NUM_STEPS, "TOTAL_TIMESTEPS": TOTAL_STEPS,
            "UPDATE_EPOCHS": 4, "NUM_MINIBATCHES": NUM_MINIBATCHES, "GAMMA": 0.99, "GAE_LAMBDA": 0.95,
            "CLIP_EPS": 0.2, "ENT_COEF": 0.01, "VF_COEF": 0.5, "MAX_GRAD_NORM": 0.5,
            "LR": 0.0002, "OPTIMISTIC_RESETS": False, "ANNEAL_LR": True,
            "LOG_INTERVAL": LOG_INTERVAL_UPDATES,
            "RNN": {
                "RNN_TYPE": rnn_type,
                "N_LAYERS": 4, "D_MODEL": 256, "D_HEAD": 32, "N_HEADS": 8,
                "MEMORY_LEN": 128, "D_K": 32, "D_V": 32, "D_FFC": 1024,
                "R": 4, "WINDOW_SIZE": NUM_STEPS, "ALPHA": 0.1, "ETA": 0.0,
            }
        }

        static_env = StaticEnvConfig(env_name="Craftax-Symbolic-v1", num_envs=NUM_ENVS, num_steps=NUM_STEPS,
                                     max_episode_steps=10000, optimistic_resets=False)
        dynamic_train = DynamicTrainingConfig(total_timesteps=TOTAL_STEPS, learning_rate=0.0002, gamma=0.99,
                                              gae_lambda=0.95, clip_eps=0.2, ent_coef=0.01, vf_coef=0.5,
                                              max_grad_norm=0.5, update_epochs=4, minibatch_size=TARGET_MINIBATCH_SIZE,
                                              log_interval=LOG_INTERVAL_UPDATES)

        env, env_params = build_purejaxrl_env(static_env, dynamic_train)
        from src.envs.wrappers import LogWrapper
        if not isinstance(env, LogWrapper): env = LogWrapper(env)
        action_dim = env.action_space(env_params).n

        if rnn_type == "sdlg_hybrid":
            network = ActorCriticSdLGHybrid(action_dim=action_dim, config=config)

            def rnn_carry_init(bs):
                return jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), bs, axis=0),
                                              BatchedSdLGHybrid.initialize_carry(batch_size=bs, n_layers=4, d_k=32,
                                                                                 d_v=32, r=4, window_size=NUM_STEPS))

        elif rnn_type == "dense":
            network = ActorCriticDense(rnn_module=None, action_dim=action_dim, config=config)

            def rnn_carry_init(bs):
                return jnp.zeros((bs, 1))

        elif rnn_type == "gru":
            network = ActorCriticGRU(rnn_module=None, action_dim=action_dim, config=config)

            def rnn_carry_init(bs):
                return BatchedGRU.initialize_carry(bs, config["RNN"]["D_MODEL"])

        elif rnn_type == "agalite":
            network = ActorCriticAGaLiTe(rnn_module=None, action_dim=action_dim, config=config)

            def rnn_carry_init(bs):
                return BatchedAGaLiTe.initialize_carry(bs, config["RNN"]["N_LAYERS"], config["RNN"]["N_HEADS"],
                                                       config["RNN"]["D_HEAD"], config["RNN"].get("ETA", 1),
                                                       config["RNN"].get("R", 1))

        else:
            network = ActorCriticGTrXL(rnn_module=None, action_dim=action_dim, config=config)

            def rnn_carry_init(bs):
                return BatchedGTrXL.initialize_carry(bs, memory_len=128, embedding_dim=256, layer_num=4)

        jax.clear_caches()
        gc.collect()

        rng = jax.random.PRNGKey(seed)
        raw_update_fn, runner_state = make_update(rng, config, env, env_params, lambda: network, rnn_carry_init)

        start_loop_idx = 0
        if latest_step is not None:
            print(f"  [CHECKPOINT] Resuming from Update Step {latest_step}...", flush=True)
            runner_state = \
            ckpt_manager.restore(latest_step, args=ocp.args.StandardRestore({"runner_state": runner_state}))[
                "runner_state"]
            start_loop_idx = latest_step // LOG_INTERVAL_UPDATES
        else:
            print("  [CHECKPOINT] Starting from scratch.", flush=True)

        os.environ["WANDB_RUN_ID"] = wandb_run_id

        init_wandb(
            project_override=WANDB_PROJECT,
            group=wandb_group,
            name=wandb_name,
            config=config, model_name=model_name, resume="allow"
        )

        @functools.partial(jax.jit, donate_argnums=(0,))
        def compiled_update_and_reduce(r_state):
            out = raw_update_fn(r_state)
            m = out["metric"]
            done_mask = m.get("returned_episode", jnp.ones_like(m.get("entropy", jnp.zeros(1))))

            reduced = {
                "train_ret": jnp.sum(
                    m.get("returned_episode_returns", jnp.zeros_like(done_mask)) * done_mask) / jnp.maximum(1.0,
                                                                                                            jnp.sum(
                                                                                                                done_mask)),
                "train_len": jnp.sum(
                    m.get("returned_episode_lengths", jnp.zeros_like(done_mask)) * done_mask) / jnp.maximum(1.0,
                                                                                                            jnp.sum(
                                                                                                                done_mask)),
                "train_ent": jnp.mean(m.get("entropy", jnp.zeros(1))),
                "train_vloss": jnp.mean(m.get("value_loss", jnp.zeros(1))),
                "max_grad_norm": jnp.max(m.get("max_gradient_norm", jnp.zeros(1))),
                "nan_detected": jnp.any(m.get("parameter_nan_detected", jnp.zeros(1, dtype=bool))),
            }
            native_achievements = out["runner_state"][1].env_state.achievements
            for i, name in enumerate(ACH_KEYS): reduced[f"train_{name}"] = jnp.mean(
                native_achievements[:, i].astype(jnp.float32))
            return out["runner_state"], reduced

        NUM_LOG_LOOPS = TOTAL_UPDATES // LOG_INTERVAL_UPDATES
        steps_processed = start_loop_idx * LOG_INTERVAL_UPDATES * STEPS_PER_UPDATE
        start_wall_time = time.time()
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
                print(f"\n[FATAL ERROR] NaN detected in model weights! Exiting.")
                wandb.log({"System/Crash": 1}, step=steps_processed)
                wandb.finish()
                sys.exit(42)

            steps_processed += LOG_INTERVAL_UPDATES * STEPS_PER_UPDATE
            current_update_idx = (loop_idx + 1) * LOG_INTERVAL_UPDATES

            active_train_time = time.time() - start_wall_time
            session_steps_processed = (loop_idx - start_loop_idx + 1) * LOG_INTERVAL_UPDATES * STEPS_PER_UPDATE
            sps = session_steps_processed / max(active_train_time, 1.0)

            train_ret = float(reduced_metric["train_ret"])
            train_total_achv = sum(float(reduced_metric[f"train_{k}"]) for k in ACH_KEYS)

            wandb_log_dict = {
                "System/SPS": sps,
                "Train/Mean_Return": train_ret,
                "Train/Normalized_Return_Pct": (train_ret / 226.0) * 100.0,
                "Train/Policy_Entropy": float(reduced_metric["train_ent"]),
                "Train/Value_Loss": float(reduced_metric["train_vloss"]),
                "Train/Mean_Episode_Length": float(reduced_metric["train_len"]),
                "Train/Max_Grad_Norm": float(reduced_metric["max_grad_norm"]),
                "Train/Total_Achievements": train_total_achv,
            }

            # Dynamically log every single achievement to its own WandB folder
            for k in ACH_KEYS:
                wandb_log_dict[f"Train_Achv/{k}"] = float(reduced_metric[f"train_{k}"])

            is_final_global_loop = (loop_idx == NUM_LOG_LOOPS - 1)
            is_session_limit_reached = (session_steps_processed >= SESSION_STEP_LIMIT)

            if is_final_global_loop or is_session_limit_reached:
                ckpt_manager.save(current_update_idx, args=ocp.args.StandardSave({"runner_state": runner_state}))
                ckpt_manager.wait_until_finished()

                print(
                    f"  [SESSION DONE] Step: {steps_processed / 1e6:>5.1f}M | Ret: {train_ret:>5.1f} | SPS: {int(sps):>5}",
                    flush=True)

                row = [model_name, seed, steps_processed, active_train_time, sps, train_ret,
                       float(reduced_metric["train_ent"]), float(reduced_metric["train_vloss"])]
                row.extend([float(reduced_metric[f"train_{k}"]) for k in ACH_KEYS])
                csv_writer.writerow(row)
                csv_file.flush()

                wandb.log(wandb_log_dict, step=steps_processed)
                wandb.finish(quiet=True)

                if is_final_global_loop:
                    print(
                        f"✅ {model_name} T={NUM_STEPS} fully completed 1 BILLION steps! Moving to next config in list...")
                    break
                else:
                    print(f"\n[WATCHDOG] 50M Session Limit Reached. Executing OS-Level RAM Flush (Exit 43)...")
                    csv_file.close()
                    os._exit(43)
            else:
                print(
                    f"  [TRAIN] Step: {steps_processed / 1e6:>5.1f}M | Ret: {train_ret:>5.1f} | Achv: {train_total_achv:>4.1f} | SPS: {int(sps):>5}",
                    flush=True)
                if loop_idx > start_loop_idx and loop_idx % 100 == 0:
                    ckpt_manager.save(current_update_idx, args=ocp.args.StandardSave({"runner_state": runner_state}))
                wandb.log(wandb_log_dict, step=steps_processed)

        try:
            ckpt_manager.wait_until_finished()
            wandb.finish()
            jax.clear_caches()
            gc.collect()
        except Exception:
            pass
        csv_file.close()

if __name__ == "__main__":
    import signal

    run_benchmark()