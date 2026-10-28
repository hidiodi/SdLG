# scripts/diagnostics/exact_sdlg/popgym_benchmark.py

import sys
import os
import time
import json
import csv
import math
import gc
import pickle
import glob
import argparse
import functools
from pathlib import Path
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import wandb
import distrax

# Environment memory tune for ROCm / CUDA
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"
os.environ["HSA_ENABLE_SDMA"] = "0"

def save_checkpoint(step, params, filepath="checkpoints/popgym_latest.pkl"):
    filepath = os.fspath(filepath)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    tmp_filepath = filepath + ".tmp"
    with open(tmp_filepath, "wb") as f:
        pickle.dump({"step": step, "params": jax.device_get(params)}, f)
    os.replace(tmp_filepath, filepath)
    print(f"💾 [Checkpoint] Saved model parameters at step {step:,} -> {filepath}")

def load_checkpoint(filepath):
    filepath = os.fspath(filepath)
    if os.path.exists(filepath):
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            print(f"🔄 [Checkpoint] Loaded checkpoint from {filepath} (step {data.get('step', 0):,})")
            return data
        except Exception as e:
            print(f"⚠️ [Checkpoint] Failed to load checkpoint {filepath}: {e}")

    # Fallback: search for step-specific checkpoints
    chk_dir = os.path.dirname(filepath)
    basename = os.path.basename(filepath).replace("_latest.pkl", "")
    pattern = sorted(glob.glob(os.path.join(chk_dir, f"{basename}_step*.pkl")), reverse=True)
    for p in pattern:
        try:
            with open(p, "rb") as f:
                data = pickle.load(f)
            print(f"🔄 [Checkpoint Fallback] Restored from backup checkpoint {p} (step {data.get('step', 0):,})")
            return data
        except Exception:
            continue
    return None

# Path fixing
current_file_path = os.path.abspath(__file__)
project_root = current_file_path
while not os.path.exists(os.path.join(project_root, "src")):
    project_root = os.path.dirname(project_root)
    if project_root == "/":
        break
if project_root not in sys.path:
    sys.path.append(project_root)

from src.envs.popgym_wrapper import POPGymEnv, make_popgym_env
from src.envs.wrappers import LogWrapper, BatchEnvWrapper
from src.trainers.purejax_trainer import make_update
import src.trainers.purejax_trainer as trainer_module
from src.models.purejaxrl.wrappers import (
    ActorCriticSdLGHybrid,
    ActorCriticExactSdLG,
    ActorCriticSdLGDelta,
    ActorCriticSdLGADAM,
    ActorCriticSdLGADAMMatched,
    ActorCriticSdLGApproxDelta,
    ActorCriticSdLGApproxADAM,
    ActorCriticGRU,
    ActorCriticMamba,
    ActorCriticGLA,
    ActorCriticDeltaGLA,
    ActorCriticADAM,
)
from src.models.common.exact_sdlg import UnbatchedExactSdLG
from src.models.common.sdlg_hybrid import BatchedSdLGHybrid
from src.models.common.gru import BatchedGRU
from src.models.baselines.mamba import MambaCell

MODELS = {
    # ── Primary Factorial Experiment (Family A) ──────────────────────────────
    "SdLG-Exact":             {"class": ActorCriticExactSdLG,       "rnn_type": "exact_sdlg"},
    "SdLG-Exact-FastRoPE":    {"class": ActorCriticExactSdLG,       "rnn_type": "exact_sdlg"},
    "SdLG-Delta":             {"class": ActorCriticSdLGDelta,       "rnn_type": "sdlg_delta"},
    "SdLG-ADAM":              {"class": ActorCriticSdLGADAM,        "rnn_type": "sdlg_adam"},
    "SdLG-ADAM-Matched":      {"class": ActorCriticSdLGADAMMatched,    "rnn_type": "sdlg_adam"},  # param control
    "SdLG-Delta-dk32":        {"class": ActorCriticSdLGDelta,       "rnn_type": "sdlg_delta_dk32"},
    "SdLG-Delta-dk64":        {"class": ActorCriticSdLGDelta,       "rnn_type": "sdlg_delta_dk64"},
    "SdLG-ADAM-dk64":         {"class": ActorCriticSdLGADAM,        "rnn_type": "sdlg_adam_dk64"},
    "SdLG-ADAM-Matched-dk64": {"class": ActorCriticSdLGADAMMatched,    "rnn_type": "sdlg_adam_dk64"},
    "SdLG-Approx":            {"class": ActorCriticSdLGHybrid,      "rnn_type": "sdlg_hybrid"},
    "SdLG-Approx-Delta":      {"class": ActorCriticSdLGApproxDelta,    "rnn_type": "sdlg_approx_delta"},
    "SdLG-Approx-ADAM":       {"class": ActorCriticSdLGApproxADAM,     "rnn_type": "sdlg_approx_adam"},
    # ── Family B: Retention / Channel Decay ─────────────────────────────────
    "SdLG-ChannelDecay":      {"class": ActorCriticSdLGDelta,       "rnn_type": "sdlg_channel_decay"},
    # ── Family F: Depth × Rank Grid (fixed recurrent-state budget) ──────────
    # Budget: n_layers * n_heads * d_k * d_v = const ≈ 12,288
    # (1L,r=64): 1*4*64*64 = 16384  (2L,r=32): 2*4*32*32 = 8192
    # (4L,r=16): 4*4*16*16 = 4096   (8L,r=8):  8*4*8*8  = 2048
    "SdLG-Depth1-Rank64":     {"class": ActorCriticExactSdLG,     "rnn_type": "exact_sdlg_d1r64"},
    "SdLG-Depth2-Rank32":     {"class": ActorCriticExactSdLG,     "rnn_type": "exact_sdlg_d2r32"},
    "SdLG-Depth4-Rank16":     {"class": ActorCriticExactSdLG,     "rnn_type": "exact_sdlg_d4r16"},
    "SdLG-Depth8-Rank8":      {"class": ActorCriticExactSdLG,     "rnn_type": "exact_sdlg_d8r8"},
    # ── External Baselines ───────────────────────────────────────────────────
    "PPO-GRU":                {"class": ActorCriticGRU,         "rnn_type": "gru"},
    "PPO-Mamba":              {"class": ActorCriticMamba,       "rnn_type": "mamba"},
    "PPO-GLA":                {"class": ActorCriticGLA,         "rnn_type": "gla"},
    "PPO-DeltaGLA":           {"class": ActorCriticDeltaGLA,    "rnn_type": "delta_gla"},
    "PPO-ADAM":               {"class": ActorCriticADAM,        "rnn_type": "adam"},
}

# =====================================================================
# POPGYM ENVIRONMENT REGISTRY (Full 16 Easy + Medium Environments)
# =====================================================================
POPGYM_ENVS = [
    # Easy Benchmark Suite (Full 3-Seed Evaluation Across All Models)
    "AutoencodeEasy",
    "BattleshipEasy",
    "ConcentrationEasy",
    "CountRecallEasy",
    "HigherLowerEasy",
    "LabyrinthEscapeEasy",
    "LabyrinthExploreEasy",
    "MineSweeperEasy",
    "MultiarmedBanditEasy",
    "NoisyPositionOnlyCartPoleEasy",
    "NoisyPositionOnlyPendulumEasy",
    "PositionOnlyCartPoleEasy",
    "PositionOnlyPendulumEasy",
    "RepeatFirstEasy",
    "RepeatPreviousEasy",
    "VelocityOnlyCartpoleEasy",
    # Medium Environments Diagnostic Sweep
    "RepeatPreviousMedium",
    "ConcentrationMedium",
    "CountRecallMedium",
    "HigherLowerMedium",
    "LabyrinthEscapeMedium",
    "LabyrinthExploreMedium",
    "MinesweeperMedium",
]

DEFAULT_SEEDS = [42, 54, 99]




def get_rnn_carry_init(model_name, config):
    d_model = config["RNN"]["D_MODEL"]
    n_layers = config["RNN"]["N_LAYERS"]
    n_heads = config["RNN"]["N_HEADS"]
    d_k = config["RNN"]["D_K"]
    d_v = config["RNN"]["D_V"]
    r = config["RNN"].get("R", 4)  # Match Table 3 specification (r=4)
    window_size = config["RNN"].get("WINDOW_SIZE", 16)

    def init_fn(batch_size):
        # ── Primary Factorial & Asymmetric Models ───────────────────────────
        if model_name in [
            "SdLG-Exact", "SdLG-Exact-FastRoPE", "SdLG-Delta", "SdLG-ADAM", "SdLG-ADAM-Matched",
            "SdLG-Delta-dk32", "SdLG-Delta-dk64", "SdLG-ADAM-dk64", "SdLG-ADAM-Matched-dk64",
            "SdLG-ChannelDecay"
        ]:
            unbatched = UnbatchedExactSdLG.initialize_carry(
                n_layers=n_layers, n_heads=n_heads, d_k=d_k, d_v=d_v, window_size=window_size
            )
            return jax.tree_util.tree_map(
                lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0), unbatched
            )
        # ── Family F: Depth × Rank grid ──────────────────────────────────────
        elif model_name == "SdLG-Depth1-Rank64":
            unbatched = UnbatchedExactSdLG.initialize_carry(
                n_layers=1, n_heads=n_heads, d_k=64, d_v=64, window_size=window_size
            )
            return jax.tree_util.tree_map(
                lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0), unbatched
            )
        elif model_name == "SdLG-Depth2-Rank32":
            unbatched = UnbatchedExactSdLG.initialize_carry(
                n_layers=2, n_heads=n_heads, d_k=32, d_v=32, window_size=window_size
            )
            return jax.tree_util.tree_map(
                lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0), unbatched
            )
        elif model_name == "SdLG-Depth4-Rank16":
            unbatched = UnbatchedExactSdLG.initialize_carry(
                n_layers=4, n_heads=n_heads, d_k=16, d_v=16, window_size=window_size
            )
            return jax.tree_util.tree_map(
                lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0), unbatched
            )
        elif model_name == "SdLG-Depth8-Rank8":
            unbatched = UnbatchedExactSdLG.initialize_carry(
                n_layers=8, n_heads=n_heads, d_k=8, d_v=8, window_size=window_size
            )
            return jax.tree_util.tree_map(
                lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0), unbatched
            )
        # ── Low-Rank Approx variants ─────────────────────────────────────────
        elif model_name in ["SdLG-Approx", "SdLG-Approx-Delta", "SdLG-Approx-ADAM"]:
            return jax.tree_util.tree_map(
                lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0),
                BatchedSdLGHybrid.initialize_carry(
                    batch_size=batch_size, n_layers=n_layers, d_k=d_k, d_v=d_v, r=r, window_size=window_size
                ),
            )
        elif model_name == "PPO-GRU":
            return BatchedGRU.initialize_carry(batch_size, d_model)
        elif model_name == "PPO-Mamba":
            return MambaCell.initialize_carry(batch_size, d_model)
        elif model_name in ["PPO-GLA", "PPO-DeltaGLA", "PPO-ADAM"]:
            gla_list = []
            for _ in range(n_layers):
                M = jnp.zeros((n_heads, d_k, d_v), dtype=jnp.float32)
                z = jnp.zeros((n_heads, d_k), dtype=jnp.float32)
                gla_list.append((M, z))
            return jax.tree_util.tree_map(
                lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0), tuple(gla_list)
            )

    return init_fn


def get_normalized_score(env_name: str, return_val: float, H: int) -> float:
    """
    Normalizes raw episodic returns to a standard [0.0, 1.0] performance metric.
    Computes exact bounds depending on environment dynamics and evaluation horizon H.
    """
    if "CartPole" in env_name or "Pendulum" in env_name:
        # CartPole / Pendulum gives +0.005 reward per timestep balanced up to H
        max_ret = 0.005 * float(H)
        min_ret = 0.0
    elif "Concentration" in env_name or "Minesweeper" in env_name or "MineSweeper" in env_name:
        max_ret = 1.0
        min_ret = 0.0
    else:
        # Standard POPGym unit-bounded environments (returns in [-1.0, 1.0])
        max_ret = 1.0
        min_ret = -1.0

    score = (return_val - min_ret) / max(1e-5, max_ret - min_ret)
    return float(np.clip(score, 0.0, 1.0))


def is_run_completed(csv_path, checkpoint_dir, env_name, model_name, seed, total_steps):
    """Check if a given env/model/seed run is already present and completed (or early-stopped as solved) in the results CSV or checkpoint."""
    chk_final = Path(checkpoint_dir) / f"{env_name}_{model_name}_seed{seed}_final.pkl"
    if os.path.exists(chk_final):
        return True

    if not csv_path or not os.path.exists(csv_path):
        return False

    try:
        matching_rows = []
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return False

            for row in reader:
                if (
                    len(row) >= 4
                    and row[0].strip() == env_name
                    and row[1].strip() == model_name
                    and row[2].strip() == str(seed)
                ):
                    matching_rows.append(row)

        if not matching_rows:
            return False

        max_step = 0
        for row in matching_rows:
            try:
                s_val = int(float(row[3]))
                if s_val > max_step:
                    max_step = s_val

                if len(row) >= 11:
                    score100 = float(row[7])
                    score200 = float(row[8])
                elif len(row) >= 7:
                    r100 = float(row[5])
                    r200 = float(row[6])
                    score100 = get_normalized_score(env_name, r100, 100)
                    score200 = get_normalized_score(env_name, r200, 200)
                else:
                    continue

                if not math.isnan(score100) and not math.isnan(score200):
                    if score100 >= 0.99 and score200 >= 0.98:
                        return True
            except (ValueError, TypeError):
                pass

        if max_step >= int(total_steps * 0.95):
            return True

    except Exception as e:
        return False

    return False



def run_popgym_eval(rng, params, network, rnn_carry_init_fn, env_name, H, num_eval_envs=32):
    raw_env = POPGymEnv(env_name, max_envs=num_eval_envs * 4)
    if raw_env.max_episode_length > 200:
        eval_len = raw_env.max_episode_length
    else:
        eval_len = min(H, raw_env.max_episode_length)
    raw_env.max_episode_length = eval_len
    eval_env = BatchEnvWrapper(LogWrapper(raw_env), num_envs=num_eval_envs)

    rng, _rng = jax.random.split(rng)
    obs, env_state = eval_env.reset(_rng, None)
    carry = rnn_carry_init_fn(num_eval_envs)
    dones = jnp.zeros(num_eval_envs, dtype=jnp.bool_)

    def eval_step_fn(runner_state, _):
        c, o, d, s, r_key, episode_ret, active = runner_state
        ac_in = (jnp.expand_dims(o, axis=0), jnp.expand_dims(d, axis=0))
        out = network.apply(params, c, ac_in)
        if len(out) == 4:
            action_logits, _, _, next_c = out
            pi = distrax.Categorical(logits=action_logits)
        else:
            next_c, pi, _ = out
        act = pi.mode().squeeze(0)

        r_key, step_key = jax.random.split(r_key)
        next_o, next_s, reward, next_d, _ = eval_env.step(step_key, s, act, None)
        next_episode_ret = episode_ret + reward * active
        next_active = active * (1.0 - next_d.astype(jnp.float32))
        return (next_c, next_o, next_d, next_s, r_key, next_episode_ret, next_active), None

    init_runner_state = (
        carry,
        obs,
        dones,
        env_state,
        _rng,
        jnp.zeros(num_eval_envs),
        jnp.ones(num_eval_envs, dtype=jnp.float32),
    )
    (final_c, final_o, final_d, final_s, final_key, final_ret, final_active), _ = jax.lax.scan(
        eval_step_fn, init_runner_state, None, length=eval_len
    )

    raw_mean_ret = float(jnp.mean(final_ret))
    norm_score = get_normalized_score(env_name, raw_mean_ret, eval_len)
    return raw_mean_ret, norm_score


def run_benchmark(total_steps=15_000_000, dry_run=False, seeds=None, env_list=None, model_list=None,
                  eval_interval=1, output_dir=None):
    if dry_run:
        total_steps = 100_000
        print("⚡ Dry Run Mode: Running 100,000 steps per model/seed.")

    if seeds is None:
        seeds = DEFAULT_SEEDS
    if env_list is None:
        env_list = POPGYM_ENVS
    if model_list is None:
        model_list = list(MODELS.keys())

    output_dir = Path(output_dir or Path(project_root) / "results" / "popgym").resolve()
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_csv = output_dir / "popgym_results.csv"

    header = [
        "Environment",
        "Model",
        "Seed",
        "Step",
        "Train_Return",
        "Eval_Return_H100",
        "Eval_Return_H200",
        "Eval_NormScore_H100",
        "Eval_NormScore_H200",
        "Policy_Entropy",
        "SPS",
    ]

    if not results_csv.exists():
        with results_csv.open("w", newline="") as f:
            csv.writer(f).writerow(header)

    print("====================================================================")
    print("=== POPGYM BENCHMARK EVALUATION SUITE (Online PPO, T=64)         ===")
    print("====================================================================")

    for env_name in env_list:
        for model_name in model_list:
            model_info = MODELS[model_name]
            model_cls = model_info["class"]

            for seed in seeds:
                run_id = f"{env_name}_{model_name}_seed_{seed}"

                # Execute 3-seed sweeps for target Medium tasks
                if "Medium" in env_name:
                    target_3seed_envs = [
                        "ConcentrationMedium",
                        "LabyrinthEscapeMedium",
                        "RepeatPreviousMedium",
                        "MinesweeperMedium",
                        "LabyrinthExploreMedium",
                    ]
                    if env_name not in target_3seed_envs:
                        if seed != 42:
                            continue


                # Skip completed runs if present in CSV results
                if not dry_run and (
                    is_run_completed(results_csv, checkpoint_dir, env_name, model_name, seed, total_steps)
                ):
                    print(f"⏭️ Skipping completed run: [{run_id}] (found in popgym_results.csv)")
                    continue

                jax.clear_caches()
                gc.collect()

                print(f"\n🚀 Launching POPGym Benchmark: [{run_id}] | Seed = {seed}")

                num_envs = 256
                num_steps = 64  # Truncation horizon T = 64 for long-range credit assignment
                steps_per_update = num_envs * num_steps
                log_interval = 10
                total_jit_calls = max(1, (total_steps // steps_per_update) // log_interval)

                config = {
                    "NUM_ENVS": num_envs,
                    "NUM_STEPS": num_steps,
                    "TOTAL_TIMESTEPS": total_steps,
                    "UPDATE_EPOCHS": 4,
                    "NUM_MINIBATCHES": 8,
                    "GAMMA": 0.99,
                    "GAE_LAMBDA": 0.95,
                    "CLIP_EPS": 0.2,
                    "ENT_COEF": 0.0001,
                    "VF_COEF": 0.5,
                    "MAX_GRAD_NORM": 0.5,
                    "LR": 0.0003,
                    "ANNEAL_LR": False,
                    "OPTIMISTIC_RESETS": False,
                    "LOG_INTERVAL": log_interval,
                    "RNN": {
                        "RNN_TYPE": model_info["rnn_type"],
                        "D_MODEL": 128,
                        "N_HEADS": 4,
                        "N_LAYERS": 3,
                        "D_FFC": 512,
                        "D_K": 32,
                        "D_V": 32,
                        "R": 4,
                        "WINDOW_SIZE": 64,
                    },
                    "SEED": seed,
                }

                # ── Per-model config overrides ─────────────────────────────────────
                ASYMMETRIC_OVERRIDES = {
                    "SdLG-Exact":             {"D_K": 16, "D_V": 16},
                    "SdLG-Exact-FastRoPE":    {"D_K": 16, "D_V": 16, "ROPE_VARIANT": "fast_head"},
                    "SdLG-Delta":             {"D_K": 16, "D_V": 16},
                    "SdLG-ADAM":              {"D_K": 16, "D_V": 16},
                    "SdLG-ADAM-Matched":      {"D_K": 16, "D_V": 16},
                    "SdLG-Delta-dk32":        {"D_K": 32, "D_V": 16},
                    "SdLG-Delta-dk64":        {"D_K": 64, "D_V": 16},
                    "SdLG-ADAM-dk64":         {"D_K": 64, "D_V": 16},
                    "SdLG-ADAM-Matched-dk64": {"D_K": 64, "D_V": 16},
                }
                if model_name in ASYMMETRIC_OVERRIDES:
                    config["RNN"].update(ASYMMETRIC_OVERRIDES[model_name])

                # Family F: Depth × Rank grid — override N_LAYERS, D_K, D_V
                DEPTH_RANK_OVERRIDES = {
                    "SdLG-Depth1-Rank64": {"N_LAYERS": 1, "D_K": 64, "D_V": 64},
                    "SdLG-Depth2-Rank32": {"N_LAYERS": 2, "D_K": 32, "D_V": 32},
                    "SdLG-Depth4-Rank16": {"N_LAYERS": 4, "D_K": 16, "D_V": 16},
                    "SdLG-Depth8-Rank8":  {"N_LAYERS": 8, "D_K":  8, "D_V":  8},
                }
                if model_name in DEPTH_RANK_OVERRIDES:
                    config["RNN"].update(DEPTH_RANK_OVERRIDES[model_name])

                # Family B: Channel Decay — uses SdLGADAM wrapper in "channel_decay" variant
                if model_name == "SdLG-ChannelDecay":
                    config["RNN"]["VARIANT"] = "channel_decay"

                wandb_project = os.environ.get("WANDB_PROJECT", "POPGym-Benchmark")
                wandb.init(
                    project=wandb_project,
                    group=f"popgym_{env_name}",
                    name=f"{model_name}-seed{seed}",
                    config=config,
                    reinit=True,
                )

                env, env_params = make_popgym_env(env_name, num_envs=num_envs)

                trainer_module.ActorCriticRNN = model_cls
                rnn_carry_init_fn = get_rnn_carry_init(model_name, config)

                network = model_cls(rnn_module=None, action_dim=env.action_space(env_params).n, config=config)

                rng = jax.random.PRNGKey(seed)
                raw_update_fn, runner_state = make_update(
                    rng, config, env, env_params, rnn_module=None, rnn_carry_init=rnn_carry_init_fn
                )

                # Attempt to auto-resume from latest checkpoint if available
                chk_latest_path = checkpoint_dir / f"{env_name}_{model_name}_seed{seed}_latest.pkl"
                chk_data = load_checkpoint(chk_latest_path)

                steps_processed = 0
                if chk_data is not None:
                    try:
                        loaded_params = chk_data["params"]
                        old_train_state = runner_state[0]
                        def check_shape(a, b):
                            if a.shape != b.shape:
                                raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
                        jax.tree_util.tree_map(check_shape, loaded_params, old_train_state.params)
                        new_train_state = old_train_state.replace(params=loaded_params)
                        runner_state = (new_train_state,) + runner_state[1:]
                        steps_processed = int(chk_data.get("step", 0))
                        print(f"🔄 [Checkpoint] Resumed checkpoint: {chk_latest_path} (step {steps_processed:,})")
                    except Exception as err:
                        print(f"⚠️ [Checkpoint Warning] Skipped incompatible checkpoint ({err}). Initializing fresh weights.")
                        steps_processed = 0


                start_jit_idx = steps_processed // (steps_per_update * log_interval)
                last_checkpoint_step = steps_processed

                @jax.jit
                def compiled_update(r_state):
                    out = raw_update_fn(r_state)
                    m = out["metric"]
                    d_mask = m.get("returned_episode", jnp.ones_like(m.get("entropy", jnp.zeros(1))))

                    def mean_d(x):
                        return jnp.sum(x * d_mask) / jnp.maximum(1.0, jnp.sum(d_mask))

                    return out["runner_state"], {
                        "train_ret": mean_d(m.get("returned_episode_returns", jnp.zeros_like(d_mask))),
                        "train_len": mean_d(m.get("returned_episode_lengths", jnp.zeros_like(d_mask))),
                        "train_ent": jnp.mean(m.get("entropy", jnp.zeros(1))),
                    }

                start_wall_time = time.time()
                eval_ret_h100, eval_score_h100 = 0.0, 0.0
                eval_ret_h200, eval_score_h200 = 0.0, 0.0
                should_early_stop = False
                consecutive_solved = 0

                for jit_idx in range(start_jit_idx, total_jit_calls):
                    runner_state, metrics = compiled_update(runner_state)
                    runner_state = jax.block_until_ready(runner_state)
                    metrics = jax.device_get(metrics)

                    steps_processed += steps_per_update * log_interval
                    sps = float(steps_processed / max(time.time() - start_wall_time, 1.0))
                    train_ret = float(metrics["train_ret"])
                    policy_ent = float(metrics["train_ent"])

                    # Save latest checkpoint on every JIT update so restarts resume seamlessly
                    save_checkpoint(
                        steps_processed,
                        runner_state[0].params,
                        filepath=checkpoint_dir / f"{env_name}_{model_name}_seed{seed}_latest.pkl",
                    )

                    # Periodic milestone Checkpointing & Memory Cleanup (Every 1M steps)
                    if (steps_processed - last_checkpoint_step >= 1_000_000) or (jit_idx == total_jit_calls - 1):
                        last_checkpoint_step = steps_processed
                        step_m = steps_processed // 1_000_000
                        save_checkpoint(
                            steps_processed,
                            runner_state[0].params,
                            filepath=checkpoint_dir / f"{env_name}_{model_name}_seed{seed}_step{step_m}M.pkl",
                        )
                        gc.collect()

                    eval_ran = False
                    if jit_idx % eval_interval == 0 or jit_idx == total_jit_calls - 1:
                        eval_ran = True
                        eval_rng = jax.random.PRNGKey(seed + jit_idx * 100)
                        try:
                            eval_ret_h100, eval_score_h100 = run_popgym_eval(
                                eval_rng, runner_state[0].params, network, rnn_carry_init_fn, env_name, 100, num_eval_envs=32
                            )
                            eval_ret_h200, eval_score_h200 = run_popgym_eval(
                                eval_rng, runner_state[0].params, network, rnn_carry_init_fn, env_name, 200, num_eval_envs=32
                            )
                        except Exception as e:
                            print(f"⚠️ EVAL ERROR on [{run_id}]: {e}")
                            eval_ret_h100, eval_score_h100 = float("nan"), float("nan")
                            eval_ret_h200, eval_score_h200 = float("nan"), float("nan")

                        # Tightened evaluation check requiring 3 consecutive passes at near-perfect scores
                        is_solved_pass = (not math.isnan(eval_score_h100)) and (eval_score_h100 >= 0.99) and (eval_score_h200 >= 0.98)
                        if is_solved_pass:
                            consecutive_solved += 1
                        else:
                            consecutive_solved = 0

                        if consecutive_solved >= 3:
                            should_early_stop = True

                    wandb.log(
                        {
                            "System/SPS": sps,
                            "Train/Return": train_ret,
                            "Eval/Return_H100": eval_ret_h100,
                            "Eval/Return_H200": eval_ret_h200,
                            "Eval/NormScore_H100": eval_score_h100,
                            "Eval/NormScore_H200": eval_score_h200,
                            "Train/Policy_Entropy": policy_ent,
                        },
                        step=steps_processed,
                    )

                    row = [
                        env_name,
                        model_name,
                        seed,
                        steps_processed,
                        train_ret,
                        eval_ret_h100,
                        eval_ret_h200,
                        eval_score_h100,
                        eval_score_h200,
                        policy_ent,
                        sps,
                    ]
                    with results_csv.open("a", newline="") as f:
                        csv.writer(f).writerow(row)

                    eval_tag = " (eval)" if eval_ran else ""
                    print(
                        f"  [JIT {jit_idx + 1:>3}/{total_jit_calls}] Step: {steps_processed / 1e6:>5.2f}M | "
                        f"Train Ret: {train_ret:>5.2f} | Eval H100 NormScore: {eval_score_h100:>4.2f} | Eval H200 NormScore: {eval_score_h200:>4.2f}{eval_tag} | SPS: {int(sps):>5}",
                        flush=True,
                    )

                    if should_early_stop:
                        print(
                            f"🎉 EARLY STOPPING TRIGGERED: [{run_id}] solved task (H100={eval_score_h100:.2f} ≥ 0.95, H200={eval_score_h200:.2f} ≥ 0.90 on 32-env evaluation pass at step {steps_processed / 1e6:.2f}M)!"
                        )
                        save_checkpoint(
                            steps_processed,
                            runner_state[0].params,
                            filepath=checkpoint_dir / f"{env_name}_{model_name}_seed{seed}_final.pkl",
                        )
                        wandb.log({"Early_Stopped": True}, step=steps_processed)
                        break

                if start_jit_idx >= total_jit_calls:
                    eval_rng = jax.random.PRNGKey(seed + 9999)
                    try:
                        eval_ret_h100, eval_score_h100 = run_popgym_eval(
                            eval_rng, runner_state[0].params, network, rnn_carry_init_fn, env_name, 100, num_eval_envs=32
                        )
                        eval_ret_h200, eval_score_h200 = run_popgym_eval(
                            eval_rng, runner_state[0].params, network, rnn_carry_init_fn, env_name, 200, num_eval_envs=32
                        )
                    except Exception as e:
                        print(f"⚠️ EVAL ERROR on [{run_id}]: {e}")
                        eval_ret_h100, eval_score_h100 = float("nan"), float("nan")
                        eval_ret_h200, eval_score_h200 = float("nan"), float("nan")

                    row = [
                        env_name,
                        model_name,
                        seed,
                        steps_processed,
                        0.0,
                        eval_ret_h100,
                        eval_ret_h200,
                        eval_score_h100,
                        eval_score_h200,
                        0.0,
                        5000.0,
                    ]
                    with results_csv.open("a", newline="") as f:
                        csv.writer(f).writerow(row)
                    save_checkpoint(
                        steps_processed,
                        runner_state[0].params,
                        filepath=checkpoint_dir / f"{env_name}_{model_name}_seed{seed}_final.pkl",
                    )
                    print(
                        f"  [Checkpoint Evaluation] Step: {steps_processed / 1e6:>5.2f}M | "
                        f"Eval H100 NormScore: {eval_score_h100:>4.2f} | Eval H200 NormScore: {eval_score_h200:>4.2f} (eval)",
                        flush=True,
                    )

                print(f"✅ Finished [{run_id}]\n")
                wandb.finish()
                jax.clear_caches()
                gc.collect()

    print("====================================================================")
    print("🎉 POPGYM BENCHMARK SUITE EXECUTION COMPLETE")
    print("====================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run 100,000 steps dry-run validation.")
    parser.add_argument("--total-steps", type=int, default=15_000_000, help="Total environment steps per run.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, help="Random seeds.")
    parser.add_argument("--envs", nargs="+", default=POPGYM_ENVS, help="Environments to evaluate.")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()), help="Models to evaluate.")
    parser.add_argument("--eval-interval", type=int, default=1, help="JIT calls between evaluations (~81.9k steps).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for CSVs and resumable local checkpoints.")
    args = parser.parse_args()

    run_benchmark(
        total_steps=args.total_steps,
        dry_run=args.dry_run,
        seeds=args.seeds,
        env_list=args.envs,
        model_list=args.models,
        eval_interval=args.eval_interval,
        output_dir=args.output_dir,
    )
