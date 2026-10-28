"""Fixed-endpoint CountRecall confirmation study for the journal revision.

This intentionally avoids the historical POPGym benchmark's checkpoint resume,
early stopping, and shared output files.  Every model/seed invocation starts
from fresh parameters and writes only to the specified revision artifact.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
os.environ.setdefault("HSA_ENABLE_SDMA", "0")

import jax
import jax.numpy as jnp
import wandb

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnostics.exact_sdlg.popgym_benchmark import (  # noqa: E402
    MODELS,
    get_normalized_score,
    get_rnn_carry_init,
    run_popgym_eval,
)
from src.envs.popgym_wrapper import make_popgym_env  # noqa: E402
from src.trainers.purejax_trainer import make_update  # noqa: E402
import src.trainers.purejax_trainer as trainer_module  # noqa: E402


STUDY_ID = "COUNTRECALL-CONFIRMATION-2026"
ENV_NAME = "CountRecallEasy"
DEFAULT_MODELS = ("SdLG-Exact", "SdLG-Approx", "PPO-GRU", "PPO-GLA")
DEFAULT_SEEDS = (42, 54, 99, 123, 456)
NUM_ENVS = 256
NUM_STEPS = 64
LOG_INTERVAL_UPDATES = 10
NOMINAL_STEPS = 15_000_000
EVAL_EVERY_LOGS = 30
EVAL_ENVS = 64


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def parse_csv(value: str, conversion):
    return tuple(conversion(part.strip()) for part in value.split(",") if part.strip())


def leaves_size(tree) -> int:
    return int(sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(tree)))


def rnn_config(model: str) -> dict:
    # Same outer width, layer count, optimizer, rollout, and training budget
    # across the four configured comparisons. State/parameter differences are
    # measured and reported rather than hidden by an inferred match claim.
    config = {
        "RNN_TYPE": MODELS[model]["rnn_type"],
        "D_MODEL": 128,
        "N_HEADS": 4,
        "N_LAYERS": 3,
        "D_FFC": 512,
        "D_K": 16,
        "D_V": 16,
        "R": 4,
        "WINDOW_SIZE": 64,
        "ALPHA": 0.1,
        "ETA": 0.0,
        "USE_ROPE_PHASE": False,
        "ROPE_VARIANT": "none",
        "USE_SOFTPLUS_DENOMINATOR": True,
        "USE_PER_RANK_NORM": True,
    }
    # This repository's PPO-GLA wrapper constructs 32-dimensional writes.
    # Its carry must match; the manifest and raw artifact expose the resulting
    # parameter/state difference instead of presenting it as matched capacity.
    if model == "PPO-GLA":
        config.update({"D_K": 32, "D_V": 32})
    return config


def config_for(model: str, nominal_steps: int, seed: int) -> dict:
    return {
        "NUM_ENVS": NUM_ENVS,
        "NUM_STEPS": NUM_STEPS,
        "TOTAL_TIMESTEPS": nominal_steps,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 8,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.0001,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "LR": 3e-4,
        "ANNEAL_LR": False,
        "OPTIMISTIC_RESETS": False,
        "LOG_INTERVAL": LOG_INTERVAL_UPDATES,
        "RNN": rnn_config(model),
        "SEED": seed,
    }


def build_manifest(path: Path, models: tuple[str, ...], seeds: tuple[int, ...], nominal_steps: int, output: Path):
    sources = [
        ROOT / "scripts/diagnostics/revision_2026_countrecall_confirmation.py",
        ROOT / "src/trainers/purejax_trainer.py",
        ROOT / "src/envs/popgym_wrapper.py",
        ROOT / "src/models/purejaxrl/wrappers.py",
        ROOT / "src/models/common/exact_sdlg.py",
        ROOT / "src/models/common/sdlg_hybrid.py",
        ROOT / "src/models/common/gru.py",
        ROOT / "src/models/common/adam_gla.py",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "study_id": STUDY_ID,
        "git_commit": git_commit(),
        "runner_sha256": source_hash(sources[0]),
        "source_sha256": {str(source.relative_to(ROOT)): source_hash(source) for source in sources if source.exists()},
        "environment": ENV_NAME,
        "models": models,
        "seeds": seeds,
        "nominal_steps": nominal_steps,
        "fixed_endpoint": "floor(nominal_steps / (NUM_ENVS*NUM_STEPS*LOG_INTERVAL_UPDATES)) completed log updates; no early stop or checkpoint resume",
        "training": {
            "num_envs": NUM_ENVS, "num_steps": NUM_STEPS,
            "log_interval_updates": LOG_INTERVAL_UPDATES,
            "ppo_epochs": 4, "minibatches": 8, "learning_rate": 3e-4,
            "anneal_lr": False, "gamma": 0.99, "gae_lambda": 0.95,
        },
        "evaluation": {
            "episodes": EVAL_ENVS,
            "length": "full environment maximum episode length, bounded by H=200 through the existing evaluator",
            "policy": "categorical mode",
            "seed_rule": "jax.PRNGKey(seed + final_log_index * 100)",
        },
        "hardware": {
            "platform": platform.platform(),
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
        },
        "output": str(output),
    }, indent=2) + "\n")


def write_header_if_needed(path: Path, overwrite: bool) -> None:
    if overwrite or not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            csv.writer(handle).writerow([
                "Study", "Environment", "Model", "Seed", "Total_Steps", "Log_Index",
                "Wall_Clock_Time", "SPS", "Train_Return", "Train_Entropy", "Eval_Return",
                "Eval_Normalized_Return", "Endpoint", "Trainable_Parameters", "State_Elements_Per_Env",
            ])


def already_complete(path: Path, model: str, seed: int) -> bool:
    if not path.exists():
        return False
    import pandas as pd
    data = pd.read_csv(path)
    rows = data[(data["Model"] == model) & (data["Seed"] == seed)]
    return bool((rows["Endpoint"] == True).any())


def run_one(model: str, seed: int, nominal_steps: int, output: Path) -> None:
    model_cls = MODELS[model]["class"]
    config = config_for(model, nominal_steps, seed)
    steps_per_log = NUM_ENVS * NUM_STEPS * LOG_INTERVAL_UPDATES
    total_logs = nominal_steps // steps_per_log
    fixed_endpoint = total_logs * steps_per_log
    if total_logs < 1:
        raise ValueError("nominal_steps does not reach one logged update")

    jax.clear_caches()
    gc.collect()
    env, env_params = make_popgym_env(ENV_NAME, num_envs=NUM_ENVS)
    trainer_module.ActorCriticRNN = model_cls
    carry_init = get_rnn_carry_init(model, config)
    network = model_cls(rnn_module=None, action_dim=env.action_space(env_params).n, config=config)
    rng = jax.random.PRNGKey(seed)
    raw_update, runner_state = make_update(rng, config, env, env_params, rnn_module=None, rnn_carry_init=carry_init)
    trainable_parameters = leaves_size(runner_state[0].params)
    state_elements = leaves_size(carry_init(1))

    @jax.jit
    def update_and_reduce(state):
        outcome = raw_update(state)
        metrics = outcome["metric"]
        mask = metrics.get("returned_episode", jnp.ones_like(metrics.get("entropy", jnp.zeros(1))))

        def mean_completed(value):
            return jnp.sum(value * mask) / jnp.maximum(1.0, jnp.sum(mask))

        return outcome["runner_state"], {
            "train_return": mean_completed(metrics.get("returned_episode_returns", jnp.zeros_like(mask))),
            "train_entropy": jnp.mean(metrics.get("entropy", jnp.zeros(1))),
        }

    run_config = dict(config)
    run_config.update({"STUDY_ID": STUDY_ID, "MODEL": model, "FIXED_ENDPOINT": fixed_endpoint})
    wandb.init(project=os.environ.get("WANDB_PROJECT", "sdlgHybrid"), mode=os.environ.get("WANDB_MODE", "offline"),
               group="countrecall-confirmation-2026", name=f"countrecall-{model}-seed-{seed}", config=run_config,
               tags=["countrecall_confirmation", "fixed_endpoint", "revision_2026"])
    start = time.time()
    with output.open("a", newline="") as handle:
        writer = csv.writer(handle)
        for log_index in range(1, total_logs + 1):
            runner_state, metrics = update_and_reduce(runner_state)
            runner_state = jax.block_until_ready(runner_state)
            metrics = jax.device_get(metrics)
            steps = log_index * steps_per_log
            endpoint = log_index == total_logs
            should_evaluate = endpoint or log_index % EVAL_EVERY_LOGS == 0
            eval_return, eval_score = "", ""
            if should_evaluate:
                eval_key = jax.random.PRNGKey(seed + log_index * 100)
                eval_return, eval_score = run_popgym_eval(eval_key, runner_state[0].params, network, carry_init,
                                                          ENV_NAME, 200, num_eval_envs=EVAL_ENVS)
            elapsed = time.time() - start
            sps = steps / max(elapsed, 1.0)
            row = [STUDY_ID, ENV_NAME, model, seed, steps, log_index, elapsed, sps,
                   float(metrics["train_return"]), float(metrics["train_entropy"]), eval_return, eval_score,
                   endpoint, trainable_parameters, state_elements]
            writer.writerow(row)
            handle.flush()
            wandb.log({"System/SPS": sps, "Train/Return": float(metrics["train_return"]),
                       "Train/Entropy": float(metrics["train_entropy"]),
                       "Eval/Return": eval_return, "Eval/Normalized_Return": eval_score}, step=steps)
            print(f"[{model} seed {seed}] {steps:,}/{fixed_endpoint:,} | train={float(metrics['train_return']):.3f} | eval={eval_score}", flush=True)
    wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--nominal-steps", type=int, default=NOMINAL_STEPS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    models = parse_csv(args.models, str)
    seeds = parse_csv(args.seeds, int)
    if set(models) - set(DEFAULT_MODELS):
        raise ValueError(f"Models must be drawn from {DEFAULT_MODELS}")
    nominal_steps = 163_840 if args.dry_run else args.nominal_steps
    build_manifest(args.manifest, models, seeds, nominal_steps, args.output)
    write_header_if_needed(args.output, args.overwrite)
    for model in models:
        for seed in seeds:
            if already_complete(args.output, model, seed):
                print(f"[skip] completed endpoint: {model} seed {seed}", flush=True)
                continue
            run_one(model, seed, nominal_steps, args.output)


if __name__ == "__main__":
    main()
