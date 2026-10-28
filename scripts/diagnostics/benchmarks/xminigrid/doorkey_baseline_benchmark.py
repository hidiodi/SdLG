"""Protocol-matched DoorKey comparison for SdLG-Exact, GRU, and GTrXL.

This study is intentionally small: two DoorKey sizes, three seeds, one shared
observation encoder and PPO configuration, a 20M-step cap, and a common early
stopping rule.  Every run receives a separate, fixed-seed terminal evaluation.

The raw trajectory and run summaries are written locally even when Weights &
Biases is disabled.  A run is considered complete only after terminal
evaluation has been recorded in the summary CSV.
"""

from __future__ import annotations

import argparse
import csv
import functools
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from flax.linen.initializers import constant, orthogonal


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH
while not (PROJECT_ROOT / "src").exists():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise RuntimeError("Could not locate repository root")
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.diagnostics.exact_sdlg.exact_sdlg_xminigrid_keycorridor import (  # noqa: E402
    PureJaxRLXMiniGridWrapper,
)
from src.envs.wrappers import BatchEnvWrapper  # noqa: E402
from src.models.common.exact_sdlg import UnbatchedExactSdLG  # noqa: E402
from src.models.common.gru import BatchedGRU  # noqa: E402
from src.models.common.gtrxl import BatchedGTrXL  # noqa: E402
from src.trainers.purejax_trainer import make_update  # noqa: E402


DEFAULT_ENVS = ("MiniGrid-DoorKey-8x8", "MiniGrid-DoorKey-16x16")
DEFAULT_MODELS = ("SdLG-Exact", "PPO-GRU", "PPO-GTrXL")
DEFAULT_SEEDS = (42, 54, 99)


class DoorKeyEncoder(nn.Module):
    """Shared MLP encoder for the 7x7x2 symbolic MiniGrid observation."""

    d_model: int

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        obs = obs.astype(jnp.float32) / 10.0
        flat = obs.reshape(obs.shape[0], obs.shape[1], -1)
        x = nn.Dense(
            self.d_model,
            kernel_init=orthogonal(jnp.sqrt(2.0)),
            bias_init=constant(0.0),
            name="encoder_fc1",
        )(flat)
        x = nn.relu(x)
        x = nn.Dense(
            self.d_model,
            kernel_init=orthogonal(jnp.sqrt(2.0)),
            bias_init=constant(0.0),
            name="encoder_fc2",
        )(x)
        return nn.LayerNorm(name="encoder_norm")(nn.relu(x))


class ActorCriticDoorKey(nn.Module):
    """One shared actor-critic shell around the three recurrent cores."""

    action_dim: int
    config: dict
    model_name: str

    @nn.compact
    def __call__(self, carry: Any, x: tuple[jnp.ndarray, jnp.ndarray]):
        obs, dones = x
        rnn_cfg = self.config["RNN"]
        emb = DoorKeyEncoder(rnn_cfg["D_MODEL"], name="shared_encoder")(obs)

        if self.model_name == "SdLG-Exact":
            emb_b = jnp.swapaxes(emb, 0, 1)
            dones_b = jnp.swapaxes(dones, 0, 1)
            batched_exact = nn.vmap(
                UnbatchedExactSdLG,
                in_axes=(0, 0, 0, None),
                out_axes=(0, 0, 0),
                variable_axes={"params": None},
                split_rngs={"params": False},
            )
            outs_b, new_carry, _ = batched_exact(
                n_layers=rnn_cfg["N_LAYERS"],
                d_model=rnn_cfg["D_MODEL"],
                n_heads=rnn_cfg["N_HEADS"],
                d_k=rnn_cfg["D_K"],
                d_v=rnn_cfg["D_V"],
                d_ffc=rnn_cfg["D_FFC"],
                window_size=rnn_cfg["WINDOW_SIZE"],
                rope_variant=rnn_cfg["ROPE_VARIANT"],
                name="exact_sdlg",
            )(emb_b, dones_b, carry, True)
            outs = jnp.swapaxes(outs_b, 0, 1)

        elif self.model_name == "PPO-GRU":
            outs = emb
            next_layers = []
            for layer_index, layer_carry in enumerate(carry):
                layer_carry, outs = BatchedGRU(name=f"gru_{layer_index}")(
                    layer_carry, (outs, dones)
                )
                next_layers.append(layer_carry)
            new_carry = tuple(next_layers)

        elif self.model_name == "PPO-GTrXL":
            new_carry, outs = BatchedGTrXL(
                head_dim=rnn_cfg["D_HEAD"],
                embedding_dim=rnn_cfg["D_MODEL"],
                head_num=rnn_cfg["N_HEADS"],
                mlp_num=2,
                layer_num=rnn_cfg["N_LAYERS"],
                memory_len=rnn_cfg["MEMORY_LEN"],
                use_embedding_layer=False,
                name="gtrxl",
            )(carry, (emb, dones))
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_head",
        )(outs)
        logits = 5.0 * jnp.tanh(logits / 5.0)
        logits = jnp.nan_to_num(logits, nan=0.0, posinf=5.0, neginf=-5.0)
        values = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic_head",
        )(outs)
        values = jnp.nan_to_num(
            jnp.squeeze(values, axis=-1), nan=0.0, posinf=10.0, neginf=-10.0
        )
        return logits, values, jnp.zeros_like(values), new_carry


def make_config(seed: int, total_steps: int, num_envs: int, log_interval: int) -> dict:
    return {
        "NUM_ENVS": num_envs,
        "NUM_STEPS": 64,
        "TOTAL_TIMESTEPS": total_steps,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 8,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "LR": 3e-4,
        "ANNEAL_LR": True,
        "OPTIMISTIC_RESETS": False,
        "LOG_INTERVAL": log_interval,
        "SEED": seed,
        "RNN": {
            # Routes the shared four-output wrapper through the existing trainer.
            "RNN_TYPE": "sdlg_hybrid",
            "D_MODEL": 128,
            "N_LAYERS": 3,
            "N_HEADS": 4,
            "D_HEAD": 32,
            "D_K": 32,
            "D_V": 32,
            "D_FFC": 512,
            "WINDOW_SIZE": 16,
            "MEMORY_LEN": 128,
            "ROPE_VARIANT": "none",
        },
    }


def make_carry(model_name: str, config: dict, batch_size: int):
    rnn_cfg = config["RNN"]
    if model_name == "SdLG-Exact":
        unbatched = UnbatchedExactSdLG.initialize_carry(
            rnn_cfg["N_LAYERS"],
            rnn_cfg["N_HEADS"],
            rnn_cfg["D_K"],
            rnn_cfg["D_V"],
            rnn_cfg["WINDOW_SIZE"],
        )
        return jax.tree_util.tree_map(
            lambda leaf: jnp.repeat(leaf[jnp.newaxis, ...], batch_size, axis=0),
            unbatched,
        )
    if model_name == "PPO-GRU":
        return tuple(
            BatchedGRU.initialize_carry(batch_size, rnn_cfg["D_MODEL"])
            for _ in range(rnn_cfg["N_LAYERS"])
        )
    if model_name == "PPO-GTrXL":
        return BatchedGTrXL.initialize_carry(
            batch_size,
            rnn_cfg["MEMORY_LEN"],
            rnn_cfg["D_MODEL"],
            rnn_cfg["N_LAYERS"],
        )
    raise ValueError(model_name)


def parameter_count(params: Any) -> int:
    return int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(params)))


def build_evaluator(
    env: PureJaxRLXMiniGridWrapper,
    network: ActorCriticDoorKey,
    model_name: str,
    config: dict,
    num_eval_envs: int,
):
    eval_env = BatchEnvWrapper(env, num_eval_envs)
    max_steps = int(env.default_params.max_steps)

    @jax.jit
    def evaluate(params, eval_seed):
        rng = jax.random.PRNGKey(eval_seed)
        rng, reset_rng = jax.random.split(rng)
        obs, env_state = eval_env.reset(reset_rng, env.default_params)
        recurrent_state = make_carry(model_name, config, num_eval_envs)
        dones = jnp.zeros((num_eval_envs,), dtype=jnp.bool_)
        active = jnp.ones((num_eval_envs,), dtype=jnp.float32)
        episode_returns = jnp.zeros((num_eval_envs,), dtype=jnp.float32)
        episode_lengths = jnp.zeros((num_eval_envs,), dtype=jnp.int32)

        def eval_step(carry, _):
            (
                obs_t,
                env_state_t,
                recurrent_state_t,
                dones_t,
                active_t,
                returns_t,
                lengths_t,
                rng_t,
            ) = carry
            rng_t, action_rng, env_rng = jax.random.split(rng_t, 3)
            logits, _, _, next_recurrent_state = network.apply(
                params,
                recurrent_state_t,
                (obs_t[jnp.newaxis, ...], dones_t[jnp.newaxis, ...]),
            )
            actions = distrax.Categorical(logits=logits.squeeze(0)).sample(seed=action_rng)
            next_obs, next_env_state, rewards, next_dones, _ = eval_env.step(
                env_rng, env_state_t, actions, env.default_params
            )
            next_returns = returns_t + rewards * active_t
            next_lengths = lengths_t + active_t.astype(jnp.int32)
            next_active = active_t * (1.0 - next_dones.astype(jnp.float32))
            return (
                next_obs,
                next_env_state,
                next_recurrent_state,
                next_dones,
                next_active,
                next_returns,
                next_lengths,
                rng_t,
            ), None

        final, _ = jax.lax.scan(
            eval_step,
            (
                obs,
                env_state,
                recurrent_state,
                dones,
                active,
                episode_returns,
                episode_lengths,
                rng,
            ),
            None,
            length=max_steps,
        )
        returns = final[5]
        lengths = final[6]
        completion = 1.0 - final[4]
        return {
            "mean_return": jnp.mean(returns),
            "success_rate": jnp.mean((returns > 0.0).astype(jnp.float32)),
            "completion_rate": jnp.mean(completion),
            "mean_length": jnp.sum(lengths * completion) / jnp.maximum(1.0, jnp.sum(completion)),
        }

    return evaluate


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def read_completed(summary_path: Path) -> set[tuple[str, str, int]]:
    if not summary_path.exists():
        return set()
    with summary_path.open(newline="") as handle:
        return {
            (row["environment"], row["model"], int(row["seed"]))
            for row in csv.DictReader(handle)
            if row.get("status") == "complete"
        }


def append_csv(path: Path, fieldnames: list[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = results_dir / "training_trajectory.csv"
    summary_path = results_dir / "run_summary.csv"
    manifest_path = results_dir / "manifest.json"

    manifest = {
        "study_id": "MINIGRID-DOORKEY-BASELINES-2026",
        "git_commit": git_commit(),
        "created_unix": time.time(),
        "environments": args.envs,
        "models": args.models,
        "seeds": args.seeds,
        "total_step_cap": args.total_steps,
        "early_stop": {
            "training_return_threshold": args.early_stop_threshold,
            "consecutive_logging_intervals": args.early_stop_patience,
        },
        "terminal_evaluation_episodes": args.eval_episodes,
        "evaluation_policy": "stochastic categorical policy; fixed disjoint seeds",
        "comparison_scope": "shared encoder, PPO settings, rollout length, training budget, and evaluation",
        "capacity_matching": "not exact; parameter counts are recorded per model",
    }
    if manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text())
        for list_field in ("environments", "models", "seeds"):
            manifest[list_field] = list(
                dict.fromkeys(previous_manifest.get(list_field, []) + manifest[list_field])
            )
        manifest["created_unix"] = previous_manifest.get(
            "created_unix", manifest["created_unix"]
        )
        manifest["updated_unix"] = time.time()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    completed = read_completed(summary_path)
    trajectory_fields = [
        "environment",
        "model",
        "seed",
        "step",
        "train_return",
        "train_length",
        "train_entropy",
        "train_value_loss",
        "sps",
        "wall_seconds",
    ]
    summary_fields = [
        "environment",
        "model",
        "seed",
        "status",
        "stop_reason",
        "terminal_step",
        "parameter_count",
        "train_return",
        "eval_mean_return",
        "eval_success_rate",
        "eval_completion_rate",
        "eval_mean_length",
        "sps",
        "wall_seconds",
        "git_commit",
    ]

    import src.models.purejaxrl.wrappers as wrappers
    import src.trainers.purejax_trainer as trainer_module

    # GTrXL carries a Boolean attention mask.  The generic trainer's reset
    # helper multiplies every carry leaf by a float mask, which otherwise
    # changes that leaf's dtype and makes lax.scan reject the carry.
    def dtype_preserving_flush(carry, last_done):
        def flush_leaf(leaf):
            extra_dims = leaf.ndim - last_done.ndim
            done = jnp.expand_dims(
                last_done, axis=tuple(range(1, 1 + extra_dims))
            )
            if leaf.dtype == jnp.bool_:
                return jnp.where(done, jnp.zeros_like(leaf), leaf)
            return (leaf * (1.0 - done)).astype(leaf.dtype)

        return jax.tree_util.tree_map(flush_leaf, carry)

    trainer_module.flush_carry_state = dtype_preserving_flush

    for env_name in args.envs:
        env = PureJaxRLXMiniGridWrapper(env_name)
        for model_name in args.models:
            if model_name not in DEFAULT_MODELS:
                raise ValueError(f"Unsupported model {model_name}; choose from {DEFAULT_MODELS}")
            for seed in args.seeds:
                key = (env_name, model_name, seed)
                if key in completed and not args.rerun:
                    print(f"[skip] {env_name} | {model_name} | seed {seed}", flush=True)
                    continue

                jax.clear_caches()
                gc.collect()
                config = make_config(seed, args.total_steps, args.num_envs, args.log_interval)
                network_factory = functools.partial(ActorCriticDoorKey, model_name=model_name)
                wrappers.ActorCriticSdLGHybrid = network_factory
                network = ActorCriticDoorKey(
                    action_dim=env.action_space().n,
                    config=config,
                    model_name=model_name,
                )
                carry_init = functools.partial(make_carry, model_name, config)

                run_name = f"{env_name}-{model_name}-seed-{seed}"
                print(f"\n[start] {run_name}", flush=True)
                wb_run = wandb.init(
                    project=args.wandb_project,
                    group="revision-2026-doorkey-baselines",
                    name=run_name,
                    config={**config, "ENV_NAME": env_name, "MODEL": model_name},
                    mode=args.wandb_mode,
                    reinit=True,
                    tags=["doorkey", "baseline", "revision-2026"],
                )

                rng = jax.random.PRNGKey(seed)
                update_fn, runner_state = make_update(
                    rng,
                    config,
                    env,
                    env.default_params,
                    lambda: network,
                    carry_init,
                )
                count = parameter_count(runner_state[0].params)
                evaluate = build_evaluator(
                    env, network, model_name, config, args.eval_episodes
                )

                @functools.partial(jax.jit, donate_argnums=(0,))
                def update_and_reduce(state):
                    output = update_fn(state)
                    metrics = output["metric"]
                    done_mask = metrics.get(
                        "returned_episode",
                        jnp.ones_like(metrics.get("entropy", jnp.zeros(1))),
                    )

                    def completed_mean(values):
                        return jnp.sum(values * done_mask) / jnp.maximum(
                            1.0, jnp.sum(done_mask)
                        )

                    return output["runner_state"], {
                        "train_return": completed_mean(
                            metrics.get(
                                "returned_episode_returns", jnp.zeros_like(done_mask)
                            )
                        ),
                        "train_length": completed_mean(
                            metrics.get(
                                "returned_episode_lengths", jnp.zeros_like(done_mask)
                            )
                        ),
                        "train_entropy": jnp.mean(metrics.get("entropy", jnp.zeros(1))),
                        "train_value_loss": jnp.mean(
                            metrics.get("value_loss", jnp.zeros(1))
                        ),
                    }

                steps_per_loop = (
                    config["NUM_ENVS"] * config["NUM_STEPS"] * config["LOG_INTERVAL"]
                )
                total_loops = config["TOTAL_TIMESTEPS"] // steps_per_loop
                start = time.time()
                early_counter = 0
                stop_reason = "step_cap"
                last_metrics = None
                steps = 0

                for loop_index in range(total_loops):
                    runner_state, reduced = update_and_reduce(runner_state)
                    runner_state = jax.block_until_ready(runner_state)
                    reduced = jax.device_get(reduced)
                    steps += steps_per_loop
                    wall = time.time() - start
                    sps = steps / max(1.0, wall)
                    last_metrics = {name: float(value) for name, value in reduced.items()}
                    train_return = last_metrics["train_return"]

                    append_csv(
                        trajectory_path,
                        trajectory_fields,
                        {
                            "environment": env_name,
                            "model": model_name,
                            "seed": seed,
                            "step": steps,
                            **last_metrics,
                            "sps": sps,
                            "wall_seconds": wall,
                        },
                    )
                    wandb.log(
                        {
                            "Train/Mean_Return": train_return,
                            "Train/Mean_Length": last_metrics["train_length"],
                            "Train/Policy_Entropy": last_metrics["train_entropy"],
                            "Train/Value_Loss": last_metrics["train_value_loss"],
                            "System/SPS": sps,
                        },
                        step=steps,
                    )
                    print(
                        f"  [{loop_index + 1:02d}/{total_loops:02d}] "
                        f"{steps / 1e6:5.2f}M | return {train_return:6.3f} | "
                        f"{sps:,.0f} SPS",
                        flush=True,
                    )

                    early_counter = (
                        early_counter + 1
                        if train_return >= args.early_stop_threshold
                        else 0
                    )
                    if early_counter >= args.early_stop_patience:
                        stop_reason = "early_stop"
                        break

                eval_seed = 100_000 + seed
                eval_metrics = jax.device_get(evaluate(runner_state[0].params, eval_seed))
                eval_metrics = {name: float(value) for name, value in eval_metrics.items()}
                wall = time.time() - start
                sps = steps / max(1.0, wall)
                wandb.log(
                    {
                        "Eval/Mean_Return": eval_metrics["mean_return"],
                        "Eval/Success_Rate": eval_metrics["success_rate"],
                        "Eval/Completion_Rate": eval_metrics["completion_rate"],
                        "Eval/Mean_Length": eval_metrics["mean_length"],
                        "Run/Terminal_Step": steps,
                        "Run/Parameter_Count": count,
                    },
                    step=steps,
                )
                append_csv(
                    summary_path,
                    summary_fields,
                    {
                        "environment": env_name,
                        "model": model_name,
                        "seed": seed,
                        "status": "complete",
                        "stop_reason": stop_reason,
                        "terminal_step": steps,
                        "parameter_count": count,
                        "train_return": last_metrics["train_return"],
                        "eval_mean_return": eval_metrics["mean_return"],
                        "eval_success_rate": eval_metrics["success_rate"],
                        "eval_completion_rate": eval_metrics["completion_rate"],
                        "eval_mean_length": eval_metrics["mean_length"],
                        "sps": sps,
                        "wall_seconds": wall,
                        "git_commit": manifest["git_commit"],
                    },
                )
                print(
                    f"[done] {run_name} | {stop_reason} at {steps / 1e6:.2f}M | "
                    f"eval return {eval_metrics['mean_return']:.3f} | "
                    f"success {eval_metrics['success_rate']:.3f}",
                    flush=True,
                )
                wb_run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", default=list(DEFAULT_ENVS))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--total-steps", type=int, default=20_000_000)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=256)
    parser.add_argument("--early-stop-threshold", type=float, default=0.96)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument(
        "--results-dir",
        default=str(PROJECT_ROOT / "results" / "minigrid_doorkey_baselines_2026"),
    )
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "sdlgHybrid"))
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.getenv("WANDB_MODE", "offline"),
    )
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("HSA_ENABLE_SDMA", "0")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    run(parse_args())
