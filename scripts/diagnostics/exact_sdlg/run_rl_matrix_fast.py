#!/usr/bin/env python3
"""
⚡ MECHANISTIC RL MATRIX EVALUATOR
========================================================================
Executes our 7-model RL mechanistic matrix directly using PureJAXRL PPO trainer:
Models:
1. SdLG-Exact             (Additive baseline, d_k=16, d_v=16)
2. SdLG-Delta             (Delta editing baseline, d_k=16, d_v=16)
3. SdLG-ADAM              (Erasable baseline, d_k=16, d_v=16)
4. SdLG-Delta-dk32        (Asymmetric capacity, d_k=32, d_v=16)
5. SdLG-Delta-dk64        (Asymmetric capacity, d_k=64, d_v=16)
6. SdLG-ADAM-dk64         (Composite: Erasable + Asymmetric, d_k=64, d_v=16)
7. SdLG-ADAM-Matched-dk64 (Parameter-matched control, d_k=64, d_v=16)

Environments:
- ConcentrationEasy  (Primary signal: Addressing Capacity d_k + Delta editing)
- RepeatPreviousEasy (Primary signal: Retention & Editing)
- CountRecallEasy    (Primary signal: Storage capacity)
- AutoencodeEasy     (Primary signal: Memory Recall)
"""

import os
import sys
import time
import functools
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import pandas as pd

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"

current_file_path = os.path.abspath(__file__)
project_root = current_file_path
while not os.path.exists(os.path.join(project_root, "src")):
    project_root = os.path.dirname(project_root)
    if project_root == "/":
        break
if project_root not in sys.path:
    sys.path.append(project_root)

from src.envs.popgym_wrapper import POPGymEnv, make_popgym_env
from src.trainers.purejax_trainer import make_update
import src.trainers.purejax_trainer as trainer_module
from src.models.purejaxrl.wrappers import (
    ActorCriticExactSdLG,
    ActorCriticSdLGDelta,
    ActorCriticSdLGADAM,
    ActorCriticSdLGADAMMatched,
)
from src.models.common.exact_sdlg import UnbatchedExactSdLG

OUTPUT_DIR = os.path.join(project_root, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_ENVS = [
    "ConcentrationEasy",
    "RepeatPreviousEasy",
    "CountRecallEasy",
    "AutoencodeEasy"
]

MODEL_SPECS = {
    "SdLG-Exact":             {"class": ActorCriticExactSdLG,       "rnn_type": "exact_sdlg", "d_k": 16, "d_v": 16, "variant": "additive"},
    "SdLG-Delta":             {"class": ActorCriticSdLGDelta,       "rnn_type": "sdlg_delta", "d_k": 16, "d_v": 16, "variant": "delta"},
    "SdLG-ADAM":              {"class": ActorCriticSdLGADAM,        "rnn_type": "sdlg_adam",  "d_k": 16, "d_v": 16, "variant": "adam"},
    "SdLG-Delta-dk32":        {"class": ActorCriticSdLGDelta,       "rnn_type": "sdlg_delta", "d_k": 32, "d_v": 16, "variant": "delta"},
    "SdLG-Delta-dk64":        {"class": ActorCriticSdLGDelta,       "rnn_type": "sdlg_delta", "d_k": 64, "d_v": 16, "variant": "delta"},
    "SdLG-ADAM-dk64":         {"class": ActorCriticSdLGADAM,        "rnn_type": "sdlg_adam",  "d_k": 64, "d_v": 16, "variant": "adam"},
    "SdLG-ADAM-Matched-dk64": {"class": ActorCriticSdLGADAMMatched, "rnn_type": "sdlg_adam",  "d_k": 64, "d_v": 16, "variant": "adam"},
}

def get_rnn_carry_init(n_layers, n_heads, d_k, d_v, window_size):
    def init_fn(batch_size):
        unbatched = UnbatchedExactSdLG.initialize_carry(
            n_layers=n_layers, n_heads=n_heads, d_k=d_k, d_v=d_v, window_size=window_size
        )
        return jax.tree_util.tree_map(
            lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0), unbatched
        )
    return init_fn

def run_single_model_env(env_name, model_name, seed=42, total_steps=200_000):
    spec = MODEL_SPECS[model_name]
    model_cls = spec["class"]
    d_k = spec["d_k"]
    d_v = spec["d_v"]
    print(f"    ▶ [{model_name}] Initializing ({d_k=}, {d_v=}, total_steps={total_steps:,})...", flush=True)
    
    num_envs = 64
    num_steps = 64
    steps_per_update = num_envs * num_steps
    log_interval = 10
    total_jit_calls = max(1, (total_steps // steps_per_update) // log_interval)
    
    config = {
        "LR": 3e-4,
        "NUM_ENVS": num_envs,
        "NUM_STEPS": num_steps,
        "TOTAL_TIMESTEPS": total_steps,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ANNEAL_LR": True,
        "DEBUG": False,
        "OPTIMISTIC_RESETS": False,
        "LOG_INTERVAL": log_interval,
        "RNN": {
            "RNN_TYPE": spec["rnn_type"],
            "N_LAYERS": 3,
            "N_HEADS": 4,
            "D_MODEL": 128,
            "D_FFC": 512,
            "D_K": d_k,
            "D_V": d_v,
            "WINDOW_SIZE": 64,
            "VARIANT": spec["variant"],
        },
        "SEED": seed,
    }
    
    env, env_params = make_popgym_env(env_name, num_envs=num_envs)
    trainer_module.ActorCriticRNN = model_cls
    rnn_carry_init_fn = get_rnn_carry_init(3, 4, d_k, d_v, 64)
    
    rng = jax.random.PRNGKey(seed)
    raw_update_fn, runner_state = make_update(
        rng, config, env, env_params, rnn_module=None, rnn_carry_init=rnn_carry_init_fn
    )
    
    def train_step(r_state):
        out = raw_update_fn(r_state)
        m = out["metric"]
        d_mask = m.get("returned_episode", jnp.ones_like(m.get("entropy", jnp.zeros(1))))
        def mean_d(x):
            return jnp.sum(x * d_mask) / jnp.maximum(1.0, jnp.sum(d_mask))
        return out["runner_state"], {
            "ret": mean_d(m.get("returned_episode_returns", jnp.zeros_like(d_mask))),
            "len": mean_d(m.get("returned_episode_lengths", jnp.zeros_like(d_mask))),
        }

    start_time = time.time()
    print(f"    ▶ [{model_name}] Initializing & JIT Compiling ({d_k=}, {d_v=}, total_steps={total_steps:,})...", flush=True)
    
    for u in range(total_jit_calls):
        runner_state, metrics = train_step(runner_state)
        runner_state = jax.block_until_ready(runner_state)
        metrics_curr = jax.device_get(metrics)
        ret_val = float(metrics_curr["ret"])
        step_num = (u + 1) * steps_per_update * log_interval
        print(f"      Step {step_num:>10,} / {total_steps:,} | Ret: {ret_val:>6.2f} | Elapsed: {(time.time() - start_time):.1f}s", flush=True)
        
    elapsed = time.time() - start_time
    
    final_ret = float(metrics["ret"])
    sps = float(total_steps / max(1e-5, elapsed))
    
    return {
        "env": env_name,
        "model": model_name,
        "d_k": d_k,
        "d_v": d_v,
        "seed": seed,
        "steps": total_steps,
        "final_return": final_ret,
        "sps": sps,
        "elapsed": elapsed
    }

def main():
    print("=" * 80)
    print("🚀 LAUNCHING PUREJAXRL MECHANISTIC MATRIX EVALUATION (200k steps/model)")
    print(f"Environments: {TARGET_ENVS}")
    print(f"Models ({len(MODEL_SPECS)}): {list(MODEL_SPECS.keys())}")
    print("=" * 80)
    
    results = []
    total_runs = len(TARGET_ENVS) * len(MODEL_SPECS)
    count = 0
    start_all = time.time()
    
    for env in TARGET_ENVS:
        print(f"\n🌍 Environment: {env}", flush=True)
        for model in MODEL_SPECS.keys():
            count += 1
            res = run_single_model_env(env, model, seed=42, total_steps=200_000)
            results.append(res)
            print(f"  [{count:2d}/{total_runs}] {model:<24} | Return: {res['final_return']:>6.2f} | SPS: {int(res['sps']):>5} ({res['elapsed']:.1f}s)", flush=True)
            
    df = pd.DataFrame(results)
    out_csv = os.path.join(OUTPUT_DIR, "mechanistic_rl_matrix_results.csv")
    df.to_csv(out_csv, index=False)
    
    print("\n" + "=" * 80)
    print(f"🎉 MATRIX EVALUATION COMPLETE in {(time.time() - start_all):.1f}s")
    print(f"Saved results to {out_csv}")
    print("=" * 80)
    
    print("\n--- Summary Table: Final Episodic Returns ---")
    piv = df.pivot(index="model", columns="env", values="final_return")
    print(piv.to_string())

if __name__ == "__main__":
    main()
