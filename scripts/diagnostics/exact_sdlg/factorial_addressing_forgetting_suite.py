#!/usr/bin/env python3
"""
⚡ FACTORIAL ADDRESSING, SELECTIVE FORGETTING & HEAD SCALING SUITE
========================================================================
1. Full (d_k x d_v) Factorial Capacity Sweep: H1 (f(d_k)) vs H2 (f(d_k * d_v))
2. Selective Forgetting Benchmark: Erase(k_1) -> Forget(k_1) vs Retain(k_2) (Delta vs ADAM)
3. Multi-Head Memory: Fixed d_k^{head}=32 (total d_k = G * 32)
4. Adaptive/Gated Key Allocation: Static d_k vs Dynamic Basis Gating d_k(N)
"""

import os
import sys
import time
import functools
import argparse
from pathlib import Path
import hashlib
import platform
import subprocess
import json
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

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

OUTPUT_DIR = os.path.join(project_root, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================================
# 1. Full (d_k x d_v) Factorial Capacity Sweep
# =====================================================================

@functools.partial(jax.jit, static_argnums=(2, 3))
def run_factorial_step(keys, vals, variant="delta", eta=0.25):
    T, d_k = keys.shape
    _, d_v = vals.shape
    M = jnp.zeros((d_k, d_v), dtype=jnp.float32)
    
    def step_fn(M_t, inputs):
        k_t, v_t = inputs
        v_hat = M_t.T @ k_t
        target_val = (v_t - v_hat) if variant == "delta" else v_t
        k_sq = jnp.sum(k_t ** 2) + 1e-5
        bet_eff = (eta * 0.5) / k_sq
        M_next = M_t + bet_eff * jnp.outer(k_t, target_val)
        return M_next, v_hat

    M_final, _ = jax.lax.scan(step_fn, M, (keys, vals))
    v_hats = keys @ M_final
    cos_sim = jnp.mean(jnp.sum(v_hats * vals, axis=-1) / (jnp.linalg.norm(v_hats, axis=-1) * jnp.linalg.norm(vals, axis=-1) + 1e-5))
    return cos_sim


def run_factorial_dk_dv_experiment():
    print("=" * 80)
    print("🧪 EXPERIMENT 1: Full (d_k x d_v) Factorial Capacity Sweep")
    print("=" * 80)
    
    rng = jax.random.PRNGKey(42)
    dk_list = [16, 32, 64, 128]
    dv_list = [8, 16, 32, 64]
    loads = [32, 64, 128]
    
    results = []
    
    for N in loads:
        for d_k in dk_list:
            for d_v in dv_list:
                rng, k1, k2 = jax.random.split(rng, 3)
                keys = jax.random.normal(k1, (N, d_k))
                vals = jax.random.normal(k2, (N, d_v))
                keys = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + 1e-5)
                vals = vals / (jnp.linalg.norm(vals, axis=-1, keepdims=True) + 1e-5)
                
                cos_add = float(run_factorial_step(keys, vals, "additive", 1.0))
                cos_delta = float(run_factorial_step(keys, vals, "delta", 0.25))
                
                results.append({
                    "N": N,
                    "d_k": d_k,
                    "d_v": d_v,
                    "matrix_params": d_k * d_v,
                    "ratio_N_dk": N / d_k,
                    "cos_additive": cos_add,
                    "cos_delta": cos_delta,
                })
                
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "factorial_dk_dv_sweep.csv"), index=False)
    
    print("\n--- Summary: Load N=128 across (d_k x d_v) ---")
    df_128 = df[df["N"] == 128].pivot(index="d_k", columns="d_v", values="cos_delta")
    print("Delta Retrieval Cosine Similarity (N=128):")
    print(df_128.to_string())
    return df


# =====================================================================
# 2. Selective Forgetting Benchmark (Delta vs ADAM Erase Gate)
# =====================================================================

@functools.partial(jax.jit, static_argnums=(3, 4))
def run_selective_forgetting_step(keys, vals, erase_mask, variant="adam", eta=0.25):
    """
    Sequence logic: Store (k_i, v_i). When erase_mask[t] == 1, issue erase command for k_t.
    Erase gate d_t = sigmoid(3.0 - 6.0 * erase_mask[t]) => 0.95 when 0, 0.05 when 1.
    """
    T, d_k = keys.shape
    _, d_v = vals.shape
    M = jnp.zeros((d_k, d_v), dtype=jnp.float32)
    
    def step_fn(M_t, inputs):
        k_t, v_t, em_t = inputs
        v_hat = M_t.T @ k_t
        
        target_val = (v_t - v_hat) if variant in ("delta", "adam") else v_t
        k_sq = jnp.sum(k_t ** 2) + 1e-5
        bet_eff = (eta * 0.5) / k_sq
        
        if variant == "adam":
            # Gated erasure: when em_t == 1, d_t drops to ~0.05
            d_t = jax.nn.sigmoid(3.0 - 6.0 * em_t)
        else:
            # Delta has no explicit erase gate (d_t = 1.0)
            d_t = 1.0
            
        M_next = d_t * M_t + bet_eff * jnp.outer(k_t, target_val)
        return M_next, v_hat

    M_final, _ = jax.lax.scan(step_fn, M, (keys, vals, erase_mask))
    return M_final


def run_selective_forgetting_experiment():
    print("\n" + "=" * 80)
    print("🧪 EXPERIMENT 2: Selective Forgetting Benchmark (Delta vs ADAM)")
    print("=" * 80)
    
    rng = jax.random.PRNGKey(42)
    d_k, d_v = 16, 16
    loads = [4, 8, 16, 32]
    
    results = []
    
    for N in loads:
        rng, k1, k2 = jax.random.split(rng, 3)
        raw_keys = jax.random.normal(k1, (N, d_k))
        raw_vals = jax.random.normal(k2, (N, d_v))
        keys = raw_keys / (jnp.linalg.norm(raw_keys, axis=-1, keepdims=True) + 1e-5)
        vals = raw_vals / (jnp.linalg.norm(raw_vals, axis=-1, keepdims=True) + 1e-5)
        
        # Step 1-N: Write N pairs
        # Step N+1: Issue Erase command for key 0 (k_0, 0_v, erase_mask=1)
        erase_keys = jnp.concatenate([keys, keys[:1]], axis=0)  # [N+1, d_k]
        erase_vals = jnp.concatenate([vals, jnp.zeros((1, d_v))], axis=0)  # [N+1, d_v]
        erase_mask = jnp.concatenate([jnp.zeros(N), jnp.ones(1)], axis=0)  # [N+1]
        
        for variant in ["additive", "delta", "adam"]:
            M_final = run_selective_forgetting_step(erase_keys, erase_vals, erase_mask, variant=variant, eta=0.25)
            
            v_hats = keys @ M_final  # Read state after erase
            
            # Forget accuracy for target key 0: 1.0 - cos(v_hat[0], vals[0]) -> high means key 0 was successfully erased!
            target_erased_cos = float(jnp.sum(v_hats[0] * vals[0]) / (jnp.linalg.norm(v_hats[0]) * jnp.linalg.norm(vals[0]) + 1e-5))
            forget_accuracy = 1.0 - max(0.0, target_erased_cos)
            
            # Retention accuracy for non-erased keys 1..N-1
            retention_cos = float(jnp.mean(jnp.sum(v_hats[1:] * vals[1:], axis=-1) / (jnp.linalg.norm(v_hats[1:], axis=-1) * jnp.linalg.norm(vals[1:], axis=-1) + 1e-5)))
            
            results.append({
                "N": N,
                "variant": variant,
                "target_residual_cos": target_erased_cos,
                "forget_accuracy": forget_accuracy,
                "retained_non_target_cos": retention_cos,
            })
            
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "selective_forgetting_study.csv"), index=False)
    print(df.to_string(index=False))
    return df


# =====================================================================
# 3. Multi-Head Memory (Fixed d_k^{head} = 32 vs Fixed Total d_k)
# =====================================================================

@functools.partial(jax.jit, static_argnums=(2, 3))
def run_fixed_head_dim_step(keys, vals, G=1, eta=0.25):
    """
    Evaluates multi-head memory where EACH head has fixed d_k_head=32 and d_v_head=16.
    Total d_k = G * 32, Total d_v = G * 16.
    """
    T, d_k_total = keys.shape
    _, d_v_total = vals.shape
    
    d_k_head = 32
    d_v_head = 16
    
    keys_h = keys.reshape(T, G, d_k_head)
    vals_h = vals.reshape(T, G, d_v_head)
    
    M_h = jnp.zeros((G, d_k_head, d_v_head), dtype=jnp.float32)
    
    def step_fn(M_t, inputs):
        k_t, v_t = inputs  # [G, d_k_head], [G, d_v_head]
        v_hat = jnp.einsum('g k v, g k -> g v', M_t, k_t)
        target_val = v_t - v_hat
        k_sq = jnp.sum(k_t ** 2, axis=-1, keepdims=True) + 1e-5
        bet_eff = (eta * 0.5) / k_sq
        updates = jnp.einsum('g k, g v -> g k v', k_t, target_val)
        M_next = M_t + bet_eff[..., None] * updates
        return M_next, v_hat

    M_final, _ = jax.lax.scan(step_fn, M_h, (keys_h, vals_h))
    
    v_hats_h = jnp.einsum('g k v, t g k -> t g v', M_final, keys_h)
    v_hats = v_hats_h.reshape(T, d_v_total)
    
    cos_sim = jnp.mean(jnp.sum(v_hats * vals, axis=-1) / (jnp.linalg.norm(v_hats, axis=-1) * jnp.linalg.norm(vals, axis=-1) + 1e-5))
    return cos_sim


def run_fixed_head_capacity_experiment():
    print("\n" + "=" * 80)
    print("🧪 EXPERIMENT 3: Multi-Head Memory under Fixed d_k^{head} = 32")
    print("=" * 80)
    
    rng = jax.random.PRNGKey(42)
    heads = [1, 2, 4, 8]
    loads = [32, 64, 128]
    
    results = []
    
    for N in loads:
        for G in heads:
            rng, k1, k2 = jax.random.split(rng, 3)
            d_k_total = G * 32
            d_v_total = G * 16
            
            keys = jax.random.normal(k1, (N, d_k_total))
            vals = jax.random.normal(k2, (N, d_v_total))
            keys = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + 1e-5)
            vals = vals / (jnp.linalg.norm(vals, axis=-1, keepdims=True) + 1e-5)
            
            cos_delta = float(run_fixed_head_dim_step(keys, vals, G=G, eta=0.25))
            
            results.append({
                "N": N,
                "G_heads": G,
                "d_k_head": 32,
                "total_d_k": d_k_total,
                "delta_cos_retrieval": cos_delta,
            })
            
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "fixed_head_capacity_study.csv"), index=False)
    print(df.to_string(index=False))
    return df


# =====================================================================
# Component-transfer diagnostic: controlled reassignment / overwrite
# =====================================================================

def _unit_normal(rng, shape):
    values = jax.random.normal(rng, shape)
    return values / (jnp.linalg.norm(values, axis=-1, keepdims=True) + 1e-6)


def _cosine_rows(predictions, targets):
    numerator = jnp.sum(predictions * targets, axis=-1)
    denominator = jnp.linalg.norm(predictions, axis=-1) * jnp.linalg.norm(targets, axis=-1) + 1e-6
    return numerator / denominator


def _write_memory(memory, key, value, formulation, delta_eta):
    """One explicitly specified synthetic memory update.

    ``gated_additive_proxy`` is deliberately a fixed-gate proxy, not a
    simulation of learned SdLG gates.  It is included only to expose the
    retention/overwrite trade-off of global decay plus additive writes under
    exactly the same draw and read protocol as the other formulations.
    """
    prediction = memory.T @ key
    if formulation == "additive":
        return memory + jnp.outer(key, value)
    if formulation == "normalized_delta":
        update = delta_eta / (jnp.dot(key, key) + 1e-6)
        return memory + update * jnp.outer(key, value - prediction)
    if formulation == "gated_additive_proxy":
        return 0.995 * memory + jnp.outer(key, value)
    raise ValueError(f"Unknown formulation: {formulation}")


def _run_overwrite_sequence(keys, original_values, replacement_values, formulation, delta_eta):
    """Write all associations, then overwrite the first quarter in order."""
    memory = jnp.zeros((keys.shape[-1], original_values.shape[-1]), dtype=jnp.float32)
    for key, value in zip(keys, original_values):
        memory = _write_memory(memory, key, value, formulation, delta_eta)

    target_count = max(1, keys.shape[0] // 4)
    for index in range(target_count):
        memory = _write_memory(memory, keys[index], replacement_values[index], formulation, delta_eta)

    predictions = keys @ memory
    target_cos = _cosine_rows(predictions[:target_count], replacement_values[:target_count]).mean()
    old_target_cos = _cosine_rows(predictions[:target_count], original_values[:target_count]).mean()
    non_target_cos = _cosine_rows(predictions[target_count:], original_values[target_count:]).mean()
    expected = jnp.concatenate([replacement_values[:target_count], original_values[target_count:]], axis=0)
    all_cos = _cosine_rows(predictions, expected).mean()
    return {
        "retrieval_cosine_after_overwrite": float(all_cos),
        "target_overwrite_fidelity": float(target_cos),
        "target_old_value_cosine": float(old_target_cos),
        "non_target_preservation": float(non_target_cos),
        "state_frobenius_norm": float(jnp.linalg.norm(memory)),
        "finite_state": bool(jnp.all(jnp.isfinite(memory))),
        "overwrite_targets": target_count,
    }


def run_component_transfer_overwrite(output_dir):
    """Run the predeclared multi-seed additive-versus-delta overwrite study."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    formulations = (("additive", 1.0), ("gated_additive_proxy", 1.0)) + tuple(
        ("normalized_delta", eta) for eta in (0.25, 0.5, 1.0)
    )
    for seed in range(10):
        for N in (32, 64, 128):
            for d_k in (16, 32, 64, 128):
                for d_v in (8, 16, 32, 64):
                    base = jax.random.PRNGKey(seed)
                    # The complete key/value/reassignment draw is shared over formulations.
                    base = jax.random.fold_in(base, N)
                    base = jax.random.fold_in(base, d_k)
                    base = jax.random.fold_in(base, d_v)
                    key_rng, value_rng, replacement_rng = jax.random.split(base, 3)
                    keys = _unit_normal(key_rng, (N, d_k))
                    original_values = _unit_normal(value_rng, (N, d_v))
                    replacement_values = _unit_normal(replacement_rng, (N, d_v))
                    for formulation, delta_eta in formulations:
                        metrics = _run_overwrite_sequence(
                            keys, original_values, replacement_values, formulation, delta_eta
                        )
                        rows.append({
                            "study_id": "CTV-A1-ASSOCIATIVE-OVERWRITE",
                            "seed": seed,
                            "N": N,
                            "d_k": d_k,
                            "d_v": d_v,
                            "load_ratio": N / d_k,
                            "formulation": formulation,
                            "delta_eta": delta_eta,
                            **metrics,
                        })
    raw = pd.DataFrame(rows)
    summary = raw.groupby(["N", "d_k", "d_v", "load_ratio", "formulation", "delta_eta"], as_index=False).agg(
        n=("seed", "count"),
        retrieval_cosine_mean=("retrieval_cosine_after_overwrite", "mean"),
        retrieval_cosine_sd=("retrieval_cosine_after_overwrite", "std"),
        overwrite_fidelity_mean=("target_overwrite_fidelity", "mean"),
        overwrite_fidelity_sd=("target_overwrite_fidelity", "std"),
        non_target_preservation_mean=("non_target_preservation", "mean"),
        non_target_preservation_sd=("non_target_preservation", "std"),
        state_norm_mean=("state_frobenius_norm", "mean"),
        all_finite=("finite_state", "all"),
    )
    raw.to_csv(output_dir / "associative_overwrite_raw.csv", index=False)
    summary.to_csv(output_dir / "associative_overwrite_summary.csv", index=False)
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
        ).strip()
    except subprocess.CalledProcessError:
        git_commit = "unavailable"
    with open(__file__, "rb") as handle:
        source_sha256 = hashlib.sha256(handle.read()).hexdigest()
    (output_dir / "associative_overwrite_manifest.json").write_text(json.dumps({
        "study_id": "CTV-A1-ASSOCIATIVE-OVERWRITE",
        "runner": os.path.relpath(__file__, project_root),
        "runner_sha256": source_sha256,
        "git_commit": git_commit,
        "jax_version": jax.__version__,
        "platform": platform.platform(),
        "backend": jax.default_backend(),
        "seeds": list(range(10)),
        "loads_N": [32, 64, 128],
        "key_dimensions": [16, 32, 64, 128],
        "value_dimensions": [8, 16, 32, 64],
        "formulations": {
            "additive": "M <- M + k v^T",
            "normalized_delta": "M <- M + eta/(||k||^2+eps) k (v-M^T k)^T",
            "gated_additive_proxy": "M <- 0.995 M + k v^T; fixed proxy, not learned SdLG gates",
        },
        "overwrite_protocol": "write N associations, then overwrite first floor(N/4) keys with replacement values",
        "shared_draws": True,
        "statistical_unit": "synthetic random draw",
    }, indent=2) + "\n")
    print(f"wrote {len(raw)} raw rows and {len(summary)} summaries to {output_dir}")
    return raw, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-transfer-overwrite", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(project_root, "papers", "writing", "generated_artifacts", "component_transfer"),
    )
    args = parser.parse_args()
    if args.component_transfer_overwrite:
        run_component_transfer_overwrite(args.output_dir)
        return

    start_time = time.time()
    run_factorial_dk_dv_experiment()
    run_selective_forgetting_experiment()
    run_fixed_head_capacity_experiment()
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"🎉 SUITE COMPLETE (Total Time: {elapsed:.2f}s)")
    print("=" * 80)


if __name__ == "__main__":
    main()
