#!/usr/bin/env python3
"""
⚡ DIAGNOSTIC MINI-STUDY SUITE
========================================================================
1. Delta Step-Size Scale Sweep (eta in {0.125, 0.25, 0.5, 1.0, 2.0})
2. Key Normalization Independence Test (Additive vs Additive+KeyNorm vs Delta+KeyNorm)
3. Overwrite Under Load Benchmark (Target Fidelity vs Non-Target Preservation)
4. ADAM Erase Gate Activation Diagnostic (E[d_t^k], Var, P(<0.5), P(>0.9))
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
import optax
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
# Standalone Matrix Recurrence Kernels for Direct Diagnostic Control
# =====================================================================

@functools.partial(jax.jit, static_argnums=(3, 4, 5))
def run_recurrent_memory_step(keys, vals, eta=1.0, variant="delta", key_norm=True, erase_bias=3.0):
    """
    Simulates matrix recurrence step-by-step over sequence T:
    M_t = diag(d_t) M_{t-1} + bet_eff * k_t (v_t - v_hat)^T
    """
    T, d_k = keys.shape
    _, d_v = vals.shape
    
    M = jnp.zeros((d_k, d_v), dtype=jnp.float32)
    z = jnp.zeros((d_k,), dtype=jnp.float32)
    
    d_t_list = []
    
    def step_fn(carry, inputs):
        M_t, z_t = carry
        k_t, v_t = inputs
        
        # Read prediction
        v_hat = M_t.T @ k_t  # [d_v]
        
        if variant in ("delta", "adam"):
            target_val = v_t - v_hat
        else:
            target_val = v_t
            
        k_sq = jnp.sum(k_t ** 2) + 1e-5
        if key_norm:
            bet_eff = (eta * 0.5) / k_sq
        else:
            bet_eff = eta * 0.5
            
        if variant == "adam":
            # Erase gate simulation: sigmoid(bias)
            d_t = jax.nn.sigmoid(erase_bias)
        else:
            d_t = 1.0
            
        M_next = d_t * M_t + bet_eff * jnp.outer(k_t, target_val)
        z_next = z_t + bet_eff * k_t
        
        return (M_next, z_next), (v_hat, d_t)

    (M_final, _), (v_hats, d_ts) = jax.lax.scan(step_fn, (M, z), (keys, vals))
    return M_final, v_hats, d_ts


# =====================================================================
# 1. Delta Step-Size Sweep & Key Normalization Independence
# =====================================================================

def run_step_size_keynorm_sweep():
    print("=" * 80)
    print("🧪 MINI-STUDY 1: Delta Step-Size Scale Sweep & Key-Norm Independence")
    print("=" * 80)
    
    rng = jax.random.PRNGKey(42)
    d_k, d_v = 16, 16
    etas = [0.125, 0.25, 0.5, 1.0, 2.0]
    loads = [4, 8, 16, 32, 64, 128]
    
    results = []
    
    for N in loads:
        rng, k1, k2 = jax.random.split(rng, 3)
        # Generate N key-value pairs
        raw_keys = jax.random.normal(k1, (N, d_k))
        raw_vals = jax.random.normal(k2, (N, d_v))
        
        # Normalize keys/vals to unit sphere for standardized test
        keys = raw_keys / (jnp.linalg.norm(raw_keys, axis=-1, keepdims=True) + 1e-5)
        vals = raw_vals / (jnp.linalg.norm(raw_vals, axis=-1, keepdims=True) + 1e-5)
        
        # Model A: Additive (raw key norm step)
        M_add_raw, _, _ = run_recurrent_memory_step(keys, vals, eta=1.0, variant="additive", key_norm=False)
        # Model B: Additive (key-normalized step)
        M_add_norm, _, _ = run_recurrent_memory_step(keys, vals, eta=1.0, variant="additive", key_norm=True)
        
        # Readout accuracy (cosine similarity across all N stored pairs)
        v_hats_add_raw = keys @ M_add_raw  # [N, d_v]
        v_hats_add_norm = keys @ M_add_norm
        
        cos_add_raw = jnp.mean(jnp.sum(v_hats_add_raw * vals, axis=-1) / (jnp.linalg.norm(v_hats_add_raw, axis=-1) * jnp.linalg.norm(vals, axis=-1) + 1e-5))
        cos_add_norm = jnp.mean(jnp.sum(v_hats_add_norm * vals, axis=-1) / (jnp.linalg.norm(v_hats_add_norm, axis=-1) * jnp.linalg.norm(vals, axis=-1) + 1e-5))
        
        for eta in etas:
            # Model C/D: Delta with key norm + step scale eta
            M_delta, _, _ = run_recurrent_memory_step(keys, vals, eta=eta, variant="delta", key_norm=True)
            v_hats_delta = keys @ M_delta
            cos_delta = jnp.mean(jnp.sum(v_hats_delta * vals, axis=-1) / (jnp.linalg.norm(v_hats_delta, axis=-1) * jnp.linalg.norm(vals, axis=-1) + 1e-5))
            
            results.append({
                "N": N,
                "eta": eta,
                "cos_add_raw": float(cos_add_raw),
                "cos_add_norm": float(cos_add_norm),
                "cos_delta": float(cos_delta),
            })
            
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "step_size_keynorm_sweep.csv"), index=False)
    
    print("\n--- Capacity Sweep across Step Scale (eta) & Key-Norm ---")
    print(df[df["eta"] == 1.0].to_string(index=False))
    return df


# =====================================================================
# 2. Overwrite Under Load Benchmark
# =====================================================================

def run_overwrite_under_load():
    print("\n" + "=" * 80)
    print("🧪 MINI-STUDY 2: Overwrite Under Load (Target Fidelity vs Non-Target Preservation)")
    print("=" * 80)
    
    rng = jax.random.PRNGKey(42)
    d_k, d_v = 16, 16
    loads = [4, 8, 16, 32, 64, 128]
    
    results = []
    
    for N in loads:
        rng, k1, k2, k3 = jax.random.split(rng, 4)
        raw_keys = jax.random.normal(k1, (N, d_k))
        raw_vals = jax.random.normal(k2, (N, d_v))
        keys = raw_keys / (jnp.linalg.norm(raw_keys, axis=-1, keepdims=True) + 1e-5)
        vals = raw_vals / (jnp.linalg.norm(raw_vals, axis=-1, keepdims=True) + 1e-5)
        
        # New value for target key (index 0)
        val_target_new = jax.random.normal(k3, (d_v,))
        val_target_new = val_target_new / (jnp.linalg.norm(val_target_new) + 1e-5)
        
        # Sequence: First write all N pairs, then overwrite key 0 with val_target_new
        seq_keys = jnp.concatenate([keys, keys[:1]], axis=0)  # [N+1, d_k]
        seq_vals = jnp.concatenate([vals, val_target_new[None, :]], axis=0)  # [N+1, d_v]
        
        for variant in ["additive", "delta"]:
            M_final, _, _ = run_recurrent_memory_step(seq_keys, seq_vals, eta=1.0, variant=variant, key_norm=True)
            
            # Read predictions for target and non-targets
            v_hats = keys @ M_final  # [N, d_v]
            
            # Target fidelity: cos(v_hat[0], val_target_new)
            target_cos = float(jnp.sum(v_hats[0] * val_target_new) / (jnp.linalg.norm(v_hats[0]) * jnp.linalg.norm(val_target_new) + 1e-5))
            
            # Non-target preservation: mean cos(v_hat[j], vals[j]) for j > 0
            non_target_cos = float(jnp.mean(jnp.sum(v_hats[1:] * vals[1:], axis=-1) / (jnp.linalg.norm(v_hats[1:], axis=-1) * jnp.linalg.norm(vals[1:], axis=-1) + 1e-5)))
            
            results.append({
                "N": N,
                "variant": variant,
                "target_fidelity": target_cos,
                "non_target_preservation": non_target_cos,
            })
            
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "overwrite_under_load.csv"), index=False)
    
    print("\n--- Overwrite Under Load (Target Fidelity vs Non-Target Preservation) ---")
    print(df.to_string(index=False))
    return df


# =====================================================================
# 3. ADAM Erase Gate Activation Diagnostic
# =====================================================================

def run_erase_gate_diagnostic():
    print("\n" + "=" * 80)
    print("🧪 MINI-STUDY 3: ADAM Erase Gate Activation Diagnostic")
    print("=" * 80)
    
    rng = jax.random.PRNGKey(42)
    d_k, d_v = 16, 16
    T = 128
    
    raw_keys = jax.random.normal(rng, (T, d_k))
    raw_vals = jax.random.normal(rng, (T, d_v))
    keys = raw_keys / (jnp.linalg.norm(raw_keys, axis=-1, keepdims=True) + 1e-5)
    vals = raw_vals / (jnp.linalg.norm(raw_vals, axis=-1, keepdims=True) + 1e-5)
    
    _, _, d_ts = run_recurrent_memory_step(keys, vals, eta=1.0, variant="adam", key_norm=True, erase_bias=3.0)
    
    mean_d = float(np.mean(d_ts))
    var_d = float(np.var(d_ts))
    p_low = float(np.mean(d_ts < 0.5))
    p_high = float(np.mean(d_ts > 0.9))
    
    print(f"  E[d_t]:     {mean_d:.4f}")
    print(f"  Var(d_t):   {var_d:.6f}")
    print(f"  P(d_t<0.5): {p_low*100:.1f}%")
    print(f"  P(d_t>0.9): {p_high*100:.1f}%")
    
    df = pd.DataFrame([{
        "mean_d": mean_d,
        "var_d": var_d,
        "p_low_0.5": p_low,
        "p_high_0.9": p_high
    }])
    df.to_csv(os.path.join(OUTPUT_DIR, "erase_gate_activation.csv"), index=False)
    return df


def main():
    start_time = time.time()
    df_step = run_step_size_keynorm_sweep()
    df_load = run_overwrite_under_load()
    df_erase = run_erase_gate_diagnostic()
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"🎉 DIAGNOSTIC MINI-STUDIES COMPLETE (Total Time: {elapsed:.2f}s)")
    print("=" * 80)


if __name__ == "__main__":
    main()
