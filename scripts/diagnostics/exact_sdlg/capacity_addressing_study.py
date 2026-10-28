#!/usr/bin/env python3
"""
⚡ ROUND 3: ADDRESSING CAPACITY & KEY SEPARATION STUDY
========================================================================
1. Non-linear Key Geometry: Linear Key (W_k x) vs Non-linear Key (f_k(x))
2. Key-Value Dimension Asymmetry: d_k in {16, 32, 64, 128} with fixed d_v = 16
3. Multi-Head Memory Subspaces: G in {1, 2, 4, 8} heads
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
# Experiment 1: Non-linear Key Geometry
# =====================================================================

class NonLinearKeyNet(nn.Module):
    d_k: int

    @nn.compact
    def __call__(self, x):
        h = nn.Dense(self.d_k * 2)(x)
        h = nn.gelu(h)
        k = nn.Dense(self.d_k)(h)
        return k


@functools.partial(jax.jit, static_argnums=(2, 3))
def run_recurrent_key_test(keys, vals, variant="delta", eta=0.25):
    """
    Evaluates key retrieval performance for a given key-value matrix.
    """
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
    
    # Read predictions
    v_hats = keys @ M_final
    cos_sim = jnp.mean(jnp.sum(v_hats * vals, axis=-1) / (jnp.linalg.norm(v_hats, axis=-1) * jnp.linalg.norm(vals, axis=-1) + 1e-5))
    return cos_sim


def run_nonlinear_key_experiment():
    print("=" * 80)
    print("🧪 EXPERIMENT 1: Non-linear Key Geometry vs Linear Keys")
    print("=" * 80)
    
    rng = jax.random.PRNGKey(42)
    d_model, d_k, d_v = 32, 16, 16
    loads = [8, 16, 32, 64, 128]
    
    key_net = NonLinearKeyNet(d_k=d_k)
    params = key_net.init(rng, jnp.ones((1, d_model)))
    
    results = []
    
    for N in loads:
        rng, k1, k2 = jax.random.split(rng, 3)
        x_inputs = jax.random.normal(k1, (N, d_model))
        vals = jax.random.normal(k2, (N, d_v))
        vals = vals / (jnp.linalg.norm(vals, axis=-1, keepdims=True) + 1e-5)
        
        # 1. Linear keys
        keys_lin = jax.random.normal(k1, (N, d_k))
        keys_lin = keys_lin / (jnp.linalg.norm(keys_lin, axis=-1, keepdims=True) + 1e-5)
        
        # 2. Non-linear keys
        keys_nonlin = key_net.apply(params, x_inputs)
        keys_nonlin = keys_nonlin / (jnp.linalg.norm(keys_nonlin, axis=-1, keepdims=True) + 1e-5)
        
        # Crosstalk metric: E_{i != j} |<k_i, k_j>|
        gram_lin = jnp.abs(keys_lin @ keys_lin.T) - jnp.eye(N)
        crosstalk_lin = float(jnp.sum(gram_lin) / (N * (N - 1)))
        
        gram_nonlin = jnp.abs(keys_nonlin @ keys_nonlin.T) - jnp.eye(N)
        crosstalk_nonlin = float(jnp.sum(gram_nonlin) / (N * (N - 1)))
        
        # Retrieval performance
        rec_add_lin = float(run_recurrent_key_test(keys_lin, vals, variant="additive", eta=1.0))
        rec_delta_lin = float(run_recurrent_key_test(keys_lin, vals, variant="delta", eta=0.25))
        
        rec_add_nonlin = float(run_recurrent_key_test(keys_nonlin, vals, variant="additive", eta=1.0))
        rec_delta_nonlin = float(run_recurrent_key_test(keys_nonlin, vals, variant="delta", eta=0.25))
        
        results.append({
            "N": N,
            "crosstalk_lin": crosstalk_lin,
            "crosstalk_nonlin": crosstalk_nonlin,
            "delta_lin_retrieval": rec_delta_lin,
            "delta_nonlin_retrieval": rec_delta_nonlin,
            "add_lin_retrieval": rec_add_lin,
            "add_nonlin_retrieval": rec_add_nonlin,
        })
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "nonlinear_key_study.csv"), index=False)
    print(df.to_string(index=False))
    return df


# =====================================================================
# Experiment 2: Key-Value Dimension Asymmetry (d_k Scaling)
# =====================================================================

def run_dk_asymmetry_experiment():
    print("\n" + "=" * 80)
    print("🧪 EXPERIMENT 2: Key-Value Dimension Asymmetry (Scaling d_k with d_v = 16)")
    print("=" * 80)
    
    rng = jax.random.PRNGKey(42)
    d_v = 16
    dk_list = [16, 32, 64, 128]
    loads = [16, 32, 64, 128]
    
    results = []
    
    for N in loads:
        for d_k in dk_list:
            rng, k1, k2 = jax.random.split(rng, 3)
            keys = jax.random.normal(k1, (N, d_k))
            vals = jax.random.normal(k2, (N, d_v))
            keys = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + 1e-5)
            vals = vals / (jnp.linalg.norm(vals, axis=-1, keepdims=True) + 1e-5)
            
            cos_add = float(run_recurrent_key_test(keys, vals, variant="additive", eta=1.0))
            cos_delta = float(run_recurrent_key_test(keys, vals, variant="delta", eta=0.25))
            
            results.append({
                "N": N,
                "d_k": d_k,
                "ratio_N_dk": N / d_k,
                "cos_additive": cos_add,
                "cos_delta": cos_delta,
            })
            
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "dk_asymmetry_study.csv"), index=False)
    
    print("\n--- Key Dimension d_k Scaling under Load N=64 & N=128 ---")
    print(df[df["N"].isin([64, 128])].to_string(index=False))
    return df


# =====================================================================
# Experiment 3: Multi-Head Memory Subspaces (G Heads)
# =====================================================================

@functools.partial(jax.jit, static_argnums=(2, 3, 4))
def run_multihead_memory_test(keys, vals, G=1, variant="delta", eta=0.25):
    """
    Partition memory matrix into G independent heads/subspaces.
    """
    T, d_k = keys.shape
    _, d_v = vals.shape
    
    d_k_head = d_k // G
    d_v_head = d_v // G
    
    # Reshape keys and vals into G heads
    keys_h = keys.reshape(T, G, d_k_head)
    vals_h = vals.reshape(T, G, d_v_head)
    
    M_h = jnp.zeros((G, d_k_head, d_v_head), dtype=jnp.float32)
    
    def step_fn(M_t, inputs):
        k_t, v_t = inputs  # [G, d_k_head], [G, d_v_head]
        
        # Read prediction per head: v_hat[g] = M_t[g]^T k_t[g]
        v_hat = jnp.einsum('g k v, g k -> g v', M_t, k_t)
        
        target_val = (v_t - v_hat) if variant == "delta" else v_t
        k_sq = jnp.sum(k_t ** 2, axis=-1, keepdims=True) + 1e-5  # [G, 1]
        bet_eff = (eta * 0.5) / k_sq  # [G, 1]
        
        # Outer product update per head
        updates = jnp.einsum('g k, g v -> g k v', k_t, target_val)
        M_next = M_t + bet_eff[..., None] * updates
        return M_next, v_hat

    M_final, _ = jax.lax.scan(step_fn, M_h, (keys_h, vals_h))
    
    # Read predictions per head
    v_hats_h = jnp.einsum('g k v, t g k -> t g v', M_final, keys_h)
    v_hats = v_hats_h.reshape(T, d_v)
    
    cos_sim = jnp.mean(jnp.sum(v_hats * vals, axis=-1) / (jnp.linalg.norm(v_hats, axis=-1) * jnp.linalg.norm(vals, axis=-1) + 1e-5))
    return cos_sim


def run_multihead_subspaces_experiment():
    print("\n" + "=" * 80)
    print("🧪 EXPERIMENT 3: Multi-Head Memory Subspaces (G in {1, 2, 4, 8})")
    print("=" * 80)
    
    rng = jax.random.PRNGKey(42)
    d_k, d_v = 32, 32  # Fixed total state dimension
    heads = [1, 2, 4, 8]
    loads = [16, 32, 64, 128]
    
    results = []
    
    for N in loads:
        for G in heads:
            rng, k1, k2 = jax.random.split(rng, 3)
            keys = jax.random.normal(k1, (N, d_k))
            vals = jax.random.normal(k2, (N, d_v))
            keys = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + 1e-5)
            vals = vals / (jnp.linalg.norm(vals, axis=-1, keepdims=True) + 1e-5)
            
            cos_add = float(run_multihead_memory_test(keys, vals, G=G, variant="additive", eta=1.0))
            cos_delta = float(run_multihead_memory_test(keys, vals, G=G, variant="delta", eta=0.25))
            
            results.append({
                "N": N,
                "G_heads": G,
                "d_k_per_head": d_k // G,
                "cos_additive": cos_add,
                "cos_delta": cos_delta,
            })
            
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "multihead_subspaces_study.csv"), index=False)
    
    print("\n--- Multi-Head Subspace Division under Load N=64 & N=128 ---")
    print(df[df["N"].isin([64, 128])].to_string(index=False))
    return df


def main():
    start_time = time.time()
    run_nonlinear_key_experiment()
    run_dk_asymmetry_experiment()
    run_multihead_subspaces_experiment()
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"🎉 ROUND 3 DIAGNOSTICS COMPLETE (Total Time: {elapsed:.2f}s)")
    print("=" * 80)


if __name__ == "__main__":
    main()
