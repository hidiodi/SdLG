#!/usr/bin/env python3
"""
Phase 0: Synthetic Mechanistic Diagnostic Suite for Associative Memory
========================================================================
Benchmarks Additive GLA, Delta GLA, Gated GLA, and Gated Delta GLA (ADAM)
on controlled synthetic memory tasks:
  1. Synthetic Capacity Ladder (Retrieval Accuracy vs N/d_k)
  2. Memory Overwrite & Collateral Damage Metric (C_t)
  3. Effective Rank Telemetry (r_eff)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Any

# Ensure output directory exists
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../../../results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================================
# Memory Cell Implementations (Numpy vectorized for fast diagnostic)
# =====================================================================

class MemoryCellBase:
    def __init__(self, d_k: int, d_v: int, eps: float = 1e-6):
        self.d_k = d_k
        self.d_v = d_v
        self.eps = eps
        self.reset()

    def reset(self):
        self.M = np.zeros((self.d_k, self.d_v), dtype=np.float64)

    def read(self, q: np.ndarray) -> np.ndarray:
        # Standard linear attention / DeltaNet matrix read
        return np.dot(self.M.T, q)

    def write(self, k: np.ndarray, v: np.ndarray, gamma: float = 1.0, beta: float = 1.0):
        raise NotImplementedError


class AdditiveGLACell(MemoryCellBase):
    """Standard SdLG / GLA Additive Outer Product Update"""
    def write(self, k: np.ndarray, v: np.ndarray, gamma: float = 1.0, beta: float = 1.0):
        self.M = gamma * self.M + beta * np.outer(k, v)


class DeltaGLACell(MemoryCellBase):
    """Official DeltaNet Error Correction Update (Yang et al., 2024)"""
    def write(self, k: np.ndarray, v: np.ndarray, gamma: float = 1.0, beta: float = 1.0):
        v_hat = self.read(k)
        error = v - v_hat
        self.M = gamma * self.M + beta * np.outer(k, error)


class GatedGLACell(MemoryCellBase):
    """Decoupled Channel-Wise Erase Gating (Additive)"""
    def write(self, k: np.ndarray, v: np.ndarray, gamma: float = 1.0, beta: float = 1.0, 
              erase_gate: np.ndarray = None, write_gate: np.ndarray = None):
        if erase_gate is None:
            erase_gate = np.ones(self.d_k) * gamma
        if write_gate is None:
            write_gate = np.ones(self.d_k) * beta
        
        # Factorized Erase Gate: diag(erase) @ M
        self.M = np.diag(erase_gate) @ self.M + np.outer(write_gate * k, v)


class GatedDeltaGLACell(MemoryCellBase):
    """Full ADAM: Decoupled Erase + Delta-Rule Correction"""
    def write(self, k: np.ndarray, v: np.ndarray, gamma: float = 1.0, beta: float = 1.0,
              erase_gate: np.ndarray = None, write_gate: np.ndarray = None):
        if erase_gate is None:
            erase_gate = np.ones(self.d_k) * gamma
        if write_gate is None:
            write_gate = np.ones(self.d_k) * beta

        v_hat = self.read(k)
        error = v - v_hat

        # Factorized Erase & Delta Write
        self.M = np.diag(erase_gate) @ self.M + np.outer(write_gate * k, error)


# =====================================================================
# Benchmark 1: Synthetic Capacity Ladder (Retrieval Accuracy vs N/d_k)
# =====================================================================

def run_recall_test(d_k: int = 64, d_v: int = 64, N: int = 16) -> pd.DataFrame:
    """Phase 0.1a: Pure recall --- store (k_i, v_i), query k_i immediately."""
    models = {
        "Additive GLA": AdditiveGLACell(d_k, d_v),
        "Delta GLA": DeltaGLACell(d_k, d_v),
        "Gated GLA": GatedGLACell(d_k, d_v),
        "Gated Delta (ADAM)": GatedDeltaGLACell(d_k, d_v)
    }
    np.random.seed(7)
    keys = np.random.randn(N, d_k)
    keys = keys / np.linalg.norm(keys, axis=1, keepdims=True)
    values = np.random.randn(N, d_v)
    values = values / np.linalg.norm(values, axis=1, keepdims=True)

    results = []
    for model_name, cell in models.items():
        cell.reset()
        cos_sims = []
        for i in range(N):
            cell.write(keys[i], values[i], gamma=0.99, beta=1.0)
            v_hat = cell.read(keys[i])
            cs = np.dot(values[i], v_hat) / (np.linalg.norm(values[i]) * np.linalg.norm(v_hat) + 1e-8)
            cos_sims.append(cs)
        results.append({"model": model_name, "recall_cos_sim": np.mean(cos_sims), "N": N})
    return pd.DataFrame(results)


def run_capacity_ladder(d_k: int = 64, d_v: int = 64,
                        N_values: List[int] = None) -> pd.DataFrame:
    """Phase 0.2: Capacity saturation sweep across explicit N values."""
    if N_values is None:
        N_values = [4, 8, 16, 32, 64, 128, 256]

    models = {
        "Additive GLA": AdditiveGLACell(d_k, d_v),
        "Delta GLA": DeltaGLACell(d_k, d_v),
        "Gated GLA": GatedGLACell(d_k, d_v),
        "Gated Delta (ADAM)": GatedDeltaGLACell(d_k, d_v)
    }

    results = []
    for N in N_values:
        np.random.seed(42 + N)
        keys = np.random.randn(N, d_k)
        keys = keys / np.linalg.norm(keys, axis=1, keepdims=True)
        values = np.random.randn(N, d_v)
        values = values / np.linalg.norm(values, axis=1, keepdims=True)

        for model_name, cell in models.items():
            cell.reset()
            for i in range(N):
                cell.write(keys[i], values[i], gamma=0.99, beta=1.0)

            cos_sim_list = []
            for i in range(N):
                v_hat = cell.read(keys[i])
                cs = np.dot(values[i], v_hat) / (np.linalg.norm(values[i]) * np.linalg.norm(v_hat) + 1e-8)
                cos_sim_list.append(cs)

            accuracy = np.mean(np.array(cos_sim_list) > 0.8)
            results.append({
                "N": N,
                "load_ratio": N / d_k,
                "model": model_name,
                "cos_sim": np.mean(cos_sim_list),
                "retrieval_accuracy": accuracy
            })

    return pd.DataFrame(results)


# =====================================================================
# Benchmark 2: Memory Overwrite & Collateral Damage Metric (C_t)
# =====================================================================

def run_overwrite_collateral_benchmark(d_k: int = 64, d_v: int = 64, N: int = 32) -> pd.DataFrame:
    models = {
        "Additive GLA": AdditiveGLACell(d_k, d_v),
        "Delta GLA": DeltaGLACell(d_k, d_v),
        "Gated GLA": GatedGLACell(d_k, d_v),
        "Gated Delta (ADAM)": GatedDeltaGLACell(d_k, d_v)
    }

    np.random.seed(99)
    keys = np.random.randn(N, d_k)
    keys = keys / np.linalg.norm(keys, axis=1, keepdims=True)

    values = np.random.randn(N, d_v)
    values = values / np.linalg.norm(values, axis=1, keepdims=True)

    # Overwrite value for key 0
    new_v0 = np.random.randn(d_v)
    new_v0 = new_v0 / np.linalg.norm(new_v0)

    results = []

    for model_name, cell in models.items():
        cell.reset()

        # Step 1: Initial population
        for i in range(N):
            cell.write(keys[i], values[i], gamma=0.99, beta=1.0)

        # Baseline reads before overwrite
        non_target_before = [cell.read(keys[i]) for i in range(1, N)]

        # Step 2: Overwrite key 0 with new_v0
        M_before = cell.M.copy()
        cell.write(keys[0], new_v0, gamma=0.99, beta=1.0)
        M_after = cell.M.copy()

        # Target retrieval quality
        target_v0_hat = cell.read(keys[0])
        target_cos_sim = np.dot(new_v0, target_v0_hat) / (np.linalg.norm(new_v0) * np.linalg.norm(target_v0_hat) + 1e-8)

        # C_t^matrix: matrix-level collateral damage
        non_target_after = [cell.read(keys[i]) for i in range(1, N)]
        non_target_drifts = [np.linalg.norm(non_target_after[i] - non_target_before[i]) for i in range(N-1)]
        ct_matrix = np.mean(non_target_drifts) / (np.linalg.norm(M_after - M_before) + 1e-8)

        # C_t^behavior: fraction of non-target associations still correctly recalled (cosine sim > 0.8)
        non_target_cos_sims = [
            np.dot(values[i], non_target_after[i-1]) /
            (np.linalg.norm(values[i]) * np.linalg.norm(non_target_after[i-1]) + 1e-8)
            for i in range(1, N)
        ]
        ct_behavior = 1.0 - np.mean(np.array(non_target_cos_sims) > 0.8)  # fraction corrupted

        # Effective Rank (r_eff at tau=0.95)
        U, s, Vt = np.linalg.svd(cell.M)
        cumulative_energy = np.cumsum(s) / np.sum(s)
        r_eff = np.searchsorted(cumulative_energy, 0.95) + 1

        results.append({
            "model": model_name,
            "target_overwrite_cos_sim": target_cos_sim,
            "ct_matrix": ct_matrix,
            "ct_behavior": ct_behavior,
            "effective_rank_95": r_eff
        })

    return pd.DataFrame(results)


# =====================================================================
# Main Execution & Plot Generation
# =====================================================================

def main():
    print("=" * 70)
    print("Phase 0: Synthetic Mechanistic Diagnostic Suite")
    print("=" * 70)

    # Phase 0.1a: Pure Recall
    print("\n[1/3] Phase 0.1a: Pure Recall Test...")
    df_recall = run_recall_test(d_k=64, d_v=64, N=16)
    print(df_recall.to_string(index=False))
    df_recall.to_csv(os.path.join(OUTPUT_DIR, "phase0_recall.csv"), index=False)

    # Phase 0.1b: Overwrite + C_t^matrix + C_t^behavior
    print("\n[2/3] Phase 0.1b+c: Overwrite Fidelity & Collateral Damage...")
    df_overwrite = run_overwrite_collateral_benchmark(d_k=64, d_v=64, N=32)
    print(df_overwrite.to_string(index=False))
    df_overwrite.to_csv(os.path.join(OUTPUT_DIR, "phase0_overwrite_collateral.csv"), index=False)

    # Phase 0.2: Capacity Saturation Sweep (N ∈ {4,8,16,32,64,128,256})
    print("\n[3/3] Phase 0.2: Capacity Saturation Sweep (N vs d_k=64)...")
    df_capacity = run_capacity_ladder(d_k=64, d_v=64)
    print(df_capacity.pivot(index="N", columns="model", values="cos_sim").round(4))
    df_capacity.to_csv(os.path.join(OUTPUT_DIR, "phase0_capacity_sweep.csv"), index=False)

    # Plot capacity saturation curve
    plt.figure(figsize=(9, 5))
    for model in df_capacity["model"].unique():
        sub = df_capacity[df_capacity["model"] == model]
        plt.plot(sub["N"], sub["cos_sim"], marker='o', label=model, linewidth=2)
    plt.axvline(x=64, color='gray', linestyle='--', label='Saturation (N = d_k = 64)')
    plt.xscale("log", base=2)
    plt.title("Phase 0.2: Capacity Saturation Sweep\nRetrieval Cosine Similarity vs N active keys (d_k=64)")
    plt.xlabel("N (number of active key-value associations)")
    plt.ylabel("Mean Retrieval Cosine Similarity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "phase0_capacity_sweep.png"), dpi=300)
    print(f"--> Plots and CSVs saved to {OUTPUT_DIR}")

    print("\n" + "=" * 70)
    print("Phase 0 Complete.")
    print("=" * 70)

if __name__ == "__main__":
    main()
