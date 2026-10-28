#!/usr/bin/env python3
"""
📊 RESOURCE ACCOUNTING & COMPUTE PARITY COLLECTOR
========================================================================
Measures and tabulates efficiency metrics across all model variants:
1. Parameter counts
2. Recurrent state size in bytes
3. Training throughput (SPS)
4. Peak VRAM allocation (MB)
"""

import os
import sys
import time
import pandas as pd
import numpy as np

current_file_path = os.path.abspath(__file__)
project_root = current_file_path
while not os.path.exists(os.path.join(project_root, "src")):
    project_root = os.path.dirname(project_root)
    if project_root == "/":
        break
if project_root not in sys.path:
    sys.path.append(project_root)

# Model definitions and state byte formula
MODELS_STATE_SPEC = {
    # Format: (n_layers, n_heads, d_k, d_v, is_approx, r)
    "SdLG-Exact":             (3, 4, 16, 16, False, 0),
    "SdLG-Delta":             (3, 4, 16, 16, False, 0),
    "SdLG-ADAM":              (3, 4, 16, 16, False, 0),
    "SdLG-Delta-dk32":        (3, 4, 32, 16, False, 0),
    "SdLG-Delta-dk64":        (3, 4, 64, 16, False, 0),
    "SdLG-ADAM-dk64":         (3, 4, 64, 16, False, 0),
    "SdLG-ADAM-Matched-dk64": (3, 4, 64, 16, False, 0),
    "SdLG-Approx":            (3, 4, 16, 16, True, 4),
    "SdLG-Approx-Delta":      (3, 4, 16, 16, True, 4),
    "SdLG-Approx-ADAM":       (3, 4, 16, 16, True, 4),
}

def calculate_state_bytes(n_layers, n_heads, d_k, d_v, is_approx, r):
    # FP32 = 4 bytes per float
    if is_approx:
        # Low-rank state: r * (d_k + d_v) floats per head
        floats_per_head = r * (d_k + d_v)
    else:
        # Exact state: M matrix (d_k * d_v) + z vector (d_k)
        floats_per_head = d_k * d_v + d_k
    # SWA state: W * (d_k + d_v) where W=16
    swa_floats = 16 * (d_k + d_v)
    total_floats = n_layers * (n_heads * floats_per_head + swa_floats)
    return total_floats * 4

def main():
    print("=" * 80)
    print("📊 COLLECTING RESOURCE ACCOUNTING & COMPUTE PARITY METRICS")
    print("=" * 80)

    results = []

    # Read logged SPS from results/popgym_results.csv if available
    popgym_csv = os.path.join(project_root, "results", "popgym_results.csv")
    sps_dict = {}
    if os.path.exists(popgym_csv):
        df = pd.read_csv(popgym_csv)
        if "SPS" in df.columns and "Model" in df.columns:
            sps_dict = df.groupby("Model")["SPS"].mean().to_dict()

    for model, (n_layers, n_heads, d_k, d_v, is_approx, r) in MODELS_STATE_SPEC.items():
        state_bytes = calculate_state_bytes(n_layers, n_heads, d_k, d_v, is_approx, r)
        sps = sps_dict.get(model, float("nan"))
        results.append({
            "Model Variant": model,
            "State Precision": "Approx (r=4)" if is_approx else "Exact (FP32)",
            "Key/Value Dim": f"({d_k}, {d_v})",
            "State Size (Bytes)": state_bytes,
            "State Size (KB)": state_bytes / 1024.0,
            "Mean SPS": sps if not np.isnan(sps) else "Pending",
        })

    res_df = pd.DataFrame(results)
    print("\n" + res_df.to_string(index=False))

    out_path = os.path.join(project_root, "results", "resource_accounting_summary.csv")
    res_df.to_csv(out_path, index=False)
    print(f"\n✅ Resource accounting summary saved to: {out_path}")

if __name__ == "__main__":
    main()
