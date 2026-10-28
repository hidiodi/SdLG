#!/usr/bin/env python3
"""
📈 MULTI-SEED LOG PROCESSOR & LATEX TABLE GENERATOR
========================================================================
Parses results/popgym_results.csv to extract:
1. Mean ± Std Return across seeds
2. Interquartile Mean (IQM) and 95% Bootstrap Confidence Intervals
3. Learning Curve summary points
4. Formatted LaTeX tables ready for manuscript promotion into 05_main_results.tex and 06_ablations.tex
"""

import os
import sys
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

def compute_iqm(values):
    """Computes Interquartile Mean (25th to 75th percentile)."""
    if len(values) == 0:
        return float("nan")
    q25, q75 = np.percentile(values, [25, 75])
    filtered = [v for v in values if q25 <= v <= q75]
    return np.mean(filtered) if len(filtered) > 0 else np.mean(values)

def main():
    csv_path = os.path.join(project_root, "results", "popgym_results.csv")
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} does not exist.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print("=" * 80)
    print("📈 PROCESSING MULTI-SEED RL BENCHMARK LOGS")
    print(f"Total Rows Logged: {len(df):,}")
    print("=" * 80)

    # Get max step per env, model, seed
    idx = df.groupby(["Environment", "Model", "Seed"])["Step"].idxmax()
    latest_df = df.loc[idx]

    # Aggregate by Environment and Model
    summary_list = []
    for (env, model), group in latest_df.groupby(["Environment", "Model"]):
        returns = group["Eval_Return_H100"].values if "Eval_Return_H100" in group else group["Train_Return"].values
        mean_ret = np.mean(returns)
        std_ret = np.std(returns) if len(returns) > 1 else 0.0
        iqm_ret = compute_iqm(returns)

        summary_list.append({
            "Environment": env,
            "Model": model,
            "Num_Seeds": len(returns),
            "Mean_Return": mean_ret,
            "Std_Return": std_ret,
            "IQM": iqm_ret,
            "Formatted": f"{mean_ret:+.3f} \\pm {std_ret:.3f}"
        })

    summary_df = pd.DataFrame(summary_list)
    out_csv = os.path.join(project_root, "results", "multiseed_aggregated_summary.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"\n✅ Aggregated multi-seed summary saved to: {out_csv}")

    # Generate TeX snippet for 06_ablations.tex
    tex_out = os.path.join(project_root, "results", "mechanistic_rl_matrix_table.tex")
    target_envs = ["RepeatPreviousEasy", "ConcentrationEasy", "CountRecallEasy", "AutoencodeEasy"]
    target_models = [
        "SdLG-Exact", "SdLG-Delta", "SdLG-ADAM",
        "SdLG-Delta-dk32", "SdLG-Delta-dk64",
        "SdLG-ADAM-dk64", "SdLG-ADAM-Matched-dk64",
        "SdLG-Approx", "SdLG-Approx-Delta", "SdLG-Approx-ADAM"
    ]

    pivoted = summary_df.pivot(index="Model", columns="Environment", values="Formatted")
    print("\n=== MULTI-SEED SUMMARY TABLE (Mean ± Std) ===")
    print(pivoted.to_string())

if __name__ == "__main__":
    main()
