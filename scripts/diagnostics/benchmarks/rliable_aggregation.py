import os
import sys
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import pandas as pd

MODELS = {
    "Exact-SdLG": {"color": "#2c7bb6", "csv_name": "Exact_LogBias_RoPE"},
    "Approx-SdLG": {"color": "#d7191c", "csv_name": "Approx_LogBias_RoPE"},
    "PPO-GRU": {"color": "#fdae61", "csv_name": "Baseline_GRU"},
    "PPO-GTrXL": {"color": "#abdda4", "csv_name": "Baseline_GTrXL_128"},
    "PPO-Mamba": {"color": "#9e0142", "csv_name": "Baseline_Mamba"},
    "PPO-GLA": {"color": "#8e44ad", "csv_name": "Baseline_GLA"}
}

def compute_iqm(scores):
    return scipy.stats.trim_mean(scores, proportiontocut=0.25)

def bootstrap_iqm_and_profile(model_task_scores, num_bootstraps=2000, alpha=0.05):
    thresholds = np.linspace(0.0, 1.2, 100)
    model_results = {model: {"iqms": [], "profiles": []} for model in MODELS.keys()}
        
    for _ in range(num_bootstraps):
        for model in MODELS.keys():
            sample = []
            for task_scores in model_task_scores[model]:
                if len(task_scores) > 0:
                    resampled = np.random.choice(task_scores, size=len(task_scores), replace=True)
                    sample.extend(resampled)
            
            sample = np.array(sample)
            if len(sample) > 0:
                model_results[model]["iqms"].append(compute_iqm(sample))
                model_results[model]["profiles"].append([np.mean(sample >= t) for t in thresholds])
                
    final_results = {}
    valid_models = [m for m in MODELS.keys() if len(model_results[m]["iqms"]) > 0]
    
    for model in valid_models:
        iqms = np.sort(model_results[model]["iqms"])
        profiles = np.array(model_results[model]["profiles"])
        
        low_idx = int(num_bootstraps * (alpha / 2))
        high_idx = int(num_bootstraps * (1.0 - alpha / 2))
        
        final_results[model] = {
            "iqm_mean": np.mean(iqms),
            "iqm_low": iqms[low_idx],
            "iqm_high": iqms[high_idx],
            "profile_mean": np.mean(profiles, axis=0),
            "profile_low": np.percentile(profiles, 2.5, axis=0),
            "profile_high": np.percentile(profiles, 97.5, axis=0)
        }
    return final_results, thresholds, valid_models

def main():
    print("==========================================================")
    print("      rliable Aggregator & Cross-Suite Visualizer         ")
    print("==========================================================")
    
    # Load data from the unified zero-shot CSV
    zs_path = "scripts/diagnostics/exact_sdlg/unified_tmaze_ablations_zeroshot.csv"
    if not os.path.exists(zs_path):
        print(f"Error: {zs_path} not found.")
        sys.exit(1)
        
    df = pd.read_csv(zs_path)
    
    # Filter only models trained on H=50
    df = df[df["H_train"] == 50]
    
    tasks = ["val_H100", "val_H150", "val_H200"]
    model_task_scores = {model: [] for model in MODELS.keys()}
    
    for model, config in MODELS.items():
        sub_df = df[df["model"] == config["csv_name"]]
        if sub_df.empty:
            print(f"Warning: No data for {model} (mapped to {config['csv_name']})")
            continue
            
        for task in tasks:
            if task in sub_df.columns:
                # Append raw scores directly (already normalized to [0, 1] return)
                scores = sub_df[task].values
                model_task_scores[model].append(scores)
            else:
                model_task_scores[model].append([])
                
    final_results, thresholds, valid_models = bootstrap_iqm_and_profile(model_task_scores)
    
    print("\n==========================================================")
    print("                    AGGREGATION RESULTS                   ")
    print("==========================================================")
    for model in valid_models:
        res = final_results[model]
        print(f"{model} IQM: {res['iqm_mean']:.3f} (95% CI: [{res['iqm_low']:.3f}, {res['iqm_high']:.3f}])")
    
    os.makedirs("plots", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for model in valid_models:
        res = final_results[model]
        color = MODELS[model]["color"]
        axes[0].plot(thresholds, res["profile_mean"], label=model, color=color, linewidth=2.5)
        axes[0].fill_between(thresholds, res["profile_low"], res["profile_high"], color=color, alpha=0.15)
        
    axes[0].set_xlabel("Normalized Score Threshold (τ)", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Fraction of Runs with Score ≥ τ", fontsize=11, fontweight='bold')
    axes[0].set_title("Performance Profiles (T-Maze Extrapolation)", fontsize=12, fontweight='bold')
    axes[0].set_xlim(0.0, 1.1)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend(loc="lower left", frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # Sort valid models by IQM for the bar chart
    valid_models.sort(key=lambda m: final_results[m]["iqm_mean"], reverse=True)
    
    means = [final_results[m]["iqm_mean"] for m in valid_models]
    lower = [max(0, final_results[m]["iqm_mean"] - final_results[m]["iqm_low"]) for m in valid_models]
    upper = [max(0, final_results[m]["iqm_high"] - final_results[m]["iqm_mean"]) for m in valid_models]
    colors = [MODELS[m]["color"] for m in valid_models]
    
    axes[1].bar(valid_models, means, yerr=[lower, upper], color=colors, edgecolor='black', capsize=10, width=0.5, alpha=0.8)
    axes[1].set_ylabel("Normalized IQM Score", fontsize=11, fontweight='bold')
    axes[1].set_title("Interquartile Mean (IQM) Comparison", fontsize=12, fontweight='bold')
    axes[1].set_ylim(0.0, 1.2)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("plots/rliable_cross_suite_aggregation.png", dpi=300)
    plt.close()
    
    print("\n✅ Dual-panel publication plot saved successfully to: plots/rliable_cross_suite_aggregation.png")

if __name__ == "__main__":
    main()
