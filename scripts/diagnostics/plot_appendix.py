import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication-quality aesthetics
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

current_file_path = os.path.abspath(__file__)
project_root = current_file_path
while not os.path.exists(os.path.join(project_root, "src")):
    project_root = os.path.dirname(project_root)
    if project_root == "/":
        break

PLOT_DIR = os.path.join(project_root, "papers/latex/thesis_style/plots/appendix")
os.makedirs(PLOT_DIR, exist_ok=True)

# Define generic color scheme
PALETTE = sns.color_palette("tab10")
COLOR_MAP = {
    # Canonical architectures
    "GTrXL": "#000000",
    "SdLG-Exact": PALETTE[0],
    "SdLG-Approx": PALETTE[2],
    "SdLG-Approx (v6)": PALETTE[3],
    "SdLG-Approx (v7)": PALETTE[4],
    # Ablation arms
    "SdLG-Exact (No-SWA)": "#d7191c",
    r"SdLG-Exact ($\Gamma=0$)": PALETTE[1],
    r"SdLG-Exact ($b_{gru}=0$)": PALETTE[5],
    # Capacity scaling
    "SdLG-Exact (Control)": PALETTE[0],
    "SdLG-Exact (FFN=2048)": "#1a9641",
    "SdLG-Exact (FFN=512)": "#a6d96a",
    "SdLG-Exact (Heads=16)": "#7b3294",
    "SdLG-Exact (Heads=2)": "#c2a5cf",
    # BPTT
    "T=32": PALETTE[0],
    "T=64": PALETTE[1],
    "T=128": "#000000",
    # T-Maze variations
    "SdLG-Exact (Dense)": "#000000",
    "SdLG-Approx (MoE)": PALETTE[8],
    # Craftax-Classic 50M Benchmark
    "SdLG-Approx (T=16)": "#1f77b4",
    "PPO-GRU (T=128)": "#d62728",
    "PPO-GRU (T=64)": "#ff7f0e",
    "PPO-GTrXL (T=64)": "#2ca02c",
    "AGaLiTe (T=64)": "#8c564b",
    "AGaLiTe (T=32)": "#e377c2",
    "PPO-Dense (T=16)": "#7f7f7f",
    "PPO-GTrXL (T=128)": "#9467bd",
    "SdLG-Exact (T=64)": "#bcbd22",
    "PPO-Mamba": "#17becf",
}

def clean_and_smooth(series, window=5):
    """Apply moving average smoothing to curves to reduce high-frequency noise."""
    if len(series) < window:
        return series
    return series.rolling(window=window, min_periods=1).mean()


# ==============================================================================
# 1. CRAFTAX-1B BENCHMARK PLOTS
# ==============================================================================
def plot_craftax_1b():
    print("Generating Craftax-1B benchmark plots...")
    dir_path = "wandb_exports/Craftax_1B_300M"
    if not os.path.exists(dir_path):
        print(f"Skipping Craftax-1B: {dir_path} not found.")
        return

    # Map file names to human-readable condition labels
    model_files = {
        "GTrXL": "PPO-GTrXL-42.csv",
        "SdLG-Exact": "ExactSdLG-42.csv",
        "SdLG-Approx": "SdLGHybrid-42.csv",
        "SdLG-Approx (v6)": "SdLGHybridv6-42-T128.csv",
        "SdLG-Approx (v7)": "SdLGHybridv7-42-T128.csv"
    }

    data = {}
    for name, fname in model_files.items():
        path = os.path.join(dir_path, fname)
        if os.path.exists(path):
            data[name] = pd.read_csv(path)

    if not data:
        print("No Craftax-1B data loaded.")
        return

    # 1.1 Episodic Return
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in ["GTrXL", "SdLG-Exact", "SdLG-Approx"]:
        if name in data:
            df = data[name]
            if "Train/Mean_Return" in df.columns:
                steps = df["Total_Steps"] / 1e6
                smoothed = clean_and_smooth(df["Train/Mean_Return"])
                ax.plot(steps, smoothed, label=name, color=COLOR_MAP.get(name, "#333333"), linewidth=2)
    ax.set_xlabel("Environment Steps (Millions)")
    ax.set_ylabel("Mean Episode Return")
    ax.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "craftax_1b_return.png"), dpi=300)
    plt.close()

    # 1.2 Episode Length
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in ["SdLG-Approx (v6)", "SdLG-Approx (v7)"]:
        if name in data:
            df = data[name]
            col = "Train/Mean_Episode_Length" if "Train/Mean_Episode_Length" in df.columns else "Train/Mean_Length"
            if col in df.columns:
                steps = df["Total_Steps"] / 1e6
                smoothed = clean_and_smooth(df[col])
                ax.plot(steps, smoothed, label=name, color=COLOR_MAP.get(name, "#333333"), linewidth=2)
    ax.set_xlabel("Environment Steps (Millions)")
    ax.set_ylabel("Mean Episode Length")
    ax.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "craftax_1b_length.png"), dpi=300)
    plt.close()

    # 1.3 Policy Entropy
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in ["SdLG-Approx (v6)", "SdLG-Approx (v7)"]:
        if name in data:
            df = data[name]
            if "Train/Policy_Entropy" in df.columns:
                steps = df["Total_Steps"] / 1e6
                smoothed = clean_and_smooth(df["Train/Policy_Entropy"])
                ax.plot(steps, smoothed, label=name, color=COLOR_MAP.get(name, "#333333"), linewidth=2)
    ax.set_xlabel("Environment Steps (Millions)")
    ax.set_ylabel("Policy Entropy")
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "craftax_1b_entropy.png"), dpi=300)
    plt.close()

    # 1.4 Achievements
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in ["GTrXL", "SdLG-Exact", "SdLG-Approx"]:
        if name in data:
            df = data[name]
            if "Train/Total_Achievements" in df.columns:
                steps = df["Total_Steps"] / 1e6
                smoothed = clean_and_smooth(df["Train/Total_Achievements"])
                ax.plot(steps, smoothed, label=name, color=COLOR_MAP.get(name, "#333333"), linewidth=2)
    ax.set_xlabel("Environment Steps (Millions)")
    ax.set_ylabel("Unique Achievements Unlocked")
    ax.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "craftax_1b_achievements.png"), dpi=300)
    plt.close()
    print("Craftax-1B plots generated successfully.")


# ==============================================================================
# 2. CRAFTAX-CLASSIC ABLATIONS: EXACT VS APPROX
# ==============================================================================
def plot_exact_vs_approx():
    print("Generating Exact vs Approx ablation plots...")
    dir_path = "wandb_exports/Craftax_Classic_Exact_vs_Approx"
    if not os.path.exists(dir_path):
        print(f"Skipping Exact vs Approx: {dir_path} not found.")
        return

    # Conditions list
    conditions = ["Control_Exact_Dense", "Ablation_Approx_Dense", "Ablation_GammaZero", "Ablation_GRUBias_0"]
    labels = {
        "Control_Exact_Dense": "SdLG-Exact",
        "Ablation_Approx_Dense": "SdLG-Approx",
        "Ablation_GammaZero": r"SdLG-Exact ($\Gamma=0$)",
        "Ablation_GRUBias_0": r"SdLG-Exact ($b_{gru}=0$)"
    }

    # Load and group by Condition
    all_files = os.listdir(dir_path)
    data_list = []
    for f in all_files:
        if not f.endswith(".csv"): continue
        path = os.path.join(dir_path, f)
        cond = None
        for c in conditions:
            if f.startswith(c):
                cond = c
                break
        if cond is None: continue
        
        df = pd.read_csv(path)
        df["Condition"] = cond
        data_list.append(df)

    if not data_list:
        print("No Exact vs Approx data found.")
        return

    full_df = pd.concat(data_list, ignore_index=True)

    metrics = {
        "Train/Mean_Return": ("craftax_classic_exact_vs_approx_return.png", "Mean Episode Return"),
        "Train/Mean_Length": ("craftax_classic_exact_vs_approx_length.png", "Mean Episode Length"),
        "Train/Policy_Entropy": ("craftax_classic_exact_vs_approx_entropy.png", "Policy Entropy")
    }

    for metric, (fname, ylabel) in metrics.items():
        if metric not in full_df.columns:
            print(f"Warning: metric {metric} missing in Exact vs Approx data.")
            continue

        fig, ax = plt.subplots(figsize=(8, 4))
        for cond in conditions:
            cond_df = full_df[full_df["Condition"] == cond]
            if cond_df.empty: continue

            # Group by step to compute mean and std
            grp = cond_df.groupby("Total_Steps")
            mean_df = grp.mean(numeric_only=True)
            std_df = grp.std(numeric_only=True).fillna(0)

            steps = mean_df.index / 1e6
            mean_vals = clean_and_smooth(mean_df[metric])
            std_vals = clean_and_smooth(std_df[metric])

            lbl = labels.get(cond, cond)
            c = COLOR_MAP.get(lbl, "#333333")
            lw = 2.5 if cond == "Control_Exact_Dense" else 2.0
            ls = "-" if cond == "Control_Exact_Dense" else "--"

            ax.plot(steps, mean_vals, label=lbl, color=c, linewidth=lw, linestyle=ls)
            ax.fill_between(steps, mean_vals - std_vals, mean_vals + std_vals, color=c, alpha=0.15)

        ax.set_xlabel("Environment Steps (Millions)")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=True, loc="lower right" if "Entropy" not in ylabel else "upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, fname), dpi=300)
        plt.close()
    print("Exact vs Approx plots generated successfully.")


# ==============================================================================
# 3. CRAFTAX-CLASSIC ABLATIONS: CAPACITY SCALING
# ==============================================================================
def plot_capacity_scaling():
    print("Generating Capacity Scaling ablation plots...")
    dir_path = "wandb_exports/Craftax_Classic_Capacity_Scaling"
    if not os.path.exists(dir_path):
        print(f"Skipping Capacity Scaling: {dir_path} not found.")
        return

    conditions = [
        "Control_Exact_Dense",
        "Capacity_WideFFN_2048",
        "Capacity_NarrowFFN_512",
        "Capacity_16Heads",
        "Capacity_2Heads"
    ]
    labels = {
        "Control_Exact_Dense": "SdLG-Exact (Control)",
        "Capacity_WideFFN_2048": "SdLG-Exact (FFN=2048)",
        "Capacity_NarrowFFN_512": "SdLG-Exact (FFN=512)",
        "Capacity_16Heads": "SdLG-Exact (Heads=16)",
        "Capacity_2Heads": "SdLG-Exact (Heads=2)"
    }

    all_files = os.listdir(dir_path)
    data_list = []
    for f in all_files:
        if not f.endswith(".csv"): continue
        path = os.path.join(dir_path, f)
        cond = None
        for c in conditions:
            if f.startswith(c):
                cond = c
                break
        if cond is None: continue

        df = pd.read_csv(path)
        df["Condition"] = cond
        data_list.append(df)

    if not data_list:
        print("No Capacity Scaling data found.")
        return

    full_df = pd.concat(data_list, ignore_index=True)

    metrics = {
        "Train/Mean_Return": ("craftax_classic_capacity_scaling_return.png", "Mean Episode Return"),
        "Train/Mean_Length": ("craftax_classic_capacity_scaling_length.png", "Mean Episode Length"),
        "Train/Policy_Entropy": ("craftax_classic_capacity_scaling_entropy.png", "Policy Entropy")
    }

    for metric, (fname, ylabel) in metrics.items():
        if metric not in full_df.columns:
            continue

        fig, ax = plt.subplots(figsize=(8, 4))
        for cond in conditions:
            cond_df = full_df[full_df["Condition"] == cond]
            if cond_df.empty: continue

            grp = cond_df.groupby("Total_Steps")
            mean_df = grp.mean(numeric_only=True)
            std_df = grp.std(numeric_only=True).fillna(0)

            steps = mean_df.index / 1e6
            mean_vals = clean_and_smooth(mean_df[metric])
            std_vals = clean_and_smooth(std_df[metric])

            lbl = labels.get(cond, cond)
            c = COLOR_MAP.get(lbl, "#333333")
            lw = 2.5 if cond == "Control_Exact_Dense" else 2.0
            ls = "-" if cond == "Control_Exact_Dense" else "--"

            ax.plot(steps, mean_vals, label=lbl, color=c, linewidth=lw, linestyle=ls)
            ax.fill_between(steps, mean_vals - std_vals, mean_vals + std_vals, color=c, alpha=0.15)

        ax.set_xlabel("Environment Steps (Millions)")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=True, loc="lower right" if "Entropy" not in ylabel else "upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, fname), dpi=300)
        plt.close()
    print("Capacity Scaling plots generated successfully.")


# ==============================================================================
# 4. CRAFTAX-CLASSIC ABLATIONS: BPTT HORIZON SCALING
# ==============================================================================
def plot_bptt_horizon():
    print("Generating BPTT Horizon ablation plots...")
    dir_path = "wandb_exports/Craftax_1B_300M"
    if not os.path.exists(dir_path):
        print(f"Skipping BPTT Horizon: {dir_path} not found.")
        return

    model_files = {
        "T=32": "SdLGHybridv7Classic-42-T32.csv",
        "T=64": "SdLGHybridv7Classic-42-T64.csv",
        "T=128": "SdLGHybridv7Classic-42-T128.csv"
    }

    data = {}
    for name, fname in model_files.items():
        path = os.path.join(dir_path, fname)
        if os.path.exists(path):
            data[name] = pd.read_csv(path)

    if not data:
        print("No BPTT Horizon scaling data loaded.")
        return

    # 4.1 Episodic Return
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, df in data.items():
        if "Train/Mean_Return" in df.columns:
            steps = df["Total_Steps"] / 1e6
            smoothed = clean_and_smooth(df["Train/Mean_Return"])
            ax.plot(steps, smoothed, label=name, color=COLOR_MAP.get(name, "#333333"), linewidth=2)
    ax.set_xlabel("Environment Steps (Millions)")
    ax.set_ylabel("Mean Episode Return")
    ax.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "craftax_classic_bptt_return.png"), dpi=300)
    plt.close()

    # 4.2 Episode Length
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, df in data.items():
        col = "Train/Mean_Episode_Length" if "Train/Mean_Episode_Length" in df.columns else "Train/Mean_Length"
        if col in df.columns:
            steps = df["Total_Steps"] / 1e6
            smoothed = clean_and_smooth(df[col])
            ax.plot(steps, smoothed, label=name, color=COLOR_MAP.get(name, "#333333"), linewidth=2)
    ax.set_xlabel("Environment Steps (Millions)")
    ax.set_ylabel("Mean Episode Length")
    ax.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "craftax_classic_bptt_length.png"), dpi=300)
    plt.close()

    # 4.3 Policy Entropy
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, df in data.items():
        if "Train/Policy_Entropy" in df.columns:
            steps = df["Total_Steps"] / 1e6
            smoothed = clean_and_smooth(df["Train/Policy_Entropy"])
            ax.plot(steps, smoothed, label=name, color=COLOR_MAP.get(name, "#333333"), linewidth=2)
    ax.set_xlabel("Environment Steps (Millions)")
    ax.set_ylabel("Policy Entropy")
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "craftax_classic_bptt_entropy.png"), dpi=300)
    plt.close()
    print("BPTT Horizon plots generated successfully.")


# ==============================================================================
# 5. MINIGRID-MEMORYS16 SWA ABLATION
# ==============================================================================
def plot_minigrid_swa():
    print("Generating MiniGrid SWA ablation plots...")
    dir_path = "wandb_exports/MiniGrid_SWA_Ablation"
    if not os.path.exists(dir_path):
        print(f"Skipping MiniGrid SWA: {dir_path} not found.")
        return

    # SWA vs None in MemoryS16 env
    swa_files = [f for f in os.listdir(dir_path) if "MemoryS16" in f]
    
    data_list = []
    for f in swa_files:
        path = os.path.join(dir_path, f)
        df = pd.read_csv(path)
        if "SdLG-SWA" in f:
            df["Condition"] = "SdLG-Exact"
        elif "SdLG-None" in f:
            df["Condition"] = "SdLG-Exact (No-SWA)"
        else:
            continue
        data_list.append(df)

    if not data_list:
        print("No MiniGrid SWA MemoryS16 data found.")
        return

    full_df = pd.concat(data_list, ignore_index=True)

    # 5.1 Training Return
    fig, ax = plt.subplots(figsize=(8, 4))
    for cond in ["SdLG-Exact", "SdLG-Exact (No-SWA)"]:
        cond_df = full_df[full_df["Condition"] == cond]
        if cond_df.empty: continue
        grp = cond_df.groupby("Total_Steps")
        mean_df = grp.mean(numeric_only=True)
        std_df = grp.std(numeric_only=True).fillna(0)
        steps = mean_df.index / 1e6
        mean_vals = clean_and_smooth(mean_df["Train/Mean_Return"])
        std_vals = clean_and_smooth(std_df["Train/Mean_Return"])
        c = COLOR_MAP.get(cond, "#333333")
        ax.plot(steps, mean_vals, label=cond, color=c, linewidth=2)
        ax.fill_between(steps, mean_vals - std_vals, mean_vals + std_vals, color=c, alpha=0.15)
    ax.set_xlabel("Environment Steps (Millions)")
    ax.set_ylabel("Mean Episode Return")
    ax.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "minigrid_swa_return.png"), dpi=300)
    plt.close()

    # 5.2 Evaluation Return (Deterministic)
    fig, ax = plt.subplots(figsize=(8, 4))
    for cond in ["SdLG-Exact", "SdLG-Exact (No-SWA)"]:
        cond_df = full_df[full_df["Condition"] == cond]
        if cond_df.empty: continue
        grp = cond_df.groupby("Total_Steps")
        mean_df = grp.mean(numeric_only=True)
        std_df = grp.std(numeric_only=True).fillna(0)
        if "Eval/Mean_Return" in mean_df.columns:
            steps = mean_df.index / 1e6
            mean_vals = clean_and_smooth(mean_df["Eval/Mean_Return"])
            std_vals = clean_and_smooth(std_df["Eval/Mean_Return"])
            c = COLOR_MAP.get(cond, "#333333")
            ax.plot(steps, mean_vals, label=cond, color=c, linewidth=2)
            ax.fill_between(steps, mean_vals - std_vals, mean_vals + std_vals, color=c, alpha=0.15)
    ax.set_xlabel("Environment Steps (Millions)")
    ax.set_ylabel("Deterministic Evaluation Return")
    ax.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "minigrid_swa_length.png"), dpi=300)
    plt.close()

    # 5.3 Policy Entropy
    fig, ax = plt.subplots(figsize=(8, 4))
    for cond in ["SdLG-Exact", "SdLG-Exact (No-SWA)"]:
        cond_df = full_df[full_df["Condition"] == cond]
        if cond_df.empty: continue
        grp = cond_df.groupby("Total_Steps")
        mean_df = grp.mean(numeric_only=True)
        std_df = grp.std(numeric_only=True).fillna(0)
        if "Train/Policy_Entropy" in mean_df.columns:
            steps = mean_df.index / 1e6
            mean_vals = clean_and_smooth(mean_df["Train/Policy_Entropy"])
            std_vals = clean_and_smooth(std_df["Train/Policy_Entropy"])
            c = COLOR_MAP.get(cond, "#333333")
            ax.plot(steps, mean_vals, label=cond, color=c, linewidth=2)
            ax.fill_between(steps, mean_vals - std_vals, mean_vals + std_vals, color=c, alpha=0.15)
    ax.set_xlabel("Environment Steps (Millions)")
    ax.set_ylabel("Policy Entropy")
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "minigrid_swa_entropy.png"), dpi=300)
    plt.close()
    print("MiniGrid SWA plots generated successfully.")


# ==============================================================================
# 6. T-MAZE EXTRAPOLATION
# ==============================================================================
COLOR_MAP = {
    # Canonical architectures & T-Maze 10-seed benchmarks
    "SdLG-Exact (Ours)": "#1f77b4",
    "SdLG-Approx": "#ff7f0e",
    "Mamba (SSM)": "#2ca02c",
    "LTI-Linear": "#d62728",
    "GRU": "#9467bd",
    "LSTM": "#8c564b",
    # Legacy & Ablation mappings
    "GTrXL": "#000000",
    "GTrXL (64)": "#e377c2",
    "GTrXL (128)": "#7f7f7f",
    "SdLG-Exact": "#1f77b4",
    "SdLG-Exact (No-SWA)": "#d7191c",
    r"SdLG-Exact ($\Gamma=0$)": PALETTE[1],
    r"SdLG-Exact ($b_{gru}=0$)": PALETTE[5],
    "SdLG-Exact (Dense)": "#000000",
    "SdLG-Approx (MoE)": PALETTE[8],
}

def save_figure_to_all(fig, filename):
    dirs = [
        os.path.join(project_root, "plots/appendix"),
        os.path.join(project_root, "papers/latex/icml_style/plots/appendix"),
        os.path.join(project_root, "papers/latex/thesis_style/plots/appendix"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, filename), dpi=300, bbox_inches='tight')

def plot_tmaze_ablations():
    print("Generating T-Maze benchmark plots with 10 seeds and Mamba baseline...")
    tr_path = os.path.join(project_root, "scripts/diagnostics/exact_sdlg/unified_tmaze_ablations_training.csv")
    zs_path = os.path.join(project_root, "scripts/diagnostics/exact_sdlg/unified_tmaze_ablations_zeroshot.csv")

    if not os.path.exists(tr_path) or not os.path.exists(zs_path):
        print(f"Skipping T-Maze plots: Unified CSV files not found.")
        return

    tr_df = pd.read_csv(tr_path)
    zs_df = pd.read_csv(zs_path)

    # Filter H_train = 50 runs (10 seeds)
    tr_50 = tr_df[tr_df["H_train"] == 50].copy()
    zs_50 = zs_df[zs_df["H_train"] == 50].copy()

    model_map = {
        "Exact_LogBias_RoPE": "SdLG-Exact (Ours)",
        "Approx_LogBias_RoPE": "SdLG-Approx",
        "Baseline_Mamba": "Mamba (SSM)",
        "Baseline_S5": "LTI-Linear",
        "Baseline_GRU": "GRU",
        "Baseline_LSTM": "LSTM",
        "Baseline_GTrXL_64": "GTrXL (64)",
        "Baseline_GTrXL_128": "GTrXL (128)",
    }

    display_order = ["SdLG-Exact (Ours)", "SdLG-Approx", "Mamba (SSM)", "LTI-Linear", "GRU", "LSTM", "GTrXL (64)", "GTrXL (128)"]

    tr_50["Condition"] = tr_50["model"].map(model_map)
    zs_50["Condition"] = zs_50["model"].map(model_map)

    tr_50 = tr_50.dropna(subset=["Condition"])
    zs_50 = zs_50.dropna(subset=["Condition"])

    # 1. Training Return Learning Curves (Mean + 95% CI)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cond_name in display_order:
        sub = tr_50[tr_50["Condition"] == cond_name]
        if sub.empty: continue
        
        grp = sub.groupby("step")["mean_return"]
        steps = grp.mean().index / 1e6
        mean_vals = clean_and_smooth(grp.mean())
        std_vals = grp.std().fillna(0)
        n_seeds = grp.count()
        sem_vals = std_vals / np.sqrt(np.maximum(1, n_seeds))
        
        c = COLOR_MAP.get(cond_name, "#333333")
        ax.plot(steps, mean_vals, label=cond_name, color=c, linewidth=2.2)
        ax.fill_between(steps, mean_vals - 1.96 * sem_vals, mean_vals + 1.96 * sem_vals, color=c, alpha=0.15)

    ax.set_xlabel("Environment Steps (Millions)")
    ax.set_ylabel("Mean Episode Return")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    save_figure_to_all(fig, "tmaze_return.png")
    plt.close()

    # 2. Episode Length
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cond_name in display_order:
        sub = tr_50[tr_50["Condition"] == cond_name]
        if sub.empty or "train_len" not in sub.columns: continue
        grp = sub.groupby("step")["train_len"]
        steps = grp.mean().index / 1e6
        mean_vals = clean_and_smooth(grp.mean())
        c = COLOR_MAP.get(cond_name, "#333333")
        ax.plot(steps, mean_vals, label=cond_name, color=c, linewidth=2.0)
    ax.set_xlabel("Environment Steps (Millions)")
    ax.set_ylabel("Mean Episode Length")
    ax.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    save_figure_to_all(fig, "tmaze_length.png")
    plt.close()

    # 3. Policy Entropy
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cond_name in display_order:
        sub = tr_50[tr_50["Condition"] == cond_name]
        if sub.empty or "train_ent" not in sub.columns: continue
        grp = sub.groupby("step")["train_ent"]
        steps = grp.mean().index / 1e6
        mean_vals = clean_and_smooth(grp.mean())
        c = COLOR_MAP.get(cond_name, "#333333")
        ax.plot(steps, mean_vals, label=cond_name, color=c, linewidth=2.0)
    ax.set_xlabel("Environment Steps (Millions)")
    ax.set_ylabel("Policy Entropy")
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    save_figure_to_all(fig, "tmaze_entropy.png")
    plt.close()

    # 4. Zero-Shot OOD Length Extrapolation Success Rates (with 95% Bootstrap CIs)
    import scipy.stats
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    eval_cols = ["val_H100", "val_H150", "val_H200"]
    x_labels = ["H=100", "H=150", "H=200"]
    x_positions = np.arange(len(eval_cols))
    
    n_models = len(display_order)
    width = 0.10
    
    for i, cond_name in enumerate(display_order):
        sub = zs_50[zs_50["Condition"] == cond_name]
        if sub.empty: continue
        
        means = []
        err_low = []
        err_high = []
        
        for col in eval_cols:
            vals = sub[col].values
            m_val = np.mean(vals)
            means.append(m_val)
            if len(vals) > 1 and np.std(vals) > 1e-6:
                res = scipy.stats.bootstrap((vals,), np.mean, confidence_level=0.95, n_resamples=1000, method='percentile')
                ci_l, ci_h = res.confidence_interval.low, res.confidence_interval.high
                err_low.append(max(0.0, m_val - ci_l))
                err_high.append(max(0.0, ci_h - m_val))
            else:
                err_low.append(0.0)
                err_high.append(0.0)
                
        c = COLOR_MAP.get(cond_name, "#333333")
        offset = (i - (n_models - 1) / 2.0) * width
        ax.bar(x_positions + offset, means, width, yerr=[err_low, err_high], capsize=3, label=cond_name, color=c, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"Eval corridor: {h}" for h in x_labels])
    ax.set_ylabel("Extrapolation Success Rate")
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=True, loc="upper right", ncol=2)
    plt.tight_layout()
    save_figure_to_all(fig, "tmaze_zeroshot_success.png")
    plt.close()
    print("T-Maze 10-seed plots with Mamba generated successfully.")


# ==============================================================================
# 7. CRAFTAX-CLASSIC 50M MACRO-BENCHMARK (RLIABLE IQM & BOOTSTRAP CIs)
# ==============================================================================
def plot_craftax_classic_macro_benchmark():
    import glob
    import csv
    import scipy.stats

    print("Generating Craftax-Classic 50M Macro-Benchmark plots with rliable IQM & Bootstrap CIs...")

    files = glob.glob(os.path.join(project_root, "checkpoints/craftax_50m_watchdog_T*/*_log.csv"))
    if not files:
        print("No Craftax-Classic 50M watchdog log CSV files found.")
        return

    name_mapping = {
        "SdLGHybrid_T16": "SdLG-Approx (Ours)",
        "PPO-GRU_T64": "PPO-GRU",
        "PPO-GTrXL_T64": "PPO-GTrXL",
        "AGaLiTe_T64": "AGaLiTe",
        "PPO-Dense_T16": "PPO-Dense",
        "PPO-Mamba_T64": "PPO-Mamba",
    }

    final_returns = {}
    sps_dict = {}

    for f in sorted(files):
        dir_name = os.path.basename(os.path.dirname(f))
        dir_t = dir_name.split("_")[-1]
        
        with open(f, "r") as fp:
            reader = list(csv.DictReader(fp))
            last = None
            for row in reversed(reader):
                if row and any(v.strip() for v in row.values() if v):
                    last = row
                    break
            if last and last.get("Train_Return"):
                raw_arch = last.get("Architecture")
                raw_key = f"{raw_arch}_{dir_t}"
                if raw_key not in name_mapping:
                    continue
                clean_name = name_mapping[raw_key]
                ret_val = float(last["Train_Return"])
                sps_val = float(last["SPS"]) if last.get("SPS") else 0.0

                if clean_name not in final_returns:
                    final_returns[clean_name] = []
                    sps_dict[clean_name] = []
                final_returns[clean_name].append(ret_val)
                sps_dict[clean_name].append(sps_val)

    if not final_returns:
        print("No valid benchmark data extracted.")
        return

    def iqm_stat(x):
        return scipy.stats.trim_mean(x, proportiontocut=0.25)

    stats_list = []
    for model_name, vals in final_returns.items():
        arr = np.array(vals)
        iqm = iqm_stat(arr)
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        mean_sps = np.mean(sps_dict[model_name])
        
        if len(arr) >= 3:
            res = scipy.stats.bootstrap((arr,), statistic=iqm_stat, n_resamples=2000, confidence_level=0.95, method='percentile', random_state=42)
            ci_low, ci_high = res.confidence_interval.low, res.confidence_interval.high
        else:
            ci_low, ci_high = iqm, iqm

        stats_list.append({
            "name": model_name,
            "iqm": iqm,
            "mean": mean_val,
            "std": std_val,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "err_low": iqm - ci_low,
            "err_high": ci_high - iqm,
            "sps": mean_sps,
            "n": len(arr)
        })

    stats_list.sort(key=lambda x: x["iqm"], reverse=True)

    # 7.1 Bar chart of IQMs with 95% Stratified Bootstrap CIs
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [s["name"] for s in stats_list]
    iqms = [s["iqm"] for s in stats_list]
    yerr = np.array([[s["err_low"], s["err_high"]] for s in stats_list]).T

    colors = [COLOR_MAP.get(n, "#333333") for n in names]
    bars = ax.bar(range(len(names)), iqms, yerr=yerr, capsize=5, color=colors, edgecolor="black", alpha=0.85, width=0.6)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Interquartile Mean (IQM) Return")
    ax.set_title("Craftax-Classic 50M Benchmark (rliable IQM & 95% Stratified Bootstrap CI)")
    ax.set_ylim(10, 18)

    for bar, s in zip(bars, stats_list):
        height = bar.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height + s["err_high"] + 0.15),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "craftax_classic_benchmark_iqm_summary.png"), dpi=300)
    plt.close()

    # 7.2 Bar chart of Throughput (SPS)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sps_vals = [s["sps"] for s in stats_list]
    bars = ax.bar(range(len(names)), sps_vals, color=colors, edgecolor="black", alpha=0.85, width=0.6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Training Speed (Steps-Per-Second / SPS)")
    ax.set_title("Hardware Execution Throughput across Architectures")
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{int(height):,}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "craftax_classic_benchmark_sps.png"), dpi=300)
    plt.close()

    print("Craftax-Classic 50M Macro-Benchmark plots generated successfully.")


if __name__ == "__main__":
    plot_craftax_1b()
    plot_exact_vs_approx()
    plot_capacity_scaling()
    plot_bptt_horizon()
    plot_minigrid_swa()
    plot_tmaze_ablations()
    plot_craftax_classic_macro_benchmark()
    print("\n🎉 All appendix plots have been generated and saved under papers/latex/thesis_style/plots/appendix/")

