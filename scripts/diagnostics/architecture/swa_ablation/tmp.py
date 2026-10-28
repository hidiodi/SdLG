import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wandb


def download_and_plot_wandb_data():
    print("Connecting to Weights & Biases API...")
    api = wandb.Api()

    # Extracting the exact entity and project from your logs
    entity = "schweitzer-tim-rwu-hochschule-ravensburg-weingarten-univ"
    project = "KeyCorridorXminigrid_Ablations"

    runs = api.runs(f"{entity}/{project}")

    data = []
    print("Fetching run histories (this may take a few seconds)...")

    for run in runs:
        # Only grab the MemoryS16 runs from today
        if "MiniGrid-MemoryS16" not in run.name:
            continue

        # Parse the name: e.g., "SdLG-SWA-MiniGrid-MemoryS16-Seed-42"
        parts = run.name.split("-")
        condition = f"{parts[0]}-{parts[1]}"  # "SdLG-SWA" or "SdLG-None"
        seed = parts[-1]

        # Download the history telemetry
        history = run.history(
            keys=["_step", "Train/Mean_Return", "Train/Policy_Entropy", "Train/Value_Loss_Var", "System/SPS"])

        # Grab the final evaluation return from the summary
        eval_ret = run.summary.get("Eval/Mean_Return", np.nan)

        if history.empty:
            continue

        max_step = history["_step"].max()

        for _, row in history.iterrows():
            step = row["_step"]
            data.append({
                "Condition": condition,
                "Seed": seed,
                "Step": step,
                "Train_Return": row.get("Train/Mean_Return", np.nan),
                "Entropy": row.get("Train/Policy_Entropy", np.nan),
                "VLoss_Var": row.get("Train/Value_Loss_Var", np.nan),
                "SPS": row.get("System/SPS", np.nan),
                "Eval_Return": eval_ret if step == max_step else np.nan
            })

    df = pd.DataFrame(data)

    if df.empty:
        print("No data found! Check your W&B entity/project names.")
        return

    print("Data downloaded successfully. Generating publication plots...")

    # Setup plot aesthetics
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Custom colors
    colors = {"SdLG-SWA": "#2c7bb6", "SdLG-None": "#d7191c"}
    styles = {"SdLG-SWA": "-", "SdLG-None": "--"}

    conditions = df["Condition"].unique()

    for cond in conditions:
        cond_df = df[df["Condition"] == cond]

        # Group by Step to get Mean and Std (Standard Deviation)
        grp = cond_df.groupby("Step")
        mean_df = grp.mean(numeric_only=True)
        std_df = grp.std(numeric_only=True, ddof=1)

        steps_millions = mean_df.index.values / 1e6
        c = colors.get(cond, "black")
        s = styles.get(cond, "-")

        # --- Panel A: Return ---
        axes[0, 0].plot(steps_millions, mean_df["Train_Return"], label=f"{cond} (Train)", color=c, linestyle=s,
                        linewidth=2.0)
        axes[0, 0].fill_between(steps_millions, mean_df["Train_Return"] - std_df["Train_Return"],
                                mean_df["Train_Return"] + std_df["Train_Return"], color=c, alpha=0.15)

        # Add Evaluation dots at the end
        eval_data = mean_df["Eval_Return"].dropna()
        if not eval_data.empty:
            axes[0, 0].plot(eval_data.index.values / 1e6, eval_data.values, label=f"{cond} (Eval)", color=c, marker="*",
                            markersize=12, linestyle="", markeredgecolor='black')

        # --- Panel B: Entropy ---
        axes[0, 1].plot(steps_millions, mean_df["Entropy"], label=cond, color=c, linestyle=s, linewidth=2.0)
        axes[0, 1].fill_between(steps_millions, mean_df["Entropy"] - std_df["Entropy"],
                                mean_df["Entropy"] + std_df["Entropy"], color=c, alpha=0.15)

        # --- Panel C: Value Loss Variance ---
        axes[1, 0].plot(steps_millions, mean_df["VLoss_Var"], label=cond, color=c, linestyle=s, linewidth=2.0)

        # --- Panel D: SPS ---
        axes[1, 1].plot(steps_millions, mean_df["SPS"], label=cond, color=c, linestyle=s, linewidth=2.0)

    # Formatting
    axes[0, 0].set_title("Generalization: Train vs Eval (MiniGrid-MemoryS16)", fontweight='bold')
    axes[0, 0].set_ylabel("Mean Episode Return")

    axes[0, 1].set_title("Exploration: Policy Entropy", fontweight='bold')
    axes[0, 1].set_ylabel("Entropy")

    axes[1, 0].set_title("Optimization Stability: Value Loss Variance", fontweight='bold')
    axes[1, 0].set_ylabel("Variance (Log Scale)")
    axes[1, 0].set_yscale('log')

    axes[1, 1].set_title("Hardware Efficiency: Throughput (SPS)", fontweight='bold')
    axes[1, 1].set_ylabel("Steps Per Second")

    for ax in axes.flatten():
        ax.set_xlabel("Environment Steps (Millions)")
        ax.legend(frameon=True)

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plot_path = "plots/swa_ablation_MiniGrid-MemoryS16_paper.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Success! High-resolution paper plot saved to: {plot_path}")


if __name__ == "__main__":
    download_and_plot_wandb_data()