import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wandb


def plot_individual_craftax_ablations():
    print("Connecting to Weights & Biases API...")
    api = wandb.Api()

    # Exact paths from your logs
    entity = "schweitzer-tim-rwu-hochschule-ravensburg-weingarten-univ"
    project = "ecraftax_classic_ablation"

    try:
        runs = api.runs(f"{entity}/{project}")
    except Exception as e:
        print(f"Error connecting to W&B: {e}")
        return

    data = []
    print("Fetching run histories for Craftax-Classic ablations (this may take a minute)...")

    for run in runs:
        # Extract the model condition by removing the "-seed-X" suffix
        run_name = run.name
        if "-seed-" not in run_name:
            continue

        condition = run_name.rsplit("-seed-", 1)[0]
        seed = run_name.rsplit("-seed-", 1)[1]

        # Download telemetry
        history = run.history(keys=["_step", "Train/Mean_Return"])
        if history.empty:
            continue

        for _, row in history.iterrows():
            data.append({
                "Condition": condition,
                "Seed": seed,
                "Step": row["_step"],
                "Train_Return": row.get("Train/Mean_Return", np.nan),
            })

    df = pd.DataFrame(data)
    if df.empty:
        print("No data found! Check W&B project/entity names.")
        return

    print("Data downloaded. Generating individual paper plots...")
    os.makedirs("plots", exist_ok=True)

    # Set aesthetics suitable for individual LaTeX insertion
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman']})

    # Define color scheme for consistency across all plots
    CONTROL_COLOR = "#000000"  # Black for the anchor control model
    COLORS = {
        "Ablation_Approx_Dense": "#d7191c",  # Red
        "Ablation_GammaZero": "#2c7bb6",  # Blue
        "Ablation_GRUBias_0": "#fdae61",  # Orange
        "Capacity_WideFFN_2048": "#1a9641",  # Green
        "Capacity_NarrowFFN_512": "#a6d96a",  # Light Green
        "Capacity_16Heads": "#7b3294",  # Purple
        "Capacity_2Heads": "#c2a5cf"  # Light Purple
    }

    LABELS = {
        "Control_Exact_Dense": "Control (Exact, Log Bias, b=2.0)",
        "Ablation_Approx_Dense": "Approximate Math (AGaLiTe)",
        "Ablation_GammaZero": "Zero Bias (No Gamma Prior)",
        "Ablation_GRUBias_0": "GRU Bias = 0.0",
        "Capacity_WideFFN_2048": "Wide FFN (2048)",
        "Capacity_NarrowFFN_512": "Narrow FFN (512)",
        "Capacity_16Heads": "16 Attention Heads",
        "Capacity_2Heads": "2 Attention Heads"
    }

    def create_single_plot(conditions_to_plot, title, filename):
        fig, ax = plt.subplots(figsize=(10, 5))  # Wide aspect ratio for single column text

        for cond in conditions_to_plot:
            cond_df = df[df["Condition"] == cond]
            if cond_df.empty: continue

            # Group by Step to get Mean and Std
            grp = cond_df.groupby("Step")
            mean_df = grp.mean(numeric_only=True)
            std_df = grp.std(numeric_only=True, ddof=1).fillna(0)

            steps = mean_df.index.values / 1e6
            c = CONTROL_COLOR if cond == "Control_Exact_Dense" else COLORS.get(cond, "#333333")
            lbl = LABELS.get(cond, cond)
            lw = 3.0 if cond == "Control_Exact_Dense" else 2.5
            ls = "-" if cond == "Control_Exact_Dense" else "--"

            ax.plot(steps, mean_df["Train_Return"], label=lbl, color=c, linewidth=lw, linestyle=ls)
            ax.fill_between(steps, mean_df["Train_Return"] - std_df["Train_Return"],
                            mean_df["Train_Return"] + std_df["Train_Return"], color=c, alpha=0.15)

        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_ylabel("Mean Episode Return", fontsize=12)
        ax.set_xlabel("Environment Steps (Millions)", fontsize=12)
        ax.legend(frameon=True, loc="lower right", fontsize=11)
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        plot_path = f"plots/{filename}"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Created: {plot_path}")

    # Generate the 4 individual plots!
    create_single_plot(
        ["Control_Exact_Dense", "Ablation_Approx_Dense"],
        "Mathematical Precision (Exact vs Approx)",
        "ch6_1_exact_vs_approx.pdf"
    )

    create_single_plot(
        ["Control_Exact_Dense", "Ablation_GammaZero"],
        "Temporal Initialization (Log Prior vs Zero Bias)",
        "ch6_4_gamma_prior.pdf"
    )

    create_single_plot(
        ["Control_Exact_Dense", "Ablation_GRUBias_0"],
        "Residual Dynamics (Markovian Start vs Random)",
        "ch6_5_gru_bias.pdf"
    )

    create_single_plot(
        ["Control_Exact_Dense", "Capacity_WideFFN_2048", "Capacity_NarrowFFN_512", "Capacity_16Heads",
         "Capacity_2Heads"],
        "Network Capacity Scaling (The Visual Bottleneck)",
        "ch6_6_capacity_scaling.pdf"
    )


if __name__ == "__main__":
    plot_individual_craftax_ablations()