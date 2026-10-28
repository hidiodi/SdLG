import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wandb


def plot_doorkey_tradeoff():
    print("Connecting to Weights & Biases API...")
    api = wandb.Api()

    entity = "schweitzer-tim-rwu-hochschule-ravensburg-weingarten-univ"
    project = "KeyCorridorXminigrid_Ablations"

    runs = api.runs(f"{entity}/{project}")

    data = []
    print("Fetching run histories for DoorKey-5x5 tradeoff...")

    for run in runs:
        # Filter by the group we used for this specific experiment
        if run.group != "exp18_keycorridor_bossfight":
            continue

        # Parse condition based on what was logged
        if "Approx_SdLG_Dense" in run.name or "Exact_SdLG_Dense" in run.name:
            condition = "Approx_SdLG_Dense"
        elif "ExactMoEv2" in run.name:
            condition = "ExactMoEv2_W32"
        else:
            continue

        seed = run.name.split("-")[-1]

        # Download telemetry
        history = run.history(keys=["_step", "Train/Mean_Return", "System/SPS"])
        if history.empty: continue

        for _, row in history.iterrows():
            data.append({
                "Condition": condition,
                "Seed": seed,
                "Step": row["_step"],
                "Train_Return": row.get("Train/Mean_Return", np.nan),
                "SPS": row.get("System/SPS", np.nan),
            })

    df = pd.DataFrame(data)
    if df.empty:
        print("No data found! Check if the runs are definitely in the 'exp18_keycorridor_bossfight' group.")
        return

    print("Data downloaded. Generating paper plot...")

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"Approx_SdLG_Dense": "#d7191c", "ExactMoEv2_W32": "#2c7bb6"}
    labels = {"Approx_SdLG_Dense": "Approx-Dense (Baseline)", "ExactMoEv2_W32": "Exact-MoE (Ours)"}

    for cond in df["Condition"].unique():
        cond_df = df[df["Condition"] == cond]

        # Group to get mean and std
        grp = cond_df.groupby("Step")
        mean_df = grp.mean(numeric_only=True)
        std_df = grp.std(numeric_only=True, ddof=1).fillna(0)

        steps = mean_df.index.values / 1e6
        c = colors.get(cond, "black")
        lbl = labels.get(cond, cond)

        # Panel A: Sample Efficiency (Return)
        axes[0].plot(steps, mean_df["Train_Return"], label=lbl, color=c, linewidth=2.5)
        axes[0].fill_between(steps, mean_df["Train_Return"] - std_df["Train_Return"],
                             mean_df["Train_Return"] + std_df["Train_Return"], color=c, alpha=0.15)

        # Panel B: Hardware Efficiency (SPS)
        axes[1].plot(steps, mean_df["SPS"], label=lbl, color=c, linewidth=2.5)
        axes[1].fill_between(steps, mean_df["SPS"] - std_df["SPS"], mean_df["SPS"] + std_df["SPS"], color=c, alpha=0.15)

    axes[0].set_title("Sample Efficiency: Mean Return", fontweight='bold')
    axes[0].set_ylabel("Episode Return")
    axes[0].set_xlabel("Environment Steps (Millions)")
    axes[0].legend(frameon=True)

    axes[1].set_title(r"Hardware Efficiency: Throughput ($\eta_{SPS}$)", fontweight='bold')
    axes[1].set_ylabel("Steps Per Second (SPS)")
    axes[1].set_xlabel("Environment Steps (Millions)")
    axes[1].legend(frameon=True)

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plot_path = "plots/doorkey_moe_tradeoff_paper.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Success! Plot saved to: {plot_path}")


if __name__ == "__main__":
    plot_doorkey_tradeoff()