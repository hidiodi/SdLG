"""Build endpoint-first GGS figures from the accepted five-seed artifact.

The paper figure shows the paired experimental unit and its paired contrast.
Checkpoint trajectories and probe telemetry are retained in the appendix.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import itertools

mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
})


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "scripts/diagnostics/architecture/gradient_shielding/gradient_shielding.csv"
OUT_DIR = Path(__file__).resolve().parent / "plots"

LABELS = {
    "SdLGHybrid-GGS-1.0": r"Full ($\alpha=1.0$)",
    "SdLGHybrid-GGS-0.1": r"Attenuated ($\alpha=0.1$)",
}
COLORS = {
    "SdLGHybrid-GGS-1.0": "#E69F00",
    "SdLGHybrid-GGS-0.1": "#0072B2",
}
STYLES = {"SdLGHybrid-GGS-1.0": "--", "SdLGHybrid-GGS-0.1": "-"}


def main() -> None:
    df = pd.read_csv(SOURCE)
    df = df[df["Architecture"].isin(LABELS)].copy()

    endpoint = (df.dropna(subset=["Eval_Return"])
                  .sort_values("Total_Steps")
                  .groupby(["Architecture", "Seed"], as_index=False)
                  .tail(1))
    paired = endpoint.pivot(index="Seed", columns="Architecture", values="Eval_Return").sort_index()
    full = paired["SdLGHybrid-GGS-1.0"].to_numpy(float)
    attenuated = paired["SdLGHybrid-GGS-0.1"].to_numpy(float)
    differences = attenuated - full
    exact_bootstrap = np.asarray([
        np.mean(differences[list(indices)])
        for indices in itertools.product(range(len(differences)), repeat=len(differences))
    ])
    ci_low, ci_high = np.quantile(exact_bootstrap, [0.025, 0.975])

    fig, axes = plt.subplots(1, 2, figsize=(4.35, 2.05), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.25, .9]})
    ax = axes[0]
    for left, right in zip(full, attenuated):
        ax.plot([0, 1], [left, right], color="#8A8A8A", linewidth=1.0,
                marker="o", markersize=3.8, markerfacecolor="white",
                markeredgecolor="#666666", zorder=2)
    condition_means = [full.mean(), attenuated.mean()]
    ax.scatter([0, 1], condition_means, marker="D", s=42, color="#0072B2",
               edgecolor="white", linewidth=.8, zorder=4, label="Condition mean")
    ax.set_xticks([0, 1], ["Full\n" + r"$\alpha=1.0$", "Attenuated\n" + r"$\alpha=0.1$"])
    ax.set_ylabel("Fixed-endpoint evaluation return")
    ax.set_title("(a) Five matched seed pairs", fontweight="bold")
    ax.set_xlim(-.25, 1.25)
    ax.set_ylim(9.1, 12.2)
    ax.grid(axis="y", color="0.88", linewidth=.7)
    ax.legend(frameon=False, loc="upper left", fontsize=8)

    ax = axes[1]
    estimate = differences.mean()
    ax.axvline(0, color="#777777", linestyle=":", linewidth=1)
    ax.errorbar(estimate, 0, xerr=[[estimate-ci_low], [ci_high-estimate]],
                fmt="D", color="#0072B2", markerfacecolor="#0072B2",
                markersize=6, linewidth=1.8, capsize=4)
    ax.set_yticks([0], [r"$\alpha=0.1 - \alpha=1.0$"])
    ax.set_xlim(-.05, 2.15)
    ax.set_ylim(-.65, .65)
    ax.set_xlabel("Paired return difference")
    ax.set_title("(b) Mean and 95% bootstrap CI", fontweight="bold")
    ax.grid(axis="x", color="0.88", linewidth=.7)
    ax.text(estimate, .27, f"{estimate:.3f}\n[{ci_low:.3f}, {ci_high:.3f}]",
            ha="center", va="bottom", fontsize=8.5, color="#0072B2",
            fontweight="bold")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "gradient_shielding_paired.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "gradient_shielding_paired.png", dpi=260, bbox_inches="tight")
    plt.close(fig)

    # Appendix: the original training-time trajectories and diagnostic cosine.
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.45), constrained_layout=True)
    for model in ["SdLGHybrid-GGS-1.0", "SdLGHybrid-GGS-0.1"]:
        model_df = df[df["Architecture"] == model]
        grouped = model_df.groupby("Total_Steps", sort=True)
        means = grouped[["Eval_Return", "Grad_Cosine_Sim"]].mean()
        sems = grouped[["Eval_Return", "Grad_Cosine_Sim"]].sem()
        steps = means.index.to_numpy() / 1e6

        for ax, metric in zip(axes, ["Eval_Return", "Grad_Cosine_Sim"]):
            values = means[metric].to_numpy()
            errors = sems[metric].to_numpy()
            ax.plot(
                steps,
                values,
                label=LABELS[model],
                color=COLORS[model],
                linestyle=STYLES[model],
                linewidth=1.2,
            )
            ax.fill_between(
                steps,
                values - errors,
                values + errors,
                color=COLORS[model],
                alpha=0.16,
                linewidth=0,
            )

    axes[0].set_title("(a) Evaluation-return trajectory", fontweight="bold")
    axes[0].set_ylabel("Mean return")
    axes[1].set_title("(b) Actor/Critic Probe-Gradient Cosine", fontweight="bold")
    axes[1].set_ylabel("Cosine similarity")
    axes[1].axhline(0.0, color="black", linewidth=1.2, linestyle=":")

    for ax in axes:
        ax.set_xlabel("Environment steps (M)")
        ax.grid(axis="y", color="0.88", linewidth=.7)
        ax.legend(frameon=False)

    appendix_dir = OUT_DIR / "appendix"
    appendix_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(appendix_dir / "gradient_shielding_trajectories.pdf", bbox_inches="tight")
    plt.close(fig)

    auxiliary_metrics = [
        ("Train_Entropy", "Policy entropy"),
        ("Train_VLoss_Var", "Value-loss variance"),
        ("SPS", "Training-loop SPS"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.55), constrained_layout=True)
    for model in ["SdLGHybrid-GGS-1.0", "SdLGHybrid-GGS-0.1"]:
        model_df = df[df["Architecture"] == model]
        grouped = model_df.groupby("Total_Steps", sort=True)
        steps = grouped.size().index.to_numpy() / 1e6
        for ax, (metric, ylabel) in zip(axes, auxiliary_metrics):
            values = grouped[metric].mean().to_numpy()
            errors = grouped[metric].sem().to_numpy()
            ax.plot(steps, values, label=LABELS[model], color=COLORS[model],
                    linestyle=STYLES[model], linewidth=1.5)
            ax.fill_between(steps, values-errors, values+errors,
                            color=COLORS[model], alpha=.16, linewidth=0)
            ax.set_ylabel(ylabel)
    for ax in axes:
        ax.set_xlabel("Environment steps (M)")
        ax.grid(axis="y", color="0.88", linewidth=.7)
    axes[0].legend(frameon=False, loc="best")
    fig.savefig(appendix_dir / "gradient_shielding_auxiliary.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
