"""Render the canonical T-Maze factorial with a publication-first hierarchy.

The terminal-step panel preserves seed-level censoring, the return panel uses
the presentation's immediately readable heatmap, and a compact third panel
shows the independent-condition H=200 contrasts from the original factorial
and the later topology/gradient-matched study.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "papers" / "writing" / "generated_artifacts"
OUT = ROOT / "papers" / "latex" / "cleanRewrite" / "plots" / "rebuilt"
TERMINAL = ART / "tmaze_rank_rope_terminal_training.csv"
HORIZONS = ART / "tmaze_rank_rope_zero_shot.csv"
CONTRASTS = ART / "revision_2026" / "tmaze_factorial_contrasts.csv"

ORDER = ["Exact_NoRoPE", "Exact_RoPE", "Approx_NoRoPE", "Approx_RoPE"]
LABELS = ["Exact\nNoRoPE", "Exact\nRoPE", "Approx.\nNoRoPE", "Approx.\nRoPE"]
COLORS = {model: ("#0072B2" if model.startswith("Exact") else "#E69F00") for model in ORDER}
LINESTYLES = {model: ("-" if model.endswith("NoRoPE") else "--") for model in ORDER}
MAX_M = 49.9712


def main() -> None:
    terminal = pd.read_csv(TERMINAL)
    horizon = pd.read_csv(HORIZONS)
    if set(terminal.model.unique()) != set(ORDER) or set(horizon.model.unique()) != set(ORDER):
        raise ValueError("Unexpected factorial conditions")
    if terminal.groupby("model").seed.nunique().to_dict() != {m: 10 for m in ORDER}:
        raise ValueError("Expected ten terminal-step runs per condition")
    if horizon.groupby("model").seed.nunique().to_dict() != {m: 10 for m in ORDER}:
        raise ValueError("Expected ten horizon evaluations per condition")

    contrasts = pd.read_csv(CONTRASTS)

    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
    })
    fig, axes = plt.subplots(1, 3, figsize=(9.25, 3.05), constrained_layout=True,
                             gridspec_kw={"width_ratios": [0.82, 1.12, 0.75]})
    rng = np.random.default_rng(20260912)

    # Panel (a): endpoint terminal step, with censored full-budget values distinct.
    ax = axes[0]
    for idx, model in enumerate(ORDER):
        values = terminal.loc[terminal.model == model, "step"].to_numpy() / 1e6
        offsets = rng.uniform(-0.13, 0.13, len(values))
        censored = np.isclose(values, MAX_M, atol=0.01)
        face = COLORS[model] if model.endswith("NoRoPE") else "white"
        ax.scatter(np.full(censored.sum(), idx) + offsets[censored], values[censored],
                   facecolor=face, edgecolor=COLORS[model], marker="^", s=28, linewidth=.9, zorder=3)
        ax.scatter(np.full((~censored).sum(), idx) + offsets[~censored], values[~censored],
                   facecolor=face, edgecolor=COLORS[model], marker="o", s=24, linewidth=.9, zorder=3)
        ax.errorbar(idx, values.mean(), yerr=values.std(ddof=1), color="black", capsize=2.5,
                    marker="_", markersize=10, linewidth=1.1, zorder=4)
    ax.axhline(MAX_M, color="#777777", linewidth=0.8, linestyle="--", zorder=1)
    ax.text(3.45, MAX_M - 1.6, "50M budget", ha="right", va="top", fontsize=7, color="#555555")
    ax.set_title("(a) Training endpoint")
    ax.set_ylabel("Environment steps (millions)")
    ax.set_xticks(range(4), LABELS)
    ax.set_ylim(0, 52)
    ax.grid(axis="y", alpha=0.22)
    ax.text(0.02, -0.25, "Triangles: right-censored at 50M.",
            transform=ax.transAxes, fontsize=7, color="#555555")

    # Panel (b): mean-return heatmap; small SD labels move to the appendix.
    ax = axes[1]
    cols = [(100, "val_H100", "H=100\ntrained"), (150, "val_H150", "H=150\nunseen"),
            (200, "val_H200", "H=200\nunseen")]
    means = np.zeros((len(ORDER), len(cols)))
    sds = np.zeros_like(means)
    for i, model in enumerate(ORDER):
        part = horizon[horizon.model == model]
        for j, (_, col, _) in enumerate(cols):
            values = part[col].to_numpy(float)
            means[i, j] = values.mean()
            sds[i, j] = values.std(ddof=1)
    image = ax.imshow(means, cmap="cividis", vmin=.45, vmax=.92, aspect="auto")
    ax.set_title("(b) Evaluation return")
    ax.set_xticks(range(len(cols)), [label for _, _, label in cols])
    ax.set_yticks(range(len(ORDER)), [label.replace("\n", " ") for label in LABELS])
    for i in range(len(ORDER)):
        for j in range(len(cols)):
            color = "white" if means[i, j] < .70 else "black"
            ax.text(j, i, f"{means[i, j]:.3f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)
    table = []
    for i, model in enumerate(ORDER):
        entries = [f"${means[i,j]:.3f}\\pm{sds[i,j]:.3f}$" for j in range(3)]
        table.append(model.replace("_", "-") + " & " + " & ".join(entries) + r" \\")
    (ROOT / "papers/latex/cleanRewrite/artifacts/tmaze_factorial_mean_sd.tex").write_text("\n".join(table) + "\n\\bottomrule\n")
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.colorbar(image, ax=ax, fraction=.045, pad=.03, label="Mean return")

    # Panel (c): make the matched-control result visible without conflating studies.
    ax = axes[2]
    selected = []
    for study, contrast, label, color in [
        ("factorial", "execution_exact_minus_approx", "Original factorial\n(marginal)", "#6A6A6A"),
        ("matched", "matched_execution_exact_minus_approx", "Matched study\n(NoRoPE)", "#0072B2"),
    ]:
        row = contrasts[(contrasts.study == study) & (contrasts.horizon == 200) &
                        (contrasts.contrast == contrast)].iloc[0]
        selected.append((label, float(row.estimate), float(row.bootstrap_ci_low),
                         float(row.bootstrap_ci_high), color))
    for y, (label, estimate, low, high, color) in enumerate(selected[::-1]):
        ax.errorbar(estimate, y, xerr=[[estimate-low], [high-estimate]], fmt="D",
                    color=color, markerfacecolor=color, capsize=3, linewidth=1.5,
                    markersize=5.5)
        ax.text(estimate, y + .22, f"{estimate:.3f}", ha="center", va="bottom",
                fontsize=7.2, color=color, fontweight="bold")
    ax.axvline(0, color="#777777", linestyle=":", linewidth=1)
    ax.set_yticks(range(2), [entry[0] for entry in selected[::-1]])
    ax.set_xlim(0, .56)
    ax.set_ylim(-.55, 1.55)
    ax.set_xlabel(r"Exact $-$ Approx. return")
    ax.set_title("(c) $H=200$ contrasts")
    ax.grid(axis="x", color="0.88", linewidth=.7)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig3_tmaze_rank_rope.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig3_tmaze_rank_rope.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'fig3_tmaze_rank_rope.pdf'}")
    print(f"wrote {OUT / 'fig3_tmaze_rank_rope.png'}")


if __name__ == "__main__":
    main()
