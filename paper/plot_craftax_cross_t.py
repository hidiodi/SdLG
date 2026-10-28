"""Render the accepted historical Craftax cross-T summary as a dot-and-whisker plot."""
from pathlib import Path
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "writing/generated_artifacts/craftax_cross_t_verified.csv"
OUT = Path(__file__).resolve().parent / "plots/rebuilt/fig7_craftax_cross_t"

mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9.5,
    "axes.labelsize": 9.5, "axes.titlesize": 10.5, "legend.fontsize": 8.5,
})

styles = {
    "PPO-GRU": ("#009E73", "s"), "SdLG-Approx": ("#E69F00", "D"),
    "PPO-Mamba": ("#56B4E9", "P"), "PPO-GLA": ("#D55E00", "v"),
    "AGaLiTe": ("#CC79A7", "^"), "PPO-GTrXL": ("#444444", "X"),
    "SdLG-Exact": ("#0072B2", "o"),
}

df = pd.read_csv(DATA)
order = ["PPO-GRU", "SdLG-Approx", "PPO-Mamba", "PPO-GLA", "AGaLiTe", "PPO-GTrXL", "SdLG-Exact"]
t_markers = {16: "o", 32: "s", 64: "D", 128: "^"}
t_offsets = {16: -.18, 32: -.06, 64: .06, 128: .18}
OUT.parent.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(4.05, 2.65), constrained_layout=True)
base_y = {family: len(order) - 1 - idx for idx, family in enumerate(order)}
for family in order:
    group = df[df["Family"] == family].sort_values("T")
    color, _ = styles[family]
    for _, row in group.iterrows():
        t_value = int(row["T"])
        ax.errorbar(row["MeanReturn"], base_y[family] + t_offsets[t_value],
                    xerr=row["SampleSD"], fmt=t_markers[t_value], color=color,
                    markerfacecolor=color, markeredgecolor="white", markeredgewidth=.55,
                    markersize=5.2, capsize=2.5, linewidth=1.25, zorder=3)

for t_value, marker in t_markers.items():
    ax.scatter([], [], marker=marker, color="#444444", s=28, label=rf"$T={t_value}$")
ax.set_yticks([base_y[family] for family in order], order)
ax.set_xlabel("Final training return (mean $\\pm$ sample SD)")
ax.grid(axis="x", color="0.88", linewidth=.7)
ax.set_xlim(12.9, 17.05)
ax.set_ylim(-.55, len(order)-.45)
ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(.5, 1.01), frameon=False,
          columnspacing=.7, handletextpad=.25)
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"), dpi=260, bbox_inches="tight")
