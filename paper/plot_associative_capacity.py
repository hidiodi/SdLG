"""Build the main-paper associative-retrieval capacity figure from cell data."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / "artifacts" / "associative_capacity_480_cells.csv"
OUT = HERE / "plots" / "rebuilt" / "fig6_associative_capacity.pdf"

rows = []
with DATA.open(newline="") as handle:
    for row in csv.DictReader(handle):
        rows.append(
            {
                "seed": int(row["seed"]),
                "N": int(row["N"]),
                "d_k": int(row["d_k"]),
                "d_v": int(row["d_v"]),
                "cosine": float(row["cosine"]),
            }
        )

loads = [32, 64, 128]
key_dims = [16, 32, 64, 128]
value_dims = [8, 16, 32, 64]
cells: dict[tuple[int, int], list[float]] = defaultdict(list)
key_marginals: dict[tuple[int, int], list[float]] = defaultdict(list)
value_marginals: dict[tuple[int, int], list[float]] = defaultdict(list)

for row in rows:
    cells[(row["N"], row["d_k"])].append(row["cosine"])
for n in loads:
    for seed in range(10):
        selected = [row for row in rows if row["N"] == n and row["seed"] == seed]
        for d_k in key_dims:
            key_marginals[(n, d_k)].append(mean(row["cosine"] for row in selected if row["d_k"] == d_k))
        for d_v in value_dims:
            value_marginals[(n, d_v)].append(mean(row["cosine"] for row in selected if row["d_v"] == d_v))


OUT.parent.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
    "font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 10.5,
    "legend.fontsize": 9.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
})
fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.25), constrained_layout=True,
                         gridspec_kw={"height_ratios": [1.22, 1]})

matrix = np.asarray([[mean(cells[(n, d_k)]) for d_k in key_dims] for n in loads])
image = axes[0].imshow(matrix, cmap="cividis", vmin=.30, vmax=.93, aspect="auto")
axes[0].set_title("(a) Mean retrieval across value dimensions")
axes[0].set_xlabel(r"Addressing dimension $d_k$")
axes[0].set_ylabel(r"Associations written $N$")
axes[0].set_xticks(range(len(key_dims)), key_dims)
axes[0].set_yticks(range(len(loads)), loads)
for i, n in enumerate(loads):
    for j, d_k in enumerate(key_dims):
        value = matrix[i, j]
        color = "white" if value < .56 else "black"
        axes[0].text(j, i-.10, f"{value:.2f}", ha="center", va="center",
                     color=color, fontsize=10, fontweight="bold")
        axes[0].text(j, i+.24, rf"${n/d_k:g}$", ha="center", va="center",
                     color=color, fontsize=8)
fig.colorbar(image, ax=axes[0], fraction=.045, pad=.03, label="Mean cosine")

key_spans, value_spans = [], []
for n in loads:
    key_means = [mean(key_marginals[(n, d_k)]) for d_k in key_dims]
    value_means = [mean(value_marginals[(n, d_v)]) for d_v in value_dims]
    key_spans.append(max(key_means)-min(key_means))
    value_spans.append(max(value_means)-min(value_means))
y = np.arange(len(loads))
height = .32
axes[1].barh(y-height/2, key_spans, height, color="#E69F00", label=r"vary $d_k$")
axes[1].barh(y+height/2, value_spans, height, color="#0072B2", label=r"vary $d_v$")
for row, values in ((y-height/2, key_spans), (y+height/2, value_spans)):
    for pos, value in zip(row, values):
        axes[1].text(value+.006, pos, f"{value:.3f}", va="center", fontsize=9)
axes[1].set_title("(b) Dimension sensitivity")
axes[1].set_xlabel("Range of marginal mean retrieval")
axes[1].set_yticks(y, [rf"$N={n}$" for n in loads])
axes[1].invert_yaxis()
axes[1].set_xlim(0, .45)
axes[1].grid(axis="x", color="0.88", linewidth=.7)
axes[1].legend(frameon=False)
fig.savefig(OUT, bbox_inches="tight")
plt.close(fig)
