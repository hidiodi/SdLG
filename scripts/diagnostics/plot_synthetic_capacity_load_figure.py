"""Plot the canonical 480-cell synthetic capacity/load analysis.

All uncertainty is computed at the seed level.  For panels (a) and (b),
each seed is first averaged over the nuisance dimension.  For panel (c),
each (N, d_k) configuration is first averaged over d_v within seed; the
same-ratio configurations are then shown separately rather than pooled.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "papers/writing/generated_artifacts/synthetic_capacity_480_cells.csv"
OUT_DIR = ROOT / "papers/latex/tmlr_review/plots/rebuilt"
ARTIFACT_DIR = ROOT / "papers/writing/generated_artifacts"


def seed_summary(df: pd.DataFrame, group_cols: list[str], nuisance: str) -> pd.DataFrame:
    """Average over nuisance within seed, then summarize across seeds."""
    per_seed = (
        df.groupby(group_cols + ["seed"], as_index=False)["cosine"]
        .mean()
        .rename(columns={"cosine": "seed_mean"})
    )
    summary = (
        per_seed.groupby(group_cols, as_index=False)["seed_mean"]
        .agg(mean="mean", sd="std", n="count")
        .rename(columns={"seed_mean": "mean"})
    )
    return summary


def main() -> None:
    df = pd.read_csv(DATA)
    expected = {"seed", "N", "d_k", "d_v", "load_ratio", "cosine"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if len(df) != 480:
        raise ValueError(f"Expected 480 rows, found {len(df)}")
    if df.groupby(["N", "d_k", "d_v"]).seed.nunique().min() != 10:
        raise ValueError("Every factorial cell must contain ten seeds")

    dk = seed_summary(df, ["N", "d_k"], "d_v")
    dv = seed_summary(df, ["N", "d_v"], "d_k")
    load = seed_summary(df, ["N", "d_k", "load_ratio"], "d_v")

    summary_rows = []
    for _, row in dk.iterrows():
        summary_rows.append({"panel": "addressing", "N": int(row.N), "d_k": int(row.d_k),
                             "d_v": "all", "load_ratio": int(row.N) / int(row.d_k),
                             "mean": row["mean"], "sd": row.sd, "n": int(row.n)})
    for _, row in dv.iterrows():
        summary_rows.append({"panel": "value", "N": int(row.N), "d_k": "all", "d_v": int(row.d_v),
                             "load_ratio": "mixed", "mean": row["mean"], "sd": row.sd,
                             "n": int(row.n)})
    for _, row in load.iterrows():
        summary_rows.append({"panel": "load_ratio", "N": int(row.N), "d_k": int(row.d_k),
                             "d_v": "all", "load_ratio": row.load_ratio, "mean": row["mean"],
                             "sd": row.sd, "n": int(row.n)})
    pd.DataFrame(summary_rows).to_csv(
        ARTIFACT_DIR / "fig6_synthetic_capacity_load_summary.csv", index=False
    )

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "legend.fontsize": 8})
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.45), constrained_layout=True)
    colors = {32: "#0072B2", 64: "#D55E00", 128: "#009E73"}
    markers = {32: "o", 64: "s", 128: "^"}

    for N in (32, 64, 128):
        part = dk[dk.N == N].sort_values("d_k")
        axes[0].errorbar(part.d_k, part["mean"], yerr=part.sd, marker=markers[N],
                         color=colors[N], capsize=2, linewidth=1.5, label=f"N={N}")
    axes[0].set_title("(a) Addressing dimension")
    axes[0].set_xlabel(r"$d_k$")
    axes[0].set_ylabel("Retrieval cosine")
    axes[0].set_xticks([16, 32, 64, 128])
    axes[0].legend(frameon=False, title="Associations")

    for N in (32, 64, 128):
        part = dv[dv.N == N].sort_values("d_v")
        axes[1].errorbar(part.d_v, part["mean"], yerr=part.sd, marker=markers[N],
                         color=colors[N], capsize=2, linewidth=1.5, label=f"N={N}")
    axes[1].set_title("(b) Value dimension")
    axes[1].set_xlabel(r"$d_v$")
    axes[1].set_ylabel("Retrieval cosine")
    axes[1].set_xticks([8, 16, 32, 64])
    axes[1].legend(frameon=False, title="Associations")

    # Plot each (N, d_k) configuration separately.  Small deterministic
    # offsets reveal repeated load ratios without implying pooled uncertainty.
    ratios = sorted(load.load_ratio.unique())
    offset_map = {}
    legend_seen = set()
    for ratio in ratios:
        configs = load[load.load_ratio == ratio].sort_values("N")
        offsets = np.linspace(-0.07, 0.07, len(configs)) if len(configs) > 1 else [0.0]
        for offset, (_, row) in zip(offsets, configs.iterrows()):
            offset_map[(int(row.N), int(row.d_k))] = float(offset)
            axes[2].errorbar(float(ratio) + offset, row["mean"], yerr=row.sd,
                             marker=markers[int(row.N)], color=colors[int(row.N)],
                             linestyle="none", capsize=2, markersize=5,
                             label=f"N={int(row.N)}" if int(row.N) not in legend_seen else None)
            legend_seen.add(int(row.N))
    axes[2].set_title("(c) Addressing load")
    axes[2].set_xlabel(r"Load ratio $N/d_k$")
    axes[2].set_ylabel("Retrieval cosine")
    axes[2].set_xticks(ratios)
    axes[2].set_xscale("log", base=2)
    axes[2].legend(frameon=False, title="Associations")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(0.25, 0.98)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig6_synthetic_capacity.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig6_synthetic_capacity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {OUT_DIR / 'fig6_synthetic_capacity.pdf'}")
    print(f"wrote {OUT_DIR / 'fig6_synthetic_capacity.png'}")
    print(f"wrote {ARTIFACT_DIR / 'fig6_synthetic_capacity_load_summary.csv'}")


if __name__ == "__main__":
    main()
