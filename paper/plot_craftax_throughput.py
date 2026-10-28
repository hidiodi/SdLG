"""Regenerate the accepted Craftax throughput figure with embedded fonts."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 9})

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "papers/writing/generated_artifacts/table2_craftax_throughput.csv"
SDLG_SOURCE = ROOT / "papers/writing/generated_artifacts/original_paper_upgrade/craftax_t64_seven_family_throughput.csv"
OUTPUT = Path(__file__).resolve().parent / "plots/rebuilt/fig4_craftax_throughput.pdf"


def main() -> None:
    order = ["PPO-GRU", "PPO-Mamba", "SdLG-Approx", "AGaLiTe", "PPO-GLA", "PPO-GTrXL", "SdLG-Exact"]
    # SdLG aggregate values confirmed by the author on 2026-10-19.
    # Preserve the five-family source and do not invent missing uncertainty.
    sdlg = pd.read_csv(SDLG_SOURCE).set_index("family").loc[["SdLG-Approx", "SdLG-Exact"]]
    data = pd.concat([pd.read_csv(SOURCE).set_index("family"), sdlg]).loc[order].reset_index()
    x = np.arange(len(data))

    fig, ax = plt.subplots(figsize=(6.1, 3.4))
    styles = {
        "PPO-GRU": ("#009E73", "s"), "PPO-Mamba": ("#56B4E9", "P"),
        "AGaLiTe": ("#CC79A7", "^"), "PPO-GLA": ("#D55E00", "v"),
        "PPO-GTrXL": ("#444444", "X"),
        "SdLG-Approx": ("#E69F00", "D"), "SdLG-Exact": ("#0072B2", "o"),
    }
    for index, row in data.iterrows():
        color, marker = styles[row["family"]]
        uncertainty = row["sd_sps"] if pd.notna(row["sd_sps"]) else None
        ax.errorbar(index, row["mean_sps"], yerr=uncertainty, fmt=marker,
                    color=color, ecolor=color, capsize=4, markersize=6, linewidth=1.4)
    ax.set_xticks(x, data["family"], rotation=23, ha="right")
    ax.set_ylabel("Cumulative training-loop SPS")
    ax.set_title("Craftax systems measurement", fontsize=12)
    ax.set_ylim(8000, 26200)
    ax.grid(axis="y", alpha=0.2)
    ax.text(
        0.01,
        0.02,
        "Includes JIT/warm-up/logging; not inference latency",
        transform=ax.transAxes,
        fontsize=8.5,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
