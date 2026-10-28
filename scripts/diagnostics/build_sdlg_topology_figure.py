"""Build Figure 1: Exact as the reference, variants as modifications."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "papers/latex/cleanRewrite/plots/rebuilt/fig1_configuration_topologies"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def box(ax, xy, width, height, text, facecolor, edgecolor="#34495e", fontsize=8.5,
        linewidth=1.2, weight="normal"):
    patch = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=linewidth, facecolor=facecolor, edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2, xy[1] + height / 2, text,
        ha="center", va="center", fontsize=fontsize, weight=weight,
    )
    return patch


def arrow(ax, start, end, color="#34495e", style="-|>", lw=1.15,
          connectionstyle="arc3"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=10, lw=lw,
        color=color, connectionstyle=connectionstyle,
    ))


def main():
    fig, ax = plt.subplots(figsize=(12.2, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.97, "SdLG-Exact reference architecture", ha="center",
            va="center", fontsize=14, weight="bold")
    ax.text(0.5, 0.925,
            "Local and associative processing are sequential within every block",
            ha="center", va="center", fontsize=9.5, color="#4b5d67")

    # Reference flow.
    box(ax, (0.012, 0.61), 0.085, 0.14, "observation\nencoder", "#dceaf7",
        fontsize=8.2, weight="bold")
    arrow(ax, (0.097, 0.68), (0.125, 0.68))

    stages = [
        (0.13, 0.61, 0.105, "LayerNorm\n+ local SWA", "#f9e3c0"),
        (0.265, 0.61, 0.095, "local gated\nresidual", "#dfe8d5"),
        (0.39, 0.61, 0.13, "LayerNorm\n+ Exact memory", "#cfe8f3"),
        (0.55, 0.61, 0.10, "memory gated\nresidual", "#dfe8d5"),
        (0.68, 0.61, 0.08, "LayerNorm\n+ FFN", "#e6ddf2"),
        (0.79, 0.61, 0.09, "FFN gated\nresidual", "#dfe8d5"),
    ]
    for i, (x, y, w, label, color) in enumerate(stages):
        box(ax, (x, y), w, 0.14, label, color, fontsize=8.0)
        if i:
            px, py, pw, _, _ = stages[i - 1]
            arrow(ax, (px + pw, py + 0.07), (x, y + 0.07))

    # State annotations make the two memory types explicit.
    box(ax, (0.132, 0.795), 0.10, 0.075, "$K_{swa},V_{swa}$\nlast $W$ items",
        "#fff7e8", edgecolor="#c8964c", fontsize=7.3)
    arrow(ax, (0.182, 0.795), (0.182, 0.755), color="#c07b22")
    box(ax, (0.40, 0.795), 0.11, 0.075, "$M,z$ per head\nfull-matrix state",
        "#eef8fc", edgecolor="#5996ad", fontsize=7.3)
    arrow(ax, (0.455, 0.795), (0.455, 0.755), color="#3e829e")

    # Clarify the residual skip/fusion semantics without overloading the path.
    ax.text(0.31, 0.575, "SWA changes the input used to form associative $q,k,v$",
            ha="center", va="center", fontsize=8.1, color="#8b3a3a")
    arrow(ax, (0.355, 0.595), (0.39, 0.635), color="#8b3a3a", lw=1.0)

    ax.add_patch(FancyBboxPatch(
        (0.115, 0.585), 0.78, 0.19,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.0, facecolor="none", edgecolor="#72828d", linestyle="--",
    ))
    ax.text(0.86, 0.785, "Exact block $\\times 3$", ha="right", va="bottom",
            fontsize=8.3, weight="bold", color="#4b5d67")
    arrow(ax, (0.88, 0.68), (0.91, 0.68))
    box(ax, (0.915, 0.61), 0.07, 0.14, "actor +\ncritic", "#dceaf7",
        fontsize=8.1, weight="bold")

    ax.text(0.5, 0.49, "Architectural variants and study controls", ha="center", va="center",
            fontsize=12, weight="bold")

    variants = [
        (0.035, 0.14, 0.205, "Positional phase",
         "RoPE rotates Exact $q,k$ before\n$\\varphi$; NoRoPE leaves them unchanged.", "#f7edf5"),
        (0.285, 0.14, 0.205, "Approximate operation",
         "Replace $(M,z)$ and its read with\n$\\widetilde K,\\widetilde V,\\theta$ and a factorized read.", "#f9dfad"),
        (0.535, 0.14, 0.205, "Original factorial",
         "Also changes layer placement and\nroutes the attenuated trunk to both heads.", "#fce4e4"),
        (0.785, 0.14, 0.18, "Matched control",
         "Restores three complete blocks\nand one unattenuated shared trunk.", "#e3f0df"),
    ]
    for x, y, w, title, body, color in variants:
        box(ax, (x, y), w, 0.24, "", color, fontsize=8)
        ax.text(x + w / 2, y + 0.18, title, ha="center", va="center",
                fontsize=9, weight="bold")
        ax.text(x + w / 2, y + 0.09, body, ha="center", va="center",
                fontsize=7.7, linespacing=1.25)

    ax.text(0.5, 0.055,
            "Memory-update gates ($\\beta,\\gamma$), residual gates, and actor-gradient scale ($\\alpha$) act at different levels.",
            ha="center", va="center", fontsize=8.5, color="#4b5d67")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.25)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
