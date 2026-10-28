#!/usr/bin/env python3
"""Generate core frozen-paper figures from standalone structured records."""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "paper-records"
DEFAULT_OUTPUT = ROOT / "results" / "figures"


def rows(relative: str) -> list[dict[str, str]]:
    with (DATA / relative).open(newline="") as stream:
        return list(csv.DictReader(stream))


def data_rows(relative: str) -> list[dict[str, str]]:
    with (ROOT / "data" / relative).open(newline="") as stream:
        return list(csv.DictReader(stream))


def display_path(path: Path) -> Path:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def save(fig: plt.Figure, output: Path, name: str) -> None:
    fig.tight_layout()
    for suffix, kwargs in (("png", {"dpi": 220}), ("pdf", {})):
        path = output / f"{name}.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        print(f"wrote {display_path(path)}")
    plt.close(fig)


def tmaze(output: Path) -> None:
    data = rows("tmaze_rank_rope_zero_shot.csv")
    horizons = (100, 150, 200)
    fig, axis = plt.subplots(figsize=(6.2, 3.8))
    for model in ("Exact_RoPE", "Exact_NoRoPE", "Approx_RoPE", "Approx_NoRoPE"):
        group = [row for row in data if row["model"] == model]
        means = [np.mean([float(row[f"val_H{h}"]) for row in group]) for h in horizons]
        sds = [np.std([float(row[f"val_H{h}"]) for row in group], ddof=1) for h in horizons]
        axis.errorbar(horizons, means, yerr=sds, marker="o", capsize=3, label=model.replace("_", " "))
    axis.set(xlabel="Evaluation horizon", ylabel="Deterministic return", ylim=(0.35, 1.05))
    axis.legend(frameon=False, ncol=2)
    save(fig, output, "tmaze_execution_phase")


def minigrid(output: Path) -> None:
    data = rows("revision_2026/minigrid_locality_seed_endpoints.csv")
    groups = {name: [float(row["Eval_Return"]) for row in data if row["Condition"] == name] for name in ("SdLG-SWA", "SdLG-None")}
    fig, axis = plt.subplots(figsize=(4.6, 3.8))
    axis.boxplot(list(groups.values()), tick_labels=["SWA", "Bypassed"])
    for index, values in enumerate(groups.values(), start=1):
        axis.scatter(np.full(len(values), index), values, color="black", s=13, alpha=0.7)
    axis.set(ylabel="Fixed-endpoint evaluation return", ylim=(-0.05, 1.05))
    save(fig, output, "minigrid_locality")


def craftax(output: Path) -> None:
    data = rows("craftax_cross_t_verified.csv")
    fig, axis = plt.subplots(figsize=(6.2, 3.8))
    for family in sorted({row["Family"] for row in data}):
        group = sorted((row for row in data if row["Family"] == family), key=lambda row: int(row["T"]))
        axis.errorbar([int(row["T"]) for row in group], [float(row["MeanReturn"]) for row in group], yerr=[float(row["SampleSD"]) for row in group], marker="o", capsize=3, label=family)
    axis.set(xlabel="Rollout / TBPTT length", ylabel="Final logged training return")
    axis.legend(frameon=False, ncol=2, fontsize=8)
    save(fig, output, "craftax_cross_t")


def ggs(output: Path) -> None:
    data = rows("ggs_endpoint_paired_difference.csv")
    seeds = [row["seed"] for row in data]
    low = [float(row["alpha_0_1_eval_return"]) for row in data]
    high = [float(row["alpha_1_0_eval_return"]) for row in data]
    fig, axis = plt.subplots(figsize=(5.2, 3.8))
    for left, right in zip(low, high):
        axis.plot((0, 1), (left, right), color="0.7", zorder=1)
    axis.scatter(np.zeros(len(low)), low, label="alpha=0.1", zorder=2)
    axis.scatter(np.ones(len(high)), high, label="alpha=1.0", zorder=2)
    axis.set(xticks=(0, 1), xticklabels=("alpha=0.1", "alpha=1.0"), ylabel="Endpoint evaluation return")
    axis.legend(frameon=False)
    save(fig, output, "craftax_ggs")


def capacity(output: Path) -> None:
    data = rows("synthetic_capacity_480_cells.csv")
    fig, axis = plt.subplots(figsize=(5.5, 3.8))
    for key_dimension in sorted({int(row["d_k"]) for row in data}):
        group = [row for row in data if int(row["d_k"]) == key_dimension]
        loads = sorted({float(row["load_ratio"]) for row in group})
        means = [np.mean([float(row["cosine"]) for row in group if float(row["load_ratio"]) == load]) for load in loads]
        axis.plot(loads, means, marker="o", label=f"d_k={key_dimension}")
    axis.set(xlabel="Associative load N / d_k", ylabel="Mean retrieval cosine")
    axis.legend(frameon=False, ncol=2)
    save(fig, output, "associative_capacity")


def tmaze_paper(output: Path) -> None:
    """Render the three-panel main-paper T-Maze summary from tracked records."""
    terminal = rows("tmaze_rank_rope_terminal_training.csv")
    zero_shot = rows("tmaze_rank_rope_zero_shot.csv")
    contrasts = rows("revision_2026/tmaze_exact_vs_external_contrasts.csv")
    models = ("Exact_RoPE", "Exact_NoRoPE", "Approx_RoPE", "Approx_NoRoPE")
    labels = [model.replace("_", " ") for model in models]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1))

    terminal_means = []
    for model in models:
        values = [float(row["step"]) / 1e6 for row in terminal if row["model"] == model]
        terminal_means.append(np.mean(values))
        axes[0].scatter(np.full(len(values), len(terminal_means) - 1), values, s=14, alpha=.65)
    axes[0].plot(range(len(models)), terminal_means, color="black", marker="D", linewidth=1)
    axes[0].set(xticks=range(len(models)), xticklabels=labels, ylabel="Terminal steps (M)", title="(a) Training termination")
    axes[0].tick_params(axis="x", rotation=30, labelsize=7)

    horizons = (100, 150, 200)
    for model in models:
        group = [row for row in zero_shot if row["model"] == model]
        means = [np.mean([float(row[f"val_H{h}"]) for row in group]) for h in horizons]
        sds = [np.std([float(row[f"val_H{h}"]) for row in group], ddof=1) for h in horizons]
        axes[1].errorbar(horizons, means, yerr=sds, marker="o", capsize=2.5, label=model.replace("_", " "))
    axes[1].set(xlabel="Evaluation horizon", ylabel="Deterministic return", ylim=(.35, 1.05), title="(b) Zero-shot evaluation")
    axes[1].legend(frameon=False, fontsize=6.7)

    target = [row for row in contrasts if int(row["horizon"]) == 200]
    y = np.arange(len(target))
    estimates = [float(row["estimate_exact_minus_baseline"]) for row in target]
    low = [float(row["bootstrap_ci_low"]) for row in target]
    high = [float(row["bootstrap_ci_high"]) for row in target]
    axes[2].errorbar(estimates, y, xerr=[np.subtract(estimates, low), np.subtract(high, estimates)], fmt="o", capsize=2.5)
    axes[2].axvline(0, color="0.5", linestyle=":")
    axes[2].set(yticks=y, yticklabels=[row["baseline_model"] for row in target], xlabel="Exact − baseline", title="(c) H=200 contrasts")
    axes[2].invert_yaxis()
    save(fig, output, "fig3_tmaze_rank_rope")


def minigrid_paper(output: Path) -> None:
    data = rows("revision_2026/minigrid_locality_seed_endpoints.csv")
    groups = {name: [float(row["Eval_Return"]) for row in data if row["Condition"] == name] for name in ("SdLG-SWA", "SdLG-None")}
    fig, axis = plt.subplots(figsize=(4.2, 3.0))
    axis.boxplot(list(groups.values()), tick_labels=("SWA enabled", "SWA bypassed"), showmeans=True)
    for index, values in enumerate(groups.values(), start=1):
        axis.scatter(np.full(len(values), index), values, color="black", s=11, alpha=.65)
    axis.set(ylabel="Fixed-endpoint evaluation return", ylim=(-.05, 1.05), title="MiniGrid local-pathway comparison")
    save(fig, output, "fig5_minigrid_locality_expanded")


def capacity_paper(output: Path) -> None:
    data = rows("synthetic_capacity_480_cells.csv")
    loads, key_dims, value_dims = (32, 64, 128), (16, 32, 64, 128), (8, 16, 32, 64)
    cells = {(n, dk): [float(row["cosine"]) for row in data if int(row["N"]) == n and int(row["d_k"]) == dk] for n in loads for dk in key_dims}
    fig, axes = plt.subplots(2, 1, figsize=(4.1, 4.8), gridspec_kw={"height_ratios": (1.25, 1)})
    matrix = np.asarray([[np.mean(cells[(n, dk)]) for dk in key_dims] for n in loads])
    image = axes[0].imshow(matrix, cmap="cividis", vmin=.30, vmax=.93, aspect="auto")
    axes[0].set(xticks=range(len(key_dims)), xticklabels=key_dims, yticks=range(len(loads)), yticklabels=loads, xlabel=r"Addressing dimension $d_k$", ylabel="Associations $N$", title="(a) Mean retrieval cosine")
    for i, n in enumerate(loads):
        for j, dk in enumerate(key_dims):
            axes[0].text(j, i, f"{matrix[i, j]:.2f}\n{n/dk:g}", ha="center", va="center", color="white" if matrix[i, j] < .56 else "black", fontsize=8)
    fig.colorbar(image, ax=axes[0], fraction=.045, pad=.03)
    key_spans, value_spans = [], []
    for n in loads:
        key_values = [np.mean(cells[(n, dk)]) for dk in key_dims]
        value_values = [np.mean([float(row["cosine"]) for row in data if int(row["N"]) == n and int(row["d_v"]) == dv]) for dv in value_dims]
        key_spans.append(max(key_values) - min(key_values))
        value_spans.append(max(value_values) - min(value_values))
    y = np.arange(len(loads))
    axes[1].barh(y - .18, key_spans, .35, label=r"vary $d_k$")
    axes[1].barh(y + .18, value_spans, .35, label=r"vary $d_v$")
    axes[1].set(yticks=y, yticklabels=[f"N={n}" for n in loads], xlabel="Range of marginal mean retrieval", title="(b) Dimension sensitivity")
    axes[1].legend(frameon=False)
    save(fig, output, "fig6_associative_capacity")


def craftax_cross_t_paper(output: Path) -> None:
    data = rows("craftax_cross_t_verified.csv")
    order = ("PPO-GRU", "SdLG-Approx", "PPO-Mamba", "PPO-GLA", "AGaLiTe", "PPO-GTrXL", "SdLG-Exact")
    markers, offsets = {16: "o", 32: "s", 64: "D", 128: "^"}, {16: -.18, 32: -.06, 64: .06, 128: .18}
    fig, axis = plt.subplots(figsize=(5.0, 3.2))
    for base, family in enumerate(reversed(order)):
        for row in [item for item in data if item["Family"] == family]:
            t = int(row["T"])
            axis.errorbar(float(row["MeanReturn"]), base + offsets[t], xerr=float(row["SampleSD"]), fmt=markers[t], capsize=2.5, label=None)
    axis.set(yticks=range(len(order)), yticklabels=list(reversed(order)), xlabel="Final training return (mean ± sample SD)", title="Craftax rollout/TBPTT-length response")
    for t, marker in markers.items():
        axis.scatter([], [], marker=marker, color="0.25", label=f"T={t}")
    axis.legend(ncol=4, loc="lower center", bbox_to_anchor=(.5, 1.01), frameon=False, fontsize=8)
    save(fig, output, "fig7_craftax_cross_t")


def craftax_throughput_paper(output: Path) -> None:
    data = rows("original_paper_upgrade/craftax_t64_seven_family_throughput.csv")
    order = ("PPO-GRU", "PPO-Mamba", "SdLG-Approx", "AGaLiTe", "PPO-GLA", "PPO-GTrXL", "SdLG-Exact")
    index = {row["family"]: row for row in data}
    selected = [index[name] for name in order]
    fig, axis = plt.subplots(figsize=(6.1, 3.3))
    axis.errorbar(range(len(selected)), [float(row["mean_sps"]) for row in selected], yerr=[float(row["sd_sps"]) if row["sd_sps"] else 0.0 for row in selected], fmt="o", capsize=3)
    axis.set(xticks=range(len(selected)), xticklabels=order, ylabel="Cumulative training-loop SPS", title="Craftax systems measurement")
    axis.tick_params(axis="x", rotation=23)
    axis.grid(axis="y", alpha=.25)
    save(fig, output, "fig4_craftax_throughput")


def ggs_paper(output: Path) -> None:
    data = rows("ggs_endpoint_paired_difference.csv")
    low = np.asarray([float(row["alpha_0_1_eval_return"]) for row in data])
    high = np.asarray([float(row["alpha_1_0_eval_return"]) for row in data])
    differences = low - high
    bootstrap = np.asarray([np.mean(differences[list(indices)]) for indices in itertools.product(range(len(differences)), repeat=len(differences))])
    ci_low, ci_high = np.quantile(bootstrap, (.025, .975))
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.7), gridspec_kw={"width_ratios": (1.25, .9)})
    for full, attenuated in zip(high, low):
        axes[0].plot((0, 1), (full, attenuated), color="0.65", marker="o", markersize=3)
    axes[0].scatter((0, 1), (np.mean(high), np.mean(low)), marker="D", zorder=3, label="Condition mean")
    axes[0].set(xticks=(0, 1), xticklabels=(r"Full\n$\alpha=1.0$", r"Attenuated\n$\alpha=0.1$"), ylabel="Endpoint evaluation return", title="(a) Matched seed pairs")
    axes[0].legend(frameon=False, fontsize=7)
    estimate = np.mean(differences)
    axes[1].axvline(0, color="0.5", linestyle=":")
    axes[1].errorbar(estimate, 0, xerr=[[estimate-ci_low], [ci_high-estimate]], fmt="D", capsize=3)
    axes[1].set(yticks=(0,), yticklabels=(r"$\alpha=0.1 - \alpha=1.0$",), xlabel="Paired return difference", title="(b) Mean and 95% CI")
    save(fig, output, "gradient_shielding_paired")


def ggs_appendix(output: Path) -> None:
    """Render the two appendix GGS telemetry panels from the retained raw log."""
    data = data_rows("legacy-records/diagnostics/architecture/gradient_shielding/gradient_shielding.csv")
    conditions = (("SdLGHybrid-GGS-0.1", r"Attenuated $\alpha=0.1$"), ("SdLGHybrid-GGS-1.0", r"Full $\alpha=1.0$"))

    def series(condition: str, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        grouped: dict[float, list[float]] = {}
        for row in data:
            if row["Architecture"] != condition or not row[metric]:
                continue
            grouped.setdefault(float(row["Total_Steps"]), []).append(float(row[metric]))
        keys = sorted(grouped)
        steps = np.asarray(keys) / 1e6
        means = np.asarray([np.mean(grouped[key]) for key in keys])
        sems = np.asarray([np.std(grouped[key], ddof=1) / np.sqrt(len(grouped[key])) if len(grouped[key]) > 1 else 0.0 for key in keys])
        return steps, means, sems

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for condition, label in conditions:
        for axis, metric, ylabel in zip(axes, ("Eval_Return", "Grad_Cosine_Sim"), ("Evaluation return", "Probe-gradient cosine")):
            steps, mean, sem = series(condition, metric)
            axis.plot(steps, mean, label=label)
            axis.fill_between(steps, mean-sem, mean+sem, alpha=.16)
            axis.set(xlabel="Environment steps (M)", ylabel=ylabel)
            axis.grid(axis="y", alpha=.25)
    axes[0].set_title("(a) Evaluation-return trajectory")
    axes[1].set_title("(b) Actor/critic probe-gradient cosine")
    axes[1].axhline(0, color="0.4", linestyle=":")
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, output, "gradient_shielding_trajectories")

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.6))
    for condition, label in conditions:
        for axis, metric, ylabel in zip(axes, ("Train_Entropy", "Train_VLoss_Var", "SPS"), ("Policy entropy", "Value-loss variance", "Training-loop SPS")):
            steps, mean, sem = series(condition, metric)
            axis.plot(steps, mean, label=label)
            axis.fill_between(steps, mean-sem, mean+sem, alpha=.16)
            axis.set(xlabel="Environment steps (M)", ylabel=ylabel)
            axis.grid(axis="y", alpha=.25)
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, output, "gradient_shielding_auxiliary")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tmaze(args.output_dir)
    minigrid(args.output_dir)
    craftax(args.output_dir)
    ggs(args.output_dir)
    capacity(args.output_dir)
    tmaze_paper(args.output_dir)
    minigrid_paper(args.output_dir)
    capacity_paper(args.output_dir)
    craftax_cross_t_paper(args.output_dir)
    craftax_throughput_paper(args.output_dir)
    ggs_paper(args.output_dir)
    ggs_appendix(args.output_dir)


if __name__ == "__main__":
    main()
