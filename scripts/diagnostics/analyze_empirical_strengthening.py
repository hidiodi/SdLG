"""Generate seed-level summaries and figures for the 2026 revision studies."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 10})

DEFAULT_DIR = Path("papers/writing/generated_artifacts/revision_2026")
DEFAULT_PLOTS = Path("papers/latex/cleanRewrite/plots/rebuilt")
RNG_SEED = 20260919
N_BOOT = 200_000


def interval(samples: np.ndarray) -> tuple[float, float]:
    return tuple(np.quantile(samples, [0.025, 0.975]))


def iqm(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    lo = int(np.floor(0.25 * len(values)))
    hi = int(np.ceil(0.75 * len(values)))
    return float(values[lo:hi].mean())


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return values[rng.integers(0, len(values), size=(N_BOOT, len(values)))].mean(axis=1)


def summarize(values: np.ndarray, rng: np.random.Generator) -> dict[str, float | int]:
    low, high = interval(bootstrap_mean(values, rng))
    return {
        "n": len(values),
        "mean": values.mean(),
        "sd": values.std(ddof=1),
        "median": np.median(values),
        "iqm": iqm(values),
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
    }


def analyze_minigrid(path: Path, output_dir: Path, plot_dir: Path, rng: np.random.Generator) -> None:
    data = pd.read_csv(path)
    endpoints = data.sort_values("Total_Steps").groupby(["Condition", "Seed"], as_index=False).tail(1)
    expected = {"SdLG-SWA", "SdLG-None"}
    if set(endpoints["Condition"]) != expected:
        raise ValueError(f"Unexpected MiniGrid conditions: {sorted(endpoints['Condition'].unique())}")
    if endpoints.duplicated(["Condition", "Seed"]).any():
        raise ValueError("MiniGrid contains duplicate condition/seed endpoints")

    summaries = []
    values: dict[str, np.ndarray] = {}
    for condition in ("SdLG-SWA", "SdLG-None"):
        vals = endpoints.loc[endpoints["Condition"] == condition, "Eval_Return"].to_numpy(float)
        values[condition] = vals
        summaries.append({"condition": condition, **summarize(vals, rng)})

    local, none = values["SdLG-SWA"], values["SdLG-None"]
    observed = local.mean() - none.mean()
    boot = bootstrap_mean(local, rng) - bootstrap_mean(none, rng)
    ci_low, ci_high = interval(boot)
    pooled = np.concatenate([local, none])
    permuted = np.empty(N_BOOT)
    for index in range(N_BOOT):
        shuffled = rng.permutation(pooled)
        permuted[index] = shuffled[: len(local)].mean() - shuffled[len(local) :].mean()
    p_value = (1 + np.sum(np.abs(permuted) >= abs(observed))) / (N_BOOT + 1)
    difference = pd.DataFrame(
        [{
            "estimand": "mean_endpoint_return_swa_enabled_minus_bypassed",
            "estimate": observed,
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "permutation_p_two_sided": p_value,
            "bootstrap_draws": N_BOOT,
            "permutation_draws": N_BOOT,
            "design": "independent_seed_level_runs",
        }]
    )
    endpoints.to_csv(output_dir / "minigrid_locality_seed_endpoints.csv", index=False)
    pd.DataFrame(summaries).to_csv(output_dir / "minigrid_locality_summary.csv", index=False)
    difference.to_csv(output_dir / "minigrid_locality_difference.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    palette = {"SdLG-SWA": "#0072B2", "SdLG-None": "#E69F00"}
    labels = {"SdLG-SWA": "SWA enabled", "SdLG-None": "SWA bypassed"}
    for x, condition in enumerate(("SdLG-SWA", "SdLG-None")):
        vals = values[condition]
        jitter = np.linspace(-0.11, 0.11, len(vals))
        ax.scatter(np.full(len(vals), x) + jitter, vals, s=29, alpha=0.82,
                   color=palette[condition], edgecolor="white", linewidth=0.45, zorder=3)
        summary = summaries[x]
        ax.errorbar(x, summary["mean"],
                    yerr=[[summary["mean"] - summary["bootstrap_ci_low"]],
                          [summary["bootstrap_ci_high"] - summary["mean"]]],
                    fmt="D", color="black", mfc="white", ms=6, capsize=4, lw=1.4, zorder=4)
    ax.set_xticks([0, 1], [labels["SdLG-SWA"], labels["SdLG-None"]])
    ax.set_ylabel("Fixed-endpoint evaluation return", fontsize=11)
    ax.set_ylim(-0.03, 1.06)
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(plot_dir / "fig5_minigrid_locality_expanded.pdf", bbox_inches="tight")
    fig.savefig(plot_dir / "fig5_minigrid_locality_expanded.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def alpha_from_label(label: str) -> float:
    return float(label.rsplit("-", 1)[-1])


def exact_sign_flip_p(differences: np.ndarray) -> float:
    observed = abs(differences.mean())
    means = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        means.append(abs(np.mean(differences * np.asarray(signs))))
    return float(np.mean(np.asarray(means) >= observed - 1e-15))


def analyze_ggs(path: Path, output_dir: Path, plot_dir: Path,
                rng: np.random.Generator) -> None:
    """Analyze the one coherent, newly run five-alpha sweep.

    Historical alpha=.1/1 runs are deliberately not merged here: they are
    retained only as reproduction-gate context, while every sweep cell has
    five fresh matched seed-level training runs from the same snapshot.
    """
    data = pd.read_csv(path)
    data["alpha"] = data["Architecture"].map(alpha_from_label)
    data.to_csv(output_dir / "ggs_alpha_sweep_raw_checked.csv", index=False)
    endpoint_rows = data.dropna(subset=["Eval_Return"]).sort_values("Total_Steps").groupby(
        ["alpha", "Seed"], as_index=False
    ).tail(1)
    seed_step_sets = data.groupby(["alpha", "Seed"])["Total_Steps"].agg(lambda x: frozenset(x))
    common_steps = sorted(set.intersection(*map(set, seed_step_sets)))
    if not common_steps:
        raise ValueError("No common probe checkpoints across the full alpha sweep")
    common_probe = data[data["Total_Steps"].isin(common_steps)]
    probe_seed = common_probe.groupby(["alpha", "Seed"], as_index=False).agg(
        probe_cosine_mean_common=("Grad_Cosine_Sim", "mean"),
        probe_cosine_final=("Grad_Cosine_Sim", "last"),
        probe_cosine_negative_fraction=("Grad_Cosine_Sim", lambda x: float(np.mean(np.asarray(x) < 0))),
        common_probe_checkpoints=("Grad_Cosine_Sim", "count"),
    )
    dense_counts = data.groupby(["alpha", "Seed"], as_index=False).agg(
        recorded_probe_checkpoints=("Grad_Cosine_Sim", "count")
    )
    probe_seed = probe_seed.merge(dense_counts, on=["alpha", "Seed"], validate="one_to_one")
    endpoint_rows = endpoint_rows.merge(probe_seed, on=["alpha", "Seed"], how="left", validate="one_to_one")

    alphas = sorted(endpoint_rows["alpha"].unique())
    expected = [0.0, 0.1, 0.25, 0.5, 1.0]
    if alphas != expected:
        raise ValueError(f"Expected alpha sweep {expected}; got {alphas}")
    expected_counts = {0.0: 5, 0.1: 5, 0.25: 5, 0.5: 5, 1.0: 5}
    actual_counts = endpoint_rows.groupby("alpha")["Seed"].nunique().to_dict()
    if actual_counts != expected_counts:
        raise ValueError(f"Expected seed counts {expected_counts}; got {actual_counts}")

    summary_rows = []
    probe_rows = []
    for alpha in alphas:
        cell = endpoint_rows.loc[endpoint_rows["alpha"] == alpha]
        summary_rows.append({"alpha": alpha, **summarize(cell["Eval_Return"].to_numpy(float), rng)})
        for metric in ("probe_cosine_mean_common", "probe_cosine_final", "probe_cosine_negative_fraction"):
            probe_rows.append({"alpha": alpha, "metric": metric,
                               **summarize(cell[metric].to_numpy(float), rng)})

    paired = endpoint_rows.pivot(index="Seed", columns="alpha", values="Eval_Return").sort_index()
    differences = paired[0.1].to_numpy() - paired[1.0].to_numpy()
    paired_boot = differences[rng.integers(0, len(differences), size=(N_BOOT, len(differences)))].mean(axis=1)
    low, high = interval(paired_boot)
    contrast = pd.DataFrame([{
        "contrast": "alpha_0.1_minus_alpha_1.0_fresh_sweep",
        "estimate": differences.mean(),
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
        "exact_sign_flip_p_two_sided": exact_sign_flip_p(differences),
        "design": "matched_common_random_numbers",
        "bootstrap_draws": N_BOOT,
    }])
    adjacent_rows = []
    for lower, upper in zip(alphas[:-1], alphas[1:]):
        common = paired[[lower, upper]].dropna()
        diff = common[upper].to_numpy(float) - common[lower].to_numpy(float)
        boot = diff[rng.integers(0, len(diff), size=(N_BOOT, len(diff)))].mean(axis=1)
        adj_low, adj_high = interval(boot)
        adjacent_rows.append({
            "contrast": f"alpha_{upper:g}_minus_alpha_{lower:g}",
            "n_pairs": len(diff),
            "estimate": diff.mean(),
            "bootstrap_ci_low": adj_low,
            "bootstrap_ci_high": adj_high,
            "design": "matched_common_random_numbers",
        })
    endpoint_rows.to_csv(output_dir / "ggs_alpha_sweep_seed_endpoints.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_dir / "ggs_alpha_sweep_endpoint_summary.csv", index=False)
    pd.DataFrame(probe_rows).to_csv(output_dir / "ggs_alpha_sweep_probe_summary.csv", index=False)
    contrast.to_csv(output_dir / "ggs_alpha_sweep_accepted_contrast.csv", index=False)
    pd.DataFrame(adjacent_rows).to_csv(output_dir / "ggs_alpha_sweep_adjacent_contrasts.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.05))
    x = np.asarray(alphas)
    for seed, row in paired.iterrows():
        axes[0].plot(x, row[x].to_numpy(float), color="#8c8c8c", alpha=0.42, lw=0.8, marker="o", ms=2.8)
    means = np.asarray([r["mean"] for r in summary_rows])
    lows = np.asarray([r["bootstrap_ci_low"] for r in summary_rows])
    highs = np.asarray([r["bootstrap_ci_high"] for r in summary_rows])
    axes[0].errorbar(x, means, yerr=[means - lows, highs - means], color="#2166ac",
                     marker="D", ms=4.5, capsize=3, lw=1.5, label="Mean (95% bootstrap CI)")
    axes[0].set_ylabel("Fixed-endpoint evaluation return")
    axes[0].set_title("Endpoint behavior")
    axes[0].legend(frameon=False, fontsize=7)

    probe_pivot = endpoint_rows.pivot(index="Seed", columns="alpha", values="probe_cosine_mean_common").sort_index()
    for seed, row in probe_pivot.iterrows():
        axes[1].plot(x, row[x].to_numpy(float), color="#8c8c8c", alpha=0.42, lw=0.8, marker="o", ms=2.8)
    probe_summary = pd.DataFrame(probe_rows)
    mean_probe = probe_summary[probe_summary["metric"] == "probe_cosine_mean_common"].sort_values("alpha")
    axes[1].errorbar(x, mean_probe["mean"],
                     yerr=[mean_probe["mean"] - mean_probe["bootstrap_ci_low"],
                           mean_probe["bootstrap_ci_high"] - mean_probe["mean"]],
                     color="#b2182b", marker="D", ms=4.5, capsize=3, lw=1.5)
    axes[1].axhline(0, color="black", lw=0.7, alpha=0.5)
    axes[1].set_ylabel("Mean probe-gradient cosine within seed")
    axes[1].set_title("Actor/Critic Probe-Gradient Cosine")
    for ax in axes:
        ax.set_xlabel(r"Actor-gradient scale $\alpha$")
        ax.set_xticks(x)
        ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(plot_dir / "fig_ggs_alpha_sweep.pdf", bbox_inches="tight")
    fig.savefig(plot_dir / "fig_ggs_alpha_sweep.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOTS)
    parser.add_argument("--study", choices=("minigrid", "ggs", "all"), default="all")
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    if args.study in ("minigrid", "all"):
        analyze_minigrid(args.artifact_dir / "minigrid_locality_raw.csv", args.artifact_dir, args.plot_dir, rng)
    if args.study in ("ggs", "all"):
        analyze_ggs(args.artifact_dir / "ggs_alpha_sweep_raw.csv", args.artifact_dir, args.plot_dir, rng)


if __name__ == "__main__":
    main()
