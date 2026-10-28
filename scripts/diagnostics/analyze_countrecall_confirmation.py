"""Analyze the fixed-endpoint CountRecall confirmation matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = ("SdLG-Exact", "SdLG-Approx", "PPO-GRU", "PPO-GLA")
N_BOOT = 200_000


def ci(samples: np.ndarray) -> tuple[float, float]:
    return tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return values[rng.integers(0, len(values), size=(N_BOOT, len(values)))].mean(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.input)
    endpoint = data[data["Endpoint"].astype(str).str.lower() == "true"].copy()
    if endpoint.duplicated(["Model", "Seed"]).any():
        raise ValueError("Duplicate CountRecall endpoints")
    counts = endpoint.groupby("Model")["Seed"].nunique().to_dict()
    if counts != {model: 5 for model in MODELS}:
        raise ValueError(f"Expected five completed seeds per model; got {counts}")
    rng = np.random.default_rng(20260920)
    summary, values = [], {}
    for model in MODELS:
        rows = endpoint[endpoint["Model"] == model]
        scores = rows["Eval_Normalized_Return"].to_numpy(float)
        values[model] = scores
        low, high = ci(bootstrap_mean(scores, rng))
        summary.append({
            "model": model, "n": len(scores), "mean": scores.mean(), "sd": scores.std(ddof=1),
            "median": np.median(scores), "bootstrap_ci_low": low, "bootstrap_ci_high": high,
            "trainable_parameters": int(rows["Trainable_Parameters"].iloc[0]),
            "state_elements_per_env": int(rows["State_Elements_Per_Env"].iloc[0]),
        })
    contrasts = []
    for model in ("SdLG-Approx", "PPO-GRU", "PPO-GLA"):
        draws = bootstrap_mean(values["SdLG-Exact"], rng) - bootstrap_mean(values[model], rng)
        low, high = ci(draws)
        contrasts.append({"contrast": f"SdLG-Exact_minus_{model}", "estimate": draws.mean(),
                          "bootstrap_ci_low": low, "bootstrap_ci_high": high,
                          "design": "independent_seed_level_runs", "bootstrap_draws": N_BOOT})
    endpoint.sort_values(["Model", "Seed"]).to_csv(args.output_dir / "countrecall_seed_endpoints.csv", index=False)
    pd.DataFrame(summary).to_csv(args.output_dir / "countrecall_summary.csv", index=False)
    pd.DataFrame(contrasts).to_csv(args.output_dir / "countrecall_contrasts.csv", index=False)

    summary_frame = pd.DataFrame(summary).set_index("model").loc[list(MODELS)].reset_index()
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    colors = {"SdLG-Exact": "#2166ac", "SdLG-Approx": "#b2182b", "PPO-GRU": "#4d9221", "PPO-GLA": "#8073ac"}
    labels = {"SdLG-Exact": "Exact", "SdLG-Approx": "Approximate", "PPO-GRU": "GRU", "PPO-GLA": "GLA"}
    for x, model in enumerate(MODELS):
        scores = values[model]
        ax.scatter(np.full(len(scores), x) + np.linspace(-.10, .10, len(scores)), scores,
                   color=colors[model], edgecolor="white", linewidth=.45, zorder=3)
        row = summary_frame[summary_frame["model"] == model].iloc[0]
        ax.errorbar(x, row["mean"], yerr=[[row["mean"] - row["bootstrap_ci_low"]],
                                           [row["bootstrap_ci_high"] - row["mean"]]],
                    fmt="D", color="black", mfc="white", ms=5.5, capsize=3.5, lw=1.3, zorder=4)
    ax.set_xticks(range(len(MODELS)), [labels[model] for model in MODELS])
    ax.set_ylabel("Fixed-endpoint normalized return")
    ax.set_ylim(-.03, 1.03)
    ax.set_title("CountRecall confirmation")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(args.plot_dir / "fig_countrecall_confirmation.pdf", bbox_inches="tight")
    fig.savefig(args.plot_dir / "fig_countrecall_confirmation.png", dpi=240, bbox_inches="tight")


if __name__ == "__main__":
    main()
