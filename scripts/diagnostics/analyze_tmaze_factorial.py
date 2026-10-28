"""Seed-level reanalysis of the accepted T-Maze studies.

The experimental unit is a seed-level training run.  Shared integer seed labels
are not treated as paired observations because the four conditions do not form
a verified coupled trajectory design.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("papers/writing/generated_artifacts/tmaze_rank_rope_zero_shot.csv")
DEFAULT_MATCHED_INPUT = Path("scripts/diagnostics/exact_sdlg/unified_tmaze_ablations_zeroshot.csv")
DEFAULT_OUTPUT = Path("papers/writing/generated_artifacts/revision_2026")
MODELS = ("Exact_NoRoPE", "Exact_RoPE", "Approx_NoRoPE", "Approx_RoPE")
MATCHED_MODELS = ("Matched_Exact_NoRoPE", "Matched_Approx_NoRoPE")
GATE_BIAS_MODELS = ("GateBias_Exact_bg2", "GateBias_Exact_bg0")


def percentile_interval(samples: np.ndarray) -> tuple[float, float]:
    return tuple(np.quantile(samples, [0.025, 0.975]))


def iqm(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    return float(np.mean(values[int(np.floor(0.25 * len(values))) : int(np.ceil(0.75 * len(values)))]))


def bootstrap_contrasts(
    cells: dict[str, np.ndarray], rng: np.random.Generator, n_boot: int
) -> dict[str, np.ndarray]:
    draws = {
        name: values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
        for name, values in cells.items()
    }
    exact = 0.5 * (draws["Exact_NoRoPE"] + draws["Exact_RoPE"])
    approx = 0.5 * (draws["Approx_NoRoPE"] + draws["Approx_RoPE"])
    no_rope = 0.5 * (draws["Exact_NoRoPE"] + draws["Approx_NoRoPE"])
    rope = 0.5 * (draws["Exact_RoPE"] + draws["Approx_RoPE"])
    exact_phase = draws["Exact_NoRoPE"] - draws["Exact_RoPE"]
    approx_phase = draws["Approx_NoRoPE"] - draws["Approx_RoPE"]
    return {
        "execution_exact_minus_approx": exact - approx,
        "phase_norope_minus_rope": no_rope - rope,
        "interaction_phase_difference_exact_minus_approx": exact_phase - approx_phase,
        "phase_effect_within_exact": exact_phase,
        "phase_effect_within_approx": approx_phase,
    }


def point_contrasts(means: dict[str, float]) -> dict[str, float]:
    exact = 0.5 * (means["Exact_NoRoPE"] + means["Exact_RoPE"])
    approx = 0.5 * (means["Approx_NoRoPE"] + means["Approx_RoPE"])
    no_rope = 0.5 * (means["Exact_NoRoPE"] + means["Approx_NoRoPE"])
    rope = 0.5 * (means["Exact_RoPE"] + means["Approx_RoPE"])
    exact_phase = means["Exact_NoRoPE"] - means["Exact_RoPE"]
    approx_phase = means["Approx_NoRoPE"] - means["Approx_RoPE"]
    return {
        "execution_exact_minus_approx": exact - approx,
        "phase_norope_minus_rope": no_rope - rope,
        "interaction_phase_difference_exact_minus_approx": exact_phase - approx_phase,
        "phase_effect_within_exact": exact_phase,
        "phase_effect_within_approx": approx_phase,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-draws", type=int, default=200_000)
    parser.add_argument("--random-seed", type=int, default=20260919)
    parser.add_argument("--study", choices=("factorial", "matched", "gate-bias"), default="factorial")
    parser.add_argument("--append", action="store_true")
    parser.add_argument(
        "--overlap-audit",
        action="store_true",
        help="Write seed-level historical-versus-matched outcome comparisons; does not infer run identity.",
    )
    args = parser.parse_args()

    input_path = args.input
    # The frozen historical factorial derivative deliberately retains its
    # original source file.  The later matched study lives in the shared raw
    # T-Maze artifact and must never be searched for in that historical export.
    if args.study in {"matched", "gate-bias"} and input_path == DEFAULT_INPUT:
        input_path = DEFAULT_MATCHED_INPUT
    data = pd.read_csv(input_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = {
        "factorial": MODELS,
        "matched": MATCHED_MODELS,
        "gate-bias": GATE_BIAS_MODELS,
    }[args.study]
    missing = set(models) - set(data["model"])
    if missing:
        raise ValueError(f"Missing factorial cells: {sorted(missing)}")

    rng = np.random.default_rng(args.random_seed)
    cell_rows: list[dict[str, object]] = []
    contrast_rows: list[dict[str, object]] = []
    for horizon in (100, 150, 200):
        column = f"val_H{horizon}"
        cells = {
            model: data.loc[data["model"] == model, column].dropna().to_numpy(float)
            for model in models
        }
        if any(len(values) != 10 for values in cells.values()):
            counts = {name: len(values) for name, values in cells.items()}
            raise ValueError(f"Expected ten seeds per cell at H={horizon}; got {counts}")

        means = {name: float(values.mean()) for name, values in cells.items()}
        for model, values in cells.items():
            boot = values[rng.integers(0, len(values), size=(args.bootstrap_draws, len(values)))].mean(axis=1)
            low, high = percentile_interval(boot)
            cell_rows.append(
                {
                    "study": args.study,
                    "horizon": horizon,
                    "model": model,
                    "n": len(values),
                    "mean": values.mean(),
                    "sd": values.std(ddof=1),
                    "median": np.median(values),
                    "iqm": iqm(values),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                }
            )

        if args.study == "factorial":
            estimates = point_contrasts(means)
            bootstraps = bootstrap_contrasts(cells, rng, args.bootstrap_draws)
        else:
            reference, treatment, contrast_name = (
                ("Matched_Exact_NoRoPE", "Matched_Approx_NoRoPE", "matched_execution_exact_minus_approx")
                if args.study == "matched"
                else ("GateBias_Exact_bg2", "GateBias_Exact_bg0", "gate_bias_plus2_minus_zero")
            )
            exact = cells[reference]
            approx = cells[treatment]
            exact_draws = exact[rng.integers(0, len(exact), size=(args.bootstrap_draws, len(exact)))].mean(axis=1)
            approx_draws = approx[rng.integers(0, len(approx), size=(args.bootstrap_draws, len(approx)))].mean(axis=1)
            estimates = {contrast_name: means[reference] - means[treatment]}
            bootstraps = {contrast_name: exact_draws - approx_draws}
        for contrast, estimate in estimates.items():
            low, high = percentile_interval(bootstraps[contrast])
            contrast_rows.append(
                {
                    "study": args.study,
                    "horizon": horizon,
                    "contrast": contrast,
                    "estimate": estimate,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "bootstrap_draws": args.bootstrap_draws,
                    "design": "independent_cells",
                }
            )

    outputs = {
        "tmaze_factorial_cell_summary.csv": pd.DataFrame(cell_rows),
        "tmaze_factorial_contrasts.csv": pd.DataFrame(contrast_rows),
        "tmaze_factorial_seed_outcomes.csv": data.loc[
            data["model"].isin(models), ["H_train", "model", "seed", "val_H100", "val_H150", "val_H200"]
        ].assign(study=args.study),
    }
    for name, frame in outputs.items():
        path = args.output_dir / name
        if args.append and path.exists():
            previous = pd.read_csv(path)
            if "study" not in previous:
                previous.insert(0, "study", "factorial")
            frame = pd.concat([previous, frame], ignore_index=True, sort=False)
        frame.to_csv(path, index=False)

    if args.overlap_audit:
        audit_rows: list[dict[str, object]] = []
        for historical, matched in (("Exact_NoRoPE", "Matched_Exact_NoRoPE"),
                                    ("Approx_NoRoPE", "Matched_Approx_NoRoPE")):
            historical_rows = data.loc[data["model"] == historical].set_index("seed")
            matched_rows = data.loc[data["model"] == matched].set_index("seed")
            for seed in sorted(set(historical_rows.index) & set(matched_rows.index)):
                for horizon in (100, 150, 200):
                    column = f"val_H{horizon}"
                    old = float(historical_rows.loc[seed, column])
                    new = float(matched_rows.loc[seed, column])
                    audit_rows.append({
                        "historical_model": historical,
                        "matched_model": matched,
                        "seed": seed,
                        "horizon": horizon,
                        "historical_return": old,
                        "matched_return": new,
                        "absolute_difference": abs(old - new),
                        "bitwise_equal_value": old == new,
                        "interpretation": "outcome comparison only; equality does not establish shared trajectories",
                    })
        pd.DataFrame(audit_rows).to_csv(args.output_dir / "tmaze_matched_overlap_audit.csv", index=False)


if __name__ == "__main__":
    main()
