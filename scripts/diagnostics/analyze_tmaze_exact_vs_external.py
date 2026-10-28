"""Independent comparison of matched Exact with the recovered T-Maze landscape.

Both inputs use the same Passive T-Maze training horizon, TBPTT length,
early-stop-or-cap rule, and evaluation horizons.  They use different seed sets
and experiment snapshots, so the analysis treats conditions as independent and
does not imply paired trajectories or single-mechanism attribution.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_BASELINES = Path("papers/writing/generated_artifacts/tmaze_external_landscape.csv")
DEFAULT_EXACT = Path(
    "papers/writing/generated_artifacts/revision_2026/tmaze_factorial_seed_outcomes.csv"
)
DEFAULT_OUTPUT = Path(
    "papers/writing/generated_artifacts/revision_2026/"
    "tmaze_exact_vs_external_contrasts.csv"
)
EXACT_MODEL = "Matched_Exact_NoRoPE"
BASELINE_ORDER = (
    "GRU",
    "LSTM",
    "Mamba",
    "LTI-Linear",
    "Markovian",
    "GLA",
    "GTrXL-64",
    "GTrXL-128",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--exact", type=Path, default=DEFAULT_EXACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-draws", type=int, default=200_000)
    parser.add_argument("--random-seed", type=int, default=20260919)
    args = parser.parse_args()

    baselines = pd.read_csv(args.baselines)
    exact_rows = pd.read_csv(args.exact)
    exact_rows = exact_rows.loc[exact_rows["model"] == EXACT_MODEL]

    if len(exact_rows) != 10:
        raise ValueError(f"Expected ten {EXACT_MODEL} rows; found {len(exact_rows)}")
    missing = set(BASELINE_ORDER) - set(baselines["family"])
    if missing:
        raise ValueError(f"Missing baseline families: {sorted(missing)}")

    rng = np.random.default_rng(args.random_seed)
    rows: list[dict[str, object]] = []
    for horizon in (100, 150, 200):
        exact = exact_rows[f"val_H{horizon}"].to_numpy(float)
        for family in BASELINE_ORDER:
            baseline = baselines.loc[baselines["family"] == family, f"H{horizon}"].to_numpy(float)
            if len(baseline) != 7:
                raise ValueError(f"Expected seven {family} rows; found {len(baseline)}")

            exact_boot = exact[
                rng.integers(0, len(exact), size=(args.bootstrap_draws, len(exact)))
            ].mean(axis=1)
            baseline_boot = baseline[
                rng.integers(0, len(baseline), size=(args.bootstrap_draws, len(baseline)))
            ].mean(axis=1)
            contrast_boot = exact_boot - baseline_boot
            ci_low, ci_high = np.quantile(contrast_boot, [0.025, 0.975])
            rows.append(
                {
                    "exact_model": EXACT_MODEL,
                    "baseline_model": family,
                    "horizon": horizon,
                    "n_exact": len(exact),
                    "n_baseline": len(baseline),
                    "exact_mean": exact.mean(),
                    "baseline_mean": baseline.mean(),
                    "estimate_exact_minus_baseline": exact.mean() - baseline.mean(),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "bootstrap_draws": args.bootstrap_draws,
                    "rng_seed": args.random_seed,
                    "design": "independent_cross_study_cells",
                    "multiplicity_adjustment": "none_pointwise_intervals",
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
