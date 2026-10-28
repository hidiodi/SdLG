"""Validate the fixed historical GGS endpoint gate before a fresh sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def endpoint(data: pd.DataFrame, architecture: str, seed: int) -> float:
    rows = data[(data["Architecture"] == architecture) & (data["Seed"] == seed)].dropna(subset=["Eval_Return"])
    if rows.empty:
        raise ValueError(f"No endpoint for {architecture}, seed {seed}")
    return float(rows.sort_values("Total_Steps").iloc[-1]["Eval_Return"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--rerun", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-absolute-difference", type=float, default=0.5)
    args = parser.parse_args()
    historical = pd.read_csv(args.historical)
    rerun = pd.read_csv(args.rerun)
    comparisons = []
    for alpha in (0.1, 1.0):
        # The accepted historical runner formatted 1.0 with one decimal;
        # the sweep runner intentionally uses compact numeric labels.  Compare
        # the same numeric condition, not its incidental CSV string spelling.
        historical_label = f"SdLGHybrid-GGS-{alpha:.1f}"
        rerun_label = f"SdLGHybrid-GGS-{alpha:g}"
        old = endpoint(historical, historical_label, args.seed)
        new = endpoint(rerun, rerun_label, args.seed)
        comparisons.append({"alpha": alpha, "seed": args.seed, "historical_endpoint": old,
                            "rerun_endpoint": new, "historical_label": historical_label,
                            "rerun_label": rerun_label, "absolute_difference": abs(new - old),
                            "pass": abs(new - old) <= args.max_absolute_difference})
    payload = {"threshold": args.max_absolute_difference, "comparisons": comparisons,
               "pass": all(row["pass"] for row in comparisons)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    if not payload["pass"]:
        raise SystemExit("GGS reproduction gate failed endpoint tolerance")


if __name__ == "__main__":
    main()
