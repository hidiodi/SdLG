"""Record endpoint-level historical-snapshot reproduction comparisons.

This intentionally records numeric agreement rather than using qualitative
phrases such as "reproduced closely".  It does not decide whether a gate
passes: that decision remains an explicit, reviewable interpretation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def final_endpoints(frame: pd.DataFrame, condition: str, seed_column: str,
                    condition_column: str, endpoint_column: str) -> pd.DataFrame:
    filtered = frame.loc[frame[condition_column].astype(str) == condition].copy()
    filtered = filtered.dropna(subset=[endpoint_column])
    if "Total_Steps" in filtered:
        filtered = filtered.sort_values("Total_Steps")
    return filtered.groupby(seed_column, as_index=False).tail(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--rerun", type=Path, required=True)
    parser.add_argument("--historical-condition-column", required=True)
    parser.add_argument("--rerun-condition-column", required=True)
    parser.add_argument("--historical-condition", required=True)
    parser.add_argument("--rerun-condition", required=True)
    parser.add_argument("--seed-column", default="seed")
    parser.add_argument("--historical-endpoint-column", default="final_return")
    parser.add_argument("--rerun-endpoint-column", default="Eval_Return")
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--environment-inventory", type=Path, required=True)
    parser.add_argument("--evaluation-protocol", required=True)
    parser.add_argument("--known-nondeterminism", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    historical = pd.read_csv(args.historical)
    rerun = pd.read_csv(args.rerun)
    seeds = [int(value) for value in args.seeds.split(",") if value]
    historical_endpoints = final_endpoints(
        historical, args.historical_condition, args.seed_column,
        args.historical_condition_column, args.historical_endpoint_column,
    ).set_index(args.seed_column)
    rerun_endpoints = final_endpoints(
        rerun, args.rerun_condition, "Seed",
        args.rerun_condition_column, args.rerun_endpoint_column,
    ).set_index("Seed")
    inventory = json.loads(args.environment_inventory.read_text())
    rows = []
    for seed in seeds:
        old = float(historical_endpoints.loc[seed, args.historical_endpoint_column])
        new = float(rerun_endpoints.loc[seed, args.rerun_endpoint_column])
        rows.append({
            "seed": seed,
            "historical_condition": args.historical_condition,
            "rerun_condition": args.rerun_condition,
            "historical_endpoint": old,
            "rerun_endpoint": new,
            "absolute_difference": abs(new - old),
            "source_commit": args.source_commit,
            "runner_sha256": args.runner_sha256,
            "environment_inventory": str(args.environment_inventory),
            "package_versions": json.dumps(inventory["packages"], sort_keys=True),
            "evaluation_protocol": args.evaluation_protocol,
            "known_nondeterminism": args.known_nondeterminism,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    if args.append and args.output.exists():
        result = pd.concat([pd.read_csv(args.output), result], ignore_index=True)
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
