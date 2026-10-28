#!/usr/bin/env python3
"""Verify the compact records for Appendix H.2's joint execution ablation.

The main release keeps original execution as its default because that is the
implementation behind the paper's primary results. The tracked patch records
the modified implementation used only for this sensitivity analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/studies/execution_semantics_ablation.json"


def read(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="") as handle:
        return list(csv.DictReader(handle))


def verify() -> None:
    config = json.loads(CONFIG.read_text())
    patch = ROOT / config["implementation_patch"]
    if not patch.is_file() or "new_swa_k = jnp.concatenate" not in patch.read_text():
        raise AssertionError("Execution-semantics patch is absent or malformed")
    endpoints = read("data/paper-records/execution_semantics_ablation/tmaze_seed_endpoints.csv")
    if len(endpoints) != 60:
        raise AssertionError(f"Expected 60 T-Maze seed endpoints, got {len(endpoints)}")
    for model in config["tmaze"]["factorial_models"] + config["tmaze"]["matched_models"]:
        count = sum(row["model"] == model for row in endpoints)
        if count != 10:
            raise AssertionError(f"{model}: expected 10 seed endpoints, got {count}")
    matched = read("data/paper-records/execution_semantics_ablation/matched_contrasts.csv")
    h200 = next(row for row in matched if int(row["horizon"]) == 200)
    if not np.isclose(float(h200["estimate"]), 0.1015625):
        raise AssertionError("Modified matched H=200 contrast does not match Appendix H.2")
    doorkey = read("data/paper-records/execution_semantics_ablation/doorkey_exact_seed_endpoints.csv")
    if len(doorkey) != 5 or any(row["stop_reason"] != "step_cap" for row in doorkey):
        raise AssertionError("Modified DoorKey record must contain five cap-terminated Exact runs")
    mean_return = np.mean([float(row["eval_mean_return"]) for row in doorkey])
    if not np.isclose(mean_return, 0.250554, atol=5e-7):
        raise AssertionError("Modified DoorKey mean return does not match Appendix H.2")
    print("PASS Appendix H.2: 60 T-Maze and 5 DoorKey modified-execution seed records")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-records", action="store_true")
    args = parser.parse_args()
    if args.verify_records:
        verify()
    else:
        raise SystemExit("Use --verify-records; full reruns apply patches/execution_semantics_ablation.patch to an isolated source copy.")


if __name__ == "__main__":
    main()
