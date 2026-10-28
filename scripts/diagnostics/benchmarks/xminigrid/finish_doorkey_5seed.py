"""Finish the missing DoorKey-8x8 five-seed cells in fresh processes."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve()
while not (PROJECT_ROOT / "src").exists():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise RuntimeError("Could not locate repository root")
    PROJECT_ROOT = PROJECT_ROOT.parent

BENCHMARK = PROJECT_ROOT / "scripts/diagnostics/benchmarks/xminigrid/doorkey_baseline_benchmark.py"
RESULTS_DIR = PROJECT_ROOT / "results/minigrid_doorkey_baselines_2026"
SUMMARY = RESULTS_DIR / "run_summary.csv"
ENVIRONMENT = "MiniGrid-DoorKey-8x8"
TARGETS = (
    ("SdLG-Exact", 456),
    ("PPO-GRU", 123),
    ("PPO-GRU", 456),
    ("PPO-GTrXL", 123),
    ("PPO-GTrXL", 456),
)


def completed() -> set[tuple[str, int]]:
    if not SUMMARY.exists():
        return set()
    with SUMMARY.open(newline="") as handle:
        return {
            (row["model"], int(row["seed"]))
            for row in csv.DictReader(handle)
            if row["environment"] == ENVIRONMENT and row["status"] == "complete"
        }


def main() -> int:
    for model, seed in TARGETS:
        if (model, seed) in completed():
            print(f"[supervisor skip] {model} seed {seed} already complete", flush=True)
            continue

        print(f"[supervisor start] {model} seed {seed}", flush=True)
        command = [
            sys.executable,
            str(BENCHMARK),
            "--envs",
            ENVIRONMENT,
            "--models",
            model,
            "--seeds",
            str(seed),
            "--total-steps",
            "20000000",
            "--num-envs",
            "512",
            "--log-interval",
            "10",
            "--eval-episodes",
            "256",
            "--results-dir",
            str(RESULTS_DIR),
            "--wandb-mode",
            "offline",
        ]
        result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if result.returncode != 0:
            print(
                f"[supervisor failed] {model} seed {seed}: exit {result.returncode}",
                flush=True,
            )
            return result.returncode
        if (model, seed) not in completed():
            print(f"[supervisor failed] no completed row for {model} seed {seed}", flush=True)
            return 2
        print(f"[supervisor done] {model} seed {seed}", flush=True)

    print("[supervisor complete] all five-seed DoorKey-8x8 cells are present", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
