# scripts/experiments/popgym_benchmark.py

import os
import sys

# Delegate directly to scripts/diagnostics/exact_sdlg/popgym_benchmark.py
current_file_path = os.path.abspath(__file__)
project_root = current_file_path
while not os.path.exists(os.path.join(project_root, "src")):
    project_root = os.path.dirname(project_root)
    if project_root == "/":
        break
if project_root not in sys.path:
    sys.path.append(project_root)

from scripts.diagnostics.exact_sdlg.popgym_benchmark import run_benchmark, DEFAULT_SEEDS, POPGYM_ENVS, MODELS
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run 100,000 steps dry-run validation.")
    parser.add_argument("--total-steps", type=int, default=10_000_000, help="Total environment steps per run.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, help="Random seeds.")
    parser.add_argument("--envs", nargs="+", default=POPGYM_ENVS, help="Environments to evaluate.")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()), help="Models to evaluate.")
    args = parser.parse_args()

    run_benchmark(
        total_steps=args.total_steps,
        dry_run=args.dry_run,
        seeds=args.seeds,
        env_list=args.envs,
        model_list=args.models,
    )
