#!/usr/bin/env python3
"""
Phase 1: Editable Memory Diagnostic Benchmark Suite
=====================================================
Benchmarks Additive GLA vs Delta GLA vs ADAM (Gated Delta) vs PPO-GRU
on POPGym ConcentrationEasy, RepeatPreviousEasy, and CountRecallEasy.
"""

import os
import sys
import argparse
import subprocess

current_file_path = os.path.abspath(__file__)
project_root = current_file_path
while not os.path.exists(os.path.join(project_root, "src")):
    project_root = os.path.dirname(project_root)
    if project_root == "/":
        break
if project_root not in sys.path:
    sys.path.append(project_root)

PYTHON_EXE = [os.path.join(project_root, ".venv/bin/python"), "-u"]

def run_phase1_benchmark(envs, models, seeds):
    print("=" * 75)
    print("PHASE 1: EDITABLE ASSOCIATIVE MEMORY BENCHMARK (POPGYM)")
    print(f"Environments: {envs}")
    print(f"Models: {models}")
    print(f"Seeds: {seeds}")
    print("=" * 75)

    for env in envs:
        for model in models:
            for seed in seeds:
                cmd = PYTHON_EXE + [
                    os.path.join(project_root, "scripts/diagnostics/exact_sdlg/popgym_benchmark.py"),
                    "--env", env,
                    "--model", model,
                    "--seed", str(seed),
                ]
                print(f"\n▶ Running Phase 1 Experiment: {env} | {model} | Seed {seed}")
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"❌ Run failed: {env} {model} seed {seed}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 Editable Memory Benchmark")
    parser.add_argument("--envs", nargs="+", default=["ConcentrationEasy", "RepeatPreviousEasy", "CountRecallEasy"], help="Environments to benchmark")
    parser.add_argument(
        "--models", nargs="+",
        default=[
            "SdLG-Exact", "SdLG-Delta", "SdLG-ADAM",
            "SdLG-Approx", "SdLG-Approx-Delta", "SdLG-Approx-ADAM",
            "PPO-GRU", "PPO-GLA", "PPO-Mamba"
        ],
        help="Models to benchmark"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 54, 99], help="Seeds to evaluate")
    args = parser.parse_args()

    run_phase1_benchmark(args.envs, args.models, args.seeds)
