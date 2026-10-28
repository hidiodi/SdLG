#!/usr/bin/env python3
"""
🏆 FULL 15M-STEP MECHANISTIC RL MATRIX SWEEP
========================================================================
Executes full-scale multi-seed 15M-step PPO evaluation across POPGym suite
for our 7-model mechanistic matrix:

Models (7):
1. SdLG-Exact             (Additive baseline, d_k=16, d_v=16)
2. SdLG-Delta             (Delta editing baseline, d_k=16, d_v=16)
3. SdLG-ADAM              (Erasable baseline, d_k=16, d_v=16)
4. SdLG-Delta-dk32        (Asymmetric capacity, d_k=32, d_v=16)
5. SdLG-Delta-dk64        (Asymmetric capacity, d_k=64, d_v=16)
6. SdLG-ADAM-dk64         (Composite: Erasable + Asymmetric, d_k=64, d_v=16)
7. SdLG-ADAM-Matched-dk64 (Parameter-matched control, d_k=64, d_v=16)

Environments:
- ConcentrationEasy  (Primary signal: Addressing Capacity d_k + Delta editing)
- RepeatPreviousEasy (Primary signal: Retention & Editing)
- CountRecallEasy    (Primary signal: Storage capacity)
- AutoencodeEasy     (Primary signal: Memory recall)
"""

import os
import sys
import subprocess
import time

current_file_path = os.path.abspath(__file__)
project_root = current_file_path
while not os.path.exists(os.path.join(project_root, "src")):
    project_root = os.path.dirname(project_root)
    if project_root == "/":
        break
if project_root not in sys.path:
    sys.path.append(project_root)

TARGET_ENVS = [
    "ConcentrationEasy",
    "RepeatPreviousEasy",
    "CountRecallEasy",
    "AutoencodeEasy"
]

TARGET_MODELS = [
    "SdLG-Delta",
    "SdLG-ADAM",
    "SdLG-Delta-dk32",
    "SdLG-Delta-dk64",
    "SdLG-ADAM-dk64",
    "SdLG-ADAM-Matched-dk64",
]


SEEDS = [42]
TOTAL_TIMESTEPS = 15_000_000


def main():
    print("=" * 80)
    print("🏆 LAUNCHING FULL 15M-STEP MECHANISTIC RL MATRIX SWEEP")
    print(f"Environments ({len(TARGET_ENVS)}): {TARGET_ENVS}")
    print(f"Models ({len(TARGET_MODELS)}): {TARGET_MODELS}")
    print(f"Seeds ({len(SEEDS)}): {SEEDS}")
    print(f"Total Steps per Run: {TOTAL_TIMESTEPS:,}")
    print("=" * 80, flush=True)

    python_bin = os.path.join(project_root, ".venv", "bin", "python")
    bench_script = os.path.join(project_root, "scripts", "diagnostics", "exact_sdlg", "popgym_benchmark.py")

    total_runs = len(TARGET_ENVS) * len(TARGET_MODELS) * len(SEEDS)
    completed = 0
    start_time = time.time()

    for env in TARGET_ENVS:
        print(f"\n🌍 Starting Environment Suite: {env}", flush=True)
        for model in TARGET_MODELS:
            for seed in SEEDS:
                completed += 1
                print(f"\n▶ [{completed:2d}/{total_runs}] Running {model} on {env} (Seed {seed})...", flush=True)
                cmd = [
                    python_bin, "-u", bench_script,
                    "--envs", env,
                    "--models", model,
                    "--seeds", str(seed),
                    "--total-steps", str(TOTAL_TIMESTEPS),
                ]
                run_start = time.time()
                res = subprocess.run(cmd, capture_output=False, text=True)
                elapsed = time.time() - run_start
                if res.returncode == 0 or res.returncode == 43:
                    print(f"  ✅ Completed [{model} / {env} / Seed {seed}] in {elapsed:.1f}s", flush=True)
                else:
                    print(f"  ❌ Returned code {res.returncode} in {elapsed:.1f}s", flush=True)

    print("\n" + "=" * 80)
    print(f"🎉 FULL 15M-STEP RL MATRIX SWEEP COMPLETE in {(time.time() - start_time):.1f}s")
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
