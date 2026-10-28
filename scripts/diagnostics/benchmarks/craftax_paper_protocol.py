#!/usr/bin/env python3
"""Launch or validate the frozen Craftax landscape and cross-T protocols.

Each full job is delegated to a fresh process because the surviving Craftax
implementation intentionally uses a watchdog exit after a long ROCm session.
The adapter turns that legacy behavior into a normal, resumable matrix launch
without treating a T=16 watchdog sweep as the paper protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "scripts" / "diagnostics" / "benchmarks" / "craftax_full" / "craftax_classic_benchmark.py"
CONFIGS = {
    "landscape": ROOT / "configs" / "studies" / "craftax_t64_landscape.json",
    "cross-t": ROOT / "configs" / "studies" / "craftax_cross_t.json",
}
RUNNER_NAMES = {
    "PPO-GRU": ("PPO-GRU", False),
    "PPO-Mamba": ("PPO-Mamba", False),
    "AGaLiTe": ("AGaLiTe", False),
    "PPO-GLA": ("PPO-GLA", False),
    "PPO-GTrXL": ("PPO-GTrXL", False),
    "SdLG-Approx": ("SdLGHybrid", True),
    "SdLG-Exact": ("ExactSdLG", False),
}


def load_protocol(profile: str) -> dict:
    protocol = json.loads(CONFIGS[profile].read_text())
    if protocol["training"]["total_environment_steps"] != 50_000_000:
        raise ValueError("Frozen Craftax studies use 50M environment steps")
    if len(protocol["training"]["seeds"]) != 5:
        raise ValueError("Frozen Craftax studies require five seeds")
    return protocol


def cells(profile: str, protocol: dict) -> list[tuple[str, int]]:
    if profile == "landscape":
        return [(family, int(protocol["training"]["rollout_tbptt_length"])) for family in protocol["families"]]
    return [(family, int(length)) for family, length in protocol["reported_cells"]]


def validate(protocol: dict, selected: list[tuple[str, int]], seeds: list[int], output_dir: Path) -> None:
    unknown = {family for family, _ in selected} - set(RUNNER_NAMES)
    if unknown:
        raise ValueError(f"No surviving Craftax runner mapping for {sorted(unknown)}")
    if not selected or not seeds:
        raise ValueError("At least one reported cell and one seed are required")
    if not set(seeds) <= set(protocol["training"]["seeds"]):
        raise ValueError("Selected seeds must be part of the frozen five-seed protocol")
    if not RUNNER.is_file():
        raise FileNotFoundError(RUNNER)
    output_dir.mkdir(parents=True, exist_ok=True)


def legacy_job_code(family: str, rollout: int, seed: int, total_steps: int, tag: str) -> str:
    runner_name, coupled = RUNNER_NAMES[family]
    return f'''import importlib.util
from pathlib import Path
path = Path({str(RUNNER)!r})
spec = importlib.util.spec_from_file_location("craftax_legacy", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.TOTAL_STEPS = {total_steps}
module.SESSION_STEP_LIMIT = {total_steps}
module.TRAINING_SEEDS = [{seed}]
module.RUN_CONFIGS = [({runner_name!r}, {rollout}, {coupled})]
module.RunTag = {tag!r}
module.run_benchmark()
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "craftax")
    parser.add_argument("--family", action="append", help="Optional reported family; repeat to select multiple")
    parser.add_argument("--t", action="append", type=int, help="Optional rollout/TBPTT length; repeat to select multiple")
    parser.add_argument("--seeds", nargs="+", type=int, help="Optional subset of the frozen seed list")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="offline")
    parser.add_argument("--validate-config", action="store_true", help="Validate protocol, runner mappings, and output location without importing JAX or training.")
    args = parser.parse_args()

    protocol = load_protocol(args.profile)
    selected = cells(args.profile, protocol)
    if args.family:
        selected = [cell for cell in selected if cell[0] in set(args.family)]
    if args.t:
        selected = [cell for cell in selected if cell[1] in set(args.t)]
    seeds = args.seeds or protocol["training"]["seeds"]
    args.output_dir = args.output_dir.resolve()
    validate(protocol, selected, seeds, args.output_dir)

    if args.validate_config:
        print(f"PASS protocol: {len(selected)} reported Craftax cell(s) x {len(seeds)} seed(s)")
        print(f"PASS runner mappings: {', '.join(sorted({family for family, _ in selected}))}")
        print(f"PASS output contract: {args.output_dir}")
        return

    (args.output_dir / "resolved_protocol.json").write_text(json.dumps({
        "profile": args.profile, "protocol": protocol, "selected_cells": selected,
        "selected_seeds": seeds, "wandb_mode": args.wandb_mode,
    }, indent=2) + "\n")
    environment = dict(os.environ, WANDB_MODE=args.wandb_mode)
    total_steps = int(protocol["training"]["total_environment_steps"])
    for family, rollout in selected:
        for seed in seeds:
            tag = f"paper-{args.profile}-{family}-T{rollout}-seed{seed}".replace(" ", "-")
            print(f"Launching {tag}", flush=True)
            completed = subprocess.run(
                [sys.executable, "-c", legacy_job_code(family, rollout, seed, total_steps, tag)],
                cwd=args.output_dir, env=environment,
            )
            # The legacy watchdog uses 43 after a successful completed session.
            if completed.returncode not in (0, 43):
                raise SystemExit(f"Craftax job failed ({tag}) with exit code {completed.returncode}")


if __name__ == "__main__":
    main()
