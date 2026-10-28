#!/usr/bin/env python3
"""Standalone launcher for the frozen eight-family T-Maze landscape.

The recurrent modules and PPO loop survive in ``rope_variant_tmaze.py``. This
adapter supplies the frozen landscape protocol, isolates all artifacts in a
chosen output directory, and writes a compact terminal-evaluation record.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "studies" / "tmaze_external_landscape.json"


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text())
    required = {"study_id", "environment", "training", "seeds", "families", "results_schema"}
    missing = required - set(protocol)
    if missing:
        raise ValueError(f"Protocol configuration is missing: {sorted(missing)}")
    if protocol["study_id"] != "T3-TMAZE-EXTERNAL-LANDSCAPE":
        raise ValueError("Unexpected study ID")
    if protocol["training"]["total_environment_steps"] != 50_000_000:
        raise ValueError("Frozen external-landscape budget must be 50M steps")
    if len(protocol["families"]) != 8 or len(protocol["seeds"]) != 7:
        raise ValueError("Frozen external landscape requires eight families and seven seeds")
    return protocol


def baseline_ablations() -> dict[str, dict]:
    """Map the frozen family names to the surviving baseline implementations."""
    return {
        "GRU": {"IS_BASELINE": True, "RNN_TYPE": "gru"},
        "LSTM": {"IS_BASELINE": True, "RNN_TYPE": "lstm"},
        "Mamba": {"IS_BASELINE": True, "RNN_TYPE": "mamba"},
        "LTI-Linear": {"IS_BASELINE": True, "RNN_TYPE": "lti_linear"},
        "Markovian": {"IS_BASELINE": True, "RNN_TYPE": "markovian"},
        "GLA": {"IS_BASELINE": True, "RNN_TYPE": "gla"},
        "GTrXL-64": {"IS_BASELINE": True, "RNN_TYPE": "gtrxl", "MEMORY_LEN": 64},
        "GTrXL-128": {"IS_BASELINE": True, "RNN_TYPE": "gtrxl", "MEMORY_LEN": 128},
    }


def validate(protocol: dict, families: list[str], seeds: list[int]) -> None:
    available = baseline_ablations()
    unknown = set(families) - set(available)
    if unknown:
        raise ValueError(f"Unsupported baseline family: {sorted(unknown)}")
    if not families or not seeds:
        raise ValueError("At least one family and seed are required")
    frozen_families = set(protocol["families"])
    if not set(families) <= frozen_families or not set(seeds) <= set(protocol["seeds"]):
        raise ValueError("Selected families/seeds must be subsets of the frozen protocol")


def materialize_terminal_records(output_dir: Path, total_steps: int) -> Path:
    """Convert the surviving runner's two CSV streams to the release schema."""
    training_path = output_dir / "rope_variants_training.csv"
    evaluation_path = output_dir / "rope_variants_zeroshot.csv"
    if not training_path.is_file() or not evaluation_path.is_file():
        raise FileNotFoundError("The underlying runner did not produce its expected CSV artifacts")

    terminal_steps: dict[tuple[str, int], int] = {}
    with training_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["model"], int(row["seed"]))
            terminal_steps[key] = max(terminal_steps.get(key, 0), int(float(row["step"])))

    records = []
    with evaluation_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["model"], int(row["seed"]))
            steps = terminal_steps.get(key)
            if steps is None:
                raise ValueError(f"No training record for final evaluation {key}")
            records.append({
                "family": row["model"],
                "seed": key[1],
                "run_id": f"{row['model']}_H{row['H_train']}_seed_{row['seed']}",
                "terminal_steps": steps,
                "terminal_status": "budget_cap" if steps >= total_steps else "early_stop",
                "H100": row["val_H100"],
                "H150": row["val_H150"],
                "H200": row["val_H200"],
            })
    path = output_dir / "terminal_evaluations.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda row: (row["family"], row["seed"])))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tmaze-external-landscape")
    parser.add_argument("--families", nargs="+", help="Optional subset of frozen baseline families")
    parser.add_argument("--seeds", nargs="+", type=int, help="Optional subset of frozen seeds")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="offline")
    parser.add_argument("--validate-config", action="store_true", help="Validate the protocol and output contract without importing JAX or training.")
    args = parser.parse_args()

    protocol = load_protocol(args.config)
    families = args.families or protocol["families"]
    seeds = args.seeds or protocol["seeds"]
    validate(protocol, families, seeds)
    args.output_dir = args.output_dir.resolve()
    if args.validate_config:
        print(f"PASS protocol: {len(families)} family/families x {len(seeds)} seed(s)")
        print(f"PASS output contract: {args.output_dir / 'terminal_evaluations.csv'}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "resolved_protocol.json").write_text(json.dumps({
        **protocol, "selected_families": families, "selected_seeds": seeds, "wandb_mode": args.wandb_mode
    }, indent=2) + "\n")
    os.environ["WANDB_MODE"] = args.wandb_mode

    # Delay importing JAX and the historical training implementation until the
    # no-training protocol check has completed.
    from scripts.diagnostics.exact_sdlg import rope_variant_tmaze as runner

    runner.current_file_path = str(args.output_dir / "tmaze_external_landscape.py")
    runner.COMPLETED_RUNS_PATH = str(args.output_dir / "completed_runs.json")
    runner.ABLATIONS = {family: baseline_ablations()[family] for family in families}
    runner.MODELS_TO_RUN = families
    runner.TRAINING_SEEDS = seeds
    runner.generate_rope_vs_cosine_plot = lambda *_: None
    runner.run_tmaze_ablations()
    terminal = materialize_terminal_records(args.output_dir, protocol["training"]["total_environment_steps"])
    print(f"wrote {terminal}")


if __name__ == "__main__":
    main()
