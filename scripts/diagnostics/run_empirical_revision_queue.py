"""Run the declared empirical-revision groups serially on one GPU.

The queue stops at the GGS compatibility gate or on a subprocess failure. Its
status JSON is intentionally small enough for the scheduled thread monitor to
inspect without reading terminal logs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import csv
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv/bin/python"
ARTIFACTS = ROOT / "papers/writing/generated_artifacts/revision_2026"
STATUS = ARTIFACTS / "empirical_revision_queue_status.json"
GGS_WORKTREE = Path("/tmp/jupyter-ggs-f374a13")


def update_status(state: str, group: str, detail: str) -> None:
    STATUS.write_text(json.dumps({
        "state": state,
        "group": group,
        "detail": detail,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n")


def run(group: str, command: list[str], cwd: Path) -> None:
    update_status("running", group, " ".join(command))
    environment = dict(os.environ)
    environment.update({"WANDB_MODE": "offline", "MPLCONFIGDIR": "/tmp/mpl-revision-2026"})
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def gate_endpoints_exist(path: Path) -> bool:
    """Allow a bookkeeping-only resume without repeating completed gate runs."""
    if not path.exists():
        return False
    complete = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("Eval_Return", "").strip() and int(float(row["Total_Steps"])) >= 20_000_000:
                complete.add((row["Architecture"], int(row["Seed"])))
    return {("SdLGHybrid-GGS-0.1", 42), ("SdLGHybrid-GGS-1", 42)} <= complete


def missing_ggs_sweep_cells(path: Path) -> dict[float, list[int]]:
    """Return only sweep cells without a persisted final endpoint.

    This makes a user-requested queue pause recoverable without overwriting
    already accepted seed-level rows.
    """
    alphas = (0.0, 0.1, 0.25, 0.5, 1.0)
    seeds = (42, 54, 99, 123, 456)
    complete: set[tuple[str, int]] = set()
    if path.exists():
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("Eval_Return", "").strip() and int(float(row["Total_Steps"])) >= 20_000_000:
                    complete.add((row["Architecture"], int(row["Seed"])))
    return {
        alpha: [seed for seed in seeds if (f"SdLGHybrid-GGS-{alpha:g}", seed) not in complete]
        for alpha in alphas
    }


def main() -> None:
    if not GGS_WORKTREE.is_dir():
        raise RuntimeError(f"Pinned GGS worktree is missing: {GGS_WORKTREE}")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    try:
        gate_raw = ARTIFACTS / "ggs_reproduction_gate_raw.csv"
        if not gate_endpoints_exist(gate_raw):
            run("ggs_reproduction_gate", [
                str(PYTHON), "scripts/diagnostics/architecture/gradient_shielding/gradient_shielding.py",
                "--run", "--alphas", "0.1,1.0", "--seeds", "42", "--overwrite",
                "--output", str(gate_raw),
                "--manifest", str(ARTIFACTS / "ggs_reproduction_gate_manifest.json"),
            ], GGS_WORKTREE)
        run("ggs_reproduction_gate_validation", [
            str(PYTHON), str(ROOT / "scripts/diagnostics/validate_ggs_reproduction_gate.py"),
            "--historical", str(ROOT / "scripts/diagnostics/architecture/gradient_shielding/gradient_shielding.csv"),
            "--rerun", str(gate_raw),
            "--output", str(ARTIFACTS / "ggs_reproduction_gate_validation.json"),
            "--max-absolute-difference", "0.5",
        ], ROOT)
        sweep_raw = ARTIFACTS / "ggs_alpha_sweep_raw.csv"
        missing_sweep = missing_ggs_sweep_cells(sweep_raw)
        for alpha, missing_seeds in missing_sweep.items():
            if not missing_seeds:
                continue
            command = [
                str(PYTHON), "scripts/diagnostics/architecture/gradient_shielding/gradient_shielding.py",
                "--run", "--alphas", f"{alpha:g}", "--seeds", ",".join(map(str, missing_seeds)),
                "--output", str(sweep_raw),
                "--manifest", str(ARTIFACTS / ("ggs_alpha_sweep_manifest.json" if not sweep_raw.exists()
                                                else f"ggs_alpha_sweep_resume_alpha_{alpha:g}_manifest.json")),
            ]
            if not sweep_raw.exists():
                command.append("--overwrite")
            run(f"ggs_alpha_sweep_alpha_{alpha:g}", command, GGS_WORKTREE)
        run("countrecall_confirmation", [
            str(PYTHON), "scripts/diagnostics/revision_2026_countrecall_confirmation.py",
            "--models", "SdLG-Exact,SdLG-Approx,PPO-GRU,PPO-GLA", "--seeds", "42,54,99,123,456", "--overwrite",
            "--output", str(ARTIFACTS / "countrecall_confirmation/raw.csv"),
            "--manifest", str(ARTIFACTS / "countrecall_confirmation/manifest.json"),
        ], ROOT)
        run("ggs_alpha_sweep_analysis", [
            str(PYTHON), str(ROOT / "scripts/diagnostics/analyze_empirical_strengthening.py"),
            "--study", "ggs",
        ], ROOT)
        run("countrecall_confirmation_analysis", [
            str(PYTHON), str(ROOT / "scripts/diagnostics/analyze_countrecall_confirmation.py"),
            "--input", str(ARTIFACTS / "countrecall_confirmation/raw.csv"),
            "--output-dir", str(ARTIFACTS / "countrecall_confirmation"),
            "--plot-dir", str(ROOT / "papers/latex/cleanRewrite/plots/rebuilt"),
        ], ROOT)
    except subprocess.CalledProcessError as error:
        update_status("failed", "subprocess", f"return code {error.returncode}: {error.cmd}")
        raise
    except Exception as error:
        update_status("failed", "queue", repr(error))
        raise
    update_status("complete", "all_groups", "GGS gate, GGS sweep, CountRecall matrix, and analyses finished")


if __name__ == "__main__":
    main()
