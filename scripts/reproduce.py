#!/usr/bin/env python3
"""The standalone command registry for every frozen-paper study.

Use `list`, `show`, and `dry-run` during setup.  `run` dispatches to the
current study implementation; runners are migrated one at a time from their
legacy locations into this public interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.release import DATA_DIR, RESULTS_DIR, write_run_manifest


@dataclass(frozen=True)
class Study:
    key: str
    identifier: str
    summary: str
    runner: str
    arguments: tuple[str, ...]
    records: tuple[str, ...]
    output_flag: str | None = None
    output_name: str | None = None
    validation_arguments: tuple[str, ...] = ()
    status: str = "migration"


STUDIES = (
    Study("tmaze-historical", "T1-TMAZE-EXECUTION-PHASE", "T-Maze execution x phase factorial", "scripts/diagnostics/exact_sdlg/exact_sdlg_tmaze_ablations.py", ("--study", "historical"), ("data/paper-records/tmaze_rank_rope_zero_shot.csv",), "--output-dir"),
    Study("tmaze-matched", "T1-TMAZE-MATCHED-EXECUTION", "Matched T-Maze Exact versus Approximate", "scripts/diagnostics/exact_sdlg/exact_sdlg_tmaze_ablations.py", ("--study", "matched"), ("data/paper-records/revision_2026/tmaze_factorial_seed_outcomes.csv",), "--output-dir", validation_arguments=("--validate-study", "matched")),
    Study("tmaze-baselines", "T3-TMAZE-EXTERNAL-LANDSCAPE", "T-Maze recurrent baseline landscape", "scripts/diagnostics/benchmarks/tmaze_external_landscape.py", (), ("data/paper-records/tmaze_external_landscape.csv",), "--output-dir", validation_arguments=("--validate-config",)),
    Study("minigrid-locality", "T1-MINIGRID-LOCAL-WINDOW", "MemoryS16 local-window intervention", "scripts/diagnostics/architecture/swa_ablation/swa_ablation.py", (), ("data/paper-records/revision_2026/minigrid_locality_seed_endpoints.csv",), "--output", "swa_ablation.csv"),
    Study("doorkey-8x8", "T3-MINIGRID-DOORKEY-8X8", "DoorKey cross-family terminal evaluation", "scripts/diagnostics/benchmarks/xminigrid/doorkey_baseline_benchmark.py", (), ("data/paper-records/doorkey_8x8/five_seed_summary.csv",), "--results-dir"),
    Study("popgym", "T3-POPGYM-SUITE", "POPGym Easy and Medium suite", "scripts/diagnostics/exact_sdlg/popgym_benchmark.py", (), ("data/paper-records/original_paper_upgrade/popgym_complete_mmer_per_seed.csv",), "--output-dir"),
    Study("craftax-landscape", "T3-CRAFTAX-EXTERNAL-LANDSCAPE", "Craftax T=64 model landscape", "scripts/diagnostics/benchmarks/craftax_paper_protocol.py", ("--profile", "landscape"), ("data/paper-records/original_paper_upgrade/craftax_t64_seven_family_performance.csv",), "--output-dir", validation_arguments=("--profile", "landscape", "--validate-config")),
    Study("craftax-cross-t", "T3-CRAFTAX-CROSS-T", "Craftax rollout-length study", "scripts/diagnostics/benchmarks/craftax_paper_protocol.py", ("--profile", "cross-t"), ("data/paper-records/craftax_cross_t_verified.csv",), "--output-dir", validation_arguments=("--profile", "cross-t", "--validate-config")),
    Study("craftax-ggs", "T1-CRAFTAX-ACTOR-GRADIENT-SCALE", "Craftax actor-gradient scale intervention", "scripts/diagnostics/architecture/gradient_shielding/gradient_shielding.py", ("--run",), ("data/paper-records/ggs_endpoint_paired_difference.csv",), "--output", "gradient_shielding.csv"),
    Study("associative-capacity", "T2-ASSOCIATIVE-CAPACITY", "Associative retrieval diagnostic", "scripts/diagnostics/exact_sdlg/persist_ten_seed_capacity_cells.py", (), ("data/paper-records/synthetic_capacity_480_cells.csv",), "--output"),
    Study("associative-overwrite", "CTV-A1-ASSOCIATIVE-OVERWRITE", "Shared-draw associative overwrite", "scripts/diagnostics/exact_sdlg/factorial_addressing_forgetting_suite.py", ("--component-transfer-overwrite",), ("data/paper-records/component_transfer/associative_overwrite_raw.csv",), "--output-dir"),
    Study("state-accounting", "CTV-B2-MATCHED-STATE-INFERENCE", "Matched carry and inference accounting", "scripts/diagnostics/benchmarks/latency/profile_throughput.py", ("--matched-tmaze-inference", "--warmup", "50", "--repeats", "300"), ("data/paper-records/component_transfer/matched_tmaze_inference.csv",), "--output", "matched_tmaze_inference.csv"),
)
BY_KEY = {study.key: study for study in STUDIES}


def require_study(key: str) -> Study:
    try:
        return BY_KEY[key]
    except KeyError as error:
        raise SystemExit(f"Unknown study {key!r}; use `list`.") from error


def dry_run(study: Study) -> bool:
    missing = [path for path in (ROOT / study.runner, *(ROOT / item for item in study.records)) if not path.is_file()]
    if missing:
        print(f"FAIL {study.key}: missing " + ", ".join(str(path.relative_to(ROOT)) for path in missing))
        return False
    subprocess.run([sys.executable, "-m", "py_compile", str(ROOT / study.runner)], check=True)
    if study.validation_arguments:
        subprocess.run([sys.executable, str(ROOT / study.runner), *study.validation_arguments], cwd=ROOT, check=True)
    level = "RECORD" if study.status == "record-only" else "PASS"
    suffix = "; standalone training runner pending" if study.status == "record-only" else ""
    print(f"{level} {study.key}: runner and {len(study.records)} frozen record(s) present{suffix}")
    return True


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "show", "dry-run", "run", "tables", "figures", "verify-records", "audit"))
    parser.add_argument("study", nargs="?")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.command == "list":
        for study in STUDIES:
            print(f"{study.key:24} {study.identifier:34} {study.summary}")
        return
    if args.command == "tables":
        subprocess.run([sys.executable, str(ROOT / "scripts/build_tables.py"), "--output-dir", str(args.output_dir or RESULTS_DIR / "tables")], cwd=ROOT, check=True)
        return
    if args.command == "figures":
        subprocess.run([sys.executable, str(ROOT / "scripts/build_figures.py"), "--output-dir", str(args.output_dir or RESULTS_DIR / "figures")], cwd=ROOT, check=True)
        return
    if args.command == "verify-records":
        subprocess.run([sys.executable, str(ROOT / "scripts/verify_paper_records.py")], cwd=ROOT, check=True)
        return
    if args.command == "audit":
        if not all(dry_run(study) for study in STUDIES):
            raise SystemExit(1)
        subprocess.run([sys.executable, str(ROOT / "scripts/verify_paper_records.py")], cwd=ROOT, check=True)
        audit_dir = RESULTS_DIR / "audit"
        subprocess.run([sys.executable, str(ROOT / "scripts/build_tables.py"), "--output-dir", str(audit_dir / "tables")], cwd=ROOT, check=True)
        subprocess.run([sys.executable, str(ROOT / "scripts/build_figures.py"), "--output-dir", str(audit_dir / "figures")], cwd=ROOT, check=True)
        report = {
            "format": 1,
            "status": "pass",
            "studies": [asdict(study) for study in STUDIES],
            "records": {
                record: sha256(ROOT / record)
                for study in STUDIES
                for record in study.records
            },
            "tables": sorted(path.name for path in (audit_dir / "tables").iterdir() if path.is_file()),
            "figures": sorted(path.name for path in (audit_dir / "figures").iterdir() if path.is_file()),
        }
        report_path = audit_dir / "audit_report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {report_path.relative_to(ROOT)}")
        return
    if not args.study:
        raise SystemExit("A study is required for show, dry-run, and run.")
    study = require_study(args.study)
    if args.command == "show":
        print(json.dumps(asdict(study), indent=2))
    elif args.command == "dry-run":
        raise SystemExit(0 if dry_run(study) else 1)
    else:
        if study.status == "record-only":
            raise SystemExit(f"{study.key} has a verified frozen record but no standalone training runner yet.")
        output_dir = (args.output_dir or RESULTS_DIR / study.key).resolve()
        manifest = write_run_manifest(study.identifier, asdict(study), output_dir)
        print(f"Wrote {manifest.relative_to(ROOT)}")
        command = [sys.executable, str(ROOT / study.runner), *study.arguments]
        if study.output_flag:
            destination = output_dir / (study.output_name or ("capacity_cells.csv" if study.key == "associative-capacity" else "outputs"))
            command.extend((study.output_flag, str(destination)))
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
