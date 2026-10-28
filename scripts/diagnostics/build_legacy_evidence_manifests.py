#!/usr/bin/env python3
"""Build conservative run-level manifests for legacy evidence recovery.

This script records what is present in raw exports. It deliberately does not
infer revision/config compatibility from scores or filenames.
"""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "papers" / "writing" / "generated_artifacts"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def numeric_max(rows, keys):
    values = []
    for row in rows:
        for key in keys:
            value = row.get(key)
            if value not in (None, ""):
                try:
                    values.append(float(value))
                except ValueError:
                    pass
                break
    return max(values) if values else ""


def seed_from_name(name: str) -> str:
    match = re.search(r"(?:seed|Seed)[-_]?(\d+)", name)
    if match:
        return match.group(1)
    # Some legacy GGS files use labels such as `GGS-0.1-42`.
    match = re.search(r"-(\d+)\.csv$", name)
    return match.group(1) if match else ""


def craftax_family(path: Path) -> str:
    name = path.name.lower()
    if "ggs" in name:
        return "GGS"
    if "approx" in name:
        return "Exact-vs-Approx / Approximate"
    if "control_exact" in name or "exact" in name:
        return "Exact-vs-Approx / Exact"
    if "gamma" in name:
        return "Gamma ablation"
    if "grubias" in name:
        return "GRU-bias ablation"
    if "capacity" in path.name.lower() or "heads" in name or "ffn" in name:
        return "Capacity / heads / FFN"
    return "Unclassified Craftax legacy"


def craftax_manifest():
    roots = [
        ROOT / "wandb_exports" / "Craftax_Classic_BPTT",
        ROOT / "wandb_exports" / "Craftax_Classic_Exact_vs_Approx",
        ROOT / "wandb_exports" / "Craftax_Classic_Capacity_Scaling",
    ]
    rows = []
    for root in roots:
        for path in sorted(root.glob("*.csv")):
            raw = read_rows(path)
            fields = set(raw[0]) if raw else set()
            row = {
                "raw_artifact": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "study_family": craftax_family(path),
                "label": path.stem,
                "seed": seed_from_name(path.name),
                "row_count": len(raw),
                "max_step": numeric_max(raw, ["_step", "Total_Steps", "env_steps", "Environment_Steps"]),
                "has_eval_return": any("Eval" in field and "Return" in field for field in fields),
                "has_train_return": any("Train" in field and "Return" in field for field in fields),
                "has_sps": any("SPS" in field for field in fields),
                "has_gradient_cosine": any("Cosine" in field for field in fields),
                "metric_fields": ";".join(sorted(fields)),
                "rollout_length": "UNRESOLVED",
                "source_revision": "UNRESOLVED",
                "config_hash": "UNRESOLVED",
                "checkpoint_rule": "UNRESOLVED",
                "hardware_precision": "UNRESOLVED",
                "promotion": "U",
                "reason": "Raw export found; endpoint semantics and revision/config identity not yet linked.",
            }
            rows.append(row)
    return rows


def synthetic_manifest():
    rows = []
    sources = [
        (ROOT / "results" / "factorial_dk_dv_sweep.csv", "historical factorial"),
        (ROOT / "results" / "synthetic_capacity_ladder.csv", "historical load ladder"),
        (ROOT / "results" / "overwrite_under_load.csv", "overwrite under load"),
        (ROOT / "results" / "synthetic_overwrite_collateral.csv", "overwrite collateral"),
        (ROOT / "results" / "dk_asymmetry_study.csv", "d_k asymmetry"),
        (ROOT / "results" / "fixed_head_capacity_study.csv", "fixed-head capacity"),
        (ROOT / "results" / "selective_forgetting_study.csv", "selective forgetting"),
    ]
    for path, family in sources:
        if not path.exists():
            continue
        raw = read_rows(path)
        fields = set(raw[0]) if raw else set()
        for index, record in enumerate(raw):
            rows.append({
                "raw_artifact": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "family": family,
                "row_index": index,
                "N": record.get("N", record.get("num_keys", "")),
                "d_k": record.get("d_k", ""),
                "d_v": record.get("d_v", ""),
                "load_ratio": record.get("load_ratio", record.get("ratio_N_dk", "")),
                "model_or_variant": record.get("model", record.get("variant", "")),
                "metric_fields": ";".join(sorted(fields)),
                "seed": record.get("seed", "UNRESOLVED"),
                "recurrence": "UNRESOLVED",
                "initialization": "UNRESOLVED",
                "normalization": "UNRESOLVED",
                "source_revision": "UNRESOLVED",
                "raw_or_aggregate": "UNRESOLVED",
                "compatibility": "UNRESOLVED",
                "promotion": "U",
            })
    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    craftax = craftax_manifest()
    synthetic = synthetic_manifest()
    write_csv(OUT / "legacy_craftax_run_manifest.csv", craftax)
    write_csv(OUT / "legacy_synthetic_row_manifest.csv", synthetic)
    print(f"craftax_rows={len(craftax)}")
    print(f"synthetic_rows={len(synthetic)}")
