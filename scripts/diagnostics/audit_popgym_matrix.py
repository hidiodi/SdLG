"""Build a provenance-first POPGym matrix audit from the surviving CSV.

This script deliberately does not infer historical revisions from model labels,
scores, or checkpoint shapes. It records what is observable in the raw CSV and
marks revision/config identity unresolved unless an external run manifest exists.
"""

from __future__ import annotations

import csv
import itertools
import glob
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "popgym_results.csv"
OUT = ROOT / "papers" / "writing"
MANIFEST = OUT / "popgym_gate4_full_matrix_manifest.csv"
REPORT = OUT / "popgym_gate4_full_matrix_audit.md"

CORE_MODELS = ["PPO-GLA", "PPO-GRU", "PPO-Mamba", "SdLG-Approx", "SdLG-Exact"]
DEFAULT_SEEDS = {42, 54, 99}
COMMON_SEEDS = {42, 43, 44}
FULL_STEP = 14_909_440

FAMILIES = {
    "AutoencodeEasy": "reconstruction",
    "BattleshipEasy": "spatial search/problem solving",
    "ConcentrationEasy": "discrete associative matching/decision memory",
    "ConcentrationMedium": "discrete associative matching/decision memory",
    "CountRecallEasy": "initial-context retention/state accumulation",
    "CountRecallMedium": "initial-context retention/state accumulation",
    "HigherLowerEasy": "discrete associative matching/decision memory",
    "HigherLowerMedium": "discrete associative matching/decision memory",
    "LabyrinthEscapeEasy": "spatial search/problem solving",
    "LabyrinthEscapeMedium": "spatial search/problem solving",
    "LabyrinthExploreEasy": "spatial search/problem solving",
    "LabyrinthExploreMedium": "spatial search/problem solving",
    "MineSweeperEasy": "spatial search/problem solving",
    "MinesweeperMedium": "spatial search/problem solving",
    "MultiarmedBanditEasy": "discrete associative matching/decision memory",
    "NoisyPositionOnlyCartPoleEasy": "continuous state tracking/noisy control",
    "NoisyPositionOnlyPendulumEasy": "continuous state tracking/noisy control",
    "PositionOnlyCartPoleEasy": "continuous state tracking/noisy control",
    "PositionOnlyPendulumEasy": "continuous state tracking/noisy control",
    "RepeatFirstEasy": "initial-context retention/sequential recall",
    "RepeatPreviousEasy": "initial-context retention/sequential recall",
    "RepeatPreviousMedium": "initial-context retention/sequential recall",
    "VelocityOnlyCartpoleEasy": "continuous state tracking/noisy control",
}


def read_groups():
    groups = defaultdict(list)
    with RAW.open(newline="") as handle:
        for row in csv.DictReader(handle):
            groups[(row["Environment"], row["Model"], int(row["Seed"]))].append(row)
    return groups


def group_summary(rows):
    steps = [int(float(row["Step"])) for row in rows]
    max_step = max(steps)
    n_at_max = sum(step == max_step for step in steps)
    complete = max_step == FULL_STEP and len(rows) >= 91
    if complete:
        endpoint = "full-budget candidate"
        observed_status = "provisional-complete"
    elif len(rows) <= 3:
        endpoint = "sparse/incomplete log"
        observed_status = "unresolved-sparse"
    else:
        endpoint = "early-stop or mixed endpoint"
        observed_status = "unresolved-endpoint"
    return {
        "observed_rows": len(rows),
        "max_step": max_step,
        "n_at_max": n_at_max,
        "endpoint_observation": endpoint,
        "observed_status": observed_status,
    }


def classify_cell(task, model, seed_summaries):
    """Classify a task/model cell without treating revision as recoverable."""
    seeds = sorted(seed_summaries)
    if not seeds:
        return "unresolved", "no observed run"
    if len(seeds) == 1:
        return "unresolved", "single-seed cell"
    if set(seeds) != COMMON_SEEDS:
        return "qualified", "noncanonical/expanded seed set"
    if all(seed_summaries[s]["observed_status"] == "provisional-complete" for s in seeds):
        return "qualified", "common seeds and endpoint observed; revision/config unresolved"
    return "qualified", "common seeds but endpoint/log coverage is mixed"


def write_manifest(groups):
    fields = [
        "task", "task_level", "memory_operation_family", "model", "seed",
        "observed_rows", "max_step", "n_at_max", "endpoint_observation", "observed_status",
        "revision_id", "config_source", "raw_artifact", "checkpoint_artifact",
        "checkpoint_candidate_count",
        "parameter_count", "capacity_class", "comparability_class", "status",
        "reason_code", "metric_candidates",
    ]
    rows = []
    for (task, model, seed), raw_rows in sorted(groups.items()):
        summary = group_summary(raw_rows)
        all_for_cell = {
            s: group_summary(v)
            for (t, m, s), v in groups.items()
            if t == task and m == model
        }
        status, reason = classify_cell(task, model, all_for_cell)
        level = "Medium" if task.endswith("Medium") or task == "MinesweeperMedium" else "Easy"
        checkpoint = f"checkpoints/{task}_{model}_seed{seed}_*.pkl"
        checkpoint_count = len(glob.glob(str(ROOT / checkpoint)))
        if summary["observed_status"] == "provisional-complete":
            reason = reason + "; endpoint observed at 14,909,440"
        else:
            reason = reason + "; endpoint/log coverage unresolved"
        rows.append({
            "task": task,
            "task_level": level,
            "memory_operation_family": FAMILIES.get(task, "unclassified"),
            "model": model,
            "seed": seed,
            **summary,
            "revision_id": "UNRESOLVED",
            "config_source": "not serialized in raw CSV",
            "raw_artifact": "results/popgym_results.csv",
            "checkpoint_artifact": checkpoint,
            "checkpoint_candidate_count": checkpoint_count,
            "parameter_count": "checkpoint-dependent; not row-mapped",
            "capacity_class": "capacity-unmatched",
            "comparability_class": status,
            "status": "hold",
            "reason_code": reason,
            "metric_candidates": "Eval_NormScore_H100;Eval_NormScore_H200;Eval_Return_H100;Eval_Return_H200",
        })
    with MANIFEST.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def common_block(groups):
    tasks = []
    for task in sorted({key[0] for key in groups}):
        ok = True
        for model in CORE_MODELS:
            for seed in sorted(COMMON_SEEDS):
                rows = groups.get((task, model, seed))
                if not rows:
                    ok = False
                    break
                if not (len(rows) >= 91 and max(int(float(r["Step"])) for r in rows) == FULL_STEP):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            tasks.append(task)
    return tasks


def best_model_block(groups, size):
    tasks = sorted({key[0] for key in groups})
    candidates = []
    for models in itertools.combinations(CORE_MODELS, size):
        valid = []
        for task in tasks:
            if all(
                (task, model, seed) in groups
                and len(groups[(task, model, seed)]) >= 91
                and max(int(float(row["Step"])) for row in groups[(task, model, seed)]) == FULL_STEP
                for model in models
                for seed in COMMON_SEEDS
            ):
                valid.append(task)
        candidates.append((len(valid), models, valid))
    return max(candidates, key=lambda item: (item[0], item[1]))


def rough_rerun_hours(groups):
    """Estimate serial hours for 23 tasks x 3 seeds using observed final SPS.

    This is explicitly a planning estimate. It is not used for any scientific
    comparison because the legacy SPS values have heterogeneous provenance.
    """
    by_model = defaultdict(list)
    for (task, model, seed), rows in groups.items():
        if model not in CORE_MODELS:
            continue
        final = max(rows, key=lambda row: int(float(row["Step"])))
        try:
            by_model[model].append(float(final["SPS"]))
        except (KeyError, ValueError):
            pass
    total = 0.0
    medians = {}
    for model in CORE_MODELS:
        median_sps = statistics.median(by_model[model])
        medians[model] = median_sps
        total += 23 * 3 * 15_000_000 / median_sps / 3600
    return total, medians


def write_report(groups, manifest_rows):
    tasks = sorted({key[0] for key in groups})
    models = sorted({key[1] for key in groups})
    block = common_block(groups)
    block4 = best_model_block(groups, 4)
    block3 = best_model_block(groups, 3)
    rerun_hours, median_sps = rough_rerun_hours(groups)
    class_counts = defaultdict(int)
    for row in manifest_rows:
        class_counts[row["comparability_class"]] += 1
    task_cells = defaultdict(set)
    for task, model, seed in groups:
        task_cells[task].add(model)
    lines = [
        "# POPGym Gate 4 — Full 23-task Matrix Audit",
        "",
        "Status: **CLOSED FOR HISTORICAL HEADLINE USE — optional fresh representative rerun remains.**",
        "",
        "## Scope and observed coverage",
        "",
        f"- Raw artifact: `results/popgym_results.csv`; {len(tasks)} tasks, {len(models)} model labels, {len(groups)} observed task/model/seed groups.",
        "- The task list is 16 Easy + 7 Medium environments. The raw file contains time series, not a serialized run manifest.",
        f"- The provisional common block is five core models (`{', '.join(CORE_MODELS)}`), seeds `{sorted(COMMON_SEEDS)}`, and observed endpoint 14,909,440 steps.",
        f"- The strict 5-model / 5-task block is: {', '.join(block)}.",
        f"- The best observed 4-model subset covers {block4[0]} tasks ({', '.join(block4[1])}); the best 3-model subset covers {block3[0]} tasks ({', '.join(block3[1])}).",
        "- This is **not** yet a fully comparable block: revision/config identity is unresolved, and checkpoint paths are not linked to individual CSV rows.",
        "",
        "## Comparability disposition",
        "",
        "| Class | Meaning in this audit | Row count |",
        "|---|---|---:|",
        f"| Fully comparable | No cells currently meet this standard; revision/config mapping is missing | 0 |",
        f"| Qualified | Observable seeds/endpoints support a provisional contrast, subject to revision/config and artifact mapping | {class_counts['qualified']} |",
        f"| Unresolved | Singleton, sparse, or otherwise insufficient observed coverage | {class_counts['unresolved']} |",
        "",
        "A qualified cell is not a publication-ready aggregate. It is a candidate for either historical provenance recovery or a fresh rerun from one frozen revision.",
        "",
        "## Rerun planning estimate",
        "",
        f"Using the legacy CSV's median final SPS only as a rough planning proxy, a fresh 23-task × 5-model × 3-seed suite at 15M steps would represent approximately **{rerun_hours:.0f} serial accelerator-hours** before parallel scheduling, failures, or reruns. The proxy is heterogeneous and is not suitable for a scientific throughput claim. A four-task representative rerun would be roughly one-fifth of that cost.",
        "",
        "## Task-family interpretation",
        "",
        "The outcome-independent task grouping remains appropriate as a map, not as a pooled score:",
        "",
        "| Family | Tasks represented |",
        "|---|---|",
    ]
    family_tasks = defaultdict(list)
    for task in tasks:
        family_tasks[FAMILIES.get(task, "unclassified")].append(task)
    for family in sorted(family_tasks):
        lines.append(f"| {family} | {', '.join(family_tasks[family])} |")
    lines += [
        "",
        "Tasks may exercise multiple operations; the family label is a primary constructor-based tag, not a claim of mechanism isolation.",
        "",
        "## Decision",
        "",
        "**Decision.** The historical POPGym matrix confirms substantial benchmark breadth but does not support a unified 23-task leaderboard, pooled task-family score, or headline cross-model comparison because revision/config identity remains unresolved. A bounded provenance-recovery pass did not establish a complete mapping for the strict 5-model / 5-task block. Historical POPGym therefore remains qualified/supporting evidence only. If POPGym is strengthened before submission, the preferred path is a fresh frozen 4-task × 5-model × 3-seed representative rerun; a full 23-task rerun is deferred unless later scientific value justifies its substantially higher cost.",
        "",
        "## Machine-readable outputs",
        "",
        f"- Row-level observed manifest: `{MANIFEST.relative_to(ROOT)}`",
        "- Bounded provenance-recovery record: `papers/writing/popgym_gate4_provenance_recovery.md`",
        "- This manifest intentionally records `UNRESOLVED` revision/config fields rather than inferring them from scores, labels, or parameter counts.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")


def main():
    groups = read_groups()
    manifest_rows = write_manifest(groups)
    write_report(groups, manifest_rows)
    print(f"wrote {MANIFEST} ({len(manifest_rows)} rows)")
    print(f"wrote {REPORT}")
    print(f"common block: {common_block(groups)}")


if __name__ == "__main__":
    main()
