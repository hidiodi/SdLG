"""Verify the balanced DoorKey-8x8 ledger before emitting final summaries.

This is deliberately independent of the training runner. It never trains,
deduplicates ledger rows, or pools the archival larger-grid pilot.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/minigrid_doorkey_baselines_2026"
OUT = BASE / "completion_20261025"
ENV = "MiniGrid-DoorKey-8x8"
MODELS = ("SdLG-Exact", "PPO-GRU", "PPO-GTrXL")
SEEDS = (42, 54, 99, 123, 456)
BLOCK = 512 * 64 * 10
CAP = (20_000_000 // BLOCK) * BLOCK


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def key(row):
    return row["environment"], row["model"], int(row["seed"])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def offline_records():
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    records = defaultdict(list)
    for directory in sorted((ROOT / "wandb").glob("offline-run-2026102[45]*")):
        log = directory / "logs/debug.log"
        if not log.exists():
            continue
        configs = [line[8:] for line in log.read_text().splitlines() if line.startswith("config: ")]
        if not configs:
            continue
        config = ast.literal_eval(configs[0])
        if config.get("ENV_NAME") != ENV:
            continue
        files = list(directory.glob("*.wandb"))
        assert len(files) == 1, directory
        store = DataStore()
        store.open_for_scan(str(files[0]))
        summary = {}
        while (data := store.scan_data()) is not None:
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            if record.HasField("summary"):
                for item in record.summary.update:
                    summary[item.key or "/".join(item.nested_key)] = json.loads(item.value_json)
        store.close()
        records[(ENV, config["MODEL"], config["SEED"])].append({
            "directory": str(directory.relative_to(ROOT)),
            "record_sha256": sha(files[0]),
            "config": config,
            "summary": summary,
            "has_terminal_evaluation": "Run/Terminal_Step" in summary,
        })
    return records


def main():
    launch = json.loads((OUT / "launch.json").read_text())
    for path, expected_hash in launch["source_sha256"].items():
        assert sha(ROOT / path) == expected_hash, f"Training source changed: {path}"
    all_rows = read_csv(BASE / "run_summary.csv")
    rows = [r for r in all_rows if r["environment"] == ENV]
    expected = {(ENV, model, seed) for model in MODELS for seed in SEEDS}
    counts = Counter(map(key, rows))
    assert set(counts) == expected and set(counts.values()) == {1}, counts
    assert all(r["status"] == "complete" for r in rows)

    # Verify that every pre-existing row is intact and only the two allowed
    # cells were appended, including no changes to the larger-grid pilot.
    before = read_csv(OUT / "before_run_summary.csv")
    assert all_rows[:len(before)] == before
    assert [key(r) for r in all_rows[len(before):]] == [
        (ENV, "PPO-GTrXL", 123), (ENV, "PPO-GTrXL", 456)]
    old_trajectory = (OUT / "before_training_trajectory.csv").read_bytes()
    assert (BASE / "training_trajectory.csv").read_bytes().startswith(old_trajectory)

    trajectories = defaultdict(list)
    for row in read_csv(BASE / "training_trajectory.csv"):
        trajectories[key(row)].append(row)
    offline = offline_records()
    provenance = []
    common_config = None
    for row in rows:
        run_key = key(row)
        attempts = [[]]
        for point in trajectories[run_key]:
            if attempts[-1] and int(point["step"]) <= int(attempts[-1][-1]["step"]):
                attempts.append([])
            attempts[-1].append(point)
        final = attempts[-1]
        terminal = int(row["terminal_step"])
        assert [int(p["step"]) for p in final] == list(range(BLOCK, terminal + 1, BLOCK)), run_key
        assert float(final[-1]["train_return"]) == float(row["train_return"])
        streak = 0
        first_stop = None
        for point in final:
            streak = streak + 1 if float(point["train_return"]) >= 0.96 else 0
            if streak >= 3:
                first_stop = int(point["step"])
                break
        if row["stop_reason"] == "early_stop":
            assert first_stop == terminal and terminal <= CAP, run_key
        else:
            assert row["stop_reason"] == "step_cap" and terminal == CAP and first_stop is None, run_key
        for metric in ("eval_mean_return", "eval_success_rate", "eval_completion_rate"):
            assert math.isfinite(float(row[metric])) and 0 <= float(row[metric]) <= 1
        assert float(row["eval_completion_rate"]) == 1.0
        assert float(row["eval_success_rate"]) * 256 == round(float(row["eval_success_rate"]) * 256)

        complete_offline = [r for r in offline[run_key] if r["has_terminal_evaluation"]]
        assert len(complete_offline) == 1, (run_key, complete_offline)
        record = complete_offline[0]
        for csv_name, wb_name in {
            "eval_mean_return": "Eval/Mean_Return",
            "eval_success_rate": "Eval/Success_Rate",
            "eval_completion_rate": "Eval/Completion_Rate",
            "eval_mean_length": "Eval/Mean_Length",
            "terminal_step": "Run/Terminal_Step",
            "parameter_count": "Run/Parameter_Count",
        }.items():
            assert math.isclose(float(row[csv_name]), record["summary"][wb_name], rel_tol=1e-12, abs_tol=1e-12), (run_key, csv_name)
        config = {k: v for k, v in record["config"].items() if k not in ("SEED", "MODEL", "_wandb")}
        if common_config is None:
            common_config = config
        assert config == common_config, run_key
        provenance.append({**row, "evaluation_seed": 100_000 + int(row["seed"]),
            "trajectory_attempt_lengths": [len(a) for a in attempts],
            "retained_trajectory_attempt": len(attempts),
            "offline_attempts": offline[run_key]})

    aggregates = []
    for model in MODELS:
        group = [r for r in rows if r["model"] == model]
        params = {int(r["parameter_count"]) for r in group}
        assert len(params) == 1
        result = {"model": model, "n": len(group), "parameter_count": params.pop(),
            "early_stop": sum(r["stop_reason"] == "early_stop" for r in group),
            "step_cap": sum(r["stop_reason"] == "step_cap" for r in group)}
        for metric in ("eval_mean_return", "eval_success_rate", "terminal_step"):
            values = [float(r[metric]) for r in group]
            result[metric + "_mean"] = statistics.mean(values)
            result[metric + "_sample_sd"] = statistics.stdev(values)
        aggregates.append(result)

    # Nothing is published until every validation above succeeds.
    OUT.mkdir(exist_ok=True)
    for filename, data in (("five_seed_summary.csv", aggregates), ("five_seed_runs.csv", sorted(rows, key=lambda r: (MODELS.index(r["model"]), int(r["seed"]))))):
        with (OUT / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    report = {"verified_utc": datetime.now(timezone.utc).isoformat(),
        "environment": ENV, "seeds": SEEDS, "n_completed": len(rows), "duplicate_completed_rows": 0,
        "previous_rows_unchanged": True, "new_cells_only": [["PPO-GTrXL", 123], ["PPO-GTrXL", 456]],
        "stopping_rule_verified_against_trajectories": True, "offline_terminal_metrics_match": True,
        "statistics": "Arithmetic mean and sample SD (n-1), across five runs per model",
        "common_config": common_config, "aggregates": aggregates, "runs": provenance,
        "input_sha256": {name: sha(BASE / name) for name in ("run_summary.csv", "training_trajectory.csv", "manifest.json")}}
    (OUT / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    summary_tex = [r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Model & Parameters & Runs & E/C & Eval. return & Success rate \\", r"\midrule"]
    main_tex = [r"\begin{tabular}{lrrr}", r"\toprule",
        r"Model & Eval. return & Success rate & E/C \\", r"\midrule"]
    md = ["# Verified five-seed DoorKey-8x8 study", "", f"Verified UTC: {report['verified_utc']}", "",
        "Exactly 15 completed rows: five unique seeds (42, 54, 99, 123, 456) per model. All original rows and trajectory bytes were preserved; only GTrXL seeds 123 and 456 were added.", "",
        "Mean ± sample SD across runs; E/C counts early-stop / step-cap outcomes. Success is the fraction of 256 terminal-policy evaluation episodes with positive return.", "",
        "| Model | Parameters | Eval. return | Success rate | E/C |", "|---|---:|---:|---:|---:|"]
    for row in aggregates:
        ret = f"{row['eval_mean_return_mean']:.3f}\\pm{row['eval_mean_return_sample_sd']:.3f}"
        success = f"{row['eval_success_rate_mean']:.3f}\\pm{row['eval_success_rate_sample_sd']:.3f}"
        ec = f"{row['early_stop']}/{row['step_cap']}"
        summary_tex.append(f"{row['model']} & {row['parameter_count']:,} & 5/5 & {ec} & ${ret}$ & ${success}$ " + r"\\")
        main_tex.append(f"{row['model']} & ${ret}$ & ${success}$ & {ec} " + r"\\")
        md.append(f"| {row['model']} | {row['parameter_count']:,} | {ret.replace(chr(92)+'pm', ' ± ')} | {success.replace(chr(92)+'pm', ' ± ')} | {ec} |")
    for name, lines in (("doorkey_8x8_summary.tex", summary_tex), ("doorkey_8x8_main.tex", main_tex)):
        (OUT / name).write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    run_tex = [r"\begin{tabular}{lrrlrrl}", r"\toprule",
        r"Model & Seed & Steps (M) & Stop & Eval. return & Success & Offline run ID \\", r"\midrule"]
    for row in sorted(provenance, key=lambda r: (MODELS.index(r["model"]), int(r["seed"]))):
        completed = next(r for r in row["offline_attempts"] if r["has_terminal_evaluation"])
        run_id = completed["directory"].rsplit("-", 1)[-1]
        stop = "early" if row["stop_reason"] == "early_stop" else "cap"
        run_tex.append(f"{row['model']} & {row['seed']} & {int(row['terminal_step']) / 1e6:.5f} & {stop} & {float(row['eval_mean_return']):.6f} & {float(row['eval_success_rate']):.6f} & \\texttt{{{run_id}}} " + r"\\")
    (OUT / "doorkey_8x8_runs.tex").write_text("\n".join(run_tex + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    md += ["", "## Verification and provenance", "",
        "- Every terminal metric agrees with the unique completed offline W&B record for that model/seed.",
        "- Every terminal step and early-stop/cap decision agrees with the final training attempt's trajectory. Cap runs end at 19,988,480 steps because logging blocks contain 327,680 steps.",
        "- The shared configuration agrees across all 15 offline runs. Evaluation uses stochastic actions, 256 first episodes, and PRNG seed 100,000 + training seed; all evaluation episodes completed.",
        "- Raw Exact seed-456 trajectories retain an earlier 40-block interrupted attempt followed by the 61-block completed attempt. The earlier attempt is not an additional statistical sample. Offline records also retain interrupted GTrXL attempts without terminal evaluations.",
        "- Full-precision results: five_seed_summary.csv and five_seed_runs.csv. Run-level configs, offline paths/hashes, metrics and attempt histories: verification.json. Launch metadata, source snapshots/hashes, package list, hardware record and supervisor output are retained alongside them.",
        "- DoorKey-16x16 pilot rows are unchanged and excluded. No DoorKey-10x10, 12x12, or 16x16 runs were launched in this completion pass.", "",
        "## Interpretation", "",
        "The comparison shares encoder, PPO settings, rollout length, evaluation and stopping rule, but capacities differ. Five-seed means and SDs describe the recorded configurations; they do not establish statistical superiority, a causal memory-mechanism effect, equal training compute, or a global ranking. The separate MemoryS16 SWA intervention is not replaced."]
    (OUT / "report.md").write_text("\n".join(md) + "\n")
    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    main()
