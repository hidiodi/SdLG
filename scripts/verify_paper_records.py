"""Verify that compact imported records reproduce headline paper values.

This is intentionally a record-level release gate, not a claim that a new PPO
training run is bitwise identical to a historical run.  Full-run comparisons
use the same aggregations and study protocols recorded in the experiment
ledger.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "paper-records"


def read_csv(relative: str) -> list[dict[str, str]]:
    with (DATA / relative).open(newline="") as stream:
        return list(csv.DictReader(stream))


def mean(rows: list[dict[str, str]], key: str) -> float:
    return statistics.mean(float(row[key]) for row in rows)


def require_close(label: str, actual: float, expected: float, tolerance: float = 5e-4) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{label}: expected {expected:.6f}, got {actual:.6f}")
    print(f"PASS  {label}: {actual:.6f}")


def verify_tmaze() -> None:
    rows = read_csv("tmaze_rank_rope_zero_shot.csv")
    exact = [row for row in rows if row["model"] == "Exact_NoRoPE"]
    approx = [row for row in rows if row["model"] == "Approx_NoRoPE"]
    all_exact = [row for row in rows if row["model"].startswith("Exact_")]
    all_approx = [row for row in rows if row["model"].startswith("Approx_")]
    require_close("T-Maze Exact-NoRoPE H=100", mean(exact, "val_H100"), 0.900)
    require_close("T-Maze Exact-NoRoPE H=150", mean(exact, "val_H150"), 0.900)
    require_close("T-Maze Exact-NoRoPE H=200", mean(exact, "val_H200"), 0.851)
    require_close(
        "T-Maze historical Exact-minus-Approximate H=200",
        mean(all_exact, "val_H200") - mean(all_approx, "val_H200"),
        0.251,
    )


def verify_tmaze_external_landscape() -> None:
    rows = read_csv("tmaze_external_landscape.csv")
    config_path = ROOT / "configs" / "studies" / "tmaze_external_landscape.json"
    config = json.loads(config_path.read_text())
    families = {
        "GRU", "LSTM", "Mamba", "LTI-Linear", "Markovian", "GLA", "GTrXL-64", "GTrXL-128"
    }
    observed = {row["family"] for row in rows}
    if observed != families:
        raise AssertionError(f"T-Maze external landscape families differ: {sorted(observed)}")
    if set(config["families"]) != families:
        raise AssertionError("T-Maze external landscape configuration families differ from the frozen record")
    seeds = {int(row["seed"]) for row in rows}
    if set(config["seeds"]) != seeds:
        raise AssertionError("T-Maze external landscape configuration seeds differ from the frozen record")
    if config["training"]["total_environment_steps"] != 50_000_000:
        raise AssertionError("T-Maze external landscape configuration has the wrong training budget")
    if len(rows) != 56 or any(sum(row["family"] == family for row in rows) != 7 for family in families):
        raise AssertionError("T-Maze external landscape must contain seven terminal records for each family")
    valid_statuses = {"early_stop", "budget_cap"}
    if any(row["terminal_status"] not in valid_statuses or int(row["terminal_steps"]) > 49_971_200 for row in rows):
        raise AssertionError("T-Maze external landscape terminal-policy record is invalid")
    budget_cap = sum(row["terminal_status"] == "budget_cap" for row in rows)
    print(f"PASS  T-Maze external landscape: 8 families x 7 terminal records ({budget_cap} budget-cap, {len(rows) - budget_cap} early-stop)")


def verify_minigrid() -> None:
    rows = read_csv("revision_2026/minigrid_locality_seed_endpoints.csv")
    swa = [row for row in rows if row["Condition"] == "SdLG-SWA"]
    bypass = [row for row in rows if row["Condition"] == "SdLG-None"]
    require_close("MiniGrid SWA endpoint", mean(swa, "Eval_Return"), 0.508)
    require_close("MiniGrid bypass endpoint", mean(bypass, "Eval_Return"), 0.782)
    require_close("MiniGrid enabled-minus-bypassed", mean(swa, "Eval_Return") - mean(bypass, "Eval_Return"), -0.274)


def verify_doorkey() -> None:
    rows = read_csv("doorkey_8x8/five_seed_summary.csv")
    exact = next(row for row in rows if row["model"] == "SdLG-Exact")
    gru = next(row for row in rows if row["model"] == "PPO-GRU")
    gtrxl = next(row for row in rows if row["model"] == "PPO-GTrXL")
    require_close("DoorKey Exact terminal return", float(exact["eval_mean_return_mean"]), 0.668)
    require_close("DoorKey GRU terminal return", float(gru["eval_mean_return_mean"]), 0.193)
    require_close("DoorKey GTrXL terminal return", float(gtrxl["eval_mean_return_mean"]), 0.468)
    if (int(exact["early_stop"]), int(gru["early_stop"]), int(gtrxl["early_stop"])) != (3, 1, 2):
        raise AssertionError("DoorKey early-stop counts differ from the paper record")
    print("PASS  DoorKey early-stop counts: Exact=3, GRU=1, GTrXL=2")


def verify_popgym() -> None:
    # The paper's scope examples use the historical three-seed subset.  The
    # larger per-seed export also contains later Exact Concentration runs and
    # must not be silently pooled into this reported comparison.
    rows = read_csv("final_verification/popgym_scope_examples.csv")

    def score(environment: str, model: str) -> float:
        return float(next(row for row in rows if row["Environment"] == environment and row["Model"] == model)["mean"])

    require_close("POPGym CountRecallEasy Exact MMER", score("CountRecallEasy", "SdLG-Exact"), 0.670)
    require_close("POPGym CountRecallEasy Approximate MMER", score("CountRecallEasy", "SdLG-Approx"), -0.712)
    require_close("POPGym ConcentrationEasy Exact MMER", score("ConcentrationEasy", "SdLG-Exact"), -0.267)
    require_close("POPGym ConcentrationEasy Approximate MMER", score("ConcentrationEasy", "SdLG-Approx"), -0.102)


def verify_craftax() -> None:
    rows = read_csv("craftax_cross_t_verified.csv")
    cross_t_config = json.loads((ROOT / "configs" / "studies" / "craftax_cross_t.json").read_text())
    expected_cells = {(family, int(length)) for family, length in cross_t_config["reported_cells"]}
    observed_cells = {(row["Family"], int(row["T"])) for row in rows}
    if observed_cells != expected_cells:
        raise AssertionError("Craftax cross-T configuration cells differ from the frozen record")
    if any(int(row["Seeds"]) != cross_t_config["training"]["runs_per_cell"] or int(row["TrainingSteps"]) != cross_t_config["training"]["total_environment_steps"] for row in rows):
        raise AssertionError("Craftax cross-T configuration budget differs from the frozen record")

    landscape = read_csv("original_paper_upgrade/craftax_t64_seven_family_performance.csv")
    landscape_config = json.loads((ROOT / "configs" / "studies" / "craftax_t64_landscape.json").read_text())
    if {row["family"] for row in landscape} != set(landscape_config["families"]):
        raise AssertionError("Craftax T=64 configuration families differ from the frozen record")

    def cell(family: str, rollout: str) -> dict[str, str]:
        return next(row for row in rows if row["Family"] == family and row["T"] == rollout)

    require_close("Craftax Approximate T=16", float(cell("SdLG-Approx", "16")["MeanReturn"]), 16.330)
    require_close("Craftax Approximate T=64", float(cell("SdLG-Approx", "64")["MeanReturn"]), 14.790)
    # The manuscript prints this cell to two decimal places.
    require_close("Craftax GLA T=64", float(cell("PPO-GLA", "64")["MeanReturn"]), 15.690, tolerance=0.005)
    print("PASS  Craftax launch configurations match the frozen seven-family and cross-T matrices")


def verify_ggs() -> None:
    rows = read_csv("ggs_endpoint_paired_difference.csv")
    require_close("GGS alpha=0.1 endpoint", mean(rows, "alpha_0_1_eval_return"), 11.210)
    require_close("GGS alpha=1.0 endpoint", mean(rows, "alpha_1_0_eval_return"), 9.757)
    require_close("GGS paired endpoint difference", mean(rows, "paired_difference"), 1.453)


def verify_diagnostics() -> None:
    capacity = read_csv("synthetic_capacity_480_cells.csv")
    if len(capacity) != 480:
        raise AssertionError(f"Capacity diagnostic: expected 480 rows, got {len(capacity)}")
    print("PASS  Associative-capacity record count: 480")

    overwrite = read_csv("component_transfer/associative_overwrite_raw.csv")
    additive = [row for row in overwrite if row["formulation"] == "additive"]
    delta_quarter = [
        row
        for row in overwrite
        if row["formulation"] == "normalized_delta" and row["delta_eta"] == "0.25"
    ]
    require_close("Overwrite additive target fidelity", mean(additive, "target_overwrite_fidelity"), 0.502)
    require_close("Overwrite normalized-Delta target fidelity", mean(delta_quarter, "target_overwrite_fidelity"), 0.685)
    require_close(
        "Overwrite normalized-Delta minus additive target fidelity",
        mean(delta_quarter, "target_overwrite_fidelity") - mean(additive, "target_overwrite_fidelity"),
        0.183,
    )

    inference = read_csv("component_transfer/matched_tmaze_inference.csv")
    exact = next(row for row in inference if row["architecture"] == "Matched_Exact_NoRoPE" and row["batch_size"] == "256")
    approx = next(row for row in inference if row["architecture"] == "Matched_Approx_NoRoPE" and row["batch_size"] == "256")
    reduction = 1.0 - float(approx["carry_bytes_per_environment"]) / float(exact["carry_bytes_per_environment"])
    require_close("Matched-state carry reduction", reduction, 0.607)
    if any(int(row["timed_iterations"]) != 300 for row in inference):
        raise AssertionError("Matched-state accounting must use 300 timed iterations per cell")
    print("PASS  Matched-state timing count: 300 per cell")


def main() -> None:
    verify_tmaze()
    verify_tmaze_external_landscape()
    verify_minigrid()
    verify_doorkey()
    verify_popgym()
    verify_craftax()
    verify_ggs()
    verify_diagnostics()
    print("All record-level headline checks passed.")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, FileNotFoundError, StopIteration) as error:
        print(f"FAILED  {error}", file=sys.stderr)
        raise SystemExit(1)
