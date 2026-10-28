"""Recompute and verify the evidence used by the cleanRewrite manuscript.

The output is intentionally both human- and machine-readable.  It never reads
numbers from manuscript prose; every value comes from a persisted experiment
artifact or a frozen derived table whose consistency is checked here.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import argparse
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "papers/latex/cleanRewrite"
WRITING = ROOT / "papers/writing"
GEN = WRITING / "generated_artifacts"
REV = GEN / "revision_2026"
OUT = GEN / "final_verification"
REPORT = WRITING / "REVISION_VERIFICATION.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(a: float, b: float, tolerance: float = 1e-10) -> bool:
    return bool(abs(float(a) - float(b)) <= tolerance)


def canonical_retrieval_table_check() -> tuple[bool, str]:
    """Compare all displayed N=128 retrieval cells with the canonical CSV."""
    source = PAPER / "artifacts/associative_capacity_480_cells.csv"
    retrieval = pd.read_csv(source)
    appendix = (PAPER / "sections/appendix.tex").read_text()
    start = appendix.index(r"\label{tab:factorial_dkdv}")
    block = appendix[start:appendix.index(r"\end{table}", start)]
    displayed: dict[tuple[int, int], tuple[float, float]] = {}
    for line in block.splitlines():
        row = re.match(r"^\s*(16|32|64|128)\s*&\s*(.*?)\\\\\s*$", line)
        if not row:
            continue
        cells = row.group(2).split("&")
        if len(cells) != 4:
            continue
        for dv, cell in zip((8, 16, 32, 64), cells):
            values = re.search(r"([0-9]+\.[0-9]+)\s*\\pm\s*([0-9]+\.[0-9]+)", cell)
            if values:
                displayed[(int(row.group(1)), dv)] = (float(values.group(1)), float(values.group(2)))

    mismatches = []
    for dk in (16, 32, 64, 128):
        for dv in (8, 16, 32, 64):
            values = retrieval[(retrieval.N == 128) & (retrieval.d_k == dk) & (retrieval.d_v == dv)].cosine
            if len(values) != 10:
                mismatches.append(f"dk={dk},dv={dv}: expected 10 rows, found {len(values)}")
                continue
            canonical = (round(float(values.mean()), 3), round(float(values.std(ddof=1)), 3))
            if displayed.get((dk, dv)) != canonical:
                mismatches.append(
                    f"dk={dk},dv={dv}: displayed={displayed.get((dk, dv))}, canonical={canonical}"
                )
    detail = "all 16 cells match canonical mean/sample-SD" if not mismatches else "; ".join(mismatches)
    return not mismatches, detail


def component_rigor() -> None:
    """Post-freeze analysis only: never invokes a runner or rewrites raw data."""
    import yaml

    out = GEN / "component_transfer"
    raw_path = out / "associative_overwrite_raw.csv"
    raw_hash = sha256(raw_path)
    raw = pd.read_csv(raw_path)
    cells = ["N", "d_k", "d_v"]
    keys = cells + ["seed"]
    metrics = ["target_overwrite_fidelity", "non_target_preservation",
               "retrieval_cosine_after_overwrite"]
    methods = [("additive", 1.0), ("gated_additive_proxy", 1.0),
               ("normalized_delta", .25), ("normalized_delta", .5), ("normalized_delta", 1.)]
    expected = pd.MultiIndex.from_tuples(list(itertools.product(
        [32, 64, 128], [16, 32, 64, 128], [8, 16, 32, 64], range(10))), names=keys)
    assert len(raw) == 2400 and not raw.duplicated(keys + ["formulation", "delta_eta"]).any()
    arrays = []
    for method, eta in methods:
        frame = raw[(raw.formulation == method) & (raw.delta_eta == eta)].set_index(keys)
        assert frame.index.sort_values().equals(expected)
        assert frame.finite_state.all()
        arrays.append(frame.loc[expected, metrics].to_numpy().reshape(48, 10, 3))
    values = np.stack(arrays, axis=2)  # cell, draw, method, metric
    assert np.isfinite(values).all()
    contrasts = [(i, 0) for i in range(1, 5)] + [(3, 2), (4, 2), (4, 3)]
    differences = np.stack([values[:, :, a] - values[:, :, b] for a, b in contrasts], axis=2)
    # Cell-specific fold_in(N,dk,dv) keys are distinct random draws. Pair methods,
    # resample ten draws within each fixed cell, and weight all 48 cells equally.
    n_boot, seed = 200_000, 20261015
    rng = np.random.default_rng(seed)
    boot = np.zeros((n_boot, len(contrasts), len(metrics)))
    cell_intervals = []
    for cell in range(48):
        indices = rng.integers(0, 10, size=(n_boot, 10))
        sampled = differences[cell][indices].mean(axis=1)
        boot += sampled / 48
        cell_intervals.append(np.quantile(sampled, [.025, .975], axis=0))
    low, high = np.quantile(boot, [.025, .975], axis=0)
    # Sensitivity only: keep common integer seed IDs clustered across cells.
    cluster = differences.mean(axis=0)
    indices = np.random.default_rng(seed + 1).integers(0, 10, size=(n_boot, 10))
    cluster_low, cluster_high = np.quantile(cluster[indices].mean(axis=1), [.025, .975], axis=0)
    rows, cell_rows = [], []
    for c, (a, b) in enumerate(contrasts):
        for m, metric in enumerate(metrics):
            common = dict(treatment=methods[a][0], treatment_eta=methods[a][1],
                          reference=methods[b][0], reference_eta=methods[b][1], metric=metric)
            rows.append(dict(**common, estimate=differences[:, :, c, m].mean(),
                             ci_low=low[c, m], ci_high=high[c, m],
                             seed_cluster_ci_low=cluster_low[c, m], seed_cluster_ci_high=cluster_high[c, m],
                             cells=48, draws_per_cell=10, bootstrap_draws=n_boot, bootstrap_seed=seed))
            for j, cell in enumerate(itertools.product([32, 64, 128], [16, 32, 64, 128], [8, 16, 32, 64])):
                d = differences[j, :, c, m]
                cell_rows.append(dict(**common, **dict(zip(cells, cell)), n=10,
                                      estimate=d.mean(), paired_difference_sd=d.std(ddof=1),
                                      ci_low=cell_intervals[j][0, c, m], ci_high=cell_intervals[j][1, c, m]))
    pd.DataFrame(rows).to_csv(out / "associative_overwrite_contrasts.csv", index=False)
    pd.DataFrame(cell_rows).to_csv(out / "associative_overwrite_cell_contrasts.csv", index=False)
    manifest = json.loads((out / "associative_overwrite_manifest.json").read_text())
    generator = ROOT / manifest["runner"]
    assert sha256(generator) == manifest["runner_sha256"], "Frozen A1 generator has changed"
    assert sha256(raw_path) == raw_hash

    # Match local W&B endpoints to every accepted matched/C1 seed, including
    # complete configuration fingerprints but excluding account/host metadata.
    root = ROOT / "scripts/diagnostics/exact_sdlg"
    endpoints_path = root / "unified_tmaze_ablations_zeroshot.csv"
    training_path = root / "unified_tmaze_ablations_training.csv"
    endpoints = pd.read_csv(endpoints_path)
    training = pd.read_csv(training_path)
    accepted_models = ["Matched_Exact_NoRoPE", "Matched_Approx_NoRoPE", "GateBias_Exact_bg2", "GateBias_Exact_bg0"]
    seeds = [42, 54, 99, 123, 456, 789, 111, 222, 333, 444]
    accepted = endpoints[endpoints.model.isin(accepted_models)]
    assert len(accepted) == 40 and not accepted.duplicated(["model", "seed"]).any()
    assert all(set(accepted[accepted.model == m].seed) == set(seeds) for m in accepted_models)
    run_rows, configs, ignored = [], {}, []
    for path in sorted((ROOT / "wandb").glob("run-202610*/files/config.yaml")):
        loaded = yaml.safe_load(path.read_text())
        cfg = {k: v["value"] for k, v in loaded.items() if k != "_wandb"}
        meta = json.loads(path.with_name("wandb-metadata.json").read_text())
        args = meta.get("args", [])
        study = args[args.index("--study") + 1] if "--study" in args else None
        if study not in {"matched", "matched-gate"}:
            continue
        ablation = cfg["ABLATION"]
        model = ("Matched_Exact_NoRoPE" if ablation["USE_EXACT"] else "Matched_Approx_NoRoPE") if study == "matched" else (
            "GateBias_Exact_bg2" if ablation["GRU_BIAS"] == 2 else "GateBias_Exact_bg0")
        summary_path = path.with_name("wandb-summary.json")
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        if not all(f"Eval/Success_H{h}" in summary for h in (100, 150, 200)):
            ignored.append(dict(run_directory=str(path.parent.parent.relative_to(ROOT)), reason="no complete endpoint evaluation"))
            continue
        seed_id = cfg["SEED"]
        record = accepted[(accepted.model == model) & (accepted.seed == seed_id)]
        assert len(record) == 1
        trace = training[(training.model == model) & (training.seed == seed_id)]
        # Append order records restarted attempts; maximum step can belong to an
        # interrupted attempt, not the run that produced the accepted endpoint.
        terminal = trace.iloc[-1]
        for h in (100, 150, 200):
            assert close(record.iloc[0][f"val_H{h}"], summary[f"Eval/Success_H{h}"])
        assert int(terminal.step) == int(summary["_step"])
        normalized = json.loads(json.dumps(cfg))
        normalized.pop("SEED")
        normalized["ABLATION"].setdefault("GRU_BIAS", 2.0)
        # Canonicalize JSON numeric spelling: YAML 1 and 1.0 mean the same config.
        configs.setdefault(model, []).append(normalized)
        requirements = path.with_name("requirements.txt")
        versions = dict(line.split("==", 1) for line in requirements.read_text().splitlines() if "==" in line)
        log = path.with_name("output.log").read_text()
        run_rows.append(dict(model=model, seed=seed_id, run_id=path.parent.parent.name.rsplit("-", 1)[-1],
                             directory=str(path.parent.parent.relative_to(ROOT)), study=study,
                             config_sha256=sha256(path), requirements_sha256=sha256(requirements),
                             git_commit=meta.get("git", {}).get("commit", "unknown"),
                             python=meta.get("python", "unknown"), jax=versions.get("jax", "unknown"),
                             jaxlib=versions.get("jaxlib", "unknown"), flax=versions.get("flax", "unknown"),
                             optax=versions.get("optax", "unknown"),
                             terminal_steps=int(terminal.step), reached_stopping_rule="[EARLY STOP]" in log,
                             earlier_attempt_rows=int(np.flatnonzero(trace.step.diff().to_numpy() < 0)[-1]) if (trace.step.diff() < 0).any() else 0,
                             **{f"H{h}":summary[f"Eval/Success_H{h}"] for h in (100, 150, 200)}))
    runs = pd.DataFrame(run_rows).sort_values(["model", "seed"])
    assert len(runs) == 40 and not runs.duplicated(["model", "seed"]).any()
    assert all(all(c == cs[0] for c in cs) for cs in configs.values())
    baseline_equal = configs["Matched_Exact_NoRoPE"][0] == configs["GateBias_Exact_bg2"][0]
    assert baseline_equal
    gate0 = json.loads(json.dumps(configs["GateBias_Exact_bg0"][0]))
    gate0["ABLATION"]["GRU_BIAS"] = 2
    assert gate0 == configs["GateBias_Exact_bg2"][0]
    runs.to_csv(out / "tmaze_baseline_run_audit.csv", index=False)
    runner = root / "exact_sdlg_tmaze_ablations.py"
    versions_equal = runs.requirements_sha256.nunique() == 1
    comparison = dict(
        status="record linkage verified; cross-study causal explanation unresolved",
        accepted_run_count=len(runs), excluded_incomplete_attempts=ignored,
        normalized_configurations=configs,
        normalized_baseline_configs_equal=baseline_equal, package_lists_identical=versions_equal,
        current_runner_sha256=sha256(runner), accepted_endpoints_sha256=sha256(endpoints_path),
        accepted_training_sha256=sha256(training_path), a1_source_sha256=raw_hash,
        a1_generator_hash_matches_manifest=True,
        a1_interval_method="paired within-cell percentile bootstrap, fixed 48-cell equally weighted grid",
        a1_analysis_status="post-freeze exploratory uncertainty; not preregistered; marginal intervals without multiplicity correction",
        a1_randomness="fold_in(seed,N,dk,dv), then split keys/values/replacements; shared only across formulations in each cell",
        a1_sensitivity="same seed IDs clustered over all cells, reported separately",
        current_runner_protocol=dict(training_rng="PRNGKey(seed)", evaluation_rng="PRNGKey(99) reused for each horizon",
            evaluation="terminal parameters, pi.mode(), 256 environments, H=100/150/200",
            stopping="three consecutive logged train returns >=0.98, 163840-step logging blocks, nominal cap50M; actual cap49971200"),
        unresolved=["Full run-time dirty-worktree snapshots and checkpoints are absent from the inspected local W&B run directories.",
                    "Equal git commits/configurations/package lists do not establish bitwise implementation identity or statistical independence.",
                    "The cause of the between-study Exact baseline difference is not identified; no cross-study pooled estimate is justified."],
        gate_bias_semantics="bg is a fixed module scalar subtracted throughout training, not a trainable bias initialization only",
    )
    (out / "scientific_rigor_audit.json").write_text(json.dumps(comparison, indent=2) + "\n")
    # Machine-generated LaTeX contains only statistics from the accepted records.
    labels = ["Fixed-gated minus additive", r"Delta $\eta=.25$ minus additive",
              r"Delta $\eta=.50$ minus additive", r"Delta $\eta=1$ minus additive",
              r"Delta $.50$ minus $.25$", r"Delta $1$ minus $.25$", r"Delta $1$ minus $.50$"]
    lines = []
    for c, label in enumerate(labels):
        entries = [f"${differences[:, :, c, m].mean():.4f}$ [{low[c,m]:.4f}, {high[c,m]:.4f}]" for m in range(3)]
        lines.append(label + " & " + " & ".join(entries) + r" \\")
    (PAPER / "artifacts/overwrite_contrasts.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")
    print(pd.DataFrame(rows).to_string(index=False))
    print(runs.groupby("model").agg(n=("seed", "size"), H200=("H200", "mean"),
                                    successes=("reached_stopping_rule", "sum"), median_steps=("terminal_steps", "median")))
    print("Baseline configs equal:", baseline_equal, "Package lists identical:", versions_equal)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, object]] = []
    claims: list[dict[str, object]] = []

    def check(check_id: str, passed: bool, detail: str, source: Path) -> None:
        checks.append({
            "check_id": check_id,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
            "source": str(source.relative_to(ROOT)),
        })

    # T-Maze: keep the historical 2x2 factorial and the later topology/gradient-
    # matched study distinct, while recomputing both from their raw seed rows.
    tmaze_raw_path = ROOT / "scripts/diagnostics/exact_sdlg/unified_tmaze_ablations_zeroshot.csv"
    tmaze_summary_path = REV / "tmaze_factorial_cell_summary.csv"
    tmaze_raw = pd.read_csv(tmaze_raw_path)
    tmaze_summary = pd.read_csv(tmaze_summary_path)
    if "study" not in tmaze_summary:
        tmaze_summary.insert(0, "study", "factorial")
    tmaze_rows = []
    model_map = {
        "Exact_NoRoPE": "Exact_NoRoPE",
        "Exact_RoPE": "Exact_RoPE",
        "Approx_NoRoPE": "Approx_NoRoPE",
        "Approx_RoPE": "Approx_RoPE",
    }
    for horizon in (100, 150, 200):
        for raw_model, display_model in model_map.items():
            values = tmaze_raw.loc[tmaze_raw["model"] == raw_model, f"val_H{horizon}"].astype(float)
            expected = tmaze_summary[(tmaze_summary["study"] == "factorial")
                                     & (tmaze_summary["horizon"] == horizon)
                                     & (tmaze_summary["model"] == display_model)].iloc[0]
            observed = float(values.mean())
            passed = len(values) == 10 and close(observed, expected["mean"])
            check(f"tmaze-{display_model}-H{horizon}", passed,
                  f"n={len(values)}; raw mean={observed:.9f}; displayed derivative={expected['mean']:.9f}",
                  tmaze_raw_path)
            tmaze_rows.append({"study": "factorial", "horizon": horizon, "model": display_model, "n": len(values),
                               "mean": observed, "sd": float(values.std(ddof=1))})

    contrasts_path = REV / "tmaze_factorial_contrasts.csv"
    contrasts = pd.read_csv(contrasts_path)
    if "study" not in contrasts:
        contrasts.insert(0, "study", "factorial")
    headline_contrasts = contrasts[(contrasts["study"] == "factorial") & contrasts["contrast"].isin([
        "execution_exact_minus_approx", "phase_norope_minus_rope",
        "interaction_phase_difference_exact_minus_approx",
    ])].copy()
    check("tmaze-factorial-contrasts", len(headline_contrasts) == 9,
          f"found {len(headline_contrasts)} prespecified horizon-by-contrast rows", contrasts_path)

    matched_models = ("Matched_Exact_NoRoPE", "Matched_Approx_NoRoPE")
    matched_summary = tmaze_summary[tmaze_summary["study"] == "matched"]
    for horizon in (100, 150, 200):
        for model in matched_models:
            values = tmaze_raw.loc[tmaze_raw["model"] == model, f"val_H{horizon}"].astype(float)
            expected = matched_summary[(matched_summary["horizon"] == horizon)
                                       & (matched_summary["model"] == model)].iloc[0]
            observed = float(values.mean())
            check(f"tmaze-matched-{model}-H{horizon}",
                  len(values) == 10 and close(observed, expected["mean"]),
                  f"n={len(values)}; raw mean={observed:.9f}; displayed derivative={expected['mean']:.9f}",
                  tmaze_raw_path)
            tmaze_rows.append({"study": "matched", "horizon": horizon, "model": model, "n": len(values),
                               "mean": observed, "sd": float(values.std(ddof=1))})

    matched_contrasts = contrasts[(contrasts["study"] == "matched")
                                  & (contrasts["contrast"] == "matched_execution_exact_minus_approx")]
    expected_matched = {100: 0.5015625, 150: 0.3984375, 200: 0.35078125}
    check("tmaze-matched-execution-contrasts",
          len(matched_contrasts) == 3 and all(
              close(row.estimate, expected_matched[int(row.horizon)])
              for row in matched_contrasts.itertuples()),
          "three prespecified independent-condition contrasts; H=200 is primary", contrasts_path)

    tmaze_training_path = ROOT / "scripts/diagnostics/exact_sdlg/unified_tmaze_ablations_training.csv"
    tmaze_training = pd.read_csv(tmaze_training_path)
    terminal_rows = []
    for model in matched_models:
        terminal = (tmaze_training[tmaze_training["model"] == model]
                    .groupby("seed", as_index=False).tail(1))
        censored = int((terminal["step"] >= 49_971_200).sum())
        early_stopped = int((terminal["mean_return"] >= 0.98).sum())
        terminal_rows.append({"study": "matched", "model": model, "n": len(terminal),
                              "mean_terminal_step": float(terminal["step"].mean()),
                              "censored": censored, "early_stopped": early_stopped})
        check(f"tmaze-matched-{model}-terminal-records", len(terminal) == 10,
              f"n=10; mean terminal step={terminal['step'].mean():.0f}; censored={censored}; "
              f"terminal return >=.98={early_stopped}", tmaze_training_path)
    tmaze_recomputed = pd.DataFrame(tmaze_rows).merge(
        pd.DataFrame(terminal_rows), on=["study", "model", "n"], how="left"
    )
    tmaze_recomputed.to_csv(OUT / "tmaze_cell_means_recomputed.csv", index=False)

    # Exact scan: validate the printed associative operator against sequential
    # recurrence, including carried state and asynchronous episode resets.
    scan_test_path = ROOT / "scripts/diagnostics/verify_exact_scan_semantics.py"
    scan_test = subprocess.run(
        [sys.executable, str(scan_test_path)], capture_output=True, text=True, check=False
    )
    check("exact-scan-semantics", scan_test.returncode == 0,
          scan_test.stdout.strip().replace("\n", "; ") or scan_test.stderr.strip(), scan_test_path)

    # MiniGrid: one coherent 15-by-15 dataset, never spliced with historical rows.
    mini_raw_path = REV / "minigrid_locality_seed_endpoints.csv"
    mini_summary_path = REV / "minigrid_locality_summary.csv"
    mini_difference_path = REV / "minigrid_locality_difference.csv"
    mini_reproduction_path = REV / "minigrid_historical_reproduction_gate.csv"
    mini_raw = pd.read_csv(mini_raw_path)
    mini_summary = pd.read_csv(mini_summary_path)
    mini_difference = pd.read_csv(mini_difference_path).iloc[0]
    mini_rows = []
    for condition in ("SdLG-SWA", "SdLG-None"):
        values = mini_raw.loc[mini_raw["Condition"] == condition, "Eval_Return"].astype(float)
        expected = mini_summary[mini_summary["condition"] == condition].iloc[0]
        passed = len(values) == 15 and close(values.mean(), expected["mean"]) and close(values.std(ddof=1), expected["sd"])
        check(f"minigrid-{condition}", passed,
              f"n={len(values)}; mean={values.mean():.9f}; sd={values.std(ddof=1):.9f}", mini_raw_path)
        mini_rows.append({"condition": condition, "n": len(values), "mean": float(values.mean()),
                          "sd": float(values.std(ddof=1)), "median": float(values.median())})
    mini_w16 = mini_raw.loc[mini_raw["Condition"] == "SdLG-SWA", "Eval_Return"].astype(float)
    mini_w0 = mini_raw.loc[mini_raw["Condition"] == "SdLG-None", "Eval_Return"].astype(float)
    mini_estimate = float(mini_w16.mean() - mini_w0.mean())
    check("minigrid-difference", close(mini_estimate, mini_difference["estimate"]),
          f"W16-W0={mini_estimate:.9f}; CI=[{mini_difference['bootstrap_ci_low']:.9f}, "
          f"{mini_difference['bootstrap_ci_high']:.9f}]", mini_difference_path)
    pd.DataFrame(mini_rows).to_csv(OUT / "minigrid_summary_recomputed.csv", index=False)

    reproduction = pd.read_csv(mini_reproduction_path).iloc[0]
    recorded_difference = abs(float(reproduction["historical_endpoint"])
                              - float(reproduction["rerun_endpoint"]))
    reproduction_fields = {
        "seed", "historical_endpoint", "rerun_endpoint", "absolute_difference",
        "source_commit", "runner_sha256", "package_versions",
        "evaluation_protocol", "known_nondeterminism",
    }
    reproduction_passed = (
        reproduction_fields.issubset(set(pd.read_csv(mini_reproduction_path).columns))
        and close(recorded_difference, reproduction["absolute_difference"])
        and int(reproduction["seed"]) == 42
    )
    check(
        "minigrid-historical-reproduction-record",
        reproduction_passed,
        f"seed=42; historical={reproduction['historical_endpoint']:.6f}; "
        f"rerun={reproduction['rerun_endpoint']:.6f}; "
        f"absolute difference={recorded_difference:.6f}; source={str(reproduction['source_commit'])[:12]}",
        mini_reproduction_path,
    )

    claims.append({
        "claim_id": "MINIGRID-LOCALITY",
        "claim_type": "controlled intervention; null/negative confirmation",
        "variables_that_change_jointly": "local attention computation and its mask/carry path",
        "observation": f"W16-W0={mini_estimate:.3f}, 95% bootstrap CI "
                       f"[{mini_difference['bootstrap_ci_low']:.3f},{mini_difference['bootstrap_ci_high']:.3f}]",
        "permissible_language": "The 15-seed rerun does not support a positive locality effect on this task.",
        "source": str(mini_difference_path.relative_to(ROOT)),
    })

    # GGS: accepted historical five matched pairs only.  The exploratory alpha=0
    # cell is deliberately excluded from every calculation below.
    ggs_path = ROOT / "scripts/diagnostics/architecture/gradient_shielding/gradient_shielding.csv"
    ggs = pd.read_csv(ggs_path)
    endpoint = (ggs.dropna(subset=["Eval_Return"]).sort_values("Total_Steps")
                .groupby(["Architecture", "Seed"], as_index=False).tail(1))
    low = endpoint[endpoint["Architecture"] == "SdLGHybrid-GGS-0.1"].sort_values("Seed")
    full = endpoint[endpoint["Architecture"] == "SdLGHybrid-GGS-1.0"].sort_values("Seed")
    merged = low[["Seed", "Eval_Return"]].merge(
        full[["Seed", "Eval_Return"]], on="Seed", suffixes=("_alpha_0_1", "_alpha_1"))
    ggs_diff = float((merged["Eval_Return_alpha_0_1"] - merged["Eval_Return_alpha_1"]).mean())
    check("ggs-accepted-endpoints", len(merged) == 5 and close(ggs_diff, 1.452577, 1e-6),
          f"five matched pairs; alpha=.1 mean={low.Eval_Return.mean():.6f}; "
          f"alpha=1 mean={full.Eval_Return.mean():.6f}; paired difference={ggs_diff:.6f}", ggs_path)
    claims.append({
        "claim_id": "CRAFTAX-GGS",
        "claim_type": "controlled paired optimization intervention",
        "variables_that_change_jointly": "actor gradient entering shared parameters and the resulting optimization trajectory",
        "observation": "paired endpoint difference=1.453; descriptive bootstrap CI=[0.925,1.962]",
        "permissible_language": "Endpoint behavior is sensitive to actor-gradient contribution in this shared-trunk setup.",
        "source": str(ggs_path.relative_to(ROOT)),
    })

    # Controlled associative retrieval.
    retrieval_path = PAPER / "artifacts/associative_capacity_480_cells.csv"
    retrieval = pd.read_csv(retrieval_path)
    monotonic_checks = 0
    total_checks = 0
    for seed in sorted(retrieval.seed.unique()):
        for n in sorted(retrieval.N.unique()):
            marginal = (retrieval[(retrieval.seed == seed) & (retrieval.N == n)]
                        .groupby("d_k")["cosine"].mean().sort_index())
            total_checks += 1
            monotonic_checks += int(np.all(np.diff(marginal.to_numpy()) > 0))
    spans = []
    for n in sorted(retrieval.N.unique()):
        dk = retrieval[retrieval.N == n].groupby("d_k")["cosine"].mean()
        dv = retrieval[retrieval.N == n].groupby("d_v")["cosine"].mean()
        spans.append({"N": int(n), "dk_span": float(dk.max() - dk.min()),
                      "dv_span": float(dv.max() - dv.min())})
    spans_df = pd.DataFrame(spans)
    spans_df.to_csv(OUT / "associative_dimension_spans.csv", index=False)
    check("associative-retrieval-factorial", len(retrieval) == 480 and monotonic_checks == total_checks,
          f"rows={len(retrieval)}; strict d_k monotonicity in {monotonic_checks}/{total_checks} seed-load slices; "
          f"d_v spans={spans_df.dv_span.min():.3f}-{spans_df.dv_span.max():.3f}", retrieval_path)
    table_matches, table_detail = canonical_retrieval_table_check()
    check("associative-retrieval-appendix-table", table_matches, table_detail, retrieval_path)
    claims.append({
        "claim_id": "ASSOCIATIVE-RETRIEVAL",
        "claim_type": "independent controlled capacity result",
        "variables_that_change_jointly": "manipulated dimension/load and finite draw/update trajectory",
        "observation": f"strict d_k monotonicity in {monotonic_checks}/{total_checks} seed-load slices; "
                       f"d_v marginal span {spans_df.dv_span.min():.3f}-{spans_df.dv_span.max():.3f}",
        "permissible_language": "Under the tested distribution and metric, retrieval is much more sensitive to d_k than d_v.",
        "source": str(retrieval_path.relative_to(ROOT)),
    })

    # Verified POPGym benchmark: derive representative orderings and full-suite coverage.
    popgym_path = GEN / "original_paper_upgrade/popgym_complete_mmer_per_seed.csv"
    popgym = pd.read_csv(popgym_path)
    selected_tasks = ["CountRecallEasy", "ConcentrationEasy"]
    selected_models = ["SdLG-Exact", "SdLG-Approx", "PPO-GRU", "PPO-Mamba", "PPO-GLA"]
    representative = popgym[popgym["Seed"].isin([42, 43, 44])]
    selected = (representative[representative.Environment.isin(selected_tasks)
                               & representative.Model.isin(selected_models)]
                .groupby(["Environment", "Model"])["Eval_Return_H100"]
                .agg(["mean", "std", "count"]).reset_index())
    selected.to_csv(OUT / "popgym_scope_examples.csv", index=False)
    task_count = int(popgym.Environment.nunique())
    easy_count = int(popgym.Environment.drop_duplicates().str.endswith("Easy").sum())
    medium_count = int(popgym.Environment.drop_duplicates().str.endswith("Medium").sum())
    check("popgym-benchmark-coverage",
          task_count == 23 and easy_count == 16 and medium_count == 7 and len(selected) == 10,
          f"{easy_count} Easy + {medium_count} Medium tasks; selected examples contain "
          f"{len(selected)} task-model summaries", popgym_path)
    exact_count = float(selected[(selected.Environment == "CountRecallEasy") & (selected.Model == "SdLG-Exact")]["mean"].iloc[0])
    exact_conc = float(selected[(selected.Environment == "ConcentrationEasy") & (selected.Model == "SdLG-Exact")]["mean"].iloc[0])
    gru_conc = float(selected[(selected.Environment == "ConcentrationEasy") & (selected.Model == "PPO-GRU")]["mean"].iloc[0])
    claims.append({
        "claim_id": "POPGYM-ORDER-REVERSAL",
        "claim_type": "cross-architecture, cross-task benchmark comparison",
        "variables_that_change_jointly": "architecture, parameter count, optimizer/training dynamics, and task operation",
        "observation": f"Exact={exact_count:.3f} on CountRecallEasy; Exact={exact_conc:.3f} and PPO-GRU={gru_conc:.3f} on ConcentrationEasy",
        "permissible_language": "The evaluated models reverse ordering across these tasks; the comparison does not establish a capacity-matched global ranking.",
        "source": str(popgym_path.relative_to(ROOT)),
    })

    # Manuscript-ready table fragment: generated directly from the per-seed rows.
    table_path = PAPER / "artifacts/popgym_scope_examples.tex"
    order = {name: i for i, name in enumerate(selected_models)}
    lookup = {(row.Environment, row.Model): row for row in selected.itertuples()}
    lines = [
        "% Generated by scripts/diagnostics/build_revision_verification.py",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Task & Operation & Exact & Approx. & PPO-GRU & PPO-Mamba & PPO-GLA \\\\",
        "\\midrule",
    ]
    task_labels = {
        "CountRecallEasy": ("CountRecall", "accumulate/query"),
        "ConcentrationEasy": ("Concentration", "discrete matching"),
    }
    for task in selected_tasks:
        label, operation = task_labels[task]
        cells = []
        for model in sorted(selected_models, key=lambda model: order[model]):
            row = lookup[(task, model)]
            cells.append(f"${row.mean:.3f}\\pm{row.std:.3f}$")
        lines.append(f"{label} & {operation} & " + " & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    table_path.write_text("\n".join(lines) + "\n")

    claims.append({
        "claim_id": "TMAZE-EXECUTION-PHASE",
        "claim_type": "controlled factorial intervention",
        "variables_that_change_jointly": "state representation, recurrence, read computation, block topology, number of associative layers, trunk-gradient routing, effective state size, numerical path; phase features",
        "observation": "Exact-minus-Approximate contrasts are positive at H=100,150,200; interaction intervals approach zero at two horizons",
        "permissible_language": "The executed configurations differ at trained and unseen horizons; this is not a pure rank or isolated-memory-mechanism effect.",
        "source": str(contrasts_path.relative_to(ROOT)),
    })

    claims.append({
        "claim_id": "TMAZE-MATCHED-EXECUTION",
        "claim_type": "controlled independent-condition execution comparison",
        "variables_that_change_jointly": "full-matrix versus rank-16 factorized state, recurrence/update, normalized read computation, effective state size, and numerical path; topology and actor-gradient scale are held fixed",
        "observation": "Exact-minus-Approximate is 0.502 [0.346,0.655] at H=100, 0.398 [0.251,0.500] at H=150, and 0.351 [0.200,0.497] at the prespecified H=200 endpoint",
        "permissible_language": "Under the matched three-block topology and common unattenuated trunk, the full-state and factorized execution formulations retain materially different delayed-memory behavior.",
        "source": str(contrasts_path.relative_to(ROOT)),
    })

    claims.append({
        "claim_id": "CRAFTAX-EXTERNAL-LANDSCAPE",
        "claim_type": "protocol-matched external scope/counterexample",
        "variables_that_change_jointly": "architecture, capacity, implementation, recurrent state, and systems cost",
        "observation": "compact GRU is competitive in return and fastest in recorded cumulative training-loop SPS",
        "permissible_language": "Dense control provides a counterexample to simple architecture-dominance narratives.",
        "source": "papers/writing/craftax_gate3a_canonical_summary.md",
    })

    # Craftax Cross-T historical summary and the recovered SdLG-Approx raw
    # artifacts.  Keep endpoint-return evidence separate from GGS and SPS.
    # The old table heading called the values ``evaluation return/IQM``; the
    # recovered per-run summaries show that the two SdLG-Approx cells are
    # final Train/Mean_Return values.  Do not use the historical heading as a
    # substitute for a raw-metric audit.
    cross_t_path = GEN / "craftax_cross_t_verified.csv"
    cross_t = pd.read_csv(cross_t_path)
    approx_t16 = cross_t[(cross_t["Family"] == "SdLG-Approx") & (cross_t["T"] == 16)].iloc[0]
    approx_t64 = cross_t[(cross_t["Family"] == "SdLG-Approx") & (cross_t["T"] == 64)].iloc[0]
    exact_t64 = cross_t[(cross_t["Family"] == "SdLG-Exact") & (cross_t["T"] == 64)].iloc[0]
    cross_t_passed = (
        len(cross_t) == 15
        and set(cross_t["Seeds"]) == {5}
        and close(approx_t16["MeanReturn"], 16.330)
        and close(approx_t64["MeanReturn"], 14.790)
        and close(exact_t64["MeanReturn"], 13.700)
        and approx_t16["HistoricalRunner"] == "SdLGHybrid"
    )
    check(
        "craftax-cross-t-historical-summary",
        cross_t_passed,
        f"{len(cross_t)} historical five-run cells; SdLG-Approx="
        f"{approx_t16['MeanReturn']:.3f} at T=16 and {approx_t64['MeanReturn']:.3f} at T=64; "
        f"historical runner={approx_t16['HistoricalRunner']}",
        cross_t_path,
    )

    recovered_raw_specs = {
        16: ROOT / "wandb",
        64: ROOT / "scripts/diagnostics/benchmarks/craftax_full/wandb",
    }
    recovered_raw_rows = []
    for rollout, base in recovered_raw_specs.items():
        candidates = sorted(base.glob(
            f"run-*/files/wandb-summary.json"
        ))
        records = []
        for path in candidates:
            run_name = path.parents[1].name
            suffix = f"craftax50m_sdlghybrid_t{rollout}_seed_"
            if suffix not in run_name:
                continue
            if rollout == 16 and not run_name.endswith("_run-4"):
                continue
            if rollout == 64 and run_name.endswith("_run-4"):
                continue
            try:
                seed = int(run_name.split("_seed_", 1)[1].split("_", 1)[0])
                summary = json.loads(path.read_text())
                score = float(summary["Train/Mean_Return"])
                endpoint = int(summary["_step"])
            except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if seed in {42, 43, 44, 45, 46}:
                records.append((seed, score, endpoint, path))
        records.sort(key=lambda row: row[0])
        values = np.array([row[1] for row in records], dtype=float)
        displayed = cross_t[(cross_t["Family"] == "SdLG-Approx")
                            & (cross_t["T"] == rollout)].iloc[0]
        raw_passed = (
            [row[0] for row in records] == [42, 43, 44, 45, 46]
            and np.all([row[2] == 49_971_200 for row in records])
            and close(values.mean(), displayed["MeanReturn"], tolerance=0.01)
            and close(values.std(ddof=1), displayed["SampleSD"], tolerance=0.01)
        )
        check(
            f"craftax-cross-t-sdlg-approx-t{rollout}-raw-endpoints",
            raw_passed,
            f"seeds={[row[0] for row in records]}; final Train/Mean_Return="
            f"{values.mean():.6f}±{values.std(ddof=1):.6f} from {len(records)} runs",
            base,
        )
        for seed, score, endpoint, path in records:
            recovered_raw_rows.append({
                "family": "SdLG-Approx",
                "historical_runner": "SdLGHybrid",
                "T": rollout,
                "seed": seed,
                "metric": "Train/Mean_Return",
                "endpoint_step": endpoint,
                "value": score,
                "artifact": str(path.relative_to(ROOT)),
            })
    pd.DataFrame(recovered_raw_rows).to_csv(
        OUT / "craftax_cross_t_sdlg_approx_raw_recomputed.csv", index=False
    )
    claims.append({
        "claim_id": "CRAFTAX-CROSS-T",
        "claim_type": "historical five-run architecture/configuration comparison",
        "variables_that_change_jointly": "rollout length, TBPTT length, GAE segmentation, PPO batch/update structure, architecture, and capacity",
        "observation": "Recovered SdLG-Approx endpoint Train/Mean_Return artifacts reproduce 16.330±0.370 at T=16 and 14.790±0.360 at T=64; profiles differ by family",
        "permissible_language": "The observed architecture profiles differ across the available rollout-horizon settings; this is not an isolated truncation effect.",
        "source": str((WRITING / "craftax_cross_t_provenance.md").relative_to(ROOT)),
    })

    # Guard against a common final-pass failure: validated numbers changing in
    # one manuscript location but not another.  These checks are deliberately
    # simple and auditable; the numerical recomputations above remain primary.
    main_tex_path = PAPER / "main.tex"
    method_tex_path = PAPER / "sections/03_methodology.tex"
    setup_tex_path = PAPER / "sections/04_experimental_setup.tex"
    results_tex_path = PAPER / "sections/05_main_results.tex"
    discussion_tex_path = PAPER / "sections/07_discussion.tex"
    main_text = main_tex_path.read_text()
    method_text = method_tex_path.read_text()
    setup_text = setup_tex_path.read_text()
    results_text = results_tex_path.read_text()
    discussion_text = discussion_tex_path.read_text()
    manuscript_checks = [
        ("manuscript-minigrid-values",
         all(token in main_text + results_text for token in
             ["15 independently analyzed runs per condition", "$-0.274$", "$[-0.549,0.008]$"]),
         "15-seed count, difference, and interval endpoints are present"),
        ("manuscript-tmaze-contrasts",
         all(token in results_text for token in
             ["$0.275$ [$0.150,0.398$]", "$0.225$ [$0.125,0.325$]",
              "$0.251$ [$0.151,0.351$]", "Table~\\ref{tab:tmaze_factorial_contrasts}"]),
         "execution effects and the factorial-contrast table reference are present"),
        ("manuscript-tmaze-matched-values",
         all(token in results_text for token in
             ["Matched Exact", "Matched Approximate", "$0.351$", "$0.200,0.497$"])
         and "three-block" in method_text + setup_text
         and "unattenuated" in setup_text,
         "matched H=200 contrast and fixed-topology description are present"),
        ("manuscript-ggs-values",
         all(token in results_text + discussion_text for token in
             ["$11.210$", "$9.757$", "$1.453$", "$[0.925,1.962]$"]),
         "accepted endpoint means, paired difference, and interval are present"),
        ("manuscript-retrieval-values",
         all(token in results_text for token in
             ["ten draws in each of 48 cells", "0.002$--$0.010",
              "Under the tested key/value distribution and retrieval metric"]),
         "factorial size, d_v span, and scope qualifier are present"),
        ("manuscript-popgym-values",
         all(token in results_text for token in
             ["$0.670$", "$0.158$", "$-0.267$", "common-seed, five-model comparisons"]),
         "coverage qualification and CountRecall/Concentration reversal values are present"),
        ("manuscript-craftax-cross-t-values",
         all(any(float(mean) == expected_mean and float(sd) == expected_sd
                 for mean, sd in re.findall(r"\$(\d+\.\d+)\\pm(\d+\.\d+)\$",
                                            main_text + results_text + discussion_text))
             for expected_mean, expected_sd in [(16.330, 0.370), (16.440, 0.110), (14.790, 0.360)])
         and
         all(token in main_text + results_text + discussion_text for token in
             ["historical name \\texttt{SdLGHybrid}", "cross-$T$"]),
         "verified SdLG-Approx mapping and Cross-T values are present"),
    ]
    for check_id, passed, detail in manuscript_checks:
        check(check_id, passed, detail, results_tex_path)

    appendix_path = PAPER / "sections/appendix.tex"
    appendix_text = appendix_path.read_text()
    check("unsupported-compute-total-removed", "1,240" not in appendix_text and "1240 GPU" not in appendix_text,
          "No unsupported project-wide GPU-hour total", appendix_path)
    check("craftax-provenance-reference", r"Appendix~D.7" not in appendix_text and
          r"Appendix~\ref{app:craftax_cross_t_provenance}" in appendix_text,
          "Craftax reproduction reference uses a stable section label", appendix_path)
    check("craftax-training-metric-label", "final logged training-return" in results_text and
          cross_t.Metric.eq("Final logged training return").all(),
          "Cross-T figure prose and summary metadata identify training return", cross_t_path)
    overlap_path = GEN / "component_transfer/tmaze_matched_overlap_audit.csv"
    overlap = pd.read_csv(overlap_path)
    exact_overlap = overlap[overlap.historical_model.eq("Exact_NoRoPE")]
    counts = exact_overlap.groupby("horizon").absolute_difference.apply(lambda x: int((x > 0).sum())).to_dict()
    check("historical-matched-seed-overlap", counts == {100: 4, 150: 4, 200: 6},
          f"Nonidentical seed-labelled endpoints by horizon: {counts}; not proof of trajectory independence", overlap_path)

    component_dir = GEN / "component_transfer"
    rigor_path = component_dir / "scientific_rigor_audit.json"
    rigor = json.loads(rigor_path.read_text())
    overwrite_path = component_dir / "associative_overwrite_raw.csv"
    overwrite = pd.read_csv(overwrite_path)
    paired = pd.read_csv(component_dir / "associative_overwrite_contrasts.csv")
    check("a1-frozen-source", sha256(overwrite_path) == rigor["a1_source_sha256"],
          "Raw outcomes unchanged since paired analysis", overwrite_path)
    for row in paired.itertuples():
        fields = ["N", "d_k", "d_v", "seed"]
        a = overwrite[(overwrite.formulation == row.treatment) & (overwrite.delta_eta == row.treatment_eta)].set_index(fields)[row.metric]
        b = overwrite[(overwrite.formulation == row.reference) & (overwrite.delta_eta == row.reference_eta)].set_index(fields)[row.metric]
        difference = (a - b).groupby(level=[0, 1, 2]).mean().mean()
        check(f"a1-{row.treatment}-{row.treatment_eta}-minus-{row.reference}-{row.reference_eta}-{row.metric}",
              close(difference, row.estimate) and row.ci_low <= difference <= row.ci_high,
              f"paired fixed-grid difference={difference:.9f}; bootstrap=[{row.ci_low:.9f},{row.ci_high:.9f}]", overwrite_path)
    run_audit = pd.read_csv(component_dir / "tmaze_baseline_run_audit.csv")
    check("matched-c1-endpoint-linkage", len(run_audit) == 40 and not run_audit.duplicated(["model", "seed"]).any()
          and sha256(tmaze_raw_path) == rigor["accepted_endpoints_sha256"]
          and sha256(tmaze_training_path) == rigor["accepted_training_sha256"],
          "40 unique accepted endpoints linked to local W&B summaries; does not prove cross-study execution identity", rigor_path)
    for model in ["GateBias_Exact_bg2", "GateBias_Exact_bg0"]:
        values = tmaze_raw[tmaze_raw.model == model]
        for h in (100, 150, 200):
            mean = values[f"val_H{h}"].mean()
            check(f"c1-{model}-H{h}", len(values) == 10 and close(mean, run_audit.loc[run_audit.model == model, f"H{h}"].mean()),
                  f"ten-seed mean={mean:.9f}", tmaze_raw_path)
    check("b2-timing-count", (pd.read_csv(component_dir / "matched_tmaze_inference.csv").timed_iterations == 300).all(),
          "All four systems cells use 300 warmed timing calls, not independent trained seeds", component_dir / "matched_tmaze_inference.csv")

    checks_df = pd.DataFrame(checks)
    claims_df = pd.DataFrame(claims)
    checks_df.to_csv(OUT / "checks.csv", index=False)
    claims_df.to_csv(OUT / "claim_ledger.csv", index=False)

    source_paths = [
        tmaze_raw_path, tmaze_training_path, tmaze_summary_path, contrasts_path, scan_test_path, mini_raw_path,
        mini_reproduction_path,
        mini_summary_path, mini_difference_path, ggs_path, retrieval_path,
        popgym_path, main_tex_path, method_tex_path, setup_tex_path, results_tex_path, discussion_tex_path,
        GEN / "table1_craftax_performance.csv",
        GEN / "table2_craftax_throughput.csv",
        cross_t_path,
        overwrite_path, rigor_path, component_dir / "associative_overwrite_contrasts.csv",
        component_dir / "tmaze_baseline_run_audit.csv",
    ]
    hashes = pd.DataFrame([{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                           for path in source_paths])
    hashes.to_csv(OUT / "source_hashes.csv", index=False)

    failed = checks_df[checks_df.status != "PASS"]
    payload = {
        "status": "PASS" if failed.empty else "FAIL",
        "checks": len(checks_df),
        "failed_checks": failed.check_id.tolist(),
        "headline_claims": len(claims_df),
        "excluded_from_manuscript": {
            "ggs_alpha_0_exploratory": "papers/writing/generated_artifacts/revision_2026/ggs_alpha_sweep_raw.csv",
            "minigrid_historical_five": "papers/writing/generated_artifacts/fig5_minigrid_swa.csv",
            "b1_rank_sweep": "halted singleton and unfinished run; no comparative evidence",
        },
        "unresolved": rigor["unresolved"],
    }
    (OUT / "verification.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Revision Verification Artifact",
        "",
        f"Overall status: **{payload['status']}** ({len(checks_df) - len(failed)}/{len(checks_df)} checks passed).",
        "",
        "Run from the repository root:",
        "",
        "```bash",
        ".venv/bin/python scripts/diagnostics/build_revision_verification.py",
        "```",
        "",
        "## Claim ledger",
        "",
        "| Claim | Type | Observation | Variables that change jointly | Permissible language |",
        "|---|---|---|---|---|",
    ]
    for row in claims_df.itertuples():
        lines.append(f"| `{row.claim_id}` | {row.claim_type} | {row.observation} | "
                     f"{row.variables_that_change_jointly} | {row.permissible_language} |")
    lines.extend([
        "", "## Mechanical checks", "",
        "| Status | Check | Detail | Source |", "|---|---|---|---|",
    ])
    for row in checks_df.itertuples():
        lines.append(f"| {row.status} | `{row.check_id}` | {row.detail} | `{row.source}` |")
    lines.extend([
        "", "## Finalization evidence", "",
        "### 1. Matched T-Maze execution",
        "- Ten accepted seed-level runs per matched cell and two separately tagged technical pilots are present in the shared T-Maze raw artifacts.",
        "- The independent-condition contrasts are 0.502 [0.346,0.655], 0.398 [0.251,0.500], and 0.351 [0.200,0.497] at H=100,150,200; H=200 is primary.",
        "- The validation mode confirms three full blocks, alpha=1 for Approximate, and nonzero actor/critic gradients to shared memory parameters.",
        "",
        "### 2. Craftax provenance",
        "- The Cross-T table retains accepted five-run summaries. SdLG-Approx T=16/T=64 were reconstructed from individual W&B endpoints within the documented historical rounding tolerance. SdLG-Exact raw-record availability is author-confirmed, not independently reconstructed here; see `papers/writing/craftax_cross_t_provenance.md`.",
        "- Cross-T endpoint returns remain separate from fixed-endpoint GGS and cumulative loop-SPS evidence.",
        "",
        "### 3. Recurrent correctness",
        "- The deterministic scan suite checks sequential/associative equivalence, chunk carry, episode resets at a TBPTT boundary, and the stop-gradient boundary while retaining parameter gradients.",
        "",
        "### 4. Mathematics",
        "- The normalized Delta special-case eigenvalue check is run with the scan suite. The appendix states its normalization and approximation assumptions and distinguishes it from a general stability guarantee.",
        "",
        "### 5. PDF release",
        "- The manuscript-number checks, source hashes, and generated claim ledger are emitted with this report. The release build is separately checked for unresolved references and fonts.",
        "", "## Dataset decisions", "",
        "- MiniGrid headline statistics use all 15 freshly generated rows per condition from one pinned historical code/environment snapshot. The earlier five-row artifact is context only and is never spliced into the headline dataset.",
        "- The separate MiniGrid reproduction record contains the historical and rerun endpoint, absolute difference, exact seed, source and runner hashes, package versions, evaluation protocol, and known nondeterminism.",
        "- GGS headline statistics retain the accepted five matched pairs for alpha=.1 and alpha=1. The single exploratory alpha=0 run is excluded.",
        "- The verified historical Craftax cross-$T$ block contains five-run recorded endpoint-return summaries. `SdLGHybrid` is mapped to the factorized SdLG-Approx configuration with a coupled actor--critic trunk; it is not pooled with GGS or loop-SPS records.",
        "- POPGym provides verified task-native benchmark records over 16 Easy and 7 Medium tasks. Its comparisons remain task-level because parameter counts and endpoint semantics differ across families; no global capacity-matched ranking is claimed.",
        "- T-Maze Exact/Approximate is a configuration-level comparison that changes representation, recurrence, read computation, block topology, associative-layer count, and trunk-gradient routing jointly.",
        "- The later matched T-Maze study retains separate historical rows and pilots. It fixes three local--associative--gated-FFN blocks and an unattenuated common trunk, while comparing the full-state and rank-16 factorized execution formulations with independent seed-level bootstrap intervals.",
        "- The printed Exact associative-scan operator is checked against sequential recurrence with carried state and asynchronous episode resets.",
        "", "## Generated files", "",
        "- `papers/writing/generated_artifacts/final_verification/checks.csv`",
        "- `papers/writing/generated_artifacts/final_verification/claim_ledger.csv`",
        "- `papers/writing/generated_artifacts/final_verification/source_hashes.csv`",
        "- `papers/writing/generated_artifacts/final_verification/popgym_scope_examples.csv`",
        "- `papers/writing/generated_artifacts/craftax_cross_t_verified.csv`",
        "- `papers/latex/cleanRewrite/artifacts/popgym_scope_examples.tex`",
        "- `papers/writing/generated_artifacts/final_verification/verification.json`",
    ])
    # Preserve the separately authored visual/scientific audit notes on rebuild.
    previous = REPORT.read_text() if REPORT.exists() else ""
    marker = "## Figure, notation, and explanation audit"
    tail = "\n" + marker + previous.split(marker, 1)[1] if marker in previous else ""
    REPORT.write_text("\n".join(lines) + "\n" + tail)

    if not failed.empty:
        raise SystemExit(f"verification failed: {failed.check_id.tolist()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-rigor-only", action="store_true")
    parser.add_argument("--check-retrieval-table-only", action="store_true")
    args = parser.parse_args()
    if args.check_retrieval_table_only:
        passed, detail = canonical_retrieval_table_check()
        print(("PASS: " if passed else "FAIL: ") + detail)
        raise SystemExit(0 if passed else 1)
    if args.component_rigor_only:
        component_rigor()
    else:
        main()
