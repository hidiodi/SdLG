# SdLG standalone supplementary-material release plan

## Objective

Turn this repository into the only code-and-data dependency a reader needs to
recreate the frozen paper's project results.  The legacy research workspace is
not a runtime, data, path, or build dependency.  The manuscript is updated
only after this release passes its verification gate.

## Definition of done

A fresh clone on the documented ROCm machine can:

1. run `uv sync` and report the resolved JAX devices;
2. run deterministic recurrent-semantics tests;
3. regenerate every project table and figure from versioned files in `data/`;
4. run a safe dry-run for every paper study;
5. launch every full study with an explicit configuration, seed list, output
   directory, and W&B mode;
6. verify frozen-record values against the paper's rounding and aggregation
   rules; and
7. produce an audit report without referring to the old workspace.

Full training does not require bitwise equality with historical PPO runs.
It must use the frozen model semantics, seed protocol, endpoint definition,
and study-specific aggregation described in the paper.

## Phase 0 - Freeze and provenance baseline

**Deliverables**

- `docs/TEMP_experiment_value_ledger.md` becomes the authoritative mapping of
  paper claim, study, runner, input records, aggregation, and acceptance
  tolerance.
- Record the current Git revision and package lockfile hash in every run
  manifest.
- Preserve the current versions of `exact_sdlg.py`, `sdlg_hybrid.py`,
  `exact_sdlg_tmaze_ablations.py`, and `swa_ablation.py` as the frozen-paper
  implementation.

**Acceptance gate**

- `python scripts/verify_paper_records.py` passes.
- No frozen claim points to an unversioned local file or an old-repository
  path.

## Phase 1 - Clean repository surface

**Deliverables**

- Reduce the current historical script pool to a small public surface:
  `src/`, `configs/`, `scripts/reproduce.py`, study runners, analysis modules,
  figure/table builders, and tests.
- Move shared model, environment, PPO, evaluation, result-manifest, and path
  utilities into explicit reusable modules.
- Remove or quarantine exploratory scripts, old watchdog logic, absolute
  paths, manuscript output paths, hard-coded checkpoint reads, and duplicate
  model/trainer copies.
- Keep only compact, citable inputs under `data/`; never commit W&B folders,
  checkpoints, logs, PDFs, generated figures, or IDE state.

**Acceptance gate**

- `rg '/home/coins|papers/|wandb_exports|checkpoints' src scripts configs`
  finds no release-path dependency.
- Package build, lint, and syntax checks pass from a clean environment.

## Phase 2 - Common reproduction infrastructure

**Deliverables**

- One command entry point: `uv run python scripts/reproduce.py`.
- Subcommands: `list`, `show STUDY`, `dry-run STUDY`, `run STUDY`,
  `tables`, `figures`, `verify-records`, and `audit`.
- Versioned YAML configuration per study/cell; no paper setting remains only
  as a Python constant.
- A result-manifest schema containing study ID, config hash, seeds, hardware,
  JAX/device versions, timestamps, Git revision, W&B run IDs, endpoint fields,
  and output checksums.
- Shared output convention: `results/<study-id>/<run-id>/`; `results/` stays
  ignored.
- `WANDB_MODE=offline` works for dry-runs and optional online tracking is
  documented without tokens in the repository.

**Acceptance gate**

- Every manifest row has a working `show` and `dry-run` command.
- Every full command writes only under `results/` and does not resume a
  historical checkpoint by default.

## Phase 3 - Experiment runners

Implement and verify the twelve paper studies below.  A checkmark requires a
clean dry-run; a full-run check requires the documented endpoint and output
schema, not completion of every expensive seed immediately.

| ID | Release command | Required full protocol |
| --- | --- | --- |
| `T1-TMAZE-EXECUTION-PHASE` | `reproduce.py run tmaze-historical` | Exact/Approximate x RoPE/NoRoPE, 10 seeds, Htrain=100, evaluation H=100/150/200. |
| `T1-TMAZE-MATCHED-EXECUTION` | `reproduce.py run tmaze-matched` | Matched three-block NoRoPE Exact/Approximate, 10 seeds each. |
| `T3-TMAZE-EXTERNAL-LANDSCAPE` | `reproduce.py run tmaze-baselines` | Eight baseline configurations, 7 seeds, shared stopping rule. |
| `T1-MINIGRID-LOCAL-WINDOW` | `reproduce.py run minigrid-locality` | MemoryS16 SWA/bypassed, 15 seeds per condition. |
| `T3-MINIGRID-DOORKEY-8X8` | `reproduce.py run doorkey-8x8` | Exact, GRU, GTrXL; 5 seeds each and terminal evaluation. |
| `T3-POPGYM-SUITE` | `reproduce.py run popgym` | 16 Easy + 7 Medium tasks; per-cell seeds and MMER aggregation. |
| `T3-CRAFTAX-EXTERNAL-LANDSCAPE` | `reproduce.py run craftax-landscape` | Seven families, T=64, 5 seeds, 50M steps. |
| `T3-CRAFTAX-CROSS-T` | `reproduce.py run craftax-cross-t` | Reported family x T cells, T in {16,32,64,128}, 5 seeds, 50M steps. |
| `T1-CRAFTAX-ACTOR-GRADIENT-SCALE` | `reproduce.py run craftax-ggs` | Alpha 0.1/1.0, matched five seeds, 20.15M endpoint. |
| `T2-ASSOCIATIVE-CAPACITY` | `reproduce.py run associative-capacity` | 480 seeded retrieval draws. |
| `CTV-A1-ASSOCIATIVE-OVERWRITE` | `reproduce.py run associative-overwrite` | 2,400 shared-draw outcomes and paired analysis. |
| `CTV-B2-MATCHED-STATE-INFERENCE` | `reproduce.py run state-accounting` | 50 warmups + 300 timed calls per documented cell. |

## Phase 4 - Tables and figures

**Deliverables**

- Keep frozen inputs in `data/` and create data-to-table/figure builders that
  do not import manuscript files.
- Write generated outputs to `results/tables/` and `results/figures/`.
- Maintain a catalog for every project table/figure: source data, generator,
  study ID, expected filename, and validation rule.
- Include the current frozen `.tex` table fragments only as reference fixtures;
  regenerated outputs must be compared against them where applicable.

**Required coverage**

- T-Maze baseline, factorial, matched, solve, and contrast tables.
- DoorKey summary and run-level tables.
- POPGym project and published-reference tables, clearly separated.
- Craftax return, throughput, cross-T, and GGS summaries.
- MiniGrid locality, associative-capacity, overwrite, and state-accounting
  tables.
- Every project figure described in the frozen paper, generated from the same
  compact inputs or fresh results.

**Acceptance gate**

- `reproduce.py tables` and `reproduce.py figures` succeed without `papers/`.
- Numeric comparisons match reference fixtures to stated display precision.

## Phase 5 - Verification and clean-room test

**Deliverables**

- `reproduce.py audit` runs environment, code-path, data-schema, table,
  figure, and frozen-value checks and writes one machine-readable report.
- Unit tests cover prefix scan equivalence, chunk carry, reset masking, TBPTT
  detachment, SWA ring-buffer/reset semantics, and result aggregation.
- One short end-to-end smoke run per environment family.
- A ROCm RX 6800 XT preflight report records driver/JAX/device information.

**Acceptance gate**

- Fresh-clone test passes using only SdLG.
- All twelve studies show `dry-run: pass`; all table/figure catalog entries
  show `generated: pass`; frozen-record verification passes.
- The audit explicitly marks heavyweight full runs as either completed or
  pending, never silently substitutes historical records for fresh runs.

## Phase 6 - Reader documentation and paper update

**Deliverables**

- README: installation, hardware note, one-command audit, quick smoke run,
  full-study commands, expected costs, W&B policy, and citation.
- `docs/reproduction.md`: detailed study matrix, outputs, data provenance,
  interpretation limits, and troubleshooting.
- Update the frozen manuscript's code-availability statement and reproduction
  appendix paths only after Phase 5 passes.
- Tag the supplementary-material release and archive its immutable revision
  before submission.

## Ordering and non-negotiables

1. Build common paths/manifests and the command registry before modifying any
   study runner.
2. Port one study at a time; add its table/figure generator and audit test in
   the same change.
3. Never claim a historical number is a fresh rerun.
4. Never require the legacy repository, local absolute paths, checkpoints, or
   private W&B files.
5. Do not edit paper paths until the standalone audit passes.
