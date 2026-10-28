# Remaining release work

This is the live, reader-facing gap register for the standalone supplementary
repository. It separates a **verified frozen record** from a **standalone
paper-protocol launcher**. A study is not called fully reproducible until it
has both.

## Optional fresh-run evidence

The release criterion for this repository is a standalone launch path,
versioned frozen record, passing configuration validation, and a clean-clone
audit. The paper's heavyweight PPO campaigns are deliberately **not** rerun as
a release prerequisite. The following are recommended post-release smoke or
provenance checks, not blockers for publishing the supplementary repository.

| Priority | Paper study | What is present | What is missing |
| --- | --- | --- | --- |
| P2 | `T3-TMAZE-EXTERNAL-LANDSCAPE` | All 56 terminal records (8 families x 7 seeds), independent-contrast analysis, versioned protocol, standalone eight-family launcher, and a passing no-training launch check. | Optionally run one selected cell and verify its normalized terminal CSV. Do not run the full 56-job campaign merely for release preparation. |
| P2 | `T3-CRAFTAX-EXTERNAL-LANDSCAPE` | Versioned five-seed T=64 records, versioned configuration, standalone adapter, and a passing no-training full-matrix launch check. | Optionally run one selected cell and verify the normalized summary output. Confirm the recovered five seed values against archived run metadata before calling a fresh run an exact historical rerun; do not run all 35 heavyweight jobs merely for release preparation. |
| P2 | `T3-CRAFTAX-CROSS-T` | Versioned cross-T record, listed 15-cell configuration, standalone adapter, and a passing no-training matrix launch check. | Optionally run one selected cell and verify the normalized summary output. Confirm the recovered five seed values against archived run metadata before calling a fresh run an exact historical rerun; do not run all 75 heavyweight jobs merely for release preparation. |

## Required release hardening

| Priority | Area | Missing work |
| --- | --- | --- |
| P2 | Per-study configurations | Replace remaining Python-only experiment constants with tracked configuration files for every study/cell: model, seeds, budget, stopping rule, evaluation protocol, and W&B mode. |
| P2 | Heavy-run launch checks | Add a non-training `--validate-config` path for T-Maze, MiniGrid, POPGym, Craftax, and GGS that checks imports, environment creation, model construction, config schema, and output writeability. `reproduce.py dry-run` currently verifies files and syntax only. |
| P2 | Output contracts | Complete the same output-directory audit for T-Maze, MiniGrid locality, DoorKey, and the recovered Craftax launchers. POPGym and the three lightweight diagnostics are already isolated. |
| P2 | Tables | Replace the remaining reference-only TeX artifacts with record-driven generators where underlying compact records suffice: DoorKey run/interim tables, overwrite contrasts, and the project Craftax summaries. Keep published POPGym reference tables as explicitly versioned external transcriptions. |
| P2 | Figure visual regression | The empirical paper figures are generated and catalogued from tracked records. Add image-level visual-regression tolerances only if the submission venue requires pixel-stable figure rendering across Matplotlib versions. |
| P2 | Automated tests | A fast release test rebuilds all tracked tables and empirical figures and rechecks every frozen headline value. Add deeper recurrent-semantics unit tests (prefix scan, chunk carry, reset masking, TBPTT detachment, and SWA ring-buffer reset) if the implementation is refactored. |
| P2 | Audit artifact | `reproduce.py audit` now writes checksummed runner, record, table, and figure inventory to `results/audit/audit_report.json`. Add per-command wall-clock and accelerator telemetry only if a release requires performance certification. |
| P2 | Clean-room rehearsal | On a fresh clone, run installation, all command validations, table/figure generation, record verification, and one short smoke run per environment family. |

## Currently validated fresh executions

- Associative capacity: all 480 cells recreated; maximum numerical difference
  from the frozen record is `1.8e-7`.
- Associative overwrite: all 2,400 raw-result rows match exactly.
- Matched state accounting: architecture names, parameter counts, carry sizes,
  50 warmups, and 300 timed calls match; latency is machine-state dependent.

## Present but not yet smoke-executed

- T-Maze historical and matched runners expose a local output directory and
  compile successfully.
- MiniGrid locality and DoorKey runners compile successfully.
- POPGym now writes CSVs and resumable checkpoints only to its supplied
  output directory; a one-task offline smoke command is documented.
- GGS writes its result CSV to the public run directory.

The status above is deliberately conservative. Frozen-record verification is
not evidence that a fresh PPO run has been completed.
