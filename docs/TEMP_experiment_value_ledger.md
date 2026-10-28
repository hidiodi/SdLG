# TEMP: experiment-to-value ledger

This temporary migration ledger is the release gate. A study may not be
called reproduced merely because its training command starts: the recorded
result data must regenerate the paper value(s), and a fresh run must be
compared with the documented tolerance before the temporary file is promoted
to `docs/reproducibility.md`.

Status meanings:

- **Imported:** source code and compact historical record are in this repo.
- **Analysis pending:** the historical record has not yet been recomputed by
  the cleaned release analysis.
- **Fresh run pending:** no post-cleanup training run has been compared with
  the paper record. This is expected for the expensive studies.

| ID | Paper result to reproduce | Canonical legacy runner | Imported record | Status |
| --- | --- | --- | --- | --- |
| `T1-TMAZE-EXECUTION-PHASE` | Exact-NoRoPE mean returns 0.900, 0.900, 0.851 at H=100/150/200; H=200 *marginal over phase* Exact-minus-Approximate contrast 0.251 [0.151, 0.351]. | `exact_sdlg_tmaze_ablations.py --study historical` | `data/legacy-records/diagnostics/exact_sdlg/unified_tmaze_ablations_zeroshot.csv`; `data/paper-records/tmaze_rank_rope_zero_shot.csv` | Imported; record verified; fresh run pending |
| `T1-TMAZE-MATCHED-EXECUTION` | H=200 Exact 0.851 vs Approximate 0.500; contrast 0.351 [0.200, 0.497]. | `exact_sdlg_tmaze_ablations.py --study matched` | `data/paper-records/revision_2026/tmaze_factorial_seed_outcomes.csv` | Imported; analysis pending; fresh run pending |
| `T3-TMAZE-EXTERNAL-LANDSCAPE` | Seven-seed H=100/150/200 terminal-policy table for GRU, LSTM, Mamba, LTI, Markovian, GLA, GTrXL-64, and GTrXL-128. | `tmaze_external_landscape.py` | `data/paper-records/tmaze_external_landscape.csv`; `data/paper-records/revision_2026/tmaze_exact_vs_external_contrasts.csv` | Standalone eight-family launcher added; frozen protocol/output validation passes. Fresh full training is intentionally not run during release preparation. |
| `T1-MINIGRID-LOCAL-WINDOW` | MemoryS16: SWA 0.508 (SD 0.445), bypass 0.782 (SD 0.354), difference -0.274 [-0.549, 0.008]. | `swa_ablation.py` | `data/paper-records/revision_2026/minigrid_locality_seed_endpoints.csv` | Imported; record verified; fresh run pending |
| `T3-MINIGRID-DOORKEY-8X8` | Five-seed terminal-policy comparison of Exact, GRU, and GTrXL, including return, success, and early-stop/cap counts. | `doorkey_baseline_benchmark.py` | `data/paper-records/doorkey_8x8/five_seed_summary.csv` | Imported; record verified; fresh run pending |
| `T3-POPGYM-SUITE` | Task-native MMER records for 16 Easy and 7 Medium tasks; examples include Exact > GRU on CountRecall and the documented task-dependent reversals. The paper's scope examples use the historical three-seed subset; later Exact Concentration runs must not be pooled into that claim. | `popgym_benchmark.py --output-dir <results-dir>` | `data/paper-records/original_paper_upgrade/popgym_complete_mmer_per_seed.csv`; `data/paper-records/final_verification/popgym_scope_examples.csv` | Imported; record verified; standalone runner now confines CSVs and resumable checkpoints to its chosen output directory; fresh protocol run pending. |
| `T3-CRAFTAX-EXTERNAL-LANDSCAPE` | Five-seed 50M-step T=64 training-return and cumulative loop-SPS landscape across the paper's model families. | `craftax_paper_protocol.py --profile landscape` | `data/paper-records/original_paper_upgrade/craftax_t64_seven_family_performance.csv`; `data/paper-records/original_paper_upgrade/craftax_t64_seven_family_throughput.csv` | Standalone protocol adapter added; full seven-family matrix and output contract pass no-training validation. |
| `T3-CRAFTAX-CROSS-T` | Five-run 50M-step cross-T summaries; e.g. Approximate 16.33 (SD 0.37) at T=16 and 14.79 (SD 0.36) at T=64; PPO-GLA 15.69 (SD 0.22) at T=64. | `craftax_paper_protocol.py --profile cross-t` | `data/paper-records/craftax_cross_t_verified.csv` | Standalone protocol adapter added; exact 15-cell reported matrix and output contract pass no-training validation. |
| `T1-CRAFTAX-ACTOR-GRADIENT-SCALE` | Five matched pairs at 20.15M steps: alpha=0.1 is 11.210 (SD 0.557), alpha=1.0 is 9.757 (SD 0.299), paired difference 1.453 [0.925, 1.962]. | `gradient_shielding.py` | `data/legacy-records/diagnostics/architecture/gradient_shielding/gradient_shielding.csv`; `data/paper-records/ggs_endpoint_paired_difference.csv` | Imported; record verified; fresh run pending |
| `T2-ASSOCIATIVE-CAPACITY` | 480 seeded Delta-rule retrieval draws; monotonic addressing-dimension pattern and load sensitivity reported in Figure 3. | `persist_ten_seed_capacity_cells.py` plus capacity analysis | `data/paper-records/synthetic_capacity_480_cells.csv` | **Fresh standalone run checked:** all 480 conditions recreated; design matches row-for-row and the largest cosine difference is 1.8e-7 (accelerator floating-point variation). |
| `CTV-A1-ASSOCIATIVE-OVERWRITE` | 2,400 outcomes: normalized Delta target fidelity 0.685 vs additive 0.502 at eta=0.25; paired difference 0.183 [0.181, 0.185]. | `factorial_addressing_forgetting_suite.py` | `data/paper-records/component_transfer/associative_overwrite_raw.csv` | **Fresh standalone run checked:** all 2,400 rows, including every numerical metric, match the frozen record exactly. |
| `CTV-B2-MATCHED-STATE-INFERENCE` | Matched Exact/Approximate carry bytes and 300-iteration warmed latency; factorized state is 60.7% smaller, without a claimed latency advantage. | `profile_throughput.py --matched-tmaze-inference --warmup 50 --repeats 300` | `data/paper-records/component_transfer/matched_tmaze_inference.csv` | **Fresh standalone run checked:** architecture labels, parameter counts, carry bytes, warmups, and timed-iteration count match exactly; latency is correctly treated as machine-state dependent. |

## Required verification sequence

1. Replace every legacy runner named above with a stable release command.
2. Run the analysis on the imported record and mark the row **record verified**
   only when its values match the paper claim to the report's stated rounding.
3. Run a small smoke configuration to validate the end-to-end runner.
4. For the full compute protocol, compare new seed-level outputs against the
   recorded aggregation using the exact study metric and seed protocol; never
   require bitwise equality for stochastic PPO training.

## Model-semantics migration note

The release tracks the current workspace versions of
`swa_ablation.py`, `exact_sdlg_tmaze_ablations.py`, `exact_sdlg.py`, and
`sdlg_hybrid.py`.  These updates change termination handling for step counts:
the terminal observation receives step count zero.  The hybrid SWA single-step
path also uses circular buffer writes and a prefix validity mask.  Any fresh
reproduction must run recurrent-semantics tests.  These versions were present
when the frozen-paper runs were produced, so they are the sole paper-release
implementation rather than a post-freeze compatibility mode.
