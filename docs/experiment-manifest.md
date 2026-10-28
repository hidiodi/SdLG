# Paper experiment manifest

This repository is the executable companion to the SdLG paper.  Every study
reported as this project's result in `main.pdf` is represented here by a
versioned configuration, a runnable command, and a declared output contract.
External published reference tables are cited as context, not reproduced as
project experiments.

## Release contract

- The runner for every entry below must execute without referring to the
  legacy research workspace.
- A `--dry-run` mode must validate each runner's configuration and dependency
  wiring without launching a full experiment.
- Full runs write their output below an ignored `results/` directory and emit
  a JSON manifest containing the configuration, seed, package versions, and
  Git revision.
- Heavyweight runs are never required to commit checkpoints or raw W&B files.
  W&B identifiers and optional artifact links belong in the run manifest.
- Aggregate analyses and figure builders consume recorded result files rather
  than hard-coded paper numbers.

## Study inventory

| ID | Study | Required release entry point | Full-run contract |
| --- | --- | --- | --- |
| `T1-TMAZE-EXECUTION-PHASE` | Passive T-Maze Exact/Approximate x RoPE/NoRoPE factorial | `uv run python scripts/reproduce.py run tmaze-historical` | 4 configurations, 10 seeds each, train at H=100 and evaluate H=100/150/200. |
| `T1-TMAZE-MATCHED-EXECUTION` | Topology- and gradient-matched T-Maze Exact vs Approximate | `uv run python scripts/reproduce.py run tmaze-matched` | 2 configurations, 10 seeds each, matched three-block NoRoPE execution. |
| `T3-TMAZE-EXTERNAL-LANDSCAPE` | T-Maze recurrent baseline landscape | `uv run python scripts/reproduce.py run tmaze-baselines` | Eight baseline configurations, seven seeds, a 50M-step budget, and frozen early-stop/evaluation protocol. |
| `T1-MINIGRID-LOCAL-WINDOW` | MemoryS16 local-window ablation | `uv run python scripts/reproduce.py run minigrid-locality` | Approximate SdLG with SWA enabled/bypassed; 15 seeds per condition. |
| `T3-MINIGRID-DOORKEY-8X8` | DoorKey-8x8 cross-family comparison | `uv run python scripts/reproduce.py run doorkey-8x8` | Exact SdLG, GRU, and GTrXL; 5 seeds per model. |
| `T3-POPGYM-SUITE` | POPGym Easy and Medium suite | `uv run python scripts/reproduce.py run popgym` | Per-task, per-model records for 16 Easy and 7 Medium tasks with task-native MMER. |
| `T3-CRAFTAX-EXTERNAL-LANDSCAPE` | Craftax-Classic model landscape | `uv run python scripts/reproduce.py run craftax-landscape` | Seven families at T=64, five seeds, 50M steps; one watchdog-compatible process per long job. |
| `T3-CRAFTAX-CROSS-T` | Craftax rollout/TBPTT-length response | `uv run python scripts/reproduce.py run craftax-cross-t` | Exact reported family x T cells, five seeds, 50M steps; no unreported grid cells are added. |
| `T1-CRAFTAX-ACTOR-GRADIENT-SCALE` | GGS actor-gradient-scale ablation | `uv run python scripts/reproduce.py run craftax-ggs` | Alpha=0.1 vs 1.0 with five matched seeds, 20M-step endpoint. |
| `T2-ASSOCIATIVE-CAPACITY` | Delta-rule retrieval diagnostic | `uv run python scripts/reproduce.py run associative-capacity` | 480 seeded retrieval draws across the documented grid. |
| `CTV-A1-ASSOCIATIVE-OVERWRITE` | Associative overwrite diagnostic | `uv run python scripts/reproduce.py run associative-overwrite` | 2,400 shared-draw method outcomes over 48 cells. |
| `CTV-B2-MATCHED-STATE-INFERENCE` | Matched carry-size and inference-latency measurement | `uv run python scripts/reproduce.py run state-accounting` | Matched Exact/Approximate construction; 50 warmups plus 300 timed iterations per cell. |

## Implementation milestones

1. Extract a single authoritative model and PPO implementation; remove the
   experiment-local copies of model classes and training loops.
2. Express every row above as a validated configuration under `configs/`.
3. Implement the stable command-line entry points and result-manifest format.
4. Port analyses, statistical aggregation, and figure regeneration.
5. Verify every dry run, unit test recurrent semantics, and run a small
   end-to-end ROCm smoke experiment before release.

## Non-release material

The clean repository excludes checkpoints, raw W&B directories, raw logs,
paper build files, duplicated figures, IDE settings, scratch work, and
historical exploratory runs.  It may link to W&B artifacts after publication.
