# SdLG supplementary material

This is the standalone reproducibility release for *Operational Boundaries of
a Modular Recurrent-Memory Architecture in Partially Observable Reinforcement
Learning*. It contains the SdLG implementation, versioned compact experiment
records, public launch commands, the frozen manuscript source, and the
compiled paper.

## Quick verification

From the repository root, install the locked environment and verify the
released records, tables, figures, and manuscript:

```bash
uv sync --frozen --extra analysis --extra dev
uv run pytest -q
uv run python scripts/reproduce.py audit
(cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)
```

The audit is analysis-only. It checks the compact records against the paper's
reported values and rebuilds derived artifacts below `results/audit/`; it does
not launch PPO training.

## Hardware and tracking

The full workloads target ROCm on an AMD Radeon RX 6800 XT with 16 GB of VRAM.
The locked environment uses the corresponding ROCm JAX packages. Confirm the
installation before starting a long run:

```bash
uv run python -c "import jax; print(jax.__version__); print(jax.devices())"
```

Craftax, POPGym, T-Maze, and MiniGrid training campaigns can take many hours or
days. The Craftax and POPGym matrices are the largest workloads. W&B is
optional; use `WANDB_MODE=offline` when no online run should be created.

The release deliberately has no Docker image. A generic CUDA image would not
represent the documented ROCm execution target.

## Reproduction command interface

All supported studies are registered in `scripts/reproduce.py`:

```bash
# List the study keys.
uv run python scripts/reproduce.py list

# Inspect one registry entry without executing it.
uv run python scripts/reproduce.py show <study-key>

# Validate its runner, records, and configuration without training.
uv run python scripts/reproduce.py dry-run <study-key>

# Start a fresh run and write below results/<study-key>/.
uv run python scripts/reproduce.py run <study-key>

# Select another output directory.
uv run python scripts/reproduce.py run <study-key> \
  --output-dir results/my-run
```

Each fresh run writes `run_manifest.json` before executing the study. The
manifest records the study configuration, Git revision, Python version, host
platform, and creation time. Generated results are ignored by Git and never
overwrite the tracked paper records.

Fresh PPO runs are stochastic. Compare the endpoint and aggregation defined
for the study rather than expecting bitwise-identical output.

## Running every experiment

Run the corresponding `dry-run` first, especially before a heavyweight PPO
run. The commands below use the complete seed sets and protocols encoded
by the release registry.

### Passive T-Maze

**Execution-by-phase factorial** (`T1-TMAZE-EXECUTION-PHASE`) runs Exact and
Approximate with RoPE enabled and disabled, using ten seeds per condition. It
trains at `H=100` and evaluates at `H=100`, `150`, and `200`.

```bash
uv run python scripts/reproduce.py dry-run tmaze-historical
uv run python scripts/reproduce.py run tmaze-historical
```

**Matched Exact/Approximate control** (`T1-TMAZE-MATCHED-EXECUTION`) runs the
two NoRoPE formulations with the matched three-block construction and ten
seeds per condition.

```bash
uv run python scripts/reproduce.py dry-run tmaze-matched
uv run python scripts/reproduce.py run tmaze-matched
```

**Recurrent baseline landscape** (`T3-TMAZE-EXTERNAL-LANDSCAPE`) runs the
eight registered baseline configurations with seven seeds, a 50-million-step
budget, and the documented stopping and evaluation protocol.

```bash
uv run python scripts/reproduce.py dry-run tmaze-baselines
uv run python scripts/reproduce.py run tmaze-baselines
```

### Appendix H.2 execution-semantics sensitivity

`H2-EXECUTION-SEMANTICS-ABLATION` is intentionally record-only in the command
registry. The released verification covers 60 modified-execution T-Maze seed
endpoints and five modified DoorKey Exact endpoints:

```bash
uv run python scripts/reproduce.py dry-run execution-semantics
```

The main source tree retains the implementation used for the main results.
`patches/execution_semantics_ablation.patch` records the two joint changes used
only for Appendix H.2: chronological right-aligned one-step local buffers and
the aligned post-reset counter update. To inspect or rerun that sensitivity,
apply the patch only in an isolated clone:

```bash
git clone . ../SdLG-h2
cd ../SdLG-h2
git apply --check patches/execution_semantics_ablation.patch
git apply patches/execution_semantics_ablation.patch
uv sync --frozen --extra analysis --extra dev

# Recreate the 40 factorial and 20 matched T-Maze endpoints.
uv run python scripts/reproduce.py run tmaze-historical \
  --output-dir results/h2-tmaze-factorial
uv run python scripts/reproduce.py run tmaze-matched \
  --output-dir results/h2-tmaze-matched

# Recreate the five modified-semantics DoorKey Exact runs.
uv run python scripts/diagnostics/benchmarks/xminigrid/doorkey_baseline_benchmark.py \
  --envs MiniGrid-DoorKey-8x8 --models SdLG-Exact \
  --seeds 42 54 99 123 456 --total-steps 20000000 \
  --early-stop-threshold 0.96 --early-stop-patience 3 \
  --results-dir results/h2-doorkey-exact --wandb-mode offline
```

Do not apply this patch to a main-result rerun. H.2 is a joint sensitivity
condition and does not identify the separate causal effect of either change.

### MiniGrid and Memory Gym

**MemoryS16 local-window intervention** (`T1-MINIGRID-LOCAL-WINDOW`) runs
Approximate SdLG with local processing enabled and bypassed, using 15 seeds per
condition and the fixed endpoint.

```bash
uv run python scripts/reproduce.py dry-run minigrid-locality
uv run python scripts/reproduce.py run minigrid-locality
```

**DoorKey-8x8 comparison** (`T3-MINIGRID-DOORKEY-8X8`) runs Exact, GRU, and
GTrXL on `MiniGrid-DoorKey-8x8`, with five seeds per model.

```bash
uv run python scripts/reproduce.py dry-run doorkey-8x8
uv run python scripts/reproduce.py run doorkey-8x8
```

### POPGym

The POPGym suite (`T3-POPGYM-SUITE`) produces per-task, per-model records for
the 16 Easy and seven Medium tasks using each task's native MMER endpoint.

```bash
uv run python scripts/reproduce.py dry-run popgym
WANDB_MODE=offline uv run python scripts/reproduce.py run popgym
```

A single-seed integration check is available before launching the full suite:

```bash
WANDB_MODE=offline uv run python scripts/diagnostics/exact_sdlg/popgym_benchmark.py \
  --dry-run --seeds 42 --envs CountRecallEasy --models SdLG-Approx \
  --output-dir results/popgym-smoke
```

### Craftax

**T=64 model landscape** (`T3-CRAFTAX-EXTERNAL-LANDSCAPE`) runs seven model
families with five seeds and 50 million environment steps. Each long job is
started in a separate watchdog-compatible process.

```bash
uv run python scripts/reproduce.py dry-run craftax-landscape
uv run python scripts/reproduce.py run craftax-landscape
```

**Rollout/TBPTT-length response** (`T3-CRAFTAX-CROSS-T`) runs the 15 reported
family-by-length cells, with five seeds and 50 million steps per cell. It does
not add unreported grid cells.

```bash
uv run python scripts/reproduce.py dry-run craftax-cross-t
uv run python scripts/reproduce.py run craftax-cross-t
```

**Actor-gradient-scale intervention** (`T1-CRAFTAX-ACTOR-GRADIENT-SCALE`)
compares `alpha=0.1` and `alpha=1.0` with five matched seeds at the 20-million-
step endpoint.

```bash
uv run python scripts/reproduce.py dry-run craftax-ggs
uv run python scripts/reproduce.py run craftax-ggs
```

### Synthetic memory diagnostics

**Associative retrieval** (`T2-ASSOCIATIVE-CAPACITY`) runs 480 seeded draws
over the documented addressing/value-dimension and load grid.

```bash
uv run python scripts/reproduce.py dry-run associative-capacity
uv run python scripts/reproduce.py run associative-capacity
```

**Corrective overwrite** (`CTV-A1-ASSOCIATIVE-OVERWRITE`) runs 2,400
shared-draw method outcomes over 48 cells.

```bash
uv run python scripts/reproduce.py dry-run associative-overwrite
uv run python scripts/reproduce.py run associative-overwrite
```

### State and latency accounting

The matched state/inference study (`CTV-B2-MATCHED-STATE-INFERENCE`)
constructs matched Exact and Approximate models, calculates carried-state
bytes, performs 50 warmups, and records 300 timed iterations per cell.

```bash
uv run python scripts/reproduce.py dry-run state-accounting
uv run python scripts/reproduce.py run state-accounting
```

Latency is machine-dependent; the carried-state calculation is deterministic,
but timings should be interpreted for the machine on which they were recorded.

## Records, tables, and figures

The tracked files below `data/paper-records/` are the compact inputs used to
verify the paper. They are not overwritten by fresh runs. Regenerate derived
material with:

```bash
uv run python scripts/reproduce.py verify-records
uv run python scripts/reproduce.py tables
uv run python scripts/reproduce.py figures
```

The detailed study inventory and output contracts are in
[`docs/experiment-manifest.md`](docs/experiment-manifest.md). The mapping from
studies to reported values is in
[`docs/experiment_value_ledger.md`](docs/experiment_value_ledger.md), with
figure and table mappings in [`docs/figure-catalog.md`](docs/figure-catalog.md)
and [`docs/table-catalog.md`](docs/table-catalog.md).

## Frozen manuscript

The manuscript source, frozen tables and figures, and compiled PDF are in
[`paper/`](paper/). To rebuild the derived material and PDF:

```bash
uv run python scripts/reproduce.py tables --output-dir paper/artifacts
uv run python scripts/reproduce.py figures --output-dir paper/plots/rebuilt
(cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)
```

## License

MIT. See [LICENSE](LICENSE).
