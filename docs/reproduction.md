# Reproducing the frozen SdLG paper

This repository is standalone supplementary material.  Commands are run from
the repository root and do not need the legacy research workspace.

## Setup

The full workloads target ROCm on an AMD Radeon RX 6800 XT (16 GB).

```bash
uv sync --extra analysis --extra dev
uv run python -c "import jax; print(jax.__version__); print(jax.devices())"
```

## Fast verification

These commands do not train an RL policy.  They validate that frozen records
and release runners are present, recompute headline claims from the compact
records, and create the reference table/figure artifacts.

```bash
uv run python scripts/reproduce.py audit
uv run python scripts/reproduce.py tables
uv run python scripts/reproduce.py figures
```

Outputs are written below `results/`, which is intentionally ignored by Git.
The audit additionally writes `results/audit/audit_report.json`, a
machine-readable inventory of the checked study configurations, record hashes,
and generated table/figure filenames.

## Study commands

Use `show` before a full run to inspect the source runner and frozen input
record.  Every full invocation writes a provenance manifest first. Its outputs
are isolated below `results/<study>/`; no command writes into the tracked
source tree.

```bash
uv run python scripts/reproduce.py show tmaze-historical
uv run python scripts/reproduce.py run tmaze-historical
uv run python scripts/reproduce.py run tmaze-matched
uv run python scripts/reproduce.py run tmaze-baselines

uv run python scripts/reproduce.py run minigrid-locality
uv run python scripts/reproduce.py run doorkey-8x8
uv run python scripts/reproduce.py run popgym

uv run python scripts/reproduce.py run craftax-landscape
uv run python scripts/reproduce.py run craftax-cross-t
uv run python scripts/reproduce.py run craftax-ggs

uv run python scripts/reproduce.py run associative-capacity
uv run python scripts/reproduce.py run associative-overwrite
uv run python scripts/reproduce.py run state-accounting
```

The external T-Maze landscape launcher validates its frozen eight-family,
seven-seed protocol before it imports JAX or trains. Its full command writes
only below the selected `results/` directory; use
`uv run python scripts/reproduce.py dry-run tmaze-baselines` for the
no-training protocol check.

The Craftax T=64 landscape and cross-T commands use a protocol adapter that
launches each long job in a fresh watchdog-compatible process. Validate either
matrix without training with `uv run python scripts/reproduce.py dry-run
craftax-landscape` or `craftax-cross-t`.

The Craftax and full POPGym protocols are expensive. They should be launched
only on the documented ROCm target after a successful audit. Use this
single-run POPGym integration check with offline tracking; it writes only
under `results/popgym-smoke/`:

```bash
WANDB_MODE=offline uv run python scripts/diagnostics/exact_sdlg/popgym_benchmark.py \
  --dry-run --seeds 42 --envs CountRecallEasy --models SdLG-Approx \
  --output-dir results/popgym-smoke
```

## Interpretation of checks

- **Frozen-record verification** checks that tracked compact records recreate
  the paper's reported rounded values and stated aggregations.
- **Dry-run** checks release wiring and required files; it does not claim the
  full training budget was run.
- **Fresh full run** produces a new seed-level result and manifest.  PPO is
  stochastic; evaluate agreement using the study's documented endpoint and
  aggregation, not bitwise equality.

## Study map

The complete paper-study inventory, seeds, budgets, and intended outputs are
in [experiment-manifest.md](experiment-manifest.md).  The current claim,
record, and verification status are tracked in
[TEMP_experiment_value_ledger.md](TEMP_experiment_value_ledger.md) until the
release audit is complete.

The empirical figure-to-record mapping is in
[figure-catalog.md](figure-catalog.md). The table equivalent is in
[table-catalog.md](table-catalog.md).
