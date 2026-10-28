# SdLG

This is the reproducible-code release for the SdLG study of local attention,
associative recurrent memory, and controlled PPO experiments in partially
observable reinforcement learning.

The complete release contract is in
[`docs/experiment-manifest.md`](docs/experiment-manifest.md): it includes
every project experiment reported in the paper, from synthetic diagnostics to
full Craftax and POPGym benchmarks. Start with the standalone
[`docs/reproduction.md`](docs/reproduction.md) guide.

## Hardware target

The full workloads target ROCm on an AMD Radeon RX 6800 XT (16 GB). Full
Craftax and POPGym jobs are intentionally heavyweight. Run manifests record
the source revision, configuration, seed, package versions, and optional W&B
run identifier; checkpoints and raw W&B files are not committed.

The release deliberately has no Docker image: the supported full-run target is
the documented ROCm host, and a generic CUDA container would be misleading.
Install with the locked `uv` command in
[`docs/reproduction.md`](docs/reproduction.md).

## Repository status

Every frozen-paper study has a public launch command, a compact versioned
record, and an auditable output contract. `uv run python
scripts/reproduce.py audit` verifies reported values and rebuilds the released
tables and empirical figures. Long PPO runs are intentionally not executed by
the audit. See [`docs/MISSING_RELEASE_ITEMS.md`](docs/MISSING_RELEASE_ITEMS.md)
for conservative limits on fresh-run validation.

## License

MIT. See [LICENSE](LICENSE).
