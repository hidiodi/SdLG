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

## Repository status

The repository provides a frozen-record audit plus table and figure builders.
The remaining migration work converts each historical study implementation to
the release command interface and removes its legacy-only dependencies.
See [`docs/MISSING_RELEASE_ITEMS.md`](docs/MISSING_RELEASE_ITEMS.md) for the
explicit release blockers and their current status.

## License

MIT. See [LICENSE](LICENSE).
