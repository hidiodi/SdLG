# SdLG supplementary material

This is the standalone reproducibility release for *Operational Boundaries of
a Modular Recurrent-Memory Architecture in Partially Observable Reinforcement
Learning*. It contains the SdLG implementation, versioned compact experiment
records, public launch commands, the frozen manuscript source, and the
compiled paper.

## Quick start

From the repository root, install the locked environment and validate the
frozen paper results:

```bash
uv sync --extra analysis --extra dev
uv run python scripts/reproduce.py audit
(cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)
```

The audit is analysis-only: it verifies the frozen headline values and
rebuilds the released table and figure artifacts under `results/audit/`. It
does not launch long PPO training jobs.

For every paper study, configuration, seeds, expected outputs, and the full
launch command, see [`docs/experiment-manifest.md`](docs/experiment-manifest.md).
[`docs/reproduction.md`](docs/reproduction.md) gives the complete guided
workflow.

## Hardware target

The full workloads target ROCm on an AMD Radeon RX 6800 XT (16 GB). Full
Craftax and POPGym jobs are intentionally heavyweight. Run manifests record
the source revision, configuration, seed, package versions, and optional W&B
run identifier; checkpoints and raw W&B files are not committed.

The release deliberately has no Docker image: the supported full-run target is
the documented ROCm host, and a generic CUDA container would be misleading.
Install with the locked `uv` command in
[`docs/reproduction.md`](docs/reproduction.md).

## What this release reproduces

Every main-result study has a public full-run command, compact versioned
records, and an auditable output contract. The release audit verifies all
reported headline values and regenerates tables and empirical figures from the
tracked records. Fresh runs are stochastic; compare their documented endpoint
aggregates rather than expecting bitwise-identical results.

Appendix H.2 is a sensitivity condition released as verified seed-level
records and a versioned implementation patch. It is intentionally separate
from the main-result training runner.

## Frozen manuscript

The standalone manuscript source, its frozen tables and figures, and the
compiled PDF are included in [`paper/`](paper/). Its reproduction appendix
uses the public commands in this release. To rebuild the derived material and
the PDF after installation, run:

```bash
uv run python scripts/reproduce.py tables --output-dir paper/artifacts
uv run python scripts/reproduce.py figures --output-dir paper/plots/rebuilt
(cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)
```

## License

MIT. See [LICENSE](LICENSE).
