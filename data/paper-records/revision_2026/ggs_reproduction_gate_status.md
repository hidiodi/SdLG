# GGS reproduction-gate status

Status: paused by user on 2026-09-20 before either condition reached a fixed-endpoint evaluation.

The first gate invocation wrote five intermediate probe-telemetry rows for `alpha=0.1`, seed 42 before it was terminated with `SIGTERM`. It contains no valid endpoint row and is **not** evidence. The reproduction gate must be restarted from scratch with `--overwrite` before any sweep is authorized.

No `alpha=1.0` gate run or full alpha-sweep run was started. The MiniGrid study is complete and unaffected.
