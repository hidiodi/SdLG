# GGS telemetry-persistence static audit

Pinned base: `f374a13431cd101b960b4b272f6bd7bd1a4f70e9`.
Revision runner: `gradient_shielding.py`, SHA-256 `b1373e6f3e484105106c2bda163b85e9428c5ee0064670bfe8e4ce9596348bbd`.

## Scope of the code change

The runner retains the pre-existing per-log computation of the actor/critic probe-gradient cosine. The revision persists a CSV row at every such logging checkpoint. At checkpoints that do not meet the already-existing evaluation condition, evaluation fields are empty rather than copied forward.

## Static checks before the reproduction gate

- The patch does not add, remove, or modify a `jax.random` call. The seed-level `jax.random.PRNGKey(seed)` construction remains at the same training-run entry point.
- The patch does not modify the imported `make_update` routine, `compiled_update_and_reduce`, optimizer configuration, model configuration, or the JIT-wrapped update function.
- The evaluation predicate is algebraically unchanged: `log_counter % EVAL_EVERY_N_LOGS == 0 or steps_processed >= TOTAL_STEPS`; it is assigned to `is_eval_checkpoint` before the same branch.
- The additional CSV write occurs after the compiled update, cosine extraction, and (when scheduled) evaluation. It cannot feed a value into a later optimizer or environment step.
- Changes outside persistence are CLI/manifest/W&B metadata and neutral diagnostic terminology.

This static audit establishes that the persistence patch does not consume the training PRNG state, modify the compiled PPO update, or alter the evaluation trigger. The paired endpoint comparison in `ggs_reproduction_gate_comparison.csv` is the empirical compatibility check once the gate finishes.
