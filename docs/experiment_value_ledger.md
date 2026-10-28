# Experiment-to-value ledger

This is the release evidence ledger. A study is not called reproduced merely
because its training command starts: tracked seed-level records must regenerate
the frozen paper values. Fresh heavyweight PPO runs are optional and are
compared by the documented aggregate metric, never bitwise equality.

| ID | Frozen-paper evidence | Release status |
| --- | --- | --- |
| `T1-TMAZE-EXECUTION-PHASE` | Ten-seed Exact/Approximate x RoPE/NoRoPE terminal records and H=100/150/200 aggregates. | Record checked by audit; standalone runner present. |
| `T1-TMAZE-MATCHED-EXECUTION` | Ten-seed matched Exact/Approximate records and independent contrasts. | Record checked by audit; model/gradient construction validated. |
| `H2-EXECUTION-SEMANTICS-ABLATION` | 60 modified-execution T-Maze endpoints, five modified DoorKey Exact endpoints, bootstrap summaries, and source patch. | Record and patch verified; sensitivity analysis only, not the main-result implementation. |
| `T3-TMAZE-EXTERNAL-LANDSCAPE` | Eight-family, seven-seed terminal record and contrast analysis. | Protocol/output contract validated without retraining. |
| `T1-MINIGRID-LOCAL-WINDOW` | Fifteen seed endpoints per SWA condition. | Record checked; public command selects the frozen 15 seeds and fixed endpoint. |
| `T3-MINIGRID-DOORKEY-8X8` | Five terminal records each for Exact, GRU, and GTrXL. | Record checked; public command selects only DoorKey-8x8 and all five seeds. |
| `T3-POPGYM-SUITE` | Task-native Easy/Medium MMER records. | Record checked; output-isolated runner present. |
| `T3-CRAFTAX-EXTERNAL-LANDSCAPE` | Seven-family T=64 performance and throughput records. | Full matrix protocol validated without retraining. |
| `T3-CRAFTAX-CROSS-T` | Fifteen reported family x rollout-length cells. | Full matrix protocol validated without retraining. |
| `T1-CRAFTAX-ACTOR-GRADIENT-SCALE` | Five matched endpoint pairs and paired bootstrap contrast. | Record checked; output-isolated runner present. |
| `T2-ASSOCIATIVE-CAPACITY` | 480 seeded retrieval draws. | Fresh standalone recreation matched within `1.8e-7`. |
| `CTV-A1-ASSOCIATIVE-OVERWRITE` | 2,400 shared-draw outcomes. | Fresh standalone recreation matched exactly. |
| `CTV-B2-MATCHED-STATE-INFERENCE` | Carry bytes and 300-iteration timed measurements. | Fresh construction/count check passed; latency remains machine-dependent. |

## Execution semantics

The default source is the frozen-paper main-result implementation: terminal
observations receive step count zero and one-step SWA uses circular writes with
a prefix validity mask. `patches/execution_semantics_ablation.patch` records
the deliberately different chronological-buffer and counter conventions used
only by Appendix H.2. Never apply that patch to recreate a main-text result.

## Fresh-run boundary

The release audit validates records, launch configuration, output contracts,
and generated artifacts. It does not claim that an expensive PPO campaign was
rerun. Any new full run must retain its manifest and compare seed-level output
against the matching frozen record using the study's stated aggregation.
