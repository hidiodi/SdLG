# Table catalog

`uv run python scripts/reproduce.py tables` writes frozen reference fixtures to
`results/tables/` without reading the manuscript source tree.

| Output | Source | Role |
| --- | --- | --- |
| `tmaze_factorial_mean_sd.tex` | `data/frozen-tables/tmaze_factorial_mean_sd.tex` | T-Maze factorial appendix table. |
| `tmaze_solve_exploratory.tex` | `data/frozen-tables/tmaze_solve_exploratory.tex` | T-Maze H=200 solve appendix table. |
| `doorkey_8x8_*.tex` | `data/frozen-tables/doorkey_8x8_*.tex` | Main, summary, and run-level DoorKey tables. |
| `popgym_*.tex` | `data/frozen-tables/popgym_*.tex` | POPGym contextual and scope tables. |
| `overwrite_contrasts.tex` | `data/frozen-tables/overwrite_contrasts.tex` | Associative overwrite contrast table. |
| `associative_capacity_480_cells.csv` | `data/paper-records/synthetic_capacity_480_cells.csv` | Capacity table/figure input; regenerated from the versioned experiment record. |

The following project-result tables are regenerated from versioned records and
byte-checked against their frozen TeX references:
`tmaze_factorial_mean_sd.tex`, `tmaze_solve_exploratory.tex`,
`doorkey_8x8_main.tex`, `doorkey_8x8_summary.tex`, and
`popgym_scope_examples.tex`. The remaining TeX artifacts are retained
verbatim because they contain run identifiers, interim results, or external
published-reference transcription not recomputed from the compact records.
No output depends on the old repository or manuscript tree.
