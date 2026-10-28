# Figure catalog

`uv run python scripts/reproduce.py figures` rebuilds every empirical figure
used by the frozen manuscript from versioned records. It writes both PNG and
PDF copies below `results/figures/` and never reads the manuscript workspace.

| Manuscript figure | Rebuilt output | Versioned source record | Status |
| --- | --- | --- | --- |
| T-Maze execution and phase | `fig3_tmaze_rank_rope.{pdf,png}` | `tmaze_rank_rope_terminal_training.csv`, `tmaze_rank_rope_zero_shot.csv`, `revision_2026/tmaze_exact_vs_external_contrasts.csv` | Rebuilt |
| Craftax systems scope | `fig4_craftax_throughput.{pdf,png}` | `original_paper_upgrade/craftax_t64_seven_family_throughput.csv` | Rebuilt |
| MiniGrid local-pathway comparison | `fig5_minigrid_locality_expanded.{pdf,png}` | `revision_2026/minigrid_locality_seed_endpoints.csv` | Rebuilt |
| Associative capacity | `fig6_associative_capacity.{pdf,png}` | `synthetic_capacity_480_cells.csv` | Rebuilt |
| Craftax cross-T | `fig7_craftax_cross_t.{pdf,png}` | `craftax_cross_t_verified.csv` | Rebuilt |
| Actor-gradient-scale endpoint | `gradient_shielding_paired.{pdf,png}` | `ggs_endpoint_paired_difference.csv` | Rebuilt |
| GGS checkpoint trajectories | `gradient_shielding_trajectories.{pdf,png}` | `legacy-records/diagnostics/architecture/gradient_shielding/gradient_shielding.csv` | Rebuilt |
| GGS auxiliary telemetry | `gradient_shielding_auxiliary.{pdf,png}` | same retained GGS raw record | Rebuilt |

The paper's architecture topology and other explanatory diagrams are static
illustrations, rather than empirical results. They are intentionally not
duplicated in this supplementary code repository; the frozen manuscript is
their canonical rendering. The table and experimental claims are independent
of those graphical assets.
