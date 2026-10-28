#!/usr/bin/env python3
"""Reconstruct exploratory Passive T-Maze solve summaries from frozen rows."""

from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "papers/writing/generated_artifacts/revision_2026/tmaze_factorial_seed_outcomes.csv"
CSV_OUT = ROOT / "papers/writing/generated_artifacts/revision_2026/tmaze_solve_exploratory.csv"
TEX_OUT = ROOT / "papers/latex/cleanRewrite/artifacts/tmaze_solve_exploratory.tex"
THRESHOLD = 0.90
Z_95 = 1.959963984540054

STUDIES = (
    ("Original factorial", "Exact_NoRoPE", "Approx_NoRoPE"),
    ("Matched control", "Matched_Exact_NoRoPE", "Matched_Approx_NoRoPE"),
)


def wilson(successes: int, total: int) -> tuple[float, float]:
    proportion = successes / total
    denominator = 1.0 + Z_95**2 / total
    center = (proportion + Z_95**2 / (2.0 * total)) / denominator
    half_width = Z_95 * math.sqrt(
        proportion * (1.0 - proportion) / total + Z_95**2 / (4.0 * total**2)
    ) / denominator
    return center - half_width, center + half_width


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    row_one = a + b
    row_two = c + d
    successes = a + c
    total = row_one + row_two

    def probability(cell_a: int) -> float:
        return (
            math.comb(row_one, cell_a)
            * math.comb(row_two, successes - cell_a)
            / math.comb(total, successes)
        )

    lower = max(0, successes - row_two)
    upper = min(row_one, successes)
    observed = probability(a)
    return sum(
        probability(cell_a)
        for cell_a in range(lower, upper + 1)
        if probability(cell_a) <= observed + 1e-15
    )


with SOURCE.open(newline="") as stream:
    source_rows = list(csv.DictReader(stream))

rows: list[dict[str, object]] = []
for study, exact_model, approximate_model in STUDIES:
    values: dict[str, list[float]] = {}
    for label, model in (("exact", exact_model), ("approximate", approximate_model)):
        values[label] = [
            float(row["val_H200"])
            for row in source_rows
            if row["model"] == model
        ]
        if len(values[label]) != 10:
            raise ValueError(f"Expected 10 {model} rows, found {len(values[label])}")

    exact_solved = sum(value >= THRESHOLD for value in values["exact"])
    approximate_solved = sum(value >= THRESHOLD for value in values["approximate"])
    exact_interval = wilson(exact_solved, 10)
    approximate_interval = wilson(approximate_solved, 10)
    fisher_p = fisher_two_sided(
        exact_solved,
        10 - exact_solved,
        approximate_solved,
        10 - approximate_solved,
    )
    rows.append(
        {
            "study": study,
            "horizon": 200,
            "solve_threshold": THRESHOLD,
            "exact_solved": exact_solved,
            "exact_total": 10,
            "exact_wilson_low": exact_interval[0],
            "exact_wilson_high": exact_interval[1],
            "approximate_solved": approximate_solved,
            "approximate_total": 10,
            "approximate_wilson_low": approximate_interval[0],
            "approximate_wilson_high": approximate_interval[1],
            "fisher_two_sided_p": fisher_p,
            "analysis_status": "post hoc exploratory; pointwise and unadjusted",
        }
    )

CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
with CSV_OUT.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

tex_lines = [
    r"\begin{tabularx}{\linewidth}{l>{\centering\arraybackslash}X>{\centering\arraybackslash}Xc}",
    r"\toprule",
    r"Study & Exact & Approx. & Fisher $p$ \\",
    r"\midrule",
]
for row in rows:
    tex_lines.append(
        f"{row['study']} & "
        f"{row['exact_solved']}/10 $[{row['exact_wilson_low']:.3f},{row['exact_wilson_high']:.3f}]$ & "
        f"{row['approximate_solved']}/10 $[{row['approximate_wilson_low']:.3f},{row['approximate_wilson_high']:.3f}]$ & "
        f"${row['fisher_two_sided_p']:.5f}$ \\\\"
    )
tex_lines.extend((r"\bottomrule", r"\end{tabularx}"))
TEX_OUT.parent.mkdir(parents=True, exist_ok=True)
TEX_OUT.write_text("\n".join(tex_lines) + "\n")

for row in rows:
    print(
        f"{row['study']}: Exact {row['exact_solved']}/10, "
        f"Approximate {row['approximate_solved']}/10, "
        f"Fisher p={row['fisher_two_sided_p']:.8f}"
    )
