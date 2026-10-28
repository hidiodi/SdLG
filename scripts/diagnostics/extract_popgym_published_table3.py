"""Extract the published POPGym Table 3 values and build appendix tables.

The source is the archived primary-paper PDF.  The parser accepts only the
13-model taxonomy and three-decimal mean/SD rows used by Morad et al.'s Table 3.
It then selects the Easy and Medium tasks represented in the thesis project.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path


DEFAULT_SOURCE = Path("papers/sources/morad2023_popgym.pdf")
DEFAULT_CSV = Path("papers/writing/generated_artifacts/popgym_morad_table3_reference.csv")
DEFAULT_TEX_DIR = Path("papers/latex/cleanRewrite/artifacts")

SOURCE_MODELS = (
    "DNC",
    "Elman",
    "FART",
    "FWP",
    "Fr.Stack",
    "GRU",
    "IndRNN",
    "LMU",
    "LSTM",
    "MLP",
    "PosMLP",
    "S4D",
    "TCN",
)
MODEL_GROUPS = (
    SOURCE_MODELS[:7],
    SOURCE_MODELS[7:],
)

# (project task, source task).  The source paper calls the project's
# PositionOnly tasks "Stateless".  VelocityOnlyCartpoleEasy was not reported in
# the source Table 3 and is retained as an explicit unavailable row.
EASY_TASKS = (
    ("AutoencodeEasy", "AutoencodeEasy"),
    ("BattleshipEasy", "BattleshipEasy"),
    ("ConcentrationEasy", "ConcentrationEasy"),
    ("CountRecallEasy", "CountRecallEasy"),
    ("HigherLowerEasy", "HigherLowerEasy"),
    ("LabyrinthEscapeEasy", "LabyrinthEscapeEasy"),
    ("LabyrinthExploreEasy", "LabyrinthExploreEasy"),
    ("MinesweeperEasy", "MineSweeperEasy"),
    ("MultiarmedBanditEasy", "MultiarmedBanditEasy"),
    ("NoisyPositionOnlyCartPoleEasy", "NoisyStatelessCartPoleEasy"),
    ("NoisyPositionOnlyPendulumEasy", "NoisyStatelessPendulumEasy"),
    ("PositionOnlyCartPoleEasy", "StatelessCartPoleEasy"),
    ("PositionOnlyPendulumEasy", "StatelessPendulumEasy"),
    ("RepeatFirstEasy", "RepeatFirstEasy"),
    ("RepeatPreviousEasy", "RepeatPreviousEasy"),
    ("VelocityOnlyCartpoleEasy", None),
)
MEDIUM_TASKS = (
    ("ConcentrationMedium", "ConcentrationMedium"),
    ("CountRecallMedium", "CountRecallMedium"),
    ("HigherLowerMedium", "HigherLowerMedium"),
    ("LabyrinthEscapeMedium", "LabyrinthEscapeMedium"),
    ("LabyrinthExploreMedium", "LabyrinthExploreMedium"),
    ("MinesweeperMedium", "MineSweeperMedium"),
    ("RepeatPreviousMedium", "RepeatPreviousMedium"),
)


def extract_layout(source: Path) -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", str(source), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def parse_table3(text: str) -> dict[tuple[str, str], tuple[float, float]]:
    start = text.rfind("Table 3: Full table of results")
    if start < 0:
        raise ValueError("Could not locate POPGym Table 3 in the source PDF")
    segment = text[start:]
    model_pattern = "|".join(re.escape(model) for model in SOURCE_MODELS)
    row_pattern = re.compile(
        rf"^\s*(?:(?P<env>[A-Za-z0-9]+(?:Easy|Medium|Hard))\s+)?"
        rf"(?P<model>{model_pattern})\s+"
        rf"(?P<mean>-?\d+\.\d+)\s+(?P<sd>\d+\.\d+)\s*$",
        re.MULTILINE,
    )

    current_env: str | None = None
    values: dict[tuple[str, str], tuple[float, float]] = {}
    for match in row_pattern.finditer(segment):
        if match.group("env"):
            current_env = match.group("env")
        if current_env is None:
            continue
        key = (current_env, match.group("model"))
        if key in values:
            raise ValueError(f"Duplicate Table 3 row: {key}")
        values[key] = (float(match.group("mean")), float(match.group("sd")))

    if len(values) != 45 * 13:
        raise ValueError(f"Expected 585 Table 3 rows; parsed {len(values)}")
    return values


def compact_number(value: float) -> str:
    rendered = f"{value:.3f}"
    if rendered.startswith("0."):
        return rendered[1:]
    if rendered.startswith("-0."):
        return "-." + rendered[3:]
    return rendered


def latex_task_name(project_task: str) -> str:
    replacements = {
        "NoisyPositionOnlyCartPoleEasy": r"NoisyPositionOnly\allowbreak{}CartPoleEasy",
        "NoisyPositionOnlyPendulumEasy": r"NoisyPositionOnly\allowbreak{}PendulumEasy",
        "PositionOnlyCartPoleEasy": r"PositionOnly\allowbreak{}CartPoleEasy",
        "PositionOnlyPendulumEasy": r"PositionOnly\allowbreak{}PendulumEasy",
        "VelocityOnlyCartpoleEasy": r"VelocityOnly\allowbreak{}CartpoleEasy",
    }
    return replacements.get(project_task, project_task)


def render_table(
    tasks: tuple[tuple[str, str | None], ...],
    models: tuple[str, ...],
    difficulty: str,
    group_index: int,
    values: dict[tuple[str, str], tuple[float, float]],
) -> str:
    model_headers = ["Frame Stack" if model == "Fr.Stack" else model for model in models]
    columns = "@{}p{0.245\\textwidth}" + "c" * len(models) + "@{}"
    label = f"table:popgym_published_{difficulty.lower()}_{group_index}"
    caption = (
        rf"\textbf{{Published POPGym {difficulty} reference ({group_index}/2).}} "
        r"MMER mean $\pm$ SD transcribed from Table~3 of \citet{morad2023popgym}. "
        r"These source trials are external context and are not pooled with the project evaluations."
    )
    lines = [
        r"\begin{table*}[p]",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{2.7pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        "Project task & " + " & ".join(model_headers) + r" \\",
        r"\midrule",
    ]
    for project_task, source_task in tasks:
        cells: list[str] = []
        for model in models:
            if source_task is None:
                cells.append("--")
            else:
                mean, sd = values[(source_task, model)]
                cells.append(rf"${compact_number(mean)}\pm{compact_number(sd)}$")
        lines.append(latex_task_name(project_task) + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if difficulty == "Easy":
        lines.extend(
            [
                r"\par\smallskip\raggedright",
                r"\textit{Note:} Morad et al.'s \texttt{Stateless} task names map to the project's "
                r"\texttt{PositionOnly} aliases. \texttt{VelocityOnlyCartpoleEasy} has no corresponding "
                r"Table~3 row and is shown as unavailable.",
            ]
        )
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--tex-dir", type=Path, default=DEFAULT_TEX_DIR)
    args = parser.parse_args()

    values = parse_table3(extract_layout(args.source))
    expected_tasks = [source for _, source in EASY_TASKS + MEDIUM_TASKS if source is not None]
    for task in expected_tasks:
        count = sum((task, model) in values for model in SOURCE_MODELS)
        if count != 13:
            raise ValueError(f"Expected 13 source models for {task}; found {count}")

    # Two previously audited cells guard against a shifted or malformed parse.
    if values[("ConcentrationEasy", "GRU")] != (-0.039, 0.005):
        raise ValueError("ConcentrationEasy/GRU does not match the primary source")
    if values[("AutoencodeEasy", "LSTM")] != (-0.312, 0.008):
        raise ValueError("AutoencodeEasy/LSTM does not match the primary source")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "project_task",
                "source_task",
                "difficulty",
                "source_model",
                "mmer_mean",
                "mmer_sd",
                "source",
            ),
        )
        writer.writeheader()
        for difficulty, tasks in (("Easy", EASY_TASKS), ("Medium", MEDIUM_TASKS)):
            for project_task, source_task in tasks:
                for model in SOURCE_MODELS:
                    if source_task is None:
                        mean = sd = ""
                    else:
                        mean, sd = values[(source_task, model)]
                    writer.writerow(
                        {
                            "project_task": project_task,
                            "source_task": source_task or "unavailable",
                            "difficulty": difficulty,
                            "source_model": model,
                            "mmer_mean": mean,
                            "mmer_sd": sd,
                            "source": "Morad et al. (2023), Table 3",
                        }
                    )

    args.tex_dir.mkdir(parents=True, exist_ok=True)
    for difficulty, tasks in (("Easy", EASY_TASKS), ("Medium", MEDIUM_TASKS)):
        for index, models in enumerate(MODEL_GROUPS, start=1):
            path = args.tex_dir / f"popgym_published_{difficulty.lower()}_{index}.tex"
            path.write_text(render_table(tasks, models, difficulty, index, values))


if __name__ == "__main__":
    main()
