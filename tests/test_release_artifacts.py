"""Fast release checks that do not launch PPO training jobs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*arguments: str) -> None:
    env = {**os.environ, "MPLBACKEND": "Agg"}
    subprocess.run([sys.executable, *arguments], cwd=ROOT, env=env, check=True)


def test_frozen_records_and_artifacts_build(tmp_path: Path) -> None:
    """All public compact records and generated artifacts remain self-contained."""
    run("scripts/verify_paper_records.py")
    tables = tmp_path / "tables"
    figures = tmp_path / "figures"
    run("scripts/build_tables.py", "--output-dir", str(tables))
    run("scripts/build_figures.py", "--output-dir", str(figures))
    assert (tables / "table_manifest.json").is_file()
    assert (figures / "fig3_tmaze_rank_rope.pdf").is_file()
    assert (figures / "gradient_shielding_auxiliary.pdf").is_file()
