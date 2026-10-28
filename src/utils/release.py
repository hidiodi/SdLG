"""Standalone release helpers shared by reproducibility commands."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def repository_revision() -> str | None:
    """Return the current Git revision when available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def config_digest(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_run_manifest(study_id: str, config: dict[str, Any], output_dir: Path) -> Path:
    """Write immutable provenance beside a new experiment's outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "study_id": study_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_revision": repository_revision(),
        "python": sys.version,
        "platform": platform.platform(),
        "config": config,
        "config_sha256": config_digest(config),
    }
    path = output_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path
