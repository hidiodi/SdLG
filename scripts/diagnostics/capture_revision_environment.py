"""Write a compact package/runtime inventory for a revision experiment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import jax


PACKAGES = ("jax", "jaxlib", "flax", "optax", "numpy", "xminigrid", "craftax", "gymnax", "distrax")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--study-id", required=True)
    args = parser.parse_args()
    try:
        rocm = subprocess.check_output(["rocminfo"], text=True, stderr=subprocess.STDOUT, timeout=15)
        rocm_summary = next((line.strip() for line in rocm.splitlines() if "Marketing Name" in line), None)
    except (OSError, subprocess.SubprocessError):
        rocm_summary = None
    payload = {
        "study_id": args.study_id,
        "git_commit": args.git_commit,
        "runner_sha256": args.runner_sha256,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "rocm_device_summary": rocm_summary,
        "packages": {package: importlib.metadata.version(package) for package in PACKAGES},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
