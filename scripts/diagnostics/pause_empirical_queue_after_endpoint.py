"""Pause the empirical queue after a requested number of GGS sweep endpoints.

This is intentionally a local process watcher, not an LLM polling loop.  The
GGS runner flushes every CSV row; the watcher waits until the next *final*
endpoint has been written before stopping the active worker and its queue
parent.  It records a paused state after both processes have been signalled.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "papers/writing/generated_artifacts/revision_2026"
RAW = ARTIFACTS / "ggs_alpha_sweep_raw.csv"
STATUS = ARTIFACTS / "empirical_revision_queue_status.json"


def completed_endpoints() -> int:
    if not RAW.exists():
        return 0
    data = pd.read_csv(RAW)
    final = data[data["Eval_Return"].notna() & (data["Total_Steps"] >= 20_000_000)]
    return int(final[["Architecture", "Seed"]].drop_duplicates().shape[0])


def matching_pids(fragment: str) -> list[int]:
    pids = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc / "cmdline").read_bytes().decode(errors="replace").replace("\x00", " ")
        except OSError:
            continue
        if fragment in command:
            pids.append(int(proc.name))
    return pids


def write_status(target: int, observed: int) -> None:
    STATUS.write_text(json.dumps({
        "state": "paused_user_requested",
        "group": "ggs_alpha_sweep",
        "detail": "Stopped after the requested next completed endpoint; resume only on user instruction.",
        "completed_ggs_endpoints": observed,
        "target_completed_ggs_endpoints": target,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-completed-endpoints", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    while completed_endpoints() < args.target_completed_endpoints:
        time.sleep(args.poll_seconds)

    observed = completed_endpoints()
    # Stop the training child before the serial queue so no subsequent seed is
    # accepted. CSV rows have already been flushed by the runner.
    for pid in matching_pids("gradient_shielding.py --run --alphas 0,0.1,0.25,0.5,1.0"):
        os.kill(pid, signal.SIGTERM)
    for pid in matching_pids("run_empirical_revision_queue.py"):
        os.kill(pid, signal.SIGTERM)
    time.sleep(3)
    write_status(args.target_completed_endpoints, observed)


if __name__ == "__main__":
    main()
