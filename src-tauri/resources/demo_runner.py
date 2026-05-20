#!/usr/bin/env python3
"""Structured stdout for the Tauri demo runner (JSON lines)."""
from __future__ import annotations

import argparse
import json
import sys
import time


def emit(obj: object) -> None:
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default="",
        help="Example file path selected in the UI (informational).",
    )
    p.add_argument(
        "--label",
        default="demo",
        help="Short text label for this run.",
    )
    p.add_argument(
        "--profile",
        default="standard",
        choices=("fast", "standard", "careful"),
        help="Example select field.",
    )
    args = p.parse_args()

    emit(
        {
            "type": "log",
            "level": "info",
            "message": (
                f"Starting demo run label={args.label!r} profile={args.profile!r} "
                f"input={args.input!r}"
            ),
        }
    )

    steps = 30
    for i in range(1, steps + 1):
        time.sleep(0.04)
        emit(
            {
                "type": "progress",
                "fraction": i / steps,
                "current": i,
                "total": steps,
                "message": f"Processing step {i} of {steps}",
            }
        )
        if i % 10 == 0:
            emit(
                {
                    "type": "log",
                    "level": "warn",
                    "message": f"Checkpoint at step {i} — memory pressure example",
                }
            )

    emit({"type": "log", "level": "info", "message": "Demo run finished successfully."})
    return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        emit({"type": "log", "level": "error", "message": "Interrupted."})
        sys.exit(130)
