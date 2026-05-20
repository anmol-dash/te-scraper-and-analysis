"""Pytest configuration and shared fixtures for IPC bridge tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
MAIN = BACKEND_ROOT / "main.py"


@dataclass
class IpcBridge:
    """A running NDJSON IPC process (``main.py``) with text-mode stdin/stdout pipes."""

    proc: subprocess.Popen[str]

    def send(self, obj: dict) -> None:
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        assert self.proc.stdin
        self.proc.stdin.write(line)
        self.proc.stdin.flush()

    def readline_obj(self, timeout: float = 5.0) -> dict:
        assert self.proc.stdout
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.proc.poll() is not None:
                line = self.proc.stdout.readline() if self.proc.stdout else ""
                if line:
                    return json.loads(line)
                raise RuntimeError("IPC process exited before a response line was read")
            line = self.proc.stdout.readline()
            if line:
                return json.loads(line)
            time.sleep(0.01)
        raise TimeoutError("timed out waiting for an IPC response line")


@pytest.fixture
def ipc_bridge() -> Iterator[IpcBridge]:
    """Spawn ``main.py`` and shut it down cleanly after the test."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(BACKEND_ROOT), env.get("PYTHONPATH", "")]).strip(
        os.pathsep
    )
    proc = subprocess.Popen(
        [sys.executable, str(MAIN)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(BACKEND_ROOT),
        env=env,
    )
    br = IpcBridge(proc=proc)
    try:
        yield br
    finally:
        if proc.poll() is None:
            try:
                if proc.stdin:
                    proc.stdin.write(
                        json.dumps({"id": "__fixture_shutdown__", "command": "shutdown", "args": {}})
                        + "\n"
                    )
                    proc.stdin.flush()
                    proc.stdin.close()
            except BrokenPipeError:
                pass
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
