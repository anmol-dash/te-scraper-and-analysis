"""IPC contract tests: spawn ``backend/main.py`` and speak NDJSON."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from main import MAX_LINE_BYTES

BACKEND = Path(__file__).resolve().parent.parent
MAIN = BACKEND / "main.py"


def _run_ipc(lines: list[str], *, timeout: float = 30.0) -> tuple[int, str, str]:
    payload = "\n".join(lines) + "\n"
    proc = subprocess.Popen(
        [sys.executable, str(MAIN)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(BACKEND),
    )
    out, err = proc.communicate(input=payload, timeout=timeout)
    assert proc.returncode is not None
    return proc.returncode, out, err


def _parse_json_lines(blob: str) -> list[dict]:
    rows: list[dict] = []
    for raw in blob.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        obj = json.loads(raw)
        assert isinstance(obj, dict)
        rows.append(obj)
    return rows


def test_ipc_ping_run_shutdown_contract() -> None:
    code, out, err = _run_ipc(
        [
            '{"id":"1","command":"ping","args":{}}',
            '{"id":"2","command":"run","args":{"argv":["noop"]}}',
            '{"id":"3","command":"shutdown","args":{}}',
        ]
    )
    assert code == 0
    assert err == ""
    msgs = _parse_json_lines(out)
    by_id = {m["id"]: m for m in msgs if m.get("id")}

    ping = by_id["1"]
    assert ping["type"] == "result"
    assert ping["payload"]["ok"] is True
    assert ping["payload"]["version"]

    run = by_id["2"]
    assert run["type"] == "result"
    assert run["payload"]["exit_code"] == 0

    shutdown = by_id["3"]
    assert shutdown["type"] == "result"
    assert shutdown["payload"]["ok"] is True


def test_echo_snippet_from_readme() -> None:
    code, out, err = _run_ipc(
        ['{"id":"1","command":"ping","args":{}}', '{"id":"z","command":"shutdown","args":{}}']
    )
    assert code == 0
    assert err == ""
    msgs = _parse_json_lines(out)
    assert any(m.get("type") == "result" and m.get("payload", {}).get("ok") is True for m in msgs)


def test_each_stdout_line_is_valid_json() -> None:
    _, out, _ = _run_ipc(
        ['{"id":"x","command":"ping","args":{}}', '{"id":"y","command":"shutdown","args":{}}']
    )
    for line in out.splitlines():
        if not line.strip():
            continue
        json.loads(line)


# --- Edge cases (malformed input, caps, cancel, shutdown ordering) -----------------


def test_malformed_json_emits_invalid_json_error(ipc_bridge) -> None:
    assert ipc_bridge.proc.stdin
    ipc_bridge.proc.stdin.write("not-json{\n")
    ipc_bridge.proc.stdin.flush()
    msg = ipc_bridge.read_until(lambda m: m.get("type") == "error")
    assert msg["type"] == "error"
    assert msg["payload"]["code"] == "INVALID_JSON"


def test_oversized_line_emits_line_too_large(ipc_bridge) -> None:
    pad = "x" * (MAX_LINE_BYTES + 1)
    payload = json.dumps({"id": "big", "command": "ping", "args": {}, "pad": pad})
    assert len(payload.encode("utf-8")) > MAX_LINE_BYTES
    assert ipc_bridge.proc.stdin
    ipc_bridge.proc.stdin.write(payload + "\n")
    ipc_bridge.proc.stdin.flush()
    msg = ipc_bridge.read_until(lambda m: m.get("type") == "error")
    assert msg["type"] == "error"
    assert msg["payload"]["code"] == "LINE_TOO_LARGE"


def test_ping_after_malformed_still_works(ipc_bridge) -> None:
    ipc_bridge.send({"id": "p0", "command": "ping", "args": {}})
    assert ipc_bridge.read_until(lambda m: m.get("id") == "p0")["payload"]["ok"] is True
    ipc_bridge.proc.stdin.write("{not json\n")
    ipc_bridge.proc.stdin.flush()
    assert ipc_bridge.read_until(lambda m: m.get("type") == "error")["type"] == "error"
    ipc_bridge.send({"id": "p1", "command": "ping", "args": {}})
    last = ipc_bridge.read_until(lambda m: m.get("id") == "p1")
    assert last["id"] == "p1"
    assert last["payload"]["ok"] is True


def test_cancel_mid_run(ipc_bridge) -> None:
    ipc_bridge.send(
        {"id": "r1", "command": "run", "args": {"argv": ["sleep", "--seconds", "30"]}}
    )
    time.sleep(0.25)
    ipc_bridge.send({"id": "c1", "command": "cancel", "args": {"target_id": "r1"}})

    results: dict[str, dict] = {}
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline and ("c1" not in results or "r1" not in results):
        msg = ipc_bridge.readline_obj(timeout=1.0)
        if msg.get("type") == "result":
            results[str(msg["id"])] = msg

    assert results["c1"]["payload"]["ok"] is True
    assert results["r1"]["payload"]["exit_code"] == 130


def test_shutdown_after_queued_run_drains_pool() -> None:
    """Shutdown may be processed while a run is still executing; pool should drain cleanly."""
    code, out, err = _run_ipc(
        [
            '{"id":"r1","command":"run","args":{"argv":["sleep","--seconds","1"]}}',
            '{"id":"sd","command":"shutdown","args":{}}',
        ],
        timeout=30.0,
    )
    assert err == ""
    assert code == 0
    msgs = _parse_json_lines(out)
    assert any(m.get("id") == "sd" and m.get("type") == "result" for m in msgs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
