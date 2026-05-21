"""
JSON-lines IPC server: reads NDJSON requests from stdin, emits NDJSON on stdout.
"""

from __future__ import annotations

import os
os.environ.setdefault("CRYPTOGRAPHY_OPENSSL_NO_LEGACY", "1")

import inspect
import json
import logging
import queue
import shutil
import socket
import subprocess
import sys
import threading
import traceback
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from io import TextIOBase
from pathlib import Path
from typing import Any, TextIO

# ── import path: backend/ contains yourtool package ───────────────────────────
_BACKEND_ROOT = Path(__file__).resolve().parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from yourtool import __version__ as _TOOL_VERSION
from yourtool.cli import main as cli_main

_REAL_STDOUT: TextIO = sys.__stdout__

_emit_lock = threading.Lock()

# ── Venv setup ─────────────────────────────────────────────────────────────────

if sys.platform == "win32":
    _GAMECA_DIR = Path(os.environ.get("APPDATA", Path.home())) / "GAMECA"
else:
    _GAMECA_DIR = Path.home() / ".gameca"

_VENV_PATH         = _GAMECA_DIR / "venv"
_SCRIPTS_DIR       = _GAMECA_DIR / "scripts"
_SETUP_DONE_SENTINEL = _GAMECA_DIR / ".setup_done"
_LAST_COMMIT_FILE  = _GAMECA_DIR / "last_commit"
_CONFIG_FILE       = _GAMECA_DIR / "config.json"
_setup_lock   = threading.Lock()
_setup_running = False

# ── Script override: downloaded updates shadow bundled scripts ─────────────────
# Must happen before any pipeline imports so lazy imports in cli.py pick them up.
if _SCRIPTS_DIR.exists() and str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Scripts downloaded on update (mirrors pyinstaller.spec _pipeline_datas).
_UPDATABLE_SCRIPTS = [
    "query.py", "hpc_client.py", "ui.py",
    "te_prep.py", "te_genome.py", "te_clustering.py",
    "te_primers.py", "te_alignment.py", "te_motif.py",
    "te_go.py", "te_expression.py", "te_enrichment.py",
    "requirements.txt",
]


def _find_requirements() -> Path | None:
    if getattr(sys, "frozen", False):
        # PyInstaller bundles requirements.txt into _MEIPASS
        candidate = Path(sys._MEIPASS) / "requirements.txt"  # type: ignore[attr-defined]
        if candidate.exists():
            return candidate
    # Dev / non-frozen: look relative to backend/ and repo root
    for candidate in (
        Path(__file__).resolve().parent.parent / "requirements.txt",
        Path(__file__).resolve().parent / "requirements.txt",
    ):
        if candidate.exists():
            return candidate
    return None


def _is_setup_done() -> bool:
    return _SETUP_DONE_SENTINEL.exists() and (_VENV_PATH / "pyvenv.cfg").exists()


def _setup_log(msg: str) -> None:
    """Append a raw line to the setup log panel without changing phase or progress."""
    _emit({"type": "setup", "phase": "log", "message": msg})


def _run_setup_background() -> None:
    global _setup_running
    with _setup_lock:
        if _setup_running:
            return
        _setup_running = True

    try:
        _emit({"type": "setup", "phase": "checking", "fraction": 0.0,
               "message": "Checking environment…"})

        if _is_setup_done():
            _setup_log(f"Venv already exists at {_VENV_PATH}")
            _emit({"type": "setup", "phase": "done", "fraction": 1.0,
                   "message": "Environment ready."})
            return

        _setup_log("First launch — setting up Python environment.")
        _setup_log(f"Platform: {sys.platform}")

        req = _find_requirements()
        if req is None:
            _emit({"type": "setup", "phase": "error",
                   "message": "requirements.txt not found in bundle."})
            return
        _setup_log(f"Requirements file: {req}")

        py = shutil.which("python3") or shutil.which("python")
        if py is None:
            _emit({"type": "setup", "phase": "error",
                   "message": "Python 3 not found on this system. "
                               "Install Python 3.11+ and relaunch."})
            return

        # Log the Python interpreter that will be used.
        _setup_log(f"Python interpreter: {py}")
        ver_r = subprocess.run([py, "--version"], capture_output=True, text=True)
        _setup_log((ver_r.stdout.strip() or ver_r.stderr.strip()) or "version unknown")

        _GAMECA_DIR.mkdir(parents=True, exist_ok=True)
        _setup_log(f"Venv destination: {_VENV_PATH}")
        _emit({"type": "setup", "phase": "creating", "fraction": 0.02,
               "message": f"Creating environment at {_VENV_PATH}…"})

        r = subprocess.run(
            [py, "-m", "venv", str(_VENV_PATH)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            for ln in r.stderr.splitlines():
                _setup_log(ln)
            _emit({"type": "setup", "phase": "error",
                   "message": f"Failed to create venv: {r.stderr.splitlines()[-1] if r.stderr.strip() else 'unknown error'}"})
            return
        _setup_log("Venv created successfully.")

        pip = str(
            _VENV_PATH / ("Scripts" if sys.platform == "win32" else "bin") / "pip"
        )

        # Upgrade pip and log its output.
        _setup_log("Upgrading pip…")
        pip_up = subprocess.run(
            [pip, "install", "--upgrade", "pip"],
            capture_output=True, text=True,
        )
        for ln in (pip_up.stdout + pip_up.stderr).splitlines():
            if ln.strip():
                _setup_log(ln)

        req_lines = [
            ln for ln in req.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        total = max(1, len(req_lines))
        done = 0

        _setup_log(f"Installing {total} requirement(s) from {req.name}…")
        _emit({"type": "setup", "phase": "installing", "fraction": 0.05,
               "message": f"Installing {total} packages…"})

        proc = subprocess.Popen(
            [pip, "install", "--no-cache-dir", "-r", str(req)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            # Every line goes to the log panel.
            _setup_log(line)
            # Progress bar advances on package-level milestones.
            if any(kw in line for kw in
                   ("Collecting", "Downloading", "Installing collected",
                    "Successfully installed")):
                done += 1
                frac = min(0.95, 0.05 + 0.90 * done / total)
                _emit({"type": "setup", "phase": "installing",
                       "fraction": frac, "message": line})

        proc.wait()
        if proc.returncode != 0:
            _setup_log(f"pip exited with code {proc.returncode}")
            _emit({"type": "setup", "phase": "error",
                   "message": "Package installation failed — check logs for details."})
            return

        _SETUP_DONE_SENTINEL.touch()
        _setup_log("Setup complete. Sentinel written.")
        _emit({"type": "setup", "phase": "done", "fraction": 1.0,
               "message": "Environment ready!"})

    except Exception as exc:
        _setup_log(traceback.format_exc())
        _emit({"type": "setup", "phase": "error",
               "message": f"Setup error: {exc}"})
    finally:
        with _setup_lock:
            _setup_running = False

# ── Auto-update from GitHub ────────────────────────────────────────────────────

def _update_log(msg: str) -> None:
    _emit({"type": "update", "phase": "log", "message": msg})


_DEFAULT_REPO   = "anmol-dash/te-scraper-and-analysis"
_DEFAULT_BRANCH = "main"


def _read_update_config() -> tuple[str, str]:
    """Return (repo, branch) — always returns something (defaults to the real repo)."""
    env_repo = os.environ.get("GAMECA_UPDATE_REPO", "").strip()
    if env_repo:
        return env_repo, os.environ.get("GAMECA_UPDATE_BRANCH", _DEFAULT_BRANCH).strip()

    if _CONFIG_FILE.exists():
        try:
            cfg = json.loads(_CONFIG_FILE.read_text())
            repo = cfg.get("update_repo", "").strip()
            if repo:
                return repo, cfg.get("update_branch", _DEFAULT_BRANCH).strip()
        except Exception:
            pass

    return _DEFAULT_REPO, _DEFAULT_BRANCH


def _has_internet(timeout: float = 3.0) -> bool:
    try:
        socket.setdefaulttimeout(timeout)
        socket.getaddrinfo("api.github.com", 443)
        return True
    except Exception:
        return False


def _run_update_check() -> None:
    """Check GitHub for new commits and download changed scripts if found."""
    repo, branch = _read_update_config()

    if not _has_internet():
        _emit({"type": "update", "phase": "offline",
               "message": "Offline — update check skipped."})
        return

    _emit({"type": "update", "phase": "checking",
           "message": f"Checking {repo} for updates…"})

    # Fetch latest commit SHA (1 API call, no auth, 60 req/hr limit).
    try:
        url = (f"https://api.github.com/repos/{repo}/commits/{branch}"
               f"?per_page=1")
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "GAMECA-updater/1.0",
                     "Accept": "application/vnd.github.v3+json"},
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read())
        latest_sha: str = data["sha"]
        short_sha = latest_sha[:7]
    except Exception as exc:
        _update_log(f"GitHub API error: {exc}")
        _emit({"type": "update", "phase": "skipped",
               "message": "Could not reach GitHub — skipping."})
        return

    stored_sha = (_LAST_COMMIT_FILE.read_text().strip()
                  if _LAST_COMMIT_FILE.exists() else "")

    if latest_sha == stored_sha:
        _emit({"type": "update", "phase": "up_to_date",
               "message": f"Up to date ({short_sha})."})
        return

    _emit({"type": "update", "phase": "downloading",
           "message": f"New commit {short_sha} — downloading changes…"})
    _SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    updated: list[str] = []
    failed:  list[str] = []

    for script in _UPDATABLE_SCRIPTS:
        raw_url = (f"https://raw.githubusercontent.com"
                   f"/{repo}/{latest_sha}/{script}")
        try:
            req = urllib.request.Request(
                raw_url, headers={"User-Agent": "GAMECA-updater/1.0"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                content: bytes = resp.read()
            dest = _SCRIPTS_DIR / script
            if not dest.exists() or dest.read_bytes() != content:
                dest.write_bytes(content)
                updated.append(script)
                _update_log(f"Updated: {script}")
            else:
                _update_log(f"Unchanged: {script}")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                _update_log(f"Not in repo (skipped): {script}")
            else:
                failed.append(script)
                _update_log(f"HTTP {exc.code} for {script}")
        except Exception as exc:
            failed.append(script)
            _update_log(f"Failed {script}: {exc}")

    # Re-run pip install if requirements changed.
    if "requirements.txt" in updated and _is_setup_done():
        _update_log("requirements.txt changed — installing new dependencies…")
        pip = str(
            _VENV_PATH / ("Scripts" if sys.platform == "win32" else "bin") / "pip"
        )
        proc = subprocess.Popen(
            [pip, "install", "--no-cache-dir", "-r",
             str(_SCRIPTS_DIR / "requirements.txt")],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                _update_log(line)
        proc.wait()

    # Ensure new scripts shadow bundled ones immediately (for this session).
    scripts_str = str(_SCRIPTS_DIR)
    if scripts_str not in sys.path:
        sys.path.insert(0, scripts_str)

    _LAST_COMMIT_FILE.write_text(latest_sha)

    if updated:
        _emit({"type": "update", "phase": "done",
               "message": f"Updated {len(updated)} file(s) to {short_sha}.",
               "sha": short_sha, "updated": updated,
               "failed": failed})
    else:
        _emit({"type": "update", "phase": "up_to_date",
               "message": f"All files current at {short_sha}."})


def _version_tuple(v: str) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in v.lstrip("v").split("."))
    except Exception:
        return (0,)


def _check_release_update(repo: str) -> None:
    """Check GitHub Releases for a newer app version and emit release_available if found."""
    import platform as _platform

    try:
        url = f"https://api.github.com/repos/{repo}/releases/latest"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "GAMECA-updater/1.0",
                     "Accept": "application/vnd.github.v3+json"},
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        _update_log(f"Release check failed: {exc}")
        return

    latest_tag: str = data.get("tag_name", "").lstrip("v")
    if not latest_tag:
        return

    if _version_tuple(latest_tag) <= _version_tuple(_TOOL_VERSION):
        _update_log(f"App is up to date (release {latest_tag}).")
        return

    # Pick the right DMG asset for this machine.
    machine = _platform.machine()
    arch_hint = "aarch64" if machine in ("arm64", "aarch64") else "x86_64"
    assets: list[dict] = data.get("assets", [])
    dmg = next(
        (a for a in assets if a["name"].endswith(".dmg") and arch_hint in a["name"]),
        next((a for a in assets if a["name"].endswith(".dmg")), None),
    )
    if dmg is None:
        _update_log(f"Release {latest_tag} has no DMG asset — cannot auto-download.")
        return

    _emit({
        "type": "update",
        "phase": "release_available",
        "message": f"GAMECA {latest_tag} is available (you have {_TOOL_VERSION}).",
        "version": latest_tag,
        "current_version": _TOOL_VERSION,
        "download_url": dmg["browser_download_url"],
        "asset_name": dmg["name"],
        "size_bytes": dmg.get("size", 0),
    })


def _startup_sequence() -> None:
    """Run setup → script update → release check sequentially."""
    from yourtool.cli import _REPO_ROOT
    _setup_log(
        f"[SIDECAR] v{_TOOL_VERSION}"
        f"  frozen={getattr(sys, 'frozen', False)}"
        f"  REPO_ROOT={_REPO_ROOT}"
        f"  scripts_dir={_GAMECA_DIR / 'scripts'}"
    )
    _run_setup_background()
    _run_update_check()
    if _has_internet():
        repo, _ = _read_update_config()
        _check_release_update(repo)


# Hard cap for a single NDJSON line read from stdin (bytes, including newline).
MAX_LINE_BYTES = 256 * 1024


def _emit(obj: dict[str, Any]) -> None:
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with _emit_lock:
        _REAL_STDOUT.write(line)
        _REAL_STDOUT.flush()


def _default_pool_capacity() -> int:
    try:
        return max(1, int(os.environ.get("PYTOOL_MAX_PARALLEL", "4")))
    except ValueError:
        return 4


def _debug_traceback() -> bool:
    return os.environ.get("PYTOOL_DEBUG", "") == "1"


@dataclass
class _IPCState:
    cancel_events: dict[str, threading.Event] = field(default_factory=dict)
    pool_capacity: int = field(default_factory=_default_pool_capacity)
    permit_sem: threading.Semaphore = field(init=False)

    def __post_init__(self) -> None:
        self.permit_sem = threading.Semaphore(self.pool_capacity)


def _permits_reserved(concurrent: bool, pool_capacity: int, max_parallel: int) -> int:
    """Reserve semaphore permits so sequential runs are exclusive; concurrent runs share fairly."""
    if not concurrent:
        return pool_capacity
    mp = max(1, min(pool_capacity, max_parallel))
    return max(1, pool_capacity // mp)


@contextmanager
def _hold_run_permits(sem: threading.Semaphore, n: int) -> Iterator[None]:
    for _ in range(n):
        sem.acquire()
    try:
        yield
    finally:
        for _ in range(n):
            sem.release()


class _LineForwardingStream(TextIOBase):
    """Buffer stdout/stderr into JSON log lines (one IPC message per newline)."""

    def __init__(self, req_id: str, stream_name: str) -> None:
        super().__init__()
        self._req_id = req_id
        self._stream_name = stream_name
        self._buf = ""

    def write(self, s: str | Any) -> int:
        if not isinstance(s, str):
            s = str(s)
        if not s:
            return 0
        self._buf += s
        parts = self._buf.split("\n")
        self._buf = parts.pop()
        for line in parts:
            _emit(
                {
                    "type": "log",
                    "id": self._req_id,
                    "payload": {"stream": self._stream_name, "line": line},
                }
            )
        return len(s)

    def flush(self) -> None:
        # Partial trailing line on flush (mirrors line-buffered text streams).
        if self._buf:
            _emit(
                {
                    "type": "log",
                    "id": self._req_id,
                    "payload": {"stream": self._stream_name, "line": self._buf},
                }
            )
            self._buf = ""

    def isatty(self) -> bool:
        return False


class _JsonLogHandler(logging.Handler):
    def __init__(self, req_id: str) -> None:
        super().__init__()
        self._req_id = req_id
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        _emit(
            {
                "type": "log",
                "id": self._req_id,
                "payload": {
                    "stream": "logging",
                    "level": record.levelname,
                    "line": msg,
                    "logger": record.name,
                },
            }
        )


@contextmanager
def _capture_stdio_and_logging(req_id: str) -> Iterator[None]:
    out = _LineForwardingStream(req_id, "stdout")
    err = _LineForwardingStream(req_id, "stderr")
    handler = _JsonLogHandler(req_id)
    root = logging.getLogger()
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.NOTSET)
    try:
        with redirect_stdout(out), redirect_stderr(err):
            yield
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
        out.flush()
        err.flush()


def _progress_proxy_if_installed(req_id: str) -> list[Callable[[], None]]:
    """If tqdm is importable, patch tqdm.tqdm to emit progress IPC (best-effort)."""
    undo: list[Callable[[], None]] = []
    try:
        import tqdm.std as tqdm_std  # type: ignore[import-untyped]
    except ImportError:
        return undo

    orig_init = tqdm_std.tqdm.__init__
    if "callbacks" not in inspect.signature(orig_init).parameters:
        return undo

    def _patched_init(self, *args: Any, **kwargs: Any) -> None:
        orig_callbacks = kwargs.pop("callbacks", None)
        callbacks = list(orig_callbacks) if orig_callbacks else []

        def _ipc_cb(inst: Any) -> None:
            try:
                total = getattr(inst, "total", None)
                n = getattr(inst, "n", 0)
                pct = 0.0
                if total not in (None, 0):
                    pct = min(1.0, max(0.0, float(n) / float(total)))
                desc = getattr(inst, "desc", "") or ""
                _emit(
                    {
                        "type": "progress",
                        "id": req_id,
                        "payload": {"pct": pct, "msg": str(desc)},
                    }
                )
            except Exception:
                pass

        callbacks.append(_ipc_cb)
        kwargs["callbacks"] = callbacks
        orig_init(self, *args, **kwargs)

    tqdm_std.tqdm.__init__ = _patched_init  # type: ignore[method-assign]

    def _restore() -> None:
        tqdm_std.tqdm.__init__ = orig_init  # type: ignore[method-assign]

    undo.append(_restore)
    return undo


def _execute_run(req_id: str, argv: list[str], cancel_event: threading.Event, state: _IPCState) -> None:
    undo_progress = _progress_proxy_if_installed(req_id)
    try:
        with _capture_stdio_and_logging(req_id):
            result = cli_main(argv, cancel_event=cancel_event)
        if isinstance(result, Mapping):
            payload = dict(result)
            payload.setdefault("exit_code", 0 if payload.get("ok", True) else 1)
        else:
            payload = {"exit_code": int(result)}
        _emit({"type": "result", "id": req_id, "payload": payload})
    except BaseException as e:
        payload: dict[str, Any] = {"code": "EXCEPTION", "msg": str(e)}
        if _debug_traceback():
            payload["traceback"] = traceback.format_exc()
        _emit({"type": "error", "id": req_id, "payload": payload})
        if not isinstance(e, Exception):
            raise
    finally:
        state.cancel_events.pop(req_id, None)
        for fn in reversed(undo_progress):
            fn()


def _normalize_request_id(req: Mapping[str, Any]) -> str:
    rid = req.get("id")
    if rid is None or rid == "":
        return str(uuid.uuid4())
    return str(rid)


def _handle_ping(req_id: str) -> None:
    _emit({"type": "result", "id": req_id, "payload": {"ok": True, "version": _TOOL_VERSION}})


def _handle_cancel(state: _IPCState, req_id: str, target: str) -> None:
    ev = state.cancel_events.get(target)
    if ev is None:
        _emit({"type": "result", "id": req_id, "payload": {"ok": False, "reason": "unknown_target"}})
        return
    ev.set()
    _emit({"type": "result", "id": req_id, "payload": {"ok": True}})


def _submit_run(
    pool: ThreadPoolExecutor,
    state: _IPCState,
    req: Mapping[str, Any],
    req_id: str,
    concurrent: bool,
    max_parallel: int,
) -> None:
    args_obj = req.get("args") or {}
    argv = args_obj.get("argv")
    if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
        _emit(
            {
                "type": "error",
                "id": req_id,
                "payload": {"code": "INVALID_ARGV", "msg": "args.argv must be a list of strings"},
            }
        )
        return

    cancel_ev = threading.Event()
    state.cancel_events[req_id] = cancel_ev
    permits = _permits_reserved(concurrent, state.pool_capacity, max_parallel)

    def task() -> None:
        with _hold_run_permits(state.permit_sem, permits):
            _execute_run(req_id, argv, cancel_ev, state)

    pool.submit(task)


def _stdin_reader(q: queue.Queue[dict[str, Any]]) -> None:
    fh = sys.stdin.buffer
    try:
        while True:
            raw = fh.readline()
            if raw == b"":
                break
            if len(raw) > MAX_LINE_BYTES:
                q.put({"command": "_oversize"})
                continue
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                req: Any = json.loads(line)
            except json.JSONDecodeError as e:
                q.put({"command": "_malformed", "error": str(e), "raw": line[:200]})
                continue
            if not isinstance(req, dict):
                q.put({"command": "_malformed", "error": "request must be a JSON object"})
                continue
            q.put(req)
    finally:
        q.put({"command": "_eof"})


def ipc_main() -> int:
    # Run setup then update check in one background thread.
    threading.Thread(
        target=_startup_sequence, daemon=True, name="gameca-startup"
    ).start()

    state = _IPCState()
    pool = ThreadPoolExecutor(max_workers=state.pool_capacity)
    q: queue.Queue[dict[str, Any]] = queue.Queue()
    reader_thread = threading.Thread(
        target=_stdin_reader, args=(q,), daemon=False, name="ipc-stdin"
    )
    reader_thread.start()
    try:
        while True:
            req = q.get()
            cmd = req.get("command")
            if cmd == "_eof":
                break
            if cmd == "_malformed":
                _emit(
                    {
                        "type": "error",
                        "id": "",
                        "payload": {"code": "INVALID_JSON", "msg": str(req.get("error", ""))},
                    }
                )
                continue
            if cmd == "_oversize":
                _emit(
                    {
                        "type": "error",
                        "id": "",
                        "payload": {
                            "code": "LINE_TOO_LARGE",
                            "msg": f"line exceeds {MAX_LINE_BYTES} bytes",
                        },
                    }
                )
                continue

            req_id = _normalize_request_id(req)
            if not isinstance(cmd, str):
                _emit(
                    {
                        "type": "error",
                        "id": req_id,
                        "payload": {"code": "INVALID_COMMAND", "msg": "command must be a string"},
                    }
                )
                continue

            try:
                if cmd == "ping":
                    _handle_ping(req_id)
                elif cmd == "cancel":
                    args = req.get("args") or {}
                    target = args.get("target_id")
                    if not target:
                        _emit(
                            {
                                "type": "error",
                                "id": req_id,
                                "payload": {"code": "INVALID_CANCEL", "msg": "args.target_id required"},
                            }
                        )
                    else:
                        _handle_cancel(state, req_id, str(target))
                elif cmd == "shutdown":
                    _emit({"type": "result", "id": req_id, "payload": {"ok": True}})
                    break
                elif cmd == "setup":
                    # Re-trigger setup (e.g. after an error) or check status.
                    # Returns immediately; progress events stream asynchronously.
                    if _is_setup_done():
                        _emit({"type": "setup", "phase": "done", "fraction": 1.0,
                               "message": "Environment ready."})
                        _emit({"type": "result", "id": req_id,
                               "payload": {"ok": True, "done": True}})
                    else:
                        threading.Thread(
                            target=_startup_sequence,
                            daemon=True, name="gameca-startup-retry",
                        ).start()
                        _emit({"type": "result", "id": req_id,
                               "payload": {"ok": True, "done": False}})
                elif cmd == "run":
                    args_obj = req.get("args") or {}
                    concurrent = bool(req.get("concurrent", False))
                    try:
                        max_parallel = int(args_obj.get("max_parallel", state.pool_capacity))
                    except (TypeError, ValueError):
                        max_parallel = state.pool_capacity
                    _submit_run(pool, state, req, req_id, concurrent, max_parallel)
                else:
                    _emit(
                        {
                            "type": "error",
                            "id": req_id,
                            "payload": {"code": "UNKNOWN_COMMAND", "msg": cmd},
                        }
                    )
            except Exception as e:
                payload: dict[str, Any] = {"code": "EXCEPTION", "msg": str(e)}
                if _debug_traceback():
                    payload["traceback"] = traceback.format_exc()
                _emit({"type": "error", "id": req_id, "payload": payload})
    finally:
        pool.shutdown(wait=True)
        reader_thread.join(timeout=30.0)
    _REAL_STDOUT.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(ipc_main())
