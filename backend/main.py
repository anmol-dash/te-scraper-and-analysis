"""
JSON-lines IPC server: reads NDJSON requests from stdin, emits NDJSON on stdout.
"""

from __future__ import annotations

import os
os.environ.setdefault("CRYPTOGRAPHY_OPENSSL_NO_LEGACY", "1")

import inspect
import json
import logging
import hashlib
import queue
import re
import shutil
import ssl
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

from yourtool import __version__ as _TOOL_VERSION  # noqa: E402  (import follows sys.path setup above)
from yourtool.cli import main as cli_main  # noqa: E402

_REAL_STDOUT: TextIO = sys.__stdout__

_emit_lock = threading.Lock()

# ── Venv setup ─────────────────────────────────────────────────────────────────

if sys.platform == "win32":
    _GAMECA_DIR = Path(os.environ.get("APPDATA", Path.home())) / "GAMECA"
else:
    _GAMECA_DIR = Path.home() / ".gameca"

_VENV_PATH         = _GAMECA_DIR / "venv"
_SCRIPTS_DIR       = _GAMECA_DIR / "scripts"
# Pipeline sources use PEP 604 unions (`str | None`) evaluated at import time,
# so anything below 3.10 dies on `import hpc_client`. pyproject asks for 3.11.
_MIN_PY = (3, 11)
# requirements.txt pins numpy<2.0, which has no wheels above 3.12 and will not
# build from source there. Prefer an interpreter inside [_MIN_PY, _MAX_PY];
# newer is only used as a last resort.
_MAX_PY = (3, 12)
# Without these the pipeline cannot run at all, so their absence is a hard
# setup failure even when other packages install cleanly.
_CORE_PACKAGES = ("paramiko", "numpy", "pandas")
_SETUP_DONE_SENTINEL = _GAMECA_DIR / ".setup_done"
_LAST_COMMIT_FILE  = _GAMECA_DIR / "last_commit"
_REQ_HASH_FILE     = _GAMECA_DIR / "requirements.hash"
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
    "te_notify.py", "fetch_jaspar.py",
    "run_fold_prediction.py",
    # ── standout analysis modules ──
    "run_phylo_analysis.py", "run_grna_analysis.py", "run_grna_offtarget.py",
    "run_transduction.py",
    "run_antisense_promoter.py", "run_ctcf_tad.py", "run_epigenetic_overlay.py",
    "run_ortholog_insertion.py", "run_multiassembly_liftover.py",
    "run_motif_gain.py", "te_motif_gain.py",
    "run_stage11_all.py", "run_divergence.py", "run_ltr_struct.py",
    "run_subfamily.py", "run_benchmark.py", "make_report_figures.py",
    # ── shared libs + infra ──
    "te_moods_scan.py", "te_divergence.py", "te_ltr_struct.py",
    "te_subfamily.py", "te_overlay.py", "te_scheduler.py", "te_provenance.py",
    "gameca_pipeline.py", "gameca.yaml",
    "requirements.txt",
]


def _find_requirements() -> Path | None:
    # The synced copy from repo HEAD wins, so a self-updated requirements.txt
    # drives the venv rebuild rather than the stale bundled one.
    synced = _SCRIPTS_DIR / "requirements.txt"
    if synced.exists():
        return synced
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


def _python_version(exe: str) -> tuple[int, ...] | None:
    """Version of the interpreter at `exe`, or None if it won't run."""
    try:
        r = subprocess.run(
            [exe, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    try:
        return tuple(int(p) for p in r.stdout.strip().split("."))
    except ValueError:
        return None


def _find_python() -> tuple[str | None, list[str]]:
    """Locate an interpreter that is at least _MIN_PY.

    A GUI-launched app inherits a minimal PATH, so a bare `python3` lookup finds
    macOS system Python 3.9 and builds a venv the pipeline cannot import. Probe
    version-qualified names and the usual install prefixes before falling back.

    Returns (interpreter, rejected) where `rejected` describes what was too old,
    so the setup panel can say *why* nothing suitable was found.
    """
    # Preferred band first (3.12 → 3.11), then anything newer as a fallback.
    preferred = [f"python3.{m}" for m in range(_MAX_PY[1], _MIN_PY[1] - 1, -1)]
    newer = [f"python3.{m}" for m in range(20, _MAX_PY[1], -1)]
    names = preferred + ["python3", "python"] + newer
    prefixes = [
        Path("/opt/homebrew/bin"), Path("/usr/local/bin"), Path("/usr/bin"),
        Path("/Library/Frameworks/Python.framework/Versions"),
    ]

    candidates: list[str] = []
    for n in names:
        found = shutil.which(n)
        if found:
            candidates.append(found)
    for prefix in prefixes:
        if prefix.name == "Versions" and prefix.is_dir():
            for ver in sorted(prefix.iterdir(), reverse=True):
                candidates.append(str(ver / "bin" / "python3"))
        else:
            candidates.extend(str(prefix / n) for n in names)

    rejected: list[str] = []
    seen: set[str] = set()
    usable: list[tuple[tuple[int, ...], str]] = []
    for exe in candidates:
        if exe in seen or not Path(exe).exists():
            continue
        seen.add(exe)
        ver = _python_version(exe)
        if ver is None:
            continue
        if ver < _MIN_PY:
            rejected.append(f"{exe} ({'.'.join(str(p) for p in ver)})")
            continue
        usable.append((ver, exe))

    # Highest version inside the supported band wins; only if the band is empty
    # do we accept something newer, which may fail to build pinned wheels.
    in_band = [(v, e) for v, e in usable if v <= _MAX_PY]
    if in_band:
        return max(in_band)[1], rejected
    if usable:
        return min(usable)[1], rejected
    return None, rejected


def _venv_python() -> Path:
    if sys.platform == "win32":
        return _VENV_PATH / "Scripts" / "python.exe"
    return _VENV_PATH / "bin" / "python"


def _is_setup_done() -> bool:
    if not (_SETUP_DONE_SENTINEL.exists() and (_VENV_PATH / "pyvenv.cfg").exists()):
        return False
    # A venv built by an older build may sit on 3.9 and fail every pipeline
    # import. Treat that as unfinished setup so it gets rebuilt rather than
    # surfacing later as a bogus "paramiko is required".
    ver = _python_version(str(_venv_python()))
    if ver is None or ver < _MIN_PY:
        _setup_log(
            f"Existing venv is Python {'.'.join(str(p) for p in ver) if ver else 'unusable'}; "
            f"rebuilding on {'.'.join(str(p) for p in _MIN_PY)}+."
        )
        _wipe_venv()
        return False
    # A changed requirements.txt (self-updated from repo HEAD) means the venv is
    # stale even though it exists and is the right Python. Rebuild it. This is
    # the automatic force-reinstall: the heavy rebuild fires only when the
    # dependency set actually changed, not on every script push.
    want = _requirements_hash(_find_requirements())
    have = _REQ_HASH_FILE.read_text().strip() if _REQ_HASH_FILE.exists() else ""
    if want and want != have:
        _setup_log(
            f"requirements.txt changed ({have[:12] or 'none'} → {want[:12]}); "
            "rebuilding the environment."
        )
        _wipe_venv()
        return False
    return True


def _wipe_venv() -> None:
    shutil.rmtree(_VENV_PATH, ignore_errors=True)
    _SETUP_DONE_SENTINEL.unlink(missing_ok=True)


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

        want = ".".join(str(p) for p in _MIN_PY)
        py, rejected = _find_python()
        if py is None:
            for r in rejected:
                _setup_log(f"Too old, skipped: {r}")
            detail = (f" Found {rejected[0]}, but {want}+ is required."
                      if rejected else "")
            _emit({"type": "setup", "phase": "error",
                   "message": f"No Python {want}+ found on this system.{detail} "
                              f"Install Python {want}+ and relaunch."})
            return
        for r in rejected:
            _setup_log(f"Too old, skipped: {r}")

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

        # Upgrade pip and log its output. setuptools and wheel are seeded
        # explicitly: 3.12+ venvs no longer ship setuptools, and any dependency
        # still using the legacy build backend then dies with
        # "Cannot import 'setuptools.build_meta'", taking the whole install
        # down — which surfaces later as a missing paramiko.
        _setup_log("Upgrading pip, setuptools, wheel…")
        pip_up = subprocess.run(
            [pip, "install", "--upgrade", "pip", "setuptools", "wheel"],
            capture_output=True, text=True,
        )
        for ln in (pip_up.stdout + pip_up.stderr).splitlines():
            if ln.strip():
                _setup_log(ln)
        if pip_up.returncode != 0:
            _setup_log(f"Build-backend bootstrap failed (exit {pip_up.returncode}); "
                       "package installs may fail.")

        # Strip inline comments too: `pip install -r` does this itself, but we
        # install per-spec, and passing "pkg>=1.0  # why" straight to
        # `pip install` is a parse error that looks like an unavailable package.
        req_lines = []
        for raw in req.read_text().splitlines():
            spec = re.sub(r"(?:^|\s)#.*$", "", raw).strip()
            if spec:
                req_lines.append(spec)
        total = max(1, len(req_lines))
        done = 0

        _setup_log(f"Installing {total} requirement(s) from {req.name}…")
        _emit({"type": "setup", "phase": "installing", "fraction": 0.05,
               "message": f"Installing {total} packages…"})

        # Install one package at a time. requirements.txt is not satisfiable as a
        # single resolve (colabfold wants numpy>=2.0.2 while the pins cap it below
        # 2.0), and `pip install -r` aborts the whole run on that conflict — so a
        # single unsatisfiable extra used to leave the venv completely empty and
        # the app reporting a missing paramiko.
        failed_pkgs: list[str] = []
        for spec in req_lines:
            proc = subprocess.Popen(
                [pip, "install", "--no-cache-dir", spec],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    _setup_log(line)
            proc.wait()

            done += 1
            frac = min(0.95, 0.05 + 0.90 * done / total)
            if proc.returncode != 0:
                failed_pkgs.append(spec)
                _setup_log(f"Skipped {spec} (pip exit {proc.returncode})")
            _emit({"type": "setup", "phase": "installing", "fraction": frac,
                   "message": f"{done}/{total} — {spec}"})

        # Optional extras are allowed to fail; the core set is not.
        vpy = str(_venv_python())
        missing_core = [
            p for p in _CORE_PACKAGES
            if subprocess.run([vpy, "-c", f"import {p}"],
                              capture_output=True).returncode != 0
        ]
        if missing_core:
            _setup_log(f"Core packages missing after install: {', '.join(missing_core)}")
            _emit({"type": "setup", "phase": "error",
                   "message": f"Setup failed — could not install: {', '.join(missing_core)}. "
                              "See the log for the pip error."})
            return
        if failed_pkgs:
            _setup_log(
                f"{len(failed_pkgs)} optional package(s) unavailable: "
                f"{', '.join(failed_pkgs)}. Core environment is usable."
            )

        _SETUP_DONE_SENTINEL.touch()
        # Record which requirements this venv was built from, so the next launch
        # rebuilds only if the file changed.
        _REQ_HASH_FILE.write_text(_requirements_hash(req))
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


def _ssl_context() -> ssl.SSLContext | None:
    """An SSL context backed by certifi's CA bundle, or None to use the default.

    The frozen binary ships no system trust store, so urllib's default context
    fails every GitHub handshake with CERTIFICATE_VERIFY_FAILED and the whole
    self-update path silently gives up. certifi is bundled for exactly this. In
    a dev run certifi may be absent — fall back to the default context, which
    finds the OS certs there.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


def _urlopen(url: str, timeout: float = 6.0):
    """urlopen with a certifi-backed context and the GitHub UA/Accept headers."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "GAMECA-updater/1.0",
                 "Accept": "application/vnd.github.v3+json"},
    )
    ctx = _ssl_context()
    if ctx is not None:
        return urllib.request.urlopen(req, timeout=timeout, context=ctx)
    return urllib.request.urlopen(req, timeout=timeout)


def _requirements_hash(req: Path | None) -> str:
    if req is None:
        return ""
    try:
        return hashlib.sha256(req.read_bytes()).hexdigest()
    except OSError:
        return ""


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
        with _urlopen(url) as resp:
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
            with _urlopen(raw_url, timeout=10) as resp:
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

    # A changed requirements.txt is not patched in place here — that path used a
    # single-shot `pip install -r`, which aborts on the unsatisfiable pins. The
    # venv is instead rebuilt by _run_setup_background(), whose requirements-hash
    # gate (see _is_setup_done) fires because update runs before setup at
    # startup and the freshly-synced file now hashes differently.
    if "requirements.txt" in updated:
        _update_log("requirements.txt changed — venv will be rebuilt during setup.")

    # Ensure new scripts shadow bundled ones immediately (for this session).
    scripts_str = str(_SCRIPTS_DIR)
    if scripts_str not in sys.path:
        sys.path.insert(0, scripts_str)

    # Only claim this commit once every script actually landed. Recording it
    # after a partial download leaves a cache that mixes files from two commits
    # yet reports itself up to date, so the next launch never retries.
    if failed:
        _update_log(
            f"{len(failed)} file(s) failed; keeping last_commit at "
            f"{stored_sha[:7] or '(none)'} so the next launch retries."
        )
        _emit({"type": "update", "phase": "partial",
               "message": f"Updated {len(updated)} file(s), {len(failed)} failed — "
                          f"will retry on next launch.",
               "sha": short_sha, "updated": updated, "failed": failed})
        return

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
        with _urlopen(url) as resp:
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
    # Update first: pull the latest scripts + requirements.txt from repo HEAD so
    # setup builds the venv from the current dependency set and its hash gate can
    # fire in this same launch. On first run or offline, the sync is a no-op and
    # setup falls back to the bundled requirements.
    _run_update_check()
    _run_setup_background()
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
    # Close any active SSH transport so blocking paramiko reads unblock immediately.
    try:
        from yourtool.cli import _HPC_CLIENT  # noqa: PLC0415
        if _HPC_CLIENT is not None:
            _HPC_CLIENT.disconnect()
    except Exception:
        pass
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
