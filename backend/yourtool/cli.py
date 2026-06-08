"""Small argparse-based CLI used by the Tauri-side IPC bridge.

Supports cooperative cancellation via an optional threading.Event.
"""

from __future__ import annotations

import argparse
import base64
import builtins
import hashlib
import json
import mimetypes
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from threading import Event
from typing import Any, Sequence

# In the PyInstaller --onefile frozen bundle, __file__ is inside sys._MEIPASS/yourtool/,
# so parents[2] lands in the OS temp dir rather than the bundle root. All pipeline
# scripts (te_prep.py etc.) are bundled at sys._MEIPASS root (spec uses dest=".").
if getattr(sys, "frozen", False):
    _REPO_ROOT = Path(sys._MEIPASS)  # type: ignore[attr-defined]
else:
    _REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_HPC_CLIENT: Any | None = None


def main(argv: Sequence[str] | None = None, cancel_event: Event | None = None) -> int | dict[str, Any]:
    if argv is None:
        argv = sys.argv[1:]
    else:
        argv = list(argv)

    parser = argparse.ArgumentParser(prog="yourtool", description="Utility CLI (IPC backend).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_noop = sub.add_parser("noop", help="Exit successfully without doing I/O.")
    p_noop.set_defaults(handler=_cmd_noop)

    p_version = sub.add_parser("version", help="Print version to stdout and exit 0.")
    p_version.set_defaults(handler=_cmd_version)

    p_sleep = sub.add_parser("sleep", help="Sleep in short slices so cancel can interrupt.")
    p_sleep.add_argument("--seconds", type=float, default=1.0, help="Total time to sleep (default: 1).")
    p_sleep.set_defaults(handler=_cmd_sleep)

    p_hpc_connect = sub.add_parser("hpc-connect", help="Connect to an SSH/HPC host.")
    p_hpc_connect.add_argument("--host", required=True)
    p_hpc_connect.add_argument("--user", required=True)
    p_hpc_connect.add_argument("--password", default="", help="Leave blank when using --key for key-file authentication.")
    p_hpc_connect.add_argument("--key", default="", help="Path to an SSH private key file (e.g. ~/.ssh/id_ed25519 or a downloaded .key/.pem).")
    p_hpc_connect.add_argument("--port", type=int, default=22)
    p_hpc_connect.add_argument("--scheduler", choices=["auto", "lsf", "slurm"], default="auto")
    p_hpc_connect.add_argument("--work-dir", default="", help="Shared writable remote directory for uploads/jobs.")
    p_hpc_connect.set_defaults(handler=_cmd_hpc_connect)

    p_hpc_disconnect = sub.add_parser("hpc-disconnect", help="Close the active HPC connection.")
    p_hpc_disconnect.set_defaults(handler=_cmd_hpc_disconnect)

    p_hpc_upload = sub.add_parser("hpc-upload", help="Upload local pipeline files to the active HPC host.")
    p_hpc_upload.add_argument("files", nargs="+")
    p_hpc_upload.set_defaults(handler=_cmd_hpc_upload)

    p_hpc_run = sub.add_parser("hpc-run", help="Run a shell command on the active HPC host.")
    p_hpc_run.add_argument("--cmd", required=True)
    p_hpc_run.add_argument("--timeout", type=int, default=3600)
    p_hpc_run.set_defaults(handler=_cmd_hpc_run)

    p_hpc_batch = sub.add_parser("hpc-batch-submit", help="Configure and submit a GAMECA batch job.")
    p_hpc_batch.add_argument("--params", required=True, help="JSON object from the desktop UI.")
    p_hpc_batch.set_defaults(handler=_cmd_hpc_batch_submit)

    p_hpc_par = sub.add_parser("hpc-batch-parallel", help="Submit pipeline stages as parallel HPC jobs and stream status.")
    p_hpc_par.add_argument("--params", required=True, help="JSON object from the desktop UI (include max_jobs).")
    p_hpc_par.set_defaults(handler=_cmd_hpc_batch_parallel)

    p_hpc_status = sub.add_parser("hpc-job-status", help="Print detailed status for the latest submitted job.")
    p_hpc_status.set_defaults(handler=_cmd_hpc_job_status)

    p_hpc_peek = sub.add_parser("hpc-job-peek", help="Print scheduler diagnostics and job log tails.")
    p_hpc_peek.add_argument("--job-id", default="")
    p_hpc_peek.set_defaults(handler=_cmd_hpc_job_peek)

    p_hpc_retrieve = sub.add_parser("hpc-retrieve", help="Retrieve the latest HPC results.")
    p_hpc_retrieve.add_argument("--local-dir", required=True)
    p_hpc_retrieve.add_argument("--remote-dir", default="")
    p_hpc_retrieve.set_defaults(handler=_cmd_hpc_retrieve)

    p_hpc_list = sub.add_parser("hpc-list-dir", help="List a remote directory.")
    p_hpc_list.add_argument("--path", required=True)
    p_hpc_list.set_defaults(handler=_cmd_hpc_list_dir)

    p_hpc_read = sub.add_parser("hpc-read-file", help="Read a remote text or binary file.")
    p_hpc_read.add_argument("--path", required=True)
    p_hpc_read.add_argument("--max-mb", type=float, default=10.0)
    p_hpc_read.set_defaults(handler=_cmd_hpc_read_file)

    p_hpc_dl = sub.add_parser("hpc-download-file", help="Download a remote file to a local path.")
    p_hpc_dl.add_argument("--remote-path", required=True)
    p_hpc_dl.add_argument("--local-path", required=True)
    p_hpc_dl.set_defaults(handler=_cmd_hpc_download_file)

    p_hpc_dld = sub.add_parser("hpc-download-dir", help="Download a remote directory as a tar archive to a local path.")
    p_hpc_dld.add_argument("--remote-path", required=True)
    p_hpc_dld.add_argument("--local-path", required=True)
    p_hpc_dld.set_defaults(handler=_cmd_hpc_download_dir)

    p_dl_update = sub.add_parser("download-update", help="Download a GAMECA release DMG to ~/Downloads.")
    p_dl_update.add_argument("--url", required=True, help="Direct download URL for the DMG.")
    p_dl_update.add_argument("--filename", default="", help="Destination filename (inferred from URL if blank).")
    p_dl_update.set_defaults(handler=_cmd_download_update)

    p_notify_get = sub.add_parser("notify-get", help="Read saved Gmail notification credentials.")
    p_notify_get.set_defaults(handler=_cmd_notify_get)

    p_notify_set = sub.add_parser("notify-set", help="Save Gmail notification credentials.")
    p_notify_set.add_argument("--email", required=True, help="Gmail sender address.")
    p_notify_set.add_argument("--password", required=True, help="Gmail App Password (16 chars).")
    p_notify_set.set_defaults(handler=_cmd_notify_set)

    args = parser.parse_args(argv)
    handler = args.handler
    result = handler(args, cancel_event)
    return result if isinstance(result, dict) else int(result)


def _cmd_noop(_args: argparse.Namespace, _cancel_event: Event | None) -> int:
    return 0


def _cmd_version(_args: argparse.Namespace, _cancel_event: Event | None) -> int:
    from yourtool import __version__

    print(__version__, flush=True)
    return 0


def _cmd_sleep(args: argparse.Namespace, cancel_event: Event | None) -> int:
    deadline = time.monotonic() + max(0.0, args.seconds)
    while time.monotonic() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            print("cancelled", flush=True)
            return 130
        time.sleep(0.05)
    return 0


def _client_required():
    if _HPC_CLIENT is None or not getattr(_HPC_CLIENT, "connected", False):
        raise RuntimeError("Not connected to HPC. Connect first.")
    return _HPC_CLIENT


def _prefer_bundled_runtime() -> None:
    """Keep bundled scripts ahead of updater-downloaded copies for runtime imports."""
    repo_root = str(_REPO_ROOT)
    sys.path[:] = [p for p in sys.path if p != repo_root]
    sys.path.insert(0, repo_root)


def _decode_base64_output(text: str) -> bytes:
    b64_clean = re.sub(r"[^A-Za-z0-9+/=]", "", text)
    b64_clean += "=" * (-len(b64_clean) % 4)
    return base64.b64decode(b64_clean)


def _download_remote_file(
    client: Any,
    remote_path: str,
    local_path: Path,
    *,
    timeout: int,
    max_bytes: int | None = None,
    progress: bool = False,
) -> tuple[bool, str, int]:
    """Download a remote file using the best transport available on this client."""
    scp_get = getattr(client, "_scp_get", None)
    if callable(scp_get):
        try:
            if scp_get(remote_path, str(local_path), timeout=timeout):
                return True, "", 0
        except TypeError:
            if scp_get(remote_path, str(local_path)):
                return True, "", 0

    if getattr(client, "use_sftp", False) and getattr(client, "sftp", None):
        if progress:
            last_pct: list[int] = [-1]

            def _progress(transferred: int, total: int) -> None:
                if not total:
                    return
                pct = int(transferred * 100 / total)
                if pct - last_pct[0] >= 10:
                    last_pct[0] = pct
                    print(f"[SCP] {pct}%  ({transferred:,} / {total:,} bytes)", flush=True)

            client.sftp.get(remote_path, str(local_path), callback=_progress)
        else:
            client.sftp.get(remote_path, str(local_path))
        return True, "", 0

    if max_bytes is None:
        cmd = f"cat {json.dumps(remote_path)} | base64"
    else:
        cmd = f"head -c {max_bytes + 1} {json.dumps(remote_path)} | base64"
    out, err, code = client.run_command(cmd, timeout=timeout)
    if code != 0:
        return False, err, code
    local_path.write_bytes(_decode_base64_output(out))
    return True, "", 0


def _cmd_hpc_connect(args: argparse.Namespace, _cancel_event: Event | None) -> dict[str, Any]:
    global _HPC_CLIENT
    _prefer_bundled_runtime()
    from hpc_client import HPCClient

    if _HPC_CLIENT is not None and getattr(_HPC_CLIENT, "connected", False):
        _HPC_CLIENT.disconnect()
    client = HPCClient()
    ok = client.connect(args.host, args.user, args.password, args.port,
                        work_dir=args.work_dir, key_path=args.key)
    if not ok:
        return {"ok": False, "error": "connection failed", "exit_code": 1}
    if args.scheduler != "auto":
        client.scheduler = args.scheduler
        print(f"Scheduler overridden: {client.scheduler.upper()}", flush=True)
    _HPC_CLIENT = client
    return {
        "ok": True,
        "host": args.host,
        "scheduler": client.scheduler,
        "home": client.remote_work_dir,
        "has_internet": bool(getattr(client, "has_internet", False)),
    }


def _cmd_hpc_disconnect(_args: argparse.Namespace, _cancel_event: Event | None) -> dict[str, Any]:
    global _HPC_CLIENT
    if _HPC_CLIENT is not None:
        try:
            _HPC_CLIENT.disconnect()
        except Exception:
            pass
        _HPC_CLIENT = None
    return {"ok": True}


def _cmd_hpc_upload(args: argparse.Namespace, _cancel_event: Event | None) -> dict[str, Any]:
    # Derive scripts dir the same way main.py does (avoids circular import).
    if sys.platform == "win32":
        _gameca_dir = Path(os.environ.get("APPDATA", Path.home())) / "GAMECA"
    else:
        _gameca_dir = Path.home() / ".gameca"
    _scripts_dir = _gameca_dir / "scripts"
    client = _client_required()
    frozen = getattr(sys, "frozen", False)
    print(
        f"[UPLOAD] frozen={frozen}  REPO_ROOT={_REPO_ROOT}"
        f"  SCRIPTS_DIR={_scripts_dir} (exists={_scripts_dir.exists()})",
        flush=True,
    )
    uploaded: list[str] = []
    for name in args.files:
        # Bundled scripts (sys._MEIPASS) are always the authoritative version
        # for this release. Downloaded OTA scripts in _scripts_dir only serve
        # as a fallback when a file isn't present in the bundle at all.
        candidates: list[Path] = []
        candidates.append(_REPO_ROOT / name)
        if _scripts_dir.exists():
            candidates.append(_scripts_dir / name)

        local_path: Path | None = None
        for c in candidates:
            if c.exists() and c.is_file():
                local_path = c
                break

        print(f"[UPLOAD] {name}: searched {[str(c) for c in candidates]} → {local_path}", flush=True)

        if local_path is None:
            searched = ", ".join(str(c) for c in candidates)
            raise FileNotFoundError(
                f"Script '{name}' not found in bundle or scripts dir.\nSearched: {searched}"
            )
        remote_path = f"{client.remote_work_dir}/{local_path.name}"
        if not client._upload_text_file(local_path, remote_path, local_path.name):
            return {"ok": False, "error": f"failed to upload {name}", "exit_code": 1}
        uploaded.append(remote_path)
    return {"ok": True, "uploaded": uploaded}


def _cmd_hpc_run(args: argparse.Namespace, _cancel_event: Event | None) -> dict[str, Any]:
    client = _client_required()
    print(f"[HPC-RUN] user={client._username}  cmd={args.cmd[:300]}", flush=True)
    out, err, code = client.run_command(args.cmd, timeout=args.timeout, stream_output="summary")
    if code != 0:
        print(f"[HPC DIAG] hpc-run command failed: {args.cmd}", flush=True)
        print(f"[HPC DIAG] exit_code={code}", flush=True)
        if out.strip():
            print("[HPC DIAG] hpc-run stdout:", flush=True)
            print(out[-20_000:], flush=True)
        if err.strip():
            print("[HPC DIAG] hpc-run stderr:", flush=True)
            print(err[-20_000:], flush=True)
    return {"ok": code == 0, "exit_code": code, "stdout": out[-20_000:], "stderr": err[-20_000:]}


def _apply_hpc_params(client: Any, params: dict[str, Any]) -> None:
    mapping = {
        "family": "FAMILY_NAME",
        "species": "SPECIES",
        "assembly": "ASSEMBLY",
        "genome": "LOCAL_ASSEMBLY_PATH",
        "out_dir": "BASE_OUT_DIR",
        "input_csv": "all_te_file",
        "te_counts": "te_counts",
        "jaspar_bed": "JASPAR_BED_PATH",
        "jaspar_tabix": "JASPAR_TABIX_PATH",
        "modules": "MODULES",
        "notify_email": "NOTIFY_EMAIL",
        "mem_mb": "MEM_MB",
        "cpus": "CPUS",
        "walltime": "WALLTIME",
        "queue": "QUEUE",
        "kmer": "K",
        "primer_k": "PRIMER_K",
        "top_n_global": "TOP_N_GLOBAL",
        "top_n_cluster": "TOP_N_CLUSTER",
        "min_seq_clustering": "MIN_SEQUENCES_FOR_CLUSTERING",
        "primer_timeout": "PRIMER_TIMEOUT",
        "fetch_workers": "FETCH_WORKERS",
        "pca_dims": "PCA_DIMS",
        "n_epochs": "N_EPOCHS",
        "random_state": "RANDOM_STATE",
        "n_neighbors": "N_NEIGHBORS",
        "min_dist": "MIN_DIST",
        "min_cluster_size": "MIN_CLUSTER_SIZE",
        "min_samples": "MIN_SAMPLES",
        "p_threshold": "P_THRESHOLD",
        "skip_jaspar": "SKIP_JASPAR",
        "skip_primers": "SKIP_PRIMERS",
        "skip_go": "SKIP_GO",
        "skip_tsne": "SKIP_TSNE",
        "annot_source": "ANNOTATION_SOURCE",
        # Stage 11 / standout analysis module options
        "target_assemblies": "TARGET_ASSEMBLIES",
        "ortholog_species": "ORTHOLOG_SPECIES",
        "liftover_cmd": "LIFTOVER_CMD",
        "epigenetic_preset": "EPIGENETIC_PRESET",
        "ctcf_preset": "CTCF_PRESET",
        "tads_preset": "TADS_PRESET",
        "grna_cas": "GRNA_CAS",
        "grna_max_mm": "GRNA_MAX_MM",
        "grna_background": "GRNA_BACKGROUND",
        "colabfold_cmd": "COLABFOLD_CMD",
        "subst_rate": "SUBST_RATE",
        "clock_divisor": "CLOCK_DIVISOR",
        "intact_orf_aa": "INTACT_ORF_AA",
        "min_ltr_identity": "MIN_LTR_IDENTITY",
        "tail_bp": "TAIL_BP",
        "promoter_bp": "PROMOTER_BP",
        "cpg_omega": "CPG_OMEGA",
        "mafft_cmd": "MAFFT_CMD",
    }
    for source, dest in mapping.items():
        if source in params and params[source] not in (None, ""):
            client.params[dest] = params[source]


def _cmd_hpc_batch_submit(args: argparse.Namespace, _cancel_event: Event | None) -> dict[str, Any]:
    params = json.loads(args.params)
    family = params.get("family", "?")
    print(f"[Submit] Building job script for {family}…", flush=True)
    client = _client_required()
    if not isinstance(params, dict):
        raise ValueError("--params must be a JSON object")
    _apply_hpc_params(client, params)

    old_input = builtins.input
    try:
        builtins.input = lambda _prompt="": "y"
        ok = bool(client.submit_batch_job())
    finally:
        builtins.input = old_input

    return {
        "ok": ok,
        "job_id": client.current_job_id,
        "output_dir": client.remote_output_dir,
        "work_dir": client.remote_work_dir,
        "exit_code": 0 if ok else 1,
    }


def _cmd_hpc_batch_parallel(args: argparse.Namespace, cancel_event: Event | None) -> dict[str, Any]:
    params = json.loads(args.params)
    if not isinstance(params, dict):
        raise ValueError("--params must be a JSON object")
    family = params.get("family", "?")
    max_jobs = int(params.get("max_jobs", 10))
    print(f"[Parallel] Preparing parallel pipeline for {family} (max_jobs={max_jobs})…", flush=True)
    client = _client_required()
    _apply_hpc_params(client, params)
    old_input = builtins.input
    try:
        builtins.input = lambda _prompt="": "y"
        job_infos = client.submit_parallel_pipeline(max_jobs=max_jobs)
    finally:
        builtins.input = old_input

    if not job_infos:
        return {"ok": False, "error": "No jobs were submitted", "exit_code": 1}

    print(f"[Parallel] Monitoring {len(job_infos)} jobs — polling every 30 s…", flush=True)

    terminal = {"DONE", "FAILED", "UNKNOWN"}
    states = {ji["job_id"]: "PENDING" for ji in job_infos}

    while not (cancel_event and cancel_event.is_set()):
        time.sleep(30)
        all_done = True
        for ji in job_infos:
            jid = ji["job_id"]
            if states[jid] in terminal:
                continue
            new_state = client._get_scheduler_state(jid)
            if new_state != states[jid]:
                print(f"[Job {ji['stage']}] {states[jid]} → {new_state}", flush=True)
                states[jid] = new_state
                if new_state in terminal:
                    print(f"[JOB DONE] {ji['stage']}: job {jid} finished ({new_state})", flush=True)
            if new_state not in terminal:
                all_done = False
        if all_done:
            break

    print("[Parallel] All jobs reached terminal state.", flush=True)
    return {"ok": True, "jobs": job_infos, "exit_code": 0}


def _cmd_hpc_job_status(_args: argparse.Namespace, _cancel_event: Event | None) -> dict[str, Any]:
    client = _client_required()
    info = client.check_job_status()
    return {"ok": info is not None, "job_id": (info or {}).get("JOB_ID"), "exit_code": 0 if info is not None else 1}


def _cmd_hpc_job_peek(args: argparse.Namespace, _cancel_event: Event | None) -> dict[str, Any]:
    client = _client_required()
    ok = bool(client.peek_job_output(args.job_id))
    return {"ok": ok, "exit_code": 0 if ok else 1}


def _cmd_hpc_retrieve(args: argparse.Namespace, _cancel_event: Event | None) -> dict[str, Any]:
    client = _client_required()
    old_input = builtins.input
    try:
        builtins.input = lambda prompt="": "n" if "Generate clustering plots" in prompt else "y"
        result = client.retrieve_results(args.local_dir, remote_out_override=args.remote_dir or None)
    finally:
        builtins.input = old_input
    ok = result is not False
    return {"ok": ok, "exit_code": 0 if ok else 1}


def _cmd_hpc_list_dir(args: argparse.Namespace, _cancel_event: Event | None) -> dict[str, Any]:
    print(f"[Nav] {args.path}", flush=True)
    client = _client_required()
    if client.use_sftp and client.sftp:
        entries = []
        for attr in client.sftp.listdir_attr(args.path):
            entries.append({
                "name": attr.filename,
                "path": f"{args.path.rstrip('/')}/{attr.filename}",
                "size": int(attr.st_size or 0),
                "is_dir": bool(attr.st_mode and (attr.st_mode & 0o040000)),
                "mtime": int(attr.st_mtime or 0),
            })
        return {"ok": True, "path": args.path, "entries": entries}
    out, err, code = client.run_command(
        f"find {json.dumps(args.path)} -maxdepth 1 -mindepth 1 -printf '%f\\t%p\\t%s\\t%y\\t%T@\\n'",
        timeout=30,
    )
    if code != 0:
        return {"ok": False, "error": err, "exit_code": code}
    entries = []
    for line in out.splitlines():
        name, path, size, kind, mtime = (line.split("\t", 4) + ["", "", "", "", ""])[:5]
        entries.append({"name": name, "path": path, "size": int(size or 0), "is_dir": kind == "d", "mtime": float(mtime or 0)})
    return {"ok": True, "path": args.path, "entries": entries}


def _cmd_hpc_download_file(args: argparse.Namespace, _cancel_event: Event | None) -> dict[str, Any]:
    client = _client_required()
    local = Path(args.local_path)
    local.parent.mkdir(parents=True, exist_ok=True)
    filename = Path(args.remote_path).name
    print(f"[SCP] Downloading {filename}…", flush=True)
    # Priority: local scp (fastest, OS-level timeout) → SFTP → SSH base64
    ok, err, code = _download_remote_file(
        client,
        args.remote_path,
        local,
        timeout=120,
        progress=True,
    )
    if not ok:
        return {"ok": False, "error": err, "exit_code": code}
    size = local.stat().st_size if local.exists() else 0
    print(f"[SCP] Done — {size:,} bytes → {local}", flush=True)
    return {"ok": True, "local_path": str(local)}


def _cmd_hpc_download_dir(args: argparse.Namespace, _cancel_event: Event | None) -> dict[str, Any]:
    client = _client_required()
    local = Path(args.local_path)
    local.parent.mkdir(parents=True, exist_ok=True)
    remote = args.remote_path.rstrip("/")
    dirname = remote.split("/")[-1] or "download"
    print(f"[SCP] Downloading directory {dirname}…", flush=True)
    out, err, code = client.run_command(
        f"tar -czf - -C {json.dumps(remote + '/..')} {json.dumps(dirname)} | base64",
        timeout=300,
    )
    if code != 0:
        return {"ok": False, "error": err, "exit_code": code}
    import re as _re
    b64_clean = _re.sub(r"[^A-Za-z0-9+/=]", "", out)
    b64_clean += "=" * (-len(b64_clean) % 4)
    local.write_bytes(base64.b64decode(b64_clean))
    size = local.stat().st_size if local.exists() else 0
    print(f"[SCP] Done — {size:,} bytes → {local}", flush=True)
    return {"ok": True, "local_path": str(local)}


def _cmd_hpc_read_file(args: argparse.Namespace, _cancel_event: Event | None) -> dict[str, Any]:
    print(f"[Open] {Path(args.path).name}", flush=True)
    client = _client_required()
    max_bytes = max(1, int(args.max_mb * 1024 * 1024))
    mime = mimetypes.guess_type(args.path)[0] or "application/octet-stream"
    ext = Path(args.path).suffix.lower()

    # Download to a local temp file.  Priority: scp (fastest) → SFTP → SSH base64.
    remote_hash = hashlib.sha256(args.path.encode("utf-8")).hexdigest()[:16]
    tmp = Path(tempfile.gettempdir()) / "gameca_preview" / f"{remote_hash}_{Path(args.path).name}"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    ok, err, code = _download_remote_file(
        client,
        args.path,
        tmp,
        timeout=60,
        max_bytes=None if ext in (".html", ".htm") else max_bytes,
    )
    if not ok:
        return {"ok": False, "error": err, "exit_code": code}

    # HTML: served from the temp file via gamecapreview:// Tauri scheme; no IPC transfer.
    if ext in (".html", ".htm"):
        return {"ok": True, "type": "local_html", "local_path": str(tmp), "mime": mime}

    data = tmp.read_bytes()
    if len(data) > max_bytes:
        return {"ok": False, "error": f"file exceeds {args.max_mb:g} MB", "exit_code": 1}

    try:
        text = data.decode("utf-8")
        return {"ok": True, "type": "text", "mime": "text/plain", "text": text, "size": len(data)}
    except UnicodeDecodeError:
        return {"ok": True, "type": "binary", "mime": mime, "data": base64.b64encode(data).decode(), "size": len(data)}


_NOTIFY_STATE_FILE = Path.home() / ".hpc_te_state.json"
_NOTIFY_EMAIL_KEY = "gmail_sender_email"
_NOTIFY_PW_KEY = "gmail_app_password"


def _load_notify_state() -> dict[str, Any]:
    try:
        return json.loads(_NOTIFY_STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_notify_state(state: dict[str, Any]) -> None:
    _NOTIFY_STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _cmd_notify_get(_args: argparse.Namespace, _cancel_event: Event | None) -> dict[str, Any]:
    state = _load_notify_state()
    email = state.get(_NOTIFY_EMAIL_KEY, "").strip()
    pw = state.get(_NOTIFY_PW_KEY, "").strip()
    return {"ok": True, "email": email, "configured": bool(email and pw)}


def _cmd_notify_set(args: argparse.Namespace, _cancel_event: Event | None) -> dict[str, Any]:
    email = args.email.strip()
    pw = args.password.strip().replace(" ", "")
    if not email or not pw:
        return {"ok": False, "error": "email and password are required"}
    state = _load_notify_state()
    state[_NOTIFY_EMAIL_KEY] = email
    state[_NOTIFY_PW_KEY] = pw
    _save_notify_state(state)
    print(f"  Notification credentials saved for {email}", flush=True)
    return {"ok": True, "email": email}


def _cmd_download_update(args: argparse.Namespace, cancel_event: Event | None) -> dict[str, Any]:
    """Download a release DMG to ~/Downloads with progress, then open it."""
    import urllib.request
    import urllib.error

    url      = args.url
    filename = args.filename.strip() or url.split("?")[0].split("/")[-1] or "GAMECA_update.dmg"
    dest_dir = Path.home() / "Downloads"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename

    print(f"Downloading {filename}…", flush=True)

    total_bytes = 0
    downloaded  = 0

    def _reporthook(count: int, block: int, total: int) -> None:
        nonlocal downloaded, total_bytes
        total_bytes = total
        downloaded  = min(total, count * block)
        if total > 0:
            pct   = int(downloaded * 100 / total)
            mb_d  = downloaded / 1_048_576
            mb_t  = total      / 1_048_576
            print(f"  {mb_d:.0f} / {mb_t:.0f} MB ({pct}%)", flush=True)
        if cancel_event and cancel_event.is_set():
            raise InterruptedError("cancelled")

    try:
        urllib.request.urlretrieve(url, str(dest), _reporthook)
    except InterruptedError:
        return {"ok": False, "error": "download cancelled", "exit_code": 130}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "exit_code": 1}

    print(f"Saved to {dest}", flush=True)

    # Open the DMG so Finder mounts it and shows the drag-to-Applications window.
    import subprocess
    subprocess.Popen(["open", str(dest)])

    return {"ok": True, "path": str(dest), "filename": filename}
