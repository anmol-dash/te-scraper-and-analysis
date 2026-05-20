"""Small argparse-based CLI used by the Tauri-side IPC bridge.

Supports cooperative cancellation via an optional threading.Event.
"""

from __future__ import annotations

import argparse
import builtins
import json
import mimetypes
import os
import sys
import time
from pathlib import Path
from threading import Event
from typing import Any, Sequence

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
    p_hpc_connect.add_argument("--password", required=True)
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


def _cmd_hpc_connect(args: argparse.Namespace, _cancel_event: Event | None) -> dict[str, Any]:
    global _HPC_CLIENT
    from hpc_client import HPCClient

    if _HPC_CLIENT is not None and getattr(_HPC_CLIENT, "connected", False):
        _HPC_CLIENT.disconnect()
    client = HPCClient()
    ok = client.connect(args.host, args.user, args.password, args.port, work_dir=args.work_dir)
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
    }


def _cmd_hpc_disconnect(_args: argparse.Namespace, _cancel_event: Event | None) -> dict[str, Any]:
    global _HPC_CLIENT
    if _HPC_CLIENT is not None and getattr(_HPC_CLIENT, "connected", False):
        _HPC_CLIENT.disconnect()
    _HPC_CLIENT = None
    return {"ok": True}


def _cmd_hpc_upload(args: argparse.Namespace, _cancel_event: Event | None) -> dict[str, Any]:
    client = _client_required()
    uploaded: list[str] = []
    for name in args.files:
        local_path = (_REPO_ROOT / name).resolve()
        try:
            local_path.relative_to(_REPO_ROOT)
        except ValueError as exc:
            raise ValueError(f"file is outside repository: {name}") from exc
        if not local_path.exists() or not local_path.is_file():
            raise FileNotFoundError(str(local_path))
        remote_path = f"{client.remote_work_dir}/{local_path.name}"
        if not client._upload_text_file(local_path, remote_path, local_path.name):
            return {"ok": False, "error": f"failed to upload {name}", "exit_code": 1}
        uploaded.append(remote_path)
    return {"ok": True, "uploaded": uploaded}


def _cmd_hpc_run(args: argparse.Namespace, _cancel_event: Event | None) -> dict[str, Any]:
    client = _client_required()
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
    }
    for source, dest in mapping.items():
        if source in params and params[source] not in (None, ""):
            client.params[dest] = params[source]


def _cmd_hpc_batch_submit(args: argparse.Namespace, _cancel_event: Event | None) -> dict[str, Any]:
    client = _client_required()
    params = json.loads(args.params)
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
    if client.use_sftp and client.sftp:
        client.sftp.get(args.remote_path, str(local))
    else:
        out, err, code = client.run_command(
            f"cat {json.dumps(args.remote_path)} | base64",
            timeout=120,
        )
        if code != 0:
            return {"ok": False, "error": err, "exit_code": code}
        import base64, re as _re
        b64_clean = _re.sub(r"[^A-Za-z0-9+/=]", "", out)
        b64_clean += "=" * (-len(b64_clean) % 4)
        local.write_bytes(base64.b64decode(b64_clean))
    return {"ok": True, "local_path": str(local)}


def _cmd_hpc_download_dir(args: argparse.Namespace, _cancel_event: Event | None) -> dict[str, Any]:
    client = _client_required()
    local = Path(args.local_path)
    local.parent.mkdir(parents=True, exist_ok=True)
    remote = args.remote_path.rstrip("/")
    dirname = remote.split("/")[-1] or "download"
    out, err, code = client.run_command(
        f"tar -czf - -C {json.dumps(remote + '/..')} {json.dumps(dirname)} | base64",
        timeout=300,
    )
    if code != 0:
        return {"ok": False, "error": err, "exit_code": code}
    import base64, re as _re
    b64_clean = _re.sub(r"[^A-Za-z0-9+/=]", "", out)
    b64_clean += "=" * (-len(b64_clean) % 4)
    local.write_bytes(base64.b64decode(b64_clean))
    return {"ok": True, "local_path": str(local)}


def _cmd_hpc_read_file(args: argparse.Namespace, _cancel_event: Event | None) -> dict[str, Any]:
    client = _client_required()
    max_bytes = max(1, int(args.max_mb * 1024 * 1024))
    if client.use_sftp and client.sftp:
        with client.sftp.file(args.path, "rb") as fh:
            data = fh.read(max_bytes + 1)
    else:
        out, err, code = client.run_command(
            f"head -c {max_bytes + 1} {json.dumps(args.path)} | base64",
            timeout=30,
        )
        if code != 0:
            return {"ok": False, "error": err, "exit_code": code}
        import base64, re as _re
        b64_clean = _re.sub(r"[^A-Za-z0-9+/=]", "", out)
        b64_clean += "=" * (-len(b64_clean) % 4)
        data = base64.b64decode(b64_clean)
    if len(data) > max_bytes:
        return {"ok": False, "error": f"file exceeds {args.max_mb:g} MB", "exit_code": 1}
    mime = mimetypes.guess_type(args.path)[0] or "application/octet-stream"
    try:
        text = data.decode("utf-8")
        return {"ok": True, "type": "text", "mime": "text/plain", "text": text, "size": len(data)}
    except UnicodeDecodeError:
        import base64
        return {"ok": True, "type": "binary", "mime": mime, "data": base64.b64encode(data).decode(), "size": len(data)}
