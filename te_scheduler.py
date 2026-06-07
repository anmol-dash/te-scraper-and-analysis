#!/usr/bin/env python3
"""
te_scheduler.py --- GAMECA shared HPC scheduler abstraction (#10)

One place for SSH + job-submission logic across all submit_*.py scripts, with
native support for LSF (bsub), Slurm (sbatch), and local (nohup) execution.
Extracted from submit_fold_prediction.py so every analysis can submit with the
same Duo-aware auth, work-dir probing, scheduler auto-detection, and venv
bootstrap --- without copy-pasting ~380 lines per feature.

Typical use in a thin submit_<feature>.py:

    import te_scheduler as sched
    args = sched.base_parser("Submit my analysis").parse_args()
    sched.submit(
        args,
        run_script="run_my_analysis.py",
        upload_files=["run_my_analysis.py", "te_scheduler.py", "requirements.txt"],
        analysis_args="--input {csv} --reports-dir {reports} --family {family}",
        job_prefix="gameca_myfeat",
    )
"""

import argparse
import base64
import datetime
import getpass
import os
import shlex
import sys
import time
from pathlib import Path

_STDERR = sys.stderr


def _log(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=_STDERR, flush=True)


try:
    import paramiko
except ImportError:                                              # pragma: no cover
    paramiko = None


# ── SSH ──────────────────────────────────────────────────────────────────────

def connect(hostname, username, password, port=22, key_path=""):
    if paramiko is None:
        _log("ERROR: paramiko required.  pip install paramiko")
        sys.exit(1)

    def kb_handler(title, instructions, prompt_list):
        responses = []
        for i, (prompt, _) in enumerate(prompt_list):
            p = prompt.lower()
            if i == 0 or "password" in p:
                responses.append(password)
            elif any(k in p for k in ("passcode", "option", "duo", "factor", "second")):
                _log("  [Duo] Sending push (approve on phone)...")
                responses.append("1")
            else:
                responses.append(password)
        return responses

    _log(f"Connecting to {hostname}:{port}...")
    transport = paramiko.Transport((hostname, port))
    transport.banner_timeout = 60
    transport.connect()

    if key_path:
        for KeyCls in (paramiko.Ed25519Key, paramiko.RSAKey, paramiko.ECDSAKey):
            try:
                key = KeyCls.from_private_key_file(key_path)
                transport.auth_publickey(username, key)
                if transport.is_authenticated():
                    _log("  Connected (key auth)")
                    return transport
            except Exception:
                continue

    transport.auth_interactive(username, kb_handler)
    if not transport.is_authenticated():
        transport.auth_password(username, password)
    if not transport.is_authenticated():
        _log("ERROR: Authentication failed")
        sys.exit(1)
    _log(f"  Connected to {hostname} as {username}")
    return transport


def run(transport, cmd, timeout=120):
    ch = transport.open_session()
    ch.get_pty()
    ch.settimeout(timeout)
    ch.exec_command(f"HOME=/tmp bash -lc {shlex.quote(cmd)}")
    out, err = b"", b""
    while True:
        if ch.recv_ready():        out += ch.recv(4096)
        if ch.recv_stderr_ready(): err += ch.recv_stderr(4096)
        if ch.exit_status_ready():
            while ch.recv_ready():        out += ch.recv(4096)
            while ch.recv_stderr_ready(): err += ch.recv_stderr(4096)
            break
        time.sleep(0.05)
    code = ch.recv_exit_status()
    ch.close()
    return out.decode(errors="replace"), err.decode(errors="replace"), code


def upload_text(transport, content, remote_path, label):
    encoded = base64.b64encode(content.encode()).decode()
    chunks = [encoded[i:i+4000] for i in range(0, len(encoded), 4000)]
    run(transport, f"rm -f {shlex.quote(remote_path)}", timeout=15)
    run(transport, f"echo {shlex.quote(chunks[0])} | base64 -d > {shlex.quote(remote_path)}",
        timeout=30)
    for chunk in chunks[1:]:
        run(transport, f"echo {shlex.quote(chunk)} | base64 -d >> {shlex.quote(remote_path)}",
            timeout=30)
    _log(f"  Uploaded {label} ({len(content):,} bytes)")


def upload_file(transport, local_path: Path, remote_path: str):
    sftp = None
    try:
        sftp = paramiko.SFTPClient.from_transport(transport)
        sftp.put(str(local_path), remote_path)
        _log(f"  SFTP upload: {local_path.name}")
        return
    except Exception as e:                                       # noqa: BLE001
        _log(f"  SFTP failed ({e}); using base64 fallback")
    finally:
        if sftp:
            sftp.close()
    upload_text(transport, local_path.read_text(errors="replace"),
                remote_path, local_path.name)


def select_work_dir(transport, username):
    probe = f"""
set +e
for d in "/project/amodzlab/{username}/gameca" "/scratch/{username}/gameca" \
          "/work/{username}/gameca" "$HOME/gameca"; do
  mkdir -p "$d" 2>/dev/null && [ -w "$d" ] && printf '%s\\n' "$d" && exit 0
done
mkdir -p "/tmp/{username}_gameca" && printf '/tmp/{username}_gameca\\n'
"""
    out, _, _ = run(transport, probe, timeout=30)
    selected = out.strip().splitlines()[-1].strip() if out.strip() else f"/tmp/{username}_gameca"
    run(transport, f"mkdir -p {shlex.quote(selected)}", timeout=20)
    _log(f"  Remote work dir: {selected}")
    return selected


# ── scheduler ────────────────────────────────────────────────────────────────

def detect_scheduler(transport):
    out, _, _ = run(transport,
                    "command -v bsub && echo HAVE_LSF; command -v sbatch && echo HAVE_SLURM",
                    timeout=15)
    if "HAVE_LSF" in out:   return "lsf"
    if "HAVE_SLURM" in out: return "slurm"
    return "none"


def job_header(scheduler, job_name, job_out, job_err,
               mem_mb, cpus, walltime, queue, gpu=False):
    gpu_lsf = '#BSUB -gpu "num=1:mode=shared:j_exclusive=no"\n' if gpu else ""
    gpu_slurm = "#SBATCH --gres=gpu:1\n" if gpu else ""
    if scheduler == "slurm":
        mem_gb = max(1, mem_mb // 1000)
        return (f"#SBATCH --job-name={job_name}\n"
                f"#SBATCH --output={job_out}\n"
                f"#SBATCH --error={job_err}\n"
                f"#SBATCH --cpus-per-task={cpus}\n"
                f"#SBATCH --mem={mem_gb}G\n"
                f"#SBATCH --time={walltime}:00\n"
                f"#SBATCH --partition={queue}\n"
                f"{gpu_slurm}")
    if scheduler == "lsf":
        return (f"#BSUB -J {job_name}\n"
                f"#BSUB -o {job_out}\n"
                f"#BSUB -e {job_err}\n"
                f"#BSUB -n {cpus}\n"
                f"#BSUB -M {mem_mb}\n"
                f"#BSUB -W {walltime}\n"
                f"#BSUB -q {queue}\n"
                f"{gpu_lsf}")
    return ""


# ── argument parser shared by submit_*.py ──────────────────────────────────────

def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", required=True)
    p.add_argument("--user", default=None)
    p.add_argument("--port", type=int, default=22)
    p.add_argument("--key", default="")
    p.add_argument("--password", default=None)
    p.add_argument("--input", required=True, help="Local or remote path to input CSV")
    p.add_argument("--reports-dir", default="/home/amodz/anmol/reports4")
    p.add_argument("--family", default="TE")
    p.add_argument("--scheduler", choices=["lsf", "slurm", "local", "auto"], default="auto")
    p.add_argument("--queue", default="normal")
    p.add_argument("--mem-mb", type=int, default=16000)
    p.add_argument("--cpus", type=int, default=4)
    p.add_argument("--walltime", default="08:00")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--modules", default="",
                   help="Space-separated HPC modules to load")
    return p


def submit(args, run_script: str, upload_files: list, analysis_args: str,
           job_prefix: str, extra_setup: str = ""):
    """Full submit flow. analysis_args may contain {csv},{reports},{family} placeholders."""
    username = args.user or os.environ.get("USER", getpass.getuser())
    if args.password is None and args.scheduler != "local-here":
        args.password = getpass.getpass(
            f"Password for {username}@{args.host}: ", stream=_STDERR)

    transport = connect(args.host, username, args.password, args.port, args.key)
    work_dir = select_work_dir(transport, username)

    scheduler = args.scheduler
    if scheduler == "auto":
        scheduler = detect_scheduler(transport)
    if scheduler == "none":
        scheduler = "local"

    _log("Uploading analysis files...")
    script_dir = Path(__file__).resolve().parent
    for fname in upload_files:
        local = script_dir / fname
        if local.exists():
            upload_file(transport, local, f"{work_dir}/{fname}")
        else:
            _log(f"  WARNING: {fname} not found locally --- skipping")

    csv_remote = args.input
    local_csv = Path(args.input)
    if local_csv.exists() and local_csv.is_file():
        csv_remote = f"{work_dir}/{local_csv.name}"
        _log(f"Uploading input CSV ({local_csv.stat().st_size/1024:.0f} KB)...")
        upload_file(transport, local_csv, csv_remote)
    else:
        _log(f"Using remote CSV path: {csv_remote}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = f"{job_prefix}_{ts}"
    job_sh = f"{work_dir}/{job_name}.sh"
    job_out = f"{work_dir}/{job_name}.out"
    job_err = f"{work_dir}/{job_name}.err"

    filled = analysis_args.format(csv=shlex.quote(csv_remote),
                                  reports=shlex.quote(args.reports_dir),
                                  family=shlex.quote(args.family))
    modules_block = "\n".join(f"module load {m}" for m in args.modules.split()) \
        if args.modules.strip() else ""
    header = job_header(scheduler, job_name, job_out, job_err,
                        args.mem_mb, args.cpus, args.walltime, args.queue, args.gpu)

    job_script = f"""#!/bin/bash
{header}
{modules_block}

echo "[$(date)] {job_prefix} started on $(hostname)"
VENV="$HOME/gameca_venv"
PYTHON_BIN="python3"; command -v python3 >/dev/null 2>&1 || PYTHON_BIN=python
[ ! -d "$VENV" ] && "$PYTHON_BIN" -m venv "$VENV" 2>&1
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
echo "[$(date)] Python: $(python --version 2>&1)"

REQ="{work_dir}/requirements.txt"
if [ -f "$REQ" ]; then
    python -m pip install --upgrade pip 2>&1 | tail -2 || true
    python -m pip install --prefer-binary -r "$REQ" 2>&1 | tail -8 || true
fi

THREADS={args.cpus}
[ -n "${{SLURM_CPUS_PER_TASK:-}}" ] && THREADS=$SLURM_CPUS_PER_TASK
[ -n "${{LSB_DJOB_NUMPROC:-}}"     ] && THREADS=$LSB_DJOB_NUMPROC
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
mkdir -p {shlex.quote(args.reports_dir)}
{extra_setup}
cd {shlex.quote(work_dir)}
echo "[$(date)] Running {run_script}..."
python -u {shlex.quote(work_dir)}/{run_script} {filled}
EXIT_CODE=$?
echo "[$(date)] Finished with exit code $EXIT_CODE"
exit $EXIT_CODE
"""

    _log("Creating job script on cluster...")
    _, _, code = run(transport,
                     f"cat > {shlex.quote(job_sh)} << 'GAMECA_EOF'\n{job_script}\nGAMECA_EOF",
                     timeout=30)
    if code != 0:
        upload_text(transport, job_script, job_sh, "job script")
    run(transport, f"chmod +x {shlex.quote(job_sh)}", timeout=10)

    _log(f"Submitting via {scheduler}...")
    if scheduler == "lsf":
        submit_cmd = f"bsub < {shlex.quote(job_sh)}"
    elif scheduler == "slurm":
        submit_cmd = f"sbatch {shlex.quote(job_sh)}"
    else:
        submit_cmd = (f"nohup bash {shlex.quote(job_sh)} "
                      f"> {shlex.quote(job_out)} 2> {shlex.quote(job_err)} & echo $!")

    out, err, code = run(transport, submit_cmd, timeout=60)
    if code != 0:
        _log(f"  Submit failed (exit {code}): {err.strip()[:200]}")
        transport.close()
        sys.exit(1)
    job_id = out.strip().splitlines()[-1].strip() if out.strip() else "(unknown)"
    _log(f"  Job submitted: {job_id}")
    transport.close()
    print(f"OUT {job_out}")
    print(f"ERR {job_err}")
