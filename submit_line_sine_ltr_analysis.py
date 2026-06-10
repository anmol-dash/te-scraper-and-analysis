#!/usr/bin/env python3
"""
submit_line_sine_ltr_analysis.py
Connect to HPC cluster, upload analysis code, and submit the LINE/SINE/LTR batch job.

Usage:
    python submit_line_sine_ltr_analysis.py \\
        --host login.cluster.edu --user myuser \\
        --l1mdt-expr  /home/amodz/anmol/L1Md_T_ultracombo.csv \\
        --b1mus2-expr /home/amodz/anmol/B1_Mus2_ultracombo.csv \\
        --iapltr1-expr /home/amodz/anmol/IAPLTR1_Mm_ultracombo.csv \\
        [--genome-fa /home/amodz/mm10.fa] \\
        [--key ~/.ssh/id_rsa] \\
        [--scheduler lsf|slurm|auto] \\
        [--queue normal] \\
        [--mem-mb 16000] \\
        [--cpus 8] \\
        [--walltime 08:00]

Output (stdout only — 2 lines):
    OUT <path>
    ERR <path>
"""

import argparse
import base64
import datetime
import getpass
import os
import shlex
import sys
import tempfile
import time
from pathlib import Path

# All progress messages go here so stdout stays clean
_STDERR = sys.stderr


def _log(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=_STDERR, flush=True)


try:
    import paramiko
except ImportError:
    print("ERROR: paramiko is required.  pip install paramiko", file=_STDERR)
    sys.exit(1)

# ── source files to upload ────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).resolve().parent
_UPLOAD_FILES = [
    # core pipeline
    "run_line_sine_ltr_analysis.py",
    "te_prep.py",
    "te_clustering.py",
    "te_expression.py",
    "requirements.txt",
    # Stage 11 runner scripts
    "run_phylo_analysis.py",
    "run_grna_offtarget.py",
    "run_transduction.py",
    "run_antisense_promoter.py",
    "run_ctcf_tad.py",
    "run_epigenetic_overlay.py",
    "run_ortholog_insertion.py",
    "run_multiassembly_liftover.py",
    "run_fold_prediction.py",
    "run_divergence.py",
    "run_ltr_struct.py",
    "run_subfamily.py",
    "run_benchmark.py",
    "run_motif_gain.py",
]


# ── SSH helpers ───────────────────────────────────────────────────────────────

def _connect(hostname: str, username: str, password: str,
             port: int = 22, key_path: str = "") -> paramiko.Transport:
    """Connect with keyboard-interactive (Duo 2FA) and optional key auth."""

    def kb_handler(title, instructions, prompt_list):
        responses = []
        for i, (prompt, _) in enumerate(prompt_list):
            p = prompt.lower()
            if i == 0 or "password" in p:
                responses.append(password)
            elif any(k in p for k in ("passcode","option","duo","factor","second")):
                _log("  [Duo] Sending push (approve on phone)…")
                responses.append("1")
            else:
                responses.append(password)
        return responses

    _log(f"Connecting to {hostname}:{port}...")
    transport = paramiko.Transport((hostname, port))
    transport.banner_timeout = 60
    transport.connect()

    # Key auth first
    if key_path:
        expanded = os.path.expanduser(key_path)
        for KeyCls in (paramiko.Ed25519Key, paramiko.RSAKey, paramiko.ECDSAKey):
            try:
                key = KeyCls.from_private_key_file(expanded)
                transport.auth_publickey(username, key)
                if transport.is_authenticated():
                    _log(f"  Key auth OK ({type(key).__name__})")
                    break
            except Exception:
                continue

    if not transport.is_authenticated():
        try:
            transport.auth_interactive(username, kb_handler)
        except paramiko.ssh_exception.AuthenticationException:
            _log("  Keyboard-interactive failed; trying password auth...")
            transport.auth_password(username, password)

    if not transport.is_authenticated():
        raise RuntimeError("Authentication failed")

    _log(f"Connected to {hostname} as {username}")
    return transport


def _run(transport: paramiko.Transport, cmd: str, timeout: int = 120) -> tuple:
    """Run a command; return (stdout, stderr, exit_code)."""
    ch = transport.open_session()
    ch.get_pty()
    ch.settimeout(timeout)
    ch.exec_command(f"HOME=/tmp bash -lc {shlex.quote(cmd)}")
    out, err = b"", b""
    while True:
        if ch.recv_ready():
            out += ch.recv(4096)
        if ch.recv_stderr_ready():
            err += ch.recv_stderr(4096)
        if ch.exit_status_ready():
            # Drain remaining
            while ch.recv_ready():
                out += ch.recv(4096)
            while ch.recv_stderr_ready():
                err += ch.recv_stderr(4096)
            break
        time.sleep(0.05)
    code = ch.recv_exit_status()
    ch.close()
    return out.decode(errors="replace"), err.decode(errors="replace"), code


def _upload_text(transport: paramiko.Transport, content: str,
                 remote_path: str, label: str):
    """Upload text content to remote_path using base64 chunk transfer."""
    encoded = base64.b64encode(content.encode()).decode()
    chunk_size = 4000
    chunks = [encoded[i:i+chunk_size] for i in range(0, len(encoded), chunk_size)]

    # Start fresh
    _run(transport, f"rm -f {shlex.quote(remote_path)}", timeout=15)
    _, _, code = _run(
        transport,
        f"echo {shlex.quote(chunks[0])} | base64 -d > {shlex.quote(remote_path)}",
        timeout=30,
    )
    if code != 0:
        raise RuntimeError(f"Upload of {label} chunk 1 failed (exit {code})")

    for i, chunk in enumerate(chunks[1:], 2):
        _, _, code = _run(
            transport,
            f"echo {shlex.quote(chunk)} | base64 -d >> {shlex.quote(remote_path)}",
            timeout=30,
        )
        if code != 0:
            raise RuntimeError(f"Upload of {label} chunk {i}/{len(chunks)} failed")

    _log(f"  Uploaded {label} ({len(content):,} bytes)")


def _upload_file(transport: paramiko.Transport, local_path: Path,
                 remote_path: str):
    """Try SFTP first, fall back to base64 upload."""
    sftp = None
    try:
        sftp = paramiko.SFTPClient.from_transport(transport)
        sftp.put(str(local_path), remote_path)
        _log(f"  SFTP upload: {local_path.name}")
        return
    except Exception as e:
        _log(f"  SFTP failed ({e}); using base64 fallback")
    finally:
        if sftp:
            sftp.close()

    content = local_path.read_text(encoding="utf-8", errors="replace")
    _upload_text(transport, content, remote_path, local_path.name)


def _detect_scheduler(transport: paramiko.Transport) -> str:
    out, _, _ = _run(transport,
        "command -v bsub && echo HAVE_LSF; command -v sbatch && echo HAVE_SLURM",
        timeout=15)
    if "HAVE_LSF" in out:
        _log("  Scheduler: LSF (bsub)")
        return "lsf"
    if "HAVE_SLURM" in out:
        _log("  Scheduler: Slurm (sbatch)")
        return "slurm"
    _log("  Scheduler: none detected — will use nohup fallback")
    return "none"


def _select_work_dir(transport: paramiko.Transport, username: str) -> str:
    probe = rf"""
set +e
for d in "/project/amodzlab/{username}/gameca" "/scratch/{username}/gameca" \
          "/work/{username}/gameca" "$HOME/gameca"; do
  mkdir -p "$d" 2>/dev/null && [ -w "$d" ] && printf '%s\n' "$d" && exit 0
done
mkdir -p "/tmp/{username}_gameca" && printf '/tmp/{username}_gameca\n'
"""
    out, _, _ = _run(transport, probe, timeout=30)
    selected = out.strip().splitlines()[-1].strip() if out.strip() else f"/tmp/{username}_gameca"
    _run(transport, f"mkdir -p {shlex.quote(selected)}", timeout=20)
    _log(f"  Remote work dir: {selected}")
    return selected


def _job_header(scheduler: str, job_name: str, job_out: str, job_err: str,
                mem_mb: int, cpus: int, walltime: str, queue: str) -> str:
    if scheduler == "slurm":
        mem_gb = max(1, mem_mb // 1000)
        return (
            f"#SBATCH --job-name={job_name}\n"
            f"#SBATCH --output={job_out}\n"
            f"#SBATCH --error={job_err}\n"
            f"#SBATCH --cpus-per-task={cpus}\n"
            f"#SBATCH --mem={mem_gb}G\n"
            f"#SBATCH --time={walltime}:00\n"
            f"#SBATCH --partition={queue}\n"
        )
    if scheduler == "lsf":
        return (
            f"#BSUB -J {job_name}\n"
            f"#BSUB -o {job_out}\n"
            f"#BSUB -e {job_err}\n"
            f"#BSUB -n {cpus}\n"
            f"#BSUB -M {mem_mb}\n"
            f"#BSUB -W {walltime}\n"
            f"#BSUB -q {queue}\n"
        )
    return ""   # nohup fallback


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Submit LINE/SINE/LTR analysis as HPC batch job",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--host",        required=True,  help="SSH hostname")
    p.add_argument("--user",        default=None,   help="SSH username (default: $USER)")
    p.add_argument("--port",        type=int, default=22)
    p.add_argument("--key",         default="",     help="SSH private key path")
    p.add_argument("--password",    default=None,
                   help="SSH password (prompted if omitted)")
    p.add_argument("--l1mdt-expr",  required=True,
                   help="Cluster path to L1Md_T_ultracombo.csv")
    p.add_argument("--b1mus2-expr", required=True,
                   help="Cluster path to B1_Mus2_ultracombo.csv")
    p.add_argument("--iapltr1-expr",required=True,
                   help="Cluster path to IAPLTR1_Mm_ultracombo.csv")
    p.add_argument("--reports-dir", default="/home/amodz/anmol/reports_line_sine_ltr",
                   help="Remote path for output figures (created on cluster if absent)")
    p.add_argument("--genome-fa",   default="",
                   help="Cluster path to mm10.fa (optional)")
    p.add_argument("--build",       default="mm10")
    p.add_argument("--source",      choices=["rmsk","dfam"], default="rmsk")
    p.add_argument("--max-loci",    type=int, default=None)
    p.add_argument("--scheduler",   choices=["lsf","slurm","auto"], default="auto")
    p.add_argument("--queue",       default="normal")
    p.add_argument("--mem-mb",      type=int, default=20000)
    p.add_argument("--cpus",        type=int, default=8)
    p.add_argument("--walltime",    default="12:00",
                   help="HH:MM wall time")
    p.add_argument("--modules",     default="",
                   help="Space-separated module load string, e.g. 'python/3.11 gcc/12'")
    return p.parse_args()


def main():
    args = parse_args()

    username = args.user or os.environ.get("USER", getpass.getuser())
    if args.password is None:
        args.password = getpass.getpass(
            f"Password for {username}@{args.host}: ", stream=_STDERR
        )

    # ── Connect — transport is closed automatically on any exit path ──────────
    transport = _connect(args.host, username, args.password, args.port, args.key)
    try:
        # ── Select work dir ──────────────────────────────────────────────────
        work_dir = _select_work_dir(transport, username)

        # ── Detect scheduler ─────────────────────────────────────────────────
        scheduler = args.scheduler
        if scheduler == "auto":
            scheduler = _detect_scheduler(transport)

        # ── Upload files ─────────────────────────────────────────────────────
        _log("Uploading analysis files...")
        for fname in _UPLOAD_FILES:
            local = _SCRIPT_DIR / fname
            if not local.exists():
                _log(f"  WARNING: {fname} not found locally — skipping")
                continue
            _upload_file(transport, local, f"{work_dir}/{fname}")

        # ── Build batch job script ────────────────────────────────────────────
        ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        job_name = f"gameca_lsl_{ts}"
        job_sh   = f"{work_dir}/{job_name}.sh"
        job_out  = f"{work_dir}/{job_name}.out"
        job_err  = f"{work_dir}/{job_name}.err"

        # Build pipeline CLI args
        analysis_args = (
            f"--l1mdt-expr  {shlex.quote(args.l1mdt_expr)} "
            f"--b1mus2-expr {shlex.quote(args.b1mus2_expr)} "
            f"--iapltr1-expr {shlex.quote(args.iapltr1_expr)} "
            f"--reports-dir {shlex.quote(args.reports_dir)} "
            f"--build {args.build} --source {args.source}"
        )
        if args.genome_fa:
            analysis_args += f" --genome-fa {shlex.quote(args.genome_fa)}"
        if args.max_loci:
            analysis_args += f" --max-loci {args.max_loci}"

        modules_block = ""
        if args.modules.strip():
            modules_block = "\n".join(f"module load {m}" for m in args.modules.split())

        header = _job_header(
            scheduler, job_name, job_out, job_err,
            args.mem_mb, args.cpus, args.walltime, args.queue,
        )

        job_script = f"""#!/bin/bash
{header}
{modules_block}

echo "[$(date)] GAMECA LINE/SINE/LTR analysis started"
echo "  Host: $(hostname)"
echo "  Work: {work_dir}"

# ── venv setup ──────────────────────────────────────────────────────────────
VENV="$HOME/gameca_venv"
PYTHON_BIN="python3"
command -v python3 >/dev/null 2>&1 || PYTHON_BIN=python

if [ ! -d "$VENV" ]; then
    echo "[$(date)] Creating venv at $VENV..."
    "$PYTHON_BIN" -m venv "$VENV" 2>&1 && echo "venv OK" || echo "venv FAILED"
fi
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
echo "[$(date)] Python: $(python --version 2>&1)"

# ── install dependencies ─────────────────────────────────────────────────────
REQ="{work_dir}/requirements.txt"
if [ -f "$REQ" ]; then
    echo "[$(date)] Installing requirements..."
    python -m pip install --upgrade pip setuptools wheel 2>&1 | tail -3 || true
    python -m pip install --prefer-binary -r "$REQ" 2>&1 | tail -10 || true
fi

# ── thread limits ────────────────────────────────────────────────────────────
THREADS={args.cpus}
[ -n "${{SLURM_CPUS_PER_TASK:-}}" ] && THREADS=$SLURM_CPUS_PER_TASK
[ -n "${{LSB_DJOB_NUMPROC:-}}"     ] && THREADS=$LSB_DJOB_NUMPROC
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS
export NUMBA_CACHE_DIR="${{TMPDIR:-/tmp}}/gameca_numba_$USER"
mkdir -p "$NUMBA_CACHE_DIR"

# ── ensure reports dir ───────────────────────────────────────────────────────
mkdir -p {shlex.quote(args.reports_dir)}

# ── run analysis ─────────────────────────────────────────────────────────────
cd {shlex.quote(work_dir)}
echo "[$(date)] Running analysis..."
python -u {shlex.quote(work_dir)}/run_line_sine_ltr_analysis.py {analysis_args}
EXIT_CODE=$?

echo ""
echo "[$(date)] Analysis finished with exit code $EXIT_CODE"
exit $EXIT_CODE
"""

        _log("Creating job script on cluster...")
        _, _, code = _run(
            transport,
            f"cat > {shlex.quote(job_sh)} << 'GAMECA_SCRIPT_EOF'\n{job_script}\nGAMECA_SCRIPT_EOF",
            timeout=30,
        )
        if code != 0:
            # Fallback: upload via base64
            _upload_text(transport, job_script, job_sh, "job script")

        _run(transport, f"chmod +x {shlex.quote(job_sh)}", timeout=10)

        # ── Submit job ───────────────────────────────────────────────────────
        _log("Submitting job...")
        if scheduler == "lsf":
            submit_cmd = f"bsub < {shlex.quote(job_sh)}"
        elif scheduler == "slurm":
            submit_cmd = f"sbatch {shlex.quote(job_sh)}"
        else:
            submit_cmd = (
                f"nohup bash {shlex.quote(job_sh)} "
                f"> {shlex.quote(job_out)} 2> {shlex.quote(job_err)} & echo $!"
            )

        out, err, code = _run(transport, submit_cmd, timeout=60)
        if code != 0:
            _log(f"  Submit failed (exit {code}): {err.strip()[:200]}")
            sys.exit(1)

        job_id = out.strip().splitlines()[-1].strip() if out.strip() else "(unknown)"
        _log(f"  Job submitted: {job_id}")

    finally:
        transport.close()
        _log("Connection closed.")

    # ── Final output (stdout only — these two lines) ──────────────────────────
    print(f"OUT {job_out}")
    print(f"ERR {job_err}")


if __name__ == "__main__":
    main()
