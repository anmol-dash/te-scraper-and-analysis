#!/usr/bin/env python3
"""
submit_fold_prediction.py
Connect to HPC, upload code, and submit the fold prediction batch job.

Usage:
    python submit_fold_prediction.py \\
        --host login.your-cluster.edu --user youruser \\
        --input /path/to/mt2_mm_ultracombo_countsv4.csv \\
        --reports-dir ~/gameca_reports \\
        [--family MT2_Mm] \\
        [--min-aa 100] \\
        [--top-n 5] \\
        [--source-seqs 100] \\
        [--colabfold-cmd colabfold_batch] \\
        [--singularity-image colabfold.sif [--singularity-source colabfold.def]] \\
        [--gpu] \\
        [--scheduler lsf|slurm|auto]

Output (stdout — 2 lines):
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
import time
from pathlib import Path

_STDERR = sys.stderr


def _log(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=_STDERR, flush=True)


try:
    import paramiko
except ImportError:
    print("ERROR: paramiko required.  pip install paramiko", file=_STDERR)
    sys.exit(1)

_SCRIPT_DIR  = Path(__file__).resolve().parent
_UPLOAD_FILES = [
    "run_fold_prediction.py",
    "requirements.txt",
    "colabfold.def",          # used when --singularity-source colabfold.def
]


# ── SSH helpers (identical pattern to submit_grna_analysis.py) ────────────────

def _connect(hostname, username, password, port=22, key_path=""):
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
    transport.connect()

    if key_path:
        for KeyCls in (paramiko.Ed25519Key, paramiko.RSAKey, paramiko.ECDSAKey):
            try:
                key = KeyCls.from_private_key_file(key_path)
                transport.auth_publickey(username, key)
                if transport.is_authenticated():
                    _log(f"  Connected (key auth)")
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


def _run(transport, cmd, timeout=120):
    ch = transport.open_session()
    ch.get_pty()
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


def _upload_text(transport, content, remote_path, label):
    encoded = base64.b64encode(content.encode()).decode()
    chunks  = [encoded[i:i+4000] for i in range(0, len(encoded), 4000)]
    _run(transport, f"rm -f {shlex.quote(remote_path)}", timeout=15)
    _run(transport,
         f"echo {shlex.quote(chunks[0])} | base64 -d > {shlex.quote(remote_path)}",
         )
    for chunk in chunks[1:]:
        _run(transport,
             f"echo {shlex.quote(chunk)} | base64 -d >> {shlex.quote(remote_path)}",
             )
    _log(f"  Uploaded {label} ({len(content):,} bytes)")


def _upload_file(transport, local_path: Path, remote_path: str):
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
    _upload_text(transport, local_path.read_text(errors="replace"),
                 remote_path, local_path.name)


def _select_work_dir(transport, username):
    probe = f"""
set +e
for d in "/project/{username}/gameca" "/scratch/{username}/gameca" \
          "/work/{username}/gameca" "$HOME/gameca"; do
  mkdir -p "$d" 2>/dev/null && [ -w "$d" ] && printf '%s\\n' "$d" && exit 0
done
mkdir -p "/tmp/{username}_gameca" && printf '/tmp/{username}_gameca\\n'
"""
    out, _, _ = _run(transport, probe, timeout=30)
    selected  = out.strip().splitlines()[-1].strip() if out.strip() else f"/tmp/{username}_gameca"
    _run(transport, f"mkdir -p {shlex.quote(selected)}", timeout=20)
    _log(f"  Remote work dir: {selected}")
    return selected


def _check_colabfold(transport, cmd: str, work_dir: str) -> str | None:
    """Return the colabfold_batch path on the cluster, or None if not found."""
    search = f"""
set +e
which {shlex.quote(cmd)} 2>/dev/null && exit 0
for p in \
  "$HOME/localcolabfold/colabfold-conda/bin/colabfold_batch" \
  "{work_dir}/localcolabfold/colabfold-conda/bin/colabfold_batch" \
  "$HOME/.conda/envs/colabfold/bin/colabfold_batch" \
  "/usr/local/bin/colabfold_batch" \
  "/opt/colabfold/bin/colabfold_batch"; do
  [ -x "$p" ] && echo "$p" && exit 0
done
exit 1
"""
    out, _, code = _run(transport, search, timeout=20)
    if code == 0 and out.strip():
        return out.strip().splitlines()[-1].strip()
    return None


def _detect_scheduler(transport):
    out, _, _ = _run(transport,
        "command -v bsub && echo HAVE_LSF; command -v sbatch && echo HAVE_SLURM",
        )
    if "HAVE_LSF"  in out: return "lsf"
    if "HAVE_SLURM" in out: return "slurm"
    return "none"


def _job_header(scheduler, job_name, job_out, job_err,
                mem_mb, cpus, walltime, queue, gpu=False):
    gpu_lsf   = '#BSUB -gpu "num=1:mode=shared:j_exclusive=no"\n' if gpu else ""
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


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Submit fold prediction batch job to HPC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--host",           required=True)
    p.add_argument("--user",           default=None)
    p.add_argument("--port",           type=int, default=22)
    p.add_argument("--key",            default="")
    p.add_argument("--password",       default=None)
    p.add_argument("--input",          required=True,
                   help="Local or remote path to input CSV")
    p.add_argument("--reports-dir",    default="~/gameca_reports")
    p.add_argument("--family",         default="MT2_Mm")
    p.add_argument("--min-aa",         type=int, default=100)
    p.add_argument("--top-n",          type=int, default=5)
    p.add_argument("--source-seqs",    type=int, default=100)
    p.add_argument("--consensus-fasta", default="",
                   help="Consensus nucleotide FASTA to fold (#11)")
    p.add_argument("--per-cluster",    action="store_true",
                   help="Build + fold a per-cluster consensus (#11)")
    p.add_argument("--colabfold-cmd",  default="colabfold_batch",
                   help="colabfold_batch executable on the cluster")
    p.add_argument("--singularity-image", default="",
                   help="Run ColabFold inside this .sif on the cluster (built "
                        "automatically if missing). Skips the host colabfold_batch "
                        "check/install entirely.")
    p.add_argument("--singularity-source", default="",
                   help="Source to build the .sif from when it is missing: a "
                        "docker:///library:// URI or 'colabfold.def' (uploaded with "
                        "the job). Blank = run_fold_prediction.py's default docker URI.")
    p.add_argument("--build-singularity", action="store_true",
                   help="Force (re)build of --singularity-image even if it exists.")
    p.add_argument("--num-recycles",   type=int, default=3)
    p.add_argument("--num-models",     type=int, default=1)
    p.add_argument("--gpu",            action="store_true",
                   help="Request a GPU node (strongly recommended for ColabFold)")
    p.add_argument("--scheduler",      choices=["lsf","slurm","auto"], default="auto")
    p.add_argument("--queue",          default="normal")
    p.add_argument("--mem-mb",         type=int, default=32000)
    p.add_argument("--cpus",           type=int, default=8)
    p.add_argument("--walltime",       default="12:00")
    p.add_argument("--modules",        default="",
                   help="Space-separated HPC modules to load (e.g. 'cuda/11.8 colabfold')")
    p.add_argument("--force",          action="store_true",
                   help="Pass --force to run script (re-run ColabFold even if cached)")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    username = args.user or os.environ.get("USER", getpass.getuser())
    if args.password is None:
        args.password = getpass.getpass(
            f"Password for {username}@{args.host}: ", stream=_STDERR)

    transport = _connect(args.host, username, args.password, args.port, args.key)
    work_dir  = _select_work_dir(transport, username)

    # ── ColabFold availability check ──────────────────────────────────────────
    # When running inside a Singularity image, colabfold_batch lives in the
    # container — skip the host-side check/install entirely.
    _install_colabfold = False
    cf_found = None if args.singularity_image else \
        _check_colabfold(transport, args.colabfold_cmd, work_dir)
    if args.singularity_image:
        _log(f"  Using Singularity image: {args.singularity_image} "
             "(host colabfold_batch check skipped)")
    if cf_found:
        _log(f"  colabfold_batch found: {cf_found}")
        if cf_found != args.colabfold_cmd:
            _log(f"  (updating --colabfold-cmd to {cf_found})")
            args.colabfold_cmd = cf_found
    elif not args.singularity_image:
        _log(f"  colabfold_batch not found on {args.host}")
        if sys.stdin.isatty():
            resp = input(
                f"  Install localColabFold on {args.host}? "
                "(downloads ~20 GB of model weights, adds 20-30 min to first run) [y/N]: "
            ).strip().lower()
        else:
            resp = "n"
            _log("  (non-interactive session; skipping ColabFold install)")
        if resp in ("y", "yes"):
            _install_colabfold = True
            cf_bin = (f"{work_dir}/localcolabfold/"
                      "colabfold-conda/bin/colabfold_batch")
            _log(f"  Will install localColabFold; binary will be at {cf_bin}")
            args.colabfold_cmd = cf_bin
        else:
            _log("  ColabFold will not be installed. "
                 "ORF extraction runs; folding will be skipped by the run script.")

    scheduler = args.scheduler
    if scheduler == "auto":
        scheduler = _detect_scheduler(transport)

    # ── Upload analysis files ─────────────────────────────────────────────────
    _log("Uploading analysis files...")
    for fname in _UPLOAD_FILES:
        local = _SCRIPT_DIR / fname
        if local.exists():
            _upload_file(transport, local, f"{work_dir}/{fname}")
        else:
            _log(f"  WARNING: {fname} not found locally — skipping")

    # ── Upload input CSV if local ─────────────────────────────────────────────
    csv_remote = args.input
    local_csv  = Path(args.input)
    if local_csv.exists() and local_csv.is_file():
        remote_csv = f"{work_dir}/{local_csv.name}"
        _log(f"Uploading input CSV ({local_csv.stat().st_size / 1024:.0f} KB)...")
        _upload_file(transport, local_csv, remote_csv)
        csv_remote = remote_csv
    else:
        _log(f"Using remote CSV path: {csv_remote}")

    # ── Build job script ──────────────────────────────────────────────────────
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = f"gameca_fold_{ts}"
    job_sh   = f"{work_dir}/{job_name}.sh"
    job_out  = f"{work_dir}/{job_name}.out"
    job_err  = f"{work_dir}/{job_name}.err"

    force_flag = "--force" if args.force else ""
    cons_flag = (f"--consensus-fasta {shlex.quote(args.consensus_fasta)} "
                 if args.consensus_fasta else "")
    per_cluster_flag = "--per-cluster " if args.per_cluster else ""

    # Singularity: forward the image + (optional) build source. The run script
    # builds the .sif on the cluster if it is missing. Without a GPU node (--gpu)
    # the container must run CPU-only, so pass --no-gpu in that case.
    sing_flag = ""
    if args.singularity_image:
        sing_flag = f"--singularity-image {shlex.quote(args.singularity_image)} "
        if args.singularity_source:
            sing_flag += f"--singularity-source {shlex.quote(args.singularity_source)} "
        if args.build_singularity:
            sing_flag += "--build-singularity "
        if not args.gpu:
            sing_flag += "--no-gpu "

    analysis_args = (
        f"--input {shlex.quote(csv_remote)} "
        f"--reports-dir {shlex.quote(args.reports_dir)} "
        f"--family {shlex.quote(args.family)} "
        f"--min-aa {args.min_aa} "
        f"--top-n {args.top_n} "
        f"--source-seqs {args.source_seqs} "
        f"{cons_flag}{per_cluster_flag}{sing_flag}"
        f"--colabfold-cmd {shlex.quote(args.colabfold_cmd)} "
        f"--num-recycles {args.num_recycles} "
        f"--num-models {args.num_models} "
        f"{force_flag}"
    ).strip()

    modules_block = ""
    if args.modules.strip():
        modules_block = "\n".join(f"module load {m}" for m in args.modules.split())

    header = _job_header(scheduler, job_name, job_out, job_err,
                         args.mem_mb, args.cpus, args.walltime, args.queue, args.gpu)

    colabfold_install_block = ""
    if _install_colabfold:
        colabfold_install_block = f"""
# --- localColabFold install (runs only if binary absent) ---
CF_BIN={shlex.quote(args.colabfold_cmd)}
if [ ! -x "$CF_BIN" ]; then
    echo "[$(date)] Installing localColabFold into {work_dir}/localcolabfold ..."
    cd {shlex.quote(work_dir)}
    wget -q https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/main/install_colabfold_linux.sh
    chmod +x install_colabfold_linux.sh
    bash install_colabfold_linux.sh 2>&1
    echo "[$(date)] localColabFold install complete"
fi
export PATH="$(dirname {shlex.quote(args.colabfold_cmd)}):$PATH"
cd {shlex.quote(work_dir)}
"""

    job_script = f"""#!/bin/bash
{header}
{modules_block}

echo "[$(date)] GAMECA fold prediction started"
echo "  Host: $(hostname)"

VENV="$HOME/gameca_venv"
PYTHON_BIN="python3"
command -v python3 >/dev/null 2>&1 || PYTHON_BIN=python
[ ! -d "$VENV" ] && "$PYTHON_BIN" -m venv "$VENV" 2>&1
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
echo "[$(date)] Python: $(python --version 2>&1)"

REQ="{work_dir}/requirements.txt"
if [ -f "$REQ" ]; then
    python -m pip install --upgrade pip setuptools wheel 2>&1 | tail -3 || true
    python -m pip install --prefer-binary -r "$REQ" 2>&1 | tail -10 || true
fi

THREADS={args.cpus}
[ -n "${{SLURM_CPUS_PER_TASK:-}}" ] && THREADS=$SLURM_CPUS_PER_TASK
[ -n "${{LSB_DJOB_NUMPROC:-}}"     ] && THREADS=$LSB_DJOB_NUMPROC
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS
export NUMBA_CACHE_DIR="${{TMPDIR:-/tmp}}/gameca_numba_$USER"
mkdir -p "$NUMBA_CACHE_DIR"

mkdir -p {shlex.quote(args.reports_dir)}
{colabfold_install_block}
# Verify colabfold is accessible (host check is irrelevant when using a container)
if [ -z "{args.singularity_image}" ] && ! command -v {shlex.quote(args.colabfold_cmd)} >/dev/null 2>&1; then
    echo "[$(date)] WARNING: {args.colabfold_cmd} not found in PATH."
    echo "  The ORF map will still be generated; structure prediction will be skipped."
    echo "  Install ColabFold:  pip install colabfold[alphafold]"
    echo "  Or load the HPC module:  module load colabfold"
fi

cd {shlex.quote(work_dir)}
echo "[$(date)] Running fold prediction..."
python -u {shlex.quote(work_dir)}/run_fold_prediction.py {analysis_args}
EXIT_CODE=$?

echo "[$(date)] Finished with exit code $EXIT_CODE"
exit $EXIT_CODE
"""

    _log("Creating job script on cluster...")
    _, _, code = _run(
        transport,
        f"cat > {shlex.quote(job_sh)} << 'GAMECA_FOLD_EOF'\n{job_script}\nGAMECA_FOLD_EOF",
    )
    if code != 0:
        _upload_text(transport, job_script, job_sh, "job script")
    _run(transport, f"chmod +x {shlex.quote(job_sh)}", timeout=10)

    # ── Submit ────────────────────────────────────────────────────────────────
    _log("Submitting...")
    if scheduler == "lsf":
        submit_cmd = f"bsub < {shlex.quote(job_sh)}"
    elif scheduler == "slurm":
        submit_cmd = f"sbatch {shlex.quote(job_sh)}"
    else:
        submit_cmd = (f"nohup bash {shlex.quote(job_sh)} "
                      f"> {shlex.quote(job_out)} 2> {shlex.quote(job_err)} & echo $!")

    out, err, code = _run(transport, submit_cmd, timeout=60)
    if code != 0:
        _log(f"  Submit failed (exit {code}): {err.strip()[:200]}")
        transport.close()
        sys.exit(1)

    job_id = out.strip().splitlines()[-1].strip() if out.strip() else "(unknown)"
    _log(f"  Job submitted: {job_id}")
    transport.close()

    print(f"OUT {job_out}")
    print(f"ERR {job_err}")


if __name__ == "__main__":
    main()
