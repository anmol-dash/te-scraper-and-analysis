#!/usr/bin/env python3
_SCRIPT_BUILD = "20260519-3"  # bump this when changing job script logic
"""
HPC Client for TE Analysis Pipeline

Interactive client that connects to an HPC cluster, submits analysis
jobs, and allows you to monitor progress and retrieve results.

Supports both LSF (bsub) and Slurm (sbatch) schedulers — auto-detected
on connect or selectable with --scheduler.

Workflow:
    1. Connect to HPC cluster (SSH)
    2. Configure parameters (family name, input files, genome path, …)
    3. Submit batch or interactive job
    4. Monitor progress / retrieve results

Usage:
    python hpc_client.py
    python hpc_client.py --host cluster.university.edu --user myuser
    python hpc_client.py --host cluster.edu --scheduler slurm

Requirements:
    pip install paramiko
"""

import base64
import datetime
import getpass
import argparse
import io
import json
import os
import re
import shlex
import stat
import sys
import tarfile
import tempfile
import zipfile
from email.mime.text import MIMEText
from pathlib import Path

_STATE_FILE = Path.home() / ".hpc_te_state.json"

try:
    import paramiko
except ImportError:
    print("Error: paramiko is required. Install with: pip install paramiko")
    sys.exit(1)


def _log(message: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


class HPCClient:
    """Interactive client for running TE analysis on HPC cluster via batch jobs."""

    def __init__(self):
        self.ssh = None
        self.sftp = None
        self.connected = False
        self.scheduler = None  # 'lsf' or 'slurm', auto-detected on connect

        # Default parameters
        self.params = {
            "FAMILY_NAME": "MT2_Mm",
            "SPECIES": "mouse",
            "ASSEMBLY": "mm10",      # UCSC assembly/build used for automatic loading
            "LOCAL_ASSEMBLY_PATH": "",  # Optional genome FASTA path on HPC
            "JASPAR_BED_PATH": "",      # Optional pre-downloaded JASPAR BED/BED.GZ on HPC
            "JASPAR_TABIX_PATH": "",    # bgzip+tabix-indexed JASPAR .bed.gz (reusable across families)
            "P_THRESHOLD": 0.05,        # Fisher p-value significance threshold for motif/GO
            "BASE_OUT_DIR": "results",
            "K": 6,
            "PCA_DIMS": 40,
            "N_EPOCHS": 120,
            "RANDOM_STATE": 0,       # 0 => multicore UMAP
            "N_NEIGHBORS": 30,
            "MIN_DIST": 0.0,
            "MIN_CLUSTER_SIZE": None,  # None => auto (N//5)
            "MIN_SAMPLES": 7,
            "SKIP_TSNE": 1,
            "PRIMER_K": 18,
            "TOP_N_GLOBAL": 8,
            "TOP_N_CLUSTER": 5,
            "TOP_N_FORWARD_PRIMERS": 3,
            "MIN_SEQUENCES_FOR_CLUSTERING": 10,
            "PRIMER_TIMEOUT": 120,
            "FETCH_WORKERS": 10,
            "all_te_file": "",       # Path to input CSV on HPC
            "te_counts": "",
            # Scheduler resources (used for both LSF and Slurm)
            "MEM_MB": 12000,
            "CPUS": 4,
            "WALLTIME": "04:00",     # HH:MM
            "QUEUE": "normal",       # LSF queue / Slurm partition
            # Optional module loads (space-separated, e.g. "python/3.11 gcc/12")
            "MODULES": "",
            # Skip flags
            "SKIP_JASPAR": 0,
            "SKIP_PRIMERS": 0,
            "SKIP_GO": 0,
            # Notification
            "NOTIFY_EMAIL": "",
        }

        self.remote_script_path = None
        self.remote_work_dir = None
        self.remote_output_dir = None  # Where results are stored on HPC
        self.current_job_id = None  # Track submitted job
        self._transport = None
        self._password = None
        self._username = None
        self.use_sftp = False

        self._state = self._load_state()
        self.local_output_dir = (
            Path(self._state["last_local_dir"])
            if self._state.get("last_local_dir")
            else None
        )

    def connect(self, hostname: str, username: str, password: str, port: int = 22, work_dir: str = ""):
        """Connect to the HPC cluster via SSH."""
        print(f"\nConnecting to {hostname}...")

        # Store password for keyboard-interactive auth
        self._password = password
        self._username = username
        self._transport = None

        def keyboard_interactive_handler(title, instructions, prompt_list):
            """Handle keyboard-interactive authentication (supports Duo 2FA).

            First prompt is always the password.  If a second prompt arrives
            (Duo passcode / option), send "1" to trigger a Duo Push — the user
            must approve on their phone before the SSH handshake completes.
            """
            responses = []
            for i, (prompt, show_input) in enumerate(prompt_list):
                p = prompt.lower()
                if i == 0 or "password" in p:
                    responses.append(self._password)
                elif "passcode" in p or "option" in p or "duo" in p or "factor" in p:
                    # Send "1" to select Duo Push; change to a TOTP code if preferred
                    print("  [Duo] Sending push request (approve on your phone)…")
                    responses.append("1")
                else:
                    # Unknown additional prompt — send password as a safe fallback
                    responses.append(self._password)
            return responses

        # Use Transport directly for more control
        try:
            print("Establishing transport...")
            self._transport = paramiko.Transport((hostname, port))
            self._transport.banner_timeout = 60
            self._transport.connect()

            # Try keyboard-interactive auth first (most common for university HPCs)
            print("Authenticating...")
            try:
                self._transport.auth_interactive(username, keyboard_interactive_handler)
            except paramiko.ssh_exception.AuthenticationException:
                # Fall back to password auth
                print("Trying password authentication...")
                self._transport.auth_password(username, password)

            if not self._transport.is_authenticated():
                print("Authentication failed")
                return False

            # Create SSHClient wrapper around the transport
            self.ssh = paramiko.SSHClient()
            self.ssh._transport = self._transport

            # Try to open SFTP (may fail on some HPC systems)
            print("Opening SFTP...")
            try:
                self.sftp = paramiko.SFTPClient.from_transport(self._transport)
                self.use_sftp = True
            except Exception as e:
                print(f"SFTP not available ({e}), will use shell commands for file transfer")
                self.sftp = None
                self.use_sftp = False

            self.connected = True
            print(f"Successfully connected to {hostname}")

            self.remote_work_dir = self._select_remote_work_dir(username, work_dir)
            print(f"Remote work directory: {self.remote_work_dir}")

            # Auto-detect scheduler
            self.scheduler = self._detect_scheduler()
            if self.scheduler:
                print(f"Scheduler detected: {self.scheduler.upper()}")
            else:
                print("Scheduler detected: NONE (bsub/sbatch not visible in login shell)")

            return True

        except paramiko.ssh_exception.AuthenticationException as e:
            print(f"Authentication failed: {e}")
            return False
        except Exception as e:
            print(f"Connection failed: {e}")
            if self._transport:
                self._transport.close()
            return False

    def _select_remote_work_dir(self, username: str, requested: str = "") -> str:
        """Pick a writable directory that is likely visible to batch compute nodes."""
        if requested.strip():
            candidate = requested.strip()
            if self._try_remote_work_dir(candidate):
                return candidate
            print(f"Warning: remote work directory is not writable, ignoring override: {candidate}")

        probe = r'''
set +e
for target in "/project/amodzlab/${USER:-__USER__}/gameca" "/project/amodzlab/gameca_${USER:-__USER__}" "/project/${USER:-__USER__}/gameca" "/scratch/${USER:-__USER__}/gameca" "/work/${USER:-__USER__}/gameca"; do
  mkdir -p "$target" >/dev/null 2>&1 || continue
  [ -d "$target" ] && [ -w "$target" ] || continue
  printf '%s\n' "$target"
  exit 0
done
for base in "${GAMECA_WORKDIR:-}" "${SCRATCH:-}" "${WORK:-}" "${PROJECT:-}" "${PWD:-}" "${HOME:-}"; do
  [ -n "$base" ] || continue
  [ "$base" = "/" ] && continue
  target="$base/gameca"
  mkdir -p "$target" >/dev/null 2>&1 || continue
  [ -d "$target" ] && [ -w "$target" ] || continue
  printf '%s\n' "$target"
  exit 0
done
target="/tmp/__USER___gameca"
mkdir -p "$target" >/dev/null 2>&1 && printf '%s\n' "$target" && exit 0
exit 1
'''.replace("__USER__", username)
        out, err, code = self.run_command(probe, timeout=30)
        if code == 0 and out.strip():
            selected = out.strip().splitlines()[-1]
            if selected.startswith("/tmp/"):
                print(
                    f"Warning: using {selected}; login-node /tmp may not be visible to batch jobs. "
                    "Set Remote work dir to a shared scratch/project path if submission files disappear."
                )
            return selected
        fallback = f"/tmp/{username}_gameca"
        print(
            "Warning: could not auto-detect a shared work directory; "
            f"falling back to {fallback}."
        )
        self.run_command(f"mkdir -p {shlex.quote(fallback)}", timeout=30)
        return fallback

    def _try_remote_work_dir(self, candidate: str) -> bool:
        if not candidate or candidate == "/":
            return False
        out, err, code = self.run_command(
            f"mkdir -p {shlex.quote(candidate)} && [ -d {shlex.quote(candidate)} ] && [ -w {shlex.quote(candidate)} ] && echo OK",
            timeout=30,
        )
        if code == 0 and "OK" in out:
            return True
        _log(f"Rejected remote work dir {candidate!r}: exit={code} stdout={out.strip()!r} stderr={err.strip()!r}")
        return False

    def _batch_work_dir_candidates(self) -> list[str]:
        user = self._username or "$USER"
        candidates = [
            f"/project/amodzlab/{user}/gameca",
            f"/project/amodzlab/gameca_{user}",
            f"/project/{user}/gameca",
            f"/scratch/{user}/gameca",
            f"/work/{user}/gameca",
        ]
        for key in ("JASPAR_BED_PATH", "JASPAR_TABIX_PATH", "LOCAL_ASSEMBLY_PATH", "all_te_file", "te_counts"):
            raw = str(self.params.get(key, "") or "").strip()
            if not raw.startswith("/"):
                continue
            p = Path(raw)
            ancestors = list(p.parents)
            for base in ancestors[:5]:
                base_s = str(base)
                if base_s in {"/", "/project", "/scratch", "/work"}:
                    continue
                candidates.extend([
                    f"{base_s}/{user}/gameca",
                    f"{base_s}/gameca_{user}",
                ])
        seen = set()
        unique = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        return unique

    def _ensure_batch_shared_work_dir(self) -> bool:
        """Batch jobs must not rely on login-node /tmp."""
        if self.remote_work_dir and not self.remote_work_dir.startswith("/tmp/"):
            return True

        print()
        print("[HPC DIAG] Current work dir is login-node /tmp; trying shared batch-safe directories.")
        for candidate in self._batch_work_dir_candidates():
            print(f"[HPC DIAG] Trying shared work dir: {candidate}")
            if self._try_remote_work_dir(candidate):
                self.remote_work_dir = candidate
                print(f"[HPC DIAG] Using shared work dir for batch: {self.remote_work_dir}")
                return True

        print("[HPC DIAG] No shared work directory candidate was writable.")
        print("[HPC DIAG] Batch submission from /tmp is unsafe because compute nodes may not see uploaded files.")
        return False

    def disconnect(self):
        """Close SSH connection."""
        if self.sftp:
            self.sftp.close()
        if self._transport:
            self._transport.close()
        self.connected = False
        print("Disconnected from HPC.")

    def _detect_scheduler(self):
        """Auto-detect LSF (bsub) or Slurm (sbatch) on the remote host."""
        for sched, cmd in [("lsf", "command -v bsub"), ("slurm", "command -v sbatch")]:
            out, _, code = self.run_command(cmd, timeout=10)
            if code == 0 and out.strip():
                return sched
        return None

    def _remote_exec_command(self, command: str) -> str:
        """Run commands through a login shell so scheduler PATH/modules match ui.py sessions.
        HOME=/tmp prevents bash -l from failing when the NFS home dir is missing/unmounted."""
        return f"HOME=/tmp bash -lc {shlex.quote(command)}"

    def _job_script_header(self, job_name, job_out, job_err, mem_mb=None, cpus=None,
                            walltime=None, queue=None):
        """Return scheduler-specific directives for the job script."""
        if self.scheduler not in {"lsf", "slurm"}:
            raise RuntimeError("Scheduler was not detected. Ensure bsub/sbatch is available after login or set scheduler explicitly.")
        mem_mb   = mem_mb   or self.params["MEM_MB"]
        cpus     = cpus     or self.params["CPUS"]
        walltime = walltime or self.params["WALLTIME"]
        queue    = queue    or self.params["QUEUE"]

        if self.scheduler == "slurm":
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
        else:  # lsf
            return (
                f"#BSUB -J {job_name}\n"
                f"#BSUB -o {job_out}\n"
                f"#BSUB -e {job_err}\n"
                f"#BSUB -n {cpus}\n"
                f"#BSUB -M {mem_mb}\n"
                f"#BSUB -W {walltime}\n"
                f"#BSUB -q {queue}\n"
            )

    def _module_load_block(self):
        """Return module load commands if MODULES param is set."""
        mods = self.params.get("MODULES", "").strip()
        if not mods:
            return "# No modules configured (set MODULES parameter if needed)"
        lines = "\n".join(f"module load {m}" for m in mods.split())
        return lines

    def _venv_setup_block(self):
        """Return robust virtualenv setup for login/compute node differences."""
        return f'''# --- Virtual environment setup ---
echo "[GAMECA] hpc_client build={_SCRIPT_BUILD}"
echo "[$(date +%H:%M:%S)] Host diagnostics before venv:"
echo "  hostname=$(hostname)"
echo "  pwd=$(pwd)"
echo "  user=$(whoami)"
echo "  shell=$SHELL"
echo "  scheduler={self.scheduler or 'unknown'}"
echo "  SLURM_JOB_ID=${{SLURM_JOB_ID:-}} LSB_JOBID=${{LSB_JOBID:-}}"
echo "  SLURM_CPUS_PER_TASK=${{SLURM_CPUS_PER_TASK:-}} LSB_DJOB_NUMPROC=${{LSB_DJOB_NUMPROC:-}}"
echo "  nproc=$(nproc 2>/dev/null || echo N/A)"
echo "  TMPDIR=${{TMPDIR:-/tmp}}"
echo "  PATH=$PATH"
VENV_DIR="$HOME/te_analysis_venv"
PYTHON_BIN="${{TE_PYTHON:-python3}}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN=python
fi

if [ -x "$VENV_DIR/bin/python" ]; then
    "$VENV_DIR/bin/python" -c "import sys; print(sys.version)" >/dev/null 2>&1 || {{
        echo "[$(date +%H:%M:%S)] Existing venv is not usable on this node; recreating $VENV_DIR ..."
        rm -rf "$VENV_DIR"
    }}
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "[$(date +%H:%M:%S)] Creating virtual environment at $VENV_DIR with $PYTHON_BIN ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
export NUMBA_CACHE_DIR="${{TMPDIR:-/tmp}}/gameca_numba_cache_$USER"
mkdir -p "$NUMBA_CACHE_DIR"
THREADS="{self.params['CPUS']}"
if [ -n "${{SLURM_CPUS_PER_TASK:-}}" ]; then
    THREADS="$SLURM_CPUS_PER_TASK"
elif [ -n "${{LSB_DJOB_NUMPROC:-}}" ]; then
    THREADS="$LSB_DJOB_NUMPROC"
fi
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export NUMBA_NUM_THREADS="$THREADS"
export VECLIB_MAXIMUM_THREADS="$THREADS"
echo "[$(date +%H:%M:%S)] NUMBA_CACHE_DIR: $NUMBA_CACHE_DIR"
echo "[$(date +%H:%M:%S)] Thread env: OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS OPENBLAS=$OPENBLAS_NUM_THREADS NUMBA=$NUMBA_NUM_THREADS"
echo "[$(date +%H:%M:%S)] Python: $(python --version 2>&1)"
echo "[$(date +%H:%M:%S)] Pip: $(python -m pip --version 2>&1)"
REQ_FILE="{self.remote_work_dir}/requirements.txt"
if [ ! -f "$REQ_FILE" ]; then
    echo "FATAL: Requirements file is missing: $REQ_FILE"
    echo "       The remote work directory must be shared between the login node and batch compute nodes."
    exit 1
fi
PIP_START=$SECONDS
echo "[$(date +%H:%M:%S)] Installing dependencies ..."
python -m pip install --quiet --upgrade pip setuptools wheel || exit 1
python -m pip install --quiet --prefer-binary -r "$REQ_FILE" || exit 1
echo "[$(date +%H:%M:%S)] Dependency install took $((SECONDS - PIP_START))s"
python - <<'PYINFO'
import importlib, os, sys
print(f"[PYINFO] executable={{sys.executable}}")
print(f"[PYINFO] version={{sys.version.split()[0]}}")
print(f"[PYINFO] NUMBA_CACHE_DIR={{os.environ.get('NUMBA_CACHE_DIR')}}")
for name in ["numpy", "pandas", "sklearn", "umap", "hdbscan", "numba"]:
    try:
        mod = importlib.import_module(name)
        print(f"[PYINFO] {{name}}={{getattr(mod, '__version__', 'unknown')}}")
    except Exception as exc:
        print(f"[PYINFO] {{name}}=IMPORT_FAILED {{type(exc).__name__}}: {{exc}}")
PYINFO
'''

    def _mafft_setup_block(self):
        """Return MAFFT setup that does not fail on broken conda installations."""
        return '''# Install MAFFT if not available
if ! command -v mafft >/dev/null 2>&1; then
    if command -v conda >/dev/null 2>&1; then
        echo "[$(date +%H:%M:%S)] Installing MAFFT via conda ..."
        conda install -y -c bioconda mafft 2>/dev/null || echo "  (MAFFT install skipped — conda failed)"
    else
        echo "  (MAFFT not found and conda is unavailable; alignment may fail if MAFFT is required)"
    fi
fi
'''

    def _submit_job_cmd(self, job_script):
        """Return the scheduler command to submit a job script."""
        if self.scheduler == "slurm":
            return f"sbatch {job_script}"
        if self.scheduler == "lsf":
            return f"bsub < {job_script}"
        raise RuntimeError("Scheduler was not detected. Ensure bsub/sbatch is available after login or set scheduler explicitly.")

    def _parse_job_id(self, submit_output):
        """Extract job ID from scheduler submission output."""
        if self.scheduler == "slurm":
            m = re.search(r"Submitted batch job (\d+)", submit_output)
            return m.group(1) if m else None
        else:
            m = re.search(r"Job <(\d+)>", submit_output)
            return m.group(1) if m else None

    def _check_running_cmd(self, job_id):
        """Return command to check if job is still running."""
        if self.scheduler == "slurm":
            return f"squeue -j {job_id} --noheader 2>&1"
        return f"bjobs {job_id} 2>&1"

    def _cancel_job_cmd(self, job_id):
        """Return command to cancel a job."""
        if self.scheduler == "slurm":
            return f"scancel {job_id}"
        return f"bkill {job_id}"

    def _interactive_alloc_cmd(self, runner_script, mem_mb, cpus, queue):
        """Return the command to allocate a node and run interactively."""
        if self.scheduler == "slurm":
            mem_gb = max(1, mem_mb // 1000)
            return (
                f"srun --mem={mem_gb}G --cpus-per-task={cpus} "
                f"--partition={queue} --pty bash {runner_script}"
            )
        return f"bsub -M {mem_mb} -n {cpus} -q {queue} -Is bash {runner_script}"

    def run_command(self, command: str, timeout: int = 300, stream_output: bool = False) -> tuple:
        """Execute a command on the remote server.

        Args:
            command: The command to execute
            timeout: Timeout in seconds
            stream_output: If True, print raw output in real-time. If "summary",
                print only stage/error/status lines plus a heartbeat.
        """
        if not self.connected:
            raise RuntimeError("Not connected to HPC")

        summary_stream = stream_output == "summary"
        raw_stream = bool(stream_output) and not summary_stream
        _log(f"Remote command start (timeout={timeout}s, stream={stream_output}): {command[:240]}")
        channel = self._transport.open_session()
        channel.settimeout(timeout)
        channel.exec_command(self._remote_exec_command(command))

        # Read output
        out = b""
        err = b""

        import time
        start_time = time.time()
        last_output_time = time.time()
        line_buffer = ""

        # Lines to always suppress — shell boilerplate, not pipeline progress
        _NOISE = re.compile(
            r"^(echo |host$|hostname|nproc|df -|ls -|cat |#|set -|"
            r"export |source |module |conda |pip |which |command -v|"
            r"Host diagnostics|venv|virtual.?env|activate|deactivate|"
            r"script running|Running script|ulimit|umask|\s*$)",
            re.IGNORECASE,
        )

        def _important_stream_line(line):
            s = line.strip()
            if not s or _NOISE.match(s):
                return False
            # Drop known verbose-only diagnostic dump lines
            if re.match(r"(^Args: \{|dtypes:|head\(2\):|Output dirs created:|"
                        r"sys\.stdin\.isatty|CWD: /)", s):
                return False
            # Pass everything that looks like real pipeline output:
            # any timestamped line, stage headers, step/arrow progress, errors
            return bool(re.match(
                r"(^\[|=== |─{5,}|STAGE |CHECKPOINT|PIPELINE|"
                r"Step [0-9]/|→ |Scanned |Fetching |Extracting |Parsing |"
                r"Scanning |Written |Saved |Running |"
                r"FATAL|ERROR|Error|Exception|Traceback|WARNING|"
                r"Exit code:|Runtime:|Results:|"
                r"sequences|clusters,|noise )",
                s, re.IGNORECASE,
            ))

        def _emit_summary(chunk):
            nonlocal line_buffer, last_output_time
            try:
                text = chunk.decode(errors="replace")
            except Exception:
                return
            line_buffer += text.replace("\r", "\n")
            while "\n" in line_buffer:
                line, line_buffer = line_buffer.split("\n", 1)
                if _important_stream_line(line):
                    print(f"[HPC OUTPUT] {line.strip()}", flush=True)
                    last_output_time = time.time()

        if summary_stream:
            print("[HPC STATUS] RUNNING — remote command has started.", flush=True)

        while True:
            # Check for stdout
            if channel.recv_ready():
                chunk = channel.recv(4096)
                out += chunk
                if raw_stream and chunk:
                    try:
                        print(chunk.decode(), end='', flush=True)
                    except UnicodeDecodeError:
                        pass
                    last_output_time = time.time()
                elif summary_stream and chunk:
                    _emit_summary(chunk)

            # Check for stderr
            if channel.recv_stderr_ready():
                chunk = channel.recv_stderr(4096)
                err += chunk
                if raw_stream and chunk:
                    try:
                        # Print stderr in a different color if possible
                        print(chunk.decode(), end='', flush=True)
                    except UnicodeDecodeError:
                        pass
                    last_output_time = time.time()
                elif summary_stream and chunk:
                    _emit_summary(chunk)

            # Check if command is done
            if channel.exit_status_ready():
                break

            # Small sleep to prevent busy waiting
            time.sleep(0.1)

            # Print a heartbeat if no output for a while (only in stream mode)
            if stream_output and (time.time() - last_output_time) > 30:
                elapsed = int(time.time() - start_time)
                if summary_stream:
                    print(f"[HPC STATUS] RUNNING — {elapsed}s elapsed.", flush=True)
                else:
                    print(f"\n[Still running... {int(time.time() - last_output_time)}s since last output]", flush=True)
                last_output_time = time.time()

        # Get any remaining data
        while channel.recv_ready():
            chunk = channel.recv(4096)
            out += chunk
            if raw_stream and chunk:
                try:
                    print(chunk.decode(), end='', flush=True)
                except UnicodeDecodeError:
                    pass
            elif summary_stream and chunk:
                _emit_summary(chunk)
        while channel.recv_stderr_ready():
            chunk = channel.recv_stderr(4096)
            err += chunk
            if raw_stream and chunk:
                try:
                    print(chunk.decode(), end='', flush=True)
                except UnicodeDecodeError:
                    pass
            elif summary_stream and chunk:
                _emit_summary(chunk)

        exit_code = channel.recv_exit_status()
        if summary_stream and line_buffer.strip() and _important_stream_line(line_buffer):
            print(f"[HPC OUTPUT] {line_buffer.strip()}", flush=True)
        channel.close()
        elapsed = time.time() - start_time
        out_text = out.decode("utf-8", errors="replace")
        err_text = err.decode("utf-8", errors="replace")
        if summary_stream:
            state = "NOT RUNNING — completed successfully" if exit_code == 0 else f"NOT RUNNING — failed with exit {exit_code}"
            print(f"[HPC STATUS] {state} after {elapsed/60:.1f} min.", flush=True)
            if exit_code != 0:
                if out_text.strip():
                    print("[HPC DEBUG] stdout tail:", flush=True)
                    for line in out_text.strip().splitlines()[-40:]:
                        print(f"[HPC DEBUG] STDOUT: {line}", flush=True)
                if err_text.strip():
                    print("[HPC DEBUG] stderr tail:", flush=True)
                    for line in err_text.strip().splitlines()[-80:]:
                        print(f"[HPC DEBUG] STDERR: {line}", flush=True)
        _log(
            f"Remote command done exit={exit_code} elapsed={elapsed:.1f}s "
            f"stdout={len(out):,}B stderr={len(err):,}B"
        )

        return out_text, err_text, exit_code

    def _diagnostic_command(self, label: str, command: str, timeout: int = 30):
        """Run a short diagnostic command and print all useful output."""
        print(f"\n[HPC DIAG] {label}")
        print(f"[HPC DIAG] $ {command}")
        out, err, code = self.run_command(command, timeout=timeout, stream_output=False)
        print(f"[HPC DIAG] exit={code} stdout={len(out)}B stderr={len(err)}B")
        if out.strip():
            print("[HPC DIAG] stdout:")
            for line in out.rstrip().splitlines():
                print(f"[HPC DIAG]   {line}")
        if err.strip():
            print("[HPC DIAG] stderr:")
            for line in err.rstrip().splitlines():
                print(f"[HPC DIAG]   {line}")
        return out, err, code

    def preview_family_count(self) -> int:
        """Preview the number of sequences matching the family name."""
        if not self.params["all_te_file"]:
            print("\nNo all_te_file set.")
            print(
                f"HPC auto-load mode will parse RepeatMasker for "
                f"family '{self.params['FAMILY_NAME']}' for "
                f"{self.params['SPECIES']} ({self.params['ASSEMBLY']}) "
                "when the job runs."
            )
            return -1

        family = self.params["FAMILY_NAME"]
        csv_path = self.params["all_te_file"]

        print(f"\nPreviewing sequences for family '{family}'...")

        # First check if file exists
        cmd = f"test -f '{csv_path}' && echo 'exists'"
        out, err, code = self.run_command(cmd)
        if 'exists' not in out:
            print(f"Error: File not found: {csv_path}")
            return -1

        # Count matching rows using grep (case-insensitive)
        # Use grep -c for count, || true to handle no matches (grep returns 1)
        cmd = f"grep -ci '{family}' '{csv_path}' || echo 0"
        out, err, code = self.run_command(cmd)

        try:
            count = int(out.strip().split('\n')[-1])
        except ValueError:
            print(f"Error parsing count: {out}")
            return -1

        print(f"\n{'='*50}")
        print(f"  FILE: {csv_path}")
        print(f"  FAMILY: {family}")
        print(f"  MATCHING SEQUENCES: {count}")
        print(f"{'='*50}")

        return count

    def set_parameter_interactive(self):
        """Interactive menu to set parameters."""
        str_params = {
            "1": "FAMILY_NAME", "2": "SPECIES", "3": "BASE_OUT_DIR",
            "11": "all_te_file", "12": "te_counts",
            "13": "QUEUE", "14": "WALLTIME", "15": "MODULES",
            "17": "ASSEMBLY",
            "19": "LOCAL_ASSEMBLY_PATH",
            "24": "JASPAR_BED_PATH",
            "25": "JASPAR_TABIX_PATH",
            "27": "NOTIFY_EMAIL",
        }
        int_params = {
            "4": "K", "5": "PRIMER_K", "6": "TOP_N_GLOBAL",
            "7": "TOP_N_CLUSTER", "8": "MIN_SEQUENCES_FOR_CLUSTERING",
            "9": "PRIMER_TIMEOUT", "10": "MEM_MB", "16": "CPUS",
            "18": "FETCH_WORKERS",
            "20": "PCA_DIMS", "21": "N_EPOCHS", "22": "RANDOM_STATE",
            "23": "SKIP_TSNE",
            "28": "SKIP_JASPAR", "29": "SKIP_PRIMERS", "30": "SKIP_GO",
        }
        float_params = {
            "26": "P_THRESHOLD",
        }

        while True:
            print("\n" + "="*65)
            print("CURRENT PARAMETERS")
            print("="*65)
            sched = getattr(self, "scheduler", "auto") or "auto"
            print(f"  Scheduler: {sched.upper()}")
            print()
            for num, key in sorted(str_params.items(), key=lambda x: int(x[0])):
                val = self.params.get(key, "") or "[NOT SET]"
                print(f"  [{num:>2}] {key:<35} = {val}")
                if key == "BASE_OUT_DIR":
                    family = str(self.params.get("FAMILY_NAME", "") or "").strip()
                    base   = str(self.params.get("BASE_OUT_DIR", "") or "").strip()
                    if base and family:
                        full = f"{self.remote_work_dir}/{base}/{family.lower()}"
                    elif base:
                        full = f"{self.remote_work_dir}/{base}/<family>"
                    else:
                        full = f"{self.remote_work_dir}/<base_out_dir>/<family>"
                    print(f"       {'':35}  → {full}")
            print()
            for num, key in sorted(int_params.items(), key=lambda x: int(x[0])):
                val = self.params.get(key, "")
                print(f"  [{num:>2}] {key:<35} = {val}")
            print()
            for num, key in sorted(float_params.items(), key=lambda x: int(x[0])):
                val = self.params.get(key, "")
                print(f"  [{num:>2}] {key:<35} = {val}")
            print()
            print("  Descriptions:")
            print("    [1]  FAMILY_NAME: TE family repName to analyse")
            print("    [2]  SPECIES: human or mouse")
            print("    [4]  K: K-mer size for clustering")
            print("    [20] PCA_DIMS: SVD dimensions before UMAP (40 is faster, 50 default quality)")
            print("    [21] N_EPOCHS: UMAP epochs (120 fast, 200 more thorough)")
            print("    [22] RANDOM_STATE: 0 enables multicore UMAP; nonzero is reproducible but slower")
            print("    [23] SKIP_TSNE: 1 skips t-SNE for speed; UMAP/PCA outputs remain")
            print("    [9]  PRIMER_TIMEOUT: Seconds before random-sampling fallback")
            print("    [10] MEM_MB: Memory request in MB (12000 = 12 GB)")
            print("    [11] all_te_file: Optional input CSV. Leave blank to auto-load by family name.")
            print("    [14] WALLTIME: Job time limit HH:MM")
            print("    [15] MODULES: Space-separated module names to load (e.g. python/3.11)")
            print("    [17] ASSEMBLY: UCSC assembly/build (human: hg38/hg19; mouse: mm10/mm39)")
            print("    [18] FETCH_WORKERS: UCSC fetch workers for automatic loading")
            print("    [19] LOCAL_ASSEMBLY_PATH: Optional local FASTA on the cluster")
            print("    [24] JASPAR_BED_PATH: Optional JASPAR BED/BED.GZ path on the cluster")
            print("    [25] JASPAR_TABIX_PATH: bgzip+tabix-indexed JASPAR .bed.gz on the cluster")
            print("         (.tbi must sit alongside it). Reusable across all TE families on the")
            print("         same build — tabix only reads your loci regions, not the whole file.")
            print("    [26] P_THRESHOLD: Fisher exact test p-value cutoff for motif/GO significance")
            print("         (default 0.05; lower = stricter, e.g. 0.01 or 0.001)")
            print("    [27] NOTIFY_EMAIL: Email address for job-complete notification (requires Gmail App Password)")
            print("    [28] SKIP_JASPAR: 1 = skip JASPAR motif/TFBS analysis")
            print("    [29] SKIP_PRIMERS: 1 = skip primer design")
            print("    [30] SKIP_GO: 1 = skip GO enrichment")
            print()
            print("  [g]  Set up Gmail App Password  (run once to enable email notifications)")
            print("  [p]  Preview family count")
            print("  [r]  Run analysis")
            print("  [q]  Back")
            print("="*65)

            choice = input("\nSelect option: ").strip().lower()

            if choice == "q":
                return False
            elif choice == "g":
                self.setup_email_auth()
            elif choice == "p":
                self.preview_family_count()
            elif choice == "r":
                return True
            elif choice in str_params:
                key = str_params[choice]
                cur = self.params.get(key, "")
                val = input(f"  {key} [{cur}]: ").strip()
                if val:
                    self.params[key] = val
                    if key == "SPECIES":
                        species = val.strip().lower()
                        if species in {"human", "homo sapiens"} and self.params.get("ASSEMBLY") in {"mm10", "mm39", ""}:
                            self.params["ASSEMBLY"] = "hg38"
                        elif species in {"mouse", "mus musculus"} and self.params.get("ASSEMBLY") in {"hg38", "hg19", ""}:
                            self.params["ASSEMBLY"] = "mm10"
            elif choice in int_params:
                key = int_params[choice]
                cur = self.params.get(key, "")
                val = input(f"  {key} [{cur}]: ").strip()
                if val:
                    try:
                        self.params[key] = int(val)
                    except ValueError:
                        print("  Must be an integer.")
            elif choice in float_params:
                key = float_params[choice]
                cur = self.params.get(key, "")
                val = input(f"  {key} [{cur}]: ").strip()
                if val:
                    try:
                        pv = float(val)
                        if not (0 < pv <= 1):
                            raise ValueError("must be between 0 and 1")
                        self.params[key] = pv
                    except ValueError as e:
                        print(f"  Invalid value: {e}")
            else:
                print("  Invalid option")

        return False

    def _using_auto_family_load(self):
        """Return True when the pipeline should build input data from family/build."""
        return not bool(str(self.params.get("all_te_file", "")).strip())

    def _pipeline_cli_args(self):
        """Build shell-safe query.py CLI args for explicit CSV or auto-load mode."""
        args = [
            "--family", self.params["FAMILY_NAME"],
            "--output", self.params["BASE_OUT_DIR"],
            "--kmer", str(self.params["K"]),
            "--pca-dims", str(self.params["PCA_DIMS"]),
            "--n-epochs", str(self.params["N_EPOCHS"]),
            "--random-state", str(self.params["RANDOM_STATE"]),
            "--n-neighbors", str(self.params.get("N_NEIGHBORS", 30)),
            "--min-dist", str(self.params.get("MIN_DIST", 0.0)),
            "--min-samples", str(self.params.get("MIN_SAMPLES", 7)),
            "--primer-kmer", str(self.params["PRIMER_K"]),
            "--top-global", str(self.params["TOP_N_GLOBAL"]),
            "--top-cluster", str(self.params["TOP_N_CLUSTER"]),
            "--min-sequences", str(self.params["MIN_SEQUENCES_FOR_CLUSTERING"]),
            "--primer-timeout", str(self.params["PRIMER_TIMEOUT"]),
        ]

        genome = str(self.params.get("LOCAL_ASSEMBLY_PATH", "")).strip()
        input_csv = str(self.params.get("all_te_file", "")).strip()

        if input_csv:
            args += ["--input", input_csv]
        else:
            args += [
                "--local",
                "--assembly", self.params["ASSEMBLY"],
                "--fetch-workers", str(self.params["FETCH_WORKERS"]),
            ]

        if genome:
            args += ["--genome", genome]
        else:
            args.append("--skip-genome")

        if int(self.params.get("SKIP_TSNE", 1)):
            args.append("--skip-tsne")

        if int(self.params.get("SKIP_JASPAR", 0)):
            args.append("--skip-motif")

        if int(self.params.get("SKIP_PRIMERS", 0)):
            args.append("--skip-primers")

        if int(self.params.get("SKIP_GO", 0)):
            args.append("--skip-go")

        # Prefer the tabix-indexed copy when set; fall back to plain BED path.
        jaspar = (str(self.params.get("JASPAR_TABIX_PATH", "")).strip()
                  or str(self.params.get("JASPAR_BED_PATH", "")).strip())
        if jaspar:
            args += ["--jaspar-bed", jaspar]

        mcs = self.params.get("MIN_CLUSTER_SIZE")
        if mcs is not None:
            args += ["--min-cluster-size", str(mcs)]

        p_thresh = self.params.get("P_THRESHOLD", 0.05)
        args += ["--p-threshold", str(p_thresh)]

        cli = " ".join(shlex.quote(str(arg)) for arg in args)
        _log(f"Pipeline CLI args: {cli}")
        return cli

    def _input_preflight_block(self, error_log=None):
        """Return bash pre-flight checks for the selected data-loading mode."""
        input_csv = str(self.params.get("all_te_file", "")).strip()
        if input_csv:
            quoted = input_csv.replace('"', '\\"')
            if error_log:
                return f'''if [ ! -f "{quoted}" ]; then
    echo "FATAL: Input file not found: {quoted}" | tee -a $ERROR_LOG
    echo "1" > {error_log}
    exit 1
fi
echo "  Input data: OK ({quoted})"'''
            return f'''if [ ! -f "{quoted}" ]; then
    echo "FATAL: Input file not found: {quoted}"
    exit 1
fi
echo "  Input data: OK ({quoted})"'''

        species = self.params["SPECIES"]
        assembly = self.params["ASSEMBLY"]
        family = self.params["FAMILY_NAME"]
        return (
            f'echo "  Input data: AUTO from RepeatMasker '
            f'(family={family}, species={species}, assembly={assembly})"'
        )

    def _upload_text_file(self, local_path: Path, remote_path: str, label: str):
        """Upload a text file through SFTP or chunked base64 shell commands."""
        with open(local_path, 'r') as f:
            content = f.read()

        if self.use_sftp and self.sftp:
            _log(f"Uploading {label} via SFTP ({len(content):,} bytes)")
            try:
                with self.sftp.file(remote_path, 'w') as f:
                    f.write(content)
                _log(f"Uploaded {label} to {remote_path}")
                return True
            except Exception as e:
                _log(f"SFTP upload failed for {label} ({e}); falling back to shell base64")
                self.use_sftp = False
                self.sftp = None

        encoded_full = base64.b64encode(content.encode()).decode()
        chunk_size = 65000
        chunks = [encoded_full[i:i+chunk_size] for i in range(0, len(encoded_full), chunk_size)] or [""]
        _log(f"Uploading {label} via shell base64 ({len(content):,} bytes, {len(chunks)} chunk(s))")

        parent = remote_path.rsplit("/", 1)[0] if "/" in remote_path else "."
        remote_q = shlex.quote(remote_path)
        self.run_command(f"mkdir -p {shlex.quote(parent)}", timeout=30)
        cmd = f"printf %s {shlex.quote(chunks[0])} | base64 -d > {remote_q}"
        out, err, code = self.run_command(cmd, timeout=30)
        if code != 0:
            print(f"Failed to upload {label} (chunk 1): {err}")
            return False

        for i, chunk in enumerate(chunks[1:], 2):
            cmd = f"printf %s {shlex.quote(chunk)} | base64 -d >> {remote_q}"
            out, err, code = self.run_command(cmd, timeout=30)
            if code != 0:
                print(f"Failed to upload {label} (chunk {i}/{len(chunks)}): {err}")
                return False

        _log(f"Uploaded {label} to {remote_path}")
        return True

    def upload_script(self):
        """Upload the analysis script to HPC with current parameters."""
        # Read local query.py
        local_script = Path(__file__).parent / "query.py"

        if not local_script.exists():
            print(f"Error: query.py not found at {local_script}")
            return False

        _log("Uploading pipeline scripts and requirements to cluster")
        # Upload to remote
        remote_script = f"{self.remote_work_dir}/te_analysis_run.py"
        if not self._upload_text_file(local_script, remote_script, "query.py"):
            return False

        self.remote_script_path = remote_script

        # Keep the cluster copy in sync with local modules imported by query.py.
        module_names = [
            "te_prep.py",
            "te_genome.py",
            "te_clustering.py",
            "te_primers.py",
            "te_alignment.py",
            "te_motif.py",
            "te_go.py",
            "te_expression.py",
            "te_enrichment.py",
            "te_fast.pyx",
            "setup_cython.py",
            "ui.py",
        ]
        for name in module_names:
            local_module = Path(__file__).parent / name
            if local_module.exists():
                remote_module = f"{self.remote_work_dir}/{name}"
                if not self._upload_text_file(local_module, remote_module, name):
                    return False
            else:
                _log(f"Optional module not found locally, skipping: {name}")

        # Upload requirements.txt for venv setup
        local_reqs = Path(__file__).parent / "requirements.txt"
        if local_reqs.exists():
            remote_reqs = f"{self.remote_work_dir}/requirements.txt"
            if not self._upload_text_file(local_reqs, remote_reqs, "requirements.txt"):
                print("Warning: Failed to upload requirements.txt")

        _log("Pipeline upload complete")
        return True

    def _log_run_configuration(self, mode: str):
        input_mode = (
            f"input_csv={self.params.get('all_te_file')}"
            if not self._using_auto_family_load()
            else f"auto family load ({self.params['SPECIES']}, {self.params['ASSEMBLY']})"
        )
        genome = self.params.get("LOCAL_ASSEMBLY_PATH") or "(none)"
        _log(
            f"{mode} configuration: family={self.params['FAMILY_NAME']}, "
            f"{input_mode}, local_assembly_path={genome}, "
            f"out={self.params['BASE_OUT_DIR']}, k={self.params['K']}, "
            f"pca_dims={self.params['PCA_DIMS']}, n_epochs={self.params['N_EPOCHS']}, "
            f"random_state={self.params['RANDOM_STATE']}, skip_tsne={self.params['SKIP_TSNE']}, "
            f"primer_k={self.params['PRIMER_K']}, mem={self.params['MEM_MB']}MB, "
            f"cpus={self.params['CPUS']}, walltime={self.params['WALLTIME']}, "
            f"queue={self.params['QUEUE']}"
        )

    def submit_batch_job(self):
        """Submit the analysis as a batch job on HPC. Returns immediately after submission."""
        self._log_run_configuration("Batch")
        if not self._ensure_batch_shared_work_dir():
            print(
                "Error: could not find a shared writable remote work directory for batch submission. "
                "Set Remote work dir to a project/scratch path visible from compute nodes."
            )
            return False
        if self._using_auto_family_load():
            print(
                "\nNo all_te_file set. The job will automatically load loci "
                f"for family '{self.params['FAMILY_NAME']}' from "
                f"RepeatMasker for {self.params['SPECIES']} ({self.params['ASSEMBLY']})."
            )
            confirm_msg = "Proceed with automatic family loading? (y/n): "
        else:
            # Preview count first
            count = self.preview_family_count()
            if count <= 0:
                confirm = input("\nNo sequences found. Continue anyway? (y/n): ").strip().lower()
                if confirm != 'y':
                    return False
            confirm_msg = f"\nProceed with analysis for {count} sequences? (y/n): "

        confirm = input(confirm_msg).strip().lower()
        if confirm != 'y':
            return False

        # Upload script
        if not self.upload_script():
            return False

        # Set up output directory on HPC
        family = self.params["FAMILY_NAME"].lower()
        self.remote_output_dir = f"{self.remote_work_dir}/{self.params['BASE_OUT_DIR']}/{family}"

        # Create bsub job script
        import datetime as _dt
        _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        job_name = f"te_analysis_{self.params['FAMILY_NAME']}"
        job_script    = f"{self.remote_work_dir}/te_analysis_job_{_ts}.sh"
        job_out       = f"{self.remote_work_dir}/te_analysis_job_{_ts}.out"
        job_err       = f"{self.remote_work_dir}/te_analysis_job_{_ts}.err"
        job_done      = f"{self.remote_work_dir}/te_analysis_job_{_ts}.done"
        job_info      = f"{self.remote_work_dir}/te_analysis_job_{_ts}.info"
        job_error_log = f"{self.remote_work_dir}/te_analysis_job_{_ts}.error.log"

        # Build the scheduler-agnostic job script
        sched_header = self._job_script_header(job_name, job_out, job_err)
        module_block = self._module_load_block()
        venv_block = self._venv_setup_block()
        mafft_block = self._mafft_setup_block()
        input_preflight = self._input_preflight_block(error_log=job_done)
        pipeline_args = self._pipeline_cli_args()
        _log(f"Creating batch job script at {job_script}")

        bsub_script = f'''#!/bin/bash
{sched_header}
{module_block}

{venv_block}
{mafft_block}

echo "=========================================================="
echo " TE Analysis Pipeline — Batch Mode"
echo "=========================================================="
echo " Job ID:      $LSB_JOBID"
echo " Host:        $(hostname)"
echo " Date:        $(date)"
echo " CPUs:        $(nproc 2>/dev/null || echo N/A)"
echo " Memory req:  12 GB"
echo " Working dir: {self.remote_work_dir}"
echo " Output dir:  {self.remote_output_dir}"
echo " Family:      {self.params['FAMILY_NAME']}"
echo " Species:     {self.params['SPECIES']}"
echo " Assembly:    {self.params['ASSEMBLY']}"
echo " Input mode:  {'auto RepeatMasker/UCSC' if self._using_auto_family_load() else 'provided CSV'}"
echo " Pipeline:    {pipeline_args}"
echo " K-mer:       {self.params['K']}"
echo " PCA dims:    {self.params['PCA_DIMS']}"
echo " UMAP epochs: {self.params['N_EPOCHS']}"
echo " Rand state:  {self.params['RANDOM_STATE']} (0 means multicore UMAP)"
echo " Skip t-SNE:  {self.params['SKIP_TSNE']}"
echo " Timeout:     {self.params['PRIMER_TIMEOUT']}s"
echo " Python:      $(python --version 2>&1)"
echo "=========================================================="

# Initialize error log
ERROR_LOG="{job_error_log}"
echo "=== TE Analysis Error Log ===" > $ERROR_LOG
echo "Job ID: $LSB_JOBID" >> $ERROR_LOG
echo "Host: $(hostname)" >> $ERROR_LOG
echo "Started: $(date)" >> $ERROR_LOG
echo "" >> $ERROR_LOG

cd {self.remote_work_dir}

# Pre-flight checks
echo ""
echo "[$(date +%H:%M:%S)] Pre-flight checks..."
echo "[$(date +%H:%M:%S)] Job script: {job_script}"
echo "[$(date +%H:%M:%S)] Error log:  $ERROR_LOG"
echo "[$(date +%H:%M:%S)] Disk space:"
df -h . "${{TMPDIR:-/tmp}}" 2>/dev/null || true
echo "[$(date +%H:%M:%S)] Working directory files:"
ls -lh {self.remote_work_dir}/te_analysis_run.py {self.remote_work_dir}/te_clustering.py {self.remote_work_dir}/requirements.txt 2>/dev/null || true

if [ ! -f "{self.remote_script_path}" ]; then
    echo "FATAL: Script not found: {self.remote_script_path}" | tee -a $ERROR_LOG
    echo "1" > {job_done}
    exit 1
fi
echo "  Script:     OK"

{input_preflight}

if [ -n "{self.params['LOCAL_ASSEMBLY_PATH']}" ] && [ -f "{self.params['LOCAL_ASSEMBLY_PATH']}" ]; then
    echo "  Genome:     OK ($(du -sh "{self.params['LOCAL_ASSEMBLY_PATH']}" 2>/dev/null | cut -f1))"
else
    echo "  Genome:     not provided — will use UCSC API fallback and skip genome-wide primer search" | tee -a $ERROR_LOG
fi

echo ""
echo "[$(date +%H:%M:%S)] Starting pipeline..."
echo "[$(date +%H:%M:%S)] Command: python -u {self.remote_script_path} {pipeline_args}"
echo "=========================================================="

SECONDS=0
python -u {self.remote_script_path} {pipeline_args} 2>&1 | tee -a $ERROR_LOG
EXIT_CODE=${{PIPESTATUS[0]}}
PIPELINE_SECONDS=$SECONDS

echo ""
echo "=========================================================="
echo " Pipeline finished"
echo " Exit code:   $EXIT_CODE"
echo " Runtime:     $((PIPELINE_SECONDS / 60))m $((PIPELINE_SECONDS % 60))s"
echo " Date:        $(date)"

if [ $EXIT_CODE -ne 0 ]; then
    echo "" >> $ERROR_LOG
    echo "=== JOB FAILED (exit code: $EXIT_CODE) ===" >> $ERROR_LOG
    echo "Ended: $(date)" >> $ERROR_LOG
    echo ""
    echo " *** JOB FAILED ***"
    echo " Error log: $ERROR_LOG"
    echo " Stderr:    {job_err}"
else
    RESULT_SIZE=$(du -sh "{self.remote_output_dir}" 2>/dev/null | cut -f1)
    echo " Results:     $RESULT_SIZE at {self.remote_output_dir}"
    echo "" >> $ERROR_LOG
    echo "=== JOB SUCCEEDED ===" >> $ERROR_LOG
    echo "Ended: $(date)" >> $ERROR_LOG
fi

echo "=========================================================="

# Create done marker file
echo $EXIT_CODE > {job_done}

exit $EXIT_CODE
'''

        # Upload the job script
        print("\nCreating bsub job script...")
        create_script_cmd = f"cat > {job_script} << 'BSUB_SCRIPT_EOF'\n{bsub_script}\nBSUB_SCRIPT_EOF"
        out, err, code = self.run_command(create_script_cmd, timeout=30)
        if code != 0:
            print(f"Error creating job script: {err}")
            return False

        # Make executable
        self.run_command(f"chmod +x {job_script}", timeout=10)

        # Remove old output files if they exist
        self.run_command(f"rm -f {job_out} {job_err} {job_done} {job_info}", timeout=10)

        # Submit the job
        submit_cmd = self._submit_job_cmd(job_script)
        sched_label = (getattr(self, "scheduler", None) or "unknown").upper()
        print(f"\nSubmitting job via {sched_label}...")
        _log(f"Submitting scheduler command: {submit_cmd}")
        out, err, code = self.run_command(submit_cmd, timeout=30)
        print("\n[HPC DIAG] Submit command completed")
        print(f"[HPC DIAG] command: {submit_cmd}")
        print(f"[HPC DIAG] exit={code} stdout={len(out)}B stderr={len(err)}B")
        if out.strip():
            print("[HPC DIAG] submit stdout:")
            for line in out.rstrip().splitlines():
                print(f"[HPC DIAG]   {line}")
        if err.strip():
            print("[HPC DIAG] submit stderr:")
            for line in err.rstrip().splitlines():
                print(f"[HPC DIAG]   {line}")

        if code != 0:
            print(f"Error submitting job: {err}")
            self._diagnostic_command("Scheduler availability", "command -v bsub; command -v sbatch; echo PATH=$PATH", timeout=15)
            self._diagnostic_command("Job script listing", f"ls -l {shlex.quote(job_script)} {shlex.quote(self.remote_work_dir)}/requirements.txt 2>&1", timeout=15)
            return False

        # Parse job ID
        job_id = self._parse_job_id(out)
        if job_id:
            self.current_job_id = job_id
            print(f"Job submitted successfully! Job ID: {job_id}")
            if self.scheduler == "lsf":
                self._diagnostic_command("LSF job immediately after submit", f"bjobs -l {shlex.quote(job_id)} 2>&1", timeout=20)
                self._diagnostic_command("LSF queue summary", f"bjobs {shlex.quote(job_id)} 2>&1", timeout=20)
            elif self.scheduler == "slurm":
                self._diagnostic_command("Slurm job immediately after submit", f"squeue -j {shlex.quote(job_id)} -o '%i %T %R' 2>&1", timeout=20)
        else:
            print(f"Job submitted but could not parse job ID from: {out}")
            job_id = "unknown"
            self.current_job_id = None

        # Save job info for later retrieval (paths embedded so finder can reconstruct them)
        job_info_content = (
            f"JOB_ID={job_id}\n"
            f"FAMILY={self.params['FAMILY_NAME']}\n"
            f"OUTPUT_DIR={self.remote_output_dir}\n"
            f"LOG_OUT={job_out}\n"
            f"LOG_ERR={job_err}\n"
            f"LOG_DONE={job_done}\n"
            f"LOG_ERRLOG={job_error_log}\n"
            f"SUBMITTED=$(date)"
        )
        self.run_command(f"echo '{job_info_content}' > {job_info}", timeout=10)
        self._diagnostic_command(
            "Submitted file inventory",
            f"ls -lh {shlex.quote(self.remote_work_dir)} | sed -n '1,120p'; "
            f"echo '--- job info ---'; cat {shlex.quote(job_info)} 2>&1; "
            f"echo '--- job script head ---'; sed -n '1,220p' {shlex.quote(job_script)} 2>&1",
            timeout=30,
        )

        print("\n" + "=" * 60)
        print("BATCH JOB SUBMITTED SUCCESSFULLY")
        print("=" * 60)
        print(f"\nJob ID: {job_id}")
        print(f"Job output: {job_out}")
        print(f"Job errors: {job_err}")
        print(f"Results will be at: {self.remote_output_dir}")
        print("\nThe job is now running on the HPC cluster.")
        print("Use 'Check job status' to monitor progress.")
        print("Use 'Retrieve results' to download when complete.")
        sched = getattr(self, "scheduler", "lsf")
        print("\nUseful HPC commands:")
        if sched == "slurm":
            print(f"  squeue -j {job_id}     # Check job status")
            print(f"  tail -f {job_out}       # View live output")
            print(f"  scancel {job_id}        # Cancel job")
        else:
            print(f"  bjobs {job_id}          # Check job status")
            print(f"  bpeek {job_id}          # View live output")
            print(f"  bkill {job_id}          # Cancel job")
        print("=" * 60)

        return True

    def run_interactive_job(self):
        """Run analysis interactively on a compute node via bsub -Is.

        Uses 'bsub -M 12000 -n 4 -Is bash' to allocate a compute node,
        then runs the pipeline with real-time output streaming.
        This keeps the SSH connection active and streams all output back.
        """
        self._log_run_configuration("Interactive")
        if self._using_auto_family_load():
            print(
                "\nNo all_te_file set. The interactive job will automatically "
                f"load loci for family '{self.params['FAMILY_NAME']}' from "
                f"RepeatMasker for {self.params['SPECIES']} ({self.params['ASSEMBLY']})."
            )
            confirm_msg = "Proceed with automatic family loading? (y/n): "
        else:
            # Preview count
            count = self.preview_family_count()
            if count <= 0:
                confirm = input("\nNo sequences found. Continue anyway? (y/n): ").strip().lower()
                if confirm != 'y':
                    return False
            confirm_msg = f"\nProceed with interactive analysis for {count} sequences? (y/n): "

        confirm = input(confirm_msg).strip().lower()
        if confirm != 'y':
            return False

        # Upload script
        if not self.upload_script():
            return False

        # Set up output directory
        family = self.params["FAMILY_NAME"].lower()
        self.remote_output_dir = f"{self.remote_work_dir}/{self.params['BASE_OUT_DIR']}/{family}"

        # Build runner script with comprehensive logging
        runner_script = f"{self.remote_work_dir}/te_analysis_runner.sh"
        module_block = self._module_load_block()
        venv_block = self._venv_setup_block()
        mafft_block = self._mafft_setup_block()
        input_preflight = self._input_preflight_block()
        pipeline_args = self._pipeline_cli_args()
        _log(f"Creating interactive runner script at {runner_script}")
        runner_content = f'''#!/bin/bash
set -e

{module_block}

{venv_block}
{mafft_block}

echo "=========================================================="
echo " TE Analysis Pipeline — Interactive Mode"
echo "=========================================================="
echo " Host:        $(hostname)"
echo " Date:        $(date)"
echo " CPUs:        $(nproc 2>/dev/null || echo N/A)"
echo " Memory req:  12 GB"
echo " Working dir: {self.remote_work_dir}"
echo " Output dir:  {self.remote_output_dir}"
echo " Family:      {self.params['FAMILY_NAME']}"
echo " Species:     {self.params['SPECIES']}"
echo " Assembly:    {self.params['ASSEMBLY']}"
echo " Input mode:  {'auto RepeatMasker/UCSC' if self._using_auto_family_load() else 'provided CSV'}"
echo " Pipeline:    {pipeline_args}"
echo " Primer K:    {self.params['PRIMER_K']}"
echo " K-mer:       {self.params['K']}"
echo " PCA dims:    {self.params['PCA_DIMS']}"
echo " UMAP epochs: {self.params['N_EPOCHS']}"
echo " Rand state:  {self.params['RANDOM_STATE']} (0 means multicore UMAP)"
echo " Skip t-SNE:  {self.params['SKIP_TSNE']}"
echo " Timeout:     {self.params['PRIMER_TIMEOUT']}s"
echo "=========================================================="
echo ""

cd {self.remote_work_dir}

# Verify files exist
echo "[$(date +%H:%M:%S)] Checking prerequisites..."
echo "[$(date +%H:%M:%S)] Runner: {runner_script}"
echo "[$(date +%H:%M:%S)] Disk space:"
df -h . "${{TMPDIR:-/tmp}}" 2>/dev/null || true
echo "[$(date +%H:%M:%S)] Working directory files:"
ls -lh {self.remote_work_dir}/te_analysis_run.py {self.remote_work_dir}/te_clustering.py {self.remote_work_dir}/requirements.txt 2>/dev/null || true
if [ ! -f "{self.remote_script_path}" ]; then
    echo "FATAL: Script not found: {self.remote_script_path}"
    exit 1
fi
echo "  Script:     OK ({self.remote_script_path})"

{input_preflight}

if [ -n "{self.params['LOCAL_ASSEMBLY_PATH']}" ] && [ -f "{self.params['LOCAL_ASSEMBLY_PATH']}" ]; then
    echo "  Genome:     OK ({self.params['LOCAL_ASSEMBLY_PATH']})"
    GENOME_SIZE=$(du -sh "{self.params['LOCAL_ASSEMBLY_PATH']}" 2>/dev/null | cut -f1)
    echo "              Size: $GENOME_SIZE"
else
    echo "  Genome:     not provided — will use UCSC API fallback and skip genome-wide primer search"
fi

echo ""
echo "[$(date +%H:%M:%S)] Python version: $(python --version 2>&1)"
echo "[$(date +%H:%M:%S)] Starting pipeline..."
echo "[$(date +%H:%M:%S)] Command: python -u {self.remote_script_path} {pipeline_args}"
echo "=========================================================="
echo ""

SECONDS=0
python -u {self.remote_script_path} {pipeline_args}
EXIT_CODE=$?
PIPELINE_SECONDS=$SECONDS

echo ""
echo "=========================================================="
echo " Pipeline finished"
echo " Exit code:   $EXIT_CODE"
echo " Runtime:     $((PIPELINE_SECONDS / 60))m $((PIPELINE_SECONDS % 60))s"
echo " Date:        $(date)"
if [ -d "{self.remote_output_dir}" ]; then
    RESULT_SIZE=$(du -sh "{self.remote_output_dir}" 2>/dev/null | cut -f1)
    echo " Results:     $RESULT_SIZE at {self.remote_output_dir}"
fi
echo "=========================================================="

exit $EXIT_CODE
'''

        # Upload runner script
        print("\nCreating runner script...")
        create_cmd = f"cat > {runner_script} << 'RUNNER_EOF'\n{runner_content}\nRUNNER_EOF"
        out, err, code = self.run_command(create_cmd, timeout=30)
        if code != 0:
            print(f"Error creating runner script: {err}")
            return False
        self.run_command(f"chmod +x {runner_script}", timeout=10)

        # Build scheduler interactive command
        mem_mb = self.params["MEM_MB"]
        cpus   = self.params["CPUS"]
        queue  = self.params["QUEUE"]
        bsub_cmd = self._interactive_alloc_cmd(runner_script, mem_mb, cpus, queue)
        sched_label = getattr(self, "scheduler", "lsf").upper()
        _log(f"Interactive scheduler command: {bsub_cmd}")

        print("\n" + "=" * 60)
        print(f"SUBMITTING INTERACTIVE JOB  ({sched_label})")
        print("=" * 60)
        print(f"  Command: {bsub_cmd}")
        print(f"  Memory:  12 GB")
        print(f"  Cores:   4")
        print(f"  Mode:    Interactive (-Is) — output streams in real-time")
        print("")
        print("Waiting for compute node allocation...")
        print("(This may take a few minutes depending on cluster load)")
        print("Press Ctrl+C to cancel.")
        print("=" * 60 + "\n")

        # Run with streaming output — long timeout for the full pipeline
        import time
        start_time = time.time()
        try:
            out, err, code = self.run_command(bsub_cmd, timeout=14400, stream_output="summary")
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Job may still be running on the cluster.")
            print("Use 'bjobs' on the HPC to check.")
            return False

        elapsed = time.time() - start_time

        print(f"\n{'=' * 60}")
        if code == 0:
            print(f"JOB COMPLETED SUCCESSFULLY ({elapsed/60:.1f} min)")
            print(f"Results at: {self.remote_output_dir}")
        else:
            print(f"JOB FAILED (exit code: {code}, elapsed: {elapsed/60:.1f} min)")
            if err:
                print(f"\nStderr (last 500 chars):\n{err[-500:]}")
        print("=" * 60)

        return code == 0

    def _find_latest_job_info(self):
        """Return info for the most recently submitted job (pipeline or motif-only).

        Globs for te_analysis_job_*.info and te_motif_job_*.info (timestamped),
        picks the one with the newest mtime, and reads log paths from its content.

        Returns
        -------
        (info_dict, out_path, err_path, done_path, errlog_path)
        All paths are None when no info file is found.
        """
        # Find the newest .info file across both job types
        glob_cmd = (
            f"ls -t {self.remote_work_dir}/te_analysis_job_*.info "
            f"{self.remote_work_dir}/te_motif_job_*.info 2>/dev/null | head -1"
        )
        latest_path, _, _ = self.run_command(glob_cmd, timeout=10)
        latest_path = latest_path.strip()
        if not latest_path:
            return None, None, None, None, None

        content, _, ccode = self.run_command(f"cat {latest_path} 2>/dev/null", timeout=10)
        if ccode != 0 or not content.strip():
            return None, None, None, None, None

        info = {}
        for line in content.strip().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                info[k.strip()] = v.strip()

        out_path    = info.get("LOG_OUT")
        err_path    = info.get("LOG_ERR")
        done_path   = info.get("LOG_DONE")
        errlog_path = info.get("LOG_ERRLOG")

        return info, out_path, err_path, done_path, errlog_path

    def _get_scheduler_state(self, job_id):
        """Return a normalised job state string by querying the scheduler.

        Returns one of: 'RUNNING', 'PENDING', 'DONE', 'FAILED', 'UNKNOWN'

        Never interprets the .done marker file — that's the caller's job.
        """
        if not job_id or job_id == "unknown":
            return "UNKNOWN"

        if self.scheduler == "slurm":
            out, err, code = self.run_command(
                f"squeue -j {job_id} --format=%T --noheader 2>&1 | head -1",
                timeout=15)
            print(f"[HPC DIAG] scheduler state probe: scheduler=slurm job={job_id} exit={code} out={out.strip()!r} err={err.strip()!r}")
            text = out.strip().upper()
            if "RUNNING" in text:
                return "RUNNING"
            if "PENDING" in text or "CF" in text:
                return "PENDING"
            if text and "INVALID" not in text and "error" not in text.lower():
                # Some other state (COMPLETING, SUSPENDED, etc.)
                return "RUNNING"
            # Job gone from squeue — check sacct for final state
            acct, acct_err, acct_code = self.run_command(
                f"sacct -j {job_id} --format=State --noheader 2>/dev/null | head -1",
                timeout=15)
            print(f"[HPC DIAG] sacct probe: job={job_id} exit={acct_code} out={acct.strip()!r} err={acct_err.strip()!r}")
            state = acct.strip().upper()
            if "COMPLETED" in state:
                return "DONE"
            if any(k in state for k in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL")):
                return "FAILED"
            return "DONE"  # gone from squeue, assume finished

        else:  # LSF
            out, err, code = self.run_command(f"bjobs {job_id} 2>&1", timeout=15)
            print(f"[HPC DIAG] scheduler state probe: scheduler=lsf job={job_id} exit={code} out={out.strip()!r} err={err.strip()!r}")
            low = out.lower()
            if "not found" in low or "is not found" in low:
                return "DONE"   # gone from LSF queue
            # Parse STAT column: header is JOBID USER STAT QUEUE ...
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 3 and parts[0] == str(job_id):
                    stat = parts[2].upper()
                    if stat == "RUN":
                        return "RUNNING"
                    if stat in ("PEND", "WAIT", "PSUSP", "SSUSP", "USUSP"):
                        return "PENDING"
                    if stat == "DONE":
                        return "DONE"
                    if stat == "EXIT":
                        return "FAILED"
                    return "RUNNING"  # unknown LSF state — still in queue
            return "UNKNOWN"

    def check_job_status(self):
        """Check the status of the most recently submitted batch job."""
        import time as _time
        _status_t0 = _time.time()
        job_info, job_out, job_err, job_done, job_errlog = self._find_latest_job_info()

        if job_info is None:
            print("\nNo job information found. Submit a batch job first.")
            return None

        job_id   = job_info.get("JOB_ID",   "unknown")
        out_dir  = job_info.get("OUTPUT_DIR", "")
        family   = job_info.get("FAMILY",    "")
        job_type = job_info.get("TYPE",      "pipeline")
        submitted = job_info.get("SUBMITTED", "unknown")

        print()
        print("=" * 60)
        print("  JOB STATUS")
        print("=" * 60)
        print(f"  Job ID    : {job_id}")
        print(f"  Type      : {job_type}" + (f"  (family: {family})" if family else ""))
        print(f"  Submitted : {submitted}")
        print(f"  Out dir   : {out_dir or '(unknown)'}")
        print(f"  Job log   : {job_out}")

        # ── 1. Ask scheduler first — never trust .done alone ─────────────────
        sched_state = self._get_scheduler_state(job_id)
        done_out, _, _ = self.run_command(f"cat {job_done} 2>/dev/null", timeout=10)
        done_val = done_out.strip()

        print()
        if sched_state == "RUNNING":
            print("  State     : RUNNING  ▶  job is active on the cluster")
        elif sched_state == "PENDING":
            print("  State     : PENDING  ⏳  waiting for resources")
        elif sched_state in ("DONE", "FAILED"):
            # Job left the queue; interpret exit code from .done marker
            if done_val == "0":
                print("  State     : COMPLETED SUCCESSFULLY  ✓")
            elif done_val:
                print(f"  State     : FAILED  ✗  (exit code: {done_val})")
            elif sched_state == "DONE":
                print("  State     : COMPLETED  (no exit-code marker written)")
            else:
                print("  State     : FAILED  (scheduler reports failure; no marker)")
        else:  # UNKNOWN — scheduler gave no clear answer; fall back to .done
            if done_val == "0":
                print("  State     : COMPLETED SUCCESSFULLY  ✓  (scheduler silent)")
            elif done_val:
                print(f"  State     : FAILED  ✗  (exit code: {done_val}, scheduler silent)")
            else:
                print("  State     : UNKNOWN  (scheduler silent, no done marker)")

        # ── 2. Stage progress from checkpoint files ───────────────────────────
        print()
        print("  --- Stage progress ---")
        if out_dir:
            # One command: list all CHECKPOINT files with timestamps
            ckpt_out, _, _ = self.run_command(
                f"find {out_dir} -maxdepth 1 -name 'CHECKPOINT_*.txt'"
                f" -printf '%TH:%TM  %f\\n' 2>/dev/null | sort -k2",
                timeout=10)
            if ckpt_out.strip():
                for cline in ckpt_out.strip().splitlines():
                    # CHECKPOINT_STAGE5_CLUSTERING.txt → "Stage5 Clustering"
                    ts, fname = cline.strip().split(None, 1)
                    label = fname.replace("CHECKPOINT_", "").replace(".txt", "").replace("_", " ").title()
                    print(f"    ✓  {ts}  {label}")
            else:
                print("    (no stage checkpoints yet — pipeline may still be initialising)")

            # What stage is currently running: last === STAGE === header in the log
            cur_stage, _, _ = self.run_command(
                f"grep -E '^=== |^\\[.+\\] STAGE [0-9]' {job_out} 2>/dev/null | tail -1",
                timeout=10)
            if cur_stage.strip():
                print(f"    ▶  Currently: {cur_stage.strip()}")

            # Result files with sizes — one find+stat call, no per-file round trips
            results_out, _, _ = self.run_command(
                f"find {out_dir} -maxdepth 4 -type f "
                r"\( -name '*.csv' -o -name '*.tsv' -o -name '*.html' -o -name '*.png' \) "
                f"-printf '%s %P\\n' 2>/dev/null | sort -k2 | head -20",
                timeout=12)
            if results_out.strip():
                print()
                print("  --- Output files ---")
                for rline in results_out.strip().splitlines():
                    parts = rline.split(None, 1)
                    if len(parts) == 2:
                        sz_bytes, relpath = int(parts[0]), parts[1]
                        sz = (f"{sz_bytes/1e6:.1f}MB" if sz_bytes > 1e6
                              else f"{sz_bytes//1024}KB" if sz_bytes > 1024
                              else f"{sz_bytes}B")
                        print(f"    {sz:>8}  {relpath}")
        else:
            print("    (output directory unknown)")

        # ── 3. What is actually happening right now ───────────────────────────
        # Strategy: skip the shell preamble (everything before Python starts),
        # strip only the handful of extremely verbose diagnostic-only lines,
        # then show the last 35 lines of real pipeline output.
        print()
        print("  --- Live pipeline output (last 35 lines) ---")
        activity_cmd = (
            # Find the line where Python was launched and keep only from there on
            f"awk '/Starting pipeline\\.\\.\\.|python -u /{{found=1}} found' {job_out} 2>/dev/null"
            # Strip blank lines and a small set of known verbose-only diag lines
            f" | grep -vE '(^\\s*$"
            f"|^Args: \\{{"
            f"|dtypes:"
            f"|head\\(2\\):"
            f"|Output dirs created:"
            f"|sys\\.stdin\\.isatty"
            f"|CWD: /"
            f"|^={60}"
            f")'"
            f" | tail -35"
        )
        activity_out, _, _ = self.run_command(activity_cmd, timeout=15)
        if activity_out.strip():
            for line in activity_out.strip().splitlines():
                print(f"  {line}")
        else:
            # Nothing after the Python start marker — job is still in preamble or hasn't started
            preamble, _, _ = self.run_command(
                f"tail -10 {job_out} 2>/dev/null || echo '(no output yet)'", timeout=10)
            for line in (preamble or "(no output yet)").rstrip().splitlines():
                print(f"  {line}")

        # ── 4. Error details only when job actually failed ───────────────────
        job_failed = done_val not in ("", "0")
        if job_failed:
            print()
            print("  --- Errors ---")
            # Show last traceback block from the output log
            tb_out, _, _ = self.run_command(
                f"grep -n 'Traceback\\|FATAL\\|Error:\\|Exception:\\|ERROR in stage' "
                f"{job_out} 2>/dev/null | tail -20",
                timeout=12)
            if tb_out.strip():
                for line in tb_out.strip().splitlines():
                    print(f"  {line}")
            # Show tail of the error log
            err_tail, _, _ = self.run_command(
                f"tail -30 {job_errlog} 2>/dev/null || echo '(empty)'", timeout=15)
            for line in (err_tail or "(empty)").rstrip().splitlines():
                print(f"  {line}")

        print()
        print(f"  Full log  : {job_out}")
        if job_failed:
            print(f"  Error log : {job_errlog}")
        elapsed = _time.time() - _status_t0
        print(f"  Checked in {elapsed:.1f}s")
        print("=" * 60)
        return job_info

    def peek_job_output(self, job_id_override: str = ""):
        """Print scheduler diagnostics and tails of known job log files."""
        job_info, job_out, job_err, job_done, job_errlog = self._find_latest_job_info()
        if job_info is None:
            print("\nNo job information found. Submit a batch job first.")
            if job_id_override:
                if self.scheduler == "lsf":
                    self._diagnostic_command("LSF job lookup without info file", f"bjobs -l {shlex.quote(job_id_override)} 2>&1", timeout=20)
                elif self.scheduler == "slurm":
                    self._diagnostic_command("Slurm job lookup without info file", f"squeue -j {shlex.quote(job_id_override)} -o '%i %T %R' 2>&1", timeout=20)
            return False

        job_id = job_id_override.strip() or job_info.get("JOB_ID", "unknown")
        print()
        print("=" * 60)
        print(f"  JOB DEBUG / LOG PEEK  {job_id}")
        print("=" * 60)
        print(f"  Work dir  : {self.remote_work_dir}")
        print(f"  Output dir: {job_info.get('OUTPUT_DIR', '(unknown)')}")
        print(f"  Log out   : {job_out}")
        print(f"  Log err   : {job_err}")
        print(f"  Done file : {job_done}")
        print(f"  Error log : {job_errlog}")

        self._diagnostic_command("Scheduler commands visible", "command -v bsub; command -v bjobs; command -v bpeek; command -v sbatch; command -v squeue; echo PATH=$PATH", timeout=20)
        if self.scheduler == "lsf":
            self._diagnostic_command("bjobs summary", f"bjobs {shlex.quote(job_id)} 2>&1", timeout=20)
            self._diagnostic_command("bjobs long", f"bjobs -l {shlex.quote(job_id)} 2>&1", timeout=30)
            self._diagnostic_command("bhist", f"bhist -l {shlex.quote(job_id)} 2>&1 | tail -120", timeout=30)
        elif self.scheduler == "slurm":
            self._diagnostic_command("squeue summary", f"squeue -j {shlex.quote(job_id)} -o '%i %T %R' 2>&1", timeout=20)
            self._diagnostic_command("sacct", f"sacct -j {shlex.quote(job_id)} --format=JobID,State,ExitCode,Elapsed,NodeList,Reason 2>&1", timeout=30)

        for label, path in [
            ("job stdout", job_out),
            ("job stderr", job_err),
            ("pipeline error log", job_errlog),
            ("done marker", job_done),
        ]:
            if path:
                self._diagnostic_command(label, f"echo FILE={shlex.quote(path)}; ls -lh {shlex.quote(path)} 2>&1; tail -200 {shlex.quote(path)} 2>&1", timeout=30)

        print("=" * 60)
        return True

    def download_error_logs(self, local_dir: str = None):
        """Download just the error logs from the HPC for debugging."""
        job_info, job_out_file, job_err_file, job_done_file, job_error_log = \
            self._find_latest_job_info()

        output_dir = (job_info or {}).get("OUTPUT_DIR")

        if not local_dir:
            local_dir = "./hpc_error_logs"

        local_path = Path(local_dir).expanduser()
        local_path.mkdir(parents=True, exist_ok=True)

        print(f"\nDownloading error logs to {local_path}")
        print("-" * 40)

        # Download various log files
        log_files = [
            (job_out_file, "job_output.log"),
            (job_err_file, "job_stderr.log"),
            (job_error_log, "job_error.log"),
        ]

        if output_dir:
            log_files.append((f"{output_dir}/pipeline_errors.log",     "pipeline_errors.log"))
            log_files.append((f"{output_dir}/pipeline_diagnostic.log", "pipeline_diagnostic.log"))
            log_files.append((f"{output_dir}/01_data/ucsc_fetch_errors.log", "ucsc_fetch_errors.log"))

        for remote_file, local_name in log_files:
            out, err, code = self.run_command(f"cat '{remote_file}' 2>/dev/null", timeout=30)
            if out.strip():
                local_file = local_path / local_name
                with open(local_file, 'w') as f:
                    f.write(out)
                print(f"  Downloaded: {local_name} ({len(out)} bytes)")
            else:
                print(f"  Skipped (empty/not found): {local_name}")

        print("-" * 40)
        print(f"Error logs saved to: {local_path}")
        return local_path

    def watch_job(self):
        """Watch the most recently submitted job live (line-based polling)."""
        import time as _time

        job_info, job_out, job_err, job_done, job_errlog = self._find_latest_job_info()

        if job_info is None:
            print("\nNo job information found. Submit a batch job first.")
            return False

        job_id   = job_info.get("JOB_ID",   "unknown")
        job_type = job_info.get("TYPE",      "pipeline")

        print()
        print("=" * 60)
        print(f"  WATCHING JOB {job_id}  [{job_type}]  (polling every 15s)")
        print(f"  Log file: {job_out}")
        print("  Press Ctrl+C to stop — job will keep running on cluster")
        print("=" * 60)

        # ── Show last 25 lines already written ───────────────────────────────
        existing, _, _ = self.run_command(
            f"tail -25 {job_out} 2>/dev/null", timeout=15)
        if existing.strip():
            print("\n  [-- last 25 lines already written --]")
            for line in existing.rstrip().splitlines():
                print(f"  {line}")
            print("  [-- live output below --]\n")
        else:
            print("\n  (output file empty — waiting for job to start...)\n")

        # Seed line counter from current file length
        lc_out, _, _ = self.run_command(
            f"wc -l < {job_out} 2>/dev/null || echo 0", timeout=10)
        try:
            last_line = int(lc_out.strip())
        except ValueError:
            last_line = 0

        poll_interval  = 15
        stale_polls    = 0   # consecutive polls with no new output
        no_output_warn = 6   # warn after ~90s of silence

        try:
            while True:
                _time.sleep(poll_interval)

                # ── Query scheduler ───────────────────────────────────────────
                sched_state = self._get_scheduler_state(job_id)

                # ── Fetch new lines ───────────────────────────────────────────
                new_lc_out, _, _ = self.run_command(
                    f"wc -l < {job_out} 2>/dev/null || echo {last_line}", timeout=10)
                try:
                    current_line = int(new_lc_out.strip())
                except ValueError:
                    current_line = last_line

                if current_line > last_line:
                    stale_polls = 0
                    new_lines, _, _ = self.run_command(
                        f"sed -n '{last_line + 1},{current_line}p' {job_out} 2>/dev/null",
                        timeout=30)
                    if new_lines.strip():
                        for line in new_lines.rstrip().splitlines():
                            print(f"  {line}")
                    last_line = current_line
                else:
                    stale_polls += 1
                    if stale_polls >= no_output_warn:
                        sched_label = {
                            "RUNNING": "still RUNNING",
                            "PENDING": "PENDING (queued)",
                        }.get(sched_state, sched_state)
                        print(f"  [no new output for {stale_polls * poll_interval}s — scheduler: {sched_label}]",
                              flush=True)
                        stale_polls = 0

                # ── Check if job has left the queue ───────────────────────────
                if sched_state in ("DONE", "FAILED"):
                    # Drain any final lines
                    final_lc, _, _ = self.run_command(
                        f"wc -l < {job_out} 2>/dev/null || echo {last_line}", timeout=10)
                    try:
                        final_line = int(final_lc.strip())
                    except ValueError:
                        final_line = last_line
                    if final_line > last_line:
                        tail_final, _, _ = self.run_command(
                            f"sed -n '{last_line + 1},{final_line}p' {job_out} 2>/dev/null",
                            timeout=30)
                        if tail_final.strip():
                            for line in tail_final.rstrip().splitlines():
                                print(f"  {line}")

                    # Read .done marker written by the job script
                    done_out, _, _ = self.run_command(
                        f"cat {job_done} 2>/dev/null", timeout=10)
                    done_val = done_out.strip()

                    print()
                    print("=" * 60)
                    if done_val == "0":
                        print("  Job COMPLETED SUCCESSFULLY  ✓")
                    elif done_val:
                        print(f"  Job FAILED  ✗  (exit code: {done_val})")
                        err_tail, _, _ = self.run_command(
                            f"tail -30 {job_errlog} 2>/dev/null", timeout=15)
                        if err_tail.strip():
                            print("\n  --- Last 30 lines of error log ---")
                            for line in err_tail.strip().splitlines():
                                print(f"  {line}")
                    else:
                        if sched_state == "DONE":
                            print("  Job left the queue (DONE state; no exit-code marker written)")
                        else:
                            print("  Job left the queue in FAILED/EXIT state; no exit-code marker written")
                    print(f"\n  Full log : {job_out}")
                    print("=" * 60)
                    return done_val == "0"

        except KeyboardInterrupt:
            print(f"\n\n  Stopped watching. Job {job_id} is still running on the cluster.")
            print(f"  Use 'Check job status' to see progress later.")
            return False

    def run_analysis(self):
        """Run the analysis - just calls submit_batch_job for backwards compatibility."""
        return self.submit_batch_job()

    # ── Motif-only batch job ──────────────────────────────────────────────────

    def submit_motif_batch_job(self):
        """Submit a standalone motif+GO analysis batch job starting from a BED file.

        Uploads te_motif.py + te_go.py, builds a comprehensive batch script
        that runs bedtools intersect, Fisher enrichment, and GO annotation, then
        submits it via the cluster scheduler (LSF bsub or Slurm sbatch).
        """
        print()
        print("=" * 60)
        print("  Submit Motif+GO Batch Job  (from TE loci BED)")
        print("=" * 60)

        # ── Collect parameters interactively ──────────────────────────────────
        default_bed = f"{self.remote_work_dir}/results/motif_analysis/te_loci.bed"
        bed_path = input(f"\nRemote path to TE loci BED [{default_bed}]: ").strip()
        if not bed_path:
            bed_path = default_bed

        # Prefer the tabix-indexed path; fall back to plain BED path.
        default_jaspar = (str(self.params.get("JASPAR_TABIX_PATH", "")).strip()
                          or str(self.params.get("JASPAR_BED_PATH", "")).strip())
        jaspar_bed = input(f"Remote path to JASPAR BED / tabix .bed.gz [{default_jaspar or 'auto-resolve'}]: ").strip()
        if not jaspar_bed:
            jaspar_bed = default_jaspar

        build_default = self.params.get("ASSEMBLY", "hg38")
        build = input(f"Genome build [{build_default}]: ").strip() or build_default

        default_out = f"{self.remote_work_dir}/results"
        out_dir = input(f"Output directory on cluster [{default_out}]: ").strip()
        if not out_dir:
            out_dir = default_out

        p_thresh = input("Fisher p-value threshold [0.05]: ").strip() or "0.05"
        run_go   = input("Also run GO annotation after motif analysis? [y]: ").strip().lower()
        run_go   = run_go not in ("n", "no", "0", "false")

        mem_mb   = int(input(f"Memory (MB) [{self.params['MEM_MB']}]: ").strip() or self.params["MEM_MB"])
        cpus     = int(input(f"CPUs [{self.params['CPUS']}]: ").strip() or self.params["CPUS"])
        walltime = input(f"Walltime HH:MM [{self.params['WALLTIME']}]: ").strip() or self.params["WALLTIME"]
        queue    = input(f"Queue/partition [{self.params['QUEUE']}]: ").strip() or self.params["QUEUE"]

        confirm = input(
            f"\nSubmit motif batch job?\n"
            f"  BED:     {bed_path}\n"
            f"  JASPAR:  {jaspar_bed or '(auto-resolve)'}\n"
            f"  build:   {build}   out: {out_dir}\n"
            f"  mem={mem_mb}MB  cpus={cpus}  walltime={walltime}  queue={queue}\n"
            f"  run-go:  {run_go}\n"
            f"\nProceed? (y/n): "
        ).strip().lower()
        if confirm != "y":
            print("Cancelled.")
            return False

        # ── Upload scripts ─────────────────────────────────────────────────────
        _log("Uploading te_motif.py, te_go.py, requirements.txt ...")
        for name in ["te_motif.py", "te_go.py", "requirements.txt"]:
            local_path = Path(__file__).parent / name
            if not local_path.exists():
                _log(f"  Warning: {name} not found locally — skipping upload")
                continue
            remote_path = f"{self.remote_work_dir}/{name}"
            if not self._upload_text_file(local_path, remote_path, name):
                print(f"Error: failed to upload {name}")
                return False

        # ── Build job script paths ─────────────────────────────────────────────
        import datetime as _dt
        _ts        = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        job_name   = "te_motif_go"
        job_script = f"{self.remote_work_dir}/te_motif_job_{_ts}.sh"
        job_out    = f"{self.remote_work_dir}/te_motif_job_{_ts}.out"
        job_err    = f"{self.remote_work_dir}/te_motif_job_{_ts}.err"
        job_done   = f"{self.remote_work_dir}/te_motif_job_{_ts}.done"
        job_info   = f"{self.remote_work_dir}/te_motif_job_{_ts}.info"
        job_errlog = f"{self.remote_work_dir}/te_motif_job_{_ts}.error.log"
        scratch    = f"{self.remote_work_dir}/tmp_motif"

        sched_header = self._job_script_header(
            job_name, job_out, job_err,
            mem_mb=mem_mb, cpus=cpus, walltime=walltime, queue=queue,
        )
        module_block = self._module_load_block()
        venv_block   = self._venv_setup_block()

        # Build te_motif.py CLI
        motif_cmd = (
            f"python -u {self.remote_work_dir}/te_motif.py"
            f" --bed-input {bed_path}"
            f" --build {build}"
            f" --out-dir {out_dir}"
            f" --p-threshold {p_thresh}"
        )
        if jaspar_bed:
            motif_cmd += f" --jaspar-bed {jaspar_bed}"
        if run_go:
            motif_cmd += f" --run-go"

        # Build te_go.py CLI (explicit fallback if --run-go fails)
        go_cmd = (
            f"python -u {self.remote_work_dir}/te_go.py"
            f" --enrichment-dir {out_dir}/enrichment_results"
            f" --build {build}"
            f" --out-dir {out_dir}"
            f" --p-threshold {p_thresh}"
        )

        batch_script = f'''#!/bin/bash
{sched_header}
{module_block}

# ── Virtual environment + package setup ───────────────────────────────────────
{venv_block}

# ── Install bedtools via conda if missing ─────────────────────────────────────
if ! command -v bedtools >/dev/null 2>&1; then
    if command -v conda >/dev/null 2>&1; then
        echo "[$(date +%H:%M:%S)] bedtools not found — installing via conda ..."
        conda install -y -c bioconda bedtools 2>/dev/null \\
            && echo "[$(date +%H:%M:%S)] bedtools installed" \\
            || echo "[$(date +%H:%M:%S)] conda bedtools install failed; job will abort at intersect"
    else
        echo "[$(date +%H:%M:%S)] WARNING: bedtools not found and conda unavailable"
    fi
fi

# ── Job banner ─────────────────────────────────────────────────────────────────
echo "=========================================================="
echo "  GAMECA — Motif+GO Batch Job"
echo "=========================================================="
echo "  Job script : {job_script}"
echo "  Job ID     : ${{LSB_JOBID:-${{SLURM_JOB_ID:-unknown}}}}"
echo "  Host       : $(hostname)"
echo "  Date       : $(date)"
echo "  CPUs alloc : $(nproc 2>/dev/null || echo N/A)"
echo "  Mem req    : {mem_mb} MB"
echo "  Python     : $(python --version 2>&1)"
echo "  bedtools   : $(bedtools --version 2>/dev/null || echo 'not found')"
echo "=========================================================="
echo "  BED input  : {bed_path}"
echo "  JASPAR BED : {jaspar_bed or '(auto-resolve)'}"
echo "  Build      : {build}"
echo "  Out dir    : {out_dir}"
echo "  p-thresh   : {p_thresh}"
echo "  Run GO     : {str(run_go).lower()}"
echo "  Scratch    : {scratch}"
echo "=========================================================="

# ── Error log init ─────────────────────────────────────────────────────────────
ERROR_LOG="{job_errlog}"
echo "=== te_motif+GO Error Log ===" > "$ERROR_LOG"
echo "Job ID   : ${{LSB_JOBID:-${{SLURM_JOB_ID:-unknown}}}}" >> "$ERROR_LOG"
echo "Host     : $(hostname)" >> "$ERROR_LOG"
echo "Started  : $(date)"     >> "$ERROR_LOG"
echo "" >> "$ERROR_LOG"

# ── Pre-flight checks ──────────────────────────────────────────────────────────
echo ""
echo "[$(date +%H:%M:%S)] === PRE-FLIGHT CHECKS ==="

# 1. BED input
if [ ! -f "{bed_path}" ]; then
    echo "[$(date +%H:%M:%S)] FATAL: BED file not found: {bed_path}" | tee -a "$ERROR_LOG"
    echo "1" > "{job_done}"; exit 1
fi
BED_LINES=$(wc -l < "{bed_path}" 2>/dev/null || echo "?")
BED_SIZE=$(du -sh "{bed_path}" 2>/dev/null | cut -f1)
echo "[$(date +%H:%M:%S)]   BED file  : OK  ($BED_LINES lines, $BED_SIZE)"
echo "[$(date +%H:%M:%S)]   BED head  :"
head -3 "{bed_path}" | sed 's/^/    /'

# 2. JASPAR BED (if provided)
JASPAR_ARG="{jaspar_bed}"
if [ -n "$JASPAR_ARG" ]; then
    if [ ! -f "$JASPAR_ARG" ]; then
        echo "[$(date +%H:%M:%S)] FATAL: JASPAR BED not found: $JASPAR_ARG" | tee -a "$ERROR_LOG"
        echo "1" > "{job_done}"; exit 1
    fi
    JASP_SIZE=$(du -sh "$JASPAR_ARG" 2>/dev/null | cut -f1)
    JASP_SIZE="${{JASP_SIZE:-$(wc -c < "$JASPAR_ARG" 2>/dev/null | awk '{{print $1" B"}}')}}"
    echo "[$(date +%H:%M:%S)]   JASPAR BED: OK ($JASP_SIZE)"
    # Check for required .tbi index alongside the .bed.gz
    JASPAR_TBI="${{JASPAR_ARG}}.tbi"
    if [ ! -f "$JASPAR_TBI" ]; then
        echo "[$(date +%H:%M:%S)] FATAL: .tbi index not found alongside JASPAR file: $JASPAR_TBI" | tee -a "$ERROR_LOG"
        echo "  Expected: ${{JASPAR_ARG}}.tbi" | tee -a "$ERROR_LOG"
        echo "  Run: tabix -p bed $JASPAR_ARG   to build the index" | tee -a "$ERROR_LOG"
        echo "1" > "{job_done}"; exit 1
    fi
    TBI_SIZE=$(du -sh "$JASPAR_TBI" 2>/dev/null | cut -f1)
    TBI_SIZE="${{TBI_SIZE:-$(wc -c < "$JASPAR_TBI" 2>/dev/null | awk '{{print $1" B"}}')}}"
    echo "[$(date +%H:%M:%S)]   JASPAR .tbi: OK ($TBI_SIZE)"
else
    echo "[$(date +%H:%M:%S)]   JASPAR BED: will be auto-resolved by te_motif.py"
fi

# 3. bedtools
if ! command -v bedtools >/dev/null 2>&1; then
    echo "[$(date +%H:%M:%S)] FATAL: bedtools not found on PATH" | tee -a "$ERROR_LOG"
    echo "  PATH=$PATH" >> "$ERROR_LOG"
    echo "1" > "{job_done}"; exit 1
fi
echo "[$(date +%H:%M:%S)]   bedtools  : $(bedtools --version 2>&1 | head -1)"

# 4. Python packages
echo "[$(date +%H:%M:%S)]   Checking Python packages ..."
python - <<'PKGCHECK'
import sys
missing = []
for pkg in ["pandas", "numpy", "scipy", "matplotlib", "requests"]:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)
if missing:
    print(f"  [WARN] Missing packages: {{missing}}")
else:
    print("  [OK] All required packages available")
PKGCHECK

# 5. Disk space
mkdir -p "{scratch}"
FREE_GB=$(python3 -c "import shutil; t,u,f=shutil.disk_usage('{scratch}'); print(f'{{f/1e9:.1f}}')" 2>/dev/null || echo "?")
echo "[$(date +%H:%M:%S)]   Scratch   : {scratch} (free: ${{FREE_GB}} GB)"

if python3 -c "import shutil,sys; t,u,f=shutil.disk_usage('{scratch}'); sys.exit(0 if f>2e9 else 1)" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)]   Disk      : OK"
else
    echo "[$(date +%H:%M:%S)] WARNING: Less than 2 GB free in scratch — bedtools may fail" | tee -a "$ERROR_LOG"
fi

# 6. Output directory
mkdir -p "{out_dir}"
echo "[$(date +%H:%M:%S)]   Out dir   : {out_dir} (created/exists)"

echo "[$(date +%H:%M:%S)] === PRE-FLIGHT COMPLETE ==="
echo ""

cd {self.remote_work_dir}

# ── Stage 1: te_motif.py ───────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] =============================="
echo "[$(date +%H:%M:%S)] STAGE 1: te_motif.py"
echo "[$(date +%H:%M:%S)]   Command: {motif_cmd}"
echo "[$(date +%H:%M:%S)] =============================="
SECONDS=0
export TMPDIR="{scratch}"

{motif_cmd} 2>&1 | tee -a "$ERROR_LOG"
MOTIF_EXIT=${{PIPESTATUS[0]}}
MOTIF_SECS=$SECONDS

echo ""
echo "[$(date +%H:%M:%S)] te_motif.py exit code: $MOTIF_EXIT  (runtime: ${{MOTIF_SECS}}s)"

if [ $MOTIF_EXIT -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] FATAL: te_motif.py failed (exit=$MOTIF_EXIT)" | tee -a "$ERROR_LOG"
    echo "1" > "{job_done}"; exit $MOTIF_EXIT
fi

# Log output file inventory after motif stage
echo ""
echo "[$(date +%H:%M:%S)] --- Motif stage output files ---"
find "{out_dir}/motif_analysis" "{out_dir}/enrichment_results" \\
     -type f \\( -name "*.csv" -o -name "*.tsv" -o -name "*.png" -o -name "*.log" \\) \\
     -exec ls -lh {{}} \\; 2>/dev/null | awk '{{print "  "$0}}'
echo ""

''' + (f'''
# ── Stage 2 (explicit): te_go.py ──────────────────────────────────────────────
# Only runs if te_motif.py did NOT already chain GO internally
GO_DONE_MARKER="{out_dir}/go_annotations/gene_functions.csv"
if [ -f "$GO_DONE_MARKER" ]; then
    echo "[$(date +%H:%M:%S)] GO annotation already completed (--run-go chained by te_motif)."
else
    echo "[$(date +%H:%M:%S)] =============================="
    echo "[$(date +%H:%M:%S)] STAGE 2: te_go.py (standalone)"
    echo "[$(date +%H:%M:%S)]   Command: {go_cmd}"
    echo "[$(date +%H:%M:%S)] =============================="
    SECONDS=0
    {go_cmd} 2>&1 | tee -a "$ERROR_LOG"
    GO_EXIT=${{PIPESTATUS[0]}}
    GO_SECS=$SECONDS
    echo "[$(date +%H:%M:%S)] te_go.py exit code: $GO_EXIT  (runtime: ${{GO_SECS}}s)"
    if [ $GO_EXIT -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] WARNING: te_go.py failed (exit=$GO_EXIT) — continuing" | tee -a "$ERROR_LOG"
    fi
fi
''' if run_go else '''
# GO annotation not requested (run_go=false)
echo "[$(date +%H:%M:%S)] GO annotation skipped (not requested)."
''') + f'''

# ── Final output inventory ─────────────────────────────────────────────────────
echo ""
echo "[$(date +%H:%M:%S)] === FINAL OUTPUT INVENTORY ==="
find "{out_dir}" -maxdepth 4 \\
     -type f \\( -name "*.csv" -o -name "*.tsv" -o -name "*.png" -o -name "*.log" -o -name "*.txt" \\) \\
     -exec ls -lh {{}} \\; 2>/dev/null | sort | awk '{{print "  "$0}}'

RESULT_SIZE=$(du -sh "{out_dir}" 2>/dev/null | cut -f1)
echo ""
echo "[$(date +%H:%M:%S)] Total output size: $RESULT_SIZE at {out_dir}"

TOTAL_SECS=$SECONDS
echo ""
echo "=========================================================="
echo "  Job complete"
echo "  Total runtime : $((TOTAL_SECS / 60))m $((TOTAL_SECS % 60))s"
echo "  Date          : $(date)"
echo "=========================================================="

echo "" >> "$ERROR_LOG"
echo "=== JOB SUCCEEDED ===" >> "$ERROR_LOG"
echo "Ended: $(date)" >> "$ERROR_LOG"

echo "0" > "{job_done}"
exit 0
'''

        # ── Write job script to cluster ────────────────────────────────────────
        _log(f"Writing batch job script to {job_script}")
        # Write via echo-to-file in chunks to avoid heredoc quoting issues
        import base64 as _b64
        encoded = _b64.b64encode(batch_script.encode()).decode()
        chunk   = 60000
        chunks  = [encoded[i:i+chunk] for i in range(0, len(encoded), chunk)]

        cmd0 = f"echo '{chunks[0]}' | base64 -d > {job_script}"
        out, err, code = self.run_command(cmd0, timeout=30)
        if code != 0:
            print(f"Error writing job script (chunk 1): {err}")
            return False
        for i, ch in enumerate(chunks[1:], 2):
            cmd_i = f"echo '{ch}' | base64 -d >> {job_script}"
            out, err, code = self.run_command(cmd_i, timeout=30)
            if code != 0:
                print(f"Error writing job script (chunk {i}/{len(chunks)}): {err}")
                return False

        self.run_command(f"chmod +x {job_script}", timeout=10)
        self.run_command(f"rm -f {job_out} {job_err} {job_done} {job_info}", timeout=10)

        # ── Submit ─────────────────────────────────────────────────────────────
        sched_label = (self.scheduler or "lsf").upper()
        print(f"\nSubmitting motif batch job via {sched_label}...")
        _log(f"Submit command: {self._submit_job_cmd(job_script)}")
        out, err, code = self.run_command(self._submit_job_cmd(job_script), timeout=30)

        if code != 0:
            print(f"Error submitting job: {err}")
            return False

        job_id = self._parse_job_id(out)
        if job_id:
            self.current_job_id = job_id
            print(f"\nJob submitted successfully!  Job ID: {job_id}")
        else:
            print(f"Job submitted (could not parse ID from: {out})")
            job_id = "unknown"
            self.current_job_id = None

        info = (f"JOB_ID={job_id}\nTYPE=motif_go\n"
                f"BED={bed_path}\nOUTPUT_DIR={out_dir}\n"
                f"LOG_OUT={job_out}\nLOG_ERR={job_err}\n"
                f"LOG_DONE={job_done}\nLOG_ERRLOG={job_errlog}\n"
                f"SUBMITTED=$(date)")
        self.run_command(f"echo '{info}' > {job_info}", timeout=10)

        print()
        print("=" * 60)
        print("MOTIF+GO BATCH JOB SUBMITTED")
        print("=" * 60)
        print(f"  Job ID     : {job_id}")
        print(f"  Job output : {job_out}")
        print(f"  Job errors : {job_err}")
        print(f"  Error log  : {job_errlog}")
        print(f"  Results    : {out_dir}")
        sched = self.scheduler or "lsf"
        print()
        if sched == "slurm":
            print(f"  squeue -j {job_id}   # status")
            print(f"  tail -f {job_out}     # live log")
            print(f"  scancel {job_id}      # cancel")
        else:
            print(f"  bjobs {job_id}        # status")
            print(f"  bpeek {job_id}        # live log")
            print(f"  bkill {job_id}        # cancel")
        print("=" * 60)
        return True

    # ── te_prep / te_enrichment remote launchers ────────────────────────────

    def _run_te_prep_interactive(self):
        """Interactively configure and run te_prep on the cluster."""
        print("\n" + "=" * 60)
        print("te_prep — Download rmsk / Extract TE sequences")
        print("=" * 60)
        species = input("Species (human/mouse) [human]: ").strip().lower() or "human"
        default_build = "mm10" if species in {"mouse", "mus musculus"} else "hg38"
        build   = input(f"Assembly/build [{default_build}]: ").strip() or default_build
        family  = input("TE family name (e.g. HERVK): ").strip()
        genome  = input("Local assembly FASTA path on cluster (blank = none): ").strip()
        out_dir = input(f"Remote output dir [{self.remote_work_dir}]: ").strip() or self.remote_work_dir
        extra   = input("Extra te_prep args (or blank): ").strip()

        fam_arg = f"--family {family}" if family else ""
        genome_arg = f"--genome-fa {genome}" if genome else ""
        cmd = (
            f"cd {self.remote_work_dir} && "
            f"python te_prep.py --build {build} {fam_arg} {genome_arg} --out-dir {out_dir} {extra}"
        )
        print(f"\nRunning: {cmd}\n")
        out, err, code = self.run_command(cmd, timeout=600)
        print(out)
        if err:
            print("[stderr]", err[:500])
        print("Exit code:", code)

    def _run_te_enrichment_interactive(self):
        """Interactively configure and run te_enrichment on the cluster."""
        print("\n" + "=" * 60)
        print("te_enrichment — UMAP / JASPAR motifs / Fisher / GO")
        print("=" * 60)
        clustered = input("Path to clustered CSV on cluster: ").strip()
        if not clustered:
            print("Clustered CSV path required.")
            return
        species  = input("Species (human/mouse) [human]: ").strip().lower() or "human"
        default_build = "mm10" if species in {"mouse", "mus musculus"} else "hg38"
        build    = input(f"Assembly/build [{default_build}]: ").strip() or default_build
        family   = input("TE family name [FAMILY]: ").strip() or "FAMILY"
        out_dir  = input(f"Remote output dir [{self.remote_work_dir}]: ").strip() or self.remote_work_dir
        jaspar   = input("JASPAR BED path (blank = auto-download): ").strip()
        p_thresh = input("Fisher p-value significance threshold [0.05]: ").strip() or "0.05"
        extra    = input("Extra te_enrichment args (or blank): ").strip()

        jaspar_arg = f"--jaspar-bed {jaspar}" if jaspar else ""
        cmd = (
            f"cd {self.remote_work_dir} && "
            f"python te_enrichment.py --input {clustered} --build {build} "
            f"--family {family} --out-dir {out_dir} {jaspar_arg} "
            f"--p-threshold {p_thresh} {extra}"
        )
        print(f"\nRunning: {cmd}\n")
        out, err, code = self.run_command(cmd, timeout=1800)
        print(out)
        if err:
            print("[stderr]", err[:500])
        print("Exit code:", code)

    def generate_clustering_plots(self):
        """Generate clustering plots after analysis."""
        if not self.connected:
            print("Not connected to HPC")
            return False

        family = self.params["FAMILY_NAME"].lower()
        base_dir = self.params['BASE_OUT_DIR']

        # Build script line-by-line — no nested .format() escaping nightmares.
        # BASEDIR_PH and FAMILY_PH are replaced via str.replace() at the end.
        script_lines = [
            "import os, sys",
            "import pandas as pd",
            "import numpy as np",
            "import matplotlib; matplotlib.use('Agg')",
            "import matplotlib.pyplot as plt",
            "from pathlib import Path",
            "from sklearn.cluster import AgglomerativeClustering",
            "from scipy.cluster.hierarchy import dendrogram, linkage",
            "",
            "def plot_dendrogram(model, **kwargs):",
            "    counts = np.zeros(model.children_.shape[0])",
            "    n_samples = len(model.labels_)",
            "    for i, merge in enumerate(model.children_):",
            "        current_count = 0",
            "        for child_idx in merge:",
            "            if child_idx < n_samples:",
            "                current_count += 1",
            "            else:",
            "                current_count += counts[child_idx - n_samples]",
            "        counts[i] = current_count",
            "    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)",
            "    dendrogram(linkage_matrix, **kwargs)",
            "",
            "base_dir = Path('BASEDIR_PH')",
            "family = 'FAMILY_PH'",
            "output_dir = base_dir / family",
            "output_dir.mkdir(parents=True, exist_ok=True)",
            "",
            "try:",
            "    data_path = output_dir / (family + '_clustering_data.csv')",
            "    if not data_path.exists():",
            "        print('Error: Clustering data not found at ' + str(data_path))",
            "        sys.exit(1)",
            "    df = pd.read_csv(data_path)",
            "    print('Loaded clustering data with ' + str(len(df)) + ' sequences')",
            "    sequences = df['sequence'].values",
            "    X = df.drop(columns=['sequence']).values",
            "    print('Performing hierarchical clustering...')",
            "    n_clusters = min(10, len(sequences))",
            "    model = AgglomerativeClustering(",
            "        n_clusters=n_clusters, affinity='euclidean',",
            "        linkage='ward', compute_distances=True)",
            "    clusters = model.fit_predict(X)",
            "    print('Generating dendrogram...')",
            "    plt.figure(figsize=(12, 8))",
            "    plot_dendrogram(model, truncate_mode='level', p=5)",
            "    plt.title('Hierarchical Clustering Dendrogram - ' + family)",
            "    plt.xlabel('Sample index or (cluster size)')",
            "    plt.ylabel('Distance')",
            "    plt.tight_layout()",
            "    plot_path = output_dir / (family + '_dendrogram.png')",
            "    plt.savefig(plot_path, dpi=300, bbox_inches='tight')",
            "    print('Saved dendrogram to ' + str(plot_path))",
            "    print('Generating consensus sequences...')",
            "    df['cluster'] = clusters",
            "    for cluster_id in range(n_clusters):",
            "        cluster_seqs = df[df['cluster'] == cluster_id]['sequence'].tolist()",
            "        if not cluster_seqs: continue",
            "        consensus = []",
            "        max_len = max(len(seq) for seq in cluster_seqs)",
            "        for i in range(max_len):",
            "            bases = {}",
            "            for seq in cluster_seqs:",
            "                if i < len(seq):",
            "                    base = seq[i].upper()",
            "                    bases[base] = bases.get(base, 0) + 1",
            "            if bases:",
            "                consensus.append(max(bases.items(), key=lambda x: x[1])[0])",
            "            else:",
            "                consensus.append('-')",
            "        consensus_seq = ''.join(consensus)",
            "        cpath = output_dir / (family + '_cluster_' + str(cluster_id+1) + '_consensus.txt')",
            "        with open(cpath, 'w') as fout:",
            "            fout.write('>consensus_cluster_' + str(cluster_id+1) + '\\n' + consensus_seq + '\\n')",
            "        gap_reduced = ''.join(b for b in consensus if b != '-')",
            "        grpath = output_dir / (family + '_cluster_' + str(cluster_id+1) + '_consensus_gap_reduced.txt')",
            "        with open(grpath, 'w') as fout:",
            "            fout.write('>consensus_cluster_' + str(cluster_id+1) + '_gap_reduced\\n' + gap_reduced + '\\n')",
            "    print('Consensus sequences generated successfully!')",
            "except Exception as e:",
            "    print('Error generating clustering plots: ' + str(e), file=sys.stderr)",
            "    sys.exit(1)",
        ]

        formatted_script = "\n".join(script_lines)
        formatted_script = formatted_script.replace("BASEDIR_PH", base_dir.replace("'", "\\'"))
        formatted_script = formatted_script.replace("FAMILY_PH", family.replace("'", "\\'"))

        # Save script to a temporary file on the remote
        script_path = f"{self.remote_work_dir}/generate_plots_{os.getpid()}.py"

        if self.use_sftp and self.sftp:
            with self.sftp.file(script_path, 'w') as f:
                f.write(formatted_script)
        else:
            encoded_full = base64.b64encode(formatted_script.encode()).decode()
            chunk_size = 65000
            chunks = [encoded_full[i:i+chunk_size] for i in range(0, len(encoded_full), chunk_size)]
            cmd = f"echo '{chunks[0]}' | base64 -d > {script_path}"
            out, err, code = self.run_command(cmd)
            if code != 0:
                print(f"Failed to upload clustering script: {err}")
                return False
            for chunk in chunks[1:]:
                cmd = f"echo '{chunk}' | base64 -d >> {script_path}"
                out, err, code = self.run_command(cmd)
                if code != 0:
                    print(f"Failed to upload clustering script chunk: {err}")
                    return False

        self.run_command(f"chmod +x {script_path}")

        print("Generating clustering plots and consensus sequences...")
        out, err, code = self.run_command(f"python {script_path}", timeout=600)

        self.run_command(f"rm -f {script_path}")

        if code != 0:
            print(f"Error generating plots: {err}")
            return False

        print(out)
        return True

    # ------------------------------------------------------------------
    # Gmail App Password + email notifications
    # ------------------------------------------------------------------

    _APP_PASSWORD_STATE_KEY = "gmail_app_password"
    _SENDER_EMAIL_STATE_KEY = "gmail_sender_email"

    def _get_email_creds(self) -> tuple[str, str]:
        """Return (sender_email, app_password), prompting and saving any missing values."""
        sender = self._state.get(self._SENDER_EMAIL_STATE_KEY, "").strip()
        pw = self._state.get(self._APP_PASSWORD_STATE_KEY, "").strip()

        if sender and pw:
            return sender, pw

        print("\n" + "="*60)
        print("GMAIL CREDENTIALS REQUIRED")
        print("="*60)
        print()

        if not sender:
            sender = input("Gmail address to send FROM: ").strip()
            if not sender:
                print("  No email entered — notifications disabled.")
                return "", ""
            self._state[self._SENDER_EMAIL_STATE_KEY] = sender
            self._save_state()
            print(f"  Sender email saved: {sender}")

        if not pw:
            print()
            print("An App Password is needed (not your regular Gmail password).")
            print("Generate one at: myaccount.google.com/apppasswords")
            print("(16-character password, spaces optional)")
            print()
            pw = getpass.getpass("Paste your Gmail App Password: ").strip().replace(" ", "")
            if not pw:
                print("  No password entered — notifications disabled.")
                return "", ""
            self._state[self._APP_PASSWORD_STATE_KEY] = pw
            self._save_state()
            print("  App password saved to state file.")

        return sender, pw

    def setup_email_auth(self):
        """Clear stored Gmail credentials and prompt for fresh ones."""
        print("\n" + "="*60)
        print("GMAIL APP PASSWORD SETUP")
        print("="*60)
        print()
        print("Steps (one-time):")
        print("  1. Go to myaccount.google.com/apppasswords")
        print("  2. Create an app password (select 'Mail' / 'Other')")
        print("  3. Copy the 16-character password shown")
        print()
        self._state.pop(self._APP_PASSWORD_STATE_KEY, None)
        self._state.pop(self._SENDER_EMAIL_STATE_KEY, None)
        sender, pw = self._get_email_creds()
        if sender and pw:
            print("  Setup complete — notifications are enabled.")

    def _send_notification_email(self, to: str, local_path: str, remote_dir: str):
        """Send job-complete notification via Gmail SMTP with App Password."""
        import smtplib

        sender, pw = self._get_email_creds()
        if not sender or not pw:
            return

        family = self.params.get("FAMILY_NAME", "")
        subject = f"TE Analysis Complete: {family}"
        body = (
            f"Your TE analysis pipeline has completed.\n\n"
            f"Family:        {family}\n"
            f"Remote output: {remote_dir}\n"
            f"Local path:    {local_path}\n"
        )

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = to

        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                smtp.login(sender, pw)
                smtp.send_message(msg)
            print(f"  Notification email sent to {to}")
        except smtplib.SMTPAuthenticationError:
            print("  Email auth failed — run option [g] to update your credentials.")
            self._state.pop(self._APP_PASSWORD_STATE_KEY, None)
            self._state.pop(self._SENDER_EMAIL_STATE_KEY, None)
            self._save_state()
        except Exception as e:
            print(f"  Email send failed: {e}")

    # ------------------------------------------------------------------
    # tmpfiles.org upload
    # ------------------------------------------------------------------

    def _upload_to_tmpfiles(self, local_path: Path) -> str:
        """Zip local_path and upload to tmpfiles.org; return download URL or ''."""
        try:
            import requests as _req
        except ImportError:
            print("  tmpfiles upload requires: pip install requests")
            return ""

        try:
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                zip_path = tmp.name

            print("  Compressing results...")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                if local_path.is_dir():
                    for f in sorted(local_path.rglob("*")):
                        if f.is_file():
                            zf.write(f, f.relative_to(local_path))
                else:
                    zf.write(local_path, local_path.name)

            size_mb = os.path.getsize(zip_path) / 1024 / 1024
            print(f"  Uploading {size_mb:.1f} MB to tmpfiles.org ...")

            with open(zip_path, "rb") as f:
                resp = _req.post(
                    "https://tmpfiles.org/api/v1/upload",
                    files={"file": f},
                    timeout=300,
                )
            resp.raise_for_status()
            url = resp.json()["data"]["url"]
            return url.replace("tmpfiles.org/", "tmpfiles.org/dl/")

        except Exception as e:
            print(f"  Upload failed: {e}")
            return ""
        finally:
            if "zip_path" in dir() and os.path.exists(zip_path):
                os.unlink(zip_path)

    def _notify_job_complete(self, local_path: Path, remote_dir: str):
        """Email a completion notification with the local path if NOTIFY_EMAIL is set."""
        to = str(self.params.get("NOTIFY_EMAIL", "")).strip()
        if not to:
            return
        print(f"\nSending completion notification to {to} ...")
        self._send_notification_email(to, str(local_path), remote_dir)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _load_state() -> dict:
        try:
            with open(_STATE_FILE) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_state(self):
        state = dict(self._state)
        if self.remote_output_dir:
            state["last_remote_dir"] = self.remote_output_dir
        if self.local_output_dir:
            state["last_local_dir"] = str(self.local_output_dir)
        try:
            with open(_STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except OSError:
            pass
        self._state = state

    # ------------------------------------------------------------------
    # Remote directory helpers
    # ------------------------------------------------------------------

    def _find_last_remote_dir(self) -> str:
        """Return the most recently created subdirectory under BASE_OUT_DIR, falling back to cached state."""
        base = f"{self.remote_work_dir}/{self.params['BASE_OUT_DIR']}"
        out, _, _ = self.run_command(
            f"ls -td {base}/*/ 2>/dev/null | head -1 | tr -d '\\n'",
            timeout=10,
        )
        live = out.strip().rstrip('/')
        if live:
            return live
        return self._state.get("last_remote_dir", "")

    def retrieve_results(self, local_dir: str, remote_out_override: str = None):
        """Download results from HPC to local directory."""
        # First check if there's a completed job
        job_info_file = f"{self.remote_work_dir}/te_analysis_job.info"
        job_done_file = f"{self.remote_work_dir}/te_analysis_job.done"

        # Check job completion status
        done_out, _, _ = self.run_command(f"cat {job_done_file} 2>/dev/null", timeout=10)
        if not done_out.strip():
            print("\nWarning: Job may not be complete yet.")
            confirm = input("Retrieve partial results anyway? (y/n): ").strip().lower()
            if confirm != 'y':
                return
        else:
            try:
                exit_code = int(done_out.strip())
                if exit_code != 0:
                    print(f"\nWarning: Job completed with error (exit code: {exit_code})")
                    confirm = input("Retrieve results anyway? (y/n): ").strip().lower()
                    if confirm != 'y':
                        return
            except ValueError:
                pass

        # Use caller-supplied path, job info, or constructed fallback
        if remote_out_override:
            remote_out = remote_out_override
        else:
            remote_out = None
            info_out, _, _ = self.run_command(f"cat {job_info_file} 2>/dev/null", timeout=10)
            if info_out.strip():
                for line in info_out.strip().split('\n'):
                    if line.startswith('OUTPUT_DIR='):
                        remote_out = line.split('=', 1)[1]
                        break

            if not remote_out:
                family = self.params["FAMILY_NAME"].lower()
                remote_out = f"{self.remote_work_dir}/{self.params['BASE_OUT_DIR']}/{family}"

        # Cache the resolved remote directory for future sessions
        self._state["last_remote_dir"] = remote_out
        self._save_state()

        # Verify remote directory exists
        check_out, _, _ = self.run_command(f"test -d '{remote_out}' && echo 'exists'", timeout=10)
        if 'exists' not in check_out:
            print(f"\nError: Remote results directory not found: {remote_out}")
            print("The job may not have completed successfully.")
            return

        # Optionally generate clustering plots
        gen_plots = input("\nGenerate clustering plots before download? (y/n): ").strip().lower()
        if gen_plots == 'y':
            print("Generating clustering plots and consensus sequences...")
            if not self.generate_clustering_plots():
                print("Warning: Failed to generate clustering plots. Continuing with available results...")

        # Expand ~ and make path absolute
        local_path = Path(local_dir).expanduser()
        if not local_path.is_absolute():
            local_path = Path.cwd() / local_path

        try:
            local_path.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            print(f"Error: Permission denied creating directory: {local_path}")
            print("Please check the path and try again.")
            return
        except OSError as e:
            print(f"Error creating directory: {e}")
            print("Hint: Use '~' for home directory (e.g., ~/Documents/output)")
            print("      Or use relative path (e.g., ./output)")
            return

        print(f"\nRetrieving results from {remote_out}")
        print(f"Saving to {local_path}")

        if self.use_sftp and self.sftp:
            def download_recursive(remote_path, local_base):
                """Recursively download directory."""
                try:
                    items = self.sftp.listdir_attr(remote_path)
                except IOError:
                    print(f"Cannot access {remote_path}")
                    return

                for item in items:
                    remote_item = f"{remote_path}/{item.filename}"
                    local_item = local_base / item.filename

                    if stat.S_ISDIR(item.st_mode):
                        local_item.mkdir(exist_ok=True)
                        download_recursive(remote_item, local_item)
                    else:
                        print(f"  Downloading: {item.filename}")
                        self.sftp.get(remote_item, str(local_item))

            download_recursive(remote_out, local_path)
        else:
            # Stream tar directly through SSH channel (no temp files, no compression)
            print("Streaming results from remote...")

            # Check if directory exists
            out, err, code = self.run_command(f"test -d '{remote_out}' && echo 'exists'")
            if 'exists' not in out:
                print(f"Error: Remote directory not found: {remote_out}")
                return

            try:
                channel = self._transport.open_session()
                channel.exec_command(f"cd '{remote_out}' && tar -cf - .")

                # Read tar stream into memory buffer and extract
                buf = io.BytesIO()
                bytes_received = 0
                while True:
                    data = channel.recv(262144)  # 256KB buffer
                    if not data:
                        break
                    buf.write(data)
                    bytes_received += len(data)
                    print(f"  Received: {bytes_received / 1024 / 1024:.1f} MB", end='\r')

                channel.close()
                print(f"\n  Total: {bytes_received / 1024 / 1024:.2f} MB")

                # Extract from memory buffer
                buf.seek(0)
                tar_file = tarfile.open(fileobj=buf, mode='r:')
                tar_file.extractall(local_path)
                tar_file.close()

            except Exception as e:
                print(f"\nError during streaming transfer: {e}")
                return

        print(f"\nResults saved to {local_path}")
        self.local_output_dir = local_path
        self._save_state()
        self._notify_job_complete(local_path, remote_out)

    # ------------------------------------------------------------------
    # Selective file download
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_size(n: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024:
                return f"{n:.0f} {unit}"
            n /= 1024
        return f"{n:.1f} GB"

    @staticmethod
    def _parse_file_selection(sel: str, max_n: int):
        """Parse '1,3-5,7' into a sorted list of 0-based indices, or None on error."""
        indices: set[int] = set()
        for part in sel.split(","):
            part = part.strip()
            if "-" in part:
                a, _, b = part.partition("-")
                try:
                    lo, hi = int(a), int(b)
                    if not (1 <= lo <= hi <= max_n):
                        raise ValueError
                    indices.update(range(lo - 1, hi))
                except ValueError:
                    print(f"Invalid range: {part!r}")
                    return None
            else:
                try:
                    i = int(part)
                    if not (1 <= i <= max_n):
                        raise ValueError
                    indices.add(i - 1)
                except ValueError:
                    print(f"Invalid selection: {part!r}")
                    return None
        return sorted(indices)

    def download_selected_files(self):
        """Browse any remote path (file or folder) and download a user-selected subset."""
        # Use last_download_path as default, falling back to last output dir
        last = self._state.get("last_download_path") or self._find_last_remote_dir()
        prompt = f"Remote path (file or folder) [{last}]: " if last else "Remote path (file or folder): "
        remote_path = input(prompt).strip() or last
        if not remote_path:
            print("No remote path specified.")
            return

        # Determine if it's a file or directory
        type_out, _, _ = self.run_command(
            f"if [ -f '{remote_path}' ]; then echo file; elif [ -d '{remote_path}' ]; then echo dir; fi",
            timeout=10,
        )
        path_type = type_out.strip()
        if not path_type:
            print(f"Path not found on HPC: {remote_path}")
            return

        if path_type == "file":
            # Single file — download directly
            files = [(os.path.basename(remote_path), 0)]
            remote_dir = os.path.dirname(remote_path)
            chosen = files
        else:
            # Directory — list contents and let user pick
            remote_dir = remote_path
            print(f"\nListing files in {remote_dir} ...")
            out, _, _ = self.run_command(
                f"find '{remote_dir}' -type f -printf '%P\\t%s\\n' 2>/dev/null | sort",
                timeout=30,
            )
            lines = [l for l in out.strip().split("\n") if l.strip()]
            if not lines:
                print("No files found.")
                return

            files: list[tuple[str, int]] = []
            for line in lines:
                parts = line.split("\t", 1)
                rel = parts[0]
                size = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                files.append((rel, size))

            print(f"\n  {'#':>4}  {'Size':>10}  File")
            print("  " + "-" * 58)
            for i, (rel, size) in enumerate(files, 1):
                print(f"  [{i:>3}]  {self._fmt_size(size):>10}  {rel}")
            print("\n  [all]  Download all files")

            sel = input("\nSelect files to download (e.g., 1,3-5 or all): ").strip().lower()
            if not sel:
                return

            if sel == "all":
                chosen = files
            else:
                indices = self._parse_file_selection(sel, len(files))
                if indices is None:
                    return
                chosen = [files[i] for i in indices]

        default_local = str(self.local_output_dir) if self.local_output_dir else ""
        prompt = f"Local directory [{default_local}]: " if default_local else "Local directory: "
        local_dir = input(prompt).strip() or default_local
        if not local_dir:
            print("No local directory specified.")
            return

        local_path = Path(local_dir).expanduser()
        local_path.mkdir(parents=True, exist_ok=True)

        print(f"\nDownloading {len(chosen)} file(s) to {local_path} ...")

        if self.use_sftp and self.sftp:
            for rel, _ in chosen:
                remote_file = f"{remote_dir}/{rel}"
                local_file = local_path / rel
                local_file.parent.mkdir(parents=True, exist_ok=True)
                print(f"  {rel}")
                try:
                    self.sftp.get(remote_file, str(local_file))
                except Exception as e:
                    print(f"    Error: {e}")
        else:
            file_args = " ".join(shlex.quote(r) for r, _ in chosen)
            try:
                channel = self._transport.open_session()
                channel.exec_command(f"cd '{remote_dir}' && tar -cf - {file_args}")
                buf = io.BytesIO()
                received = 0
                while True:
                    data = channel.recv(262144)
                    if not data:
                        break
                    buf.write(data)
                    received += len(data)
                    print(f"  Received: {received / 1024 / 1024:.1f} MB", end="\r")
                channel.close()
                print()
                buf.seek(0)
                tf = tarfile.open(fileobj=buf, mode="r:")
                tf.extractall(local_path)
                tf.close()
            except Exception as e:
                print(f"\nTransfer error: {e}")
                return

        self.local_output_dir = local_path
        self._state["last_download_path"] = remote_path
        self._save_state()
        print(f"\nDone. {len(chosen)} file(s) saved to {local_path}")

    def test_email_interactive(self):
        """Interactively set sender/receiver/password, print the test script, then run it."""
        import smtplib

        print("\n" + "="*60)
        print("EMAIL TEST")
        print("="*60)
        print()

        # Sender
        stored_sender = self._state.get(self._SENDER_EMAIL_STATE_KEY, "").strip()
        prompt = f"  Sender Gmail address [{stored_sender}]: " if stored_sender else "  Sender Gmail address: "
        sender = input(prompt).strip() or stored_sender
        if not sender:
            print("  No sender address — aborting.")
            return
        self._state[self._SENDER_EMAIL_STATE_KEY] = sender

        # Receiver
        stored_receiver = self._state.get("test_receiver_email", "").strip()
        prompt = f"  Receiver email address [{stored_receiver}]: " if stored_receiver else "  Receiver email address: "
        receiver = input(prompt).strip() or stored_receiver
        if not receiver:
            print("  No receiver address — aborting.")
            return
        self._state["test_receiver_email"] = receiver

        # App password (masked input; show if already stored)
        stored_pw = self._state.get(self._APP_PASSWORD_STATE_KEY, "").strip()
        if stored_pw:
            replace = input(f"  App password already stored. Replace it? (y/n) [n]: ").strip().lower()
            if replace == "y":
                stored_pw = ""
        if not stored_pw:
            print("  Generate an App Password at: myaccount.google.com/apppasswords")
            stored_pw = getpass.getpass("  Paste Gmail App Password: ").strip().replace(" ", "")
            if not stored_pw:
                print("  No password entered — aborting.")
                return
        self._state[self._APP_PASSWORD_STATE_KEY] = stored_pw
        self._save_state()

        # Print the exact script that will run
        script = f"""\
import smtplib
from email.mime.text import MIMEText

SENDER   = "{sender}"
RECEIVER = "{receiver}"
PASSWORD = "{'*' * len(stored_pw)}"  # actual password loaded from state file

msg = MIMEText("Test email from HPC TE Analysis Client.")
msg["Subject"] = "HPC TE Client — Email Test"
msg["From"]    = SENDER
msg["To"]      = RECEIVER

with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
    smtp.login(SENDER, PASSWORD)
    smtp.send_message(msg)

print("Email sent successfully.")"""

        print()
        print("-"*60)
        print("TEST SCRIPT (password masked):")
        print("-"*60)
        print(script)
        print("-"*60)
        print()

        confirm = input("Send test email now? (y/n) [y]: ").strip().lower()
        if confirm == "n":
            print("  Aborted.")
            return

        msg = MIMEText("Test email from HPC TE Analysis Client.")
        msg["Subject"] = "HPC TE Client — Email Test"
        msg["From"] = sender
        msg["To"] = receiver

        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                smtp.login(sender, stored_pw)
                smtp.send_message(msg)
            print(f"  Email sent successfully to {receiver}.")
        except smtplib.SMTPAuthenticationError:
            print("  Auth failed — check that the App Password is correct and that")
            print("  2-Step Verification is enabled on the sender account.")
            self._state.pop(self._APP_PASSWORD_STATE_KEY, None)
            self._save_state()
        except Exception as e:
            print(f"  Send failed: {e}")

    def main_menu(self):
        """Main interactive menu after connection."""
        while True:
            try:
                print("\n" + "="*60)
                print("HPC TE ANALYSIS CLIENT")
                print("="*60)
                sched = getattr(self, "scheduler", None) or "lsf"
                sched_label = sched.upper()
                print("  [1]  Configure parameters")
                print("  [2]  Preview family count")
                print("  --- Prepare & Fetch Data ---")
                print("  [3]  Run te_prep  (download rmsk / extract sequences)")
                print("  --- Core Analysis ---")
                print(f"  [4]  Submit batch job  ({sched_label})")
                print("  --- Download ---")
                print("  [5]  Browse & download files  (any path, selective)")
                print("  --- Enrichment & Motifs ---")
                print("  [6]  Run te_enrichment (UMAP / JASPAR / Fisher / GO)")
                print("  --- Monitor & Retrieve ---")
                print("  [7]  Check batch job status")
                print("  [8]  Watch batch job progress (live)")
                print("  [9]  Retrieve results  (all files)")
                print("  [10] Download error logs only")
                print("  [11] Disconnect and exit")
                print("  [12] Test email (set sender / receiver / password and send test)")
                print("="*60)

                choice = input("\nSelect option (1-12): ").strip()

                if choice == '1':
                    if self.set_parameter_interactive():
                        self.submit_batch_job()
                elif choice == '2':
                    self.preview_family_count()
                elif choice == '3':
                    self._run_te_prep_interactive()
                elif choice == '4':
                    self.submit_batch_job()
                elif choice == '5':
                    self.download_selected_files()
                elif choice == '6':
                    self._run_te_enrichment_interactive()
                elif choice == '7':
                    self.check_job_status()
                elif choice == '8':
                    self.watch_job()
                elif choice == '9':
                    last_remote = self._find_last_remote_dir()
                    remote_prompt = (
                        f"Remote results directory [{last_remote}]: "
                        if last_remote else "Remote results directory: "
                    )
                    remote_path = input(remote_prompt).strip() or last_remote

                    if self.local_output_dir:
                        default_local = str(self.local_output_dir)
                        local_dir = input(f"Local output directory [{default_local}]: ").strip() or default_local
                    else:
                        local_dir = input("Local output directory (e.g., ~/Documents/output): ").strip()
                    if local_dir:
                        self.retrieve_results(local_dir, remote_out_override=remote_path or None)
                elif choice == '10':
                    local_dir = input("Enter local directory for error logs [./hpc_error_logs]: ").strip()
                    self.download_error_logs(local_dir if local_dir else None)
                elif choice == '11':
                    break
                elif choice == '12':
                    self.test_email_interactive()
                else:
                    print("Invalid option")

            except KeyboardInterrupt:
                print("\n\nInterrupted by user.")
                confirm = input("Exit? (y/n): ").strip().lower()
                if confirm == 'y':
                    break
            except Exception as e:
                print(f"\nError: {e}")
                print("Returning to main menu...")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive HPC client for TE analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hpc_client.py
  python hpc_client.py --host cluster.university.edu --user myusername
  python hpc_client.py -H cluster.edu -u myuser -o ./results
        """
    )
    parser.add_argument("-H", "--host", help="HPC hostname")
    parser.add_argument("-p", "--port", type=int, default=22, help="SSH port (default: 22)")
    parser.add_argument("-u", "--user", help="Username")
    parser.add_argument("-o", "--output", help="Local output directory for results")
    parser.add_argument(
        "--setup-email", action="store_true",
        help="Run one-time Gmail OAuth2 setup for email notifications, then exit",
    )
    args = parser.parse_args()

    print("="*60)
    print("  HPC TE ANALYSIS CLIENT")
    print("  Interactive client for running TE analysis on HPC")
    print("="*60)

    client = HPCClient()

    if args.setup_email:
        client.setup_email_auth()
        sys.exit(0)

    state = client._state

    # Get connection details (use args > cached state > prompt)
    print("\nEnter HPC connection details:")

    hostname = args.host
    if not hostname:
        cached_host = state.get("last_hostname", "")
        prompt = f"  Hostname [{cached_host}]: " if cached_host else "  Hostname: "
        hostname = input(prompt).strip() or cached_host
    else:
        print(f"  Hostname: {hostname}")

    if not hostname:
        print("Hostname is required")
        sys.exit(1)

    port = args.port
    if port == 22:
        port_str = input("  Port [22]: ").strip()
        port = int(port_str) if port_str else 22

    username = args.user
    if not username:
        cached_user = state.get("last_username", "")
        prompt = f"  Username [{cached_user}]: " if cached_user else "  Username: "
        username = input(prompt).strip() or cached_user
    else:
        print(f"  Username: {username}")

    if not username:
        print("Username is required")
        sys.exit(1)

    password = getpass.getpass("  Password: ")

    # Connect
    if not client.connect(hostname, username, password, port):
        print("Failed to connect. Exiting.")
        sys.exit(1)

    # Persist connection details for next session
    client._state["last_hostname"] = hostname
    client._state["last_username"] = username
    client._save_state()

    # Store output directory if provided
    if args.output:
        client.local_output_dir = Path(args.output)
        print(f"Results will be saved to: {args.output}")

    try:
        # Show main menu
        client.main_menu()
    finally:
        client.disconnect()

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
