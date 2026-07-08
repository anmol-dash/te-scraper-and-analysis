#!/usr/bin/env python3
_SCRIPT_BUILD = "20260610-1"  # bump this when changing job script logic
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
import threading
import zipfile
from pathlib import Path

_STATE_FILE = Path.home() / ".hpc_te_state.json"
_GHCR_IMAGE = "ghcr.io/anmol-dash/gameca:latest"  # published by .github/workflows/docker-image.yml

try:
    import paramiko
except ImportError:
    print("Error: paramiko is required. Install with: pip install paramiko")
    sys.exit(1)


def _log(message: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def _remote_path(work_dir: str, sub: str) -> str:
    """Join work_dir with sub, but treat sub as absolute if it starts with '/'."""
    sub = sub.strip()
    return sub if sub.startswith("/") else f"{work_dir}/{sub}"


class HPCClient:
    """Interactive client for running TE analysis on HPC cluster via batch jobs."""

    def __init__(self):
        self.ssh = None
        self.sftp = None
        self.connected = False
        self.scheduler = None       # 'lsf' or 'slurm', auto-detected on connect
        self._scheduler_live = False  # True only if the daemon responds
        self._python = "python3"    # remote python binary, detected on connect
        self._current_job_info_path = None  # info file for the most recently submitted job
        self.has_internet = False   # whether the connected node can reach UCSC/Dfam

        # Default parameters
        self.params = {
            "FAMILY_NAME": "MT2_Mm",
            "SPECIES": "mouse",
            "ASSEMBLY": "mm10",      # UCSC assembly/build used for automatic loading
            "SOURCE_DB": "rmsk",     # Coordinate source: rmsk (RepeatMasker) or dfam
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
            # Singularity/Apptainer container mode. When CONTAINER_SIF is set to an
            # absolute path of gameca.sif on the cluster, jobs run every `python`
            # call through `singularity exec <binds> gameca.sif python ...` instead
            # of building a venv — no glibc/import issues, nothing to pip-install.
            # Left blank here; connect() auto-populates it via _ensure_container_sif()
            # (local ./gameca.sif upload, or a docker://ghcr build on the cluster)
            # unless the user has already set it explicitly.
            "CONTAINER_SIF": "",
            # Extra bind mounts, space-separated "src:dst" (or "src") entries. The
            # work dir, output dir and genome dir are bound automatically.
            "CONTAINER_BINDS": "",
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
        self._hostname = None
        self._port = 22
        self._key_path = ""
        self.use_sftp = False

        self._state = self._load_state()
        self.local_output_dir = (
            Path(self._state["last_local_dir"])
            if self._state.get("last_local_dir")
            else None
        )

    def connect(self, hostname: str, username: str, password: str = "", port: int = 22,
                work_dir: str = "", key_path: str = ""):
        """Connect to the HPC cluster via SSH."""
        print(f"\nConnecting to {hostname}...")

        # Store credentials for reconnection
        self._password = password
        self._username = username
        self._hostname = hostname
        self._port = port
        self._key_path = key_path
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

            # Key-file auth (highest priority when a key is supplied)
            if key_path:
                _key = None
                _key_path_expanded = os.path.expanduser(key_path)
                for _KeyClass in (
                    paramiko.Ed25519Key,
                    paramiko.RSAKey,
                    paramiko.ECDSAKey,
                ):
                    try:
                        _key = _KeyClass.from_private_key_file(_key_path_expanded)
                        break
                    except (paramiko.ssh_exception.SSHException, ValueError):
                        continue
                if _key is None:
                    print(f"  Warning: could not load key {key_path!r} — trying password auth")
                else:
                    self._transport.auth_publickey(username, _key)
                    if self._transport.is_authenticated():
                        print(f"  Key authentication OK ({type(_key).__name__})")

            if not self._transport.is_authenticated():
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
                print("Scheduler detected: NONE — jobs will run inside tmux sessions on this host")

            # Check whether the scheduler daemon is actually reachable
            self._scheduler_live = self._check_scheduler_live()
            if self.scheduler and not self._scheduler_live:
                print(f"  Warning: {self.scheduler.upper()} binaries found but daemon is DOWN — will use nohup fallback")

            # Auto-detect python binary
            self._python = self._detect_python()

            # Check whether nextflow is available for --nextflow /
            # --post-alignment-analyses-nextflow (query.py falls back to the
            # in-process pipeline if it isn't, but warn up front).
            self.has_nextflow = self._detect_nextflow()
            if self.has_nextflow:
                print("Nextflow detected — --nextflow / --post-alignment-analyses-nextflow available")
            else:
                print("Nextflow: NOT found on PATH — --nextflow will fall back to the in-process pipeline")

            # Check whether this node can reach the internet (login nodes usually can,
            # compute nodes usually cannot).  Uses curl to avoid Python DNS quirks.
            print("Checking internet connectivity (UCSC/Dfam)...")
            self.has_internet = self._check_internet()
            if self.has_internet:
                print("  Internet: reachable — UCSC API fetch will work on this node")
            else:
                print("  Internet: NOT reachable from this node")
                print("  Sequence fetching will require a local genome FASTA (--genome).")

            # Auto-provision gameca.sif so CONTAINER_SIF doesn't have to be set by
            # hand. Never let this block a successful connection.
            try:
                self._ensure_container_sif()
            except Exception as e:
                print(f"[GAMECA] Container auto-setup skipped ({e}); "
                      "falling back to the venv path.")

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
for target in "/project/${USER:-__USER__}/gameca" "/scratch/${USER:-__USER__}/gameca" "/work/${USER:-__USER__}/gameca"; do
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
        out, err, code = self.run_command(probe)
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
        self.run_command(f"mkdir -p {shlex.quote(fallback)}")
        return fallback

    def _try_remote_work_dir(self, candidate: str) -> bool:
        if not candidate or candidate == "/":
            return False
        out, err, code = self.run_command(
            f"mkdir -p {shlex.quote(candidate)} && [ -d {shlex.quote(candidate)} ] && [ -w {shlex.quote(candidate)} ] && echo OK",
        )
        if code == 0 and "OK" in out:
            return True
        _log(f"Rejected remote work dir {candidate!r}: exit={code} stdout={out.strip()!r} stderr={err.strip()!r}")
        return False

    def _batch_work_dir_candidates(self) -> list[str]:
        user = self._username or "$USER"
        candidates = [
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
        """Close SSH connection, force-killing the socket if it hangs (e.g. WiFi dropped)."""
        self.connected = False  # block new commands immediately

        sftp      = self.sftp
        transport = self._transport
        self.sftp        = None
        self._transport  = None

        def _close():
            for obj in (sftp, transport):
                try:
                    if obj:
                        obj.close()
                except Exception:
                    pass

        t = threading.Thread(target=_close, daemon=True)
        t.start()
        t.join(timeout=3)
        if t.is_alive():
            # Graceful close is hung — nuke the underlying TCP socket directly
            try:
                transport.sock.close()
            except Exception:
                pass
        print("Disconnected from HPC.", flush=True)

    def _ensure_connected(self) -> bool:
        """Check if the SSH transport is alive; silently reconnect if not."""
        if self._transport and self._transport.is_active():
            return True
        if not self._hostname or not self._username:
            return False
        _log("SSH transport lost — attempting reconnect")
        print("SSH connection dropped — reconnecting...", flush=True)
        ok = self.connect(
            self._hostname, self._username,
            password=self._password or "",
            port=self._port,
            work_dir=self.remote_work_dir or "",
            key_path=self._key_path or "",
        )
        if ok:
            print("Reconnected.", flush=True)
        else:
            print("Reconnect failed — please restart and re-enter credentials.", flush=True)
        return ok

    def _scp_get(self, remote_path: str, local_path: str, timeout: int = 60) -> bool:
        """Download a remote file as fast as possible.

        Tier 1 — local scp subprocess with SSH_ASKPASS (OpenSSH, no sshpass needed).
        Tier 2 — SCP protocol over the existing paramiko channel (no re-auth).
        Falls back to False so the caller can try SFTP / base64 as a last resort.
        """
        if self._scp_get_local(remote_path, local_path, timeout):
            return True
        return self._scp_get_paramiko(remote_path, local_path, timeout)

    def _scp_get_local(self, remote_path: str, local_path: str, timeout: int) -> bool:
        """Run the local scp binary, injecting password via SSH_ASKPASS."""
        import os as _os, stat as _stat, subprocess as _subprocess, tempfile as _tempfile

        port = getattr(self, "_port", 22) or 22
        fd, askpass = _tempfile.mkstemp(suffix=".sh")
        try:
            with _os.fdopen(fd, "w") as f:
                f.write(f"#!/bin/sh\necho {shlex.quote(self._password or '')}\n")
            _os.chmod(askpass, _stat.S_IRWXU)

            env = _os.environ.copy()
            env["SSH_ASKPASS"]         = askpass
            env["SSH_ASKPASS_REQUIRE"] = "force"   # OpenSSH ≥ 8.4
            env["DISPLAY"]             = ":0"      # older OpenSSH fallback

            cmd = [
                "scp", "-O",
                "-o", "StrictHostKeyChecking=no",
                "-o", "BatchMode=no",
                "-P", str(port),
                f"{self._username}@{self._hostname}:{remote_path}",
                str(local_path),
            ]
            result = _subprocess.run(
                cmd, env=env, capture_output=True,
                stdin=_subprocess.DEVNULL,
            )
            if result.returncode == 0:
                return True
            _log(f"local scp rc={result.returncode}: {result.stderr.decode(errors='replace')[:200]}")
            return False
        except Exception as exc:
            _log(f"local scp error: {exc}")
            return False
        finally:
            try:
                _os.unlink(askpass)
            except Exception:
                pass

    def _scp_get_paramiko(self, remote_path: str, local_path: str, timeout: int) -> bool:
        """SCP receive protocol over the existing authenticated paramiko transport.

        Runs 'scp -f <path>' on the server via exec_command — no password needed
        because it reuses the already-established SSH session.
        """
        try:
            channel = self._transport.open_session()
            channel.exec_command(f"scp -f {shlex.quote(remote_path)}")

            channel.sendall(b"\x00")  # signal: ready

            # Read file header: "C<mode> <size> <name>\n"
            header = bytearray()
            while True:
                b = channel.recv(1)
                if not b or b == b"\n":
                    break
                header.extend(b)

            if not header or header[0:1] != b"C":
                channel.close()
                return False

            _, size_str, _ = header.decode().split(" ", 2)
            file_size = int(size_str)

            channel.sendall(b"\x00")  # ready to receive data

            received = 0
            with open(local_path, "wb") as fh:
                while received < file_size:
                    chunk = channel.recv(min(65536, file_size - received))
                    if not chunk:
                        break
                    fh.write(chunk)
                    received += len(chunk)

            channel.recv(1)           # trailing null from server
            channel.sendall(b"\x00")  # ack
            channel.close()

            ok = received == file_size
            if not ok:
                _log(f"paramiko scp incomplete: got {received}/{file_size}")
            return ok
        except Exception as exc:
            _log(f"paramiko scp error: {exc}")
            return False

    def _detect_scheduler(self):
        """Auto-detect LSF (bsub) or Slurm (sbatch) on the remote host."""
        for sched, cmd in [("lsf", "command -v bsub"), ("slurm", "command -v sbatch")]:
            out, _, code = self.run_command(cmd)
            if code == 0 and out.strip():
                return sched
        return None

    def _detect_python(self):
        """Return the name of the available python binary on the remote host."""
        for binary in ("python3", "python"):
            out, _, code = self.run_command(f"command -v {binary}")
            if code == 0 and out.strip():
                return binary
        return "python3"  # best guess if neither found

    def _detect_nextflow(self) -> bool:
        """Check whether nextflow (and the java it needs) is on the remote PATH.

        query.py's --nextflow / --post-alignment-analyses-nextflow flags shell
        out to `nextflow run ...` from inside the submitted job; if it's
        missing there, query.py itself falls back to the in-process pipeline,
        but surfacing that here lets the UI warn before a job is even
        submitted rather than only after it's deep into a running bsub job.
        """
        out, _, code = self.run_command("command -v nextflow && command -v java")
        return code == 0 and out.strip() != ""

    def _check_internet(self) -> bool:
        """Test whether this node can reach UCSC via curl (more reliable than Python DNS).

        Uses curl because Python's socket.getaddrinfo can fail on some HPC nodes
        even when curl resolves names correctly (different libc/nsswitch paths).
        Returns True if the node has outbound HTTPS access.
        """
        out, _, code = self.run_command(
            "curl -s -o /dev/null "
            "-w '%{http_code}' 'https://api.genome.ucsc.edu/' 2>/dev/null || echo FAIL",
        )
        return code == 0 and out.strip() not in ("", "FAIL", "000")

    def ensure_venv_cmd(self) -> str:
        """Return a shell block that creates/activates the venv and installs deps.

        Venv lives in remote_work_dir (not $HOME) to avoid the HOME=/tmp override
        set by _remote_exec_command. Hash-gated so pip only runs when
        requirements.txt changes.
        """
        venv_dir = f"{self.remote_work_dir}/venv"
        req_file = f"{self.remote_work_dir}/requirements.txt"
        return f"""
GAMECA_VENV="{venv_dir}"
GAMECA_PY="{self._python}"
GAMECA_REQ="{req_file}"
mkdir -p "$(dirname "$GAMECA_VENV")"
# Recreate the venv if it predates --system-site-packages (or is broken). The
# inherited site-packages let us reuse the cluster conda env's prebuilt numpy/
# pandas/scipy/matplotlib instead of compiling them from source.
if [ -x "$GAMECA_VENV/bin/python" ] && ! grep -qi "include-system-site-packages = true" "$GAMECA_VENV/pyvenv.cfg" 2>/dev/null; then
    echo "[GAMECA] Rebuilding venv with --system-site-packages ..."
    rm -rf "$GAMECA_VENV"
fi
if [ ! -x "$GAMECA_VENV/bin/python" ]; then
    echo "[GAMECA] Creating virtualenv (--system-site-packages) at $GAMECA_VENV ..."
    "$GAMECA_PY" -m venv --system-site-packages "$GAMECA_VENV" || {{
        echo "[GAMECA] venv creation failed — trying to install python3-venv ..."
        apt-get install -y python3-venv python3-pip >/dev/null 2>&1 || true
        "$GAMECA_PY" -m venv --system-site-packages "$GAMECA_VENV" || {{ echo "[GAMECA] ERROR: cannot create venv"; exit 1; }}
    }}
fi
source "$GAMECA_VENV/bin/activate" || {{ echo "[GAMECA] ERROR: cannot activate venv at $GAMECA_VENV"; exit 1; }}
if [ -f "$GAMECA_REQ" ]; then
    _req_hash=$(md5sum "$GAMECA_REQ" 2>/dev/null | cut -d' ' -f1 || sha1sum "$GAMECA_REQ" 2>/dev/null | cut -d' ' -f1 || echo none)
    _last_hash=$(cat "$GAMECA_VENV/.req_hash" 2>/dev/null || echo "")
    if [ "$_req_hash" != "$_last_hash" ]; then
        echo "[GAMECA] Installing dependencies (requirements changed) ..."
        python -m pip install --quiet --upgrade pip setuptools wheel || true
        # Install each requirement INDEPENDENTLY, preferring prebuilt wheels.
        # Rationale for old-glibc HPC nodes:
        #   * a single un-installable package (pybigtools = manylinux_2_28 only,
        #     or the huge colabfold stack) must NOT abort the whole install and
        #     take scikit-learn etc. down with it (pip -r is all-or-nothing);
        #   * --prefer-binary avoids source builds; packages already provided by
        #     the conda base env are detected as satisfied and skipped.
        _failed=""
        while IFS= read -r _line || [ -n "$_line" ]; do
            _pkg="${{_line%%#*}}"
            _pkg="$(echo $_pkg | xargs)"
            [ -z "$_pkg" ] && continue
            if ! python -m pip install --prefer-binary "$_pkg" > /tmp/gameca_pip.$$ 2>&1; then
                echo "[GAMECA] WARN: could not install '$_pkg' (skipping):"
                tail -3 /tmp/gameca_pip.$$ | sed 's/^/[GAMECA]     /'
                _failed="$_failed $_pkg"
            fi
        done < "$GAMECA_REQ"
        rm -f /tmp/gameca_pip.$$
        echo "$_req_hash" > "$GAMECA_VENV/.req_hash"
        if [ -n "$_failed" ]; then
            echo "[GAMECA] Dependencies installed (skipped:$_failed )"
        else
            echo "[GAMECA] Dependencies installed."
        fi
    fi
else
    echo "[GAMECA] Warning: $GAMECA_REQ not found — skipping pip install"
fi
""".strip()

    def _remote_exec_command(self, command: str) -> str:
        """Run commands through a login shell so scheduler PATH/modules match ui.py sessions.
        HOME=/tmp prevents bash -l from failing when the NFS home dir is missing/unmounted."""
        return f"HOME=/tmp bash -lc {shlex.quote(command)}"

    def _job_script_header(self, job_name, job_out, job_err, mem_mb=None, cpus=None,
                            walltime=None, queue=None):
        """Return scheduler-specific directives for the job script."""
        if self.scheduler not in {"lsf", "slurm"}:
            # No scheduler — job runs inside a detached tmux session (see submit_batch_job
            # / submit_command_as_batch_job). The header is just an informational comment.
            return f"# (no job scheduler detected — {job_name} runs inside a tmux session)\n"
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
            # span[hosts=1] forces all N slots onto ONE host. The pipeline is a
            # single shared-memory process (threaded BLAS/MAFFT + a ~3GB genome
            # cache loaded once), so slots scattered across hosts would be paid
            # for but unusable and would over-subscribe the one host we land on.
            return (
                f"#BSUB -J {job_name}\n"
                f"#BSUB -o {job_out}\n"
                f"#BSUB -e {job_err}\n"
                f"#BSUB -n {cpus}\n"
                f'#BSUB -R "span[hosts=1]"\n'
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

    def _container_sif(self):
        """Return the configured gameca.sif path, or '' if container mode is off."""
        return str(self.params.get("CONTAINER_SIF", "") or "").strip()

    def _ensure_container_sif(self):
        """Auto-provision gameca.sif on connect so CONTAINER_SIF need not be set by hand.

        Precedence (skipped entirely if the user already set CONTAINER_SIF):
          1. A local ./gameca.sif (built via build_sif.sh) is uploaded via SFTP
             to a fixed path inside remote_work_dir, and reused across connects
             if the remote copy already matches its size.
          2. Otherwise, if singularity/apptainer is on the remote PATH, build
             gameca.sif there directly from the published GHCR image
             (see .github/workflows/docker-image.yml) and cache it at the same
             fixed path for future connects.
          3. If neither is possible, CONTAINER_SIF stays unset and jobs use the
             existing venv path exactly as before.
        """
        if self._container_sif():
            return  # user already configured it explicitly — don't override
        if not self.remote_work_dir:
            return

        remote_sif = f"{self.remote_work_dir}/gameca.sif"
        local_sif = Path("gameca.sif")

        if local_sif.exists():
            if self._upload_sif_if_stale(local_sif, remote_sif):
                self.params["CONTAINER_SIF"] = remote_sif
                print(f"[GAMECA] CONTAINER_SIF auto-set from local {local_sif} -> {remote_sif}")
            return

        out, _, code = self.run_command(
            "command -v singularity 2>/dev/null || command -v apptainer 2>/dev/null"
        )
        runtime = out.strip().splitlines()[0] if code == 0 and out.strip() else ""
        if not runtime:
            print("[GAMECA] No local gameca.sif and no singularity/apptainer on the "
                  "cluster PATH — container mode stays off (venv path used).")
            return

        out, _, code = self.run_command(f"test -f {shlex.quote(remote_sif)} && echo EXISTS")
        if code == 0 and "EXISTS" in out:
            self.params["CONTAINER_SIF"] = remote_sif
            print(f"[GAMECA] Reusing cached {remote_sif}")
            return

        print(f"[GAMECA] Building gameca.sif on the cluster from {_GHCR_IMAGE} "
              "(this can take several minutes)...")
        build_cmd = (
            f"cd {shlex.quote(self.remote_work_dir)} && "
            f"{shlex.quote(runtime)} build gameca.sif docker://{_GHCR_IMAGE}"
        )
        out, err, code = self.run_command(build_cmd, timeout=1800, stream_output="summary")
        if code == 0:
            self.params["CONTAINER_SIF"] = remote_sif
            print(f"[GAMECA] Built and cached {remote_sif}; CONTAINER_SIF set automatically.")
        else:
            print(f"[GAMECA] singularity build failed (exit {code}); container mode stays off.")
            if err.strip():
                print(err.strip()[-500:])

    def _upload_sif_if_stale(self, local_path: Path, remote_path: str) -> bool:
        """Upload local_path to remote_path via SFTP, skipping if sizes already match."""
        local_size = local_path.stat().st_size
        out, _, code = self.run_command(
            f"stat -c %s {shlex.quote(remote_path)} 2>/dev/null "
            f"|| stat -f %z {shlex.quote(remote_path)} 2>/dev/null"
        )
        remote_size = int(out.strip()) if code == 0 and out.strip().isdigit() else -1
        if remote_size == local_size:
            print(f"[GAMECA] Remote {remote_path} already matches local gameca.sif "
                  f"({local_size:,} bytes) — reusing.")
            return True

        if not (self.use_sftp and self.sftp):
            print("[GAMECA] SFTP unavailable — cannot upload local gameca.sif "
                  f"({local_size / 1e9:.1f} GB); leaving CONTAINER_SIF unset.")
            return False

        print(f"[GAMECA] Uploading local gameca.sif ({local_size / 1e9:.2f} GB) "
              f"to {remote_path} ...")
        self.run_command(f"mkdir -p {shlex.quote(str(Path(remote_path).parent))}")
        last_pct = [-1]

        def _cb(sent, total):
            pct = int(100 * sent / total) if total else 0
            if pct != last_pct[0] and pct % 10 == 0:
                last_pct[0] = pct
                print(f"[GAMECA]   upload {pct}%")

        try:
            self.sftp.put(str(local_path), remote_path, callback=_cb)
            return True
        except Exception as e:
            print(f"[GAMECA] Upload of gameca.sif failed: {e}")
            return False

    def _container_setup_block(self):
        """Return a shell block that makes every subsequent `python` call run
        inside gameca.sif via `singularity exec`.

        Rather than patch the four job-script templates, we define a bash function
        named `python` (and `python3`) that wraps `singularity exec <binds> sif
        python "$@"`. Existing `python -u query.py ...` call sites then transparently
        execute inside the container — where mafft/bedtools/liftOver and every
        Python dep already live, so no venv, pip, or module loads are needed.
        """
        sif = self._container_sif()
        # Auto-bind work dir, output dir and (parent of) the genome FASTA, plus any
        # user-supplied extras. Bind "src:src" so absolute paths match inside/out.
        bind_srcs = []
        for p in (self.remote_work_dir, self.remote_output_dir):
            if p:
                bind_srcs.append(str(p))
        genome = str(self.params.get("LOCAL_ASSEMBLY_PATH", "") or "").strip()
        if genome:
            bind_srcs.append(str(Path(genome).parent))
        for extra in str(self.params.get("CONTAINER_BINDS", "") or "").split():
            bind_srcs.append(extra)
        # De-dup while preserving order; format each as a -B argument.
        seen, binds = set(), []
        for b in bind_srcs:
            if b and b not in seen:
                seen.add(b)
                binds.append(f"-B {shlex.quote(b if ':' in b else f'{b}:{b}')}")
        binds_str = " ".join(binds)
        pwd_dir = self.remote_work_dir or "."
        return f'''# ── Container (Singularity/Apptainer) mode ──────────────────────────────────
# CONTAINER_SIF is set: run all Python through gameca.sif — no venv/pip/modules.
GAMECA_SIF={shlex.quote(sif)}
GAMECA_RT="$(command -v singularity 2>/dev/null || command -v apptainer 2>/dev/null || echo singularity)"
GAMECA_BINDS="{binds_str}"
if [ ! -f "$GAMECA_SIF" ]; then
    echo "[GAMECA] FATAL: CONTAINER_SIF not found: $GAMECA_SIF" >&2
    exit 1
fi
echo "[GAMECA] Container mode: $GAMECA_RT exec $GAMECA_SIF"
echo "[GAMECA] Binds: $GAMECA_BINDS"
# Route bare `python`/`python3` into the image. --pwd keeps the working dir; the
# bind-mounted work dir means the repo's own code is used (bake default, override).
python()  {{ "$GAMECA_RT" exec $GAMECA_BINDS --pwd {shlex.quote(pwd_dir)} "$GAMECA_SIF" python "$@"; }}
python3() {{ python "$@"; }}
export GAMECA_SIF GAMECA_RT GAMECA_BINDS
export -f python python3
echo "[GAMECA] Container python: $(python --version 2>&1)"
'''.strip()

    def _venv_setup_block(self):
        """Return robust virtualenv setup. Never exits — logs errors and continues.

        In container mode (CONTAINER_SIF set) this is replaced by a block that
        routes `python` through gameca.sif instead of building a venv.
        """
        if self._container_sif():
            return self._container_setup_block()
        return f'''# ── Host diagnostics ────────────────────────────────────────────────────────
echo "[GAMECA] build={_SCRIPT_BUILD}  host=$(hostname)  user=$(whoami)"
echo "[GAMECA] SLURM_JOB_ID=${{SLURM_JOB_ID:-<unset>}}  LSB_JOBID=${{LSB_JOBID:-<unset>}}"
echo "[GAMECA] PATH=$PATH"

# ── Virtual environment ──────────────────────────────────────────────────────
VENV_DIR="${{GAMECA_VENV:-$HOME/gameca_venv}}"
PYTHON_BIN="python3"
command -v python3 >/dev/null 2>&1 || PYTHON_BIN=python

# Validate existing venv
if [ -x "$VENV_DIR/bin/python" ]; then
    "$VENV_DIR/bin/python" -c "import sys" >/dev/null 2>&1 || {{
        echo "[GAMECA] Existing venv broken — removing $VENV_DIR"
        rm -rf "$VENV_DIR"
    }}
fi

# Create venv if needed
USING_VENV=false
if [ ! -d "$VENV_DIR" ]; then
    echo "[GAMECA] Creating venv at $VENV_DIR ..."
    if "$PYTHON_BIN" -m venv "$VENV_DIR" 2>&1; then
        echo "[GAMECA] venv created OK"
        USING_VENV=true
    else
        echo "[GAMECA] WARNING: python3 -m venv failed — trying python3-venv install ..."
        apt-get install -y python3-venv python3-pip 2>&1 || true
        if "$PYTHON_BIN" -m venv "$VENV_DIR" 2>&1; then
            echo "[GAMECA] venv created OK after apt-get"
            USING_VENV=true
        else
            echo "[GAMECA] WARNING: venv unavailable — using system Python (packages may be missing)"
        fi
    fi
else
    USING_VENV=true
fi

# Activate
if $USING_VENV && [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
    echo "[GAMECA] venv activated: $(python --version 2>&1)"
else
    echo "[GAMECA] Using system Python: $("$PYTHON_BIN" --version 2>&1)"
fi

# Thread counts
THREADS="{self.params['CPUS']}"
[ -n "${{SLURM_CPUS_PER_TASK:-}}" ] && THREADS="$SLURM_CPUS_PER_TASK"
[ -n "${{LSB_DJOB_NUMPROC:-}}"     ] && THREADS="$LSB_DJOB_NUMPROC"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
export NUMBA_NUM_THREADS="$THREADS" VECLIB_MAXIMUM_THREADS="$THREADS"
export NUMBA_CACHE_DIR="${{TMPDIR:-/tmp}}/gameca_numba_$USER"
mkdir -p "$NUMBA_CACHE_DIR"

# Install requirements
REQ_FILE="{self.remote_work_dir}/requirements.txt"
if [ -f "$REQ_FILE" ]; then
    echo "[GAMECA] Installing requirements (this may take a few minutes on first run) ..."
    PIP_START=$SECONDS
    python -m pip install --upgrade pip setuptools wheel 2>&1 | tail -5 || true
    python -m pip install --prefer-binary -r "$REQ_FILE" 2>&1 | tail -20 || {{
        echo "[GAMECA] WARNING: pip install had errors — packages may be partially installed"
    }}
    echo "[GAMECA] pip done in $((SECONDS - PIP_START))s"
else
    echo "[GAMECA] WARNING: requirements.txt not found at $REQ_FILE"
fi

echo "[GAMECA] Python: $(python --version 2>&1)  Pip: $(python -m pip --version 2>&1 | cut -d' ' -f1-2)"
'''

    def _mafft_setup_block(self):
        """Return MAFFT setup that does not fail on broken conda installations.

        In container mode mafft is baked into gameca.sif, so host-side install is
        unnecessary (and would run outside the image anyway)."""
        if self._container_sif():
            return "# MAFFT provided by gameca.sif (container mode) — no host install needed"
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

    def _tmux_launch_cmd(self, session_name, job_script, job_out, job_err):
        """Return the command to launch a job script inside a detached tmux session.

        Used as the no-scheduler execution mode (e.g. plain SSH cloud VMs without
        LSF/Slurm) — the session keeps the job running across SSH disconnects and
        can be inspected live via `tmux capture-pane` / `tmux attach`.
        """
        return (
            f"tmux new-session -d -s {shlex.quote(session_name)} "
            f"{shlex.quote(f'bash {job_script} >{job_out} 2>{job_err}')}"
        )

    def _has_tmux(self) -> bool:
        """Return True if the `tmux` binary is available on the remote host."""
        _, _, code = self.run_command("command -v tmux >/dev/null 2>&1")
        return code == 0

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
        if self.scheduler == "lsf":
            return f"bjobs {job_id} 2>&1"
        return f"tmux has-session -t {shlex.quote(str(job_id))} 2>&1 && echo RUNNING || echo GONE"

    def _cancel_job_cmd(self, job_id):
        """Return command to cancel a job."""
        if self.scheduler == "slurm":
            return f"scancel {job_id}"
        if self.scheduler == "lsf":
            return f"bkill {job_id}"
        return f"tmux kill-session -t {shlex.quote(str(job_id))} 2>&1"

    def _check_scheduler_live(self) -> bool:
        """Return True only if the scheduler daemon is actually reachable."""
        if self.scheduler == "slurm":
            out, _, code = self.run_command("scontrol ping 2>&1")
            return code == 0 and "UP" in out
        if self.scheduler == "lsf":
            _, _, code = self.run_command("bjobs -h 2>&1")
            return code == 0
        return False

    def _detect_slurm_partition(self) -> str:
        """Return the first available SLURM partition, falling back to 'normal'."""
        out, _, code = self.run_command(
            "sinfo --noheader -o '%P' 2>/dev/null | tr -d '*' | head -1", timeout=10
        )
        part = out.strip().splitlines()[0].strip() if code == 0 and out.strip() else ""
        return part or "normal"

    def _interactive_alloc_cmd(self, runner_script, mem_mb, cpus, queue):
        """Return the command to allocate a node and run interactively."""
        if self.scheduler == "slurm":
            mem_gb = max(1, mem_mb // 1000)
            return (
                f"srun --mem={mem_gb}G --cpus-per-task={cpus} "
                f"--partition={queue} --pty bash {runner_script}"
            )
        # span[hosts=1]: keep all slots on one host (shared-memory single process).
        return (f'bsub -M {mem_mb} -n {cpus} -R "span[hosts=1]" '
                f"-q {queue} -Is bash {runner_script}")

    def run_command(self, command: str, timeout: int = 300, stream_output: bool = False, cancel_event=None) -> tuple:
        """Execute a command on the remote server.

        Args:
            command: The command to execute
            timeout: Timeout in seconds
            stream_output: If True, print raw output in real-time. If "summary",
                print only stage/error/status lines plus a heartbeat.
        """
        if not self.connected:
            raise RuntimeError("Not connected to HPC")

        if not self._ensure_connected():
            raise RuntimeError("SSH session not active and reconnect failed")

        summary_stream = stream_output == "summary"
        raw_stream = bool(stream_output) and not summary_stream
        _log(f"Remote command start (timeout={timeout}s, stream={stream_output}): {command[:240]}")
        channel = self._transport.open_session()
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

            if cancel_event is not None and cancel_event.is_set():
                channel.close()
                raise InterruptedError("run_command cancelled by user")

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
        out, err, code = self.run_command(command, stream_output=False)
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
            "31": "SOURCE_DB",
            "32": "CONTAINER_SIF",
            "33": "CONTAINER_BINDS",
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
                        full = _remote_path(self.remote_work_dir, base) + f"/{family.lower()}"
                    elif base:
                        full = _remote_path(self.remote_work_dir, base) + "/<family>"
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
            print("    [31] SOURCE_DB: coordinate source — rmsk (RepeatMasker) or dfam")
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
            print("    [27] NOTIFY_EMAIL: Email address for job-complete notification (sent via Resend — no setup needed)")
            print("    [28] SKIP_JASPAR: 1 = skip JASPAR motif/TFBS analysis")
            print("    [29] SKIP_PRIMERS: 1 = skip primer design")
            print("    [30] SKIP_GO: 1 = skip GO enrichment")
            print("    [32] CONTAINER_SIF: absolute path to gameca.sif on the cluster. When set, jobs")
            print("         run every python step inside the container (no venv/pip/glibc issues).")
            print("         Auto-provisioned on connect: uploads a local ./gameca.sif if present, else")
            print("         builds one on the cluster from ghcr.io/anmol-dash/gameca:latest via")
            print("         singularity/apptainer. Set this manually to override or force a specific path.")
            print("    [33] CONTAINER_BINDS: extra Singularity bind mounts, space-separated src[:dst]")
            print("         (work dir, output dir and genome dir are bound automatically).")
            print()
            print("  [g]  Send a test email  (Resend via cluster — confirm notifications work)")
            print("  [p]  Preview family count")
            print("  [r]  Run analysis")
            print("  [q]  Back")
            print("="*65)

            choice = input("\nSelect option: ").strip().lower()

            if choice == "q":
                return False
            elif choice == "g":
                self.submit_email_diagnostic_job()
            elif choice == "p":
                self.preview_family_count()
            elif choice == "r":
                return True
            elif choice in str_params:
                key = str_params[choice]
                cur = self.params.get(key, "")
                val = input(f"  {key} [{cur}]: ").strip()
                if val:
                    if key == "SOURCE_DB" and val not in {"rmsk", "dfam"}:
                        print(f"  Invalid source '{val}' — must be rmsk or dfam")
                        continue
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
                "--source",   self.params.get("SOURCE_DB", "rmsk"),
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

        # ── Stage 11 / standout analysis module options ────────────────────
        def _listval(key):
            v = self.params.get(key)
            if v is None:
                return []
            if isinstance(v, (list, tuple)):
                return [str(x).strip() for x in v if str(x).strip()]
            return [tok for tok in str(v).replace(",", " ").split() if tok]

        def _strval(key):
            v = self.params.get(key)
            v = "" if v is None else str(v).strip()
            return v

        target_assemblies = _listval("TARGET_ASSEMBLIES")
        if target_assemblies:
            args += ["--target-assemblies"] + target_assemblies

        ortholog_species = _listval("ORTHOLOG_SPECIES")
        if ortholog_species:
            args += ["--ortholog-species"] + ortholog_species

        liftover_cmd = _strval("LIFTOVER_CMD")
        if liftover_cmd:
            args += ["--liftover-cmd", liftover_cmd]

        epigenetic_preset = _strval("EPIGENETIC_PRESET")
        if epigenetic_preset:
            args += ["--epigenetic-preset", epigenetic_preset]

        ctcf_preset = _strval("CTCF_PRESET")
        if ctcf_preset:
            args += ["--ctcf-preset", ctcf_preset]

        tads_preset = _strval("TADS_PRESET")
        if tads_preset:
            args += ["--tads-preset", tads_preset]

        grna_cas = _strval("GRNA_CAS")
        if grna_cas:
            args += ["--grna-cas", grna_cas]

        grna_max_mm = self.params.get("GRNA_MAX_MM")
        if grna_max_mm is not None and str(grna_max_mm).strip() != "":
            args += ["--grna-max-mm", str(grna_max_mm)]

        grna_background = _strval("GRNA_BACKGROUND")
        if grna_background:
            args += ["--grna-background", grna_background]

        colabfold_cmd = _strval("COLABFOLD_CMD")
        if colabfold_cmd:
            args += ["--colabfold-cmd", colabfold_cmd]

        subst_rate = _strval("SUBST_RATE")
        if subst_rate:
            args += ["--subst-rate", subst_rate]

        clock_divisor = _strval("CLOCK_DIVISOR")
        if clock_divisor:
            args += ["--clock-divisor", clock_divisor]

        intact_orf_aa = self.params.get("INTACT_ORF_AA")
        if intact_orf_aa is not None and str(intact_orf_aa).strip() != "":
            args += ["--intact-orf-aa", str(intact_orf_aa)]

        min_ltr_identity = _strval("MIN_LTR_IDENTITY")
        if min_ltr_identity:
            args += ["--min-ltr-identity", min_ltr_identity]

        tail_bp = self.params.get("TAIL_BP")
        if tail_bp is not None and str(tail_bp).strip() != "":
            args += ["--tail-bp", str(tail_bp)]

        promoter_bp = self.params.get("PROMOTER_BP")
        if promoter_bp is not None and str(promoter_bp).strip() != "":
            args += ["--promoter-bp", str(promoter_bp)]

        cpg_omega = _strval("CPG_OMEGA")
        if cpg_omega:
            args += ["--cpg-omega", cpg_omega]

        mafft_cmd = _strval("MAFFT_CMD")
        if mafft_cmd:
            args += ["--mafft-cmd", mafft_cmd]

        args += ["--force"]  # always re-run cached stages in batch mode

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
                # SFTP failure often kills the transport — reconnect before shell fallback
                self._ensure_connected()

        encoded_full = base64.b64encode(content.encode()).decode()
        chunk_size = 65000
        chunks = [encoded_full[i:i+chunk_size] for i in range(0, len(encoded_full), chunk_size)] or [""]
        _log(f"Uploading {label} via shell base64 ({len(content):,} bytes, {len(chunks)} chunk(s))")

        parent = remote_path.rsplit("/", 1)[0] if "/" in remote_path else "."
        remote_q = shlex.quote(remote_path)
        self.run_command(f"mkdir -p {shlex.quote(parent)}")
        cmd = f"printf %s {shlex.quote(chunks[0])} | base64 -d > {remote_q}"
        out, err, code = self.run_command(cmd)
        if code != 0:
            print(f"Failed to upload {label} (chunk 1): {err}")
            return False

        for i, chunk in enumerate(chunks[1:], 2):
            cmd = f"printf %s {shlex.quote(chunk)} | base64 -d >> {remote_q}"
            out, err, code = self.run_command(cmd)
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
            "te_notify.py",
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

    def submit_batch_job(self, cancel_event=None):
        """Submit the analysis as a batch job on HPC. Returns immediately after submission."""
        self._log_run_configuration("Batch")
        if cancel_event is not None and cancel_event.is_set():
            return False
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
        if cancel_event is not None and cancel_event.is_set():
            return False
        if not self.upload_script():
            return False

        # Set up output directory on HPC
        family = self.params["FAMILY_NAME"].lower()
        self.remote_output_dir = _remote_path(self.remote_work_dir, self.params['BASE_OUT_DIR']) + f"/{family}"

        # Pre-create remote output directory so logs land inside it
        self.run_command(f"mkdir -p {self.remote_output_dir}")

        # Create bsub job script
        import datetime as _dt
        _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        job_name = f"te_analysis_{self.params['FAMILY_NAME']}"
        job_script    = f"{self.remote_work_dir}/te_analysis_job_{_ts}.sh"
        job_done      = f"{self.remote_work_dir}/te_analysis_job_{_ts}.done"
        job_info      = f"{self.remote_work_dir}/te_analysis_job_{_ts}.info"
        job_out       = f"{self.remote_output_dir}/pipeline.out"
        job_err       = f"{self.remote_output_dir}/pipeline.err"
        job_error_log = f"{self.remote_output_dir}/pipeline_error.log"

        # Build the scheduler-agnostic job script
        sched_header = self._job_script_header(job_name, job_out, job_err)
        module_block = self._module_load_block()
        venv_block = self._venv_setup_block()
        mafft_block = self._mafft_setup_block()
        input_preflight = self._input_preflight_block(error_log=job_done)
        pipeline_args = self._pipeline_cli_args()
        _log(f"Creating batch job script at {job_script}")

        # Fire-and-forget completion email, sent from the compute node via the
        # Resend API over HTTPS (through the proxy). Body is built at run time so
        # it reflects the real exit status / timing.
        notify_to  = str(self.params.get("NOTIFY_EMAIL", "")).strip()
        notify_b64 = self._email_notify_b64() if notify_to else ""
        if notify_b64:
            fam = self.params.get("FAMILY_NAME", "")
            notify_section = (
                '\n# --- GAMECA completion email (Resend API over HTTPS/proxy) ---\n'
                'GAMECA_NOTIFY_PY="$(mktemp "${TMPDIR:-/tmp}/gameca_notify_XXXXXX.py")"\n'
                'chmod 600 "$GAMECA_NOTIFY_PY"\n'
                f'echo {shlex.quote(notify_b64)} | base64 -d > "$GAMECA_NOTIFY_PY"\n'
                'if [ "$EXIT_CODE" -eq 0 ]; then GAMECA_RESULT="SUCCEEDED"; else GAMECA_RESULT="FAILED (exit $EXIT_CODE)"; fi\n'
                'GAMECA_PMIN=$((PIPELINE_SECONDS/60)); GAMECA_PSEC=$((PIPELINE_SECONDS%60))\n'
                'GAMECA_WMIN=$((WALL_SECONDS/60)); GAMECA_WSEC=$((WALL_SECONDS%60))\n'
                f'export GAMECA_MAIL_TO={shlex.quote(notify_to)}\n'
                f'export GAMECA_MAIL_SUBJECT="GAMECA {fam}: pipeline $GAMECA_RESULT"\n'
                "GAMECA_MAIL_BODY=\"$(printf 'TE analysis for %s %s\\non host %s at %s.\\n\\n"
                "Output dir: %s\\nPipeline time: %sm %ss\\nWall time: %sm %ss\\n' "
                f'"{fam}" "$GAMECA_RESULT" "$(hostname)" "$(date)" "{self.remote_output_dir}" '
                '"$GAMECA_PMIN" "$GAMECA_PSEC" "$GAMECA_WMIN" "$GAMECA_WSEC")"\n'
                'export GAMECA_MAIL_BODY\n'
                f'{self._python} "$GAMECA_NOTIFY_PY" || echo "  (email notification step failed)"\n'
                'rm -f "$GAMECA_NOTIFY_PY"\n'
            )
        else:
            notify_section = ""

        bsub_script = f'''#!/bin/bash
{sched_header}
{module_block}

JOB_START_TIME=$(date +%s)

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
WALL_SECONDS=$(( $(date +%s) - JOB_START_TIME ))

echo ""
echo "=========================================================="
echo " Pipeline finished"
echo " Exit code:   $EXIT_CODE"
echo " Pipeline:    $((PIPELINE_SECONDS / 60))m $((PIPELINE_SECONDS % 60))s"
echo " Wall time:   $((WALL_SECONDS / 60))m $((WALL_SECONDS % 60))s"
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
{notify_section}
exit $EXIT_CODE
'''

        # Upload the job script
        print("\nCreating bsub job script...")
        create_script_cmd = f"cat > {job_script} << 'BSUB_SCRIPT_EOF'\n{bsub_script}\nBSUB_SCRIPT_EOF"
        out, err, code = self.run_command(create_script_cmd)
        if code != 0:
            print(f"Error creating job script: {err}")
            return False

        # Make executable
        self.run_command(f"chmod +x {job_script}")

        # Remove old output files if they exist
        self.run_command(f"rm -f {job_out} {job_err} {job_done} {job_info}")

        # Submit the job
        if cancel_event is not None and cancel_event.is_set():
            return False
        if self.scheduler in ("lsf", "slurm"):
            submit_cmd = self._submit_job_cmd(job_script)
            sched_label = self.scheduler.upper()
            print(f"\nSubmitting job via {sched_label}...")
            _log(f"Submitting scheduler command: {submit_cmd}")
            out, err, code = self.run_command(submit_cmd)
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
        else:
            # No scheduler — launch inside a detached tmux session so the job
            # survives SSH disconnects and can be monitored live.
            session_name = f"gameca_{family}_{_ts}"
            print("\nNo scheduler detected — launching via tmux...")
            if not self._has_tmux():
                print("Error submitting job: tmux is not available on this host.")
                self._diagnostic_command("tmux availability", "command -v tmux; echo PATH=$PATH", timeout=15)
                return False

            launch_cmd = self._tmux_launch_cmd(session_name, job_script, job_out, job_err)
            _log(f"Submitting tmux command: {launch_cmd}")
            out, err, code = self.run_command(launch_cmd)
            print("\n[HPC DIAG] tmux launch completed")
            print(f"[HPC DIAG] command: {launch_cmd}")
            print(f"[HPC DIAG] exit={code} stdout={len(out)}B stderr={len(err)}B")
            if err.strip():
                print("[HPC DIAG] tmux stderr:")
                for line in err.rstrip().splitlines():
                    print(f"[HPC DIAG]   {line}")

            verify_out, _, verify_code = self.run_command(
                f"tmux has-session -t {shlex.quote(session_name)} 2>&1", timeout=15)
            if code != 0 or verify_code != 0:
                print(f"Error submitting job: tmux session did not start ({err.strip() or verify_out.strip()})")
                self._diagnostic_command("tmux sessions", "tmux list-sessions 2>&1", timeout=15)
                return False

            job_id = session_name
            self.current_job_id = job_id
            print(f"Job submitted successfully! tmux session: {job_id}")
            self._diagnostic_command("tmux session immediately after submit", "tmux list-sessions 2>&1", timeout=15)

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
        self.run_command(f"echo '{job_info_content}' > {job_info}")
        self._current_job_info_path = job_info
        self._diagnostic_command(
            "Submitted file inventory",
            f"ls -lh {shlex.quote(self.remote_work_dir)} | sed -n '1,120p'; "
            f"echo '--- job info ---'; cat {shlex.quote(job_info)} 2>&1; "
            f"echo '--- job script head ---'; sed -n '1,220p' {shlex.quote(job_script)} 2>&1",
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
        sched = getattr(self, "scheduler", None)
        print("\nUseful HPC commands:")
        if sched == "slurm":
            print(f"  squeue -j {job_id}     # Check job status")
            print(f"  tail -f {job_out}       # View live output")
            print(f"  scancel {job_id}        # Cancel job")
        elif sched == "lsf":
            print(f"  bjobs {job_id}          # Check job status")
            print(f"  bpeek {job_id}          # View live output")
            print(f"  bkill {job_id}          # Cancel job")
        else:
            print(f"  tmux capture-pane -p -t {job_id}   # View live output")
            print(f"  tmux attach -t {job_id}            # Attach to the session")
            print(f"  tmux kill-session -t {job_id}      # Cancel job")
        print("=" * 60)

        return True

    def _submit_job_cmd_dep(self, job_script, dep_job_id=None):
        """Return the scheduler command to submit a job, optionally after dep_job_id completes.

        dep_job_id may be a single job-id string or a list/tuple of job-id strings for
        multi-dependency (job runs after ALL listed jobs complete successfully).
        """
        if isinstance(dep_job_id, (list, tuple)):
            dep_ids = [str(d) for d in dep_job_id if d]
        elif dep_job_id:
            dep_ids = [str(dep_job_id)]
        else:
            dep_ids = []

        if self.scheduler == "slurm":
            dep = f" --dependency=afterok:{':'.join(dep_ids)}" if dep_ids else ""
            return f"sbatch{dep} {job_script}"
        if self.scheduler == "lsf":
            if len(dep_ids) == 1:
                dep = f" -w 'done({dep_ids[0]})'"
            elif dep_ids:
                dep = " -w '" + " && ".join(f"done({d})" for d in dep_ids) + "'"
            else:
                dep = ""
            return f"bsub{dep} < {job_script}"
        raise RuntimeError("Scheduler was not detected.")

    def _submit_stage_job(self, stage_label, extra_pipeline_args, dep_job_id=None):
        """Build and submit a single-stage pipeline job. Returns job_id or None."""
        import datetime as _dt
        _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        family = self.params["FAMILY_NAME"].lower()
        job_name   = f"te_{family}_{stage_label}"
        job_script = f"{self.remote_work_dir}/{job_name}_{_ts}.sh"
        job_out    = f"{self.remote_work_dir}/{job_name}_{_ts}.out"
        job_err    = f"{self.remote_work_dir}/{job_name}_{_ts}.err"
        job_done   = f"{self.remote_work_dir}/{job_name}_{_ts}.done"

        sched_header   = self._job_script_header(job_name, job_out, job_err)
        module_block   = self._module_load_block()
        venv_block     = self._venv_setup_block()
        mafft_block    = self._mafft_setup_block()
        base_args      = self._pipeline_cli_args()
        full_args      = base_args + " " + " ".join(shlex.quote(a) for a in extra_pipeline_args)

        script = f"""#!/bin/bash
{sched_header}
{module_block}
{venv_block}
{mafft_block}

echo "Stage: {stage_label}"
echo "Date: $(date)"
echo "Host: $(hostname)"

{self._remote_exec_command(f'python -u {self.remote_script_path} {full_args}')}

EXIT_CODE=$?
echo $EXIT_CODE > {job_done}
exit $EXIT_CODE
"""
        create_cmd = f"cat > {shlex.quote(job_script)} << 'STAGE_SCRIPT_EOF'\n{script}\nSTAGE_SCRIPT_EOF"
        out, err, code = self.run_command(create_cmd)
        if code != 0:
            print(f"[Parallel] Error creating {stage_label} script: {err}")
            return None
        self.run_command(f"chmod +x {shlex.quote(job_script)}")

        submit_cmd = self._submit_job_cmd_dep(job_script, dep_job_id)
        print(f"[Parallel] Submitting {stage_label} job…", flush=True)
        out, err, code = self.run_command(submit_cmd)
        if code != 0:
            print(f"[Parallel] Error submitting {stage_label}: {err}")
            return None
        job_id = self._parse_job_id(out)
        if job_id:
            dep_note = f" (after {dep_job_id})" if dep_job_id else ""
            print(f"[Parallel]   → Job {job_id} [{stage_label}]{dep_note}", flush=True)
        return job_id

    def submit_parallel_pipeline(self, max_jobs=10):
        """Submit the pipeline as parallel LSF/Slurm jobs.

        Job 1:  sequences → clustering → dashboard  (--stop-after dashboard)
        Jobs 2-4 run in parallel once Job 1 completes (LSF/Slurm dependency):
          Job 2: alignment   (--resume-from alignment --stop-after alignment)
          Job 3: motif + go  (--resume-from motif --stop-after go)
          Job 4: primers     (--resume-from primers --stop-after primers)

        Returns list of {stage, job_id} dicts.
        """
        if not self._ensure_batch_shared_work_dir():
            print("[Parallel] Error: could not find a shared writable work directory.")
            return []
        if not self.upload_script():
            return []

        family = self.params["FAMILY_NAME"].lower()
        import datetime as _dt
        _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.remote_output_dir = _remote_path(self.remote_work_dir, self.params["BASE_OUT_DIR"]) + f"/{family}"

        job_infos = []
        running = 0

        # Job 1: seq → clustering → dashboard
        jid1 = self._submit_stage_job("stage1_clustering", ["--stop-after", "dashboard"])
        if not jid1:
            return []
        job_infos.append({"stage": "clustering", "job_id": jid1})
        self.current_job_id = jid1
        running += 1

        skip_alignment = int(self.params.get("SKIP_ALIGNMENT", 0))
        skip_motif     = int(self.params.get("SKIP_JASPAR", 0))
        skip_primers   = int(self.params.get("SKIP_PRIMERS", 0))

        # Jobs 2-4: all depend on Job 1
        if not skip_alignment and running < max_jobs:
            jid = self._submit_stage_job(
                "stage2_alignment",
                ["--resume-from", "alignment", "--stop-after", "alignment",
                 "--skip-motif", "--skip-primers"],
                dep_job_id=jid1,
            )
            if jid:
                job_infos.append({"stage": "alignment", "job_id": jid})
                running += 1

        if not skip_motif and running < max_jobs:
            jid = self._submit_stage_job(
                "stage3_motif",
                ["--resume-from", "motif", "--stop-after", "go",
                 "--skip-alignment", "--skip-primers"],
                dep_job_id=jid1,
            )
            if jid:
                job_infos.append({"stage": "motif+go", "job_id": jid})
                running += 1

        if not skip_primers and running < max_jobs:
            jid = self._submit_stage_job(
                "stage4_primers",
                ["--resume-from", "primers", "--stop-after", "primers",
                 "--skip-alignment", "--skip-motif"],
                dep_job_id=jid1,
            )
            if jid:
                job_infos.append({"stage": "primers", "job_id": jid})
                running += 1

        # Job 5: Stage 11 standout analysis — depends on ALL parallel jobs completing.
        # Collects job IDs from Jobs 2-4 (whichever were actually submitted).
        parallel_jids = [ji["job_id"] for ji in job_infos if ji["stage"] != "clustering"]
        if parallel_jids and running < max_jobs:
            # If no parallel jobs were submitted (all skipped), depend on Job 1.
            dep_ids = parallel_jids if parallel_jids else [jid1]
            jid = self._submit_stage_job(
                "stage5_standout",
                ["--resume-from", "standout"],
                dep_job_id=dep_ids,
            )
            if jid:
                job_infos.append({"stage": "standout", "job_id": jid})
                running += 1
        elif running < max_jobs:
            # All main stages skipped — run standout after clustering job
            jid = self._submit_stage_job(
                "stage5_standout",
                ["--resume-from", "standout"],
                dep_job_id=jid1,
            )
            if jid:
                job_infos.append({"stage": "standout", "job_id": jid})

        print(f"\n[Parallel] {len(job_infos)} jobs submitted.", flush=True)
        return job_infos

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
        self.remote_output_dir = _remote_path(self.remote_work_dir, self.params['BASE_OUT_DIR']) + f"/{family}"

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

JOB_START_TIME=$(date +%s)

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
WALL_SECONDS=$(( $(date +%s) - JOB_START_TIME ))

echo ""
echo "=========================================================="
echo " Pipeline finished"
echo " Exit code:   $EXIT_CODE"
echo " Pipeline:    $((PIPELINE_SECONDS / 60))m $((PIPELINE_SECONDS % 60))s"
echo " Wall time:   $((WALL_SECONDS / 60))m $((WALL_SECONDS % 60))s"
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
        out, err, code = self.run_command(create_cmd)
        if code != 0:
            print(f"Error creating runner script: {err}")
            return False
        self.run_command(f"chmod +x {runner_script}")

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
            out, err, code = self.run_command(bsub_cmd, stream_output="summary")
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

    def _write_job_info(self, info_path: str, job_id: str, out: str, err: str,
                        done: str, label: str = "", errlog: str = ""):
        """Write a key=value .info file readable by _find_latest_job_info."""
        import io as _io, datetime as _dt
        content = (
            f"JOB_ID={job_id}\n"
            f"LABEL={label}\n"
            f"LOG_OUT={out}\n"
            f"LOG_ERR={err}\n"
            f"LOG_DONE={done}\n"
            f"LOG_ERRLOG={errlog or err}\n"
            f"SUBMITTED={_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        written = False
        try:
            if self.sftp:
                self.sftp.putfo(_io.BytesIO(content.encode()), info_path)
                written = True
        except Exception:
            pass
        if not written:
            import base64 as _b64
            b64 = _b64.b64encode(content.encode()).decode()
            _, _, code = self.run_command(
                f"echo {shlex.quote(b64)} | base64 -d > {shlex.quote(info_path)}",
            )
            written = (code == 0)
        if written:
            self._current_job_info_path = info_path
        else:
            _log(f"WARNING: failed to write job info file: {info_path}")

    def _find_latest_job_info(self):
        """Return info for the most recently submitted job.

        Prefers the in-session job (self._current_job_info_path) so that
        check_job_status always reflects the job you just submitted, not
        whatever old .info file has the newest mtime on disk.

        Returns
        -------
        (info_dict, out_path, err_path, done_path, errlog_path)
        All paths are None when no info file is found.
        """
        # Prefer the job submitted in this session
        preferred = getattr(self, "_current_job_info_path", None)
        if preferred:
            chk, _, chk_code = self.run_command(
                f"test -f {shlex.quote(preferred)} && echo ok", timeout=10)
            if chk_code == 0 and chk.strip() == "ok":
                latest_path = preferred
            else:
                _log(f"_current_job_info_path {preferred!r} not found on remote — falling back to glob")
                latest_path = None
        else:
            latest_path = None

        if not latest_path:
            # Fall back: newest .info file across all job types
            glob_cmd = (
                f"ls -t {self.remote_work_dir}/te_analysis_job_*.info "
                f"{self.remote_work_dir}/te_motif_job_*.info "
                f"{self.remote_work_dir}/gameca_*.info 2>/dev/null | head -1"
            )
            out, _, _ = self.run_command(glob_cmd)
            latest_path = out.strip()
        if not latest_path:
            return None, None, None, None, None

        content, _, ccode = self.run_command(f"cat {latest_path} 2>/dev/null")
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
                )
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
                )
            print(f"[HPC DIAG] sacct probe: job={job_id} exit={acct_code} out={acct.strip()!r} err={acct_err.strip()!r}")
            state = acct.strip().upper()
            if "COMPLETED" in state:
                return "DONE"
            if any(k in state for k in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL")):
                return "FAILED"
            return "DONE"  # gone from squeue, assume finished

        elif self.scheduler == "lsf":
            out, err, code = self.run_command(f"bjobs {job_id} 2>&1")
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

        else:  # No scheduler — job runs inside a tmux session
            out, err, code = self.run_command(
                f"tmux has-session -t {shlex.quote(str(job_id))} 2>&1", timeout=15)
            print(f"[HPC DIAG] scheduler state probe: scheduler=none(tmux) job={job_id} exit={code} out={out.strip()!r} err={err.strip()!r}")
            return "RUNNING" if code == 0 else "DONE"

    def check_job_status(self):
        """Check the status of the most recently submitted batch job."""
        import time as _time
        _status_t0 = _time.time()

        is_session_job = bool(getattr(self, "_current_job_info_path", None))
        job_info, job_out, job_err, job_done, job_errlog = self._find_latest_job_info()

        if job_info is None:
            print()
            print("=" * 60)
            print("  No job information found.")
            print("  Submit a batch job first (option 9 / 16).")
            print("=" * 60)
            return None

        job_id    = job_info.get("JOB_ID",   "unknown")
        out_dir   = job_info.get("OUTPUT_DIR", "")
        family    = job_info.get("FAMILY",    "")
        label     = job_info.get("LABEL",     "")
        job_type  = job_info.get("TYPE",      "pipeline" if family else label or "unknown")
        submitted = job_info.get("SUBMITTED", "unknown")

        print()
        print("=" * 60)
        if not is_session_job:
            print("  ⚠  HISTORICAL JOB  (no job submitted in this session)")
            print("     Showing last job found on disk — this is NOT your current run.")
            print("     Submit a new job, then check status again.")
            print("=" * 60)
        else:
            print("  JOB STATUS")
            print("=" * 60)
        print(f"  Job ID    : {job_id}")
        print(f"  Type      : {job_type}" + (f"  (family: {family})" if family else ""))
        print(f"  Submitted : {submitted}")
        print(f"  Out dir   : {out_dir or '(not set — motif/command job)'}")
        print(f"  Job log   : {job_out or '(unknown)'}")

        # ── 0. Raw scheduler snapshot (always shown, regardless of job_id) ─────
        print()
        print("  --- Raw scheduler queue ---")
        if self.scheduler == "slurm":
            sq_out, sq_err, sq_rc = self.run_command(
                "squeue -u \"$USER\" -o '%.10i %.8T %.12M %.6D %R' 2>&1 || squeue 2>&1 | head -30",
                )
            print(f"  squeue -u $USER  (exit {sq_rc}):")
            for ln in (sq_out or sq_err or "(no output)").rstrip().splitlines():
                print(f"    {ln}")
        elif self.scheduler == "lsf":
            bj_out, bj_err, bj_rc = self.run_command("bjobs 2>&1")
            print(f"  bjobs  (exit {bj_rc}):")
            for ln in (bj_out or bj_err or "(no output)").rstrip().splitlines():
                print(f"    {ln}")
        else:
            tx_out, tx_err, tx_rc = self.run_command(
                "tmux list-sessions 2>&1 | grep gameca_ || echo '(no active tmux sessions)'",
                )
            print(f"  tmux list-sessions  (exit {tx_rc}):")
            for ln in (tx_out or tx_err or "(no output)").rstrip().splitlines():
                print(f"    {ln}")

        py_out, _, _ = self.run_command(
            "ps aux 2>/dev/null | grep -E '[p]ython.*te_|[p]ython.*query' | head -10",
            )
        if py_out.strip():
            print("  Running python pipeline processes:")
            for ln in py_out.rstrip().splitlines():
                print(f"    {ln}")
        else:
            print("  No python pipeline processes found in ps.")

        # ── 1. Ask scheduler first — never trust .done alone ─────────────────
        sched_state = self._get_scheduler_state(job_id)
        done_out, _, _ = self.run_command(f"cat {job_done} 2>/dev/null")
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
                # Could be a nohup background job — check by PID
                ps_out, _, ps_code = self.run_command(
                    f"ps -p {shlex.quote(str(job_id))} -o pid= 2>/dev/null", timeout=8)
                if ps_code == 0 and ps_out.strip():
                    print("  State     : RUNNING  ▶  (background process, not in scheduler queue)")
                else:
                    print("  State     : UNKNOWN  (scheduler silent, process not found, no done marker)")

        # ── 2. Stage progress from checkpoint files ───────────────────────────
        print()
        print("  --- Stage progress ---")
        if out_dir:
            # One command: list all CHECKPOINT files with timestamps
            ckpt_out, _, _ = self.run_command(
                f"find {out_dir} -maxdepth 1 -name 'CHECKPOINT_*.txt'"
                f" -printf '%TH:%TM  %f\\n' 2>/dev/null | sort -k2",
                )
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
                )
            if cur_stage.strip():
                print(f"    ▶  Currently: {cur_stage.strip()}")

            # Result files with sizes — one find+stat call, no per-file round trips
            results_out, _, _ = self.run_command(
                f"find {out_dir} -maxdepth 4 -type f "
                r"\( -name '*.csv' -o -name '*.tsv' -o -name '*.html' -o -name '*.png' \) "
                f"-printf '%s %P\\n' 2>/dev/null | sort -k2 | head -20",
                )
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
        activity_out, _, _ = self.run_command(activity_cmd)
        if activity_out.strip():
            for line in activity_out.strip().splitlines():
                print(f"  {line}")
        else:
            # Nothing after the Python start marker — job is still in preamble or hasn't started
            preamble, _, _ = self.run_command(
                f"tail -10 {job_out} 2>/dev/null || echo '(no output yet)'", timeout=10)
            for line in (preamble or "(no output yet)").rstrip().splitlines():
                print(f"  {line}")

        # ── 4. Stderr / error details ────────────────────────────────────────
        job_failed  = done_val not in ("", "0")
        job_missing_done = done_val == ""   # script didn't finish / still running
        show_err = job_failed or job_missing_done

        if show_err:
            # Always show stderr when the job didn't produce a clean .done marker
            err_file = job_err or job_errlog
            if err_file:
                err_tail, _, err_read_code = self.run_command(
                    f"wc -l < {err_file} 2>/dev/null; tail -40 {err_file} 2>/dev/null",
                    )
                if err_tail.strip() and err_tail.strip() != "0":
                    print()
                    print("  --- Stderr / setup errors ---")
                    lines = err_tail.strip().splitlines()
                    # First line is wc -l output; rest is tail
                    line_count = lines[0].strip() if lines else "?"
                    for line in lines[1:]:
                        print(f"  {line}")
                    print(f"  ({line_count} total lines in stderr log)")

        if job_failed:
            print()
            print("  --- Fatal errors in stdout ---")
            tb_out, _, _ = self.run_command(
                f"grep -n 'Traceback\\|FATAL\\|Error:\\|Exception:\\|ERROR in stage' "
                f"{job_out} 2>/dev/null | tail -20",
                )
            if tb_out.strip():
                for line in tb_out.strip().splitlines():
                    print(f"  {line}")

        print()
        print(f"  Full log  : {job_out}")
        if job_err and job_err != job_out:
            print(f"  Stderr    : {job_err}")
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

        # Raw queue snapshot — always run so user can see what's actually there
        if self.scheduler == "slurm":
            self._diagnostic_command(
                "squeue all user jobs",
                "squeue -u \"$USER\" -o '%.10i %.8T %.12M %.6D %R' 2>&1 || squeue 2>&1 | head -30",
                )
            self._diagnostic_command(
                "running python pipeline processes",
                "ps aux 2>/dev/null | grep -E '[p]ython.*te_|[p]ython.*query' | head -10 || echo '(none)'",
                )
        elif self.scheduler == "lsf":
            self._diagnostic_command("bjobs all", "bjobs 2>&1", timeout=15)
            self._diagnostic_command(
                "running python pipeline processes",
                "ps aux 2>/dev/null | grep -E '[p]ython.*te_|[p]ython.*query' | head -10 || echo '(none)'",
                )
        else:
            self._diagnostic_command("tmux sessions", "tmux list-sessions 2>&1", timeout=15)
            self._diagnostic_command(
                "running python pipeline processes",
                "ps aux 2>/dev/null | grep -E '[p]ython.*te_|[p]ython.*query' | head -10 || echo '(none)'",
                )

        if self.scheduler == "lsf":
            self._diagnostic_command("bjobs summary", f"bjobs {shlex.quote(job_id)} 2>&1", timeout=20)
            self._diagnostic_command("bjobs long", f"bjobs -l {shlex.quote(job_id)} 2>&1", timeout=30)
            self._diagnostic_command("bhist", f"bhist -l {shlex.quote(job_id)} 2>&1 | tail -120", timeout=30)
        elif self.scheduler == "slurm":
            self._diagnostic_command("squeue summary", f"squeue -j {shlex.quote(job_id)} -o '%i %T %R' 2>&1", timeout=20)
            self._diagnostic_command("sacct", f"sacct -j {shlex.quote(job_id)} --format=JobID,State,ExitCode,Elapsed,NodeList,Reason 2>&1", timeout=30)
        else:
            self._diagnostic_command("tmux session state", f"tmux has-session -t {shlex.quote(job_id)} 2>&1 && echo RUNNING || echo GONE", timeout=15)
            self._diagnostic_command("tmux pane tail", f"tmux capture-pane -p -t {shlex.quote(job_id)} 2>&1 | tail -120", timeout=20)

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
            out, err, code = self.run_command(f"cat '{remote_file}' 2>/dev/null")
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
                        )
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
                            )
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

    # ── Submission diagnostics ────────────────────────────────────────────────

    def diagnose_job_submission(self):
        """Run a trivial test job (echo) through the full submission pipeline.

        Prints a step-by-step trace so the user can see exactly where things
        break — SLURM not running, wrong partition, nohup not surviving, etc.
        """
        import datetime as _dt, io as _io, time as _time
        print()
        print("=" * 60)
        print("  Job Submission Diagnostic")
        print("=" * 60)

        _ts  = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        scr  = f"{self.remote_work_dir}/diag_{_ts}.sh"
        out  = f"{self.remote_work_dir}/diag_{_ts}.out"
        err  = f"{self.remote_work_dir}/diag_{_ts}.err"
        done = f"{self.remote_work_dir}/diag_{_ts}.done"

        script = (
            "#!/bin/bash\n"
            f'echo "[diag] started at $(date)"\n'
            f'echo "[diag] hostname=$(hostname)"\n'
            f'echo "[diag] user=$(whoami)"\n'
            f'sleep 5\n'
            f'echo "[diag] finished"\n'
            f'touch {done}\n'
        )

        # Step 1: write script
        print("\n[1/5] Writing test script via SFTP ...")
        try:
            if self.sftp:
                self.sftp.putfo(_io.BytesIO(script.encode()), scr)
                self.run_command(f"chmod +x {scr}")
                print(f"      OK  → {scr}")
            else:
                print("      SFTP not available")
                return
        except Exception as e:
            print(f"      FAILED: {e}")
            return

        # Step 2: check SLURM
        print("\n[2/5] Checking SLURM daemon (scontrol ping) ...")
        ping_out, ping_err, ping_code = self.run_command("scontrol ping 2>&1")
        slurm_up = ping_code == 0 and "UP" in ping_out
        print(f"      exit={ping_code}  output={ping_out.strip()!r}")
        print(f"      SLURM running: {'YES' if slurm_up else 'NO'}")

        # Step 3: detect partition
        print("\n[3/5] Detecting SLURM partition (sinfo) ...")
        part_out, _, part_code = self.run_command(
            "sinfo --noheader -o '%P %a %D' 2>&1", timeout=8)
        print(f"      exit={part_code}  output={part_out.strip()!r}")
        detected_part = self._detect_slurm_partition() if slurm_up else "(N/A)"
        print(f"      Will use partition: {detected_part!r}")

        # Step 4: try sbatch
        submitted_job_id = None
        if slurm_up:
            queue = detected_part
            mem_mb = self.params["MEM_MB"]
            cpus   = self.params["CPUS"]
            header = self._job_script_header("diag_test", out, err,
                                             mem_mb=mem_mb, cpus=cpus,
                                             walltime="00:05", queue=queue)
            full_script = "#!/bin/bash\n" + header + "\n" + script
            try:
                self.sftp.putfo(_io.BytesIO(full_script.encode()), scr)
            except Exception as e:
                print(f"      Script overwrite failed: {e}")
            print(f"\n[4/5] Submitting via sbatch ...")
            sub_out, sub_err, sub_code = self.run_command(f"sbatch {scr}")
            print(f"      exit={sub_code}")
            print(f"      stdout={sub_out.strip()!r}")
            print(f"      stderr={sub_err.strip()!r}")
            if sub_code == 0:
                submitted_job_id = self._parse_job_id(sub_out + sub_err)
                print(f"      parsed job_id={submitted_job_id!r}")
                if submitted_job_id:
                    _time.sleep(2)
                    q_out, q_err, q_code = self.run_command(
                        f"squeue -j {submitted_job_id} 2>&1", timeout=8)
                    print(f"      squeue output={q_out.strip()!r}")
        else:
            print("\n[4/5] Skipping sbatch (SLURM not running)")

        # Step 5: nohup fallback
        if not submitted_job_id:
            print(f"\n[5/5] Trying nohup fallback ...")
            # Rewrite script without scheduler headers
            plain = "#!/bin/bash\n" + script
            try:
                self.sftp.putfo(_io.BytesIO(plain.encode()), scr)
                self.run_command(f"chmod +x {scr}")
            except Exception as e:
                print(f"      Script write failed: {e}")
                return
            nohup_cmd = (
                f"setsid nohup bash {scr} >{out} 2>{err} </dev/null & "
                f"PID=$!; disown $PID 2>/dev/null || true; echo $PID"
            )
            pid_out, pid_err, pid_code = self.run_command(nohup_cmd)
            pid = pid_out.strip().split()[-1] if pid_out.strip() else None
            print(f"      exit={pid_code}  pid={pid!r}  err={pid_err.strip()!r}")
            if pid:
                print(f"      Waiting 8s then checking log ...")
                _time.sleep(8)
                log_out, _, _ = self.run_command(f"cat {out} 2>/dev/null")
                print(f"      Log contents: {log_out.strip()!r}")
                done_out, _, _ = self.run_command(f"ls {done} 2>/dev/null")
                print(f"      Done marker:  {'FOUND' if done_out.strip() else 'MISSING'}")
            else:
                print("      nohup also failed — cannot run background jobs on this host")
        else:
            print("\n[5/5] sbatch succeeded — skipping nohup check")

        print("\n" + "=" * 60)
        print("  Diagnostic complete")
        print("=" * 60)

    # ── Generic batch submission ──────────────────────────────────────────────

    def submit_command_as_batch_job(self, label: str, command: str,
                                    mem_mb: int = None, cpus: int = None,
                                    walltime: str = None, queue: str = None) -> str | None:
        """Wrap any shell command in a scheduler job script and submit it.

        Returns the job ID (scheduler) or PID (nohup fallback) string, or None
        on total failure.  Every step prints a visible status line so failures
        are never silent.
        """
        import datetime as _dt
        import io as _io
        import base64 as _b64

        _ts      = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe     = re.sub(r"[^a-z0-9_]", "_", label.lower())[:20]
        job_name   = f"gameca_{safe}"
        job_script = f"{self.remote_work_dir}/{safe}_{_ts}.sh"
        job_out    = f"{self.remote_work_dir}/{safe}_{_ts}.out"
        job_err    = f"{self.remote_work_dir}/{safe}_{_ts}.err"
        job_done   = f"{self.remote_work_dir}/{safe}_{_ts}.done"
        job_info   = f"{self.remote_work_dir}/gameca_{safe}_{_ts}.info"

        mem_mb   = mem_mb   or self.params["MEM_MB"]
        cpus     = cpus     or self.params["CPUS"]
        walltime = walltime or self.params["WALLTIME"]
        queue    = queue    or self.params["QUEUE"]

        print(f"\n[SUBMIT] ── Job submission: {label} ─────────────────────────")
        print(f"[SUBMIT] Script : {job_script}")
        print(f"[SUBMIT] Output : {job_out}")
        print(f"[SUBMIT] Error  : {job_err}")
        print(f"[SUBMIT] Sched  : {self.scheduler or 'none'}  live={getattr(self, '_scheduler_live', False)}")
        print(f"[SUBMIT] Params : mem={mem_mb}MB  cpus={cpus}  walltime={walltime}  queue={queue}")

        if self.scheduler == "slurm" and queue in ("normal", ""):
            detected = self._detect_slurm_partition()
            if detected and detected != queue:
                print(f"[SUBMIT] Partition auto-detected: {detected!r} (was {queue!r})")
                queue = detected

        sched_header = self._job_script_header(
            job_name, job_out, job_err,
            mem_mb=mem_mb, cpus=cpus, walltime=walltime, queue=queue,
        )
        venv_block   = self._venv_setup_block()
        module_block = self._module_load_block()

        script = (
            "#!/bin/bash\n"
            f"{sched_header}\n"
            "# setup phase — venv errors are logged but do not abort the job\n"
            "set -uo pipefail\n"
            f"{module_block}\n"
            f"{venv_block}\n"
            "# command phase — hard exit on any failure from here on\n"
            "set -euo pipefail\n"
            f'echo "[$(date +%H:%M:%S)] Starting {label} ..."\n'
            f"cd {self.remote_work_dir}\n"
            f"{command}\n"
            f'echo "[$(date +%H:%M:%S)] {label} complete."\n'
            f"touch {job_done}\n"
        )
        print(f"[SUBMIT] Script size: {len(script)} bytes")

        # ── Step 1: Upload job script ─────────────────────────────────────────
        script_written = False
        if self.sftp:
            print("[SUBMIT] Uploading script via SFTP...")
            try:
                self.sftp.putfo(_io.BytesIO(script.encode()), job_script)
                print("[SUBMIT] SFTP upload OK")
                script_written = True
            except Exception as sftp_exc:
                print(f"[SUBMIT] SFTP upload failed: {sftp_exc}")
                print("[SUBMIT] Falling back to base64 shell write...")

        if not script_written:
            print("[SUBMIT] Writing script via base64 shell command...")
            b64 = _b64.b64encode(script.encode()).decode()
            wr_cmd = f"echo {shlex.quote(b64)} | base64 -d > {shlex.quote(job_script)}"
            wr_out, wr_err, wr_code = self.run_command(wr_cmd)
            if wr_code != 0:
                print(f"[SUBMIT] FAILED to write script (exit={wr_code}): {wr_err.strip()!r}")
                return None
            print("[SUBMIT] Base64 write OK")
            script_written = True

        cx_out, cx_err, cx_code = self.run_command(f"chmod +x {shlex.quote(job_script)}")
        if cx_code != 0:
            print(f"[SUBMIT] chmod failed (exit={cx_code}): {cx_err.strip()!r}")
        else:
            print("[SUBMIT] chmod +x OK")

        # Verify the script actually landed on disk
        vf_out, _, vf_code = self.run_command(
            f"wc -c < {shlex.quote(job_script)} 2>/dev/null", timeout=10)
        print(f"[SUBMIT] Remote script size: {vf_out.strip() or '?'} bytes  (exit={vf_code})")

        # ── Step 2: Submit via scheduler ──────────────────────────────────────
        job_id   = None
        combined = ""

        if getattr(self, "_scheduler_live", False):
            submit_cmd = self._submit_job_cmd(job_script)
            print(f"[SUBMIT] Scheduler command: {submit_cmd}")
            sub_out, sub_err, sub_code = self.run_command(submit_cmd)
            combined = (sub_out + "\n" + sub_err).strip()
            print(f"[SUBMIT] Scheduler exit={sub_code}")
            if sub_out.strip():
                print(f"[SUBMIT] Scheduler stdout: {sub_out.strip()!r}")
            if sub_err.strip():
                print(f"[SUBMIT] Scheduler stderr: {sub_err.strip()!r}")

            if sub_code == 0:
                job_id = self._parse_job_id(combined)
                print(f"[SUBMIT] Parsed job ID: {job_id!r}")
                if job_id:
                    import time as _time
                    _time.sleep(2)
                    q_out, q_err, q_code = self.run_command(
                        self._check_running_cmd(job_id), timeout=15)
                    print(f"[SUBMIT] Queue check (exit={q_code}): {q_out.strip()!r}")
                    # Only discard job_id if queue reports a definitive "not found"
                    # (empty output means the scheduler accepted but hasn't registered yet
                    # — keep the ID so the user can monitor it)
            else:
                print(f"[SUBMIT] Scheduler submission FAILED (exit={sub_code})")
        else:
            print("[SUBMIT] Scheduler daemon not live — skipping scheduler, using tmux")

        # ── Step 3: tmux fallback (keeps the job alive across SSH disconnects) ─
        if not job_id:
            if combined:
                print(f"[SUBMIT] Scheduler gave no valid job ID — trying tmux fallback")

            if self._has_tmux():
                session_name = f"{job_name}_{_ts}"
                tmux_cmd = self._tmux_launch_cmd(session_name, job_script, job_out, job_err)
                print(f"[SUBMIT] tmux command: {tmux_cmd}")
                tx_out, tx_err, tx_code = self.run_command(tmux_cmd)
                verify_out, _, verify_code = self.run_command(
                    f"tmux has-session -t {shlex.quote(session_name)} 2>&1", timeout=15)
                print(f"[SUBMIT] tmux exit={tx_code}  session={session_name!r}  verify_exit={verify_code}  stderr={tx_err.strip()!r}")
                if tx_code == 0 and verify_code == 0:
                    self._write_job_info(job_info, session_name, job_out, job_err, job_done, label)
                    print(f"\n  Job running in tmux session  ({session_name})")
                    print(f"  Output: tmux capture-pane -p -t {session_name}")
                    print(f"  Attach: tmux attach -t {session_name}")
                    return session_name
                print(f"[SUBMIT] tmux launch failed — falling back to nohup")
            else:
                print(f"[SUBMIT] tmux not available on remote host — falling back to nohup")

            nohup_cmd = (
                f"nohup bash {shlex.quote(job_script)} "
                f">{shlex.quote(job_out)} 2>{shlex.quote(job_err)} </dev/null & "
                f"echo $!"
            )
            print(f"[SUBMIT] nohup command: {nohup_cmd}")
            pid_out, pid_err, pid_code = self.run_command(nohup_cmd)
            pid = pid_out.strip().split()[-1] if pid_out.strip() else None
            print(f"[SUBMIT] nohup exit={pid_code}  PID={pid!r}  stderr={pid_err.strip()!r}")
            if pid_code != 0 or not pid or not pid.isdigit():
                print(f"[SUBMIT] FAILED: nohup launch failed")
                return None
            self._write_job_info(job_info, pid, job_out, job_err, job_done, label)
            print(f"\n  Job running in background  (PID {pid})")
            print(f"  Output: tail -f {job_out}")
            print(f"  Errors: tail -f {job_err}")
            return pid

        # ── Step 4: Write info file ───────────────────────────────────────────
        print(f"[SUBMIT] Writing info file: {job_info}")
        self._write_job_info(job_info, job_id, job_out, job_err, job_done, label)
        print(f"\n  Job submitted  (ID {job_id})")
        print(f"  Queue:  {queue}   CPUs: {cpus}   Mem: {mem_mb}MB")
        print(f"  Output: {job_out}")
        print(f"  Errors: {job_err}")
        sched = self.scheduler or "lsf"
        print(f"\n  Monitor:  {'squeue -j' if sched == 'slurm' else 'bjobs'} {job_id}")
        print(f"  Live log: tail -f {job_out}")
        return job_id

    # ── Motif-only batch job ──────────────────────────────────────────────────

    def submit_motif_batch_job(self):
        """Submit a standalone motif+GO analysis batch job starting from a BED file."""
        print()
        print("=" * 60)
        print("  Submit Motif+GO Batch Job  (from TE loci BED)")
        print("=" * 60)

        # ── Collect parameters interactively ──────────────────────────────────
        default_bed = f"{self.remote_work_dir}/results/motif_analysis/te_loci.bed"
        bed_path = input(f"\nRemote path to TE loci BED [{default_bed}]: ").strip() or default_bed

        default_jaspar = (str(self.params.get("JASPAR_TABIX_PATH", "")).strip()
                          or str(self.params.get("JASPAR_BED_PATH", "")).strip())
        jaspar_bed = input(f"Remote path to JASPAR tabix .bed.gz [{default_jaspar or 'none — skip JASPAR'}]: ").strip()
        if not jaspar_bed:
            jaspar_bed = default_jaspar  # may still be "" if not configured

        build = input(f"Genome build [{self.params.get('ASSEMBLY', 'hg38')}]: ").strip() or self.params.get("ASSEMBLY", "hg38")

        default_out = f"{self.remote_work_dir}/results"
        out_dir = input(f"Output directory on cluster [{default_out}]: ").strip() or default_out

        p_thresh = input("Fisher p-value threshold [0.05]: ").strip() or "0.05"
        run_go   = input("Also run GO annotation? [y]: ").strip().lower() not in ("n", "no", "0", "false")

        mem_mb   = int(input(f"Memory (MB) [{self.params['MEM_MB']}]: ").strip() or self.params["MEM_MB"])
        cpus     = int(input(f"CPUs [{self.params['CPUS']}]: ").strip() or self.params["CPUS"])
        walltime = input(f"Walltime HH:MM [{self.params['WALLTIME']}]: ").strip() or self.params["WALLTIME"]
        queue    = input(f"Queue/partition [{self.params['QUEUE']}]: ").strip() or self.params["QUEUE"]

        confirm = input(
            f"\nSubmit motif batch job?\n"
            f"  BED:     {bed_path}\n"
            f"  JASPAR:  {jaspar_bed or '(none — JASPAR skipped)'}\n"
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

        # ── Build command ──────────────────────────────────────────────────────
        scratch = f"{self.remote_work_dir}/tmp_motif"

        motif_cmd = (
            f"{self._python} -u {self.remote_work_dir}/te_motif.py"
            f" --bed-input {shlex.quote(bed_path)}"
            f" --build {shlex.quote(build)}"
            f" --out-dir {shlex.quote(out_dir)}"
            f" --p-threshold {p_thresh}"
        )
        if jaspar_bed:
            motif_cmd += f" --jaspar-bed {shlex.quote(jaspar_bed)}"
        if run_go:
            motif_cmd += " --run-go"

        go_fallback = ""
        if run_go:
            go_cmd = (
                f"{self._python} -u {self.remote_work_dir}/te_go.py"
                f" --enrichment-dir {shlex.quote(out_dir)}/enrichment_results"
                f" --build {shlex.quote(build)}"
                f" --out-dir {shlex.quote(out_dir)}"
                f" --p-threshold {p_thresh}"
            )
            go_fallback = (
                f"\n# Run GO explicitly if te_motif didn't chain it\n"
                f"if [ ! -f {shlex.quote(out_dir)}/go_annotations/gene_functions.csv ]; then\n"
                f"  {go_cmd}\nfi"
            )

        preflight = (
            f"[ -f {shlex.quote(bed_path)} ] || "
            f"{{ echo 'FATAL: BED not found: {bed_path}'; exit 1; }}\n"
        )
        if jaspar_bed:
            preflight += (
                f"[ -f {shlex.quote(jaspar_bed)} ] || "
                f"{{ echo 'FATAL: JASPAR BED not found: {jaspar_bed}'; exit 1; }}\n"
                f"[ -f {shlex.quote(jaspar_bed)}.tbi ] || "
                f"{{ echo 'FATAL: JASPAR .tbi not found — run: tabix -p bed {jaspar_bed}'; exit 1; }}\n"
            )
        preflight += (
            f"command -v bedtools >/dev/null 2>&1 || "
            f"{{ echo 'FATAL: bedtools not in PATH'; exit 1; }}\n"
            f"mkdir -p {shlex.quote(scratch)} {shlex.quote(out_dir)}\n"
            f"export TMPDIR={shlex.quote(scratch)}"
        )

        cmd = f"{preflight}\n{motif_cmd}{go_fallback}"

        job_id = self.submit_command_as_batch_job(
            "te_motif_go", cmd,
            mem_mb=mem_mb, cpus=cpus, walltime=walltime, queue=queue,
        )
        if job_id:
            self.current_job_id = job_id
            print(f"\n  Motif job submitted (ID/PID: {job_id})")
            print("  Use 'Check batch job status' (option 11) to monitor it.")
        else:
            print("\n" + "!" * 60)
            print("  SUBMISSION FAILED — no job is running.")
            print("  Check the output above for error details.")
            print("  Common causes: script upload failed, scheduler down,")
            print("  nohup launch failed (try logging in to the cluster")
            print("  and running the command manually).")
            print("!" * 60)
        return bool(job_id)

    # ── te_prep / te_enrichment remote launchers ────────────────────────────

    def _send_completion_email_ssh(self, label: str, exit_code: int) -> None:
        """Fire a completion email from the cluster for an SSH-run (non-bsub) command.

        Uses the same Resend / HTTPS-proxy path as the batch job notification.
        Silently skips if NOTIFY_EMAIL is not set or no API key is configured.
        """
        notify_to = str(self.params.get("NOTIFY_EMAIL", "")).strip()
        if not notify_to:
            return
        result_word = "SUCCEEDED" if exit_code == 0 else f"FAILED (exit {exit_code})"
        subject = f"GAMECA {label}: {result_word}"
        body = f"{label} {result_word}."
        block = self._email_notify_block(notify_to, subject, body)
        if not block:
            return
        print("  Sending completion email...")
        self.run_command(block)

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
            f"{self._python} te_prep.py --build {build} {fam_arg} {genome_arg} --out-dir {out_dir} {extra}"
        )
        print(f"\nRunning: {cmd}\n")
        out, err, code = self.run_command(cmd)
        print(out)
        if err:
            print("[stderr]", err[:500])
        print("Exit code:", code)
        self._send_completion_email_ssh(f"te_prep {family or build}", code)

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
            f"{self._python} te_enrichment.py --input {clustered} --build {build} "
            f"--family {family} --out-dir {out_dir} {jaspar_arg} "
            f"--p-threshold {p_thresh} {extra}"
        )
        print(f"\nRunning: {cmd}\n")
        out, err, code = self.run_command(cmd)
        print(out)
        if err:
            print("[stderr]", err[:500])
        print("Exit code:", code)
        self._send_completion_email_ssh(f"te_enrichment {family}", code)

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
        out, err, code = self.run_command(f"{self._python} {script_path}")

        self.run_command(f"rm -f {script_path}")

        if code != 0:
            print(f"Error generating plots: {err}")
            return False

        print(out)
        return True

    # ------------------------------------------------------------------
    # Gmail OAuth + cluster-side email notifications
    # ------------------------------------------------------------------

    _APP_PASSWORD_STATE_KEY = "gmail_app_password"   # legacy (SMTP path retired)
    _SENDER_EMAIL_STATE_KEY = "gmail_sender_email"
    _OAUTH_CLIENT_ID_KEY     = "gmail_oauth_client_id"
    _OAUTH_CLIENT_SECRET_KEY = "gmail_oauth_client_secret"
    _OAUTH_REFRESH_TOKEN_KEY = "gmail_oauth_refresh_token"

    # --- App-shipped OAuth client (the "installed app" pattern) ---------------
    # Create ONE Desktop OAuth client a single time (Google Cloud Console ->
    # APIs & Services -> Credentials -> Create credentials -> OAuth client ID ->
    # Application type "Desktop app"), then paste its two values here. After that
    # NO end user ever touches the console: option [17] just opens the Google
    # consent popup, exactly like gcloud/gsutil.
    #
    # For Desktop clients Google treats the secret as non-confidential, so
    # shipping it in the app is the expected, supported pattern. Caveat: while
    # the OAuth app's publishing status is "Testing", only added Test users can
    # authorize and their refresh tokens expire after 7 days. Click "Publish app"
    # on the consent screen to lift both limits (the gmail.send scope shows an
    # "unverified app" interstitial until Google verifies it, which you can click
    # through for your own / test accounts).
    _DEFAULT_OAUTH_CLIENT_ID     = ""
    _DEFAULT_OAUTH_CLIENT_SECRET = ""

    # --- Resend (active email provider) ---------------------------------------
    # The cluster sends completion mail via a single HTTPS POST to Resend, which
    # rides the same proxy your downloads use. The API key is embedded on purpose
    # (per user request). From-address uses the verified anmol-dash.com domain, so
    # notifications can be delivered to ANY recipient (not just the Resend account
    # email). If you ever switch domains, verify it at resend.com/domains and set
    # _RESEND_FROM to an address on it.
    _RESEND_API_KEY = "re_VNAgkap7_KFTasPNHnQeMu3QuED4iDDtW"
    _RESEND_FROM    = "GAMECA <no-reply@anmol-dash.com>"

    # "gmail.send" lets the cluster send on your behalf; "userinfo.email" lets us
    # record which address authorized (used as the From: header).
    _OAUTH_SCOPES = ("https://www.googleapis.com/auth/gmail.send "
                     "https://www.googleapis.com/auth/userinfo.email")

    def _resolve_oauth_client(self):
        """Return (client_id, client_secret): a per-user override stored in state
        if present, otherwise the app-shipped default. Either may be "" if nothing
        is configured yet."""
        cid  = (self._state.get(self._OAUTH_CLIENT_ID_KEY, "").strip()
                or self._DEFAULT_OAUTH_CLIENT_ID.strip())
        csec = (self._state.get(self._OAUTH_CLIENT_SECRET_KEY, "").strip()
                or self._DEFAULT_OAUTH_CLIENT_SECRET.strip())
        return cid, csec

    def _oauth_creds_present(self) -> bool:
        cid, csec = self._resolve_oauth_client()
        return bool(cid and csec and self._state.get(self._OAUTH_REFRESH_TOKEN_KEY, "").strip())

    def _get_oauth_creds(self, interactive: bool = True):
        """Return (client_id, client_secret, refresh_token, sender_email).

        Runs the one-time browser authorization if no refresh token is stored.
        Returns empty strings if setup is declined or fails.
        """
        if not self._oauth_creds_present():
            if not interactive:
                return "", "", "", ""
            self.setup_email_auth()
        if not self._oauth_creds_present():
            return "", "", "", ""
        cid, csec = self._resolve_oauth_client()
        return (
            cid, csec,
            self._state.get(self._OAUTH_REFRESH_TOKEN_KEY, "").strip(),
            self._state.get(self._SENDER_EMAIL_STATE_KEY, "").strip(),
        )

    def _oauth_loopback_authorize(self, client_id: str, client_secret: str):
        """Open the browser for Google consent; return (refresh_token, email).

        Uses a localhost redirect and stdlib only. This runs on YOUR machine,
        which has normal internet; only the resulting refresh token is later
        used on the cluster (over HTTPS, through the proxy).
        """
        import http.server, socket, secrets, webbrowser
        import urllib.parse, urllib.request, json as _json

        s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
        redirect_uri = f"http://localhost:{port}/"
        state = secrets.token_urlsafe(16)
        auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode({
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": self._OAUTH_SCOPES,
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
        })

        holder = {}

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                holder["code"]  = (qs.get("code")  or [None])[0]
                holder["state"] = (qs.get("state") or [None])[0]
                holder["error"] = (qs.get("error") or [None])[0]
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<h2>GAMECA: Gmail authorization received.</h2>"
                                 b"<p>You can close this tab and return to the terminal.</p>")

            def log_message(self, *a):
                pass

        httpd = http.server.HTTPServer(("127.0.0.1", port), _Handler)
        print("\n  Opening your browser to authorize Gmail access ...")
        print("  (Sign in, then click Allow. If no browser opens, paste this URL:)")
        print(f"    {auth_url}\n")
        try:
            webbrowser.open(auth_url)
        except Exception:
            pass
        try:
            httpd.handle_request()  # serve the single redirect, then stop
        finally:
            httpd.server_close()

        if holder.get("error"):
            print(f"  Authorization failed: {holder['error']}")
            return "", ""
        if not holder.get("code") or holder.get("state") != state:
            print("  Authorization failed (no code returned or state mismatch).")
            return "", ""

        try:
            data = urllib.parse.urlencode({
                "code": holder["code"],
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            }).encode()
            req = urllib.request.Request("https://oauth2.googleapis.com/token", data=data)
            with urllib.request.urlopen(req) as r:
                tok = _json.loads(r.read().decode())
        except Exception as e:
            print(f"  Token exchange failed: {e}")
            return "", ""

        refresh = tok.get("refresh_token", "")
        access  = tok.get("access_token", "")
        if not refresh:
            print("  Google did not return a refresh token. Revoke prior access at")
            print("  myaccount.google.com/permissions and re-run this setup.")
            return "", ""

        email = ""
        try:
            req2 = urllib.request.Request(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": "Bearer " + access},
            )
            with urllib.request.urlopen(req2) as r:
                email = _json.loads(r.read().decode()).get("email", "")
        except Exception:
            pass
        return refresh, email

    def setup_email_auth(self):
        """Authorize Gmail via Google OAuth — just a browser consent popup.

        If the app ships (or you've stored) an OAuth client, this opens Google's
        sign-in/consent screen directly; the user never visits the console. The
        stored refresh token lets the cluster send mail over HTTPS (Gmail API)
        through the same proxy your downloads use.
        """
        print("\n" + "=" * 64)
        print("GMAIL AUTHORIZATION  (Google sign-in popup — no SMTP, no password)")
        print("=" * 64)

        client_id, client_secret = self._resolve_oauth_client()

        if not (client_id and client_secret):
            # No client shipped/stored yet — this only happens for whoever sets
            # the app up the first time. Offer the one-time client creation.
            print()
            print("  No OAuth client is configured yet. This is a ONE-TIME step for")
            print("  whoever sets up the app; end users never see it afterward.")
            print()
            print("  Create one Desktop OAuth client (~3 min):")
            print("    1. https://console.cloud.google.com/ -> create/pick a project.")
            print("    2. APIs & Services -> Library -> 'Gmail API' -> ENABLE.")
            print("    3. APIs & Services -> OAuth consent screen -> 'External' ->")
            print("       add your Gmail under 'Test users' (or click 'Publish app').")
            print("    4. Credentials -> Create credentials -> OAuth client ID ->")
            print("       Application type 'Desktop app' -> Create. Copy both values.")
            print()
            print("  Tip: paste them into _DEFAULT_OAUTH_CLIENT_ID / _SECRET in")
            print("  hpc_client.py to make this fully automatic for everyone. Or enter")
            print("  them now to store them just for you:")
            print()
            client_id = input("  OAuth Client ID: ").strip()
            if not client_id:
                print("  No client ID — aborting.")
                return
            client_secret = getpass.getpass("  OAuth Client secret: ").strip()
            if not client_secret:
                print("  No client secret — aborting.")
                return
            self._state[self._OAUTH_CLIENT_ID_KEY] = client_id
            self._state[self._OAUTH_CLIENT_SECRET_KEY] = client_secret
            self._save_state()

        refresh, email = self._oauth_loopback_authorize(client_id, client_secret)
        if not refresh:
            print("\n  Authorization did not complete (no refresh token returned).")
            return
        self._state[self._OAUTH_REFRESH_TOKEN_KEY] = refresh
        if email:
            self._state[self._SENDER_EMAIL_STATE_KEY] = email
        self._save_state()

        print("\n  Gmail authorized" + (f" as {email}" if email else "") + ".")
        print("  Now run option [22] to verify the cluster can actually send")
        print("  (it submits a job that mails you via the Gmail API over HTTPS).")

    def _email_send_py(self, result_path: str = "") -> str:
        """Return standalone Python (stdlib only) that sends one email via the
        Resend HTTPS API, routed through any detected proxy (same egress curl
        uses). Returns "" if no Resend API key is configured.

        Recipient / subject / body are read at run time from the environment
        (GAMECA_MAIL_TO / GAMECA_MAIL_SUBJECT / GAMECA_MAIL_BODY) so the same
        script serves both static tests and the pipeline's dynamic status mail.
        """
        key    = self._RESEND_API_KEY.strip()
        sender = self._RESEND_FROM.strip()
        if not key:
            return ""
        rp = repr(result_path) if result_path else "None"
        # NOTE: subject/body/recipient come from the environment at run time.
        return f'''import os, json, urllib.request, urllib.error
from pathlib import Path

def _detect_proxy():
    for v in ("https_proxy","HTTPS_PROXY","http_proxy","HTTP_PROXY","all_proxy","ALL_PROXY"):
        x = os.environ.get(v)
        if x:
            return x.strip()
    home = Path.home()
    rc = home / ".curlrc"
    if rc.exists():
        for raw in rc.read_text(errors="ignore").splitlines():
            ln = raw.strip()
            if not ln or ln.startswith("#"):
                continue
            if ln.lower().startswith(("proxy","-x","--proxy")):
                for sep in ("=", " "):
                    if sep in ln:
                        c = ln.split(sep,1)[1].strip().strip('"').strip("'")
                        if c:
                            return c if "://" in c else "http://"+c
    cc = home / ".condarc"
    if cc.exists():
        txt = cc.read_text(errors="ignore")
        for key in ("https:","http:"):
            if key in txt:
                seg = txt.split(key,1)[1].splitlines()[0].strip().strip('"').strip("'")
                if seg.startswith(("http://","https://")):
                    return seg
    for wg in (home / ".wgetrc", Path("/etc/wgetrc")):
        if wg.exists():
            for raw in wg.read_text(errors="ignore").splitlines():
                ln = raw.strip()
                if ln.lower().startswith(("https_proxy","http_proxy")) and "=" in ln:
                    c = ln.split("=",1)[1].strip().strip('"').strip("'")
                    if c:
                        return c if "://" in c else "http://"+c
    return None

_proxy = _detect_proxy()
_handler = urllib.request.ProxyHandler({{"http": _proxy, "https": _proxy}}) if _proxy else urllib.request.ProxyHandler({{}})
_opener = urllib.request.build_opener(_handler)

_API_KEY = {key!r}
_FROM    = {sender!r}
_TO      = os.environ.get("GAMECA_MAIL_TO", "")
_SUBJECT = os.environ.get("GAMECA_MAIL_SUBJECT", "GAMECA notification")
_BODY    = os.environ.get("GAMECA_MAIL_BODY", "")

def _send():
    if not _TO:
        raise RuntimeError("GAMECA_MAIL_TO is empty")
    payload = json.dumps({{
        "from": _FROM, "to": [_TO], "subject": _SUBJECT, "text": _BODY,
    }}).encode("utf-8")
    # NB: a real User-Agent is REQUIRED. Resend's API is fronted by Cloudflare,
    # which returns "403 error code: 1010" (banned by client signature) for the
    # default "Python-urllib/x.y" UA. A normal browser UA passes the WAF.
    req = urllib.request.Request(
        "https://api.resend.com/emails", data=payload, method="POST",
        headers={{"Authorization": "Bearer " + _API_KEY,
                  "Content-Type": "application/json",
                  "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                  "Accept": "application/json"}})
    with _opener.open(req) as r:
        resp = json.loads(r.read().decode())
    if not resp.get("id"):
        raise RuntimeError("no id in Resend response: %s" % resp)

try:
    _send()
    _status = "resend_send=PASS (proxy=%s)" % (_proxy or "none")
except urllib.error.HTTPError as e:
    _status = "resend_send=FAIL (HTTP %s: %s)" % (e.code, e.read().decode(errors="ignore")[:300])
except Exception as e:
    _status = "resend_send=FAIL (%s)" % e

print(_status)
_rp = {rp}
if _rp:
    try:
        with open(_rp, "w") as _fh:
            _fh.write(_status + "\\n")
    except Exception:
        pass
'''

    def _email_notify_b64(self, result_path: str = "") -> str:
        """base64 of the env-driven Resend send script, or "" if not configured."""
        py = self._email_send_py(result_path)
        return base64.b64encode(py.encode()).decode() if py else ""

    def _email_notify_block(self, to: str, subject: str, body: str,
                            result_path: str = "") -> str:
        """Return a bash snippet that sends a Resend notification from the compute
        node (HTTPS via proxy) with a STATIC subject/body. Returns "" if no API
        key or no recipient.

        For a dynamic body (e.g. exit code), use _email_notify_b64 directly and
        set GAMECA_MAIL_* in the surrounding shell.
        """
        to = (to or "").strip()
        if not to:
            return ""
        b64 = self._email_notify_b64(result_path)
        if not b64:
            return ""
        return (
            '\n# --- GAMECA email notification (Resend API over HTTPS/proxy) ---\n'
            'GAMECA_NOTIFY_PY="$(mktemp "${TMPDIR:-/tmp}/gameca_notify_XXXXXX.py")"\n'
            'chmod 600 "$GAMECA_NOTIFY_PY"\n'
            f'echo {shlex.quote(b64)} | base64 -d > "$GAMECA_NOTIFY_PY"\n'
            f'export GAMECA_MAIL_TO={shlex.quote(to)}\n'
            f'export GAMECA_MAIL_SUBJECT={shlex.quote(subject)}\n'
            f'export GAMECA_MAIL_BODY={shlex.quote(body)}\n'
            f'{self._python} "$GAMECA_NOTIFY_PY" || echo "  (email notification step failed)"\n'
            'rm -f "$GAMECA_NOTIFY_PY"\n'
        )

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
        """Local-side completion email is intentionally disabled.

        Notification is sent fire-and-forget from the compute node via the Resend
        API (see _email_notify_block, wired into the batch job script). Sending
        again from the laptop here was removed so a working local send can't mask
        a broken cluster-side path (a false positive). Run option [22] to verify
        cluster delivery.
        """
        return

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
        done_out, _, _ = self.run_command(f"cat {job_done_file} 2>/dev/null")
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
            info_out, _, _ = self.run_command(f"cat {job_info_file} 2>/dev/null")
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
        check_out, _, _ = self.run_command(f"test -d '{remote_out}' && echo 'exists'")
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

            if not self._ensure_connected():
                print("Cannot stream results: SSH session lost and reconnect failed.")
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
            if not self._ensure_connected():
                print("Cannot stream files: SSH session lost and reconnect failed.")
                return
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

    def submit_email_diagnostic_job(self):
        """Verify cluster-side email delivery via the Resend HTTPS API.

        SMTP can't leave these compute nodes (raw sockets get ENETUNREACH), but
        HTTPS does — through the same proxy curl/downloads use. This submits ONE
        tiny job that, on the compute node, POSTs a test email to the Resend API
        through the proxy and reports PASS/FAIL. No per-user setup: the API key is
        embedded in the app.
        """
        import datetime as _dt
        import time as _time

        print("\n" + "=" * 64)
        print("EMAIL DELIVERY TEST  (Resend API over HTTPS — cluster-side)")
        print("=" * 64)
        print()

        if not getattr(self, "_scheduler_live", False) or self.scheduler not in {"lsf", "slurm"}:
            print("  No live LSF/Slurm scheduler detected — this test targets")
            print("  scheduler-based clusters. Aborting.")
            return

        # Step 1 — confirm Resend is configured (API key is embedded in the app).
        if not self._RESEND_API_KEY.strip():
            print("\n  No Resend API key set (_RESEND_API_KEY in hpc_client.py). Aborting.")
            return
        print(f"  Sending via Resend, From: {self._RESEND_FROM}")
        print("  (anmol-dash.com is verified, so delivery to any recipient is allowed.)")

        # Step 2 — recipient.
        stored_to = self._state.get("test_receiver_email", "").strip()
        prompt = (f"\n  Send the test email to [{stored_to}]: " if stored_to
                  else "\n  Send the test email to (your Resend account email): ")
        to = input(prompt).strip() or stored_to
        if not to:
            print("  No recipient — aborting.")
            return
        self._state["test_receiver_email"] = to
        self._save_state()

        _ts        = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        job_name   = "gameca_email_diag"
        job_script = f"{self.remote_work_dir}/gameca_email_diag_{_ts}.sh"
        job_out    = f"{self.remote_work_dir}/gameca_email_diag_{_ts}.out"
        job_err    = f"{self.remote_work_dir}/gameca_email_diag_{_ts}.err"
        job_done   = f"{self.remote_work_dir}/gameca_email_diag_{_ts}.done"
        job_result = f"{self.remote_work_dir}/gameca_email_diag_{_ts}.result"

        subject = f"GAMECA cluster email test {_ts}"
        body    = (f"This message was sent from a compute node via the Resend API "
                   f"over HTTPS (through the cluster proxy).\n\nIf you received it, "
                   f"cluster-side notifications work. Test id: {_ts}")

        notify_b64 = self._email_notify_b64(result_path=job_result)
        if not notify_b64:
            print("  Internal error: Resend send script could not be built. Aborting.")
            return

        header = self._job_script_header(
            job_name, job_out, job_err,
            mem_mb=500, cpus=1, walltime="00:05", queue=self.params.get("QUEUE", "normal"),
        )

        script = f'''#!/bin/bash
{header}
echo "GAMECA email delivery test (Resend API / HTTPS)"
echo "Host: $(hostname)"
echo "Date: $(date)"

GAMECA_NOTIFY_PY="$(mktemp "${{TMPDIR:-/tmp}}/gameca_notify_XXXXXX.py")"
chmod 600 "$GAMECA_NOTIFY_PY"
echo {shlex.quote(notify_b64)} | base64 -d > "$GAMECA_NOTIFY_PY"
export GAMECA_MAIL_TO={shlex.quote(to)}
export GAMECA_MAIL_SUBJECT={shlex.quote(subject)}
export GAMECA_MAIL_BODY={shlex.quote(body)}
{self._python} "$GAMECA_NOTIFY_PY"
rm -f "$GAMECA_NOTIFY_PY"

touch {job_done}
echo "Test complete."
exit 0
'''

        print("\n  This submits one 1-CPU job that, on the compute node:")
        print("    - detects the proxy (env / ~/.curlrc / ~/.condarc / wgetrc), and")
        print("    - POSTs the test email to the Resend API over HTTPS.")
        print("  Your refresh token is written only to a 0600 temp file on the node,")
        print("  never printed here.")
        confirm = input("\n  Submit test job? (y/n) [y]: ").strip().lower()
        if confirm == "n":
            print("  Aborted.")
            return

        # Write + submit the job.
        create_cmd = f"cat > {job_script} << 'GAMECA_DIAG_EOF'\n{script}\nGAMECA_DIAG_EOF"
        out, err, code = self.run_command(create_cmd)
        if code != 0:
            print(f"  Error creating script: {err}")
            return
        self.run_command(f"chmod 700 {job_script}")
        self.run_command(f"rm -f {job_out} {job_err} {job_done} {job_result}")

        submit_cmd = self._submit_job_cmd(job_script)
        print(f"\n  Submitting: {submit_cmd}")
        out, err, code = self.run_command(submit_cmd)
        job_id = self._parse_job_id(out + err)
        if code != 0:
            print(f"  Submission failed (exit {code}): {(out + err).strip()}")
            return
        print(f"  Job submitted — ID: {job_id or '?'}")
        if job_id:
            self.current_job_id = job_id

        # Poll for the done marker (up to 5 minutes).
        print("  Waiting for job to finish (polling every 15 s, timeout 5 min) ...")
        deadline = _time.time() + 300
        done = False
        while _time.time() < deadline:
            chk_out, _, chk_code = self.run_command(f"cat {job_done} 2>/dev/null")
            if chk_code == 0 and chk_out.strip():
                done = True
                break
            _time.sleep(15)

        # Read back the server-side result line.
        res_out, _, res_code = self.run_command(f"cat {job_result} 2>/dev/null")
        result_line = res_out.strip() if res_code == 0 else ""

        print("\n" + "=" * 64)
        print("RESULT")
        print("=" * 64)
        if not done:
            print("  (job did not finish within 5 min — result may be partial)")
        print(f"  Compute-node Resend send : {result_line or 'no result file written'}")
        print("=" * 64)

        sent_ok = result_line.startswith("resend_send=PASS")
        if sent_ok:
            print(f"\n  The compute node reported a successful send to {to}.")
            print(f"  Check that inbox for subject: \"{subject}\"")
            ans = input("  Did the email ARRIVE? (y/n): ").strip().lower()
            if ans == "y":
                self._state["email_delivery_method"] = "resend_cluster"
                self._save_state()
                print("\n  Saved email_delivery_method = 'resend_cluster'.")
                print("  Real pipeline jobs with NOTIFY_EMAIL set will now email you on finish,")
                print("  sent fire-and-forget from the compute node — no need to keep this client open.")
            else:
                print("\n  Resend reported success but you didn't see it — check Spam. The sender")
                print("  domain (anmol-dash.com) is verified, so any recipient is allowed; a missing")
                print("  message is almost always spam-filtering or a typo in the recipient address.")
        else:
            self._state["email_delivery_method"] = "none"
            self._save_state()
            if "FAIL (HTTP" in result_line:
                print("\n  HTTPS reached Resend but it rejected the request (see code/body above).")
                print("  Common causes: 403 'error code: 1010' = Cloudflare blocked the client")
                print("  signature (needs a real User-Agent — handled by the app); 403 other =")
                print("  sender domain not verified for this key; 401 = bad API key; 422 = invalid")
                print("  From/To address. Fix and re-run.")
            else:
                print("\n  The HTTPS request never completed — likely the proxy wasn't detected on the")
                print("  compute node. Check what makes curl work there:")
                print("    echo $https_proxy $HTTPS_PROXY ; grep -i proxy ~/.curlrc ~/.condarc 2>/dev/null")
                print("  If a proxy shows up only in a config this script doesn't read, tell me the")
                print("  value and I'll wire it in. The full error is in the result line above.")

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
                print("  [12] Send a test email (Resend via cluster — confirm notifications work)")
                print("  [13] Test email delivery (submit job → Resend API send via cluster proxy)")
                print("="*60)

                choice = input("\nSelect option (1-13): ").strip()

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
                    self.submit_email_diagnostic_job()
                elif choice == '13':
                    self.submit_email_diagnostic_job()
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
  python hpc_client.py --host 45.128.119.52 --user root --key ~/.ssh/id_ed25519
  python hpc_client.py --host cluster.edu --scheduler slurm
  python hpc_client.py --host cluster.edu --scheduler lsf
        """
    )
    parser.add_argument("-H", "--host", help="HPC hostname or IP")
    parser.add_argument("-p", "--port", type=int, default=22, help="SSH port (default: 22)")
    parser.add_argument("-u", "--user", help="Username")
    parser.add_argument("-i", "--key", metavar="KEY_FILE",
                        help="Path to SSH private key (default: auto-detect from ~/.ssh/)")
    parser.add_argument("-o", "--output", help="Local output directory for results")
    parser.add_argument(
        "--setup-email", action="store_true",
        help="Show email (Resend) notification status, then exit",
    )
    parser.add_argument(
        "--scheduler", choices=["lsf", "slurm"],
        help="Force scheduler type (lsf or slurm). Auto-detected from PATH if omitted.",
    )
    args = parser.parse_args()

    print("="*60)
    print("  HPC TE ANALYSIS CLIENT")
    print("  Interactive client for running TE analysis on HPC")
    print("="*60)

    client = HPCClient()

    if args.setup_email:
        if client._RESEND_API_KEY.strip():
            print("\n  Email notifications use Resend (HTTPS) — no setup required.")
            print(f"  From: {client._RESEND_FROM}")
            print("  Set NOTIFY_EMAIL to your recipient, then use menu option to send a")
            print("  test once connected. Sender domain anmol-dash.com is verified, so any")
            print("  recipient address is allowed.")
        else:
            print("\n  No Resend API key configured (_RESEND_API_KEY in hpc_client.py).")
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

    key_path = args.key or ""
    if key_path:
        print(f"  Key file: {key_path}")
        password = ""
    else:
        password = getpass.getpass("  Password (leave blank to use SSH key): ")

    # Connect
    if not client.connect(hostname, username, password, port, key_path=key_path):
        print("Failed to connect. Exiting.")
        sys.exit(1)

    # Apply --scheduler override, or prompt if auto-detection failed
    if args.scheduler:
        client.scheduler = args.scheduler
        print(f"Scheduler override applied: {client.scheduler.upper()}")
    elif not client.scheduler:
        print("\nCould not auto-detect a scheduler (bsub/sbatch not found on PATH).")
        while True:
            choice = input("  Choose scheduler [lsf/slurm]: ").strip().lower()
            if choice in ("lsf", "slurm"):
                client.scheduler = choice
                print(f"  Using: {choice.upper()}")
                break
            print("  Please enter 'lsf' or 'slurm'.")

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
