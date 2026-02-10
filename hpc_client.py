#!/usr/bin/env python3
"""
HPC Client for TE Analysis Pipeline (Batch Job Mode)

An interactive client that connects to an HPC cluster, submits analysis
jobs via bsub, and allows you to monitor progress and retrieve results.

Workflow:
    1. Connect to HPC cluster
    2. Configure parameters (family name, input files, primer timeout, etc.)
    3. Submit batch job - job runs on HPC, outputs stay on cluster
    4. Check job status or watch progress live
    5. Retrieve results when job completes

Key Features:
    - Batch job submission via bsub (doesn't require active connection)
    - Results stored on HPC cluster, downloaded on demand
    - Primer timeout with random sampling fallback (default: 2 min)
      If genome search takes too long, uses random chromosome sampling
      to estimate hits instead of hanging indefinitely

Usage:
    python hpc_client.py
    python hpc_client.py --host cluster.edu --user myuser

Requirements:
    pip install paramiko
"""

import os
import sys
import stat
import getpass
import argparse
import base64
import io
import re
import tarfile
from pathlib import Path

try:
    import paramiko
except ImportError:
    print("Error: paramiko is required. Install with: pip install paramiko")
    sys.exit(1)


class HPCClient:
    """Interactive client for running TE analysis on HPC cluster via batch jobs."""

    def __init__(self):
        self.ssh = None
        self.sftp = None
        self.connected = False

        # Default parameters
        self.params = {
            "FAMILY_NAME": "HERVK9",
            "HG38_FA": "/project/amodzlab/index/human/hg38/hg38.fa",
            "BASE_OUT_DIR": "collab_rna",
            "K": 18,  # K-mer size for clustering
            "PRIMER_K": 18,  # K-mer size for primer design (e.g., 8, 12, 18)
            "TOP_N_GLOBAL": 8,
            "TOP_N_CLUSTER": 5,
            "TOP_N_FORWARD_PRIMERS": 3,
            "MIN_SEQUENCES_FOR_CLUSTERING": 10,
            "PRIMER_TIMEOUT": 120,  # Timeout for primer genome search (seconds)
            "all_te_file": "",  # Path to CSV on HPC
            "te_counts": "",    # Path to CSV on HPC
        }

        self.local_output_dir = None
        self.remote_script_path = None
        self.remote_work_dir = None
        self.remote_output_dir = None  # Where results are stored on HPC
        self.current_job_id = None  # Track submitted job
        self._transport = None
        self._password = None
        self.use_sftp = False

    def connect(self, hostname: str, username: str, password: str, port: int = 22):
        """Connect to the HPC cluster via SSH."""
        print(f"\nConnecting to {hostname}...")

        # Store password for keyboard-interactive auth
        self._password = password
        self._transport = None

        def keyboard_interactive_handler(title, instructions, prompt_list):
            """Handle keyboard-interactive authentication."""
            responses = []
            for prompt, show_input in prompt_list:
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

            # Get user's home directory
            channel = self._transport.open_session()
            channel.exec_command("echo $HOME")
            self.remote_work_dir = channel.recv(1024).decode().strip()
            channel.close()
            print(f"Remote home directory: {self.remote_work_dir}")

            return True

        except paramiko.ssh_exception.AuthenticationException as e:
            print(f"Authentication failed: {e}")
            return False
        except Exception as e:
            print(f"Connection failed: {e}")
            if self._transport:
                self._transport.close()
            return False

    def disconnect(self):
        """Close SSH connection."""
        if self.sftp:
            self.sftp.close()
        if self._transport:
            self._transport.close()
        self.connected = False
        print("Disconnected from HPC.")

    def run_command(self, command: str, timeout: int = 300, stream_output: bool = False) -> tuple:
        """Execute a command on the remote server.

        Args:
            command: The command to execute
            timeout: Timeout in seconds
            stream_output: If True, print output in real-time as it's received
        """
        if not self.connected:
            raise RuntimeError("Not connected to HPC")

        channel = self._transport.open_session()
        channel.settimeout(timeout)
        channel.exec_command(command)

        # Read output
        out = b""
        err = b""

        import time
        last_output_time = time.time()

        while True:
            # Check for stdout
            if channel.recv_ready():
                chunk = channel.recv(4096)
                out += chunk
                if stream_output and chunk:
                    try:
                        print(chunk.decode(), end='', flush=True)
                    except UnicodeDecodeError:
                        pass
                    last_output_time = time.time()

            # Check for stderr
            if channel.recv_stderr_ready():
                chunk = channel.recv_stderr(4096)
                err += chunk
                if stream_output and chunk:
                    try:
                        # Print stderr in a different color if possible
                        print(chunk.decode(), end='', flush=True)
                    except UnicodeDecodeError:
                        pass
                    last_output_time = time.time()

            # Check if command is done
            if channel.exit_status_ready():
                break

            # Small sleep to prevent busy waiting
            time.sleep(0.1)

            # Print a heartbeat if no output for a while (only in stream mode)
            if stream_output and (time.time() - last_output_time) > 30:
                print(f"\n[Still running... {int(time.time() - last_output_time)}s since last output]", flush=True)
                last_output_time = time.time()

        # Get any remaining data
        while channel.recv_ready():
            chunk = channel.recv(4096)
            out += chunk
            if stream_output and chunk:
                try:
                    print(chunk.decode(), end='', flush=True)
                except UnicodeDecodeError:
                    pass
        while channel.recv_stderr_ready():
            chunk = channel.recv_stderr(4096)
            err += chunk
            if stream_output and chunk:
                try:
                    print(chunk.decode(), end='', flush=True)
                except UnicodeDecodeError:
                    pass

        exit_code = channel.recv_exit_status()
        channel.close()

        return out.decode(), err.decode(), exit_code

    def preview_family_count(self) -> int:
        """Preview the number of sequences matching the family name."""
        if not self.params["all_te_file"]:
            print("Error: all_te_file path not set")
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
        while True:
            print("\n" + "="*60)
            print("CURRENT PARAMETERS")
            print("="*60)

            param_list = [
                ("1", "FAMILY_NAME", self.params["FAMILY_NAME"]),
                ("2", "HG38_FA", self.params["HG38_FA"]),
                ("3", "BASE_OUT_DIR", self.params["BASE_OUT_DIR"]),
                ("4", "K", self.params["K"]),
                ("5", "PRIMER_K", self.params["PRIMER_K"]),
                ("6", "TOP_N_GLOBAL", self.params["TOP_N_GLOBAL"]),
                ("7", "TOP_N_CLUSTER", self.params["TOP_N_CLUSTER"]),
                ("8", "TOP_N_FORWARD_PRIMERS", self.params["TOP_N_FORWARD_PRIMERS"]),
                ("9", "MIN_SEQUENCES_FOR_CLUSTERING", self.params["MIN_SEQUENCES_FOR_CLUSTERING"]),
                ("10", "PRIMER_TIMEOUT", self.params["PRIMER_TIMEOUT"]),
                ("11", "all_te_file", self.params["all_te_file"] or "[NOT SET - REQUIRED]"),
                ("12", "te_counts", self.params["te_counts"] or "[NOT SET - optional]"),
            ]

            # Add descriptions below the parameter list
            print("\n  Parameter descriptions:")
            print("    [4]  K: K-mer size for clustering analysis (default: 18)")
            print("    [5]  PRIMER_K: K-mer size for primer design (e.g., 8, 12, 18)")
            print("    [10] PRIMER_TIMEOUT: Timeout (seconds) for each primer search")
            print("         If exceeded, uses random chromosome sampling instead")
            print("    [11] all_te_file: TE annotations CSV with columns:")
            print("         chr, start, stop, TE_name, family, strand, + expression columns")
            print("    [12] te_counts: Optional summary CSV (not required for analysis)")

            for num, name, value in param_list:
                print(f"  [{num:>2}] {name:<30} = {value}")

            print("\n  [p]  Preview family count")
            print("  [r]  Run analysis (submit batch job)")
            print("  [q]  Quit")
            print("="*60)

            choice = input("\nSelect option (1-12, p, r, q): ").strip().lower()

            if choice == 'q':
                return False
            elif choice == 'p':
                self.preview_family_count()
            elif choice == 'r':
                return True
            elif choice in ['1', '2', '3', '11', '12']:
                # String parameters
                key = param_list[int(choice)-1][1].split()[0]  # Handle "all_te_file (CSV path)"
                current = self.params[key]
                new_val = input(f"Enter new value for {key} [{current}]: ").strip()
                if new_val:
                    self.params[key] = new_val
            elif choice in ['4', '5', '6', '7', '8', '9', '10']:
                # Integer parameters
                key = param_list[int(choice)-1][1]
                current = self.params[key]
                new_val = input(f"Enter new value for {key} [{current}]: ").strip()
                if new_val:
                    try:
                        self.params[key] = int(new_val)
                    except ValueError:
                        print("Invalid integer value")
            else:
                print("Invalid option")

        return False

    def upload_script(self):
        """Upload the analysis script to HPC with current parameters."""
        # Read local query.py
        local_script = Path(__file__).parent / "query.py"

        if not local_script.exists():
            print(f"Error: query.py not found at {local_script}")
            return False

        with open(local_script, 'r') as f:
            script_content = f.read()

        # Create modified script with parameters injected
        param_block = f'''# ==================== CONFIG ====================
FAMILY_NAME = "{self.params['FAMILY_NAME']}"
HG38_FA = "{self.params['HG38_FA']}"
BASE_OUT_DIR = Path("{self.params['BASE_OUT_DIR']}")
K = {self.params['K']}  # K-mer size for clustering
PRIMER_K = {self.params['PRIMER_K']}  # K-mer size for primer design
TOP_N_GLOBAL = {self.params['TOP_N_GLOBAL']}
TOP_N_CLUSTER = {self.params['TOP_N_CLUSTER']}
TOP_N_FORWARD_PRIMERS = {self.params['TOP_N_FORWARD_PRIMERS']}
MIN_SEQUENCES_FOR_CLUSTERING = {self.params['MIN_SEQUENCES_FOR_CLUSTERING']}
PRIMER_SEARCH_TIMEOUT_OVERRIDE = {self.params['PRIMER_TIMEOUT']}  # Timeout for primer search (uses random sampling if exceeded)
DEBUG = True  # Enable progress output
'''

        # Add data loading code
        data_load_block = f'''
# ==================== LOAD INPUT DATA ====================
import pandas as pd
print("Loading input data...")
df = pd.read_csv("{self.params['all_te_file']}")
print(f"Loaded all_te_file: {{len(df)}} rows")
'''

        if self.params['te_counts']:
            data_load_block += f'''
df2 = pd.read_csv("{self.params['te_counts']}")
print(f"Loaded te_counts: {{len(df2)}} rows")
'''

        # Find and replace the config section
        # Replace the config block
        config_pattern = r'# ==================== CONFIG ====================.*?# ================================================'
        modified_script = re.sub(config_pattern, param_block + '\n# ================================================',
                                 script_content, flags=re.DOTALL)

        # Insert data loading before "# ==================== LOAD AND FILTER DATA ===================="
        load_marker = "# ==================== LOAD AND FILTER DATA ===================="
        modified_script = modified_script.replace(load_marker, data_load_block + "\n" + load_marker)

        # Upload to remote
        remote_script = f"{self.remote_work_dir}/te_analysis_run.py"

        if self.use_sftp and self.sftp:
            with self.sftp.file(remote_script, 'w') as f:
                f.write(modified_script)
        else:
            # Upload via chunked base64 to avoid "Argument list too long" errors.
            # Shell argument limits are typically 128KB-2MB, and our script can exceed that
            # after base64 encoding. We split into ~48KB raw chunks (~64KB base64).
            encoded_full = base64.b64encode(modified_script.encode()).decode()
            chunk_size = 65000  # base64 chars per chunk — safe for any shell
            chunks = [encoded_full[i:i+chunk_size] for i in range(0, len(encoded_full), chunk_size)]

            print(f"Uploading script ({len(modified_script)} bytes, {len(chunks)} chunks)...")

            # First chunk: overwrite file
            cmd = f"echo '{chunks[0]}' | base64 -d > {remote_script}"
            out, err, code = self.run_command(cmd, timeout=30)
            if code != 0:
                print(f"Failed to upload script (chunk 1): {err}")
                return False

            # Remaining chunks: append
            for i, chunk in enumerate(chunks[1:], 2):
                cmd = f"echo '{chunk}' | base64 -d >> {remote_script}"
                out, err, code = self.run_command(cmd, timeout=30)
                if code != 0:
                    print(f"Failed to upload script (chunk {i}/{len(chunks)}): {err}")
                    return False

            print(f"  Uploaded {len(chunks)} chunks successfully")

        self.remote_script_path = remote_script
        print(f"Uploaded analysis script to {remote_script}")
        return True

    def submit_batch_job(self):
        """Submit the analysis as a batch job on HPC. Returns immediately after submission."""
        if not self.params["all_te_file"]:
            print("Error: all_te_file path must be set")
            return False

        # Preview count first
        count = self.preview_family_count()
        if count <= 0:
            confirm = input("\nNo sequences found. Continue anyway? (y/n): ").strip().lower()
            if confirm != 'y':
                return False

        confirm = input(f"\nProceed with analysis for {count} sequences? (y/n): ").strip().lower()
        if confirm != 'y':
            return False

        # Upload script
        if not self.upload_script():
            return False

        # Set up output directory on HPC
        family = self.params["FAMILY_NAME"].lower()
        self.remote_output_dir = f"{self.remote_work_dir}/{self.params['BASE_OUT_DIR']}/{family}"

        # Create bsub job script
        job_name = f"te_analysis_{self.params['FAMILY_NAME']}"
        job_script = f"{self.remote_work_dir}/te_analysis_job.sh"
        job_out = f"{self.remote_work_dir}/te_analysis_job.out"
        job_err = f"{self.remote_work_dir}/te_analysis_job.err"
        job_done = f"{self.remote_work_dir}/te_analysis_job.done"
        job_info = f"{self.remote_work_dir}/te_analysis_job.info"

        # Error log file
        job_error_log = f"{self.remote_work_dir}/te_analysis_job.error.log"

        # Build the bsub script content
        bsub_script = f'''#!/bin/bash
#BSUB -J {job_name}
#BSUB -o {job_out}
#BSUB -e {job_err}
#BSUB -n 4
#BSUB -M 12000
#BSUB -W 04:00
#BSUB -q normal

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

if [ ! -f "{self.remote_script_path}" ]; then
    echo "FATAL: Script not found: {self.remote_script_path}" | tee -a $ERROR_LOG
    echo "1" > {job_done}
    exit 1
fi
echo "  Script:     OK"

if [ ! -f "{self.params['all_te_file']}" ]; then
    echo "FATAL: Input file not found: {self.params['all_te_file']}" | tee -a $ERROR_LOG
    echo "1" > {job_done}
    exit 1
fi
echo "  Input data: OK"

if [ -f "{self.params['HG38_FA']}" ]; then
    echo "  Genome:     OK ($(du -sh "{self.params['HG38_FA']}" 2>/dev/null | cut -f1))"
else
    echo "  Genome:     NOT FOUND — will use UCSC API fallback" | tee -a $ERROR_LOG
fi

echo ""
echo "[$(date +%H:%M:%S)] Starting pipeline..."
echo "=========================================================="

SECONDS=0
python -u {self.remote_script_path} --primer-timeout {self.params['PRIMER_TIMEOUT']} 2>&1 | tee -a $ERROR_LOG
EXIT_CODE=${{PIPESTATUS[0]}}

echo ""
echo "=========================================================="
echo " Pipeline finished"
echo " Exit code:   $EXIT_CODE"
echo " Runtime:     $((SECONDS / 60))m $((SECONDS % 60))s"
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
        print("\nSubmitting job to cluster via bsub...")
        submit_cmd = f"bsub < {job_script}"
        out, err, code = self.run_command(submit_cmd, timeout=30)

        if code != 0:
            print(f"Error submitting job: {err}")
            return False

        # Parse job ID from bsub output
        job_id = None
        match = re.search(r'Job <(\d+)>', out)
        if match:
            job_id = match.group(1)
            self.current_job_id = job_id
            print(f"Job submitted successfully! Job ID: {job_id}")
        else:
            print(f"Job submitted but could not parse job ID from: {out}")
            job_id = "unknown"
            self.current_job_id = None

        # Save job info for later retrieval
        job_info_content = f"JOB_ID={job_id}\nFAMILY={self.params['FAMILY_NAME']}\nOUTPUT_DIR={self.remote_output_dir}\nSUBMITTED=$(date)"
        self.run_command(f"echo '{job_info_content}' > {job_info}", timeout=10)

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
        print("\nUseful HPC commands:")
        print(f"  bjobs {job_id}        # Check job status")
        print(f"  bpeek {job_id}        # View live output")
        print(f"  bkill {job_id}        # Cancel job")
        print("=" * 60)

        return True

    def run_interactive_job(self):
        """Run analysis interactively on a compute node via bsub -Is.

        Uses 'bsub -M 12000 -n 4 -Is bash' to allocate a compute node,
        then runs the pipeline with real-time output streaming.
        This keeps the SSH connection active and streams all output back.
        """
        if not self.params["all_te_file"]:
            print("Error: all_te_file path must be set")
            return False

        # Preview count
        count = self.preview_family_count()
        if count <= 0:
            confirm = input("\nNo sequences found. Continue anyway? (y/n): ").strip().lower()
            if confirm != 'y':
                return False

        confirm = input(f"\nProceed with interactive analysis for {count} sequences? (y/n): ").strip().lower()
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
        runner_content = f'''#!/bin/bash
set -e

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
echo " Primer K:    {self.params['PRIMER_K']}"
echo " Timeout:     {self.params['PRIMER_TIMEOUT']}s"
echo "=========================================================="
echo ""

cd {self.remote_work_dir}

# Verify files exist
echo "[$(date +%H:%M:%S)] Checking prerequisites..."
if [ ! -f "{self.remote_script_path}" ]; then
    echo "FATAL: Script not found: {self.remote_script_path}"
    exit 1
fi
echo "  Script:     OK ({self.remote_script_path})"

if [ ! -f "{self.params['all_te_file']}" ]; then
    echo "FATAL: Input file not found: {self.params['all_te_file']}"
    exit 1
fi
echo "  Input data: OK ({self.params['all_te_file']})"

if [ -f "{self.params['HG38_FA']}" ]; then
    echo "  Genome:     OK ({self.params['HG38_FA']})"
    GENOME_SIZE=$(du -sh "{self.params['HG38_FA']}" 2>/dev/null | cut -f1)
    echo "              Size: $GENOME_SIZE"
else
    echo "  Genome:     NOT FOUND — will use UCSC API fallback"
fi

echo ""
echo "[$(date +%H:%M:%S)] Python version: $(python --version 2>&1)"
echo "[$(date +%H:%M:%S)] Starting pipeline..."
echo "=========================================================="
echo ""

SECONDS=0
python -u {self.remote_script_path} --primer-timeout {self.params['PRIMER_TIMEOUT']}
EXIT_CODE=$?

echo ""
echo "=========================================================="
echo " Pipeline finished"
echo " Exit code:   $EXIT_CODE"
echo " Runtime:     $((SECONDS / 60))m $((SECONDS % 60))s"
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

        # Build bsub command
        bsub_cmd = (
            f"bsub -M 12000 -n 4 -Is "
            f"bash {runner_script}"
        )

        print("\n" + "=" * 60)
        print("SUBMITTING INTERACTIVE JOB")
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
            out, err, code = self.run_command(bsub_cmd, timeout=14400, stream_output=True)
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

    def check_job_status(self):
        """Check the status of the submitted batch job."""
        job_info_file = f"{self.remote_work_dir}/te_analysis_job.info"
        job_done_file = f"{self.remote_work_dir}/te_analysis_job.done"
        job_out_file = f"{self.remote_work_dir}/te_analysis_job.out"
        job_err_file = f"{self.remote_work_dir}/te_analysis_job.err"
        job_error_log = f"{self.remote_work_dir}/te_analysis_job.error.log"

        # Try to get job info
        out, err, code = self.run_command(f"cat {job_info_file} 2>/dev/null", timeout=10)
        if code != 0 or not out.strip():
            print("No job information found. Have you submitted a job?")
            return None

        # Parse job info
        job_info = {}
        for line in out.strip().split('\n'):
            if '=' in line:
                key, val = line.split('=', 1)
                job_info[key] = val

        job_id = job_info.get('JOB_ID', 'unknown')
        family = job_info.get('FAMILY', 'unknown')
        output_dir = job_info.get('OUTPUT_DIR', '')

        print("\n" + "=" * 60)
        print("JOB STATUS")
        print("=" * 60)
        print(f"Job ID: {job_id}")
        print(f"Family: {family}")
        print(f"Output dir: {output_dir}")

        # Check if job is done
        done_out, _, _ = self.run_command(f"cat {job_done_file} 2>/dev/null", timeout=10)
        job_failed = False
        if done_out.strip():
            try:
                exit_code = int(done_out.strip())
                if exit_code == 0:
                    print(f"\nStatus: COMPLETED SUCCESSFULLY")
                else:
                    print(f"\nStatus: FAILED (exit code: {exit_code})")
                    job_failed = True
            except ValueError:
                print(f"\nStatus: COMPLETED (exit code: {done_out.strip()})")

            # Show output size
            size_out, _, _ = self.run_command(f"du -sh {output_dir} 2>/dev/null || echo 'unknown'", timeout=10)
            print(f"Results size: {size_out.strip()}")
            if not job_failed:
                print("\nResults are ready to retrieve!")
        else:
            # Check if job is still running
            if job_id != 'unknown':
                bjobs_out, _, _ = self.run_command(f"bjobs {job_id} 2>&1", timeout=10)
                if 'not found' in bjobs_out.lower():
                    # Job finished but no done marker - check output
                    print("\nStatus: COMPLETED (checking results...)")
                else:
                    print(f"\nbjobs output:\n{bjobs_out}")
            else:
                print("\nStatus: UNKNOWN (no job ID available)")

        # If job failed, show error information
        if job_failed:
            print("\n" + "-" * 60)
            print("ERROR DETAILS")
            print("-" * 60)

            # Show last part of error log
            err_log_out, _, _ = self.run_command(f"tail -100 {job_error_log} 2>/dev/null", timeout=30)
            if err_log_out.strip():
                print("\n--- Error Log (last 100 lines) ---")
                print(err_log_out)
                print("--- End Error Log ---")

            # Show stderr
            stderr_out, _, _ = self.run_command(f"tail -50 {job_err_file} 2>/dev/null", timeout=30)
            if stderr_out.strip():
                print("\n--- STDERR (last 50 lines) ---")
                print(stderr_out)
                print("--- End STDERR ---")

            # Look for Python tracebacks in output
            traceback_out, _, _ = self.run_command(
                f"grep -A 20 'Traceback\\|Error:\\|Exception:' {job_out_file} 2>/dev/null | tail -50",
                timeout=30
            )
            if traceback_out.strip():
                print("\n--- Python Errors Found ---")
                print(traceback_out)
                print("--- End Python Errors ---")

            print("\nFull logs available at:")
            print(f"  Output: {job_out_file}")
            print(f"  Stderr: {job_err_file}")
            print(f"  Error log: {job_error_log}")

        else:
            # Optionally show recent output for non-failed jobs
            show_output = input("\nShow recent job output? (y/n): ").strip().lower()
            if show_output == 'y':
                tail_out, _, _ = self.run_command(f"tail -50 {job_out_file} 2>/dev/null", timeout=30)
                print("\n--- Recent Output ---")
                print(tail_out if tail_out else "(no output yet)")
                print("--- End Output ---")

        # Check for pipeline error log in output directory
        if output_dir:
            family = job_info.get('FAMILY', 'unknown').lower()
            pipeline_error_log = f"{output_dir}/pipeline_errors.log"
            pipe_err_out, _, _ = self.run_command(f"cat {pipeline_error_log} 2>/dev/null", timeout=30)
            if pipe_err_out.strip():
                print("\n" + "-" * 60)
                print("PIPELINE ERROR LOG")
                print("-" * 60)
                print(pipe_err_out)
                print("-" * 60)

        print("=" * 60)
        return job_info

    def download_error_logs(self, local_dir: str = None):
        """Download just the error logs from the HPC for debugging."""
        job_info_file = f"{self.remote_work_dir}/te_analysis_job.info"
        job_out_file = f"{self.remote_work_dir}/te_analysis_job.out"
        job_err_file = f"{self.remote_work_dir}/te_analysis_job.err"
        job_error_log = f"{self.remote_work_dir}/te_analysis_job.error.log"

        # Get job info
        info_out, _, _ = self.run_command(f"cat {job_info_file} 2>/dev/null", timeout=10)
        output_dir = None
        if info_out.strip():
            for line in info_out.strip().split('\n'):
                if line.startswith('OUTPUT_DIR='):
                    output_dir = line.split('=', 1)[1]
                    break

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

        # Add pipeline error log if output dir exists
        if output_dir:
            log_files.append((f"{output_dir}/pipeline_errors.log", "pipeline_errors.log"))
            # Also try UCSC fetch errors
            family = output_dir.split('/')[-1]
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
        """Watch job progress in real-time (polling mode)."""
        job_info_file = f"{self.remote_work_dir}/te_analysis_job.info"
        job_done_file = f"{self.remote_work_dir}/te_analysis_job.done"
        job_out_file = f"{self.remote_work_dir}/te_analysis_job.out"

        # Get job info
        out, err, code = self.run_command(f"cat {job_info_file} 2>/dev/null", timeout=10)
        if code != 0 or not out.strip():
            print("No job information found. Have you submitted a job?")
            return False

        job_id = "unknown"
        for line in out.strip().split('\n'):
            if line.startswith('JOB_ID='):
                job_id = line.split('=', 1)[1]
                break

        print("\n" + "=" * 60)
        print(f"WATCHING JOB {job_id} (polling every 10 seconds)")
        print("Press Ctrl+C to stop watching (job will continue running)")
        print("=" * 60 + "\n")

        import time
        last_output_size = 0
        poll_interval = 10

        try:
            while True:
                # Check if job is done
                done_out, _, _ = self.run_command(f"cat {job_done_file} 2>/dev/null || echo 'running'", timeout=10)
                if done_out.strip() != 'running':
                    print(f"\n\nJob completed with exit code: {done_out.strip()}")
                    break

                # Check job status with bjobs
                if job_id != 'unknown':
                    status_out, _, _ = self.run_command(f"bjobs {job_id} 2>&1 | tail -1", timeout=10)
                    if 'not found' in status_out.lower():
                        time.sleep(2)
                        done_out, _, _ = self.run_command(f"cat {job_done_file} 2>/dev/null || echo 'running'", timeout=10)
                        if done_out.strip() != 'running':
                            print(f"\n\nJob completed with exit code: {done_out.strip()}")
                            break
                        print("\nJob finished but no done marker found.")
                        break

                # Stream new output
                size_out, _, _ = self.run_command(f"wc -c < {job_out_file} 2>/dev/null || echo 0", timeout=10)
                try:
                    current_size = int(size_out.strip())
                except ValueError:
                    current_size = 0

                if current_size > last_output_size:
                    skip_bytes = last_output_size
                    new_out, _, _ = self.run_command(
                        f"tail -c +{skip_bytes + 1} {job_out_file} 2>/dev/null",
                        timeout=30
                    )
                    if new_out:
                        print(new_out, end='', flush=True)
                    last_output_size = current_size

                time.sleep(poll_interval)

        except KeyboardInterrupt:
            print(f"\n\nStopped watching. Job {job_id} is still running on the cluster.")
            print(f"Use 'Check job status' to check progress later.")
            return False

        return True

    def run_analysis(self):
        """Run the analysis - just calls submit_batch_job for backwards compatibility."""
        return self.submit_batch_job()

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

    def retrieve_results(self, local_dir: str):
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

        # Try to get output directory from job info
        remote_out = None
        info_out, _, _ = self.run_command(f"cat {job_info_file} 2>/dev/null", timeout=10)
        if info_out.strip():
            for line in info_out.strip().split('\n'):
                if line.startswith('OUTPUT_DIR='):
                    remote_out = line.split('=', 1)[1]
                    break

        # Fall back to constructed path
        if not remote_out:
            family = self.params["FAMILY_NAME"].lower()
            remote_out = f"{self.remote_work_dir}/{self.params['BASE_OUT_DIR']}/{family}"

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

    def main_menu(self):
        """Main interactive menu after connection."""
        while True:
            try:
                print("\n" + "="*60)
                print("HPC TE ANALYSIS CLIENT")
                print("="*60)
                print("  [1] Configure parameters")
                print("  [2] Preview family count")
                print("  --- Run Analysis ---")
                print("  [3] Run interactively (bsub -Is, real-time output)")
                print("  [4] Submit batch job  (bsub, runs in background)")
                print("  --- Monitor & Retrieve ---")
                print("  [5] Check batch job status")
                print("  [6] Watch batch job progress (live)")
                print("  [7] Retrieve results")
                print("  [8] Download error logs only")
                print("  [9] Disconnect and exit")
                print("="*60)

                choice = input("\nSelect option (1-9): ").strip()

                if choice == '1':
                    if self.set_parameter_interactive():
                        # User chose to run — offer interactive mode first
                        mode = input("\nRun interactively (i) or submit batch job (b)? [i]: ").strip().lower()
                        if mode == 'b':
                            self.submit_batch_job()
                        else:
                            self.run_interactive_job()
                elif choice == '2':
                    self.preview_family_count()
                elif choice == '3':
                    self.run_interactive_job()
                elif choice == '4':
                    self.submit_batch_job()
                elif choice == '5':
                    self.check_job_status()
                elif choice == '6':
                    self.watch_job()
                elif choice == '7':
                    if self.local_output_dir:
                        default_dir = str(self.local_output_dir)
                        local_dir = input(f"Enter local output directory [{default_dir}]: ").strip()
                        local_dir = local_dir or default_dir
                    else:
                        local_dir = input("Enter local output directory (e.g., ~/Documents/output): ").strip()
                    if local_dir:
                        self.retrieve_results(local_dir)
                elif choice == '8':
                    local_dir = input("Enter local directory for error logs [./hpc_error_logs]: ").strip()
                    self.download_error_logs(local_dir if local_dir else None)
                elif choice == '9':
                    break
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
    args = parser.parse_args()

    print("="*60)
    print("  HPC TE ANALYSIS CLIENT")
    print("  Interactive client for running TE analysis on HPC")
    print("="*60)

    client = HPCClient()

    # Get connection details (use args or prompt)
    print("\nEnter HPC connection details:")

    hostname = args.host
    if not hostname:
        hostname = input("  Hostname: ").strip()
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
        username = input("  Username: ").strip()
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
