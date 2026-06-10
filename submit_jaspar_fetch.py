#!/usr/bin/env python3
"""
submit_jaspar_fetch.py  —  From your laptop, log in to the HPC cluster and
submit FOUR parallel, NO-TIME-LIMIT JASPAR pre-fetch jobs (one per build).

Each job runs  fetch_jaspar.py --build <B> --cache-dir <CACHE>  on a compute
node and writes a genome-wide, reusable  JASPAR2022_<B>.sorted.bed.gz  into the
shared cache. Later pipeline runs reuse it via  --jaspar-dir <CACHE>.

Login is handled by the existing hpc_client.HPCClient (supports password +
Duo push), so you connect "from this side" exactly like the main pipeline.

NO TIME LIMIT: the generated job script intentionally OMITS the LSF -W /
Slurm --time directive, so no wall-clock cap is imposed from our side. (The
queue's own default still applies — pass --queue to pick an unlimited queue.)

Example:
    python3 submit_jaspar_fetch.py \\
        --host hpclogin1.hpc.upenn.edu --user amodz \\
        --remote-dir /home/amodz/anmol/te-scraper-and-analysis \\
        --cache-dir  /home/amodz/anmol/jaspar_cache \\
        --queue normal --mem-mb 16000 --cpus 4 --upload

Then point the pipeline at the cache:
    python3 query.py --local --family LTR5_Hs --assembly hg38 \\
        --output ... --jaspar-dir /home/amodz/anmol/jaspar_cache
"""

import argparse
import base64
import getpass
import shlex
import sys
from pathlib import Path

DEFAULT_BUILDS = ["hg38", "hg19", "mm10", "mm39"]


def _build_job_script(scheduler, build, remote_dir, cache_dir, mem_mb, cpus,
                      queue, workers):
    """Generate a scheduler job script with NO wall-clock limit."""
    name = f"jaspar_{build}"
    out  = f"{cache_dir}/fetch_{build}.out"
    err  = f"{cache_dir}/fetch_{build}.err"
    cmd  = (f"python3 {shlex.quote(remote_dir)}/fetch_jaspar.py "
            f"--build {shlex.quote(build)} "
            f"--cache-dir {shlex.quote(cache_dir)} "
            f"--workers {workers}")

    if scheduler == "slurm":
        # --time intentionally omitted → partition default (no cap from us).
        header = (
            f"#SBATCH --job-name={name}\n"
            f"#SBATCH --output={out}\n"
            f"#SBATCH --error={err}\n"
            f"#SBATCH --cpus-per-task={cpus}\n"
            f"#SBATCH --mem={max(1, mem_mb // 1000)}G\n"
            f"#SBATCH --partition={queue}\n"
        )
    else:  # lsf
        # -W intentionally omitted → no wall-clock limit imposed by us.
        header = (
            f"#BSUB -J {name}\n"
            f"#BSUB -o {out}\n"
            f"#BSUB -e {err}\n"
            f"#BSUB -n {cpus}\n"
            f"#BSUB -M {mem_mb}\n"
            f"#BSUB -q {queue}\n"
        )

    return (
        "#!/bin/bash\n"
        f"{header}"
        "# NOTE: time-limit directive deliberately omitted (NO TIME LIMIT).\n"
        "set -uo pipefail\n"
        f"cd {shlex.quote(remote_dir)}\n"
        f'echo "[$(date)] JASPAR fetch start build={build} cache={cache_dir}"\n'
        f"{cmd}\n"
        "rc=$?\n"
        f'echo "[$(date)] JASPAR fetch done build={build} rc=$rc"\n'
        "exit $rc\n"
    )


def _put_remote_file(client, content_bytes, remote_path):
    """Upload bytes to remote_path: prefer SFTP, fall back to base64 over SSH."""
    import io
    if getattr(client, "sftp", None):
        try:
            client.sftp.putfo(io.BytesIO(content_bytes), remote_path)
            return True
        except Exception as exc:
            print(f"  SFTP put failed ({exc}); falling back to base64 over SSH")
    b64 = base64.b64encode(content_bytes).decode()
    out, err, rc = client.run_command(
        f"echo {b64} | base64 -d > {shlex.quote(remote_path)}", timeout=120)
    if rc != 0:
        print(f"  base64 upload failed (rc={rc}): {err[:200]}")
        return False
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Login to HPC and submit 4 parallel no-time-limit JASPAR fetch jobs.")
    ap.add_argument("--host", required=True, help="HPC login hostname.")
    ap.add_argument("--user", required=True, help="HPC username.")
    ap.add_argument("--port", type=int, default=22)
    ap.add_argument("--key", default="", help="Path to SSH private key (optional).")
    ap.add_argument("--remote-dir", required=True,
                    help="Remote dir holding te_motif.py (+ fetch_jaspar.py, or use --upload).")
    ap.add_argument("--cache-dir", required=True,
                    help="Remote shared dir for the JASPAR2022_<build>.sorted.bed.gz caches.")
    ap.add_argument("--builds", nargs="+", default=DEFAULT_BUILDS,
                    help=f"Builds → one job each (default: {' '.join(DEFAULT_BUILDS)}).")
    ap.add_argument("--queue", default="normal", help="LSF queue / Slurm partition.")
    ap.add_argument("--mem-mb", type=int, default=16000)
    ap.add_argument("--cpus", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4,
                    help="HTTP workers within each build job (default 4).")
    ap.add_argument("--upload", action="store_true",
                    help="SFTP-upload fetch_jaspar.py from the local repo to --remote-dir.")
    args = ap.parse_args()

    try:
        from hpc_client import HPCClient
    except Exception as exc:
        print(f"Could not import hpc_client: {exc}")
        sys.exit(1)

    client = HPCClient()
    password = "" if args.key else getpass.getpass(f"Password for {args.user}@{args.host}: ")
    client.connect(args.host, args.user, password=password, port=args.port,
                   key_path=args.key, work_dir=args.remote_dir)
    if not client.connected:
        print("Connection failed — aborting.")
        sys.exit(1)

    scheduler = client.scheduler or "lsf"
    print(f"\nScheduler: {scheduler.upper()}   remote-dir: {args.remote_dir}")
    print(f"Cache dir: {args.cache_dir}")

    # Ensure the cache dir exists remotely.
    client.run_command(f"mkdir -p {shlex.quote(args.cache_dir)}", timeout=60)

    # Optionally upload the local fetch_jaspar.py so the remote repo doesn't
    # have to be synced first. (te_motif.py is assumed already present remotely.)
    if args.upload:
        local = Path(__file__).resolve().parent / "fetch_jaspar.py"
        if not local.exists():
            print(f"--upload requested but {local} not found locally.")
            sys.exit(1)
        remote = f"{args.remote_dir}/fetch_jaspar.py"
        print(f"Uploading {local.name} → {remote}")
        if not _put_remote_file(client, local.read_bytes(), remote):
            print("Upload failed — aborting.")
            sys.exit(1)

    submit_cmd = "bsub <" if scheduler == "lsf" else "sbatch"
    submitted = []
    for build in args.builds:
        script = _build_job_script(scheduler, build, args.remote_dir, args.cache_dir,
                                   args.mem_mb, args.cpus, args.queue, args.workers)
        remote_sh = f"{args.cache_dir}/submit_jaspar_{build}.sh"
        print(f"\n[{build}] writing job script → {remote_sh}")
        if not _put_remote_file(client, script.encode(), remote_sh):
            print(f"[{build}] could not write job script — skipping.")
            continue
        out, err, rc = client.run_command(f"{submit_cmd} {shlex.quote(remote_sh)}",
                                          timeout=120)
        line = (out or err).strip().splitlines()[-1] if (out or err).strip() else ""
        if rc == 0:
            print(f"[{build}] SUBMITTED: {line}")
            submitted.append(build)
        else:
            print(f"[{build}] submit FAILED (rc={rc}): {(err or out)[:200]}")

    print("\n" + "=" * 60)
    print(f"Submitted {len(submitted)}/{len(args.builds)} jobs: {', '.join(submitted) or '(none)'}")
    print("Watch progress with:")
    print(f"  bpeek -f <jobid>     # or:  tail -f {args.cache_dir}/fetch_<build>.out")
    print("When done, run the pipeline with:")
    print(f"  python3 query.py --local --family <FAM> --assembly <BUILD> "
          f"--output <OUT> --jaspar-dir {args.cache_dir}")
    print("=" * 60)

    client.disconnect()


if __name__ == "__main__":
    main()
