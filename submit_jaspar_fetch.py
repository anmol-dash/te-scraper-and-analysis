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
                      queue, workers, notify_email=""):
    """Generate a scheduler job script with NO wall-clock limit."""
    name = f"jaspar_{build}"
    out  = f"{cache_dir}/fetch_{build}.out"
    err  = f"{cache_dir}/fetch_{build}.err"
    cmd  = (f"python3 {shlex.quote(remote_dir)}/fetch_jaspar.py "
            f"--build {shlex.quote(build)} "
            f"--cache-dir {shlex.quote(cache_dir)} "
            f"--workers {workers}")
    if notify_email:
        # Each job emails when its own build finishes (runs on the cluster, so
        # the cluster home's ~/.hpc_te_state.json must hold the Gmail App Password).
        cmd += f" --notify-email {shlex.quote(notify_email)}"

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
    ap.add_argument("--notify-email", default="", metavar="EMAIL",
                    help="Each build job emails this address on completion (needs a Gmail "
                         "App Password in the cluster home's ~/.hpc_te_state.json).")
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

    submit_cmd = "bsub <" if scheduler == "lsf" else "sbatch"

    # Build ONE remote driver script that creates the cache dir, writes every
    # job script (base64-embedded), and submits them all. Running this in a
    # SINGLE SSH exec avoids the multi-session fragility that caused only the
    # first build to submit before the transport dropped.
    driver = ["set -uo pipefail", f"mkdir -p {shlex.quote(args.cache_dir)}"]

    if args.upload:
        local = Path(__file__).resolve().parent / "fetch_jaspar.py"
        if not local.exists():
            print(f"--upload requested but {local} not found locally.")
            sys.exit(1)
        b64 = base64.b64encode(local.read_bytes()).decode()
        remote_fetch = f"{args.remote_dir}/fetch_jaspar.py"
        driver.append(f'echo {b64} | base64 -d > {shlex.quote(remote_fetch)} '
                      f'&& echo "[upload] fetch_jaspar.py -> {remote_fetch}"')

    for build in args.builds:
        script = _build_job_script(scheduler, build, args.remote_dir, args.cache_dir,
                                   args.mem_mb, args.cpus, args.queue, args.workers,
                                   notify_email=args.notify_email)
        remote_sh = f"{args.cache_dir}/submit_jaspar_{build}.sh"
        b64 = base64.b64encode(script.encode()).decode()
        driver.append(f'echo {b64} | base64 -d > {shlex.quote(remote_sh)}')
        driver.append(f'echo "[{build}] submitting {remote_sh}"')
        # `|| true` so one bad submit never aborts the rest of the driver.
        driver.append(f'{submit_cmd} {shlex.quote(remote_sh)} || echo "[{build}] SUBMIT FAILED"')

    driver_script = "\n".join(driver) + "\n"
    print(f"\nSubmitting {len(args.builds)} jobs in a single SSH session: "
          f"{', '.join(args.builds)}")
    out, err, rc = client.run_command(driver_script)
    if out:
        print(out.rstrip())
    if err.strip():
        print("---- stderr ----")
        print(err.rstrip())

    # Honest reporting: failed builds print an explicit marker; job ids are
    # counted from the scheduler's own confirmation lines.
    blob = out + "\n" + err
    failed = [b for b in args.builds if f"[{b}] SUBMIT FAILED" in blob]
    n_jobids = blob.count("Job <") if scheduler == "lsf" else blob.count("Submitted batch job")

    print("\n" + "=" * 60)
    print(f"Driver exit rc={rc}.  Scheduler confirmations: {n_jobids}/{len(args.builds)}")
    if failed:
        print(f"  FAILED builds: {', '.join(failed)}")
    if n_jobids < len(args.builds):
        print("  Some builds may not have submitted — check the output above.")
    print("Watch progress with:")
    print(f"  bpeek -f <jobid>     # or:  tail -f {args.cache_dir}/fetch_<build>.out")
    print("When done, run the pipeline with:")
    print(f"  python3 query.py --local --family <FAM> --assembly <BUILD> "
          f"--output <OUT> --jaspar-dir {args.cache_dir}")
    print("=" * 60)

    client.disconnect()


if __name__ == "__main__":
    main()
