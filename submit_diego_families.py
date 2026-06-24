#!/usr/bin/env python3
"""
submit_diego_families.py

Submits one pair of bsub jobs per TE family from the Diego enrichment list:

  Job A  gamXXX_main   — query.py core pipeline (clustering, consensus,
                          primers) + all Stage 11 modules EXCEPT ColabFold
                          → emails anmoldash@gmail.com when done, noting
                            that ColabFold is still running.

  Job B  gamXXX_fold   — ColabFold structure prediction (Stage 11 fold
                          module only); has an LSF dependency on Job A.
                          → emails anmoldash@gmail.com when done.

Each family is a separate entry in bjobs.  Run this script from the
directory that contains query.py, run_stage11_all.py, etc.

Usage (on the cluster login node):
    python3 submit_diego_families.py [--dry-run]

Flags:
    --dry-run   Print generated job scripts without submitting anything.
    --outdir    Override output root (default: /home/amodz/anmol/diegojfamiliesv1).
    --queue     LSF queue name (default: normal).
    --colabfold-cmd  Path to colabfold_batch binary (default: colabfold_batch).
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# ── Family list ───────────────────────────────────────────────────────────────
# Exactly the 10 primary families the user asked to run as the top-level jobs.
# The "enriched over" relatives are captured by the analysis of the primary family
# (they share sequence similarity and are noted in the pipeline report).
#
# GSAT and HSAT4 are tandem satellite repeats rather than dispersed TEs.
# The pipeline handles them — clustering + consensus + primers will work, though
# cluster sizes may be large and sequences short.  Stage 11 modules that expect
# ERV/LTR structure (ltr_struct) will gracefully report N/A for these.

FAMILIES = [
    "THE1A",      # + THE1B/C/D and their -int regions
    "MSTA1",      # + MSTA1-int, MSTB-int, MST-int
    "HERV9-int",  # + HERVK9-int, HERV9N-int, HERV9NC-int
    "LTR1D",      # + LTR1D1
    "LTR12C",     # + LTR12F, LTR12E, LTR12D
    "LTR2752",
    "MER52A",     # + MER52C, MER52D, MER52-int
    "MER66D",
    "GSAT",       # satellite — + GSATII, GSATX (primer design included)
    "HSAT4",      # satellite — + HSAT5   (primer design included)
]

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_OUTDIR       = "/home/amodz/anmol/diegojfamiliesv1"
DEFAULT_QUEUE        = "normal"
DEFAULT_COLABFOLD    = "colabfold_batch"
ASSEMBLY             = "hg38"
NOTIFY_EMAIL         = "anmoldash@gmail.com"

# LSF resources
MAIN_CPUS   = 16
MAIN_MEM_MB = 24000   # 24 GB — clustering + consensus + full Stage 11
MAIN_WALL   = "12:00" # HH:MM

FOLD_CPUS   = 8
FOLD_MEM_MB = 16000   # 16 GB — ColabFold CPU mode
FOLD_WALL   = "24:00"

# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_name(family: str) -> str:
    """LSF job name: letters/digits/underscores only, ≤ 30 chars."""
    return re.sub(r"[^A-Za-z0-9_]", "_", family)[:30]


def fam_lower(family: str) -> str:
    """Directory name query.py uses: family.lower()."""
    return family.lower()


def venv_activation_block() -> str:
    """Shell fragment that activates the GAMECA venv if present, else falls
    back to system Python.  Mirrors hpc_client.py's _venv_setup_block."""
    return (
        'GAMECA_VENV="${GAMECA_VENV:-$HOME/gameca_venv}"\n'
        'if [ -f "$GAMECA_VENV/bin/activate" ]; then\n'
        '    source "$GAMECA_VENV/bin/activate"\n'
        '    echo "[GAMECA] venv activated: $(python --version 2>&1)"\n'
        'else\n'
        '    echo "[GAMECA] venv not found at $GAMECA_VENV — using system Python"\n'
        'fi\n'
        'PYTHON=$(command -v python3 || command -v python)\n'
    )


def mafft_block() -> str:
    return (
        "module load mafft 2>/dev/null || module load MAFFT 2>/dev/null || true\n"
        'command -v mafft >/dev/null 2>&1 || echo "[GAMECA] WARNING: mafft not on PATH"\n'
    )


def make_main_script(family: str, outdir: str, queue: str, workdir: str) -> str:
    sn      = safe_name(family)
    fl      = fam_lower(family)
    fam_dir = f"{outdir}/{fl}"
    # Path to the clustered CSV that query.py writes after Stage 7
    clustered_csv = f"{fam_dir}/01_data/{fl}_clustered.csv"
    reports_dir   = f"{fam_dir}/reports"
    log_out = f"{fam_dir}/bsub_main.out"
    log_err = f"{fam_dir}/bsub_main.err"
    venv    = venv_activation_block()
    mafft   = mafft_block()

    # Directives must be at column 0 — build without indentation
    lines = [
        "#!/bin/bash",
        f"#BSUB -J gam_{sn}_main",
        f"#BSUB -o {log_out}",
        f"#BSUB -e {log_err}",
        f"#BSUB -n {MAIN_CPUS}",
        f"#BSUB -M {MAIN_MEM_MB}",
        f"#BSUB -W {MAIN_WALL}",
        f"#BSUB -q {queue}",
        "",
        "set -euo pipefail",
        'echo "============================================================"',
        f'echo " GAMECA — {family}  (main job)"',
        'echo " Job ID : $LSB_JOBID"',
        'echo " Host   : $(hostname)"',
        'echo " Started: $(date)"',
        'echo "============================================================"',
        "",
        venv,
        mafft,
        f"mkdir -p {fam_dir}",
        f"cd {workdir}",
        "",
        f"# ── Stage 1-10: core pipeline (clustering, consensus, primers) ──",
        f'echo "[$(date +%H:%M:%S)] Running core pipeline for {family}..."',
        f'"$PYTHON" query.py \\',
        f'    --local \\',
        f'    --family "{family}" \\',
        f'    --assembly {ASSEMBLY} \\',
        f'    --output  {outdir} \\',
        f'    --skip-standout \\',
        f'    --skip-tsne',
        "",
        f'echo "[$(date +%H:%M:%S)] Core pipeline done."',
        "",
        f"# ── Stage 11: all modules except ColabFold fold prediction ───────",
        f'echo "[$(date +%H:%M:%S)] Running Stage 11 (no ColabFold)..."',
        f'"$PYTHON" run_stage11_all.py \\',
        f'    --input       "{clustered_csv}" \\',
        f'    --family      "{family}" \\',
        f'    --assembly    {ASSEMBLY} \\',
        f'    --reports-dir "{reports_dir}" \\',
        f'    --skip        fold \\',
        f'    --notify-email "{NOTIFY_EMAIL}"',
        "",
        f'echo "[$(date +%H:%M:%S)] {family} main job complete."',
    ]
    return "\n".join(lines) + "\n"


def make_fold_script(family: str, outdir: str, queue: str,
                     workdir: str, colabfold_cmd: str) -> str:
    sn      = safe_name(family)
    fl      = fam_lower(family)
    fam_dir = f"{outdir}/{fl}"
    clustered_csv = f"{fam_dir}/01_data/{fl}_clustered.csv"
    reports_dir   = f"{fam_dir}/reports"
    log_out = f"{fam_dir}/bsub_fold.out"
    log_err = f"{fam_dir}/bsub_fold.err"
    cons_fa = f"{fam_dir}/cluster_alignments/all_cluster_consensuses.fa"
    venv    = venv_activation_block()

    lines = [
        "#!/bin/bash",
        f"#BSUB -J gam_{sn}_fold",
        f"#BSUB -o {log_out}",
        f"#BSUB -e {log_err}",
        f"#BSUB -n {FOLD_CPUS}",
        f"#BSUB -M {FOLD_MEM_MB}",
        f"#BSUB -W {FOLD_WALL}",
        f"#BSUB -q {queue}",
        "",
        "set -euo pipefail",
        'echo "============================================================"',
        f'echo " GAMECA — {family}  (ColabFold job)"',
        'echo " Job ID : $LSB_JOBID"',
        'echo " Host   : $(hostname)"',
        'echo " Started: $(date)"',
        'echo "============================================================"',
        "",
        venv,
        f"cd {workdir}",
        "",
        f'echo "[$(date +%H:%M:%S)] Running ColabFold structure prediction for {family}..."',
        f'"$PYTHON" run_stage11_all.py \\',
        f'    --input           "{clustered_csv}" \\',
        f'    --family          "{family}" \\',
        f'    --assembly        {ASSEMBLY} \\',
        f'    --reports-dir     "{reports_dir}" \\',
        f'    --consensus-fasta "{cons_fa}" \\',
        f'    --only            fold \\',
        f'    --colabfold-cmd   "{colabfold_cmd}" \\',
        f'    --notify-email    "{NOTIFY_EMAIL}"',
        "",
        f'echo "[$(date +%H:%M:%S)] {family} ColabFold job complete."',
    ]
    return "\n".join(lines) + "\n"


def submit(script_text: str, dry_run: bool,
           dependency_job_id: str | None = None) -> str | None:
    """Write script to a temp file and bsub it.  Returns the LSF job ID string."""
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", delete=False, prefix="gameca_"
    ) as fh:
        fh.write(script_text)
        tmp = fh.name
    os.chmod(tmp, 0o755)

    cmd = ["bsub"]
    if dependency_job_id:
        cmd += ["-w", f"done({dependency_job_id})"]
    cmd += ["<", tmp]   # bsub reads from stdin

    if dry_run:
        print(f"\n{'='*60}")
        print(f"[DRY RUN] Would submit:\n{script_text}")
        print(f"Command: bsub {'<' + tmp}")
        return "DRY_RUN_ID"

    # bsub < file needs shell=True (stdin redirect)
    shell_cmd = " ".join(cmd[:-2]) + f" < {tmp}"
    result = subprocess.run(shell_cmd, shell=True, capture_output=True, text=True)
    os.unlink(tmp)

    if result.returncode != 0:
        print(f"  ERROR submitting: {result.stderr.strip()}", file=sys.stderr)
        return None

    # LSF prints: Job <12345> is submitted to queue <normal>.
    m = re.search(r"Job <(\d+)>", result.stdout)
    job_id = m.group(1) if m else None
    print(f"  Submitted: {result.stdout.strip()} → job_id={job_id}")
    return job_id


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run",       action="store_true",
                   help="Print scripts without submitting.")
    p.add_argument("--outdir",        default=DEFAULT_OUTDIR)
    p.add_argument("--queue",         default=DEFAULT_QUEUE)
    p.add_argument("--colabfold-cmd", default=DEFAULT_COLABFOLD)
    args = p.parse_args()

    workdir = str(Path(__file__).resolve().parent)

    # Ensure output root exists (login node can create it; compute nodes will
    # also mkdir -p the family subdir in the job script).
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    print(f"Output root : {args.outdir}")
    print(f"Queue       : {args.queue}")
    print(f"ColabFold   : {args.colabfold_cmd}")
    print(f"Notify      : {NOTIFY_EMAIL}")
    print(f"Families    : {len(FAMILIES)}")
    print(f"Total jobs  : {len(FAMILIES) * 2}  ({len(FAMILIES)} main + {len(FAMILIES)} fold)")
    print()

    submitted = []
    failed    = []

    for family in FAMILIES:
        print(f"[{family}]")

        main_script = make_main_script(family, args.outdir, args.queue, workdir)
        fold_script = make_fold_script(family, args.outdir, args.queue,
                                       workdir, args.colabfold_cmd)

        main_id = submit(main_script, args.dry_run)
        if main_id is None:
            print(f"  !! Failed to submit main job for {family}")
            failed.append(family)
            continue

        fold_id = submit(fold_script, args.dry_run, dependency_job_id=main_id)
        if fold_id is None:
            print(f"  !! Failed to submit fold job for {family} (main={main_id} still running)")
            failed.append(f"{family}(fold)")
            submitted.append((family, main_id, None))
        else:
            submitted.append((family, main_id, fold_id))

        print()

    print("=" * 60)
    print(f"Submitted {len(submitted)} families:")
    for fam, mid, fid in submitted:
        fold_str = fid if fid else "FAILED"
        print(f"  {fam:<16}  main={mid:<10}  fold={fold_str}")
    if failed:
        print(f"\nFailed to submit: {', '.join(failed)}")
    print()
    print("Monitor with:  bjobs -w")
    print(f"Outputs in  :  {args.outdir}/<family>/")
    print(f"  Core log  :  <family>/bsub_main.out")
    print(f"  Fold log  :  <family>/bsub_fold.out")
    print(f"  Stage 11  :  <family>/reports/standout_analysis.log")


if __name__ == "__main__":
    main()
