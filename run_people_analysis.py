#!/usr/bin/env python3
"""
run_people_analysis.py — submit one GAMECA ``query.py`` job per TE family into
per-person output folders, all from a single login-node invocation.

Layout produced under ``--base-dir``::

    <base>/Diego/<L1Md subfamily>/      # all L1Md* subfamilies (mm10)
    <base>/Claire/MT2_Mm/               # MT2_Mm (mm10)
    <base>/Claire/<B2 subfamily>/       # all B2* subfamilies (mm10)
    <base>/Katie/<IAP subfamily>/       # all IAP* subfamilies (mm10)
    <base>/Zach/AluSx1/                 # AluSx1 (hg38)

Subfamilies are discovered live from the RepeatMasker (rmsk) table for the
assembly (downloaded/cached via te_prep), so "all L1Md/B2/IAP subfamilies" tracks
whatever the assembly actually contains rather than a hard-coded list.

The ColabFold protein-fold calculation is heavy, so by default exactly ONE family
per group runs it (``--fold-scope group``) — the most-abundant subfamily of each
group is the representative.  Fold is only actually performed when a colabfold
command is supplied (``--colabfold-cmd`` / ``$COLABFOLD_CMD``); without it every
job skips structure prediction.

This script runs on the cluster LOGIN node (where ``bsub``/``sbatch`` and outbound
internet are available).  It detects LSF vs Slurm, writes one job script per
family under ``<base>/_jobs/``, submits them all, and writes a manifest.

Usage (on the login node, after you log in)::

    python run_people_analysis.py --base-dir /home/amodz/anmol/people_runs
    python run_people_analysis.py --base-dir OUT --dry-run        # show, don't submit
    python run_people_analysis.py --base-dir OUT \
        --colabfold-cmd colabfold_batch --genome-mm10 /ref/mm10.fa \
        --genome-hg38 /ref/hg38.fa --max-loci 2000
"""

import argparse
import datetime
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

# ── The plan ────────────────────────────────────────────────────────────────
# Each group → one person folder + either an explicit family list or a discovery
# regex matched against rmsk repNames for the assembly.
#   person   : output sub-folder name
#   label    : human label for logs / fold-representative grouping
#   assembly : genome build (drives rmsk table + std chroms)
#   discover : regex string matched against repName (mutually exclusive w/ families)
#   families : explicit repName list (mutually exclusive w/ discover)
PLAN = [
    {"person": "Diego",  "label": "L1Md", "assembly": "mm10", "discover": r"^L1Md"},
    {"person": "Claire", "label": "MT2_Mm", "assembly": "mm10", "families": ["MT2_Mm"]},
    {"person": "Claire", "label": "B2",   "assembly": "mm10", "discover": r"^B2(_|$)"},
    {"person": "Katie",  "label": "IAP",  "assembly": "mm10", "discover": r"^IAP"},
    {"person": "Zach",   "label": "AluSx1", "assembly": "hg38", "families": ["AluSx1"]},
]


def _std_chroms(assembly):
    """Standard chromosome set for an assembly (reuse te_prep's definitions)."""
    try:
        import te_prep
        if assembly in ("mm10", "mm39"):
            return te_prep.STD_CHROMS_MOUSE
        return te_prep.STD_CHROMS_HUMAN
    except Exception:
        if assembly in ("mm10", "mm39"):
            return set([f"chr{i}" for i in range(1, 20)] + ["chrX", "chrY"])
        return set([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"])


def discover_families(regex, assembly, rmsk_dir, min_count):
    """Return {repName: locus_count} for repNames matching *regex* in the assembly's
    rmsk table, restricted to standard chromosomes and to >= *min_count* loci."""
    import gzip
    import te_prep
    rmsk_dir = rmsk_dir or getattr(te_prep, "_DEFAULT_RMSK", None) \
        or os.environ.get("TE_RMSK_DIR")
    rmsk_path = te_prep.get_rmsk_path(assembly, rmsk_dir)
    rx = re.compile(regex)
    std = _std_chroms(assembly)
    counts = Counter()
    with gzip.open(rmsk_path, "rt") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) <= te_prep.COL_REPNAME:
                continue
            if std and fields[te_prep.COL_CHROM] not in std:
                continue
            name = fields[te_prep.COL_REPNAME]
            if rx.search(name):
                counts[name] += 1
    return {k: v for k, v in counts.items() if v >= min_count}


def resolve_group(group, rmsk_dir, min_count):
    """Resolve a PLAN group to a sorted list of (family, locus_count)."""
    if group.get("families"):
        # Explicit families: count loci so the manifest is informative, but never
        # drop an explicitly-requested family even if the count lookup fails.
        out = []
        for fam in group["families"]:
            try:
                fams = discover_families(rf"^{re.escape(fam)}$",
                                         group["assembly"], rmsk_dir, 1)
                out.append((fam, fams.get(fam, 0)))
            except Exception:
                out.append((fam, 0))
        return out
    fams = discover_families(group["discover"], group["assembly"],
                             rmsk_dir, min_count)
    # Largest first so the fold representative (index 0) is the most abundant.
    return sorted(fams.items(), key=lambda kv: (-kv[1], kv[0]))


def detect_scheduler():
    if shutil.which("bsub"):
        return "lsf"
    if shutil.which("sbatch"):
        return "slurm"
    return None


def _proxy_export_line():
    """If the login shell has an HTTPS proxy, forward it into the job so the UCSC
    fetch in te_prep works from compute nodes that only allow proxied egress."""
    p = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
    if not p:
        return ""
    q = shlex.quote(p)
    return (f"export https_proxy={q} http_proxy={q} "
            f"HTTPS_PROXY={q} HTTP_PROXY={q}\n")


def build_job_script(jobs_dir, script_dir, family, assembly, out_dir, do_fold,
                     args):
    """Write a per-family job script that runs query.py; return its path."""
    q = [args.python, str(Path(script_dir) / "query.py"),
         "--local", "--family", family, "--assembly", assembly,
         "--output", str(out_dir)]
    if args.source:
        q += ["--source", args.source]
    if args.rmsk_dir:
        q += ["--rmsk-dir", args.rmsk_dir]
    if args.max_loci:
        q += ["--max-loci", str(args.max_loci)]
    genome = args.genome_mm10 if assembly in ("mm10", "mm39") else args.genome_hg38
    if genome:
        q += ["--genome", genome]
    if do_fold and args.colabfold_cmd:
        q += ["--colabfold-cmd", args.colabfold_cmd]
    if args.extra_query_args:
        q += shlex.split(args.extra_query_args)
    qstr = " ".join(shlex.quote(str(c)) for c in q)

    script = jobs_dir / f"{out_dir.parent.name}_{family}.sh"
    with open(script, "w") as fh:
        fh.write("#!/usr/bin/env bash\n")
        fh.write("set -uo pipefail\n")
        # Silence the non-fatal conda-libmamba ICU load error on older nodes.
        fh.write("export CONDA_SOLVER=classic\n")
        fh.write(f"cd {shlex.quote(str(script_dir))}\n")
        prx = _proxy_export_line()
        if prx:
            fh.write(prx)
        fh.write('echo "Host: $(hostname)   Started: $(date)"\n')
        fh.write(f'echo "Family: {family}   Assembly: {assembly}   '
                 f'Fold: {"yes" if (do_fold and args.colabfold_cmd) else "no"}"\n')
        fh.write(f"mkdir -p {shlex.quote(str(out_dir))}\n")
        fh.write(qstr + "\n")
        fh.write("rc=$?\n")
        fh.write('echo "Finished: $(date)  rc=$rc"\n')
        fh.write("exit $rc\n")
    os.chmod(script, 0o755)
    return script


def submit(scheduler, job_script, job_name, out_dir, args):
    """Submit one job; return (job_id_or_None, submit_cmd_string)."""
    log_out = out_dir / "sched.%J.out" if scheduler == "lsf" else out_dir / "sched.out"
    log_err = out_dir / "sched.%J.err" if scheduler == "lsf" else out_dir / "sched.err"
    if scheduler == "lsf":
        cmd = ["bsub", "-J", job_name, "-q", args.queue,
               "-n", str(args.cores),
               "-M", str(args.mem_mb), "-R", f"rusage[mem={args.mem_mb}]",
               "-W", args.wall, "-o", str(log_out), "-e", str(log_err),
               "bash", str(job_script)]
    else:  # slurm
        cmd = ["sbatch", "-J", job_name, "-p", args.queue,
               "-n", "1", "-c", str(args.cores),
               f"--mem={args.mem_mb}", "-t", args.wall,
               "-o", str(out_dir / "sched.%j.out"),
               "-e", str(out_dir / "sched.%j.err"),
               str(job_script)]
    cmd_str = " ".join(shlex.quote(c) for c in cmd)
    if args.dry_run:
        return None, cmd_str
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = (res.stdout or "") + (res.stderr or "")
    job_id = None
    m = re.search(r"Job <(\d+)>", out) or re.search(r"Submitted batch job (\d+)", out)
    if m:
        job_id = m.group(1)
    else:
        print(f"  WARNING: could not parse job id from: {out.strip()[:200]}")
    return job_id, cmd_str


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-dir", required=True,
                   help="Top-level output dir; per-person sub-folders are created under it.")
    p.add_argument("--rmsk-dir", default=os.environ.get("TE_RMSK_DIR"),
                   help="Dir holding rmsk_<build>.txt.gz (also passed to query.py). "
                        "Default: te_prep's default / $TE_RMSK_DIR.")
    p.add_argument("--colabfold-cmd", default=os.environ.get("COLABFOLD_CMD", ""),
                   help="colabfold_batch command/path; enables the fold calc on the "
                        "fold representative(s). Empty => no job folds.")
    p.add_argument("--fold-scope", choices=["group", "folder", "none"],
                   default="group",
                   help="Which jobs get the fold calc: one per family group "
                        "(default), one per person folder, or none.")
    p.add_argument("--genome-mm10", default=os.environ.get("GENOME_MM10", ""),
                   help="mm10 genome FASTA for in-memory primer search (optional).")
    p.add_argument("--genome-hg38", default=os.environ.get("GENOME_HG38", ""),
                   help="hg38 genome FASTA for in-memory primer search (optional).")
    p.add_argument("--max-loci", type=int, default=None,
                   help="Cap loci per family (default: no cap = full family).")
    p.add_argument("--min-count", type=int, default=10,
                   help="Skip discovered subfamilies with fewer than this many loci "
                        "(default: 10). Explicit families are never skipped.")
    p.add_argument("--source", choices=["rmsk", "dfam"], default="rmsk")
    p.add_argument("--queue", default=os.environ.get("QUEUE", "normal"))
    p.add_argument("--cores", type=int, default=int(os.environ.get("N_CORES", "4")))
    p.add_argument("--mem-mb", type=int, default=int(os.environ.get("MEM_MB", "24000")))
    p.add_argument("--wall", default=os.environ.get("WALL", "24:00"),
                   help="Wall-clock limit (LSF HH:MM / Slurm time string).")
    p.add_argument("--extra-query-args", default=os.environ.get("EXTRA_QUERY_ARGS", ""),
                   help="Extra args appended verbatim to every query.py call "
                        "(e.g. '--skip-genome').")
    p.add_argument("--python", default=shutil.which("python3") or shutil.which("python")
                   or sys.executable)
    p.add_argument("--scheduler", choices=["lsf", "slurm"], default=None,
                   help="Force scheduler (default: auto-detect bsub/sbatch).")
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve families and print the submit commands without submitting.")
    args = p.parse_args()

    scheduler = args.scheduler or detect_scheduler()
    if scheduler is None and not args.dry_run:
        sys.exit("ERROR: neither bsub nor sbatch found. Run on a login node, or use "
                 "--dry-run, or pass --scheduler.")
    if scheduler is None:
        scheduler = "lsf"  # dry-run preview default

    script_dir = Path(__file__).resolve().parent
    base = Path(args.base_dir).resolve()
    jobs_dir = base / "_jobs"
    base.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GAMECA per-person family submission")
    print(f"  Base dir:   {base}")
    print(f"  Scheduler:  {scheduler}{'  (DRY RUN)' if args.dry_run else ''}")
    print(f"  Fold scope: {args.fold_scope}"
          f"{'  (no colabfold-cmd => fold disabled)' if not args.colabfold_cmd else ''}")
    print(f"  Resources:  -q {args.queue} -n {args.cores} -M {args.mem_mb} -W {args.wall}")
    print("=" * 70)

    submitted, folded_persons = [], set()
    for group in PLAN:
        person, label, assembly = group["person"], group["label"], group["assembly"]
        try:
            fam_counts = resolve_group(group, args.rmsk_dir, args.min_count)
        except SystemExit:
            raise
        except Exception as e:
            print(f"  [{person}/{label}] DISCOVERY FAILED: {e}")
            continue
        if not fam_counts:
            print(f"  [{person}/{label}] no families matched — skipping")
            continue

        print(f"\n[{person}/{label}] ({assembly}) {len(fam_counts)} family(ies):")
        for fam, cnt in fam_counts:
            print(f"    {fam:30s} {cnt:>8,} loci")

        for idx, (fam, cnt) in enumerate(fam_counts):
            out_dir = base / person / fam
            out_dir.mkdir(parents=True, exist_ok=True)
            # Fold representative = most-abundant family in the group (idx 0).
            do_fold = False
            if args.fold_scope == "group":
                do_fold = (idx == 0)
            elif args.fold_scope == "folder":
                do_fold = (idx == 0 and person not in folded_persons)
            if do_fold:
                folded_persons.add(person)

            job_script = build_job_script(jobs_dir, script_dir, fam, assembly,
                                          out_dir, do_fold, args)
            job_name = f"gam_{person}_{fam}"[:60]
            job_id, cmd_str = submit(scheduler, job_script, job_name, out_dir, args)
            tag = " [FOLD]" if (do_fold and args.colabfold_cmd) else \
                  (" [fold-rep, no colabfold-cmd]" if do_fold else "")
            print(f"    submit {fam:30s} -> {out_dir}"
                  f"  job={job_id or '(dry-run)'}{tag}")
            submitted.append({
                "person": person, "label": label, "assembly": assembly,
                "family": fam, "loci": cnt, "fold": bool(do_fold and args.colabfold_cmd),
                "fold_representative": do_fold, "out_dir": str(out_dir),
                "job_id": job_id, "job_script": str(job_script),
                "submit_cmd": cmd_str,
            })

    manifest = base / "people_analysis_manifest.json"
    manifest.write_text(json.dumps({
        "created": datetime.datetime.now().isoformat(),
        "base_dir": str(base),
        "scheduler": scheduler,
        "dry_run": args.dry_run,
        "fold_scope": args.fold_scope,
        "colabfold_cmd": args.colabfold_cmd or None,
        "max_loci": args.max_loci,
        "n_jobs": len(submitted),
        "jobs": submitted,
    }, indent=2))

    n_fold = sum(1 for j in submitted if j["fold"])
    print("\n" + "=" * 70)
    print(f"  {'Would submit' if args.dry_run else 'Submitted'}: {len(submitted)} jobs "
          f"({n_fold} with fold)")
    print(f"  Manifest: {manifest}")
    if scheduler == "lsf" and not args.dry_run:
        print("  Track with:  bjobs -J 'gam_*'")
    elif scheduler == "slurm" and not args.dry_run:
        print("  Track with:  squeue -u $USER")
    print("=" * 70)


if __name__ == "__main__":
    main()
