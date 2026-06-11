#!/usr/bin/env python3
"""
run_stage11_all.py — run every GAMECA Stage 11 ("standout analysis") module
against a single input CSV, outside the main pipeline.

This mirrors query.py's `_run_standout_analysis` / `_MODULES` registry exactly
(see query.py around the "STAGE 11 HELPER" section) so the standalone and
in-pipeline code paths stay in sync. It exists so you can point all Stage 11
modules at an arbitrary loci/expression CSV (e.g. <family>_ultracombo.csv)
without first running the clustering pipeline or being on the HPC cluster.

Each module is a sibling run_*.py script invoked with:
    --input <csv> --reports-dir <dir> --family <name> [module-specific extras]

The input CSV needs a `Seq` column (per-copy sequence); a `Cluster` column is
used when present so cluster-aware modules (phylo tree, DE, etc.) run fully.

Usage:
    python run_stage11_all.py --input <loci.csv> --family <NAME> \
        [--assembly hg38] [--reports-dir ./stage11_outputs]

Preset (MT2_Mm, mouse mm10):
    python run_stage11_all.py --input mt2_mm_ultracombo.csv \
        --family MT2_Mm --assembly mm10 --reports-dir stage11_mt2_mm
"""

import argparse
import datetime
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _build_modules(args, cons_fa: Path):
    """Return the Stage 11 module registry as (name, script, extra_args) tuples.

    Mirrors query.py `_MODULES`. Optional knobs fall back to each module's own
    default exactly as the pipeline does (only passed when the user set them).
    """
    _opt = lambda flag, val: [flag, str(val)] if val is not None else []
    _opt_list = lambda flag, vals: ([flag] + [str(v) for v in vals]) if vals else []
    _cons = (["--consensus-fasta", str(cons_fa)] if cons_fa and cons_fa.exists() else [])
    asm = args.assembly

    return [
        ("phylo",         "run_phylo_analysis.py",
         ["--subst-rate", str(args.subst_rate),
          "--clock-divisor", str(args.clock_divisor),
          "--intact-orf-aa", str(args.intact_orf_aa)]
         + _opt("--mafft-cmd", args.mafft_cmd)),
        ("grna",          "run_grna_offtarget.py",
         ["--cas", str(args.grna_cas), "--max-mm", str(args.grna_max_mm)]
         + _opt("--background", args.grna_background)),
        ("transduction",  "run_transduction.py",
         ["--tail-bp", str(args.tail_bp), "--min-shared", "3"]),
        ("antisense",     "run_antisense_promoter.py",
         ["--promoter-bp", str(args.promoter_bp)]),
        ("ctcf_tad",      "run_ctcf_tad.py",
         ["--motif-mismatch", "3"]
         + _opt("--ctcf-preset", args.ctcf_preset)
         + _opt("--tads-preset", args.tads_preset)),
        ("epigenetic",    "run_epigenetic_overlay.py",
         _opt("--preset", args.epigenetic_preset)),
        ("ortholog",      "run_ortholog_insertion.py",
         _opt_list("--species", args.ortholog_species)
         + _opt("--liftover-cmd", args.liftover_cmd)),
        ("multiassembly", "run_multiassembly_liftover.py",
         ["--source-assembly", asm]
         + _opt_list("--target-assemblies", args.target_assemblies)
         + _opt("--liftover-cmd", args.liftover_cmd)),
        ("fold",          "run_fold_prediction.py",
         ["--per-cluster", "--min-aa", "100", "--top-n", "5"]
         + _opt("--colabfold-cmd", args.colabfold_cmd)
         + _cons),
        ("divergence",    "run_divergence.py",
         ["--assembly", asm]
         + _opt("--cpg-omega", args.cpg_omega)
         + _cons),
        ("ltr_struct",    "run_ltr_struct.py",
         _opt("--min-ltr-identity", args.min_ltr_identity)),
        ("subfamily",     "run_subfamily.py",
         ["--assembly", asm]
         + _opt("--cpg-omega", args.cpg_omega)
         + _cons),
        ("benchmark",     "run_benchmark.py",
         ["--assembly", asm]),
        ("motif_gain",    "run_motif_gain.py",
         ["--assembly", asm]
         + _cons),
        ("expression",    "te_expression.py",
         ["--out-dir", str(args.reports_dir.parent)]),
    ]


def main():
    p = argparse.ArgumentParser(
        description="Run all GAMECA Stage 11 standout-analysis modules on one CSV.")
    p.add_argument("--input", required=True,
                   help="Loci CSV (needs a Seq column; Cluster column used if present).")
    p.add_argument("--family", required=True, help="TE family name (e.g. MT2_Mm).")
    p.add_argument("--assembly", default="hg38",
                   help="Genome assembly (e.g. hg38, mm10). Default: hg38.")
    p.add_argument("--reports-dir", default="stage11_outputs",
                   help="Output directory for figures/.tex (default: stage11_outputs).")
    p.add_argument("--consensus-fasta", default=None,
                   help="Optional all_cluster_consensuses.fa; passed to modules that use it.")
    p.add_argument("--only", default=None,
                   help="Comma-separated module names to run (default: all).")
    p.add_argument("--skip", default=None,
                   help="Comma-separated module names to skip.")
    # Module knobs (default None → each module's own default, matching query.py).
    p.add_argument("--subst-rate", default="2.2e-9")
    p.add_argument("--clock-divisor", default="2")
    p.add_argument("--intact-orf-aa", default="100")
    p.add_argument("--mafft-cmd", default=None)
    p.add_argument("--grna-cas", default="SpCas9")
    p.add_argument("--grna-max-mm", default="2")
    p.add_argument("--grna-background", default=None)
    p.add_argument("--tail-bp", default="150")
    p.add_argument("--promoter-bp", default="200")
    p.add_argument("--ctcf-preset", default=None)
    p.add_argument("--tads-preset", default=None)
    p.add_argument("--epigenetic-preset", default=None)
    p.add_argument("--ortholog-species", nargs="*", default=None)
    p.add_argument("--liftover-cmd", default=None)
    p.add_argument("--target-assemblies", nargs="*", default=None)
    p.add_argument("--colabfold-cmd", default=None)
    p.add_argument("--min-ltr-identity", default=None)
    p.add_argument("--cpg-omega", default=None)
    args = p.parse_args()

    input_csv = Path(args.input).resolve()
    if not input_csv.exists():
        sys.exit(f"ERROR: input CSV not found: {input_csv}")

    args.reports_dir = Path(args.reports_dir).resolve()
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    sdir = Path(__file__).resolve().parent
    py = shutil.which("python3") or shutil.which("python") or sys.executable
    cons_fa = Path(args.consensus_fasta).resolve() if args.consensus_fasta else None

    modules = _build_modules(args, cons_fa)
    only = {m.strip() for m in args.only.split(",")} if args.only else None
    skip = {m.strip() for m in args.skip.split(",")} if args.skip else set()
    if only:
        modules = [m for m in modules if m[0] in only]
    if skip:
        modules = [m for m in modules if m[0] not in skip]

    log_path = args.reports_dir / "standout_analysis.log"

    print("=" * 60)
    print("GAMECA STAGE 11 — STANDOUT ANALYSIS (standalone)")
    print("=" * 60)
    print(f"  Family:   {args.family}")
    print(f"  Assembly: {args.assembly}")
    print(f"  Input:    {input_csv}")
    print(f"  Reports:  {args.reports_dir}")
    print(f"  Log:      {log_path}")
    print(f"  Modules:  {len(modules)}")
    if cons_fa and not cons_fa.exists():
        print(f"  NOTE: consensus FASTA {cons_fa} not found — modules will run without it.")
    print("=" * 60)

    ok = fail = skipped = 0
    results: list[tuple[str, str, float]] = []

    with open(log_path, "w", buffering=1) as lf:
        lf.write("GAMECA Stage 11 Standalone Log\n")
        lf.write(f"Family:   {args.family}\n")
        lf.write(f"Assembly: {args.assembly}\n")
        lf.write(f"Input:    {input_csv}\n")
        lf.write(f"Python:   {py}\n")
        lf.write(f"Started:  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
        lf.write("=" * 60 + "\n")

        for name, script, extra in modules:
            script_path = sdir / script
            lf.write(f"\n{'='*60}\n[{name}]\n{'='*60}\n")
            lf.flush()
            if not script_path.exists():
                msg = f"  [{name}] SKIP (script not found: {script_path})"
                print(msg); lf.write(msg + "\n")
                skipped += 1
                results.append((name, "SKIP", 0.0))
                continue

            cmd = ([py, str(script_path),
                    "--input",       str(input_csv),
                    "--reports-dir", str(args.reports_dir),
                    "--family",      args.family]
                   + extra)
            lf.write(f"CMD: {' '.join(cmd)}\n\n"); lf.flush()
            print(f"  [{name}] running...", end="", flush=True)
            t0 = time.time()
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
            elapsed = time.time() - t0
            lf.write(f"\n[EXIT {rc}  {elapsed:.0f}s]\n"); lf.flush()
            if rc == 0:
                print(f" ok ({elapsed:.0f}s)"); ok += 1
                results.append((name, "OK", elapsed))
            else:
                print(f" FAILED rc={rc} ({elapsed:.0f}s)"); fail += 1
                results.append((name, f"FAILED rc={rc}", elapsed))

        lf.write(f"\n{'='*60}\nSTAGE 11 SUMMARY\n{'='*60}\n")
        for nm, st, el in results:
            lf.write(f"  {nm:<20} {st:<18} {el:.0f}s\n")
        lf.write(f"\nDone: ok={ok} failed={fail} skipped={skipped}\n")
        lf.write(f"Finished: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for nm, st, el in results:
        print(f"  {nm:<20} {st:<18} {el:.0f}s")
    print(f"\n  Done: {ok} ok / {fail} failed / {skipped} skipped")
    if fail:
        print("  FAILED modules: "
              + ", ".join(n for n, s, _ in results if s.startswith("FAILED")))
    print(f"  Outputs in: {args.reports_dir}")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
