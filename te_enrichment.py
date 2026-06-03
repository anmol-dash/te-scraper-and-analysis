#!/usr/bin/env python3
"""
te_enrichment.py  —  GAMECA enrichment orchestrator
─────────────────────────────────────────────────────────────────────────────
Chains the three enrichment steps in order:

  Step M — te_motif.py      JASPAR overlap + Fisher enrichment
  Step G — te_go.py         GO annotation via mygene.info
  Step E — te_expression.py Expression boxplots

Each step can also be run independently:

  python te_motif.py     --input clustered.csv --build hg38 --out-dir ./results
  python te_go.py        --enrichment-dir ./results/enrichment_results --build hg38
  python te_expression.py --input clustered.csv --out-dir ./results

Usage:
    python te_enrichment.py \\
        --input  ./results/clustered.csv \\
        --build  hg38 \\
        --out-dir ./results

    # Skip individual steps:
    python te_enrichment.py --input data.csv --skip-go
    python te_enrichment.py --input data.csv --skip-expression

    # Re-run specific steps only:
    python te_enrichment.py --input data.csv --only motif

    # Check what has run:
    python te_enrichment.py --input data.csv --status

HPC (LSF):
    bsub -q normal -M 12000 -n 4 -o enrich.log \\
        python te_enrichment.py --input clustered.csv --build hg38

HPC (Slurm):
    sbatch --mem=12G --cpus-per-task=4 -o enrich.log \\
        --wrap "python te_enrichment.py --input clustered.csv --build hg38"
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# ── Step imports ──────────────────────────────────────────────────────────────
from te_motif      import run_motif_analysis
from te_go         import run_go_annotation
from te_expression import run_expression_analysis


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _cp_path(out_dir):
    return Path(out_dir) / "enrichment_checkpoints.json"


def load_checkpoints(out_dir):
    p = _cp_path(out_dir)
    return json.loads(p.read_text()) if p.exists() else {}


def save_checkpoint(out_dir, step, metadata=None):
    p  = _cp_path(out_dir)
    cp = load_checkpoints(out_dir)
    cp[step] = {"completed_at": datetime.now().isoformat(), "metadata": metadata or {}}
    p.write_text(json.dumps(cp, indent=2))
    print(f"  [checkpoint] '{step}' saved")


def _is_done(cp, step, force_steps):
    return step not in force_steps and step in cp


def _print_status(out_dir, steps):
    cp = load_checkpoints(out_dir)
    print(f"\nEnrichment status — {out_dir}")
    for step in steps:
        if step in cp:
            ts   = cp[step]["completed_at"]
            meta = cp[step].get("metadata", {})
            ms   = ", ".join(f"{k}={v}" for k, v in meta.items())
            print(f"  ✓ {step:<14} {ts}" + (f"  [{ms}]" if ms else ""))
        else:
            print(f"  – {step:<14} (not run)")
    remaining = [s for s in steps if s not in cp]
    print(f"\n  Remaining: {remaining if remaining else 'none'}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="GAMECA enrichment pipeline (motif + GO + expression)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input",       required=True, help="Clustered CSV (from te_clustering.py)")
    p.add_argument("--build",       default="hg38",
                   help="Genome build: hg38/hg19/mm10/mm39 (default: hg38)")
    p.add_argument("--out-dir",     default=".",   help="Output directory")
    p.add_argument("--jaspar-bed",  default=None,
                   help="JASPAR BED path (auto-downloaded if omitted)")
    p.add_argument("--jaspar-dir",  default=None,
                   help="Directory to cache JASPAR BED files")
    p.add_argument("--p-threshold", type=float, default=0.05)

    # Step toggles
    p.add_argument("--skip-motif",      action="store_true")
    p.add_argument("--skip-go",         action="store_true")
    p.add_argument("--skip-expression", action="store_true")
    p.add_argument("--only",            metavar="STEP",
                   help="Run only this step: motif | go | expression")

    # Re-run / status
    p.add_argument("--force",       action="store_true", help="Re-run all steps")
    p.add_argument("--force-steps", nargs="+", default=[], metavar="STEP",
                   help="Re-run specific steps (motif go expression)")
    p.add_argument("--status",      action="store_true",
                   help="Print checkpoint status and exit")
    p.add_argument("--notify-email", default="", metavar="EMAIL",
                   help="Send a completion email to this address (requires Gmail App Password setup).")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_steps    = ["motif", "go", "expression"]
    force_steps  = set(all_steps) if args.force else set(args.force_steps)

    if args.status:
        _print_status(out_dir, all_steps)
        return

    if args.only:
        if args.only not in all_steps:
            print(f"Unknown step '{args.only}'. Choose from: {all_steps}")
            sys.exit(1)
        run_single = args.only
        skip = {s for s in all_steps if s != run_single}
    else:
        run_single = None
        skip = set()
        if args.skip_motif:      skip.add("motif")
        if args.skip_go:         skip.add("go")
        if args.skip_expression: skip.add("expression")

    cp      = load_checkpoints(out_dir)
    enrich_dir = out_dir / "enrichment_results"
    jaspar_dir = args.jaspar_dir or str(Path(out_dir) / "jaspar")

    print("=" * 60)
    print("GAMECA — Enrichment Pipeline")
    print(f"  Input:  {args.input}")
    print(f"  Build:  {args.build}")
    print(f"  OutDir: {out_dir}")
    print(f"  Steps:  {[s for s in all_steps if s not in skip]}")
    print("=" * 60)

    # ── M: Motif analysis ─────────────────────────────────────────────────────
    motif_result = None
    if "motif" not in skip:
        if _is_done(cp, "motif", force_steps):
            print(f"\n[SKIP] motif: {cp['motif']['completed_at']}")
        else:
            print("\n" + "=" * 60)
            print("STEP M — Motif Analysis")
            print("=" * 60)
            motif_result = run_motif_analysis(
                input_csv      = args.input,
                build          = args.build,
                out_dir        = str(out_dir),
                jaspar_bed_arg = args.jaspar_bed,
                jaspar_dir     = jaspar_dir,
                p_threshold    = args.p_threshold,
                force          = "motif" in force_steps,
            )
            n_sig = sum(len(v) for v in motif_result["significant_tfs"].values())
            save_checkpoint(out_dir, "motif", {"significant_tfs": n_sig})

    # ── G: GO annotation ──────────────────────────────────────────────────────
    if "go" not in skip:
        if not enrich_dir.exists():
            print("\n[SKIP] go: enrichment_results dir not found — run motif step first")
        elif _is_done(cp, "go", force_steps):
            print(f"\n[SKIP] go: {cp['go']['completed_at']}")
        else:
            print("\n" + "=" * 60)
            print("STEP G — GO Annotation")
            print("=" * 60)
            go_result = run_go_annotation(
                enrichment_dir = str(enrich_dir),
                build          = args.build,
                out_dir        = str(out_dir),
                clustered_csv  = args.input,
                p_threshold    = args.p_threshold,
                force          = "go" in force_steps,
            )
            save_checkpoint(out_dir, "go", {"n_genes": go_result["n_genes"]})

    # ── E: Expression analysis ────────────────────────────────────────────────
    if "expression" not in skip:
        if _is_done(cp, "expression", force_steps):
            print(f"\n[SKIP] expression: {cp['expression']['completed_at']}")
        else:
            print("\n" + "=" * 60)
            print("STEP E — Expression Analysis")
            print("=" * 60)
            expr_result = run_expression_analysis(
                input_csv = args.input,
                out_dir   = str(out_dir),
                force     = "expression" in force_steps,
            )
            save_checkpoint(out_dir, "expression",
                            {"n_clusters": expr_result["n_clusters"],
                             "n_expr_cols": len(expr_result["expr_cols"])})

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Enrichment pipeline complete")
    _print_status(out_dir, all_steps)
    if args.notify_email:
        from te_notify import send_completion_email
        send_completion_email(args.notify_email, "te_enrichment", str(out_dir))


if __name__ == "__main__":
    main()
