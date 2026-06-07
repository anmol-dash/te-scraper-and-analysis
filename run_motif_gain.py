#!/usr/bin/env python3
"""
run_motif_gain.py — GAMECA TF motif gain/loss runner (Stage 11 standout module)

Thin wrapper around te_motif_gain.run_motif_gain_analysis.

Usage:
    python run_motif_gain.py --input clustered.csv --reports-dir ./reports \\
        --family LTR5_Hs --assembly hg38

For HPC submission use submit_motif_gain.py.
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="GAMECA TF motif gain/loss annotation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--input",           required=True)
    p.add_argument("--reports-dir",     default="./reports")
    p.add_argument("--family",          default="TE")
    p.add_argument("--assembly",        default="hg38")
    p.add_argument("--overlaps-tsv",    default=None)
    p.add_argument("--consensus-fasta", default=None)
    p.add_argument("--max-mm",          type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()

    # Allow import from the bundled PyInstaller extraction dir
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))

    try:
        from te_motif_gain import run_motif_gain_analysis
    except ImportError as exc:
        print(f"[run_motif_gain] ERROR: cannot import te_motif_gain: {exc}", flush=True)
        sys.exit(1)

    run_motif_gain_analysis(
        input_csv       = args.input,
        reports_dir     = args.reports_dir,
        family_name     = args.family,
        assembly        = args.assembly,
        overlaps_tsv    = args.overlaps_tsv,
        max_mm          = args.max_mm,
        consensus_fasta = args.consensus_fasta,
    )


if __name__ == "__main__":
    main()
