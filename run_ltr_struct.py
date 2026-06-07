#!/usr/bin/env python3
"""
run_ltr_struct.py — GAMECA standout module: LTR structural annotation

Annotates each LTR retrotransposon locus with:
  • classification  full-length | solo-LTR | internal-only | truncated
  • has_5ltr / has_3ltr   boolean
  • ltr_identity           fractional identity between 5'/3' ends
  • pbs_found / pbs_position / pbs_motif   primer binding site
  • ppt_found / ppt_position / ppt_purine_frac   polypurine tract
  • tsd_found / tsd_sequence   target site duplication (when flanking seq available)

No external tools required — pure Python.

Outputs (in <reports_dir>/):
  ltr_struct_annotated.csv     per-locus annotation (all original columns preserved)
  ltr_struct_summary.csv       per-cluster breakdown of structural classes
  fig_ltr_struct.png           stacked bar: structural classification by cluster
  ltr_struct_values.tex        LaTeX macros for the report

Usage (called automatically by query.py Stage 11):
    python run_ltr_struct.py --input clustered.csv --reports-dir ./reports --family THE1D-int
"""

import argparse
import sys
from pathlib import Path


def _parse_args():
    p = argparse.ArgumentParser(
        description="LTR structural annotation — PBS, PPT, TSD, LTR boundaries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input",            required=True,
                   help="Clustered CSV (needs 'Seq' column)")
    p.add_argument("--reports-dir",      required=True,
                   help="Output directory for reports and figures")
    p.add_argument("--family",           default="FAMILY",
                   help="TE family name (used to select PBS probe, plot titles)")
    p.add_argument("--min-ltr-identity", type=float, default=0.65,
                   help="Minimum fractional identity between 5'/3' ends to call an LTR "
                        "(default 0.65)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    from te_ltr_struct import run_ltr_struct_analysis
    run_ltr_struct_analysis(
        input_csv        = args.input,
        reports_dir      = args.reports_dir,
        family_name      = args.family,
        min_ltr_identity = args.min_ltr_identity,
    )
