#!/usr/bin/env python3
"""
run_divergence.py — GAMECA standout module: repeat landscape / Kimura divergence

Computes CpG-corrected Kimura 2-parameter divergence per locus against the
cluster consensus sequence and generates repeat landscape histograms.

Outputs (in <reports_dir>/):
  divergence_per_locus.csv     per-locus K2P distance + alignment length
  divergence_stats.csv         per-cluster summary statistics
  fig_repeat_landscape.pdf     divergence histogram, combined + per-cluster
  fig_repeat_landscape.png     same, raster
  fig_repeat_landscape_per_cluster.png  faceted per-cluster version
  repeat_landscape_values.tex  LaTeX macros for the report

Usage (called automatically by query.py Stage 11):
    python run_divergence.py --input clustered.csv --reports-dir ./reports --family HERVK9

Standalone:
    python run_divergence.py \\
        --input results/hervk9/01_data/hervk9_clustered.csv \\
        --reports-dir results/hervk9/reports \\
        --family HERVK9
"""

import argparse
import sys
from pathlib import Path


def _parse_args():
    p = argparse.ArgumentParser(
        description="CpG-corrected Kimura divergence + repeat landscape (GAMECA standout)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input",          required=True,
                   help="Clustered CSV (needs 'Seq' and 'Cluster' columns)")
    p.add_argument("--reports-dir",    required=True,
                   help="Output directory for reports and figures")
    p.add_argument("--family",         default="FAMILY",
                   help="TE family name (used in plot titles)")
    p.add_argument("--consensus-fasta",default=None,
                   help="Path to all_cluster_consensuses.fa (auto-detected if omitted)")
    p.add_argument("--assembly",       default="hg38",
                   help="Genome assembly (informational; default hg38)")
    p.add_argument("--cpg-omega",      type=float, default=10.0,
                   help="CpG transition correction factor (default 10)")
    p.add_argument("--bin-width",      type=float, default=1.0,
                   help="Histogram bin width in %% divergence (default 1.0)")
    p.add_argument("--max-div",        type=float, default=40.0,
                   help="X-axis cap for histogram (default 40%%)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    from te_divergence import run_divergence_analysis
    run_divergence_analysis(
        input_csv       = args.input,
        reports_dir     = args.reports_dir,
        family_name     = args.family,
        consensus_fasta = args.consensus_fasta,
        cpg_omega       = args.cpg_omega,
        bin_width       = args.bin_width,
        max_div         = args.max_div,
    )
