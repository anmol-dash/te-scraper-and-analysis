#!/usr/bin/env python3
"""
run_subfamily.py — GAMECA standout module: automatic subfamily resolution

Computes pairwise CpG-corrected K2P distances between cluster consensus
sequences, attempts to fetch the Dfam canonical for divergence measurement,
then builds a UPGMA dendrogram and assigns provisional subfamily labels.

Outputs (in <reports_dir>/):
  subfamily_table.csv          per-cluster: divergence from canonical,
                               subfamily label, is_canonical_type flag
  fig_subfamily_tree.png       UPGMA dendrogram of cluster consensuses
  fig_subfamily_divergence.png per-cluster bar chart vs Dfam canonical
  subfamily_tree.nwk           Newick format tree string
  subfamily_values.tex         LaTeX macros for the report

Usage (called automatically by query.py Stage 11):
    python run_subfamily.py --input clustered.csv --reports-dir ./reports --family HERVK9
"""

import argparse
import sys
from pathlib import Path


def _parse_args():
    p = argparse.ArgumentParser(
        description="Subfamily resolution: K2P divergence + UPGMA tree",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input",           required=True,
                   help="Clustered CSV (needs 'Seq' and 'Cluster' columns)")
    p.add_argument("--reports-dir",     required=True,
                   help="Output directory for reports and figures")
    p.add_argument("--family",          default="FAMILY",
                   help="TE family name (used for Dfam lookup + plot titles)")
    p.add_argument("--consensus-fasta", default=None,
                   help="Path to all_cluster_consensuses.fa (auto-detected if omitted)")
    p.add_argument("--assembly",        default="hg38",
                   help="Genome assembly (used for Dfam API; default hg38)")
    p.add_argument("--cpg-omega",       type=float, default=10.0,
                   help="CpG transition correction factor (default 10)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    from te_subfamily import run_subfamily_analysis
    run_subfamily_analysis(
        input_csv       = args.input,
        reports_dir     = args.reports_dir,
        family_name     = args.family,
        consensus_fasta = args.consensus_fasta,
        assembly        = args.assembly,
        cpg_omega       = args.cpg_omega,
    )
