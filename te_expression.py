#!/usr/bin/env python3
"""
te_expression.py  —  GAMECA step E: Expression Analysis
─────────────────────────────────────────────────────────────────────────────
Generates per-cluster expression boxplots from a clustered CSV.

Auto-detects expression columns as any numeric column that is not a known
coordinate / cluster / embedding column. Columns can also be specified
explicitly via --stage-cols.

Input:   clustered CSV produced by te_clustering.py
         (needs columns: Cluster, plus ≥1 numeric expression columns)

Output:
  <out_dir>/expression_plots/
    boxplot_all.png        all data
    boxplot_mid80.png      10th–90th percentile (outliers removed)
    expression_stats.csv   per-cluster mean/median/std per column

Usage:
    python te_expression.py \\
        --input ./results/clustered.csv \\
        --out-dir ./results

    # Specify exact expression columns:
    python te_expression.py \\
        --input ./results/clustered.csv \\
        --stage-cols pronuc twocell fourcell eightcell morulacell \\
        --stage-labels "Pro-nucleus" "2-cell" "4-cell" "8-cell" "Morula" \\
        --out-dir ./results
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = [
    "#4C9BE8","#E8604C","#2ECC71","#9B59B6","#F39C12",
    "#1ABC9C","#E74C3C","#3498DB","#E67E22","#8E44AD",
]

# Columns that are never expression data
_NON_EXPR_COLS = {
    "chr", "chromosome", "chrom", "#chrom",
    "start", "chromstart", "stop", "end", "chromend",
    "strand", "te_name", "repname", "seq",
    "cluster", "cluster_id",
    "umap_x", "umap_y", "pca_x", "pca_y", "tsne_x", "tsne_y",
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="GAMECA step E: Expression analysis per cluster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input",        required=True, help="Clustered CSV (from te_clustering.py)")
    p.add_argument("--out-dir",      default=".",  help="Output directory")
    p.add_argument("--stage-cols",   nargs="+",    default=None,
                   help="Explicit expression column names")
    p.add_argument("--stage-labels", nargs="+",    default=None,
                   help="Display labels for --stage-cols (same length)")
    p.add_argument("--log1p",        action="store_true", default=True,
                   help="Apply log1p transform to expression values (default: on)")
    p.add_argument("--no-log1p",     dest="log1p", action="store_false")
    p.add_argument("--force",        action="store_true",
                   help="Re-run even if output already exists")
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────

def _cluster_col(df):
    for c in ("Cluster", "cluster"):
        if c in df.columns:
            return c
    return None


def _auto_expr_cols(df):
    """Return numeric columns that look like expression data."""
    out = []
    for c in df.columns:
        if c.lower() in _NON_EXPR_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def run_expression_analysis(input_csv, out_dir, stage_cols=None, stage_labels=None,
                             log1p=True, force=False):
    """
    Generate per-cluster expression boxplots.

    Parameters
    ----------
    input_csv    : path to clustered CSV
    out_dir      : root output directory
    stage_cols   : list of column names to use as expression; auto-detected if None
    stage_labels : display labels corresponding to stage_cols
    log1p        : apply log1p transform before plotting
    force        : re-run even if outputs exist

    Returns
    -------
    dict with keys:
        expr_dir      – output directory path
        expr_cols     – list of expression columns used
        n_clusters    – number of clusters plotted
    """
    out_dir   = Path(out_dir)
    expr_dir  = out_dir / "expression_plots"
    stats_csv = expr_dir / "expression_stats.csv"
    expr_dir.mkdir(parents=True, exist_ok=True)

    if (expr_dir / "boxplot_all.png").exists() and not force:
        print("  [SKIP] Expression plots already exist (use --force to re-run)")
        return {"expr_dir": str(expr_dir), "expr_cols": [], "n_clusters": 0}

    print(f"\n  Loading {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"  {len(df):,} rows")

    cl_col = _cluster_col(df)
    if cl_col is None:
        print("FATAL: No Cluster column found. Run te_clustering.py first.")
        sys.exit(1)

    df[cl_col] = df[cl_col].astype(int)
    cluster_ids = sorted([c for c in df[cl_col].unique() if c >= 0])
    print(f"  {len(cluster_ids)} clusters: {cluster_ids}")

    # ── Resolve expression columns ────────────────────────────────────────────
    if stage_cols:
        missing = [c for c in stage_cols if c not in df.columns]
        if missing:
            print(f"FATAL: Expression columns not found: {missing}")
            sys.exit(1)
        expr_cols = stage_cols
    else:
        expr_cols = _auto_expr_cols(df)

    if not expr_cols:
        print("  No expression columns found — skipping.")
        return {"expr_dir": str(expr_dir), "expr_cols": [], "n_clusters": 0}

    labels = stage_labels if (stage_labels and len(stage_labels) == len(expr_cols)) else expr_cols
    print(f"  Expression columns ({len(expr_cols)}): {expr_cols}")

    # ── Build per-cluster data dicts ──────────────────────────────────────────
    cluster_labels = {}
    cluster_colors = {}
    data_dict = {}
    for i, cid in enumerate(cluster_ids):
        lbl = f"Cluster {cid}"
        col = PALETTE[i % len(PALETTE)]
        cluster_labels[cid] = lbl
        cluster_colors[lbl] = col
        sub = df[df[cl_col] == cid]
        if log1p:
            data_dict[lbl] = {c: np.log1p(sub[c].values) for c in expr_cols}
        else:
            data_dict[lbl] = {c: sub[c].values for c in expr_cols}

    # ── Boxplot function ──────────────────────────────────────────────────────
    def _boxplot(data_d, title, fname, p_low=0, p_high=100, showfliers=True):
        labs = list(data_d.keys())
        w, g = 0.3, 1.2
        pos  = {lb: [i*g + li*w for i in range(len(expr_cols))]
                for li, lb in enumerate(labs)}
        fig, ax = plt.subplots(figsize=(max(10, len(expr_cols)*2), 7))
        for lb in labs:
            c = cluster_colors[lb]
            for col, p in zip(expr_cols, pos[lb]):
                raw = data_d[lb][col]
                lo  = np.percentile(raw, p_low)
                hi  = np.percentile(raw, p_high)
                d   = raw[(raw >= lo) & (raw <= hi)]
                ax.boxplot(d, positions=[p], widths=w*0.85,
                           patch_artist=True, showfliers=showfliers,
                           boxprops=dict(facecolor=c, alpha=0.75),
                           medianprops=dict(color="white", linewidth=2),
                           whiskerprops=dict(color=c, linestyle="--"),
                           capprops=dict(color=c))
        tick_pos = [i*g + (len(labs)-1)*w/2 for i in range(len(expr_cols))]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(labels, fontsize=11, rotation=30 if len(labels) > 6 else 0, ha="right")
        ax.set_ylabel("log1p(counts)" if log1p else "counts")
        ax.set_title(title, fontweight="bold")
        patches = [mpatches.Patch(facecolor=cluster_colors[lb], alpha=0.75, label=lb)
                   for lb in labs]
        ax.legend(handles=patches, fontsize=10, framealpha=0.3)
        ax.spines[["top","right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(expr_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {fname}")

    _boxplot(data_dict, "Expression per Cluster — All Data",
             "boxplot_all.png", showfliers=True)
    _boxplot(data_dict, "Expression per Cluster — Mid 80%",
             "boxplot_mid80.png", p_low=10, p_high=90, showfliers=False)

    # ── Per-cluster per-stage stats ───────────────────────────────────────────
    stat_rows = []
    for cid in cluster_ids:
        sub = df[df[cl_col] == cid]
        for c in expr_cols:
            vals = sub[c].dropna().values
            if log1p:
                vals = np.log1p(vals)
            stat_rows.append({
                "Cluster": cid,
                "Column":  c,
                "n":       len(vals),
                "mean":    round(float(np.mean(vals)),   4) if len(vals) else None,
                "median":  round(float(np.median(vals)), 4) if len(vals) else None,
                "std":     round(float(np.std(vals)),    4) if len(vals) else None,
            })
    pd.DataFrame(stat_rows).to_csv(stats_csv, index=False)
    print(f"  Saved {stats_csv.name}")

    return {
        "expr_dir":  str(expr_dir),
        "expr_cols": expr_cols,
        "n_clusters": len(cluster_ids),
    }


def main():
    args = parse_args()
    print("=" * 60)
    print("GAMECA — Expression Analysis")
    print(f"  Input:  {args.input}")
    print(f"  OutDir: {args.out_dir}")
    print("=" * 60)
    run_expression_analysis(
        input_csv    = args.input,
        out_dir      = args.out_dir,
        stage_cols   = args.stage_cols,
        stage_labels = args.stage_labels,
        log1p        = args.log1p,
        force        = args.force,
    )


if __name__ == "__main__":
    main()
