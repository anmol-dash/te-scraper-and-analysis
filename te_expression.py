#!/usr/bin/env python3
"""
te_expression.py  —  GAMECA step E: Expression Analysis
─────────────────────────────────────────────────────────────────────────────
Generates per-cluster expression plots and statistics from a clustered CSV.

Auto-detects expression columns as any column (numeric, numeric-string, or
list-like SQUIRE cell) that is not coordinate / score / cluster / embedding
metadata. Sample names are arbitrary — there is no developmental-stage
whitelist, so "K562_rep2", "Liver_donorA", "siCTRL_0h", "2-cell", etc. are all
treated identically. Columns can also be specified explicitly via --stage-cols.

When run as part of the query.py pipeline, --stage-cols/--stage-labels are
always supplied: query.py requires the user to designate expression column
names and figure order up front (via --expr-cols/--expr-labels, or an
interactive prompt) before the pipeline proceeds, and passes that resolved,
ordered list to this module. The auto-detection above only kicks in for
standalone/direct invocations of this script.

Input:   clustered CSV produced by te_clustering.py
         (needs columns: Cluster, plus ≥1 numeric expression columns)

Output:
  <out_dir>/expression_plots/
    boxplot_all.png            expression per cluster — all data
    boxplot_mid80.png          10th–90th percentile (outliers removed)
    stage_profile.png          mean expression across stages + per-cluster lines
    peak_stage_summary.png     which stage peaks in each cluster
    chromosomal_expression.png mean expression by chromosome
    chromosomal_heatmap.png    stage × chromosome heatmap
    primer_expression.png      expression profile by top primer (if available)
    expression_stats.csv       per-cluster mean/median/std/cv/pct_zero per column
    peak_stage_summary.csv     peak stage per cluster
    chromosomal_expression.csv mean expression per chromosome per stage
    primer_expression.csv      mean expression per primer per stage

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
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

PALETTE = [
    "#4C9BE8","#E8604C","#2ECC71","#9B59B6","#F39C12",
    "#1ABC9C","#E74C3C","#3498DB","#E67E22","#8E44AD",
]

_CHROM_ORDER = (
    [f"chr{i}" for i in range(1, 23)]
    + ["chrX", "chrY", "chrM"]
)

_NON_EXPR_COLS = {
    "chr", "chromosome", "chrom", "#chrom", "genoname",
    "start", "chromstart", "stop", "end", "chromend",
    "genostart", "genoend", "genoleft",
    "strand", "te_name", "te_id", "repname", "repclass", "repfamily",
    "name", "id", "bin", "seq", "sequence", "ucsc_url",
    "cluster", "cluster_id", "_te_row_id", "_expr_row_id",
    "length", "gc_content", "gc",
    "swscore", "millidiv", "milliins", "milldel", "millidel",
    "repstart", "repend", "repleft",
    "divergence", "kimura", "age", "age_mya", "subst_rate",
    "umap_x", "umap_y", "pca_x", "pca_y", "tsne_x", "tsne_y",
    "expr_match_count", "expr_overlap_bp",
}

# Substring patterns (matched against the lower-cased column name) that mark a
# numeric column as *metadata*, never an expression sample — regardless of how
# the actual samples happen to be named. This is what lets the module tolerate
# arbitrary sample names (e.g. "K562_rep2", "Liver_donorA", "siCTRL_0h"):
# any numeric column that is NOT obvious coordinate/score/embedding metadata is
# treated as a sample, instead of relying on a fixed list of stage names.
_NON_EXPR_PATTERNS = (
    "_row_id", "row_id", "embedding", "_pca", "_umap", "_tsne",
    "coord", "millidiv", "milli_div", "swscore", "sw_score",
    "overlap_bp", "match_count", "_url",
)


def _is_metadata_col(col):
    """True when *col* is coordinate/score/embedding bookkeeping, not a sample."""
    name = str(col).strip().lower()
    if name in _NON_EXPR_COLS:
        return True
    return any(pat in name for pat in _NON_EXPR_PATTERNS)


def _listlike_count(v):
    """Count elements in a list-like cell ('[a, b, c]', 'x;y;z', 'p,q').

    SQUIRE / ultracombo expression assemblies store the per-sample signal as a
    list of cell-barcodes rather than a scalar count. We treat the *number* of
    elements as the expression value so those columns become usable samples.
    Returns np.nan for genuine scalars (handled by numeric coercion instead).
    """
    if isinstance(v, (list, tuple, set)):
        return float(len(v))
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0.0
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return 0.0
    if s.startswith("[") and s.endswith("]"):
        try:
            import ast
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple, set)):
                return float(len(parsed))
        except Exception:
            pass
        # Brackets present but not valid Python literal (e.g. unquoted tokens):
        # count comma/semicolon-separated tokens inside the brackets.
        inner = s[1:-1].strip()
        if not inner:
            return 0.0
        for sep in (",", ";", "|"):
            if sep in inner:
                return float(len([t for t in inner.split(sep) if t.strip()]))
        return 1.0
    # Delimited list of tokens (commas / semicolons / pipes) with no brackets.
    for sep in (";", "|"):
        if sep in s:
            return float(len([t for t in s.split(sep) if t.strip()]))
    return np.nan


def _coerce_expr_series(s):
    """Return *s* as a float Series, tolerant of list-like / string-number cells.

    Order of attempts: (1) already numeric → as-is; (2) plain numeric strings →
    pd.to_numeric; (3) list-like cells → element counts. Returns None if the
    column cannot be interpreted as expression at all.
    """
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() >= 0.5:        # mostly real numbers → treat as numeric
        return num.fillna(0.0).astype(float)
    counts = s.map(_listlike_count)
    if counts.notna().any() and float(counts.fillna(0).sum()) > 0:
        return counts.fillna(0.0).astype(float)
    return None


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    """Parse the standalone expression-analysis CLI (see module docstring)."""
    p = argparse.ArgumentParser(
        description="GAMECA step E: Expression analysis per cluster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input",        required=True, help="Clustered CSV (from te_clustering.py)")
    p.add_argument("--out-dir",      default=".",  help="Output directory")
    p.add_argument("--reports-dir",  default=None, help=argparse.SUPPRESS)
    p.add_argument("--family",       default=None, help=argparse.SUPPRESS)
    p.add_argument("--stage-cols",   nargs="+",    default=None,
                   help="Explicit expression column names")
    p.add_argument("--stage-labels", nargs="+",    default=None,
                   help="Display labels for --stage-cols (same length)")
    p.add_argument("--expr-pattern", default=None,
                   help="Regex — columns whose name matches are used for bar plots. "
                        "Without --stage-cols or --expr-pattern, bar plots are skipped.")
    p.add_argument("--log1p",        action="store_true", default=True,
                   help="Apply log1p transform to expression values (default: on)")
    p.add_argument("--no-log1p",     dest="log1p", action="store_false")
    p.add_argument("--force",        action="store_true",
                   help="Re-run even if output already exists")
    p.add_argument("--notify-email", default="", metavar="EMAIL",
                   help="Send a completion email to this address (requires Gmail App Password setup).")
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────

def _cluster_col(df):
    for c in ("Cluster", "cluster"):
        if c in df.columns:
            return c
    return None


def _auto_expr_cols(df):
    """Detect expression-sample columns under *arbitrary* sample naming.

    A column is a sample if it is NOT coordinate/score/embedding metadata and
    its values can be interpreted as expression (numeric, numeric-string, or
    list-like). Columns the pipeline already prefixed ``expr_`` are always kept
    (minus the ``expr_match_count`` / ``expr_overlap_bp`` bookkeeping pair).
    No sample-name whitelist is used, so names like "K562_rep2" or "2-cell"
    are treated identically.
    """
    out = []
    for c in df.columns:
        name = str(c).strip().lower()
        if name in ("expr_match_count", "expr_overlap_bp"):
            continue
        forced = name.startswith("expr_")           # pipeline-merged sample
        if not forced and _is_metadata_col(c):
            continue
        if _coerce_expr_series(df[c]) is not None:
            out.append(c)
    return out


def _apply(vals, log1p):
    return np.log1p(vals) if log1p else vals


def _ylabel(log1p):
    return "log1p(counts)" if log1p else "counts"


def _chrom_col(df):
    wanted = ("chr", "chromosome", "chrom", "#chrom", "genoname")
    lower = {str(c).strip().lower(): c for c in df.columns}
    for w in wanted:
        if w in lower:
            return lower[w]
    return None


# ── 1. Boxplots (existing) ─────────────────────────────────────────────────────

def _boxplots(df, cl_col, expr_cols, labels, log1p, expr_dir):
    """Write the per-cluster expression boxplots (all-data and mid-80% variants).

    Noise cluster -1 is excluded. `log1p` toggles a log1p transform; `labels`
    supplies display names for the expr_cols.
    """
    cluster_ids = sorted([c for c in df[cl_col].unique() if c >= 0])
    cluster_colors = {}
    data_dict = {}
    for i, cid in enumerate(cluster_ids):
        lbl = f"Cluster {cid}"
        col = PALETTE[i % len(PALETTE)]
        cluster_colors[lbl] = col
        sub = df[df[cl_col] == cid]
        data_dict[lbl] = {c: _apply(sub[c].values, log1p) for c in expr_cols}

    def _one(data_d, title, fname, p_low=0, p_high=100, showfliers=True):
        labs = list(data_d.keys())
        w, g = 0.3, 1.2
        pos = {lb: [i * g + li * w for i in range(len(expr_cols))]
               for li, lb in enumerate(labs)}
        fig, ax = plt.subplots(figsize=(max(10, len(expr_cols) * 2), 7))
        for lb in labs:
            c = cluster_colors[lb]
            for col, p in zip(expr_cols, pos[lb]):
                raw = data_d[lb][col]
                lo = np.percentile(raw, p_low)
                hi = np.percentile(raw, p_high)
                d = raw[(raw >= lo) & (raw <= hi)]
                ax.boxplot(d, positions=[p], widths=w * 0.85,
                           patch_artist=True, showfliers=showfliers,
                           boxprops=dict(facecolor=c, alpha=0.75),
                           medianprops=dict(color="white", linewidth=2),
                           whiskerprops=dict(color=c, linestyle="--"),
                           capprops=dict(color=c))
        tick_pos = [i * g + (len(labs) - 1) * w / 2 for i in range(len(expr_cols))]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(labels, fontsize=11,
                           rotation=30 if len(labels) > 6 else 0, ha="right")
        ax.set_ylabel(_ylabel(log1p))
        ax.set_title(title, fontweight="bold")
        patches = [mpatches.Patch(facecolor=cluster_colors[lb], alpha=0.75, label=lb)
                   for lb in labs]
        ax.legend(handles=patches, fontsize=10, framealpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(expr_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {fname}")

    _one(data_dict, "Expression per Cluster — All Data",
         "boxplot_all.png", showfliers=True)
    _one(data_dict, "Expression per Cluster — Mid 80%",
         "boxplot_mid80.png", p_low=10, p_high=90, showfliers=False)


# ── 2. Stage profile: mean per stage + per-cluster lines ─────────────────────

def _plot_stage_profile(df, cl_col, expr_cols, labels, log1p, expr_dir):
    """Bar chart of overall mean expression per stage with per-cluster line overlay."""
    cluster_ids = sorted([c for c in df[cl_col].unique() if c >= 0])

    overall_means = [_apply(df[c].dropna().values, log1p).mean() for c in expr_cols]
    overall_stds  = [_apply(df[c].dropna().values, log1p).std()  for c in expr_cols]
    peak_idx = int(np.argmax(overall_means))
    peak_label = labels[peak_idx]

    fig, ax = plt.subplots(figsize=(max(9, len(expr_cols) * 1.5), 6))
    x = np.arange(len(expr_cols))
    bars = ax.bar(x, overall_means, yerr=overall_stds, capsize=4,
                  color="#4C9BE8", alpha=0.7, label="All TEs (mean ± std)")
    bars[peak_idx].set_facecolor("#E8604C")
    bars[peak_idx].set_alpha(0.9)

    for i, cid in enumerate(cluster_ids):
        sub = df[df[cl_col] == cid]
        cl_means = [_apply(sub[c].dropna().values, log1p).mean() for c in expr_cols]
        ax.plot(x, cl_means, marker="o", linewidth=1.5, markersize=5,
                color=PALETTE[i % len(PALETTE)], label=f"Cluster {cid}", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30 if len(labels) > 6 else 0,
                       ha="right", fontsize=11)
    ax.set_ylabel(_ylabel(log1p))
    ax.set_title(f"Expression Profile Across Stages\n(peak stage: {peak_label})",
                 fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.3, ncol=max(1, len(cluster_ids) // 4 + 1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(expr_dir / "stage_profile.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved stage_profile.png  (peak stage: {peak_label})")

    # Peak stage per cluster — horizontal grouped bar
    peak_rows = []
    for cid in cluster_ids:
        sub = df[df[cl_col] == cid]
        cl_means = [_apply(sub[c].dropna().values, log1p).mean() for c in expr_cols]
        pk_idx = int(np.argmax(cl_means))
        peak_rows.append({
            "Cluster": cid,
            "peak_stage": labels[pk_idx],
            "peak_col": expr_cols[pk_idx],
            "peak_mean": round(cl_means[pk_idx], 4),
            **{f"mean_{labels[i]}": round(cl_means[i], 4) for i in range(len(expr_cols))},
        })
    df_peak = pd.DataFrame(peak_rows)
    df_peak.to_csv(expr_dir / "peak_stage_summary.csv", index=False)
    print(f"  Saved peak_stage_summary.csv")

    # Heatmap: cluster × stage
    heat = np.array([
        [_apply(df[df[cl_col] == cid][c].dropna().values, log1p).mean()
         for c in expr_cols]
        for cid in cluster_ids
    ])
    fig2, ax2 = plt.subplots(figsize=(max(8, len(expr_cols) * 1.2),
                                      max(4, len(cluster_ids) * 0.6 + 1.5)))
    im = ax2.imshow(heat, aspect="auto", cmap="YlOrRd")
    ax2.set_xticks(range(len(expr_cols)))
    ax2.set_xticklabels(labels, rotation=30 if len(labels) > 6 else 0,
                        ha="right", fontsize=10)
    ax2.set_yticks(range(len(cluster_ids)))
    ax2.set_yticklabels([f"Cluster {c}" for c in cluster_ids], fontsize=10)
    plt.colorbar(im, ax=ax2, label=_ylabel(log1p), shrink=0.8)
    ax2.set_title("Mean Expression Heatmap: Cluster × Stage", fontweight="bold")
    # Annotate cells
    for r in range(len(cluster_ids)):
        for c in range(len(expr_cols)):
            ax2.text(c, r, f"{heat[r, c]:.2f}", ha="center", va="center",
                     fontsize=7, color="black" if heat[r, c] < heat.max() * 0.7 else "white")
    plt.tight_layout()
    plt.savefig(expr_dir / "peak_stage_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved peak_stage_summary.png")

    return df_peak


# ── 3. Chromosomal expression ─────────────────────────────────────────────────

def _plot_chromosomal_expression(df, cl_col, expr_cols, labels, log1p, expr_dir):
    """Write chromosomal-expression figures (per-chromosome means + stage×chrom heatmap).

    No-ops with a [SKIP] message when the input has no recognizable chromosome
    column.
    """
    chr_col = _chrom_col(df)
    if chr_col is None:
        print("  [SKIP] No chromosome column found — skipping chromosomal expression")
        return

    df = df.copy()
    df["_total_expr"] = df[expr_cols].sum(axis=1)
    if log1p:
        df["_total_expr"] = np.log1p(df["_total_expr"])

    chroms = df[chr_col].astype(str).str.strip()
    df["_chrom_norm"] = chroms.apply(
        lambda c: c if c.startswith("chr") else f"chr{c}"
    )

    # Keep standard chromosomes, sort by canonical order
    std = set(_CHROM_ORDER)
    df_std = df[df["_chrom_norm"].isin(std)].copy()
    if df_std.empty:
        df_std = df.copy()

    chrom_means = (df_std.groupby("_chrom_norm")["_total_expr"].mean()
                   .reindex(_CHROM_ORDER).dropna())
    chrom_counts = df_std.groupby("_chrom_norm").size().reindex(_CHROM_ORDER).dropna()

    # ── Bar chart: mean total expression per chromosome ───────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                              gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    x = np.arange(len(chrom_means))
    ax.bar(x, chrom_means.values, color="#4C9BE8", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(chrom_means.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(f"Mean total expression ({_ylabel(log1p)})")
    ax.set_title("Chromosomal Expression — Mean Total Expression per Chromosome",
                 fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    ax2 = axes[1]
    ax2.bar(x, chrom_counts.values, color="#2ECC71", alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(chrom_means.index, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("TE loci count")
    ax2.set_title("Number of TE Loci per Chromosome")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(expr_dir / "chromosomal_expression.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved chromosomal_expression.png")

    # ── Heatmap: stage × chromosome ───────────────────────────────────────────
    heat_rows = []
    for c in expr_cols:
        col_vals = _apply(df_std[c].values, log1p)
        df_std = df_std.copy()
        df_std[f"_v_{c}"] = col_vals
        row = df_std.groupby("_chrom_norm")[f"_v_{c}"].mean().reindex(_CHROM_ORDER).dropna()
        heat_rows.append(row)

    heat_df = pd.DataFrame(heat_rows, index=labels)
    heat_arr = heat_df.values

    fig3, ax3 = plt.subplots(figsize=(max(12, len(heat_df.columns) * 0.7),
                                       max(4, len(expr_cols) * 0.6 + 1.5)))
    im = ax3.imshow(heat_arr, aspect="auto", cmap="Blues")
    ax3.set_xticks(range(len(heat_df.columns)))
    ax3.set_xticklabels(heat_df.columns, rotation=45, ha="right", fontsize=9)
    ax3.set_yticks(range(len(labels)))
    ax3.set_yticklabels(labels, fontsize=10)
    plt.colorbar(im, ax=ax3, label=_ylabel(log1p), shrink=0.8)
    ax3.set_title("Mean Expression Heatmap: Stage × Chromosome", fontweight="bold")
    plt.tight_layout()
    plt.savefig(expr_dir / "chromosomal_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved chromosomal_heatmap.png")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    chrom_csv_rows = []
    for chrom in chrom_means.index:
        sub = df_std[df_std["_chrom_norm"] == chrom]
        row = {"chromosome": chrom, "n_loci": len(sub),
               "mean_total_expr": round(float(sub["_total_expr"].mean()), 4)}
        for c, lbl in zip(expr_cols, labels):
            vals = _apply(sub[c].dropna().values, log1p)
            row[f"mean_{lbl}"] = round(float(vals.mean()), 4) if len(vals) else None
        chrom_csv_rows.append(row)
    pd.DataFrame(chrom_csv_rows).to_csv(expr_dir / "chromosomal_expression.csv", index=False)
    print(f"  Saved chromosomal_expression.csv")


# ── 4. Expression by primer ───────────────────────────────────────────────────

def _plot_primer_expression(df, cl_col, expr_cols, labels, log1p, expr_dir, out_dir):
    """Cross-reference top primers with expression: show which primers cover
    highly-expressed loci and what their stage profile looks like."""
    primers_dir = out_dir / "06_primers"
    summary_csv = primers_dir / "selected_primers_summary.csv"
    if not summary_csv.exists():
        # Try adjacent location
        for alt in [out_dir / "primers" / "selected_primers_summary.csv",
                    out_dir.parent / "06_primers" / "selected_primers_summary.csv"]:
            if alt.exists():
                summary_csv = alt
                break
        else:
            print("  [SKIP] selected_primers_summary.csv not found — skipping primer expression")
            return

    df_primers = pd.read_csv(summary_csv)
    if "primer" not in df_primers.columns or "rows_covered" not in df_primers.columns:
        print("  [SKIP] primer CSV missing required columns — skipping primer expression")
        return

    df = df.reset_index(drop=True)

    primer_records = []
    for _, pr_row in df_primers.iterrows():
        primer_seq = str(pr_row["primer"])
        rows_str = str(pr_row.get("rows_covered", ""))
        if not rows_str or rows_str == "nan":
            continue
        try:
            row_ids = [int(x) for x in rows_str.split(",") if x.strip().isdigit()]
        except ValueError:
            continue
        valid_ids = [r for r in row_ids if r < len(df)]
        if not valid_ids:
            continue
        sub = df.iloc[valid_ids]
        rec = {
            "primer": primer_seq,
            "n_loci": len(valid_ids),
            "coverage": pr_row.get("coverage", len(valid_ids)),
            "total_expr_score": pr_row.get("total_expr", None),
            "strategy": pr_row.get("strategy", ""),
        }
        for c, lbl in zip(expr_cols, labels):
            vals = _apply(sub[c].dropna().values, log1p)
            rec[f"mean_{lbl}"] = round(float(vals.mean()), 4) if len(vals) else 0.0
        primer_records.append(rec)

    if not primer_records:
        print("  [SKIP] No valid primer rows found — skipping primer expression plot")
        return

    df_pexpr = pd.DataFrame(primer_records)
    df_pexpr.to_csv(expr_dir / "primer_expression.csv", index=False)
    print(f"  Saved primer_expression.csv  ({len(df_pexpr)} primers)")

    # ── Grouped bar chart: stage expression for each primer ──────────────────
    mean_cols = [f"mean_{lbl}" for lbl in labels]
    n_primers = len(df_pexpr)
    n_stages  = len(labels)
    x = np.arange(n_stages)
    width = 0.8 / max(n_primers, 1)

    fig, ax = plt.subplots(figsize=(max(10, n_stages * 1.5), 6))
    for i, (_, pr_row) in enumerate(df_pexpr.iterrows()):
        means = [pr_row[c] for c in mean_cols]
        offset = (i - n_primers / 2 + 0.5) * width
        short_seq = pr_row["primer"][:12] + "…" if len(pr_row["primer"]) > 12 else pr_row["primer"]
        ax.bar(x + offset, means, width * 0.9,
               color=PALETTE[i % len(PALETTE)], alpha=0.8,
               label=f"{short_seq} (n={int(pr_row['n_loci'])})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30 if n_stages > 6 else 0,
                       ha="right", fontsize=11)
    ax.set_ylabel(f"Mean expression ({_ylabel(log1p)})")
    ax.set_title("Expression Profile by Top Primer\n(each bar group = one stage; colors = primers)",
                 fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.3, ncol=max(1, n_primers // 6 + 1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(expr_dir / "primer_expression.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved primer_expression.png")

    # ── Heatmap: primer × stage ───────────────────────────────────────────────
    heat = df_pexpr[mean_cols].values
    short_names = [
        (str(r["primer"])[:14] + "…" if len(str(r["primer"])) > 14 else str(r["primer"]))
        for _, r in df_pexpr.iterrows()
    ]
    fig2, ax2 = plt.subplots(figsize=(max(8, n_stages * 1.2),
                                       max(4, n_primers * 0.55 + 1.5)))
    im = ax2.imshow(heat, aspect="auto", cmap="RdYlGn")
    ax2.set_xticks(range(n_stages))
    ax2.set_xticklabels(labels, rotation=30 if n_stages > 6 else 0,
                        ha="right", fontsize=10)
    ax2.set_yticks(range(n_primers))
    ax2.set_yticklabels(short_names, fontsize=9)
    plt.colorbar(im, ax=ax2, label=_ylabel(log1p), shrink=0.8)
    ax2.set_title("Expression Heatmap: Primer × Stage", fontweight="bold")
    for r in range(n_primers):
        for c in range(n_stages):
            ax2.text(c, r, f"{heat[r, c]:.2f}", ha="center", va="center",
                     fontsize=7, color="black" if heat[r, c] < heat.max() * 0.6 else "white")
    plt.tight_layout()
    plt.savefig(expr_dir / "primer_expression_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved primer_expression_heatmap.png")


# ── 5. Extended stats CSV ─────────────────────────────────────────────────────

def _compute_stats(df, cl_col, expr_cols, labels, log1p, expr_dir):
    """Compute per-cluster expression summary statistics and write them to CSV."""
    cluster_ids = sorted([c for c in df[cl_col].unique() if c >= 0])
    rows = []
    for cid in cluster_ids:
        sub = df[df[cl_col] == cid]
        for c, lbl in zip(expr_cols, labels):
            vals = sub[c].dropna().values
            raw_vals = _apply(vals, log1p)
            n = len(raw_vals)
            mean_ = float(np.mean(raw_vals)) if n else None
            std_  = float(np.std(raw_vals))  if n else None
            rows.append({
                "Cluster":      cid,
                "Column":       c,
                "Label":        lbl,
                "n":            n,
                "mean":         round(mean_, 4)  if mean_ is not None else None,
                "median":       round(float(np.median(raw_vals)), 4) if n else None,
                "std":          round(std_,  4)  if std_  is not None else None,
                "cv":           round(std_ / mean_, 4) if (mean_ and mean_ != 0) else None,
                "min":          round(float(np.min(raw_vals)),  4) if n else None,
                "max":          round(float(np.max(raw_vals)),  4) if n else None,
                "pct_zero":     round(float((vals == 0).mean() * 100), 2) if n else None,
                "p10":          round(float(np.percentile(raw_vals, 10)), 4) if n else None,
                "p90":          round(float(np.percentile(raw_vals, 90)), 4) if n else None,
            })
    df_stats = pd.DataFrame(rows)
    df_stats.to_csv(expr_dir / "expression_stats.csv", index=False)
    print(f"  Saved expression_stats.csv")
    return df_stats


# ── 6. Differential expression between clusters ───────────────────────────────

def _de_between_clusters(df, cl_col, expr_cols, log1p, expr_dir):
    """Pairwise Wilcoxon rank-sum test between all cluster pairs with BH correction.

    For each expression column, tests every ordered pair of clusters (A vs B)
    using scipy.stats.mannwhitneyu (two-sided).  P-values are corrected across
    all tests (columns × pairs) using the Benjamini-Hochberg procedure.

    Outputs:
        de_pairwise.csv     all pairwise results with adjusted p-values
        de_heatmap.png      −log10(p_adj) heatmap (cluster × cluster, per column sum)
        de_significant.csv  subset passing p_adj < 0.05
    """
    try:
        from scipy import stats as _stats
    except ImportError:
        print("  [SKIP] DE testing requires scipy — install with: pip install scipy")
        return

    cluster_ids = sorted([c for c in df[cl_col].unique() if c >= 0])
    if len(cluster_ids) < 2:
        print("  [SKIP] Need ≥2 clusters for DE testing")
        return

    pairs = [(a, b) for i, a in enumerate(cluster_ids)
             for b in cluster_ids[i + 1:]]

    rows = []
    for col in expr_cols:
        for cA, cB in pairs:
            grpA = _apply(df[df[cl_col] == cA][col].dropna().values, log1p)
            grpB = _apply(df[df[cl_col] == cB][col].dropna().values, log1p)
            if len(grpA) < 3 or len(grpB) < 3:
                continue
            try:
                stat, pval = _stats.mannwhitneyu(grpA, grpB, alternative="two-sided")
            except Exception:
                stat, pval = np.nan, 1.0
            meanA = float(np.mean(grpA))
            meanB = float(np.mean(grpB))
            rows.append({
                "column":   col,
                "clusterA": cA,
                "clusterB": cB,
                "n_A":      len(grpA),
                "n_B":      len(grpB),
                "mean_A":   round(meanA, 4),
                "mean_B":   round(meanB, 4),
                "log2fc":   round(np.log2((meanA + 1e-9) / (meanB + 1e-9)), 4),
                "stat_U":   round(float(stat), 2) if not np.isnan(stat) else None,
                "p_value":  pval,
            })

    if not rows:
        print("  [SKIP] No valid cluster pairs for DE testing")
        return

    de_df = pd.DataFrame(rows)

    # BH correction across all tests
    pvals = de_df["p_value"].values
    n_tests = len(pvals)
    sorted_idx = np.argsort(pvals)
    bh_thresh = np.arange(1, n_tests + 1) * 0.05 / n_tests
    p_adj = np.ones(n_tests)
    for k in range(n_tests - 1, -1, -1):
        i = sorted_idx[k]
        if k == n_tests - 1:
            p_adj[i] = min(1.0, pvals[i] * n_tests / (k + 1))
        else:
            p_adj[i] = min(p_adj[sorted_idx[k + 1]],
                           pvals[i] * n_tests / (k + 1))

    de_df["p_adj_BH"] = np.round(p_adj, 6)
    de_df["significant"] = de_df["p_adj_BH"] < 0.05

    de_csv = expr_dir / "de_pairwise.csv"
    de_df.to_csv(de_csv, index=False)
    print(f"  Saved de_pairwise.csv  ({len(de_df)} tests, "
          f"{int(de_df['significant'].sum())} sig at p_adj<0.05)")

    sig_df = de_df[de_df["significant"]].sort_values("p_adj_BH")
    sig_df.to_csv(expr_dir / "de_significant.csv", index=False)
    print(f"  Saved de_significant.csv  ({len(sig_df)} rows)")

    n_cols = de_df["column"].nunique()
    tex = (
        "% Auto-generated by te_expression.py (_de_between_clusters)\n"
        f"\\newcommand{{\\deNClusters}}{{{len(cluster_ids)}}}\n"
        f"\\newcommand{{\\deNTests}}{{{len(de_df):,}}}\n"
        f"\\newcommand{{\\deNColumns}}{{{n_cols}}}\n"
        f"\\newcommand{{\\deNSignificant}}{{{int(de_df['significant'].sum()):,}}}\n"
    )
    (expr_dir / "de_values.tex").write_text(tex)
    print(f"  Saved de_values.tex")

    # ── Heatmap: average −log10(p_adj) per cluster pair summed over columns ──
    try:
        mat = np.zeros((len(cluster_ids), len(cluster_ids)))
        cid_idx = {cid: i for i, cid in enumerate(cluster_ids)}
        for _, row_ in de_df.iterrows():
            i_ = cid_idx[row_["clusterA"]]
            j_ = cid_idx[row_["clusterB"]]
            val = -np.log10(max(row_["p_adj_BH"], 1e-300))
            mat[i_, j_] = max(mat[i_, j_], val)
            mat[j_, i_] = max(mat[j_, i_], val)

        fig, ax = plt.subplots(figsize=(max(5, len(cluster_ids) * 0.9 + 2),
                                         max(4, len(cluster_ids) * 0.9 + 1.5)))
        im = ax.imshow(mat, aspect="auto", cmap="Reds")
        plt.colorbar(im, ax=ax, label=r"$-\log_{10}(p_{\mathrm{adj}})$", shrink=0.8)
        ticks = range(len(cluster_ids))
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        lbl = [f"C{c}" for c in cluster_ids]
        ax.set_xticklabels(lbl, fontsize=10)
        ax.set_yticklabels(lbl, fontsize=10)
        ax.set_title("Differential Expression Between Clusters\n"
                     r"($-\log_{10}$ BH-adjusted Wilcoxon p-value)",
                     fontweight="bold")
        for i_ in range(len(cluster_ids)):
            for j_ in range(len(cluster_ids)):
                if i_ != j_:
                    ax.text(j_, i_, f"{mat[i_, j_]:.1f}", ha="center", va="center",
                            fontsize=8,
                            color="white" if mat[i_, j_] > mat.max() * 0.6 else "black")
        plt.tight_layout()
        hm_path = expr_dir / "de_heatmap.png"
        plt.savefig(hm_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved de_heatmap.png")
    except Exception as e:
        print(f"  WARNING: DE heatmap failed: {e}")

    # ── Volcano-style plot: log2FC vs −log10(p_adj), all columns combined ────
    try:
        sig_mask = de_df["significant"].values
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.scatter(de_df.loc[~sig_mask, "log2fc"],
                    -np.log10(de_df.loc[~sig_mask, "p_adj_BH"].clip(lower=1e-300)),
                    s=12, alpha=0.4, color="#aaaaaa", label="n.s.")
        ax2.scatter(de_df.loc[sig_mask, "log2fc"],
                    -np.log10(de_df.loc[sig_mask, "p_adj_BH"].clip(lower=1e-300)),
                    s=18, alpha=0.7, color="#E8604C", label=f"sig (n={sig_mask.sum()})")
        ax2.axhline(-np.log10(0.05), color="black", linestyle="--", linewidth=1,
                    label="p_adj = 0.05")
        ax2.axvline(0, color="gray", linestyle="-", linewidth=0.5)
        ax2.set_xlabel("log₂ fold change (cluster A / B)")
        ax2.set_ylabel(r"$-\log_{10}(p_{\mathrm{adj}})$")
        ax2.set_title(f"DE Volcano — {df.shape[0]:,} loci, all columns × all cluster pairs",
                      fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(expr_dir / "de_volcano.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved de_volcano.png")
    except Exception as e:
        print(f"  WARNING: DE volcano failed: {e}")


# ── main ──────────────────────────────────────────────────────────────────────

def run_expression_analysis(input_csv, out_dir, stage_cols=None, stage_labels=None,
                             log1p=True, force=False, expr_pattern=None):
    """
    Generate per-cluster expression plots and statistics.

    Returns
    -------
    dict with keys: expr_dir, expr_cols, n_clusters
    """
    out_dir   = Path(out_dir)
    expr_dir  = out_dir / "expression_plots"
    expr_dir.mkdir(parents=True, exist_ok=True)

    if (expr_dir / "boxplot_all.png").exists() and not force:
        print("  [SKIP] Expression plots already exist (use --force to re-run)")
        return {"expr_dir": str(expr_dir), "expr_cols": [], "n_clusters": 0}

    print(f"\n  Loading {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    cl_col = _cluster_col(df)
    if cl_col is None:
        print("FATAL: No Cluster column found. Run te_clustering.py first.")
        sys.exit(1)

    df[cl_col] = df[cl_col].astype(int)
    cluster_ids = sorted([c for c in df[cl_col].unique() if c >= 0])
    print(f"  {len(cluster_ids)} clusters: {cluster_ids}")

    explicit = bool(stage_cols or expr_pattern)

    if stage_cols:
        missing = [c for c in stage_cols if c not in df.columns]
        if missing:
            print(f"FATAL: Expression columns not found: {missing}")
            sys.exit(1)
        expr_cols = list(stage_cols)
    elif expr_pattern:
        import re as _re
        try:
            _pat = _re.compile(expr_pattern)
        except _re.error as e:
            print(f"FATAL: --expr-pattern is not a valid regex: {e}")
            sys.exit(1)
        expr_cols = [c for c in _auto_expr_cols(df) if _pat.search(c)]
        if not expr_cols:
            print(f"  WARNING: --expr-pattern {expr_pattern!r} matched no columns — skipping bar plots")
    else:
        expr_cols = _auto_expr_cols(df)

    if not expr_cols:
        print("  No expression columns found — skipping.")
        return {"expr_dir": str(expr_dir), "expr_cols": [], "n_clusters": 0}

    # Coerce every selected column to a float Series (handles numeric strings
    # and list-like SQUIRE cells), dropping any that cannot be interpreted or
    # that are entirely empty/constant-NaN so weird inputs never crash plotting.
    kept_cols, kept_labels = [], []
    pre_labels = (stage_labels if (stage_labels and len(stage_labels) == len(expr_cols))
                  else expr_cols)
    for c, lbl in zip(expr_cols, pre_labels):
        coerced = _coerce_expr_series(df[c])
        if coerced is None or coerced.dropna().empty:
            print(f"  [DROP] column {c!r} is not usable as expression — skipping")
            continue
        df[c] = coerced
        kept_cols.append(c)
        kept_labels.append(lbl)
    expr_cols, labels = kept_cols, kept_labels

    if not expr_cols:
        print("  No usable expression columns after coercion — skipping.")
        return {"expr_dir": str(expr_dir), "expr_cols": [], "n_clusters": 0}
    print(f"  Expression columns ({len(expr_cols)}): {expr_cols}")

    # ── Run all analyses ──────────────────────────────────────────────────────
    print("\n  --- Boxplots ---")
    _boxplots(df, cl_col, expr_cols, labels, log1p, expr_dir)

    if explicit:
        print("\n  --- Stage profile ---")
        _plot_stage_profile(df, cl_col, expr_cols, labels, log1p, expr_dir)
    else:
        print("\n  --- Stage profile [SKIP: use --stage-cols or --expr-pattern to enable bar plots] ---")

    print("\n  --- Chromosomal expression ---")
    _plot_chromosomal_expression(df, cl_col, expr_cols, labels, log1p, expr_dir)

    if explicit:
        print("\n  --- Primer expression ---")
        _plot_primer_expression(df, cl_col, expr_cols, labels, log1p, expr_dir, out_dir)
    else:
        print("\n  --- Primer expression [SKIP: use --stage-cols or --expr-pattern to enable bar plots] ---")

    print("\n  --- Extended stats ---")
    _compute_stats(df, cl_col, expr_cols, labels, log1p, expr_dir)

    print("\n  --- Differential expression between clusters ---")
    _de_between_clusters(df, cl_col, expr_cols, log1p, expr_dir)

    return {
        "expr_dir":   str(expr_dir),
        "expr_cols":  expr_cols,
        "n_clusters": len(cluster_ids),
    }


def main():
    """Standalone entry point: load the clustered CSV, resolve expression
    columns, and emit all boxplot/stage/chromosomal figures and stats."""
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
        expr_pattern = args.expr_pattern,
    )
    if args.notify_email:
        from te_notify import send_completion_email
        send_completion_email(args.notify_email, "te_expression", args.out_dir)


if __name__ == "__main__":
    main()
