#!/usr/bin/env python3
"""
run_line_sine_ltr_analysis.py --- GAMECA cross-family batch analysis
Analyzes LINE (L1Md_T), SINE (B1_Mus2), and LTR (IAPLTR1_Mm) TE families.

Runs as a batch job submitted by submit_line_sine_ltr_analysis.py.
All figures saved as PDFs to --reports-dir; writes measured_values.tex there too.

Usage:
    python run_line_sine_ltr_analysis.py \
        --l1mdt-expr  /path/L1Md_T_ultracombo.csv \
        --b1mus2-expr /path/B1_Mus2_ultracombo.csv \
        --iapltr1-expr /path/IAPLTR1_Mm_ultracombo.csv \
        --reports-dir /home/amodz/anmol/reports4 \
        [--genome-fa /path/mm10.fa] \
        [--rmsk-dir ~/te_analysis/rmsk] \
        [--source rmsk|dfam] \
        [--build mm10] \
        [--max-loci 5000]
"""

import argparse
import datetime
import json
import os
import shutil
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ─── configuration ────────────────────────────────────────────────────────────

FAMILIES = {
    "L1Md_T":    {"type": "LINE", "color": "#E8604C"},
    "B1_Mus2":   {"type": "SINE", "color": "#4C9BE8"},
    "IAPLTR1_Mm":{"type": "LTR",  "color": "#2ECC71"},
}

_PALETTE = [
    "#4C9BE8","#E8604C","#2ECC71","#9B59B6","#F39C12",
    "#1ABC9C","#E74C3C","#3498DB","#E67E22","#8E44AD",
]

_NON_EXPR = {
    "chr","chromosome","chrom","#chrom","genoname","genostart","genoend",
    "start","chromstart","stop","end","chromend","strand","te_name",
    "repname","seq","cluster","cluster_id","umap_x","umap_y",
    "pca_x","pca_y","tsne_x","tsne_y","length","gc_content",
    "te_id","repclass","repfamily","swscore","millidiv","ucsc_url",
    "name","score","locus","te_chromosome","te_start","te_end",
    "te_length","te_strand","unnamed: 0","_total_expr",
}


# ─── helpers ──────────────────────────────────────────────────────────────────

def _pp(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _exists(path: Path, force: bool) -> bool:
    """Return True if path exists and --force was not passed (skip this step)."""
    if not force and path.exists():
        _pp(f"  [skip] {path.name} already exists (use --force to overwrite)")
        return True
    return False


def _auto_expr_cols(df: pd.DataFrame) -> list:
    out = []
    for c in df.columns:
        if c.lower().strip() in _NON_EXPR:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def _norm_chrom(s: str) -> str:
    s = str(s).strip()
    return s if s.startswith("chr") else f"chr{s}"


# ─── expression file parsing ──────────────────────────────────────────────────

def _read_expression_file(path: str) -> pd.DataFrame:
    """Read SQuIRE ultracombo CSV. Normalises column names for coordinate matching."""
    df = pd.read_csv(path, sep=None, engine="python", comment=None)
    df.columns = [str(c).strip() for c in df.columns]

    # Normalise chromosome / start / end column names
    for variant, target in [
        (["#chrom","chrom","chromosome","genoname","te_chromosome","tx_chr"], "expr_chr"),
        (["chromstart","start","genostart","chromStart","te_start","TE_start","tx_start"], "expr_start"),
        (["chromend","end","genoend","chromEnd","te_end","stop","TE_end","tx_stop"],       "expr_stop"),
    ]:
        for v in variant:
            for col in df.columns:
                if col.lower() == v.lower():
                    df.rename(columns={col: target}, inplace=True)
                    break

    if "expr_chr" not in df.columns:
        raise ValueError(f"Expression file {path}: cannot find chromosome column. "
                         f"Columns: {list(df.columns)[:10]}")
    df["expr_chr"]   = df["expr_chr"].apply(_norm_chrom)
    df["expr_start"] = pd.to_numeric(df["expr_start"], errors="coerce")
    df["expr_stop"]  = pd.to_numeric(df["expr_stop"],  errors="coerce")
    return df


def match_expression(loci_df: pd.DataFrame, expr_path: str) -> pd.DataFrame:
    """Merge per-locus expression data into loci_df by coordinate overlap.

    Tries exact (chr, start, stop) match first; falls back to 1-bp tolerance,
    then any overlap.
    """
    _pp(f"  Matching expression file: {expr_path}")
    expr_df = _read_expression_file(expr_path)
    expr_cols = _auto_expr_cols(expr_df)
    _pp(f"    Expression columns detected ({len(expr_cols)}): {expr_cols[:6]}...")

    loci_df = loci_df.copy()
    loci_df["_chr_norm"] = loci_df["Chromosome"].apply(_norm_chrom)

    # Build expression lookup: chr -> list of (start, stop, {col: val})
    from collections import defaultdict
    expr_idx: dict = defaultdict(list)
    for _, row in expr_df.iterrows():
        ch = str(row["expr_chr"])
        es = int(row["expr_start"]) if pd.notna(row["expr_start"]) else -1
        ee = int(row["expr_stop"])  if pd.notna(row["expr_stop"])  else -1
        vals = {c: row[c] for c in expr_cols if c in row and pd.notna(row[c])}
        expr_idx[ch].append((es, ee, vals))

    matched = 0
    expr_records = []

    for _, lrow in loci_df.iterrows():
        ch   = lrow["_chr_norm"]
        ls   = int(lrow["Start"])
        le   = int(lrow["Stop"])
        cands = expr_idx.get(ch, [])

        best_vals = None
        # 1. exact match
        for es, ee, vals in cands:
            if es == ls and ee == le:
                best_vals = vals
                break
        # 2. ±1 bp tolerance
        if best_vals is None:
            for es, ee, vals in cands:
                if abs(es - ls) <= 1 and abs(ee - le) <= 1:
                    best_vals = vals
                    break
        # 3. any overlap
        if best_vals is None:
            for es, ee, vals in cands:
                if es < le and ee > ls:
                    best_vals = vals
                    break

        if best_vals is not None:
            matched += 1
        expr_records.append(best_vals or {})

    pct = 100 * matched / max(len(loci_df), 1)
    _pp(f"    Matched {matched}/{len(loci_df)} loci ({pct:.1f}%) to expression data")

    # Merge expression columns into loci_df
    expr_df_merged = pd.DataFrame(expr_records, index=loci_df.index)
    for col in expr_cols:
        if col in expr_df_merged.columns:
            loci_df[col] = pd.to_numeric(expr_df_merged[col], errors="coerce").fillna(0.0)

    loci_df.drop(columns=["_chr_norm"], inplace=True)
    return loci_df, expr_cols, matched


# ─── adaptive clustering ──────────────────────────────────────────────────────

def adaptive_clustering(df: pd.DataFrame, family: str, out_dir: Path,
                        kmer: int = 6, n_neighbors_list=(15, 30),
                        pca_dims: int = 50, n_epochs: int = 200,
                        random_state: int = 42) -> tuple:
    """Try min_cluster_size = n//5, n//6, ..., n//15 × n_neighbors 15,30.

    Returns (clustered_df, labels, divisor_used, n_neighbors_used, n_clusters).
    Stops at the first (divisor, n_neighbors) that yields ≥2 real clusters.
    """
    from te_clustering import clustering_analysis

    n = len(df)
    _pp(f"  Adaptive clustering: n={n}, trying divisors 5..15, n_neighbors {list(n_neighbors_list)}")

    for divisor in range(5, 16):
        mcs = max(5, n // divisor)
        for nn in n_neighbors_list:
            _pp(f"    divisor={divisor} (mcs={mcs}), n_neighbors={nn}")
            df_c, labels = clustering_analysis(
                df,
                kmer=kmer,
                min_cluster_size=mcs,
                n_neighbors=nn,
                pca_dims=pca_dims,
                n_epochs=n_epochs,
                random_state=random_state,
                out_dir=None,
                family_name=family,
                compute_tsne=False,
            )
            real_clusters = sorted(set(labels[labels >= 0]))
            n_real = len(real_clusters)
            _pp(f"      → {n_real} cluster(s) found")
            if n_real >= 2:
                _pp(f"  ✓ Accepted: divisor={divisor}, n_neighbors={nn}, clusters={n_real}")
                return df_c, labels, divisor, nn, n_real

    # Fallback: use n//15
    _pp(f"  ⚠ No config gave ≥2 clusters; using divisor=15, n_neighbors={n_neighbors_list[-1]}")
    mcs = max(5, n // 15)
    df_c, labels = clustering_analysis(
        df,
        kmer=kmer,
        min_cluster_size=mcs,
        n_neighbors=n_neighbors_list[-1],
        pca_dims=pca_dims,
        n_epochs=n_epochs,
        random_state=random_state,
        out_dir=None,
        family_name=family,
        compute_tsne=False,
    )
    n_real = len(set(labels[labels >= 0]))
    return df_c, labels, 15, n_neighbors_list[-1], n_real


# ─── Kimura / repeat landscape ────────────────────────────────────────────────

def plot_kimura_divergence(family_frames: dict, out_path: Path):
    """Stacked repeat-landscape (Kimura divergence) histograms for three families.

    milliDiv from RMSK = Kimura divergence × 1000 (per mil); convert to %.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle("Repeat Landscape --- Kimura Substitution Level", fontsize=14,
                 fontweight="bold", y=1.02)

    for ax, (family, df) in zip(axes, family_frames.items()):
        meta = FAMILIES[family]
        div_vals = df["milliDiv"].dropna().values / 10.0  # → % divergence
        age_label = "Kimura substitution level (%)"
        color = meta["color"]

        ax.hist(div_vals, bins=50, color=color, edgecolor="white",
                linewidth=0.4, alpha=0.85)
        ax.axvline(np.median(div_vals), color="black", linestyle="--",
                   linewidth=1.2, label=f"Median {np.median(div_vals):.1f}%")
        ax.set_xlabel(age_label, fontsize=10)
        ax.set_ylabel("Locus count", fontsize=10)
        ax.set_title(f"{family}\n({meta['type']})", fontweight="bold")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    _pp(f"  Saved {out_path}")


# ─── differential expression ─────────────────────────────────────────────────

def differential_expression(df: pd.DataFrame, expr_cols: list) -> pd.DataFrame:
    """Pairwise Wilcoxon rank-sum tests between HDBSCAN clusters, BH-corrected."""
    try:
        from scipy.stats import mannwhitneyu
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        _pp("  [DE] scipy/statsmodels not available --- skipping DE analysis")
        return pd.DataFrame()

    cluster_col = "Cluster" if "Cluster" in df.columns else "cluster"
    clusters = sorted(c for c in df[cluster_col].unique() if c >= 0)
    if len(clusters) < 2:
        _pp("  [DE] fewer than 2 clusters --- skipping DE")
        return pd.DataFrame()

    rows = []
    for c1, c2 in combinations(clusters, 2):
        g1 = df[df[cluster_col] == c1]
        g2 = df[df[cluster_col] == c2]
        for col in expr_cols:
            v1 = g1[col].dropna().values
            v2 = g2[col].dropna().values
            if len(v1) < 3 or len(v2) < 3:
                continue
            try:
                stat, p = mannwhitneyu(v1, v2, alternative="two-sided")
                rows.append({
                    "cluster1": c1, "cluster2": c2, "stage": col,
                    "n1": len(v1), "n2": len(v2),
                    "mean1": float(np.mean(v1)), "mean2": float(np.mean(v2)),
                    "stat": stat, "pval": p,
                })
            except Exception:
                pass

    if not rows:
        return pd.DataFrame()

    de_df = pd.DataFrame(rows)
    _, padj, _, _ = multipletests(de_df["pval"].values, method="fdr_bh")
    de_df["padj"] = padj
    de_df["sig"] = de_df["padj"] < 0.05
    n_sig = int(de_df["sig"].sum())
    _pp(f"  [DE] {n_sig}/{len(de_df)} comparisons significant (BH-FDR < 0.05)")
    return de_df


def plot_de_heatmap(de_df: pd.DataFrame, family: str, out_path: Path):
    """Plot -log10(padj) heatmap of DE results between cluster pairs."""
    if de_df.empty:
        _pp(f"  [DE] no results to plot for {family}")
        return

    pairs = [(r["cluster1"], r["cluster2"]) for _, r in de_df.drop_duplicates(
        subset=["cluster1","cluster2"]).iterrows()]
    stages = de_df["stage"].unique().tolist()

    mat = np.zeros((len(pairs), len(stages)))
    for i, (c1, c2) in enumerate(pairs):
        for j, st in enumerate(stages):
            sub = de_df[(de_df["cluster1"]==c1) & (de_df["cluster2"]==c2) & (de_df["stage"]==st)]
            if not sub.empty:
                mat[i, j] = -np.log10(max(sub["padj"].values[0], 1e-300))

    fig, ax = plt.subplots(figsize=(max(6, len(stages)*0.9), max(3, len(pairs)*0.7 + 1.5)))
    im = ax.imshow(mat, aspect="auto", cmap="OrRd")
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([f"C{c1} vs C{c2}" for c1, c2 in pairs], fontsize=9)
    plt.colorbar(im, ax=ax, label="-log10(BH-adjusted p)", shrink=0.8)
    ax.set_title(f"{family} --- Cluster DE (Wilcoxon, BH-corrected)", fontweight="bold")
    ax.axhline(0.5, color="white", linewidth=0.5)

    # Mark significant cells
    for i in range(len(pairs)):
        for j in range(len(stages)):
            c1, c2 = pairs[i]
            sub = de_df[(de_df["cluster1"]==c1) & (de_df["cluster2"]==c2) & (de_df["stage"]==stages[j])]
            if not sub.empty and sub["sig"].values[0]:
                ax.text(j, i, "*", ha="center", va="center", fontsize=11, color="black")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    _pp(f"  Saved {out_path}")


# ─── per-family 3-panel result figure ────────────────────────────────────────

def plot_family_results(df: pd.DataFrame, family: str, expr_cols: list,
                        out_path: Path):
    """Three-panel figure: UMAP clusters | expression heatmap | Kimura divergence."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    meta = FAMILIES[family]
    color = meta["color"]

    cluster_col = "Cluster" if "Cluster" in df.columns else "cluster"
    clusters = sorted(c for c in df[cluster_col].unique() if c >= 0)
    cmap_vals = {c: i for i, c in enumerate(clusters + [-1])}

    # ── Panel a: UMAP ──────────────────────────────────────────────────────────
    ax = axes[0]
    for cid in sorted(set(df[cluster_col].unique())):
        sub = df[df[cluster_col] == cid]
        label = f"Cluster {cid}" if cid >= 0 else "Noise"
        c = _PALETTE[cid % len(_PALETTE)] if cid >= 0 else "#cccccc"
        ax.scatter(sub["umap_x"], sub["umap_y"], s=6, c=c, alpha=0.7, label=label)
    ax.legend(fontsize=7, markerscale=2)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(f"(a) {family} UMAP clustering\n({len(clusters)} clusters)", fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)

    # ── Panel b: per-cluster expression heatmap ────────────────────────────────
    ax = axes[1]
    if expr_cols:
        heat = np.array([
            [np.log1p(df[df[cluster_col]==c][col].dropna().mean())
             for col in expr_cols]
            for c in clusters
        ])
        if heat.size:
            im = ax.imshow(heat, aspect="auto", cmap="YlOrRd")
            ax.set_xticks(range(len(expr_cols)))
            ax.set_xticklabels(expr_cols, rotation=35, ha="right", fontsize=8)
            ax.set_yticks(range(len(clusters)))
            ax.set_yticklabels([f"Cluster {c}" for c in clusters], fontsize=9)
            plt.colorbar(im, ax=ax, label="log1p(mean counts)", shrink=0.8)
            ax.set_title(f"(b) Expression per cluster\n(log1p mean)", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No expression data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("(b) Expression (no data)")

    # ── Panel c: Kimura divergence ─────────────────────────────────────────────
    ax = axes[2]
    if "milliDiv" in df.columns:
        div_vals = df["milliDiv"].dropna().values / 10.0
        ax.hist(div_vals, bins=40, color=color, edgecolor="white",
                linewidth=0.4, alpha=0.85)
        ax.axvline(np.median(div_vals), color="black", linestyle="--",
                   linewidth=1.2, label=f"Median {np.median(div_vals):.1f}%")
        ax.set_xlabel("Kimura divergence (%)"); ax.set_ylabel("Locus count")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "milliDiv not available", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_title(f"(c) Repeat landscape\n({meta['type']})", fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    plt.suptitle(f"{family} ({meta['type']}) --- GAMECA Analysis",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    _pp(f"  Saved {out_path}")


# ─── combined cross-family figure ─────────────────────────────────────────────

def plot_combined_kimura(family_frames: dict, out_path: Path):
    """Single combined figure: Kimura divergence for LINE, SINE, LTR side-by-side."""
    plot_kimura_divergence(family_frames, out_path)


def plot_combined_clustering(family_results: dict, out_path: Path):
    """UMAP panels for all three families in one figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("UMAP Clustering --- LINE, SINE, LTR TE Families", fontsize=14,
                 fontweight="bold")

    for ax, (family, (df, _)) in zip(axes, family_results.items()):
        meta = FAMILIES[family]
        cluster_col = "Cluster" if "Cluster" in df.columns else "cluster"
        for cid in sorted(set(df[cluster_col].unique())):
            sub = df[df[cluster_col] == cid]
            label = f"C{cid}" if cid >= 0 else "Noise"
            c = _PALETTE[cid % len(_PALETTE)] if cid >= 0 else "#cccccc"
            ax.scatter(sub["umap_x"], sub["umap_y"], s=5, c=c, alpha=0.7, label=label)
        ax.legend(fontsize=7, markerscale=2)
        ax.set_title(f"{family}\n({meta['type']})", fontweight="bold")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    _pp(f"  Saved {out_path}")


# ─── prep: RMSK / Dfam ───────────────────────────────────────────────────────

def prep_family(family: str, build: str, source: str, rmsk_dir: str,
                genome_fa: str, max_loci: int) -> pd.DataFrame:
    """Fetch loci and sequences for a TE family. Wraps te_prep functions."""
    import te_prep as _prep

    std_chroms = _prep.STD_CHROMS_MOUSE if build.startswith("mm") else _prep.STD_CHROMS_HUMAN

    _pp(f"  [{family}] fetching loci from {source.upper()}...")
    if source == "dfam":
        hits = _prep.parse_dfam_family(None, family, build=build, std_chroms=std_chroms)
    else:
        rmsk_path = _prep.get_rmsk_path(build, rmsk_dir)
        hits = _prep.parse_rmsk_family(rmsk_path, family, std_chroms=std_chroms)

    if not hits:
        _pp(f"  [{family}] FATAL: no loci found")
        sys.exit(1)

    df = pd.DataFrame(hits).sort_values(["Chromosome","Start"]).reset_index(drop=True)
    if max_loci and len(df) > max_loci:
        _pp(f"  [{family}] capping to {max_loci} loci (--max-loci)")
        df = df.head(max_loci).reset_index(drop=True)

    _pp(f"  [{family}] {len(df):,} loci across {df['Chromosome'].nunique()} chromosomes")

    # Extract sequences
    _pp(f"  [{family}] extracting sequences...")
    if genome_fa and os.path.exists(genome_fa):
        seqs = _prep.extract_sequences(genome_fa, df)
    else:
        _pp(f"  [{family}] no genome FASTA --- using UCSC API (build={build})")
        seqs, _ = _prep.fetch_sequences_ucsc(df, assembly=build, n_workers=3)

    df["Seq"] = seqs
    df["chr"]   = df["Chromosome"]
    df["start"] = df["Start"]
    df["stop"]  = df["Stop"]

    before = len(df)
    df = df[df["Seq"].str.len() > 0].reset_index(drop=True)
    if len(df) < before:
        _pp(f"  [{family}] dropped {before - len(df)} empty sequences")

    df["Length"] = df["Seq"].str.len()
    df["GC_Content"] = df["Seq"].apply(
        lambda s: (s.upper().count("G") + s.upper().count("C")) / len(s) if len(s) > 0 else 0
    )
    return df


# ─── measured values / text report ───────────────────────────────────────────

def write_measured_values(results: dict, reports_dir: Path):
    """Write measured_values.tex (LaTeX macros) and report_numbers.txt (human-readable)."""
    lines_tex = [
        "% Auto-generated by run_line_sine_ltr_analysis.py",
        f"% {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "% ─── existing MT2_Mm macros (placeholders --- run full pipeline to fill) ───",
        r"\providecommand{\mtTwoLoci}{(pending run)}",
        r"\providecommand{\mtTwoClusters}{(pending run)}",
        r"\providecommand{\mtTwoExprLoci}{(pending run)}",
        r"\providecommand{\genomecacheSpeedup}{(pending run)}",
        r"\providecommand{\genomecacheReads}{(pending run)}",
        r"\providecommand{\cythonSpeedupGC}{(pending run)}",
        r"\providecommand{\cythonSpeedupKmer}{(pending run)}",
        "",
        "% ─── LINE / SINE / LTR analysis macros ─────────────────────────────────",
    ]
    lines_txt = [
        "=" * 60,
        "GAMECA LINE / SINE / LTR Analysis --- Measured Values",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
    ]

    macro_prefix = {
        "L1Md_T":     "lOneT",
        "B1_Mus2":    "bOneMT",
        "IAPLTR1_Mm": "iapltrOne",
    }

    for family, res in results.items():
        pfx = macro_prefix[family]
        n_loci   = res.get("n_loci",    0)
        n_clust  = res.get("n_clusters", 0)
        n_expr   = res.get("n_expr",     0)
        divisor  = res.get("divisor",    "---")
        n_neigh  = res.get("n_neighbors","---")
        n_de_sig = res.get("n_de_sig",   0)
        med_div  = res.get("median_div", 0.0)
        mcs      = res.get("min_cluster_size", "---")

        lines_tex += [
            f"% {family} ({FAMILIES[family]['type']})",
            rf"\providecommand{{\{pfx}Loci}}{{{n_loci:,}}}",
            rf"\providecommand{{\{pfx}Clusters}}{{{n_clust}}}",
            rf"\providecommand{{\{pfx}ExprLoci}}{{{n_expr:,}}}",
            rf"\providecommand{{\{pfx}Divisor}}{{{divisor}}}",
            rf"\providecommand{{\{pfx}Neighbors}}{{{n_neigh}}}",
            rf"\providecommand{{\{pfx}MinClusterSize}}{{{mcs}}}",
            rf"\providecommand{{\{pfx}DeSig}}{{{n_de_sig}}}",
            rf"\providecommand{{\{pfx}MedianDiv}}{{{med_div:.1f}}}",
            "",
        ]
        lines_txt += [
            "",
            f"── {family} ({FAMILIES[family]['type']}) ──────────────────────────────",
            f"  Loci:               {n_loci:,}",
            f"  Clusters:           {n_clust}",
            f"  Expression loci:    {n_expr:,}",
            f"  Clustering divisor: n/{divisor}  (min_cluster_size={mcs})",
            f"  UMAP n_neighbors:   {n_neigh}",
            f"  Median Kimura div.: {med_div:.1f}%",
            f"  DE sig comparisons: {n_de_sig} (BH-FDR < 0.05)",
        ]

    # Write
    tex_path = reports_dir / "measured_values.tex"
    tex_path.write_text("\n".join(lines_tex) + "\n")
    _pp(f"  Written: {tex_path}")

    txt_path = reports_dir / "report_numbers.txt"
    txt_path.write_text("\n".join(lines_txt) + "\n")
    _pp(f"  Written: {txt_path}")

    return tex_path, txt_path


# ─── argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="LINE/SINE/LTR TE family analysis --- batch mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--l1mdt-expr",  required=True,
                   help="Path to L1Md_T_ultracombo.csv on cluster")
    p.add_argument("--b1mus2-expr", required=True,
                   help="Path to B1_Mus2_ultracombo.csv on cluster")
    p.add_argument("--iapltr1-expr",required=True,
                   help="Path to IAPLTR1_Mm_ultracombo.csv on cluster")
    p.add_argument("--reports-dir", default="/home/amodz/anmol/reports4",
                   help="Directory to save figures and measured_values.tex")
    p.add_argument("--genome-fa",   default="",
                   help="Path to mm10.fa on cluster (optional; UCSC API fallback)")
    p.add_argument("--rmsk-dir",    default=os.path.expanduser("~/te_analysis/rmsk"),
                   help="Directory for cached RMSK gz files")
    p.add_argument("--source",      choices=["rmsk","dfam"], default="rmsk",
                   help="Annotation source (default: rmsk)")
    p.add_argument("--build",       default="mm10",
                   help="Genome build (default: mm10)")
    p.add_argument("--max-loci",    type=int, default=None,
                   help="Cap number of loci per family (for testing)")
    p.add_argument("--kmer",        type=int, default=6)
    p.add_argument("--pca-dims",    type=int, default=50)
    p.add_argument("--n-epochs",    type=int, default=200)
    p.add_argument("--random-state",type=int, default=42)
    p.add_argument("--force", action="store_true",
                   help="Ignore all cached outputs and rerun every step")
    return p.parse_args()


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    expr_paths = {
        "L1Md_T":     args.l1mdt_expr,
        "B1_Mus2":    args.b1mus2_expr,
        "IAPLTR1_Mm": args.iapltr1_expr,
    }

    _pp("=" * 60)
    _pp("GAMECA --- LINE / SINE / LTR Cross-Family Analysis")
    _pp(f"  Build:       {args.build}")
    _pp(f"  Source:      {args.source}")
    _pp(f"  Genome FA:   {args.genome_fa or '(none --- UCSC API)'}")
    _pp(f"  Reports dir: {reports_dir}")
    _pp(f"  Force rerun: {args.force}")
    _pp("=" * 60)

    family_loci: dict = {}
    family_results: dict = {}
    analysis_meta: dict = {}

    # ─── Step 1: Prep, expression match, and cluster each family ───────────────
    for family, expr_path in expr_paths.items():
        _pp(f"\n{'='*60}")
        _pp(f"FAMILY: {family}  ({FAMILIES[family]['type']})")
        _pp(f"{'='*60}")

        loci_cache   = reports_dir / f"cache_{family.lower()}_loci.pkl"
        clust_cache  = reports_dir / f"cache_{family.lower()}_clustered.pkl"
        labels_cache = reports_dir / f"cache_{family.lower()}_labels.npy"
        meta_cache   = reports_dir / f"cache_{family.lower()}_meta.json"
        de_path      = reports_dir / f"de_{family.lower()}.csv"

        # ── 1a. Loci + sequences ──────────────────────────────────────────────
        if not args.force and loci_cache.exists():
            _pp(f"  [cache] loading loci from {loci_cache.name}")
            df = pd.read_pickle(loci_cache)
        else:
            df = prep_family(
                family, args.build, args.source, args.rmsk_dir,
                args.genome_fa, args.max_loci,
            )
            df.to_pickle(loci_cache)
            _pp(f"  [cache] saved loci → {loci_cache.name}")

        n_loci  = len(df)
        med_div = float(df["milliDiv"].median()) / 10.0 if "milliDiv" in df.columns else 0.0
        family_loci[family] = df.copy()

        # ── 1b + 1c. Expression match + clustering ────────────────────────────
        # All three cluster artefacts must exist together to be a valid cache hit.
        clust_cached = (
            not args.force
            and clust_cache.exists()
            and labels_cache.exists()
            and meta_cache.exists()
        )
        if clust_cached:
            _pp(f"  [cache] loading clustered data from {clust_cache.name}")
            df_c        = pd.read_pickle(clust_cache)
            labels      = np.load(labels_cache)
            cached_meta = json.loads(meta_cache.read_text())
            divisor     = cached_meta["divisor"]
            nn          = cached_meta["n_neighbors"]
            n_clusters  = cached_meta["n_clusters"]
            expr_cols   = cached_meta.get("expr_cols", [])
            n_expr      = cached_meta.get("n_expr", 0)
        else:
            # Need expression-matched df before clustering
            if os.path.exists(expr_path):
                df, expr_cols, n_expr = match_expression(df, expr_path)
            else:
                _pp(f"  WARNING: expression file not found: {expr_path}")
                expr_cols, n_expr = [], 0

            df_c, labels, divisor, nn, n_clusters = adaptive_clustering(
                df, family, reports_dir,
                kmer=args.kmer,
                pca_dims=args.pca_dims,
                n_epochs=args.n_epochs,
                random_state=args.random_state,
            )
            df_c.to_pickle(clust_cache)
            np.save(labels_cache, labels)
            _pp(f"  [cache] saved clustering → {clust_cache.name}")

        family_results[family] = (df_c, labels)
        mcs = max(5, n_loci // divisor)

        # ── 1d. Differential expression ───────────────────────────────────────
        if not args.force and de_path.exists():
            _pp(f"  [cache] loading DE results from {de_path.name}")
            de_df    = pd.read_csv(de_path)
            n_de_sig = int(de_df["sig"].sum()) if "sig" in de_df else 0
        else:
            de_df    = differential_expression(df_c, expr_cols)
            n_de_sig = int(de_df["sig"].sum()) if not de_df.empty and "sig" in de_df else 0
            if not de_df.empty:
                de_df.to_csv(de_path, index=False)
                _pp(f"  DE results → {de_path}")

        # ── 1e. DE heatmap ────────────────────────────────────────────────────
        de_fig_path = reports_dir / f"fig_de_{family.lower().replace('_','')}.pdf"
        if not _exists(de_fig_path, args.force):
            plot_de_heatmap(de_df, family, de_fig_path)

        # ── 1f. Per-family results figure ─────────────────────────────────────
        fig_path = reports_dir / f"fig_{family.lower().replace('_','')}_results.pdf"
        if not _exists(fig_path, args.force):
            plot_family_results(df_c, family, expr_cols, fig_path)

        # ── Save per-family meta cache ─────────────────────────────────────────
        analysis_meta[family] = {
            "n_loci":           n_loci,
            "n_clusters":       n_clusters,
            "n_expr":           n_expr,
            "divisor":          divisor,
            "n_neighbors":      nn,
            "min_cluster_size": mcs,
            "n_de_sig":         n_de_sig,
            "median_div":       round(med_div, 2),
        }
        if args.force or not meta_cache.exists():
            meta_to_save = dict(analysis_meta[family])
            meta_to_save["expr_cols"] = expr_cols
            meta_cache.write_text(json.dumps(meta_to_save, indent=2))
            _pp(f"  [cache] saved meta → {meta_cache.name}")

        _pp(f"  [{family}] DONE: {n_loci:,} loci, {n_clusters} clusters, "
            f"divisor=n/{divisor}, nn={nn}, {n_expr} expr loci")

    # ─── Step 2: Combined / multi-family figures ───────────────────────────────
    _pp(f"\n{'='*60}")
    _pp("Generating combined figures...")
    _pp(f"{'='*60}")

    kimura_path = reports_dir / "fig_kimura_divergence.pdf"
    if not _exists(kimura_path, args.force):
        plot_combined_kimura(family_loci, kimura_path)

    clustering_path = reports_dir / "fig_line_sine_ltr_clustering.pdf"
    if not _exists(clustering_path, args.force):
        plot_combined_clustering(family_results, clustering_path)

    # ─── Step 3: Named figures matching the report ─────────────────────────────
    for family, out_name in [
        ("L1Md_T",     "fig_line_results"),
        ("B1_Mus2",    "fig_sine_results"),
        ("IAPLTR1_Mm", "fig_ltr_results"),
    ]:
        src = reports_dir / f"fig_{family.lower().replace('_','')}_results.pdf"
        dst = reports_dir / f"{out_name}.pdf"
        if src.exists() and src != dst:
            if not _exists(dst, args.force):
                shutil.copy(src, dst)
                _pp(f"  Copied {src.name} → {dst.name}")

    # ─── Step 4: Write measured values ─────────────────────────────────────────
    tex_path = reports_dir / "measured_values.tex"
    txt_path = reports_dir / "report_numbers.txt"
    if not args.force and tex_path.exists() and txt_path.exists():
        _pp(f"  [skip] measured_values.tex and report_numbers.txt already exist")
    else:
        _pp(f"\n{'='*60}")
        _pp("Writing measured values...")
        tex_path, txt_path = write_measured_values(analysis_meta, reports_dir)

    _pp(f"\n{'='*60}")
    _pp("ALL DONE")
    _pp(f"  Figures:           {reports_dir}")
    _pp(f"  measured_values:   {tex_path}")
    _pp(f"  report_numbers:    {txt_path}")
    _pp(f"{'='*60}")

    print("\n" + "=" * 60)
    print("TEXT REPORT (copy-paste into GAMECA report):")
    print("=" * 60)
    print(txt_path.read_text())


if __name__ == "__main__":
    main()
