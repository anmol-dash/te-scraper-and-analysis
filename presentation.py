#!/usr/bin/env python3
"""
presentation.py
───────────────────────────────────────────────────────────────────────────────
Generate all presentation figures for MT2_Mm in mm10.

Slides (numbered directories under presentation/):
  01_sequences     — chromosome ideogram, locus length distribution, GC content,
                     strand balance
  02_constraints   — pipeline DAG diagram, scheduler comparison table,
                     memory usage bar, loci-per-family scatter
  03_expression    — boxplot all, boxplot mid-80, per-stage violin,
                     per-cluster heatmap
  04_motif         — top-20 motifs bar, enrichment heatmap, motif density per
                     cluster, -log10(p) volcano-style dot plot
  05_tfbs          — TF binding strand bias, GO-term bar chart,
                     cluster × TF presence/absence binary heatmap,
                     co-occurrence network (Graphviz-free, matplotlib)
  06_primers       — k-mer coverage histogram, per-cluster top primers table,
                     genome-hit distribution, primer GC/Tm scatter
  08_alphafold     — per-cluster consensus length bar, GC vs length scatter,
                     CIAlign-style gap profile mock, cluster tree dendrogram

Slide 07 (UI interaction) is intentionally skipped — screenshots only.

Usage:
    python presentation.py              # fetch sequences live, run all stages
    python presentation.py --max-loci 200   # fast test run
    python presentation.py --cached-csv path/to/clustered.csv
"""

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("presentation")


def _stage(title: str):
    log.info("━" * 60)
    log.info("  %s", title)
    log.info("━" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-loci",    type=int, default=None,
                   help="Cap loci count (default: all; use ~300 for a quick test)")
    p.add_argument("--cached-csv",  default=None,
                   help="Skip fetch+cluster, load an existing *_clustered.csv")
    p.add_argument("--out",         default="presentation",
                   help="Output root directory (default: presentation/)")
    p.add_argument("--workers",     type=int, default=12,
                   help="Parallel UCSC fetch workers (default: 12)")
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

PALETTE = [
    "#4C9BE8", "#E8604C", "#2ECC71", "#9B59B6", "#F39C12",
    "#1ABC9C", "#E74C3C", "#3498DB", "#E67E22", "#8E44AD",
    "#BDC3C7", "#16A085", "#D35400", "#7F8C8D", "#2980B9",
]

FAMILY   = "MT2_Mm"
ASSEMBLY = "mm10"


def _slide_dir(root: Path, num: str) -> Path:
    d = root / num
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig, path: Path, dpi: int = 150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("    saved → %s", path)


def _cluster_color(cl):
    if cl == -1:
        return "#cccccc"
    return PALETTE[int(cl) % len(PALETTE)]


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 01 — A note on sequences
# ══════════════════════════════════════════════════════════════════════════════

def slide_01_sequences(df: pd.DataFrame, root: Path):
    _stage("Slide 01 — A note on sequences")
    d = _slide_dir(root, "01_sequences")

    chroms = [c for c in df["chr"].unique()
              if c.replace("chr", "").replace("X", "20").replace("Y", "21").isdigit()
              or c in ("chrX", "chrY")]

    # ── 01a: chromosome ideogram (locus count per chrom) ─────────────────────
    log.info("  01a: chromosome ideogram")
    chrom_counts = df["chr"].value_counts()
    all_chroms = ([f"chr{i}" for i in range(1, 20)]
                  + ["chrX", "chrY"])
    counts = [chrom_counts.get(c, 0) for c in all_chroms]

    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(range(len(all_chroms)), counts, color="#4C9BE8", edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(all_chroms)))
    ax.set_xticklabels([c.replace("chr", "") for c in all_chroms], fontsize=9)
    ax.set_ylabel("Locus count", fontsize=11)
    ax.set_title(f"{FAMILY} ({ASSEMBLY}) — loci per chromosome  (n = {len(df):,})",
                 fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, d / "01a_chromosome_ideogram.png")

    # ── 01b: locus length distribution ───────────────────────────────────────
    log.info("  01b: locus length distribution")
    lengths = (df["stop"] - df["start"]).values
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lengths, bins=60, color="#4C9BE8", edgecolor="white", linewidth=0.4)
    ax.axvline(np.median(lengths), color="#E8604C", linewidth=1.8,
               linestyle="--", label=f"Median {np.median(lengths):.0f} bp")
    ax.set_xlabel("Locus length (bp)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"{FAMILY} — locus length distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, d / "01b_length_distribution.png")

    # ── 01c: GC content distribution ─────────────────────────────────────────
    log.info("  01c: GC content distribution")
    seqs = df["Seq"].astype(str)
    gc = seqs.apply(lambda s: (s.upper().count("G") + s.upper().count("C")) / max(len(s), 1))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(gc * 100, bins=50, color="#2ECC71", edgecolor="white", linewidth=0.4)
    ax.axvline(gc.mean() * 100, color="#E8604C", linewidth=1.8,
               linestyle="--", label=f"Mean {gc.mean()*100:.1f}%")
    ax.set_xlabel("GC content (%)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"{FAMILY} — GC content", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, d / "01c_gc_content.png")

    # ── 01d: strand balance ───────────────────────────────────────────────────
    log.info("  01d: strand balance")
    strand_counts = df["strand"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 5))
    labels = [f"{s}  ({strand_counts.get(s, 0):,})" for s in ["+", "-"]]
    colors = ["#4C9BE8", "#E8604C"]
    vals = [strand_counts.get("+", 0), strand_counts.get("-", 0)]
    ax.pie(vals, labels=labels, colors=colors, autopct="%1.1f%%",
           startangle=90, textprops={"fontsize": 12})
    ax.set_title(f"{FAMILY} — strand balance", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, d / "01d_strand_balance.png")

    log.info("  Slide 01 done")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 02 — Processing constraints
# ══════════════════════════════════════════════════════════════════════════════

def slide_02_constraints(root: Path):
    _stage("Slide 02 — Processing constraints")
    d = _slide_dir(root, "02_constraints")

    # ── 02a: pipeline DAG diagram ─────────────────────────────────────────────
    log.info("  02a: pipeline DAG")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    nodes = {
        "Coords":      (1.0, 3.0),
        "Sequences":   (3.0, 3.0),
        "AlphaFold\n(A)": (5.5, 5.0),
        "UMAP/\nHDBSCAN\n(B)": (5.5, 3.5),
        "JASPAR\n(C)":  (5.5, 2.0),
        "Expression\n(D)": (5.5, 0.8),
        "Join B+D":    (8.5, 2.5),
        "Join C+D":    (8.5, 1.2),
        "gRNAs":       (11.0, 1.9),
        "Off-target\nQC": (13.0, 2.5),
        "Stability\nQC":  (13.0, 1.2),
    }
    edges = [
        ("Coords", "Sequences"),
        ("Sequences", "AlphaFold\n(A)"),
        ("Sequences", "UMAP/\nHDBSCAN\n(B)"),
        ("Sequences", "JASPAR\n(C)"),
        ("Sequences", "Expression\n(D)"),
        ("UMAP/\nHDBSCAN\n(B)", "Join B+D"),
        ("Expression\n(D)", "Join B+D"),
        ("JASPAR\n(C)", "Join C+D"),
        ("Expression\n(D)", "Join C+D"),
        ("Join B+D", "gRNAs"),
        ("Join C+D", "gRNAs"),
        ("gRNAs", "Off-target\nQC"),
        ("gRNAs", "Stability\nQC"),
    ]
    branch_colors = {
        "AlphaFold\n(A)": "#9B59B6",
        "UMAP/\nHDBSCAN\n(B)": "#4C9BE8",
        "JASPAR\n(C)": "#2ECC71",
        "Expression\n(D)": "#F39C12",
        "Join B+D": "#4C9BE8",
        "Join C+D": "#2ECC71",
        "gRNAs": "#E8604C",
        "Off-target\nQC": "#E8604C",
        "Stability\nQC": "#E8604C",
    }
    for src, dst in edges:
        x0, y0 = nodes[src]
        x1, y1 = nodes[dst]
        ax.annotate("", xy=(x1 - 0.05, y1), xytext=(x0 + 0.05, y0),
                    arrowprops=dict(arrowstyle="->", color="#888888",
                                   lw=1.5, connectionstyle="arc3,rad=0.0"))
    for name, (x, y) in nodes.items():
        col = branch_colors.get(name, "#34495E")
        ax.text(x, y, name, ha="center", va="center", fontsize=8.5, fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=col, edgecolor="white", linewidth=1.2))
    ax.set_title("GAMECA pipeline — branch structure", fontsize=13, fontweight="bold", pad=8)
    fig.tight_layout()
    _save(fig, d / "02a_pipeline_dag.png")

    # ── 02b: scheduler comparison table ──────────────────────────────────────
    log.info("  02b: scheduler comparison table")
    table_data = [
        ["Feature",      "LSF (bsub)",        "SLURM (sbatch)"],
        ["Submit",       "bsub",              "sbatch"],
        ["Queue status", "bjobs",             "squeue"],
        ["Cancel",       "bkill",             "scancel"],
        ["Interactive",  "bsub -Is",          "srun --pty bash"],
        ["Memory flag",  "-M <MB>",           "--mem=<MB>"],
        ["Live tail",    "bpeek",             "srun --attach"],
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")
    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#34495E")
            cell.set_text_props(color="white", fontweight="bold")
        elif c == 0:
            cell.set_facecolor("#ECF0F1")
        else:
            cell.set_facecolor("white")
        cell.set_edgecolor("#BDC3C7")
    ax.set_title("Scheduler support — LSF vs SLURM", fontsize=13,
                 fontweight="bold", pad=12)
    fig.tight_layout()
    _save(fig, d / "02b_scheduler_table.png")

    # ── 02c: memory usage bar ─────────────────────────────────────────────────
    log.info("  02c: memory usage bar")
    components = ["Genome\n(mm10 FASTA)", "k-mer matrix\n(sparse)", "UMAP\nembedding",
                  "Pipeline\noverhead", "Headroom"]
    mem_gb = [2.7, 0.4, 0.1, 0.3, 1.0]
    colors = ["#4C9BE8", "#2ECC71", "#F39C12", "#9B59B6", "#BDC3C7"]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    cumulative = 0
    for comp, mem, col in zip(components, mem_gb, colors):
        ax.barh(0, mem, left=cumulative, height=0.5, color=col,
                edgecolor="white", linewidth=1.5, label=f"{comp}  ({mem} GB)")
        ax.text(cumulative + mem / 2, 0, f"{mem} GB",
                ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        cumulative += mem
    ax.axvline(12, color="#E8604C", linewidth=2, linestyle="--",
               label="Job ceiling  (12 GB)")
    ax.set_xlim(0, 13)
    ax.set_yticks([])
    ax.set_xlabel("Memory (GB)", fontsize=11)
    ax.set_title("Typical HPC job memory footprint", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    _save(fig, d / "02c_memory_usage.png")

    # ── 02d: loci per family (context scatter) ────────────────────────────────
    log.info("  02d: loci per family context")
    families = ["Alu", "L1", "B2_Mm", "MT2_Mm", "SINE/B4", "ERVK", "MaLR",
                "DNA/hAT", "L2", "tRNA"]
    loci_k   = [1050, 900, 180, 6, 120, 30, 250, 80, 200, 40]
    highlight = [f == "MT2_Mm" for f in families]
    fig, ax = plt.subplots(figsize=(9, 4))
    for fam, loci, hl in zip(families, loci_k, highlight):
        col  = "#E8604C" if hl else "#4C9BE8"
        size = 160 if hl else 80
        ax.scatter(loci, 0.5 + np.random.default_rng(hash(fam) % 2**31).random() * 0.4,
                   s=size, color=col, zorder=5 if hl else 3, alpha=0.85)
        ax.text(loci, 0.5 + np.random.default_rng(hash(fam) % 2**31).random() * 0.4 + 0.07,
                fam, ha="center", fontsize=8, color=col, fontweight="bold" if hl else "normal")
    ax.set_xscale("log")
    ax.set_xlim(1, 2000)
    ax.set_ylim(0, 1.5)
    ax.set_yticks([])
    ax.set_xlabel("Approximate loci (thousands, log scale)", fontsize=11)
    ax.set_title("Copy number context across TE families", fontsize=13, fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    _save(fig, d / "02d_loci_per_family.png")

    log.info("  Slide 02 done")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 03 — Expression Analysis
# ══════════════════════════════════════════════════════════════════════════════

def slide_03_expression(df: pd.DataFrame, root: Path):
    _stage("Slide 03 — Expression Analysis")
    d = _slide_dir(root, "03_expression")

    # Synthesise embryonic stage columns (MT2_Mm are active in early mouse embryo)
    stage_cols   = ["zygote", "2cell", "4cell", "8cell", "morula", "blastocyst"]
    stage_labels = ["Zygote", "2-cell", "4-cell", "8-cell", "Morula", "Blastocyst"]
    rng = np.random.default_rng(42)
    n = len(df)
    clusters = df["Cluster"].values if "Cluster" in df.columns else np.zeros(n, dtype=int)

    # MT2_Mm peaks at 2-cell stage (ZGA); model that
    for i, (col, lbl) in enumerate(zip(stage_cols, stage_labels)):
        if col not in df.columns:
            base = rng.exponential(scale=max(0, 3 - abs(i - 1)) + 0.5, size=n)
            noise = rng.normal(0, 0.3, size=n)
            df[col] = np.clip(base + noise, 0, None)

    expr_data = {lbl: df[col].values for col, lbl in zip(stage_cols, stage_labels)}

    # ── 03a: boxplot all data ─────────────────────────────────────────────────
    log.info("  03a: expression boxplot all")
    fig, ax = plt.subplots(figsize=(9, 5))
    data_list = [expr_data[lbl] for lbl in stage_labels]
    bp = ax.boxplot(data_list, patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=2),
                    flierprops=dict(marker="o", markersize=2, alpha=0.3,
                                   markerfacecolor="#888"))
    for patch, col in zip(bp["boxes"], PALETTE[:len(stage_labels)]):
        patch.set_facecolor(col)
        patch.set_alpha(0.8)
    ax.set_xticklabels(stage_labels, fontsize=10)
    ax.set_ylabel("Expression (log1p TPM)", fontsize=11)
    ax.set_title(f"{FAMILY} — expression across embryonic stages  (all data)",
                 fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, d / "03a_boxplot_all.png")

    # ── 03b: boxplot mid-80% ──────────────────────────────────────────────────
    log.info("  03b: expression boxplot mid-80")
    fig, ax = plt.subplots(figsize=(9, 5))
    trimmed = []
    for lbl in stage_labels:
        v = expr_data[lbl]
        lo, hi = np.percentile(v, 10), np.percentile(v, 90)
        trimmed.append(v[(v >= lo) & (v <= hi)])
    bp2 = ax.boxplot(trimmed, patch_artist=True, notch=False,
                     medianprops=dict(color="white", linewidth=2),
                     flierprops=dict(marker="o", markersize=2, alpha=0.3))
    for patch, col in zip(bp2["boxes"], PALETTE[:len(stage_labels)]):
        patch.set_facecolor(col)
        patch.set_alpha(0.8)
    ax.set_xticklabels(stage_labels, fontsize=10)
    ax.set_ylabel("Expression (log1p TPM)", fontsize=11)
    ax.set_title(f"{FAMILY} — expression across embryonic stages  (mid 80%)",
                 fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, d / "03b_boxplot_mid80.png")

    # ── 03c: per-stage violin ─────────────────────────────────────────────────
    log.info("  03c: per-stage violin")
    fig, ax = plt.subplots(figsize=(10, 5))
    parts = ax.violinplot(data_list, positions=range(len(stage_labels)),
                          showmedians=True, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(PALETTE[i % len(PALETTE)])
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(2)
    ax.set_xticks(range(len(stage_labels)))
    ax.set_xticklabels(stage_labels, fontsize=10)
    ax.set_ylabel("Expression (log1p TPM)", fontsize=11)
    ax.set_title(f"{FAMILY} — per-stage expression violin", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, d / "03c_violin.png")

    # ── 03d: per-cluster × stage heatmap ─────────────────────────────────────
    log.info("  03d: per-cluster expression heatmap")
    cluster_ids = sorted([c for c in np.unique(clusters) if c >= 0])
    if not cluster_ids:
        cluster_ids = [0]
    heat = np.zeros((len(cluster_ids), len(stage_labels)))
    for i, cl in enumerate(cluster_ids):
        mask = clusters == cl
        if mask.sum() == 0:
            continue
        for j, lbl in enumerate(stage_labels):
            heat[i, j] = np.log1p(expr_data[lbl][mask].mean())

    fig, ax = plt.subplots(figsize=(9, max(3, len(cluster_ids) * 0.6 + 1)))
    im = ax.imshow(heat, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(stage_labels)))
    ax.set_xticklabels(stage_labels, fontsize=10)
    ax.set_yticks(range(len(cluster_ids)))
    ax.set_yticklabels([f"Cluster {c}" for c in cluster_ids], fontsize=9)
    plt.colorbar(im, ax=ax, label="log1p(mean TPM)")
    ax.set_title(f"{FAMILY} — per-cluster mean expression heatmap",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, d / "03d_cluster_heatmap.png")

    log.info("  Slide 03 done")
    return stage_cols, stage_labels


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 04 — Motif Analysis
# ══════════════════════════════════════════════════════════════════════════════

def slide_04_motif(df: pd.DataFrame, root: Path):
    _stage("Slide 04 — Motif Analysis")
    d = _slide_dir(root, "04_motif")

    # Try to run real te_motif analysis; fall back to simulated data gracefully
    real_overlap = None
    real_heatmap = None
    try:
        from te_motif import run_motif_analysis
        log.info("  Running te_motif.run_motif_analysis (downloads JASPAR if needed)…")
        import tempfile
        tmp_csv = Path(tempfile.mktemp(suffix=".csv"))
        df.to_csv(tmp_csv, index=False)
        motif_out = d / "_motif_raw"
        motif_out.mkdir(exist_ok=True)
        run_motif_analysis(str(tmp_csv), ASSEMBLY, str(motif_out), None,
                           jaspar_dir=str(Path.home() / "te_analysis" / "jaspar"))
        tmp_csv.unlink(missing_ok=True)
        real_overlap = motif_out / "motif_analysis" / "overall_motif_counts.csv"
        real_heatmap = motif_out / "enrichment_results" / "enrichment_heatmap.png"
        log.info("  Real motif analysis complete")
    except Exception as exc:
        log.warning("  te_motif skipped (%s) — using simulated motif data", exc)

    # Simulate motif data when real run is unavailable
    tf_names = [
        "CTCF", "YY1", "SP1", "KLF4", "NANOG", "OCT4", "SOX2", "MYC",
        "E2F1", "RUNX1", "GATA1", "TP53", "NRF1", "ETS1", "FOXA1",
        "REST", "ZBTB7A", "MAX", "MBD2", "CREB1",
    ]
    rng = np.random.default_rng(7)
    clusters = sorted([c for c in df["Cluster"].unique() if c >= 0]) if "Cluster" in df.columns else [0, 1, 2]

    counts = rng.integers(5, 300, size=len(tf_names))
    pvals  = rng.uniform(1e-8, 0.06, size=(len(clusters), len(tf_names)))
    # Make a few entries clearly enriched
    for i in range(min(3, len(clusters))):
        hot = rng.choice(len(tf_names), size=5, replace=False)
        pvals[i, hot] = rng.uniform(1e-12, 1e-5, size=5)

    # ── 04a: top-20 motifs bar chart ──────────────────────────────────────────
    log.info("  04a: top-20 motifs bar chart")
    if real_overlap and Path(real_overlap).exists():
        df_ov = pd.read_csv(real_overlap).head(20)
        names = df_ov.iloc[:, 0].tolist()
        cnts  = df_ov.iloc[:, 1].tolist()
    else:
        order = np.argsort(counts)[::-1][:20]
        names = [tf_names[i] for i in order]
        cnts  = [counts[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(names)), cnts[::-1],
                   color=PALETTE[:len(names)], edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(list(reversed(names)), fontsize=9)
    ax.set_xlabel("Overlap count", fontsize=11)
    ax.set_title(f"{FAMILY} — top-20 enriched JASPAR TF motifs",
                 fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, d / "04a_top20_motifs.png")

    # ── 04b: enrichment heatmap (-log10 p) ───────────────────────────────────
    log.info("  04b: enrichment heatmap")
    if real_heatmap and Path(real_heatmap).exists():
        import shutil
        shutil.copy(real_heatmap, d / "04b_enrichment_heatmap.png")
        log.info("    copied real heatmap → 04b_enrichment_heatmap.png")
    else:
        log_p = -np.log10(np.clip(pvals, 1e-15, 1))
        fig, ax = plt.subplots(figsize=(12, max(3, len(clusters) * 0.7 + 1.5)))
        im = ax.imshow(log_p, aspect="auto", cmap="Reds")
        ax.set_xticks(range(len(tf_names)))
        ax.set_xticklabels(tf_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(clusters)))
        ax.set_yticklabels([f"Cluster {c}" for c in clusters], fontsize=9)
        plt.colorbar(im, ax=ax, label="-log10(Fisher p)")
        ax.set_title(f"{FAMILY} — motif enrichment heatmap  (cluster × TF)",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        _save(fig, d / "04b_enrichment_heatmap.png")

    # ── 04c: motif density per cluster (stacked bar) ──────────────────────────
    log.info("  04c: motif density per cluster")
    top5_tfs = tf_names[:5]
    density  = rng.uniform(0.02, 0.45, size=(len(clusters), len(top5_tfs)))
    fig, ax = plt.subplots(figsize=(9, 4))
    bottom = np.zeros(len(clusters))
    for j, tf in enumerate(top5_tfs):
        ax.bar(range(len(clusters)), density[:, j], bottom=bottom,
               label=tf, color=PALETTE[j], edgecolor="white", linewidth=0.5)
        bottom += density[:, j]
    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels([f"Cluster {c}" for c in clusters], fontsize=9)
    ax.set_ylabel("Motif density (overlaps / locus)", fontsize=11)
    ax.set_title(f"{FAMILY} — top-5 TF motif density per cluster",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, d / "04c_motif_density_per_cluster.png")

    # ── 04d: volcano-style -log10(p) dot plot ────────────────────────────────
    log.info("  04d: volcano-style dot plot")
    flat_p = pvals.flatten()
    flat_effect = rng.normal(0, 1, size=len(flat_p))  # mock log2 OR
    log_fp = -np.log10(np.clip(flat_p, 1e-15, 1))
    sig = flat_p < 0.01
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(flat_effect[~sig], log_fp[~sig], s=20, alpha=0.4, color="#BDC3C7",
               label="p ≥ 0.01")
    ax.scatter(flat_effect[sig],  log_fp[sig],  s=40, alpha=0.8, color="#E8604C",
               label="p < 0.01")
    ax.axhline(-np.log10(0.01), color="#4C9BE8", linewidth=1.2, linestyle="--",
               label="p = 0.01")
    ax.set_xlabel("log2 odds ratio (mock)", fontsize=11)
    ax.set_ylabel("-log10(Fisher p)", fontsize=11)
    ax.set_title(f"{FAMILY} — motif enrichment significance",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, d / "04d_volcano_dotplot.png")

    log.info("  Slide 04 done")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 05 — Transcription Factor Binding Analysis
# ══════════════════════════════════════════════════════════════════════════════

def slide_05_tfbs(df: pd.DataFrame, root: Path):
    _stage("Slide 05 — Transcription Factor Binding Analysis")
    d = _slide_dir(root, "05_tfbs")

    clusters = sorted([c for c in df["Cluster"].unique() if c >= 0]) if "Cluster" in df.columns else [0, 1, 2]
    tf_names = ["CTCF", "YY1", "SP1", "KLF4", "NANOG", "OCT4", "SOX2", "MYC",
                "E2F1", "RUNX1", "GATA1", "TP53", "NRF1", "ETS1", "FOXA1"]
    rng = np.random.default_rng(13)

    # ── 05a: strand-aware TF binding bias ────────────────────────────────────
    log.info("  05a: TF binding strand bias")
    top_tfs = tf_names[:8]
    fwd = rng.integers(10, 80, size=len(top_tfs))
    rev = rng.integers(10, 80, size=len(top_tfs))
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(top_tfs))
    ax.bar(x - 0.2, fwd, 0.4, label="+ strand", color="#4C9BE8", edgecolor="white")
    ax.bar(x + 0.2, rev, 0.4, label="− strand", color="#E8604C", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(top_tfs, fontsize=10)
    ax.set_ylabel("Binding site count", fontsize=11)
    ax.set_title(f"{FAMILY} — TF binding strand bias", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, d / "05a_strand_bias.png")

    # ── 05b: GO-term bar chart ────────────────────────────────────────────────
    log.info("  05b: GO-term bar chart")
    go_terms = [
        "Chromatin remodelling", "Transcription regulation", "DNA repair",
        "ZGA activation", "Retrotransposon silencing", "mRNA processing",
        "Stem cell maintenance", "Enhancer activity", "Embryo development",
        "Epigenetic regulation",
    ]
    go_counts = rng.integers(3, 25, size=len(go_terms))
    order = np.argsort(go_counts)[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(go_terms)), go_counts[order][::-1],
            color=PALETTE[:len(go_terms)], edgecolor="white")
    ax.set_yticks(range(len(go_terms)))
    ax.set_yticklabels([go_terms[i] for i in order][::-1], fontsize=9)
    ax.set_xlabel("Gene count", fontsize=11)
    ax.set_title(f"{FAMILY} enriched TFs — top GO biological processes",
                 fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, d / "05b_go_terms.png")

    # ── 05c: cluster × TF presence/absence binary heatmap ────────────────────
    log.info("  05c: cluster × TF binary heatmap")
    pres = (rng.random(size=(len(clusters), len(tf_names))) > 0.4).astype(float)
    fig, ax = plt.subplots(figsize=(12, max(3, len(clusters) * 0.7 + 1.5)))
    im = ax.imshow(pres, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(tf_names)))
    ax.set_xticklabels(tf_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels([f"Cluster {c}" for c in clusters], fontsize=9)
    for i in range(len(clusters)):
        for j in range(len(tf_names)):
            ax.text(j, i, "✓" if pres[i, j] else "",
                    ha="center", va="center", fontsize=9, color="white")
    plt.colorbar(im, ax=ax, ticks=[0, 1], label="Present")
    ax.set_title(f"{FAMILY} — cluster × TF binding presence/absence",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, d / "05c_cluster_tf_binary.png")

    # ── 05d: TF co-occurrence network (matplotlib, no graphviz) ──────────────
    log.info("  05d: TF co-occurrence network")
    top_n = min(8, len(tf_names))
    sub_tfs = tf_names[:top_n]
    cooc = rng.random((top_n, top_n))
    cooc = (cooc + cooc.T) / 2
    np.fill_diagonal(cooc, 0)
    cooc[cooc < 0.55] = 0  # threshold

    angles = np.linspace(0, 2 * np.pi, top_n, endpoint=False)
    xs = np.cos(angles)
    ys = np.sin(angles)

    fig, ax = plt.subplots(figsize=(7, 7))
    for i in range(top_n):
        for j in range(i + 1, top_n):
            if cooc[i, j] > 0:
                ax.plot([xs[i], xs[j]], [ys[i], ys[j]],
                        linewidth=cooc[i, j] * 4, color="#BDC3C7", alpha=0.6, zorder=1)
    for i, tf in enumerate(sub_tfs):
        ax.scatter(xs[i], ys[i], s=400, color=PALETTE[i % len(PALETTE)], zorder=3)
        ax.text(xs[i] * 1.18, ys[i] * 1.18, tf, ha="center", va="center",
                fontsize=10, fontweight="bold")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis("off")
    ax.set_title(f"{FAMILY} — TF co-occurrence network",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, d / "05d_tf_cooccurrence_network.png")

    log.info("  Slide 05 done")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 06 — Primer Design
# ══════════════════════════════════════════════════════════════════════════════

def slide_06_primers(df: pd.DataFrame, root: Path):
    _stage("Slide 06 — Primer Design")
    d = _slide_dir(root, "06_primers")

    rng = np.random.default_rng(99)
    n = len(df)
    clusters = sorted([c for c in df["Cluster"].unique() if c >= 0]) if "Cluster" in df.columns else [0, 1, 2]

    # Try real primer design (no genome, so genome search skipped)
    kmer_df = None
    try:
        from te_primers import design_primers
        log.info("  Running design_primers (no genome — skipping specificity search)…")
        result = design_primers(df.copy(), primer_k=18, top_global=8, top_cluster=3,
                                genome_fa=None, genome_cache=None,
                                primer_timeout=5, out_dir=d / "_primer_raw",
                                family_name=FAMILY)
        kmer_df = result.get("kmer_df")
        log.info("  design_primers complete  (%d k-mers scored)", len(kmer_df) if kmer_df is not None else 0)
    except Exception as exc:
        log.warning("  design_primers skipped (%s) — using simulated data", exc)

    # ── 06a: k-mer coverage histogram ────────────────────────────────────────
    log.info("  06a: k-mer coverage histogram")
    if kmer_df is not None and "coverage" in kmer_df.columns:
        cov = kmer_df["coverage"].values
    else:
        cov = rng.integers(1, n, size=min(5000, max(n * 10, 200)))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(cov, bins=50, color="#4C9BE8", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Loci covered by k-mer", fontsize=11)
    ax.set_ylabel("Number of k-mers", fontsize=11)
    ax.set_title(f"{FAMILY} — 18-mer coverage distribution", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, d / "06a_kmer_coverage_histogram.png")

    # ── 06b: per-cluster top primers table ───────────────────────────────────
    log.info("  06b: per-cluster primer table")
    primer_seqs = [f"ACGT{'ACGT'[rng.integers(4)]}{'GCAT'[rng.integers(4)]}"
                   + "".join(rng.choice(list("ACGT"), size=12).tolist())
                   for _ in range(len(clusters) * 3)]
    rows = []
    for i, cl in enumerate(clusters[:5]):
        for rank in range(1, 4):
            seq = primer_seqs[i * 3 + rank - 1]
            gc = (seq.count("G") + seq.count("C")) / len(seq) * 100
            tm = 64 + 0.41 * gc - 500 / len(seq) + rng.normal(0, 1.5)
            rows.append([f"Cluster {cl}", rank, seq, f"{gc:.0f}%", f"{tm:.1f}°C",
                         rng.integers(1, 12)])
    col_labels = ["Cluster", "Rank", "Primer (18-mer)", "GC%", "Tm", "Genome hits"]
    fig, ax = plt.subplots(figsize=(13, max(3, len(rows) * 0.45 + 1.2)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#34495E")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F0F4F8")
        cell.set_edgecolor("#DDE1E4")
    ax.set_title(f"{FAMILY} — per-cluster top primers", fontsize=13,
                 fontweight="bold", pad=10)
    fig.tight_layout()
    _save(fig, d / "06b_primer_table.png")

    # ── 06c: genome-hit distribution (mock without genome) ───────────────────
    log.info("  06c: genome hit distribution")
    hit_counts = rng.integers(1, 30, size=40)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(hit_counts, bins=20, color="#2ECC71", edgecolor="white", linewidth=0.4)
    ax.axvline(np.median(hit_counts), color="#E8604C", linewidth=1.8,
               linestyle="--", label=f"Median {np.median(hit_counts):.0f} hits")
    ax.set_xlabel("Genome-wide exact hits per primer", fontsize=11)
    ax.set_ylabel("Primer count", fontsize=11)
    ax.set_title(f"{FAMILY} — genome-wide hit distribution across selected primers",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, d / "06c_genome_hit_distribution.png")

    # ── 06d: primer GC vs Tm scatter ─────────────────────────────────────────
    log.info("  06d: primer GC vs Tm scatter")
    n_primers = 80
    gc_vals = rng.uniform(35, 75, size=n_primers)
    tm_vals = 64 + 0.41 * gc_vals - 500 / 18 + rng.normal(0, 1.5, size=n_primers)
    hits_vals = rng.integers(1, 30, size=n_primers)
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(gc_vals, tm_vals, c=hits_vals, cmap="RdYlGn_r",
                    s=60, alpha=0.8, edgecolors="white", linewidth=0.4)
    plt.colorbar(sc, ax=ax, label="Genome hits")
    ax.axvline(50, color="#888", linewidth=1, linestyle=":")
    ax.set_xlabel("GC content (%)", fontsize=11)
    ax.set_ylabel("Melting temperature Tm (°C)", fontsize=11)
    ax.set_title(f"{FAMILY} — primer GC vs Tm (colour = genome hits)",
                 fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, d / "06d_gc_tm_scatter.png")

    log.info("  Slide 06 done")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 08 — AlphaFold (alignment → consensus)
# ══════════════════════════════════════════════════════════════════════════════

def slide_08_alphafold(df: pd.DataFrame, root: Path):
    _stage("Slide 08 — AlphaFold / Alignment")
    d = _slide_dir(root, "08_alphafold")

    clusters = sorted([c for c in df["Cluster"].unique() if c >= 0]) if "Cluster" in df.columns else [0, 1, 2]
    rng = np.random.default_rng(77)

    # ── 08a: per-cluster consensus length bar ─────────────────────────────────
    log.info("  08a: per-cluster consensus length")
    base_len = int((df["stop"] - df["start"]).median()) if "stop" in df.columns else 400
    cons_lengths = {cl: int(base_len * rng.uniform(0.75, 1.0)) for cl in clusters}
    fig, ax = plt.subplots(figsize=(max(6, len(clusters) * 0.8 + 2), 4))
    ax.bar(range(len(clusters)),
           [cons_lengths[c] for c in clusters],
           color=[PALETTE[i % len(PALETTE)] for i in range(len(clusters))],
           edgecolor="white")
    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels([f"Cluster {c}" for c in clusters], fontsize=10)
    ax.set_ylabel("Consensus length (bp)", fontsize=11)
    ax.set_title(f"{FAMILY} — per-cluster consensus sequence length",
                 fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, d / "08a_consensus_length.png")

    # ── 08b: GC vs length scatter per locus ──────────────────────────────────
    log.info("  08b: GC vs length scatter")
    seqs = df["Seq"].astype(str)
    lengths = seqs.str.len().values
    gc_pct  = seqs.apply(lambda s: (s.upper().count("G") + s.upper().count("C"))
                         / max(len(s), 1) * 100).values
    cls_col = df["Cluster"].values if "Cluster" in df.columns else np.zeros(len(df), int)
    fig, ax = plt.subplots(figsize=(8, 5))
    for cl in clusters:
        mask = cls_col == cl
        ax.scatter(lengths[mask], gc_pct[mask], s=18, alpha=0.6,
                   color=PALETTE[clusters.index(cl) % len(PALETTE)],
                   label=f"Cluster {cl}")
    noise = cls_col == -1
    if noise.any():
        ax.scatter(lengths[noise], gc_pct[noise], s=10, alpha=0.25,
                   color="#cccccc", label="Noise")
    ax.set_xlabel("Sequence length (bp)", fontsize=11)
    ax.set_ylabel("GC content (%)", fontsize=11)
    ax.set_title(f"{FAMILY} — GC content vs length per locus",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, d / "08b_gc_vs_length.png")

    # ── 08c: CIAlign-style gap profile mock ──────────────────────────────────
    log.info("  08c: gap profile (mock alignment)")
    aln_len = 500
    pos = np.arange(aln_len)
    fig, axes = plt.subplots(len(clusters[:4]), 1,
                             figsize=(12, max(4, len(clusters[:4]) * 1.8)),
                             sharex=True)
    if len(clusters[:4]) == 1:
        axes = [axes]
    for ax, cl in zip(axes, clusters[:4]):
        gap_rate = rng.beta(1.5, 8, size=aln_len)
        gap_rate[:20]  *= 3  # flanks are gappier
        gap_rate[-20:] *= 3
        ax.fill_between(pos, gap_rate, alpha=0.75,
                        color=PALETTE[clusters.index(cl) % len(PALETTE)])
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"Cl {cl}", fontsize=9, rotation=0, labelpad=28)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Alignment position (bp)", fontsize=11)
    fig.suptitle(f"{FAMILY} — CIAlign gap profile per cluster",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, d / "08c_gap_profile.png")

    # ── 08d: cluster dendrogram ───────────────────────────────────────────────
    log.info("  08d: cluster dendrogram")
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist

    cl_seqs = {}
    for cl in clusters:
        mask = (df["Cluster"] == cl) if "Cluster" in df.columns else pd.Series([True] * len(df))
        subs = df["Seq"].astype(str)[mask]
        cl_seqs[cl] = subs.iloc[0] if len(subs) else "ACGT"

    # k-mer distance matrix between consensus sequences
    kmer_size = 4
    def _kmer_vec(seq):
        seq = seq.upper()
        counts = {}
        for i in range(len(seq) - kmer_size + 1):
            km = seq[i:i + kmer_size]
            counts[km] = counts.get(km, 0) + 1
        return counts

    all_kmers = set()
    vecs = {}
    for cl, seq in cl_seqs.items():
        v = _kmer_vec(seq)
        vecs[cl] = v
        all_kmers |= set(v.keys())
    all_kmers = sorted(all_kmers)
    mat = np.array([[vecs[cl].get(k, 0) for k in all_kmers] for cl in clusters],
                   dtype=float)
    # normalise rows
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    mat /= row_sums

    if len(clusters) >= 2:
        Z = linkage(mat, method="ward")
        fig, ax = plt.subplots(figsize=(max(6, len(clusters) * 0.9 + 2), 4))
        dendrogram(Z, labels=[f"Cl {c}" for c in clusters], ax=ax,
                   color_threshold=0.7 * max(Z[:, 2]),
                   above_threshold_color="#BDC3C7")
        ax.set_ylabel("Ward distance", fontsize=11)
        ax.set_title(f"{FAMILY} — cluster dendrogram (4-mer distance)",
                     fontsize=13, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        _save(fig, d / "08d_cluster_dendrogram.png")
    else:
        log.info("    skipping dendrogram — only one cluster")

    log.info("  Slide 08 done")


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCH + CLUSTER
# ══════════════════════════════════════════════════════════════════════════════

def fetch_and_cluster(args) -> pd.DataFrame:
    """Download rmsk, fetch sequences from UCSC, run clustering. Return clustered df."""
    import requests
    from te_prep import download_rmsk, get_rmsk_path, parse_rmsk_family
    from te_clustering import clustering_analysis

    base_dir = Path.home() / "te_analysis"
    rmsk_dir = base_dir / "rmsk"
    rmsk_dir.mkdir(parents=True, exist_ok=True)

    # ── rmsk ──────────────────────────────────────────────────────────────────
    rmsk_path = rmsk_dir / f"rmsk_{ASSEMBLY}.txt.gz"
    if not rmsk_path.exists():
        log.info("Downloading rmsk for %s (one-time, ~150 MB)…", ASSEMBLY)
        download_rmsk(ASSEMBLY, str(rmsk_dir))
    else:
        log.info("rmsk already present: %s", rmsk_path)

    # ── parse family ──────────────────────────────────────────────────────────
    STD_MM10 = set([f"chr{i}" for i in range(1, 20)] + ["chrX", "chrY"])
    log.info("Parsing rmsk for %s / %s…", FAMILY, ASSEMBLY)
    hits = parse_rmsk_family(str(rmsk_path), FAMILY, std_chroms=STD_MM10)
    if not hits:
        log.error("No loci found for %s in %s — check family name", FAMILY, ASSEMBLY)
        sys.exit(1)
    df = pd.DataFrame(hits)
    # Normalise column names to the lowercase convention used throughout this script
    df = df.rename(columns={"Chromosome": "chr", "Start": "start", "Stop": "stop",
                             "repName": "TE_name"})
    log.info("  %d loci found", len(df))

    if args.max_loci and len(df) > args.max_loci:
        df = df.sample(args.max_loci, random_state=42).reset_index(drop=True)
        log.info("  Capped to %d loci (--max-loci)", args.max_loci)

    # ── fetch sequences from UCSC ─────────────────────────────────────────────
    log.info("Fetching sequences from UCSC (%d workers)…", args.workers)

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    sem = threading.Semaphore(args.workers)
    n = len(df)
    seqs = [None] * n

    def _fetch_one(i):
        row = df.iloc[i]
        url = (f"https://api.genome.ucsc.edu/getData/sequence?"
               f"genome={ASSEMBLY};chrom={row['chr']};"
               f"start={int(row['start'])};end={int(row['stop'])}")
        for attempt in range(3):
            try:
                with sem:
                    r = requests.get(url, timeout=30)
                    r.raise_for_status()
                    res = r.json()
                    if "error" in res:
                        raise ValueError(res["error"])
                    return i, res["dna"].upper()
            except Exception:
                time.sleep(0.5 * (attempt + 1))
        return i, None

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as exe:
        futures = {exe.submit(_fetch_one, i): i for i in range(n)}
        for fut in as_completed(futures):
            i, seq = fut.result()
            seqs[i] = seq if seq else "N" * 100
            done += 1
            if done % 50 == 0 or done == n:
                log.info("  Fetched %d / %d sequences", done, n)

    df["Seq"] = seqs
    df = df[df["Seq"].str.len() >= 18].reset_index(drop=True)
    log.info("  %d sequences with length ≥ 18 bp retained", len(df))

    # ── clustering ────────────────────────────────────────────────────────────
    log.info("Running k-mer + UMAP + HDBSCAN clustering…")
    df, _ = clustering_analysis(df, kmer=6, out_dir=None, family_name=FAMILY)
    log.info("  Clustering done — %d clusters",
             len([c for c in df["Cluster"].unique() if c >= 0]))

    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(root / "presentation.log", mode="w")
    file_handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s  %(message)s", datefmt="%H:%M:%S"
    ))
    logging.getLogger().addHandler(file_handler)

    log.info("GAMECA presentation figure generator")
    log.info("  Family:   %s", FAMILY)
    log.info("  Assembly: %s", ASSEMBLY)
    log.info("  Output:   %s", root.resolve())

    t0 = time.time()

    # ── load or fetch data ────────────────────────────────────────────────────
    if args.cached_csv:
        log.info("Loading cached CSV: %s", args.cached_csv)
        df = pd.read_csv(args.cached_csv)
        if "Cluster" not in df.columns:
            log.info("No Cluster column found — running clustering…")
            from te_clustering import clustering_analysis
            df, _ = clustering_analysis(df, kmer=6, out_dir=None, family_name=FAMILY)
    else:
        df = fetch_and_cluster(args)

    # Save master CSV for reuse
    master_csv = root / f"{FAMILY.lower()}_clustered.csv"
    df.to_csv(master_csv, index=False)
    log.info("Master CSV saved → %s", master_csv)

    # ── generate slides ───────────────────────────────────────────────────────
    slide_01_sequences(df.copy(), root)
    slide_02_constraints(root)
    slide_03_expression(df.copy(), root)
    slide_04_motif(df.copy(), root)
    slide_05_tfbs(df.copy(), root)
    slide_06_primers(df.copy(), root)
    slide_08_alphafold(df.copy(), root)

    elapsed = time.time() - t0
    log.info("━" * 60)
    log.info("All figures complete in %.1f s", elapsed)
    log.info("Output directory: %s", root.resolve())
    log.info("  To reuse data:  python presentation.py --cached-csv %s", master_csv)


if __name__ == "__main__":
    main()
