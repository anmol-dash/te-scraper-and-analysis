#!/usr/bin/env python3
"""make_report_figures.py — build EVERY figure used by gameca_report.tex.

Single entry point for the manuscript's figures. It has two phases:

  1. Schematics (always run, need no data): the pipeline flowchart, the Stage-11
     standout-analysis overview, the software-architecture diagram, and the
     Nextflow DSL2 DAG. Rendered directly here into --out-dir (report_figs/).

  2. Data figures (run with --data, given real inputs): the phylogenetic trees,
     Kimura/repeat-landscape plots, gRNA Pareto fronts, DE heatmaps, LTR/subfamily
     figures, per-stage benchmarks, and the cross-family LINE/SINE/LTR plots. These
     are NOT drawn here — they are produced by invoking the real analysis scripts
     (run_line_sine_ltr_analysis.py, which itself drives every Stage-11 module per
     family, and optionally run_stage11_all.py for a single worked example) on
     genuine cluster inputs, writing into the paths the report reads (reports8/…).

No figure is ever fabricated. Data figures require real loci/expression/genome
inputs; when an input or a real generator is missing, the figure is reported as
SKIPPED with the reason — never invented. A manifest at the end lists every report
figure and whether it was produced.

Usage:
    # schematics only (no cluster data needed):
    python make_report_figures.py

    # everything — schematics + all data figures (typical cluster run):
    python make_report_figures.py --data \
        --reports-dir reports8 --build mm10 --genome-fa ~/te_analysis/mm10.fa \
        --l1mdt-expr L1Md_T_ultracombo.csv --b1mus2-expr B1_Mus2_ultracombo.csv \
        --iapltr1-expr IAPLTR1_Mm_ultracombo.csv

    # plus a single-family worked example (Stage-11 on one loci CSV):
    python make_report_figures.py --data --family MT2_Mm --assembly mm10 \
        --input mt2_mm_ultracombo.csv
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Palette (shared with make_nextflow_dag_fig.py so the manuscript reads as one)
# --------------------------------------------------------------------------- #
INPUT = "#555555"
CORE  = "#2f6f8f"   # core query.py stages
STAND = "#3f8f5f"   # Stage-11 standout modules
GATH  = "#8f6f2f"   # gather / report
ACCENT = "#8f3f6f"  # execution environment
DELIV = "#b0413e"   # deliverables (primers + CRISPR reagents) — the payoff colour

# The single worked-example TE used as the prototype throughout the manuscript
# (AJM comment C33/C158: one named TE from database to reagents; the three-family
# LINE/SINE/LTR set becomes supplementary). IAPLTR1_Mm is the LTR family already
# carried in Table 3 / Figs 3-4, so its real data is reused rather than regenerated.
PROTOTYPE_FAMILY = "IAPLTR1_Mm"
PROTOTYPE_ASSEMBLY = "mm10"
PROTOTYPE_DIR = "reports8/stage11_iapltr1"


def apply_house_style():
    """One brand for every figure (AJM comment C144: consistent colour + font).

    Sets matplotlib rcParams shared by all schematics drawn here; the palette
    constants above remain the single source of brand colour, and README.md
    publishes the same hexes so the data-figure generators can match them.
    """
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "text.color": "#222222",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 150,
    })


def _round_box(ax, x, y, w, h, text, color, fs=10, sub=None, lw=1.4, alpha="22"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=lw, edgecolor=color, facecolor=color + alpha))
    if sub:
        ax.text(x + w / 2, y + h * 0.63, text, ha="center", va="center",
                fontsize=fs, weight="bold", color="black")
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center",
                fontsize=fs - 2.5, style="italic", color="#333333")
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, weight="bold", color="black")


def _arrow(ax, x1, y1, x2, y2, color="#444444", style="-|>", lw=1.6, ls="-"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
        linewidth=lw, color=color, linestyle=ls, shrinkA=2, shrinkB=2))


def _save(fig, out_dir, name):
    png = os.path.join(out_dir, name + ".png")
    pdf = os.path.join(out_dir, name + ".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", png, "and", pdf)


# --------------------------------------------------------------------------- #
# 1. Pipeline flowchart
# --------------------------------------------------------------------------- #
def make_pipeline_flowchart(out_dir):
    """End-to-end flowchart: input -> core stages -> Stage 11 -> report,
    wrapped by the Nextflow/Singularity execution environment."""
    W, H = 12.5, 7.6
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    # Execution-environment band behind everything.
    ax.add_patch(FancyBboxPatch(
        (0.15, 0.65), W - 0.30, H - 1.4,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.3, edgecolor=ACCENT, facecolor=ACCENT + "0d",
        linestyle=(0, (6, 4))))
    ax.text(W / 2, 0.95,
            "Execution environment  ·  Nextflow DSL2 orchestration  ·  "
            "Singularity container (gameca.sif)  ·  LSF / SLURM",
            ha="center", va="center", fontsize=9.5, weight="bold", color=ACCENT)

    # Core stages (query.py Stages 1-7): a vertical stack in the middle band.
    stages = [
        ("1  Prep",       "loci · sequences · Kimura div."),
        ("2  Clustering", "k-mers -> SVD -> UMAP -> HDBSCAN"),
        ("3  Alignment",  "MAFFT + CIAlign consensus"),
        ("4  Motif",      "JASPAR enrichment + Fisher"),
        ("5  GO",         "mygene.info gene context"),
        ("6  Expression", "per-cluster counts + Wilcoxon DE"),
        ("7  Primers",    "k-mer candidates + genome-wide spec."),
    ]
    bx, bw, bh = 3.55, 3.55, 0.60
    top = 5.55
    gap = 0.12
    ys = []
    for i, (title, sub) in enumerate(stages):
        y = top - i * (bh + gap)
        ys.append(y)
        _round_box(ax, bx, y, bw, bh, title, CORE, fs=9.5, sub=sub)
        if i > 0:
            _arrow(ax, bx + bw / 2, ys[i - 1], bx + bw / 2, y + bh,
                   color=CORE, lw=1.3)
    ax.text(bx + bw / 2, top + bh + 0.30, "Core pipeline  (query.py, Stages 1-7)",
            ha="center", va="center", fontsize=10.5, weight="bold", color=CORE)

    # Input (left, aligned with the middle of the stack).
    mid_y = (ys[0] + ys[-1] + bh) / 2
    _round_box(ax, 0.55, mid_y - 0.5, 2.35, 1.0, "Input", INPUT, fs=11,
               sub="family + assembly")
    _arrow(ax, 2.90, mid_y, bx, top + bh / 2, color=INPUT, lw=1.6)

    # Stage 11 block (expands into the standout-analyses figure).
    s11x, s11w = 7.75, 3.15
    s11y, s11h = 1.75, 3.4
    _round_box(ax, s11x, s11y, s11w, s11h, "Stage 11", STAND, fs=13,
               sub="16 standout\nanalysis modules")
    ax.text(s11x + s11w / 2, s11y - 0.28, "run in parallel  (see Fig. 2)",
            ha="center", va="center", fontsize=8.5, style="italic", color=STAND)
    # core pipeline feeds Stage 11.
    _arrow(ax, bx + bw, mid_y, s11x, s11y + s11h / 2, color=CORE, lw=1.8)

    # Report / output (top-right, above Stage 11).
    ry = top + bh - 0.72
    _round_box(ax, s11x, ry, s11w, 0.9, "Report + family folder", GATH, fs=10,
               sub="LaTeX PDF · figures · macros")
    _arrow(ax, s11x + s11w / 2, s11y + s11h, s11x + s11w / 2, ry,
           color=GATH, lw=1.7)

    legend = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=CORE + "55",
               markeredgecolor=CORE, markersize=11, label="core stage (query.py)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=STAND + "55",
               markeredgecolor=STAND, markersize=11, label="Stage-11 standout modules"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=GATH + "55",
               markeredgecolor=GATH, markersize=11, label="gather / report"),
    ]
    ax.legend(handles=legend, loc="lower center", ncol=3, fontsize=8.6,
              frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(pad=0.4)
    _save(fig, out_dir, "fig_pipeline_flowchart")


# --------------------------------------------------------------------------- #
# 2. Stage-11 standout-analysis overview
# --------------------------------------------------------------------------- #
def make_stage11_overview(out_dir):
    """Grid of all 16 Stage-11 modules grouped by biological theme."""
    # (theme, color, [(title, script), ...])
    groups = [
        ("Evolution & age", "#2f6f8f", [
            ("Subfamily phylogenetics", "run_phylo_analysis.py"),
            ("Repeat landscape / divergence", "run_divergence.py"),
            ("Automatic subfamily resolution", "run_subfamily.py"),
            ("Consensus distance", "plot_consensus_distance.py"),
        ]),
        ("Regulation", "#3f8f5f", [
            ("Motif gain / turnover", "run_motif_gain.py"),
            ("Antisense / bidirectional promoter", "run_antisense_promoter.py"),
            ("CTCF sites & TAD boundaries", "run_ctcf_tad.py"),
            ("Epigenetic / regulatory overlay", "run_epigenetic_overlay.py"),
        ]),
        ("Structure & engineering", "#8f6f2f", [
            ("LTR structural annotation", "run_ltr_struct.py"),
            ("Protein structure (ColabFold)", "run_fold_prediction.py"),
            ("Allele-aware gRNA off-target", "run_grna_offtarget.py"),
        ]),
        ("Comparative & mobilization", "#8f3f6f", [
            ("Orthologous-insertion calling", "run_ortholog_insertion.py"),
            ("Multi-assembly liftover", "run_multiassembly_liftover.py"),
            ("3' transduction lineages", "run_transduction.py"),
        ]),
        ("Expression & benchmark", "#6f4f8f", [
            ("Differential expression (clusters)", "te_expression.py"),
            ("Benchmark vs manual workflow", "run_benchmark.py"),
        ]),
    ]

    ncol = len(groups)
    W, H = 15.5, 7.2
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    ax.text(W / 2, H - 0.28,
            "Stage 11 — standout analysis modules  (each a standalone run_*.py, "
            "run in parallel as an independent Nextflow task)",
            ha="center", va="center", fontsize=11, weight="bold", color="#222222")

    col_w = W / ncol
    pad = 0.30
    card_w = col_w - 2 * pad
    card_h = 0.92
    card_gap = 0.24
    top = H - 1.35

    for c, (theme, color, mods) in enumerate(groups):
        cx = c * col_w + pad
        # column header
        ax.add_patch(FancyBboxPatch(
            (cx, top + 0.15), card_w, 0.5,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=0, facecolor=color + "33"))
        ax.text(cx + card_w / 2, top + 0.40, theme, ha="center", va="center",
                fontsize=9.8, weight="bold", color=color)
        for i, (title, script) in enumerate(mods):
            y = top - (i + 1) * (card_h + card_gap) + 0.15
            ax.add_patch(FancyBboxPatch(
                (cx, y), card_w, card_h,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                linewidth=1.3, edgecolor=color, facecolor=color + "14"))
            ax.text(cx + card_w / 2, y + card_h * 0.62, title,
                    ha="center", va="center", fontsize=8.6, weight="bold",
                    color="black", wrap=True)
            ax.text(cx + card_w / 2, y + card_h * 0.22, script,
                    ha="center", va="center", fontsize=7.2, style="italic",
                    color="#444444", family="monospace")

    ax.text(W / 2, 0.22,
            "All modules consume the clustered loci + consensus from the core "
            "pipeline and write their own figures and measured-value macros.",
            ha="center", va="center", fontsize=8.6, style="italic", color="#666666")

    fig.tight_layout(pad=0.4)
    _save(fig, out_dir, "fig_stage11_analyses")


# --------------------------------------------------------------------------- #
# 3. Software architecture diagram
# --------------------------------------------------------------------------- #
def make_architecture(out_dir):
    """Layered architecture: desktop/CLI front-end -> HPC client -> cluster."""
    W, H = 11.0, 6.0
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    # Client layer.
    _round_box(ax, 0.5, 4.55, 4.4, 1.1, "Client layer", INPUT, fs=11,
               sub="Tauri desktop app  ·  query.py CLI")
    # Orchestration.
    _round_box(ax, 0.5, 3.05, 4.4, 1.1, "Orchestration", CORE, fs=11,
               sub="hpc_client.py  (SSH / bsub, channel-per-command)")
    # Core engine.
    _round_box(ax, 0.5, 1.55, 4.4, 1.1, "Core engine", STAND, fs=11,
               sub="query.py Stages 1-7 + Stage 11 modules")
    # Shared resource.
    _round_box(ax, 0.5, 0.35, 4.4, 0.85, "Shared genome cache", GATH, fs=10,
               sub="GenomeCache (hg38 in RAM, pickled)")

    for y1, y2 in [(4.55, 4.15), (3.05, 2.65), (1.55, 1.20)]:
        _arrow(ax, 2.7, y1, 2.7, y2, color="#555555", lw=1.6, style="<|-|>")

    # HPC side.
    ax.add_patch(FancyBboxPatch(
        (5.7, 0.35, ), 4.8, 5.3,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.3, edgecolor=ACCENT, facecolor=ACCENT + "0d",
        linestyle=(0, (6, 4))))
    ax.text(8.1, 5.35, "LSF / SLURM HPC cluster", ha="center", va="center",
            fontsize=10.5, weight="bold", color=ACCENT)
    hpc = [
        ("Login node", "job submission · Nextflow"),
        ("Compute nodes", "singularity exec gameca.sif"),
        ("Genome / RMSK / JASPAR", "shared filesystem"),
        ("ColabFold node", "GPU folding (separate)"),
    ]
    for i, (t, s) in enumerate(hpc):
        y = 4.35 - i * 1.02
        _round_box(ax, 6.0, y, 4.2, 0.82, t, ACCENT, fs=10, sub=s)

    _arrow(ax, 4.9, 3.6, 6.0, 4.35 + 0.4, color="#555555", lw=1.8, style="-|>")
    ax.text(5.45, 4.35, "SSH", ha="center", va="bottom", fontsize=8.5,
            style="italic", color="#555555")

    fig.tight_layout(pad=0.4)
    _save(fig, out_dir, "fig_architecture")


# --------------------------------------------------------------------------- #
# 4. Nextflow DSL2 DAG (schematic; mirrors make_nextflow_dag_fig.py)
# --------------------------------------------------------------------------- #
def make_nextflow_dag(out_dir):
    W, H = 11.0, 5.4
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    _round_box(ax, 0.25, 2.35, 1.7, 0.9, "samplesheet", INPUT, fs=9.5,
               sub="family, assembly")
    _arrow(ax, 1.95, 2.8, 2.55, 2.8)
    _round_box(ax, 2.55, 2.25, 2.0, 1.1, "GAMECA_CORE", CORE, fs=10.5,
               sub="query.py  Stages 1-10")

    ax.text(5.55, 4.95, "STANDOUT subworkflow  —  scatter / gather",
            ha="center", va="center", fontsize=9.5, style="italic", color=STAND)
    ax.add_patch(FancyBboxPatch(
        (4.75, 0.55), 3.7, 4.15,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.1, edgecolor=STAND, facecolor="none", linestyle=(0, (5, 4))))

    task_y = [3.75, 3.05, 2.35, 1.65, 0.95]
    task_labels = ["run_phylo_analysis.py", "run_grna_offtarget.py",
                   "run_divergence.py", "run_ltr_struct.py", "...  (16 modules)"]
    for ty, tl in zip(task_y, task_labels):
        _round_box(ax, 5.05, ty, 3.05, 0.55, tl, STAND, fs=8.2)
        _arrow(ax, 4.55, 2.8, 5.05, ty + 0.27, color=STAND, lw=1.1)

    _round_box(ax, 8.75, 2.25, 1.95, 1.1, "gather", GATH, fs=10.5,
               sub="merge -> family folder")
    for ty in task_y:
        _arrow(ax, 8.10, ty + 0.27, 8.75, 2.8, color=GATH, lw=1.0)

    ax.annotate("", xy=(8.62, 0.30), xytext=(2.5, 0.30),
                arrowprops=dict(arrowstyle="-", color="#999999", lw=1.0))
    ax.text(5.6, 0.08, "GAMECA subworkflow  (include { GAMECA })",
            ha="center", va="center", fontsize=8.6, color="#666666")
    ax.text(W / 2, 5.28,
            "Nextflow DSL2  ·  profiles: lsf / slurm · singularity / docker · test",
            ha="center", va="center", fontsize=9, weight="bold", color="#222222")

    legend = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=CORE + "55",
               markeredgecolor=CORE, markersize=11, label="core engine (process_high)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=STAND + "55",
               markeredgecolor=STAND, markersize=11, label="parallel module (process_medium)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=GATH + "55",
               markeredgecolor=GATH, markersize=11, label="gather / collect (process_low)"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=7.6, frameon=False,
              bbox_to_anchor=(1.0, 0.16))

    fig.tight_layout(pad=0.4)
    _save(fig, out_dir, "fig_nextflow_dag")


# --------------------------------------------------------------------------- #
# 5. Prototype-TE journey: database numbers -> reagents  (AJM C158/C159/C33)
# --------------------------------------------------------------------------- #
def make_prototype_journey(out_dir):
    """The deliverable-forward storyline for the IAPLTR1_Mm prototype: start from
    RepeatMasker/Dfam locus numbers, resolve which insertions matter, and end on
    the reagents users actually want — family/locus primers and allele-aware
    CRISPRa/i guides. Schematic only (no measured values are drawn here); the
    real per-step panels are produced by the Stage-11 generators and listed in
    the manifest under reports8/stage11_iapltr1/."""
    W, H = 15.6, 4.6
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    ax.text(W / 2, H - 0.3,
            f"Prototype walk-through — {PROTOTYPE_FAMILY} (LTR, {PROTOTYPE_ASSEMBLY}):  "
            "from database annotation to ready-to-order reagents",
            ha="center", va="center", fontsize=12, weight="bold", color="#222222")

    # Five narrative steps, then a highlighted deliverables end-cap.
    steps = [
        ("RMSK / Dfam", "named family ->\nlocus coordinates", INPUT),
        ("Sub-group", "alignment-free\nk-mer clustering", CORE),
        ("Which matter", "expression filter:\nmeaningful vs inert", STAND),
        ("Characterise", "per-group consensus\n+ motif enrichment", CORE),
    ]
    bw, bh = 2.35, 1.5
    gap = 0.5
    x = 0.35
    y = 1.9
    centres = []
    for title, sub, col in steps:
        _round_box(ax, x, y, bw, bh, title, col, fs=11, sub=sub)
        centres.append((x, x + bw))
        x += bw + gap

    # Deliverables end-cap — emphasised (thicker border, filled, DELIV colour).
    dx = x
    dw = 3.1
    ax.add_patch(FancyBboxPatch(
        (dx, y - 0.18), dw, bh + 0.36,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=2.6, edgecolor=DELIV, facecolor=DELIV + "1f"))
    ax.text(dx + dw / 2, y + bh - 0.28, "DELIVERABLES",
            ha="center", va="center", fontsize=11.5, weight="bold", color=DELIV)
    ax.text(dx + dw / 2, y + bh * 0.42, "family / locus-specific primers",
            ha="center", va="center", fontsize=9.6, color="#222222")
    ax.text(dx + dw / 2, y + bh * 0.14, "allele-aware CRISPRa/i guides",
            ha="center", va="center", fontsize=9.6, weight="bold", color=DELIV)

    # Arrows between steps and into the deliverables.
    for (l, r), (nl, nr) in zip(centres, centres[1:] + [(dx, dx + dw)]):
        _arrow(ax, r, y + bh / 2, nl, y + bh / 2, color="#666666", lw=1.8)

    ax.text(W / 2, 0.9,
            "Steps 1-4 are the tested core workflow; the CRISPR/primer reagents are "
            "the payoff that makes the family analysis actionable.",
            ha="center", va="center", fontsize=9, style="italic", color="#666666")
    ax.text(W / 2, 0.5,
            "Schematic — measured panels for each step are in reports8/stage11_iapltr1/ "
            "(see README.md).",
            ha="center", va="center", fontsize=8, style="italic", color="#999999")

    fig.tight_layout(pad=0.4)
    _save(fig, out_dir, "fig_prototype_journey")


# --------------------------------------------------------------------------- #
# 6. Why resolve copies: meaningful vs inert loci  (AJM C149/C107)
# --------------------------------------------------------------------------- #
def make_locus_filtering(out_dir):
    """Conceptual contrast (AJM C149): lumping every copy of a family averages
    real signal across meaningful, low-function and inert loci; GAMECA instead
    resolves copies by cluster + expression and distinguishes the expressed
    consensus from the common consensus (C107). Diagram only — no numbers."""
    W, H = 12.5, 5.2
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    ax.text(W / 2, H - 0.3,
            "Resolving copies, not lumping them  —  a core conceptual choice",
            ha="center", va="center", fontsize=12, weight="bold", color="#222222")

    # Left panel: naive lumping.
    ax.add_patch(FancyBboxPatch(
        (0.3, 0.5), 5.4, H - 1.4, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.3, edgecolor="#999999", facecolor="#9999990d"))
    ax.text(3.0, H - 1.15, "Lump the whole family",
            ha="center", va="center", fontsize=10.5, weight="bold", color="#777777")
    # a bag of mixed loci — mostly inert grey, a few meaningful/low-function
    ys = [3.3, 2.85, 2.4, 1.95]
    palette_cycle = [STAND, "#aaaaaa", GATH, "#aaaaaa", "#aaaaaa", STAND, "#aaaaaa",
                     "#aaaaaa", GATH, "#aaaaaa", "#aaaaaa", "#aaaaaa",
                     "#aaaaaa", STAND, "#aaaaaa", "#aaaaaa"]
    pc = iter(palette_cycle)
    for row in ys:
        for cx in [1.1, 2.1, 3.1, 4.1]:
            col = next(pc, "#aaaaaa")
            ax.add_patch(FancyBboxPatch(
                (cx, row), 0.7, 0.32, boxstyle="round,pad=0.01,rounding_size=0.05",
                linewidth=0, facecolor=col + "cc"))
    _arrow(ax, 3.0, 1.85, 3.0, 1.25, color="#777777", lw=1.6)
    ax.text(3.0, 0.95, "averaged signal:\nmeaningful copies diluted by inert ones",
            ha="center", va="center", fontsize=8.8, style="italic", color="#777777")

    # Right panel: GAMECA resolution.
    ax.add_patch(FancyBboxPatch(
        (6.8, 0.5), 5.4, H - 1.4, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4, edgecolor=STAND, facecolor=STAND + "0f"))
    ax.text(9.5, H - 1.15, "GAMECA: cluster + expression resolve copies",
            ha="center", va="center", fontsize=10.5, weight="bold", color=STAND)
    lanes = [
        ("meaningful (expressed)", STAND),
        ("low-function", GATH),
        ("inert / relic", "#aaaaaa"),
    ]
    for i, (lbl, col) in enumerate(lanes):
        ly = 3.15 - i * 0.72
        ax.add_patch(FancyBboxPatch(
            (7.1, ly), 4.6, 0.5, boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.3, edgecolor=col, facecolor=col + "22"))
        ax.text(9.4, ly + 0.25, lbl, ha="center", va="center",
                fontsize=9, weight="bold", color="#333333")
    ax.text(9.5, 0.95,
            "expressed consensus  ≠  common consensus\n"
            "downstream analysis targets the copies that matter",
            ha="center", va="center", fontsize=8.8, style="italic", color=STAND)

    fig.tight_layout(pad=0.4)
    _save(fig, out_dir, "fig_locus_filtering")


# --------------------------------------------------------------------------- #
# 7. CRISPR reagent design deliverable  (AJM C131/C157/C159)
# --------------------------------------------------------------------------- #
def make_crispr_deliverable(out_dir):
    """Concept diagram for the allele-aware CRISPRa/i guide-design module — the
    deliverable AJM wants surfaced prominently. Repetitive targets make guide
    design hard: a guide hits many copies, so GAMECA scores candidates against
    the full copy landscape with seed-anchored, mismatch-tolerant matching and
    reports the coverage-vs-off-target trade-off. Drawn as a labelled concept
    diagram, NOT a data scatter — the measured Pareto front is
    fig_grna_offtarget_pareto from run_grna_offtarget.py."""
    W, H = 12.5, 5.4
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    ax.text(W / 2, H - 0.3,
            "Allele-aware CRISPRa/i guide design for a repetitive family",
            ha="center", va="center", fontsize=12, weight="bold", color=DELIV)

    # Left: the design logic as a small flow.
    _round_box(ax, 0.35, 3.6, 3.4, 0.95, "Candidate guides", CORE, fs=10,
               sub="PAM-aware, from consensus\n& per-copy sequence")
    _round_box(ax, 0.35, 2.15, 3.4, 0.95, "Score vs copy landscape", STAND, fs=10,
               sub="seed-anchored,\nmismatch-tolerant match")
    _round_box(ax, 0.35, 0.7, 3.4, 0.95, "Coverage vs off-target", DELIV, fs=10,
               sub="on-family hits vs\noff-family hits per guide")
    _arrow(ax, 2.05, 3.6, 2.05, 3.1, color="#666666", lw=1.6)
    _arrow(ax, 2.05, 2.15, 2.05, 1.65, color="#666666", lw=1.6)

    # Right: a clearly-labelled *schematic* trade-off axis (illustrative curve).
    ox, oy = 5.0, 1.0            # axis origin
    aw, ah = 6.6, 3.4           # axis extent
    ax.add_patch(FancyArrowPatch((ox, oy), (ox, oy + ah), arrowstyle="-|>",
                                 mutation_scale=14, color="#444444", lw=1.6))
    ax.add_patch(FancyArrowPatch((ox, oy), (ox + aw, oy), arrowstyle="-|>",
                                 mutation_scale=14, color="#444444", lw=1.6))
    ax.text(ox - 0.15, oy + ah / 2, "on-family coverage", rotation=90,
            ha="center", va="center", fontsize=9.5, color="#333333")
    ax.text(ox + aw / 2, oy - 0.35, "off-target hits", ha="center", va="center",
            fontsize=9.5, color="#333333")
    # illustrative concave frontier (schematic — not measured points)
    import numpy as np
    t = np.linspace(0.08, 0.95, 40)
    fx = ox + 0.4 + t * (aw - 0.9)
    fy = oy + 0.4 + (ah - 0.9) * (1 - np.exp(-3.2 * t))
    ax.plot(fx, fy, color=DELIV, lw=2.2, ls=(0, (5, 3)))
    ax.scatter(fx[6::9], fy[6::9], s=42, color=DELIV, zorder=5,
               edgecolor="white", linewidth=0.8)
    ax.annotate("non-dominated guides\n(favourable trade-off)",
                xy=(fx[6], fy[6]), xytext=(fx[6] + 1.6, fy[6] - 0.9),
                fontsize=8.6, color=DELIV,
                arrowprops=dict(arrowstyle="->", color=DELIV, lw=1.2))
    ax.text(ox + aw / 2, oy + ah + 0.15,
            "Coverage / off-target Pareto frontier  (schematic; measured version "
            "= fig_grna_offtarget_pareto)",
            ha="center", va="center", fontsize=8.4, style="italic", color="#888888")

    fig.tight_layout(pad=0.4)
    _save(fig, out_dir, "fig_crispr_deliverable")


# --------------------------------------------------------------------------- #
# Data-figure orchestration (invokes the real analysis scripts; no fabrication)
# --------------------------------------------------------------------------- #

# Every figure the report references, with how it is produced. `source` is either
# "schematic" (drawn here) or the script that emits it. `path` is relative to the
# report .tex directory and matches the \figorbox argument in gameca_report.tex.
REPORT_FIGURES = [
    # schematics (this script) — written to --out-dir
    ("report_figs/fig_pipeline_flowchart", "schematic"),
    ("report_figs/fig_stage11_analyses",   "schematic"),
    ("report_figs/fig_architecture",       "schematic"),
    ("report_figs/fig_nextflow_dag",       "schematic"),
    # deliverable-forward schematics added for the AJM revision
    ("report_figs/fig_prototype_journey",  "schematic"),
    ("report_figs/fig_locus_filtering",    "schematic"),
    ("report_figs/fig_crispr_deliverable", "schematic"),
    # ------------------------------------------------------------------ #
    # IAPLTR1_Mm PROTOTYPE — the real collaborator-style procedural outputs
    # AJM comment C33 asks for: clusters, expression, consensus, alignment,
    # motif, primers, guide off-targets. Each maps to a genuine generator; see
    # README.md for the C33 coverage checklist. Nothing here is fabricated.
    # ------------------------------------------------------------------ #
    # -- clusters: see fig_ltr_results + fig_line_sine_ltr_clustering above --
    # -- expression (per-cluster DE + expression panels) --
    ("reports8/fig_de_iapltr1mm",                       "run_line_sine_ltr_analysis.py"),
    ("reports8/expression_plots/stage_profile",         "run_stage11_all.py (te_expression.py)"),
    ("reports8/expression_plots/chromosomal_heatmap",   "run_stage11_all.py (te_expression.py)"),
    ("reports8/expression_plots/primer_expression",     "run_stage11_all.py (te_expression.py)"),
    # -- new consensus sequences (per-cluster + global consensus distance) --
    ("reports8/stage11_iapltr1/fig_cluster_consensus_distance", "run_stage11_all.py (plot_consensus_distance.py)"),
    ("reports8/stage11_iapltr1/fig_global_consensus_distance",  "run_stage11_all.py (plot_consensus_distance.py)"),
    # -- alignment-derived phylogeny / molecular-clock age --
    ("reports8/stage11_iapltr1/fig_phylo_tree",          "run_stage11_all.py"),
    ("reports8/stage11_iapltr1/fig_phylo_divergence",    "run_stage11_all.py"),
    ("reports8/stage11_iapltr1/fig_phylo_master",        "run_stage11_all.py"),
    # -- motif analysis (evolutionary motif gain / turnover) --
    ("reports8/stage11_iapltr1/fig_motif_gains_bar",     "run_stage11_all.py"),
    ("reports8/stage11_iapltr1/fig_motif_gains_heatmap", "run_stage11_all.py"),
    # -- LTR structure (IAPLTR is an LTR/ERV) + subfamily consensus tree --
    ("reports8/stage11_iapltr1/fig_ltr_struct",          "run_stage11_all.py"),
    ("reports8/stage11_iapltr1/fig_subfamily_tree",      "run_stage11_all.py"),
    ("reports8/stage11_iapltr1/fig_subfamily_divergence","run_stage11_all.py"),
    # -- repeat landscape / divergence for the prototype --
    ("reports8/stage11_iapltr1/fig_repeat_landscape",    "run_stage11_all.py"),
    # -- regulation (antisense promoter, CTCF) --
    ("reports8/stage11_iapltr1/fig_antisense_motifs",    "run_stage11_all.py"),
    ("reports8/stage11_iapltr1/fig_antisense_expr",      "run_stage11_all.py"),
    ("reports8/stage11_iapltr1/fig_ctcf_overlap",        "run_stage11_all.py"),
    # -- CRISPR guide off-target: the headline deliverable (C131/C158) --
    ("reports8/stage11_iapltr1/fig_grna_offtarget_pareto","run_stage11_all.py"),
    ("reports8/stage11_iapltr1/fig_grna_offtarget_top",  "run_stage11_all.py"),
    # -- 3' transduction lineages --
    ("reports8/stage11_iapltr1/fig_transduction_groups", "run_stage11_all.py"),
    # cross-family LINE/SINE/LTR (run_line_sine_ltr_analysis.py -> --reports-dir)
    ("reports8/fig_line_sine_ltr_clustering", "run_line_sine_ltr_analysis.py"),
    ("reports8/fig_line_results",             "run_line_sine_ltr_analysis.py"),
    ("reports8/fig_sine_results",             "run_line_sine_ltr_analysis.py"),
    ("reports8/fig_ltr_results",              "run_line_sine_ltr_analysis.py"),
    ("reports8/fig_kimura_divergence",        "run_line_sine_ltr_analysis.py"),
    ("reports8/fig_de_l1mdt",                 "run_line_sine_ltr_analysis.py"),
    ("reports8/fig_de_b1mus2",                "run_line_sine_ltr_analysis.py"),
    # per-family Stage-11 (driven per family into stage11_<fam>/ subdirs)
    ("reports8/stage11_l1mdt/fig_phylo_tree",            "run_line_sine_ltr_analysis.py"),
    ("reports8/stage11_l1mdt/fig_phylo_divergence",      "run_line_sine_ltr_analysis.py"),
    ("reports8/stage11_l1mdt/fig_phylo_master",          "run_line_sine_ltr_analysis.py"),
    ("reports8/stage11_l1mdt/fig_grna_offtarget_pareto", "run_line_sine_ltr_analysis.py"),
    ("reports8/stage11_l1mdt/fig_transduction_groups",   "run_line_sine_ltr_analysis.py"),
    ("reports8/stage11_l1mdt/fig_antisense_motifs",      "run_line_sine_ltr_analysis.py"),
    ("reports8/stage11_l1mdt/fig_benchmark_steps",       "run_line_sine_ltr_analysis.py"),
    # single-family figures the report reads from the root (from a Stage-11 run)
    ("fig_ctcf_overlap",     "run_stage11_all.py"),
    ("fig_repeat_landscape", "run_stage11_all.py"),
    ("fig_ltr_struct",       "run_stage11_all.py"),
    ("fig_subfamily_tree",   "run_stage11_all.py"),
    ("fig_benchmark",        "run_stage11_all.py"),
    # no automated generator in the repo — manual/conceptual figures
    ("fig_mt2_results",       "manual"),
    ("fig_bench_genomecache", "manual"),
    ("fig_bench_cython",      "manual"),
]

# figures with source == "manual" that genuinely have no generator
NO_GENERATOR = {"fig_mt2_results", "fig_bench_genomecache", "fig_bench_cython"}


def _python():
    return shutil.which("python3") or shutil.which("python") or sys.executable


def run_cross_family(args):
    """Drive run_line_sine_ltr_analysis.py — the master data-figure generator.

    It clusters each LINE/SINE/LTR family and runs every Stage-11 module per
    family into <reports-dir>/stage11_<family>/, plus the cross-family plots.
    Passes inputs straight through; the script itself skips any figure whose
    data (expression, genome) is absent rather than inventing it.
    """
    script = HERE / "run_line_sine_ltr_analysis.py"
    if not script.exists():
        print(f"  SKIP cross-family: {script} not found")
        return False
    cmd = [_python(), str(script), "--reports-dir", args.reports_dir,
           "--build", args.build, "--source", args.source]
    for flag, val in [("--genome-fa", args.genome_fa),
                      ("--rmsk-dir", args.rmsk_dir),
                      ("--l1mdt-expr", args.l1mdt_expr),
                      ("--b1mus2-expr", args.b1mus2_expr),
                      ("--iapltr1-expr", args.iapltr1_expr)]:
        if val:
            cmd += [flag, val]
    if args.max_loci:
        cmd += ["--max-loci", str(args.max_loci)]
    print("  RUN:", " ".join(cmd))
    return subprocess.run(cmd).returncode == 0


def run_single_family(args):
    """Optional worked example: run all Stage-11 modules on one loci CSV via
    run_stage11_all.py. Figures land in --reports-dir (default the repo root so
    the report's root-relative figorbox references resolve)."""
    script = HERE / "run_stage11_all.py"
    if not script.exists():
        print(f"  SKIP single-family: {script} not found")
        return False
    if not args.input or not Path(args.input).exists():
        print(f"  SKIP single-family: --input not provided or missing "
              f"({args.input!r})")
        return False
    out = args.single_reports_dir or "."
    cmd = [_python(), str(script), "--input", args.input,
           "--family", args.family, "--assembly", args.assembly,
           "--reports-dir", out]
    if args.consensus_fasta:
        cmd += ["--consensus-fasta", args.consensus_fasta]
    print("  RUN:", " ".join(cmd))
    return subprocess.run(cmd).returncode == 0


def print_manifest():
    """Report the status of every figure the manuscript references."""
    print("\n" + "=" * 68)
    print("FIGURE MANIFEST  (relative to the report .tex directory)")
    print("=" * 68)
    made = missing = skipped = 0
    for rel, source in REPORT_FIGURES:
        base = os.path.basename(rel)
        if source == "manual" and base not in NO_GENERATOR:
            continue  # internal bookkeeping entry
        pdf = HERE / (rel + ".pdf")
        png = HERE / (rel + ".png")
        exists = pdf.exists() or png.exists()
        if base in NO_GENERATOR:
            status = "MANUAL   (no automated generator — provide the figure by hand)"
            skipped += 1
        elif exists:
            status = "MADE     [%s]" % source
            made += 1
        else:
            status = "MISSING  [%s — needs real inputs; not fabricated]" % source
            missing += 1
        print(f"  {rel:<48} {status}")
    print("-" * 68)
    print(f"  made={made}  missing={missing}  manual={skipped}")
    print("=" * 68)


def write_readme(out_dir):
    """Write out_dir/README.md documenting every manuscript figure, the shared
    brand, and the IAPLTR1_Mm prototype convention. Static text — regenerated on
    each run so it travels with the figures (AJM comments C33/C131/C144/C159)."""
    readme = f"""# GAMECA report figures

This folder holds the schematic figures for `gameca_report.tex` and is the
documented home of the figure pipeline. Every figure the manuscript references is
built by `make_report_figures.py` — schematics directly, data figures by invoking
the real analysis scripts. **No figure is ever fabricated:** a data figure that
lacks real inputs is reported `MISSING` in the manifest, never invented
(see the no-fabrication policy below).

## Prototype TE: {PROTOTYPE_FAMILY} ({PROTOTYPE_ASSEMBLY})

Per the review, the paper follows **one named TE as a prototype throughout**:
`{PROTOTYPE_FAMILY}`, an LTR/endogenous-retrovirus family. It is the LTR family
already carried in the cross-family results, so its real data is reused rather than
regenerated. The three-family LINE/SINE/LTR set (L1Md_T, B1_Mus2, {PROTOTYPE_FAMILY})
is intended as **supplementary** breadth; the main text should walk the reader
through the single prototype from database annotation to ordered reagents. The
prototype's measured Stage-11 panels are written to `{PROTOTYPE_DIR}/` by
`run_stage11_all.py`.

## Deliverables come first

The reagents are the payoff and should appear early and prominently:
**family/locus-specific primers** and **allele-aware CRISPRa/i guides**. The
CRISPR module is what moves the tool from "useful" to "necessary" for the field —
`fig_prototype_journey` and `fig_crispr_deliverable` exist to make that case.

## Brand / house style (keep every figure consistent)

All figures share one look. Match these when producing data figures too:

| Role | Hex |
|------|-----|
| Input / neutral | `{INPUT}` |
| Core stage (query.py) | `{CORE}` |
| Stage-11 standout module | `{STAND}` |
| Gather / report | `{GATH}` |
| Execution environment | `{ACCENT}` |
| **Deliverables (primers + CRISPR)** | `{DELIV}` |

Font: DejaVu Sans. Rounded boxes, thin arrows, PNG at 150 dpi + vector PDF.
`apply_house_style()` sets these as matplotlib rcParams for the schematics.

## Figures in this folder (schematics — always generated)

- **fig_pipeline_flowchart** — end-to-end flow: input to core Stages 1-7 to
  Stage 11 to report, inside the Nextflow/Singularity execution band.
  *Section: Design/architecture.*
- **fig_architecture** — three-layer software architecture (desktop/CLI to HPC
  client to cluster). *Section: Design and architecture (Figure 1).*
- **fig_stage11_analyses** — grid of the 16 Stage-11 copy-resolved modules by
  theme. *Section: Copy-resolved analysis modules.*
- **fig_nextflow_dag** — the Nextflow DSL2 scatter/gather DAG. *Section: HPC.*
- **fig_prototype_journey** — NEW. The {PROTOTYPE_FAMILY} storyline: RMSK/Dfam
  loci to clustering to expression filter (meaningful vs inert) to consensus+motif
  to the highlighted **primers + CRISPRa/i deliverables**. Schematic spine of the
  results narrative. *Should open the demonstration/results section.*
- **fig_locus_filtering** — NEW. Why GAMECA resolves copies instead of lumping
  them: lumping averages signal across meaningful, low-function and inert loci,
  whereas cluster+expression resolution separates them; also expressed-consensus
  vs common-consensus. Concept diagram (no numbers). *Frame this as a conceptual
  contribution.*
- **fig_crispr_deliverable** — NEW. Concept diagram of allele-aware CRISPRa/i
  guide design for repetitive targets (seed-anchored, mismatch-tolerant coverage
  vs off-target; Pareto trade-off). The measured version is
  `fig_grna_offtarget_pareto`. *Section: deliverables / CRISPR.*

## Prototype data panels (generated on the cluster into `{PROTOTYPE_DIR}/`)

Produced by `python make_report_figures.py --data --input <{PROTOTYPE_FAMILY} loci
csv> --iapltr1-expr <expr csv> ...`, which drives `run_stage11_all.py`. These are
the *real* per-family outputs (the kind sent to collaborators):

- **fig_phylo_tree / fig_phylo_divergence / fig_phylo_master** — subfamily
  phylogeny, per-copy molecular-clock age, and the combined age/intactness master
  panel.
- **fig_grna_offtarget_pareto / fig_grna_offtarget_top** — measured guide
  coverage-vs-off-target frontier and the top guides.
- **fig_transduction_groups** — 3' transduction lineage groups.
- **fig_antisense_motifs / fig_antisense_expr** — bidirectional-promoter core
  motifs and antisense expression.

## C33 coverage checklist — the collaborator-style outputs

AJM (comment C33) asked for the real procedural graphs sent to Katie/Claire/Diego —
clusters, expression, consensus, alignment, motif, primers, guide off-targets — for
the prototype. Every one is produced by a genuine generator and registered in the
manifest (`MADE` once real inputs run; `MISSING` otherwise — never fabricated):

| C33 item | Figure(s) | Generator |
|----------|-----------|-----------|
| Clusters (UMAP/HDBSCAN) | `reports8/fig_ltr_results`, `reports8/fig_line_sine_ltr_clustering` | `run_line_sine_ltr_analysis.py` |
| Expression (per-cluster DE) | `reports8/fig_de_iapltr1mm`, `reports8/expression_plots/stage_profile`, `.../chromosomal_heatmap` | cross-family + `te_expression.py` |
| New consensus sequences | `.../fig_cluster_consensus_distance`, `.../fig_global_consensus_distance`, `.../fig_phylo_tree` | `plot_consensus_distance.py`, `run_phylo_analysis.py` |
| Motif analysis (turnover) | `.../fig_motif_gains_bar`, `.../fig_motif_gains_heatmap` | `run_motif_gain.py` |
| Guide off-targets (CRISPR) | `.../fig_grna_offtarget_pareto`, `.../fig_grna_offtarget_top` | `run_grna_offtarget.py` |
| LTR structure (prototype is an LTR) | `.../fig_ltr_struct`, `.../fig_subfamily_tree` | `run_ltr_struct.py`, `run_subfamily.py` |
| Regulation (antisense, CTCF) | `.../fig_antisense_motifs`, `.../fig_antisense_expr`, `.../fig_ctcf_overlap` | `run_antisense_promoter.py`, `run_ctcf_tad.py` |
| 3' transduction | `.../fig_transduction_groups` | `run_transduction.py` |

Three C33 items are deliberately **not** matplotlib figures — flagged here so they are
not silently assumed:

- **Alignments** — the multiple alignment is cleaned/visualised by **CIAlign** (its own
  PNG in the alignment output dir); the alignment-derived *figure* in the report is the
  phylogeny/consensus panel (`fig_phylo_tree`).
- **JASPAR motif *enrichment*** (`overall_top_motifs.png`, `enrichment_heatmap.png`) is a
  **core `query.py` Stage-4** output, not produced by the two generators driven here; the
  registered motif figure is the evolutionary motif-**turnover** (`fig_motif_gains_*`).
  Include the enrichment panel from a core run or the collaborator report if the paper
  needs it.
- **Primers** are a CSV table (`06_primers/selected_primers_summary.csv`); the primer
  *figure* is `expression_plots/primer_expression` (primers × expression).

## Regenerate

```bash
# schematics + README only (no cluster data needed)
python make_report_figures.py

# everything, prototype-first (real inputs; nothing fabricated)
python make_report_figures.py --data \\
    --input {PROTOTYPE_FAMILY}_loci.csv --iapltr1-expr IAPLTR1_Mm_ultracombo.csv \\
    --genome-fa ~/te_analysis/mm10.fa --build {PROTOTYPE_ASSEMBLY}
```

The run prints a **manifest** marking each figure `MADE`, `MISSING` (needs real
inputs — not fabricated), or `MANUAL` (no automated generator).

## No-fabrication policy

Data figures require genuine loci/expression/genome inputs. When an input or a
real generator is missing, the figure is reported `MISSING` with the reason and
left unbuilt. Numbers, plots and measured panels are never invented.
"""
    path = os.path.join(out_dir, "README.md")
    with open(path, "w") as fh:
        fh.write(readme)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="report_figs",
                    help="Output directory for schematic figures (default: report_figs).")
    ap.add_argument("--data", action="store_true",
                    help="Also produce the data figures by running the real analysis "
                         "scripts (needs genuine cluster inputs).")
    # cross-family (run_line_sine_ltr_analysis.py) pass-through
    ap.add_argument("--reports-dir", default="reports8",
                    help="Where the data figures are written; must match the report's "
                         "graphicspath (default: reports8).")
    ap.add_argument("--build", default="mm10", help="Genome build (default: mm10).")
    ap.add_argument("--source", choices=["rmsk", "dfam"], default="rmsk")
    ap.add_argument("--genome-fa", default="", help="Path to genome FASTA (optional).")
    ap.add_argument("--rmsk-dir", default=os.path.expanduser("~/te_analysis/rmsk"))
    ap.add_argument("--l1mdt-expr", default="")
    ap.add_argument("--b1mus2-expr", default="")
    ap.add_argument("--iapltr1-expr", default="")
    ap.add_argument("--max-loci", type=int, default=None)
    # optional single-family worked example (run_stage11_all.py)
    ap.add_argument("--input", default="",
                    help="Loci CSV for a single-family Stage-11 worked example.")
    ap.add_argument("--family", default=PROTOTYPE_FAMILY,
                    help=f"Prototype/worked-example family (default: {PROTOTYPE_FAMILY}).")
    ap.add_argument("--assembly", default=PROTOTYPE_ASSEMBLY)
    ap.add_argument("--consensus-fasta", default="")
    ap.add_argument("--single-reports-dir", default=PROTOTYPE_DIR,
                    help=f"Output dir for the single-family prototype run "
                         f"(default: {PROTOTYPE_DIR}).")
    args = ap.parse_args()

    # Phase 1 — schematics (always).
    apply_house_style()
    os.makedirs(args.out_dir, exist_ok=True)
    print("Phase 1: schematic figures ->", os.path.abspath(args.out_dir))
    make_pipeline_flowchart(args.out_dir)
    make_stage11_overview(args.out_dir)
    make_architecture(args.out_dir)
    make_nextflow_dag(args.out_dir)
    make_prototype_journey(args.out_dir)
    make_locus_filtering(args.out_dir)
    make_crispr_deliverable(args.out_dir)
    write_readme(args.out_dir)

    # Phase 2 — data figures (opt-in; invokes the real generators).
    if args.data:
        print("\nPhase 2: data figures (real analysis scripts)")
        run_cross_family(args)
        if args.input:
            run_single_family(args)
    else:
        print("\nPhase 2 skipped (pass --data with real inputs to generate the "
              "data figures).")

    print_manifest()


if __name__ == "__main__":
    main()
