#!/usr/bin/env python3
"""make_report_figures.py — render every schematic figure used by gameca_report.tex.

This produces the *diagram* figures for the manuscript (pipeline flowchart, the
Stage-11 standout-analysis overview, the software architecture diagram, and the
Nextflow DSL2 DAG) and writes them all — .png and .pdf — into ./report_figs/.

These are structural diagrams of the pipeline itself, not plots of run data;
the data-driven figures (mt2_results, kimura_divergence, phylo trees, DE
heatmaps, gRNA Pareto, ...) are emitted by the individual run_*.py modules on
real cluster output and are deliberately NOT regenerated here.

Usage:
    python make_report_figures.py                 # -> ./report_figs/
    python make_report_figures.py --out-dir DIR
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------- #
# Palette (shared with make_nextflow_dag_fig.py so the manuscript reads as one)
# --------------------------------------------------------------------------- #
INPUT = "#555555"
CORE  = "#2f6f8f"   # core query.py stages
STAND = "#3f8f5f"   # Stage-11 standout modules
GATH  = "#8f6f2f"   # gather / report
ACCENT = "#8f3f6f"  # execution environment


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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="report_figs",
                    help="Output directory for figures (default: report_figs).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    make_pipeline_flowchart(args.out_dir)
    make_stage11_overview(args.out_dir)
    make_architecture(args.out_dir)
    make_nextflow_dag(args.out_dir)
    print("\nAll schematic figures written to", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
