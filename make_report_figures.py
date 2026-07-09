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
    ap.add_argument("--family", default="FAMILY")
    ap.add_argument("--assembly", default="mm10")
    ap.add_argument("--consensus-fasta", default="")
    ap.add_argument("--single-reports-dir", default="",
                    help="Output dir for the single-family run (default: repo root).")
    args = ap.parse_args()

    # Phase 1 — schematics (always).
    os.makedirs(args.out_dir, exist_ok=True)
    print("Phase 1: schematic figures ->", os.path.abspath(args.out_dir))
    make_pipeline_flowchart(args.out_dir)
    make_stage11_overview(args.out_dir)
    make_architecture(args.out_dir)
    make_nextflow_dag(args.out_dir)

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
