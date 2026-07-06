#!/usr/bin/env python3
"""Render fig_nextflow_dag.pdf — schematic of the GAMECA Nextflow DSL2 orchestration.

Static architectural diagram for the manuscript (gameca_report.tex, Section
"Workflow orchestration with Nextflow"). Regenerate with:  python make_nextflow_dag_fig.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

W, H = 11.0, 5.4
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

CORE  = "#2f6f8f"
STAND = "#3f8f5f"
GATH  = "#8f6f2f"
INPUT = "#555555"


def box(x, y, w, h, text, color, fs=10, sub=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.10",
                 linewidth=1.4, edgecolor=color, facecolor=color + "22"))
    if sub:
        ax.text(x + w / 2, y + h * 0.62, text, ha="center", va="center",
                fontsize=fs, weight="bold", color="black")
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                fontsize=fs - 2.5, style="italic", color="#333333")
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, weight="bold", color="black")


def arrow(x1, y1, x2, y2, color="#444444", style="-|>", lw=1.5, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=13, linewidth=lw, color=color, linestyle=ls,
                 shrinkA=1, shrinkB=1))


# Input / samplesheet
box(0.25, 2.35, 1.7, 0.9, "samplesheet", INPUT, fs=9.5,
    sub="family, assembly")
arrow(1.95, 2.8, 2.55, 2.8)

# GAMECA_CORE
box(2.55, 2.25, 2.0, 1.1, "GAMECA_CORE", CORE, fs=10.5,
    sub="query.py  Stages 1-10")

# scatter label
ax.text(5.55, 4.95, "STANDOUT subworkflow  —  scatter / gather",
        ha="center", va="center", fontsize=9.5, style="italic", color=STAND)
ax.add_patch(FancyBboxPatch((4.75, 0.55), 3.7, 4.15,
             boxstyle="round,pad=0.02,rounding_size=0.08",
             linewidth=1.1, edgecolor=STAND, facecolor="none", linestyle=(0, (5, 4))))

# scatter fan-out into standout tasks
task_y = [3.75, 3.05, 2.35, 1.65, 0.95]
task_labels = ["run_phylo_analysis.py", "run_grna_offtarget.py",
               "run_divergence.py", "run_ltr_struct.py", "…  (15 modules)"]
for ty, tl in zip(task_y, task_labels):
    box(5.05, ty, 3.05, 0.55, tl, STAND, fs=8.2)
    arrow(4.55, 2.8, 5.05, ty + 0.27, color=STAND, lw=1.1)

# gather into report/collect
box(8.75, 2.25, 1.95, 1.1, "gather", GATH, fs=10.5,
    sub="merge → family folder")
for ty in task_y:
    arrow(8.10, ty + 0.27, 8.75, 2.8, color=GATH, lw=1.0)

# GAMECA subworkflow bracket (CORE -> STANDOUT)
ax.annotate("", xy=(8.62, 0.30), xytext=(2.5, 0.30),
            arrowprops=dict(arrowstyle="-", color="#999999", lw=1.0))
ax.text(5.6, 0.08, "GAMECA subworkflow  (importable:  include { GAMECA })",
        ha="center", va="center", fontsize=8.6, color="#666666")

# profiles caption strip
ax.text(W / 2, 5.28, "Nextflow DSL2  ·  profiles: lsf / slurm · singularity / docker · test",
        ha="center", va="center", fontsize=9, weight="bold", color="#222222")

legend = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor=CORE + "55",
           markeredgecolor=CORE, markersize=11, label="core engine (process_high)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=STAND + "55",
           markeredgecolor=STAND, markersize=11, label="parallel module task (process_medium)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=GATH + "55",
           markeredgecolor=GATH, markersize=11, label="gather / collect (process_low)"),
]
ax.legend(handles=legend, loc="lower right", fontsize=7.6, frameon=False,
          bbox_to_anchor=(1.0, 0.16))

fig.tight_layout(pad=0.4)
fig.savefig("fig_nextflow_dag.pdf", bbox_inches="tight")
fig.savefig("fig_nextflow_dag.png", dpi=150, bbox_inches="tight")
print("wrote fig_nextflow_dag.pdf / .png")
