#!/usr/bin/env python3
"""
run_benchmark.py — GAMECA standout module: benchmarking vs manual workflow
─────────────────────────────────────────────────────────────────────────────
Records real GAMECA stage wall-clock timings from pipeline_diagnostic.log
and produces a comparison table against a documented manual workflow step
count.  Runtime figures are measured — never estimated or fabricated.

Manual workflow step counts represent the minimal POSIX/R commands required
for an equivalent analysis (download, filter, align, cluster, motif) as
documented in the literature and UCSC/Bioconductor vignettes; no runtime is
claimed for the manual path (it depends on hardware and experience).

Outputs (in <reports_dir>/):
  benchmark_table.csv          stage timings + manual step count comparison
  fig_benchmark.png            bar chart of GAMECA wall-clock time per stage
  benchmark_values.tex         LaTeX macros (total time, n_stages, n_manual_steps)

Usage (called automatically by query.py Stage 11):
    python run_benchmark.py --input clustered.csv --reports-dir ./reports \\
        --family HERVK9 --assembly hg38
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Manual workflow step count (documented, not fabricated runtime) ─────────
#
# Each row represents one shell/R command a human would run to achieve the
# same result as the corresponding GAMECA stage.  The step count is the
# minimum number of distinct commands; the description cites the tool.
#
MANUAL_STEPS = [
    ("Download RMSK",        1, "UCSC Table Browser → export → save file"),
    ("Filter by family",     1, "awk/grep on repName field"),
    ("Extract sequences",    2, "bedtools getfasta (requires indexed genome FASTA)"),
    ("Quality filter",       1, "custom Python/R script"),
    ("Multiple alignment",   1, "mafft --auto input.fa > aligned.fa"),
    ("Clustering",           3, "R: umap() + hdbscan() + manual parameter tuning"),
    ("Divergence",           2, "R/Python: pairwise2 + CpG correction script"),
    ("Motif analysis",       3, "fimo/HOMER + bedtools intersect + custom enrichment"),
    ("LTR annotation",       4, "LTRharvest setup + run + parse output + reformat"),
    ("Expression overlay",   2, "bedtools intersect + custom R join"),
    ("Report generation",    3, "Rmarkdown/LaTeX compilation"),
]

# Total GAMECA user-steps: one command (python query.py --local --family X --assembly Y)
GAMECA_USER_STEPS = 1


def _parse_stage_times(log_path):
    """Extract stage wall-clock times from pipeline_diagnostic.log.

    Returns: dict {stage_name: seconds} for all stages found.
    """
    times = {}
    if not log_path or not Path(log_path).exists():
        return times

    # Pattern: [TIMING] stage_name: 12.3s total
    pattern = re.compile(r"\[TIMING\]\s+(.+?):\s+([\d.]+)s\s+total")
    # Also match STAGE N: ... done in Xs format
    pattern2 = re.compile(r"STAGE.*?done in\s+([\d.]+)s", re.IGNORECASE)

    prev_t = 0.0
    with open(log_path) as fh:
        for line in fh:
            m = pattern.search(line)
            if m:
                name = m.group(1).strip()
                t    = float(m.group(2))
                # Convert cumulative to incremental if monotonically increasing
                delta = t - prev_t
                times[name] = delta if delta > 0 else t
                prev_t = t

    # Also try reading CHECKPOINT files
    log_dir = Path(log_path).parent
    for ckpt in sorted(log_dir.glob("CHECKPOINT_*.txt")):
        name = ckpt.stem.replace("CHECKPOINT_", "").replace("_", " ").title()
        if name not in times:
            times[name] = None  # present but timing not captured

    return times


def run_benchmark(input_csv, reports_dir, family_name, assembly="hg38"):
    """Record GAMECA stage timings and build comparison table.

    Args:
        input_csv:   path to the clustered CSV (used to locate log file)
        reports_dir: directory to write benchmark outputs
        family_name: family name string
        assembly:    genome assembly string
    """
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== BENCHMARKING ({family_name} / {assembly}) ===")

    # Locate pipeline_diagnostic.log
    _base = Path(input_csv).parent.parent
    log_path = _base / "pipeline_diagnostic.log"
    if not log_path.exists():
        log_path = _base.parent / "pipeline_diagnostic.log"

    stage_times = _parse_stage_times(log_path if log_path.exists() else None)

    if stage_times:
        print(f"  Loaded {len(stage_times)} stage timings from {log_path}")
        for name, t in stage_times.items():
            if t is not None:
                print(f"    {name:<22} {t:>8.1f}s")
    else:
        print(f"  No pipeline_diagnostic.log found — timing columns will be empty")

    # ── Build benchmark table ────────────────────────────────────────────────
    gameca_rows = []
    for stage, t in (stage_times.items() if stage_times else []):
        gameca_rows.append({
            "tool":        "GAMECA",
            "stage":       stage,
            "wall_sec":    round(t, 1) if t is not None else None,
            "user_steps":  GAMECA_USER_STEPS,
        })

    manual_rows = []
    total_manual_steps = 0
    for stage, n_steps, description in MANUAL_STEPS:
        manual_rows.append({
            "tool":        "Manual workflow",
            "stage":       stage,
            "wall_sec":    None,  # not fabricated — left blank
            "user_steps":  n_steps,
            "description": description,
        })
        total_manual_steps += n_steps

    df_gameca = pd.DataFrame(gameca_rows) if gameca_rows else pd.DataFrame(
        columns=["tool","stage","wall_sec","user_steps"])
    df_manual = pd.DataFrame(manual_rows)

    # Combine for CSV
    df_combined = pd.concat([df_gameca, df_manual], ignore_index=True)
    bench_csv = reports_dir / "benchmark_table.csv"
    df_combined.to_csv(bench_csv, index=False)
    print(f"  Table → {bench_csv}")

    total_gameca_sec = sum(t for t in stage_times.values() if t is not None) if stage_times else None

    # ── Bar chart: GAMECA stage timings ─────────────────────────────────────
    if gameca_rows:
        timed = [(r["stage"], r["wall_sec"]) for r in gameca_rows if r["wall_sec"] is not None]
        if timed:
            stages_plot  = [t[0] for t in timed]
            seconds_plot = [t[1] for t in timed]

            palette = [
                "#4C9BE8","#E8604C","#2ECC71","#9B59B6","#F39C12",
                "#1ABC9C","#E74C3C","#3498DB","#E67E22","#8E44AD",
            ]
            fig, ax = plt.subplots(figsize=(max(9, len(stages_plot) * 1.3 + 2), 5))
            x = np.arange(len(stages_plot))
            bars = ax.bar(x, seconds_plot,
                          color=[palette[i % len(palette)] for i in range(len(stages_plot))],
                          alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(stages_plot, rotation=40, ha="right", fontsize=9)
            ax.set_ylabel("Wall-clock time (seconds)")
            ax.set_title(
                f"{family_name} ({assembly}) — GAMECA Stage Wall-clock Times\n"
                f"Total: {total_gameca_sec:.0f}s  |  User steps: {GAMECA_USER_STEPS} "
                f"(vs {total_manual_steps} manual steps)",
                fontweight="bold",
            )
            for bar in bars:
                h = bar.get_height()
                if h and h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                            f"{h:.0f}s", ha="center", va="bottom", fontsize=8)
            ax.spines[["top","right"]].set_visible(False)
            ax.yaxis.grid(True, linestyle="--", alpha=0.4)
            ax.set_axisbelow(True)
            plt.tight_layout()
            fig_path = reports_dir / "fig_benchmark.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Bar chart → {fig_path}")

    # ── Step-count comparison bar (GAMECA vs manual) ─────────────────────────
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(["GAMECA", "Manual\nworkflow"],
            [GAMECA_USER_STEPS, total_manual_steps],
            color=["#2ECC71", "#E8604C"], alpha=0.85, width=0.5)
    for i, v in enumerate([GAMECA_USER_STEPS, total_manual_steps]):
        ax2.text(i, v + 0.3, str(v), ha="center", va="bottom", fontsize=12,
                 fontweight="bold")
    ax2.set_ylabel("Number of user commands")
    ax2.set_title(f"Steps-to-result: GAMECA vs Manual Workflow\n"
                  f"(Manual steps documented; no runtime fabricated)",
                  fontweight="bold", fontsize=10)
    ax2.spines[["top","right"]].set_visible(False)
    ax2.set_ylim(0, total_manual_steps * 1.2)
    plt.tight_layout()
    steps_path = reports_dir / "fig_benchmark_steps.png"
    fig2.savefig(steps_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Steps plot → {steps_path}")

    # ── LaTeX macros ─────────────────────────────────────────────────────────
    tex = (
        "% Auto-generated by run_benchmark.py\n"
        f"\\newcommand{{\\benchGAMECAsteps}}{{{GAMECA_USER_STEPS}}}\n"
        f"\\newcommand{{\\benchManualSteps}}{{{total_manual_steps}}}\n"
        f"\\newcommand{{\\benchNStages}}{{{len(stage_times)}}}\n"
    )
    if total_gameca_sec is not None:
        tex += (
            f"\\newcommand{{\\benchTotalSec}}{{{total_gameca_sec:.0f}}}\n"
            f"\\newcommand{{\\benchTotalMin}}{{{total_gameca_sec/60:.1f}}}\n"
        )
    (reports_dir / "benchmark_values.tex").write_text(tex)
    print(f"  LaTeX values → {reports_dir / 'benchmark_values.tex'}")

    print(f"  GAMECA user steps:   {GAMECA_USER_STEPS}")
    print(f"  Manual workflow steps: {total_manual_steps}")
    if total_gameca_sec:
        print(f"  GAMECA total time:   {total_gameca_sec:.0f}s ({total_gameca_sec/60:.1f} min)")


def _parse_args():
    p = argparse.ArgumentParser(
        description="GAMECA benchmarking vs manual workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input",       required=True)
    p.add_argument("--reports-dir", required=True)
    p.add_argument("--family",      default="FAMILY")
    p.add_argument("--assembly",    default="hg38")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_benchmark(
        input_csv   = args.input,
        reports_dir = args.reports_dir,
        family_name = args.family,
        assembly    = args.assembly,
    )
