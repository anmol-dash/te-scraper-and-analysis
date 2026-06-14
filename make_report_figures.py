#!/usr/bin/env python3
"""
make_report_figures.py — collect or generate the figures gameca_report.tex needs

For a given family and results directory (a <output>/<family_lower>/ produced by
query.py --local or HPC batch mode), this script ensures every figure and
measured-values macro file referenced in gameca_report.tex is present in
./figures/ (the directory the report's \\graphicspath looks in):

  • If a file already exists under <results-dir>/reports/ (or expression_plots/
    for the DE figures), it is copied straight into figures/ — no analysis is
    re-run.
  • If it is missing, the corresponding GAMECA standout module is invoked
    against the family's clustered CSV to produce it, then the result is copied.

Usage:
    python make_report_figures.py --family THE1D-int \\
        --results-dir results/the1d-int [--figures-dir figures] [--force]

The script is idempotent: run it repeatedly and only missing figures are
regenerated.  Pass --force to regenerate everything regardless.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Figure groups — each standout module writes into <results-dir>/reports/
# Runner scripts share the uniform CLI:
#   python <runner> --input <clustered_csv> --reports-dir <reports_dir>
#                   --family <family> [extra_args...]
# ---------------------------------------------------------------------------
STANDOUT_GROUPS = [
    dict(
        key="phylo",
        runner="run_phylo_analysis.py",
        extra=["--subst-rate", "2.2e-9", "--clock-divisor", "2",
               "--intact-orf-aa", "100"],
        outputs=["fig_phylo_tree.png", "fig_phylo_divergence.png",
                 "fig_phylo_master.png", "phylo_measured_values.tex"],
        needs_consensus=True,
    ),
    dict(
        key="grna_offtarget",
        runner="run_grna_offtarget.py",
        extra=["--cas", "SpCas9", "--max-mm", "2"],
        outputs=["fig_grna_offtarget_pareto.png",
                 "grna_offtarget_measured_values.tex"],
        needs_consensus=False,
    ),
    dict(
        key="transduction",
        runner="run_transduction.py",
        extra=["--tail-bp", "150", "--min-shared", "3"],
        outputs=["fig_transduction_groups.png",
                 "transduction_measured_values.tex"],
        needs_consensus=False,
    ),
    dict(
        key="antisense",
        runner="run_antisense_promoter.py",
        extra=["--promoter-bp", "200"],
        outputs=["fig_antisense_motifs.png", "antisense_measured_values.tex"],
        needs_consensus=False,
    ),
    dict(
        key="ctcf_tad",
        runner="run_ctcf_tad.py",
        extra=["--motif-mismatch", "3"],
        outputs=["fig_ctcf_overlap.png", "ctcf_tad_measured_values.tex"],
        needs_consensus=False,
    ),
    # ── New Stage 11 modules (Jun 2026) ─────────────────────────────────────
    dict(
        key="divergence",
        runner="run_divergence.py",
        extra=[],
        outputs=["fig_repeat_landscape.png", "fig_repeat_landscape.pdf",
                 "fig_repeat_landscape_per_cluster.png",
                 "divergence_per_locus.csv", "divergence_stats.csv",
                 "repeat_landscape_values.tex"],
        needs_consensus=True,
    ),
    dict(
        key="ltr_struct",
        runner="run_ltr_struct.py",
        extra=[],
        outputs=["fig_ltr_struct.png", "ltr_struct_annotated.csv",
                 "ltr_struct_summary.csv", "ltr_struct_values.tex"],
        needs_consensus=False,
    ),
    dict(
        key="subfamily",
        runner="run_subfamily.py",
        extra=[],
        outputs=["fig_subfamily_tree.png", "fig_subfamily_divergence.png",
                 "subfamily_table.csv", "subfamily_tree.nwk",
                 "subfamily_values.tex"],
        needs_consensus=True,
    ),
    dict(
        key="benchmark",
        runner="run_benchmark.py",
        extra=[],
        outputs=["fig_benchmark.png", "fig_benchmark_steps.png",
                 "benchmark_table.csv", "benchmark_values.tex"],
        needs_consensus=False,
    ),
    dict(
        key="motif_gain",
        runner="run_motif_gain.py",
        extra=[],
        outputs=["fig_motif_gains_bar.png", "fig_motif_gains_heatmap.png",
                 "motif_gain_per_copy.csv", "motif_gain_summary.csv",
                 "motif_gain_values.tex"],
        needs_consensus=True,
    ),
]

# Modules whose outputs live in expression_plots/ instead of reports/
# Runner: te_expression.py --input <csv> --reports-dir <rep> --family <fam>
#                           --out-dir <results_dir>
EXPRESSION_GROUP = dict(
    key="expression_de",
    runner="te_expression.py",
    extra=[],
    outputs=["de_heatmap.png", "de_volcano.png",
             "de_pairwise.csv", "de_significant.csv", "de_values.tex"],
)

# Modules with optional/extra args that need user-supplied presets — we only
# copy their figures through if present; we don't auto-run them (missing args).
PASSTHROUGH_MODULES = [
    "epigenetic_measured_values.tex",
    "ortholog_measured_values.tex",
    "multiassembly_measured_values.tex",
    "fold_measured_values.tex",
    "provenance_measured_values.tex",
]


def _pp(msg: str):
    print(msg, flush=True)


def _copy(src: Path, dst_dir: Path):
    dst = dst_dir / src.name
    if dst.resolve() != src.resolve():
        shutil.copy2(src, dst)


def _skip(path: Path, force: bool) -> bool:
    """Return True (and print a skip message) if path exists and --force is off."""
    if not force and path.exists():
        _pp(f"  [skip] {path.name} already present")
        return True
    return False


def _run_module(key, runner_path, cmd, reports_dir):
    """Invoke a standout module subprocess; return True on success."""
    _pp(f"  [{key}] running {runner_path.name} ...")
    rc = subprocess.run(cmd, stdout=None, stderr=None).returncode
    if rc != 0:
        _pp(f"  [{key}] FAILED (exit {rc})")
        return False
    _pp(f"  [{key}] ok")
    return True


def ensure_standout_group(group, clustered_csv, reports_dir, cons_fa,
                          family, assembly, script_dir, python, figures_dir, force):
    src_dir = reports_dir
    primary_fig = group["outputs"][0]

    if _skip(src_dir / primary_fig, force):
        for fname in group["outputs"]:
            p = src_dir / fname
            if p.exists():
                _copy(p, figures_dir)
        return

    runner = script_dir / group["runner"]
    if not runner.exists():
        _pp(f"  [{group['key']}] SKIP — {runner.name} not found in {script_dir}")
        return

    cmd = [python, str(runner),
           "--input",       str(clustered_csv),
           "--reports-dir", str(reports_dir),
           "--family",      family]
    if group.get("needs_consensus") and cons_fa and cons_fa.exists():
        cmd += ["--consensus-fasta", str(cons_fa)]
    cmd += group["extra"]
    if assembly and group["key"] in ("divergence", "subfamily", "motif_gain"):
        cmd += ["--assembly", assembly]

    ok = _run_module(group["key"], runner, cmd, reports_dir)
    if ok:
        for fname in group["outputs"]:
            p = src_dir / fname
            if p.exists():
                _copy(p, figures_dir)


def ensure_expression_de(clustered_csv, results_dir, reports_dir,
                          family, script_dir, python, figures_dir, force):
    expr_dir = results_dir / "expression_plots"
    primary = expr_dir / "de_heatmap.png"

    if _skip(primary, force):
        for fname in EXPRESSION_GROUP["outputs"]:
            p = expr_dir / fname
            if p.exists():
                _copy(p, figures_dir)
        return

    runner = script_dir / EXPRESSION_GROUP["runner"]
    if not runner.exists():
        _pp("  [expression_de] SKIP — te_expression.py not found")
        return

    cmd = [python, str(runner),
           "--input",       str(clustered_csv),
           "--reports-dir", str(reports_dir),
           "--family",      family,
           "--out-dir",     str(results_dir)]
    ok = _run_module("expression_de", runner, cmd, expr_dir)
    if ok:
        for fname in EXPRESSION_GROUP["outputs"]:
            p = expr_dir / fname
            if p.exists():
                _copy(p, figures_dir)


def collect_passthrough(reports_dir, figures_dir):
    """Copy over any already-present measured-values tex files not auto-run here."""
    for fname in PASSTHROUGH_MODULES:
        p = reports_dir / fname
        if p.exists():
            _copy(p, figures_dir)
            _pp(f"  [passthrough] copied {fname}")


def merge_macro_files(reports_dir, expr_dir, figures_dir):
    """
    Collect all *_values.tex files from reports/ and expression_plots/ and
    concatenate them into figures/measured_values.tex so the report's single
    \\input{figures/measured_values.tex} picks up all measured values.
    """
    import datetime

    parts = [
        "% Auto-generated by make_report_figures.py",
        f"% {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    tex_files = sorted(reports_dir.glob("*_values.tex"))
    for tf in tex_files:
        parts.append(f"% ── {tf.name} ────────────────────────────")
        parts.append(tf.read_text())
        parts.append("")

    if expr_dir.exists():
        for tf in sorted(expr_dir.glob("*_values.tex")):
            parts.append(f"% ── {tf.name} (expression_plots) ────")
            parts.append(tf.read_text())
            parts.append("")

    out = figures_dir / "measured_values.tex"
    out.write_text("\n".join(parts))
    _pp(f"  Wrote {out}  ({len(tex_files)} report + expression sources)")


def _parse_args():
    p = argparse.ArgumentParser(
        description="Collect or generate figures for gameca_report.tex",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--family", required=True,
                   help="TE family name (e.g. THE1D-int, HERVK9-int)")
    p.add_argument("--results-dir", required=True,
                   help="Path to the pipeline output directory for this family "
                        "(contains 01_data/, reports/, expression_plots/, ...)")
    p.add_argument("--figures-dir", default="figures",
                   help="Destination directory for the report (default: figures/)")
    p.add_argument("--assembly", default="hg38",
                   help="Genome assembly (default: hg38)")
    p.add_argument("--force", action="store_true",
                   help="Regenerate figures even if they already exist")
    p.add_argument("--script-dir", default=None,
                   help="Directory containing run_*.py scripts "
                        "(default: same directory as this script)")
    return p.parse_args()


def main():
    args = _parse_args()

    results_dir = Path(args.results_dir).resolve()
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        _pp(f"ERROR: results directory not found: {results_dir}")
        sys.exit(1)

    # Derive standard sub-paths from the results directory
    reports_dir = results_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    expr_dir = results_dir / "expression_plots"

    family_lower = args.family.lower().replace("-", "_").replace(" ", "_")
    clustered_csv = results_dir / "01_data" / f"{family_lower}_clustered.csv"
    if not clustered_csv.exists():
        # Fallback: search for any *_clustered.csv under 01_data/
        candidates = sorted((results_dir / "01_data").glob("*_clustered.csv"))
        if candidates:
            clustered_csv = candidates[0]
            _pp(f"  Note: using clustered CSV {clustered_csv.name}")

    if not clustered_csv.exists():
        _pp(f"ERROR: clustered CSV not found under {results_dir}/01_data/")
        _pp("  Run the full GAMECA pipeline first (query.py --local --family ...)")
        sys.exit(1)

    cons_fa = results_dir / "cluster_alignments" / "all_cluster_consensuses.fa"

    script_dir = Path(args.script_dir) if args.script_dir else Path(__file__).resolve().parent
    python = shutil.which("python3") or shutil.which("python") or sys.executable

    _pp("=" * 60)
    _pp(f" make_report_figures  family={args.family}")
    _pp(f" results-dir : {results_dir}")
    _pp(f" figures-dir : {figures_dir.resolve()}")
    _pp(f" force       : {args.force}")
    _pp("=" * 60)

    # ── Standout modules ────────────────────────────────────────────────────
    for group in STANDOUT_GROUPS:
        _pp(f"\n[{group['key']}]")
        ensure_standout_group(
            group, clustered_csv, reports_dir, cons_fa,
            args.family, args.assembly, script_dir, python,
            figures_dir, args.force,
        )

    # ── Expression DE ────────────────────────────────────────────────────────
    _pp("\n[expression_de]")
    ensure_expression_de(
        clustered_csv, results_dir, reports_dir,
        args.family, script_dir, python, figures_dir, args.force,
    )

    # ── Passthrough (epigenetic, ortholog, fold, provenance, motif_gain) ────
    _pp("\n[passthrough — copy existing measured-value tex files]")
    collect_passthrough(reports_dir, figures_dir)

    # ── Consolidate all *_values.tex into figures/measured_values.tex ───────
    _pp("\n[merge macro files]")
    merge_macro_files(reports_dir, expr_dir, figures_dir)

    _pp("\n" + "=" * 60)
    _pp(f" Done.  Place '{figures_dir}/' next to gameca_report.tex and compile.")
    _pp("=" * 60)


if __name__ == "__main__":
    main()
