#!/usr/bin/env python3
"""
ui.py — GAMECA terminal launcher and results viewer.

Usage:
    python ui.py                    # interactive menu
    python ui.py --results <dir>    # display results summary for an output directory
    python ui.py --help-workflow    # print workflow overview
"""

import argparse
import sys
from pathlib import Path


# ── ANSI helpers ──────────────────────────────────────────────────────────────

def _c(text, code):
    if sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text

def bold(t):    return _c(t, "1")
def green(t):   return _c(t, "32")
def yellow(t):  return _c(t, "33")
def cyan(t):    return _c(t, "36")
def red(t):     return _c(t, "31")
def dim(t):     return _c(t, "2")


# ── ASCII art ─────────────────────────────────────────────────────────────────

GAMECA_ART = r"""
  ██████╗  █████╗ ███╗   ███╗███████╗ ██████╗ █████╗
 ██╔════╝ ██╔══██╗████╗ ████║██╔════╝██╔════╝██╔══██╗
 ██║  ███╗███████║██╔████╔██║█████╗  ██║     ███████║
 ██║   ██║██╔══██║██║╚██╔╝██║██╔══╝  ██║     ██╔══██║
 ╚██████╔╝██║  ██║██║ ╚═╝ ██║███████╗╚██████╗██║  ██║
  ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝

   Gene Alignment, Motif, Expression & Clustering Analysis
"""

PIPELINE_STEPS = """
  te_prep.py ──→ te_clustering.py ──→ te_alignment.py
                       │
                       ├──→ te_motif.py ──→ te_go.py
                       │
                       └──→ te_expression.py
"""


# ── Workflow docs ─────────────────────────────────────────────────────────────

WORKFLOW = """
╔══════════════════════════════════════════════════════════════════╗
║              GAMECA Pipeline — Workflow Overview                 ║
╚══════════════════════════════════════════════════════════════════╝

Each step is a self-contained script with its own CLI.
Steps pass data forward via CSV files.

──────────────────────────────────────────────────────────────────
STEP 1 — Prepare  (te_prep.py)
──────────────────────────────────────────────────────────────────
  Downloads RepeatMasker annotations from UCSC (rmsk.txt.gz),
  extracts sequences for a TE family, and writes the input CSV.

  OUTPUT → {family}_sequences.csv
           columns: chr, start, stop, TE_name, strand, Seq

  python te_prep.py --build hg38 --family HERVK --out-dir ./data

  Key env vars:
    HG38_FA   — path to local hg38 FASTA (fast pysam extraction)
    TE_BASE_DIR, TE_RMSK_DIR

──────────────────────────────────────────────────────────────────
STEP C — Clustering  (te_clustering.py)
──────────────────────────────────────────────────────────────────
  k-mer encoding → PCA / UMAP / t-SNE → HDBSCAN clustering.

  INPUT  → sequences CSV (needs Seq column)
  OUTPUT → clustered.csv (adds Cluster, umap_x, umap_y, pca_x, …)
           clustering_visualization.html
           clustering_coordinates.csv

  python te_clustering.py --input sequences.csv --kmer 18

──────────────────────────────────────────────────────────────────
STEP A — Alignment  (te_alignment.py)
──────────────────────────────────────────────────────────────────
  MAFFT global + per-cluster alignment, CIAlign visualisation,
  majority-vote consensus generation.

  INPUT  → clustered CSV (needs Seq + Cluster)
  OUTPUT → cluster_alignments/*.fa, cialign_plots/index.html
           cleaned_consensus/*.fa

  python te_alignment.py --input clustered.csv --family HERVK

──────────────────────────────────────────────────────────────────
STEP M — Motif  (te_motif.py)
──────────────────────────────────────────────────────────────────
  bedtools intersect TE loci against JASPAR TFBS predictions,
  then Fisher's exact test per cluster × motif.
  JASPAR BED auto-downloaded from jaspar.elixir.no if not provided.

  INPUT  → clustered CSV (needs chr/start/stop/Cluster)
  OUTPUT → motif_analysis/all_overlaps.tsv
           enrichment_results/cluster_N_enrichment.csv
           enrichment_results/enrichment_heatmap.png

  python te_motif.py --input clustered.csv --build hg38

  Key env vars:
    TE_JASPAR_HG38 — pre-downloaded JASPAR BED path

──────────────────────────────────────────────────────────────────
STEP G — Gene / GO  (te_go.py)
──────────────────────────────────────────────────────────────────
  Looks up GO terms for enriched TF motifs via mygene.info API.

  INPUT  → enrichment_results/ directory (from te_motif.py)
  OUTPUT → go_annotations/gene_functions.csv
           go_annotations/cluster_N_enrichment_annotated.csv
           go_annotations/strand_plots/*.png  (if clustered CSV given)

  python te_go.py --enrichment-dir ./results/enrichment_results \\
                  --clustered-csv  ./results/clustered.csv \\
                  --build hg38

──────────────────────────────────────────────────────────────────
STEP E — Expression  (te_expression.py)
──────────────────────────────────────────────────────────────────
  Per-cluster expression boxplots from numeric columns.
  Auto-detects expression columns or accepts explicit names.

  INPUT  → clustered CSV (needs Cluster + numeric expression cols)
  OUTPUT → expression_plots/boxplot_all.png
           expression_plots/boxplot_mid80.png
           expression_plots/expression_stats.csv

  python te_expression.py --input clustered.csv

──────────────────────────────────────────────────────────────────
ORCHESTRATION
──────────────────────────────────────────────────────────────────
  Run the full core pipeline:
    python query.py --family HERVK --genome hg38.fa --out-dir ./results

  Run all enrichment steps (M + G + E):
    python te_enrichment.py --input clustered.csv --build hg38

  HPC (auto-detects LSF / Slurm):
    python hpc_client.py
"""


# ── Results summary ───────────────────────────────────────────────────────────

def _exists(p):
    return green("✔") if Path(p).exists() else dim("–")


def display_results(out_dir: Path):
    out_dir = Path(out_dir).expanduser()
    if not out_dir.exists():
        print(red(f"Directory not found: {out_dir}"))
        return

    print(bold(f"\nResults in {out_dir}\n"))

    print(bold("Core:"))
    for name, label in [
        ("clustered_data.csv",           "Clustered sequences CSV"),
        ("clustering_coordinates.csv",    "UMAP / PCA / t-SNE coords"),
        ("clustering_visualization.html", "Clustering interactive plot"),
    ]:
        p = out_dir / name
        print(f"  {_exists(p)}  {label:<46} {dim(name)}")

    print(bold("\nAlignment:"))
    for name, label in [
        ("cluster_alignments",  "Per-cluster alignment FASTAs"),
        ("cialign_plots",       "CIAlign plots  (open index.html)"),
        ("cleaned_consensus",   "CIAlign-cleaned consensus FASTAs"),
    ]:
        p = out_dir / name
        status = green("✔") if p.exists() else dim("–")
        print(f"  {status}  {label}")

    print(bold("\nPrimers:"))
    for name, label in [
        ("selected_primers_summary.csv",  "Global primer summary"),
        ("cluster_top_primers.csv",       "Per-cluster top primers"),
        ("primer_genome_hits_summary.csv","Genome-wide hit counts"),
    ]:
        p = out_dir / name
        # also check 06_primers/
        if not p.exists():
            p = out_dir / "06_primers" / name
        print(f"  {_exists(p)}  {label:<46} {dim(name)}")

    print(bold("\nMotif (te_motif.py):"))
    for name, label in [
        ("motif_analysis/all_overlaps.tsv",                "bedtools overlap TSV"),
        ("motif_analysis/overall_top_motifs.png",          "Top-20 motifs bar chart"),
        ("enrichment_results/enrichment_heatmap.png",      "Enrichment heatmap"),
    ]:
        p = out_dir / name
        print(f"  {_exists(p)}  {label:<46} {dim(name.split('/')[-1])}")

    print(bold("\nGO annotation (te_go.py):"))
    for name, label in [
        ("go_annotations/gene_functions.csv", "Gene → GO terms"),
    ]:
        p = out_dir / name
        print(f"  {_exists(p)}  {label:<46} {dim(name.split('/')[-1])}")

    print(bold("\nExpression (te_expression.py):"))
    for name, label in [
        ("expression_plots/boxplot_all.png",   "Boxplot — all data"),
        ("expression_plots/boxplot_mid80.png", "Boxplot — mid 80%"),
        ("expression_plots/expression_stats.csv", "Per-cluster stats"),
    ]:
        p = out_dir / name
        print(f"  {_exists(p)}  {label:<46} {dim(name.split('/')[-1])}")

    print(bold("\nCheckpoints:"))
    cp_path = out_dir / "enrichment_checkpoints.json"
    if cp_path.exists():
        import json
        cp = json.loads(cp_path.read_text())
        for step, info in cp.items():
            print(f"  {green('✔')}  {step:<16} {dim(info.get('completed_at',''))}")
    else:
        print(f"  {dim('–')}  No checkpoint file")

    # Cluster count table
    cluster_csv = out_dir / "cluster_alignments" / "cluster_consensus_summary.csv"
    if cluster_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(cluster_csv)
            print(bold(f"\nClusters ({len(df)}):"))
            disp_cols = [c for c in ["cluster","size","consensus_length"] if c in df.columns]
            print(df[disp_cols].to_string(index=False))
        except Exception:
            pass

    print()


# ── Interactive menu ──────────────────────────────────────────────────────────

def _ask(prompt, default=""):
    val = input(f"{cyan('?')} {prompt}" + (f" [{default}]" if default else "") + ": ").strip()
    return val or default


def _run(label, cmd):
    import subprocess
    print(dim(f"\n  $ {cmd}"))
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(red(f"  {label} exited with code {result.returncode}"))
    else:
        print(green(f"  {label} complete"))


def interactive_menu():
    print(bold(GAMECA_ART))
    print(dim(PIPELINE_STEPS))

    while True:
        print(f"""
{bold("Main Menu")}
  {cyan('[1]')}  Show workflow overview
  {bold("─── Data prep ───────────────────────────────────")}
  {cyan('[2]')}  te_prep.py         Download rmsk + extract sequences
  {bold("─── Core analysis ───────────────────────────────")}
  {cyan('[3]')}  te_clustering.py   k-mer + UMAP + HDBSCAN clustering
  {cyan('[4]')}  te_alignment.py    MAFFT + CIAlign + consensus
  {bold("─── Enrichment ──────────────────────────────────")}
  {cyan('[5]')}  te_motif.py        JASPAR overlap + Fisher enrichment
  {cyan('[6]')}  te_go.py           GO annotation (mygene.info)
  {cyan('[7]')}  te_expression.py   Expression boxplots
  {cyan('[8]')}  te_enrichment.py   Run all enrichment steps (M+G+E)
  {bold("─── Utils ───────────────────────────────────────")}
  {cyan('[9]')}  View results summary
  {cyan('[10]')} Open HPC client
  {cyan('[11]')} Exit
""")
        choice = input(bold("Select (1-11): ")).strip()

        if choice == '1':
            print(WORKFLOW)

        elif choice == '2':
            print(bold("\n— te_prep —"))
            build   = _ask("Genome build", "hg38")
            family  = _ask("TE family (e.g. HERVK)")
            out_dir = _ask("Output directory", "./te_data")
            cmd = f"python te_prep.py --build {build} --out-dir {out_dir}"
            if family:
                cmd += f" --family {family}"
            _run("te_prep", cmd)

        elif choice == '3':
            print(bold("\n— te_clustering —"))
            inp     = _ask("Input CSV (sequences)")
            out     = _ask("Output CSV", inp)
            out_dir = _ask("Visualization directory", "./results")
            kmer    = _ask("k-mer size", "18")
            family  = _ask("Family name", "FAMILY")
            _run("te_clustering",
                 f"python te_clustering.py --input {inp} --output {out} "
                 f"--kmer {kmer} --out-dir {out_dir} --family {family}")

        elif choice == '4':
            print(bold("\n— te_alignment —"))
            inp     = _ask("Input clustered CSV")
            out_dir = _ask("Output directory", "./results")
            family  = _ask("Family name", "FAMILY")
            skip_al = input(f"{cyan('?')} Skip CIAlign? (y/N): ").strip().lower() == 'y'
            cmd = f"python te_alignment.py --input {inp} --out-dir {out_dir} --family {family}"
            if skip_al:
                cmd += " --no-cialign"
            _run("te_alignment", cmd)

        elif choice == '5':
            print(bold("\n— te_motif —"))
            inp     = _ask("Input clustered CSV")
            build   = _ask("Genome build", "hg38")
            out_dir = _ask("Output directory", "./results")
            jaspar  = _ask("JASPAR BED path (blank = auto-download)", "")
            cmd = (f"python te_motif.py --input {inp} --build {build} --out-dir {out_dir}")
            if jaspar:
                cmd += f" --jaspar-bed {jaspar}"
            _run("te_motif", cmd)

        elif choice == '6':
            print(bold("\n— te_go —"))
            enrich_dir = _ask("enrichment_results directory")
            clustered  = _ask("Clustered CSV for strand plots (blank to skip)", "")
            build      = _ask("Genome build", "hg38")
            out_dir    = _ask("Output directory", "./results")
            cmd = (f"python te_go.py --enrichment-dir {enrich_dir} "
                   f"--build {build} --out-dir {out_dir}")
            if clustered:
                cmd += f" --clustered-csv {clustered}"
            _run("te_go", cmd)

        elif choice == '7':
            print(bold("\n— te_expression —"))
            inp     = _ask("Input clustered CSV")
            out_dir = _ask("Output directory", "./results")
            _run("te_expression",
                 f"python te_expression.py --input {inp} --out-dir {out_dir}")

        elif choice == '8':
            print(bold("\n— te_enrichment (M + G + E) —"))
            inp     = _ask("Input clustered CSV")
            build   = _ask("Genome build", "hg38")
            out_dir = _ask("Output directory", "./results")
            jaspar  = _ask("JASPAR BED path (blank = auto-download)", "")
            skip_go   = input(f"{cyan('?')} Skip GO annotation? (y/N): ").strip().lower() == 'y'
            skip_expr = input(f"{cyan('?')} Skip expression plots? (y/N): ").strip().lower() == 'y'
            cmd = (f"python te_enrichment.py --input {inp} --build {build} --out-dir {out_dir}")
            if jaspar:    cmd += f" --jaspar-bed {jaspar}"
            if skip_go:   cmd += " --skip-go"
            if skip_expr: cmd += " --skip-expression"
            _run("te_enrichment", cmd)

        elif choice == '9':
            out_dir = _ask("Results directory", "./results")
            display_results(Path(out_dir))

        elif choice == '10':
            import subprocess
            print(bold("\n  Starting HPC client…"))
            subprocess.run([sys.executable, "hpc_client.py"])

        elif choice == '11':
            print("\nGoodbye!")
            sys.exit(0)

        else:
            print(red("  Invalid option."))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GAMECA pipeline launcher and results viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results",       metavar="DIR",
                        help="Display results summary for DIR and exit")
    parser.add_argument("--help-workflow", action="store_true",
                        help="Print workflow overview and exit")
    args = parser.parse_args()

    if args.help_workflow:
        print(bold(GAMECA_ART))
        print(WORKFLOW)
        return

    if args.results:
        display_results(Path(args.results))
        return

    interactive_menu()


if __name__ == "__main__":
    main()
