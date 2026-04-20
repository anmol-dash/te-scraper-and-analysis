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
import os
from pathlib import Path


# ── ANSI helpers ──────────────────────────────────────────────────────────────

def _c(text, code):
    if sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text

def bold(t):      return _c(t, "1")
def green(t):     return _c(t, "32")
def yellow(t):    return _c(t, "33")
def cyan(t):      return _c(t, "36")
def red(t):       return _c(t, "31")
def dim(t):       return _c(t, "2")
def magenta(t):   return _c(t, "35")
def blue(t):      return _c(t, "34")
def bright(t):    return _c(t, "97")
def bg_dark(t):   return _c(t, "40")

def _header(title: str, width: int = 66) -> str:
    pad = width - len(title) - 4
    left = pad // 2
    right = pad - left
    return (
        cyan("╔" + "═" * (width - 2) + "╗") + "\n"
        + cyan("║") + " " * left + bold(bright(title)) + " " * right + cyan("║") + "\n"
        + cyan("╚" + "═" * (width - 2) + "╝")
    )

def _box(lines: list[str], width: int = 66) -> str:
    result = cyan("┌" + "─" * (width - 2) + "┐") + "\n"
    for line in lines:
        visible = _strip_ansi(line)
        pad = width - 2 - len(visible)
        result += cyan("│") + " " + line + " " * max(0, pad - 1) + cyan("│") + "\n"
    result += cyan("└" + "─" * (width - 2) + "┘")
    return result

def _strip_ansi(s: str) -> str:
    import re
    return re.sub(r"\033\[[0-9;]*m", "", s)

def _divider(label: str = "", width: int = 66, char: str = "─") -> str:
    if label:
        label_str = f" {label} "
        side = (width - len(label_str)) // 2
        return dim(char * side) + bold(label_str) + dim(char * (width - side - len(label_str)))
    return dim(char * width)


# ── Native file/directory picker ───────────────────────────────────────────────

def _pick_path(prompt: str, pick_dir: bool = False, default: str = "") -> str:
    """Open a native OS file picker (Finder on macOS). Falls back to text input."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        if pick_dir:
            path = filedialog.askdirectory(title=prompt)
        else:
            path = filedialog.askopenfilename(
                title=prompt,
                filetypes=[
                    ("CSV files", "*.csv"),
                    ("TSV files", "*.tsv"),
                    ("FASTA files", "*.fa *.fasta *.fna"),
                    ("BED files", "*.bed *.bed.gz"),
                    ("All files", "*.*"),
                ],
            )
        root.destroy()
        if path:
            return path
    except Exception:
        pass

    # Fallback to text input
    val = input(f"  {cyan('›')} {prompt}" + (f" [{default}]" if default else "") + ": ").strip()
    return val or default


def _ask(prompt: str, default: str = "", pick: bool = False, pick_dir: bool = False) -> str:
    if pick or pick_dir:
        print(f"  {dim('Opening file picker…')}")
        return _pick_path(prompt, pick_dir=pick_dir, default=default)
    val = input(f"  {cyan('›')} {prompt}" + (f" [{default}]" if default else "") + ": ").strip()
    return val or default


def _confirm(prompt: str, default: bool = False) -> bool:
    hint = "Y/n" if default else "y/N"
    val = input(f"  {cyan('›')} {prompt} ({hint}): ").strip().lower()
    if not val:
        return default
    return val.startswith("y")


# ── ASCII art ─────────────────────────────────────────────────────────────────

GAMECA_ART = r"""
  ██████╗  █████╗ ███╗   ███╗███████╗ ██████╗  █████╗
 ██╔════╝ ██╔══██╗████╗ ████║██╔════╝██╔════╝ ██╔══██╗
 ██║  ███╗███████║██╔████╔██║█████╗  ██║      ███████║
 ██║   ██║██╔══██║██║╚██╔╝██║██╔══╝  ██║      ██╔══██║
 ╚██████╔╝██║  ██║██║ ╚═╝ ██║███████╗╚██████╗ ██║  ██║
  ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝
"""

TAGLINE = "Gene Alignment, Motif, Expression & Clustering Analysis"

PIPELINE_DIAGRAM = """
  Coords ──→ Get Sequences ──┬──→ MAFFT ──→ CIAlign ──→ AlphaFold   [Branch A]
                             ├──→ UMAP / HDBSCAN ──→ Cluster QC      [Branch B]
                             ├──→ JASPAR overlap ──→ Motif analysis   [Branch C]
                             └──→ Expression collation                [Branch D]
                                        │                   │
                                  [Join B+D]           [Join C+D]
                                        └────────┬──────────┘
                                           Generate gRNAs
                                        ┌────────┴──────────┐
                                   Off-target QC       Stability QC
"""


# ── Workflow docs ─────────────────────────────────────────────────────────────

WORKFLOW = """
Each step is a self-contained script with its own CLI.
Steps pass data forward via CSV files.

STEP 1 — Prepare  (te_prep.py)
  Downloads RepeatMasker annotations from UCSC (rmsk.txt.gz),
  extracts sequences for a TE family, and writes the input CSV.

  OUTPUT → {family}_sequences.csv
           columns: chr, start, stop, TE_name, strand, Seq

  python te_prep.py --build hg38 --family HERVK --out-dir ./data

  Key env vars:
    HG38_FA   — path to local hg38 FASTA (fast pysam extraction)
    TE_BASE_DIR, TE_RMSK_DIR

STEP C — Clustering  (te_clustering.py)
  k-mer encoding → PCA / UMAP / t-SNE → HDBSCAN clustering.

  INPUT  → sequences CSV (needs Seq column)
  OUTPUT → clustered.csv (adds Cluster, umap_x, umap_y, pca_x, …)
           clustering_visualization.html
           clustering_coordinates.csv

  python te_clustering.py --input sequences.csv --kmer 18

STEP A — Alignment  (te_alignment.py)
  MAFFT global + per-cluster alignment, CIAlign visualisation,
  majority-vote consensus generation.

  INPUT  → clustered CSV (needs Seq + Cluster)
  OUTPUT → cluster_alignments/*.fa, cialign_plots/index.html
           cleaned_consensus/*.fa

  python te_alignment.py --input clustered.csv --family HERVK

STEP M — Motif  (te_motif.py)
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

STEP G — Gene / GO  (te_go.py)
  Looks up GO terms for enriched TF motifs via mygene.info API.

  INPUT  → enrichment_results/ directory (from te_motif.py)
  OUTPUT → go_annotations/gene_functions.csv
           go_annotations/cluster_N_enrichment_annotated.csv
           go_annotations/strand_plots/*.png

  python te_go.py --enrichment-dir ./results/enrichment_results \\
                  --clustered-csv  ./results/clustered.csv \\
                  --build hg38

STEP E — Expression  (te_expression.py)
  Per-cluster expression boxplots from numeric columns.
  Auto-detects expression columns or accepts explicit names.

  INPUT  → clustered CSV (needs Cluster + numeric expression cols)
  OUTPUT → expression_plots/boxplot_all.png
           expression_plots/boxplot_mid80.png
           expression_plots/expression_stats.csv

  python te_expression.py --input clustered.csv

ORCHESTRATION
  Run the full core pipeline:
    python query.py --family HERVK --genome hg38.fa --out-dir ./results

  Run all enrichment steps (M + G + E):
    python te_enrichment.py --input clustered.csv --build hg38

  HPC (auto-detects LSF / Slurm):
    python hpc_client.py
"""


# ── Results summary ───────────────────────────────────────────────────────────

def _tick(p: Path) -> str:
    return green("✔") if p.exists() else dim("·")


def display_results(out_dir: Path):
    out_dir = Path(out_dir).expanduser()
    if not out_dir.exists():
        print(red(f"\n  Directory not found: {out_dir}"))
        return

    print()
    print(_header(f"Results  ·  {out_dir.name}"))
    print()

    sections = [
        ("Core outputs", [
            ("clustered_data.csv",           "Clustered sequences CSV"),
            ("clustering_coordinates.csv",    "UMAP / PCA / t-SNE coords"),
            ("clustering_visualization.html", "Clustering interactive plot"),
        ]),
        ("Alignment", [
            ("cluster_alignments",  "Per-cluster alignment FASTAs"),
            ("cialign_plots",       "CIAlign plots  (open index.html)"),
            ("cleaned_consensus",   "CIAlign-cleaned consensus FASTAs"),
        ]),
        ("Primers", [
            ("selected_primers_summary.csv",   "Global primer summary"),
            ("cluster_top_primers.csv",        "Per-cluster top primers"),
            ("primer_genome_hits_summary.csv", "Genome-wide hit counts"),
        ]),
        ("Motif  (te_motif.py)", [
            ("motif_analysis/all_overlaps.tsv",           "bedtools overlap TSV"),
            ("motif_analysis/overall_top_motifs.png",     "Top-20 motifs bar chart"),
            ("enrichment_results/enrichment_heatmap.png", "Enrichment heatmap"),
        ]),
        ("GO annotation  (te_go.py)", [
            ("go_annotations/gene_functions.csv", "Gene → GO terms"),
        ]),
        ("Expression  (te_expression.py)", [
            ("expression_plots/boxplot_all.png",      "Boxplot — all data"),
            ("expression_plots/boxplot_mid80.png",    "Boxplot — mid 80%"),
            ("expression_plots/expression_stats.csv", "Per-cluster stats"),
        ]),
    ]

    for section_label, items in sections:
        print(_divider(section_label))
        for name, label in items:
            p = out_dir / name
            if not p.exists() and "/" not in name:
                alt = out_dir / "06_primers" / name
                if alt.exists():
                    p = alt
            basename = name.split("/")[-1]
            print(f"  {_tick(p)}  {label:<46} {dim(basename)}")
        print()

    # Checkpoints
    print(_divider("Checkpoints"))
    cp_path = out_dir / "enrichment_checkpoints.json"
    if cp_path.exists():
        import json
        cp = json.loads(cp_path.read_text())
        for step, info in cp.items():
            ts = info.get("completed_at", "")
            print(f"  {green('✔')}  {bold(step):<20} {dim(ts)}")
    else:
        print(f"  {dim('·')}  No checkpoint file found")
    print()

    # Cluster table
    cluster_csv = out_dir / "cluster_alignments" / "cluster_consensus_summary.csv"
    if cluster_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(cluster_csv)
            disp_cols = [c for c in ["cluster", "size", "consensus_length"] if c in df.columns]
            print(_divider(f"Clusters  ({len(df)})"))
            for _, row in df[disp_cols].iterrows():
                parts = [f"{c}: {bold(str(row[c]))}" for c in disp_cols]
                print(f"  {cyan('·')}  {'   '.join(parts)}")
            print()
        except Exception:
            pass


# ── Step runners ──────────────────────────────────────────────────────────────

def _run(label: str, cmd: str):
    import subprocess
    print()
    print(f"  {dim('$')} {dim(cmd)}")
    print(_divider())
    result = subprocess.run(cmd, shell=True)
    print(_divider())
    if result.returncode != 0:
        print(f"  {red('✘')} {bold(label)} exited with code {result.returncode}")
    else:
        print(f"  {green('✔')} {bold(label)} complete")
    print()


def _step_header(label: str):
    print()
    print(_divider(label))


# ── Menu steps ────────────────────────────────────────────────────────────────

def _step_prep():
    _step_header("te_prep.py  ·  Download rmsk + extract sequences")
    build   = _ask("Genome build", "hg38")
    family  = _ask("TE family name (e.g. HERVK)")
    out_dir = _ask("Output directory  [or pick]", "./te_data", pick_dir=True)
    cmd = f"python te_prep.py --build {build} --out-dir {out_dir}"
    if family:
        cmd += f" --family {family}"
    _run("te_prep", cmd)


def _step_clustering():
    _step_header("te_clustering.py  ·  k-mer + UMAP + HDBSCAN")
    inp     = _ask("Input sequences CSV  [pick file]", pick=True)
    out     = _ask("Output CSV path", inp or "clustered.csv")
    out_dir = _ask("Visualization directory  [or pick]", "./results", pick_dir=True)
    kmer    = _ask("k-mer size", "18")
    family  = _ask("Family name", "FAMILY")
    _run("te_clustering",
         f"python te_clustering.py --input {inp} --output {out} "
         f"--kmer {kmer} --out-dir {out_dir} --family {family}")


def _step_alignment():
    _step_header("te_alignment.py  ·  MAFFT + CIAlign + consensus")
    inp     = _ask("Input clustered CSV  [pick file]", pick=True)
    out_dir = _ask("Output directory  [or pick]", "./results", pick_dir=True)
    family  = _ask("Family name", "FAMILY")
    skip_ci = _confirm("Skip CIAlign?", default=False)
    cmd = f"python te_alignment.py --input {inp} --out-dir {out_dir} --family {family}"
    if skip_ci:
        cmd += " --no-cialign"
    _run("te_alignment", cmd)


def _step_motif():
    _step_header("te_motif.py  ·  JASPAR overlap + Fisher enrichment")
    inp     = _ask("Input clustered CSV  [pick file]", pick=True)
    build   = _ask("Genome build", "hg38")
    out_dir = _ask("Output directory  [or pick]", "./results", pick_dir=True)
    if _confirm("Provide a JASPAR BED file?", default=False):
        jaspar = _ask("JASPAR BED path  [pick file]", pick=True)
    else:
        jaspar = ""
    cmd = f"python te_motif.py --input {inp} --build {build} --out-dir {out_dir}"
    if jaspar:
        cmd += f" --jaspar-bed {jaspar}"
    _run("te_motif", cmd)


def _step_go():
    _step_header("te_go.py  ·  GO annotation via mygene.info")
    enrich_dir = _ask("enrichment_results directory  [pick]", pick_dir=True)
    if _confirm("Include clustered CSV for strand plots?", default=False):
        clustered = _ask("Clustered CSV  [pick file]", pick=True)
    else:
        clustered = ""
    build   = _ask("Genome build", "hg38")
    out_dir = _ask("Output directory  [or pick]", "./results", pick_dir=True)
    cmd = (f"python te_go.py --enrichment-dir {enrich_dir} "
           f"--build {build} --out-dir {out_dir}")
    if clustered:
        cmd += f" --clustered-csv {clustered}"
    _run("te_go", cmd)


def _step_expression():
    _step_header("te_expression.py  ·  Per-cluster expression boxplots")
    inp     = _ask("Input clustered CSV  [pick file]", pick=True)
    out_dir = _ask("Output directory  [or pick]", "./results", pick_dir=True)
    _run("te_expression",
         f"python te_expression.py --input {inp} --out-dir {out_dir}")


def _step_enrichment():
    _step_header("te_enrichment.py  ·  Motif + GO + Expression (M+G+E)")
    inp     = _ask("Input clustered CSV  [pick file]", pick=True)
    build   = _ask("Genome build", "hg38")
    out_dir = _ask("Output directory  [or pick]", "./results", pick_dir=True)
    if _confirm("Provide a JASPAR BED file?", default=False):
        jaspar = _ask("JASPAR BED path  [pick file]", pick=True)
    else:
        jaspar = ""
    skip_go   = _confirm("Skip GO annotation?",      default=False)
    skip_expr = _confirm("Skip expression plots?",   default=False)
    cmd = f"python te_enrichment.py --input {inp} --build {build} --out-dir {out_dir}"
    if jaspar:    cmd += f" --jaspar-bed {jaspar}"
    if skip_go:   cmd += " --skip-go"
    if skip_expr: cmd += " --skip-expression"
    _run("te_enrichment", cmd)


# ── Interactive menu ──────────────────────────────────────────────────────────

MENU = [
    # (key, label, category)
    ("1",  "Workflow overview",                           "info"),
    (None, "Data prep",                                   "section"),
    ("2",  "te_prep.py       Download rmsk + sequences",  "action"),
    (None, "Core analysis",                               "section"),
    ("3",  "te_clustering.py k-mer · UMAP · HDBSCAN",    "action"),
    ("4",  "te_alignment.py  MAFFT · CIAlign · consensus","action"),
    (None, "Enrichment",                                  "section"),
    ("5",  "te_motif.py      JASPAR overlap + Fisher",    "action"),
    ("6",  "te_go.py         GO annotation (mygene.info)","action"),
    ("7",  "te_expression.py Expression boxplots",        "action"),
    ("8",  "te_enrichment.py All enrichment  (M+G+E)",   "action"),
    (None, "Utilities",                                   "section"),
    ("9",  "View results summary",                        "action"),
    ("10", "Open HPC client",                             "action"),
    ("11", "Exit",                                        "action"),
]


def _print_menu():
    os.system("clear" if sys.stdout.isatty() else ":")
    print()
    print(cyan(GAMECA_ART.rstrip()))
    print(f"  {dim(TAGLINE)}")
    print()
    print(dim(PIPELINE_DIAGRAM))
    print()
    print(_divider("Menu"))
    for key, label, kind in MENU:
        if kind == "section":
            print(f"\n  {dim('┄' * 3)}  {bold(label)}")
        elif kind == "info":
            print(f"    {cyan(f'[{key}]')}  {dim(label)}")
        else:
            num = f"[{key}]"
            print(f"    {cyan(num):<14}  {label}")
    print()


def interactive_menu():
    while True:
        _print_menu()
        choice = input(f"  {cyan('›')} {bold('Select')}: ").strip()

        if choice == '1':
            print()
            print(_header("GAMECA Pipeline  ·  Workflow Overview"))
            print(WORKFLOW)
            input(f"  {dim('Press Enter to return to menu…')}")

        elif choice == '2':  _step_prep()
        elif choice == '3':  _step_clustering()
        elif choice == '4':  _step_alignment()
        elif choice == '5':  _step_motif()
        elif choice == '6':  _step_go()
        elif choice == '7':  _step_expression()
        elif choice == '8':  _step_enrichment()

        elif choice == '9':
            _step_header("Results summary")
            out_dir = _ask("Results directory  [or pick]", "./results", pick_dir=True)
            display_results(Path(out_dir))
            input(f"  {dim('Press Enter to return to menu…')}")

        elif choice == '10':
            import subprocess
            print(f"\n  {dim('Starting HPC client…')}")
            subprocess.run([sys.executable, "hpc_client.py"])

        elif choice == '11':
            print(f"\n  {green('Goodbye!')}  {dim('— GAMECA')}\n")
            sys.exit(0)

        else:
            print(red("  Invalid selection — please choose a number from the menu."))
            import time; time.sleep(1)


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
        print(cyan(GAMECA_ART))
        print(_header("GAMECA Pipeline  ·  Workflow Overview"))
        print(WORKFLOW)
        return

    if args.results:
        display_results(Path(args.results))
        return

    interactive_menu()


if __name__ == "__main__":
    main()
