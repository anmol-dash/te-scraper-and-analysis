# GAMECA — Gene Alignment, Motif, Expression & Clustering Analysis

```
  ██████╗  █████╗ ███╗   ███╗███████╗ ██████╗ █████╗
 ██╔════╝ ██╔══██╗████╗ ████║██╔════╝██╔════╝██╔══██╗
 ██║  ███╗███████║██╔████╔██║█████╗  ██║     ███████║
 ██║   ██║██╔══██║██║╚██╔╝██║██╔══╝  ██║     ██╔══██║
 ╚██████╔╝██║  ██║██║ ╚═╝ ██║███████╗╚██████╗██║  ██║
  ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝
```

A modular TE analysis pipeline where every step is a self-contained script
that reads from and writes to CSV files, so steps can be run independently,
rerun in isolation, or swapped out. Supports **LSF** and **Slurm** HPC clusters.

---
![alt text](https://github.com/anmol-dash/te-scraper-and-analysis/blob/main/ERD_Updated.png)

## Pipeline

```
te_prep.py ──→ te_clustering.py ──→ te_alignment.py
                     │
                     ├──→ te_motif.py ──→ te_go.py
                     │
                     └──→ te_expression.py
```

---

## Parallelization

The pipeline is structured as a **DAG** with well-defined fan-out and join points, enabling significant wall-clock speedups when run with a DAG scheduler (Snakemake, Nextflow, or Python `concurrent.futures`).

### Execution Groups

After **Stage 1** (sequence retrieval) completes, four branches fan out simultaneously:

| Group | Branch | Steps | Dependency |
|---|---|---|---|
| **1** | Alignment & Structure | MAFFT → CIAlign → AlphaFold consensus | Only needs sequences |
| **2** | Clustering | k-mer vectorization → UMAP → HDBSCAN → stability QC | Only needs sequences |
| **3** | Motif Analysis | JASPAR BED check → bedtools/FIMO overlap | Needs sequences + JASPAR BED |
| **4** | Expression Collation | Parse RNA-seq / scRNA-seq TPM matrices | Only needs sequences |

### Join Barriers

Two synchronization points exist before gRNA generation:

- **Cluster expression analysis** — waits for Group 2 (clustering) **and** Group 4 (expression collation)
- **Motif expression analysis** — waits for Group 3 (motif overlap) **and** Group 4 (expression collation)

After both analyses complete, **gRNA generation** fans out again to two fully independent sub-branches:

- Off-target scoring (Cas-OFFinder / bowtie)
- gRNA stability / thermodynamic QC

### Critical Path

```
Get Sequences → MAFFT → CIAlign → AlphaFold
```

AlphaFold structural prediction is the dominant wall-clock bottleneck. All other branches should complete well before it finishes, meaning the pipeline's total runtime is effectively gated by Branch A.

### Recommended Scheduler

```python
# Conceptual concurrent.futures DAG
with ThreadPoolExecutor() as pool:
    seqs   = pool.submit(get_sequences, coords)          # Stage 1

    branch_a = pool.submit(run_alignment_structure, seqs)  # ─┐
    branch_b = pool.submit(run_clustering,          seqs)  #  ├ parallel fan-out
    branch_c = pool.submit(run_motif_analysis,      seqs)  #  │
    branch_d = pool.submit(collate_expression,      seqs)  # ─┘

    # join barriers
    cluster_expr = pool.submit(analyze_clusters, branch_b, branch_d)
    motif_expr   = pool.submit(analyze_motifs,   branch_c, branch_d)

    grnas = pool.submit(generate_grnas, cluster_expr, motif_expr)

    # final parallel QC
    pool.submit(offtarget_scoring,  grnas)
    pool.submit(stability_scoring,  grnas)
```

For HPC use, submit each group as a separate job array and use scheduler dependencies (`#BSUB -w done(jobA) && done(jobB)` for LSF, `--dependency=afterok:` for Slurm) to enforce the join barriers automatically.

## Script Overview

| Script | GAMECA step | Input → Output |
|---|---|---|
| `ui.py` | launcher | interactive menu + results viewer |
| `te_prep.py` | Prepare | UCSC rmsk → `sequences.csv` |
| `te_clustering.py` | **C** Clustering | `sequences.csv` → `clustered.csv` + plots |
| `te_alignment.py` | **A** Alignment | `clustered.csv` → alignment FASTAs + CIAlign plots |
| `te_motif.py` | **M** Motif | `clustered.csv` → `all_overlaps.tsv` + enrichment CSVs |
| `te_go.py` | **G** Gene/GO | `enrichment_results/` → annotated CSVs + GO plots |
| `te_expression.py` | **E** Expression | `clustered.csv` → boxplot PNGs + stats CSV |
| `te_enrichment.py` | M+G+E orchestrator | chains motif → GO → expression |
| `query.py` | core orchestrator | chains prep → clustering → alignment → primers |
| `te_genome.py` | shared library | `GenomeCache` class (indexed FASTA access) |
| `te_primers.py` | shared library | k-mer primer design + genome specificity search |
| `hpc_client.py` | HPC client | SSH → LSF/Slurm job submission |

---

## Step-by-step Data Flow

Each script has a `--input` / `--out-dir` interface. Outputs of one step become
inputs of the next.

| Step | Script | Key input | Key output |
|---|---|---|---|
| Prepare | `te_prep.py` | genome build + family name | `{family}_sequences.csv` |
| Cluster | `te_clustering.py` | `sequences.csv` (needs `Seq`) | `clustered.csv` + HTML plot |
| Align | `te_alignment.py` | `clustered.csv` (needs `Seq` + `Cluster`) | FASTA alignments, CIAlign PNGs |
| Motif | `te_motif.py` | `clustered.csv` (needs coords + `Cluster`) | `all_overlaps.tsv`, per-cluster enrichment CSVs |
| GO | `te_go.py` | `enrichment_results/` directory | `gene_functions.csv`, annotated enrichment CSVs |
| Expression | `te_expression.py` | `clustered.csv` (needs `Cluster` + numeric cols) | boxplot PNGs, `expression_stats.csv` |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Bioinformatics tools (best installed via conda):

```bash
conda install -c bioconda mafft pysam pybedtools
pip install cialign
```

### 2. Interactive launcher

```bash
python ui.py
```

The launcher walks you through each step with prompts. It also provides a
workflow overview (`--help-workflow`) and can display a results summary for any
output directory (`--results <dir>`).

### 3. Step-by-step (command line)

**Step 1 — Prepare data**
```bash
python te_prep.py --build hg38 --family HERVK --out-dir ./hervk_data
```

**Step 2 — Core analysis**
```bash
python query.py \
  --family HERVK \
  --genome /path/to/hg38.fa \
  --out-dir ./results
```

**Step 3 — Enrichment**
```bash
python te_enrichment.py \
  --input ./results/clustered_data.csv \
  --build hg38 \
  --family HERVK \
  --out-dir ./results
# JASPAR BED auto-downloaded if not supplied
```

### 4. HPC cluster (LSF or Slurm)

```bash
python hpc_client.py
```

The client detects whether the remote cluster runs **LSF** (`bsub/bjobs/bkill`) or
**Slurm** (`sbatch/squeue/scancel/srun`) and uses the appropriate commands
automatically.  Menu options include te_prep, interactive run, batch submission,
job monitoring, and result retrieval.

---

## Environment Variables

All paths are configurable — no hardcoded site-specific locations.

| Variable | Description |
|---|---|
| `HG38_FA` | Path to hg38 FASTA (enables fast local extraction) |
| `MM10_FA` | Path to mm10 FASTA |
| `TE_BASE_DIR` | Default working directory for te_prep output |
| `TE_RMSK_DIR` | Directory where rmsk.txt.gz files are cached |
| `TE_JASPAR_HG38` | Pre-downloaded JASPAR BED for hg38 |
| `TE_JASPAR_HG19` | Pre-downloaded JASPAR BED for hg19 |
| `TE_JASPAR_MM10` | Pre-downloaded JASPAR BED for mm10 |
| `TE_JASPAR_MM39` | Pre-downloaded JASPAR BED for mm39 |

---

## te_prep.py

Downloads RepeatMasker annotations from UCSC and extracts sequences.

```
python te_prep.py --build hg38 --family HERVK [options]

Options:
  --build BUILD         Genome build (hg38, hg19, mm10, mm39)
  --family FAMILY       TE family name to extract
  --out-dir DIR         Output directory
  --genome-fa FA        Path to local genome FASTA (uses pysam for speed)
  --download BUILD      Download rmsk for BUILD and exit
  --list-families       List all TE families in downloaded rmsk table
  --search TERM         Search for families matching TERM
```

Output CSV has columns: `chr, start, stop, TE_name, Seq` (plus optional expression columns).

---

## query.py

Core pipeline: sequence fetch → clustering → alignment → primers.

```
python query.py --family HERVK [options]

Key options:
  --family FAMILY         TE family name
  --genome FA             Genome FASTA for local extraction
  --out-dir DIR           Output directory
  --kmer K                k-mer size [18]
  --min-sequences N       Minimum sequences for clustering [10]
  --skip-alignment        Skip MAFFT / CIAlign
  --skip-primers          Skip primer design
```

---

## te_motif.py

JASPAR motif overlap + Fisher's exact test per cluster.

```
python te_motif.py --input clustered.csv --build hg38 [options]

Key options:
  --input FILE         Clustered CSV (from te_clustering.py)
  --build BUILD        Genome build for JASPAR download
  --out-dir DIR        Output directory
  --jaspar-bed FILE    Pre-downloaded JASPAR BED (auto-downloaded if omitted)
  --jaspar-dir DIR     Directory to cache downloaded JASPAR files
  --p-threshold FLOAT  Fisher p-value cutoff (default: 0.05)
  --force              Re-run even if overlap file exists
```

### JASPAR BED resolution

If `--jaspar-bed` is not supplied, looks in order:

1. `TE_JASPAR_<BUILD>` environment variable
2. `<jaspar-dir>/JASPAR2024_<build>.sorted.bed.gz`
3. Automatic download from `jaspar.elixir.no`

---

## te_go.py

GO annotation for enriched TF motifs via mygene.info API.

```
python te_go.py --enrichment-dir ./results/enrichment_results --build hg38 [options]

Key options:
  --enrichment-dir DIR  Directory with cluster_N_enrichment.csv files
  --build BUILD         Genome build (sets mygene species)
  --out-dir DIR         Output directory
  --clustered-csv FILE  Clustered CSV — enables per-strand GO bar charts
  --p-threshold FLOAT   Load only motifs below this p-value (default: 0.05)
  --top-motifs N        Motifs per cluster sent to mygene.info (default: 30)
  --force               Re-query even if gene_functions.csv exists
```

---

## te_expression.py

Per-cluster expression boxplots. Auto-detects numeric columns as expression data.

```
python te_expression.py --input clustered.csv [options]

Key options:
  --input FILE           Clustered CSV (from te_clustering.py)
  --out-dir DIR          Output directory
  --stage-cols COL ...   Explicit expression column names
  --stage-labels LBL ... Display labels for --stage-cols
  --no-log1p             Disable log1p transform
  --force                Re-run even if outputs exist
```

---

## te_enrichment.py

Orchestrator that chains te_motif → te_go → te_expression in one command.

```
python te_enrichment.py --input clustered.csv --build hg38 [options]

Key options:
  --input FILE         Clustered CSV
  --build BUILD        Genome build
  --out-dir DIR        Output directory
  --jaspar-bed FILE    JASPAR BED path (auto-downloaded if omitted)
  --skip-motif         Skip te_motif step
  --skip-go            Skip te_go step
  --skip-expression    Skip te_expression step
  --only STEP          Run only: motif | go | expression
  --force              Re-run all steps
  --force-steps STEP … Re-run specific steps
  --status             Print checkpoint status without running
```

---

## hpc_client.py

SSH-based client for LSF/Slurm HPC clusters.

```bash
python hpc_client.py [-H hostname] [-u username] [-o output_dir]
```

### Scheduler auto-detection

After connecting, `hpc_client.py` runs `command -v bsub` and `command -v sbatch`
to detect the available scheduler.  All subsequent job scripts, submission commands,
status queries, and cancellations use the appropriate syntax automatically.

### Configurable parameters

Use option `[1] Configure parameters` in the menu. Key parameters:

| Parameter | Description |
|---|---|
| `FAMILY_NAME` | TE family to analyse |
| `all_te_file` | Input CSV path on cluster |
| `HG38_FA` | Genome FASTA path on cluster |
| `QUEUE` | Scheduler queue / partition |
| `MEM_MB` | Memory request in MB |
| `CPUS` | CPU core count |
| `WALLTIME` | Job walltime (HH:MM or HH:MM:SS) |
| `MODULES` | Space-separated module names to load |

---

## Output Structure

```
{out_dir}/
├── clustered_data.csv              # Sequences with cluster assignments
├── clustering_coordinates.csv      # UMAP / PCA / t-SNE embeddings
├── clustering_visualization.html   # Interactive cluster plot
│
├── cluster_alignments/
│   ├── cluster_N_aligned.fa        # Per-cluster MAFFT alignment
│   ├── cluster_N_consensus.fa      # Per-cluster consensus
│   └── cluster_consensus_summary.csv
│
├── cialign_plots/
│   ├── index.html                  # Open this to browse all plots
│   └── *.png                       # Input / output / markup plots
│
├── cleaned_consensus/
│   └── *_cleaned_consensus.fa      # CIAlign-filtered consensuses
│
├── 04_alignments/
│   └── alignment_stats.txt
│
├── 06_primers/
│   ├── selected_primers_summary.csv
│   ├── cluster_top_primers.csv
│   └── primer_genome_hits_summary.csv
│
├── 07_visualizations/
│   └── index.html                  # Main interactive dashboard
│
├── motif_enrichment_results.csv    # JASPAR Fisher enrichment
├── go_annotation_results.csv       # GO terms (if --go used)
└── enrichment_checkpoints.json     # Step completion tracking
```

---

## Input CSV Format

The core pipeline accepts any CSV with:

| Column | Description |
|---|---|
| `chr` | Chromosome (`chr1`, `chrX`, …) |
| `start` | Genomic start (0-based) |
| `stop` | Genomic end |
| `TE_name` | TE identifier (used for family filtering) |
| `Seq` | DNA sequence (added by te_prep or query.py) |
| `strand` | `+` or `-` (optional) |
| Expression columns | Numeric — any column not in the above list |

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `MAFFT not found` | Not installed or not in PATH | `conda install -c bioconda mafft` |
| `CIAlign not found` | Not installed | `pip install cialign` |
| `pysam import error` | Not installed / platform issue | `conda install -c bioconda pysam` |
| `pybedtools import error` | Requires bedtools binary | `conda install -c bioconda pybedtools` |
| `JASPAR download failed` | No internet access | Supply `--jaspar-bed` or set `TE_JASPAR_<BUILD>` |
| `All sequences → cluster -1` | Dataset too small / too similar | Pipeline auto-assigns to cluster 0 |
| Scheduler not detected | Both bsub and sbatch absent | Set `scheduler` param manually via menu option 1 |

---

## Citations

- **MAFFT**: Katoh & Standley (2013) *Mol Biol Evol* 30:772-780
- **HDBSCAN**: McInnes et al. (2017) *JOSS* 2(11):205
- **UMAP**: McInnes et al. (2018) arXiv:1802.03426
- **CIAlign**: Tumescheit et al. (2022) *PeerJ* 10:e12983
- **JASPAR 2024**: Castro-Mondragon et al. (2022) *Nucleic Acids Res* 50:D165-D173

---

## License

[Add your license here]
