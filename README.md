# TE Analysis Pipeline

A comprehensive pipeline for analyzing Transposable Elements (TEs), performing sequence clustering, multiple sequence alignment, and primer design for experimental validation.

## Overview

This pipeline:
- Fetches TE sequences from the UCSC Genome Browser API (hg38)
- Clusters sequences using k-mer analysis with UMAP/PCA/t-SNE dimensionality reduction
- Performs multiple sequence alignment using MAFFT
- Designs optimal primers with genome-wide specificity checking
- Generates interactive visualizations and comprehensive reports

## Features

- **Clustering Analysis**: HDBSCAN clustering on k-mer frequency vectors with multiple embedding methods
- **Primer Design**: Two optimization strategies (coverage-first, expression-first) with hg38 specificity search
- **Alignment**: Global and per-cluster alignments with consensus sequence generation
- **Visualization**: Interactive Plotly dashboards and CIAlign alignment plots
- **HPC Support**: Interactive client for running analyses on remote HPC clusters

---

## Installation

### Prerequisites

- Python 3.8+
- MAFFT (for sequence alignment)
- Access to hg38 reference genome (for primer specificity search)

### Step 1: Clone or Download

```bash
git clone <repository-url>
cd utility_result
```

### Step 2: Install Python Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or install manually
pip install pandas numpy requests beautifulsoup4 plotly scikit-learn umap-learn hdbscan biopython paramiko
```

### Step 3: Install MAFFT (for alignments)

**macOS (Homebrew):**
```bash
brew install mafft
```

**Ubuntu/Debian:**
```bash
sudo apt-get install mafft
```

**Conda:**
```bash
conda install -c bioconda mafft
```

### Step 4: Install CIAlign (optional, for alignment visualization)

```bash
pip install cialign
```

### Step 5: Verify Installation

```bash
# Check Python dependencies
python -c "import pandas, numpy, plotly, sklearn, umap, hdbscan; print('All dependencies OK')"

# Check MAFFT
mafft --version

# Check CIAlign (optional)
CIAlign --version
```

---

## Input Files

### `all_te_file` (Required)

Main CSV file containing transposable element (TE) annotations with genomic coordinates and expression data. This file is the primary input for the entire pipeline.

#### Required Columns

| Column | Type | Description | Example | Used For |
|--------|------|-------------|---------|----------|
| `chr` | string | Chromosome name (must match hg38 naming) | `chr1`, `chr2`, `chrX`, `chrY` | Fetching sequences from UCSC API |
| `start` | integer | Genomic start position (0-based or 1-based) | `100000` | Fetching sequences from UCSC API |
| `stop` | integer | Genomic end position | `105000` | Fetching sequences from UCSC API |
| `TE_name` | string | TE identifier/name containing the family name | `HERVK9-int`, `LTR12C` | Filtering rows by FAMILY_NAME parameter |
| `strand` | string | Strand orientation | `+` or `-` | Primer genome hit analysis |

#### Optional Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `family` | string | TE family classification | `HERVK9`, `LTR` |
| Expression columns | numeric | RNA-seq read counts or normalized expression | `45.0`, `0`, `120.5` |

#### Expression Columns (Recommended)

Expression data enables primer optimization by expression level and comparative analysis across conditions.

**Naming Convention:** `{SampleID}_{Condition}_{Replicate}`

| Component | Description | Examples |
|-----------|-------------|----------|
| SampleID | Unique sample identifier | `A1`, `A2`, `B1`, `Sample01` |
| Condition | Experimental condition | `siCTRL`, `siRBMX`, `Rescue`, `WT`, `KO` |
| Replicate | Replicate number | `r1`, `r2`, `r3`, `rep1` |

**Example column names:**
- `A1_siCTRL_r1`, `A2_siCTRL_r2`, `A3_siCTRL_r3` (Control replicates)
- `B1_siRBMX_r1`, `B2_siRBMX_r2`, `B3_siRBMX_r3` (Treatment replicates)
- `C1_Rescue_r1`, `C2_Rescue_r2`, `C3_Rescue_r3` (Rescue replicates)

**How expression columns are detected:**
- All numeric columns except `start`, `stop`, `chr`, `Cluster` are treated as expression data
- Conditions are extracted by splitting column names on `_` and taking the second element
- Expression values are summed per row for primer ranking

#### Complete Example CSV

```csv
chr,start,stop,TE_name,family,strand,A1_siCTRL_r1,A2_siCTRL_r2,A3_siCTRL_r3,B1_siRBMX_r1,B2_siRBMX_r2,B3_siRBMX_r3,C1_Rescue_r1,C2_Rescue_r2,C3_Rescue_r3
chr1,100000,105000,HERVK9-int,HERVK9,+,45,52,38,120,115,108,62,58,55
chr1,250000,258500,HERVK9-int,HERVK9,-,12,15,10,45,42,48,20,18,22
chr2,500000,507500,HERVK9-int,HERVK9,+,0,2,1,8,6,9,3,2,4
chr3,750000,756000,LTR12C,LTR12,+,100,95,102,200,210,195,150,145,155
chr5,1000000,1008000,HERVK9-int,HERVK9,-,5,8,6,25,30,28,12,10,14
chrX,2000000,2007000,HERVK9-int,HERVK9,+,0,0,0,3,2,4,1,0,2
```

#### Column Requirements & Validation

| Requirement | Details |
|-------------|---------|
| **Header row** | First row must contain column names |
| **Delimiter** | Comma-separated (CSV format) |
| **Chromosome format** | Must match UCSC hg38 naming: `chr1`-`chr22`, `chrX`, `chrY`, `chrM` |
| **Coordinates** | Must be valid hg38 genomic positions; `stop` > `start` |
| **TE_name filtering** | Rows are included if `TE_name` contains `FAMILY_NAME` (case-insensitive) |
| **No missing coordinates** | `chr`, `start`, `stop` cannot be empty or NA |
| **Expression values** | Can be 0, integers, or floats; missing values treated as 0 |

#### How Each Column is Used

```
chr + start + stop  →  Fetch DNA sequence from UCSC Genome Browser API (hg38)
        ↓
    TE_name         →  Filter rows matching FAMILY_NAME parameter
        ↓
    strand          →  Report strand in primer genome hit analysis
        ↓
Expression columns  →  Calculate total expression per TE instance
                       Rank primers by expression coverage
                       Generate expression heatmaps and comparisons
                       Group by condition for differential analysis
```

#### Minimum Viable File (No Expression Data)

If you only have genomic coordinates without expression data:

```csv
chr,start,stop,TE_name,strand
chr1,100000,105000,HERVK9-int,+
chr1,250000,258500,HERVK9-int,-
chr2,500000,507500,HERVK9-int,+
```

**Note:** Without expression data, primer selection will be based solely on sequence coverage.

#### Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Wrong chromosome format | "Failed to fetch sequence" | Use `chr1` not `1` or `Chr1` |
| Coordinates out of range | "Failed to fetch sequence" | Verify coordinates are valid hg38 positions |
| TE_name doesn't match | "Found 0 instances" | Check that TE_name contains your FAMILY_NAME |
| Missing header | Column data misaligned | Ensure first row has column names |
| Tab-separated file | Parsing errors | Convert to comma-separated CSV |
| Extra whitespace | Matching failures | Trim whitespace from values |

#### Generating This File

This file is typically generated from:
- **RepeatMasker output** + RNA-seq quantification (featureCounts, TEtranscripts)
- **TElocal/TEtranscripts** output with genomic coordinates added
- **Custom TE annotation pipeline** combining coordinates with expression counts

---

### `te_counts` (Optional)

Summary CSV with TE family-level counts. This file is loaded but **not actively used** in the current pipeline. It may be useful for reference or future enhancements.

**Not required for analysis** - you can leave this parameter unset.

---

## Usage

### Option 1: HPC Client (Recommended for Remote Clusters)

Interactive client for running analyses on HPC systems.

```bash
python hpc_client.py
```

**Connection:**
```
Enter HPC connection details:
  Hostname: cluster.university.edu
  Port [22]:
  Username: myusername
  Password: ********
```

**Menu Options:**
1. **Configure parameters** - Set analysis parameters and input file paths
2. **Preview family count** - Check how many sequences match your target family
3. **Run full analysis** - Execute the pipeline on the HPC
4. **Retrieve results** - Download results to local machine
5. **Disconnect and exit**

**Required Parameters:**
- `FAMILY_NAME`: Target TE family to analyze (e.g., "HERVK9", "THE1A", "LTR12C")
- `all_te_file`: Path to input CSV on the HPC
- `HG38_FA`: Path to hg38.fa reference genome on the HPC

### Option 2: Direct Execution

For local runs or manual HPC submission.

1. **Edit configuration** in `query.py` (lines 32-40):
```python
FAMILY_NAME = "HERVK9"  # Target TE family
HG38_FA = "/path/to/hg38.fa"  # Reference genome
BASE_OUT_DIR = Path("collab_rna")  # Output directory
K = 18  # K-mer size
TOP_N_GLOBAL = 8  # Top global primers
TOP_N_CLUSTER = 5  # Top primers per cluster
MIN_SEQUENCES_FOR_CLUSTERING = 10  # Minimum for clustering
```

2. **Prepare your data** and load it before running:
```python
import pandas as pd
df = pd.read_csv("path/to/your/te_data.csv")
# Then run query.py
```

3. **Run the analysis:**
```bash
python query.py
```

---

## Output Structure

```
{family_name}/
├── 01_data/
│   ├── {family}_with_sequences.csv    # Raw data with fetched sequences
│   └── {family}_clustered.csv         # Data with cluster assignments
│
├── 02_statistics/
│   ├── overall_statistics.txt         # Global statistics
│   ├── cluster_comparison.txt         # Cluster comparison summary
│   ├── cluster_summary_table.csv      # Cluster metrics table
│   └── per_cluster/
│       └── cluster_{n}_statistics.txt # Per-cluster statistics
│
├── 03_clustering/
│   ├── clustering_visualization.html  # Interactive PCA/UMAP/t-SNE plot
│   └── clustering_skipped.txt         # (if clustering was skipped)
│
├── 04_alignments/
│   └── alignment_stats.txt            # Alignment statistics
│
├── 05_consensus/
│   └── (consensus sequences)
│
├── 06_primers/
│   ├── selected_primers_summary.csv   # Top global primers
│   ├── cluster_top5_primers.csv       # Top primers per cluster
│   ├── all_12mer_candidates_metrics.csv
│   ├── {primer}_genome_hits.csv       # Genome hits per primer
│   └── primer_genome_hits_summary.csv
│
├── 07_visualizations/
│   ├── index.html                     # Main dashboard
│   ├── cluster_distribution.html
│   ├── sequence_characteristics.html
│   ├── expression_heatmap.html
│   └── expression_comparison.html
│
├── cluster_alignments/
│   ├── cluster_{n}_seqs.fa
│   ├── cluster_{n}_aligned.fa
│   ├── cluster_{n}_consensus.fa
│   └── all_cluster_consensuses.fa
│
├── cialign_plots/
│   ├── index.html                     # CIAlign visualization index
│   ├── global_alignment_input.png
│   ├── global_alignment_output.png
│   └── cluster_{n}_alignment_*.png
│
├── cleaned_consensus/
│   ├── {family}_cleaned_consensus.fa
│   ├── cluster_{n}_cleaned_consensus.fa
│   └── all_clusters_cleaned_consensus.fa
│
├── {family}_seqs.fa                   # All sequences FASTA
├── {family}_aligned.fa                # Global alignment
├── {family}_consensus.fa              # Global consensus
└── README.txt                         # Auto-generated guide
```

---

## Key Output Files

| File | Description |
|------|-------------|
| `07_visualizations/index.html` | Interactive dashboard - **start here** |
| `cialign_plots/index.html` | Alignment visualization plots |
| `selected_primers_summary.csv` | Best primers for experimental use |
| `cluster_top5_primers.csv` | Cluster-specific primers |
| `primer_genome_hits_summary.csv` | Primer specificity (genome hit counts) |
| `{family}_consensus.fa` | Global consensus sequence |
| `cluster_alignments/all_cluster_consensuses.fa` | All cluster consensuses |

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FAMILY_NAME` | "HERVK9" | Target TE family name to filter |
| `HG38_FA` | - | Path to hg38 reference FASTA |
| `BASE_OUT_DIR` | "collab_rna" | Base output directory |
| `K` | 18 | K-mer size for encoding |
| `TOP_N_GLOBAL` | 8 | Number of top global primers |
| `TOP_N_CLUSTER` | 5 | Number of top primers per cluster |
| `MIN_SEQUENCES_FOR_CLUSTERING` | 10 | Minimum sequences for clustering |

---

## Supported TE Families

Pre-configured families (can analyze any family in your data):
- THE1A, MSTA1, HERV9-int, LTR1D, LTR12C
- LTR2752, MER52A, MER66D, GSAT, HSAT4
- HERVK9, LTR8B

---

## Troubleshooting

### UCSC API Errors
**Symptom:** "Failed to fetch sequence for row X: 'dna'"

**Solution:** Check your input coordinates are valid hg38 positions. The API may be temporarily unavailable - retry after a few minutes.

### No K-mers Found
**Symptom:** "No valid 12-mers found in sequences"

**Cause:** All sequences contain only N's (failed fetches) or sequences are too short.

**Solution:** Verify UCSC API is working and coordinates are correct.

### MAFFT Not Found
**Symptom:** "MAFFT not found"

**Solution:** Install MAFFT (see Installation section) and ensure it's in your PATH.

### SFTP Not Available (HPC)
**Symptom:** "SFTP not available, will use shell commands"

**This is normal** - the client will fall back to shell-based file transfer.

### Clustering Produces Only Noise (-1)
**Symptom:** All sequences assigned to cluster -1

**Cause:** Sequences are too similar or dataset is too small for HDBSCAN to find density clusters.

**Solution:** The pipeline automatically reassigns to Cluster 0 and continues analysis.

---

## Requirements File

Create `requirements.txt`:
```
pandas>=1.3.0
numpy>=1.20.0
requests>=2.25.0
beautifulsoup4>=4.9.0
plotly>=5.0.0
scikit-learn>=0.24.0
umap-learn>=0.5.0
hdbscan>=0.8.0
biopython>=1.79
paramiko>=2.7.0
```

---

## Citation

If you use this pipeline, please cite:
- MAFFT: Katoh & Standley (2013) Mol Biol Evol 30:772-780
- HDBSCAN: McInnes et al. (2017) JOSS 2(11):205
- UMAP: McInnes et al. (2018) arXiv:1802.03426
- CIAlign: Tumescheit et al. (2022) PeerJ 10:e12983

---

## License

[Add your license here]

## Contact

[Add contact information here]
