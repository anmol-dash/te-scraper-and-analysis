# GAMECA — Gene Alignment, Motif, Expression & Clustering Analysis

GAMECA is a modular transposable-element (TE) analysis pipeline — prepare → cluster → align → motif → GO → expression → primers — with integrated **LSF** and **Slurm** HPC support. This repository ships the Python tooling plus a **Tauri** desktop shell that wraps the workflow in a native UI and communicates with a bundled Python sidecar over newline-delimited JSON (NDJSON) IPC.

Current version: **v0.4.44**

## Download

Installers are published on GitHub Releases:

- **Latest release:** https://github.com/anmol-dash/te-scraper-and-analysis/releases
- macOS: `GAMECA_<version>_universal.dmg` — a single universal build covering both Apple Silicon and Intel.
- Windows: `GAMECA_<version>_x64-setup.exe`.

### macOS: "Apple could not verify GAMECA is free of malware"

Expected on first launch. The installers are unsigned — the release pipeline has no Apple
Developer certificate — so macOS quarantines anything downloaded through a browser. The app
is fine; Gatekeeper just has no signature to check. To open it:

1. Drag **GAMECA** from the `.dmg` into Applications.
2. **Right-click** (or Control-click) GAMECA in Applications and choose **Open**.
3. Click **Open** in the dialog.

Only needed once — macOS remembers the exception. Double-clicking without this first step will
keep showing the warning with no way past it.

Alternatively, download with the GitHub CLI, which doesn't set the quarantine flag at all:

```bash
gh release download --repo anmol-dash/te-scraper-and-analysis --pattern "*.dmg"
```

## Features

| Feature | Details |
|---------|---------|
| **Full TE pipeline** | k-mer → SVD → UMAP/t-SNE → HDBSCAN clustering; MAFFT alignment; JASPAR + Fisher motif enrichment; GO annotation; expression boxplots; k-mer primer design |
| **HPC integration** | SSH client auto-detects LSF vs Slurm; uploads scripts, submits batch jobs, streams live output, retrieves results |
| **Annotation sources** | **RMSK** (UCSC RepeatMasker) and **Dfam** (REST API — no bulk download required) for TE locus coordinates |
| **Local mode** | Runs entirely on your Mac; sequences fetched from UCSC API when no genome FASTA is available |
| **Cython acceleration** | `te_fast.pyx` speeds up k-mer matrix construction, GC content, and length calculations ~3–5× when compiled |
| **Desktop shell** | Tauri + React frontend; Python sidecar streams logs/progress/cancellation over IPC |
| **Auto-update** | On launch, scripts update silently from the latest commit; app update prompts when a new GitHub Release is published |
| **File browser** | Remote filesystem browser with sort-by-name/date, CSV table viewer, inline Plotly HTML visualisation |
| **Smart GO step** | GO annotation only needs a family name and species — auto-detects enrichment results and chains clustering → motif → GO if they are absent |

## Quick start

1. Download and install `GAMECA_<version>_universal.dmg` from Releases (see [Download](#download) if macOS blocks it).
2. Launch GAMECA — first launch installs Python dependencies automatically (~2–3 min).
3. Connect to your HPC cluster (SSH credentials) or run locally with **Local Pipeline**.
4. Choose a pipeline step from the sidebar and hit **Run on Cluster**.

### Locus count (before running the full pipeline)

Use **RMSK locus count** or **Dfam locus count** in the *Data prep* step group to verify your family name and get the number of loci before committing to a full job.

- RMSK: downloads `rmsk_<assembly>.txt.gz` (~150 MB, cached) to the HPC work dir on first run.
- Dfam: hits the Dfam REST API per-family — no bulk file needed, result in seconds.

## Pipeline steps

```
Data prep
  RMSK locus count      – verify family name + locus count (RepeatMasker)
  Dfam locus count      – same via Dfam REST API
  Download + extract    – download rmsk/Dfam coords, extract sequences (UCSC API fallback)

Core analysis
  Clustering            – k-mer · SVD · UMAP · HDBSCAN  → cluster_summary.csv
  Alignment             – MAFFT · CIAlign (trimAl fallback) · consensus

Enrichment
  Motif                 – JASPAR + Fisher + HOMER
  GO annotation         – family + species only; chains clustering → motif → GO automatically
                          if enrichment results are not yet present
  Expression            – boxplots per cluster (column names + figure order
                          must be designated — see "Designating expression
                          columns" below)
  Full enrichment       – motif + GO + expression combined

Primer design
  Primers               – k-mer candidate generation; genome-wide specificity via GenomeCache;
                          per-cluster top-primer summary

Full pipeline
  Batch job             – bsub / sbatch with all steps

Results
  Retrieve              – rsync results to local machine
  File browser          – remote filesystem with sort, CSV viewer, HTML plots
```

### Designating expression columns

If the input data has candidate expression columns (any numeric per-sample
column not already used for coordinates/scores/clustering), `query.py`
requires you to name them and set their figure order before the pipeline
proceeds — it never silently guesses:

```bash
python query.py --input clustered.csv --family MT2_Mm --output results \
    --expr-cols pronuc twocell fourcell eightcell morulacell \
    --expr-labels "Pro-nucleus" "2-cell" "4-cell" "8-cell" "Morula"
```

- `--expr-cols` is required for non-interactive/batch runs (e.g. `bsub`/`sbatch`
  jobs, which have no TTY) whenever candidate columns are present; the pipeline
  exits early with the detected candidate list if it's missing.
- Running interactively without `--expr-cols` prompts for the column order
  instead of exiting.
- `--expr-labels` is optional — omit it to reuse the column names as labels.
- Families with no candidate expression columns are unaffected — no prompt or
  flag is required.

The resolved order is reused everywhere downstream: the dashboard expression
heatmap, `te_expression.py`'s stage-profile/primer-expression figures, and the
primer design summary tables.

### Design choice: CIAlign failure/timeout fallback

CIAlign's own cleaning + plotting pass can fail outright or hang (capped at a
1-hour timeout) on very large alignments — massive families like `AluSx1` can
have tens of thousands of loci per cluster, and CIAlign wasn't built for that
scale. Rather than silently ending up with no alignment visualization at all,
`te_alignment.run_cialign()` falls back once, automatically:

1. If CIAlign fails (non-zero exit) or times out (>3600s), clean the same
   alignment with **trimAl** (`-automated1` heuristic) instead — trimAl's
   column-trimming is far cheaper than CIAlign's own cleaning step and
   finishes in minutes even on huge alignments.
2. Retry CIAlign once on the trimAl-cleaned alignment, this time capped at
   **1,000 sequences** for display (down from the normal 25,000-sequence
   subsample cap) so the retry itself can't time out the same way.

If trimAl isn't installed, or the trimAl-cleaned retry also fails, the stage
logs the failure and moves on (alignment/consensus output is unaffected —
only the CIAlign visualization plots are skipped).

## Copy-resolved analysis modules

Beyond the core pipeline, a family of standalone `run_*.py` modules take the clustered,
per-locus output and characterise each copy. Each writes its own figures, a CSV/JSON of
per-copy results, and a `*_measured_values.tex` macro file, and each has a matching HPC
submitter. `run_stage11_all.py` runs them together for one family.

| Module | What it produces |
|--------|------------------|
| `run_phylo_analysis.py` | Per-cluster MAFFT alignment, majority-rule consensus, Kimura molecular-clock age, ORF-integrity call, and a nominated master element |
| `run_grna_offtarget.py` / `run_grna_analysis.py` | Allele-aware CRISPRa/i guides scored against the full copy landscape; coverage-versus-off-target Pareto frontier |
| `run_transduction.py` | 3′ transduction lineages (shared downstream-tail k-mers) and poly-A signal detection |
| `run_antisense_promoter.py` | Antisense / bidirectional promoter scan of each 5′ region on both strands |
| `run_divergence.py` | CpG-corrected Kimura divergence landscape per family/cluster |
| `run_ltr_struct.py` | LTR structural classification (full-length / solo-LTR / internal-only / truncated) |
| `run_ctcf_tad.py`, `run_epigenetic_overlay.py` | CTCF-site / TAD-boundary proximity and epigenetic/regulatory-track overlay |
| `run_ortholog_insertion.py`, `run_multiassembly_liftover.py` | Orthologous-insertion calling and liftover across `hg38` / T2T-CHM13 / `mm10` / `mm39` |
| `run_fold_prediction.py`, `run_subfamily.py` | ColabFold structure prediction of consensus ORFs; subfamily dendrogram of cluster consensuses |
| `make_report_figures.py` | Cross-family LINE/SINE/LTR batch analysis (clusters, expression, DE, per-family report figures) |
| `run_cluster_search.py` | Search for an N-cluster configuration at minimal noise; `--regen-only` rebuilds expression/DE and figures from an existing clustered CSV |

## Supported assemblies

| Species | Assemblies |
|---------|------------|
| Human   | `hg38`, `hg19` |
| Mouse   | `mm10`, `mm39` |

## Repository layout

```
backend/                  Python IPC sidecar (PyInstaller → pytool binary)
  yourtool/
    cli.py                CLI handlers: hpc-connect, hpc-upload, hpc-run, hpc-batch-submit …
  tests/
    test_cli_hpc_files.py HPC file-sync integration tests
    test_ipc.py           IPC protocol tests
frontend/                 React + Tailwind UI (Tauri webview)
src-tauri/                Rust shell

hpc_client.py             SSH/bsub/sbatch client; auto-detects LSF vs Slurm;
                          venv created in remote_work_dir (writable project directory)
query.py                  Main pipeline driver (~2400 lines); stages 1–10 with checkpoints
ui.py                     Terminal UI: HPC menu, rmsk/Dfam query, batch submit

te_prep.py                TE coordinate fetching (RMSK + Dfam REST API) + sequence extraction
te_clustering.py          k-mer → SVD → UMAP/t-SNE → HDBSCAN
te_alignment.py           MAFFT + CIAlign + majority-vote consensus
te_motif.py               JASPAR bedtools intersect + Fisher enrichment + optional HOMER
te_go.py                  GO annotation via mygene.info
te_expression.py          Expression boxplots, stage profiles, chromosomal heatmaps
te_enrichment.py          Full enrichment orchestrator (motif + GO + expression)
te_primers.py             k-mer primer design with genome-wide specificity checking
te_genome.py              Shared genome utilities: GenomeCache (in-memory FASTA),
                          primer search, reverse-complement helpers
te_fast.pyx               Cython-accelerated hot paths (k-mer matrix, GC, lengths)
te_fast.c                 Generated C source (committed so the cluster doesn't need Cython)
setup_cython.py           Build script: python setup_cython.py build_ext --inplace

run_*.py                  Copy-resolved analysis modules run after the core pipeline
                          (see "Copy-resolved analysis modules" below): phylogeny,
                          allele-aware gRNA, transduction, antisense, divergence,
                          LTR structure, CTCF/TAD, epigenetic, ortholog/liftover, etc.
run_stage11_all.py        Orchestrates the copy-resolved modules for one family
make_report_figures.py    Cross-family LINE/SINE/LTR batch analysis; emits the per-family
                          report data figures (clusters, expression, DE, divergence)
generate_latex_report.py  Builds the per-family LaTeX report (exercised by test_gameca.sh)

requirements.txt          Python dependencies (pinned ranges)

test_gameca.sh            Comprehensive end-to-end pipeline test (multi-stage, cluster or local)
backend/tests/            IPC protocol + HPC file-sync integration tests
```

The pipeline is described in a short methods manuscript under
[`terra_report/`](terra_report/) (`terra_report.tex`, compiled `terra_report.pdf`),
which walks one prototype family (`L1Md_T`) from RepeatMasker annotation to ordered
primers and CRISPRa/i guides. See [`terra_report/README.md`](terra_report/README.md).

## System requirements

| Role | Requirement |
|------|-------------|
| Desktop app | macOS 12+ (arm64 or x86_64) |
| Python tooling | Python 3.11+ (auto-installed in venv on first launch) |
| HPC cluster | LSF (`bsub`) or Slurm (`sbatch`); SSH access from your Mac |
| Bioinformatics | MAFFT, bedtools, CIAlign on the cluster; trimAl recommended (CIAlign failure/timeout fallback — see Design choices below); see `requirements.txt` |

## Building from source

```bash
# Prerequisites: Node 20+, pnpm, Rust stable, Xcode CLT (macOS)
pnpm install

# Build the Python sidecar first
cd backend
pip install pyinstaller
pyinstaller pyinstaller.spec --noconfirm
cp dist/pytool ../src-tauri/binaries/pytool-aarch64-apple-darwin
cd ..

# Build the Tauri app
pnpm tauri build
# → src-tauri/target/release/bundle/dmg/GAMECA_<version>_aarch64.dmg
```

> **Note:** Always rebuild PyInstaller before `pnpm tauri build` if you changed any Python file. The sidecar binary must be copied into `src-tauri/binaries/` manually.

### Optional: compile the Cython extension

```bash
python setup_cython.py build_ext --inplace
```

If `te_fast.cpython-*.so` is present it is used automatically; the pipeline falls back to pure Python otherwise. The pre-generated `te_fast.c` is committed so cluster nodes only need a C compiler (`gcc`), not Cython.

## Auto-update behaviour

- **Scripts** (`te_prep.py`, `hpc_client.py`, `ui.py`, etc.): updated silently from the latest commit on `main` every time the app launches.
- **App binary**: when a new GitHub Release is published with a version tag higher than the running version, users see an in-app prompt to download and install the new DMG.

## Nextflow / HPC pipeline

Besides the desktop app, GAMECA ships a **Nextflow DSL2** workflow that runs the
full pipeline on an LSF/Slurm cluster with real per-module parallelism, `-resume`,
and a reproducible container. This is the recommended path for running GAMECA as a
shared, multi-user tool on HPC.

```bash
# on the cluster: build the container image (pulls the prebuilt GHCR image, no root)
REMOTE_IMAGE=docker://ghcr.io/anmol-dash/gameca:latest ./build_sif.sh

# run a family through Nextflow + the container
nextflow run nextflow/main.nf --family HERVK9 --assembly hg38 \
    --genome_fasta /path/to/hg38.fa --container_sif "$PWD/gameca.sif" \
    --outdir results -profile lsf,singularity
```

See **[nextflow/README.md](nextflow/README.md)** for the full setup, profiles,
samplesheet format, and how to embed GAMECA as a subworkflow in another pipeline.

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | IPC contracts, sidecar lifecycle, event flow |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Sidecar, PyInstaller, macOS gatekeeper |
| [docs/ADDING_A_COMMAND.md](docs/ADDING_A_COMMAND.md) | End-to-end: Python CLI → IPC → UI |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Dev setup, tests, PR checklist |

## License

Released under the [MIT License](LICENSE).
