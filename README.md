# GAMECA — Gene Alignment, Motif, Expression & Clustering Analysis

GAMECA is a modular transposable-element (TE) analysis pipeline — prepare → cluster → align → motif → GO → expression — with integrated **LSF** and **Slurm** HPC support. This repository ships the Python tooling plus a **Tauri** desktop shell that wraps the workflow in a native UI and communicates with a bundled Python sidecar over newline-delimited JSON (NDJSON) IPC.

Current version: **v0.4.8**

## Download

Installers are published on GitHub Releases:

- **Latest release:** https://github.com/anmol-dash/te-scraper-and-analysis/releases
- Pick the `.dmg` (macOS) artifact matching your architecture (`aarch64` for Apple Silicon, `x86_64` for Intel).

## Features

| Feature | Details |
|---------|---------|
| **Full TE pipeline** | k-mer → SVD → UMAP/t-SNE → HDBSCAN clustering; MAFFT alignment; JASPAR + Fisher motif enrichment; GO annotation; expression boxplots |
| **HPC integration** | SSH client auto-detects LSF vs Slurm; uploads scripts, submits batch jobs, streams live output, retrieves results |
| **Annotation sources** | **RMSK** (UCSC RepeatMasker) and **Dfam** (REST API — no bulk download required) for TE locus coordinates |
| **Local mode** | Runs entirely on your Mac; sequences fetched from UCSC API when no genome FASTA is available |
| **Desktop shell** | Tauri + React frontend; Python sidecar streams logs/progress/cancellation over IPC |
| **Auto-update** | On launch, scripts update silently from the latest commit; app update prompts when a new GitHub Release is published |
| **File browser** | Remote filesystem browser with sort-by-name/date, CSV table viewer, inline Plotly HTML visualisation |

## Quick start

1. Download and install `GAMECA_<version>_aarch64.dmg` from Releases.
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
  Alignment             – MAFFT · CIAlign · consensus

Enrichment
  Motif                 – JASPAR + Fisher + HOMER
  GO annotation         – mygene.info
  Expression            – boxplots per cluster
  Full enrichment       – motif + GO + expression combined

Full pipeline
  Batch job             – bsub / sbatch with all steps

Results
  Retrieve              – rsync results to local machine
  File browser          – remote filesystem with sort, CSV viewer, HTML plots
```

## Repository layout

```
backend/            Python IPC sidecar (PyInstaller → pytool binary)
  main.py           NDJSON server: routes commands, streams logs, manages setup/updates
  yourtool/cli.py   CLI handlers: hpc-connect, hpc-upload, hpc-run, hpc-batch-submit …
frontend/           React + Tailwind UI (Tauri webview)
src-tauri/          Rust shell
hpc_client.py       SSH/bsub/sbatch client
te_prep.py          TE coordinate fetching (RMSK + Dfam REST API) + sequence extraction
te_clustering.py    k-mer → SVD → UMAP/t-SNE → HDBSCAN
te_alignment.py     MAFFT + CIAlign consensus
te_motif.py         JASPAR motif enrichment
te_go.py            GO annotation
te_expression.py    Expression analysis
te_enrichment.py    Full enrichment orchestrator
ui.py               Terminal UI (HPC menu, rmsk/Dfam query, batch submit)
query.py            Main pipeline driver (~2400 lines)
```

## System requirements

| Role | Requirement |
|------|-------------|
| Desktop app | macOS 12+ (arm64 or x86_64) |
| Python tooling | Python 3.11+ (auto-installed in venv on first launch) |
| HPC cluster | LSF (`bsub`) or Slurm (`sbatch`); SSH access from your Mac |
| Bioinformatics | MAFFT, bedtools on the cluster; see `requirements.txt` |

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

## Auto-update behaviour

- **Scripts** (`te_prep.py`, `hpc_client.py`, `ui.py`, etc.): updated silently from the latest commit on `main` every time the app launches.
- **App binary**: when a new GitHub Release is published with a version tag higher than the running version, users see an in-app prompt to download and install the new DMG.

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | IPC contracts, sidecar lifecycle, event flow |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Sidecar, PyInstaller, macOS gatekeeper |
| [docs/ADDING_A_COMMAND.md](docs/ADDING_A_COMMAND.md) | End-to-end: Python CLI → IPC → UI |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Dev setup, tests, PR checklist |
