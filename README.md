# GAMECA — Gene Alignment, Motif, Expression & Clustering Analysis

GAMECA is a modular transposable-element analysis pipeline (prepare → cluster → align → motif → GO → expression) with optional **LSF** and **Slurm** HPC integration. This repository ships the Python tooling plus a **Tauri** desktop shell that embeds the workflow behind a native UI and talks to a bundled Python worker over newline-delimited JSON (NDJSON) IPC.

![Screenshot placeholder — replace with app window showing pipeline or dashboard](./docs/assets/screenshot-placeholder.png)

## Download

Installers and checksums are published on GitHub Releases:

- **Latest releases:** https://github.com/anmol-dash/te-scraper-and-analysis/releases  
- Pick the artifact for your OS/architecture from the release assets.

## Key features

- **DAG-friendly pipeline:** Steps read/write CSV checkpoints so stages can rerun independently or swap implementations.
- **HPC-aware:** SSH client (`hpc_client.py`) auto-detects LSF vs Slurm for batch submission and monitoring.
- **Interactive launcher & dashboards:** Terminal UI (`ui.py`) and optional local clustering/results dashboards.
- **Desktop shell:** Tauri wraps the webview UI, spawns a Python sidecar/worker, and streams logs/progress/cancellation over IPC (see [Architecture](docs/ARCHITECTURE.md)).
- **Built-in updater:** Release artifacts can be wired to Tauri’s updater (signatures + public key — see [Troubleshooting](docs/TROUBLESHOOTING.md)).

## System requirements

| Role | Requirement |
|------|-------------|
| **Desktop app** | OS-supported WebView stack per Tauri (varies by platform); bundled runtime included in installers where applicable. |
| **Python tooling** | **Python 3.11+** for scripts and IPC backend (`backend/`). |
| **Bioinformatics deps** | MAFFT, bedtools, optional conda packages — see `requirements.txt` and pipeline docs in-repo. |

Developer toolchain (building from source): **Node 20+**, **pnpm**, **Rust stable**, platform kits for Tauri (Linux webkit2gtk, macOS Xcode CLT, Windows VS Build Tools). See [CONTRIBUTING.md](CONTRIBUTING.md).

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Stack diagram, IPC contracts, sidecar lifecycle, event flow, state ownership |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Sidecar, PyInstaller, updater, macOS gatekeeper, SmartScreen |
| [docs/ADDING_A_COMMAND.md](docs/ADDING_A_COMMAND.md) | End-to-end recipe: Python CLI → Rust command → TS IPC → UI |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Local dev, tests, branches, commits, PR checklist |

Pipeline-oriented usage (Docker, CLI examples, scripts overview) remains available throughout the Python modules and historical README sections may be consolidated here over time; start from `ui.py --help` and `requirements.txt` for command-line workflows.
