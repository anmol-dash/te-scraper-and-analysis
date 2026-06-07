#!/usr/bin/env bash
# run-container.sh — one-command GAMECA containerized pipeline (#9)
#
# Builds (if needed) and runs the GAMECA Apptainer/Singularity image against a
# YAML config, bind-mounting the repo + your data. Falls back to Docker if no
# Apptainer/Singularity is available.
#
# Usage:
#   scripts/run-container.sh gameca.yaml [extra gameca_pipeline.py args...]
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-gameca.yaml}"; shift || true
SIF="$ROOT/gameca.sif"

if command -v apptainer >/dev/null 2>&1; then RUNTIME=apptainer
elif command -v singularity >/dev/null 2>&1; then RUNTIME=singularity
else RUNTIME=""; fi

if [ -n "$RUNTIME" ]; then
    if [ ! -f "$SIF" ]; then
        echo "[container] Building $SIF with $RUNTIME (one-time)…"
        "$RUNTIME" build "$SIF" "$ROOT/gameca.def"
    fi
    echo "[container] Running pipeline via $RUNTIME…"
    exec "$RUNTIME" run \
        -B "$ROOT":/opt/gameca/code \
        --pwd /opt/gameca/code \
        "$SIF" --config "$CONFIG" "$@"
elif command -v docker >/dev/null 2>&1; then
    echo "[container] No Apptainer/Singularity; using Docker…"
    docker build -t gameca:latest "$ROOT"
    exec docker run --rm -v "$ROOT":/app -w /app gameca:latest \
        python gameca_pipeline.py --config "$CONFIG" "$@"
else
    echo "ERROR: no apptainer/singularity/docker found on PATH." >&2
    exit 1
fi
