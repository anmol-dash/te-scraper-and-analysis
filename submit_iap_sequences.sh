#!/usr/bin/env bash
#
# submit_iap_sequences.sh — bsub a single job that fetches sequences for ALL
# IAP repNames from rmsk (strand-aware) and writes one combined CSV.
#
# Run on the cluster LOGIN node:
#   chmod +x submit_iap_sequences.sh
#   ./submit_iap_sequences.sh                       # uses defaults below
#   OUT=/home/me/iap.csv ./submit_iap_sequences.sh  # override via env
#
# Internet note: with no GENOME_MM10 set, sequences are pulled from the UCSC API
# (needs outbound HTTPS from the compute node). Provide a local FASTA via
# GENOME_MM10 for fully offline extraction. Pre-cache rmsk to skip the download:
#   python te_prep.py --download mm10 --rmsk-dir "$RMSK_DIR"

set -uo pipefail

BUILD="${BUILD:-mm10}"
PATTERN="${PATTERN:-IAP}"
OUT="${OUT:-$PWD/iap_${BUILD}_sequences.csv}"
RMSK_DIR="${RMSK_DIR:-$HOME/te_analysis/rmsk}"
GENOME_MM10="${GENOME_MM10:-}"          # optional local FASTA -> offline extraction
QUEUE="${QUEUE:-normal}"
N_CORES="${N_CORES:-4}"
MEM_MB="${MEM_MB:-24000}"
WALL="${WALL:-12:00}"
PYTHON="${PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-$PWD/iap_logs}"
mkdir -p "$LOG_DIR"

PYCMD="$PYTHON $SCRIPT_DIR/fetch_iap_sequences.py --build $BUILD --pattern $PATTERN --out $OUT --rmsk-dir $RMSK_DIR"
[ -n "$GENOME_MM10" ] && PYCMD="$PYCMD --genome-fa $GENOME_MM10"

echo "============================================================"
echo "Submitting IAP sequence fetch"
echo "  Build:     $BUILD   Pattern: repName contains '$PATTERN'"
echo "  Output:    $OUT"
echo "  Genome FA: ${GENOME_MM10:-<none - UCSC API>}"
echo "  Resources: -q $QUEUE -n $N_CORES -M $MEM_MB -W $WALL"
echo "  Command:   $PYCMD"
echo "============================================================"

bsub -q "$QUEUE" -n "$N_CORES" -M "$MEM_MB" \
     -R "rusage[mem=$MEM_MB] span[hosts=1]" -W "$WALL" \
     -J "iap_seqs_${BUILD}" \
     -o "$LOG_DIR/iap_seqs_%J.out" -e "$LOG_DIR/iap_seqs_%J.err" \
     bash -lc "cd $SCRIPT_DIR && $PYCMD"
