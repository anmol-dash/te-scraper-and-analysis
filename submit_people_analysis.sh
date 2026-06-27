#!/usr/bin/env bash
#
# submit_people_analysis.sh — one command, run on the cluster LOGIN node, that
# submits a GAMECA query.py job per TE family into per-person folders:
#
#   Diego  : all L1Md  subfamilies (mm10)
#   Claire : MT2_Mm (mm10) + all B2 subfamilies (mm10)
#   Katie  : all IAP   subfamilies (mm10)
#   Zach   : AluSx1 (hg38)
#
# Subfamilies are discovered live from the rmsk table. Exactly one family per
# group runs the heavy ColabFold protein-fold calc (the most-abundant subfamily),
# and only if COLABFOLD_CMD is set.
#
# Usage:
#   chmod +x submit_people_analysis.sh
#   ./submit_people_analysis.sh                 # submit everything
#   ./submit_people_analysis.sh --dry-run       # show what would be submitted
#
# Internet note: discovery + UCSC sequence fetch need outbound HTTPS. Login nodes
# usually have it; if your login shell sets http_proxy/https_proxy it is forwarded
# into each job. Alternatively pre-cache rmsk and point at it with RMSK_DIR:
#   python te_prep.py --download mm10 --rmsk-dir "$RMSK_DIR"
#   python te_prep.py --download hg38 --rmsk-dir "$RMSK_DIR"

set -uo pipefail

# ── Config (override via env) ──────────────────────────────────────────────────
BASE_DIR="${BASE_DIR:-/home/amodz/anmol/people_runs}"
COLABFOLD_CMD="${COLABFOLD_CMD:-}"     # e.g. colabfold_batch ; empty => no folding
FOLD_SCOPE="${FOLD_SCOPE:-group}"      # group | folder | none
MAX_LOCI="${MAX_LOCI:-}"               # empty => no cap (full family)
MIN_COUNT="${MIN_COUNT:-10}"           # skip discovered subfamilies below this size
RMSK_DIR="${RMSK_DIR:-}"               # optional pre-downloaded rmsk_<build>.txt.gz dir
GENOME_MM10="${GENOME_MM10:-}"         # optional mm10 FASTA for primer search
GENOME_HG38="${GENOME_HG38:-}"         # optional hg38 FASTA for primer search
EXTRA_QUERY_ARGS="${EXTRA_QUERY_ARGS:-}"   # appended to every query.py call
QUEUE="${QUEUE:-normal}"
N_CORES="${N_CORES:-4}"
MEM_MB="${MEM_MB:-24000}"
WALL="${WALL:-24:00}"
PYTHON="${PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYCMD=("$PYTHON" "$SCRIPT_DIR/run_people_analysis.py"
       --base-dir "$BASE_DIR"
       --fold-scope "$FOLD_SCOPE"
       --min-count "$MIN_COUNT"
       --queue "$QUEUE" --cores "$N_CORES" --mem-mb "$MEM_MB" --wall "$WALL")
[ -n "$COLABFOLD_CMD" ]    && PYCMD+=(--colabfold-cmd "$COLABFOLD_CMD")
[ -n "$MAX_LOCI" ]         && PYCMD+=(--max-loci "$MAX_LOCI")
[ -n "$RMSK_DIR" ]         && PYCMD+=(--rmsk-dir "$RMSK_DIR")
[ -n "$GENOME_MM10" ]      && PYCMD+=(--genome-mm10 "$GENOME_MM10")
[ -n "$GENOME_HG38" ]      && PYCMD+=(--genome-hg38 "$GENOME_HG38")
[ -n "$EXTRA_QUERY_ARGS" ] && PYCMD+=(--extra-query-args "$EXTRA_QUERY_ARGS")
# Pass through any extra flags (e.g. --dry-run, --scheduler slurm).
PYCMD+=("$@")

echo "============================================================"
echo "GAMECA per-person submission"
echo "  Base dir:    $BASE_DIR"
echo "  Fold scope:  $FOLD_SCOPE   Colabfold: ${COLABFOLD_CMD:-<none>}"
echo "  Max loci:    ${MAX_LOCI:-<all>}   Min subfamily size: $MIN_COUNT"
echo "  Resources:   -q $QUEUE -n $N_CORES -M $MEM_MB -W $WALL"
echo "============================================================"

exec "${PYCMD[@]}"
