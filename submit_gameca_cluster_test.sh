#!/usr/bin/env bash
#
# submit_gameca_cluster_test.sh — submit the comprehensive GAMECA cluster test
# to LSF and collect a single detailed output file.
#
# It runs run_gameca_cluster_test.py over a LINE/SINE/LTR × hg38/mm10 matrix:
# for each family it pulls loci + sequences straight from the UCSC browser,
# clusters them, injects SYNTHETIC expression columns with varied (non
# developmental-stage) sample names, then runs every Stage 11 module. A master
# log + JSON summary land in $OUT_DIR.
#
# Usage:
#   chmod +x submit_gameca_cluster_test.sh
#   ./submit_gameca_cluster_test.sh                 # batch bsub (default)
#   ./submit_gameca_cluster_test.sh -Is             # interactive (recommended if
#                                                   # only login nodes reach UCSC)
#
# ── INTERNET REQUIREMENT (read this) ───────────────────────────────────────────
# The test fetches from hgdownload.soe.ucsc.edu and api.genome.ucsc.edu. On many
# clusters compute nodes have NO direct outbound internet. Two ways to cope:
#   (a) HTTPS proxy: if your login shell sets http_proxy/https_proxy, this script
#       forwards them into the job (curl in te_prep.py rides the proxy).
#   (b) Pre-cache rmsk on the login node, then point te_prep at it:
#         python te_prep.py --download hg38 --rmsk-dir "$RMSK_DIR"
#         python te_prep.py --download mm10 --rmsk-dir "$RMSK_DIR"
#       (sequence fetching still needs outbound HTTPS — see note in the log).
# If neither works on batch nodes, submit with `-Is` to run on an interactive
# node that can reach UCSC.

set -uo pipefail

# ── Config ─────────────────────────────────────────────────────────────────────
OUT_DIR="${OUT_DIR:-/home/amodz/anmol/gamecatestv624}"
EMAIL="${EMAIL:-anmoldash@gmail.com}"
MATRIX="${MATRIX:-}"                 # empty → built-in matrix; or "L1HS:hg38,MT2_Mm:mm10"
MAX_LOCI="${MAX_LOCI:-300}"          # cap loci/family for tractable UCSC + MAFFT
RMSK_DIR="${RMSK_DIR:-}"             # optional pre-downloaded rmsk_<build>.txt.gz dir
INCLUDE_FOLD="${INCLUDE_FOLD:-0}"    # 1 → also run ColabFold (heavy/GPU)
QUEUE="${QUEUE:-normal}"
N_CORES="${N_CORES:-4}"
MEM_MB="${MEM_MB:-24000}"            # peak observed ~19 GB (L1HS); 16 GB overran
WALL="${WALL:-24:00}"                # HH:MM
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

mkdir -p "$OUT_DIR"

# ── Build the python command ───────────────────────────────────────────────────
PYCMD=("$PYTHON" "$SCRIPT_DIR/run_gameca_cluster_test.py"
       --out-dir "$OUT_DIR"
       --max-loci "$MAX_LOCI"
       --notify-email "$EMAIL")
[ -n "$MATRIX" ]   && PYCMD+=(--matrix "$MATRIX")
[ -n "$RMSK_DIR" ] && PYCMD+=(--rmsk-dir "$RMSK_DIR")
[ "$INCLUDE_FOLD" = "1" ] && PYCMD+=(--include-fold)

# Forward an HTTPS proxy into the job if the login shell has one (lets the UCSC
# fetch work from compute nodes that only allow proxied outbound HTTPS).
PROXY_PREFIX=""
if [ -n "${https_proxy:-${HTTPS_PROXY:-}}" ]; then
  P="${https_proxy:-$HTTPS_PROXY}"
  PROXY_PREFIX="export https_proxy='$P' http_proxy='$P' HTTPS_PROXY='$P' HTTP_PROXY='$P'; "
  echo "Forwarding HTTPS proxy into job: $P"
fi

JOB_SH="$OUT_DIR/_run_cluster_test.sh"
PYCMD_STR="$(printf '%q ' "${PYCMD[@]}")"
{
  echo "#!/usr/bin/env bash"
  echo "set -uo pipefail"
  # Avoid the noisy (non-fatal) 'conda-libmamba-solver (libicui18n.so.*)' load
  # error on cluster nodes with an older system ICU than the conda env expects.
  echo "export CONDA_SOLVER=classic"
  printf 'cd %q\n' "$SCRIPT_DIR"
  [ -n "$PROXY_PREFIX" ] && echo "$PROXY_PREFIX"
  echo 'echo "Host: $(hostname)   Started: $(date)"'
  echo "$PYCMD_STR"
  echo 'echo "Finished: $(date)"'
} > "$JOB_SH"
chmod +x "$JOB_SH"

BSUB_OUT="$OUT_DIR/bsub.%J.out"
BSUB_ERR="$OUT_DIR/bsub.%J.err"

BSUB_ARGS=(-J "gamecatest"
           -q "$QUEUE"
           -n "$N_CORES"
           -M "$MEM_MB" -R "rusage[mem=$MEM_MB]"
           -W "$WALL"
           -o "$BSUB_OUT" -e "$BSUB_ERR")

echo "============================================================"
echo "GAMECA cluster test submission"
echo "  Out dir:  $OUT_DIR"
echo "  Matrix:   ${MATRIX:-<built-in LINE/SINE/LTR x hg38/mm10>}"
echo "  Max loci: $MAX_LOCI   Fold: $INCLUDE_FOLD"
echo "  Resources: -q $QUEUE -n $N_CORES -M $MEM_MB -W $WALL"
echo "  Job script: $JOB_SH"
echo "============================================================"

if [ "${1:-}" = "-Is" ]; then
  echo "Submitting INTERACTIVE (streams output; runs on an internet-capable node)…"
  exec bsub "${BSUB_ARGS[@]}" -Is bash "$JOB_SH"
else
  echo "Submitting BATCH job…"
  bsub "${BSUB_ARGS[@]}" bash "$JOB_SH"
  echo
  echo "Track with:  bjobs -J gamecatest"
  echo "Live log:    tail -f $OUT_DIR/gameca_cluster_test.log"
  echo "Summary:     $OUT_DIR/gameca_cluster_test_summary.json"
fi
