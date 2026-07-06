#!/usr/bin/env bash
#
# submit_clustering_test.sh — validate the new clustering defaults
# (kmer=10, UMAP n_neighbors=15/min_dist=0.0, HDBSCAN min_samples=5/
# min_cluster_size=100) on a FROM-SCRATCH MT2_Mm run: UCSC fetch → clustering
# → full pipeline → LaTeX report, as a single LSF job.
#
# Usage:
#   chmod +x submit_clustering_test.sh
#   ./submit_clustering_test.sh            # batch
#   ./submit_clustering_test.sh -Is        # interactive (streams output)

set -uo pipefail

# ── Config ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAMILY="${FAMILY:-MT2_Mm}"
ASSEMBLY="${ASSEMBLY:-mm10}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/results_clustering_test}"
PYTHON="${PYTHON:-python3}"
QUEUE="${QUEUE:-normal}"
N_CORES="${N_CORES:-4}"
MEM_MB="${MEM_MB:-24000}"
WALL="${WALL:-06:00}"                                 # HH:MM

mkdir -p "$OUT_DIR"

# Forward an HTTPS proxy into the job so the UCSC fetch works from compute nodes.
PROXY_PREFIX=""
if [ -n "${https_proxy:-${HTTPS_PROXY:-}}" ]; then
  P="${https_proxy:-$HTTPS_PROXY}"
  PROXY_PREFIX="export https_proxy='$P' http_proxy='$P' HTTPS_PROXY='$P' HTTP_PROXY='$P'; "
  echo "Forwarding HTTPS proxy into job: $P"
fi

# ── Build the job script ───────────────────────────────────────────────────────
JOB_SH="$OUT_DIR/_run_clustering_test.sh"
{
  echo "#!/usr/bin/env bash"
  echo "set -uo pipefail"
  echo "export CONDA_SOLVER=classic"          # silence non-fatal libmamba/ICU error
  printf 'cd %q\n' "$SCRIPT_DIR"
  [ -n "$PROXY_PREFIX" ] && echo "$PROXY_PREFIX"
  echo 'echo "Host: $(hostname)   Started: $(date)"'
  echo 'rc=0'
  printf '%s\n' "$PYTHON query.py --local --family $FAMILY --assembly $ASSEMBLY --output $(printf '%q' "$OUT_DIR") --force || rc=1"
  cat <<'CHECK'
echo "──────── verifying clustering output ────────"
coords=$(find "$OUT_DIR_ESC" -name clustering_coordinates.csv | head -1)
if [ -s "$coords" ]; then
  n_clusters=$(tail -n +2 "$coords" | cut -d, -f5 | sort -u | wc -l)
  echo "PASS: clustering_coordinates.csv present ($(( $(wc -l < "$coords") - 1 )) rows, $n_clusters distinct cluster labels)"
else
  echo "FAIL: clustering_coordinates.csv missing"; rc=1
fi
CHECK
  echo 'echo "Finished: $(date)   exit=$rc"'
  echo 'exit $rc'
} > "$JOB_SH"
# Inject the resolved OUT_DIR into the heredoc'd check block.
sed -i.bak "s#\$OUT_DIR_ESC#$OUT_DIR#g" "$JOB_SH" && rm -f "$JOB_SH.bak"
chmod +x "$JOB_SH"

# ── Submit ─────────────────────────────────────────────────────────────────────
BSUB_OUT="$OUT_DIR/clustering_test.%J.out"
BSUB_ERR="$OUT_DIR/clustering_test.%J.err"
BSUB_ARGS=(-J "gameca_clustering_test"
           -q "$QUEUE"
           -n "$N_CORES"
           -M "$MEM_MB" -R "rusage[mem=$MEM_MB]"
           -W "$WALL"
           -o "$BSUB_OUT" -e "$BSUB_ERR")

echo "============================================================"
echo "GAMECA clustering-defaults test submission"
echo "  Family:    $FAMILY ($ASSEMBLY)"
echo "  Out dir:   $OUT_DIR"
echo "  Logs:      clustering_test.<JOBID>.out / .err"
echo "  Resources: -q $QUEUE -n $N_CORES -M $MEM_MB -W $WALL"
echo "  Job script: $JOB_SH"
echo "============================================================"

if [ "${1:-}" = "-Is" ]; then
  echo "Submitting INTERACTIVE (streams output; internet-capable node)…"
  exec bsub "${BSUB_ARGS[@]}" -Is bash "$JOB_SH"
else
  echo "Submitting BATCH job…"
  bsub "${BSUB_ARGS[@]}" bash "$JOB_SH"
  echo
  echo "Track with:  bjobs -J gameca_clustering_test"
  echo "Stdout:      $OUT_DIR/clustering_test.<JOBID>.out"
  echo "Stderr:      $OUT_DIR/clustering_test.<JOBID>.err"
fi
