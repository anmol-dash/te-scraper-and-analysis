#!/usr/bin/env bash
# rerun_qpcr_multipair.sh --- redesign qPCR primers as K per-sub-cluster pairs
# for already-designed families, reusing their existing query.py sequences CSV
# (no re-run of clustering or gRNA). Then annotate per-pair + union coverage.
#
# Submits to the rhel9 BATCH queue: the interactive nodes panic the apptainer
# binary with a FIPS/OpenSSL error, but singularity works on rhel9 batch (that's
# where the first pass ran). Do NOT run this inside an interactive session.
#
#   FAMILIES="LTR5 HERVK-int" NCLUST=3 bash scripts/rerun_qpcr_multipair.sh
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
SIF=${SIF:-$REPO/gameca.sif}
REF=${REF:-$HOME/hervk_ccle/ref}
GENOME=${GENOME:-$REF/hg38.fa}
WORK=${REAGENT_WORK:-$HOME/hervk_reagents}
PYLIB="$WORK/pylib"
FAMILIES=${FAMILIES:-"LTR5 HERVK-int"}
NCLUST=${NCLUST:-3}
NPAIRS=${NPAIRS:-3}        # max pairs in the greedy union set per family
MAXMM=${MAXMM:-3}          # 5'-body mismatch tolerance (3' end stays exact)
EXPR_COLS=${EXPR_COLS:-}   # per-locus expression column name(s) to weight by, if present
QUEUE=${QUEUE:-rhel9}
WALL=${WALL:-4:00}
JOB=${JOB:-qpcr_multi}
export SINGULARITYENV_PYTHONNOUSERSITE=1 APPTAINERENV_PYTHONNOUSERSITE=1

do_work() {
  read -r -a FAM_ARR <<< "$FAMILIES"
  for fam in "${FAM_ARR[@]}"; do
    local out="$WORK/$fam"
    local seqcsv
    # prefer a CSV with BOTH Seq and Cluster (the clustered file) over Seq-only
    seqcsv=$(find "$out" -name '*.csv' | while read -r f; do
        h=$(head -1 "$f" 2>/dev/null)
        echo "$h" | grep -qiw Seq || continue
        if echo "$h" | grep -qiw Cluster; then echo "2 $f"; else echo "1 $f"; fi
      done | sort -rn | head -1 | cut -d' ' -f2-)
    [ -z "$seqcsv" ] && { echo "[$fam] no sequences CSV under $out -- skip"; continue; }
    echo "== [$fam] greedy $NPAIRS-pair union (clusters=$NCLUST, max-mm=$MAXMM) seqs: $seqcsv =="
    local exprarg=()
    [ -n "$EXPR_COLS" ] && exprarg=(--expr-cols $EXPR_COLS)
    SINGULARITYENV_PYTHONPATH="$PYLIB" APPTAINERENV_PYTHONPATH="$PYLIB" \
      singularity exec -B "$HOME" "$SIF" python "$REPO/te_qpcr_primers.py" \
        --input "$seqcsv" --family "$fam" --genome "$GENOME" --out "$out" \
        --n-pairs "$NPAIRS" --n-clusters "$NCLUST" --max-mm "$MAXMM" "${exprarg[@]}" \
        || { echo "[$fam] primer design failed"; continue; }
  done
}

if [ "${1:-}" = "--run" ]; then do_work; exit 0; fi

# submit to rhel9 batch (interactive nodes FIPS-panic the apptainer binary)
LOG="$WORK/${JOB}.%J.log"
bsub -q "$QUEUE" -n 4 -M 20000 -R "rusage[mem=20000]" -W "$WALL" \
     -J "$JOB" -o "$LOG" -e "$LOG" \
     env SIF="$SIF" REF="$REF" GENOME="$GENOME" REAGENT_WORK="$WORK" \
         FAMILIES="$FAMILIES" NCLUST="$NCLUST" NPAIRS="$NPAIRS" MAXMM="$MAXMM" \
         EXPR_COLS="$EXPR_COLS" \
     bash "$REPO/scripts/$(basename "$0")" --run
echo "submitted $JOB (-q $QUEUE) for: $FAMILIES  (NCLUST=$NCLUST)"
echo "  watch: bjobs -J $JOB ; log: $WORK/${JOB}.*.log"
