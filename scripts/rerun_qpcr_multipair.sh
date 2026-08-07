#!/usr/bin/env bash
# rerun_qpcr_multipair.sh --- redesign qPCR primers as K per-sub-cluster pairs
# for already-designed families, reusing their existing query.py sequences CSV
# (no re-run of clustering or gRNA). Then annotate per-pair + union coverage.
#
# Submits to the rhel9 BATCH queue. NOTE: the apptainer FIPS/OpenSSL panic is
# NODE-dependent, not queue-dependent -- node188/rhel9 panics on the same image
# other rhel9 nodes run fine. lib_container.sh sets GOFIPS=0 to avoid it and
# resolves singularity-or-apptainer, since the compute nodes may have only one.
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
EXPR_TSV=${EXPR_TSV:-}     # locus_expression.tsv: if set, attach per-copy expression first
REAGENT_TAG=${REAGENT_TAG:-}  # output suffix to keep datasets separate (e.g. gse, ccle)
TOP_N=${TOP_N:-}          # if set, also write <family>_top<N>_primers.csv (ranked list)
# Amplicon window and primer3 seed depth. Matter most for FRAGMENTED families:
# a copy shorter than AMP_MIN can never be amplified, so it caps achievable
# coverage. Empty = each tool's own default (80/150/15).
AMP_MIN=${AMP_MIN:-}
AMP_MAX=${AMP_MAX:-}
MAX_REPS=${MAX_REPS:-}    # real copies per group used to seed primer3
QUEUE=${QUEUE:-rhel9}
WALL=${WALL:-4:00}
JOB=${JOB:-qpcr_multi}
# shellcheck source=/dev/null
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_container.sh"
container_module_load
container_init || exit 1
container_probe "$SIF" || echo "  WARNING: container unusable on $(hostname -s); this will fail"

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
    # REAGENT_TAG keeps datasets separate: e.g. TAG=gse -> LTR5_Hs_gse_qpcr_pairs.csv
    local label="$fam"; [ -n "$REAGENT_TAG" ] && label="${fam}_${REAGENT_TAG}"
    local designcsv="$seqcsv"
    # attach per-copy expression to a TAGGED copy (not in place) so CCLE/GSE don't clobber
    if [ -n "$EXPR_TSV" ]; then
      designcsv="$out/${label}.seqs.csv"
      echo "[$label] attaching per-locus expression from $EXPR_TSV"
      "$GAMECA_RT" exec $CONTAINER_EXEC_FLAGS -B "$HOME" "$SIF" python "$REPO/attach_locus_expression.py" \
          --expr "$EXPR_TSV" --seqs "$seqcsv" --out "$designcsv" \
          || { echo "[$label] expression attach failed"; designcsv="$seqcsv"; }
    fi
    echo "== [$label] greedy $NPAIRS-pair union (clusters=$NCLUST, max-mm=$MAXMM) seqs: $designcsv =="
    local exprarg=()
    [ -n "$EXPR_COLS" ] && exprarg=(--expr-cols $EXPR_COLS)
    [ -n "$AMP_MIN" ]  && exprarg+=(--amp-min "$AMP_MIN")
    [ -n "$AMP_MAX" ]  && exprarg+=(--amp-max "$AMP_MAX")
    [ -n "$MAX_REPS" ] && exprarg+=(--max-reps "$MAX_REPS")
    SINGULARITYENV_PYTHONPATH="$PYLIB" APPTAINERENV_PYTHONPATH="$PYLIB" \
      "$GAMECA_RT" exec $CONTAINER_EXEC_FLAGS -B "$HOME" "$SIF" python "$REPO/te_qpcr_primers.py" \
        --input "$designcsv" --family "$label" --genome "$GENOME" --out "$out" \
        --n-pairs "$NPAIRS" --n-clusters "$NCLUST" --max-mm "$MAXMM" ${exprarg[@]+"${exprarg[@]}"} \
        || { echo "[$label] primer design failed"; continue; }
    if [ -n "$TOP_N" ]; then
      echo "[$label] ranking top-$TOP_N primers"
      SINGULARITYENV_PYTHONPATH="$PYLIB" APPTAINERENV_PYTHONPATH="$PYLIB" \
        "$GAMECA_RT" exec $CONTAINER_EXEC_FLAGS -B "$HOME" "$SIF" python "$REPO/top_primers.py" \
          --input "$designcsv" --family "$label" --genome "$GENOME" --out "$out" \
          --top-n "$TOP_N" --max-mm "$MAXMM" ${exprarg[@]+"${exprarg[@]}"} \
          || echo "[$label] top-primers ranking failed"
    fi
  done
}

if [ "${1:-}" = "--run" ]; then do_work; exit 0; fi

# submit to rhel9 batch (interactive nodes FIPS-panic the apptainer binary)
# BSUB_DEP: LSF dependency expression, e.g. 'done(a) && done(b)'. Kept as ONE
# argument -- word-splitting it would hand bsub a broken '-w done(a)' plus junk.
dep_args=(); [ -n "${BSUB_DEP:-}" ] && dep_args=( -w "$BSUB_DEP" )
LOG="$WORK/${JOB}.%J.log"
bsub -q "$QUEUE" -n 4 -M 20000 -R "rusage[mem=20000]" -W "$WALL" \
     ${dep_args[@]+"${dep_args[@]}"} \
     -J "$JOB" -o "$LOG" -e "$LOG" \
     env SIF="$SIF" REF="$REF" GENOME="$GENOME" REAGENT_WORK="$WORK" \
         FAMILIES="$FAMILIES" NCLUST="$NCLUST" NPAIRS="$NPAIRS" MAXMM="$MAXMM" \
         EXPR_COLS="$EXPR_COLS" EXPR_TSV="$EXPR_TSV" REAGENT_TAG="$REAGENT_TAG" TOP_N="$TOP_N" \
         AMP_MIN="$AMP_MIN" AMP_MAX="$AMP_MAX" MAX_REPS="$MAX_REPS" \
     bash "$REPO/scripts/$(basename "$0")" --run
echo "submitted $JOB (-q $QUEUE) for: $FAMILIES  (NCLUST=$NCLUST)"
echo "  watch: bjobs -J $JOB ; log: $WORK/${JOB}.*.log"
