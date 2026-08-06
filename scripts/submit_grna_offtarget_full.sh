#!/usr/bin/env bash
# submit_grna_offtarget_full.sh
# COMPREHENSIVE off-target check for the ordered guides, as an LSF job.
#
# The fast scan in scripts/submit_grna_combos.sh searches canonical NGG PAMs and
# mismatches on the primary chromosomes. This one additionally searches:
#
#   * DNA and RNA BULGES (an unpaired base on either side of the duplex) --- a
#     bulged off-target looks like ~15 mismatches to a mismatch-only search and
#     is silently discarded by it
#   * NON-CANONICAL PAMs (NAG, NGA), which SpCas9 cuts at reduced efficiency;
#     an NAG site with 0 mismatches can out-cut an NGG site with 4
#   * EVERY CONTIG, including alt haplotypes and unplaced scaffolds
#   * up to 4 mismatches at any position (no seed heuristic anywhere)
#
# That is ~40-120x the work --- hours of CPU and a 3.3 GB genome resident --- so
# run_grna_offtarget_full.py refuses to start outside a scheduler job. This
# script is the intended way to run it.
#
#     bash scripts/submit_grna_offtarget_full.sh              # size it, then submit
#     ESTIMATE_ONLY=1 bash scripts/submit_grna_offtarget_full.sh   # size it only
#
# After the scan it re-derives every combination/coverage table on top of the
# comprehensive site list (run_grna_combos.py --reuse-sites), so the combination
# rankings account for bulged and non-canonical off-targets too.
#
# The guide spreadsheet is tracked in the repo, so a git pull is all it takes:
#     git pull && bash scripts/submit_grna_offtarget_full.sh
# (GUIDES=/path/to/other.xlsx overrides which spreadsheet is used.)
#
# Env overrides: SIF GENOME REF GUIDES COMBO_WORK OTFULL_JOB FAMILIES OT_MM
#                BULGE BULGE_MM PAM_SET WORKERS RMSK_DIR QUEUE WALL MEM
#                ESTIMATE_ONLY
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
SIF=${SIF:-$REPO/gameca.sif}
REF=${REF:-$HOME/hervk_ccle/ref}
GENOME=${GENOME:-$REF/hg38.fa}
WORK=${COMBO_WORK:-$HOME/hervk_reagents}
OUT="$WORK/offtarget_full"
JOB=${OTFULL_JOB:-otfull}
FAMILIES=${FAMILIES:-"LTR5_Hs HERVK-int"}
ASSEMBLY=${ASSEMBLY:-hg38}
OT_MM=${OT_MM:-4}                  # mismatches anywhere in the protospacer
BULGE=${BULGE:-1}                  # max DNA/RNA bulge size (0 disables)
BULGE_MM=${BULGE_MM:-2}            # extra mismatches allowed WITH a bulge.
                                   # Deliberately below OT_MM: a bulged alignment
                                   # drops a position and multiplies alignments
                                   # ~37x, so at 4 mismatches it returns ~800
                                   # random sites per guide per 47 Mb --- the
                                   # real hits drown in background.
PAM_SET=${PAM_SET:-NGG,NAG,NGA}    # canonical + the two SpCas9 cuts weakly
WORKERS=${WORKERS:-16}             # keep == bsub -n
RMSK_DIR=${RMSK_DIR:-$HOME/te_analysis/rmsk}
TARGET_COV=${TARGET_COV:-0.5}
QUEUE=${QUEUE:-rhel9}
WALL=${WALL:-24:00}
MEM=${MEM:-64000}
mkdir -p "$WORK" "$OUT" "$RMSK_DIR"
read -r -a FAM_ARR <<< "$FAMILIES"

export SINGULARITYENV_PYTHONNOUSERSITE=1
sing() { singularity exec -B "$HOME" "$SIF" "$@"; }

find_guides() {
  local c
  for c in "${GUIDES:-}" \
           "$WORK"/*guide*candidates*.xlsx "$WORK"/*guide*.xlsx \
           "$REPO"/*guide*candidates*.xlsx "$REPO"/*guide*.xlsx \
           "$HOME"/*guide*candidates*.xlsx; do
    [ -n "$c" ] && [ -s "$c" ] && { printf '%s' "$c"; return 0; }
  done
  return 1
}

scan_args() {
  printf '%s\n' --guides "$1" --genome "$GENOME" --out "$OUT" \
    --assembly "$ASSEMBLY" --rmsk-dir "$RMSK_DIR" \
    --ot-mm "$OT_MM" --bulge "$BULGE" --bulge-mm "$BULGE_MM" \
    --pam-set "$PAM_SET" --workers "$WORKERS"
}

run_scan() {
  local guides="$1"
  local args=()
  while IFS= read -r a; do args+=("$a"); done < <(scan_args "$guides")

  echo "== 1/2 comprehensive genome scan =="
  sing python "$REPO/run_grna_offtarget_full.py" "${args[@]}"

  # Re-rank the guide COMBINATIONS on the comprehensive site list. --reuse-sites
  # reads offtarget_sites_all.csv instead of rescanning, so this is seconds.
  echo "== 2/2 combination tables on the comprehensive sites =="
  local famargs=() f
  for f in "${FAM_ARR[@]}"; do famargs+=(--family "$f"); done
  sing python "$REPO/run_grna_combos.py" \
      --guides "$guides" "${famargs[@]}" \
      --seqs-dir "$WORK" --out "$OUT" --reuse-sites \
      --target-cov "$TARGET_COV" --rmsk-dir "$RMSK_DIR"
}

ensure_copies() {   # family copies CSV (Seq column); regenerate if absent
  local fam="$1"
  local out="$WORK/$fam"
  local csv
  csv=$(find "$out" -name '*_with_sequences.csv' 2>/dev/null | head -1 || true)
  if [ -n "$csv" ]; then echo "  [$fam] copies: $csv"; return 0; fi
  echo "  [$fam] no sequences CSV --- regenerating with query.py"
  mkdir -p "$out"
  sing python "$REPO/query.py" --local --family "$fam" --assembly "$ASSEMBLY" \
      --genome "$GENOME" --output "$out" --stop-after primers
  csv=$(find "$out" -name '*_with_sequences.csv' 2>/dev/null | head -1 || true)
  [ -n "$csv" ] || { echo "  [$fam] FAILED to produce a sequences CSV"; return 1; }
  echo "  [$fam] copies: $csv"
}

preflight() {
  echo "== preflight =="
  [ -s "$SIF" ]    && echo "  sif OK:    $SIF"    || { echo "  MISSING sif: $SIF"; exit 1; }
  [ -s "$GENOME" ] && echo "  genome OK: $GENOME" || { echo "  MISSING genome: $GENOME"; exit 1; }

  local guides
  if ! guides=$(find_guides); then
    echo "  MISSING the guide spreadsheet. It is tracked in the repo, so:"
    echo "    git -C $REPO pull        # then re-run this script"
    echo "  (or point at one explicitly: GUIDES=/path/to/guides.xlsx bash \$0)"
    exit 1
  fi
  echo "  guides OK: $guides"

  # Everything below is advisory. Only the sif, the genome and the guides are
  # required to submit --- a flaky self-test, a failed annotation download or a
  # missing copies CSV must NOT stop the job from reaching the queue, so each is
  # allowed to fail with a warning (the compute node has internet too).
  echo "  self-test (bulge maths, non-canonical PAMs, strands, tiers):"
  if sing python "$REPO/run_grna_offtarget_full.py" --self-test >"$OUT/.selftest.log" 2>&1; then
    tail -2 "$OUT/.selftest.log"
  else
    echo "  !! SELF-TEST FAILED --- see $OUT/.selftest.log (submitting anyway)"
  fi

  for fam in "${FAM_ARR[@]}"; do
    ensure_copies "$fam" || echo "  WARNING: no copies for $fam; the combination"\
      "step will skip it, the off-target scan is unaffected"
  done

  echo "  annotations -> $RMSK_DIR (fetched here; the login node has internet)"
  sing python -c "
import sys; sys.path.insert(0,'$REPO')
from te_prep import get_rmsk_path
from run_grna_combos import download_refgene
print('  rmsk:   ', get_rmsk_path('$ASSEMBLY', '$RMSK_DIR'))
print('  refGene:', download_refgene('$ASSEMBLY', '$RMSK_DIR'))
" || echo "  WARNING: annotation prefetch failed; the job will fetch them itself"

  echo "== job size =="
  local args=()
  while IFS= read -r a; do args+=("$a"); done < <(scan_args "$guides")
  sing python "$REPO/run_grna_offtarget_full.py" "${args[@]}" --estimate \
    || echo "  WARNING: could not size the job (submitting anyway)"

  printf '%s' "$guides" > "$OUT/.guides_path"
  echo "  outputs -> $OUT   (job: $JOB)"
}

if [ "${1:-}" = "--run-one" ]; then
  run_scan "$(cat "$OUT/.guides_path")"
  exit 0
fi

preflight
[ "${ESTIMATE_ONLY:-0}" = "1" ] && { echo "ESTIMATE_ONLY=1 --- not submitting."; exit 0; }

command -v bsub >/dev/null 2>&1 || {
  echo "ERROR: bsub not found --- run this on an LSF submit/login node."
  echo "       There is no local fallback here on purpose: this scan is ~9"
  echo "       core-hours. To override anyway, call the scanner directly with"
  echo "       --force-local (see --estimate first)."
  exit 1
}

LOG="$OUT/${JOB}.%J.log"
set +e
SUB=$(bsub -n "$WORKERS" -M "$MEM" -R "rusage[mem=$MEM]" -W "$WALL" -q "$QUEUE" \
     -J "$JOB" -o "$LOG" -e "$LOG" \
     env SIF="$SIF" GENOME="$GENOME" REF="$REF" COMBO_WORK="$WORK" OTFULL_JOB="$JOB" \
         FAMILIES="$FAMILIES" ASSEMBLY="$ASSEMBLY" OT_MM="$OT_MM" BULGE="$BULGE" \
         BULGE_MM="$BULGE_MM" \
         PAM_SET="$PAM_SET" WORKERS="$WORKERS" RMSK_DIR="$RMSK_DIR" \
         TARGET_COV="$TARGET_COV" \
     bash "$REPO/scripts/$(basename "$0")" --run-one 2>&1)
RC=$?
set -e
echo "$SUB"
# Confirm a job really entered the queue rather than assuming bsub worked.
JOBID=$(printf '%s' "$SUB" | sed -n 's/.*[Jj]ob <\([0-9][0-9]*\)>.*/\1/p' | head -1)
if [ "$RC" -ne 0 ] || [ -z "$JOBID" ]; then
  echo "ERROR: bsub did not submit a job (exit $RC). Nothing is queued."
  exit 1
fi
echo "submitted job $JOBID as '$JOB' (-n $WORKERS, -M $MEM, -W $WALL) for: ${FAM_ARR[*]}"
bjobs "$JOBID" 2>/dev/null || true
echo "  watch:   bjobs -J $JOB"
echo "  log:     $OUT/${JOB}.*.log"
echo "  results: $OUT/offtarget_full_report.txt   (comprehensive off-target)"
echo "           $OUT/ANSWER.txt                  (combinations, re-ranked on those sites)"
echo "           $OUT/offtarget_sites_all.csv     (every site, annotated)"
echo "           $OUT/offtarget_by_guide_full.csv (per-guide burden)"
