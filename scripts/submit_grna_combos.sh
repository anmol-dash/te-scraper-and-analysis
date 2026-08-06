#!/usr/bin/env bash
# submit_grna_combos.sh
# Answer the two questions about the guides that were ordered for HERVK:
#
#   1. Which COMBINATION of the sent guides covers the most family copies?
#        - LTR5_Hs   : every 2-guide combination, ranked
#        - HERVK-int : the smallest set that reaches TARGET_COV (default 50%)
#      Every subset is enumerated, so the answers are exact, not greedy.
#
#   2. Do the best combinations hit GENES?
#      Each guide is scanned against the whole genome (exhaustive, mismatches
#      anywhere, PAM required) and every site is annotated with RepeatMasker
#      and RefSeq. Hitting LTR5A/LTR5B/HERVK-int (same repFamily = ERVK) is
#      reported as on-family and fine; coding-exon hits are flagged.
#
# Everything runs inside gameca.sif and reuses the existing GAMECA scripts:
#   query.py (copies, only if the sequences CSV is missing) and
#   run_grna_combos.py (which imports run_grna_offtarget.py + te_prep.py).
#
# Run it on the login node --- it preflights, pre-downloads rmsk/refGene, then
# submits ONE job that scans the genome once for both families:
#
#     bash scripts/submit_grna_combos.sh
#
# The guide spreadsheet is tracked in the repo, so a git pull is all it takes.
# To use a different one:  GUIDES=/path/to/guides.xlsx bash scripts/submit_grna_combos.sh
#
# Env overrides: SIF GENOME REF GUIDES COMBO_WORK COMBO_JOB FAMILIES TARGET_COV
#                COV_MM OT_MM RMSK_DIR QUEUE WALL RUN_LOCAL
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
SIF=${SIF:-$REPO/gameca.sif}
REF=${REF:-$HOME/hervk_ccle/ref}
GENOME=${GENOME:-$REF/hg38.fa}
WORK=${COMBO_WORK:-$HOME/hervk_reagents}
OUT="$WORK/combos"
JOB=${COMBO_JOB:-grnacombo}
FAMILIES=${FAMILIES:-"LTR5_Hs HERVK-int"}
ASSEMBLY=${ASSEMBLY:-hg38}
CAS=${CAS:-SpCas9}
TARGET_COV=${TARGET_COV:-0.5}       # HERVK-int: how much of the family to cover
COV_MM=${COV_MM:-2}                 # on-family coverage: PAM-distal mismatches
OT_MM=${OT_MM:-3}                   # off-target: mismatches anywhere
RMSK_DIR=${RMSK_DIR:-$HOME/te_analysis/rmsk}
QUEUE=${QUEUE:-rhel9}
WALL=${WALL:-4:00}
mkdir -p "$WORK" "$OUT" "$RMSK_DIR"
read -r -a FAM_ARR <<< "$FAMILIES"

# host ~/.local numpy breaks the container's stack (see submit_reagent_design.sh)
export SINGULARITYENV_PYTHONNOUSERSITE=1
sing() { singularity exec -B "$HOME" "$SIF" "$@"; }

find_guides() {   # the .xlsx is git-ignored, so look in the usual landing spots
  local c
  for c in "${GUIDES:-}" \
           "$WORK"/*guide*candidates*.xlsx "$WORK"/*guide*.xlsx \
           "$REPO"/*guide*candidates*.xlsx "$REPO"/*guide*.xlsx \
           "$HOME"/*guide*candidates*.xlsx; do
    [ -n "$c" ] && [ -s "$c" ] && { printf '%s' "$c"; return 0; }
  done
  return 1
}

ensure_copies() {  # the family copies CSV (Seq column) --- regenerate if absent
  local fam="$1"
  local out="$WORK/$fam"
  local csv
  csv=$(find "$out" -name '*_with_sequences.csv' 2>/dev/null | head -1 || true)
  if [ -n "$csv" ]; then
    echo "  [$fam] copies: $csv"
    return 0
  fi
  echo "  [$fam] no sequences CSV --- regenerating with query.py (--stop-after primers)"
  mkdir -p "$out"
  sing python "$REPO/query.py" --local --family "$fam" --assembly "$ASSEMBLY" \
      --genome "$GENOME" --output "$out" --stop-after primers
  csv=$(find "$out" -name '*_with_sequences.csv' 2>/dev/null | head -1 || true)
  [ -n "$csv" ] || { echo "  [$fam] FAILED to produce a sequences CSV"; return 1; }
  echo "  [$fam] copies: $csv"
}

run_analysis() {
  local guides="$1"
  local famargs=()
  local f
  for f in "${FAM_ARR[@]}"; do famargs+=(--family "$f"); done
  sing python "$REPO/run_grna_combos.py" \
      --guides "$guides" \
      "${famargs[@]}" \
      --seqs-dir "$WORK" \
      --genome "$GENOME" \
      --assembly "$ASSEMBLY" \
      --rmsk-dir "$RMSK_DIR" \
      --cas "$CAS" \
      --cov-mm "$COV_MM" \
      --ot-mm "$OT_MM" \
      --target-cov "$TARGET_COV" \
      --out "$OUT"
}

preflight() {
  echo "== preflight =="
  [ -s "$SIF" ]    && echo "  sif OK:    $SIF"    || { echo "  MISSING sif: $SIF"; exit 1; }
  [ -s "$GENOME" ] && echo "  genome OK: $GENOME" || { echo "  MISSING genome: $GENOME"; exit 1; }

  local guides
  if ! guides=$(find_guides); then
    echo "  MISSING the guide spreadsheet. It is tracked in the repo, so:"
    echo "    git -C $REPO pull        # then re-run this script"
    echo "  (or run with GUIDES=/path/to/guides.xlsx)"
    exit 1
  fi
  echo "  guides OK: $guides"

  echo "  self-test (synthetic controls for coverage / scan / annotation):"
  sing python "$REPO/run_grna_combos.py" --self-test | sed -n '/SELF-TEST/,$p' | tail -3

  for fam in "${FAM_ARR[@]}"; do ensure_copies "$fam"; done

  # fetch annotations here: the login node definitely has internet
  echo "  annotations -> $RMSK_DIR"
  sing python -c "
import sys; sys.path.insert(0,'$REPO')
from te_prep import get_rmsk_path
from run_grna_combos import download_refgene
print('  rmsk:   ', get_rmsk_path('$ASSEMBLY', '$RMSK_DIR'))
print('  refGene:', download_refgene('$ASSEMBLY', '$RMSK_DIR'))
"
  echo "  outputs -> $OUT   (job: $JOB)"
  printf '%s' "$guides" > "$OUT/.guides_path"
}

if [ "${1:-}" = "--run-one" ]; then
  run_analysis "$(cat "$OUT/.guides_path")"
  exit 0
fi

preflight
GUIDES_PATH=$(cat "$OUT/.guides_path")

if [ "${RUN_LOCAL:-0}" = "1" ]; then
  echo "== running here (RUN_LOCAL=1) =="
  run_analysis "$GUIDES_PATH"
  exit 0
fi

LOG="$OUT/${JOB}.%J.log"
bsub -n 4 -M 24000 -R "rusage[mem=24000]" -W "$WALL" -q "$QUEUE" \
     -J "$JOB" -o "$LOG" -e "$LOG" \
     env SIF="$SIF" GENOME="$GENOME" REF="$REF" COMBO_WORK="$WORK" COMBO_JOB="$JOB" \
         FAMILIES="$FAMILIES" ASSEMBLY="$ASSEMBLY" CAS="$CAS" TARGET_COV="$TARGET_COV" \
         COV_MM="$COV_MM" OT_MM="$OT_MM" RMSK_DIR="$RMSK_DIR" \
     bash "$REPO/scripts/$(basename "$0")" --run-one
echo "submitted $JOB for: ${FAM_ARR[*]}"
echo "  watch:   bjobs -J $JOB"
echo "  log:     $OUT/${JOB}.*.log"
echo "  ANSWER:  $OUT/ANSWER.txt      (also printed at the end of the log)"
echo "  tables:  $OUT/combos_<family>.csv, guide_coverage_<family>.csv,"
echo "           best_by_size_<family>.csv, offtarget_sites_all.csv"
