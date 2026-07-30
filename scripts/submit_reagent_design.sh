#!/usr/bin/env bash
# submit_reagent_design.sh
# Generate qPCR primers + CRISPR (a/i/KO) gRNA reagents for HERVK subfamilies,
# using the existing GAMECA scripts, one LSF array element per family.
#
# Per family it runs (all inside gameca.sif, reusing the cluster hg38.fa):
#   1. query.py --stop-after primers  -> family sequences CSV + specific primers
#   2. run_grna_offtarget.py          -> coverage-vs-off-target Pareto guide set
#   3. run_grna_analysis.py           -> PAM-aware greedy guide set (SpCas9)
#   4. te_qpcr_primers.py             -> primer3 fwd/rev qPCR pairs + specificity
#
# Reagent mapping (from the CCLE HERVK expression):
#   LTR5_Hs, LTR5  = 5' LTR / promoter  -> CRISPRa (dCas9-VPR) or CRISPRi (dCas9-KRAB)
#   HERVK-int      = internal/coding    -> knockout guides + qPCR of transcript body
# (same guides serve CRISPRa and CRISPRi; only the dCas9 effector differs.)
#
# Run on login (query.py --local fetches rmsk, needs internet; compute nodes here
# also have internet):  bash scripts/submit_reagent_design.sh
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
SIF=${SIF:-$REPO/gameca.sif}
REF=${REF:-$HOME/hervk_ccle/ref}          # reuse the downloaded hg38.fa
GENOME=${GENOME:-$REF/hg38.fa}
WORK=${WORK:-$HOME/hervk_reagents}
CAS=${CAS:-SpCas9}
FAMILIES=${FAMILIES:-"LTR5_Hs LTR5 HERVK-int"}   # add LTR5A LTR5B for full variant set
QUEUE=${QUEUE:-rhel9}
WALL=${WALL:-8:00}
JOB=${JOB:-reagents}
mkdir -p "$WORK"
read -r -a FAM_ARR <<< "$FAMILIES"
sing() { singularity exec -B "$HOME" "$SIF" "$@"; }

design_one() {
  local fam="$1" out="$WORK/$fam"
  mkdir -p "$out"
  echo "== [$fam] 1/4 query.py primers + sequences =="
  sing python "$REPO/query.py" --local --family "$fam" --assembly hg38 \
      --genome "$GENOME" --output "$out" --stop-after primers

  # locate the family sequences CSV (the one with a 'Seq' column)
  local seqcsv
  seqcsv=$(find "$out" -name '*.csv' -print0 | xargs -0 -I{} sh -c \
      'head -1 "{}" | grep -qiw Seq && echo "{}"' | head -1 || true)
  [ -z "$seqcsv" ] && { echo "[$fam] no sequences CSV with a Seq column under $out"; return 1; }
  echo "[$fam] sequences CSV: $seqcsv"

  echo "== [$fam] 2/4 gRNA off-target/coverage Pareto ($CAS) =="
  sing python "$REPO/run_grna_offtarget.py" --input "$seqcsv" --family "$fam" \
      --cas "$CAS" --reports-dir "$out" || echo "[$fam] offtarget step failed (see log)"

  echo "== [$fam] 3/4 gRNA greedy set ($CAS) =="
  sing python "$REPO/run_grna_analysis.py" --input "$seqcsv" --family "$fam" \
      --cas "$CAS" --reports-dir "$out" || echo "[$fam] grna_analysis step failed (see log)"

  echo "== [$fam] 4/4 primer3 qPCR pairs =="
  sing python "$REPO/te_qpcr_primers.py" --input "$seqcsv" --family "$fam" \
      --genome "$GENOME" --out "$out" || echo "[$fam] qpcr step failed (see log)"

  echo "[$fam] done -> $out"
}

preflight() {
  echo "== preflight =="
  [ -s "$SIF" ]    && echo "  sif OK: $SIF"       || { echo "  MISSING sif: $SIF"; exit 1; }
  [ -s "$GENOME" ] && echo "  genome OK: $GENOME" || { echo "  MISSING genome: $GENOME"; exit 1; }
  echo "  families: ${FAM_ARR[*]}"
  echo "  python deps in sif:"
  for m in numpy pandas sklearn primer3; do
    sing python -c "import $m" 2>/dev/null && echo "    $m: OK" \
      || echo "    $m: MISSING (needed for $( [ "$m" = primer3 ] && echo qPCR || echo pipeline ))"
  done
}

if [ "${1:-}" = "--run-one" ]; then
  idx="${LSB_JOBINDEX:?--run-one needs an array index}"
  design_one "${FAM_ARR[$((idx-1))]}"; exit 0
fi

preflight
N=${#FAM_ARR[@]}
bsub_args=( -n 4 -M 20000 -R "rusage[mem=20000]" -W "$WALL" -q "$QUEUE" )
LOG="$WORK/${JOB}.%J_%I.log"
bsub "${bsub_args[@]}" -J "${JOB}[1-$N]" -o "$LOG" -e "$LOG" \
     env SIF="$SIF" REF="$REF" GENOME="$GENOME" WORK="$WORK" CAS="$CAS" \
         FAMILIES="$FAMILIES" REPO="$REPO" \
     bash "$REPO/scripts/$(basename "$0")" --run-one
echo "submitted ${JOB}[1-$N] for: ${FAM_ARR[*]}"
echo "  watch: bjobs -J $JOB ; logs: $WORK/${JOB}.*_*.log ; outputs under $WORK/<family>/"
