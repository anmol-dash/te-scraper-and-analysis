#!/usr/bin/env bash
# rerun_qpcr_multipair.sh --- redesign qPCR primers as K per-sub-cluster pairs
# for already-designed families, reusing their existing query.py sequences CSV
# (no re-run of clustering or gRNA). Then annotate per-pair + union coverage.
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
export SINGULARITYENV_PYTHONNOUSERSITE=1
read -r -a FAM_ARR <<< "$FAMILIES"

for fam in "${FAM_ARR[@]}"; do
  out="$WORK/$fam"
  seqcsv=$(find "$out" -name '*.csv' -print0 | xargs -0 -I{} sh -c \
      'head -1 "{}" | grep -qiw Seq && echo "{}"' 2>/dev/null | head -1 || true)
  [ -z "$seqcsv" ] && { echo "[$fam] no sequences CSV under $out -- skip"; continue; }
  echo "== [$fam] redesign as $NCLUST per-cluster pairs (seqs: $seqcsv) =="
  SINGULARITYENV_PYTHONPATH="$PYLIB" singularity exec -B "$HOME" "$SIF" \
      python "$REPO/te_qpcr_primers.py" --input "$seqcsv" --family "$fam" \
      --genome "$GENOME" --out "$out" --n-clusters "$NCLUST" \
      || { echo "[$fam] primer design failed"; continue; }
  python3 "$REPO/te_qpcr_coverage.py" \
      --pairs "$out/${fam}_qpcr_pairs.csv" --seqs "$seqcsv"
done
