#!/usr/bin/env bash
# submit_hervk_ccle_requant.sh
# TE-aware HERVK requantification for MCF-7, PC-3, PANC-1 from public CCLE
# RNA-seq (SRP186687 / PRJNA523380). FULL DEPTH, one sample at a time with
# immediate cleanup of FASTQ/BAM.
#
# Prereq: run scripts/setup_hervk_ccle.sh first (login node) and let the
# star_index job finish. Tools come from the micromamba env it created
# ($WORK/env) because gameca.sif has none of them. The compute node needs
# NO internet: .sra files were pre-downloaded by setup; fasterq-dump reads
# them locally.
#
# Submit:  bash scripts/submit_hervk_ccle_requant.sh
set -euo pipefail

WORK=${WORK:-$HOME/hervk_ccle}
ENVBIN=$WORK/env/bin
REF=$WORK/ref
THREADS=${THREADS:-12}
MEM_MB=${MEM_MB:-45000}          # STAR needs ~32-40 GB for hg38
QUEUE=${QUEUE:-normal}

STAR_INDEX=$REF/star_hg38
GENE_GTF=$REF/hg38.knownGene.gtf
TE_GTF=$REF/hg38_rmsk_TE.gtf
MANIFEST=$(cd "$(dirname "$0")" && pwd)/hervk_ccle_manifest.tsv
mkdir -p "$WORK/fastq" "$WORK/bam" "$WORK/counts" "$WORK/tmp"

preflight() {
  echo "== preflight =="
  for t in fasterq-dump STAR TEcount; do
    [ -x "$ENVBIN/$t" ] && echo "  tool $t OK" || { echo "  tool $t MISSING (run setup first)"; exit 1; }
  done
  for f in "$STAR_INDEX/SAindex" "$GENE_GTF" "$TE_GTF"; do
    [ -e "$f" ] && echo "  ref OK: $f" || { echo "  ref MISSING: $f (run setup / wait for index)"; exit 1; }
  done
  tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r cl smp run rest; do
    [ -z "${run:-}" ] && continue
    [ -s "$WORK/sra/$run/$run.sra" ] && echo "  sra OK: $cl $run" || { echo "  sra MISSING: $cl $run (run setup)"; exit 1; }
  done
}

process_one() {
  local cl="$1" run="$2"
  local fq1="$WORK/fastq/${run}_1.fastq" fq2="$WORK/fastq/${run}_2.fastq"
  local pref="$WORK/bam/${cl}." bam
  echo "[$cl] fasterq-dump (offline, full depth)"
  "$ENVBIN/fasterq-dump" --split-files -e "$THREADS" -t "$WORK/tmp" \
      -O "$WORK/fastq" "$WORK/sra/$run/$run.sra"
  echo "[$cl] STAR (multimappers retained)"
  "$ENVBIN/STAR" --runThreadN "$THREADS" --genomeDir "$STAR_INDEX" \
      --readFilesIn "$fq1" "$fq2" \
      --outSAMtype BAM Unsorted --outFileNamePrefix "$pref" \
      --outFilterMultimapNmax 100 --winAnchorMultimapNmax 100 \
      --outSAMprimaryFlag AllBestScore
  rm -f "$fq1" "$fq2"
  bam="${pref}Aligned.out.bam"
  echo "[$cl] TEcount"
  "$ENVBIN/TEcount" --mode multi --format BAM --sortByPos \
      -b "$bam" --GTF "$GENE_GTF" --TE "$TE_GTF" \
      --project "$WORK/counts/${cl}"
  rm -f "$bam"
  echo "[$cl] done -> $WORK/counts/${cl}.cntTable"
}

main() {
  preflight
  echo; echo "=== full-depth requant, one sample at a time ==="
  tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r cl smp run rest; do
    [ -z "${run:-}" ] && continue
    process_one "$cl" "$run"
  done
  echo; echo "=== HERVK subfamily counts across the three lines ==="
  "$ENVBIN/python" - "$WORK/counts" <<'PY'
import sys, glob, os, csv
d=sys.argv[1]
want={"HERVK-int","HERVK9-int","HERVK11-int","LTR5","LTR5_Hs","LTR5A","LTR5B"}
tables={}
for f in sorted(glob.glob(os.path.join(d,"*.cntTable"))):
    cl=os.path.basename(f).split(".")[0]
    for row in csv.reader(open(f), delimiter="\t"):
        if len(row)<2: continue
        feat=row[0]; sub=feat.split(":")[0]
        if sub in want or feat in want:
            tables.setdefault(sub,{})[cl]=row[1]
cols=sorted({c for v in tables.values() for c in v})
print("subfamily\t"+"\t".join(cols))
for feat in sorted(tables):
    print(feat+"\t"+"\t".join(tables[feat].get(c,"0") for c in cols))
PY
}

if [ "${1:-}" = "--run" ]; then main; exit 0; fi
preflight   # fail fast on the login node before submitting
LOG="$WORK/hervk_ccle.%J.log"
bsub -q "$QUEUE" -n "$THREADS" -M "$MEM_MB" -R "rusage[mem=$MEM_MB]" \
     -J hervk_ccle -o "$LOG" -e "$LOG" \
     bash "$(cd "$(dirname "$0")" && pwd)/$(basename "$0")" --run
echo "submitted; log -> $LOG"
