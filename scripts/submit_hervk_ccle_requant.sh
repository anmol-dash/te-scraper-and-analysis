#!/usr/bin/env bash
# submit_hervk_ccle_requant.sh
# TE-aware HERVK requantification for MCF-7, PC-3, PANC-1 from public CCLE
# RNA-seq (SRP186687 / PRJNA523380). FULL DEPTH, one sample at a time; BAMs
# are deleted as soon as counted (gz FASTQ kept for re-runs).
#
# Prereq: run scripts/setup_hervk_ccle.sh first (login node; downloads gz FASTQ
# from ENA) and let the star_index job finish. Tools run from Singularity images
# (host is old-glibc, so bioconda binaries won't run natively). Compute node
# needs NO internet: FASTQ were pre-downloaded; STAR reads the .gz directly.
#
# Submit:  bash scripts/submit_hervk_ccle_requant.sh
set -euo pipefail

WORK=${WORK:-$HOME/hervk_ccle}
REF=$WORK/ref
CONT=$WORK/containers
THREADS=${THREADS:-12}
MEM_MB=${MEM_MB:-45000}          # STAR needs ~32-40 GB for hg38
QUEUE=${QUEUE:-}                 # real queue from `bqueues` (empty -> LSF default)
LSF_SELECT=${LSF_SELECT:-}       # e.g. "rhel90" to force RHEL9 nodes (see `lshosts -w`)

STAR_SIF=$CONT/star.sif
TE_SIF=$CONT/tetranscripts.sif
STAR_INDEX=$REF/star_hg38
GENE_GTF=$REF/hg38.knownGene.gtf
TE_GTF=$REF/hg38_rmsk_TE.gtf
MANIFEST=$(cd "$(dirname "$0")" && pwd)/hervk_ccle_manifest.tsv
mkdir -p "$WORK/fastq" "$WORK/bam" "$WORK/counts" "$WORK/tmp"
sing() { singularity exec -B "$WORK" "$@"; }

preflight() {
  echo "== preflight =="
  for s in "$STAR_SIF" "$TE_SIF"; do
    [ -s "$s" ] && echo "  img OK: $(basename "$s")" || { echo "  img MISSING: $s (run setup)"; exit 1; }
  done
  for f in "$STAR_INDEX/SAindex" "$GENE_GTF" "$TE_GTF"; do
    [ -e "$f" ] && echo "  ref OK: $f" || { echo "  ref MISSING: $f (run setup / wait for index)"; exit 1; }
  done
  tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r cl smp run rest; do
    [ -z "${run:-}" ] && continue
    { [ -s "$WORK/fastq/${run}_1.fastq.gz" ] && [ -s "$WORK/fastq/${run}_2.fastq.gz" ]; } \
      && echo "  fastq OK: $cl $run" || { echo "  fastq MISSING: $cl $run (run setup)"; exit 1; }
  done
}

process_one() {
  local cl="$1" run="$2"
  local fq1="$WORK/fastq/${run}_1.fastq.gz" fq2="$WORK/fastq/${run}_2.fastq.gz"
  local pref="$WORK/bam/${cl}." bam
  echo "[$cl] STAR (multimappers retained; reads gz directly)"
  sing "$STAR_SIF" STAR --runThreadN "$THREADS" --genomeDir "$STAR_INDEX" \
      --readFilesIn "$fq1" "$fq2" --readFilesCommand zcat \
      --outSAMtype BAM Unsorted --outFileNamePrefix "$pref" \
      --outFilterMultimapNmax 100 --winAnchorMultimapNmax 100 \
      --outSAMprimaryFlag AllBestScore
  bam="${pref}Aligned.out.bam"
  echo "[$cl] TEcount"
  sing "$TE_SIF" TEcount --mode multi --format BAM --sortByPos \
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
  sing "$TE_SIF" python - "$WORK/counts" <<'PY'
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
preflight   # fail fast before submitting
bsub_args=( -n "$THREADS" -M "$MEM_MB" -R "rusage[mem=$MEM_MB]" )
[ -n "$QUEUE" ]      && bsub_args+=( -q "$QUEUE" )
[ -n "$LSF_SELECT" ] && bsub_args+=( -R "select[$LSF_SELECT]" )
LOG="$WORK/hervk_ccle.%J.log"
bsub "${bsub_args[@]}" -J hervk_ccle -o "$LOG" -e "$LOG" \
     bash "$(cd "$(dirname "$0")" && pwd)/$(basename "$0")" --run
echo "submitted (${bsub_args[*]}); log -> $LOG"
