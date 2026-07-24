#!/usr/bin/env bash
# submit_hervk_ccle_requant.sh
# TE-aware HERVK requantification for MCF-7, PC-3, PANC-1 from public CCLE
# RNA-seq (SRP186687 / PRJNA523380). Runs FULL DEPTH but processes one sample
# at a time and deletes each intermediate as soon as it is consumed, so peak
# scratch use ~= a single sample's FASTQ+BAM (not all three at once).
#
# All tools run inside gameca.sif via `singularity exec`.
# Submit:  bash scripts/submit_hervk_ccle_requant.sh
# (or run the body directly on an interactive node for testing)
set -euo pipefail

# ---- configuration (edit to your cluster) --------------------------------
SIF=${SIF:-$HOME/gameca.sif}
WORK=${WORK:-${SCRATCH:-$HOME}/hervk_ccle}
THREADS=${THREADS:-12}
MEM_MB=${MEM_MB:-40000}                 # STAR needs ~32-40 GB for hg38
QUEUE=${QUEUE:-normal}

# reference bundle (set these; build once, reuse) --------------------------
STAR_INDEX=${STAR_INDEX:-$WORK/ref/star_hg38}          # STAR genomeDir
GENE_GTF=${GENE_GTF:-$WORK/ref/gencode.v44.hg38.gtf}   # gene models
TE_GTF=${TE_GTF:-$WORK/ref/GRCh38_Ensembl_rmsk_TE.gtf} # <- carries HERVK-int/LTR5/LTR5_Hs labels
MANIFEST=${MANIFEST:-$(cd "$(dirname "$0")" && pwd)/hervk_ccle_manifest.tsv}

mkdir -p "$WORK" "$WORK/fastq" "$WORK/bam" "$WORK/counts" "$WORK/ref"
SEXEC=(singularity exec "$SIF")

# ---- preflight (also runnable standalone; see chat) ----------------------
preflight() {
  echo "== RAM ==";     free -h | awk '/Mem/{print "  "$2" total, "$7" available"}' || true
  echo "== Scratch =="; df -h "$WORK" | awk 'NR==2{print "  "$4" free on "$6}'
  echo "== Container =="; ls -lh "$SIF" 2>/dev/null || { echo "  MISSING: $SIF"; exit 1; }
  for t in prefetch fasterq-dump STAR TEcount samtools; do
    "${SEXEC[@]}" bash -lc "command -v $t" >/dev/null 2>&1 \
      && echo "  $t: OK" || echo "  $t: MISSING in sif (add to Dockerfile)"
  done
  for f in "$STAR_INDEX" "$GENE_GTF" "$TE_GTF"; do
    [ -e "$f" ] && echo "  ref OK: $f" || echo "  ref MISSING: $f"
  done
}

# ---- per-sample worker: download -> align -> count -> clean --------------
process_one() {
  local cl="$1" run="$2"
  local fq1="$WORK/fastq/${run}_1.fastq" fq2="$WORK/fastq/${run}_2.fastq"
  local pref="$WORK/bam/${cl}."
  local bam="${pref}Aligned.out.bam"

  echo "[$cl] prefetch $run"
  "${SEXEC[@]}" prefetch -O "$WORK/sra" "$run"
  echo "[$cl] fasterq-dump (full depth)"
  "${SEXEC[@]}" fasterq-dump --split-files -e "$THREADS" \
      -t "$WORK/tmp" -O "$WORK/fastq" "$WORK/sra/$run/$run.sra"
  rm -rf "$WORK/sra/$run"                       # free .sra immediately

  echo "[$cl] STAR align (multimappers retained)"
  "${SEXEC[@]}" STAR \
      --runThreadN "$THREADS" --genomeDir "$STAR_INDEX" \
      --readFilesIn "$fq1" "$fq2" \
      --outSAMtype BAM Unsorted --outFileNamePrefix "$pref" \
      --outFilterMultimapNmax 100 --winAnchorMultimapNmax 100 \
      --outSAMprimaryFlag AllBestScore
  rm -f "$fq1" "$fq2"                            # free FASTQ immediately

  echo "[$cl] TEcount"
  "${SEXEC[@]}" TEcount --mode multi --format BAM --sortByPos \
      -b "$bam" --GTF "$GENE_GTF" --TE "$TE_GTF" \
      --project "$WORK/counts/${cl}"
  rm -f "$bam"                                   # free BAM immediately
  echo "[$cl] done -> $WORK/counts/${cl}.cntTable"
}

main() {
  preflight
  echo; echo "=== full-depth requant, one sample at a time ==="
  # skip header; columns: cell_line ccle_sample run ...
  tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r cl smp run rest; do
    [ -z "${run:-}" ] && continue
    process_one "$cl" "$run"
  done

  echo; echo "=== HERVK subfamily rows across the three lines ==="
  "${SEXEC[@]}" python3 - "$WORK/counts" <<'PY'
import sys, glob, os, csv
d=sys.argv[1]
want=("HERVK-int","LTR5","LTR5_Hs","LTR5B","LTR5A")
tables={}
for f in sorted(glob.glob(os.path.join(d,"*.cntTable"))):
    cl=os.path.basename(f).split(".")[0]
    for row in csv.reader(open(f), delimiter="\t"):
        if len(row)<2: continue
        feat,val=row[0],row[1]
        if feat.split(":")[0] in want or feat in want:
            tables.setdefault(feat,{})[cl]=val
cols=sorted({c for v in tables.values() for c in v})
print("feature\t"+"\t".join(cols))
for feat in sorted(tables):
    print(feat+"\t"+"\t".join(tables[feat].get(c,"NA") for c in cols))
PY
}

# If submitted via bsub the body runs on a compute node; -M/-n/-q below.
if [ "${1:-}" = "--run" ]; then main; exit 0; fi
BSUB_OUT="$WORK/hervk_ccle.%J.log"
bsub -q "$QUEUE" -n "$THREADS" -M "$MEM_MB" -R "rusage[mem=${MEM_MB}]" \
     -J hervk_ccle -o "$BSUB_OUT" -e "$BSUB_OUT" \
     bash "$(cd "$(dirname "$0")" && pwd)/$(basename "$0")" --run
echo "submitted; log -> $BSUB_OUT"
