#!/usr/bin/env bash
# submit_hervk_ccle_requant.sh
# TE-aware HERVK requantification, submitted as an LSF JOB ARRAY (one element
# per manifest row, run in parallel). STAR (multimappers retained) + TEcount;
# each BAM deleted as soon as counted, gz FASTQ kept. Resumable: an element
# whose <label>.cntTable already exists is skipped.
#
# Works for any study via env overrides (defaults = CCLE in $HOME/hervk_ccle):
#   WORK / MANIFEST / REF / CONT / JOB / WALL / QUEUE / MEM_MB / THREADS
# e.g. GSE157927 reusing the CCLE index+containers:
#   WORK=$HOME/gse157927 MANIFEST=.../gse157927_manifest.tsv \
#   REF=$HOME/hervk_ccle/ref CONT=$HOME/hervk_ccle/containers JOB=gse_requant \
#   bash scripts/submit_hervk_ccle_requant.sh
#
# Prereq: refs/containers built (setup_hervk_ccle.sh) and all FASTQ downloaded.
# Compute node needs NO internet: STAR reads pre-downloaded .gz directly.
# After the array finishes: bash scripts/hervk_normalize_plot.sh (table+figure).
set -euo pipefail

WORK=${WORK:-$HOME/hervk_ccle}
REF=${REF:-$WORK/ref}            # override to reuse another study's STAR index/GTFs
CONT=${CONT:-$WORK/containers}   # override to reuse another study's containers
THREADS=${THREADS:-12}
MEM_MB=${MEM_MB:-45000}          # STAR needs ~32-40 GB for hg38
QUEUE=${QUEUE:-rhel9}            # PennHPC/PMACS default queue post-Q3-2026 (lowercase)
LSF_SELECT=${LSF_SELECT:-}       # optional select[]; not needed on PennHPC
WALL=${WALL:-72:00}             # LSF wall clock HH:MM per array element
JOB=${JOB:-requant}             # LSF job-array name (set per study to avoid clashes)

STAR_SIF=$CONT/star.sif
TE_SIF=$CONT/tetranscripts.sif
STAR_INDEX=$REF/star_hg38
GENE_GTF=$REF/hg38.knownGene.gtf
TE_GTF=$REF/hg38_rmsk_TE.gtf
MANIFEST=${MANIFEST:-$(cd "$(dirname "$0")" && pwd)/hervk_ccle_manifest.tsv}
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
  if [ -s "$WORK/counts/${cl}.cntTable" ]; then
    echo "[$cl] already counted -> skip (delete cntTable to redo)"; return 0
  fi
  for f in "$fq1" "$fq2" "$STAR_INDEX/SAindex" "$GENE_GTF" "$TE_GTF" "$STAR_SIF" "$TE_SIF"; do
    [ -e "$f" ] || { echo "[$cl] MISSING $f"; exit 1; }
  done
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

# one array element: process the LSB_JOBINDEX-th data row of the manifest
run_one() {
  local idx="${LSB_JOBINDEX:?--run-one must be an LSF array element}"
  local row cl run
  row=$(sed -n "$((idx+1))p" "$MANIFEST")     # +1 skips header
  cl=$(printf '%s' "$row" | cut -f1)
  run=$(printf '%s' "$row" | cut -f3)
  [ -z "$run" ] && { echo "no run at index $idx"; exit 0; }
  echo "== array[$idx] $cl $run on $(hostname) =="
  process_one "$cl" "$run"
}

if [ "${1:-}" = "--run-one" ]; then run_one; exit 0; fi
if [ "${1:-}" = "--run" ]; then                # legacy sequential path
  preflight; tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r cl smp run rest; do
    [ -z "${run:-}" ] && continue; process_one "$cl" "$run"; done; exit 0
fi

# ---- submit as an LSF job array: one element per manifest row --------------
preflight   # fail fast: refs, containers, and every FASTQ present
N=$(($(grep -c . "$MANIFEST") - 1))            # data rows (excl header)
bsub_args=( -n "$THREADS" -M "$MEM_MB" -R "rusage[mem=$MEM_MB]" -W "$WALL" )
[ -n "$QUEUE" ]      && bsub_args+=( -q "$QUEUE" )
[ -n "$LSF_SELECT" ] && bsub_args+=( -R "select[$LSF_SELECT]" )
LOG="$WORK/${JOB}.%J_%I.log"
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
bsub "${bsub_args[@]}" -J "${JOB}[1-$N]" -o "$LOG" -e "$LOG" \
     bash "$SELF" --run-one
echo "submitted array ${JOB}[1-$N]  (${bsub_args[*]})"
echo "  watch:  bjobs -J $JOB       logs: $WORK/${JOB}.*_*.log"
echo "  when all elements DONE:  bash scripts/hervk_normalize_plot.sh   (builds the table + figure)"
