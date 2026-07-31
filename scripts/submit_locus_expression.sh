#!/usr/bin/env bash
# submit_locus_expression.sh --- per-LOCUS TE expression for the CCLE samples,
# so reagent design can weight by which HERVK copies are actually expressed
# (TEcount gave only subfamily-level totals).
#
# Per sample (LSF array over hervk_ccle_manifest.tsv): STAR-align the RNA FASTQ
# (multimappers retained) then featureCounts per rmsk locus (-M --fraction -O,
# -g transcript_id = one count per copy). Then `--merge` combines the samples
# into locus_expression.tsv (CPM per sample + mean), filtered to HERVK/LTR5 loci.
#
#   bash scripts/submit_locus_expression.sh            # submit the 3-sample array
#   bash scripts/submit_locus_expression.sh --merge    # after it finishes
set -euo pipefail

WORK=${WORK:-$HOME/hervk_ccle}
REF=${REF:-$WORK/ref}
CONT=${CONT:-$WORK/containers}
THREADS=${THREADS:-12}
MEM_MB=${MEM_MB:-45000}
QUEUE=${QUEUE:-rhel9}
WALL=${WALL:-24:00}
JOB=${JOB:-locus_expr}
FILTER=${FILTER:-'HERVK|LTR5'}    # keep only these loci in the merged table
STAR_SIF=$CONT/star.sif
SUB_SIF=$CONT/subread.sif
STAR_INDEX=$REF/star_hg38
TE_GTF=$REF/hg38_rmsk_TE.gtf
MANIFEST=${MANIFEST:-$(cd "$(dirname "$0")" && pwd)/hervk_ccle_manifest.tsv}
mkdir -p "$WORK/locus" "$WORK/bam" "$CONT"
sing() { singularity exec -B "$WORK" "$@"; }

# --- merge mode: combine per-sample featureCounts into locus_expression.tsv ---
if [ "${1:-}" = "--merge" ]; then
  python3 - "$WORK/locus" "$MANIFEST" "$FILTER" > "$WORK/locus_expression.tsv" <<'PY'
import sys, os, csv, re
locdir, manifest, filt = sys.argv[1], sys.argv[2], re.compile(sys.argv[3])
cells = [l.split("\t")[0] for l in open(manifest).read().splitlines()[1:] if l.strip()]
per = {}          # locus -> {chrom,start,end, cell: cpm}
meta = {}
for cell in cells:
    fc = os.path.join(locdir, f"{cell}.featureCounts.txt")
    if not os.path.exists(fc):
        sys.stderr.write(f"missing {fc}\n"); continue
    rows = [r for r in csv.reader(open(fc), delimiter="\t") if r and not r[0].startswith("#")]
    hdr = rows[0]; cnt_i = len(hdr) - 1              # last col = the BAM's counts
    counts = {}
    for r in rows[1:]:
        gid = r[0]
        try: c = float(r[cnt_i])
        except (ValueError, IndexError): continue
        counts[gid] = c
        meta.setdefault(gid, (r[1].split(";")[0], r[2].split(";")[0], r[3].split(";")[-1]))
    lib = sum(counts.values()) or 1.0
    for gid, c in counts.items():
        if filt.search(gid):
            per.setdefault(gid, {})[cell] = c / lib * 1e6
out = csv.writer(sys.stdout, delimiter="\t")
out.writerow(["locus", "chrom", "start", "end"] + [f"cpm_{c}" for c in cells] + ["cpm_mean"])
for gid in sorted(per):
    chrom, start, end = meta.get(gid, ("", "", ""))
    vals = [per[gid].get(c, 0.0) for c in cells]
    out.writerow([gid, chrom, start, end] + [f"{v:.4f}" for v in vals] + [f"{sum(vals)/len(vals):.4f}"])
sys.stderr.write(f"[merge] wrote {len(per)} {sys.argv[3]} loci across {len(cells)} samples\n")
PY
  echo "wrote $WORK/locus_expression.tsv"
  exit 0
fi

# --- per-sample worker ------------------------------------------------------
run_one() {
  local idx="${LSB_JOBINDEX:?}"
  local row cl run
  row=$(sed -n "$((idx+1))p" "$MANIFEST")
  cl=$(printf '%s' "$row" | cut -f1); run=$(printf '%s' "$row" | cut -f3)
  [ -z "$run" ] && exit 0
  local fq1="$WORK/fastq/${run}_1.fastq.gz" fq2="$WORK/fastq/${run}_2.fastq.gz"
  local pref="$WORK/bam/${cl}.locus." bam="$WORK/bam/${cl}.locus.Aligned.out.bam"
  local out="$WORK/locus/${cl}.featureCounts.txt"
  [ -s "$out" ] && { echo "[$cl] featureCounts exists -> skip"; exit 0; }
  echo "[$cl] STAR align"
  sing "$STAR_SIF" STAR --runThreadN "$THREADS" --genomeDir "$STAR_INDEX" \
      --readFilesIn "$fq1" "$fq2" --readFilesCommand zcat \
      --outSAMtype BAM Unsorted --outFileNamePrefix "$pref" \
      --outFilterMultimapNmax 100 --winAnchorMultimapNmax 100 --outSAMprimaryFlag AllBestScore
  echo "[$cl] featureCounts per locus"
  sing "$SUB_SIF" featureCounts -a "$TE_GTF" -o "$out" \
      -M --fraction -O -T "$THREADS" -p --countReadPairs \
      -t exon -g transcript_id "$bam"
  rm -f "$bam"
  echo "[$cl] done -> $out"
}
if [ "${1:-}" = "--run-one" ]; then run_one; exit 0; fi

# --- submit ----------------------------------------------------------------
[ -s "$SUB_SIF" ] || { echo "pulling subread container..."; \
  curl -fSL -o "$SUB_SIF" https://depot.galaxyproject.org/singularity/subread:2.0.6--he4a0461_2; }
for f in "$STAR_SIF" "$SUB_SIF" "$STAR_INDEX/SAindex" "$TE_GTF"; do
  [ -e "$f" ] || { echo "MISSING $f"; exit 1; }
done
N=$(($(grep -c . "$MANIFEST") - 1))
LOG="$WORK/${JOB}.%J_%I.log"
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
bsub -q "$QUEUE" -n "$THREADS" -M "$MEM_MB" -R "rusage[mem=$MEM_MB]" -W "$WALL" \
     -J "${JOB}[1-$N]" -o "$LOG" -e "$LOG" bash "$SELF" --run-one
echo "submitted ${JOB}[1-$N]; when DONE run: bash scripts/submit_locus_expression.sh --merge"
