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

# GAMECA_ENV: path to a file of `export VAR='value'` lines written at submit time.
# Job settings travel this way instead of on the bsub command line, because bsub
# re-emits the command as a STRING and only quotes arguments containing spaces --
# so a value like FILTER='^(LTR66|LTR10G)_dup' arrives unquoted and the node dies
# with a shell syntax error before any of this script runs.
if [ -n "${GAMECA_ENV:-}" ] && [ -r "${GAMECA_ENV}" ]; then
  # shellcheck source=/dev/null
  . "$GAMECA_ENV"
fi

WORK=${WORK:-$HOME/hervk_ccle}
REF=${REF:-$WORK/ref}
CONT=${CONT:-$WORK/containers}
THREADS=${THREADS:-12}
MEM_MB=${MEM_MB:-45000}
QUEUE=${QUEUE:-rhel9}
WALL=${WALL:-24:00}
JOB=${JOB:-locus_expr}
FILTER=${FILTER:-'HERVK|LTR5'}    # keep only these loci in the merged table
# ALSO_TECOUNT=1 runs TEcount on the SAME BAM before it is deleted, so one STAR
# pass yields both per-locus counts and the subfamily-level .cntTable. Without
# it you must also run submit_hervk_ccle_requant.sh, i.e. align every sample
# twice. Default off so the existing HERVK flow is unchanged.
ALSO_TECOUNT=${ALSO_TECOUNT:-0}
# Strandedness. Both defaults below are the tools' own defaults (unstranded);
# set STRANDED=reverse + FC_STRAND=2 for a dUTP/TruSeq-stranded library.
STRANDED=${STRANDED:-no}          # TEcount: no | forward | reverse
FC_STRAND=${FC_STRAND:-0}         # featureCounts -s: 0 | 1 | 2
STAR_SIF=$CONT/star.sif
SUB_SIF=$CONT/subread.sif
TE_SIF=$CONT/tetranscripts.sif
STAR_INDEX=$REF/star_hg38
TE_GTF=$REF/hg38_rmsk_TE.gtf
GENE_GTF=$REF/hg38.knownGene.gtf
MANIFEST=${MANIFEST:-$(cd "$(dirname "$0")" && pwd)/hervk_ccle_manifest.tsv}
mkdir -p "$WORK/locus" "$WORK/bam" "$CONT"
[ "$ALSO_TECOUNT" = "1" ] && mkdir -p "$WORK/counts"
# Resolve singularity/apptainer on THIS host (login and compute nodes differ --
# a missing runtime is what makes every array element exit 127).
# shellcheck source=/dev/null
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_container.sh"
CONTAINER_BIND="${CONTAINER_BIND:-}${CONTAINER_BIND:+:}$WORK:$REF:$CONT"
container_module_load
container_init || exit 1

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
  local cnt="$WORK/counts/${cl}.cntTable"
  # only skip when EVERY requested output is already there
  if [ -s "$out" ] && { [ "$ALSO_TECOUNT" != "1" ] || [ -s "$cnt" ]; }; then
    echo "[$cl] outputs exist -> skip"; exit 0
  fi
  echo "[$cl] STAR align"
  sing "$STAR_SIF" STAR --runThreadN "$THREADS" --genomeDir "$STAR_INDEX" \
      --readFilesIn "$fq1" "$fq2" --readFilesCommand zcat \
      --outSAMtype BAM Unsorted --outFileNamePrefix "$pref" \
      --outFilterMultimapNmax 100 --winAnchorMultimapNmax 100 --outSAMprimaryFlag AllBestScore
  if [ ! -s "$out" ]; then
    echo "[$cl] featureCounts per locus (-s $FC_STRAND)"
    sing "$SUB_SIF" featureCounts -a "$TE_GTF" -o "$out" \
        -M --fraction -O -T "$THREADS" -p --countReadPairs -s "$FC_STRAND" \
        -t exon -g transcript_id "$bam"
  fi
  if [ "$ALSO_TECOUNT" = "1" ] && [ ! -s "$cnt" ]; then
    echo "[$cl] TEcount subfamily level (--stranded $STRANDED) on the same BAM"
    sing "$TE_SIF" TEcount --mode multi --format BAM --sortByPos \
        --stranded "$STRANDED" -b "$bam" --GTF "$GENE_GTF" --TE "$TE_GTF" \
        --project "$WORK/counts/${cl}"
  fi
  rm -f "$bam"
  if [ "$ALSO_TECOUNT" = "1" ]; then echo "[$cl] done -> $out + $cnt"
  else echo "[$cl] done -> $out"; fi
}
if [ "${1:-}" = "--run-one" ]; then run_one; exit 0; fi

# --- submit ----------------------------------------------------------------
[ -s "$SUB_SIF" ] || { echo "pulling subread container..."; \
  curl -fSL -o "$SUB_SIF" https://depot.galaxyproject.org/singularity/subread:2.0.6--he4a0461_2; }
# Containers and GTFs come from the (synchronous) setup step, so they must
# exist now no matter what.
for f in "$STAR_SIF" "$SUB_SIF" "$TE_GTF"; do
  [ -e "$f" ] || { echo "MISSING $f"; exit 1; }
done
if [ "$ALSO_TECOUNT" = "1" ]; then
  for f in "$TE_SIF" "$GENE_GTF"; do
    [ -e "$f" ] || { echo "MISSING $f"; exit 1; }
  done
fi
# The STAR index and the FASTQ, by contrast, are produced by OTHER LSF jobs.
# REQUIRE_INPUTS=0 downgrades them to warnings so this array can be submitted
# with a -w dependency on those jobs before they have run.
REQUIRE_INPUTS=${REQUIRE_INPUTS:-${REQUIRE_FASTQ:-1}}
miss=0
[ -e "$STAR_INDEX/SAindex" ] || { echo "  STAR index not built yet: $STAR_INDEX"; miss=1; }
while IFS=$'\t' read -r cl _ run _; do
  [ -z "${run:-}" ] && continue
  { [ -s "$WORK/fastq/${run}_1.fastq.gz" ] && [ -s "$WORK/fastq/${run}_2.fastq.gz" ]; } \
    || { echo "  fastq not yet present: $cl $run"; miss=1; }
done < <(tail -n +2 "$MANIFEST")
if [ "$miss" = 1 ]; then
  [ "$REQUIRE_INPUTS" = "1" ] && \
    { echo "build the index / download the FASTQ first (or REQUIRE_INPUTS=0)"; exit 1; }
  echo "  REQUIRE_INPUTS=0 -> submitting anyway (expects upstream jobs to produce these)"
fi
N=$(($(grep -c . "$MANIFEST") - 1))
LOG="$WORK/${JOB}.%J_%I.log"
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
# BSUB_DEP: an LSF dependency EXPRESSION, e.g. 'done(ena_dl) && done(star_index)'.
# It must reach bsub as a single argument, so it is never word-split.
dep_args=(); [ -n "${BSUB_DEP:-}" ] && dep_args=( -w "$BSUB_DEP" )
# Job settings go in a FILE, not on the bsub command line -- see GAMECA_ENV at the
# top. printf %q quotes each value, and only the (metacharacter-free) file path
# travels through bsub.
ENVFILE="$WORK/${JOB}.env"
{
  # NB: GAMECA_RT is deliberately NOT forwarded -- the compute node must resolve
  # its own container binary; the login node's path may not exist there.
  for v in WORK REF CONT MANIFEST THREADS ALSO_TECOUNT STRANDED FC_STRAND FILTER \
           CONTAINER_MODULE; do
    eval "_val=\${$v:-}"
    [ -n "$_val" ] && printf 'export %s=%q\n' "$v" "$_val"
  done
} > "$ENVFILE"
echo "  job settings -> $ENVFILE"
bsub -q "$QUEUE" -n "$THREADS" -M "$MEM_MB" -R "rusage[mem=$MEM_MB]" -W "$WALL" \
     ${dep_args[@]+"${dep_args[@]}"} \
     -J "${JOB}[1-$N]" -o "$LOG" -e "$LOG" \
     env GAMECA_ENV="$ENVFILE" bash "$SELF" --run-one
echo "submitted ${JOB}[1-$N]; when DONE run: MANIFEST=$MANIFEST WORK=$WORK bash scripts/submit_locus_expression.sh --merge"
