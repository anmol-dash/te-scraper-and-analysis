#!/usr/bin/env bash
# setup_hervk_ccle.sh  --  RUN ON THE LOGIN NODE (needs internet).
# Provisions everything the requant job needs, using only stable URLs.
#
# NOTE ON GLIBC: this cluster is old-glibc (RHEL7, glibc 2.17). Modern bioconda
# BINARIES require glibc>=2.28 and segfault here (prefetch: GLIBC_2.27 not found).
# So the compiled tools run from prebuilt Singularity images (Galaxy depot),
# which carry their own glibc -- host glibc no longer matters. Singularity
# already works here (it runs gameca.sif).
#
#   1. pull sra-tools / STAR / TEcount Singularity images
#   2. references from UCSC (hg38 fasta, knownGene GTF) + TE GTF generated
#      deterministically from UCSC rmsk.txt.gz (no Dropbox link-rot)
#   3. SRA .sra files pre-downloaded here (compute node needs no internet)
#   4. submit the STAR genome-index build as an LSF job
#
# After this finishes and the star_index job completes:
#   bash scripts/submit_hervk_ccle_requant.sh
set -euo pipefail

WORK=${WORK:-$HOME/hervk_ccle}
# REF/CONT are overridable so a second study can point at an already-built
# hg38 + STAR index + container set instead of downloading its own copy.
REF=${REF:-$WORK/ref}
CONT=${CONT:-$WORK/containers}
THREADS=${THREADS:-12}
MEM_MB=${MEM_MB:-45000}
# QUEUE: PennHPC/PMACS post-2026-Q3-maintenance default is 'rhel9' (lowercase).
# LSF_SELECT: optional LSF select[] expr; not needed on PennHPC (rhel9 is a queue).
QUEUE=${QUEUE:-rhel9}
LSF_SELECT=${LSF_SELECT:-}
JOB=${JOB:-ena_dl}          # name of the download job (inherited by download_fastq_ena.sh)
MANIFEST=${MANIFEST:-$(cd "$(dirname "$0")" && pwd)/hervk_ccle_manifest.tsv}
# Subfamilies to report as a sanity check on the generated TE GTF (ERE-style
# regex). Override per study, e.g. FAM_GREP='^(LTR66|LTR10G)$'.
FAM_GREP=${FAM_GREP:-'^(HERVK.*-int|LTR5|LTR5_Hs|LTR5A|LTR5B)$'}
# assemble bsub resource/queue args once
bsub_args=( -n "$THREADS" -M "$MEM_MB" -R "rusage[mem=$MEM_MB]" )
[ -n "$QUEUE" ]      && bsub_args+=( -q "$QUEUE" )
[ -n "$LSF_SELECT" ] && bsub_args+=( -R "select[$LSF_SELECT]" )
mkdir -p "$WORK" "$REF" "$CONT" "$WORK/fastq"

DEPOT=https://depot.galaxyproject.org/singularity
STAR_SIF=$CONT/star.sif
TE_SIF=$CONT/tetranscripts.sif
LIBC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_container.sh"
# shellcheck source=/dev/null
. "$LIBC"
CONTAINER_BIND="${CONTAINER_BIND:-}${CONTAINER_BIND:+:}$WORK:$REF:$CONT"
container_module_load
container_init || exit 1
# FASTQ come straight from EBI ENA (deterministic HTTP, gzipped) -- no sra-tools
# (its 3.4.1 build segfaults on exit here even though the download succeeds).

# --- 1. tool containers (carry their own glibc) ---------------------------
pull() {  # $1=target sif  $2=depot image name
  [ -s "$1" ] && { echo "  have $(basename "$1")"; return; }
  echo "  pull $2"
  curl -fSL -o "$1" "$DEPOT/$2"
}
echo "[setup] pulling tool images ..."
pull "$STAR_SIF" "star:2.7.11b--h5ca1c30_8"
pull "$TE_SIF"   "tetranscripts:2.2.4--pyh106432d_0"
echo "[setup] tool check (runtime, inside containers):"
sing "$STAR_SIF" STAR --version
sing "$TE_SIF" TEcount --version 2>&1 | head -1

# --- 2. references --------------------------------------------------------
cd "$REF"
UCSC=https://hgdownload.soe.ucsc.edu/goldenPath/hg38
echo "[setup] hg38 fasta ..."
[ -s hg38.fa ] || { curl -fSL -O "$UCSC/bigZips/hg38.fa.gz" && gunzip -f hg38.fa.gz; }
echo "[setup] gene GTF (UCSC knownGene) ..."
[ -s hg38.knownGene.gtf ] || { curl -fSL -O "$UCSC/bigZips/genes/hg38.knownGene.gtf.gz" && gunzip -f hg38.knownGene.gtf.gz; }
echo "[setup] TE GTF from UCSC rmsk.txt.gz ..."
if [ ! -s hg38_rmsk_TE.gtf ]; then
  curl -fSL -O "$UCSC/database/rmsk.txt.gz"
  # rmsk cols: 6=chrom 7=start(0-based) 8=end 10=strand 11=repName 12=repClass 13=repFamily
  zcat rmsk.txt.gz | awk -F'\t' '
    $12 ~ /Simple_repeat|Low_complexity|Satellite|^RNA$|rRNA|scRNA|snRNA|srpRNA|tRNA|Unknown|\?/ {next}
    { n[$11]++;
      printf "%s\trmsk\texon\t%d\t%d\t.\t%s\t.\tgene_id \"%s\"; transcript_id \"%s_dup%d\"; family_id \"%s\"; class_id \"%s\";\n",
             $6,$7+1,$8,$10,$11,$11,n[$11],$13,$12 }' > hg38_rmsk_TE.gtf
fi
echo "  TE GTF: $(wc -l < hg38_rmsk_TE.gtf) lines; subfamilies of interest present:"
# `|| true`: with pipefail a non-matching grep would otherwise abort the script.
# A miss here means the repName is wrong or was dropped by the class filter --
# report it loudly rather than dying three lines later for no obvious reason.
fam_counts=$(awk -F'gene_id "' '{split($2,a,"\"");print a[1]}' hg38_rmsk_TE.gtf \
  | grep -E "$FAM_GREP" | sort | uniq -c || true)
if [ -n "$fam_counts" ]; then
  echo "$fam_counts" | sed 's/^/  /'
else
  echo "  WARNING: no gene_id in the TE GTF matches /$FAM_GREP/ --"
  echo "  check the repName spelling; downstream counting will find nothing."
fi

# --- 3. submit STAR index build first (runs on a node while we download) ---
IDX=$REF/star_hg38
if [ -s "$IDX/SAindex" ]; then
  echo "[setup] STAR index already built at $IDX"
elif bjobs -J star_index 2>/dev/null | grep -qE '\b(PEND|RUN)\b'; then
  # don't submit a 2nd build into the same genomeDir (would corrupt it)
  echo "[setup] STAR index job already PEND/RUN (bjobs -J star_index); not resubmitting"
else
  mkdir -p "$IDX"
  cat > "$WORK/build_index.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
# resolve the container runtime ON THE COMPUTE NODE -- it may be apptainer there,
# or absent from the batch PATH entirely (that shows up as exit code 127)
. "$LIBC"
CONTAINER_BIND="$WORK:$REF"
container_module_load
container_init || exit 1
sing "$STAR_SIF" STAR --runMode genomeGenerate \\
  --runThreadN $THREADS --genomeDir "$IDX" --genomeFastaFiles "$REF/hg38.fa" \\
  --sjdbGTFfile "$REF/hg38.knownGene.gtf" --sjdbOverhang 100
echo "STAR index done: $IDX"
EOF
  chmod +x "$WORK/build_index.sh"
  echo "[setup] submitting STAR index build (${bsub_args[*]}) ..."
  bsub "${bsub_args[@]}" \
       -J star_index -o "$WORK/star_index.%J.log" -e "$WORK/star_index.%J.log" \
       bash "$WORK/build_index.sh"
  echo "  -> watch: bjobs -J star_index ; log: $WORK/star_index.*.log"
fi

# --- 4. submit the ENA FASTQ download as its own LSF job (aria2, parallel) --
# PC-3 ~87 GB and PANC-1 ~64 GB gz -- long + TLS-drop-prone, so it runs on a
# compute node via bsub (aria2c -x16, native resume; curl fallback).
DL=$(cd "$(dirname "$0")" && pwd)/download_fastq_ena.sh
echo "[setup] submitting ENA FASTQ download job ..."
# these are plain shell vars here, so hand them over explicitly -- otherwise the
# download job silently falls back to its own default (CCLE) manifest/WORK.
env MANIFEST="$MANIFEST" WORK="$WORK" CONT="$CONT" QUEUE="$QUEUE" bash "$DL"

n_runs=$(($(grep -c . "$MANIFEST") - 1))
echo; echo "[setup] DONE. Two jobs are now running: star_index + $JOB."
echo "  watch:  bjobs"
echo "  when the index is built AND all $((n_runs * 2)) fastq.gz are present, run:"
echo "  bash scripts/submit_hervk_ccle_requant.sh"
