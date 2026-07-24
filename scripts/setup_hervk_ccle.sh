#!/usr/bin/env bash
# setup_hervk_ccle.sh  --  RUN ON THE LOGIN NODE (needs internet).
# Provisions everything the requant job needs, using only stable URLs:
#   1. a self-contained micromamba env with the tools (sif has none of them,
#      and the base conda is broken: libmamba/libicui18n.so.75 missing)
#   2. references from UCSC (hg38 fasta, knownGene GTF) + a TE GTF generated
#      deterministically from UCSC rmsk.txt.gz (no Dropbox link-rot)
#   3. SRA .sra files pre-downloaded here (so the compute node needs no net)
#   4. submits the STAR genome-index build as an LSF job
#
# After this finishes and the index job completes:
#   bash scripts/submit_hervk_ccle_requant.sh
set -euo pipefail

WORK=${WORK:-$HOME/hervk_ccle}
ENV=$WORK/env
ENVBIN=$ENV/bin
REF=$WORK/ref
THREADS=${THREADS:-12}
MEM_MB=${MEM_MB:-45000}
QUEUE=${QUEUE:-normal}
MANIFEST=$(cd "$(dirname "$0")" && pwd)/hervk_ccle_manifest.tsv
mkdir -p "$WORK" "$REF" "$WORK/sra"

# --- 1. micromamba + tools ------------------------------------------------
if [ ! -x "$ENVBIN/STAR" ]; then
  if ! command -v micromamba >/dev/null 2>&1 && [ ! -x "$WORK/bin/micromamba" ]; then
    echo "[setup] installing micromamba (static binary) ..."
    mkdir -p "$WORK/bin"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xvj -C "$WORK/bin" --strip-components=1 bin/micromamba
  fi
  MM=$(command -v micromamba || echo "$WORK/bin/micromamba")
  echo "[setup] creating tools env at $ENV ..."
  "$MM" create -y -p "$ENV" -c conda-forge -c bioconda \
    sra-tools star samtools seqtk subread tetranscripts
fi
echo "[setup] tool check:"
for t in prefetch fasterq-dump STAR TEcount samtools; do
  [ -x "$ENVBIN/$t" ] && echo "  $t OK" || { echo "  $t MISSING"; exit 1; }
done

# --- 2. references --------------------------------------------------------
cd "$REF"
UCSC=https://hgdownload.soe.ucsc.edu/goldenPath/hg38
echo "[setup] hg38 fasta ..."
[ -s hg38.fa ] || { curl -fSL -O "$UCSC/bigZips/hg38.fa.gz" && gunzip -f hg38.fa.gz; }
echo "[setup] gene GTF (UCSC knownGene) ..."
[ -s hg38.knownGene.gtf ] || { curl -fSL -O "$UCSC/bigZips/genes/hg38.knownGene.gtf.gz" && gunzip -f hg38.knownGene.gtf.gz; }

echo "[setup] building TE GTF from UCSC rmsk.txt.gz ..."
if [ ! -s hg38_rmsk_TE.gtf ]; then
  curl -fSL -O "$UCSC/database/rmsk.txt.gz"
  # rmsk.txt cols: 6=chrom 7=start(0-based) 8=end 10=strand 11=repName 12=repClass 13=repFamily
  # emit one "exon" per repeat instance; gene_id=subfamily (e.g. LTR5_Hs) -> subfamily-level counts.
  # drop non-TE classes to match the standard TEtranscripts GTF.
  zcat rmsk.txt.gz | awk -F'\t' '
    $12 ~ /Simple_repeat|Low_complexity|Satellite|^RNA$|rRNA|scRNA|snRNA|srpRNA|tRNA|Unknown|\?/ {next}
    { n[$11]++;
      printf "%s\trmsk\texon\t%d\t%d\t.\t%s\t.\tgene_id \"%s\"; transcript_id \"%s_dup%d\"; family_id \"%s\"; class_id \"%s\";\n",
             $6, $7+1, $8, $10, $11, $11, n[$11], $13, $12 }
  ' > hg38_rmsk_TE.gtf
fi
echo "  TE GTF: $(wc -l < hg38_rmsk_TE.gtf) lines; HERVK subfamilies present:"
awk -F'gene_id "' '{split($2,a,"\"");print a[1]}' hg38_rmsk_TE.gtf \
  | grep -E '^(HERVK.*-int|LTR5|LTR5_Hs|LTR5A|LTR5B)$' | sort | uniq -c

# --- 3. pre-download SRA (.sra) on the login node -------------------------
echo "[setup] prefetch SRA runs (this is the big network step) ..."
tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r cl smp run rest; do
  [ -z "${run:-}" ] && continue
  if [ -s "$WORK/sra/$run/$run.sra" ]; then echo "  $run already present"; continue; fi
  echo "  prefetch $cl $run"
  "$ENVBIN/prefetch" --max-size u -O "$WORK/sra" "$run"
done

# --- 4. submit STAR index build (needs ~40 GB RAM) ------------------------
IDX=$REF/star_hg38
if [ -s "$IDX/SAindex" ]; then
  echo "[setup] STAR index already built at $IDX"
else
  mkdir -p "$IDX"
  cat > "$WORK/build_index.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
"$ENVBIN/STAR" --runMode genomeGenerate --runThreadN $THREADS \
  --genomeDir "$IDX" --genomeFastaFiles "$REF/hg38.fa" \
  --sjdbGTFfile "$REF/hg38.knownGene.gtf" --sjdbOverhang 100
echo "STAR index done: $IDX"
EOF
  chmod +x "$WORK/build_index.sh"
  echo "[setup] submitting STAR index build ..."
  bsub -q "$QUEUE" -n "$THREADS" -M "$MEM_MB" -R "rusage[mem=$MEM_MB]" \
       -J star_index -o "$WORK/star_index.%J.log" -e "$WORK/star_index.%J.log" \
       bash "$WORK/build_index.sh"
  echo "  -> watch: bjobs -J star_index ; log: $WORK/star_index.*.log"
fi

echo
echo "[setup] DONE. When the star_index job finishes, run:"
echo "  bash scripts/submit_hervk_ccle_requant.sh"
