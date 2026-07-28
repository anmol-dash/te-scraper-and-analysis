#!/usr/bin/env bash
# fetch_gse157927.sh  --  archive GSE157927 (Triptolide / pancreatic SE study,
# BioProject PRJNA663345 / SRA SRP282356) into its OWN location, separate from
# the CCLE HERVK work in $HOME/hervk_ccle. This study is unrelated to CCLE and
# shares only the PANC1 line (no MCF-7, no PC-3).
#
#   default:  pull all PROCESSED supplementary files (~20 MB) + the SRA run
#             manifest (raw-read map). No large downloads.
#   --raw RE: also download raw RNA-seq FASTQ (from ENA) for runs whose sample
#             title matches regex RE, e.g.:
#               bash scripts/fetch_gse157927.sh --raw 'PANC1 '     # just PANC1
#               bash scripts/fetch_gse157927.sh --raw 'RNA-Seq'    # all human RNA
set -euo pipefail

DEST=${DEST:-$HOME/gse157927}
mkdir -p "$DEST/processed" "$DEST/meta" "$DEST/fastq"
SUPPL=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE157nnn/GSE157927/suppl
ENA="https://www.ebi.ac.uk/ena/portal/api/filereport?accession=SRP282356&result=read_run"

# --- 1. processed supplementary files (FPKM, htseq counts, peak BEDs) -------
echo "[gse] processed supplementary files -> $DEST/processed"
curl -s "$SUPPL/" | grep -oE 'GSE157927[^"/]*\.(gz|txt)' | sort -u | while read -r f; do
  [ -s "$DEST/processed/$f" ] && { echo "  have $f"; continue; }
  echo "  get $f"; curl -fSL -o "$DEST/processed/$f" "$SUPPL/$f"
done

# --- 2. SRA run manifest (library type per run; the raw-read map) -----------
echo "[gse] SRA run manifest -> $DEST/meta/SRP282356_runs.tsv"
curl -s "${ENA}&fields=run_accession,library_strategy,library_source,scientific_name,read_count,base_count,fastq_ftp,sample_title&format=tsv&limit=0" \
  > "$DEST/meta/SRP282356_runs.tsv"
echo "  $(($(wc -l < "$DEST/meta/SRP282356_runs.tsv")-1)) runs recorded"

# --- 3. optional raw FASTQ for selected runs (RNA-seq only) -----------------
if [ "${1:-}" = "--raw" ]; then
  RE=${2:?usage: --raw <sample_title regex>}
  echo "[gse] raw RNA-seq FASTQ for titles matching /$RE/ -> $DEST/fastq"
  awk -F'\t' -v re="$RE" 'NR>1 && $3=="TRANSCRIPTOMIC" && $8 ~ re {print $1"\t"$7"\t"$8}' \
      "$DEST/meta/SRP282356_runs.tsv" | while IFS=$'\t' read -r run ftp title; do
    [ -z "$ftp" ] && { echo "  no fastq_ftp for $run ($title)"; continue; }
    echo "  == $run  $title =="
    IFS=';' read -ra urls <<< "$ftp"
    for u in "${urls[@]}"; do
      out="$DEST/fastq/$(basename "$u")"
      [ -s "$out" ] && { echo "    have $(basename "$out")"; continue; }
      echo "    get $(basename "$out")"
      curl -fSL -C - -o "$out" "https://${u}" --retry 15 --retry-delay 5 --retry-all-errors
    done
  done
fi

echo
echo "[gse] done. Layout under $DEST :"
echo "  processed/  GEO supplementary (FPKM, htseq counts, MACS peaks)"
echo "  meta/       SRP282356_runs.tsv (run -> library type + sample title)"
echo "  fastq/      raw RNA-seq (only if --raw was used)"
