#!/usr/bin/env bash
# hervk_gse157927.sh
# Pull GSE157927 processed RNA-seq (gene-level FPKM) on the HPC cluster and
# extract the annotated ERV/ERVK gene expression across the cell panel.
#
# IMPORTANT CAVEATS (see README notes at bottom):
#   * GSE157927 is a PANCREATIC cancer / CAF study. It contains Panc1 but does
#     NOT contain MCF-7 or PC3.
#   * The processed matrix is GENE-LEVEL (Ensembl genes). It does NOT quantify
#     HERVK repeat subfamilies (HERVK-int / LTR5 / LTR5Hs). Only a handful of
#     single annotated ERVK loci (ERVK3-1, ERVK13-1, ERV3-1, ERVW-1) exist here.
#   * Subfamily-level HERVK expression requires re-quantifying the RAW reads
#     (SRA SRP282356 / BioProject PRJNA663345) against RepeatMasker TE
#     annotations (SQUIRE / TEtranscripts / GAMECA). That is a separate, heavy
#     job — stub commands are provided at the bottom, gated off by default.
set -euo pipefail

GSE=GSE157927
OUT=${1:-$PWD/${GSE}_hervk}
mkdir -p "$OUT" && cd "$OUT"

BASE="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE157nnn/${GSE}/suppl"
FPKM="${GSE}_all_samples_FPKM_final.txt.gz"

echo "[1/3] Downloading processed gene-level FPKM matrix ..."
[ -s "$FPKM" ] || curl -fSL -o "$FPKM" "${BASE}/${FPKM}"

echo "[2/3] Panel + presence check ..."
python3 - "$FPKM" <<'PY'
import gzip, sys
f = gzip.open(sys.argv[1], "rt")
hdr = f.readline().rstrip("\n").split("\t")
samples = hdr[2:hdr.index("GeneDescription")]
print("  samples (%d): %s" % (len(samples), ", ".join(samples)))
def norm(s):  # drop separators/case so 'MIA-PaCa-2'->'miapaca2'
    return "".join(c for c in s.lower() if c.isalnum())
# exact normalized match — 'PC3' must NOT match 'BxPC-3' (a different line)
for want in ("Panc1", "PC3", "MCF7"):
    hit = [s for s in samples if norm(s) == norm(want)]
    print("  requested '%s' present: %s %s" % (want, bool(hit), hit or ""))
PY

echo "[3/3] Extracting ERV/ERVK gene FPKM -> ervk_fpkm.tsv ..."
python3 - "$FPKM" > ervk_fpkm.tsv <<'PY'
import gzip, sys
f = gzip.open(sys.argv[1], "rt")
hdr = f.readline().rstrip("\n").split("\t")
samples = hdr[2:hdr.index("GeneDescription")]
targets = {"ERVK3-1","ERVK13-1","ERV3-1","ERVW-1","ERVWE2","ERV9-1"}
print("GeneName\tGeneID\t" + "\t".join(samples))
for line in f:
    p = line.rstrip("\n").split("\t")
    if p[1] in targets:
        print(p[1] + "\t" + p[0] + "\t" + "\t".join(p[2:2+len(samples)]))
PY
column -t ervk_fpkm.tsv
echo
echo "Done. Wrote: $OUT/ervk_fpkm.tsv"

# ---------------------------------------------------------------------------
# OPTIONAL — real HERVK SUBFAMILY expression from RAW reads (heavy; disabled).
# Requires: sra-tools, a TE-aware quantifier, RepeatMasker hg38 GTF.
# Only Panc1 is relevant here (no MCF-7/PC3 in this study).
#
#   esearch -db sra -query PRJNA663345 | efetch -format runinfo > runinfo.csv
#   # pick the Panc1 RNA-seq run(s), then:
#   prefetch <SRR>; fasterq-dump --split-files <SRR>
#   # STAR --outFilterMultimapNmax 100 --winAnchorMultimapNmax 100 ...
#   # TEcount --GTF genes.gtf --TE hg38_rmsk_TE.gtf -b Aligned.bam
#   #   -> then sum HERVK-int / LTR5 / LTR5_Hs rows
# ---------------------------------------------------------------------------
