#!/usr/bin/env bash
# submit_rbmx_recluster.sh
#
# One-command LSF submission for RBMX family re-clustering and JASPAR motif
# analysis.  For each of the 33 RBMX families whose sequences failed to fetch
# in the original run, this script:
#
#   1. Verifies / downloads hg38.fa (checks common HPC paths, then UCSC).
#   2. Downloads the JASPAR 2022 hg38 BED (once, shared across all jobs).
#   3. Submits a job array — one LSF job per family — that calls
#      recluster_rbmx.py (sits alongside this script in the project root).
#
# Usage:
#   chmod +x submit_rbmx_recluster.sh
#   ./submit_rbmx_recluster.sh [results_dir]
#
#   results_dir  defaults to /home/amodz/anmol/results_rbmx_families
#
# After submission monitor with:
#   bjobs -J "rbmx_recluster"
#   tail -f logs_recluster/THE1A.log

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${1:-/home/amodz/anmol/results_rbmx_families}"   # source: existing family dirs
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${SCRIPT_DIR}/rbmx_reclustered_${TIMESTAMP}"         # fresh output each run
GENOME_DIR="${SCRIPT_DIR}"
JASPAR_DIR="${SCRIPT_DIR}/jaspar_cache"
LOG_DIR="${OUT_DIR}/logs"
PYTHON="${PYTHON:-python3}"

mkdir -p "$OUT_DIR" "$JASPAR_DIR" "$LOG_DIR"

# Keep RESULTS_DIR pointing at the source for family enumeration
RESULTS_DIR="$SOURCE_DIR"

# ── 0. Pre-flight: verify bedtools is available and functional ───────────────
echo "  [preflight] Checking bedtools..."
if ! command -v bedtools &>/dev/null; then
    echo ""
    echo "ERROR: bedtools not found on PATH."
    echo "  Try:  module load bedtools"
    echo "  Or:   conda install -c bioconda bedtools"
    echo ""
    exit 1
fi
# Quick functional test — pipe two overlapping records through intersect
_BT_TEST=$(printf 'chr1\t100\t200\nchr1\t150\t250\n' \
           | bedtools intersect -a stdin -b stdin 2>&1) || {
    echo ""
    echo "ERROR: bedtools intersect self-test failed:"
    echo "  ${_BT_TEST}"
    echo "  bedtools may be installed but broken or missing libstdc++."
    echo ""
    exit 1
}
BT_VER=$(bedtools --version 2>&1 | head -1)
echo "  [preflight] bedtools OK  (${BT_VER})"

# Verify python is reachable too — nothing else to do if it isn't
echo "  [preflight] Checking python..."
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: python not found at '${PYTHON}'.  Set PYTHON=/path/to/python3."
    exit 1
fi
PY_VER=$("$PYTHON" --version 2>&1)
echo "  [preflight] python OK  (${PY_VER})"
echo ""

echo "============================================================"
echo " RBMX re-clustering + motif submission"
echo "  Source dir  : $SOURCE_DIR"
echo "  Output dir  : $OUT_DIR"
echo "  Project dir : $SCRIPT_DIR"
echo "  JASPAR cache: $JASPAR_DIR"
echo "  Logs        : $LOG_DIR"
echo "============================================================"

# ── 1. Locate or download hg38.fa ────────────────────────────────────────────
HG38_FA=""

# Common HPC genome locations — checked in priority order
GENOME_CANDIDATES=(
    "${SCRIPT_DIR}/hg38.fa"
    "${HOME}/hg38.fa"
    "/data/genomes/hg38/hg38.fa"
    "/data/genomes/hg38/genome.fa"
    "/scratch/genomes/hg38/hg38.fa"
    "/reference/human/hg38/hg38.fa"
    "/refdata/hg38/hg38.fa"
    "/shares/hpcc/genomes/hg38/hg38.fa"
    "/work/reference/hg38.fa"
    "/gpfs/data/genomes/hg38/hg38.fa"
    "/cluster/ref/hg38/hg38.fa"
)

for candidate in "${GENOME_CANDIDATES[@]}"; do
    if [[ -f "$candidate" ]]; then
        HG38_FA="$candidate"
        echo "  [genome] Found at: $HG38_FA"
        break
    fi
done

if [[ -z "$HG38_FA" ]]; then
    HG38_FA="${SCRIPT_DIR}/hg38.fa"
    echo "  [genome] Not found in standard locations."
    echo "  [genome] Downloading hg38.fa.gz from UCSC → ${HG38_FA}.gz"
    echo "  [genome] (This is ~3.1 GB; interrupt and set HG38_FA manually if you have it)"
    echo ""

    UCSC_HG38="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
    curl -L --retry 5 --retry-delay 10 -o "${HG38_FA}.gz" "$UCSC_HG38"
    echo "  [genome] Decompressing..."
    gunzip "${HG38_FA}.gz"
    echo "  [genome] Done: ${HG38_FA}"
fi

if [[ ! -f "$HG38_FA" ]]; then
    echo "ERROR: hg38.fa not found and download failed.  Set HG38_FA manually."
    exit 1
fi

# ── 2. Download JASPAR 2022 hg38 BED (once, shared) ─────────────────────────
# We download the full pre-scored BED from JASPAR.elixir.no (JASPAR 2024 hg38).
# The file is ~500 MB gzipped.  If already present, skip.
JASPAR_BED="${JASPAR_DIR}/JASPAR2024_hg38.bed.gz"

if [[ -f "$JASPAR_BED" ]]; then
    echo "  [jaspar] Found: $JASPAR_BED"
else
    echo "  [jaspar] Downloading JASPAR 2024 hg38 BED..."
    JASPAR_URL="https://jaspar.elixir.no/static/data/beds/JASPAR2024_hg38.bed.gz"
    curl -L --retry 5 --retry-delay 10 -o "$JASPAR_BED" "$JASPAR_URL"
    echo "  [jaspar] Done: $JASPAR_BED"
fi

# Sort + index the BED once (required by bedtools intersect -sorted)
JASPAR_BED_SORTED="${JASPAR_DIR}/JASPAR2024_hg38.sorted.bed.gz"
if [[ ! -f "$JASPAR_BED_SORTED" ]]; then
    echo "  [jaspar] Sorting and indexing BED..."
    # Use bgzip + tabix if available (preferred), else plain gzip sort
    if command -v tabix &>/dev/null && command -v bgzip &>/dev/null; then
        zcat "$JASPAR_BED" \
            | sort -k1,1 -k2,2n \
            | bgzip -c > "$JASPAR_BED_SORTED"
        tabix -p bed "$JASPAR_BED_SORTED"
        echo "  [jaspar] tabix index written"
    else
        zcat "$JASPAR_BED" \
            | sort -k1,1 -k2,2n \
            | gzip -c > "$JASPAR_BED_SORTED"
        echo "  [jaspar] sorted BED written (no tabix)"
    fi
    # Use the sorted version from now on
    JASPAR_BED="$JASPAR_BED_SORTED"
else
    JASPAR_BED="$JASPAR_BED_SORTED"
    echo "  [jaspar] Sorted BED already present: $JASPAR_BED"
fi

# ── 3. Build family list from existing directories ───────────────────────────
mapfile -t FAMILIES < <(
    ls -1d "${RESULTS_DIR}"/*/  2>/dev/null \
    | grep -v '_logs\|_status\|_summary' \
    | xargs -I{} basename {} \
    | sort
)

N="${#FAMILIES[@]}"
if [[ "$N" -eq 0 ]]; then
    echo "ERROR: No family directories found under ${RESULTS_DIR}"
    exit 1
fi

echo ""
echo "  Found $N families:"
printf '    %s\n' "${FAMILIES[@]}"
echo ""

# Write the family list to a file so the job array can look it up by index
FAM_LIST_FILE="${LOG_DIR}/family_list.txt"
printf '%s\n' "${FAMILIES[@]}" > "$FAM_LIST_FILE"
echo "  Family list → $FAM_LIST_FILE"

# ── 4. Write the per-job script (heredoc) ────────────────────────────────────
JOB_SCRIPT="${LOG_DIR}/job_worker.sh"
cat > "$JOB_SCRIPT" <<HEREDOC
#!/usr/bin/env bash
# Auto-generated by submit_rbmx_recluster.sh — do not edit by hand
set -euo pipefail

IDX="\${LSB_JOBINDEX:-1}"
FAM_LIST="${FAM_LIST_FILE}"
FAMILY=\$(sed -n "\${IDX}p" "\$FAM_LIST")

if [[ -z "\$FAMILY" ]]; then
    echo "ERROR: empty family for index \$IDX"
    exit 1
fi

echo "=== Job \$IDX / \$FAMILY ==="
cd "${SCRIPT_DIR}"

${PYTHON} recluster_rbmx.py \\
    --results-dir "${SOURCE_DIR}" \\
    --out-dir     "${OUT_DIR}/\${FAMILY}" \\
    --family       "\$FAMILY" \\
    --genome       "${HG38_FA}" \\
    --jaspar-bed   "${JASPAR_BED}" \\
    --jaspar-dir   "${JASPAR_DIR}" \\
    --max-divisor  15 \\
    --assembly     hg38 \\
    --kmer         6 \\
    --pca-dims     50 \\
    --n-epochs     200 \\
    --min-samples  7 \\
    --p-threshold  0.05
HEREDOC
chmod +x "$JOB_SCRIPT"

# ── 5. Submit job array ───────────────────────────────────────────────────────
# Resource allocation:
#   -M 20000  20 GB RAM  (GenomeCache ~3.5 GB + UMAP/HDBSCAN + motif analysis)
#   -n 4      4 cores    (UMAP and HDBSCAN both use n_jobs=-1)
#   -W 3:00   3-hour wall-clock limit (generous; typical run <30 min per family)
#   %8        run at most 8 jobs concurrently (avoids overloading the cluster)
#
# Adjust -q, -M, -W to match your cluster policy.

QUEUE="${LSF_QUEUE:-normal}"
echo "Submitting job array [1-${N}] to queue '${QUEUE}'..."
echo ""

bsub \
    -J "rbmx_recluster[1-${N}]%8" \
    -q  "${QUEUE}" \
    -n  4 \
    -M  20000 \
    -R  "rusage[mem=20000] span[hosts=1]" \
    -W  3:00 \
    -o  "${LOG_DIR}/%J_%I.out" \
    -e  "${LOG_DIR}/%J_%I.err" \
    "${JOB_SCRIPT}"

echo ""
echo "============================================================"
echo " Submitted.  Monitor with:"
echo "   bjobs -J rbmx_recluster"
echo "   bpeek  <jobid>[<idx>]"
echo ""
echo " Per-family output:"
echo "   Root      : ${OUT_DIR}/<family>/"
echo "   Sequences : ${OUT_DIR}/<family>/01_data/<family>_with_sequences.csv"
echo "   Clustered : ${OUT_DIR}/<family>/01_data/<family>_clustered.csv"
echo "   UMAP HTML : ${OUT_DIR}/<family>/03_clustering/clustering_visualization.html"
echo "   Motifs    : ${OUT_DIR}/<family>/motif_analysis/"
echo "   Logs      : ${LOG_DIR}/<jobid>_<idx>.out"
echo "============================================================"
