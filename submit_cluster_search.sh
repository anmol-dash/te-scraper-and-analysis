#!/usr/bin/env bash
# submit_cluster_search.sh — search clustering parameters for N clusters with
# minimal noise, then apply the winner and regenerate expression/DE.
#
# Why: the core pipeline uses min_cluster_size = floor(N/5) and stops at the
# first config giving >=2 clusters. For L1Md_T that demands 4,727 loci per
# cluster, yielding 2 blobs and 53.8% noise. This searches for a real partition.
#
# Usage:
#   ./submit_cluster_search.sh                       # L1Md_T, target 5-6 clusters
#   ./submit_cluster_search.sh B1_Mus2 b1mus2        # another family
#   FAMILY=IAPLTR1_Mm TAG=iapltr1mm ./submit_cluster_search.sh
set -euo pipefail

FAMILY="${1:-${FAMILY:-L1Md_T}}"
TAG="${2:-${TAG:-l1mdt}}"
MIN_CLUSTERS="${MIN_CLUSTERS:-5}"
MAX_CLUSTERS="${MAX_CLUSTERS:-6}"

REPO="${REPO:-$HOME/anmol/te-scraper-and-analysis}"
DIR="reports8/stage11_${TAG}"
INPUT="${DIR}/$(echo "$FAMILY" | tr '[:upper:]' '[:lower:]')_clustered.csv"

# TE_NO_PYSAM=1 is required: pysam's bundled libcrypto fails its FIPS self-test
# on these nodes and abort()s at import. See project_fips_pysam_container memory.
bsub -J "csearch_${TAG}" -M 32000 -n 8 -o "cluster_search_${TAG}.%J.log" \
  "cd ${REPO} && singularity exec --env TE_NO_PYSAM=1 gameca.sif \
   python run_cluster_search.py \
     --input ${INPUT} \
     --family ${FAMILY} \
     --reports-dir ${DIR} \
     --min-clusters ${MIN_CLUSTERS} --max-clusters ${MAX_CLUSTERS} \
     --apply"

echo "submitted: ${FAMILY} -> ${DIR}"
echo "  target: ${MIN_CLUSTERS}-${MAX_CLUSTERS} clusters, minimal noise"
echo "  watch:  bjobs ; tail -f cluster_search_${TAG}.*.log"
echo
echo "on completion, results land in ${DIR}/:"
echo "  cluster_search.csv          every configuration tried"
echo "  cluster_search_best.json    winning parameters"
echo "  fig_cluster_search.pdf      noise vs cluster count"
echo "  ${FAMILY,,}_clustered.csv   re-clustered loci"
echo "  fig_${TAG}_results.pdf      regenerated expression (corrected columns)"
echo "  fig_de_${TAG}.pdf           regenerated DE"
