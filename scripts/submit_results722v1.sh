#!/usr/bin/env bash
# submit_results722v1.sh
# Re-run the LTR5_Hs (hg38) + MT2_Mm (mm10) test pair, replacing the contents of
# $OUTPUT (default /home/amodz/anmol/results722v1) from scratch.
#
# Same query.py invocations as the 2026-07-23 run, plus the fixes for:
#   • tabix "0 rows"  → empty enrichment_results/ + go_annotations/
#   • KeyError 'chr'  → MT2_Mm collapsed to a single Cluster 0 in every CSV
#   • alignment/consensus/CIAlign output split across 6 directories
#   • clustering_visualization_expr.html produced with no expression data
#   • motif failure aborting the job before primers + Stage 11 ran
#
# Prereq (once, and worth doing first — it is what fixes the JASPAR failure):
#   bash scripts/submit_fetch_jaspar.sh      # caches hg38/hg19/mm10/mm39
#
# Submit:  bash scripts/submit_results722v1.sh
set -euo pipefail

GAMECA_HOME=${GAMECA_HOME:-$HOME/anmol/te-scraper-and-analysis}
OUTPUT=${OUTPUT:-$HOME/anmol/results722v1}
JASPAR_DIR=${JASPAR_DIR:-$HOME/anmol/jaspar_cache}   # shared cache, not per-family
MT2_INPUT=${MT2_INPUT:-$HOME/anmol/mt2_mm_ultracombo_counts.csv}
LOGS=${LOGS:-$OUTPUT/logs}

QUEUE=${QUEUE:-rhel9}            # PennHPC/PMACS default post-Q3-2026
THREADS=${THREADS:-12}
MEM_MB=${MEM_MB:-48000}
WALL=${WALL:-48:00}

# The 2026-07-23 run went through Nextflow and every attempt failed rc=1
# (query.py not visible inside gameca.sif), silently falling back to the
# in-process loop. The bind fix is committed (533ae81); this script pins the
# in-process path so the re-test measures the pipeline, not the orchestrator.
# Set USE_NEXTFLOW=1 to exercise the Nextflow path instead.
USE_NEXTFLOW=${USE_NEXTFLOW:-0}
if [ "$USE_NEXTFLOW" = "1" ]; then NF_FLAG="--nextflow"; else NF_FLAG="--no-nextflow"; fi

echo "== GAMECA re-test =="
echo "  GAMECA_HOME : $GAMECA_HOME"
echo "  OUTPUT      : $OUTPUT   (will be REPLACED)"
echo "  JASPAR_DIR  : $JASPAR_DIR"
echo "  MT2 input   : $MT2_INPUT"
echo "  queue=$QUEUE threads=$THREADS mem=${MEM_MB}MB wall=$WALL nextflow=$USE_NEXTFLOW"
echo

# ── preflight ───────────────────────────────────────────────────────────────
[ -f "$GAMECA_HOME/query.py" ] || { echo "ERROR: no query.py under $GAMECA_HOME" >&2; exit 1; }
[ -s "$MT2_INPUT" ]            || { echo "ERROR: MT2_Mm input not found: $MT2_INPUT" >&2; exit 1; }

# Kept as a plain string, not an array: it is interpolated into the job script
# text below, where it must land as TWO argv words ("--jaspar-dir" and the path).
JASPAR_ARG=""
if [ -s "$JASPAR_DIR/JASPAR2022_hg38.sorted.bed.gz" ] || \
   [ -s "$JASPAR_DIR/JASPAR2024_hg38.sorted.bed.gz" ]; then
  JASPAR_ARG="--jaspar-dir '$JASPAR_DIR'"
  echo "  JASPAR cache found — passing --jaspar-dir"
else
  echo "  WARNING: no JASPAR cache in $JASPAR_DIR."
  echo "           Each family will fetch its own copy (slow, and the network path"
  echo "           is what failed last time). Recommended: run"
  echo "             bash scripts/submit_fetch_jaspar.sh"
  echo "           first, then re-submit this script."
fi
echo

# ── replace the output tree ─────────────────────────────────────────────────
if bjobs -w 2>/dev/null | grep -qE "gameca_(ltr5_hs|mt2_mm)"; then
  echo "ERROR: a gameca_ltr5_hs / gameca_mt2_mm job is already PEND/RUN." >&2
  echo "       Kill it first (bkill) so it cannot write into the tree being wiped." >&2
  exit 1
fi

echo "Removing previous results:"
for d in "$OUTPUT/ltr5_hs" "$OUTPUT/mt2_mm" "$OUTPUT/pipeline_info"; do
  [ -e "$d" ] && { echo "  rm -rf $d"; rm -rf "$d"; } || echo "  (absent) $d"
done
mkdir -p "$OUTPUT" "$LOGS"
echo

# ── LTR5_Hs / hg38  (local rmsk mode, no expression) ────────────────────────
echo "[submit] LTR5_Hs (hg38)"
bsub -J gameca_ltr5_hs \
     -q "$QUEUE" -n "$THREADS" -M "$MEM_MB" -R "rusage[mem=${MEM_MB}]" -W "$WALL" \
     -o "$LOGS/ltr5_hs.%J.out" -e "$LOGS/ltr5_hs.%J.err" \
     /bin/bash -lc "
set -euo pipefail
cd '$GAMECA_HOME'
echo \"[gameca] LTR5_Hs host=\$(hostname) start=\$(date)\"
export HTTPS_PROXY=\${HTTPS_PROXY:-\${https_proxy:-}}
export HTTP_PROXY=\${HTTP_PROXY:-\${http_proxy:-}}
python3 query.py \
  --local --source rmsk \
  --family LTR5_Hs \
  --assembly hg38 \
  --output '$OUTPUT' \
  $JASPAR_ARG \
  --skip-genome \
  --fetch-workers 10 \
  --kmer 10 --pca-dims 40 --n-epochs 200 --n-neighbors 30 \
  --min-cluster-size 100 --min-samples 5 --min-sequences 10 --random-state 42 \
  --primer-kmer 18 --top-global 8 --top-cluster 5 --primer-timeout 120 \
  --parallel-alignment --parallel-primers \
  --target-assemblies hg19 panTro6 rheMac10 \
  --ortholog-species panTro6 gorGor6 ponAbe3 \
  --epigenetic-preset K562 --ctcf-preset K562 --tads-preset K562 \
  --grna-cas SpCas9 --grna-max-mm 2 \
  --clock-divisor 1 --intact-orf-aa 100 --min-ltr-identity 0.65 \
  --tail-bp 150 --promoter-bp 200 --cpg-omega 10 \
  --p-threshold 0.05 \
  $NF_FLAG --debug --force
echo \"[gameca] LTR5_Hs DONE rc=\$? \$(date)\"
"

# ── MT2_Mm / mm10  (CSV input, 5 expression stages) ─────────────────────────
# NOTE: no --expression-assembly. It takes a PATH to an expression interval
# table; passing 'hg38' there is what produced
# "FileNotFoundError: Expression assembly not found: hg38" last time. The
# per-stage counts are already columns of $MT2_INPUT and are named via
# --expr-cols / --expr-labels below.
echo "[submit] MT2_Mm (mm10)"
bsub -J gameca_mt2_mm \
     -q "$QUEUE" -n "$THREADS" -M "$MEM_MB" -R "rusage[mem=${MEM_MB}]" -W "$WALL" \
     -o "$LOGS/mt2_mm.%J.out" -e "$LOGS/mt2_mm.%J.err" \
     /bin/bash -lc "
set -euo pipefail
cd '$GAMECA_HOME'
echo \"[gameca] MT2_Mm host=\$(hostname) start=\$(date)\"
export HTTPS_PROXY=\${HTTPS_PROXY:-\${https_proxy:-}}
export HTTP_PROXY=\${HTTP_PROXY:-\${http_proxy:-}}
python3 query.py \
  --input '$MT2_INPUT' \
  --family MT2_Mm \
  --assembly mm10 \
  --output '$OUTPUT' \
  $JASPAR_ARG \
  --skip-genome \
  --fetch-workers 10 \
  --expr-cols twocell fourcell eightcell morulacell pronuc \
  --expr-labels '2-cell' '4-cell' '8-cell' 'Morula' 'Pronucleus' \
  --kmer 10 --pca-dims 40 --n-epochs 200 --n-neighbors 30 \
  --min-cluster-size 100 --min-samples 5 --min-sequences 10 --random-state 42 \
  --primer-kmer 18 --top-global 8 --top-cluster 5 --primer-timeout 120 \
  --parallel-alignment --parallel-primers \
  --target-assemblies mm39 rn7 \
  --ortholog-species mm39 rn7 \
  --grna-cas SpCas9 --grna-max-mm 2 \
  --clock-divisor 1 --intact-orf-aa 100 --min-ltr-identity 0.65 \
  --tail-bp 150 --promoter-bp 200 --cpg-omega 10 \
  --p-threshold 0.05 \
  $NF_FLAG --debug --force
echo \"[gameca] MT2_Mm DONE rc=\$? \$(date)\"
"

cat <<EOF

Watch:  bjobs -w | grep gameca_
Logs:   $LOGS
Status: cat $OUTPUT/{ltr5_hs,mt2_mm}/PIPELINE_STATUS.txt

── What to check when they finish ──────────────────────────────────────────
 1. 10/10 stages in PIPELINE_STATUS.txt (was 7/10 — motif aborted the job)
 2. MT2_Mm has >1 cluster in the CSVs, matching its clustering HTML:
      cut -d, -f\$(head -1 $OUTPUT/mt2_mm/01_data/mt2_mm_clustered.csv \\
        | tr ',' '\\n' | grep -n '^Cluster\$' | cut -d: -f1) \\
        $OUTPUT/mt2_mm/01_data/mt2_mm_clustered.csv | sort -u | head
      diff <(cut -d, -f1 $OUTPUT/mt2_mm/03_clustering/cluster_summary.csv) /dev/null
 3. Enrichment + GO are populated:
      ls -la $OUTPUT/*/enrichment_results/ $OUTPUT/*/go_annotations/
 4. Alignment output is in ONE tree, whole family + every cluster,
    cleaned and not cleaned:
      ls $OUTPUT/*/04_alignments/            # fasta/ images/ logs/ index.html
      ls $OUTPUT/*/04_alignments/fasta/      # whole_family_* and cluster_N_*
      ls -d $OUTPUT/*/cialign_plots $OUTPUT/*/05_consensus 2>/dev/null \\
        && echo "UNEXPECTED: old split dirs still present"
 5. LTR5_Hs has NO clustering_visualization_expr.html (it has no expression);
    MT2_Mm does have one:
      ls $OUTPUT/*/03_clustering/clustering_visualization_expr.html
 6. cluster_summary.csv strand columns are non-zero for MT2_Mm (recovered
    from TE_ID, which has no explicit strand column):
      cat $OUTPUT/mt2_mm/03_clustering/cluster_summary.csv
EOF
