#!/usr/bin/env bash
# submit_fetch_jaspar.sh
# Pre-fetch the genome-wide JASPAR 2022 BED cache for all four mammalian
# builds — one LSF job per build, so the four run in parallel.
#
# WHY: without a warm cache every pipeline run re-fetches JASPAR into its own
# per-family results/<family>/jaspar/ directory. Worse, the on-the-fly bigBed
# path writes an *uncompressed* JASPAR2022_<build>.te_loci.bed, which used to be
# fed to `zcat` and silently produced an EMPTY indexed BED — the "tabix returned
# 0 rows" failure that left enrichment_results/ and go_annotations/ empty.
# A shared, genome-wide, pre-validated cache avoids that path entirely.
#
# Submit:  bash scripts/submit_fetch_jaspar.sh
# Then:    pass  --jaspar-dir "$CACHE"  to query.py (see submit_results722v1.sh)
set -euo pipefail

GAMECA_HOME=${GAMECA_HOME:-$HOME/anmol/te-scraper-and-analysis}
CACHE=${CACHE:-$HOME/anmol/jaspar_cache}     # shared across ALL families/runs
LOGS=${LOGS:-$CACHE/logs}
BUILDS=${BUILDS:-"hg38 hg19 mm10 mm39"}
QUEUE=${QUEUE:-rhel9}                        # PennHPC/PMACS default post-Q3-2026
MEM_MB=${MEM_MB:-16000}
THREADS=${THREADS:-4}
WORKERS=${WORKERS:-8}                        # parallel HTTP workers per build

mkdir -p "$CACHE" "$LOGS"

echo "== JASPAR pre-fetch =="
echo "  GAMECA_HOME : $GAMECA_HOME"
echo "  cache dir   : $CACHE"
echo "  builds      : $BUILDS"
echo "  queue       : $QUEUE   mem=${MEM_MB}MB  threads=$THREADS  workers=$WORKERS"

[ -f "$GAMECA_HOME/fetch_jaspar.py" ] || {
  echo "ERROR: fetch_jaspar.py not found under $GAMECA_HOME" >&2; exit 1; }

for B in $BUILDS; do
  OUT="$CACHE/JASPAR2022_${B}.sorted.bed.gz"
  ALT="$CACHE/JASPAR2024_${B}.sorted.bed.gz"
  # Skip builds already cached and non-empty (the resolver accepts either name).
  if [ -s "$OUT" ] || [ -s "$ALT" ]; then
    echo "  [skip] $B already cached"
    continue
  fi
  # Don't stack a second job for a build that is already queued/running.
  if bjobs -w 2>/dev/null | grep -q "jaspar_${B}"; then
    echo "  [skip] $B already PEND/RUN"
    continue
  fi

  echo "  [submit] $B"
  bsub -J "jaspar_${B}" \
       -q "$QUEUE" \
       -n "$THREADS" \
       -M "$MEM_MB" \
       -R "rusage[mem=${MEM_MB}]" \
       -o "$LOGS/jaspar_${B}.%J.out" \
       -e "$LOGS/jaspar_${B}.%J.err" \
       /bin/bash -lc "
set -euo pipefail
cd '$GAMECA_HOME'
echo \"[jaspar] build=$B host=\$(hostname) start=\$(date)\"

# The fetch needs outbound HTTPS; PMACS compute nodes reach it via the proxy.
export HTTPS_PROXY=\${HTTPS_PROXY:-\${https_proxy:-}}
export HTTP_PROXY=\${HTTP_PROXY:-\${http_proxy:-}}

python3 fetch_jaspar.py --build '$B' --cache-dir '$CACHE' --workers $WORKERS

# Validate: a cache that exists but is empty is the exact failure mode this
# script exists to prevent, so fail loudly rather than leaving a poisoned file.
F=\$(ls -1 '$CACHE'/JASPAR20*_${B}.sorted.bed.gz 2>/dev/null | head -1)
if [ -z \"\$F\" ]; then echo '[jaspar] FAILED: no cache file written' >&2; exit 1; fi
N=\$(gzip -dc \"\$F\" | head -100000 | wc -l)
echo \"[jaspar] \$F  bytes=\$(stat -c%s \"\$F\" 2>/dev/null || stat -f%z \"\$F\")  first-lines=\$N\"
if [ \"\$N\" -eq 0 ]; then
  echo '[jaspar] FAILED: cache file is EMPTY — removing so it is not reused' >&2
  rm -f \"\$F\"; exit 1
fi

# Build the tabix index once, here, so pipeline runs only ever seek.
if command -v tabix >/dev/null && command -v bgzip >/dev/null; then
  tabix -f -p bed \"\$F\" || tabix -f -p bed -C \"\$F\" || true
  ls -la \"\$F\".tbi \"\$F\".csi 2>/dev/null || true
fi
echo \"[jaspar] build=$B DONE \$(date)\"
"
done

echo
echo "Watch:   bjobs -w | grep jaspar_"
echo "Logs:    $LOGS"
echo "Then run the pipeline with:  --jaspar-dir $CACHE"
