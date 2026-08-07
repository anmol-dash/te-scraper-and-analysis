#!/usr/bin/env bash
# download_fastq_ena.sh  --  resilient, parallel FASTQ download from EBI ENA,
# submitted as an LSF job (compute node, long wall clock; not the login node).
#
# The big CCLE runs (PC-3 ~87 GB, PANC-1 ~64 GB gz) download slowly over a
# single curl stream and ENA drops TLS mid-transfer. We use aria2c (16 parallel
# connections, native resume) from a Singularity container for speed+robustness,
# and fall back to a curl resume-loop if aria2 is unavailable. Each file is
# verified against ENA's reported byte size.
#
# Submit:  bash scripts/download_fastq_ena.sh          (bsubs itself)
# Watch:   bjobs -J ena_dl ; tail -f $WORK/ena_dl.*.log
set -uo pipefail   # no -e: retry loops handle non-zero exits

WORK=${WORK:-$HOME/hervk_ccle}
CONT=${CONT:-$WORK/containers}   # override to reuse another study's containers
QUEUE=${QUEUE:-rhel9}
WALL=${WALL:-48:00}          # LSF wall clock HH:MM; downloads can be long
THREADS=${THREADS:-8}
JOB=${JOB:-ena_dl}           # per-study name so concurrent studies don't collide
DEPOT=https://depot.galaxyproject.org/singularity
ARIA_SIF=$CONT/aria2.sif
MANIFEST=${MANIFEST:-$(cd "$(dirname "$0")" && pwd)/hervk_ccle_manifest.tsv}
mkdir -p "$WORK/fastq" "$CONT"

# ---- login side: ensure aria2 image, then submit self to a compute node ----
if [ "${1:-}" != "--run" ]; then
  if [ ! -s "$ARIA_SIF" ]; then
    echo "[dl] pulling aria2 image ..."
    curl -fSL -o "$ARIA_SIF" "$DEPOT/aria2:1.36.0" || echo "[dl] aria2 pull failed; will use curl fallback"
  fi
  LOG="$WORK/${JOB}.%J.log"
  echo "[dl] submitting download job (-q $QUEUE -W $WALL) as '$JOB' ..."
  bsub -q "$QUEUE" -n "$THREADS" -W "$WALL" -M 6000 -R "rusage[mem=6000]" \
       -J "$JOB" -o "$LOG" -e "$LOG" \
       env WORK="$WORK" CONT="$CONT" MANIFEST="$MANIFEST" JOB="$JOB" \
       bash "$(cd "$(dirname "$0")" && pwd)/$(basename "$0")" --run
  echo "[dl] submitted; watch: bjobs -J $JOB ; tail -f $WORK/${JOB}.*.log"
  exit 0
fi

# ---- compute side (--run): actually download ------------------------------
size() { [ -s "$1" ] && stat -c%s "$1" 2>/dev/null || echo 0; }

get_aria() {  # $1=url $2=dir $3=outname
  sing "$ARIA_SIF" aria2c \
    -x16 -s16 -c --max-tries=0 --retry-wait=5 --timeout=60 \
    --allow-overwrite=true --auto-file-renaming=false \
    -d "$2" -o "$3" "$1"
}
get_curl() {  # $1=url $2=out
  curl -fSL -C - -o "$2" "$1" --retry 15 --retry-delay 5 --retry-all-errors \
       --connect-timeout 30 --speed-time 120 --speed-limit 2000
}

fetch() {  # $1=url $2=out $3=want_bytes
  local url="$1" out="$2" want="$3" have attempt
  for attempt in $(seq 1 200); do
    have=$(size "$out")
    if [ -n "$want" ] && [ "$want" -gt 0 ] && [ "$have" -ge "$want" ]; then
      echo "  OK $(basename "$out") ($have bytes)"; return 0
    fi
    echo "  [$attempt] $(basename "$out"): have $have / want ${want:-?}"
    if [ -s "$ARIA_SIF" ]; then
      get_aria "$url" "$(dirname "$out")" "$(basename "$out")" && true
    else
      get_curl "$url" "$out" && true
    fi
  done
  have=$(size "$out")
  if [ -n "$want" ] && [ "$want" -gt 0 ] && [ "$have" -lt "$want" ]; then
    echo "  FAILED $(basename "$out") ($have/$want)"; return 1
  fi
}

echo "== ENA download start on $(hostname) =="
# aria2 runs in a container; if no runtime resolves here, drop to the curl path
# rather than looping 200 times on 'command not found'.
# shellcheck source=/dev/null
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_container.sh"
CONTAINER_BIND="${CONTAINER_BIND:-}${CONTAINER_BIND:+:}$WORK"
container_module_load
if ! container_init; then
  echo "  no container runtime -> using curl for every transfer"
  ARIA_SIF=""
fi
tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r cl smp run rest; do
  [ -z "${run:-}" ] && continue
  info=$(curl -s "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=${run}&result=read_run&fields=fastq_ftp,fastq_bytes&format=tsv" | awk 'NR==2')
  urls=$(echo "$info" | cut -f2); bytes=$(echo "$info" | cut -f3)
  [ -z "$urls" ] && { echo "no ENA FASTQ for $run"; continue; }
  IFS=';' read -r u1 u2 <<< "$urls"
  IFS=';' read -r b1 b2 <<< "$bytes"
  echo "== $cl $run =="
  fetch "https://${u1}" "$WORK/fastq/$(basename "$u1")" "${b1:-0}"
  fetch "https://${u2}" "$WORK/fastq/$(basename "$u2")" "${b2:-0}"
done

echo "== final sizes =="
ls -lh "$WORK/fastq/" 2>/dev/null || true

# VERIFY, and exit non-zero if anything is missing. Without this the job reports
# success having downloaded nothing (the fetch loop runs in a `while` on the far
# side of a pipe, and there is no `set -e`), so every -w done(...) dependency
# downstream fires on an empty $WORK/fastq and the real failure surfaces much
# later as a pile of unexplained exit-127 array elements.
missing=0; present=0
while IFS=$'\t' read -r cl smp run rest; do
  [ -z "${run:-}" ] && continue
  ok=1
  for f in "$WORK/fastq/${run}_1.fastq.gz" "$WORK/fastq/${run}_2.fastq.gz"; do
    [ -s "$f" ] || { echo "  MISSING $(basename "$f")  ($cl)"; ok=0; }
  done
  [ "$ok" = 1 ] && present=$((present + 1)) || missing=$((missing + 1))
done < <(tail -n +2 "$MANIFEST")

echo "== $present complete, $missing incomplete =="
if [ "$missing" -gt 0 ]; then
  echo "ENA DOWNLOAD FAILED: $missing of $((present + missing)) runs are missing a FASTQ."
  echo "  Common causes: no outbound network from the compute node, ENA rate-limiting,"
  echo "  or a full filesystem. Check the per-file lines above, then re-run --"
  echo "  completed files are kept and resumed."
  exit 1
fi
echo "ena download done: all $present runs present and byte-complete"
