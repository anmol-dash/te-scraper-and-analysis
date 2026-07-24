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
CONT=$WORK/containers
QUEUE=${QUEUE:-rhel9}
WALL=${WALL:-48:00}          # LSF wall clock HH:MM; downloads can be long
THREADS=${THREADS:-8}
DEPOT=https://depot.galaxyproject.org/singularity
ARIA_SIF=$CONT/aria2.sif
MANIFEST=$(cd "$(dirname "$0")" && pwd)/hervk_ccle_manifest.tsv
mkdir -p "$WORK/fastq" "$CONT"

# ---- login side: ensure aria2 image, then submit self to a compute node ----
if [ "${1:-}" != "--run" ]; then
  if [ ! -s "$ARIA_SIF" ]; then
    echo "[dl] pulling aria2 image ..."
    curl -fSL -o "$ARIA_SIF" "$DEPOT/aria2:1.36.0" || echo "[dl] aria2 pull failed; will use curl fallback"
  fi
  LOG="$WORK/ena_dl.%J.log"
  echo "[dl] submitting download job (-q $QUEUE -W $WALL) ..."
  bsub -q "$QUEUE" -n "$THREADS" -W "$WALL" -M 6000 -R "rusage[mem=6000]" \
       -J ena_dl -o "$LOG" -e "$LOG" \
       bash "$(cd "$(dirname "$0")" && pwd)/$(basename "$0")" --run
  echo "[dl] submitted; watch: bjobs -J ena_dl ; tail -f $WORK/ena_dl.*.log"
  exit 0
fi

# ---- compute side (--run): actually download ------------------------------
size() { [ -s "$1" ] && stat -c%s "$1" 2>/dev/null || echo 0; }

get_aria() {  # $1=url $2=dir $3=outname
  singularity exec -B "$WORK" "$ARIA_SIF" aria2c \
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
echo "ena download done; verify sizes above match ENA (then run submit_hervk_ccle_requant.sh)"
