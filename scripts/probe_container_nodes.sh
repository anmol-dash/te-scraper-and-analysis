#!/usr/bin/env bash
# probe_container_nodes.sh --- find out, per compute node, whether the container
# runtime can actually start -- and if not, whether --userns rescues it.
#
# Background: on pennhpc rhel9, `apptainer --version` succeeds while
# `apptainer exec` panics in the Go FIPS/OpenSSL backend on SOME nodes. That
# kills any pipeline stage needing STAR/featureCounts/TEcount, which have no
# host fallback. This maps the damage in about a minute so you can either
# confirm a workaround or exclude the bad nodes.
#
#   bash scripts/probe_container_nodes.sh           # 20 probes
#   N=40 bash scripts/probe_container_nodes.sh      # cast a wider net
#   bash scripts/probe_container_nodes.sh --report  # summarise when they finish
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
CONT=${CONT:-$HOME/hervk_ccle/containers}
IMG=${IMG:-$CONT/star.sif}
OUT=${OUT:-$HOME/container_probe}
QUEUE=${QUEUE:-rhel9}
N=${N:-20}
JOB=${JOB:-cprobe}
mkdir -p "$OUT"

if [ "${1:-}" = "--report" ]; then
  echo "== container probe results =="
  # NB: every count below is guarded. A bare `grep ... | wc -l` inside $( ) exits
  # non-zero when nothing matches, and under `set -e` that kills the script --
  # which is exactly why this printed a bare header and quit.
  shopt -s nullglob
  res=( "$OUT"/*.res )
  shopt -u nullglob
  if [ ${#res[@]} -eq 0 ]; then
    echo "  no results in $OUT"
    echo
    if bjobs -J "$JOB" >/dev/null 2>&1; then
      echo "  the probe jobs are still queued/running:"
      bjobs -J "$JOB" 2>/dev/null | head -5 | sed 's/^/    /'
      echo "  re-run --report when they clear."
    else
      echo "  no probe jobs are queued either -- you need to submit them first:"
      echo "      bash scripts/probe_container_nodes.sh"
      echo "  (then wait ~1 min and re-run --report)"
    fi
    exit 0
  fi
  count() { grep -l "RESULT=$1" "${res[@]}" 2>/dev/null | wc -l | tr -d ' '; }
  ok=$(count ok); un=$(count userns); bad=$(count broken)
  printf '  plain exec works : %s\n  needs --userns   : %s\n  broken entirely  : %s\n' \
    "${ok:-0}" "${un:-0}" "${bad:-0}"
  echo
  echo "  by node:"
  sort -u "${res[@]}" 2>/dev/null | sed 's/^/    /' || true
  echo
  broken=$(grep -h 'RESULT=broken' "${res[@]}" 2>/dev/null | sed 's/ .*//' | sort -u | paste -sd' ' - || true)
  if [ -n "$broken" ]; then
    echo "  exclude these when submitting:"
    echo "    EXCLUDE_HOSTS='$broken' bash scripts/run_endometrium_ltr.sh --clean --go"
  elif [ "$un" -gt 0 ]; then
    echo "  --userns rescues every node tested; the pipeline will use it automatically."
  else
    echo "  every node tested is fine -- the earlier failure may have been one bad node."
  fi
  exit 0
fi

if [ "${1:-}" = "--run-one" ]; then
  h=$(hostname -s)
  res="$OUT/${h}.res"
  # deliberately NOT using lib_container.sh here: this probe must report the raw
  # behaviour, not the behaviour after the library's workarounds
  export GOFIPS=0 GOLANG_FIPS=0
  RT=$(command -v singularity 2>/dev/null || command -v apptainer 2>/dev/null || true)
  if [ -z "$RT" ]; then echo "$h RESULT=broken reason=no-runtime" > "$res"; exit 0; fi
  ver=$("$RT" --version 2>&1 | head -1)
  if "$RT" exec "$IMG" true >/dev/null 2>&1; then
    echo "$h RESULT=ok runtime='$ver'" > "$res"
  elif "$RT" exec --userns "$IMG" true >/dev/null 2>&1; then
    echo "$h RESULT=userns runtime='$ver'" > "$res"
  else
    err=$("$RT" exec "$IMG" true 2>&1 | head -1 | tr -d "'")
    echo "$h RESULT=broken runtime='$ver' err='$err'" > "$res"
  fi
  exit 0
fi

[ -s "$IMG" ] || { echo "FATAL: probe image not found: $IMG (set IMG=)"; exit 1; }
rm -f "$OUT"/*.res
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
# span[ptile=1] spreads the array over distinct hosts rather than packing it
bsub -q "$QUEUE" -n 1 -W 5 -M 2000 -R "rusage[mem=2000]" -R "span[ptile=1]" \
     -J "${JOB}[1-$N]" -o "$OUT/${JOB}.%J_%I.log" -e "$OUT/${JOB}.%J_%I.log" \
     env OUT="$OUT" IMG="$IMG" bash "$SELF" --run-one
echo "submitted ${JOB}[1-$N]; these are 5-second jobs."
echo "  when they clear:  bash scripts/probe_container_nodes.sh --report"
