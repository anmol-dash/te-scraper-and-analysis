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

# How many of our probe jobs are still queued/running? Judged from the OUTPUT,
# not the exit status: this LSF returns 0 from `bjobs -J <name>` even when the
# name matches nothing, so an exit-code test reports "still running" forever.
probe_jobs_running() {
  local out
  out=$(bjobs -J "$JOB" 2>/dev/null || true)
  printf '%s\n' "$out" | grep -cE '\b(PEND|RUN)\b' || true
}

report() {   # $1=just_submitted (1 after a submit, else empty)
  local submitted="${1:-}"
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
    if [ "$(probe_jobs_running)" -gt 0 ]; then
      echo "  the probe jobs are still queued/running:"
      bjobs -J "$JOB" 2>/dev/null | head -5 | sed 's/^/    /' || true
      echo "  re-run --report when they clear."
    elif [ -n "$submitted" ]; then
      echo "  The probe jobs finished but wrote no results -- they failed before"
      echo "  reaching the container test. Look at the job logs:"
      echo "      tail -n 20 $OUT/${JOB}.*.log"
    else
      echo "  NOTHING WAS SUBMITTED. Run it without --report first:"
      echo "      bash scripts/probe_container_nodes.sh"
      echo "  (that submits, waits, and prints this report for you)"
    fi
    return 0
  fi
  # No pipelines here. `grep ... | wc -l` inside $( ) exits non-zero when there
  # are no matches, and under `set -e` + pipefail that aborts the whole report --
  # which is what made this print a bare header twice.
  count() {
    local n=0 f body
    for f in "${res[@]}"; do
      body=$(cat "$f" 2>/dev/null || true)
      case "$body" in *"RESULT=$1"*) n=$((n + 1)) ;; esac
    done
    printf '%s' "$n"
  }
  ok=$(count ok); un=$(count userns); bad=$(count broken)
  printf '  plain exec works : %s\n  needs --userns   : %s\n  broken entirely  : %s\n' \
    "${ok:-0}" "${un:-0}" "${bad:-0}"
  echo
  echo "  by node:"
  sort -u "${res[@]}" 2>/dev/null | sed 's/^/    /' || true
  echo
  broken=""
  for f in "${res[@]}"; do
    body=$(cat "$f" 2>/dev/null || true)
    case "$body" in *"RESULT=broken"*) broken="$broken${broken:+ }${body%% *}" ;; esac
  done
  if [ -n "$broken" ]; then
    echo "  exclude these when submitting:"
    echo "    EXCLUDE_HOSTS='$broken' bash scripts/run_endometrium_ltr.sh --clean --go"
  elif [ "$un" -gt 0 ]; then
    echo "  --userns rescues every node tested; the pipeline will use it automatically."
  else
    echo "  every node tested is fine -- the earlier failure may have been one bad node."
  fi
}

if [ "${1:-}" = "--report" ]; then report; exit 0; fi

if [ "${1:-}" = "--run-one" ]; then
  # A diagnostic must ALWAYS leave a verdict behind, so this worker runs with
  # -e and pipefail OFF. The previous version piped the panic text through
  # `head -1`: head exits after one line, apptainer takes SIGPIPE, pipefail
  # reports failure and -e killed the worker before it wrote anything -- so
  # every BROKEN node (the ones we most need to see) silently vanished.
  set +e +o pipefail
  h=$(hostname -s)
  res="$OUT/${h}.res"
  echo "$h RESULT=broken reason=probe-did-not-finish" > "$res"   # overwritten below
  # deliberately NOT using lib_container.sh here: this probe must report the raw
  # behaviour, not the behaviour after the library's workarounds
  export GOFIPS=0 GOLANG_FIPS=0
  RT=$(command -v singularity 2>/dev/null || command -v apptainer 2>/dev/null)
  if [ -z "$RT" ]; then echo "$h RESULT=broken reason=no-runtime" > "$res"; exit 0; fi
  # first line via parameter expansion, not `| head -1` -- no pipe, no SIGPIPE
  ver=$("$RT" --version 2>&1); ver=${ver%%$'\n'*}
  if "$RT" exec "$IMG" true >/dev/null 2>&1; then
    echo "$h RESULT=ok runtime='$ver'" > "$res"
  elif "$RT" exec --userns "$IMG" true >/dev/null 2>&1; then
    echo "$h RESULT=userns runtime='$ver'" > "$res"
  else
    err=$("$RT" exec "$IMG" true 2>&1); err=${err%%$'\n'*}; err=${err//\'/}
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
echo "submitted ${JOB}[1-$N]; these are 5-second jobs. Waiting for them..."
for i in $(seq 1 60); do            # up to 10 minutes
  running=$(probe_jobs_running)
  [ "$running" -eq 0 ] && break
  printf '\r  %s still queued/running (%ds elapsed) ' "$running" "$((i * 10))"
  sleep 10
done
echo
report 1
