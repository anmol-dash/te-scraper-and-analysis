#!/usr/bin/env bash
# lib_container.sh --- sourced by the submit scripts to get a WORKING container
# runtime on whichever host they happen to be running on.
#
# Two pennhpc failure modes this exists to survive:
#
#   1. `singularity` is not on PATH on the compute nodes (the login node has it,
#      the batch shell does not inherit that PATH). The binary is often called
#      `apptainer` there instead. Symptom: every array element dies instantly
#      with exit code 127 and an empty-looking log.
#
#   2. Some nodes ship apptainer as a Go build linked against a FIPS OpenSSL it
#      cannot initialise, and it panics before the container even starts:
#        panic: opensslcrypto: can't enable FIPS mode for OpenSSL 3.2.2
#      Symptom: exit code 2, ~15 s runtime. Node-dependent, not queue-dependent.
#      GOFIPS=0/GOLANG_FIPS=0 disables that backend and is harmless elsewhere.
#
# Usage:
#   source "$(dirname "$0")/lib_container.sh"
#   container_init            # sets GAMECA_RT, exits 1 with a real message if none
#   sing "$SOME_SIF" cmd ...  # runs cmd in that image
#
# GAMECA_RT=/path/to/apptainer forces a specific binary.

# apptainer ignores SINGULARITYENV_*; set both so ~/.local never shadows the image
export SINGULARITYENV_PYTHONNOUSERSITE=1
export APPTAINERENV_PYTHONNOUSERSITE=1
export GOFIPS=0 GOLANG_FIPS=0

# Extra bind paths (colon-separated) for filesystems outside $HOME
CONTAINER_BIND=${CONTAINER_BIND:-}

container_init() {
  if [ -n "${GAMECA_RT:-}" ] && command -v "$GAMECA_RT" >/dev/null 2>&1; then
    :
  else
    GAMECA_RT=$(command -v singularity 2>/dev/null \
                || command -v apptainer 2>/dev/null || true)
  fi
  if [ -z "${GAMECA_RT:-}" ]; then
    echo "FATAL: no container runtime on $(hostname -s): neither 'singularity'" >&2
    echo "       nor 'apptainer' is on PATH." >&2
    echo "       PATH=$PATH" >&2
    echo "       If this is a compute node, the batch shell may need a module:" >&2
    echo "         module avail 2>&1 | grep -i -E 'singularity|apptainer'" >&2
    echo "       then re-submit with CONTAINER_MODULE='<name>' (or set GAMECA_RT=)." >&2
    return 1
  fi
  export GAMECA_RT
  echo "  container runtime: $GAMECA_RT ($("$GAMECA_RT" --version 2>&1 | head -1))"
}

# If the site needs `module load`, do it before probing. CONTAINER_MODULE is
# forwarded into the job by the submit scripts.
container_module_load() {
  [ -n "${CONTAINER_MODULE:-}" ] || return 0
  # `module` is a shell function from /etc/profile.d; batch shells often skip it
  if ! command -v module >/dev/null 2>&1; then
    for f in /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
      [ -r "$f" ] && . "$f" && break
    done
  fi
  command -v module >/dev/null 2>&1 || { echo "  (no 'module' command; skipping)"; return 0; }
  echo "  module load $CONTAINER_MODULE"
  module load $CONTAINER_MODULE || echo "  WARNING: module load failed; continuing"
}

# container_probe <image.sif> -- does the runtime ACTUALLY start on this host?
# `command -v` is not enough: apptainer can be present and still panic before the
# container starts (the Go FIPS/OpenSSL panic). GOFIPS=0 is meant to prevent that,
# but it is a mitigation, not a guarantee -- so anything with a non-container
# fallback should probe first rather than retry into a wall.
container_probe() {
  local img="$1"
  [ -n "${GAMECA_RT:-}" ] || return 1
  [ -s "$img" ] || return 1
  sing "$img" true >/dev/null 2>&1
}

# sing <image.sif> <command...>
sing() {
  local img="$1"; shift
  local binds=( -B "$HOME" )
  if [ -n "$CONTAINER_BIND" ]; then
    local IFS=':'
    for p in $CONTAINER_BIND; do [ -n "$p" ] && binds+=( -B "$p" ); done
  fi
  "$GAMECA_RT" exec "${binds[@]}" "$img" "$@"
}
