#!/usr/bin/env bash
#
# submit_nextflow_test.sh — smoke-test the GAMECA Nextflow pipeline on the HPC.
#
# It (1) turns the pushed container image into a Singularity/Apptainer .sif (once),
# then (2) runs `nextflow run nextflow/main.nf -profile test,lsf,singularity`. The
# `test` profile sets stubRun=true, so every process (GAMECA_CORE + the 15
# post-alignment analyses modules + GAMECA_REPORT) runs as a lightweight stub —
# this validates config parsing, channel wiring, LSF submission, and
# `singularity exec` end-to-end WITHOUT needing a genome, network, or heavy deps.
#
# IMPORTANT: Nextflow is the *driver* — it submits the child bsub jobs itself, so
# run this ON A LOGIN NODE. Do NOT wrap it in `bsub` (that would nest bsub inside
# a job, which most LSF sites forbid).
#
# Usage:
#   chmod +x submit_nextflow_test.sh
#   ./submit_nextflow_test.sh                    # stub smoke test (default)
#   FAMILY=MT2_Mm ASSEMBLY=mm10 GENOME=/path/mm10.fa ./submit_nextflow_test.sh
#                                                # real single-family end-to-end run
#
# Common overrides (env vars):
#   IMAGE     container image to pull       (default ghcr.io/anmol-dash/gameca:latest)
#   SIF       output/existing .sif          (default ./gameca.sif; build skipped if present)
#   PROFILE   nextflow -profile             (default: test,lsf,singularity; auto-drops
#                                            'test' when FAMILY is set)
#   OUTDIR    results dir                   (default ./nextflow_test_results)
#   WORKDIR   nextflow work dir             (default ./nf_work)
#   QUEUE     LSF queue for child jobs      (default normal)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Config ─────────────────────────────────────────────────────────────────────
IMAGE="${IMAGE:-ghcr.io/anmol-dash/gameca:latest}"
SIF="${SIF:-$SCRIPT_DIR/gameca.sif}"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/nextflow_test_results}"
WORKDIR="${WORKDIR:-$SCRIPT_DIR/nf_work}"
QUEUE="${QUEUE:-normal}"

FAMILY="${FAMILY:-}"
ASSEMBLY="${ASSEMBLY:-hg38}"
GENOME="${GENOME:-}"

# Real run when FAMILY is given; otherwise the stub smoke test.
if [ -n "$FAMILY" ]; then
  PROFILE="${PROFILE:-lsf,singularity}"
else
  PROFILE="${PROFILE:-test,lsf,singularity}"
fi

# ── 1. Ensure Nextflow + Java are on PATH ───────────────────────────────────────
module load nextflow 2>/dev/null || true
module load java 2>/dev/null || module load openjdk 2>/dev/null \
  || module load Java 2>/dev/null || true

if ! command -v nextflow >/dev/null 2>&1; then
  echo "[nf-test] nextflow not found — installing to \$HOME/.local/bin ..."
  command -v java >/dev/null 2>&1 || {
    echo "ERROR: Java 11+ required for Nextflow. 'module load java' (or openjdk) first." >&2
    exit 1
  }
  mkdir -p "$HOME/.local/bin"
  ( cd "$HOME/.local/bin" && curl -s https://get.nextflow.io | bash )
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v nextflow >/dev/null 2>&1 || { echo "ERROR: nextflow still not on PATH" >&2; exit 1; }
echo "[nf-test] nextflow: $(command -v nextflow)  ($(nextflow -version 2>&1 | awk '/version/{print $3; exit}'))"

# ── 2. Build the .sif from the pushed image (once) ──────────────────────────────
RUNTIME="$(command -v singularity || command -v apptainer || echo singularity)"
if [ ! -f "$SIF" ]; then
  echo "[nf-test] Building $SIF from docker://$IMAGE via $RUNTIME ..."
  "$RUNTIME" build "$SIF" "docker://$IMAGE" \
    || { echo "ERROR: failed to build $SIF (is $IMAGE pushed + public?)" >&2; exit 1; }
else
  echo "[nf-test] Reusing existing $SIF"
fi

# ── 3. Assemble and run the nextflow command ────────────────────────────────────
# Route child LSF jobs to $QUEUE without editing the committed config.
mkdir -p "$WORKDIR"
QCFG="$WORKDIR/_queue.config"
echo "process.queue = '$QUEUE'" > "$QCFG"

NF_ARGS=(run "$SCRIPT_DIR/nextflow/main.nf"
         -profile "$PROFILE"
         -c "$QCFG"
         --container_sif "$SIF"
         --outdir "$OUTDIR"
         -work-dir "$WORKDIR"
         -resume)

if [ -n "$FAMILY" ]; then
  NF_ARGS+=(--family "$FAMILY" --assembly "$ASSEMBLY")
  [ -n "$GENOME" ] && NF_ARGS+=(--genome_fasta "$GENOME")
fi

echo "============================================================"
echo "GAMECA Nextflow test"
echo "  Image:   $IMAGE"
echo "  SIF:     $SIF"
echo "  Profile: $PROFILE"
echo "  Mode:    ${FAMILY:+real run — family=$FAMILY assembly=$ASSEMBLY}${FAMILY:-stub smoke test}"
echo "  Outdir:  $OUTDIR"
echo "  Queue:   $QUEUE  (child jobs)"
echo "  Cmd:     nextflow ${NF_ARGS[*]}"
echo "============================================================"

exec nextflow "${NF_ARGS[@]}"
