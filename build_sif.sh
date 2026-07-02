#!/usr/bin/env bash
#
# build_sif.sh — build the GAMECA Singularity/Apptainer image (gameca.sif).
#
# You CANNOT build a .sif on macOS — Singularity/Apptainer needs Linux. Pick a
# path below. The docker:// path is the project default (no root on the cluster).
#
# ── Path A (default): Docker → docker:// ────────────────────────────────────
#   On any machine with Docker (your Mac is fine for THIS step):
#       docker build -t "$IMAGE" .
#       docker push  "$IMAGE"                 # to a registry the cluster can pull
#   On the cluster login node (no root needed):
#       REMOTE_IMAGE="docker://$IMAGE" ./build_sif.sh
#
# ── Path B: fakeroot build on the cluster ───────────────────────────────────
#   Copy the repo to the cluster, then on a login/compute node:
#       DEF=gameca.def ./build_sif.sh
#   (requires user-namespace / --fakeroot support; ask your admins if unsure.)
#
# After building, run the pipeline via `singularity exec` — see run_in_sif() at
# the bottom of this file, or set CONTAINER_SIF in the HPC client / submit args.

set -euo pipefail

IMAGE="${IMAGE:-gameca:latest}"                 # docker tag (Path A build/push)
SIF="${SIF:-gameca.sif}"                        # output image
RUNTIME="${RUNTIME:-$(command -v singularity || command -v apptainer || echo singularity)}"
REMOTE_IMAGE="${REMOTE_IMAGE:-}"                 # e.g. docker://ghcr.io/you/gameca:latest
DEF="${DEF:-}"                                   # e.g. gameca.def for fakeroot build

if [ -n "$REMOTE_IMAGE" ]; then
    echo "[build_sif] Building $SIF from $REMOTE_IMAGE (docker:// path) ..."
    "$RUNTIME" build "$SIF" "$REMOTE_IMAGE"
elif [ -n "$DEF" ]; then
    echo "[build_sif] Building $SIF from $DEF (--fakeroot path) ..."
    "$RUNTIME" build --fakeroot "$SIF" "$DEF"
else
    cat >&2 <<EOF
[build_sif] Nothing to build. Choose one:
  Path A:  docker build -t $IMAGE . && docker push $IMAGE
           REMOTE_IMAGE=docker://$IMAGE $0
  Path B:  DEF=gameca.def $0
EOF
    exit 2
fi

echo "[build_sif] Built $SIF. Verifying imports inside the container ..."
"$RUNTIME" exec "$SIF" python -c \
  "import numpy,pandas,scipy,sklearn,umap,hdbscan,numba,Bio,pysam,MOODS,te_fast; \
   print('gameca.sif import smoke test OK')"

cat <<EOF

[build_sif] Done: $SIF

Run the pipeline through the image (bind your work dir + genome):

  WORK=\$PWD
  GENOME_DIR=/path/to/genomes           # dir containing hg38.fa / mm10.fa
  $RUNTIME exec \\
    -B "\$WORK":"\$WORK" -B "\$GENOME_DIR":"\$GENOME_DIR" --pwd "\$WORK" \\
    $SIF python query.py --family MT2_Mm --assembly mm10 \\
      --local-assembly "\$GENOME_DIR/mm10.fa"

Or let the HPC client wrap every job automatically by pointing it at the .sif:
  set the CONTAINER_SIF parameter to the absolute path of $SIF on the cluster.
EOF
