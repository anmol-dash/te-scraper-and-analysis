#!/usr/bin/env bash
# Deploy the TERRA web app to a Hugging Face Docker Space.
#
# HF Docker Spaces build from a `Dockerfile` at the *root* of the Space repo.
# This repo's root Dockerfile is the heavy HPC/.sif image, so we don't push this
# repo directly. Instead we assemble a minimal, self-contained Space repo that
# mirrors only the paths the web Dockerfile needs, then git-push it to HF.
#
# Usage:
#   webapp/hf/deploy_hf.sh <hf-username>/<space-name>
#   # e.g. webapp/hf/deploy_hf.sh anmol-dash/terra
#
# Prereqs (one-time):
#   pip install -U "huggingface_hub[cli]"
#   hf auth login          # paste a WRITE token from hf.co/settings/tokens
#   hf repos create <hf-username>/<space-name> --type space --space-sdk docker
#
# Re-run this script any time to redeploy; HF rebuilds the container on push.
set -euo pipefail

SPACE="${1:?usage: deploy_hf.sh <hf-username>/<space-name>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo ">> assembling Space in $STAGE"
git -C "$STAGE" clone "https://huggingface.co/spaces/$SPACE" . 2>/dev/null \
  || { echo "!! could not clone https://huggingface.co/spaces/$SPACE"; \
       echo "   create it first: hf repos create $SPACE --type space --space-sdk docker"; \
       exit 1; }

# Root Dockerfile for the Space = the web Dockerfile (build context is the Space
# root, and we mirror the exact paths it COPYs from below).
cp "$REPO_ROOT/webapp/api/Dockerfile" "$STAGE/Dockerfile"
# HF Space landing README (front-matter drives the Space config).
cp "$REPO_ROOT/webapp/hf/README.md"   "$STAGE/README.md"

# Pipeline modules the API imports + the API + the static front-end.
cp "$REPO_ROOT/te_clustering.py" "$REPO_ROOT/run_grna_analysis.py" \
   "$REPO_ROOT/te_prep.py" "$REPO_ROOT/te_notify.py" "$STAGE/"
mkdir -p "$STAGE/webapp"
rsync -a --delete "$REPO_ROOT/webapp/api/"      "$STAGE/webapp/api/"
rsync -a --delete "$REPO_ROOT/webapp/frontend/" "$STAGE/webapp/frontend/"
rm -rf "$STAGE/webapp/api/__pycache__"

echo ">> committing + pushing to $SPACE"
git -C "$STAGE" add -A
git -C "$STAGE" -c user.email="deploy@terra" -c user.name="terra-deploy" \
    commit -m "Deploy TERRA web app" --quiet || { echo "nothing to deploy"; exit 0; }
git -C "$STAGE" push

echo ">> done — build at https://huggingface.co/spaces/$SPACE  (Logs tab shows progress)"
