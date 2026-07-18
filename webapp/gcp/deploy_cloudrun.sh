#!/usr/bin/env bash
# Deploy the TERRA web app (backend + bundled front-end, one origin) to Google
# Cloud Run. One public URL serves the whole app — no CORS, no separate frontend.
#
# Usage:
#   webapp/gcp/deploy_cloudrun.sh [gcp-project-id] [region]
#   # defaults: project = `gcloud config get-value project`, region = us-central1
#
# Prereqs (one-time):
#   - Install the gcloud CLI + `gcloud auth login`
#   - A GCP project with billing enabled (Cloud Run's free tier covers light use;
#     you're only billed while an instance is actually running).
#
# Re-run any time to redeploy — it rebuilds the image and rolls out a new revision.
#
# WHY THESE FLAGS (see webapp/api/app.py):
#   --no-cpu-throttling : job work runs in a background thread AFTER the HTTP
#       response is sent; Cloud Run throttles CPU to ~0 between requests by
#       default, which would stall the clustering/guide thread. This keeps CPU
#       allocated so the job actually progresses while the browser polls.
#   --max-instances 1   : the job registry (JOBS) is an in-memory dict, so every
#       request for a given job must land on the same instance. One instance
#       keeps status/log/file polling consistent.
#   --min-instances 0   : scale to zero when idle → no cost when nobody's using it.
set -euo pipefail

PROJECT="${1:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${2:-us-central1}"
SERVICE="terra"
REPO="terra"                       # Artifact Registry repo
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${SERVICE}:latest"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

[ -n "$PROJECT" ] || { echo "!! no project — pass it as arg 1 or run: gcloud config set project <id>"; exit 1; }
echo ">> project=$PROJECT region=$REGION image=$IMAGE"

echo ">> enabling required APIs (idempotent)"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
    artifactregistry.googleapis.com --project "$PROJECT"

echo ">> ensuring Artifact Registry repo '$REPO' exists"
gcloud artifacts repositories describe "$REPO" --location "$REGION" --project "$PROJECT" >/dev/null 2>&1 \
  || gcloud artifacts repositories create "$REPO" --repository-format docker \
       --location "$REGION" --project "$PROJECT" --description "TERRA web images"

echo ">> building image via Cloud Build (context = repo root)"
( cd "$REPO_ROOT" && gcloud builds submit \
    --config webapp/gcp/cloudbuild.yaml \
    --substitutions "_IMAGE=${IMAGE}" \
    --project "$PROJECT" . )

echo ">> deploying to Cloud Run"
gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --project "$PROJECT" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --memory 4Gi \
  --cpu 2 \
  --no-cpu-throttling \
  --min-instances 0 \
  --max-instances 1 \
  --timeout 3600 \
  --concurrency 20

URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --project "$PROJECT" --format 'value(status.url)')"
echo ">> done — TERRA is live at: $URL"
