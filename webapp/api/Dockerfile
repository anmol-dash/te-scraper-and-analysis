# TERRA Web App — container for Hugging Face Spaces / Render / Fly.io / any PaaS.
#
# Serves BOTH the FastAPI backend and the static front-end from one origin, so
# no API-URL configuration is needed: the UI talks to /api/* on itself.
#
# Build context MUST be the repository root (the API imports te_clustering.py,
# run_grna_analysis.py and te_prep.py from there):
#
#   docker build -f webapp/api/Dockerfile -t terra-web .
#   docker run -p 7860:7860 terra-web      # open http://localhost:7860
#
FROM python:3.11-slim

# Build tools for hdbscan / numba wheels that may compile from source.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for layer caching.
COPY webapp/api/requirements.txt /app/webapp/api/requirements.txt
RUN pip install --no-cache-dir -r /app/webapp/api/requirements.txt

# Copy the pipeline modules the API imports + the API + the static front-end.
COPY te_clustering.py run_grna_analysis.py te_prep.py te_notify.py /app/
COPY webapp/api /app/webapp/api
COPY webapp/frontend /app/webapp/frontend

# HF Spaces runs the container as a non-root user (uid 1000) and only guarantees
# /tmp is writable. Point every cache/scratch dir there so numba, matplotlib and
# the job store never try to write into a read-only home.
ENV PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    TERRA_JOBS_DIR=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba-cache \
    PORT=7860

# HF Spaces expects the app on port 7860 (declared as app_port in the Space
# README). $PORT lets Render/Fly override it.
EXPOSE 7860

CMD ["sh", "-c", "cd /app/webapp/api && uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
