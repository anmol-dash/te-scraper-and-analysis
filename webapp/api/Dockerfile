# TERRA Web API — container for Render / Fly.io / Hugging Face Spaces / any PaaS.
#
# Build context MUST be the repository root (the API imports te_clustering.py,
# run_grna_analysis.py and te_prep.py from there):
#
#   docker build -f webapp/api/Dockerfile -t terra-api .
#   docker run -p 8000:8000 terra-api
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

# Copy the pipeline modules the API imports + the API itself.
COPY te_clustering.py run_grna_analysis.py te_prep.py te_notify.py /app/
COPY webapp/api /app/webapp/api

ENV PYTHONUNBUFFERED=1 \
    TERRA_JOBS_DIR=/tmp \
    PORT=8000

EXPOSE 8000

# Honour the platform-provided $PORT (Render/Fly/HF set it); default 8000.
CMD ["sh", "-c", "cd /app/webapp/api && uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
