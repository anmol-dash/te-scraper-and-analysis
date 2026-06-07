FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba-cache \
    GAMECA_HOST=0.0.0.0 \
    GAMECA_PORT=8765

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bedtools \
        build-essential \
        ca-certificates \
        gcc \
        git \
        libbz2-dev \
        libcurl4-openssl-dev \
        liblzma-dev \
        libssl-dev \
        mafft \
        procps \
        wget \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && wget -q http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/liftOver \
        -O /usr/local/bin/liftOver && chmod +x /usr/local/bin/liftOver \
        || echo "WARNING: liftOver download failed; ortholog/multiassembly degrade gracefully"

COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements.txt

COPY . .

EXPOSE 8765

CMD ["python", "ui.py", "--local"]
