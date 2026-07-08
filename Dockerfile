FROM python:3.11-slim-bookworm

# GAMECA analysis container.
#
# This image is the SOURCE OF TRUTH for the Singularity/Apptainer .sif used on
# HPC. The container ships its own glibc 2.36, so the manylinux_2_28 wheels that
# fail to install on old-glibc login/compute nodes (pybigtools, MOODS-python,
# colabfold, numba/llvmlite) install cleanly here. Once inside the .sif the host
# glibc is irrelevant — that is the whole point of wrapping the pipeline.
#
# Build the .sif from this image (see build_sif.sh):
#   docker build -t ghcr.io/anmol-dash/gameca:latest .
#   docker push ghcr.io/anmol-dash/gameca:latest
#   # on the cluster (no root needed):
#   singularity build gameca.sif docker://ghcr.io/anmol-dash/gameca:latest
#
# NOTE: ColabFold fold prediction is intentionally NOT in this image — it needs
# a CUDA base and runs from the separate colabfold.sif (see colabfold.def).

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba-cache \
    GAMECA_HOME=/opt/gameca/code \
    GAMECA_HOST=0.0.0.0 \
    GAMECA_PORT=8765

# System toolchain + bioinformatics binaries the pipeline shells out to
# (mafft, bedtools, liftOver, bigBedToBed/bigWigToBedGraph for JASPAR bigBed).
# trimal is the fallback alignment cleaner when CIAlign fails or times out
# (>1h) on huge alignments — see te_alignment.run_cialign()/README.md.
# default-jre-headless + curl are here so the baked-in Nextflow (installed below)
# can run — needed when query.py --stage11-nextflow orchestrates the
# post-alignment analyses from *inside* the container.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bedtools \
        build-essential \
        ca-certificates \
        curl \
        default-jre-headless \
        gcc \
        git \
        libbz2-dev \
        libcurl4-openssl-dev \
        liblzma-dev \
        libssl-dev \
        mafft \
        procps \
        tabix \
        wget \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && for _b in liftOver bigBedToBed bigWigToBedGraph; do \
         wget -q "http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/$_b" \
           -O "/usr/local/bin/$_b" && chmod +x "/usr/local/bin/$_b" \
           || echo "WARNING: $_b download failed; stages that use it degrade gracefully"; \
       done

# trimAl is no longer packaged in Debian bookworm's apt repos ("Unable to
# locate package trimal"), which broke every image build since it was added
# as the CIAlign fallback cleaner (see te_alignment.run_trimal()). Build the
# small C++ binary from source instead — it has no external deps beyond g++.
RUN curl -sL https://github.com/inab/trimal/archive/refs/tags/v1.5.0.tar.gz \
      -o /tmp/trimal.tar.gz \
    && tar xzf /tmp/trimal.tar.gz -C /tmp \
    && make -C /tmp/trimal-1.5.0/source \
    && install -m 755 /tmp/trimal-1.5.0/source/trimal /usr/local/bin/trimal \
    && rm -rf /tmp/trimal.tar.gz /tmp/trimal-1.5.0 \
    && trimal --version

# Nextflow — GAMECA's orchestration layer (see nextflow/). Baked in so the image
# can BOTH be the per-process container for an external `nextflow run` AND run the
# pipeline itself (query.py --stage11-nextflow) when exec'd standalone on HPC.
ENV NXF_HOME=/opt/gameca/.nextflow \
    NXF_OFFLINE=false
RUN curl -s https://get.nextflow.io | bash \
    && mv nextflow /usr/local/bin/nextflow \
    && chmod +x /usr/local/bin/nextflow \
    && nextflow -version

WORKDIR /opt/gameca/code

# Install Python deps first so the layer caches across code edits. numpy/Cython
# must be present before the te_fast extension is built below.
#
# ColabFold is deliberately excluded here: colabfold[alphafold] pins
# jax/alphafold/tensorflow with no CPU-only linux wheels, which makes the resolve
# impossible, and fold prediction runs from the separate CUDA colabfold.sif
# anyway (run_fold_prediction.py --colabfold-cmd). We strip just that line and
# keep requirements.txt as the single source of truth.
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
    && grep -viE '^[[:space:]]*colabfold' requirements.txt > /tmp/requirements.docker.txt \
    && python -m pip install -r /tmp/requirements.docker.txt

# Bake the repository into the image. .dockerignore keeps out the desktop app,
# genomes, caches and prior results (those get bind-mounted at runtime).
COPY . .

# Build the te_fast Cython extension FOR LINUX. The only committed .so is a
# macOS build (and is .dockerignore'd); without this the pipeline silently falls
# back to pure-Python and runs much slower.
RUN python setup_cython.py build_ext --inplace \
    && python -c "import te_fast; print('te_fast built:', te_fast.__file__)"

# Fail the build early if any critical import is missing — this is the glibc /
# import smoke test the whole container exists to guarantee.
RUN python - <<'PY'
import importlib
mods = ["numpy", "pandas", "scipy", "sklearn", "umap", "hdbscan",
        "numba", "Bio", "pysam", "MOODS", "requests", "yaml", "te_fast"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:  # noqa: BLE001
        missing.append(f"{m}: {e}")
try:
    importlib.import_module("pybigtools")
except Exception as e:  # noqa: BLE001
    print(f"NOTE: pybigtools unavailable ({e}); JASPAR bigBed path degrades to bigBedToBed")
if missing:
    raise SystemExit("MISSING IMPORTS:\n  " + "\n  ".join(missing))
print("import smoke test OK")
PY

ENV PATH=/usr/local/bin:$PATH

EXPOSE 8765

# Default entrypoint launches the local dashboard; on HPC we override this with
# `singularity exec gameca.sif python query.py ...`.
CMD ["python", "ui.py", "--local"]
