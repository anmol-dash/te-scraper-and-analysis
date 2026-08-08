#!/usr/bin/env bash
# setup_host_tools.sh --- install STAR + featureCounts as plain host binaries,
# so the pipeline can run with NO container at all.
#
# Why this exists: on pennhpc rhel9 the apptainer package is broken on every
# node tested (19/19, Aug 2026) --
#   panic: opensslcrypto: can't enable FIPS mode for OpenSSL 3.2.2
# `apptainer --version` works, `apptainer exec` panics, and --userns does not
# help. That is a broken system package, not something a flag fixes. Until RIS
# rebuild it, containers cannot be used here for anything.
#
# Both tools ship official self-contained Linux x86_64 builds:
#   STAR           - upstream publishes a STATIC binary (no shared-lib deps)
#   featureCounts  - subread ships a precompiled tarball
# Neither needs conda, pip, root, or a container.
#
# Deliberately NOT installed: TEtranscripts/TEcount. It needs pysam, which
# SIGABRTs on this cluster for the same FIPS reason. featureCounts with
# -g gene_id gives the same subfamily-level table (the TE GTF's gene_id IS the
# repName), so TEcount is not required -- see submit_locus_expression.sh.
#
#   bash scripts/setup_host_tools.sh          # install into ~/tools
#   TOOLS=/some/where bash scripts/setup_host_tools.sh
set -euo pipefail

TOOLS=${TOOLS:-$HOME/tools}
BIN="$TOOLS/bin"
STAR_VER=${STAR_VER:-2.7.11b}
SUBREAD_VER=${SUBREAD_VER:-2.0.6}
mkdir -p "$BIN" "$TOOLS/src"

echo "== host tools -> $BIN =="

# --- STAR (static upstream binary) ------------------------------------------
if [ -x "$BIN/STAR" ] && "$BIN/STAR" --version >/dev/null 2>&1; then
  echo "  STAR already installed: $("$BIN/STAR" --version)"
else
  echo "  downloading STAR $STAR_VER (static build) ..."
  curl -fSL -o "$BIN/STAR.tmp" \
    "https://raw.githubusercontent.com/alexdobin/STAR/${STAR_VER}/bin/Linux_x86_64_static/STAR"
  chmod +x "$BIN/STAR.tmp"
  if "$BIN/STAR.tmp" --version >/dev/null 2>&1; then
    mv "$BIN/STAR.tmp" "$BIN/STAR"
    echo "  STAR OK: $("$BIN/STAR" --version)"
  else
    rm -f "$BIN/STAR.tmp"
    echo "  STAR binary did not run here; falling back to the non-static build"
    curl -fSL -o "$BIN/STAR" \
      "https://raw.githubusercontent.com/alexdobin/STAR/${STAR_VER}/bin/Linux_x86_64/STAR"
    chmod +x "$BIN/STAR"
    "$BIN/STAR" --version >/dev/null 2>&1 \
      && echo "  STAR OK: $("$BIN/STAR" --version)" \
      || { echo "  FATAL: neither STAR build runs on this host"; exit 1; }
  fi
fi

# --- featureCounts (subread precompiled) ------------------------------------
if [ -x "$BIN/featureCounts" ] && "$BIN/featureCounts" -v >/dev/null 2>&1; then
  echo "  featureCounts already installed"
else
  echo "  downloading subread $SUBREAD_VER ..."
  tgz="$TOOLS/src/subread-${SUBREAD_VER}-Linux-x86_64.tar.gz"
  [ -s "$tgz" ] || curl -fSL -o "$tgz" \
    "https://downloads.sourceforge.net/project/subread/subread-${SUBREAD_VER}/subread-${SUBREAD_VER}-Linux-x86_64.tar.gz"
  tar xzf "$tgz" -C "$TOOLS/src"
  src="$TOOLS/src/subread-${SUBREAD_VER}-Linux-x86_64/bin"
  [ -x "$src/featureCounts" ] || { echo "  FATAL: featureCounts not in the tarball"; exit 1; }
  cp "$src/featureCounts" "$BIN/"
  chmod +x "$BIN/featureCounts"
fi
# featureCounts -v prints to stdout and exits non-zero on some builds
fc_ver=$("$BIN/featureCounts" -v 2>&1 | tr -d '\n' || true)
echo "  featureCounts OK: ${fc_ver:-installed}"

# --- python venv for the reagent-design side --------------------------------
# query.py guards umap/numba/plotly/matplotlib/pybigtools/te_fast behind
# try/except, so the hard requirements are small. Installed per-package and
# tolerantly: one wheel failing must not sink the rest.
VENV="$TOOLS/venv"
if [ "${SKIP_VENV:-0}" = "1" ]; then
  echo "  SKIP_VENV=1 -> not building the python env"
else
  if [ ! -x "$VENV/bin/python" ]; then
    echo "  creating python venv at $VENV ..."
    python3 -m venv "$VENV" || { echo "  venv creation FAILED (python3 -m venv)"; exit 1; }
    "$VENV/bin/pip" install --quiet --upgrade pip setuptools wheel 2>&1 | tail -1 || true
  fi
  echo "  installing python packages (per-package, failures tolerated) ..."
  for pkg in numpy pandas scipy scikit-learn requests primer3-py umap-learn matplotlib; do
    if "$VENV/bin/python" -c "import ${pkg//-/_}" >/dev/null 2>&1; then
      echo "    $pkg: already present"; continue
    fi
    if "$VENV/bin/pip" install --quiet "$pkg" >/dev/null 2>&1; then
      echo "    $pkg: installed"
    else
      echo "    $pkg: FAILED (optional ones are fine to skip)"
    fi
  done
  echo "  python env check:"
  for m in numpy pandas scipy sklearn primer3; do
    "$VENV/bin/python" -c "import $m" 2>/dev/null && echo "    $m: OK" || echo "    $m: MISSING"
  done
fi

echo
echo "== done =="
echo "  STAR          : $BIN/STAR"
echo "  featureCounts : $BIN/featureCounts"
[ -x "$VENV/bin/python" ] && echo "  python        : $VENV/bin/python"
echo
echo "  Run the pipeline without containers:"
echo "      USE_CONTAINER=0 bash scripts/run_endometrium_ltr.sh --clean --go"
echo "  (TOOLS=$TOOLS is picked up automatically; set TOOLS= if you moved it.)"
