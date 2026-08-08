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
  # Which interpreter to build the venv from. The conda 'base' env on this login
  # node is itself broken (it prints a libicui18n.so.75 error on every command),
  # so prefer a real system python and only fall back to whatever is on PATH.
  BASE_PY=${BASE_PY:-}
  if [ -z "$BASE_PY" ]; then
    for c in /usr/bin/python3.11 /usr/bin/python3.10 /usr/bin/python3.9 /usr/bin/python3 python3; do
      if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,8) else 1)' 2>/dev/null; then
        BASE_PY=$(command -v "$c"); break
      fi
    done
  fi
  [ -n "$BASE_PY" ] || { echo "  FATAL: no python3 >= 3.8 found (set BASE_PY=)"; exit 1; }
  echo "  base interpreter: $BASE_PY ($("$BASE_PY" -V 2>&1))"

  if [ ! -x "$VENV/bin/python" ] || [ "${REBUILD_VENV:-0}" = "1" ]; then
    rm -rf "$VENV"
    echo "  creating python venv at $VENV ..."
    "$BASE_PY" -m venv "$VENV" || { echo "  venv creation FAILED"; exit 1; }
  fi
  "$VENV/bin/python" -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1 || true
  echo "  pip: $("$VENV/bin/python" -m pip --version 2>&1 | cut -d' ' -f1-2)"

  # --no-cache-dir: $HOME here already holds ~58 GB of FASTQ, and a full quota
  # shows up as an inscrutable wheel failure.
  echo "  installing python packages ..."
  fails=0
  for pkg in numpy pandas scipy scikit-learn requests primer3-py umap-learn matplotlib; do
    mod=${pkg//-/_}; mod=${mod/scikit_learn/sklearn}; mod=${mod/primer3_py/primer3}; mod=${mod/umap_learn/umap}
    if "$VENV/bin/python" -c "import $mod" >/dev/null 2>&1; then
      echo "    $pkg: already present"; continue
    fi
    log="$TOOLS/src/pip-$pkg.log"
    if "$VENV/bin/python" -m pip install --no-cache-dir "$pkg" > "$log" 2>&1; then
      echo "    $pkg: installed"
    else
      fails=$((fails + 1))
      echo "    $pkg: FAILED -- reason:"
      # show the actual error rather than hiding it
      grep -iE 'error|no matching distribution|no space|quota|denied|killed' "$log" \
        | tail -3 | sed 's/^/        /' || tail -3 "$log" | sed 's/^/        /'
      echo "        full log: $log"
    fi
  done

  echo "  python env check:"
  missing=""
  for m in numpy pandas scipy sklearn primer3 requests; do
    if "$VENV/bin/python" -c "import $m" 2>/dev/null; then echo "    $m: OK"
    else echo "    $m: MISSING"; missing="$missing $m"; fi
  done
  if [ -n "$missing" ]; then
    echo
    echo "  The reagent-design stages need numpy/pandas/scipy/sklearn."
    echo "  Common causes, in order:"
    echo "    * disk quota in \$HOME  -> check:  du -sh ~ ; quota -s 2>/dev/null"
    echo "    * no outbound network from this host to pypi.org"
    echo "    * the base interpreter is unusable (try BASE_PY=/usr/bin/python3.9)"
    echo "  Retry after fixing:  REBUILD_VENV=1 bash scripts/setup_host_tools.sh"
    echo "  NOTE: STAR + featureCounts are already installed, so the EXPRESSION"
    echo "        half of the pipeline can run regardless of this."
  fi
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
