#!/usr/bin/env python3
"""
te_motif.py  —  GAMECA step M: Motif Analysis
─────────────────────────────────────────────────────────────────────────────
Overlaps TE loci against JASPAR TFBS predictions (via bedtools intersect)
then runs a per-cluster Fisher's exact test to find enriched TF motifs.

Input:   clustered CSV produced by te_clustering.py
         (needs columns: chr/Chromosome, start/Start, stop/Stop/End, Cluster)

Output:
  <out_dir>/motif_analysis/
    te_motif.log                   full run log
    all_overlaps.tsv               raw bedtools overlap output
    overall_motif_counts.csv       motif frequency across all loci
    overall_top_motifs.png         top-20 bar chart
    cluster_motif_counts.csv       per-cluster motif counts + sites_per_locus
    cluster_motif_counts_matrix.csv  motif x cluster wide matrix (+ total)
    cluster_N_top_motifs.png       top-20 bar chart per cluster
  <out_dir>/enrichment_results/
    cluster_N_enrichment.csv       Fisher p-values per cluster
    enrichment_heatmap.png         -log10(p) heatmap across clusters

JASPAR BED resolution order (--jaspar-bed is optional; auto-download always works)
  1. --jaspar-bed FILE
  2. TE_JASPAR_<BUILD> environment variable
  3. <jaspar-dir>/JASPAR2022_<build>.sorted.bed.gz  (CMMT cache)
  4. <jaspar-dir>/JASPAR2024_<build>.sorted.bed.gz  (legacy cache)
  5. Auto-query CMMT JASPAR 2022 BigBed (bigBedToBed, efficient)
  6. Auto-download CMMT per-motif TSV.gz files (no external tools needed)

Usage:
    python te_motif.py \\
        --input ./results/clustered.csv \\
        --build hg38 \\
        --out-dir ./results

    python te_motif.py --input data.csv --build mm10 \\
        --jaspar-bed /path/JASPAR2024_mm10.bed.gz
"""

import argparse
import gzip
import logging
import os
import sys
import time
import tempfile
import traceback
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# module-level logger — configured per-run by _setup_logger()
log = logging.getLogger("te_motif")


# ── Network self-detection ────────────────────────────────────────────────────
# HPC compute nodes frequently reach the internet only through a proxy that is
# configured in a curl-specific place (~/.curlrc, ~/.condarc) and NOT exported
# as the standard *_proxy environment variables that Python/pip honour. The
# helpers below discover whatever route works and apply it to the process
# environment so requests, pip and curl all use the same path — no user setup.

_NETWORK_ENV_APPLIED = False


def _detect_proxy():
    """Return an https proxy URL discovered from the environment/config, or None.

    Checks, in order: standard env vars (any case), ~/.curlrc, ~/.condarc,
    ~/.wgetrc and /etc/wgetrc. Returns the first proxy found.
    """
    # 1. Standard environment variables (any capitalisation, http or https)
    for var in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY",
                "all_proxy", "ALL_PROXY"):
        val = os.environ.get(var)
        if val:
            return val.strip()

    home = Path.home()

    # 2. ~/.curlrc  —  `proxy = http://host:port`  or  `--proxy http://host:port`
    curlrc = home / ".curlrc"
    if curlrc.exists():
        try:
            for raw in curlrc.read_text(errors="ignore").splitlines():
                line = raw.strip()
                if line.startswith("#") or not line:
                    continue
                low = line.lower()
                if low.startswith(("proxy", "-x", "--proxy")):
                    # forms: "proxy = url", "proxy=url", "-x url", "--proxy url"
                    for sep in ("=", " "):
                        if sep in line:
                            cand = line.split(sep, 1)[1].strip().strip('"').strip("'")
                            if cand:
                                return cand
        except Exception:
            pass

    # 3. ~/.condarc  —  proxy_servers: { https: url, http: url }
    condarc = home / ".condarc"
    if condarc.exists():
        try:
            txt = condarc.read_text(errors="ignore")
            for key in ("https:", "http:"):
                if key in txt:
                    seg = txt.split(key, 1)[1].splitlines()[0].strip()
                    seg = seg.strip('"').strip("'")
                    if seg.startswith(("http://", "https://")):
                        return seg
        except Exception:
            pass

    # 4. wgetrc files
    for wgetrc in (home / ".wgetrc", Path("/etc/wgetrc")):
        if wgetrc.exists():
            try:
                for raw in wgetrc.read_text(errors="ignore").splitlines():
                    line = raw.strip()
                    low = line.lower()
                    if low.startswith(("https_proxy", "http_proxy")) and "=" in line:
                        cand = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if cand:
                            return cand if "://" in cand else "http://" + cand
            except Exception:
                pass

    return None


def _apply_network_env():
    """Normalise any discovered proxy into the standard *_proxy env vars so that
    Python (requests/urllib), pip and curl all use the same route. Idempotent."""
    global _NETWORK_ENV_APPLIED
    if _NETWORK_ENV_APPLIED:
        return os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    _NETWORK_ENV_APPLIED = True
    proxy = _detect_proxy()
    if proxy:
        for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
            os.environ.setdefault(var, proxy)
        log.info("Network: using proxy %s (auto-detected)", proxy)
    else:
        log.info("Network: no proxy detected (checked *_proxy env, ~/.curlrc, "
                 "~/.condarc, ~/.wgetrc, /etc/wgetrc)")
    return proxy


def _curl_download(url, dest, timeout=120):
    """Download `url` to `dest` using curl (which works on HPC nodes where the
    Python socket stack is blocked). curl natively reads ~/.curlrc; we also pass
    any detected proxy explicitly. Returns True on success."""
    import subprocess
    proxy = _apply_network_env()
    cmd = ["curl", "-fsSL", 
           "-o", str(dest)]
    if proxy:
        cmd += ["--proxy", proxy]
    cmd.append(url)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and Path(dest).exists() and Path(dest).stat().st_size > 0:
            return True
        log.warning("curl download failed (%s): rc=%s %s",
                    url, r.returncode, (r.stderr or "")[:200])
    except Exception as exc:
        log.warning("curl download errored (%s): %s", url, exc)
    return False


class _GlibcIncompatibleError(RuntimeError):
    """Raised when bigBedToBed fails due to GLIBC mismatch and pybigtools is absent."""


# ── JASPAR download URLs ──────────────────────────────────────────────────────

# CMMT (UBC) hosts per-assembly JASPAR 2022 UCSC tracks.  These are the
# canonical, reliably-available source used for automatic download.
CMMT_BASE_URL = "https://expdata.cmmt.ubc.ca/JASPAR/downloads/UCSC_tracks/2022"
CMMT_ASSEMBLIES = {
    "araTha1", "ce10", "ce11", "ci3", "danRer11", "dm6",
    "hg19", "hg38", "mm10", "mm39", "sacCer3",
}

JASPAR_URLS = {
    # Legacy JASPAR 2024 BED URLs — kept only for error messages; server
    # routinely removes old releases so these may return 404.
    "hg38": "https://jaspar.elixir.no/static/data/beds/JASPAR2024_hg38.bed.gz",
    "hg19": "https://jaspar.elixir.no/static/data/beds/JASPAR2024_hg19.bed.gz",
    "mm10": "https://jaspar.elixir.no/static/data/beds/JASPAR2024_mm10.bed.gz",
    "mm39": "https://jaspar.elixir.no/static/data/beds/JASPAR2024_mm39.bed.gz",
}

JASPAR_BIGBED_URLS = {
    "hg38":     f"{CMMT_BASE_URL}/JASPAR2022_hg38.bb",
    "hg19":     f"{CMMT_BASE_URL}/JASPAR2022_hg19.bb",
    "mm10":     f"{CMMT_BASE_URL}/JASPAR2022_mm10.bb",
    "mm39":     f"{CMMT_BASE_URL}/JASPAR2022_mm39.bb",
    "danRer11": f"{CMMT_BASE_URL}/JASPAR2022_danRer11.bb",
    "dm6":      f"{CMMT_BASE_URL}/JASPAR2022_dm6.bb",
    "sacCer3":  f"{CMMT_BASE_URL}/JASPAR2022_sacCer3.bb",
    "araTha1":  f"{CMMT_BASE_URL}/JASPAR2022_araTha1.bb",
    "ce10":     f"{CMMT_BASE_URL}/JASPAR2022_ce10.bb",
    "ce11":     f"{CMMT_BASE_URL}/JASPAR2022_ce11.bb",
    "mm39":     f"{CMMT_BASE_URL}/JASPAR2022_mm39.bb",
}

def jaspar_source_reachable(build):
    """Return True if a no-binary JASPAR source (CMMT host) is reachable.

    The bulk-BED and per-motif download fallbacks need only network — no
    bigBedToBed and no pybigtools. This lets the preflight decide whether the
    motif stage can succeed via those paths before committing to a run.
    Tries curl (works where Python sockets are blocked), then urllib.
    """
    import subprocess
    _apply_network_env()
    url = JASPAR_BIGBED_URLS.get(build) or (CMMT_BASE_URL + "/")
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    # 1. curl HEAD
    cmd = ["curl", "-sI"]
    if proxy:
        cmd += ["--proxy", proxy]
    cmd.append(url)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        first = (r.stdout or "").splitlines()[:1]
        if r.returncode == 0 and first and any(c in first[0] for c in ("200", "206", "301", "302", "403")):
            return True
    except Exception:
        pass
    # 2. urllib fallback (some nodes block curl but not Python, or vice-versa)
    try:
        import urllib.request
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "te_motif/1.0"})
        with urllib.request.urlopen(req) as resp:
            return 200 <= resp.status < 400
    except Exception as exc:
        log.info("JASPAR source not reachable via urllib either: %s", exc)
    return False

_DEFAULT_BASE = os.environ.get("TE_BASE_DIR",   str(Path.home() / "te_analysis"))
_DEFAULT_JASP = os.environ.get("TE_JASPAR_DIR", str(Path(_DEFAULT_BASE) / "jaspar"))


# ── Logging setup ─────────────────────────────────────────────────────────────

def _setup_logger(out_dir: Path) -> Path:
    """Configure root te_motif logger: timestamped console + file handler.

    Returns the path to the log file.
    """
    log_path = out_dir / "motif_analysis" / "te_motif.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s",
                            datefmt="%H:%M:%S")

    # Console: INFO and above
    if not any(isinstance(h, logging.StreamHandler) and
               not isinstance(h, logging.FileHandler) for h in log.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        log.addHandler(sh)

    # File: DEBUG and above (overwrites on each run)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)

    return log_path


def _log_system_info(scratch: str) -> None:
    """Log Python version, key env vars, and scratch disk space."""
    import platform, shutil as _shutil
    log.info("=== SYSTEM INFO ===")
    log.info("  Python %s  |  platform: %s", sys.version.split()[0], platform.platform())
    for var in ("TMPDIR", "SLURM_JOB_ID", "LSB_JOBID", "HOSTNAME", "USER"):
        val = os.environ.get(var, "(not set)")
        log.debug("  env %s = %s", var, val)
    try:
        total, used, free = _shutil.disk_usage(scratch)
        log.info("  Scratch dir: %s  |  free: %.1f GB / %.1f GB total",
                 scratch, free / 1e9, total / 1e9)
        if free < 2 * 1e9:
            log.warning("  LOW DISK: only %.1f GB free in scratch — bedtools may fail", free / 1e9)
    except Exception as exc:
        log.warning("  Could not check disk usage for %s: %s", scratch, exc)
    try:
        import psutil
        vm = psutil.virtual_memory()
        log.info("  RAM: %.1f GB free / %.1f GB total", vm.available / 1e9, vm.total / 1e9)
    except ImportError:
        log.debug("  psutil not installed — skipping RAM check")
    except Exception as exc:
        log.debug("  psutil error: %s", exc)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="GAMECA step M: JASPAR motif overlap + Fisher enrichment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Input: one of --input (clustered CSV) or --bed-input (existing BED file)
    p.add_argument("--input", default=None,
                   help="Clustered CSV (from te_clustering.py)")
    p.add_argument("--bed-input", default=None,
                   help="Existing TE loci BED file (chr/start/stop, 3+ columns). "
                        "All loci are assigned to cluster 0. "
                        "Alternative to --input when no clustered CSV is available.")
    p.add_argument("--build",      default="hg38", help="Genome build: hg38/hg19/mm10/mm39")
    p.add_argument("--out-dir",    default=".", help="Output directory")
    p.add_argument("--jaspar-bed", default=None,
                   help="Path to JASPAR BED (auto-downloaded if omitted)")
    p.add_argument("--jaspar-dir", default=_DEFAULT_JASP,
                   help=f"Cache directory for JASPAR BED files (default: {_DEFAULT_JASP})")
    p.add_argument("--p-threshold", type=float, default=0.05,
                   help="Fisher p-value significance threshold (default: 0.05)")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if overlap file already exists")
    p.add_argument("--homer", action="store_true",
                   help="Also run HOMER findMotifsGenome.pl on each cluster")
    p.add_argument("--homer-genome", default=None,
                   help="HOMER genome name or FASTA path (e.g. hg38, mm10); "
                        "defaults to --build value")
    p.add_argument("--homer-size", default="200",
                   help="HOMER -size parameter (default: 200)")
    p.add_argument("--homer-threads", type=int, default=4,
                   help="HOMER -p threads per cluster (default: 4)")
    # GO chaining
    p.add_argument("--run-go", action="store_true",
                   help="Chain te_go.py GO annotation immediately after motif analysis")
    p.add_argument("--go-top-motifs", type=int, default=30,
                   help="Top motifs per cluster to annotate in chained GO run (default: 30)")
    p.add_argument("--notify-email", default="", metavar="EMAIL",
                   help="Send a completion email to this address (requires Gmail App Password setup).")
    args = p.parse_args()
    if not args.input and not args.bed_input:
        p.error("Provide --input (clustered CSV) or --bed-input (existing BED file).")
    if args.input and args.bed_input:
        p.error("--input and --bed-input are mutually exclusive.")
    return args


# ── JASPAR BED helpers ────────────────────────────────────────────────────────

def _get_bigbed_to_bed(jaspar_dir):
    """Return a bigBedToBed executable, downloading the Linux binary if possible."""
    import platform
    import shutil
    import stat

    env_path = os.environ.get("BIGBEDTOBED", "")
    if env_path and Path(env_path).exists():
        log.debug("bigBedToBed from env: %s", env_path)
        return env_path

    found = shutil.which("bigBedToBed")
    if found:
        log.debug("bigBedToBed on PATH: %s", found)
        return found

    system = platform.system().lower()
    machine = platform.machine().lower()
    ucsc_platform = None
    if system == "linux" and machine in {"x86_64", "amd64"}:
        ucsc_platform = "linux.x86_64"
    elif system == "darwin" and machine == "arm64":
        ucsc_platform = "macOSX.arm64"
    elif system == "darwin" and machine in {"x86_64", "amd64"}:
        ucsc_platform = "macOSX.x86_64"
    if not ucsc_platform:
        log.warning("bigBedToBed: unsupported platform %s/%s", system, machine)
        return None

    bin_dir = Path(jaspar_dir) / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    exe = bin_dir / "bigBedToBed"
    if exe.exists():
        # Verify the cached binary actually runs on this system's glibc
        import subprocess as _sp
        _probe = _sp.run([str(exe)], capture_output=True, text=True)
        _combined = _probe.stdout + _probe.stderr
        if "GLIBC" in _combined or "version `" in _combined or "version not found" in _combined:
            log.warning("Cached bigBedToBed is incompatible with this system's glibc; removing")
            exe.unlink(missing_ok=True)
        else:
            log.debug("bigBedToBed cached: %s", exe)
            return str(exe)

    url = f"https://hgdownload.soe.ucsc.edu/admin/exe/{ucsc_platform}/bigBedToBed"
    log.info("bigBedToBed not found — downloading UCSC utility: %s", url)
    # Prefer curl (works on HPC nodes where Python sockets are blocked); fall
    # back to requests if curl is unavailable.
    if _curl_download(url, exe, timeout=120):
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        log.info("bigBedToBed downloaded (curl) → %s", exe)
        return str(exe)
    try:
        import requests
        _apply_network_env()
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(exe, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        log.info("bigBedToBed downloaded → %s", exe)
        return str(exe)
    except Exception as exc:
        log.error("Could not download bigBedToBed: %s", exc)
        log.debug(traceback.format_exc())
        return None


def _query_bigbed_pybigtools(bb_url, loci, out):
    """Query bigBed via pybigtools (no external binary, no glibc dependency).

    Returns (total_records_written, had_error).
    """
    import pybigtools  # noqa: PLC0415
    bb = pybigtools.open(bb_url, "r")
    seen: set = set()
    for _, row in loci.iterrows():
        chrom = str(row["chrom"])
        start = max(0, int(row["start"]) - 1)
        end   = int(row["end"]) + 1
        try:
            # pybigtools API: .records(chrom, start, end) yields (start, end, *rest)
            # for bigBed; rest fields (name, score, strand, …) are split by space.
            for rec in bb.records(chrom, start, end):
                s, e = rec[0], rec[1]
                rest = "\t".join(str(x) for x in rec[2:])
                line = f"{chrom}\t{s}\t{e}" + (f"\t{rest}" if rest else "")
                if line in seen:
                    continue
                seen.add(line)
                out.write(line + "\n")
        except Exception:
            pass
    return len(seen)


def _ensure_pybigtools():
    """Import pybigtools, attempting an automatic pip install if it is missing.

    pybigtools is a Rust-based wheel with no glibc dependency, so it is the
    correct fix when the bundled bigBedToBed binary is GLIBC-incompatible.
    Returns True if pybigtools is importable after this call, else False.
    """
    try:
        import pybigtools  # noqa: F401, PLC0415
        return True
    except ImportError:
        pass

    import importlib
    import subprocess
    import sys

    log.info("pybigtools not installed — attempting automatic pip install ...")
    proxy = _apply_network_env()  # normalise proxy into env for pip + requests
    proxy_args = ["--proxy", proxy] if proxy else []

    def _retry_import():
        importlib.invalidate_caches()
        import pybigtools  # noqa: F401, PLC0415
        return True

    # Try a normal install first, then a --user install (works without write
    # access to the conda/site-packages dir, common on shared HPC nodes).
    # --only-binary=:all: prevents pip from falling back to a (slow, usually
    # failing) Rust source build when no compatible wheel exists.
    base = [sys.executable, "-m", "pip", "install", "--quiet",
            "--disable-pip-version-check", "--only-binary=:all:"]
    attempts = (
        base + proxy_args + ["pybigtools"],
        base + proxy_args + ["--user", "pybigtools"],
    )
    for cmd in attempts:
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                log.warning("  pip install failed: %s",
                            (res.stderr or res.stdout)[-300:])
                continue
            log.info("  pybigtools installed via pip%s", " (proxy)" if proxy else "")
            return _retry_import()
        except subprocess.TimeoutExpired:
            log.warning("  pip install timed out (>600s)")
        except Exception as exc:
            log.warning("  pip install attempt errored: %s", exc)

    # Last resort: pip's Python socket stack is blocked but curl has a route.
    # Fetch a compatible wheel from PyPI via curl and install it offline.
    if _install_wheel_via_curl("pybigtools"):
        try:
            return _retry_import()
        except Exception:
            pass

    log.error("  Could not auto-install pybigtools (no network / no pip / restricted env).")
    return False


def _install_wheel_via_curl(pkg):
    """Fetch a platform-compatible wheel for `pkg` from PyPI using curl, then
    pip-install it from the local file. Works when only curl can reach the net.
    Returns True if a wheel was downloaded and installed."""
    import json
    import subprocess
    import sys
    import sysconfig

    log.info("  Falling back to curl wheel fetch for %s ...", pkg)
    meta = Path(tempfile.gettempdir()) / f"{pkg}_pypi.json"
    if not _curl_download(f"https://pypi.org/pypi/{pkg}/json", meta, timeout=60):
        return False
    try:
        data = json.loads(meta.read_text())
    except Exception as exc:
        log.warning("  Could not parse PyPI metadata: %s", exc)
        return False

    pyver = f"cp{sys.version_info.major}{sys.version_info.minor}"
    plat = (sysconfig.get_platform() or "").lower()  # e.g. linux-x86_64

    # Use pip/packaging's own tag logic so we only accept a wheel this
    # interpreter+glibc can actually install (e.g. a manylinux_2_28 wheel is
    # rejected on a glibc<2.28 node instead of being downloaded and failing).
    try:
        from packaging.tags import sys_tags
        from packaging.utils import parse_wheel_filename
        _compat = {str(t) for t in sys_tags()}

        def _wheel_compatible(fname):
            if not fname.endswith(".whl"):
                return False
            try:
                _, _, _, tagset = parse_wheel_filename(fname)
            except Exception:
                return False
            return any(str(t) in _compat for t in tagset)
    except Exception:
        # packaging unavailable — fall back to a conservative string heuristic.
        arch = "x86_64" if "x86_64" in plat or "amd64" in plat else (
            "aarch64" if "aarch64" in plat or "arm64" in plat else "")

        def _wheel_compatible(fname):
            f = fname.lower()
            if not f.endswith(".whl"):
                return False
            if pyver not in f and "abi3" not in f and "py3" not in f:
                return False
            if "linux" in plat and arch and arch not in f:
                return False
            return True

    # Pick the newest release version that has an installable wheel.
    best = None
    versions = sorted(data.get("releases", {}), reverse=True)
    # Prefer PEP 440 ordering when packaging is available.
    try:
        from packaging.version import Version, InvalidVersion

        def _vkey(v):
            try:
                return Version(v)
            except InvalidVersion:
                return Version("0")
        versions = sorted(data.get("releases", {}), key=_vkey, reverse=True)
    except Exception:
        pass
    for ver in versions:
        for rel in data["releases"].get(ver, []):
            if _wheel_compatible(rel.get("filename", "")):
                best = rel
                break
        if best:
            break

    if not best:
        log.warning("  No installable %s wheel exists for this node "
                    "(python=%s, platform=%s). Prebuilt wheels are incompatible "
                    "with this glibc — skipping pybigtools.", pkg, pyver, plat)
        return False

    wheel = Path(tempfile.gettempdir()) / best["filename"]
    if not _curl_download(best["url"], wheel, timeout=300):
        return False
    log.info("  Downloaded wheel: %s", best["filename"])
    cmd = [sys.executable, "-m", "pip", "install", "--quiet",
           "--disable-pip-version-check", "--no-index", str(wheel)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            res = subprocess.run(cmd + ["--user"], capture_output=True, text=True)
        if res.returncode == 0:
            log.info("  Installed %s from local wheel.", pkg)
            return True
        log.warning("  Offline wheel install failed: %s", (res.stderr or res.stdout)[-300:])
    except Exception as exc:
        log.warning("  Offline wheel install errored: %s", exc)
    return False


def _build_locus_jaspar_bed_from_bigbed(build, jaspar_dir, loci_bed):
    """Query remote JASPAR bigBed by TE chromosome spans and write a BED file.

    Tries pybigtools first (pure Python/Rust, no glibc dependency), then falls
    back to the bigBedToBed binary if pybigtools is not installed.
    """
    import subprocess

    bb_url = JASPAR_BIGBED_URLS.get(build)
    if not bb_url or not loci_bed:
        return None

    try:
        loci = pd.read_csv(loci_bed, sep="\t", header=None, names=["chrom", "start", "end"])
    except Exception as exc:
        log.error("Failed to read loci BED %s: %s", loci_bed, exc)
        log.debug(traceback.format_exc())
        return None

    loci["start"] = pd.to_numeric(loci["start"], errors="coerce")
    loci["end"]   = pd.to_numeric(loci["end"],   errors="coerce")
    loci = loci.dropna(subset=["chrom", "start", "end"])
    if loci.empty:
        log.warning("Loci BED is empty after parsing; cannot query bigBed.")
        return None

    out_path = Path(jaspar_dir) / f"JASPAR2022_{build}.te_loci.bed"
    tmp_path = out_path.with_suffix(".bed.part")
    log.info("Auto-querying JASPAR bigBed for %d TE loci", len(loci))
    log.info("  Source: %s", bb_url)
    log.info("  Output: %s", out_path)
    t0 = time.time()

    # ── Path 1: pybigtools (Rust-based, works on any glibc / macOS / Windows) ──
    # Auto-install if missing — this is the glibc-independent fix.
    _ensure_pybigtools()
    _pbt_available = False
    try:
        import pybigtools as _pbt  # noqa: F401, PLC0415
        import concurrent.futures as _cf
        _pbt_available = True
        log.info("  Using pybigtools for bigBed query (timeout=300s)")
        with open(tmp_path, "w") as out:
            _ex = _cf.ThreadPoolExecutor(max_workers=1)
            _fut = _ex.submit(_query_bigbed_pybigtools, bb_url, loci, out)
            try:
                total = _fut.result()
            except _cf.TimeoutError:
                log.warning("  pybigtools bigBed query timed out (>300s)")
                total = 0
            finally:
                _ex.shutdown(wait=False)
        elapsed = time.time() - t0
        if total > 0:
            tmp_path.replace(out_path)
            log.info("bigBed query complete (pybigtools): %d records in %.1fs", total, elapsed)
            if validate_jaspar_bed(out_path):
                log.info("JASPAR locus BED ready: %s", out_path)
                return str(out_path)
            out_path.unlink(missing_ok=True)
            log.error("bigBed-derived BED failed validation; discarding")
        else:
            tmp_path.unlink(missing_ok=True)
            log.warning("pybigtools returned 0 records (elapsed=%.1fs) — trying bigBedToBed", elapsed)
    except ImportError:
        log.info("  pybigtools not installed — trying bigBedToBed binary")
    except Exception as exc:
        log.warning("  pybigtools query failed (%s) — trying bigBedToBed binary", exc)
        log.debug(traceback.format_exc())
        tmp_path.unlink(missing_ok=True)

    # ── Path 2: bigBedToBed binary (fallback) ────────────────────────────────
    tool = _get_bigbed_to_bed(jaspar_dir)
    if not tool:
        log.warning("bigBedToBed is not available; cannot auto-query JASPAR bigBed.")
        return None

    seen: set = set()
    errors = 0
    total = 0
    with open(tmp_path, "w") as out:
        for row_i, row in loci.iterrows():
            chrom = str(row["chrom"])
            start = max(0, int(row["start"]) - 1)
            end   = int(row["end"]) + 1
            cmd   = [tool, bb_url, "stdout",
                     f"-chrom={chrom}", f"-start={start}", f"-end={end}"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                errors += 1
                log.warning("bigBedToBed failed for %s:%d-%d: %s",
                            chrom, start, end, result.stderr[:200])
                if "GLIBC" in result.stderr or "version `" in result.stderr or \
                        "version not found" in result.stderr:
                    try:
                        Path(tool).unlink(missing_ok=True)
                    except Exception:
                        pass
                    if not _pbt_available:
                        raise _GlibcIncompatibleError(
                            "bigBedToBed requires GLIBC_2.29+ (not available on this node) "
                            "and pybigtools is not installed — cannot auto-fetch JASPAR. "
                            "Fix: pip install pybigtools  OR  pass --jaspar-bed /path/to/JASPAR.bed.gz"
                        )
                    log.error("bigBedToBed is incompatible with this system's glibc — "
                              "pybigtools was also unavailable")
                    break
                continue
            lines = [ln for ln in result.stdout.splitlines()
                     if ln and not ln.startswith(("#", "track"))]
            for line in lines:
                if line in seen:
                    continue
                seen.add(line)
                out.write(line + "\n")
            total = len(seen)
            if (row_i + 1) % 100 == 0 or row_i == len(loci) - 1:
                log.info("  queried %d/%d loci; %d unique TFBS records (errors=%d)",
                         row_i + 1, len(loci), total, errors)

    elapsed = time.time() - t0
    if total == 0:
        tmp_path.unlink(missing_ok=True)
        log.error("No JASPAR records extracted from bigBed (errors=%d, elapsed=%.1fs).",
                  errors, elapsed)
        return None

    tmp_path.replace(out_path)
    log.info("bigBed query complete (binary): %d records in %.1fs", total, elapsed)
    if validate_jaspar_bed(out_path):
        log.info("JASPAR locus BED ready: %s", out_path)
        return str(out_path)
    out_path.unlink(missing_ok=True)
    log.error("bigBed-derived BED failed validation; discarding %s", out_path)
    return None


# ── CMMT per-motif TSV.gz helpers ────────────────────────────────────────────

def _list_cmmt_files(build):
    """Return list of (filename, url) for all per-motif TSV.gz files on CMMT."""
    import re, urllib.request
    url = f"{CMMT_BASE_URL}/{build}/"
    req = urllib.request.Request(url, headers={"User-Agent": "te_motif/1.0"})
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    # Match href with single or double quotes; strip any leading path component
    files = re.findall(r"""href=["'](?:[^"']*/)?(MA\d+[\w.]+\.tsv\.gz)["']""", html)
    if not files:
        log.debug("CMMT directory listing sample (first 500 chars): %s", html[:500])
    return [(f, f"{url}{f}") for f in files]


def _build_loci_lookup(loci_bed):
    """Return chr → sorted list of (start, end) for fast overlap testing."""
    loci = {}
    try:
        with open(loci_bed) as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                try:
                    start, end = int(parts[1]), int(parts[2])
                except ValueError:
                    continue
                loci.setdefault(parts[0], []).append((start, end))
    except Exception as exc:
        log.warning("_build_loci_lookup: could not read %s: %s", loci_bed, exc)
        return None
    for chr_ in loci:
        loci[chr_].sort()
    return loci


def _overlaps_any(chr_, start, end, loci_lookup):
    """Return True if (chr_, start, end) overlaps any interval in loci_lookup."""
    import bisect
    intervals = loci_lookup.get(chr_)
    if not intervals:
        return False
    idx = bisect.bisect_left(intervals, (start,))
    if idx > 0 and intervals[idx - 1][1] > start:
        return True
    while idx < len(intervals) and intervals[idx][0] < end:
        if intervals[idx][1] > start:
            return True
        idx += 1
    return False


def _download_cmmt_per_motif(build, jaspar_dir, loci_bed=None):
    """Stream-download JASPAR 2022 per-motif TSV.gz files from CMMT.

    Each file is decompressed on the fly.  When loci_bed is provided only rows
    that overlap the TE loci are kept, avoiding multi-GB local storage.  The
    merged result is sorted and cached as JASPAR2022_{build}.sorted.bed.gz.

    Runs to completion (no overall time cap); returns the path to the cached
    BED.gz, or None on failure.
    """
    import re, urllib.request

    if build not in CMMT_ASSEMBLIES:
        log.warning("Build '%s' not in CMMT assemblies; skipping per-motif download.", build)
        return None

    jaspar_dir = Path(jaspar_dir)
    out_path = jaspar_dir / f"JASPAR2022_{build}.sorted.bed.gz"

    log.info("CMMT per-motif download: build=%s  dir=%s", build, jaspar_dir)

    # Fetch directory listing
    try:
        file_list = _list_cmmt_files(build)
    except Exception as exc:
        log.error("Could not list CMMT files for %s: %s", build, exc)
        return None

    if not file_list:
        log.error("No .tsv.gz files found for build '%s' on CMMT", build)
        return None
    log.info("  %d motif files found for %s", len(file_list), build)

    # Build interval lookup for filtering
    loci_lookup = None
    if loci_bed and Path(loci_bed).exists():
        loci_lookup = _build_loci_lookup(loci_bed)
        if loci_lookup:
            log.info("  Loci filter active: %d chromosomes in TE loci", len(loci_lookup))
        else:
            log.warning("  Could not build loci lookup; downloading all rows (may be large)")

    tmp_path = out_path.with_suffix(".bed.gz.part")
    t0 = time.time()
    total_written = 0

    # Modest parallelism: each motif file is an independent HTTP GET. Kept low
    # (default 4) to stay polite to the CMMT server; override TE_CMMT_WORKERS.
    # Download+parse happen in worker threads; gzip writing stays single-threaded.
    import concurrent.futures as _cf
    n_workers = max(1, int(os.environ.get("TE_CMMT_WORKERS", "4")))
    n_total = len(file_list)

    def _fetch_one(item):
        fname, url = item
        rows = []
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "te_motif/1.0"})
            with urllib.request.urlopen(req) as resp:
                with gzip.open(resp, "rt", encoding="utf-8", errors="replace") as gz:
                    for line in gz:
                        parts = line.rstrip("\n").split("\t")
                        if len(parts) < 7:
                            continue
                        try:
                            start, end = int(parts[1]), int(parts[2])
                        except ValueError:
                            continue
                        chr_ = parts[0]
                        if loci_lookup is not None and not _overlaps_any(
                                chr_, start, end, loci_lookup):
                            continue
                        # 6-col BED: chr start end motif_name score strand
                        rows.append(
                            f"{chr_}\t{start}\t{end}\t{parts[3]}\t{parts[4]}\t{parts[6]}\n")
            return fname, rows, None
        except Exception as exc:
            return fname, None, exc

    try:
        done = errors = 0
        log.info("  Downloading %d files with %d parallel workers ...", n_total, n_workers)
        with gzip.open(str(tmp_path), "wt") as out_fh:
            with _cf.ThreadPoolExecutor(max_workers=n_workers) as ex:
                futures = [ex.submit(_fetch_one, item) for item in file_list]
                for fut in _cf.as_completed(futures):
                    fname, rows, exc = fut.result()
                    done += 1
                    if exc is not None:
                        errors += 1
                        log.warning("  Failed to download/parse %s: %s", fname, exc)
                        continue
                    for r in rows:
                        out_fh.write(r)
                    total_written += len(rows)
                    if done == 1 or done % 50 == 0 or done == n_total:
                        log.info("  [%d/%d] %d rows so far (%d workers, %d errors)",
                                 done, n_total, total_written, n_workers, errors)

        if total_written == 0:
            tmp_path.unlink(missing_ok=True)
            log.error("CMMT per-motif: no rows written (loci filter too strict or download failed)")
            return None

        log.info("  %d rows written in %.1fs — sorting ...", total_written, time.time() - t0)

        # Sort by chr, start  (subprocess pipeline if available, else pure-Python)
        import shutil, subprocess as _sp
        if shutil.which("sort") and shutil.which("bgzip"):
            with open(out_path, "wb") as fh:
                p1 = _sp.Popen(["zcat", str(tmp_path)], stdout=_sp.PIPE)
                p2 = _sp.Popen(["sort", "-k1,1", "-k2,2n", "-S", "1G"],
                                stdin=p1.stdout, stdout=_sp.PIPE)
                p1.stdout.close()
                p3 = _sp.Popen(["bgzip", "-c"], stdin=p2.stdout, stdout=fh)
                p2.stdout.close()
                p3.communicate(); p2.wait(); p1.wait()
                if p3.returncode != 0:
                    log.warning("bgzip pipeline failed; falling back to plain gzip sort")
                    out_path.unlink(missing_ok=True)
                else:
                    tmp_path.unlink(missing_ok=True)
        if not out_path.exists():
            # Pure-Python sort fallback
            rows = []
            with gzip.open(str(tmp_path), "rt") as fh:
                for line in fh:
                    p = line.strip().split("\t")
                    if len(p) >= 3:
                        try:
                            rows.append((p[0], int(p[1]), line))
                        except ValueError:
                            pass
            rows.sort(key=lambda x: (x[0], x[1]))
            with gzip.open(str(out_path), "wt") as fh:
                for _, _, line in rows:
                    fh.write(line)
            tmp_path.unlink(missing_ok=True)

        log.info("CMMT per-motif download complete: %.1f MB in %.1fs → %s",
                 out_path.stat().st_size / 1e6, time.time() - t0, out_path)
        return str(out_path)

    except Exception as exc:
        log.error("CMMT per-motif download failed: %s", exc)
        log.debug(traceback.format_exc())
        tmp_path.unlink(missing_ok=True)
        return None


def _download_jaspar_bulk_bed(build, jaspar_dir, loci_bed=None):
    """Download the single bulk JASPAR 2024 BED from jaspar.elixir.no (~500 MB).

    Much faster than per-motif streaming.  Filters to TE loci when loci_bed is
    provided.  Saves as JASPAR2024_{build}.sorted.bed.gz.
    Returns path or None on failure.
    """
    import urllib.request, shutil as _sh

    url = JASPAR_URLS.get(build)
    if not url:
        log.info("No bulk JASPAR URL known for build '%s'; skipping.", build)
        return None

    jaspar_dir = Path(jaspar_dir)
    out_path = jaspar_dir / f"JASPAR2024_{build}.sorted.bed.gz"
    raw_gz   = jaspar_dir / f"JASPAR2024_{build}.raw.bed.gz"

    if not raw_gz.exists():
        log.info("Bulk JASPAR 2024 BED download (~500 MB): %s", url)
        t0 = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "te_motif/1.0"})
            with urllib.request.urlopen(req) as resp, \
                 open(raw_gz, "wb") as fh:
                _sh.copyfileobj(resp, fh)
            mb = raw_gz.stat().st_size / 1e6
            log.info("  Downloaded %.1f MB in %.1fs", mb, time.time() - t0)
            # Content sniff: a dead/redirected URL (e.g. the retired
            # jaspar.elixir.no/static/data/beds path) returns a tiny HTML 404
            # page with HTTP 200 via some proxies. Reject anything that is not a
            # real gzip stream so a 404 page is never saved as a "cache".
            with open(raw_gz, "rb") as _fh:
                magic = _fh.read(2)
            if magic != b"\x1f\x8b":
                log.error("Bulk JASPAR download is not gzip (magic=%s) — likely an "
                          "HTML error page; aborting: %s", magic.hex(), url)
                raw_gz.unlink(missing_ok=True)
                return None
            if mb < 1.0:
                log.error("Bulk JASPAR download too small (%.1f MB); aborting", mb)
                raw_gz.unlink(missing_ok=True)
                return None
        except Exception as exc:
            log.warning("Bulk JASPAR download failed: %s", exc)
            raw_gz.unlink(missing_ok=True)
            return None
    else:
        log.info("Bulk JASPAR raw BED already present: %s", raw_gz)

    loci_lookup = _build_loci_lookup(loci_bed) if loci_bed else None
    try:
        log.info("  Sorting bulk JASPAR BED%s ...",
                 " (with loci filter)" if loci_lookup else "")
        rows = []
        with gzip.open(str(raw_gz), "rt", encoding="utf-8", errors="replace") as gz:
            for line in gz:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                try:
                    chrom, start, end = parts[0], int(parts[1]), int(parts[2])
                except ValueError:
                    continue
                if loci_lookup and not _overlaps_any(chrom, start, end, loci_lookup):
                    continue
                rows.append((chrom, start, line))
        if not rows:
            log.error("Bulk JASPAR BED: no rows after filter")
            return None
        rows.sort(key=lambda x: (x[0], x[1]))
        with gzip.open(str(out_path), "wt") as fh:
            for _, _, line in rows:
                fh.write(line)
        raw_gz.unlink(missing_ok=True)
        log.info("Bulk JASPAR BED ready: %d rows → %s", len(rows), out_path)
        return str(out_path) if validate_jaspar_bed(out_path) else None
    except Exception as exc:
        log.warning("Bulk JASPAR sort/filter failed: %s", exc)
        raw_gz.unlink(missing_ok=True)
        return None


def resolve_jaspar_bed(build, jaspar_bed_arg, jaspar_dir, loci_bed=None):
    """Return path to a valid JASPAR BED, downloading if necessary."""
    jaspar_dir = Path(jaspar_dir)
    jaspar_dir.mkdir(parents=True, exist_ok=True)
    log.debug("resolve_jaspar_bed: build=%s jaspar_dir=%s arg=%s", build, jaspar_dir, jaspar_bed_arg)

    if jaspar_bed_arg and Path(jaspar_bed_arg).exists():
        log.info("Using provided --jaspar-bed: %s", jaspar_bed_arg)
        if not validate_jaspar_bed(jaspar_bed_arg, raise_on_error=True):
            log.error("Provided --jaspar-bed is not a valid BED/BED.GZ: %s", jaspar_bed_arg)
            sys.exit(1)
        return jaspar_bed_arg

    env_path = os.environ.get(f"TE_JASPAR_{build.upper()}", "")
    if env_path and Path(env_path).exists():
        log.info("Using JASPAR BED from env TE_JASPAR_%s: %s", build.upper(), env_path)
        if not validate_jaspar_bed(env_path, raise_on_error=True):
            log.error("TE_JASPAR_%s is not a valid BED/BED.GZ: %s", build.upper(), env_path)
            sys.exit(1)
        return env_path

    # ── 1. CMMT 2022 cache (from per-motif or bigBed auto-download) ───────────
    cmmt_cache = jaspar_dir / f"JASPAR2022_{build}.sorted.bed.gz"
    if cmmt_cache.exists():
        log.info("Found CMMT-cached JASPAR BED: %s (%.1f MB)",
                 cmmt_cache, cmmt_cache.stat().st_size / 1e6)
        if validate_jaspar_bed(cmmt_cache):
            log.info("Cached JASPAR 2022 BED is valid — using it.")
            return str(cmmt_cache)
        log.warning("CMMT cache invalid; deleting and re-downloading: %s", cmmt_cache)
        cmmt_cache.unlink()

    # ── 2. Legacy JASPAR 2024 cache (backward-compat) ─────────────────────────
    local_path = jaspar_dir / f"JASPAR2024_{build}.sorted.bed.gz"
    if local_path.exists():
        log.info("Found legacy-cached JASPAR BED: %s (%.1f MB)",
                 local_path, local_path.stat().st_size / 1e6)
        if validate_jaspar_bed(local_path):
            log.info("Legacy cached JASPAR BED is valid — using it.")
            return str(local_path)
        log.warning("Legacy cached JASPAR file is invalid; deleting: %s", local_path)
        local_path.unlink()

    # ── 3. CMMT BigBed remote query (efficient; needs bigBedToBed) ────────────
    log.info("No cached JASPAR BED found — auto-downloading from CMMT (JASPAR 2022) ...")
    log.info("Tip: set TE_JASPAR_%s=/path/to/file to skip future downloads", build.upper())
    try:
        locus_bed = _build_locus_jaspar_bed_from_bigbed(build, jaspar_dir, loci_bed)
    except _GlibcIncompatibleError as exc:
        # bigBed needs glibc>=2.29 / pybigtools, neither available here. Do NOT
        # abort — fall through to the no-binary bulk / per-motif downloads below,
        # which only need network (and produce identical JASPAR BED output).
        log.warning("bigBed path unavailable (%s)", exc)
        log.warning("Falling back to no-binary JASPAR download (bulk / per-motif) ...")
        locus_bed = None
    if locus_bed:
        return locus_bed

    # ── 3.5. Single bulk BED from jaspar.elixir.no (~500 MB, one request) ───────
    log.info("Trying single-file bulk JASPAR 2024 BED download ...")
    bulk_path = _download_jaspar_bulk_bed(build, jaspar_dir, loci_bed=loci_bed)
    if bulk_path:
        return bulk_path

    # ── 4. CMMT per-motif TSV.gz streaming (last resort; capped at 30 min) ────
    log.info("Bulk download failed — falling back to per-motif CMMT download ...")
    cmmt_path = _download_cmmt_per_motif(build, jaspar_dir, loci_bed=loci_bed)
    if cmmt_path:
        return cmmt_path

    log.error("FATAL: all JASPAR BED download methods failed.")
    bb = JASPAR_BIGBED_URLS.get(build)
    if bb:
        log.error("CMMT hosts JASPAR 2022 data for this build.")
        log.error("  BigBed URL : %s", bb)
        log.error("  Per-motif  : %s/%s/", CMMT_BASE_URL, build)
        log.error("")
        log.error("To fix: download the bigBed on a machine with glibc >= 2.29 and convert:")
        log.error("  wget -O JASPAR2022_%s.bb '%s'", build, bb)
        log.error("  bigBedToBed JASPAR2022_%s.bb JASPAR2022_%s.bed", build, build)
        log.error("  bgzip JASPAR2022_%s.bed && tabix -p bed JASPAR2022_%s.bed.gz", build, build)
        log.error("  # then rerun with:  --jaspar-bed /path/to/JASPAR2022_%s.bed.gz", build)
    log.error("Or pass --skip-motif to skip JASPAR analysis entirely.")
    sys.exit(1)


def validate_jaspar_bed(bed_path, sample=500, raise_on_error=False):
    bed_path = Path(bed_path)
    size = bed_path.stat().st_size if bed_path.exists() else 0
    if not bed_path.exists() or size < 64:
        msg = f"file missing or too small ({size} bytes)"
        log.warning("Cannot validate BED %s: %s", bed_path, msg)
        if raise_on_error:
            raise ValueError(msg)
        return False
    with open(bed_path, "rb") as raw:
        head = raw.read(16)
    if head.startswith(b"<!") or head.lower().startswith(b"<html"):
        msg = "file is HTML, not BED"
        log.error("BED validation failed for %s: %s", bed_path, msg)
        if raise_on_error:
            raise ValueError(msg)
        return False
    if str(bed_path).endswith(".gz") and not head.startswith(b"\x1f\x8b"):
        msg = "not a gzipped file"
        log.error("BED validation failed for %s: %s (magic=%s)",
                  bed_path, msg, head[:4].hex())
        if raise_on_error:
            raise ValueError(msg)
        return False
    def _is_gz(p):
        try:
            with open(p, "rb") as _f: return _f.read(2) == b"\x1f\x8b"
        except Exception: return False
    opener = gzip.open if _is_gz(bed_path) else open
    col_counts = []
    try:
        with opener(str(bed_path), "rt") as fh:
            for i, line in enumerate(fh):
                if i >= sample:
                    break
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("track"):
                    continue
                col_counts.append(len(line.split("\t")))
    except Exception as exc:
        log.error("Cannot validate BED %s: %s", bed_path, exc)
        log.debug(traceback.format_exc())
        if raise_on_error:
            raise
        return False
    if col_counts:
        modal = Counter(col_counts).most_common(1)[0][0]
        if modal < 4:
            log.warning("BED has too few columns: modal_cols=%d (path=%s)", modal, bed_path)
            return False
        log.info("JASPAR BED validated: modal_cols=%d, sampled %d lines, size=%.1f MB",
                 modal, len(col_counts), size / 1e6)
        return True
    log.warning("BED validation found no data rows in %s", bed_path)
    return False


def _normalise_bed(src, dst, n_cols=6):
    opener_r = gzip.open if str(src).endswith(".gz") else open
    opener_w = gzip.open if str(dst).endswith(".gz") else open
    n = 0
    log.info("Normalising BED to %d columns: %s → %s", n_cols, src, dst)
    try:
        with opener_r(str(src), "rt") as fin, opener_w(str(dst), "wt") as fout:
            for line in fin:
                s = line.rstrip("\n")
                if not s or s.startswith("#") or s.startswith("track"):
                    fout.write(s + "\n"); continue
                parts = s.split("\t")
                if len(parts) >= n_cols:
                    fout.write("\t".join(parts[:n_cols]) + "\n")
                    n += 1
        log.info("Normalised BED → %s (%d lines)", dst, n)
    except Exception as exc:
        log.error("_normalise_bed failed: %s", exc)
        log.debug(traceback.format_exc())
        raise


def ensure_tabix_index(bed_gz, scratch=None):
    """Guarantee a usable tabix index for a bgzipped, coordinate-sorted BED.

    Returns the path to an indexed ``.bed.gz`` (the original when the index can
    be built in place, or a scratch symlink/re-bgzipped copy when the source
    dir is read-only or the source is plain gzip), or ``None`` if indexing is
    impossible (no tabix/bgzip on PATH).

    The index is built once and cached, so subsequent runs reuse it. This is
    what turns the genome-wide JASPAR intersect from a full multi-GB streaming
    scan into a few-MB ``tabix -R`` seek.
    """
    import subprocess, shutil
    bed_gz = Path(bed_gz)
    if shutil.which("tabix") is None or shutil.which("bgzip") is None:
        log.info("tabix/bgzip not on PATH — cannot index %s (will full-scan)", bed_gz)
        return None

    def _has_index(p):
        return Path(f"{p}.tbi").exists() or Path(f"{p}.csi").exists()

    def _is_bgzip(p):
        try:
            with open(p, "rb") as f:
                magic = f.read(4)
            return len(magic) == 4 and magic[0] == 0x1f and magic[1] == 0x8b \
                   and magic[2] == 0x08 and magic[3] == 0x04
        except Exception:
            return False

    def _build(p):
        """Index p in place; try .tbi then fall back to .csi (large coords)."""
        for extra in ([], ["-C"]):
            r = subprocess.run(["tabix", "-f", "-p", "bed", *extra, str(p)],
                               capture_output=True, text=True)
            if r.returncode == 0 and _has_index(p):
                return True
            log.debug("tabix %s failed for %s: %s",
                      extra or ["tbi"], p, (r.stderr or "")[:300])
        return False

    # 1. Already indexed.
    if _has_index(bed_gz):
        return str(bed_gz)

    nthreads = str(os.cpu_count() or 4)
    tmp_dir = Path(scratch or os.environ.get("TMPDIR") or bed_gz.parent)

    # 2. True BGZF + writable dir → index in place (common, cheap).
    if _is_bgzip(bed_gz) and os.access(bed_gz.parent, os.W_OK):
        if _build(bed_gz):
            log.info("Built tabix index for %s", bed_gz)
            return str(bed_gz)

    # 3. True BGZF but read-only dir → symlink into scratch and index the link
    #    (avoids copying a tens-of-GB file just to drop a tiny index).
    if _is_bgzip(bed_gz):
        tmp_dir.mkdir(parents=True, exist_ok=True)
        link = tmp_dir / bed_gz.name
        if _has_index(link):
            return str(link)
        try:
            if link.exists() or link.is_symlink():
                link.unlink()
            os.symlink(bed_gz.resolve(), link)
        except OSError:
            shutil.copy2(bed_gz, link)  # symlinks unsupported on this FS
        if _build(link):
            log.info("Built tabix index via scratch symlink → %s.tbi", link)
            return str(link)
        return None

    # 4. Plain gzip *or* uncompressed BED → decompress if needed, coord-sort,
    #    bgzip once, then index.
    #
    #    The decompressor must be chosen from the file's magic bytes, not its
    #    name: the bigBed path writes an *uncompressed* JASPAR2022_<build>.te_loci.bed,
    #    and running `zcat` on that emits nothing while still exiting the
    #    pipeline cleanly — producing a valid but EMPTY .bed.gz that tabix
    #    indexes happily. The downstream `tabix -R` then returns 0 rows and the
    #    motif/enrichment/GO stages all come back empty. Hence both the
    #    magic-byte dispatch and the row-count guard below.
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cached = tmp_dir / (bed_gz.name if bed_gz.name.endswith(".bed.gz")
                        else bed_gz.stem + ".sorted.bed.gz")
    if _has_index(cached):
        return str(cached)

    def _is_gzip(p):
        try:
            with open(p, "rb") as f:
                return f.read(2) == b"\x1f\x8b"
        except Exception:
            return False

    n_in = None
    if _is_gzip(bed_gz):
        # `gzip -dc`, not `zcat`: macOS/BSD zcat only accepts .Z and fails on
        # .gz, which used to yield the same silent empty-output failure.
        decomp = ["gzip", "-dc", str(bed_gz)]
        log.info("Re-bgzipping plain-gzip %s (one-time) ...", bed_gz)
    else:
        decomp = ["cat", str(bed_gz)]
        log.info("bgzipping uncompressed BED %s (one-time) ...", bed_gz)
        try:
            with open(bed_gz, "rb") as _f:
                n_in = sum(1 for _ in _f)
            log.info("  source has %d lines", n_in)
        except Exception:
            pass
    if n_in == 0:
        log.error("Cannot index %s — source file is empty (0 lines). "
                  "The JASPAR fetch upstream produced no records.", bed_gz)
        return None

    t0 = time.time()
    try:
        with open(cached, "wb") as fout:
            p1 = subprocess.Popen(decomp, stdout=subprocess.PIPE)
            p2 = subprocess.Popen(
                ["sort", "-k1,1", "-k2,2n", "-S", "1G", "-T", str(tmp_dir)],
                stdin=p1.stdout, stdout=subprocess.PIPE)
            p1.stdout.close()
            p3 = subprocess.Popen(["bgzip", "-@", nthreads, "-c"],
                                  stdin=p2.stdout, stdout=fout)
            p2.stdout.close()
            p3.communicate(); p2.wait(); p1.wait()
            if p1.returncode != 0:
                raise RuntimeError(
                    f"{decomp[0]} exit {p1.returncode} on {bed_gz}")
            if p2.returncode != 0:
                raise RuntimeError(f"sort exit {p2.returncode}")
            if p3.returncode != 0:
                raise RuntimeError(f"bgzip pipeline exit {p3.returncode}")
    except Exception as exc:
        log.warning("re-bgzip pipeline failed for %s: %s", bed_gz, exc)
        cached.unlink(missing_ok=True)
        return None

    # Guard: never hand back an indexed-but-empty BED. A silent empty cache is
    # indistinguishable from "no motifs overlap our loci" much further
    # downstream, where it surfaces as a bare "tabix returned 0 rows".
    try:
        n_out = 0
        with gzip.open(cached, "rt") as _f:
            for _ in _f:
                n_out += 1
        log.info("  bgzipped %d lines → %s", n_out, cached)
        if n_out == 0:
            log.error("bgzip produced an EMPTY %s from %s — refusing to index it. "
                      "Check that the JASPAR source file actually contains records.",
                      cached, bed_gz)
            cached.unlink(missing_ok=True)
            return None
    except Exception as exc:
        log.warning("Could not verify line count of %s: %s", cached, exc)

    if _build(cached):
        log.info("Built tabix index (re-bgzipped copy) in %.1fs → %s",
                 time.time() - t0, cached)
        return str(cached)
    cached.unlink(missing_ok=True)
    return None


def bedtools_intersect_safe(v_bed, jaspar_bed, scratch=None):
    """Run bedtools intersect, writing output to a temp file to avoid OOM/SIGBUS.

    Uses the bedtools CLI directly. This avoids pybedtools' Python packaging
    overhead and its temp-file/mmap failure modes on HPC scratch filesystems.
    """
    import subprocess, shutil as _shutil

    class _OverlapResult(list):
        def to_dataframe(self, names=None, header=None):
            rows = [str(x).rstrip("\n").split("\t") for x in self]
            return pd.DataFrame(rows, columns=names)

    def _merge_loci(a, tmp_dir):
        """Coord-sort + merge the -a loci into a compact, non-overlapping region
        file used ONLY for the tabix region query (fewer redundant bgzip block
        seeks). The original -a is still used for the final bedtools intersect,
        so per-locus output rows are preserved. Falls back to raw -a if bedtools
        merge is unavailable."""
        if _shutil.which("bedtools") is None:
            return str(a)
        sorted_tmp = Path(tmp_dir) / f"loci_sorted_{os.getpid()}.bed"
        merged = Path(tmp_dir) / f"loci_merged_{os.getpid()}.bed"
        try:
            with open(sorted_tmp, "w") as fh:
                subprocess.run(
                    ["sort", "-k1,1", "-k2,2n", "-T", str(tmp_dir), str(a)],
                    stdout=fh, env=dict(os.environ, LC_ALL="C"), check=True)
            with open(merged, "w") as fh:
                subprocess.run(["bedtools", "merge", "-i", str(sorted_tmp)],
                               stdout=fh, check=True)
            sorted_tmp.unlink(missing_ok=True)
            if merged.stat().st_size > 0:
                return str(merged)
        except Exception as exc:
            log.debug("loci merge failed (%s); using raw -a for tabix", exc)
        sorted_tmp.unlink(missing_ok=True)
        merged.unlink(missing_ok=True)
        return str(a)

    def _tabix_subset(a, b):
        if not str(b).endswith(".gz"):
            return b
        if _shutil.which("tabix") is None:
            log.info("tabix not found; bedtools will scan full JASPAR BED")
            return b
        # Build the index on demand (cached) so we always seek, never stream the
        # whole multi-GB file. ensure_tabix_index may relocate to scratch.
        indexed = ensure_tabix_index(b, scratch=scratch)
        if indexed is None:
            log.warning("Could not index %s; bedtools will scan the full BED "
                        "(slow). Install tabix/bgzip to enable fast seeks.", b)
            return b
        b = indexed

        tmp_dir = scratch or os.environ.get("TMPDIR") or str(Path(a).parent)
        region_bed = _merge_loci(a, tmp_dir)
        subset = Path(tmp_dir) / f"jaspar_subset_{os.getpid()}.bed"
        cmd = ["tabix", "-R", str(region_bed), str(b)]
        log.info("Subsetting indexed JASPAR BED with tabix before bedtools ...")
        log.debug("tabix command: %s", " ".join(cmd))
        t0 = time.time()
        with open(subset, "w") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.PIPE, text=True)
        if region_bed != str(a):
            Path(region_bed).unlink(missing_ok=True)
        if proc.returncode != 0:
            subset.unlink(missing_ok=True)
            log.warning("tabix subset failed; falling back to full JASPAR BED: %s",
                        proc.stderr[:1000])
            return b
        if subset.stat().st_size == 0:
            subset.unlink(missing_ok=True)
            raise RuntimeError(
                "tabix returned zero JASPAR rows for the TE loci. This usually "
                "means chromosome names or genome build do not match."
            )
        log.info("tabix subset done in %.1fs: %.1f MB at %s",
                 time.time() - t0, subset.stat().st_size / 1e6, subset)
        return str(subset)

    def _sort_a(a):
        """Sort the -a (TE loci) BED with the SAME ordering as the cached JASPAR
        BED (`LC_ALL=C sort -k1,1 -k2,2n`, see bb_to_sorted_bed.py) so bedtools
        can use the low-memory `-sorted` sweep-line algorithm."""
        tmp_dir = scratch or os.environ.get("TMPDIR") or str(Path(a).parent)
        a_sorted = Path(tmp_dir) / f"{Path(a).stem}.csorted_{os.getpid()}.bed"
        env = dict(os.environ, LC_ALL="C")
        with open(a_sorted, "w") as fh:
            proc = subprocess.run(
                ["sort", "-k1,1", "-k2,2n", "-T", str(tmp_dir), str(a)],
                stdout=fh, stderr=subprocess.PIPE, env=env)
        if proc.returncode != 0:
            a_sorted.unlink(missing_ok=True)
            log.warning("sort of -a BED failed (%s); using unsorted -a",
                        proc.stderr.decode(errors="replace")[:500])
            return str(a)
        return str(a_sorted)

    def _run_cli(a, b, sorted_stream=True):
        """Stream bedtools output to a temp file; return (_OverlapResult, stderr).

        With sorted_stream=True both inputs are assumed `LC_ALL=C` chrom/start
        sorted and `-sorted` is passed so bedtools sweeps the (multi-GB) JASPAR
        BED instead of loading it fully into RAM (the historical SIGKILL/-9 OOM).
        """
        tmp_dir = scratch or os.environ.get("TMPDIR") or str(Path(a).parent)
        tmp_out = Path(tmp_dir) / f"bedtools_overlaps_{os.getpid()}.tsv"
        a_use = _sort_a(a) if sorted_stream else str(a)

        # Full-scan fallback (no usable index): -b is still a multi-GB .gz.
        # Pipe through multi-threaded bgzip -d so decompression — the real
        # bottleneck of the streaming sweep — isn't single-threaded.
        gz_fallback = (str(b).endswith(".gz")
                       and _shutil.which("bgzip") is not None)
        cmd = ["bedtools", "intersect", "-a", a_use,
               "-b", ("-" if gz_fallback else str(b)), "-wa", "-wb"]
        if sorted_stream:
            cmd.append("-sorted")
        log.debug("bedtools CLI command: %s%s", " ".join(cmd),
                  f"  (b piped via bgzip -d from {b})" if gz_fallback else "")
        log.debug("bedtools output temp file: %s", tmp_out)
        t0 = time.time()
        if gz_fallback:
            nthreads = str(os.cpu_count() or 4)
            with open(tmp_out, "w") as fh:
                dec = subprocess.Popen(["bgzip", "-d", "-@", nthreads, "-c", str(b)],
                                       stdout=subprocess.PIPE)
                proc = subprocess.Popen(cmd, stdin=dec.stdout, stdout=fh,
                                        stderr=subprocess.PIPE)
                dec.stdout.close()
                _, _stderr_b = proc.communicate()
                dec.wait()
                proc.stderr = _stderr_b  # normalise for the shared code below
        else:
            with open(tmp_out, "w") as fh:
                proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.PIPE)
        elapsed = time.time() - t0
        stderr_text = proc.stderr.decode(errors="replace")
        if a_use != str(a):
            Path(a_use).unlink(missing_ok=True)
        if stderr_text.strip():
            log.debug("bedtools stderr: %s", stderr_text[:2000])
        if proc.returncode != 0:
            log.error("bedtools intersect exited %d after %.1fs. stderr: %s",
                      proc.returncode, elapsed, stderr_text[:1000])
            tmp_out.unlink(missing_ok=True)
            return None, stderr_text
        out_size = tmp_out.stat().st_size
        log.info("bedtools intersect done in %.1fs — output %.1f MB at %s",
                 elapsed, out_size / 1e6, tmp_out)
        lines = tmp_out.read_text(errors="replace").splitlines(keepends=True)
        tmp_out.unlink(missing_ok=True)
        result = _OverlapResult([ln for ln in lines if ln.strip()])
        log.info("Parsed %d overlap lines from bedtools output", len(result))
        return result, ""

    if _shutil.which("bedtools") is None:
        raise EnvironmentError(
            "bedtools not found on PATH. Load it on the cluster with "
            "`module load bedtools`, install it via conda/bioconda, or use a "
            "container that includes bedtools."
        )

    log.info("Running bedtools intersect (CLI, streaming to disk) ...")
    intersect_b = _tabix_subset(v_bed, jaspar_bed)
    log.debug("  -a: %s  -b: %s", v_bed, intersect_b)
    overlaps, stderr_text = _run_cli(v_bed, intersect_b, sorted_stream=True)
    if overlaps is None and any(k in stderr_text.lower() for k in
                                ("sorted", "chromosomes", "chrom order", "order")):
        log.warning("`-sorted` sweep failed on chrom ordering; retrying without "
                    "-sorted (higher memory). stderr head: %s", stderr_text[:300])
        overlaps, stderr_text = _run_cli(v_bed, intersect_b, sorted_stream=False)
    if overlaps is None and "fields" in stderr_text.lower():
        log.warning("Column mismatch detected — normalising JASPAR BED to 6 cols ...")
        norm = str(intersect_b).replace(".bed.gz", ".norm6.bed").replace(".bed", ".norm6.bed")
        _normalise_bed(str(intersect_b), norm, 6)
        overlaps, stderr_text = _run_cli(v_bed, norm)
    if overlaps is not None:
        log.info("bedtools intersect: %d overlaps", len(overlaps))
        return overlaps

    raise RuntimeError(f"bedtools intersect failed:\n{stderr_text[:2000]}")


# ── coordinate column detection ───────────────────────────────────────────────

def _detect_coords(df):
    def _fc(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None
    return (
        _fc(["chr", "Chromosome", "chrom", "Chr", "#chrom"]),
        _fc(["start", "Start", "chromStart"]),
        _fc(["stop", "Stop", "End", "end", "chromEnd"]),
    )


def _cluster_col(df):
    for c in ("Cluster", "cluster"):
        if c in df.columns:
            return c
    return None


# ── main ──────────────────────────────────────────────────────────────────────

def run_motif_analysis(input_csv, build, out_dir, jaspar_bed_arg,
                       jaspar_dir, p_threshold=0.05, force=False,
                       bed_input=None):
    """
    Run JASPAR motif overlap and Fisher enrichment.

    Parameters
    ----------
    input_csv    : path to clustered CSV (output of te_clustering.py), or None
    build        : genome build string, e.g. "hg38"
    out_dir      : root output directory
    jaspar_bed_arg : explicit JASPAR BED path or None
    jaspar_dir   : directory for caching downloaded JASPAR files
    p_threshold  : Fisher p-value cutoff for reporting significance
    force        : re-run even if outputs already exist
    bed_input    : path to an existing TE loci BED file; when provided,
                   skips CSV loading and assigns all loci to cluster 0.
                   Mutually exclusive with input_csv.

    Returns
    -------
    dict with keys:
        overlaps_path      – path to all_overlaps.tsv
        enrichment_dir     – path to directory with per-cluster CSVs
        significant_tfs    – {cluster_id: DataFrame of significant TFs}
    """
    out_dir = Path(out_dir)
    motif_dir  = out_dir / "motif_analysis"
    enrich_dir = out_dir / "enrichment_results"
    motif_dir.mkdir(parents=True, exist_ok=True)
    enrich_dir.mkdir(parents=True, exist_ok=True)

    log_path = _setup_logger(out_dir)

    log.info("=" * 60)
    log.info("GAMECA — Motif Analysis")
    log.info("  input_csv   : %s", input_csv or "(not used)")
    log.info("  bed_input   : %s", bed_input  or "(not used)")
    log.info("  build       : %s", build)
    log.info("  out_dir     : %s", out_dir)
    log.info("  jaspar_bed  : %s", jaspar_bed_arg or "(auto-resolve)")
    log.info("  jaspar_dir  : %s", jaspar_dir)
    log.info("  p_threshold : %s", p_threshold)
    log.info("  force       : %s", force)
    log.info("  log file    : %s", log_path)
    log.info("=" * 60)

    pipeline_t0 = time.time()
    overlaps_path = motif_dir / "all_overlaps.tsv"

    # ── Load data ─────────────────────────────────────────────────────────────
    chr_col   = "chr"
    start_col = "start"
    stop_col  = "stop"
    cl_col    = "Cluster"

    if bed_input:
        # ── BED-input mode: skip CSV, treat all loci as cluster 0 ─────────────
        log.info("--- STAGE: Load BED ---")
        t0 = time.time()
        bed_input = Path(bed_input)
        if not bed_input.exists():
            log.error("BED input file not found: %s", bed_input)
            sys.exit(1)

        bed_size = bed_input.stat().st_size
        log.info("BED file: %s  (%.2f MB)", bed_input, bed_size / 1e6)

        # Count lines without loading everything
        try:
            import subprocess as _sp
            wc = _sp.run(["wc", "-l", str(bed_input)],
                         capture_output=True, text=True)
            if wc.returncode == 0:
                log.info("  Line count (wc -l): %s", wc.stdout.strip().split()[0])
        except Exception:
            pass

        try:
            # Read with up to 10 columns; ignore extras; no header
            df_raw = pd.read_csv(bed_input, sep="\t", header=None,
                                 usecols=range(min(3, 3)), engine="python",
                                 on_bad_lines="skip")
            # Ensure at least 3 columns
            if df_raw.shape[1] < 3:
                log.error("BED file has fewer than 3 columns (%d); "
                          "expected chr/start/stop", df_raw.shape[1])
                sys.exit(1)
            df_raw.columns = list(df_raw.columns)
        except Exception as exc:
            # pandas may reject usecols=range(3) if file has exactly 3 cols; retry
            try:
                df_raw = pd.read_csv(bed_input, sep="\t", header=None, engine="python",
                                     on_bad_lines="skip")
            except Exception as exc2:
                log.error("Failed to read BED file %s: %s", bed_input, exc2)
                log.debug(traceback.format_exc())
                sys.exit(1)

        if df_raw.shape[1] < 3:
            log.error("BED file has fewer than 3 columns (%d); "
                      "expected chr/start/stop", df_raw.shape[1])
            sys.exit(1)

        df = pd.DataFrame({
            "chr":   df_raw.iloc[:, 0].astype(str),
            "start": pd.to_numeric(df_raw.iloc[:, 1], errors="coerce"),
            "stop":  pd.to_numeric(df_raw.iloc[:, 2], errors="coerce"),
        })
        # Drop rows with non-numeric coordinates or header-like rows
        n_before = len(df)
        df = df.dropna(subset=["start", "stop"])
        df = df[~df["chr"].str.startswith(("chrom", "Chrom", "#"))]
        n_dropped = n_before - len(df)
        if n_dropped:
            log.warning("Dropped %d non-numeric / header rows from BED", n_dropped)

        df["start"] = df["start"].astype(int)
        df["stop"]  = df["stop"].astype(int)
        df[cl_col]  = 0

        log.info("Loaded %d loci from BED in %.1fs (all assigned to cluster 0)",
                 len(df), time.time() - t0)
        log.debug("Sample (first 5 rows):\n%s", df.head().to_string(index=False))
        log.debug("Coordinate range summary: "
                  "chr unique=%d, start min=%d max=%d, stop min=%d max=%d",
                  df["chr"].nunique(),
                  df["start"].min(), df["start"].max(),
                  df["stop"].min(),  df["stop"].max())

        cluster_ids = [0]
        has_strand  = False
        noise_n     = 0
        log.info("Cluster layout: 1 cluster (cluster 0)  |  noise (cluster -1): 0 rows")

        # Use the supplied BED directly as te_loci.bed (copy if different path)
        v_bed = motif_dir / "te_loci.bed"
        if Path(bed_input).resolve() != v_bed.resolve():
            log.info("Copying supplied BED → %s", v_bed)
            try:
                import shutil as _shutil
                _shutil.copy2(str(bed_input), str(v_bed))
            except Exception as exc:
                log.warning("Could not copy BED: %s — writing fresh from dataframe", exc)
                df[["chr", "start", "stop"]].to_csv(
                    v_bed, sep="\t", header=False, index=False)
        else:
            log.debug("BED input is already at canonical path: %s", v_bed)

    else:
        # ── CSV mode ──────────────────────────────────────────────────────────
        log.info("--- STAGE: Load CSV ---")
        t0 = time.time()
        if not Path(input_csv).exists():
            log.error("Input CSV not found: %s", input_csv)
            sys.exit(1)

        csv_size = Path(input_csv).stat().st_size
        log.info("CSV file: %s  (%.2f MB)", input_csv, csv_size / 1e6)

        try:
            df = pd.read_csv(input_csv)
        except Exception as exc:
            log.error("Failed to read input CSV %s: %s", input_csv, exc)
            log.debug(traceback.format_exc())
            sys.exit(1)

        log.info("Loaded %d rows, %d columns from %s (%.1fs)",
                 len(df), len(df.columns), input_csv, time.time() - t0)
        log.debug("Columns: %s", list(df.columns))

        chr_col, start_col, stop_col = _detect_coords(df)
        log.debug("Detected coordinate columns: chr=%s start=%s stop=%s",
                  chr_col, start_col, stop_col)
        if not all([chr_col, start_col, stop_col]):
            log.error("Cannot find coordinate columns (chr/start/stop). "
                      "Available columns: %s", list(df.columns))
            sys.exit(1)

        cl_col = _cluster_col(df)
        if cl_col is None:
            log.error("No Cluster column found. Available columns: %s", list(df.columns))
            log.error("Run te_clustering.py first.")
            sys.exit(1)

        df[cl_col] = df[cl_col].astype(int)
        cluster_ids = sorted([c for c in df[cl_col].unique() if c >= 0])
        has_strand  = "strand" in df.columns
        noise_n     = int((df[cl_col] == -1).sum())

        log.info("%d clusters: %s  |  noise (cluster -1): %d rows",
                 len(cluster_ids), cluster_ids, noise_n)
        for cid in cluster_ids:
            log.debug("  Cluster %d: %d sequences", cid, int((df[cl_col] == cid).sum()))

        # ── Write TE loci BED ──────────────────────────────────────────────────
        v_bed = motif_dir / "te_loci.bed"
        log.info("Writing TE loci BED → %s (%d loci)", v_bed, len(df))
        try:
            df[[chr_col, start_col, stop_col]].to_csv(
                v_bed, sep="\t", header=False, index=False)
        except Exception as exc:
            log.error("Failed to write te_loci.bed: %s", exc)
            log.debug(traceback.format_exc())
            sys.exit(1)

    # ── Resolve JASPAR BED ────────────────────────────────────────────────────
    log.info("--- STAGE: Resolve JASPAR BED ---")
    t0 = time.time()
    try:
        jaspar_bed = resolve_jaspar_bed(build, jaspar_bed_arg, jaspar_dir, loci_bed=v_bed)
    except SystemExit:
        raise
    except Exception as exc:
        log.error("resolve_jaspar_bed raised unexpectedly: %s", exc)
        log.debug(traceback.format_exc())
        sys.exit(1)

    log.info("JASPAR BED resolved in %.1fs: %s", time.time() - t0, jaspar_bed)
    validate_jaspar_bed(jaspar_bed)

    # ── Bedtools intersect ────────────────────────────────────────────────────
    log.info("--- STAGE: Bedtools Intersect ---")
    if overlaps_path.exists() and not force:
        log.info("[SKIP] Overlaps file exists: %s (use --force to rerun)", overlaps_path)
        try:
            df_ov = pd.read_csv(overlaps_path, sep="\t")
            log.info("Loaded existing overlaps: %d rows", len(df_ov))
        except Exception as exc:
            log.error("Failed to read existing overlaps %s: %s — rerunning intersect",
                      overlaps_path, exc)
            log.debug(traceback.format_exc())
            overlaps_path.unlink(missing_ok=True)
            df_ov = None
    else:
        df_ov = None

    if df_ov is None:
        t0 = time.time()
        scratch = os.environ.get("TMPDIR", str(out_dir / "tmp"))
        Path(scratch).mkdir(parents=True, exist_ok=True)
        tempfile.tempdir = scratch
        log.debug("Scratch dir for bedtools temp files: %s", scratch)
        _log_system_info(scratch)

        try:
            overlaps = bedtools_intersect_safe(str(v_bed), jaspar_bed, scratch=scratch)
        except SystemExit:
            raise
        except Exception as exc:
            log.error("bedtools_intersect_safe raised: %s", exc)
            log.debug(traceback.format_exc())
            sys.exit(1)

        if len(overlaps) == 0:
            log.error("FATAL: Zero overlaps returned. "
                      "Check genome build (%s) and JASPAR BED path (%s).", build, jaspar_bed)
            sys.exit(1)

        log.info("Parsing overlap column layout from first line ...")
        try:
            first = str(overlaps[0]).strip().split("\t")
        except Exception as exc:
            log.error("Cannot read first overlap line: %s", exc)
            log.debug(traceback.format_exc())
            sys.exit(1)

        n_mc   = len(first) - 3
        cols_v = [chr_col, start_col, stop_col]
        log.debug("First overlap line has %d columns (%d motif-side)", len(first), n_mc)
        log.debug("First line preview: %s", "\t".join(first[:10]))

        if   n_mc == 4: cols_m = ["Motif_chr","Motif_start","Motif_end","Motif_name"]
        elif n_mc == 5: cols_m = ["Motif_chr","Motif_start","Motif_end","Motif_name","Motif_score"]
        elif n_mc == 6: cols_m = ["Motif_chr","Motif_start","Motif_end","Motif_name","Motif_score","Motif_strand"]
        elif n_mc == 7: cols_m = ["Motif_chr","Motif_start","Motif_end","Motif_ID","Motif_score","Motif_strand","Motif_name"]
        else:           cols_m = [f"motif_col_{i}" for i in range(n_mc)]
        log.debug("Motif columns assigned: %s", cols_m)

        log.info("Converting overlaps to DataFrame ...")
        try:
            df_ov = overlaps.to_dataframe(names=cols_v + cols_m, header=None)
        except Exception as exc:
            log.error("Failed to convert overlaps to DataFrame: %s", exc)
            log.debug(traceback.format_exc())
            sys.exit(1)

        if "Motif_name" not in df_ov.columns:
            nc = [c for c in df_ov.columns if "name" in c.lower() or "id" in c.lower()]
            if nc:
                df_ov["Motif_name"] = df_ov[nc[0]]
                log.info("Derived Motif_name from column: %s", nc[0])
            else:
                df_ov["Motif_name"] = df_ov.iloc[:, 3]
                log.warning("Motif_name not found; using column index 3 as fallback")

        # Merge cluster info
        log.info("Merging cluster assignments into overlap table ...")
        mcols = [chr_col, start_col, stop_col, cl_col] + (["strand"] if has_strand else [])
        w = df[mcols].copy()
        try:
            for c in [start_col, stop_col]:
                df_ov[c] = pd.to_numeric(df_ov[c], errors="coerce")
                w[c]     = pd.to_numeric(w[c],     errors="coerce")
            df_ov[chr_col] = df_ov[chr_col].astype(str)
            w[chr_col]     = w[chr_col].astype(str)
            pre_merge = len(df_ov)
            df_ov = df_ov.merge(w, on=[chr_col, start_col, stop_col], how="left",
                                suffixes=("", "_new"))
            log.debug("Merge: %d → %d rows (left join on coords)", pre_merge, len(df_ov))
            for col in [cl_col, "strand"]:
                if f"{col}_new" in df_ov.columns:
                    df_ov[col] = df_ov[f"{col}_new"].combine_first(
                        df_ov.get(col, pd.Series(dtype=object)))
            df_ov.drop(columns=[c for c in df_ov.columns if c.endswith("_new")], inplace=True)

            null_cluster = df_ov[cl_col].isna().sum() if cl_col in df_ov.columns else "N/A"
            if null_cluster:
                log.warning("%s rows in overlaps have no cluster assignment after merge",
                            null_cluster)
        except Exception as exc:
            log.error("Cluster merge failed: %s", exc)
            log.debug(traceback.format_exc())
            sys.exit(1)

        try:
            df_ov.to_csv(overlaps_path, sep="\t", index=False)
            log.info("Saved %s (%d rows, %.1f MB) in %.1fs",
                     overlaps_path.name, len(df_ov),
                     overlaps_path.stat().st_size / 1e6, time.time() - t0)
        except Exception as exc:
            log.error("Failed to save overlaps TSV: %s", exc)
            log.debug(traceback.format_exc())

        # Overall motif counts
        try:
            overall = df_ov["Motif_name"].value_counts()
            overall.to_csv(motif_dir / "overall_motif_counts.csv")
            log.info("%d unique motifs. Top 5: %s",
                     len(overall),
                     ", ".join(f"{k}={v}" for k, v in overall.head(5).items()))
        except Exception as exc:
            log.warning("overall_motif_counts.csv failed: %s", exc)
            log.debug(traceback.format_exc())

        # Top-20 bar chart
        try:
            tc = overall.head(20).reset_index()
            tc.columns = ["Motif", "Count"]
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(range(len(tc)), tc["Count"], color="#3498DB", alpha=0.85)
            ax.set_xticks(range(len(tc)))
            ax.set_xticklabels(tc["Motif"], rotation=45, ha="right", fontsize=9)
            ax.set_ylabel("Count")
            ax.set_title(f"Top 20 Motifs ({build})", fontweight="bold")
            ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            out_png = motif_dir / "overall_top_motifs.png"
            plt.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close()
            log.info("Saved %s", out_png)
        except Exception as exc:
            log.warning("Top-20 bar chart failed: %s", exc)
            log.debug(traceback.format_exc())

        # Per-cluster motif counts + charts. all_overlaps.tsv already carries a
        # Cluster column, so the family-level rollup above was throwing away a
        # breakdown that was sitting right there. The Fisher stage below reports
        # ENRICHMENT (cluster vs rest); these are the raw per-cluster counts that
        # enrichment is computed from, which is what you want when comparing
        # absolute motif load between subfamilies.
        loci_per_cluster = {}
        try:
            per_cluster = (df_ov.groupby([cl_col, "Motif_name"])
                                .size().rename("count").reset_index()
                                .sort_values([cl_col, "count"],
                                             ascending=[True, False]))

            # Wide motif x cluster matrix — the convenient shape for heatmaps
            # and for diffing subfamilies side by side.
            matrix = (per_cluster.pivot(index="Motif_name", columns=cl_col,
                                        values="count")
                                 .fillna(0).astype(int))
            matrix.columns = [f"cluster_{c}" for c in matrix.columns]
            matrix["total"] = matrix.sum(axis=1)
            matrix.sort_values("total", ascending=False, inplace=True)
            matrix.to_csv(motif_dir / "cluster_motif_counts_matrix.csv")
            log.info("Saved cluster_motif_counts.csv + matrix (%d motifs x %d clusters)",
                     len(matrix), max(len(matrix.columns) - 1, 0))

            # Normalised per-cluster share: cluster sizes differ, so raw counts
            # alone make the biggest cluster look motif-rich by construction.
            loci_per_cluster = df.groupby(cl_col).size().to_dict()
            per_cluster["sites_per_locus"] = [
                (row["count"] / loci_per_cluster[row[cl_col]])
                if loci_per_cluster.get(row[cl_col]) else float("nan")
                for _, row in per_cluster.iterrows()
            ]
            per_cluster.to_csv(motif_dir / "cluster_motif_counts.csv", index=False)
        except Exception as exc:
            log.warning("cluster_motif_counts.csv failed: %s", exc)
            log.debug(traceback.format_exc())

        try:
            _cids = sorted(df_ov[cl_col].dropna().unique())
            for cid in _cids:
                sub = (df_ov[df_ov[cl_col] == cid]["Motif_name"]
                       .value_counts().head(20).reset_index())
                if sub.empty:
                    continue
                sub.columns = ["Motif", "Count"]
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.bar(range(len(sub)), sub["Count"], color="#E67E22", alpha=0.85)
                ax.set_xticks(range(len(sub)))
                ax.set_xticklabels(sub["Motif"], rotation=45, ha="right", fontsize=9)
                ax.set_ylabel("Count")
                _n = loci_per_cluster.get(cid, "?")
                ax.set_title(f"Top 20 Motifs — cluster {cid} (n={_n} loci, {build})",
                             fontweight="bold")
                ax.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                plt.savefig(motif_dir / f"cluster_{cid}_top_motifs.png",
                            dpi=150, bbox_inches="tight")
                plt.close()
            log.info("Saved per-cluster top-motif charts for %d cluster(s)", len(_cids))
        except Exception as exc:
            log.warning("Per-cluster top-motif charts failed: %s", exc)
            log.debug(traceback.format_exc())

    # ── Fisher's exact test ───────────────────────────────────────────────────
    log.info("--- STAGE: Fisher's Exact Test (%d clusters) ---", len(cluster_ids))
    t0 = time.time()
    all_motifs = df_ov["Motif_name"].unique()
    total_n    = len(df)
    significant_tfs = {}
    log.info("Total motifs to test: %d  |  total sequences: %d", len(all_motifs), total_n)

    for cid in cluster_ids:
        cn = int((df[cl_col] == cid).sum())
        log.info("  Cluster %d (n=%d): running Fisher test on %d motifs ...",
                 cid, cn, len(all_motifs))
        t_cluster = time.time()
        results = []
        for m_i, motif in enumerate(all_motifs):
            mi   = len(df_ov[(df_ov[cl_col]==cid) & (df_ov["Motif_name"]==motif)]
                        [[chr_col, start_col, stop_col]].drop_duplicates())
            mt   = len(df_ov[df_ov["Motif_name"]==motif]
                        [[chr_col, start_col, stop_col]].drop_duplicates())
            nmc  = cn - mi
            mnc  = mt - mi
            nmnc = max((total_n - cn) - mnc, 0)
            try:
                odds, pv = stats.fisher_exact([[mi, mnc],[nmc, nmnc]], alternative="greater")
            except Exception as exc:
                log.debug("fisher_exact failed for motif %s cluster %d: %s", motif, cid, exc)
                odds, pv = 1.0, 1.0
            results.append({
                "Motif": motif, "In_Cluster": mi, "Total": mt,
                "Cluster_Size": cn, "Odds_Ratio": round(odds, 4), "P_Value": pv,
            })
            if (m_i + 1) % 500 == 0:
                log.debug("    Cluster %d: tested %d/%d motifs ...", cid, m_i + 1, len(all_motifs))

        rdf = pd.DataFrame(results).sort_values("P_Value")
        out_csv = enrich_dir / f"cluster_{cid}_enrichment.csv"
        try:
            rdf.to_csv(out_csv, index=False)
        except Exception as exc:
            log.error("Failed to save enrichment CSV for cluster %d: %s", cid, exc)
            log.debug(traceback.format_exc())

        sig = rdf[rdf["P_Value"] < p_threshold]
        significant_tfs[cid] = sig
        elapsed_c = time.time() - t_cluster
        log.info("  Cluster %d: %d significant TFs (p<%.3f) in %.1fs → %s",
                 cid, len(sig), p_threshold, elapsed_c, out_csv.name)
        if len(sig) > 0:
            log.debug("  Top 5 for cluster %d:\n%s", cid,
                      sig[["Motif","P_Value","Odds_Ratio"]].head(5).to_string(index=False))

    # Enrichment heatmap
    log.info("--- STAGE: Enrichment Heatmap ---")
    try:
        all_sig = []
        for cid, sdf in significant_tfs.items():
            if len(sdf):
                t = sdf.head(10).copy(); t["cluster"] = cid; all_sig.append(t)
        if all_sig:
            comb = pd.concat(all_sig)
            comb["nlp"] = -np.log10(comb["P_Value"].clip(lower=1e-300))
            piv = comb.pivot_table(index="Motif", columns="cluster",
                                   values="nlp", aggfunc="max").fillna(0)
            top_m = piv.max(axis=1).sort_values(ascending=False).head(30).index
            piv   = piv.loc[top_m]
            log.debug("Heatmap: %d motifs × %d clusters", len(top_m), len(piv.columns))
            fig, ax = plt.subplots(figsize=(8, max(6, len(top_m) * 0.4)))
            im = ax.imshow(piv.values, aspect="auto", cmap="Reds")
            plt.colorbar(im, ax=ax, label="-log10(p)")
            ax.set_xticks(range(len(piv.columns)))
            ax.set_xticklabels([f"Cl {c}" for c in piv.columns])
            ax.set_yticks(range(len(top_m))); ax.set_yticklabels(top_m, fontsize=8)
            ax.set_title("Motif Enrichment per Cluster", fontweight="bold")
            plt.tight_layout()
            heatmap_path = enrich_dir / "enrichment_heatmap.png"
            plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
            plt.close()
            log.info("Saved %s", heatmap_path)
        else:
            log.warning("No significant TFs in any cluster — heatmap not generated")
    except Exception as exc:
        log.warning("Enrichment heatmap failed: %s", exc)
        log.debug(traceback.format_exc())

    total_sig    = sum(len(s) for s in significant_tfs.values())
    total_elapsed = time.time() - pipeline_t0
    log.info("Fisher step: %.1fs", time.time() - t0)
    log.info("Total significant TF hits (p<%s): %d", p_threshold, total_sig)
    log.info("Motif analysis complete in %.1fs", total_elapsed)
    log.info("Log written to: %s", log_path)

    return {
        "overlaps_path":   str(overlaps_path),
        "enrichment_dir":  str(enrich_dir),
        "significant_tfs": significant_tfs,
    }


def run_homer(input_csv, build, out_dir, genome=None, size="200",
              threads=4, force=False):
    """
    Run HOMER findMotifsGenome.pl on each cluster's loci.

    Parameters
    ----------
    input_csv : path to clustered CSV (output of te_clustering.py)
    build     : genome build string used as fallback genome name
    out_dir   : root output directory; results go in <out_dir>/homer_results/
    genome    : HOMER genome name or FASTA path (defaults to build)
    size      : HOMER -size value, e.g. "200" or "given"
    threads   : HOMER -p (parallel threads per cluster run)
    force     : re-run even if per-cluster output already exists

    Returns
    -------
    dict mapping cluster_id -> path to knownResults.txt (or None if failed)
    """
    import shutil, subprocess

    if not shutil.which("findMotifsGenome.pl"):
        log.error("HOMER not found. Install via http://homer.ucsd.edu/homer/introduction/install.html")
        sys.exit(1)

    genome  = genome or build
    out_dir = Path(out_dir)
    homer_root = out_dir / "homer_results"
    homer_root.mkdir(parents=True, exist_ok=True)

    log.info("--- STAGE: HOMER ---")
    log.info("  genome=%s  size=%s  threads=%d", genome, size, threads)

    try:
        df = pd.read_csv(input_csv)
    except Exception as exc:
        log.error("Failed to read input CSV for HOMER: %s", exc)
        log.debug(traceback.format_exc())
        sys.exit(1)

    chr_col, start_col, stop_col = _detect_coords(df)
    if not all([chr_col, start_col, stop_col]):
        log.error("Cannot find coordinate columns for HOMER. Columns: %s", list(df.columns))
        sys.exit(1)
    cl_col = _cluster_col(df)
    if cl_col is None:
        log.error("No Cluster column found for HOMER. Run te_clustering.py first.")
        sys.exit(1)

    df[cl_col] = df[cl_col].astype(int)
    cluster_ids = sorted([c for c in df[cl_col].unique() if c >= 0])
    log.info("HOMER: %d clusters: %s", len(cluster_ids), cluster_ids)

    results = {}
    for cid in cluster_ids:
        cluster_dir = homer_root / f"cluster_{cid}"
        known_txt   = cluster_dir / "knownResults.txt"

        if known_txt.exists() and not force:
            log.info("[SKIP] Cluster %d HOMER output exists: %s", cid, known_txt)
            results[cid] = str(known_txt)
            continue

        cluster_dir.mkdir(parents=True, exist_ok=True)

        sub = df[df[cl_col] == cid][[chr_col, start_col, stop_col]].copy()
        sub.insert(0, "name", [f"locus_{i}" for i in range(len(sub))])
        sub["strand"] = "+"
        bed_path = cluster_dir / f"cluster_{cid}.bed"
        try:
            sub[["name", chr_col, start_col, stop_col, "strand"]].to_csv(
                bed_path, sep="\t", header=False, index=False)
        except Exception as exc:
            log.error("Failed to write HOMER BED for cluster %d: %s", cid, exc)
            log.debug(traceback.format_exc())
            results[cid] = None
            continue

        cmd = [
            "findMotifsGenome.pl",
            str(bed_path), genome, str(cluster_dir),
            "-size", str(size), "-p", str(threads), "-nomotif",
        ]
        log.info("Cluster %d (n=%d): running HOMER — %s", cid, len(sub), " ".join(cmd))
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        except Exception as exc:
            log.error("HOMER subprocess failed for cluster %d: %s", cid, exc)
            log.debug(traceback.format_exc())
            results[cid] = None
            continue
        elapsed = time.time() - t0

        if proc.returncode != 0:
            log.warning("Cluster %d HOMER failed (exit=%d, %.1fs)\nstderr:\n%s",
                        cid, proc.returncode, elapsed, proc.stderr[-1000:])
            results[cid] = None
            continue

        if not known_txt.exists():
            log.warning("Cluster %d: knownResults.txt not produced after %.1fs", cid, elapsed)
            results[cid] = None
            continue

        try:
            kr = pd.read_csv(known_txt, sep="\t")
            kr.columns = [c.strip() for c in kr.columns]
            p_col    = next((c for c in kr.columns if "p-value" in c.lower() or "pvalue" in c.lower()), None)
            name_col = kr.columns[0]
            if p_col:
                kr = kr.sort_values(p_col)
                top = kr[[name_col, p_col]].head(10).to_string(index=False)
            else:
                top = kr.head(10).to_string(index=False)
            log.info("Cluster %d HOMER done in %.1fs. Top known motifs:\n%s", cid, elapsed, top)
            kr.to_csv(cluster_dir / "knownResults_summary.csv", index=False)
        except Exception as exc:
            log.warning("Cluster %d: HOMER done in %.1fs (parse warning: %s)", cid, elapsed, exc)
            log.debug(traceback.format_exc())

        results[cid] = str(known_txt)

    n_ok = sum(1 for v in results.values() if v)
    log.info("HOMER complete: %d/%d clusters succeeded → %s", n_ok, len(cluster_ids), homer_root)
    return results


def main():
    args = parse_args()
    # Logger not yet wired to file here; run_motif_analysis sets it up once out_dir is known
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)-8s] %(message)s",
                        datefmt="%H:%M:%S",
                        stream=sys.stdout)

    motif_result = run_motif_analysis(
        input_csv      = args.input,
        build          = args.build,
        out_dir        = args.out_dir,
        jaspar_bed_arg = args.jaspar_bed,
        jaspar_dir     = args.jaspar_dir,
        p_threshold    = args.p_threshold,
        force          = args.force,
        bed_input      = args.bed_input,
    )

    if args.homer:
        run_homer(
            input_csv = args.input,
            build     = args.build,
            out_dir   = args.out_dir,
            genome    = args.homer_genome,
            size      = args.homer_size,
            threads   = args.homer_threads,
            force     = args.force,
        )

    if args.run_go:
        log.info("--- STAGE: Chained GO Annotation (--run-go) ---")
        try:
            import te_go
        except ImportError:
            log.error("te_go.py not found on PYTHONPATH; cannot chain GO analysis.")
            sys.exit(1)
        enrich_dir = motif_result.get("enrichment_dir", str(Path(args.out_dir) / "enrichment_results"))
        log.info("Running te_go.run_go_annotation(enrichment_dir=%s, build=%s)", enrich_dir, args.build)
        te_go.run_go_annotation(
            enrichment_dir = enrich_dir,
            build          = args.build,
            out_dir        = args.out_dir,
            clustered_csv  = args.input,   # None in bed-input mode — strand plots skipped
            p_threshold    = args.p_threshold,
            top_motifs     = args.go_top_motifs,
            force          = args.force,
        )
        log.info("GO annotation complete.")

    if args.notify_email:
        from te_notify import send_completion_email
        send_completion_email(args.notify_email, "te_motif", args.out_dir)


if __name__ == "__main__":
    main()
