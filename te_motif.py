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
  <out_dir>/enrichment_results/
    cluster_N_enrichment.csv       Fisher p-values per cluster
    enrichment_heatmap.png         -log10(p) heatmap across clusters

JASPAR BED resolution order
  1. --jaspar-bed FILE
  2. TE_JASPAR_<BUILD> environment variable
  3. <jaspar-dir>/JASPAR2024_<build>.sorted.bed.gz  (cached)
  4. Auto-download from jaspar.elixir.no

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

# ── JASPAR download URLs ──────────────────────────────────────────────────────

JASPAR_URLS = {
    # Historical BED URLs are no longer available on the JASPAR server. Keep
    # these here only so failures are explicit and validated instead of silently
    # accepting a 404 HTML page as a BED file.
    "hg38": "https://jaspar.elixir.no/static/data/beds/JASPAR2024_hg38.bed.gz",
    "hg19": "https://jaspar.elixir.no/static/data/beds/JASPAR2024_hg19.bed.gz",
    "mm10": "https://jaspar.elixir.no/static/data/beds/JASPAR2024_mm10.bed.gz",
    "mm39": "https://jaspar.elixir.no/static/data/beds/JASPAR2024_mm39.bed.gz",
}

JASPAR_BIGBED_URLS = {
    "hg38": "https://mencius.uio.no/JASPAR/JASPAR_familial_binding_sites/2024/hg38/JASPAR2024_hg38.bb",
    "hg19": "https://mencius.uio.no/JASPAR/JASPAR_familial_binding_sites/2024/hg19/JASPAR2024_hg19.bb",
    "mm10": "https://mencius.uio.no/JASPAR/JASPAR_familial_binding_sites/2024/mm10/JASPAR2024_mm10.bb",
    "mm39": "https://mencius.uio.no/JASPAR/JASPAR_familial_binding_sites/2024/mm39/JASPAR2024_mm39.bb",
}

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
        log.debug("bigBedToBed cached: %s", exe)
        return str(exe)

    url = f"https://hgdownload.soe.ucsc.edu/admin/exe/{ucsc_platform}/bigBedToBed"
    log.info("bigBedToBed not found — downloading UCSC utility: %s", url)
    try:
        import requests
        with requests.get(url, stream=True, timeout=60) as r:
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


def _build_locus_jaspar_bed_from_bigbed(build, jaspar_dir, loci_bed):
    """Query remote JASPAR bigBed by TE chromosome spans and write a BED file."""
    import subprocess

    bb_url = JASPAR_BIGBED_URLS.get(build)
    if not bb_url or not loci_bed:
        return None

    tool = _get_bigbed_to_bed(jaspar_dir)
    if not tool:
        log.warning("bigBedToBed is not available; cannot auto-query JASPAR bigBed.")
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

    out_path = Path(jaspar_dir) / f"JASPAR2024_{build}.te_loci.bed"
    tmp_path = out_path.with_suffix(".bed.part")
    total = 0
    log.info("Auto-querying JASPAR bigBed for %d TE loci", len(loci))
    log.info("  Source: %s", bb_url)
    log.info("  Output: %s", out_path)
    seen = set()
    errors = 0
    t0 = time.time()
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
    log.info("bigBed query complete: %d records in %.1fs", total, elapsed)
    if validate_jaspar_bed(out_path):
        log.info("JASPAR locus BED ready: %s", out_path)
        return str(out_path)
    out_path.unlink(missing_ok=True)
    log.error("bigBed-derived BED failed validation; discarding %s", out_path)
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

    local_path = jaspar_dir / f"JASPAR2024_{build}.sorted.bed.gz"
    if local_path.exists():
        log.info("Found cached JASPAR BED: %s (%.1f MB)",
                 local_path, local_path.stat().st_size / 1e6)
        if validate_jaspar_bed(local_path):
            log.info("Cached JASPAR BED is valid — using it.")
            return str(local_path)
        log.warning("Cached JASPAR file is invalid; deleting: %s", local_path)
        local_path.unlink()

    url = JASPAR_URLS.get(build)
    if not url:
        log.error("No JASPAR URL for build '%s'. Provide --jaspar-bed or set TE_JASPAR_%s.",
                  build, build.upper())
        sys.exit(1)

    log.info("JASPAR BED not found locally — attempting download from: %s", url)
    log.info("Destination: %s  (hg38 ~1-2 GB; may take several minutes)", local_path)
    log.info("Tip: set TE_JASPAR_%s=/path/to/file to skip future downloads", build.upper())

    import shutil
    import subprocess
    tmp_path = local_path.with_suffix(local_path.suffix + ".part")
    for tool in [
        ["wget", "-q", "--show-progress", "-O", str(tmp_path), url],
        ["curl", "-L", "-o", str(tmp_path), url],
    ]:
        if shutil.which(tool[0]) is None:
            log.debug("%s not found — trying next downloader", tool[0])
            continue
        log.info("Downloading with %s ...", tool[0])
        try:
            if subprocess.run(tool).returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size > 0:
                if validate_jaspar_bed(tmp_path):
                    tmp_path.replace(local_path)
                    log.info("Downloaded %.0f MB → %s",
                             local_path.stat().st_size / 1e6, local_path)
                    return str(local_path)
                log.warning("Downloaded file is not BED/GZIP; discarding %s", tmp_path)
                tmp_path.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("%s failed to start: %s", tool[0], exc)

    log.info("wget/curl unavailable or failed — trying Python requests downloader")
    try:
        import requests
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            downloaded = 0
            with open(tmp_path, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if downloaded and downloaded % (100 * 1024 * 1024) < 1024 * 1024:
                        log.info("  downloaded %.0f MB", downloaded / 1e6)
        if tmp_path.exists() and tmp_path.stat().st_size > 0:
            if validate_jaspar_bed(tmp_path):
                tmp_path.replace(local_path)
                log.info("Downloaded %.0f MB → %s",
                         local_path.stat().st_size / 1e6, local_path)
                return str(local_path)
            log.warning("Downloaded file is not BED/GZIP; discarding %s", tmp_path)
    except Exception as exc:
        log.error("Python requests download failed: %s", exc)
        log.debug(traceback.format_exc())
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    locus_bed = _build_locus_jaspar_bed_from_bigbed(build, jaspar_dir, loci_bed)
    if locus_bed:
        return locus_bed

    bb = JASPAR_BIGBED_URLS.get(build)
    log.error("FATAL: all JASPAR BED download methods failed.")
    if bb:
        log.error("JASPAR now publishes this build as bigBed: %s", bb)
        log.error("Convert/query that bigBed to BED, then rerun with:")
        log.error("  --jaspar-bed /path/to/JASPAR2024_%s.bed.gz", build)
    log.error("Manual wget command: wget -O %s '%s'", local_path, url)
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


def bedtools_intersect_safe(v_bed, jaspar_bed, scratch=None):
    """Run bedtools intersect, writing output to a temp file to avoid OOM/SIGBUS.

    Always uses the CLI path (streaming to disk) to avoid pybedtools mmap issues
    on HPC scratch filesystems.  pybedtools is only used as a last resort.
    """
    import subprocess, shutil as _shutil

    class _OverlapResult(list):
        def to_dataframe(self, names=None, header=None):
            rows = [str(x).rstrip("\n").split("\t") for x in self]
            return pd.DataFrame(rows, columns=names)

    def _run_cli(a, b):
        """Stream bedtools output to a temp file; return (_OverlapResult, stderr)."""
        tmp_dir = scratch or os.environ.get("TMPDIR") or str(Path(a).parent)
        tmp_out = Path(tmp_dir) / f"bedtools_overlaps_{os.getpid()}.tsv"
        cmd = ["bedtools", "intersect", "-a", str(a), "-b", str(b), "-wa", "-wb"]
        log.debug("bedtools CLI command: %s", " ".join(cmd))
        log.debug("bedtools output temp file: %s", tmp_out)
        t0 = time.time()
        with open(tmp_out, "w") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.PIPE)
        elapsed = time.time() - t0
        stderr_text = proc.stderr.decode(errors="replace")
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
        log.warning("bedtools not found on PATH — falling back to pybedtools")
    else:
        log.info("Running bedtools intersect (CLI, streaming to disk) ...")
        log.debug("  -a: %s  -b: %s", v_bed, jaspar_bed)
        overlaps, stderr_text = _run_cli(v_bed, jaspar_bed)
        if overlaps is None and "fields" in stderr_text.lower():
            log.warning("Column mismatch detected — normalising JASPAR BED to 6 cols ...")
            norm = str(jaspar_bed).replace(".bed.gz", ".norm6.bed").replace(".bed", ".norm6.bed")
            _normalise_bed(str(jaspar_bed), norm, 6)
            overlaps, stderr_text = _run_cli(v_bed, norm)
        if overlaps is not None:
            log.info("bedtools intersect: %d overlaps", len(overlaps))
            return overlaps
        log.error("bedtools CLI failed; stderr: %s", stderr_text[:1000])

    # pybedtools fallback — only reached if bedtools CLI is unavailable or failed
    try:
        import pybedtools
    except ImportError:
        log.error("FATAL: bedtools not found on PATH and pybedtools not installed.")
        sys.exit(1)

    log.info("Running bedtools intersect (pybedtools fallback) ...")
    if scratch:
        pybedtools.set_tempdir(scratch)
        log.debug("pybedtools tempdir: %s", scratch)
    v_bt = pybedtools.BedTool(str(v_bed))
    m_bt = pybedtools.BedTool(str(jaspar_bed))
    t0 = time.time()
    try:
        overlaps = v_bt.intersect(m_bt, wa=True, wb=True)
        log.info("pybedtools intersect: %d overlaps in %.1fs", len(overlaps), time.time() - t0)
        return overlaps
    except Exception as exc:
        if "fields" in str(exc).lower():
            log.warning("pybedtools column mismatch — normalising JASPAR BED to 6 cols ...")
            norm = str(jaspar_bed).replace(".bed.gz", ".norm6.bed.gz").replace(".bed", ".norm6.bed")
            _normalise_bed(str(jaspar_bed), norm, 6)
            overlaps = v_bt.intersect(pybedtools.BedTool(norm), wa=True, wb=True)
            log.info("pybedtools retry: %d overlaps", len(overlaps))
            return overlaps
        log.error("pybedtools intersect failed: %s", exc)
        log.debug(traceback.format_exc())
        raise


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


if __name__ == "__main__":
    main()
