#!/usr/bin/env python3
"""
te_prep.py
─────────────────────────────────────────────────────────────────────────────
Fetch TE family loci from a local RepeatMasker table and extract sequences
from a local genome FASTA.

Setup (run once on a node with internet access to get the rmsk table):
    python te_prep.py --download hg38
    python te_prep.py --download mm10

Typical run:
    python te_prep.py AluSz  hg38 --genome-fa /data/hg38.fa
    python te_prep.py B2_Mm2 mm10 --genome-fa /data/mm10.fa --max-loci 500
    python te_prep.py L1HS   hg38 --list-families
    python te_prep.py --search Alu --build hg38

Output:
    <base-dir>/<family>_analysis_results/clustered_data.csv
    (compatible with te_enrichment.py and query.py)

HPC submission (LSF / bsub):
    bsub -q normal -M 8000 -o prep.log python te_prep.py AluSz hg38

HPC submission (Slurm):
    sbatch --mem=8G -o prep.log --wrap="python te_prep.py AluSz hg38"
"""

import argparse
import gzip
import os
import re
import sys
import time
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# numpy and pandas are only needed for the full analysis path (not --count-only).
# Import lazily so the HPC can run count/query modes without these packages.
try:
    import numpy as np
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    np = None  # type: ignore[assignment]
    pd = None  # type: ignore[assignment]
    _HAS_PANDAS = False

# requests, threading, and ThreadPoolExecutor are imported lazily inside
# fetch_sequences_ucsc() so the HPC can run te_prep.py without them.

# Optional pysam for fast indexed FASTA access
try:
    import pysam
    _PYSAM = True
except ImportError:
    _PYSAM = False

# ═══════════════════════════════════════════════════════════════════════════
# PATHS (all overridable via CLI args or environment variables)
# ═══════════════════════════════════════════════════════════════════════════

# Default data root — override with --base-dir or TE_BASE_DIR env var
_DEFAULT_BASE = os.environ.get("TE_BASE_DIR", str(Path.home() / "te_analysis"))

# Default rmsk directory — override with --rmsk-dir or TE_RMSK_DIR env var
_DEFAULT_RMSK = os.environ.get("TE_RMSK_DIR", str(Path(_DEFAULT_BASE) / "rmsk"))

# Known genome FASTA paths (env-variable overridable)
GENOME_FA = {
    "hg38": os.environ.get("HG38_FA", ""),
    "hg19": os.environ.get("HG19_FA", ""),
    "mm10": os.environ.get("MM10_FA", ""),
    "mm39": os.environ.get("MM39_FA", ""),
}

# UCSC rmsk download URLs
RMSK_URLS = {
    "hg38": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/rmsk.txt.gz",
    "hg19": "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/rmsk.txt.gz",
    "mm10": "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/database/rmsk.txt.gz",
    "mm39": "https://hgdownload.soe.ucsc.edu/goldenPath/mm39/database/rmsk.txt.gz",
}

# Dfam REST API base — per-family annotations, no bulk file download needed
DFAM_API_BASE = "https://www.dfam.org/api"

# Assembly ID mapping for the Dfam API
DFAM_ASSEMBLY_IDS = {
    "hg38": "hg38", "hg19": "hg19",
    "mm10": "mm10", "mm39": "mm39",
}

# rmsk.txt.gz column indices (tab-separated, no header)
COL_CHROM = 5; COL_START = 6; COL_END = 7; COL_STRAND = 9
COL_REPNAME = 10; COL_REPCLASS = 11; COL_REPFAM = 12
COL_SWSCORE = 1; COL_MILLIDIV = 2

STD_CHROMS_HUMAN = set([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"])
STD_CHROMS_MOUSE = set([f"chr{i}" for i in range(1, 20)] + ["chrX", "chrY"])


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="TE family data prep — local rmsk + local FASTA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python te_prep.py --download hg38
  python te_prep.py AluSz  hg38 --genome-fa /data/hg38.fa
  python te_prep.py B2_Mm2 mm10 --genome-fa /data/mm10.fa --max-loci 500
  python te_prep.py L1HS   hg38 --list-families
  python te_prep.py --search Alu --build hg38
        """,
    )
    p.add_argument("family", nargs="?", default=None,
                   help="TE repName (e.g. AluSz, B2_Mm2, L1HS)")
    p.add_argument("build",  nargs="?", default=None,
                   help="Genome build: hg38, hg19, mm10, mm39 (default: hg38)")
    # Named-flag aliases so the GUI can pass --family / --build explicitly
    p.add_argument("--family", dest="family_flag", default=None,
                   help="TE repName (named-flag form; overrides positional)")
    p.add_argument("--build",  dest="build_flag",  default=None,
                   help="Genome build (named-flag form; overrides positional)")
    p.add_argument("--download",  metavar="BUILD",
                   help="Download rmsk.txt.gz for BUILD (run on login node with internet)")
    p.add_argument("--base-dir",  default=_DEFAULT_BASE,
                   help=f"Output base directory (default: {_DEFAULT_BASE})")
    p.add_argument("--rmsk-dir",  default=_DEFAULT_RMSK,
                   help=f"Directory holding rmsk_<build>.txt.gz files (default: {_DEFAULT_RMSK})")
    p.add_argument("--genome-fa", default=None,
                   help="Path to genome FASTA (auto-detected from TE_<BUILD>_FA env var)")
    p.add_argument("--max-loci",  type=int, default=None,
                   help="Cap number of loci (useful for testing)")
    p.add_argument("--list-families", action="store_true",
                   help="List all repNames and their counts, then exit")
    p.add_argument("--search", default=None,
                   help="Search for repNames containing this substring")
    p.add_argument("--no-filter-chroms", action="store_true",
                   help="Include non-standard chromosomes (default: standard only)")
    p.add_argument("--count-only", action="store_true",
                   help="Print locus count for the family and exit (no sequence extraction)")
    p.add_argument("--source", choices=["rmsk", "dfam"], default="rmsk",
                   help="Annotation source: rmsk (UCSC RepeatMasker) or dfam (Dfam nrph hits) [default: rmsk]")
    p.add_argument("--dfam-url", default=None,
                   help="Override Dfam download URL for this build")
    p.add_argument("--fetch-workers", type=int, default=3,
                   help="Parallel UCSC fetch workers when no genome FASTA is available (default: 3; capped at 3 concurrent regardless)")
    p.add_argument("--notify-email", default="", metavar="EMAIL",
                   help="Send a completion email to this address (requires Gmail App Password setup).")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════

def download_rmsk(build, rmsk_dir):
    """Download rmsk.txt.gz from UCSC. Requires internet."""
    url = RMSK_URLS.get(build)
    if not url:
        print(f"FATAL: No rmsk URL for build '{build}'")
        print(f"Known builds: {', '.join(RMSK_URLS)}")
        sys.exit(1)

    rmsk_dir = Path(rmsk_dir)
    rmsk_dir.mkdir(parents=True, exist_ok=True)
    outpath = rmsk_dir / f"rmsk_{build}.txt.gz"

    if outpath.exists():
        size_mb = outpath.stat().st_size / 1e6
        print(f"Already exists: {outpath} ({size_mb:.0f} MB). Delete to re-download.")
        return str(outpath)

    print(f"Downloading {url} → {outpath}")
    print("  (hg38 ~150 MB, may take a few minutes)")

    import subprocess, shlex

    # Use shell=True so DNS resolution goes through /bin/sh — required on nodes
    # where list-form subprocess DNS fails. Also try -4 to force IPv4 in case
    # IPv6 resolution is broken on this node.
    for tool_cmd in [
        f"curl -4 -L --progress-bar -o {shlex.quote(str(outpath))} {shlex.quote(url)}",
        f"curl -L --progress-bar -o {shlex.quote(str(outpath))} {shlex.quote(url)}",
        f"wget -q --show-progress -O {shlex.quote(str(outpath))} {shlex.quote(url)}",
    ]:
        try:
            ret = subprocess.run(tool_cmd, shell=True)
            if ret.returncode == 0 and outpath.exists() and outpath.stat().st_size > 0:
                size_mb = outpath.stat().st_size / 1e6
                print(f"Done: {outpath} ({size_mb:.0f} MB)")
                return str(outpath)
            if outpath.exists():
                outpath.unlink()  # remove partial download
        except Exception:
            continue

    raise RuntimeError(
        f"rmsk bulk download failed for {build}: hgdownload.soe.ucsc.edu unreachable. "
        f"Will fall back to UCSC REST API."
    )


# ═══════════════════════════════════════════════════════════════════════════
# PARSE rmsk
# ═══════════════════════════════════════════════════════════════════════════

def get_rmsk_path(build, rmsk_dir):
    path = Path(rmsk_dir) / f"rmsk_{build}.txt.gz"
    if not path.exists():
        if build not in RMSK_URLS:
            print(f"FATAL: rmsk file not found: {path}")
            print(f"Run first:  python te_prep.py --download {build}")
            sys.exit(1)
        print(f"rmsk file not found for {build} — downloading automatically...")
        download_rmsk(build, rmsk_dir)
    return str(path)


def parse_rmsk_family(rmsk_path, family, std_chroms=None):
    """Stream rmsk.txt.gz and return list of dicts for rows matching repName."""
    hits = []
    t0 = time.time()
    family_lc = str(family).lower()
    print(f"  Scanning {Path(rmsk_path).name} for repName='{family}'...")

    with gzip.open(rmsk_path, "rt") as f:
        for n_lines, line in enumerate(f, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 13:
                continue
            if fields[COL_REPNAME] != family and fields[COL_REPNAME].lower() != family_lc:
                continue
            chrom = fields[COL_CHROM]
            if std_chroms and chrom not in std_chroms:
                continue
            hits.append({
                "Chromosome": chrom,
                "Start":      int(fields[COL_START]),
                "Stop":       int(fields[COL_END]),
                "strand":     fields[COL_STRAND],
                "repName":    fields[COL_REPNAME],
                "repClass":   fields[COL_REPCLASS],
                "repFamily":  fields[COL_REPFAM],
                "swScore":    int(fields[COL_SWSCORE]),
                "milliDiv":   int(fields[COL_MILLIDIV]),
            })
            if len(hits) % 10_000 == 0:
                print(f"    {len(hits):,} hits so far...", flush=True)

    print(f"  Scanned {n_lines:,} lines in {time.time()-t0:.1f}s → {len(hits):,} hits")
    return hits


def fetch_rmsk_from_api(family, build, std_chroms=None):
    """Fetch rmsk loci for one family from the UCSC REST API, chromosome by chromosome.

    Fallback when hgdownload.soe.ucsc.edu is unreachable.
    Uses api.genome.ucsc.edu/getData/track which is accessible from more networks.
    Returns the same list-of-dicts format as parse_rmsk_family().
    """
    if build.startswith("hg"):
        all_chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    else:
        all_chroms = [f"chr{i}" for i in range(1, 20)] + ["chrX", "chrY"]

    chroms = [c for c in all_chroms if std_chroms is None or c in std_chroms]
    hits = []
    print(f"  UCSC REST API fallback: querying rmsk '{family}' in {build} "
          f"across {len(chroms)} chromosomes...", flush=True)

    for chrom in chroms:
        base = (f"https://api.genome.ucsc.edu/getData/track"
                f"?genome={build};track=rmsk;chrom={chrom}")
        offset = None
        chrom_hits = 0
        while True:
            url = base + (f";offset={offset}" if offset is not None else "")
            try:
                data = _curl_json(url, timeout=120)
            except Exception as exc:
                print(f"    {chrom}: fetch error — {exc}", flush=True)
                break
            for rec in data.get("rmsk", []):
                if rec.get("repName") == family:
                    hits.append({
                        "Chromosome": rec["genoName"],
                        "Start":      int(rec["genoStart"]),
                        "Stop":       int(rec["genoEnd"]),
                        "strand":     rec.get("strand", "+"),
                        "repName":    rec["repName"],
                        "repClass":   rec.get("repClass", ""),
                        "repFamily":  rec.get("repFamily", ""),
                        "swScore":    int(rec.get("swScore", 0)),
                        "milliDiv":   int(rec.get("milliDiv", 0)),
                    })
                    chrom_hits += 1
            next_off = data.get("nextOffset")
            if not next_off or next_off == offset:
                break
            offset = next_off
        if chrom_hits:
            print(f"    {chrom}: {chrom_hits} hits", flush=True)

    print(f"  API rmsk done: {len(hits)} loci for '{family}'", flush=True)
    return hits


def count_family_rmsk(rmsk_path, family, std_chroms=None):
    """Stream rmsk.txt.gz and return the hit count without building a list."""
    count = 0
    family_lc = str(family).lower()
    with gzip.open(rmsk_path, "rt") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 13:
                continue
            if fields[COL_REPNAME] != family and fields[COL_REPNAME].lower() != family_lc:
                continue
            if std_chroms and fields[COL_CHROM] not in std_chroms:
                continue
            count += 1
    return count


def list_families(rmsk_path, search=None):
    """Print top repNames and their counts."""
    counts = Counter()
    print(f"  Scanning {Path(rmsk_path).name}...")
    t0 = time.time()
    with gzip.open(rmsk_path, "rt") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) > COL_REPNAME:
                counts[fields[COL_REPNAME]] += 1
    print(f"  Done [{time.time()-t0:.1f}s]  {len(counts)} unique repNames\n")

    if search:
        sl = search.lower()
        counts = {k: v for k, v in counts.items() if sl in k.lower()}
        print(f"  Matches for '{search}': {len(counts)} families\n")

    for name, cnt in sorted(counts.items(), key=lambda x: -x[1])[:50]:
        print(f"  {name:35s}  {cnt:>9,}")
    if len(counts) > 50:
        print(f"  ... and {len(counts)-50} more")


# ═══════════════════════════════════════════════════════════════════════════
# DFAM
# ═══════════════════════════════════════════════════════════════════════════

## Dfam uses a REST API per family — no bulk file download needed.
## API: GET /api/families/{accession}/assemblies/{assembly}/annotations?nrph=true
## Returns a gzip-compressed TSV with columns:
##   #sequence name | model accession | model name | bit score | e-value |
##   hmm start | hmm end | hmm length | strand |
##   alignment start | alignment end | envelope start | envelope end | sequence length

# Dfam REST API column indices (0-based, fixed format)
_DFAM_COL_CHROM  = 0   # #sequence name
_DFAM_COL_FAMILY = 2   # model name
_DFAM_COL_STRAND = 8   # strand
_DFAM_COL_START  = 9   # alignment start
_DFAM_COL_END    = 10  # alignment end


def _curl_bytes(url, timeout=120):
    """Fetch a URL with curl (shell=True) and return the raw response bytes."""
    import subprocess, shlex
    result = subprocess.run(
        f"curl -s -L --connect-timeout 10 --max-time {timeout} {shlex.quote(url)}",
        shell=True, capture_output=True, timeout=timeout + 15,
    )
    if result.returncode != 0:
        raise ConnectionError(
            f"curl exit={result.returncode}: {result.stderr.decode(errors='replace').strip()[:120]}"
        )
    return result.stdout  # bytes


def _dfam_lookup_accession(family_name):
    """Return Dfam accession for a family name via the REST API."""
    import urllib.parse
    url = (f"{DFAM_API_BASE}/families"
           f"?name_prefix={urllib.parse.quote(family_name)}&limit=10")
    try:
        data = _curl_json(url, timeout=20)
        for entry in data.get("results", []):
            if entry.get("name") == family_name:
                return entry["accession"]
        results = data.get("results", [])
        if results:
            print(f"  WARNING: exact match for '{family_name}' not found; "
                  f"using closest: {results[0]['name']} ({results[0]['accession']})", flush=True)
            return results[0]["accession"]
        return None
    except Exception as e:
        print(f"FATAL: Dfam API lookup failed for '{family_name}': {e}", flush=True)
        sys.exit(1)


def _dfam_fetch_raw(family_name, build):
    """Fetch per-family Dfam annotations from the REST API.
    Returns the raw gzip bytes (the API returns a gzip-compressed TSV).
    """
    accession = _dfam_lookup_accession(family_name)
    if not accession:
        print(f"FATAL: Family '{family_name}' not found in Dfam.", flush=True)
        sys.exit(1)

    assembly = DFAM_ASSEMBLY_IDS.get(build, build)
    url = (f"{DFAM_API_BASE}/families/{accession}"
           f"/assemblies/{assembly}/annotations?nrph=true")
    print(f"  Dfam REST API: {url}", flush=True)
    try:
        return _curl_bytes(url, timeout=120)
    except Exception as e:
        print(f"FATAL: Dfam API fetch failed: {e}", flush=True)
        sys.exit(1)


def _dfam_open_raw(raw_bytes):
    """Open raw gzip bytes from the Dfam API as a text stream."""
    import io as _io
    return gzip.open(_io.BytesIO(raw_bytes), "rt")


def download_dfam(build, rmsk_dir, url_override=None):
    """Cache per-family Dfam data is not applicable for bulk download.
    This stub exists for CLI compatibility; actual data is fetched per-family via API.
    """
    print("  NOTE: Dfam uses a per-family REST API — no bulk file to pre-download.")
    print("  Run  python te_prep.py <family> <build> --source dfam  to fetch and process.")


def get_dfam_path(build, rmsk_dir):
    """Stub for CLI compatibility — Dfam data is fetched live from the REST API."""
    return None


def parse_dfam_family(_, family, build="hg38", std_chroms=None):
    """Fetch family hits from the Dfam REST API and return list of dicts."""
    t0 = time.time()
    print(f"  Fetching Dfam hits for '{family}' in {build} via REST API…", flush=True)
    raw = _dfam_fetch_raw(family, build)
    hits = []
    n_lines = 0
    with _dfam_open_raw(raw) as f:
        for line in f:
            if line.startswith("#"):
                continue
            n_lines += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) <= _DFAM_COL_END:
                continue
            chrom = fields[_DFAM_COL_CHROM]
            if std_chroms and chrom not in std_chroms:
                continue
            start = int(fields[_DFAM_COL_START])
            end   = int(fields[_DFAM_COL_END])
            # Dfam reports - strand with start > end; normalise
            if start > end:
                start, end = end, start
            hits.append({
                "Chromosome": chrom,
                "Start":      start,
                "Stop":       end,
                "strand":     fields[_DFAM_COL_STRAND],
                "repName":    fields[_DFAM_COL_FAMILY],
                "repClass":   "",
                "repFamily":  "",
                "swScore":    0,
                "milliDiv":   0,
            })
            if len(hits) % 10_000 == 0:
                print(f"    {len(hits):,} hits so far...", flush=True)
    print(f"  {n_lines:,} data lines → {len(hits):,} hits [{time.time()-t0:.1f}s]")
    return hits


def count_family_dfam(_, family, build="hg38", std_chroms=None):
    """Count Dfam hits for *family* via the REST API."""
    t0 = time.time()
    print(f"  Fetching Dfam hits for '{family}' in {build} via REST API…", flush=True)
    raw = _dfam_fetch_raw(family, build)
    count = 0
    with _dfam_open_raw(raw) as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) <= _DFAM_COL_CHROM:
                continue
            if std_chroms and fields[_DFAM_COL_CHROM] not in std_chroms:
                continue
            count += 1
    return count


def list_families_dfam(_, search=None):
    """Dfam family search is not supported via the bulk-file path; use the API directly."""
    print("  Dfam family listing requires the REST API — use --search with --source rmsk,")
    print("  or visit https://www.dfam.org to browse families.")


# ═══════════════════════════════════════════════════════════════════════════
# SEQUENCE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def _revcomp(seq):
    comp = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")
    return seq.translate(comp)[::-1]


def _row_strand(row):
    for col in ("strand", "Strand"):
        if col in row and pd.notna(row.get(col)):
            val = str(row.get(col)).strip()
            if val in {"+", "-"}:
                return val
    for col in ("TE_ID", "TE_name", "name", "Name", "id", "ID"):
        if col not in row or pd.isna(row.get(col)):
            continue
        text = str(row.get(col)).strip()
        if not text:
            continue
        m = re.search(r"(?:^|[|:,\s])([+-])$", text)
        if m:
            return m.group(1)
    return "+"


def _orient_sequence(seq, row):
    seq = str(seq).upper()
    return _revcomp(seq) if _row_strand(row) == "-" else seq


def _ucsc_api_url(chrom, start, stop, assembly):
    return (
        f"https://api.genome.ucsc.edu/getData/sequence?"
        f"genome={assembly};chrom={chrom};start={int(start)};end={int(stop)}"
    )


def _check_ucsc_connectivity():
    """Return (ok, error_msg). Uses curl — more reliable than Python socket on HPC nodes
    where nsswitch/libc DNS may differ from the system curl resolver."""
    import subprocess
    try:
        r = subprocess.run(
            "curl -s --connect-timeout 6 --max-time 8 -o /dev/null "
            "-w '%{http_code}' 'https://api.genome.ucsc.edu/'",
            shell=True, capture_output=True, text=True, timeout=12,
        )
        code = r.stdout.strip()
        if r.returncode == 0 and code not in ("", "000"):
            return True, None
        return False, f"curl exit={r.returncode} http={code} stderr={r.stderr.strip()[:120]}"
    except Exception as e:
        return False, str(e)


_CONNECTIVITY_ERROR = """\
  FATAL: This node cannot reach api.genome.ucsc.edu (DNS resolution failed).
  HPC compute nodes typically block outbound internet — UCSC API is unreachable.

  Options:
    1) Provide a local genome FASTA with --genome /path/to/{build}.fa
       (fastest, no network required)
    2) On a login node that has internet access, run te_prep.py manually,
       then pass the resulting CSV as --input to te_prep / query.py on the cluster.
    3) Use --source dfam only if your cluster allows outbound HTTPS to dfam.org.
"""


def _curl_json(url, timeout=30):
    """Fetch a URL with curl (via shell=True) and return the parsed JSON dict.

    shell=True is critical: it routes DNS through /bin/sh, matching the same
    resolver path as interactive curl. Without it, subprocess curl uses a
    different libc DNS path that fails on some HPC nodes (exit 6 / exit 28).
    """
    import subprocess, json as _json, shlex
    cmd = f"curl -s --connect-timeout 8 --max-time {timeout} {shlex.quote(url)}"
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=timeout + 8,
    )
    if result.returncode != 0:
        raise ConnectionError(
            f"curl exit={result.returncode}: {(result.stderr or result.stdout).strip()[:120]}"
        )
    try:
        return _json.loads(result.stdout)
    except _json.JSONDecodeError as exc:
        raise ValueError(f"JSON parse error ({exc}): body={result.stdout[:80]!r}")


def fetch_sequences_ucsc(df, assembly="hg38", n_workers=10):
    """Fetch sequences from UCSC API in parallel via curl. Returns (seqs, urls) lists."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    n = len(df)
    seqs = [None] * n
    urls = [""] * n
    failed = []
    # Cap concurrency at 3 — UCSC rate-limits aggressively
    _concurrency = min(n_workers, 3)
    sem = threading.Semaphore(_concurrency)
    _print_lock = threading.Lock()

    def _log(msg):
        with _print_lock:
            print(msg, flush=True)

    def _fetch_one(i):
        row   = df.iloc[i]
        chrom = str(row.get("chr", row.get("Chromosome", "")))
        start = int(row.get("start", row.get("Start", 0)))
        stop  = int(row.get("stop",  row.get("Stop",  0)))
        url   = _ucsc_api_url(chrom, start, stop, assembly)
        for attempt in range(5):
            try:
                with sem:
                    time.sleep(0.4)  # ≤2.5 req/sec across all workers
                    _log(f"    [{i}] GET {url}")
                    res = _curl_json(url)
                http_code = res.get("statusCode", 200)
                _log(f"    [{i}] HTTP {http_code}")
                if http_code == 429:
                    wait = min(120, 30 * 2 ** attempt)
                    _log(f"    [{i}] rate-limited — sleeping {wait}s before retry {attempt+1}")
                    time.sleep(wait)
                    continue
                if "error" in res:
                    raise ValueError(res["error"])
                dna = res.get("dna", "")
                _log(f"    [{i}] OK  len={len(dna)}  seq={dna}")
                return i, _orient_sequence(dna, row), url
            except Exception as exc:
                _log(f"    [{i}] attempt {attempt+1}/5 FAILED: {type(exc).__name__}: {exc}")
                if attempt == 4:
                    _log(f"    [{i}] giving up — filling with Ns")
                    return i, None, url
                wait = min(60, 2 ** attempt)
                _log(f"    [{i}] retrying in {wait}s")
                time.sleep(wait)
        return i, None, url

    print(f"  Fetching {n:,} sequences from UCSC ({_concurrency} workers, assembly={assembly})...", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as exe:
        futures = {exe.submit(_fetch_one, i): i for i in range(n)}
        for fut in as_completed(futures):
            if _abort.is_set():
                # Cancel remaining submitted work
                for f in futures:
                    f.cancel()
                print(_CONNECTIVITY_ERROR.format(build=assembly), flush=True)
                sys.exit(1)
            i, seq, url = fut.result()
            urls[i] = url
            if seq is None:
                failed.append(i)
                seqs[i] = "N" * 100
            else:
                seqs[i] = seq
            done += 1
            if done % max(1, n // 20) == 0 or done == n:
                print(f"  --- progress: {done:,}/{n:,} done, {len(failed)} failed so far ---", flush=True)

    if failed:
        print(f"  WARNING: {len(failed)} sequences failed to fetch (filled with Ns)")
    return seqs, urls


def extract_sequences_pysam(genome_fa, df):
    """Use pysam for fast indexed FASTA access (auto-builds .fai index)."""
    fai = genome_fa + ".fai"
    if not os.path.exists(fai):
        print("  Building FASTA index (.fai)...")
        pysam.faidx(genome_fa)

    fasta = pysam.FastaFile(genome_fa)
    available = set(fasta.references)
    seqs = []
    skipped = 0
    t0 = time.time()
    n = len(df)
    print(f"  Extracting {n:,} sequences (pysam)...")

    for i, row in df.iterrows():
        chrom = row["Chromosome"]
        if chrom not in available:
            seqs.append("")
            skipped += 1
            continue
        try:
            seq = fasta.fetch(chrom, int(row["Start"]), int(row["Stop"]))
            if _row_strand(row) == "-":
                seq = _revcomp(seq)
            seqs.append(seq)
        except Exception:
            seqs.append("")
            skipped += 1
        if (i + 1) % 10_000 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"    {i+1:,}/{n:,}  ({rate:.0f}/s)", flush=True)

    fasta.close()
    print(f"  Done: {n - skipped:,}/{n:,} extracted [{time.time()-t0:.1f}s]")
    return seqs


def extract_sequences_fasta(genome_fa, df):
    """Pure-Python FASTA extraction (no pysam). Loads genome into memory once."""
    print("  Loading genome into memory (no pysam, may take a minute)...")
    t0 = time.time()
    genomes = {}
    chrom = None
    buf = []
    with open(genome_fa) as fh:
        for line in fh:
            if line.startswith(">"):
                if chrom:
                    genomes[chrom] = "".join(buf).upper()
                chrom = line[1:].strip().split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if chrom:
        genomes[chrom] = "".join(buf).upper()
    print(f"  Genome loaded: {len(genomes)} chroms [{time.time()-t0:.1f}s]")

    seqs = []
    for _, row in df.iterrows():
        ch = row["Chromosome"]
        seq = genomes.get(ch, "")[int(row["Start"]):int(row["Stop"])]
        if seq and _row_strand(row) == "-":
            seq = _revcomp(seq)
        seqs.append(seq)
    return seqs


def extract_sequences(genome_fa, df):
    """Extract sequences from genome FASTA for all rows in df."""
    if _PYSAM:
        return extract_sequences_pysam(genome_fa, df)
    else:
        print("  NOTE: pysam not available, using in-memory FASTA extraction")
        return extract_sequences_fasta(genome_fa, df)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Named flags (--family / --build) override positional args when provided.
    if args.family_flag:
        args.family = args.family_flag
    if args.build_flag:
        args.build = args.build_flag
    if not args.build:
        args.build = "hg38"

    # ── Download mode ───────────────────────────────────────────────────────
    if args.download:
        if args.source == "dfam":
            download_dfam(args.download, args.rmsk_dir, url_override=args.dfam_url)
        else:
            download_rmsk(args.download, args.rmsk_dir)
        return

    # ── List / search mode ──────────────────────────────────────────────────
    if args.list_families or args.search:
        build = args.build
        if args.source == "dfam":
            list_families_dfam(None, search=args.search)
        else:
            rmsk_path = get_rmsk_path(build, args.rmsk_dir)
            list_families(rmsk_path, search=args.search)
        return

    # ── Normal mode ─────────────────────────────────────────────────────────
    if not args.family:
        print("Usage: python te_prep.py <family> <build> [options]")
        print("       python te_prep.py --download hg38")
        print("       python te_prep.py --download hg38 --source dfam")
        print("       python te_prep.py --list-families --build hg38")
        sys.exit(1)

    family = args.family
    build  = args.build
    source = args.source

    std_chroms = None
    if not args.no_filter_chroms:
        std_chroms = STD_CHROMS_HUMAN if build.startswith("hg") else STD_CHROMS_MOUSE

    # ── Count-only mode ─────────────────────────────────────────────────────
    if args.count_only:
        print(
            f"[COUNT-ONLY] family={family}  build={build}  source={source}"
            f"  rmsk_dir={args.rmsk_dir}  chroms={'std' if std_chroms else 'all'}",
            flush=True,
        )
        if source == "dfam":
            t0 = time.time()
            count = count_family_dfam(None, family, build=build, std_chroms=std_chroms)
        else:
            rmsk_candidate = Path(args.rmsk_dir) / f"rmsk_{build}.txt.gz"
            if not rmsk_candidate.exists():
                print(f"  rmsk_{build}.txt.gz not found — downloading to {args.rmsk_dir} (~150 MB, one-time)…", flush=True)
                download_rmsk(build, args.rmsk_dir)
            rmsk_path = get_rmsk_path(build, args.rmsk_dir)
            t0 = time.time()
            count = count_family_rmsk(rmsk_path, family, std_chroms)
        print(f"Locus count for '{family}' ({build}, {source}): {count:,}  [{time.time()-t0:.1f}s]")
        return

    # Resolve genome FASTA — optional; UCSC API is the fallback
    genome_fa = (args.genome_fa
                 or GENOME_FA.get(build, "")
                 or os.environ.get(f"TE_{build.upper()}_FA", ""))
    use_ucsc = False
    if genome_fa and not os.path.exists(genome_fa):
        print(f"WARNING: Genome FASTA not found at {genome_fa} — falling back to UCSC API")
        genome_fa = ""
    if not genome_fa:
        print(f"  No local genome FASTA — sequences will be fetched from UCSC API (assembly={build})")
        use_ucsc = True

    # Absolute output path: if --base-dir starts with "/" use it directly as
    # BASE_DIR; otherwise treat it as a parent and append the family subdir.
    _base = Path(args.base_dir)
    BASE_DIR = _base if _base.is_absolute() else _base / f"{family}_analysis_results"
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    CLUSTERED_PATH = BASE_DIR / "clustered_data.csv"

    # Annotation path (only for rmsk; dfam uses REST API)
    ann_path = None
    _rmsk_api_fallback = False
    if source == "rmsk":
        try:
            ann_path = get_rmsk_path(build, args.rmsk_dir)
        except RuntimeError as _dl_err:
            print(f"\n  {_dl_err}", flush=True)
            print(f"  → Falling back to UCSC REST API (api.genome.ucsc.edu)...\n", flush=True)
            _rmsk_api_fallback = True

    if _rmsk_api_fallback:
        _input_str = f"UCSC REST API — rmsk track, {build} (hgdownload unreachable)"
    elif ann_path:
        _input_str = str(Path(ann_path).resolve())
    else:
        _input_str = "Dfam REST API"

    print("=" * 60)
    print("TE Family Data Prep")
    print(f"  Family:    {family}")
    print(f"  Build:     {build}")
    print(f"  Source:    {source}{' (REST API fallback)' if _rmsk_api_fallback else ''}")
    print(f"  Input:     {_input_str}")
    print(f"  Genome FA: {genome_fa or '(none — UCSC API fallback)'}")
    print(f"  Output:    {BASE_DIR.resolve()}")
    print(f"  Date:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── Step 1: Parse annotation source for coordinates ─────────────────────
    src_label = "RepeatMasker" if source == "rmsk" else "Dfam"
    print(f"\n{'='*60}\nSTEP 1: Fetch {src_label} coordinates\n{'='*60}")

    if source == "dfam":
        hits = parse_dfam_family(None, family, build=build, std_chroms=std_chroms)
    elif _rmsk_api_fallback:
        hits = fetch_rmsk_from_api(family, build, std_chroms=std_chroms)
    else:
        hits = parse_rmsk_family(ann_path, family, std_chroms)

    if not hits:
        src_hint = "dfam" if source == "dfam" else "rmsk"
        print(f"\nFATAL: No loci for family='{family}' in {build} ({source}).")
        print(f"Name is case-sensitive. To search:  python te_prep.py --search {family.lower()} --build {build} --source {src_hint}")
        sys.exit(1)

    if not _HAS_PANDAS:
        print("FATAL: numpy/pandas not installed — run: pip install numpy pandas", flush=True)
        sys.exit(1)

    df = pd.DataFrame(hits).sort_values(["Chromosome", "Start"]).reset_index(drop=True)
    print(f"  {len(df):,} loci across {df['Chromosome'].nunique()} chromosomes")
    print(f"  Coordinates source: {src_label}")

    if args.max_loci and len(df) > args.max_loci:
        print(f"  Capping to {args.max_loci} (--max-loci)")
        df = df.head(args.max_loci).reset_index(drop=True)

    # ── Step 2: TE_ID ──────────────────────────────────────────────────────
    print(f"\n{'='*60}\nSTEP 2: Build TE identifiers\n{'='*60}")
    df["TE_ID"] = (
        df["Chromosome"] + "|" +
        df["Start"].astype(str) + "|" +
        df["Stop"].astype(str) + "|" +
        df["repName"] + ":" + df["repFamily"] + ":" + df["repClass"] + "|" +
        df["swScore"].astype(str) + "|" +
        df["strand"]
    )
    df["TE_name"] = df["TE_ID"]
    df["chr"]   = df["Chromosome"]
    df["start"] = df["Start"]
    df["stop"]  = df["Stop"]
    print(f"  Example ID: {df['TE_ID'].iloc[0]}")

    # ── Step 3: Sequences ──────────────────────────────────────────────────
    if use_ucsc:
        print(f"\n{'='*60}\nSTEP 3: Fetch sequences from UCSC API\n{'='*60}")
        print(f"  Coordinates sourced from {src_label}; sequences fetched from UCSC (genome={build})")
        seqs, ucsc_urls = fetch_sequences_ucsc(df, assembly=build, n_workers=args.fetch_workers)
        df["Seq"] = seqs
        df["ucsc_url"] = ucsc_urls
        n_all_n = sum(1 for s in seqs if s and set(s.upper()) <= {"N"})
        if n_all_n:
            pct = 100 * n_all_n / len(seqs)
            print(f"  WARNING: {n_all_n}/{len(seqs)} ({pct:.1f}%) sequences are all-N (UCSC fetch failed)")
        if n_all_n == len(seqs):
            sys.exit(
                "FATAL: Every sequence fetch failed.\n"
                "  Check network access to api.genome.ucsc.edu from the compute node,\n"
                "  or provide a local genome FASTA via --genome."
            )
    else:
        print(f"\n{'='*60}\nSTEP 3: Extract sequences from local FASTA\n{'='*60}")
        seqs = extract_sequences(genome_fa, df)
        df["Seq"] = seqs
        df["ucsc_url"] = ""

    before = len(df)
    df = df[df["Seq"].str.len() > 0].reset_index(drop=True)
    if len(df) < before:
        print(f"  Dropped {before - len(df)} empty sequences")

    # ── Step 4: Stats ──────────────────────────────────────────────────────
    print(f"\n{'='*60}\nSTEP 4: Stats\n{'='*60}")
    df["Length"] = df["Seq"].str.len()
    df["GC_Content"] = df["Seq"].apply(
        lambda s: (s.upper().count("G") + s.upper().count("C")) / len(s) if len(s) > 0 else 0
    )
    print(f"  Length: mean={df['Length'].mean():.0f}  "
          f"median={df['Length'].median():.0f}  "
          f"range=[{df['Length'].min()}, {df['Length'].max()}]")
    print(f"  GC:     mean={df['GC_Content'].mean():.3f}")

    # ── Step 5: Save ───────────────────────────────────────────────────────
    print(f"\n{'='*60}\nSTEP 5: Save\n{'='*60}")
    keep = [c for c in ["Chromosome", "Start", "Stop", "chr", "start", "stop",
                         "Seq", "TE_ID", "TE_name", "strand",
                         "repName", "repClass", "repFamily",
                         "swScore", "milliDiv", "Length", "GC_Content", "ucsc_url"]
             if c in df.columns]
    df[keep].to_csv(CLUSTERED_PATH, index=False)
    print(f"  Saved: {CLUSTERED_PATH}")
    print(f"  {len(df):,} loci, {len(keep)} columns")

    print(f"\n{'='*60}\nDONE\n{'='*60}")
    print(f"  {family} ({build}): {len(df):,} loci")
    print(f"\nNext steps:")
    print(f"  python te_enrichment.py {family} {build} --base-dir {args.base_dir}")
    print(f"  python query.py --input {CLUSTERED_PATH} --family {family}")
    if args.notify_email:
        from te_notify import send_completion_email
        send_completion_email(args.notify_email, "te_prep", str(BASE_DIR))


if __name__ == "__main__":
    main()
