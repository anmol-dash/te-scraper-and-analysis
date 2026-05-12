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
import sys
import time
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

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
    p.add_argument("build",  nargs="?", default="hg38",
                   help="Genome build: hg38, hg19, mm10, mm39 (default: hg38)")
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

    import subprocess

    # Try system download tools; skip gracefully when the binary is missing
    for tool in [
        ["curl", "-L", "--progress-bar", "-o", str(outpath), url],
        ["wget", "-q", "--show-progress", "-O", str(outpath), url],
    ]:
        try:
            ret = subprocess.run(tool)
            if ret.returncode == 0:
                size_mb = outpath.stat().st_size / 1e6
                print(f"Done: {outpath} ({size_mb:.0f} MB)")
                return str(outpath)
        except FileNotFoundError:
            continue  # binary not available on this system

    # Pure-Python fallback using urllib (no external tools required)
    print("  curl/wget not found — using Python urllib (no progress bar)...")
    import urllib.request
    try:
        urllib.request.urlretrieve(url, str(outpath))
        size_mb = outpath.stat().st_size / 1e6
        print(f"Done: {outpath} ({size_mb:.0f} MB)")
        return str(outpath)
    except Exception as e:
        if outpath.exists():
            outpath.unlink()
        print(f"FATAL: Download failed: {e}")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# PARSE rmsk
# ═══════════════════════════════════════════════════════════════════════════

def get_rmsk_path(build, rmsk_dir):
    path = Path(rmsk_dir) / f"rmsk_{build}.txt.gz"
    if not path.exists():
        print(f"FATAL: rmsk file not found: {path}")
        print(f"Run first:  python te_prep.py --download {build}")
        sys.exit(1)
    return str(path)


def parse_rmsk_family(rmsk_path, family, std_chroms=None):
    """Stream rmsk.txt.gz and return list of dicts for rows matching repName."""
    hits = []
    t0 = time.time()
    print(f"  Scanning {Path(rmsk_path).name} for repName='{family}'...")

    with gzip.open(rmsk_path, "rt") as f:
        for n_lines, line in enumerate(f, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 13:
                continue
            if fields[COL_REPNAME] != family:
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
        import re
        m = re.search(r"(?:^|[|:,\s])([+-])$", text)
        if m:
            return m.group(1)
    return "+"


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

    # ── Download mode ───────────────────────────────────────────────────────
    if args.download:
        download_rmsk(args.download, args.rmsk_dir)
        return

    # ── List / search mode ──────────────────────────────────────────────────
    if args.list_families or args.search:
        build = args.build or "hg38"
        rmsk_path = get_rmsk_path(build, args.rmsk_dir)
        list_families(rmsk_path, search=args.search)
        return

    # ── Normal mode ─────────────────────────────────────────────────────────
    if not args.family:
        print("Usage: python te_prep.py <family> <build> [options]")
        print("       python te_prep.py --download hg38")
        print("       python te_prep.py --list-families --build hg38")
        sys.exit(1)

    family = args.family
    build  = args.build

    # Resolve genome FASTA
    genome_fa = (args.genome_fa
                 or GENOME_FA.get(build, "")
                 or os.environ.get(f"TE_{build.upper()}_FA", ""))
    if not genome_fa:
        print(f"FATAL: Genome FASTA not specified for {build}.")
        print(f"  Use --genome-fa /path/to/{build}.fa")
        print(f"  Or set environment variable TE_{build.upper()}_FA=/path/to/{build}.fa")
        sys.exit(1)
    if not os.path.exists(genome_fa):
        print(f"FATAL: Genome FASTA not found: {genome_fa}")
        sys.exit(1)

    BASE_DIR = Path(args.base_dir) / f"{family}_analysis_results"
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    CLUSTERED_PATH = BASE_DIR / "clustered_data.csv"

    print("=" * 60)
    print("TE Family Data Prep")
    print(f"  Family:    {family}")
    print(f"  Build:     {build}")
    print(f"  Genome FA: {genome_fa}")
    print(f"  Output:    {BASE_DIR}")
    print(f"  Date:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    rmsk_path = get_rmsk_path(build, args.rmsk_dir)

    # ── Step 1: Parse rmsk ─────────────────────────────────────────────────
    print(f"\n{'='*60}\nSTEP 1: Parse RepeatMasker table\n{'='*60}")
    std_chroms = None
    if not args.no_filter_chroms:
        std_chroms = STD_CHROMS_HUMAN if build.startswith("hg") else STD_CHROMS_MOUSE

    hits = parse_rmsk_family(rmsk_path, family, std_chroms)

    if not hits:
        print(f"\nFATAL: No loci for repName='{family}' in {build}.")
        print(f"repName is case-sensitive. To search:  python te_prep.py --search {family.lower()} --build {build}")
        sys.exit(1)

    df = pd.DataFrame(hits).sort_values(["Chromosome", "Start"]).reset_index(drop=True)
    print(f"  {len(df):,} loci across {df['Chromosome'].nunique()} chromosomes")

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
    # Alias so query.py recognizes it
    df["TE_name"] = df["TE_ID"]
    df["chr"]   = df["Chromosome"]
    df["start"] = df["Start"]
    df["stop"]  = df["Stop"]
    print(f"  Example ID: {df['TE_ID'].iloc[0]}")

    # ── Step 3: Sequences ──────────────────────────────────────────────────
    print(f"\n{'='*60}\nSTEP 3: Extract sequences\n{'='*60}")
    seqs = extract_sequences(genome_fa, df)
    df["Seq"] = seqs
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
                         "swScore", "milliDiv", "Length", "GC_Content"]
             if c in df.columns]
    df[keep].to_csv(CLUSTERED_PATH, index=False)
    print(f"  Saved: {CLUSTERED_PATH}")
    print(f"  {len(df):,} loci, {len(keep)} columns")

    print(f"\n{'='*60}\nDONE\n{'='*60}")
    print(f"  {family} ({build}): {len(df):,} loci")
    print(f"\nNext steps:")
    print(f"  python te_enrichment.py {family} {build} --base-dir {args.base_dir}")
    print(f"  python query.py --input {CLUSTERED_PATH} --family {family}")


if __name__ == "__main__":
    main()
