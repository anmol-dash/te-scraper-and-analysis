#!/usr/bin/env python3
"""
te_genome.py
Shared genome utilities: GenomeCache for in-memory FASTA access and primer search.
Imported by query.py, te_primers.py, and te_prep.py.
"""

import os
import sys
from pathlib import Path


def progress_print(msg, newline=True):
    import datetime
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    if newline:
        print(f"[{ts}] {msg}", flush=True)
    else:
        print(f"[{ts}] {msg}", end="", flush=True)


def reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]


def _search_single_chrom(primer, rc, plen, chrom, seq, max_hits=0, current_count=0):
    """Search a single chromosome sequence for primer (forward + RC) hits."""
    hits = []
    count = current_count
    seq = seq.upper()
    idx = seq.find(primer)
    while idx != -1:
        hits.append((chrom, idx + 1, idx + plen, "+"))
        count += 1
        if max_hits and count >= max_hits:
            return hits, count
        idx = seq.find(primer, idx + 1)
    idx = seq.find(rc)
    while idx != -1:
        hits.append((chrom, idx + 1, idx + plen, "-"))
        count += 1
        if max_hits and count >= max_hits:
            return hits, count
        idx = seq.find(rc, idx + 1)
    return hits, count


class GenomeCache:
    """Load a reference genome FASTA into memory once, reuse for all operations.

    Supports:
    - Local sequence extraction (replaces UCSC API calls)
    - Fast exact-match primer search (no re-reading the file)
    - Pickle cache for fast subsequent loads

    Usage:
        gc = GenomeCache("/path/to/genome.fa", cache_dir="/path/to/cache")
        gc.load()
        seq = gc.extract_sequence("chr1", 1000, 2000)
        hits = gc.search_primer("ATCGATCG")
    """

    MAX_GENOME_HITS = 10_000  # stop searching once this many hits are found

    def __init__(self, fasta_path, cache_dir=None):
        self.genomes = {}
        self.fasta_path = Path(fasta_path) if fasta_path else None
        self._loaded = False
        self._total_bp = 0

        if self.fasta_path and self.fasta_path.exists():
            cache_base = Path(cache_dir) if cache_dir else Path(".")
            cache_base.mkdir(parents=True, exist_ok=True)
            self.cache_path = cache_base / f"{self.fasta_path.stem}.genome_cache.pkl"
        else:
            self.cache_path = None

    # ------------------------------------------------------------------
    def load(self):
        """Load genome into memory (pickle cache speeds up subsequent runs)."""
        if self._loaded:
            return

        if not self.fasta_path or not self.fasta_path.exists():
            progress_print(f"  WARNING: Genome file not found: {self.fasta_path}")
            progress_print("  Genome-dependent features will be skipped or use UCSC API fallback")
            self._loaded = True
            return

        import pickle
        import time as _t

        # Try pickle cache
        if self.cache_path and self.cache_path.exists():
            try:
                if self.cache_path.stat().st_mtime > self.fasta_path.stat().st_mtime:
                    progress_print(f"  Loading genome from cache: {self.cache_path.name}")
                    t0 = _t.time()
                    with open(self.cache_path, "rb") as f:
                        self.genomes = pickle.load(f)
                    self._total_bp = sum(len(s) for s in self.genomes.values())
                    progress_print(
                        f"  Loaded {len(self.genomes)} chromosomes "
                        f"({self._total_bp:,} bp) from cache in {_t.time()-t0:.1f}s"
                    )
                    self._loaded = True
                    return
            except Exception as e:
                progress_print(f"  Cache load failed ({e}), falling back to FASTA parsing")

        # Parse FASTA
        progress_print(f"  Loading genome from FASTA: {self.fasta_path.name}")
        progress_print("  First-time load — will cache as pickle for future runs")
        t0 = _t.time()
        chrom = None
        seq_buf = []
        chrom_count = 0

        with open(self.fasta_path) as fh:
            for line in fh:
                if line.startswith(">"):
                    if chrom is not None:
                        self.genomes[chrom] = "".join(seq_buf).upper()
                        self._total_bp += len(self.genomes[chrom])
                        chrom_count += 1
                        if chrom_count % 10 == 0:
                            progress_print(
                                f"    Loaded {chrom_count} chromosomes ({self._total_bp:,} bp)..."
                            )
                    chrom = line[1:].strip().split()[0]
                    seq_buf = []
                else:
                    seq_buf.append(line.strip())
            if chrom is not None:
                self.genomes[chrom] = "".join(seq_buf).upper()
                self._total_bp += len(self.genomes[chrom])

        elapsed = _t.time() - t0
        progress_print(
            f"  Loaded {len(self.genomes)} chromosomes ({self._total_bp:,} bp) in {elapsed:.1f}s"
        )

        # Write pickle cache
        if self.cache_path:
            try:
                progress_print("  Caching genome as pickle for faster future loads...")
                t0 = _t.time()
                with open(self.cache_path, "wb") as f:
                    pickle.dump(self.genomes, f, protocol=pickle.HIGHEST_PROTOCOL)
                cache_mb = self.cache_path.stat().st_size / 1024 / 1024
                progress_print(
                    f"  Cached to {self.cache_path.name} ({cache_mb:.0f} MB) in {_t.time()-t0:.1f}s"
                )
            except Exception as e:
                progress_print(f"  WARNING: Could not write cache: {e}")

        self._loaded = True

    # ------------------------------------------------------------------
    def extract_sequence(self, chrom, start, stop):
        """Return uppercase sequence string for chrom:start-stop (0-based), or None."""
        seq = self.genomes.get(chrom)
        if seq is None:
            return None
        if start < 0 or stop < 0 or stop <= start or stop > len(seq):
            return None
        return seq[start:stop]

    # ------------------------------------------------------------------
    def search_primer(self, primer, max_hits=None):
        """Search all chromosomes for exact forward + RC matches of *primer*.

        Returns list of (chrom, start, stop, strand) tuples.
        Stops early when max_hits is reached (primer deemed non-specific).
        """
        if max_hits is None:
            max_hits = self.MAX_GENOME_HITS
        primer = primer.upper()
        rc = reverse_complement(primer)
        plen = len(primer)
        hits = []
        total = 0
        for i, (chrom, seq) in enumerate(self.genomes.items()):
            chrom_hits, total = _search_single_chrom(
                primer, rc, plen, chrom, seq, max_hits=max_hits, current_count=total
            )
            hits.extend(chrom_hits)
            if chrom_hits:
                progress_print(f"    {chrom}: {len(chrom_hits)} hits (running total: {total:,})")
            if max_hits and total >= max_hits:
                remaining = len(self.genomes) - i - 1
                progress_print(
                    f"    HIT CAP ({max_hits:,}) reached — skipping {remaining} remaining chromosomes"
                )
                break
        return hits

    # ------------------------------------------------------------------
    @property
    def is_loaded(self):
        return self._loaded and bool(self.genomes)
