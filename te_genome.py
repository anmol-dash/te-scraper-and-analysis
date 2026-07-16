#!/usr/bin/env python3
"""
te_genome.py
Shared genome utilities: GenomeCache for in-memory FASTA access and primer search.
Imported by query.py, te_primers.py, and te_prep.py.

Speed path:
  - te_fast (Cython) is used when compiled; falls back to pure Python otherwise.
  - search_primer_parallel() uses ProcessPoolExecutor for per-chromosome tasks.
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


# ── Cython fast-path (optional) ─────────────────────────────────────────────
try:
    import te_fast as _tf
    _HAS_CYTHON = True
except ImportError:
    _HAS_CYTHON = False


def reverse_complement(seq: str) -> str:
    if _HAS_CYTHON:
        return _tf.reverse_complement(seq)
    return seq.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]


# ── Module-level worker function (must be picklable) ────────────────────────

def _search_single_chrom(primer, rc, plen, chrom, seq, max_hits=0, current_count=0):
    """Search one chromosome for exact primer matches (forward + RC)."""
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


def _parallel_chrom_worker(args):
    """Top-level picklable worker for ProcessPoolExecutor."""
    primer, rc, plen, chrom, seq, max_hits = args
    hits, total = _search_single_chrom(primer, rc, plen, chrom, seq, max_hits, 0)
    return hits, total


# ── GenomeCache ─────────────────────────────────────────────────────────────

class GenomeCache:
    """Load a reference genome FASTA into memory once, reuse for all operations.

    Supports:
    - Local sequence extraction (replaces UCSC API calls)
    - Fast exact-match primer search, optionally parallelized across chromosomes
    - Pickle cache for fast subsequent loads

    Usage:
        gc = GenomeCache("/path/to/genome.fa", cache_dir="/path/to/cache")
        gc.load()
        seq = gc.extract_sequence("chr1", 1000, 2000)
        hits = gc.search_primer("ATCGATCG")
        hits = gc.search_primer("ATCGATCG", parallel=True)   # parallel search
    """

    MAX_GENOME_HITS = 10_000

    def __init__(self, fasta_path, cache_dir=None):
        """Create an unloaded cache for the given FASTA (call load() to populate).

        cache_dir is where the parsed-genome pickle is read from / written to so
        subsequent runs skip re-parsing the FASTA.
        """
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

    def extract_sequence(self, chrom, start, stop):
        """Return uppercase sequence string for chrom:start-stop (0-based), or None."""
        seq = self.genomes.get(chrom)
        if seq is None:
            return None
        if start < 0 or stop < 0 or stop <= start or stop > len(seq):
            return None
        return seq[start:stop]

    def search_primer(self, primer, max_hits=None, parallel=False, n_workers=None):
        """Search all chromosomes for exact forward + RC matches of *primer*.

        Args:
            primer   : primer sequence (str)
            max_hits : stop early when this many hits found (default MAX_GENOME_HITS)
            parallel : use ProcessPoolExecutor for per-chromosome parallelism
            n_workers: worker count for parallel mode (default: cpu_count, capped at 8)

        Returns list of (chrom, start, stop, strand) tuples.
        """
        if max_hits is None:
            max_hits = self.MAX_GENOME_HITS

        primer = primer.upper()
        rc = reverse_complement(primer)
        plen = len(primer)

        if parallel and len(self.genomes) > 1:
            return self._search_primer_parallel(primer, rc, plen, max_hits, n_workers)

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

    def _search_primer_parallel(self, primer, rc, plen, max_hits, n_workers):
        """Per-chromosome parallel search using ProcessPoolExecutor.

        Each chromosome is sent to a worker process as a separate task.
        Serialization overhead is real (~0.5s per large chromosome) but the
        per-chromosome parallelism provides ~N_WORKERS× speedup on the search itself.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing as mp

        if n_workers is None:
            n_workers = min(mp.cpu_count() or 4, 8, len(self.genomes))

        tasks = [
            (primer, rc, plen, chrom, seq, max_hits)
            for chrom, seq in self.genomes.items()
        ]

        all_hits = []
        total = 0

        with ProcessPoolExecutor(max_workers=n_workers) as exe:
            futures = {exe.submit(_parallel_chrom_worker, t): t[3] for t in tasks}
            for fut in as_completed(futures):
                chrom_name = futures[fut]
                try:
                    chrom_hits, chrom_total = fut.result()
                    all_hits.extend(chrom_hits)
                    total += chrom_total
                    if chrom_hits:
                        progress_print(
                            f"    {chrom_name}: {chrom_total} hits (running total: {total:,})"
                        )
                    if max_hits and total >= max_hits:
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        break
                except Exception as e:
                    progress_print(f"    WARNING: worker failed for {chrom_name}: {e}")

        return all_hits[:max_hits] if max_hits else all_hits

    @property
    def is_loaded(self):
        return self._loaded and bool(self.genomes)
