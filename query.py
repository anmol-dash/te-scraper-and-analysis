#!/usr/bin/env python3
"""
TE Analysis Pipeline - Transposable Element clustering, alignment, and primer design

Usage:
    python query.py                    # Run with pre-loaded df (interactive/notebook)
    python query.py --test             # Run with mock test data (standalone)
    python query.py --input FILE.csv   # Run with input CSV file
"""
import os
import sys
import argparse
import json
import requests
import traceback
import random
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from bs4 import BeautifulSoup
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

# Fix matplotlib backend BEFORE any plotting imports
os.environ['MPLBACKEND'] = 'Agg'

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import hdbscan

# ==================== HELPER FUNCTIONS (DO NOT MOVE) ====================
def debug_print(msg):
    """Print debug message if DEBUG mode is enabled"""
    if globals().get('DEBUG', False):
        print(f"[DEBUG] {msg}", flush=True)

def progress_print(msg, newline=True):
    """Print progress message with timestamp"""
    import datetime
    import sys
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    if newline:
        print(f"[{timestamp}] {msg}", flush=True)
    else:
        print(f"[{timestamp}] {msg}", end="", flush=True)
    sys.stdout.flush()

def progress_bar(current, total, prefix="Progress", suffix="", length=40):
    """Display a progress bar"""
    percent = current / total if total > 0 else 0
    filled = int(length * percent)
    bar = "█" * filled + "░" * (length - filled)
    print(f"\r{prefix} |{bar}| {current}/{total} ({percent*100:.1f}%) {suffix}", end="", flush=True)
    if current >= total:
        print()  # New line when complete

def log_error(stage, error, context=None):
    """Log detailed error information for debugging."""
    import datetime
    import traceback as tb

    print("\n" + "=" * 70, flush=True)
    print(f"ERROR in stage: {stage}", flush=True)
    print("=" * 70, flush=True)
    print(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Error type: {type(error).__name__}", flush=True)
    print(f"Error message: {str(error)}", flush=True)

    if context:
        print(f"\nContext:", flush=True)
        for key, value in context.items():
            # Truncate long values
            str_val = str(value)
            if len(str_val) > 200:
                str_val = str_val[:200] + "..."
            print(f"  {key}: {str_val}", flush=True)

    print(f"\nFull traceback:", flush=True)
    print("-" * 70, flush=True)
    tb.print_exc()
    print("-" * 70, flush=True)
    print("=" * 70 + "\n", flush=True)

    # Also write to a dedicated error log file if OUT_DIR exists
    try:
        if 'OUT_DIR' in globals() and OUT_DIR.exists():
            error_log_path = OUT_DIR / "pipeline_errors.log"
            with open(error_log_path, "a") as f:
                f.write("\n" + "=" * 70 + "\n")
                f.write(f"ERROR in stage: {stage}\n")
                f.write(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Error type: {type(error).__name__}\n")
                f.write(f"Error message: {str(error)}\n")
                if context:
                    f.write(f"\nContext:\n")
                    for key, value in context.items():
                        str_val = str(value)
                        if len(str_val) > 500:
                            str_val = str_val[:500] + "..."
                        f.write(f"  {key}: {str_val}\n")
                f.write(f"\nFull traceback:\n")
                f.write(tb.format_exc())
                f.write("=" * 70 + "\n")
            print(f"Error details also written to: {error_log_path}", flush=True)
    except Exception:
        pass  # Don't fail if we can't write the error log

def stage_timer(stage_name):
    """Context manager to time pipeline stages."""
    import time as _time
    class _Timer:
        def __init__(self):
            self.start = None
            self.elapsed = 0
        def __enter__(self):
            self.start = _time.time()
            progress_print(f">>> STAGE START: {stage_name}")
            return self
        def __exit__(self, *args):
            self.elapsed = _time.time() - self.start
            progress_print(f"<<< STAGE DONE: {stage_name} ({self.elapsed:.1f}s)")
    return _Timer()
# ========================================================================

# ==================== GENOME CACHE ====================
class GenomeCache:
    """Load reference genome into memory once, reuse for all operations.

    Avoids re-reading the multi-GB FASTA for each primer search.
    Caches parsed genome as pickle for faster loading on subsequent runs.
    Also used for local sequence extraction (replacing UCSC API calls).
    """

    def __init__(self, fasta_path, cache_dir=None):
        self.genomes = {}  # {chrom: sequence_string}
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
        """Load genome into memory, using pickle cache if available."""
        if self._loaded:
            return

        if not self.fasta_path or not self.fasta_path.exists():
            progress_print(f"  WARNING: Genome file not found: {self.fasta_path}")
            progress_print(f"  Genome-dependent features will be skipped or use UCSC API fallback")
            self._loaded = True
            return

        import pickle
        import time as _time

        # Try loading from pickle cache first (much faster than parsing FASTA)
        if self.cache_path and self.cache_path.exists():
            try:
                cache_mtime = self.cache_path.stat().st_mtime
                fasta_mtime = self.fasta_path.stat().st_mtime
                if cache_mtime > fasta_mtime:
                    progress_print(f"  Loading genome from cache: {self.cache_path.name}")
                    t0 = _time.time()
                    with open(self.cache_path, 'rb') as f:
                        self.genomes = pickle.load(f)
                    self._total_bp = sum(len(s) for s in self.genomes.values())
                    elapsed = _time.time() - t0
                    progress_print(f"  Loaded {len(self.genomes)} chromosomes ({self._total_bp:,} bp) from cache in {elapsed:.1f}s")
                    self._loaded = True
                    return
            except Exception as e:
                progress_print(f"  Cache load failed ({e}), falling back to FASTA parsing")

        # Parse from FASTA (first run or cache miss)
        progress_print(f"  Loading genome from FASTA: {self.fasta_path.name}")
        progress_print(f"  First-time load — will cache as pickle for future runs")
        t0 = _time.time()
        chrom = None
        seq_buf = []
        chrom_count = 0

        with open(self.fasta_path) as fh:
            for line in fh:
                if line.startswith('>'):
                    if chrom is not None:
                        self.genomes[chrom] = ''.join(seq_buf).upper()
                        self._total_bp += len(self.genomes[chrom])
                        chrom_count += 1
                        if chrom_count % 10 == 0:
                            progress_print(f"    Loaded {chrom_count} chromosomes ({self._total_bp:,} bp)...")
                    chrom = line[1:].strip().split()[0]
                    seq_buf = []
                else:
                    seq_buf.append(line.strip())
            if chrom is not None:
                self.genomes[chrom] = ''.join(seq_buf).upper()
                self._total_bp += len(self.genomes[chrom])

        elapsed = _time.time() - t0
        progress_print(f"  Loaded {len(self.genomes)} chromosomes ({self._total_bp:,} bp) in {elapsed:.1f}s")

        # Save pickle cache for future runs
        if self.cache_path:
            try:
                progress_print(f"  Caching genome as pickle for faster future loads...")
                t0 = _time.time()
                with open(self.cache_path, 'wb') as f:
                    pickle.dump(self.genomes, f, protocol=pickle.HIGHEST_PROTOCOL)
                cache_mb = self.cache_path.stat().st_size / 1024 / 1024
                elapsed = _time.time() - t0
                progress_print(f"  Cached to {self.cache_path.name} ({cache_mb:.0f} MB) in {elapsed:.1f}s")
            except Exception as e:
                progress_print(f"  WARNING: Could not write cache: {e}")

        self._loaded = True

    def extract_sequence(self, chrom, start, stop):
        """Extract a sequence from the in-memory genome. Returns uppercase string or None."""
        chrom_seq = self.genomes.get(chrom)
        if chrom_seq is None:
            return None
        if start < 0 or stop < 0 or stop <= start:
            return None
        if stop > len(chrom_seq):
            return None
        return chrom_seq[start:stop]

    def search_primer(self, primer, max_hits=None):
        """Search all loaded chromosomes for exact matches of primer and its RC.
        Returns list of (chrom, start, stop, strand) tuples.
        Stops early if max_hits is reached (primer is non-specific).
        """
        if max_hits is None:
            max_hits = globals().get('MAX_GENOME_HITS', 10000)
        primer = primer.upper()
        rc = reverse_complement(primer)
        plen = len(primer)
        hits = []
        total_count = 0
        capped = False
        for i, (chrom, seq) in enumerate(self.genomes.items()):
            chrom_hits, total_count = _search_single_chrom(
                primer, rc, plen, chrom, seq,
                max_hits=max_hits, current_count=total_count
            )
            hits.extend(chrom_hits)
            if chrom_hits:
                progress_print(f"    {chrom}: {len(chrom_hits)} hits (running total: {total_count:,})")
            if max_hits and total_count >= max_hits:
                remaining = len(self.genomes) - i - 1
                progress_print(f"    HIT CAP ({max_hits:,}) reached — skipping {remaining} remaining chromosomes")
                progress_print(f"    (Primer is non-specific; exact count not needed)")
                capped = True
                break
        return hits

    @property
    def is_loaded(self):
        return self._loaded and bool(self.genomes)

# Global genome cache — initialized after config, reused across all stages
genome_cache = None
# ========================================================================

# ==================== CONFIG ====================
FAMILY_NAME = "HERVK9"  # CHANGE THIS TO YOUR TARGET FAMILY
HG38_FA = "/project/amodzlab/index/human/hg38/hg38.fa"
BASE_OUT_DIR = Path("collab_rna")
K = 18  # K-mer size for clustering analysis
PRIMER_K = 18  # K-mer size for primer design (can be different, e.g., 8, 12, 18)
TOP_N_GLOBAL = 8
TOP_N_CLUSTER = 5
TOP_N_FORWARD_PRIMERS = 3  # Number of top sequences to generate forward primers from
MIN_SEQUENCES_FOR_CLUSTERING = 10  # Minimum sequences required for clustering
DEBUG = False  # Set to True for verbose debugging output
# ================================================

def create_test_data():
    """Create mock test data for standalone testing"""
    import random
    random.seed(42)
    np.random.seed(42)

    def random_seq(length, gc_bias=0.5):
        bases = []
        for _ in range(length):
            if random.random() < gc_bias:
                bases.append(random.choice(['G', 'C']))
            else:
                bases.append(random.choice(['A', 'T']))
        return ''.join(bases)

    def mutate_seq(seq, mutation_rate=0.1):
        bases = list(seq)
        for i in range(len(bases)):
            if random.random() < mutation_rate:
                bases[i] = random.choice(['A', 'C', 'G', 'T'])
        return ''.join(bases)

    base_seq1 = random_seq(500, gc_bias=0.45)
    base_seq2 = random_seq(480, gc_bias=0.55)

    test_data = []

    # Cluster 1: 8 sequences
    for i in range(8):
        seq = mutate_seq(base_seq1, mutation_rate=0.05 + i*0.01)
        test_data.append({
            'chr': f'chr{(i % 5) + 1}',
            'start': 1000000 + i * 10000,
            'stop': 1000000 + i * 10000 + len(seq),
            'TE_name': f'{FAMILY_NAME}_element_{i+1}',
            'family': FAMILY_NAME,
            'strand': '+' if i % 2 == 0 else '-',
            'Seq': seq,
            'A1_siCTRL_r1': np.random.uniform(0, 50),
            'A1_siCTRL_r2': np.random.uniform(0, 50),
            'A1_siKD_r1': np.random.uniform(0, 30),
            'A1_siKD_r2': np.random.uniform(0, 30),
        })

    # Cluster 2: 7 sequences
    for i in range(7):
        seq = mutate_seq(base_seq2, mutation_rate=0.05 + i*0.01)
        test_data.append({
            'chr': f'chr{(i % 5) + 6}',
            'start': 2000000 + i * 10000,
            'stop': 2000000 + i * 10000 + len(seq),
            'TE_name': f'{FAMILY_NAME}_element_{i+9}',
            'family': FAMILY_NAME,
            'strand': '+' if i % 2 == 0 else '-',
            'Seq': seq,
            'A1_siCTRL_r1': np.random.uniform(20, 100),
            'A1_siCTRL_r2': np.random.uniform(20, 100),
            'A1_siKD_r1': np.random.uniform(10, 50),
            'A1_siKD_r2': np.random.uniform(10, 50),
        })

    return pd.DataFrame(test_data)

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='TE Analysis Pipeline')
    parser.add_argument('--test', action='store_true', help='Run with mock test data')
    parser.add_argument('--input', type=str, help='Input CSV file with TE data')
    parser.add_argument('--family', type=str, default=FAMILY_NAME, help='TE family name to analyze')
    parser.add_argument('--output', type=str, default=str(BASE_OUT_DIR), help='Output directory')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--skip-genome-search', action='store_true', help='Skip genome-wide primer search')
    parser.add_argument('--primer-timeout', type=int, default=120,
                        help='Timeout in seconds for each primer genome search (default: 120)')
    return parser.parse_args() if len(sys.argv) > 1 else argparse.Namespace(
        test=False, input=None, family=FAMILY_NAME, output=str(BASE_OUT_DIR), debug=False,
        skip_genome_search=False, primer_timeout=120
    )

# Only parse args when running as script
if __name__ == "__main__" or len(sys.argv) > 1:
    args = parse_args()
    if args.test:
        FAMILY_NAME = "TEST_TE"
        BASE_OUT_DIR = Path("test_output")
        DEBUG = True
        print("\n" + "="*60)
        print("RUNNING IN TEST MODE WITH MOCK DATA")
        print("="*60)
    else:
        FAMILY_NAME = args.family
        BASE_OUT_DIR = Path(args.output)
        DEBUG = args.debug
# ================================================

# Setup output directory with improved structure
OUT_DIR = BASE_OUT_DIR / FAMILY_NAME.lower()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Create organized subdirectories
DIRS = {
    'data': OUT_DIR / '01_data',
    'stats': OUT_DIR / '02_statistics',
    'clustering': OUT_DIR / '03_clustering',
    'alignments': OUT_DIR / '04_alignments',
    'consensus': OUT_DIR / '05_consensus',
    'primers': OUT_DIR / '06_primers',
    'visualizations': OUT_DIR / '07_visualizations'
}

for dir_path in DIRS.values():
    dir_path.mkdir(parents=True, exist_ok=True)

print(f"Output directory: {OUT_DIR.resolve()}")
print(f"Analyzing family: {FAMILY_NAME}")
print(f"\nOrganized directory structure:")
for name, path in DIRS.items():
    print(f"  {name:15s}: {path.name}")

# ==================== LOAD REFERENCE GENOME ====================
print("\n=== LOADING REFERENCE GENOME ===")
progress_print(f"Genome path: {HG38_FA}")
genome_cache = GenomeCache(HG38_FA, cache_dir=str(OUT_DIR))
genome_cache.load()
if genome_cache.is_loaded:
    progress_print("Genome loaded — local sequence extraction and fast primer searches enabled")
else:
    progress_print("Genome not available — will use UCSC API and skip genome primer searches")

# ==================== PIPELINE TIMING ====================
import time as _pipeline_time
_pipeline_start = _pipeline_time.time()
_stage_times = {}

def _record_stage(name, elapsed):
    _stage_times[name] = elapsed
    progress_print(f"  [TIMING] {name}: {elapsed:.1f}s")

# ==================== LOAD AND FILTER DATA ====================
print("\n=== LOADING DATA ===")

try:
    # Handle different data sources
    if __name__ == "__main__" and len(sys.argv) > 1:
        args = parse_args()
        if args.test:
            print("Creating mock test data...")
            df = create_test_data()
            print(f"Created {len(df)} mock sequences")
        elif args.input:
            print(f"Loading data from {args.input}...")
            if not Path(args.input).exists():
                raise FileNotFoundError(f"Input file not found: {args.input}")
            df = pd.read_csv(args.input)
            print(f"Loaded {len(df)} rows from input file")
            print(f"Columns found: {list(df.columns)}")
        else:
            # Check if df exists in global scope (notebook/interactive mode)
            if 'df' not in dir():
                print("ERROR: No data source specified.")
                print("Usage:")
                print("  python query.py --test              # Run with mock test data")
                print("  python query.py --input FILE.csv   # Run with input CSV file")
                print("  Or ensure 'df' is loaded in interactive mode")
                sys.exit(1)
    else:
        # Interactive/notebook mode - df should be pre-loaded
        if 'df' not in dir():
            print("WARNING: 'df' not found. Creating test data for demonstration.")
            df = create_test_data()
            FAMILY_NAME = "TEST_TE"
            BASE_OUT_DIR = Path("test_output")
            DEBUG = True

    # Validate required columns
    required_cols = ['TE_name']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {list(df.columns)}")

    # Filter for target family
    print(f"Filtering for family: {FAMILY_NAME}")
    df_family = df[df['TE_name'].str.contains(FAMILY_NAME, case=False, na=False)].copy()
    df_family = df_family.reset_index(drop=True)
    print(f"Found {len(df_family)} instances of {FAMILY_NAME}")

    if len(df_family) == 0:
        print(f"ERROR: No instances found for family {FAMILY_NAME}")
        print(f"Available TE names (first 20): {df['TE_name'].unique()[:20].tolist()}")
        print(f"Total unique TE names: {df['TE_name'].nunique()}")
        sys.exit(1)

except Exception as e:
    log_error("LOAD AND FILTER DATA", e, {
        "FAMILY_NAME": FAMILY_NAME,
        "input_file": args.input if 'args' in dir() and hasattr(args, 'input') else "N/A",
        "df_shape": df.shape if 'df' in dir() else "N/A",
        "df_columns": list(df.columns) if 'df' in dir() else "N/A"
    })
    sys.exit(1)

# ==================== FETCH SEQUENCES ====================
_stage_t0 = _pipeline_time.time()
print("\n=== FETCHING SEQUENCES ===")

try:
    # Check if sequences are already provided (e.g., test mode or pre-processed data)
    if 'Seq' in df_family.columns and df_family['Seq'].notna().all():
        print("Sequences already present in data, skipping UCSC fetch")
        # Ensure sequences are uppercase
        df_family['Seq'] = df_family['Seq'].str.upper()
        failed_indices = []
    else:
        # Validate required columns for sequence extraction
        ucsc_required = ['chr', 'start', 'stop']
        ucsc_missing = [c for c in ucsc_required if c not in df_family.columns]
        if ucsc_missing:
            raise ValueError(f"Missing columns required for sequence fetch: {ucsc_missing}. "
                           f"Available: {list(df_family.columns)}")

        seqlist = []
        failed_indices = []
        fetch_errors = []  # Track detailed errors

        # === FAST PATH: Extract from local genome (no network calls) ===
        if genome_cache is not None and genome_cache.is_loaded:
            import time as _time
            t0 = _time.time()
            progress_print(f"Extracting {len(df_family)} sequences from local genome (fast path)...")
            for i in range(len(df_family)):
                if (i+1) % max(1, len(df_family)//20) == 0 or i == len(df_family)-1:
                    progress_bar(i+1, len(df_family), prefix="Extracting sequences")
                try:
                    chrom = df_family['chr'].iloc[i]
                    start = int(df_family['start'].iloc[i])
                    stop = int(df_family['stop'].iloc[i])

                    if start < 0 or stop < 0:
                        raise ValueError(f"Invalid coordinates: start={start}, stop={stop}")
                    if stop <= start:
                        raise ValueError(f"stop ({stop}) must be > start ({start})")

                    seq = genome_cache.extract_sequence(chrom, start, stop)
                    if seq is None:
                        raise ValueError(f"Could not extract {chrom}:{start}-{stop} from genome")
                    seqlist.append(seq)
                except Exception as e:
                    error_msg = f"Row {i}: {chrom}:{start}-{stop} - {type(e).__name__}: {str(e)}"
                    fetch_errors.append(error_msg)
                    seqlist.append("N" * 100)
                    failed_indices.append(i)
            elapsed = _time.time() - t0
            progress_print(f"Extracted {len(df_family) - len(failed_indices)} sequences in {elapsed:.1f}s (local genome)")

        # === SLOW PATH: Fetch from UCSC API (network, ~1-2s per sequence) ===
        else:
            progress_print(f"Fetching {len(df_family)} sequences from UCSC API (slow path — no local genome)...")
            progress_print(f"  TIP: Provide HG38_FA path to skip API calls in future runs")
            for i in range(len(df_family)):
                progress_bar(i+1, len(df_family), prefix="Fetching sequences")
                try:
                    chrom = df_family['chr'].iloc[i]
                    start = int(df_family['start'].iloc[i])
                    stop = int(df_family['stop'].iloc[i])

                    if start < 0 or stop < 0:
                        raise ValueError(f"Invalid coordinates: start={start}, stop={stop}")
                    if stop <= start:
                        raise ValueError(f"stop ({stop}) must be > start ({start})")

                    link = (f"https://api.genome.ucsc.edu/getData/sequence?"
                            f"genome=hg38;chrom={chrom};"
                            f"start={start};end={stop}")
                    r = requests.get(link, timeout=30)
                    r.raise_for_status()

                    res = r.json()

                    if 'error' in res:
                        raise ValueError(f"API error: {res['error']}")

                    if 'dna' in res:
                        seqlist.append(res['dna'].upper())
                    else:
                        raise KeyError(f"'dna' not in response. Keys: {list(res.keys())}")
                except Exception as e:
                    error_msg = f"Row {i}: {chrom}:{start}-{stop} - {type(e).__name__}: {str(e)}"
                    fetch_errors.append(error_msg)
                    seqlist.append("N" * 100)
                    failed_indices.append(i)

        df_family['Seq'] = seqlist
        print(f"\nSuccessfully fetched {len(df_family) - len(failed_indices)} sequences")
        if failed_indices:
            print(f"Failed to fetch {len(failed_indices)} sequences:")
            # Show first 10 errors
            for err in fetch_errors[:10]:
                print(f"  - {err}")
            if len(fetch_errors) > 10:
                print(f"  ... and {len(fetch_errors) - 10} more errors")

            # Save fetch errors to log
            error_log_path = DIRS['data'] / "ucsc_fetch_errors.log"
            with open(error_log_path, "w") as f:
                f.write(f"UCSC Fetch Errors for {FAMILY_NAME}\n")
                f.write("=" * 60 + "\n")
                for err in fetch_errors:
                    f.write(err + "\n")
            print(f"Full error log saved to: {error_log_path}")

    # Save raw data with sequences
    save_path = DIRS['data'] / f"{FAMILY_NAME.lower()}_with_sequences.csv"
    df_family.to_csv(save_path, index=False)
    print(f"Saved sequences to {save_path}")

    _record_stage("Sequence Fetching", _pipeline_time.time() - _stage_t0)

except Exception as e:
    log_error("FETCH SEQUENCES", e, {
        "FAMILY_NAME": FAMILY_NAME,
        "df_family_shape": df_family.shape if 'df_family' in dir() else "N/A",
        "sequences_fetched": len(seqlist) if 'seqlist' in dir() else 0,
        "failed_count": len(failed_indices) if 'failed_indices' in dir() else "N/A"
    })
    sys.exit(1)

# ==================== BASIC STATISTICS ====================
_stage_t0 = _pipeline_time.time()
print("\n=== COMPUTING BASIC STATISTICS ===")

def compute_basic_stats(df, label="", output_file=None):
    """Compute and save comprehensive statistics for sequences and expression data"""
    seqs = df['Seq'].astype(str)
    lengths = seqs.apply(len)
    gc = seqs.apply(lambda s: (s.count('G') + s.count('C')) / len(s) if len(s) > 0 else np.nan)
    
    # Detect expression columns
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    exclude = {"start", "stop", "Unnamed: 0", "chr", "Cluster", "_total_expr"}
    expr_cols = [c for c in numeric_cols if c not in exclude]
    
    # Build statistics string
    stats_lines = []
    stats_lines.append(f"{'='*60}")
    stats_lines.append(f"STATISTICS FOR {FAMILY_NAME}{label}")
    stats_lines.append(f"{'='*60}\n")
    
    stats_lines.append(f"Dataset size: {len(df)} sequences\n")
    
    stats_lines.append("=== SEQUENCE LENGTH STATISTICS ===")
    stats_lines.append(f"Count:   {len(lengths)}")
    stats_lines.append(f"Mean:    {lengths.mean():.2f} bp")
    stats_lines.append(f"Median:  {lengths.median():.2f} bp")
    stats_lines.append(f"Std Dev: {lengths.std():.2f} bp")
    stats_lines.append(f"Min:     {lengths.min()} bp")
    stats_lines.append(f"Max:     {lengths.max()} bp")
    stats_lines.append(f"Q1:      {lengths.quantile(0.25):.2f} bp")
    stats_lines.append(f"Q3:      {lengths.quantile(0.75):.2f} bp\n")
    
    stats_lines.append("=== GC CONTENT STATISTICS ===")
    stats_lines.append(f"Mean:    {gc.mean():.3f} ({gc.mean()*100:.1f}%)")
    stats_lines.append(f"Median:  {gc.median():.3f} ({gc.median()*100:.1f}%)")
    stats_lines.append(f"Std Dev: {gc.std():.3f}")
    stats_lines.append(f"Min:     {gc.min():.3f} ({gc.min()*100:.1f}%)")
    stats_lines.append(f"Max:     {gc.max():.3f} ({gc.max()*100:.1f}%)")
    stats_lines.append(f"Q1:      {gc.quantile(0.25):.3f} ({gc.quantile(0.25)*100:.1f}%)")
    stats_lines.append(f"Q3:      {gc.quantile(0.75):.3f} ({gc.quantile(0.75)*100:.1f}%)\n")
    
    if expr_cols:
        stats_lines.append("=== EXPRESSION DATA STATISTICS ===")
        stats_lines.append(f"Number of expression columns: {len(expr_cols)}\n")
        
        for col in expr_cols:
            stats_lines.append(f"--- {col} ---")
            stats_lines.append(f"Mean:    {df[col].mean():.2f}")
            stats_lines.append(f"Median:  {df[col].median():.2f}")
            stats_lines.append(f"Std Dev: {df[col].std():.2f}")
            stats_lines.append(f"Min:     {df[col].min():.2f}")
            stats_lines.append(f"Max:     {df[col].max():.2f}")
            stats_lines.append(f"Q1:      {df[col].quantile(0.25):.2f}")
            stats_lines.append(f"Q3:      {df[col].quantile(0.75):.2f}")
            stats_lines.append(f"Non-zero: {(df[col] > 0).sum()} ({(df[col] > 0).sum()/len(df)*100:.1f}%)")
            stats_lines.append("")
        
        # Overall expression statistics
        total_expr = df[expr_cols].sum(axis=1)
        stats_lines.append("=== TOTAL EXPRESSION (sum across all columns) ===")
        stats_lines.append(f"Mean:    {total_expr.mean():.2f}")
        stats_lines.append(f"Median:  {total_expr.median():.2f}")
        stats_lines.append(f"Std Dev: {total_expr.std():.2f}")
        stats_lines.append(f"Min:     {total_expr.min():.2f}")
        stats_lines.append(f"Max:     {total_expr.max():.2f}\n")
        
        # Expression by condition (if applicable)
        condition_groups = {}
        for col in expr_cols:
            parts = col.split('_')
            if len(parts) >= 2:
                condition = parts[1]
                if condition not in condition_groups:
                    condition_groups[condition] = []
                condition_groups[condition].append(col)
        
        if len(condition_groups) > 1:
            stats_lines.append("=== EXPRESSION BY CONDITION ===")
            for condition, cols in sorted(condition_groups.items()):
                condition_expr = df[cols].sum(axis=1)
                stats_lines.append(f"--- {condition} ({len(cols)} replicates) ---")
                stats_lines.append(f"Mean:    {condition_expr.mean():.2f}")
                stats_lines.append(f"Median:  {condition_expr.median():.2f}")
                stats_lines.append(f"Std Dev: {condition_expr.std():.2f}")
                stats_lines.append("")
    
    stats_text = "\n".join(stats_lines)
    
    # Print to console
    print(stats_text)
    
    # Save to file if specified
    if output_file:
        with open(output_file, "w") as f:
            f.write(stats_text)
        print(f"\n✓ Statistics saved to {output_file}")
    
    return expr_cols

# Compute overall statistics
overall_stats_file = DIRS['stats'] / "overall_statistics.txt"
expr_cols = compute_basic_stats(df_family, label=" - OVERALL", output_file=overall_stats_file)

_record_stage("Basic Statistics", _pipeline_time.time() - _stage_t0)

# ==================== CLUSTERING ANALYSIS ====================
_stage_t0 = _pipeline_time.time()
print("\n=== PERFORMING CLUSTERING ANALYSIS ===")

try:

    # Check if dataset is large enough for clustering
    if len(df_family) < MIN_SEQUENCES_FOR_CLUSTERING:
        print(f"\n⚠️  WARNING: Only {len(df_family)} sequences found.")
        print(f"   Clustering analysis requires at least {MIN_SEQUENCES_FOR_CLUSTERING} sequences.")
        print(f"   Skipping clustering and assigning all sequences to Cluster 0.")

        # Assign all sequences to a single cluster
        df_family['Cluster'] = 0
        cluster_labels = np.array([0] * len(df_family))
        embedding = None

        # Save data without clustering visualization
        df_family.to_csv(DIRS['data'] / f"{FAMILY_NAME.lower()}_clustered.csv", index=False)
        print(f"Saved data to {DIRS['data'] / f'{FAMILY_NAME.lower()}_clustered.csv'}")

        # Create a simple note file explaining why clustering was skipped
        with open(DIRS['clustering'] / "clustering_skipped.txt", "w") as f:
            f.write(f"Clustering Analysis Skipped\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Dataset size: {len(df_family)} sequences\n")
            f.write(f"Minimum required: {MIN_SEQUENCES_FOR_CLUSTERING} sequences\n\n")
            f.write(f"All sequences have been assigned to Cluster 0.\n")
            f.write(f"Clustering requires more sequences for meaningful results.\n")

        print(f"Explanation saved to {DIRS['clustering'] / 'clustering_skipped.txt'}")

        # Create a simple HTML visualization even when clustering is skipped
        skip_viz_html = f"""<!DOCTYPE html>
    <html>
    <head>
        <title>{FAMILY_NAME} Clustering - Skipped</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .card {{ background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 600px; margin: auto; }}
            h1 {{ color: #ff9800; }}
            .info {{ background: #fff3e0; padding: 15px; border-radius: 4px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>⚠️ Clustering Analysis Skipped</h1>
            <div class="info">
                <p><strong>Dataset size:</strong> {len(df_family)} sequences</p>
                <p><strong>Minimum required:</strong> {MIN_SEQUENCES_FOR_CLUSTERING} sequences</p>
            </div>
            <p>All sequences have been assigned to <strong>Cluster 0</strong>.</p>
            <p>Clustering requires more sequences for meaningful results.</p>
            <p>Other analyses (alignments, primers, visualizations) will still be generated.</p>
        </div>
    </body>
    </html>"""
        with open(DIRS['clustering'] / "clustering_visualization.html", "w") as f:
            f.write(skip_viz_html)
        print(f"Saved clustering visualization to {DIRS['clustering'] / 'clustering_visualization.html'}")

    else:
        def clustering_analysis(df):
            seqs = df["Seq"].tolist()
            n_samples = len(seqs)

            # Step 1: K-mer encoding
            progress_print(f"Step 1/5: Encoding {n_samples} sequences as {K}-mer vectors...")
            vectorizer = CountVectorizer(analyzer="char", ngram_range=(K, K))
            encoded = vectorizer.fit_transform(seqs).toarray()
            n_features = encoded.shape[1]
            progress_print(f"  ✓ Encoded into {n_features} unique {K}-mers")

            # Check if we have enough features for dimensionality reduction
            if n_features < 2:
                progress_print(f"  ⚠ Only {n_features} unique {K}-mers found - sequences too similar")
                progress_print(f"  Assigning all sequences to Cluster 0")
                return np.zeros(n_samples, dtype=int), np.column_stack([np.arange(n_samples), np.zeros(n_samples)])

            # Determine number of PCA components (must be <= min(n_samples, n_features))
            n_components = min(2, n_features, n_samples)

            # Step 2: PCA embedding
            progress_print(f"Step 2/5: Computing PCA embedding (n_components={n_components})...")
            pca_emb = PCA(n_components=n_components, random_state=42).fit_transform(encoded)
            progress_print(f"  ✓ PCA complete")

            # If only 1 component, add a zero column to make it 2D for visualization
            if pca_emb.shape[1] == 1:
                pca_emb = np.column_stack([pca_emb, np.zeros(n_samples)])

            # Step 3: UMAP embedding
            n_neighbors = min(30, n_samples - 1)
            if n_neighbors < 2:
                n_neighbors = 2
            umap_components = min(2, n_features)
            progress_print(f"Step 3/5: Computing UMAP embedding (n_neighbors={n_neighbors})...")
            progress_print(f"  This may take a moment for large datasets...")

            try:
                umap_emb = umap.UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=0.0,
                    n_components=umap_components,
                    random_state=42,
                    verbose=DEBUG
                ).fit_transform(encoded)
                progress_print(f"  ✓ UMAP complete")
            except Exception as e:
                progress_print(f"  ⚠ UMAP failed with default settings: {e}")
                progress_print(f"  Retrying with n_jobs=1 (single-threaded)...")
                try:
                    umap_emb = umap.UMAP(
                        n_neighbors=n_neighbors,
                        min_dist=0.0,
                        n_components=umap_components,
                        random_state=42,
                        verbose=DEBUG,
                        n_jobs=1
                    ).fit_transform(encoded)
                    progress_print(f"  ✓ UMAP complete (single-threaded)")
                except Exception as e2:
                    progress_print(f"  ❌ UMAP failed again: {e2}")
                    progress_print(f"  Skipping UMAP and using PCA as fallback for clustering")
                    # Fallback: use PCA embedding (padded if needed)
                    umap_emb = pca_emb.copy()

            # If only 1 component, add a zero column
            if umap_emb.shape[1] == 1:
                umap_emb = np.column_stack([umap_emb, np.zeros(n_samples)])

            # Step 4: t-SNE embedding
            perplexity = min(30, n_samples - 1)
            if perplexity < 5:
                perplexity = 5
            tsne_components = min(2, n_features)
            progress_print(f"Step 4/5: Computing t-SNE embedding (perplexity={perplexity})...")
            progress_print(f"  This may take a moment for large datasets...")

            try:
                tsne_emb = TSNE(
                    n_components=tsne_components,
                    perplexity=perplexity,
                    init='random' if n_features < 2 else 'pca',
                    random_state=42,
                    verbose=1 if DEBUG else 0
                ).fit_transform(encoded)
                progress_print(f"  ✓ t-SNE complete")
            except Exception as e:
                progress_print(f"  ⚠ t-SNE failed: {e}")
                progress_print(f"  Skipping t-SNE and using PCA as fallback")
                tsne_emb = pca_emb.copy()

            # If only 1 component, add a zero column
            if tsne_emb.shape[1] == 1:
                tsne_emb = np.column_stack([tsne_emb, np.zeros(n_samples)])

            # Step 5: HDBSCAN Clustering
            min_cluster_size = max(5, int(len(df)/7))
            progress_print(f"Step 5/5: Running HDBSCAN clustering (min_cluster_size={min_cluster_size})...")

            progress_print(f"  Clustering PCA embedding...", newline=False)
            pca_clusterer = hdbscan.HDBSCAN(min_samples=7, min_cluster_size=min_cluster_size)
            pca_labels = pca_clusterer.fit_predict(pca_emb)
            print(f" found {len(set(pca_labels)) - (1 if -1 in pca_labels else 0)} clusters")

            progress_print(f"  Clustering UMAP embedding...", newline=False)
            umap_clusterer = hdbscan.HDBSCAN(min_samples=7, min_cluster_size=min_cluster_size)
            umap_labels = umap_clusterer.fit_predict(umap_emb)
            print(f" found {len(set(umap_labels)) - (1 if -1 in umap_labels else 0)} clusters")

            progress_print(f"  Clustering t-SNE embedding...", newline=False)
            tsne_clusterer = hdbscan.HDBSCAN(min_samples=7, min_cluster_size=min_cluster_size)
            tsne_labels = tsne_clusterer.fit_predict(tsne_emb)
            print(f" found {len(set(tsne_labels)) - (1 if -1 in tsne_labels else 0)} clusters")

            # Check if UMAP clustering failed (all noise)
            if len(set(umap_labels)) == 1 and -1 in umap_labels:
                progress_print("  ⚠ HDBSCAN assigned all to noise - using Cluster 0 instead")
                umap_labels = np.zeros(len(df), dtype=int)

            progress_print(f"  ✓ Clustering complete")

            progress_print("Generating clustering visualization...")
            # Create visualization with IMPROVED CLUSTER LABELING
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=[f"PCA (k={K})", f"UMAP (k={K})", f"t-SNE (k={K})"]
            )

            def add_panel(emb, col, labels, method_name, df):
                # Create color map for clusters with clear labeling
                unique_clusters = sorted(set(labels))
                cluster_colors = {c: i for i, c in enumerate(unique_clusters)}

                # Create hover text with cluster information and coordinates
                hover_texts = []
                for i, l in enumerate(labels):
                    row = df.iloc[i]
                    coord = f"{row.get('chr','?')}:{row.get('start','?')}-{row.get('stop','?')}"
                    hover_texts.append(
                        f"<b>Row {i}</b><br>"
                        f"{coord}<br>"
                        f"Cluster: {l}<br>"
                        f"Method: {method_name}<br>"
                        f"x: %{{x:.2f}}<br>"
                        f"y: %{{y:.2f}}"
                    )

                fig.add_trace(
                    go.Scatter(
                        x=emb[:, 0], y=emb[:, 1],
                        mode="markers",
                        marker=dict(
                            size=5,
                            color=[cluster_colors[l] for l in labels],
                            colorscale="Viridis",
                            showscale=(col==3),
                            colorbar=dict(
                                title="Cluster ID",
                                tickvals=list(range(len(unique_clusters))),
                                ticktext=[f"C{c}" if c != -1 else "Noise" for c in unique_clusters]
                            ) if col==3 else None
                        ),
                        text=hover_texts,
                        hovertemplate="%{text}<extra></extra>",
                        name=method_name
                    ),
                    row=1, col=col
                )

            add_panel(pca_emb, 1, pca_labels, "PCA", df)
            add_panel(umap_emb, 2, umap_labels, "UMAP", df)
            add_panel(tsne_emb, 3, tsne_labels, "t-SNE", df)

            fig.update_layout(
                width=1500,
                height=450,
                showlegend=False,
                title=f"Clustering Analysis for {FAMILY_NAME} (k={K}) - Colors indicate Cluster IDs"
            )

            # Save figure
            fig.write_html(DIRS['clustering'] / "clustering_visualization.html")
            print(f"Saved clustering visualization to {DIRS['clustering'] / 'clustering_visualization.html'}")

            # Save coordinates CSV with embeddings and cluster assignments
            coord_rows = []
            for i in range(len(df)):
                row = df.iloc[i]
                coord_rows.append({
                    "row": i,
                    "chr": row.get("chr", ""),
                    "start": row.get("start", ""),
                    "stop": row.get("stop", ""),
                    "strand": row.get("strand", ""),
                    "cluster": int(umap_labels[i]),
                    "pca_x": pca_emb[i, 0], "pca_y": pca_emb[i, 1],
                    "umap_x": umap_emb[i, 0], "umap_y": umap_emb[i, 1],
                    "tsne_x": tsne_emb[i, 0], "tsne_y": tsne_emb[i, 1],
                })
            pd.DataFrame(coord_rows).to_csv(DIRS['clustering'] / "clustering_coordinates.csv", index=False)
            print(f"Saved clustering coordinates to {DIRS['clustering'] / 'clustering_coordinates.csv'}")

            # Use UMAP labels as primary clustering (most common in literature)
            return umap_labels, umap_emb

        # Perform clustering
        cluster_labels, embedding = clustering_analysis(df_family)
        df_family['Cluster'] = cluster_labels

        # Save clustered data
        df_family.to_csv(DIRS['data'] / f"{FAMILY_NAME.lower()}_clustered.csv", index=False)
        print(f"Saved clustered data to {DIRS['data'] / f'{FAMILY_NAME.lower()}_clustered.csv'}")

    # Rename for consistency with primer script
    df2 = df_family.copy()

except Exception as e:
    log_error("CLUSTERING ANALYSIS", e, {
        "FAMILY_NAME": FAMILY_NAME,
        "K": K,
        "df_family_shape": df_family.shape if 'df_family' in dir() else "N/A",
        "MIN_SEQUENCES": MIN_SEQUENCES_FOR_CLUSTERING
    })
    # Try to continue with single cluster assignment
    print("Attempting to continue with single cluster assignment...")
    try:
        df_family['Cluster'] = 0
        df2 = df_family.copy()
        print("Assigned all sequences to Cluster 0 due to clustering error")
    except Exception as e2:
        log_error("CLUSTERING FALLBACK", e2, {})
        sys.exit(1)

_record_stage("Clustering Analysis", _pipeline_time.time() - _stage_t0)

# ==================== PER-CLUSTER STATISTICS ====================
_stage_t0 = _pipeline_time.time()
print("\n=== COMPUTING PER-CLUSTER STATISTICS ===")

cluster_stats_dir = DIRS['stats'] / "per_cluster"
cluster_stats_dir.mkdir(exist_ok=True)

cluster_summary_data = []

for cluster in sorted(df_family['Cluster'].unique()):
    cluster_df = df_family[df_family['Cluster'] == cluster]
    cluster_size = len(cluster_df)
    
    if cluster == -1:
        cluster_label = f" - CLUSTER {cluster} (NOISE, n={cluster_size})"
    else:
        cluster_label = f" - CLUSTER {cluster} (n={cluster_size})"
    
    print(f"\nProcessing Cluster {cluster} ({cluster_size} sequences)...")
    
    # Compute stats for this cluster
    cluster_stats_file = cluster_stats_dir / f"cluster_{cluster}_statistics.txt"
    compute_basic_stats(cluster_df, label=cluster_label, output_file=cluster_stats_file)
    
    # Collect summary data
    seqs = cluster_df['Seq'].astype(str)
    lengths = seqs.apply(len)
    gc = seqs.apply(lambda s: (s.count('G') + s.count('C')) / len(s) if len(s) > 0 else np.nan)
    
    if expr_cols:
        total_expr = cluster_df[expr_cols].sum(axis=1)
        cluster_summary_data.append({
            'cluster': cluster,
            'size': cluster_size,
            'mean_length': lengths.mean(),
            'median_length': lengths.median(),
            'mean_gc': gc.mean(),
            'median_gc': gc.median(),
            'mean_total_expr': total_expr.mean(),
            'median_total_expr': total_expr.median(),
            'sum_total_expr': total_expr.sum()
        })
    else:
        cluster_summary_data.append({
            'cluster': cluster,
            'size': cluster_size,
            'mean_length': lengths.mean(),
            'median_length': lengths.median(),
            'mean_gc': gc.mean(),
            'median_gc': gc.median()
        })

# Save cluster summary table
cluster_summary_df = pd.DataFrame(cluster_summary_data)
cluster_summary_file = DIRS['stats'] / "cluster_summary_table.csv"
cluster_summary_df.to_csv(cluster_summary_file, index=False)
print(f"\n✓ Cluster summary table saved to {cluster_summary_file}")

# Create a comprehensive cluster comparison file
cluster_comparison_file = DIRS['stats'] / "cluster_comparison.txt"
with open(cluster_comparison_file, "w") as f:
    f.write(f"{'='*60}\n")
    f.write(f"CLUSTER COMPARISON FOR {FAMILY_NAME}\n")
    f.write(f"{'='*60}\n\n")
    f.write(f"Total sequences: {len(df_family)}\n")
    f.write(f"Number of clusters: {len([c for c in df_family['Cluster'].unique() if c != -1])}\n")
    if -1 in df_family['Cluster'].unique():
        f.write(f"Noise points (cluster -1): {len(df_family[df_family['Cluster'] == -1])}\n")
    f.write("\n")
    
    f.write("CLUSTER SIZE DISTRIBUTION:\n")
    f.write("-" * 40 + "\n")
    for _, row in cluster_summary_df.sort_values('cluster').iterrows():
        f.write(f"Cluster {int(row['cluster']):3d}: {int(row['size']):5d} sequences "
                f"({row['size']/len(df_family)*100:5.1f}%)\n")
    f.write("\n")
    
    f.write("CLUSTER CHARACTERISTICS:\n")
    f.write("-" * 40 + "\n")
    f.write(f"{'Cluster':<10} {'Size':<8} {'Mean Len':<12} {'Mean GC':<10}")
    if 'mean_total_expr' in cluster_summary_df.columns:
        f.write(f" {'Mean Expr':<12}\n")
    else:
        f.write("\n")
    
    for _, row in cluster_summary_df.sort_values('cluster').iterrows():
        f.write(f"{int(row['cluster']):<10} {int(row['size']):<8} "
                f"{row['mean_length']:<12.1f} {row['mean_gc']:<10.3f}")
        if 'mean_total_expr' in row:
            f.write(f" {row['mean_total_expr']:<12.1f}\n")
        else:
            f.write("\n")

print(f"✓ Cluster comparison saved to {cluster_comparison_file}")

_record_stage("Per-Cluster Statistics", _pipeline_time.time() - _stage_t0)

# ==================== GENERATE VISUALIZATION DASHBOARD ====================
_stage_t0 = _pipeline_time.time()
print("\n=== GENERATING VISUALIZATION DASHBOARD ===")

try:
    import plotly.express as px

    # Use the proper 07_visualizations directory
    vis_dir = DIRS['visualizations']
    vis_dir.mkdir(exist_ok=True)
    
    # Plot 1: Cluster distribution
    cluster_counts = df_family['Cluster'].value_counts().sort_index()
    noise_count = cluster_counts.get(-1, 0)
    cluster_counts = cluster_counts[cluster_counts.index != -1] if -1 in cluster_counts.index else cluster_counts
    
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Bar(
        x=cluster_counts.index,
        y=cluster_counts.values,
        name='Clusters',
        marker_color='steelblue',
        text=cluster_counts.values,
        textposition='auto'
    ))
    if noise_count > 0:
        fig_dist.add_trace(go.Bar(
            x=[-1], y=[noise_count], name='Noise',
            marker_color='lightgray', text=[noise_count], textposition='auto'
        ))
    fig_dist.update_layout(
        title="Cluster Size Distribution",
        xaxis_title="Cluster ID",
        yaxis_title="Number of Sequences",
        height=400
    )
    fig_dist.write_html(vis_dir / "cluster_distribution.html")
    print(f"✓ Saved: cluster_distribution.html")
    
    # Plot 2: Sequence characteristics
    df_family['length'] = df_family['Seq'].astype(str).apply(len)
    df_family['gc_content'] = df_family['Seq'].astype(str).apply(
        lambda s: (s.count('G') + s.count('C')) / len(s) if len(s) > 0 else 0
    )
    
    fig_chars = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Sequence Length by Cluster", "GC Content by Cluster",
                       "Length Distribution", "GC Distribution"),
        specs=[[{"type": "box"}, {"type": "box"}],
               [{"type": "histogram"}, {"type": "histogram"}]]
    )
    
    # Include ALL clusters including noise (-1) to ensure visualizations are always populated
    clusters = sorted(df_family['Cluster'].unique())
    for cluster in clusters:
        cluster_df = df_family[df_family['Cluster'] == cluster]
        cluster_label = "Noise" if cluster == -1 else f"C{cluster}"
        fig_chars.add_trace(go.Box(y=cluster_df['length'], name=cluster_label, showlegend=False), row=1, col=1)
        fig_chars.add_trace(go.Box(y=cluster_df['gc_content'], name=cluster_label, showlegend=False), row=1, col=2)
    
    fig_chars.add_trace(go.Histogram(x=df_family['length'], nbinsx=30, showlegend=False, marker_color='steelblue'), row=2, col=1)
    fig_chars.add_trace(go.Histogram(x=df_family['gc_content'], nbinsx=30, showlegend=False, marker_color='coral'), row=2, col=2)
    
    fig_chars.update_xaxes(title_text="Cluster", row=1, col=1)
    fig_chars.update_xaxes(title_text="Cluster", row=1, col=2)
    fig_chars.update_xaxes(title_text="Length (bp)", row=2, col=1)
    fig_chars.update_xaxes(title_text="GC Content", row=2, col=2)
    fig_chars.update_yaxes(title_text="Length (bp)", row=1, col=1)
    fig_chars.update_yaxes(title_text="GC Content", row=1, col=2)
    fig_chars.update_yaxes(title_text="Count", row=2, col=1)
    fig_chars.update_yaxes(title_text="Count", row=2, col=2)
    fig_chars.update_layout(height=800, title_text="Sequence Characteristics")
    fig_chars.write_html(vis_dir / "sequence_characteristics.html")
    print(f"✓ Saved: sequence_characteristics.html")
    
    # Plot 3: Expression heatmap (if expression data exists)
    if expr_cols:
        cluster_expr = df_family.groupby('Cluster')[expr_cols].mean()
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=cluster_expr.values,
            x=cluster_expr.columns,
            y=[f"Cluster {c}" for c in cluster_expr.index],
            colorscale='Viridis',
            colorbar=dict(title="Mean Expression")
        ))
        fig_heatmap.update_layout(
            title="Mean Expression by Cluster",
            xaxis_title="Sample",
            yaxis_title="Cluster",
            height=max(400, len(cluster_expr) * 30)
        )
        fig_heatmap.write_html(vis_dir / "expression_heatmap.html")
        print(f"✓ Saved: expression_heatmap.html")
        
        # Plot 4: Expression comparison by condition
        condition_groups = {}
        for col in expr_cols:
            parts = col.split('_')
            if len(parts) >= 2:
                condition = parts[1]
                if condition not in condition_groups:
                    condition_groups[condition] = []
                condition_groups[condition].append(col)
        
        if len(condition_groups) > 1:
            data_for_plot = []
            for cluster in sorted(df_family['Cluster'].unique()):
                cluster_df = df_family[df_family['Cluster'] == cluster]
                for condition, cols in condition_groups.items():
                    total_expr = cluster_df[cols].sum(axis=1).mean()
                    data_for_plot.append({
                        'Cluster': f"C{cluster}",
                        'Condition': condition,
                        'Expression': total_expr
                    })
            
            plot_df = pd.DataFrame(data_for_plot)
            fig_expr = px.bar(plot_df, x='Cluster', y='Expression', color='Condition',
                             barmode='group', title="Mean Expression by Cluster and Condition")
            fig_expr.update_layout(height=500)
            fig_expr.write_html(vis_dir / "expression_comparison.html")
            print(f"✓ Saved: expression_comparison.html")
    
    # Create dashboard index
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{FAMILY_NAME} Analysis Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        .viz-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-top: 20px; }}
        .viz-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .viz-card h2 {{ margin-top: 0; color: #4CAF50; }}
        .viz-card a {{ display: inline-block; margin-top: 10px; padding: 10px 20px; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 4px; }}
        .viz-card a:hover {{ background-color: #45a049; }}
        iframe {{ width: 100%; height: 500px; border: 1px solid #ddd; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>{FAMILY_NAME} Analysis Dashboard</h1>
    <p>Comprehensive visualization of clustering and expression analysis</p>
    <div class="viz-grid">
        <div class="viz-card"><h2>Cluster Distribution</h2><iframe src="cluster_distribution.html"></iframe><a href="cluster_distribution.html" target="_blank">Open</a></div>
        <div class="viz-card"><h2>Sequence Characteristics</h2><iframe src="sequence_characteristics.html"></iframe><a href="sequence_characteristics.html" target="_blank">Open</a></div>
        {'<div class="viz-card"><h2>Expression Heatmap</h2><iframe src="expression_heatmap.html"></iframe><a href="expression_heatmap.html" target="_blank">Open</a></div>' if expr_cols else ''}
        {'<div class="viz-card"><h2>Expression Comparison</h2><iframe src="expression_comparison.html"></iframe><a href="expression_comparison.html" target="_blank">Open</a></div>' if expr_cols and len(condition_groups) > 1 else ''}
    </div>
    <div style="margin-top: 40px; padding: 20px; background: white; border-radius: 8px;">
        <h2>Related Files</h2>
        <ul>
            <li><a href="../overall_statistics.txt">Overall Statistics</a></li>
            <li><a href="../cluster_comparison.txt">Cluster Comparison</a></li>
            <li><a href="../cluster_statistics/">Per-Cluster Statistics</a></li>
            <li><a href="../clustering_visualization.html">Original Clustering (PCA/UMAP/t-SNE)</a></li>
        </ul>
    </div>
</body>
</html>"""
    
    with open(vis_dir / "index.html", "w") as f:
        f.write(html_content)
    
    print(f"✓ Dashboard created: {vis_dir / 'index.html'}")

except Exception as e:
    print(f"Warning: Could not create full visualization dashboard: {e}")
    # Create a minimal fallback visualization to ensure folder is never empty
    try:
        fallback_html = f"""<!DOCTYPE html>
<html>
<head><title>{FAMILY_NAME} Analysis</title></head>
<body>
<h1>{FAMILY_NAME} Analysis Results</h1>
<p>Total sequences: {len(df_family)}</p>
<p>Clusters: {sorted(df_family['Cluster'].unique())}</p>
<p>Note: Full visualizations could not be generated. Error: {e}</p>
</body>
</html>"""
        with open(DIRS['visualizations'] / "index.html", "w") as f:
            f.write(fallback_html)
        print(f"✓ Fallback visualization created: {DIRS['visualizations'] / 'index.html'}")
    except Exception as e2:
        print(f"Warning: Could not create fallback visualization: {e2}")

_record_stage("Visualization Dashboard", _pipeline_time.time() - _stage_t0)

# ==================== GENERATE README ====================
print("\n=== GENERATING README ===")

readme_content = f"""# {FAMILY_NAME} Analysis Results

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Quick Start

1. Open **visualizations/index.html** in your browser for interactive dashboard
2. Read **overall_statistics.txt** for summary statistics
3. Check **selected_primers_summary.csv** for best primers

## Dataset Summary

- Total sequences: {len(df2)}
- Number of clusters: {len([c for c in df2['Cluster'].unique() if c != -1])}
- Noise sequences: {len(df2[df2['Cluster'] == -1]) if -1 in df2['Cluster'].values else 0}

## Directory Structure

```
{FAMILY_NAME.lower()}/
├── README.txt                          (this file)
├── overall_statistics.txt              (complete statistics)
├── cluster_comparison.txt              (cluster summary)
├── {FAMILY_NAME.lower()}_clustered.csv (data with clusters)
├── {FAMILY_NAME.lower()}_seqs.fa       (all sequences)
├── {FAMILY_NAME.lower()}_aligned.fa    (alignment)
├── {FAMILY_NAME.lower()}_consensus.fa  (consensus)
├── selected_primers_summary.csv        (best primers)
├── visualizations/                     (interactive plots)
│   └── index.html                      (🎯 START HERE)
├── cialign_plots/                      (alignment visualizations)
│   └── index.html                      (🎨 ALIGNMENT PLOTS)
├── cleaned_consensus/                  (CIAlign cleaned consensus)
│   ├── {{family}}_cleaned_consensus.fa
│   ├── cluster_*_cleaned_consensus.fa
│   └── all_clusters_cleaned_consensus.fa
├── cluster_statistics/                 (per-cluster stats)
└── cluster_alignments/                 (per-cluster alignments)
```

## Key Output Files

### Statistics
- **overall_statistics.txt** - Length, GC, expression stats for all sequences
- **cluster_comparison.txt** - Quick comparison across clusters
- **cluster_statistics/cluster_{{n}}_statistics.txt** - Stats for each cluster

### Sequences & Alignments
- **{{family}}_seqs.fa** - All sequences in FASTA format
- **{{family}}_aligned.fa** - Multiple sequence alignment (MAFFT)
- **{{family}}_consensus.fa** - Consensus sequence from alignment
- **cluster_alignments/** - Per-cluster alignments and consensuses
- **cleaned_consensus/** - CIAlign cleaned consensus sequences

### Primers
- **selected_primers_summary.csv** - Top {TOP_N_GLOBAL} primers (global)
- **cluster_top5_primers.csv** - Top {TOP_N_CLUSTER} primers per cluster
- **{{primer}}_genome_hits.csv** - Genome locations for each primer
- **primer_genome_hits_summary.csv** - Summary of genome hits

### Visualizations
- **visualizations/index.html** - Interactive dashboard
- **cialign_plots/index.html** - CIAlign alignment visualizations
- **clustering_visualization.html** - PCA/UMAP/t-SNE plots
- **visualizations/cluster_distribution.html** - Cluster sizes
- **visualizations/sequence_characteristics.html** - Length/GC plots
- **visualizations/expression_heatmap.html** - Expression by cluster
- **visualizations/expression_comparison.html** - Condition comparison

## Understanding Your Results

### Clusters
Sequences were grouped by sequence similarity using:
- 12-mer frequency analysis
- UMAP dimensionality reduction  
- HDBSCAN clustering

Cluster -1 = noise/outliers that don't fit into any cluster.

### Primers
Two optimization strategies were used:
1. **cov_then_expr**: Maximize coverage first, then expression
2. **expr_then_cov**: Maximize expression first, then coverage

Best primers typically rank high in both strategies.

### Expression (if available)
Expression data across conditions shows:
- Which sequences are highly expressed
- Differential expression between conditions
- Cluster-specific expression patterns

## Common Next Steps

1. **Experimental Design**
   - Check selected_primers_summary.csv
   - Verify genome specificity in primer_genome_hits_summary.csv
   - Design PCR/qPCR experiments

2. **Sequence Analysis**
   - View alignments in Jalview/AliView
   - BLAST consensus sequences
   - Compare cluster consensuses

3. **Publication**
   - Export plots from visualizations/
   - Copy statistics from .txt files
   - Include primer tables

## Parameters Used

- K-mer size: {K}
- Top primers (global): {TOP_N_GLOBAL}
- Top primers (per cluster): {TOP_N_CLUSTER}
- Genome reference: {HG38_FA}

## Tools Compatibility

These files work with:
- **Alignment Viewers**: Jalview, AliView, Geneious
- **BLAST**: Use consensus.fa at NCBI
- **Primer Design**: Primer3, OligoAnalyzer
- **Statistics**: R, Python pandas, Excel
- **Genome Browsers**: UCSC, IGV

## Questions or Issues?

- Check OUTPUT_GUIDE.md for detailed documentation
- Verify input data format
- Ensure MAFFT is installed for alignments

Analysis pipeline version: 1.0
"""

readme_file = OUT_DIR / "README.txt"
with open(readme_file, "w") as f:
    f.write(readme_content)

print(f"✓ README generated: {readme_file}")




# ==================== PRIMER DESIGN ====================
_stage_t0 = _pipeline_time.time()
print("\n=== STARTING PRIMER DESIGN ===")

# --- helper funcs ---
def reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]

# Timeout constant for primer genome search (in seconds)
# Priority: 1) PRIMER_SEARCH_TIMEOUT_OVERRIDE (from hpc_client), 2) args.primer_timeout, 3) default 120s
if 'PRIMER_SEARCH_TIMEOUT_OVERRIDE' in dir() and PRIMER_SEARCH_TIMEOUT_OVERRIDE:
    PRIMER_SEARCH_TIMEOUT = PRIMER_SEARCH_TIMEOUT_OVERRIDE
    print(f"  Using primer timeout from HPC client: {PRIMER_SEARCH_TIMEOUT}s")
elif 'args' in dir() and hasattr(args, 'primer_timeout'):
    PRIMER_SEARCH_TIMEOUT = args.primer_timeout
else:
    PRIMER_SEARCH_TIMEOUT = 120  # Default 2 minutes

# Major chromosomes for sampling (excludes tiny scaffolds)
MAJOR_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

MAX_GENOME_HITS = 10000  # Stop searching after this many hits — primer is non-specific

print(f"  Primer search settings: timeout={PRIMER_SEARCH_TIMEOUT}s, max_hits={MAX_GENOME_HITS}")

def _search_single_chrom(primer, rc, plen, chrom, seq, max_hits=0, current_count=0):
    """Search a single chromosome for primer hits.
    Returns (hits_list, total_count). Stops early if max_hits exceeded.
    """
    hits = []
    count = current_count
    seq = seq.upper()
    # forward
    idx = seq.find(primer)
    while idx != -1:
        hits.append((chrom, idx+1, idx+plen, "+"))
        count += 1
        if max_hits and count >= max_hits:
            return hits, count
        idx = seq.find(primer, idx+1)
    # reverse
    idx = seq.find(rc)
    while idx != -1:
        hits.append((chrom, idx+1, idx+plen, "-"))
        count += 1
        if max_hits and count >= max_hits:
            return hits, count
        idx = seq.find(rc, idx+1)
    return hits, count

def _search_genome_full(primer, fasta_path, stop_event=None):
    """
    Full genome search - called in a thread with optional stop_event for cancellation.
    Returns list of hits or None if stopped.
    """
    primer = primer.upper()
    rc = reverse_complement(primer)
    plen = len(primer)
    hits = []
    total_count = 0
    max_hits = globals().get('MAX_GENOME_HITS', 10000)

    with open(fasta_path, "r") as f:
        chrom = None
        seq_buf = []
        for line in f:
            if stop_event and stop_event.is_set():
                return None

            if line.startswith(">"):
                if chrom is not None:
                    seq = "".join(seq_buf)
                    chrom_hits, total_count = _search_single_chrom(
                        primer, rc, plen, chrom, seq,
                        max_hits=max_hits, current_count=total_count
                    )
                    hits.extend(chrom_hits)
                    if max_hits and total_count >= max_hits:
                        return hits
                chrom = line[1:].strip().split()[0]
                seq_buf = []
            else:
                seq_buf.append(line.strip())
        if chrom is not None and (not stop_event or not stop_event.is_set()):
            seq = "".join(seq_buf)
            chrom_hits, total_count = _search_single_chrom(
                primer, rc, plen, chrom, seq,
                max_hits=max_hits, current_count=total_count
            )
            hits.extend(chrom_hits)

    return hits

def _estimate_genome_hits_by_sampling(primer, fasta_path, sample_chroms=5, seed=None):
    """
    Estimate genome-wide hits by sampling a few chromosomes and extrapolating.
    Much faster than full genome search.

    Uses random sampling of major chromosomes to estimate genome-wide primer specificity.
    This is called when the full genome search times out (default: 2 minutes).

    Args:
        primer: The primer sequence to search for
        fasta_path: Path to the genome FASTA file
        sample_chroms: Number of chromosomes to sample (default: 5)
        seed: Random seed for reproducibility (default: None for random)
    """
    primer = primer.upper()
    rc = reverse_complement(primer)
    plen = len(primer)

    # Read and index chromosome positions in the FASTA
    chrom_positions = {}  # chrom -> (file_offset_start, file_offset_end)

    print(f"    [RANDOM SAMPLING MODE] Sampling {sample_chroms} chromosomes to estimate coverage...")
    print(f"    This is faster than full genome search and provides a reasonable estimate.")

    # Parse FASTA to find chromosome boundaries
    progress_print(f"    Indexing genome chromosomes...", newline=False)
    with open(fasta_path, "r") as f:
        current_chrom = None
        current_start = 0
        line_pos = 0
        for line in f:
            if line.startswith(">"):
                if current_chrom is not None:
                    chrom_positions[current_chrom] = (current_start, line_pos)
                current_chrom = line[1:].strip().split()[0]
                current_start = line_pos + len(line)
            line_pos += len(line)
        # last chrom
        if current_chrom is not None:
            chrom_positions[current_chrom] = (current_start, line_pos)
    print(f" found {len(chrom_positions)} chromosomes/contigs")

    # Filter to major chromosomes that exist in the file
    available_major = [c for c in MAJOR_CHROMS if c in chrom_positions]

    if not available_major:
        # Fall back to any available chromosomes
        available_major = list(chrom_positions.keys())[:24]  # Limit to first 24

    if not available_major:
        print(f"    WARNING: No chromosomes found in {fasta_path}")
        return pd.DataFrame(columns=["chrom", "start", "stop", "strand"]), True

    # Randomly sample chromosomes (use random seed if not provided)
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    sample_size = min(sample_chroms, len(available_major))
    sampled_chroms = random.sample(available_major, sample_size)

    print(f"    Sampling chromosomes: {', '.join(sampled_chroms)}")

    hits = []
    total_sampled_length = 0

    # Read and search only sampled chromosomes
    with open(fasta_path, "r") as f:
        for i, chrom in enumerate(sampled_chroms):
            progress_print(f"    Searching {chrom} ({i+1}/{sample_size})...", newline=False)
            start_pos, end_pos = chrom_positions[chrom]
            f.seek(start_pos)
            data = f.read(end_pos - start_pos)
            seq = "".join(line for line in data.split('\n') if not line.startswith('>'))
            total_sampled_length += len(seq)
            chrom_hits, _ = _search_single_chrom(primer, rc, plen, chrom, seq)
            hits.extend(chrom_hits)
            print(f" {len(chrom_hits)} hits ({len(seq):,} bp)")

    # Estimate total genome hits based on sampling
    # Approximate total genome length (hg38 ~3.1 billion bp)
    ESTIMATED_GENOME_SIZE = 3_100_000_000

    if total_sampled_length > 0:
        scaling_factor = ESTIMATED_GENOME_SIZE / total_sampled_length
        estimated_total_hits = int(len(hits) * scaling_factor)
    else:
        estimated_total_hits = 0

    print(f"    ----------------------------------------")
    print(f"    Sampled {sample_size} chromosomes ({total_sampled_length:,} bp)")
    print(f"    Found {len(hits)} actual hits in sample")
    print(f"    Estimated genome-wide: ~{estimated_total_hits:,} hits (extrapolated)")
    print(f"    ----------------------------------------")

    dfh = pd.DataFrame(hits, columns=["chrom", "start", "stop", "strand"])
    dfh["estimated_total"] = estimated_total_hits  # Add metadata column
    dfh["sampling_note"] = f"Estimated from {sample_size} chromosomes (seed={seed})"
    return dfh, True  # True indicates this was estimated

def search_seq_chromosomal(primer, fasta=HG38_FA, timeout=PRIMER_SEARCH_TIMEOUT, _gcache=None):
    """
    Search hg38 for exact matches of primer and its RC.
    Returns DataFrame of hits columns [chrom, start, stop, strand].

    If _gcache (GenomeCache) is provided and loaded, uses fast in-memory search.
    Otherwise falls back to file-based search with timeout + random sampling fallback.
    """
    primer = primer.upper()

    # === FAST PATH: In-memory genome search ===
    if _gcache is not None and _gcache.is_loaded:
        import time as _time
        t0 = _time.time()
        rc = reverse_complement(primer)
        progress_print(f"  Searching in-memory genome for {primer} (rc: {rc})...")
        hits = _gcache.search_primer(primer)
        elapsed = _time.time() - t0
        dfh = pd.DataFrame(hits, columns=["chrom", "start", "stop", "strand"]) if hits else \
              pd.DataFrame(columns=["chrom", "start", "stop", "strand"])
        progress_print(f"  -> {len(dfh):,} genomic hits in {elapsed:.1f}s (in-memory)")
        return dfh

    # === SLOW PATH: File-based search with timeout ===
    fasta_path = Path(fasta) if fasta else None

    if fasta_path is None or not fasta_path.exists():
        debug_print(f"Genome file not found: {fasta}. Skipping genome search.")
        return pd.DataFrame(columns=["chrom", "start", "stop", "strand"])

    rc = reverse_complement(primer)
    print(f"Searching genome for primer {primer} (rc: {rc})")
    print(f"  Genome: {fasta_path.name}")
    print(f"  Timeout: {timeout}s (will use random sampling if exceeded)")

    stop_event = threading.Event()
    search_start_time = None

    def search_task():
        nonlocal search_start_time
        search_start_time = __import__('time').time()
        return _search_genome_full(primer, fasta_path, stop_event)

    import time
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(search_task)
        try:
            hits = future.result(timeout=timeout)
            elapsed = time.time() - start_time
            if hits is None:
                raise FuturesTimeoutError("Search cancelled")
            dfh = pd.DataFrame(hits, columns=["chrom", "start", "stop", "strand"])
            print(f"  -> COMPLETE: Found {len(dfh):,} genomic hits in {elapsed:.1f}s")
            return dfh
        except FuturesTimeoutError:
            stop_event.set()
            elapsed = time.time() - start_time
            print(f"\n  *** TIMEOUT after {elapsed:.1f}s ***")
            print(f"  Falling back to random chromosome sampling for estimation...")
            dfh, _ = _estimate_genome_hits_by_sampling(primer, fasta_path)
            return dfh

# ----------------- Build k-mer -> rows map -----------------
try:
    progress_print(f"Building {PRIMER_K}-mer index for primer design across {len(df2)} sequences...")

    # detect expression columns
    numeric_cols = list(df2.select_dtypes(include=[np.number]).columns)
    exclude = {"start", "stop", "Unnamed: 0", "chr", "Cluster"}
    expr_cols = [c for c in numeric_cols if c not in exclude]
    if not expr_cols:
        expr_cols = [c for c in df2.columns if any(prefix in c for prefix in
                     ("A1_","A2_","A3_","B1_","B2_","B3_","C1_","C2_","C3_"))]
    progress_print(f"  Expression columns detected: {len(expr_cols)}")

    # per-row total expression
    df2 = df2.reset_index(drop=True)
    row_total_expr = df2[expr_cols].sum(axis=1) if expr_cols else pd.Series(0, index=df2.index)
    df2["_total_expr"] = row_total_expr

    kmer_to_rows = defaultdict(set)
    progress_print(f"  Scanning sequences for {PRIMER_K}-mers (this may take a moment)...")
    total_seqs = len(df2)
    for idx, (ridx, seq) in enumerate(df2["Seq"].astype(str).items()):
        if idx % max(1, total_seqs // 10) == 0:
            progress_bar(idx, total_seqs, prefix="  Scanning", suffix="")
        s = seq.upper()
        L = len(s)
        if L < PRIMER_K:
            continue
        seen_local = set()
        for i in range(0, L - PRIMER_K + 1):
            kmer = s[i:i+PRIMER_K]
            if "N" in kmer:
                continue
            if kmer not in seen_local:
                kmer_to_rows[kmer].add(ridx)
                seen_local.add(kmer)
    progress_bar(total_seqs, total_seqs, prefix="  Scanning", suffix="complete")

    progress_print(f"  ✓ Found {len(kmer_to_rows):,} unique {PRIMER_K}-mers")

    # ----------------- compute metrics for each kmer -----------------
    progress_print("Computing coverage and expression metrics for each primer...")
    rows_total_expr = df2["_total_expr"].to_dict()

    kmer_records = []
    for kmer, rows_set in kmer_to_rows.items():
        coverage = len(rows_set)
        total_expr = sum(rows_total_expr[r] for r in rows_set)
        kmer_records.append((kmer, coverage, total_expr, rows_set))

    kmer_df = pd.DataFrame(
        [{"primer": r[0], "coverage": r[1], "total_expr": r[2], "rows": r[3]} for r in kmer_records]
    )

    # Check if any k-mers were found
    if len(kmer_df) == 0:
        print(f"\nWARNING: No valid {PRIMER_K}-mers found in sequences.")
        print("This can happen if sequences are too short or contain too many N's.")
        print("Skipping primer design section.")
        selected_primers = []
        top_by_cov_expr = pd.DataFrame()
        top_by_expr_cov = pd.DataFrame()
    else:
        # Add ranking columns
        kmer_df["rank_by_cov_expr"] = kmer_df.sort_values(["coverage","total_expr"], ascending=[False,False])\
                                             .reset_index().index + 1
        kmer_df["rank_by_expr_cov"] = kmer_df.sort_values(["total_expr","coverage"], ascending=[False,False])\
                                             .reset_index().index + 1

        # select top primers (global)
        top_by_cov_expr = kmer_df.sort_values(["coverage","total_expr"], ascending=[False,False]).head(TOP_N_GLOBAL).copy()
        top_by_expr_cov = kmer_df.sort_values(["total_expr","coverage"], ascending=[False,False]).head(TOP_N_GLOBAL).copy()

    if len(kmer_df) > 0:
        print("\nTop primers by (coverage desc, total_expr desc):")
        print(top_by_cov_expr[["primer","coverage","total_expr"]])

        print("\nTop primers by (total_expr desc, coverage desc):")
        print(top_by_expr_cov[["primer","coverage","total_expr"]])

        # ----------------- For selected primers, find genome hits -----------------
        print("\nFinding genomic hits (hg38) for selected primers...")
        selected_primers = pd.concat([top_by_cov_expr["primer"], top_by_expr_cov["primer"]]).unique().tolist()

    primer_hits = {}
    if selected_primers:
        progress_print(f"Searching genome for {len(selected_primers)} selected primers...")
        for pidx, pr in enumerate(selected_primers):
            progress_print(f"  Primer {pidx+1}/{len(selected_primers)}: {pr}")
            df_hits = search_seq_chromosomal(pr, fasta=HG38_FA, _gcache=genome_cache)
            primer_hits[pr] = df_hits
            fn = DIRS['primers'] / f"{pr}_genome_hits.csv"
            df_hits.to_csv(fn, index=False)
            print(f"Saved {fn.name}")

    # ----------------- Build results DataFrame for global primers -----------------
    def rows_set_to_str(rows_set):
        return ",".join(map(str, sorted(rows_set)))

    def rows_set_to_coords(rows_set, df):
        coords = []
        for r in sorted(rows_set):
            row = df.loc[r]
            coords.append(f"{row.get('chr','?')}:{row.get('start','?')}-{row.get('stop','?')}")
        return ",".join(coords)

    if len(kmer_df) > 0:
        all_primers_df = kmer_df.copy()
        all_primers_df["rows_covered"] = all_primers_df["rows"].apply(rows_set_to_str)
        all_primers_df["rows_coordinates"] = all_primers_df["rows"].apply(lambda rs: rows_set_to_coords(rs, df2))
        all_primers_df.sort_values(["coverage","total_expr"], ascending=[False,False])\
                      .to_csv(DIRS['primers'] / f"all_{PRIMER_K}mer_candidates_metrics.csv", index=False)
        print(f"Saved all candidates to {DIRS['primers'] / f'all_{PRIMER_K}mer_candidates_metrics.csv'}")

    if selected_primers:
        # Save top primers summary
        top_summary = pd.DataFrame({
            "primer": selected_primers,
            "coverage": [kmer_df.loc[kmer_df["primer"]==p,"coverage"].values[0] for p in selected_primers],
            "total_expr": [kmer_df.loc[kmer_df["primer"]==p,"total_expr"].values[0] for p in selected_primers],
            "rows_covered": [rows_set_to_str(kmer_df.loc[kmer_df["primer"]==p,"rows"].values[0]) for p in selected_primers],
            "rows_coordinates": [rows_set_to_coords(kmer_df.loc[kmer_df["primer"]==p,"rows"].values[0], df2) for p in selected_primers],
            "strategy": ["cov_then_expr" if p in set(top_by_cov_expr["primer"]) else "expr_then_cov"
                         for p in selected_primers]
        })
        top_summary.to_csv(DIRS['primers'] / "selected_primers_summary.csv", index=False)
        print(f"Saved selected primers to {DIRS['primers'] / 'selected_primers_summary.csv'}")
        print(top_summary)

    # ----------------- Primer-genome overlap table -----------------
    primer_overlap_rows = []
    for pr in selected_primers if selected_primers else []:
        rows_set = kmer_df.loc[kmer_df["primer"]==pr,"rows"].values[0]
        df_genome_hits = primer_hits.get(pr, pd.DataFrame(columns=["chrom","start","stop","strand"]))
        for rid in sorted(rows_set):
            row = df2.loc[rid]
            row_chr = row.get("chr", None)
            row_start = row.get("start", None)
            row_stop = row.get("stop", None)
            row_strand = row.get("strand", None)
            for _, gh in df_genome_hits.iterrows():
                primer_overlap_rows.append({
                    "primer": pr,
                    "seq_row": rid,
                    "row_chr": row_chr,
                    "row_start": row_start,
                    "row_stop": row_stop,
                    "row_strand": row_strand,
                    "genome_chr": gh["chrom"],
                    "genome_start": gh["start"],
                    "genome_stop": gh["stop"],
                    "genome_strand": gh["strand"]
                })

    df_primer_overlaps = pd.DataFrame(primer_overlap_rows)
    if len(df_primer_overlaps) > 0:
        df_primer_overlaps.to_csv(OUT_DIR / "primer_to_genome_overlap_hits.csv", index=False)
        print(f"Saved overlap table to {OUT_DIR / 'primer_to_genome_overlap_hits.csv'}")

    # ----------------- Per-cluster: top 5 primers per cluster -----------------
    print("\nComputing top primers per Cluster...")
    cluster_groups = df2.groupby("Cluster").indices
    cluster_top_primers = {}

    for cluster, rows_idx_list in cluster_groups.items():
        # Process ALL clusters including noise (-1) to ensure primers are always generated
        cluster_label = "noise" if cluster == -1 else str(cluster)
        print(f"\nProcessing cluster {cluster_label} for primers...")

        kmer_map_local = defaultdict(set)
        for rid in rows_idx_list:
            s = str(df2.at[rid,"Seq"]).upper()
            if len(s) < PRIMER_K:
                continue
            seen_local = set()
            for i in range(0, len(s)-PRIMER_K+1):
                k = s[i:i+PRIMER_K]
                if "N" in k:
                    continue
                if k not in seen_local:
                    kmer_map_local[k].add(rid)
                    seen_local.add(k)

        recs = []
        for k, rset in kmer_map_local.items():
            cov = len(rset)
            tot_expr = sum(rows_total_expr[r] for r in rset)
            recs.append((k, cov, tot_expr, rset))

        if not recs:
            continue

        df_local = pd.DataFrame([{"primer":r[0],"coverage":r[1],"total_expr":r[2],"rows":r[3]}
                                 for r in recs])
        top5 = df_local.sort_values(["coverage","total_expr"], ascending=[False,False]).head(TOP_N_CLUSTER)
        cluster_top_primers[cluster] = top5

        # find genome hits for these primers
        for p in top5["primer"].tolist():
            if p not in primer_hits:
                primer_hits[p] = search_seq_chromosomal(p, fasta=HG38_FA, _gcache=genome_cache)
                primer_hits[p].to_csv(DIRS['primers'] / f"{p}_genome_hits.csv", index=False)

    # save cluster results
    cluster_summary_rows = []
    for cl, dfp in cluster_top_primers.items():
        for _, r in dfp.iterrows():
            cluster_summary_rows.append({
                "cluster": cl,
                "primer": r["primer"],
                "coverage": r["coverage"],
                "total_expr": r["total_expr"],
                "rows_covered": rows_set_to_str(r["rows"])
            })

    cluster_summary_df = pd.DataFrame(cluster_summary_rows)
    cluster_summary_df.to_csv(DIRS['primers'] / "cluster_top5_primers.csv", index=False)
    print(f"Saved cluster primers to {DIRS['primers'] / 'cluster_top5_primers.csv'}")

    # ----------------- Write fasta of all sequences -----------------
    fasta_path = OUT_DIR / f"{FAMILY_NAME.lower()}_seqs.fa"
    with open(fasta_path, "w") as fh:
        for rid, row in df2.iterrows():
            header = f">row{rid}|{row.get('TE_name','')[:50]}|cluster{row.get('Cluster','')}"
            seq = str(row["Seq"]).strip().upper()
            fh.write(header + "\n")
            for i in range(0, len(seq), 80):
                fh.write(seq[i:i+80] + "\n")
    print(f"Wrote FASTA to {fasta_path}")

except Exception as e:
    log_error("PRIMER DESIGN", e, {
        "FAMILY_NAME": FAMILY_NAME,
        "PRIMER_K": PRIMER_K,
        "df2_shape": df2.shape if 'df2' in dir() else "N/A",
        "kmer_count": len(kmer_to_rows) if 'kmer_to_rows' in dir() else "N/A",
        "selected_primers": selected_primers if 'selected_primers' in dir() else "N/A",
        "HG38_FA": HG38_FA,
        "PRIMER_TIMEOUT": PRIMER_SEARCH_TIMEOUT if 'PRIMER_SEARCH_TIMEOUT' in dir() else "N/A"
    })
    print("Continuing with other analyses despite primer design error...")
    primer_hits = {}
    selected_primers = []

_record_stage("Primer Design", _pipeline_time.time() - _stage_t0)

# ==================== MULTIPLE SEQUENCE ALIGNMENT ====================
_stage_t0 = _pipeline_time.time()
print("\n=== PERFORMING MULTIPLE SEQUENCE ALIGNMENT ===")

try:
    from Bio import AlignIO
    from Bio.Align import AlignInfo
    from Bio import SeqIO
    from io import StringIO
    import subprocess

    # Check if MAFFT is available
    progress_print("Checking for MAFFT installation...")
    try:
        subprocess.run(["mafft", "--version"], capture_output=True, check=True)
        mafft_available = True
        progress_print("  ✓ MAFFT found")
    except:
        progress_print("  MAFFT not found. Attempting to install...")
        try:
            subprocess.run(["conda", "install", "-y", "-c", "bioconda", "mafft"], check=True)
            mafft_available = True
        except:
            progress_print("  ✗ Could not install MAFFT. Skipping alignment.")
            mafft_available = False

    if mafft_available:
        # ----------------- Global alignment -----------------
        progress_print(f"Aligning {len(df2)} sequences globally with MAFFT...")
        progress_print("  This may take several minutes for large datasets...")
        input_fasta = fasta_path
        output_aligned = OUT_DIR / f"{FAMILY_NAME.lower()}_aligned.fa"

        # Run MAFFT alignment
        mafft_cmd = f"mafft --auto --thread -1 {input_fasta} > {output_aligned}"
        debug_print(f"Running: {mafft_cmd}")
        result = subprocess.run(mafft_cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            progress_print(f"  ✓ Global alignment saved to {output_aligned.name}")

            # Generate consensus sequence
            progress_print("  Generating global consensus sequence...")
            alignment = AlignIO.read(output_aligned, "fasta")
            summary_align = AlignInfo.SummaryInfo(alignment)
            consensus = summary_align.dumb_consensus(threshold=0.5, ambiguous='N')
            
            # Save consensus
            consensus_file = DIRS['consensus'] / f"{FAMILY_NAME.lower()}_consensus.fa"
            with open(consensus_file, "w") as f:
                f.write(f">{FAMILY_NAME}_consensus\n")
                consensus_str = str(consensus)
                for i in range(0, len(consensus_str), 80):
                    f.write(consensus_str[i:i+80] + "\n")
            print(f"Global consensus saved to {consensus_file}")
            
            # Calculate alignment statistics
            alignment_stats_file = DIRS['alignments'] / "alignment_stats.txt"
            with open(alignment_stats_file, "w") as f:
                f.write(f"=== GLOBAL ALIGNMENT STATISTICS ===\n\n")
                f.write(f"Number of sequences: {len(alignment)}\n")
                f.write(f"Alignment length: {alignment.get_alignment_length()}\n")
                f.write(f"Consensus length (no gaps): {len(consensus_str.replace('-', ''))}\n\n")
                
                # Calculate column conservation
                gap_counts = []
                for i in range(alignment.get_alignment_length()):
                    column = alignment[:, i]
                    gap_count = column.count('-')
                    gap_counts.append(gap_count)
                
                f.write(f"Average gaps per column: {np.mean(gap_counts):.2f}\n")
                f.write(f"Columns with no gaps: {sum(1 for g in gap_counts if g == 0)}\n")
                f.write(f"Columns with >50% gaps: {sum(1 for g in gap_counts if g > len(alignment)/2)}\n")
            
            print(f"Alignment statistics saved to {alignment_stats_file}")
        else:
            print(f"MAFFT failed with error: {result.stderr}")
        
        # ----------------- Per-cluster alignments -----------------
        print("\n=== ALIGNING SEQUENCES PER CLUSTER ===")
        cluster_alignment_dir = OUT_DIR / "cluster_alignments"
        cluster_alignment_dir.mkdir(exist_ok=True)
        
        cluster_consensus_sequences = []
        
        for cluster in sorted(df2['Cluster'].unique()):
            # Process ALL clusters including noise (-1) to ensure alignments are always generated
            cluster_label = "noise" if cluster == -1 else str(cluster)
            cluster_df = df2[df2['Cluster'] == cluster]
            cluster_size = len(cluster_df)

            if cluster_size < 2:
                print(f"Cluster {cluster_label} has only {cluster_size} sequence(s), skipping alignment")
                continue

            print(f"\nProcessing Cluster {cluster_label} ({cluster_size} sequences)...")
            
            # Write cluster-specific FASTA
            cluster_fasta = cluster_alignment_dir / f"cluster_{cluster}_seqs.fa"
            with open(cluster_fasta, "w") as f:
                for rid, row in cluster_df.iterrows():
                    header = f">row{rid}|{row.get('TE_name','')[:50]}|cluster{cluster}"
                    seq = str(row["Seq"]).strip().upper()
                    f.write(header + "\n")
                    for i in range(0, len(seq), 80):
                        f.write(seq[i:i+80] + "\n")
            
            # Align cluster sequences
            cluster_aligned = cluster_alignment_dir / f"cluster_{cluster}_aligned.fa"
            mafft_cmd = f"mafft --auto --thread -1 {cluster_fasta} > {cluster_aligned}"
            result = subprocess.run(mafft_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"  Cluster {cluster} alignment saved to {cluster_aligned.name}")
                
                # Generate consensus for this cluster
                try:
                    alignment = AlignIO.read(cluster_aligned, "fasta")
                    summary_align = AlignInfo.SummaryInfo(alignment)
                    consensus = summary_align.dumb_consensus(threshold=0.5, ambiguous='N')
                    
                    # Save cluster consensus
                    cluster_consensus_file = cluster_alignment_dir / f"cluster_{cluster}_consensus.fa"
                    consensus_str = str(consensus)
                    with open(cluster_consensus_file, "w") as f:
                        f.write(f">{FAMILY_NAME}_cluster{cluster}_consensus\n")
                        for i in range(0, len(consensus_str), 80):
                            f.write(consensus_str[i:i+80] + "\n")
                    
                    print(f"  Cluster {cluster} consensus saved to {cluster_consensus_file.name}")
                    
                    # Store for summary
                    cluster_consensus_sequences.append({
                        'cluster': cluster,
                        'size': cluster_size,
                        'consensus_length': len(consensus_str.replace('-', '')),
                        'alignment_length': alignment.get_alignment_length()
                    })
                    
                    # Cluster-specific alignment stats
                    cluster_stats_file = cluster_alignment_dir / f"cluster_{cluster}_stats.txt"
                    with open(cluster_stats_file, "w") as f:
                        f.write(f"=== CLUSTER {cluster} ALIGNMENT STATISTICS ===\n\n")
                        f.write(f"Number of sequences: {len(alignment)}\n")
                        f.write(f"Alignment length: {alignment.get_alignment_length()}\n")
                        f.write(f"Consensus length (no gaps): {len(consensus_str.replace('-', ''))}\n\n")
                        
                        # Calculate column conservation
                        gap_counts = []
                        for i in range(alignment.get_alignment_length()):
                            column = alignment[:, i]
                            gap_count = column.count('-')
                            gap_counts.append(gap_count)
                        
                        f.write(f"Average gaps per column: {np.mean(gap_counts):.2f}\n")
                        f.write(f"Columns with no gaps: {sum(1 for g in gap_counts if g == 0)}\n")
                        f.write(f"Columns with >50% gaps: {sum(1 for g in gap_counts if g > len(alignment)/2)}\n")
                
                except Exception as e:
                    print(f"  Warning: Could not generate consensus for cluster {cluster}: {e}")
            else:
                print(f"  MAFFT failed for cluster {cluster}: {result.stderr}")
        
        # Save summary of all cluster consensuses
        if cluster_consensus_sequences:
            cluster_consensus_summary = pd.DataFrame(cluster_consensus_sequences)
            cluster_consensus_summary.to_csv(
                cluster_alignment_dir / "cluster_consensus_summary.csv", 
                index=False
            )
            print(f"\nCluster consensus summary saved to {cluster_alignment_dir / 'cluster_consensus_summary.csv'}")
            
            # Create a combined FASTA with all cluster consensuses (including noise cluster -1)
            all_consensus_file = cluster_alignment_dir / "all_cluster_consensuses.fa"
            with open(all_consensus_file, "w") as outf:
                for cluster in sorted(df2['Cluster'].unique()):
                    # Include ALL clusters including noise (-1)
                    consensus_file = cluster_alignment_dir / f"cluster_{cluster}_consensus.fa"
                    if consensus_file.exists():
                        with open(consensus_file, "r") as inf:
                            outf.write(inf.read())
            print(f"Combined cluster consensuses saved to {all_consensus_file}")
    
    # ==================== CIALIGN VISUALIZATION ====================
    print("\n=== GENERATING CIALIGN PLOTS ===")

    try:
        # Check if CIAlign is available
        progress_print("Checking for CIAlign installation...")
        cialign_check = subprocess.run(["CIAlign", "--version"],
                                      capture_output=True, text=True)
        cialign_available = True
        progress_print("  ✓ CIAlign found")
    except FileNotFoundError:
        progress_print("  CIAlign not found. Attempting installation...")
        try:
            subprocess.run(["pip", "install", "cialign"], check=True)
            cialign_available = True
            progress_print("  ✓ CIAlign installed successfully")
        except:
            progress_print("  ✗ Could not install CIAlign. Skipping CIAlign plots.")
            cialign_available = False

    if cialign_available and mafft_available:
        # Create CIAlign output directory
        cialign_dir = OUT_DIR / "cialign_plots"
        cialign_dir.mkdir(exist_ok=True)

        # Set matplotlib backend to Agg (headless) to avoid Jupyter/inline backend issues
        import os
        os.environ['MPLBACKEND'] = 'Agg'

        # Function to run CIAlign
        def run_cialign(input_fasta, output_stem, label):
            """Run CIAlign with standard options for visualization"""
            progress_print(f"  Running CIAlign for {label}...")

            cmd = [
                "CIAlign",
                "--infile", str(input_fasta),
                "--outfile_stem", str(output_stem),
                "--remove_insertions",
                "--remove_divergent",
                "--remove_short",
                "--plot_input",
                "--plot_output",
                "--plot_markup"
            ]

            debug_print(f"Command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True,
                                   env={**os.environ, 'MPLBACKEND': 'Agg'})

            if result.returncode == 0:
                generated = list(Path(output_stem).parent.glob(f"{Path(output_stem).name}*"))
                png_count = sum(1 for f in generated if f.suffix == '.png')
                progress_print(f"    ✓ Generated {png_count} plots for {label}")
                return True
            else:
                progress_print(f"    ✗ CIAlign failed for {label}")
                debug_print(f"Error: {result.stderr[:200] if result.stderr else 'unknown'}")
                return False
        
        # 1. Global alignment CIAlign
        if output_aligned.exists():
            global_stem = cialign_dir / "global_alignment"
            run_cialign(output_aligned, global_stem, "global alignment")
        
        # 2. Per-cluster alignments CIAlign
        print("\n=== GENERATING CIALIGN PLOTS FOR CLUSTERS ===")

        for cluster in sorted(df2['Cluster'].unique()):
            # Process ALL clusters including noise (-1) to ensure CIAlign plots are always generated
            cluster_aligned = cluster_alignment_dir / f"cluster_{cluster}_aligned.fa"
            if cluster_aligned.exists():
                cluster_stem = cialign_dir / f"cluster_{cluster}_alignment"
                run_cialign(cluster_aligned, cluster_stem, f"cluster {cluster}")
        
        # 3. Create an index HTML for CIAlign plots
        cialign_index = cialign_dir / "index.html"
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>CIAlign Alignment Plots - {FAMILY_NAME}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #4CAF50; margin-top: 30px; }}
        .plot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; margin: 20px 0; }}
        .plot-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .plot-card h3 {{ margin-top: 0; color: #333; }}
        .plot-card img {{ width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
        .plot-card a {{ display: inline-block; margin: 5px 5px 0 0; padding: 8px 16px; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 4px; font-size: 14px; }}
        .plot-card a:hover {{ background-color: #45a049; }}
    </style>
</head>
<body>
    <h1>CIAlign Alignment Visualization - {FAMILY_NAME}</h1>
    <p>Interactive alignment plots showing conservation, insertions, and sequence quality.</p>
    
    <h2>Global Alignment</h2>
    <div class="plot-grid">
"""
        
        # Add global plots
        global_plots = list(cialign_dir.glob("global_alignment*"))
        if global_plots:
            for plot_file in sorted(global_plots):
                if plot_file.suffix in ['.png', '.svg']:
                    plot_name = plot_file.stem.replace('global_alignment_', '').replace('_', ' ').title()
                    html_content += f"""
        <div class="plot-card">
            <h3>{plot_name}</h3>
            <img src="{plot_file.name}" alt="{plot_name}">
            <a href="{plot_file.name}" download>Download</a>
        </div>
"""
        else:
            html_content += "<p>No global alignment plots generated.</p>"
        
        html_content += """
    </div>
    
    <h2>Per-Cluster Alignments</h2>
"""
        
        # Add cluster plots
        clusters_with_plots = sorted(set([
            int(f.name.split('_')[1]) 
            for f in cialign_dir.glob("cluster_*_alignment*") 
            if f.suffix in ['.png', '.svg']
        ]))
        
        for cluster in clusters_with_plots:
            html_content += f"""
    <h3>Cluster {cluster}</h3>
    <div class="plot-grid">
"""
            cluster_plots = sorted(cialign_dir.glob(f"cluster_{cluster}_alignment*"))
            for plot_file in cluster_plots:
                if plot_file.suffix in ['.png', '.svg']:
                    plot_name = plot_file.stem.replace(f'cluster_{cluster}_alignment_', '').replace('_', ' ').title()
                    html_content += f"""
        <div class="plot-card">
            <h3>{plot_name}</h3>
            <img src="{plot_file.name}" alt="{plot_name}">
            <a href="{plot_file.name}" download>Download</a>
        </div>
"""
            html_content += """
    </div>
"""
        
        html_content += """
    <div style="margin-top: 40px; padding: 20px; background: white; border-radius: 8px;">
        <h2>About These Plots</h2>
        <p>Generated with CIAlign - a tool for cleaning and visualizing multiple sequence alignments.</p>
        <ul>
            <li><strong>Input plot</strong>: Shows the original alignment before cleaning</li>
            <li><strong>Output plot</strong>: Shows the cleaned alignment after removing insertions, divergent sequences, and short sequences</li>
            <li><strong>Markup plot</strong>: Highlights regions that were removed during cleaning</li>
        </ul>
        <p><a href="https://github.com/KatyBrown/CIAlign" target="_blank">CIAlign GitHub</a></p>
    </div>
</body>
</html>
"""
        
        with open(cialign_index, 'w') as f:
            f.write(html_content)
        
        print(f"\n✓ CIAlign visualization index created: {cialign_index}")
        print(f"  Open {cialign_index} to view all alignment plots")
        
        # ==================== EXTRACT CLEANED CONSENSUS SEQUENCES ====================
        print("\n=== GENERATING CLEANED CONSENSUS SEQUENCES FROM CIALIGN ===")
        
        cleaned_consensus_dir = OUT_DIR / "cleaned_consensus"
        cleaned_consensus_dir.mkdir(exist_ok=True)
        
        def generate_consensus_from_cleaned(cleaned_fasta, output_fasta, label):
            """Generate consensus from CIAlign cleaned alignment"""
            try:
                from Bio import AlignIO
                from Bio.Align import AlignInfo
                
                if not cleaned_fasta.exists():
                    print(f"  ⚠ Cleaned alignment not found: {cleaned_fasta.name}")
                    return None
                
                alignment = AlignIO.read(cleaned_fasta, "fasta")
                summary_align = AlignInfo.SummaryInfo(alignment)
                consensus = summary_align.dumb_consensus(threshold=0.5, ambiguous='N')
                consensus_str = str(consensus)
                
                # Save consensus
                with open(output_fasta, "w") as f:
                    f.write(f">{label}_cleaned_consensus\n")
                    for i in range(0, len(consensus_str), 80):
                        f.write(consensus_str[i:i+80] + "\n")
                
                print(f"  ✓ Generated {label} cleaned consensus: {len(consensus_str.replace('-', ''))} bp")
                return consensus_str
            
            except Exception as e:
                print(f"  ✗ Failed to generate consensus for {label}: {e}")
                return None
        
        # Global cleaned consensus
        global_cleaned = cialign_dir / "global_alignment_cleaned.fasta"
        if global_cleaned.exists():
            global_cleaned_consensus = cleaned_consensus_dir / f"{FAMILY_NAME.lower()}_cleaned_consensus.fa"
            generate_consensus_from_cleaned(global_cleaned, global_cleaned_consensus, FAMILY_NAME)
        
        # Per-cluster cleaned consensus
        all_cleaned_consensuses = []
        for cluster in sorted(df2['Cluster'].unique()):
            # Process ALL clusters including noise (-1) to ensure cleaned consensus is always generated
            cluster_cleaned = cialign_dir / f"cluster_{cluster}_alignment_cleaned.fasta"
            if cluster_cleaned.exists():
                cluster_cleaned_consensus = cleaned_consensus_dir / f"cluster_{cluster}_cleaned_consensus.fa"
                consensus_str = generate_consensus_from_cleaned(
                    cluster_cleaned, 
                    cluster_cleaned_consensus,
                    f"{FAMILY_NAME}_cluster{cluster}"
                )
                
                if consensus_str:
                    all_cleaned_consensuses.append({
                        'cluster': cluster,
                        'consensus_seq': consensus_str,
                        'length_no_gaps': len(consensus_str.replace('-', '')),
                        'length_with_gaps': len(consensus_str)
                    })
        
        # Create combined file with all cleaned cluster consensuses
        if all_cleaned_consensuses:
            all_cleaned_file = cleaned_consensus_dir / "all_clusters_cleaned_consensus.fa"
            with open(all_cleaned_file, "w") as f:
                for item in all_cleaned_consensuses:
                    cluster = item['cluster']
                    consensus_str = item['consensus_seq']
                    f.write(f">{FAMILY_NAME}_cluster{cluster}_cleaned_consensus\n")
                    for i in range(0, len(consensus_str), 80):
                        f.write(consensus_str[i:i+80] + "\n")
            print(f"\n✓ Combined cleaned consensuses: {all_cleaned_file}")
            
            # Save summary table
            summary_df = pd.DataFrame([{
                'cluster': item['cluster'],
                'length_no_gaps': item['length_no_gaps'],
                'length_with_gaps': item['length_with_gaps']
            } for item in all_cleaned_consensuses])
            
            summary_file = cleaned_consensus_dir / "cleaned_consensus_summary.csv"
            summary_df.to_csv(summary_file, index=False)
            print(f"✓ Cleaned consensus summary: {summary_file}")
    
    elif cialign_available and not mafft_available:
        print("CIAlign available but no alignments to visualize (MAFFT not run)")
    
except ImportError as e:
    print(f"Warning: BioPython not available. Skipping alignment. Error: {e}")
    print("Install with: conda install -y -c conda-forge biopython")

# ----------------- Save additional outputs -----------------
if 'kmer_df' in dir() and len(kmer_df) > 0:
    kmer_out = kmer_df.copy()
    kmer_out["rows_covered"] = kmer_out["rows"].apply(rows_set_to_str)
    kmer_out["rows_coordinates"] = kmer_out["rows"].apply(lambda rs: rows_set_to_coords(rs, df2))
    kmer_out.drop(columns=["rows"], inplace=True)
    kmer_out.to_csv(DIRS['primers'] / "kmer_candidate_metrics_full.csv", index=False)

if primer_hits:
    primer_hits_summary = []
    for p, dfh in primer_hits.items():
        # Check if results were estimated (has estimated_total column)
        is_estimated = "estimated_total" in dfh.columns
        estimated_total = dfh["estimated_total"].iloc[0] if is_estimated and len(dfh) > 0 else None
        primer_hits_summary.append({
            "primer": p,
            "genome_hits": len(dfh),
            "estimated": is_estimated,
            "estimated_total": estimated_total if is_estimated else len(dfh)
        })
    primer_hits_summary_df = pd.DataFrame(primer_hits_summary).sort_values("genome_hits", ascending=False)
    primer_hits_summary_df.to_csv(DIRS['primers'] / "primer_genome_hits_summary.csv", index=False)

_record_stage("Alignment & CIAlign", _pipeline_time.time() - _stage_t0)

# Check for any errors that occurred
error_log_path = OUT_DIR / "pipeline_errors.log"
errors_occurred = error_log_path.exists() and error_log_path.stat().st_size > 0

_total_elapsed = _pipeline_time.time() - _pipeline_start

print("\n" + "="*60)
if errors_occurred:
    print("PIPELINE COMPLETED WITH WARNINGS/ERRORS")
else:
    print("PIPELINE COMPLETED SUCCESSFULLY")
print("="*60)
print(f" - Family analyzed: {FAMILY_NAME}")
print(f" - All outputs in: {OUT_DIR.resolve()}")
print(f" - Number of sequences: {len(df2) if 'df2' in dir() else 'N/A'}")
try:
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f" - Number of clusters: {n_clusters}")
except:
    print(f" - Number of clusters: N/A")
print(f" - Unique {K}-mers found: {len(kmer_to_rows) if 'kmer_to_rows' in dir() else 'N/A'}")
print(f" - Top primers selected: {len(selected_primers) if 'selected_primers' in dir() else 'N/A'}")

# ==================== TIMING SUMMARY ====================
print(f"\n{'='*60}")
print(f"TIMING SUMMARY (total: {_total_elapsed:.1f}s = {_total_elapsed/60:.1f}min)")
print(f"{'='*60}")
for stage_name, elapsed in _stage_times.items():
    pct = (elapsed / _total_elapsed * 100) if _total_elapsed > 0 else 0
    bar_len = int(pct / 2)
    bar = '#' * bar_len + '.' * (50 - bar_len)
    print(f"  {stage_name:<25s} {elapsed:7.1f}s ({pct:5.1f}%) |{bar}|")
print(f"{'='*60}")

if errors_occurred:
    print("\n*** WARNINGS/ERRORS OCCURRED ***")
    print(f"Check error log: {error_log_path}")
    try:
        with open(error_log_path, "r") as f:
            error_content = f.read()
            import re
            error_stages = re.findall(r'ERROR in stage: (.+)', error_content)
            if error_stages:
                print("Stages with errors:")
                for stage in set(error_stages):
                    print(f"  - {stage}")
    except:
        pass
    print("=" * 60)

print("\nKEY OUTPUT FILES:")
print(f" - DASHBOARD: 07_visualizations/index.html (OPEN THIS FIRST!)")
print(f" - CIALIGN PLOTS: cialign_plots/index.html (Alignment visualizations)")
print(f" - Overall statistics: 02_statistics/overall_statistics.txt")
print(f" - Cluster statistics: 02_statistics/per_cluster/ directory")
print(f" - Cluster comparison: 02_statistics/cluster_comparison.txt")
print(f" - Clustering visualization: 03_clustering/clustering_visualization.html")
print(f" - Selected primers: 06_primers/selected_primers_summary.csv")
print(f" - Cluster-specific primers: 06_primers/cluster_top5_primers.csv")
print(f" - FASTA sequences: {FAMILY_NAME.lower()}_seqs.fa")
print(f" - Genome hits summary: 06_primers/primer_genome_hits_summary.csv")
print(f" - Global alignment: 04_alignments/{FAMILY_NAME.lower()}_aligned.fa")
print(f" - Global consensus: 05_consensus/{FAMILY_NAME.lower()}_consensus.fa")
print(f" - Cleaned consensus (CIAlign): 05_consensus/cleaned_consensus/ directory")
print(f" - Cluster alignments: cluster_alignments/ directory")
print(f" - All cluster consensuses: cluster_alignments/all_cluster_consensuses.fa")

if errors_occurred:
    print(f"\n*** ERROR LOG: {error_log_path} ***")
print("="*60)

 
