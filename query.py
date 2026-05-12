#!/usr/bin/env python3
"""
query.py — TE Analysis Pipeline Orchestrator
─────────────────────────────────────────────────────────────────────────────
Stages:
  1. Load reference genome (GenomeCache)
  2. Load & filter TE data
  3. Extract sequences (local genome or parallel UCSC API)
  4. Basic statistics (Cython-accelerated when te_fast is compiled)
  5. Clustering (UMAP / HDBSCAN) → te_clustering.py
  6. Visualization dashboard
  7. Primer design                → te_primers.py
  8. Alignment + CIAlign          → te_alignment.py

LOCAL MODE (no HPC, no file paths needed):
  python query.py --local --family THE1D-int --assembly hg38
  python query.py --local --family HERVK9 --assembly hg38 --genome /path/hg38.fa
  python query.py --local --family THE1D-int --assembly hg38 --max-loci 100

HPC MODE (default):
  python query.py --input te_data.csv --family HERVK9 --genome /path/hg38.fa
  python query.py --test

Speed options:
  --fetch-workers 20      # more parallel UCSC workers (default 10)
  --parallel-primers      # parallel genome-wide primer search
  python setup_cython.py build_ext --inplace   # compile Cython for batch ops
"""

import argparse
import datetime
import ast
import os
import re
import sys
import time
import traceback
from pathlib import Path

# ── HPC-only guard ────────────────────────────────────────────────────────────
# query.py is designed to run on a SLURM or LSF cluster.  Bypass with --local
# for local execution or --test for CI/unit-test mode.
def _check_hpc_environment():
    on_slurm = bool(os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_NODELIST"))
    on_lsf   = bool(os.environ.get("LSB_JOBID")    or os.environ.get("LSB_HOSTS"))
    if on_slurm or on_lsf:
        return
    if "--test" in sys.argv or "--local" in sys.argv or "--help" in sys.argv or "-h" in sys.argv:
        return
    print(
        "\n"
        "  ╔══════════════════════════════════════════════════════════════╗\n"
        "  ║  ERROR: query.py must run on an HPC cluster                  ║\n"
        "  ║                                                              ║\n"
        "  ║  No SLURM or LSF environment detected.                       ║\n"
        "  ║                                                              ║\n"
        "  ║  Run locally (auto-downloads rmsk, fetches seqs via UCSC):   ║\n"
        "  ║    python query.py --local --family HERVK9 --assembly hg38  ║\n"
        "  ║                                                              ║\n"
        "  ║  Or submit via HPC:                                          ║\n"
        "  ║    python ui.py --host <cluster> --user <username>           ║\n"
        "  ╚══════════════════════════════════════════════════════════════╝\n",
        file=sys.stderr,
    )
    sys.exit(1)

# Fix matplotlib backend before any plotting imports
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
import requests

# Pipeline modules
from te_genome import GenomeCache, reverse_complement
from te_clustering import clustering_analysis
from te_primers import design_primers
from te_alignment import run_alignment_pipeline
from te_motif import run_motif_analysis
from te_go import run_go_annotation

# ═══════════════════════════════════════════════════════════════════════════
# CLI / CONFIG
# ═══════════════════════════════════════════════════════════════════════════

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="TE Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Local mode (no HPC, no file paths needed):\n"
            "  python query.py --local --family THE1D-int --assembly hg38\n"
            "  python query.py --local --family HERVK9 --assembly hg38 --genome /path/hg38.fa\n"
        ),
    )
    p.add_argument("--input",    type=str, default=None, help="Input CSV file")
    p.add_argument("--family",   type=str, default="HERVK9", help="TE family repName")
    p.add_argument("--genome",   type=str, default=None,
                   help="Path to genome FASTA (enables local extraction + primer search)")
    p.add_argument("--output",   type=str, default="results",
                   help="Base output directory (default: results)")
    p.add_argument("--kmer",     type=int, default=6,
                   help="K-mer size for clustering (default: 6; use --primer-kmer for primers)")
    p.add_argument("--pca-dims", type=int, default=50,
                   help="SVD components fed into UMAP/t-SNE clustering (default: 50)")
    p.add_argument("--n-epochs", type=int, default=200,
                   help="UMAP optimisation epochs for clustering (default: 200)")
    p.add_argument("--random-state", type=int, default=42,
                   help="Clustering random seed; pass 0 to enable multicore UMAP")
    p.add_argument("--skip-tsne", action="store_true",
                   help="Skip t-SNE during clustering for faster UMAP/PCA pipeline runs")
    p.add_argument("--primer-kmer",  type=int, default=18, help="Primer k-mer size")
    p.add_argument("--top-global",   type=int, default=8)
    p.add_argument("--top-cluster",  type=int, default=5)
    p.add_argument("--min-sequences",type=int, default=10,
                   help="Minimum sequences needed for clustering (default: 10)")
    p.add_argument("--primer-timeout", type=int, default=120,
                   help="Timeout for primer genome search in seconds (default: 120)")
    # ── Local mode ─────────────────────────────────────────────────────────────
    p.add_argument("--local",    action="store_true",
                   help="Run locally: auto-download rmsk, fetch seqs via UCSC API")
    p.add_argument("--assembly", type=str, default="hg38",
                   help="Genome assembly for local mode: hg38, hg19, mm10, mm39 (default: hg38)")
    p.add_argument("--rmsk-dir", type=str, default=None,
                   help="Directory for rmsk cache files (default: ~/te_analysis/rmsk)")
    p.add_argument("--max-loci", type=int, default=None,
                   help="Cap number of TE loci (useful for quick tests)")
    p.add_argument("--expression-assembly", type=str, default=None,
                   help="Optional CSV/TSV/BED-like expression assembly with chr/start/stop columns")
    p.add_argument("--expression-buffer", type=int, default=50,
                   help="Expand expression start/stop by this many bp when matching TE loci (default: 50)")
    p.add_argument("--fetch-workers", type=int, default=10,
                   help="Parallel UCSC fetch workers (default: 10)")
    p.add_argument("--parallel-primers", action="store_true",
                   help="Parallelize primer genome search across chromosomes")
    # ── Misc ───────────────────────────────────────────────────────────────────
    p.add_argument("--test",          action="store_true", help="Run with mock test data")
    p.add_argument("--debug",         action="store_true")
    p.add_argument("--skip-genome",   action="store_true", help="Skip genome-wide primer search")
    p.add_argument("--skip-alignment",action="store_true")
    p.add_argument("--skip-primers",  action="store_true")
    p.add_argument("--skip-motif",    action="store_true",
                   help="Skip JASPAR motif / TF binding analysis")
    p.add_argument("--skip-go",       action="store_true",
                   help="Skip GO annotation plots")
    p.add_argument("--jaspar-bed",    type=str, default=None,
                   help="Path to pre-downloaded JASPAR BED")
    p.add_argument("--jaspar-dir",    type=str, default=None,
                   help="Directory for cached/downloaded JASPAR BED files")
    p.add_argument("--p-threshold",   type=float, default=0.05,
                   help="P-value threshold for motif/GO reporting")
    p.add_argument("--resume-from", choices=["sequences", "stats", "clustering", "dashboard", "primers", "alignment", "motif", "tfbs", "go"],
                   default=None, help="Resume an existing output folder from this stage")
    p.add_argument("--validate-existing", action="store_true",
                   help="Report which expected output files already exist, then continue/resume")
    return p.parse_args(argv)


# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def progress_print(msg, newline=True):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    if newline:
        print(f"[{ts}] {msg}", flush=True)
    else:
        print(f"[{ts}] {msg}", end="", flush=True)


def progress_bar(cur, tot, prefix="Progress", length=40):
    pct = cur / tot if tot else 0
    filled = int(length * pct)
    bar = "█" * filled + "░" * (length - filled)
    print(f"\r{prefix} |{bar}| {cur}/{tot} ({pct*100:.1f}%)", end="", flush=True)
    if cur >= tot:
        print()


def _row_strand(row):
    """Return '+' or '-' from explicit strand fields or terminal TE_ID tokens."""
    for col in ("strand", "Strand"):
        if col in row and pd.notna(row.get(col)):
            val = str(row.get(col)).strip()
            if val in {"+", "-"}:
                return val

    for col in ("TE_ID", "TE_name", "repName", "name", "Name", "id", "ID"):
        if col not in row or pd.isna(row.get(col)):
            continue
        text = str(row.get(col)).strip()
        if not text:
            continue
        # te_prep IDs end with "|+" or "|-"; tolerate colon/space/comma too.
        m = re.search(r"(?:^|[|:,\s])([+-])$", text)
        if m:
            return m.group(1)
    return "+"


def _orient_sequence(seq, row):
    """Orient a sequence to the TE strand requested by row metadata."""
    seq = str(seq).upper()
    return reverse_complement(seq).upper() if _row_strand(row) == "-" else seq


def log_error(stage, err, context=None):
    print("\n" + "=" * 70, flush=True)
    print(f"ERROR in stage: {stage}", flush=True)
    print(f"  {type(err).__name__}: {err}", flush=True)
    if context:
        for k, v in context.items():
            s = str(v)
            print(f"  {k}: {s[:200]}", flush=True)
    traceback.print_exc()
    print("=" * 70 + "\n", flush=True)

    # Also write to pipeline_errors.log
    try:
        if "OUT_DIR" in globals() and OUT_DIR.exists():
            with open(OUT_DIR / "pipeline_errors.log", "a") as f:
                f.write(f"\n{'='*70}\nERROR: {stage}\n{type(err).__name__}: {err}\n")
                f.write(traceback.format_exc())
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# TEST DATA
# ═══════════════════════════════════════════════════════════════════════════

def create_test_data():
    import random as _r
    _r.seed(42); np.random.seed(42)

    def _seq(l, gc=0.5):
        return "".join(
            _r.choice("GC") if _r.random() < gc else _r.choice("AT")
            for _ in range(l)
        )

    def _mutate(s, rate=0.1):
        bases = list(s)
        for i in range(len(bases)):
            if _r.random() < rate:
                bases[i] = _r.choice("ACGT")
        return "".join(bases)

    b1 = _seq(500, 0.45); b2 = _seq(480, 0.55)
    rows = []
    for i in range(8):
        s = _mutate(b1, 0.05 + i * 0.01)
        rows.append({"chr": f"chr{(i%5)+1}", "start": 1_000_000 + i*10000,
                     "stop": 1_000_000 + i*10000 + len(s),
                     "TE_name": f"TEST_TE_element_{i+1}", "strand": "+" if i%2==0 else "-",
                     "Seq": s,
                     "A1_siCTRL_r1": np.random.uniform(0, 50),
                     "A1_siKD_r1": np.random.uniform(0, 30)})
    for i in range(7):
        s = _mutate(b2, 0.05 + i * 0.01)
        rows.append({"chr": f"chr{(i%5)+6}", "start": 2_000_000 + i*10000,
                     "stop": 2_000_000 + i*10000 + len(s),
                     "TE_name": f"TEST_TE_element_{i+9}", "strand": "+" if i%2==0 else "-",
                     "Seq": s,
                     "A1_siCTRL_r1": np.random.uniform(20, 100),
                     "A1_siKD_r1": np.random.uniform(10, 50)})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# LOCAL MODE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _load_local_data(args):
    """Download rmsk (if needed), parse family loci, return DataFrame with coords.

    Called when --local is set and --input is not provided.
    """
    from te_prep import download_rmsk, get_rmsk_path, parse_rmsk_family
    from pathlib import Path

    assembly = args.assembly
    family   = args.family
    rmsk_dir = args.rmsk_dir or str(Path.home() / "te_analysis" / "rmsk")

    progress_print(f"LOCAL MODE: {family} on {assembly}")

    # Auto-download rmsk if not present
    rmsk_path_candidate = Path(rmsk_dir) / f"rmsk_{assembly}.txt.gz"
    if not rmsk_path_candidate.exists():
        progress_print(f"  rmsk_{assembly}.txt.gz not found — downloading (~150 MB, one-time)...")
        download_rmsk(assembly, rmsk_dir)
    else:
        progress_print(f"  rmsk found: {rmsk_path_candidate}")

    rmsk_path = get_rmsk_path(assembly, rmsk_dir)

    # Standard chromosomes filter
    if assembly.startswith("hg"):
        std = set([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"])
    else:
        std = set([f"chr{i}" for i in range(1, 20)] + ["chrX", "chrY"])

    progress_print(f"  Parsing rmsk for '{family}'...")
    hits = parse_rmsk_family(rmsk_path, family, std_chroms=std)

    if not hits:
        print(f"ERROR: No loci found for repName='{family}' in {assembly}.")
        print(f"  repName is case-sensitive. Search families with:")
        print(f"    python te_prep.py --search {family.lower()} --build {assembly}")
        sys.exit(1)

    df = pd.DataFrame(hits)
    df["TE_name"] = df["repName"]
    df["chr"]     = df["Chromosome"]
    df["start"]   = df["Start"]
    df["stop"]    = df["Stop"]

    if args.max_loci and len(df) > args.max_loci:
        progress_print(f"  Capping to {args.max_loci} loci (--max-loci)")
        df = df.head(args.max_loci).reset_index(drop=True)

    progress_print(f"  {len(df):,} loci across {df['chr'].nunique()} chromosomes")
    if getattr(args, "expression_assembly", None):
        df = _attach_expression_assembly(
            df,
            args.expression_assembly,
            buffer_bp=getattr(args, "expression_buffer", 50),
        )
    return df.reset_index(drop=True)


def _detect_coord_columns(df, label="table"):
    """Detect chr/start/stop columns across common BED/CSV spellings."""
    def _pick(candidates):
        lower = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand.lower() in lower:
                return lower[cand.lower()]
        return None

    chr_col = _pick(["chr", "chrom", "chromosome", "#chrom", "seqnames"])
    start_col = _pick(["start", "chromStart", "txStart", "begin"])
    stop_col = _pick(["stop", "end", "chromEnd", "txEnd"])
    if not all([chr_col, start_col, stop_col]):
        raise ValueError(
            f"Could not detect coordinate columns in {label}. "
            f"Need chr/start/stop-like columns; found {list(df.columns)}"
        )
    return chr_col, start_col, stop_col


def _read_expression_assembly(path):
    """Read expression assembly as CSV/TSV/BED-like table."""
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Expression assembly not found: {path}")

    sep = "\t" if path.suffix.lower() in {".tsv", ".bed"} else None
    try:
        kwargs = {"sep": sep}
        if sep is None:
            kwargs["engine"] = "python"
        df = pd.read_csv(path, **kwargs)
    except Exception:
        df = pd.read_csv(path, sep="\t", header=None)

    # Headerless BED fallback.
    if all(isinstance(c, int) for c in df.columns) and df.shape[1] >= 3:
        cols = ["chr", "start", "stop"] + [f"expr_field_{i}" for i in range(4, df.shape[1] + 1)]
        df.columns = cols[:df.shape[1]]
    return df


def _attach_expression_assembly(df_te, expression_path, buffer_bp=50):
    """Attach expression assembly rows to RepeatMasker loci using buffered overlap.

    Expression intervals are expanded by +/- buffer_bp before overlap testing:
      TE_start <= expr_stop + buffer AND TE_stop >= expr_start - buffer

    Numeric expression columns are summed across all matching expression rows and
    prefixed with expr_.  Extra metadata summarizes match count and matched IDs.
    """
    progress_print(
        f"  Loading expression assembly: {expression_path} "
        f"(buffer +/- {buffer_bp:,} bp)"
    )
    df_expr = _read_expression_assembly(expression_path)
    te_chr, te_start, te_stop = _detect_coord_columns(df_te, "RepeatMasker loci")
    ex_chr, ex_start, ex_stop = _detect_coord_columns(df_expr, "expression assembly")

    df_te = df_te.copy().reset_index(drop=True)
    df_te["_te_row_id"] = np.arange(len(df_te), dtype=int)
    df_expr = df_expr.copy().reset_index(drop=True)
    df_expr["_expr_row_id"] = np.arange(len(df_expr), dtype=int)

    for col in (te_start, te_stop):
        df_te[col] = pd.to_numeric(df_te[col], errors="coerce")
    for col in (ex_start, ex_stop):
        df_expr[col] = pd.to_numeric(df_expr[col], errors="coerce")

    df_te = df_te.dropna(subset=[te_chr, te_start, te_stop]).copy()
    df_expr = df_expr.dropna(subset=[ex_chr, ex_start, ex_stop]).copy()
    df_te[te_chr] = df_te[te_chr].astype(str)
    df_expr[ex_chr] = df_expr[ex_chr].astype(str)

    expr_exclude = {ex_chr, ex_start, ex_stop, "_expr_row_id", "Seq", "sequence", "Unnamed: 0"}
    numeric_expr_cols = [
        c for c in df_expr.columns
        if c not in expr_exclude
        and not str(c).lower().startswith("unnamed")
        and pd.api.types.is_numeric_dtype(df_expr[c])
    ]

    def _listlike_count(v):
        if pd.isna(v):
            return 0
        if isinstance(v, (list, tuple, set)):
            return len(v)
        s = str(v).strip()
        if not s or s.lower() == "nan":
            return 0
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple, set)):
                    return len(parsed)
            except Exception:
                return 0
        return np.nan

    derived_expr_cols = []
    for c in list(df_expr.columns):
        if c in expr_exclude or str(c).lower().startswith("unnamed"):
            continue
        if c in numeric_expr_cols:
            continue
        counts = df_expr[c].map(_listlike_count)
        if counts.notna().any() and counts.fillna(0).sum() > 0:
            safe = re.sub(r"[^0-9A-Za-z_]+", "_", str(c)).strip("_")
            new_col = f"{safe}_count"
            df_expr[new_col] = counts.fillna(0).astype(float)
            derived_expr_cols.append(new_col)
    numeric_expr_cols.extend(derived_expr_cols)

    if not numeric_expr_cols:
        for c in df_expr.columns:
            if c in expr_exclude or str(c).lower().startswith("unnamed"):
                continue
            converted = pd.to_numeric(df_expr[c], errors="coerce")
            if converted.notna().any():
                df_expr[c] = converted
                numeric_expr_cols.append(c)

    id_col = next(
        (c for c in ["TE_ID", "name", "Name", "gene", "gene_id", "transcript_id", "id", "ID"]
         if c in df_expr.columns),
        None,
    )

    df_te["expr_match_count"] = 0
    df_te["expr_overlap_bp"] = 0
    df_te["expr_ids"] = ""
    seq_col = next((c for c in ["Seq", "seq", "sequence"] if c in df_expr.columns), None)
    if seq_col and "Seq" not in df_te.columns:
        df_te["Seq"] = pd.NA
        df_te["expr_seq_source"] = ""
    for col in numeric_expr_cols:
        out_col = f"expr_{col}" if not str(col).startswith("expr_") else str(col)
        if out_col not in df_te.columns:
            df_te[out_col] = 0.0

    total_matches = 0
    matched_te = set()
    buffer_bp = max(0, int(buffer_bp or 0))

    for chrom, te_chr_df in df_te.groupby(te_chr, sort=False):
        expr_chr_df = df_expr[df_expr[ex_chr] == chrom]
        if expr_chr_df.empty:
            continue

        expr_starts = expr_chr_df[ex_start].astype(int).to_numpy() - buffer_bp
        expr_stops = expr_chr_df[ex_stop].astype(int).to_numpy() + buffer_bp
        expr_starts = np.maximum(expr_starts, 0)

        for te_idx, te_row in te_chr_df.iterrows():
            ts = int(te_row[te_start])
            te = int(te_row[te_stop])
            mask = (ts <= expr_stops) & (te >= expr_starts)
            if not mask.any():
                continue

            hits = expr_chr_df.loc[mask]
            n_hits = len(hits)
            total_matches += n_hits
            matched_te.add(te_idx)
            df_te.at[te_idx, "expr_match_count"] = n_hits

            overlap_starts = np.maximum(ts, expr_starts[mask])
            overlap_stops = np.minimum(te, expr_stops[mask])
            df_te.at[te_idx, "expr_overlap_bp"] = int(np.maximum(0, overlap_stops - overlap_starts).sum())

            if id_col:
                vals = [str(v) for v in hits[id_col].dropna().astype(str).unique()[:8]]
                df_te.at[te_idx, "expr_ids"] = ";".join(vals)

            if seq_col:
                seq_hits = hits[seq_col].dropna().astype(str)
                seq_hits = seq_hits[seq_hits.str.len() > 0]
                if not seq_hits.empty:
                    seq = seq_hits.iloc[0].upper()
                    if set(seq) <= set("ACGTNacgtn"):
                        df_te.at[te_idx, "Seq"] = seq
                        df_te.at[te_idx, "expr_seq_source"] = str(hits.iloc[0].get(id_col, hits.iloc[0]["_expr_row_id"]))

            for col in numeric_expr_cols:
                out_col = f"expr_{col}" if not str(col).startswith("expr_") else str(col)
                df_te.at[te_idx, out_col] = float(pd.to_numeric(hits[col], errors="coerce").fillna(0).sum())

    progress_print(
        f"  Expression overlap: {len(matched_te):,}/{len(df_te):,} TE loci matched "
        f"({total_matches:,} expression interval overlaps)"
    )
    if numeric_expr_cols:
        progress_print(
            f"  Attached expression columns: "
            f"{', '.join([('expr_' + str(c)) if not str(c).startswith('expr_') else str(c) for c in numeric_expr_cols])}"
        )
    else:
        progress_print("  No numeric expression columns found; attached match metadata only")
    if seq_col and "Seq" in df_te.columns:
        progress_print(f"  Expression assembly supplied sequences for {df_te['Seq'].notna().sum():,} TE loci")

    return df_te.drop(columns=["_te_row_id"], errors="ignore")


def _fetch_sequences_parallel(df, assembly="hg38", n_workers=10):
    """Fetch sequences from UCSC API in parallel using ThreadPoolExecutor.

    ~10x faster than the original sequential loop for large families.
    Returns list of sequences (same order as df rows).
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    n = len(df)
    seqs = [None] * n
    failed = []
    lock = threading.Lock()
    # Semaphore to respect UCSC rate limits (~10 concurrent OK)
    sem = threading.Semaphore(n_workers)

    def _fetch_one(i):
        chrom = df["chr"].iloc[i]
        start = int(df["start"].iloc[i])
        stop  = int(df["stop"].iloc[i])
        url = (
            f"https://api.genome.ucsc.edu/getData/sequence?"
            f"genome={assembly};chrom={chrom};start={start};end={stop}"
        )
        retries = 3
        for attempt in range(retries):
            try:
                with sem:
                    r = requests.get(url, timeout=30)
                    r.raise_for_status()
                    res = r.json()
                    if "error" in res:
                        raise ValueError(res["error"])
                    return i, _orient_sequence(res["dna"], df.iloc[i])
            except Exception as exc:
                if attempt == retries - 1:
                    return i, None
                time.sleep(0.5 * (attempt + 1))
        return i, None

    progress_print(f"  Fetching {n} sequences in parallel ({n_workers} workers)...")
    done = 0

    with ThreadPoolExecutor(max_workers=n_workers) as exe:
        futures = {exe.submit(_fetch_one, i): i for i in range(n)}
        for fut in as_completed(futures):
            i, seq = fut.result()
            if seq is None:
                failed.append(i)
                seqs[i] = "N" * 100
            else:
                seqs[i] = seq
            done += 1
            progress_bar(done, n, "  Fetching")

    if failed:
        progress_print(f"  {len(failed)} sequences failed to fetch (filled with Ns)")
    return seqs


def _fill_missing_sequences_parallel(df, assembly="hg38", n_workers=10):
    """Fetch only rows with missing Seq, preserving any sequences already present."""
    if "Seq" not in df.columns:
        return _fetch_sequences_parallel(df, assembly=assembly, n_workers=n_workers)

    existing = df["Seq"].copy()
    valid = existing.notna() & existing.astype(str).str.len().gt(0)
    missing_idx = df.index[~valid].tolist()
    if not missing_idx:
        progress_print("All sequences already present — skipping fetch")
        return existing.astype(str).str.upper().tolist()

    progress_print(
        f"Preserving {int(valid.sum()):,} supplied sequences; "
        f"fetching {len(missing_idx):,} missing sequence(s)"
    )
    sub = df.loc[missing_idx].reset_index(drop=True)
    fetched = _fetch_sequences_parallel(sub, assembly=assembly, n_workers=n_workers)
    seqs = existing.astype("object").tolist()
    for original_idx, seq in zip(missing_idx, fetched):
        seqs[original_idx] = seq
    return [str(s).upper() if pd.notna(s) else "N" * 100 for s in seqs]


# ═══════════════════════════════════════════════════════════════════════════
# SQL-BACKED STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

def _cluster_summary_sql(df):
    """Per-cluster stats via SQLite — replaces pandas groupby for large datasets.

    Returns a DataFrame: cluster, size, mean_length, mean_gc, min_length, max_length.
    """
    import sqlite3

    conn = sqlite3.connect(":memory:")
    seqs = df["Seq"].astype(str).tolist()

    # Build a compact stats table without pulling Seq strings into SQL
    rows = []
    for i, row in df.iterrows():
        s = row["Seq"] if isinstance(row["Seq"], str) else str(row["Seq"])
        ln = len(s)
        gc = (s.count("G") + s.count("C")) / ln if ln > 0 else 0.0
        rows.append((int(row["Cluster"]), ln, gc))

    conn.execute("CREATE TABLE seq_stats (cluster INTEGER, length INTEGER, gc REAL)")
    conn.executemany("INSERT INTO seq_stats VALUES (?, ?, ?)", rows)
    conn.commit()

    result = conn.execute("""
        SELECT
            cluster,
            COUNT(*)        AS size,
            AVG(length)     AS mean_length,
            AVG(gc)         AS mean_gc,
            MIN(length)     AS min_length,
            MAX(length)     AS max_length
        FROM seq_stats
        GROUP BY cluster
        ORDER BY cluster
    """).fetchall()
    conn.close()

    return pd.DataFrame(result,
                        columns=["cluster", "size", "mean_length", "mean_gc",
                                 "min_length", "max_length"])


# ═══════════════════════════════════════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_basic_stats(df, label="", output_file=None):
    seqs = df["Seq"].astype(str)
    seqs_list = seqs.tolist()

    # Use Cython batch functions if available, otherwise numpy
    try:
        import te_fast as _tf
        lengths = _tf.batch_lengths(seqs_list)
        gc = _tf.batch_gc_content(seqs_list)
    except ImportError:
        import numpy as _np
        lengths = seqs.apply(len).values
        gc = seqs.apply(
            lambda s: (s.count("G") + s.count("C")) / len(s) if len(s) > 0 else np.nan
        ).values

    numeric = list(df.select_dtypes(include=[np.number]).columns)
    exclude = {"start", "stop", "Unnamed: 0", "chr", "Cluster", "_total_expr"}
    expr_cols = [c for c in numeric if c not in exclude]

    lines = [
        f"{'='*60}", f"STATISTICS{label}", f"{'='*60}",
        f"Dataset size: {len(df)} sequences",
        "", "=== SEQUENCE LENGTH ===",
        f"Mean: {np.mean(lengths):.1f}  Median: {np.median(lengths):.1f}",
        f"Std:  {np.std(lengths):.1f}  Range: [{lengths.min()}, {lengths.max()}]",
        "", "=== GC CONTENT ===",
        f"Mean: {np.mean(gc):.3f} ({np.mean(gc)*100:.1f}%)",
        f"Std:  {np.std(gc):.3f}  Range: [{gc.min():.3f}, {gc.max():.3f}]",
    ]
    if expr_cols:
        lines += ["", "=== EXPRESSION COLUMNS ===",
                  f"Columns: {len(expr_cols)}"]
        total_expr = df[expr_cols].sum(axis=1)
        lines += [f"Total expression — mean: {total_expr.mean():.2f}  "
                  f"max: {total_expr.max():.2f}"]
    text = "\n".join(lines)
    print(text)
    if output_file:
        Path(output_file).write_text(text)
    return expr_cols


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════

def build_dashboard(df, expr_cols, vis_dir, family_name):
    """Generate Plotly interactive dashboard in vis_dir/."""
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots

        vis_dir = Path(vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)

        # 1. Cluster distribution
        vc = df["Cluster"].value_counts().sort_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[str(i) for i in vc.index],
            y=vc.values, text=vc.values, textposition="auto",
            marker_color=["#cccccc" if i == -1 else "#4C9BE8" for i in vc.index],
        ))
        fig.update_layout(title="Cluster Size Distribution",
                          xaxis_title="Cluster", yaxis_title="Count", height=400)
        fig.write_html(vis_dir / "cluster_distribution.html")

        # 2. Sequence characteristics
        df2 = df.copy()
        df2["length"] = df2["Seq"].astype(str).apply(len)
        df2["gc_content"] = df2["Seq"].astype(str).apply(
            lambda s: (s.count("G") + s.count("C")) / len(s) if len(s) > 0 else 0
        )
        fig2 = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Length by Cluster", "GC by Cluster",
                            "Length Distribution", "GC Distribution"),
            specs=[[{"type":"box"},{"type":"box"}],
                   [{"type":"histogram"},{"type":"histogram"}]]
        )
        for cl in sorted(df2["Cluster"].unique()):
            sub = df2[df2["Cluster"] == cl]
            lbl = "Noise" if cl == -1 else f"C{cl}"
            fig2.add_trace(go.Box(y=sub["length"], name=lbl, showlegend=False), row=1, col=1)
            fig2.add_trace(go.Box(y=sub["gc_content"], name=lbl, showlegend=False), row=1, col=2)
        fig2.add_trace(go.Histogram(x=df2["length"], showlegend=False,
                                    marker_color="steelblue"), row=2, col=1)
        fig2.add_trace(go.Histogram(x=df2["gc_content"], showlegend=False,
                                    marker_color="coral"), row=2, col=2)
        fig2.update_layout(height=800, title="Sequence Characteristics")
        fig2.write_html(vis_dir / "sequence_characteristics.html")

        # 3. Expression heatmap
        expr_files = {}
        if expr_cols:
            cluster_expr = df.groupby("Cluster")[expr_cols].mean()
            fig3 = go.Figure(data=go.Heatmap(
                z=cluster_expr.values,
                x=cluster_expr.columns,
                y=[f"Cluster {c}" for c in cluster_expr.index],
                colorscale="Viridis",
            ))
            fig3.update_layout(title="Mean Expression by Cluster", height=max(400, len(cluster_expr)*30))
            fig3.write_html(vis_dir / "expression_heatmap.html")
            expr_files["heatmap"] = True

            # 4. Condition comparison
            cond_groups = {}
            for c in expr_cols:
                parts = c.split("_")
                if len(parts) >= 2:
                    cond_groups.setdefault(parts[1], []).append(c)
            if len(cond_groups) > 1:
                rows_ = []
                for cl in sorted(df["Cluster"].unique()):
                    c_df = df[df["Cluster"] == cl]
                    for cond, cols in cond_groups.items():
                        rows_.append({"Cluster": f"C{cl}", "Condition": cond,
                                      "Expression": c_df[cols].sum(axis=1).mean()})
                fig4 = px.bar(pd.DataFrame(rows_), x="Cluster", y="Expression",
                               color="Condition", barmode="group",
                               title="Mean Expression by Cluster and Condition")
                fig4.write_html(vis_dir / "expression_comparison.html")
                expr_files["comparison"] = True

        # Dashboard index
        expr_cards = ""
        if expr_files.get("heatmap"):
            expr_cards += ('<div class="viz-card"><h2>Expression Heatmap</h2>'
                           '<iframe src="expression_heatmap.html"></iframe>'
                           '<a href="expression_heatmap.html" target="_blank">Open</a></div>')
        if expr_files.get("comparison"):
            expr_cards += ('<div class="viz-card"><h2>Expression Comparison</h2>'
                           '<iframe src="expression_comparison.html"></iframe>'
                           '<a href="expression_comparison.html" target="_blank">Open</a></div>')

        idx_html = f"""<!DOCTYPE html>
<html>
<head>
  <title>{family_name} Analysis Dashboard</title>
  <style>
    body {{font-family:Arial,sans-serif;margin:20px;background:#f5f5f5;}}
    h1 {{color:#333;border-bottom:3px solid #4CAF50;padding-bottom:8px;}}
    .viz-grid {{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:20px;margin-top:20px;}}
    .viz-card {{background:#fff;padding:20px;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,.1);}}
    .viz-card h2 {{margin-top:0;color:#4CAF50;}}
    .viz-card a {{display:inline-block;margin-top:10px;padding:8px 16px;background:#4CAF50;
                  color:#fff;text-decoration:none;border-radius:4px;}}
    iframe {{width:100%;height:500px;border:1px solid #ddd;border-radius:4px;}}
  </style>
</head>
<body>
  <h1>{family_name} Analysis Dashboard</h1>
  <div class="viz-grid">
    <div class="viz-card"><h2>Cluster Distribution</h2>
      <iframe src="cluster_distribution.html"></iframe>
      <a href="cluster_distribution.html" target="_blank">Open</a></div>
    <div class="viz-card"><h2>Sequence Characteristics</h2>
      <iframe src="sequence_characteristics.html"></iframe>
      <a href="sequence_characteristics.html" target="_blank">Open</a></div>
    {expr_cards}
  </div>
</body>
</html>"""
        (vis_dir / "index.html").write_text(idx_html)
        progress_print(f"  Dashboard → {vis_dir / 'index.html'}")

    except Exception as e:
        progress_print(f"  WARNING: dashboard generation failed: {e}")


def build_tfbs_analysis(out_dir, family_name):
    """Generate TF binding summaries from motif_analysis/all_overlaps.tsv."""
    try:
        import matplotlib.pyplot as plt
        import plotly.graph_objects as go

        out_dir = Path(out_dir)
        overlaps_path = out_dir / "motif_analysis" / "all_overlaps.tsv"
        tfbs_dir = out_dir / "05_tfbs"
        tfbs_dir.mkdir(parents=True, exist_ok=True)

        if not overlaps_path.exists():
            (tfbs_dir / "README.txt").write_text(
                "TFBS plots were not generated because motif_analysis/all_overlaps.tsv "
                "does not exist. Run JASPAR motif analysis first.\n"
            )
            progress_print("  TFBS skipped: motif overlaps missing")
            return False

        df_ov = pd.read_csv(overlaps_path, sep="\t")
        motif_col = "Motif_name" if "Motif_name" in df_ov.columns else None
        cluster_col = "Cluster" if "Cluster" in df_ov.columns else "cluster" if "cluster" in df_ov.columns else None
        strand_col = "strand" if "strand" in df_ov.columns else None
        if motif_col is None or cluster_col is None:
            raise ValueError(f"overlaps missing motif/cluster columns: {list(df_ov.columns)}")

        counts = df_ov.groupby([cluster_col, motif_col]).size().reset_index(name="Binding_Sites")
        counts.to_csv(tfbs_dir / "cluster_tf_binding_counts.csv", index=False)

        top = counts.sort_values("Binding_Sites", ascending=False).head(30)
        fig = go.Figure(data=go.Bar(
            x=top["Binding_Sites"],
            y=[f"C{r[cluster_col]} | {r[motif_col]}" for _, r in top.iterrows()],
            orientation="h",
            marker_color="#4169E1",
        ))
        fig.update_layout(
            title=f"{family_name}: Top TF Binding Site Overlaps",
            xaxis_title="Overlap count",
            yaxis_title="Cluster | TF motif",
            height=max(500, 24 * len(top)),
            margin=dict(l=220, r=30, t=60, b=40),
        )
        fig.write_html(tfbs_dir / "tf_binding_top_overlaps.html")

        piv = counts.pivot_table(
            index=motif_col, columns=cluster_col, values="Binding_Sites",
            aggfunc="sum", fill_value=0,
        )
        top_motifs = piv.sum(axis=1).sort_values(ascending=False).head(40).index
        piv = piv.loc[top_motifs]
        fig2 = go.Figure(data=go.Heatmap(
            z=np.log10(piv.values + 1),
            x=[f"Cluster {c}" for c in piv.columns],
            y=piv.index.tolist(),
            colorscale="Blues",
            colorbar=dict(title="log10(count+1)"),
        ))
        fig2.update_layout(
            title=f"{family_name}: Cluster x TF Binding Heatmap",
            height=max(600, 18 * len(piv)),
            margin=dict(l=220, r=30, t=60, b=40),
        )
        fig2.write_html(tfbs_dir / "cluster_tf_binding_heatmap.html")

        if strand_col:
            strand_counts = (
                df_ov.groupby([cluster_col, strand_col]).size()
                .reset_index(name="Binding_Sites")
            )
            strand_counts.to_csv(tfbs_dir / "strand_tf_binding_counts.csv", index=False)
            fig3 = go.Figure()
            for strand in sorted(strand_counts[strand_col].dropna().astype(str).unique()):
                sub = strand_counts[strand_counts[strand_col].astype(str) == strand]
                fig3.add_trace(go.Bar(
                    x=[f"Cluster {c}" for c in sub[cluster_col]],
                    y=sub["Binding_Sites"],
                    name=strand,
                ))
            fig3.update_layout(
                title=f"{family_name}: TF Binding by TE Strand",
                xaxis_title="Cluster",
                yaxis_title="Binding site overlaps",
                barmode="group",
            )
            fig3.write_html(tfbs_dir / "strand_tf_binding.html")

        readme = (
            f"TF binding analysis for {family_name}\n"
            f"Input overlaps: {overlaps_path}\n"
            "Outputs:\n"
            "- cluster_tf_binding_counts.csv\n"
            "- tf_binding_top_overlaps.html\n"
            "- cluster_tf_binding_heatmap.html\n"
            "- strand_tf_binding_counts.csv / strand_tf_binding.html when strand exists\n"
        )
        (tfbs_dir / "README.txt").write_text(readme)
        progress_print(f"  TFBS analysis → {tfbs_dir}")
        return True
    except Exception as e:
        log_error("TF BINDING ANALYSIS", e)
        return False


def validate_existing_outputs(out_dir, family_name):
    """Print a compact checklist of restartable outputs."""
    out_dir = Path(out_dir)
    family_lower = family_name.lower()
    checks = [
        ("sequences", out_dir / "01_data" / f"{family_lower}_with_sequences.csv"),
        ("clustered", out_dir / "01_data" / f"{family_lower}_clustered.csv"),
        ("clustering_html", out_dir / "03_clustering" / "clustering_visualization.html"),
        ("consensus", out_dir / "05_consensus"),
        ("motif_overlaps", out_dir / "motif_analysis" / "all_overlaps.tsv"),
        ("motif_enrichment", out_dir / "enrichment_results"),
        ("tfbs", out_dir / "05_tfbs" / "cluster_tf_binding_heatmap.html"),
        ("go", out_dir / "go_annotations"),
    ]
    print("\n=== EXISTING OUTPUT VALIDATION ===")
    for label, path in checks:
        ok = path.exists() and (not path.is_dir() or any(path.iterdir()))
        print(f"  {'OK ' if ok else 'MISS'}  {label:<18} {path}")


def run_motif_stage_full(args, out_dir, dirs, family_name,
                         clustered_csv=None, bed_input=None):
    """
    Full motif + JASPAR + GO stage with tabix acceleration and comprehensive logging.

    Two interactive prompts are shown at stage entry:
      • Path to a pre-built bgzip+tabix JASPAR .bed.gz (reusable across families/builds)
      • Fisher p-value significance threshold

    Can be driven from either a clustered CSV (normal pipeline) or a raw BED file
    (bed_input, for when no clustered CSV is available).
    """
    import gzip as _gzip, shutil as _shutil
    from collections import Counter as _Counter
    from scipy import stats as _stats
    import requests as _requests

    build      = getattr(args, "assembly", "hg38") or "hg38"
    SPECIES_ID = {"hg38":"9606","hg19":"9606","mm10":"10090","mm39":"10090"}.get(build,"9606")
    FORCE      = False

    motif_dir  = dirs["motif"]
    enrich_dir = dirs["enrichment"]
    go_dir_    = dirs["go"]
    scratch    = out_dir / "tmp_motif"
    for _d in [motif_dir, enrich_dir, go_dir_, scratch]:
        _d.mkdir(parents=True, exist_ok=True)

    t_motif_pipeline = time.time()

    # ══════════════════════════════════════════════════════════
    #  Interactive setup prompts
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  MOTIF + JASPAR + GO  —  Stage Setup")
    print("=" * 65)
    print("  Press Enter to accept the default shown in [brackets].\n")

    # ── 1. Pre-built tabix-indexed JASPAR file ───────────────
    jaspar_bed_arg = getattr(args, "jaspar_bed", None) or ""
    print("  [Tabix JASPAR index]")
    print("  Supply a pre-built bgzip+tabix JASPAR .bed.gz (.tbi alongside it)")
    print("  to skip the re-index step.  The index is reusable across ALL TE")
    print(f"  families on the same build ({build}) — tabix only reads your loci.")
    _tx_raw = input(
        f"  Path to pre-built tabix .bed.gz  [{jaspar_bed_arg or 'blank = auto-build'}]: "
    ).strip()

    PREBUILT_TABIX = None
    if _tx_raw:
        _tp  = Path(_tx_raw)
        _tbi = Path(str(_tp) + ".tbi")
        if not _tp.exists():
            print(f"  ⚠  Not found: {_tp}  — falling back to auto-build / --jaspar-bed")
        elif not _tbi.exists():
            print(f"  ⚠  No .tbi alongside {_tp.name}  — falling back to auto-build / --jaspar-bed")
        else:
            PREBUILT_TABIX = str(_tp)
            progress_print(f"  ✓  Pre-built tabix: {_tp}  ({_tp.stat().st_size/1e6:.0f} MB)")
            progress_print(f"     Index           : {_tbi}  ({_tbi.stat().st_size/1e3:.1f} KB)")
    elif jaspar_bed_arg:
        print(f"  → Using --jaspar-bed from args: {jaspar_bed_arg}")
    else:
        print(f"  → Auto-build mode  (cached under {motif_dir}/jaspar_indexed/)")
    print()

    # ── 2. Significance threshold ────────────────────────────
    p_default = getattr(args, "p_threshold", 0.05) or 0.05
    print("  [Significance threshold]")
    _pv_raw = input(f"  Fisher p-value threshold  [{p_default}]: ").strip()
    P_THRESHOLD = p_default
    if _pv_raw:
        try:
            _pv = float(_pv_raw)
            if not (0 < _pv <= 1):
                raise ValueError("must be in (0, 1]")
            P_THRESHOLD = _pv
            print(f"  ✓  p-threshold: {P_THRESHOLD}")
        except ValueError as _e:
            print(f"  ⚠  '{_pv_raw}' invalid ({_e})  — using default {P_THRESHOLD}")
    else:
        print(f"  → p-threshold: {P_THRESHOLD}")

    print()
    print("=" * 65)
    print(f"  build       : {build}  |  species: {SPECIES_ID}")
    print(f"  out_dir     : {out_dir}")
    print(f"  motif_dir   : {motif_dir}")
    print(f"  jaspar src  : {PREBUILT_TABIX or jaspar_bed_arg or '(auto-resolve)'}")
    print(f"  p_threshold : {P_THRESHOLD}")
    print(f"  input       : {'BED → ' + str(bed_input) if bed_input else 'CSV → ' + str(clustered_csv)}")
    print("=" * 65)

    # ══════════════════════════════════════════════════════════
    #  STAGE M1 — Load loci
    # ══════════════════════════════════════════════════════════
    print("\n" + "─" * 65)
    print("  STAGE M1 — Load loci")
    print("─" * 65)
    t0 = time.time()

    chr_col, start_col, stop_col, cl_col = "chr", "start", "stop", "Cluster"
    df_loci = None

    if bed_input:
        _bp = Path(bed_input)
        if not _bp.exists():
            raise FileNotFoundError(f"BED input not found: {_bp}")
        _sz = _bp.stat().st_size
        print(f"  Mode   : BED input  (all loci → cluster 0)")
        print(f"  File   : {_bp}")
        print(f"  Size   : {_sz/1e6:.2f} MB")
        _wc = subprocess.run(["wc","-l",str(_bp)], capture_output=True, text=True)
        print(f"  Lines  : {_wc.stdout.strip().split()[0] if _wc.returncode==0 else '?'}")
        _raw = pd.read_csv(_bp, sep="\t", header=None, engine="python", on_bad_lines="skip")
        if _raw.shape[1] < 3:
            raise ValueError(f"BED must have ≥3 columns; got {_raw.shape[1]}")
        df_loci = pd.DataFrame({
            "chr"  : _raw.iloc[:,0].astype(str),
            "start": pd.to_numeric(_raw.iloc[:,1], errors="coerce"),
            "stop" : pd.to_numeric(_raw.iloc[:,2], errors="coerce"),
        })
        _nb = len(df_loci)
        df_loci = df_loci.dropna(subset=["start","stop"])
        df_loci = df_loci[~df_loci["chr"].str.lower().str.startswith(("chrom","#"))]
        df_loci["start"]   = df_loci["start"].astype(int)
        df_loci["stop"]    = df_loci["stop"].astype(int)
        df_loci["Cluster"] = 0
        print(f"  Loaded : {len(df_loci):,} loci  ({_nb - len(df_loci)} dropped)")
        cluster_ids, has_strand = [0], False
    else:
        if not clustered_csv or not Path(clustered_csv).exists():
            raise FileNotFoundError(f"Clustered CSV not found: {clustered_csv}")
        _sz = Path(clustered_csv).stat().st_size
        print(f"  Mode   : clustered CSV")
        print(f"  File   : {clustered_csv}")
        print(f"  Size   : {_sz/1e6:.2f} MB")
        df_loci = pd.read_csv(clustered_csv)
        print(f"  Loaded : {len(df_loci):,} rows × {len(df_loci.columns)} columns")
        print(f"  Cols   : {list(df_loci.columns)}")
        # Detect coordinate columns
        def _fc(cands):
            return next((c for c in cands if c in df_loci.columns), None)
        chr_col   = _fc(["chr","Chromosome","chrom","Chr","#chrom"])
        start_col = _fc(["start","Start","chromStart"])
        stop_col  = _fc(["stop","Stop","End","end","chromEnd"])
        cl_col    = _fc(["Cluster","cluster"])
        if not all([chr_col, start_col, stop_col, cl_col]):
            raise ValueError(
                f"Cannot find required columns.  Found: {list(df_loci.columns)}\n"
                "Need chr/start/stop/Cluster — run te_clustering.py first.")
        df_loci[cl_col] = df_loci[cl_col].astype(int)
        cluster_ids = sorted(c for c in df_loci[cl_col].unique() if c >= 0)
        has_strand  = "strand" in df_loci.columns
        noise_n     = int((df_loci[cl_col] == -1).sum())
        print(f"  Clusters     : {cluster_ids}")
        print(f"  Noise (-1)   : {noise_n:,} rows")
        print(f"  Has strand   : {has_strand}")

    print(f"  Chromosomes  : {sorted(df_loci[chr_col].unique())[:10]}")
    print(f"  Locus length : median {int((df_loci[stop_col]-df_loci[start_col]).median()):,} bp")
    print(f"  M1 done in {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════════════════════
    #  STAGE M2 — Resolve JASPAR BED source
    # ══════════════════════════════════════════════════════════
    print("\n" + "─" * 65)
    print("  STAGE M2 — Resolve JASPAR BED source")
    print("─" * 65)
    t0 = time.time()

    # Determine the raw JASPAR file we'll validate / index
    jasp_raw = Path(PREBUILT_TABIX or jaspar_bed_arg or "")
    if not jasp_raw.exists() or not jaspar_bed_arg and not PREBUILT_TABIX:
        # Fall back to te_motif's auto-resolve logic
        _jasp_dir = getattr(args, "jaspar_dir", None) or str(out_dir / "jaspar")
        Path(_jasp_dir).mkdir(parents=True, exist_ok=True)
        from te_motif import resolve_jaspar_bed, validate_jaspar_bed
        _v_bed_tmp = motif_dir / "te_loci_tmp.bed"
        df_loci[[chr_col, start_col, stop_col]].to_csv(
            _v_bed_tmp, sep="\t", header=False, index=False)
        jasp_raw = Path(resolve_jaspar_bed(build, jaspar_bed_arg, _jasp_dir, loci_bed=_v_bed_tmp))
        _v_bed_tmp.unlink(missing_ok=True)
    else:
        from te_motif import validate_jaspar_bed

    jasp_bytes = jasp_raw.stat().st_size
    print(f"  File  : {jasp_raw}")
    print(f"  Size  : {jasp_bytes/1e6:.1f} MB")

    _opener = _gzip.open if str(jasp_raw).endswith(".gz") else open
    _sample, _ccounts = [], []
    with _opener(str(jasp_raw), "rt") as _fh:
        for _i, _ln in enumerate(_fh):
            if _i >= 500: break
            _ln = _ln.rstrip()
            if not _ln or _ln.startswith(("#","track")): continue
            _sample.append(_ln)
            _ccounts.append(len(_ln.split("\t")))
    if not _ccounts:
        raise ValueError("JASPAR BED appears empty or unreadable")
    _modal = _Counter(_ccounts).most_common(1)[0][0]
    print(f"  Sampled {len(_sample)} lines  —  modal cols: {_modal}")
    print(f"  First record: {_sample[0][:120]}")
    print(f"  M2 done in {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════════════════════
    #  STAGE M3 — Write sorted canonical te_loci.bed
    # ══════════════════════════════════════════════════════════
    print("\n" + "─" * 65)
    print("  STAGE M3 — Write sorted te_loci.bed")
    print("─" * 65)
    t0 = time.time()

    v_bed = motif_dir / "te_loci.bed"
    _df_sort = df_loci[[chr_col, start_col, stop_col]].sort_values(
        by=[chr_col, start_col, stop_col],
        key=lambda c: c if c.name != chr_col else c.astype(str)
    )
    _df_sort.to_csv(v_bed, sep="\t", header=False, index=False)
    print(f"  Wrote  : {v_bed}  ({v_bed.stat().st_size:,} bytes, {len(_df_sort):,} loci)")
    print(f"  M3 done in {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════════════════════
    #  STAGE M3b — Resolve bgzip + tabix index
    # ══════════════════════════════════════════════════════════
    print("\n" + "─" * 65)
    print("  STAGE M3b — Resolve bgzip + tabix index for JASPAR")
    print("─" * 65)
    t0 = time.time()

    def _have(cmd): return _shutil.which(cmd) is not None

    use_tabix = _have("tabix") and _have("bgzip")
    tabix_target = None

    if not use_tabix:
        progress_print("  tabix/bgzip not found — will use full bedtools scan (slower)")
    elif PREBUILT_TABIX:
        tabix_target = Path(PREBUILT_TABIX)
        print(f"  Using pre-built tabix file (no rebuild needed):")
        print(f"    {tabix_target}")
        _tbi = Path(str(tabix_target) + ".tbi")
        print(f"    {_tbi}  ({_tbi.stat().st_size/1e3:.1f} KB)")
        print(f"  M3b done in {time.time()-t0:.2f}s")
    else:
        cache_dir = motif_dir / "jaspar_indexed"
        cache_dir.mkdir(parents=True, exist_ok=True)

        def _is_bgzip(p):
            try:
                with open(p,"rb") as _f: magic = _f.read(4)
                return len(magic)==4 and magic[0]==0x1f and magic[1]==0x8b \
                       and magic[2]==0x08 and magic[3]==0x04
            except Exception: return False

        indexed_gz  = cache_dir / "JASPAR_indexed.bed.gz"
        indexed_tbi = Path(str(indexed_gz) + ".tbi")
        sibling_tbi = Path(str(jasp_raw) + ".tbi")
        src_bgzip   = _is_bgzip(jasp_raw)
        print(f"  Source bgzip? {src_bgzip}  |  cache: {cache_dir}")

        if src_bgzip and sibling_tbi.exists():
            print(f"  Reusing source .tbi at {sibling_tbi}")
            tabix_target = jasp_raw
        elif indexed_gz.exists() and indexed_tbi.exists() and not FORCE:
            print(f"  Reusing cached indexed copy:")
            print(f"    {indexed_gz}  ({indexed_gz.stat().st_size/1e6:.1f} MB)")
            print(f"    {indexed_tbi}  ({indexed_tbi.stat().st_size/1e3:.1f} KB)")
            tabix_target = indexed_gz
        else:
            if src_bgzip:
                print(f"  Copying bgzip source to cache ...")
                _shutil.copy2(jasp_raw, indexed_gz)
                print(f"    → {indexed_gz}  ({indexed_gz.stat().st_size/1e6:.1f} MB)")
            else:
                print(f"  Plain gzip detected — decompressing + re-bgzipping + sorting ...")
                print(f"  (one-time cost; subsequent runs reuse the cached index)")
                _t_bg = time.time()
                with open(indexed_gz, "wb") as _fout:
                    _p1 = subprocess.Popen(["zcat", str(jasp_raw)], stdout=subprocess.PIPE)
                    _p2 = subprocess.Popen(
                        ["sort","-k1,1","-k2,2n","-S","2G","-T",str(scratch)],
                        stdin=_p1.stdout, stdout=subprocess.PIPE)
                    _p1.stdout.close()
                    _p3 = subprocess.Popen(["bgzip","-c"], stdin=_p2.stdout, stdout=_fout)
                    _p2.stdout.close()
                    _p3.communicate(); _p2.wait(); _p1.wait()
                    if _p3.returncode != 0:
                        raise RuntimeError(f"bgzip pipeline failed (exit {_p3.returncode})")
                print(f"    Wrote {indexed_gz}  "
                      f"({indexed_gz.stat().st_size/1e6:.1f} MB) in {time.time()-_t_bg:.1f}s")
            print(f"  Building tabix index ...")
            _t_tb = time.time()
            _res = subprocess.run(
                ["tabix","-p","bed","-f",str(indexed_gz)],
                capture_output=True, text=True)
            if _res.returncode != 0:
                raise RuntimeError(f"tabix indexing failed:\n{_res.stderr}")
            print(f"    Wrote {indexed_tbi}  "
                  f"({indexed_tbi.stat().st_size/1e3:.1f} KB) in {time.time()-_t_tb:.1f}s")
            tabix_target = indexed_gz

        print(f"  Tabix target: {tabix_target}")
        print(f"  M3b done in {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════════════════════
    #  STAGE M4 — bedtools intersect  (tabix-accelerated or full)
    # ══════════════════════════════════════════════════════════
    print("\n" + "─" * 65)
    print("  STAGE M4 — JASPAR intersect")
    print("─" * 65)
    t0 = time.time()

    overlaps_path = motif_dir / "all_overlaps.tsv"
    df_ov = None

    if overlaps_path.exists() and not FORCE:
        print(f"  [SKIP] overlaps file exists: {overlaps_path}")
        print(f"         Delete it or set FORCE=True to rerun.")
        try:
            df_ov = pd.read_csv(overlaps_path, sep="\t")
            print(f"  Loaded {len(df_ov):,} rows  ({overlaps_path.stat().st_size/1e6:.1f} MB)")
        except Exception as _e:
            print(f"  WARNING: reload failed ({_e}); rerunning intersect")
            overlaps_path.unlink(missing_ok=True)
            df_ov = None

    if df_ov is None:
        _bt = subprocess.run(["bedtools","--version"], capture_output=True, text=True)
        if _bt.returncode != 0:
            raise EnvironmentError("bedtools not found on PATH — load it with `module load bedtools`")
        print(f"  bedtools: {_bt.stdout.strip()}")

        if use_tabix and tabix_target:
            # ── Step 1: tabix subset ──────────────────────────
            subset_path = scratch / f"jaspar_subset_{os.getpid()}.bed"
            print(f"\n  [Step 1/2] tabix -R (pulls only JASPAR rows overlapping our loci)")
            print(f"    tabix -R {v_bed} {tabix_target}")
            _t_tx = time.time()
            with open(subset_path,"w") as _fo:
                _res = subprocess.run(
                    ["tabix","-R",str(v_bed),str(tabix_target)],
                    stdout=_fo, stderr=subprocess.PIPE, text=True)
            if _res.returncode != 0:
                raise RuntimeError(f"tabix -R failed:\n{_res.stderr}")
            _n_sub = sum(1 for _ in open(subset_path))
            print(f"    tabix done in {time.time()-_t_tx:.1f}s  "
                  f"|  {_n_sub:,} motif rows  ({subset_path.stat().st_size/1e6:.1f} MB)")
            if _n_sub == 0:
                raise RuntimeError(
                    "tabix returned 0 rows — likely a chromosome naming mismatch.\n"
                    f"  Check: head -2 {v_bed}  vs  zcat {tabix_target} | head -2")
            intersect_b = str(subset_path)
        else:
            progress_print("  Skipping tabix subset — using full JASPAR file with bedtools")
            intersect_b = str(jasp_raw)
            subset_path = None

        # ── Step 2: bedtools intersect ────────────────────────
        tmp_out = scratch / f"bt_overlaps_{os.getpid()}.tsv"
        _cmd = ["bedtools","intersect",
                "-a", str(v_bed), "-b", intersect_b,
                "-sorted", "-wa", "-wb"]
        print(f"\n  [Step 2/2] bedtools intersect")
        print(f"    {' '.join(_cmd)}")
        _t_bt = time.time()
        with open(tmp_out,"w") as _fo:
            _proc = subprocess.run(_cmd, stdout=_fo, stderr=subprocess.PIPE)
        _stderr = _proc.stderr.decode(errors="replace")
        print(f"    exit={_proc.returncode}  elapsed={time.time()-_t_bt:.1f}s")
        if _stderr.strip():
            print(f"    stderr: {_stderr.strip()[:500]}")

        # Retry without -sorted if chrom order complaint
        if _proc.returncode != 0 and "sort" in _stderr.lower():
            print("    retrying without -sorted ...")
            _cmd2 = [c for c in _cmd if c != "-sorted"]
            with open(tmp_out,"w") as _fo:
                _proc = subprocess.run(_cmd2, stdout=_fo, stderr=subprocess.PIPE)
            _stderr = _proc.stderr.decode(errors="replace")
            print(f"    retry exit={_proc.returncode}")

        # Retry with 6-col normalised subset on column-count mismatch
        if _proc.returncode != 0 and ("fields" in _stderr.lower()
                                       or "different number" in _stderr.lower()):
            print("    column mismatch — normalising to 6 cols ...")
            _norm = scratch / "jaspar_norm6.bed"
            _nn = 0
            with open(intersect_b) as _fi, open(_norm,"w") as _fo:
                for _l in _fi:
                    _s = _l.rstrip("\n")
                    if not _s or _s.startswith(("#","track")): continue
                    _p = _s.split("\t")
                    if len(_p) >= 6:
                        _fo.write("\t".join(_p[:6]) + "\n"); _nn += 1
            print(f"    wrote {_nn:,} normalised rows")
            _cmd3 = ["bedtools","intersect","-a",str(v_bed),"-b",str(_norm),"-wa","-wb"]
            with open(tmp_out,"w") as _fo:
                _proc = subprocess.run(_cmd3, stdout=_fo, stderr=subprocess.PIPE)
            _stderr = _proc.stderr.decode(errors="replace")
            print(f"    normalised retry exit={_proc.returncode}")

        if _proc.returncode != 0:
            raise RuntimeError(f"bedtools intersect failed (exit {_proc.returncode})\n{_stderr[:2000]}")

        _out_sz = tmp_out.stat().st_size
        print(f"  Output size: {_out_sz/1e6:.1f} MB")
        if _out_sz == 0:
            raise RuntimeError(
                "bedtools returned zero output — check chromosome naming "
                f"matches between te_loci.bed and JASPAR (build={build})")

        # Parse column layout
        _fl = open(tmp_out).readline().rstrip("\n").split("\t")
        _n_mc = len(_fl) - 3
        _cv   = [chr_col, start_col, stop_col]
        if   _n_mc == 4: _cm = ["Motif_chr","Motif_start","Motif_end","Motif_name"]
        elif _n_mc == 5: _cm = ["Motif_chr","Motif_start","Motif_end","Motif_name","Motif_score"]
        elif _n_mc == 6: _cm = ["Motif_chr","Motif_start","Motif_end","Motif_name","Motif_score","Motif_strand"]
        elif _n_mc == 7: _cm = ["Motif_chr","Motif_start","Motif_end","Motif_ID","Motif_score","Motif_strand","Motif_name"]
        else:            _cm = [f"motif_col_{_i}" for _i in range(_n_mc)]
        print(f"  Column layout: {len(_fl)} total  ({len(_cv)} TE-side + {_n_mc} JASPAR-side)")
        print(f"  JASPAR cols  : {_cm}")
        print(f"  Preview      : {_fl[:8]}")

        print(f"  Reading into DataFrame ...")
        df_ov = pd.read_csv(tmp_out, sep="\t", header=None, names=_cv+_cm,
                            dtype=str, engine="python", on_bad_lines="skip")
        tmp_out.unlink(missing_ok=True)
        if subset_path and subset_path.exists():
            subset_path.unlink(missing_ok=True)
        print(f"  Rows: {len(df_ov):,}")

        if "Motif_name" not in df_ov.columns:
            _nc = [c for c in df_ov.columns if "name" in c.lower()]
            df_ov["Motif_name"] = df_ov[_nc[0]] if _nc else df_ov.iloc[:,3]
            print(f"  Motif_name derived from: {_nc[0] if _nc else 'col-index-3'}")

        for _c in [start_col, stop_col]:
            df_ov[_c] = pd.to_numeric(df_ov[_c], errors="coerce")

        # Merge cluster assignments
        _w = df_loci[[chr_col, start_col, stop_col, cl_col]].copy()
        for _c in [start_col, stop_col]:
            _w[_c] = pd.to_numeric(_w[_c], errors="coerce")
        df_ov[chr_col] = df_ov[chr_col].astype(str)
        _w[chr_col]    = _w[chr_col].astype(str)
        df_ov = df_ov.merge(_w, on=[chr_col, start_col, stop_col], how="left")
        df_ov[cl_col] = df_ov[cl_col].fillna(0).astype(int)
        print(f"  After cluster merge: {len(df_ov):,} rows  "
              f"({df_ov[cl_col].isna().sum()} missing cluster)")

        df_ov.to_csv(overlaps_path, sep="\t", index=False)
        print(f"  Saved → {overlaps_path}  ({overlaps_path.stat().st_size/1e6:.1f} MB)")

    print(f"\n  Unique TE loci     : {df_ov[[chr_col,start_col,stop_col]].drop_duplicates().shape[0]:,}")
    print(f"  Unique JASPAR motifs: {df_ov['Motif_name'].nunique():,}")
    print(f"  Mean overlaps/locus : {len(df_ov)/max(len(df_loci),1):.1f}")
    _top5 = df_ov["Motif_name"].value_counts().head(5)
    print(f"  Top 5 motifs by raw overlap count:")
    for _m, _cnt in _top5.items():
        print(f"    {_m:<40}  {_cnt:>8,}")

    # Overall motif frequency
    _vc = df_ov["Motif_name"].value_counts()
    _vc.to_csv(motif_dir / "overall_motif_counts.csv")
    try:
        import matplotlib.pyplot as _plt
        _vc20 = _vc.head(20).reset_index(); _vc20.columns = ["Motif","Count"]
        _fig, _ax = _plt.subplots(figsize=(13,5))
        _ax.bar(range(len(_vc20)), _vc20["Count"], color="#3498DB", alpha=0.85, edgecolor="white")
        _ax.set_xticks(range(len(_vc20)))
        _ax.set_xticklabels(_vc20["Motif"], rotation=45, ha="right", fontsize=8)
        _ax.set_ylabel("Overlap count"); _ax.set_xlabel("")
        _ax.set_title(f"Top 20 JASPAR Motifs — {build}  ({family_name})", fontweight="bold")
        _ax.spines[["top","right"]].set_visible(False)
        _plt.tight_layout()
        _out_png = motif_dir / "overall_top_motifs.png"
        _plt.savefig(_out_png, dpi=150, bbox_inches="tight"); _plt.close()
        print(f"  Saved → {_out_png}")
    except Exception as _e:
        print(f"  WARNING: top-20 bar chart failed: {_e}")
    print(f"  M4 total time: {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════════════════════
    #  STAGE M5 — Fisher's exact test
    # ══════════════════════════════════════════════════════════
    print("\n" + "─" * 65)
    print("  STAGE M5 — Fisher's exact test  (cluster × motif)")
    print("─" * 65)
    t0 = time.time()

    total_n    = len(df_loci)
    all_motifs = df_ov["Motif_name"].unique()
    significant_tfs = {}

    print(f"  Clusters   : {cluster_ids}")
    print(f"  Total loci : {total_n:,}")
    print(f"  Motifs     : {len(all_motifs):,}")
    print(f"  Total tests: {len(cluster_ids) * len(all_motifs):,}")

    for cid in cluster_ids:
        cn = int((df_loci[cl_col] == cid).sum())
        print(f"\n  ── Cluster {cid}  (n={cn:,}) ──")
        _t_cl = time.time()
        _results = []
        _logged  = set()

        for _mi, motif in enumerate(all_motifs):
            _in_cl = len(
                df_ov[(df_ov[cl_col]==cid) & (df_ov["Motif_name"]==motif)]
                [[chr_col, start_col, stop_col]].drop_duplicates()
            )
            _total_m = len(
                df_ov[df_ov["Motif_name"]==motif]
                [[chr_col, start_col, stop_col]].drop_duplicates()
            )
            _nmc  = cn - _in_cl
            _mnc  = _total_m - _in_cl
            _nmnc = max((total_n - cn) - _mnc, 0)
            try:
                _odds, _pv = _stats.fisher_exact(
                    [[_in_cl, _mnc], [_nmc, _nmnc]], alternative="greater")
            except Exception:
                _odds, _pv = 1.0, 1.0
            _results.append({
                "Motif": motif, "In_Cluster": _in_cl, "Total_With_Motif": _total_m,
                "Cluster_Size": cn, "Total_Loci": total_n,
                "Odds_Ratio": round(_odds,4), "P_Value": _pv,
            })
            _ms = (_mi+1) // 500
            if _ms not in _logged and (_mi+1) % 500 == 0:
                _logged.add(_ms)
                _el = time.time() - _t_cl
                _rate = (_mi+1) / max(_el, 1e-9)
                _eta  = (len(all_motifs) - _mi - 1) / max(_rate, 1e-9)
                progress_print(f"    tested {_mi+1:,}/{len(all_motifs):,}  "
                               f"|  {_el:.0f}s elapsed  |  ETA ~{_eta:.0f}s")

        _rdf = pd.DataFrame(_results).sort_values("P_Value")
        _out_csv = enrich_dir / f"cluster_{cid}_enrichment.csv"
        _rdf.to_csv(_out_csv, index=False)

        _sig = _rdf[_rdf["P_Value"] < P_THRESHOLD]
        significant_tfs[cid] = _sig
        print(f"    Done in {time.time()-_t_cl:.1f}s  |  "
              f"{len(_sig):,} significant TFs (p<{P_THRESHOLD})")
        print(f"    Saved → {_out_csv}  ({_out_csv.stat().st_size/1e3:.0f} KB)")
        if len(_sig):
            print(f"    Top 10:")
            for _, _row in _sig.head(10).iterrows():
                print(f"      {_row['Motif']:<40}  "
                      f"p={_row['P_Value']:.2e}  OR={_row['Odds_Ratio']:.2f}  "
                      f"in={int(_row['In_Cluster'])}/{cn}")
        else:
            print(f"    No significant TFs at p<{P_THRESHOLD}")

    print(f"\n  Fisher stage: {time.time()-t0:.1f}s  |  "
          f"total sig hits: {sum(len(s) for s in significant_tfs.values()):,}")

    # Enrichment heatmap
    try:
        import matplotlib.pyplot as _plt
        import numpy as _np
        _frames = [(_s.head(15).assign(cluster=_c))
                   for _c, _s in significant_tfs.items() if len(_s)]
        if _frames:
            _comb = pd.concat(_frames)
            _comb["nlp"] = -_np.log10(_comb["P_Value"].clip(lower=1e-300))
            _piv = _comb.pivot_table(index="Motif", columns="cluster",
                                     values="nlp", aggfunc="max").fillna(0)
            _top = _piv.max(axis=1).sort_values(ascending=False).head(30).index
            _piv = _piv.loc[_top]
            _fig, _ax = _plt.subplots(figsize=(max(6, len(_piv.columns)*2),
                                               max(6, len(_top)*0.38)))
            _im = _ax.imshow(_piv.values, aspect="auto", cmap="Reds")
            _plt.colorbar(_im, ax=_ax, label="−log₁₀(p)")
            _ax.set_xticks(range(len(_piv.columns)))
            _ax.set_xticklabels([f"Cl {c}" for c in _piv.columns], fontsize=9)
            _ax.set_yticks(range(len(_top))); _ax.set_yticklabels(_top, fontsize=8)
            _ax.set_title(f"Motif Enrichment — {build}  ({family_name})", fontweight="bold")
            _plt.tight_layout()
            _hp = enrich_dir / "enrichment_heatmap.png"
            _plt.savefig(_hp, dpi=150, bbox_inches="tight"); _plt.close()
            print(f"  Heatmap saved → {_hp}")
        else:
            print("  No significant TFs — heatmap skipped")
    except Exception as _e:
        print(f"  WARNING: heatmap failed: {_e}")

    # ══════════════════════════════════════════════════════════
    #  STAGE M6 — GO annotation
    # ══════════════════════════════════════════════════════════
    if not getattr(args, "skip_go", False):
        print("\n" + "─" * 65)
        print("  STAGE M6 — GO annotation  (mygene.info API)")
        print("─" * 65)
        t0 = time.time()

        _gf_path = go_dir_ / "gene_functions.csv"
        _all_sig_motifs = set()
        _go_top = getattr(args, "go_top_motifs", 30) or 30
        for _s in significant_tfs.values():
            _all_sig_motifs.update(_s.head(_go_top)["Motif"].tolist())
        print(f"  Unique sig motifs to annotate: {len(_all_sig_motifs)}")

        _annotated = []

        if not _all_sig_motifs:
            print("  No significant motifs — GO skipped")
            (go_dir_ / "GO_SKIPPED.txt").write_text("No significant motifs passed the p-value threshold.\n")
        else:
            if _gf_path.exists() and not FORCE:
                print(f"  [SKIP] gene_functions.csv exists — loading cache")
                _gf_df = pd.read_csv(_gf_path)
                print(f"  Loaded {len(_gf_df):,} records from cache")
            else:
                def _qgene(name):
                    _c = re.sub(r"[^A-Za-z0-9]","",name)
                    _h = re.sub(r"[^A-Za-z0-9-]","",name)
                    for _q in [f"symbol:{_h}",f"symbol:{_c}",_h,_c]:
                        if not _q: continue
                        try:
                            _r = _requests.get(
                                f"https://mygene.info/v3/query?q={_q}"
                                f"&species={SPECIES_ID}&fields=symbol,name,summary,go&size=1",
                                timeout=15)
                            if _r.status_code == 200:
                                _hits = _r.json().get("hits",[])
                                if _hits: return _hits[0]
                        except Exception: pass
                    return None

                def _goterms(gd, key, n=5):
                    if not isinstance(gd, dict): return ""
                    _t = gd.get(key,[])
                    return "; ".join(x.get("term","") for x in _t[:n]) if isinstance(_t,list) else ""

                _gf = {}; _errs = 0; _t_go = time.time()
                for _i, _m in enumerate(sorted(_all_sig_motifs), 1):
                    _gn  = _m.split("::")[0] if "::" in _m else _m
                    _key = re.sub(r"[^A-Za-z0-9-]","",_gn)
                    if not _key or _key in _gf: continue
                    _info = _qgene(_gn)
                    if _info: _gf[_key] = _info
                    else:     _errs += 1
                    if _i % 20 == 0:
                        _el = time.time() - _t_go
                        _eta = (len(_all_sig_motifs) - _i) / max(_i/max(_el,1e-9), 1e-9)
                        progress_print(f"    [{_i}/{len(_all_sig_motifs)}]  "
                                       f"hits={len(_gf)}  errors={_errs}  "
                                       f"elapsed={_el:.0f}s  ETA~{_eta:.0f}s")
                    time.sleep(0.12)

                print(f"\n  mygene.info: {len(_gf)} hits / {len(_all_sig_motifs)} queried  "
                      f"|  {_errs} no-hit  |  {time.time()-_t_go:.1f}s")
                _rows = []
                for _gn, _info in _gf.items():
                    _gd = _info.get("go",{})
                    _rows.append({
                        "Gene":_gn, "Symbol":_info.get("symbol",""),
                        "Name":_info.get("name",""),
                        "Summary":((_info.get("summary","") or "")[:300]),
                        "GO_BP":_goterms(_gd,"BP"), "GO_MF":_goterms(_gd,"MF"),
                        "GO_CC":_goterms(_gd,"CC"),
                    })
                _gf_df = pd.DataFrame(_rows)
                _gf_df.to_csv(_gf_path, index=False)
                print(f"  Saved {len(_gf_df):,} records → {_gf_path}")

            # Annotate enrichment CSVs
            _gf_idx = _gf_df.set_index("Gene") if len(_gf_df) else pd.DataFrame()
            def _lkp(motif, col):
                _g = re.sub(r"[^A-Za-z0-9-]","",motif.split("::")[0] if "::" in motif else motif)
                if len(_gf_idx) and _g in _gf_idx.index:
                    _v = _gf_idx.loc[_g]
                    return str(_v[col]) if col in _v.index else ""
                return ""

            for _cid, _sig in significant_tfs.items():
                if not len(_sig): continue
                _sig = _sig.copy()
                _sig["Cluster"]   = _cid
                for _col in ["Name","Summary","GO_BP","GO_MF","GO_CC"]:
                    _sig[_col] = _sig["Motif"].apply(lambda m, c=_col: _lkp(m,c))
                _out_ann = enrich_dir / f"cluster_{_cid}_enrichment_annotated.csv"
                _sig.to_csv(_out_ann, index=False)
                _annotated.append(_sig)
                print(f"  Cluster {_cid}: {len(_sig):,} sig TFs  "
                      f"|  {(_sig['GO_BP']!='').sum()} with GO_BP  → {_out_ann.name}")

        # ── GO plots ──────────────────────────────────────────
        try:
            import matplotlib.pyplot as _plt
            def _split_go(v):
                return [t.strip() for t in str(v or "").split(";") if t.strip()]
            _bp_rows = []
            if len(_gf_df) and "GO_BP" in _gf_df.columns:
                for _t in _gf_df["GO_BP"].dropna():
                    for _term in _split_go(_t):
                        _bp_rows.append({"Source":"all_genes","GO_Term":_term})
            for _fr in _annotated:
                for _, _row in _fr.iterrows():
                    for _term in _split_go(_row.get("GO_BP","")):
                        _bp_rows.append({"Source":f"cluster_{_row['Cluster']}","GO_Term":_term})

            if _bp_rows:
                _bpc = pd.DataFrame(_bp_rows)
                _og  = _bpc["GO_Term"].value_counts().head(20).reset_index()
                _og.columns = ["GO_Term","Count"]
                _og.to_csv(go_dir_ / "go_bp_top_terms.csv", index=False)
                print(f"  Top GO-BP terms:")
                for _, _r in _og.head(10).iterrows():
                    print(f"    [{_r['Count']:>4}]  {_r['GO_Term']}")
                _fig, _ax = _plt.subplots(figsize=(10, max(5, len(_og)*0.38)))
                _ax.barh(_og["GO_Term"][::-1], _og["Count"][::-1], color="#4C9BE8")
                _ax.set_xlabel("Count")
                _ax.set_title(f"Top GO-BP Terms — {build}  ({family_name})", fontweight="bold")
                _ax.spines[["top","right"]].set_visible(False)
                _plt.tight_layout()
                _gop = go_dir_ / "go_bp_top_terms.png"
                _plt.savefig(_gop, dpi=150, bbox_inches="tight"); _plt.close()
                print(f"  Saved → {_gop}")

                for _fr in _annotated:
                    _cid2 = _fr["Cluster"].iloc[0] if "Cluster" in _fr.columns else "?"
                    _cl_terms = [t for _, _r in _fr.iterrows()
                                 for t in _split_go(_r.get("GO_BP",""))]
                    if not _cl_terms: continue
                    _clc = pd.Series(_cl_terms).value_counts().head(15).reset_index()
                    _clc.columns = ["GO_Term","Count"]
                    _fig2, _ax2 = _plt.subplots(figsize=(10, max(4, len(_clc)*0.38)))
                    _ax2.barh(_clc["GO_Term"][::-1], _clc["Count"][::-1], color="#2ECC71")
                    _ax2.set_xlabel("Count")
                    _ax2.set_title(f"GO-BP — Cluster {_cid2}  ({build})", fontweight="bold")
                    _ax2.spines[["top","right"]].set_visible(False)
                    _plt.tight_layout()
                    _clp = go_dir_ / f"cluster_{_cid2}_go_bp.png"
                    _plt.savefig(_clp, dpi=150, bbox_inches="tight"); _plt.close()
                    print(f"  Cluster {_cid2} GO-BP saved → {_clp}")
            else:
                (go_dir_ / "go_plot_status.txt").write_text(
                    "No GO-BP terms available — no motifs passed the p-value cutoff "
                    "or mygene.info returned no GO annotations.\n")
                print("  No GO-BP terms to plot")
        except Exception as _e:
            print(f"  WARNING: GO plots failed: {_e}")
            traceback.print_exc()

        print(f"  M6 done in {time.time()-t0:.1f}s")

    # ── TFBS summary plots ────────────────────────────────────
    print("\n" + "─" * 65)
    print("  STAGE M7 — TFBS binding summary plots")
    print("─" * 65)
    build_tfbs_analysis(out_dir, family_name)

    # ── Final output inventory ────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  MOTIF STAGE COMPLETE  ({time.time()-t_motif_pipeline:.1f}s total)")
    print("=" * 65)
    _all = sorted(out_dir.rglob("*"))
    for _f in _all:
        if _f.is_file() and _f.suffix in {".csv",".tsv",".png",".log",".txt"}:
            _sz = _f.stat().st_size
            print(f"    {str(_f.relative_to(out_dir)):<55}  {_sz/1e3:>8.1f} KB")


def run_motif_tfbs_go(args, out_dir, dirs, family_name, run_motif=True, run_tfbs=True, run_go=True):
    """Run restartable motif, TFBS, and GO stages from an existing clustered CSV."""
    clustered_csv = dirs["data"] / f"{family_name.lower()}_clustered.csv"
    if not clustered_csv.exists():
        candidates = sorted(Path(out_dir).glob("**/*_clustered.csv"))
        if candidates:
            clustered_csv = candidates[0]
    if not clustered_csv.exists():
        raise FileNotFoundError(f"No clustered CSV found under {out_dir}; cannot resume motif/TFBS/GO")

    if run_motif and not args.skip_motif:
        run_motif_stage_full(args, out_dir, dirs, family_name,
                             clustered_csv=str(clustered_csv))

    if run_tfbs and not run_motif:
        print("\n=== TRANSCRIPTION FACTOR BINDING ANALYSIS ===")
        build_tfbs_analysis(out_dir, family_name)

    if run_go and not args.skip_go and not run_motif:
        print("\n=== GO PLOT GENERATION ===")
        if any(dirs["enrichment"].glob("cluster_*_enrichment.csv")):
            run_go_annotation(
                enrichment_dir=str(dirs["enrichment"]),
                build=getattr(args, "assembly", "hg38") or "hg38",
                out_dir=out_dir,
                clustered_csv=str(clustered_csv),
                p_threshold=args.p_threshold,
                top_motifs=30,
                force=False,
            )
        else:
            progress_print("  GO skipped: no enrichment CSV files found")
            (dirs["go"] / "GO_SKIPPED.txt").write_text(
                "GO annotation was skipped because no cluster_*_enrichment.csv "
                "files were generated by motif analysis.\n"
            )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(args):
    """Execute the full TE analysis pipeline."""
    global OUT_DIR

    FAMILY_NAME = args.family
    HG38_FA = args.genome
    BASE_OUT_DIR = Path(args.output)

    OUT_DIR = BASE_OUT_DIR / FAMILY_NAME.lower()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    DIRS = {
        "data":          OUT_DIR / "01_data",
        "stats":         OUT_DIR / "02_statistics",
        "clustering":    OUT_DIR / "03_clustering",
        "alignments":    OUT_DIR / "04_alignments",
        "consensus":     OUT_DIR / "05_consensus",
        "primers":       OUT_DIR / "06_primers",
        "visualizations": OUT_DIR / "07_visualizations",
        "motif":         OUT_DIR / "motif_analysis",
        "enrichment":    OUT_DIR / "enrichment_results",
        "go":            OUT_DIR / "go_annotations",
        "tfbs":          OUT_DIR / "05_tfbs",
    }
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    if args.validate_existing:
        validate_existing_outputs(OUT_DIR, FAMILY_NAME)

    if args.resume_from in {"motif", "tfbs", "go"}:
        try:
            run_motif_tfbs_go(
                args, OUT_DIR, DIRS, FAMILY_NAME,
                run_motif=args.resume_from == "motif",
                run_tfbs=args.resume_from in {"motif", "tfbs"},
                run_go=args.resume_from in {"motif", "tfbs", "go"},
            )
        except SystemExit:
            raise
        except Exception as e:
            log_error(f"RESUME FROM {args.resume_from.upper()}", e)
            sys.exit(1)
        print("\nResume run complete.")
        return None

    pipeline_start = time.time()
    stage_times = {}

    def _record(name):
        t = time.time() - pipeline_start
        stage_times[name] = t
        progress_print(f"  [TIMING] {name}: {t:.1f}s total")

    print("\n" + "=" * 60)
    print(f"TE ANALYSIS PIPELINE")
    print(f"  Family:  {FAMILY_NAME}")
    print(f"  Output:  {OUT_DIR.resolve()}")
    print(f"  Genome:  {HG38_FA or '(not provided)'}")
    if getattr(args, "local", False):
        print(f"  Mode:    LOCAL  (assembly={getattr(args, 'assembly', 'hg38')})")
    print(f"  K-mer:   {args.kmer}")
    print(f"  UMAP:    pca_dims={args.pca_dims}, n_epochs={args.n_epochs}, "
          f"random_state={'None/multicore' if args.random_state == 0 else args.random_state}, "
          f"skip_tsne={args.skip_tsne}")
    print(f"  Threads: OMP={os.environ.get('OMP_NUM_THREADS', '(unset)')} "
          f"MKL={os.environ.get('MKL_NUM_THREADS', '(unset)')} "
          f"OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS', '(unset)')} "
          f"NUMBA={os.environ.get('NUMBA_NUM_THREADS', '(unset)')}")
    print(f"  Cache:   NUMBA_CACHE_DIR={os.environ.get('NUMBA_CACHE_DIR', '(unset)')}")
    print(f"  Date:    {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── 1. Load genome ──────────────────────────────────────────────────────
    print("\n=== LOADING REFERENCE GENOME ===")
    progress_print(f"Genome path argument: {HG38_FA or '(none)'}")
    genome_cache = GenomeCache(HG38_FA, cache_dir=str(OUT_DIR))
    genome_cache.load()
    if genome_cache.is_loaded:
        progress_print("Genome loaded — local extraction + fast primer search enabled")
    else:
        progress_print("Genome not available — UCSC API fallback for sequences")

    # ── 2. Load data ────────────────────────────────────────────────────────
    t0 = time.time()
    print("\n=== LOADING DATA ===")
    try:
        progress_print(
            f"Data source: test={args.test}, local={getattr(args, 'local', False)}, "
            f"input={args.input or '(auto)'}"
        )
        if args.test:
            progress_print("Creating mock test data...")
            df_family = create_test_data()
            FAMILY_NAME = "TEST_TE"
        elif getattr(args, "local", False) and not args.input:
            # Local mode: auto-download rmsk and build DataFrame from coords
            df_family = _load_local_data(args)
        elif args.input:
            if not Path(args.input).exists():
                raise FileNotFoundError(f"Input file not found: {args.input}")
            df_raw = pd.read_csv(args.input)
            progress_print(f"Loaded {len(df_raw)} rows, columns: {list(df_raw.columns)}")
            if "TE_name" not in df_raw.columns:
                if "TE_ID" in df_raw.columns:
                    progress_print("Deriving 'TE_name' from 'TE_ID' (field 3, pipe-delimited)")
                    df_raw["TE_name"] = df_raw["TE_ID"].str.split("|").str[3]
                else:
                    raise ValueError(f"Missing required column 'TE_name'. Columns: {list(df_raw.columns)}")
            df_family = df_raw[
                df_raw["TE_name"].str.contains(FAMILY_NAME, case=False, na=False)
            ].copy().reset_index(drop=True)
            progress_print(f"Filtered: {len(df_family)} instances of {FAMILY_NAME}")
        elif "df" in globals():
            df_raw = globals()["df"]
            df_family = df_raw[
                df_raw["TE_name"].str.contains(FAMILY_NAME, case=False, na=False)
            ].copy().reset_index(drop=True)
        else:
            raise RuntimeError(
                "No data source. Options:\n"
                "  --local --family <NAME> --assembly hg38  (auto-download + UCSC API)\n"
                "  --input <CSV>\n"
                "  --test"
            )

        if len(df_family) == 0:
            print(f"ERROR: No sequences found for family '{FAMILY_NAME}'")
            sys.exit(1)
        progress_print(f"  {len(df_family)} sequences to analyze")
    except Exception as e:
        log_error("LOAD DATA", e)
        sys.exit(1)
    stage_times["Load Data"] = time.time() - t0

    # ── 3. Fetch sequences ──────────────────────────────────────────────────
    t0 = time.time()
    print("\n=== FETCHING SEQUENCES ===")
    try:
        if "Seq" in df_family.columns and df_family["Seq"].notna().all():
            progress_print("Sequences already present — normalizing strand orientation")
            df_family["Seq"] = [
                _orient_sequence(seq, row)
                for seq, (_, row) in zip(df_family["Seq"], df_family.iterrows())
            ]
        else:
            for col in ["chr", "start", "stop"]:
                if col not in df_family.columns:
                    raise ValueError(f"Missing column '{col}' for sequence extraction")

            if genome_cache.is_loaded:
                # Fast local extraction — vectorized per-row, no network
                progress_print(f"Extracting {len(df_family)} sequences from local genome...")
                failed = []
                seqlist = []
                for i in range(len(df_family)):
                    progress_bar(i + 1, len(df_family), "  Extracting")
                    try:
                        chrom = df_family["chr"].iloc[i]
                        start = int(df_family["start"].iloc[i])
                        stop  = int(df_family["stop"].iloc[i])
                        seq = genome_cache.extract_sequence(chrom, start, stop)
                        if seq is None:
                            raise ValueError(f"No sequence for {chrom}:{start}-{stop}")
                        seqlist.append(_orient_sequence(seq, df_family.iloc[i]))
                    except Exception:
                        seqlist.append("N" * 100)
                        failed.append(i)
                df_family["Seq"] = seqlist
                if failed:
                    progress_print(f"  {len(failed)} sequences failed extraction")
            else:
                # Parallel UCSC API fetch — ~10x faster than sequential
                assembly = getattr(args, "assembly", "hg38") or "hg38"
                n_workers = getattr(args, "fetch_workers", 10)
                progress_print(f"UCSC fetch settings: assembly={assembly}, workers={n_workers}")
                seqlist = _fill_missing_sequences_parallel(
                    df_family,
                    assembly=assembly,
                    n_workers=n_workers,
                )
                df_family["Seq"] = seqlist

        df_family.to_csv(DIRS["data"] / f"{FAMILY_NAME.lower()}_with_sequences.csv", index=False)
    except Exception as e:
        log_error("FETCH SEQUENCES", e)
        sys.exit(1)
    stage_times["Sequences"] = time.time() - t0

    # ── 4. Statistics ───────────────────────────────────────────────────────
    t0 = time.time()
    print("\n=== BASIC STATISTICS ===")
    expr_cols = compute_basic_stats(
        df_family,
        label=f" — {FAMILY_NAME}",
        output_file=DIRS["stats"] / "overall_statistics.txt"
    )
    stage_times["Statistics"] = time.time() - t0

    # ── 5. Clustering ───────────────────────────────────────────────────────
    t0 = time.time()
    print("\n=== CLUSTERING ===")
    try:
        if len(df_family) < args.min_sequences:
            progress_print(
                f"  Too few sequences ({len(df_family)} < {args.min_sequences}) "
                "— assigning all to Cluster 0"
            )
            df_family["Cluster"] = 0
        else:
            rs = args.random_state if args.random_state != 0 else None
            progress_print(
                "Clustering settings: "
                f"kmer={args.kmer}, pca_dims={args.pca_dims}, n_epochs={args.n_epochs}, "
                f"random_state={rs}, compute_tsne={not args.skip_tsne}"
            )
            df_family, cluster_labels = clustering_analysis(
                df_family, kmer=args.kmer,
                out_dir=DIRS["clustering"],
                family_name=FAMILY_NAME,
                debug=args.debug,
                pca_dims=args.pca_dims,
                n_epochs=args.n_epochs,
                random_state=rs,
                compute_tsne=not args.skip_tsne,
            )

        df_family.to_csv(DIRS["data"] / f"{FAMILY_NAME.lower()}_clustered.csv", index=False)
        n_clusters = len([c for c in df_family["Cluster"].unique() if c >= 0])
        progress_print(f"  {n_clusters} clusters, {(df_family['Cluster']==-1).sum()} noise")
    except Exception as e:
        log_error("CLUSTERING", e)
        df_family["Cluster"] = 0
    stage_times["Clustering"] = time.time() - t0

    # Per-cluster stats: text files written per cluster, summary via SQL
    cs_dir = DIRS["stats"] / "per_cluster"
    cs_dir.mkdir(exist_ok=True)
    for cl in sorted(df_family["Cluster"].unique()):
        c_df = df_family[df_family["Cluster"] == cl]
        compute_basic_stats(
            c_df, label=f" — Cluster {cl}",
            output_file=cs_dir / f"cluster_{cl}_statistics.txt"
        )
    sql_summary = _cluster_summary_sql(df_family)
    sql_summary.to_csv(DIRS["stats"] / "cluster_summary.csv", index=False)

    # ── 6. Dashboard ────────────────────────────────────────────────────────
    t0 = time.time()
    print("\n=== VISUALIZATION DASHBOARD ===")
    build_dashboard(df_family, expr_cols, DIRS["visualizations"], FAMILY_NAME)
    stage_times["Dashboard"] = time.time() - t0

    # ── 7. Alignment ────────────────────────────────────────────────────────
    if not args.skip_alignment:
        t0 = time.time()
        print("\n=== ALIGNMENT ===")
        try:
            run_alignment_pipeline(df_family, OUT_DIR, FAMILY_NAME)
        except Exception as e:
            log_error("ALIGNMENT", e)
        stage_times["Alignment"] = time.time() - t0

    # ── 9. JASPAR motif / TF binding / GO analysis ─────────────────────────
    if not args.skip_motif:
        t0 = time.time()
        clustered_csv = DIRS["data"] / f"{FAMILY_NAME.lower()}_clustered.csv"
        try:
            run_motif_stage_full(args, OUT_DIR, DIRS, FAMILY_NAME,
                                 clustered_csv=str(clustered_csv))
        except SystemExit as e:
            progress_print(f"  Motif/GO stage did not complete (exit={e.code})")
            (DIRS["motif"] / "MOTIF_ANALYSIS_FAILED.txt").write_text(
                f"Motif stage exited with code {e.code}.\n"
                "Check the log, genome build, JASPAR BED path, and bedtools availability.\n"
            )
        except Exception as e:
            log_error("MOTIF / TFBS / GO", e)
            (DIRS["motif"] / "MOTIF_ANALYSIS_FAILED.txt").write_text(
                f"Motif stage failed: {type(e).__name__}: {e}\n"
            )
        stage_times["Motif+TFBS+GO"] = time.time() - t0

    # ── 10. Primer design ───────────────────────────────────────────────────
    if not args.skip_primers:
        t0 = time.time()
        print("\n=== PRIMER DESIGN ===")
        try:
            # Enable parallel genome search when --parallel-primers is set
            if getattr(args, "parallel_primers", False) and genome_cache.is_loaded:
                genome_cache._default_parallel = True
            design_primers(
                df_family,
                primer_k=args.primer_kmer,
                top_global=args.top_global,
                top_cluster=args.top_cluster,
                genome_fa=None if args.skip_genome else HG38_FA,
                genome_cache=None if args.skip_genome else genome_cache,
                primer_timeout=args.primer_timeout,
                out_dir=DIRS["primers"],
                family_name=FAMILY_NAME,
            )
        except Exception as e:
            log_error("PRIMER DESIGN", e)
        stage_times["Primers"] = time.time() - t0

    # ── Summary ─────────────────────────────────────────────────────────────
    total = time.time() - pipeline_start
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Family:    {FAMILY_NAME}")
    print(f"  Sequences: {len(df_family)}")
    print(f"  Clusters:  {n_clusters if 'n_clusters' in dir() else 'N/A'}")
    print(f"  Output:    {OUT_DIR.resolve()}")
    print(f"  Total:     {total:.1f}s ({total/60:.1f} min)")
    print()
    print("  Stage timings:")
    for stage, seconds in stage_times.items():
        print(f"    {stage:<14} {seconds:>8.1f}s")
    print()
    print(f"  Dashboard:     07_visualizations/index.html")
    print(f"  CIAlign:       cialign_plots/index.html")
    print(f"  Primers:       06_primers/selected_primers_summary.csv")
    print(f"  Clustering:    03_clustering/clustering_visualization.html")
    print(f"  Consensus:     05_consensus/")
    print(f"  Motifs:        motif_analysis/overall_top_motifs.png")
    print(f"  TF binding:    05_tfbs/cluster_tf_binding_heatmap.html")
    print(f"  GO:            go_annotations/")
    print("=" * 60)

    return df_family


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _check_hpc_environment()
    args = parse_args()
    run_pipeline(args)
