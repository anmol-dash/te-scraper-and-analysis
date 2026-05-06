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
import os
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
    if "--test" in sys.argv or "--local" in sys.argv:
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
    return df.reset_index(drop=True)


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
                    return i, res["dna"].upper()
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
    }
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

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
            progress_print("Sequences already present — skipping fetch")
            df_family["Seq"] = df_family["Seq"].str.upper()
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
                        seqlist.append(seq)
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
                seqlist = _fetch_sequences_parallel(df_family, assembly=assembly,
                                                    n_workers=n_workers)
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

    # ── 7. Primer design ────────────────────────────────────────────────────
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

    # ── 8. Alignment ────────────────────────────────────────────────────────
    if not args.skip_alignment:
        t0 = time.time()
        print("\n=== ALIGNMENT ===")
        try:
            run_alignment_pipeline(df_family, OUT_DIR, FAMILY_NAME)
        except Exception as e:
            log_error("ALIGNMENT", e)
        stage_times["Alignment"] = time.time() - t0

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
    print("=" * 60)

    return df_family


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _check_hpc_environment()
    args = parse_args()
    run_pipeline(args)
