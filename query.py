#!/usr/bin/env python3
"""
query.py — TE Analysis Pipeline Orchestrator
─────────────────────────────────────────────────────────────────────────────
Stages:
  1. Load reference genome (GenomeCache)
  2. Load & filter TE data
  3. Extract sequences (local genome or UCSC API)
  4. Basic statistics
  5. Clustering (UMAP / HDBSCAN) → te_clustering.py
  6. Visualization dashboard
  7. Primer design                → te_primers.py
  8. Alignment + CIAlign          → te_alignment.py

Usage:
  python query.py --input te_data.csv --family HERVK9 --genome /path/hg38.fa
  python query.py --test                          # mock test data, no genome needed
  python query.py --input data.csv --skip-genome  # skip primer genome search

Can also be called programmatically after setting df in the calling scope.
"""

import argparse
import datetime
import os
import sys
import time
import traceback
from pathlib import Path

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
    )
    p.add_argument("--input",  type=str, default=None, help="Input CSV file")
    p.add_argument("--family", type=str, default="HERVK9", help="TE family name")
    p.add_argument("--genome", type=str, default=None,
                   help="Path to genome FASTA (enables local extraction + primer search)")
    p.add_argument("--output", type=str, default="results",
                   help="Base output directory (default: results)")
    p.add_argument("--kmer",   type=int, default=18, help="K-mer size (default: 18)")
    p.add_argument("--primer-kmer", type=int, default=18, help="Primer k-mer size")
    p.add_argument("--top-global",  type=int, default=8)
    p.add_argument("--top-cluster", type=int, default=5)
    p.add_argument("--min-sequences", type=int, default=10,
                   help="Minimum sequences needed for clustering (default: 10)")
    p.add_argument("--primer-timeout", type=int, default=120,
                   help="Timeout for primer genome search in seconds (default: 120)")
    p.add_argument("--test",        action="store_true", help="Run with mock test data")
    p.add_argument("--debug",       action="store_true")
    p.add_argument("--skip-genome", action="store_true", help="Skip genome-wide primer search")
    p.add_argument("--skip-alignment", action="store_true")
    p.add_argument("--skip-primers",   action="store_true")
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
# STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_basic_stats(df, label="", output_file=None):
    seqs = df["Seq"].astype(str)
    lengths = seqs.apply(len)
    gc = seqs.apply(lambda s: (s.count("G") + s.count("C")) / len(s) if len(s) > 0 else np.nan)

    numeric = list(df.select_dtypes(include=[np.number]).columns)
    exclude = {"start", "stop", "Unnamed: 0", "chr", "Cluster", "_total_expr"}
    expr_cols = [c for c in numeric if c not in exclude]

    lines = [
        f"{'='*60}", f"STATISTICS{label}", f"{'='*60}",
        f"Dataset size: {len(df)} sequences",
        "", "=== SEQUENCE LENGTH ===",
        f"Mean: {lengths.mean():.1f}  Median: {lengths.median():.1f}",
        f"Std:  {lengths.std():.1f}  Range: [{lengths.min()}, {lengths.max()}]",
        "", "=== GC CONTENT ===",
        f"Mean: {gc.mean():.3f} ({gc.mean()*100:.1f}%)",
        f"Std:  {gc.std():.3f}  Range: [{gc.min():.3f}, {gc.max():.3f}]",
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
    print(f"  Date:    {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── 1. Load genome ──────────────────────────────────────────────────────
    print("\n=== LOADING REFERENCE GENOME ===")
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
        if args.test:
            progress_print("Creating mock test data...")
            df_family = create_test_data()
            FAMILY_NAME = "TEST_TE"
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
            raise RuntimeError("No data source: use --input, --test, or set df= in interactive mode")

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

            seqlist = []
            failed = []

            if genome_cache.is_loaded:
                progress_print(f"Extracting {len(df_family)} sequences from local genome...")
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
                    except Exception as e2:
                        seqlist.append("N" * 100)
                        failed.append(i)
            else:
                progress_print(f"Fetching {len(df_family)} sequences from UCSC API...")
                for i in range(len(df_family)):
                    progress_bar(i + 1, len(df_family), "  Fetching")
                    try:
                        chrom = df_family["chr"].iloc[i]
                        start = int(df_family["start"].iloc[i])
                        stop  = int(df_family["stop"].iloc[i])
                        url = (f"https://api.genome.ucsc.edu/getData/sequence?"
                               f"genome=hg38;chrom={chrom};start={start};end={stop}")
                        r = requests.get(url, timeout=30)
                        r.raise_for_status()
                        res = r.json()
                        if "error" in res:
                            raise ValueError(res["error"])
                        seqlist.append(res["dna"].upper())
                    except Exception as e2:
                        seqlist.append("N" * 100)
                        failed.append(i)

            df_family["Seq"] = seqlist
            if failed:
                progress_print(f"  {len(failed)} sequences failed to fetch")

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
            df_family, cluster_labels = clustering_analysis(
                df_family, kmer=args.kmer,
                out_dir=DIRS["clustering"],
                family_name=FAMILY_NAME,
                debug=args.debug
            )

        df_family.to_csv(DIRS["data"] / f"{FAMILY_NAME.lower()}_clustered.csv", index=False)
        n_clusters = len([c for c in df_family["Cluster"].unique() if c >= 0])
        progress_print(f"  {n_clusters} clusters, {(df_family['Cluster']==-1).sum()} noise")
    except Exception as e:
        log_error("CLUSTERING", e)
        df_family["Cluster"] = 0
    stage_times["Clustering"] = time.time() - t0

    # Per-cluster stats
    cs_dir = DIRS["stats"] / "per_cluster"
    cs_dir.mkdir(exist_ok=True)
    cluster_summary = []
    for cl in sorted(df_family["Cluster"].unique()):
        c_df = df_family[df_family["Cluster"] == cl]
        compute_basic_stats(
            c_df, label=f" — Cluster {cl}",
            output_file=cs_dir / f"cluster_{cl}_statistics.txt"
        )
        lengths = c_df["Seq"].astype(str).apply(len)
        gc = c_df["Seq"].astype(str).apply(
            lambda s: (s.count("G") + s.count("C")) / len(s) if len(s) > 0 else 0
        )
        cluster_summary.append({"cluster": cl, "size": len(c_df),
                                  "mean_length": lengths.mean(), "mean_gc": gc.mean()})
    pd.DataFrame(cluster_summary).to_csv(DIRS["stats"] / "cluster_summary.csv", index=False)

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
    args = parse_args()
    run_pipeline(args)
