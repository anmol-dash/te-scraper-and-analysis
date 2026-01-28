#!/usr/bin/env python3
"""
Test Pipeline Script - Run with small mock dataset to debug each stage
"""
import os
import sys
import json
import requests
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import traceback

# Set matplotlib backend before any imports that might use it
os.environ['MPLBACKEND'] = 'Agg'

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import hdbscan

# ==================== CONFIG ====================
FAMILY_NAME = "TEST_TE"  # Test family name
HG38_FA = None  # Skip genome search for testing
BASE_OUT_DIR = Path("test_output")
K = 12  # Smaller k-mer for test
TOP_N_GLOBAL = 5
TOP_N_CLUSTER = 3
MIN_SEQUENCES_FOR_CLUSTERING = 3  # Lower threshold for testing
DEBUG = True

def debug_print(msg):
    if DEBUG:
        print(f"[DEBUG] {msg}")

# ==================== CREATE MOCK DATA ====================
print("\n" + "="*60)
print("CREATING MOCK TEST DATA")
print("="*60)

# Generate random DNA sequences that look like TEs
def random_seq(length, gc_bias=0.5):
    """Generate random DNA sequence with GC bias"""
    import random
    bases = []
    for _ in range(length):
        if random.random() < gc_bias:
            bases.append(random.choice(['G', 'C']))
        else:
            bases.append(random.choice(['A', 'T']))
    return ''.join(bases)

def mutate_seq(seq, mutation_rate=0.1):
    """Mutate a sequence with given rate"""
    import random
    bases = list(seq)
    for i in range(len(bases)):
        if random.random() < mutation_rate:
            bases[i] = random.choice(['A', 'C', 'G', 'T'])
    return ''.join(bases)

# Create a base sequence (like an ancestral TE)
np.random.seed(42)
import random
random.seed(42)

base_seq1 = random_seq(500, gc_bias=0.45)  # Cluster 1 ancestor
base_seq2 = random_seq(480, gc_bias=0.55)  # Cluster 2 ancestor

# Generate test data - 15 sequences in 2 clusters
test_data = []

# Cluster 1: 8 sequences derived from base_seq1
for i in range(8):
    seq = mutate_seq(base_seq1, mutation_rate=0.05 + i*0.01)
    test_data.append({
        'chr': f'chr{(i % 5) + 1}',
        'start': 1000000 + i * 10000,
        'stop': 1000000 + i * 10000 + len(seq),
        'TE_name': f'TEST_TE_element_{i+1}',
        'family': 'TEST_TE',
        'strand': '+' if i % 2 == 0 else '-',
        'Seq': seq,
        'A1_siCTRL_r1': np.random.uniform(0, 50),
        'A1_siCTRL_r2': np.random.uniform(0, 50),
        'A1_siKD_r1': np.random.uniform(0, 30),
        'A1_siKD_r2': np.random.uniform(0, 30),
    })

# Cluster 2: 7 sequences derived from base_seq2
for i in range(7):
    seq = mutate_seq(base_seq2, mutation_rate=0.05 + i*0.01)
    test_data.append({
        'chr': f'chr{(i % 5) + 6}',
        'start': 2000000 + i * 10000,
        'stop': 2000000 + i * 10000 + len(seq),
        'TE_name': f'TEST_TE_element_{i+9}',
        'family': 'TEST_TE',
        'strand': '+' if i % 2 == 0 else '-',
        'Seq': seq,
        'A1_siCTRL_r1': np.random.uniform(20, 100),
        'A1_siCTRL_r2': np.random.uniform(20, 100),
        'A1_siKD_r1': np.random.uniform(10, 50),
        'A1_siKD_r2': np.random.uniform(10, 50),
    })

df = pd.DataFrame(test_data)
print(f"Created mock dataset with {len(df)} sequences")
print(f"Columns: {list(df.columns)}")
print(f"Sequence lengths: {df['Seq'].str.len().describe()}")

# ==================== SETUP OUTPUT DIRECTORY ====================
OUT_DIR = BASE_OUT_DIR / FAMILY_NAME.lower()
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    debug_print(f"Created directory: {dir_path}")

print(f"\nOutput directory: {OUT_DIR.resolve()}")
print("Directory structure created:")
for name, path in DIRS.items():
    print(f"  {name:15s}: {path}")

# ==================== STAGE 1: SAVE DATA ====================
print("\n" + "="*60)
print("STAGE 1: SAVING DATA")
print("="*60)

try:
    df_family = df.copy()
    df_family.to_csv(DIRS['data'] / f"{FAMILY_NAME.lower()}_with_sequences.csv", index=False)
    print(f"✓ Saved {len(df_family)} sequences to 01_data/")
except Exception as e:
    print(f"✗ ERROR in Stage 1: {e}")
    traceback.print_exc()

# ==================== STAGE 2: BASIC STATISTICS ====================
print("\n" + "="*60)
print("STAGE 2: COMPUTING STATISTICS")
print("="*60)

try:
    def compute_basic_stats(df, label="", output_file=None):
        seqs = df['Seq'].astype(str)
        lengths = seqs.apply(len)
        gc = seqs.apply(lambda s: (s.count('G') + s.count('C')) / len(s) if len(s) > 0 else np.nan)

        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        exclude = {"start", "stop", "Unnamed: 0", "chr", "Cluster", "_total_expr"}
        expr_cols = [c for c in numeric_cols if c not in exclude]

        stats_lines = []
        stats_lines.append(f"{'='*60}")
        stats_lines.append(f"STATISTICS FOR {FAMILY_NAME}{label}")
        stats_lines.append(f"{'='*60}\n")
        stats_lines.append(f"Dataset size: {len(df)} sequences\n")

        stats_lines.append("=== SEQUENCE LENGTH STATISTICS ===")
        stats_lines.append(f"Mean:    {lengths.mean():.2f} bp")
        stats_lines.append(f"Median:  {lengths.median():.2f} bp")
        stats_lines.append(f"Min:     {lengths.min()} bp")
        stats_lines.append(f"Max:     {lengths.max()} bp\n")

        stats_lines.append("=== GC CONTENT STATISTICS ===")
        stats_lines.append(f"Mean:    {gc.mean():.3f} ({gc.mean()*100:.1f}%)")
        stats_lines.append(f"Min:     {gc.min():.3f}")
        stats_lines.append(f"Max:     {gc.max():.3f}\n")

        if expr_cols:
            stats_lines.append("=== EXPRESSION STATISTICS ===")
            for col in expr_cols:
                stats_lines.append(f"{col}: mean={df[col].mean():.2f}, max={df[col].max():.2f}")

        stats_text = "\n".join(stats_lines)

        if output_file:
            with open(output_file, "w") as f:
                f.write(stats_text)
            print(f"✓ Saved statistics to {output_file}")

        return expr_cols

    expr_cols = compute_basic_stats(df_family, label=" - OVERALL",
                                    output_file=DIRS['stats'] / "overall_statistics.txt")
    print(f"✓ Expression columns detected: {expr_cols}")

except Exception as e:
    print(f"✗ ERROR in Stage 2: {e}")
    traceback.print_exc()
    expr_cols = []

# ==================== STAGE 3: CLUSTERING ====================
print("\n" + "="*60)
print("STAGE 3: CLUSTERING ANALYSIS")
print("="*60)

try:
    def clustering_analysis(df):
        seqs = df["Seq"].tolist()
        debug_print(f"Clustering {len(seqs)} sequences")

        # Encode sequences as k-mer count vectors
        vectorizer = CountVectorizer(analyzer="char", ngram_range=(K, K))
        encoded = vectorizer.fit_transform(seqs).toarray()
        n_features = encoded.shape[1]
        n_samples = encoded.shape[0]
        debug_print(f"Encoded into {n_features} unique {K}-mers ({n_samples} samples)")

        if n_features < 2:
            print("⚠ Not enough k-mer diversity for clustering")
            return np.zeros(n_samples, dtype=int), np.column_stack([np.arange(n_samples), np.zeros(n_samples)])

        n_components = min(2, n_features, n_samples)

        # PCA
        debug_print("Computing PCA...")
        pca_emb = PCA(n_components=n_components, random_state=42).fit_transform(encoded)
        if pca_emb.shape[1] == 1:
            pca_emb = np.column_stack([pca_emb, np.zeros(n_samples)])

        # UMAP
        debug_print("Computing UMAP...")
        n_neighbors = min(15, n_samples - 1)
        if n_neighbors < 2:
            n_neighbors = 2
        umap_emb = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2,
                            random_state=42).fit_transform(encoded)

        # t-SNE
        debug_print("Computing t-SNE...")
        perplexity = min(5, n_samples - 1)
        if perplexity < 2:
            perplexity = 2
        tsne_emb = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(encoded)

        # HDBSCAN clustering
        debug_print("Running HDBSCAN...")
        min_cluster_size = max(3, int(len(df)/5))
        debug_print(f"min_cluster_size={min_cluster_size}")

        clusterer = hdbscan.HDBSCAN(min_samples=2, min_cluster_size=min_cluster_size)
        umap_labels = clusterer.fit_predict(umap_emb)
        debug_print(f"UMAP clusters found: {set(umap_labels)}")

        # If all noise, assign to cluster 0
        if len(set(umap_labels)) == 1 and -1 in umap_labels:
            print("⚠ All sequences assigned as noise, using Cluster 0")
            umap_labels = np.zeros(len(df), dtype=int)

        # Create visualization
        fig = make_subplots(rows=1, cols=3, subplot_titles=["PCA", "UMAP", "t-SNE"])

        def add_panel(emb, col, labels, method):
            unique_clusters = sorted(set(labels))
            colors = [unique_clusters.index(l) for l in labels]
            fig.add_trace(
                go.Scatter(x=emb[:, 0], y=emb[:, 1], mode="markers",
                          marker=dict(size=8, color=colors, colorscale="Viridis"),
                          text=[f"Row {i}, Cluster {l}" for i, l in enumerate(labels)],
                          name=method),
                row=1, col=col
            )

        add_panel(pca_emb, 1, umap_labels, "PCA")
        add_panel(umap_emb, 2, umap_labels, "UMAP")
        add_panel(tsne_emb, 3, umap_labels, "t-SNE")

        fig.update_layout(width=1200, height=400, title=f"Clustering Analysis - {FAMILY_NAME}")
        fig.write_html(DIRS['clustering'] / "clustering_visualization.html")
        debug_print(f"Saved clustering visualization")

        return umap_labels, umap_emb

    if len(df_family) >= MIN_SEQUENCES_FOR_CLUSTERING:
        cluster_labels, embedding = clustering_analysis(df_family)
        df_family['Cluster'] = cluster_labels
    else:
        print(f"⚠ Only {len(df_family)} sequences, skipping clustering")
        df_family['Cluster'] = 0
        cluster_labels = np.array([0] * len(df_family))

    # Save clustered data
    df_family.to_csv(DIRS['data'] / f"{FAMILY_NAME.lower()}_clustered.csv", index=False)
    print(f"✓ Clustering complete. Clusters found: {sorted(df_family['Cluster'].unique())}")
    print(f"✓ Saved clustering visualization to 03_clustering/")

except Exception as e:
    print(f"✗ ERROR in Stage 3: {e}")
    traceback.print_exc()
    df_family['Cluster'] = 0
    cluster_labels = np.array([0] * len(df_family))

# ==================== STAGE 4: PER-CLUSTER STATISTICS ====================
print("\n" + "="*60)
print("STAGE 4: PER-CLUSTER STATISTICS")
print("="*60)

try:
    cluster_stats_dir = DIRS['stats'] / "per_cluster"
    cluster_stats_dir.mkdir(exist_ok=True)

    for cluster in sorted(df_family['Cluster'].unique()):
        cluster_df = df_family[df_family['Cluster'] == cluster]
        cluster_label = f" - CLUSTER {cluster} (n={len(cluster_df)})"
        cluster_file = cluster_stats_dir / f"cluster_{cluster}_statistics.txt"
        compute_basic_stats(cluster_df, label=cluster_label, output_file=cluster_file)

    # Cluster comparison
    with open(DIRS['stats'] / "cluster_comparison.txt", "w") as f:
        f.write(f"CLUSTER COMPARISON FOR {FAMILY_NAME}\n")
        f.write("="*40 + "\n\n")
        for cluster in sorted(df_family['Cluster'].unique()):
            cluster_df = df_family[df_family['Cluster'] == cluster]
            f.write(f"Cluster {cluster}: {len(cluster_df)} sequences\n")

    print(f"✓ Per-cluster statistics saved to 02_statistics/")

except Exception as e:
    print(f"✗ ERROR in Stage 4: {e}")
    traceback.print_exc()

# ==================== STAGE 5: ALIGNMENTS ====================
print("\n" + "="*60)
print("STAGE 5: MULTIPLE SEQUENCE ALIGNMENT")
print("="*60)

try:
    import subprocess
    from Bio import AlignIO
    from Bio.Align import AlignInfo

    # Check MAFFT
    mafft_check = subprocess.run(["mafft", "--version"], capture_output=True, text=True)
    debug_print(f"MAFFT version check: returncode={mafft_check.returncode}")

    # Write FASTA
    fasta_path = OUT_DIR / f"{FAMILY_NAME.lower()}_seqs.fa"
    with open(fasta_path, "w") as fh:
        for rid, row in df_family.iterrows():
            header = f">row{rid}|{row.get('TE_name','')}|cluster{row.get('Cluster','')}"
            seq = str(row["Seq"]).strip().upper()
            fh.write(header + "\n")
            for i in range(0, len(seq), 80):
                fh.write(seq[i:i+80] + "\n")
    debug_print(f"Wrote FASTA to {fasta_path}")

    # Global alignment
    output_aligned = OUT_DIR / f"{FAMILY_NAME.lower()}_aligned.fa"
    mafft_cmd = f"mafft --auto --thread -1 {fasta_path} > {output_aligned} 2>/dev/null"
    debug_print(f"Running: {mafft_cmd}")
    result = subprocess.run(mafft_cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0 and output_aligned.exists() and output_aligned.stat().st_size > 0:
        print(f"✓ Global alignment saved to {output_aligned.name}")

        # Generate consensus
        alignment = AlignIO.read(output_aligned, "fasta")
        summary_align = AlignInfo.SummaryInfo(alignment)
        consensus = summary_align.dumb_consensus(threshold=0.5, ambiguous='N')

        consensus_file = OUT_DIR / f"{FAMILY_NAME.lower()}_consensus.fa"
        with open(consensus_file, "w") as f:
            f.write(f">{FAMILY_NAME}_consensus\n")
            consensus_str = str(consensus)
            for i in range(0, len(consensus_str), 80):
                f.write(consensus_str[i:i+80] + "\n")
        print(f"✓ Consensus saved to {consensus_file.name}")

        # Also copy to 05_consensus
        import shutil
        shutil.copy(consensus_file, DIRS['consensus'] / f"{FAMILY_NAME.lower()}_consensus.fa")

        # Alignment stats
        alignment_stats_file = DIRS['alignments'] / "alignment_stats.txt"
        with open(alignment_stats_file, "w") as f:
            f.write(f"=== GLOBAL ALIGNMENT STATISTICS ===\n\n")
            f.write(f"Number of sequences: {len(alignment)}\n")
            f.write(f"Alignment length: {alignment.get_alignment_length()}\n")
            f.write(f"Consensus length (no gaps): {len(consensus_str.replace('-', ''))}\n")
        print(f"✓ Alignment stats saved to 04_alignments/")

        # Per-cluster alignments
        cluster_alignment_dir = OUT_DIR / "cluster_alignments"
        cluster_alignment_dir.mkdir(exist_ok=True)

        for cluster in sorted(df_family['Cluster'].unique()):
            cluster_df = df_family[df_family['Cluster'] == cluster]
            if len(cluster_df) < 2:
                continue

            cluster_fasta = cluster_alignment_dir / f"cluster_{cluster}_seqs.fa"
            with open(cluster_fasta, "w") as f:
                for rid, row in cluster_df.iterrows():
                    f.write(f">row{rid}|cluster{cluster}\n")
                    seq = str(row["Seq"]).strip().upper()
                    for i in range(0, len(seq), 80):
                        f.write(seq[i:i+80] + "\n")

            cluster_aligned = cluster_alignment_dir / f"cluster_{cluster}_aligned.fa"
            cmd = f"mafft --auto --thread -1 {cluster_fasta} > {cluster_aligned} 2>/dev/null"
            subprocess.run(cmd, shell=True)

            if cluster_aligned.exists() and cluster_aligned.stat().st_size > 0:
                try:
                    aln = AlignIO.read(cluster_aligned, "fasta")
                    summary = AlignInfo.SummaryInfo(aln)
                    cons = summary.dumb_consensus(threshold=0.5, ambiguous='N')

                    cluster_cons_file = cluster_alignment_dir / f"cluster_{cluster}_consensus.fa"
                    with open(cluster_cons_file, "w") as f:
                        f.write(f">{FAMILY_NAME}_cluster{cluster}_consensus\n")
                        cons_str = str(cons)
                        for i in range(0, len(cons_str), 80):
                            f.write(cons_str[i:i+80] + "\n")
                    print(f"  ✓ Cluster {cluster} alignment and consensus complete")
                except Exception as e:
                    debug_print(f"Cluster {cluster} consensus failed: {e}")

        # Also save stats in 05_consensus
        shutil.copy(alignment_stats_file, DIRS['consensus'] / "alignment_stats.txt")

    else:
        print(f"✗ MAFFT failed: {result.stderr}")

except ImportError as e:
    print(f"✗ BioPython not available: {e}")
except Exception as e:
    print(f"✗ ERROR in Stage 5: {e}")
    traceback.print_exc()

# ==================== STAGE 6: PRIMER DESIGN ====================
print("\n" + "="*60)
print("STAGE 6: PRIMER DESIGN")
print("="*60)

try:
    df2 = df_family.copy()

    # Detect expression columns
    numeric_cols = list(df2.select_dtypes(include=[np.number]).columns)
    exclude = {"start", "stop", "Unnamed: 0", "chr", "Cluster", "_total_expr"}
    expr_cols = [c for c in numeric_cols if c not in exclude]

    # Per-row total expression
    df2 = df2.reset_index(drop=True)
    row_total_expr = df2[expr_cols].sum(axis=1) if expr_cols else pd.Series(0, index=df2.index)
    df2["_total_expr"] = row_total_expr

    # Build k-mer index
    kmer_to_rows = defaultdict(set)
    debug_print("Building k-mer index...")
    for ridx, seq in df2["Seq"].astype(str).items():
        s = seq.upper()
        L = len(s)
        if L < K:
            continue
        seen_local = set()
        for i in range(0, L - K + 1):
            kmer = s[i:i+K]
            if "N" in kmer:
                continue
            if kmer not in seen_local:
                kmer_to_rows[kmer].add(ridx)
                seen_local.add(kmer)

    debug_print(f"Total unique {K}-mers found: {len(kmer_to_rows)}")

    if len(kmer_to_rows) == 0:
        print("⚠ No valid k-mers found, skipping primer design")
    else:
        # Compute metrics
        rows_total_expr = df2["_total_expr"].to_dict()
        kmer_records = []
        for kmer, rows_set in kmer_to_rows.items():
            coverage = len(rows_set)
            total_expr = sum(rows_total_expr[r] for r in rows_set)
            kmer_records.append({"primer": kmer, "coverage": coverage, "total_expr": total_expr,
                               "rows": ",".join(map(str, sorted(rows_set)))})

        kmer_df = pd.DataFrame(kmer_records)

        # Top primers by coverage
        top_primers = kmer_df.sort_values(["coverage", "total_expr"], ascending=[False, False]).head(TOP_N_GLOBAL)

        # Save results
        top_primers.to_csv(DIRS['primers'] / "selected_primers_summary.csv", index=False)
        print(f"✓ Top {len(top_primers)} primers saved to 06_primers/")
        print("\nTop primers by coverage:")
        print(top_primers[["primer", "coverage", "total_expr"]].to_string())

        # Save all k-mer candidates
        kmer_df.to_csv(DIRS['primers'] / "all_kmer_candidates.csv", index=False)

        # Per-cluster primers
        cluster_primers = []
        for cluster in sorted(df2['Cluster'].unique()):
            cluster_rows = df2[df2['Cluster'] == cluster].index.tolist()
            cluster_kmers = []
            for kmer, rows_set in kmer_to_rows.items():
                cluster_overlap = rows_set & set(cluster_rows)
                if cluster_overlap:
                    cov = len(cluster_overlap)
                    expr = sum(rows_total_expr[r] for r in cluster_overlap)
                    cluster_kmers.append({"cluster": cluster, "primer": kmer, "coverage": cov, "total_expr": expr})

            if cluster_kmers:
                cluster_df = pd.DataFrame(cluster_kmers)
                top_cluster = cluster_df.sort_values(["coverage", "total_expr"], ascending=[False, False]).head(TOP_N_CLUSTER)
                cluster_primers.extend(top_cluster.to_dict('records'))

        if cluster_primers:
            pd.DataFrame(cluster_primers).to_csv(DIRS['primers'] / "cluster_top_primers.csv", index=False)
            print(f"✓ Per-cluster primers saved")

except Exception as e:
    print(f"✗ ERROR in Stage 6: {e}")
    traceback.print_exc()

# ==================== STAGE 7: VISUALIZATIONS ====================
print("\n" + "="*60)
print("STAGE 7: GENERATING VISUALIZATIONS")
print("="*60)

try:
    import plotly.express as px

    vis_dir = DIRS['visualizations']

    # Cluster distribution
    cluster_counts = df_family['Cluster'].value_counts().sort_index()
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Bar(
        x=[f"C{c}" for c in cluster_counts.index],
        y=cluster_counts.values,
        marker_color='steelblue',
        text=cluster_counts.values,
        textposition='auto'
    ))
    fig_dist.update_layout(title="Cluster Size Distribution", xaxis_title="Cluster", yaxis_title="Count")
    fig_dist.write_html(vis_dir / "cluster_distribution.html")
    print(f"✓ cluster_distribution.html")

    # Sequence characteristics
    df_family['length'] = df_family['Seq'].str.len()
    df_family['gc_content'] = df_family['Seq'].apply(
        lambda s: (s.count('G') + s.count('C')) / len(s) if len(s) > 0 else 0
    )

    fig_chars = make_subplots(rows=2, cols=2,
        subplot_titles=("Length by Cluster", "GC by Cluster", "Length Distribution", "GC Distribution"))

    for cluster in sorted(df_family['Cluster'].unique()):
        cdf = df_family[df_family['Cluster'] == cluster]
        fig_chars.add_trace(go.Box(y=cdf['length'], name=f"C{cluster}", showlegend=False), row=1, col=1)
        fig_chars.add_trace(go.Box(y=cdf['gc_content'], name=f"C{cluster}", showlegend=False), row=1, col=2)

    fig_chars.add_trace(go.Histogram(x=df_family['length'], nbinsx=20, showlegend=False), row=2, col=1)
    fig_chars.add_trace(go.Histogram(x=df_family['gc_content'], nbinsx=20, showlegend=False), row=2, col=2)
    fig_chars.update_layout(height=700, title="Sequence Characteristics")
    fig_chars.write_html(vis_dir / "sequence_characteristics.html")
    print(f"✓ sequence_characteristics.html")

    # Expression heatmap
    if expr_cols:
        cluster_expr = df_family.groupby('Cluster')[expr_cols].mean()
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=cluster_expr.values,
            x=cluster_expr.columns,
            y=[f"Cluster {c}" for c in cluster_expr.index],
            colorscale='Viridis'
        ))
        fig_heatmap.update_layout(title="Mean Expression by Cluster")
        fig_heatmap.write_html(vis_dir / "expression_heatmap.html")
        print(f"✓ expression_heatmap.html")

    # Index page
    html_index = f"""<!DOCTYPE html>
<html>
<head>
    <title>{FAMILY_NAME} Analysis Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
        .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .card h2 {{ color: #4CAF50; margin-top: 0; }}
        iframe {{ width: 100%; height: 400px; border: 1px solid #ddd; }}
        a {{ color: #4CAF50; }}
    </style>
</head>
<body>
    <h1>{FAMILY_NAME} Analysis Dashboard</h1>
    <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>Total sequences: {len(df_family)} | Clusters: {sorted(df_family['Cluster'].unique())}</p>

    <div class="grid">
        <div class="card">
            <h2>Cluster Distribution</h2>
            <iframe src="cluster_distribution.html"></iframe>
        </div>
        <div class="card">
            <h2>Sequence Characteristics</h2>
            <iframe src="sequence_characteristics.html"></iframe>
        </div>
        {'<div class="card"><h2>Expression Heatmap</h2><iframe src="expression_heatmap.html"></iframe></div>' if expr_cols else ''}
    </div>

    <h2>Other Files</h2>
    <ul>
        <li><a href="../03_clustering/clustering_visualization.html">Clustering Visualization</a></li>
        <li><a href="../02_statistics/overall_statistics.txt">Overall Statistics</a></li>
        <li><a href="../06_primers/selected_primers_summary.csv">Selected Primers</a></li>
    </ul>
</body>
</html>"""

    with open(vis_dir / "index.html", "w") as f:
        f.write(html_index)
    print(f"✓ index.html (dashboard)")

except Exception as e:
    print(f"✗ ERROR in Stage 7: {e}")
    traceback.print_exc()

# ==================== FINAL SUMMARY ====================
print("\n" + "="*60)
print("PIPELINE COMPLETE - CHECKING OUTPUT")
print("="*60)

for name, path in DIRS.items():
    files = list(path.glob("*"))
    if files:
        print(f"✓ {name:15s}: {len(files)} files")
        for f in files[:3]:
            print(f"    - {f.name}")
        if len(files) > 3:
            print(f"    ... and {len(files)-3} more")
    else:
        print(f"✗ {name:15s}: EMPTY!")

print(f"\n📂 All output in: {OUT_DIR.resolve()}")
print(f"🎯 Open: {DIRS['visualizations'] / 'index.html'}")
