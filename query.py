import os
import json
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import hdbscan

ll018234089312 = [
    "THE1A",
    "MSTA1",
    "HERV9-int",
    "LTR1D",
    "LTR12C",
    "LTR2752",
    "MER52A",
    "MER66D",
    "GSAT",
    "HSAT4",
    "HERVK9",
    "LTR8B"
]

# ==================== CONFIG ====================
FAMILY_NAME = "HERVK9"  # CHANGE THIS TO YOUR TARGET FAMILY
HG38_FA = "/project/amodzlab/index/human/hg38/hg38.fa"
BASE_OUT_DIR = Path("collab_rna")
K = 18
TOP_N_GLOBAL = 8
TOP_N_CLUSTER = 5
TOP_N_FORWARD_PRIMERS = 3  # NEW: Number of top sequences to generate forward primers from
MIN_SEQUENCES_FOR_CLUSTERING = 10  # NEW: Minimum sequences required for clustering

# Fix matplotlib backend for CIAlign (prevents Jupyter inline backend issues)
import os
os.environ['MPLBACKEND'] = 'Agg'
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

# ==================== LOAD AND FILTER DATA ====================
print("\n=== LOADING DATA ===")
# Assuming df is already loaded with the structure you provided
# df should have columns: chr, start, stop, TE_name, family, strand, and expression columns (A1_siCTRL_r1, etc.)

# Filter for target family
df_family = df[df['TE_name'].str.contains(FAMILY_NAME, case=False, na=False)].copy()
df_family = df_family.reset_index(drop=True)
print(f"Found {len(df_family)} instances of {FAMILY_NAME}")

if len(df_family) == 0:
    print(f"ERROR: No instances found for family {FAMILY_NAME}")
    exit(1)

# ==================== FETCH SEQUENCES FROM UCSC ====================
print("\n=== FETCHING SEQUENCES FROM UCSC ===")
seqlist = []
failed_indices = []

for i in range(len(df_family)):
    print(f"{i+1}/{len(df_family)}", end="\r")
    try:
        # UCSC API uses semicolons as parameter separators
        link = (f"https://api.genome.ucsc.edu/getData/sequence?"
                f"genome=hg38;chrom={df_family['chr'].iloc[i]};"
                f"start={int(df_family['start'].iloc[i])};end={int(df_family['stop'].iloc[i])}")
        r = requests.get(link, timeout=30)
        r.raise_for_status()  # Raise an error for bad HTTP status codes

        # Parse JSON response directly (no BeautifulSoup needed)
        res = r.json()

        # Check for error in response
        if 'error' in res:
            raise ValueError(f"API error: {res['error']}")

        # Extract DNA sequence
        if 'dna' in res:
            seqlist.append(res['dna'].upper())
        else:
            # Some UCSC API versions return sequence differently
            raise KeyError(f"'dna' not in response. Keys: {list(res.keys())}")
    except Exception as e:
        print(f"\nWarning: Failed to fetch sequence for row {i}: {e}")
        seqlist.append("N" * 100)  # placeholder for failed fetches
        failed_indices.append(i)

df_family['Seq'] = seqlist
print(f"\nSuccessfully fetched {len(df_family) - len(failed_indices)} sequences")
if failed_indices:
    print(f"Failed to fetch {len(failed_indices)} sequences")

# Save raw data with sequences
df_family.to_csv(DIRS['data'] / f"{FAMILY_NAME.lower()}_with_sequences.csv", index=False)
print(f"Saved sequences to {DIRS['data'] / f'{FAMILY_NAME.lower()}_with_sequences.csv'}")

# ==================== BASIC STATISTICS ====================
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

# ==================== CLUSTERING ANALYSIS ====================
# ==================== CLUSTERING ANALYSIS ====================
print("\n=== PERFORMING CLUSTERING ANALYSIS ===")

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

        # Encode sequences as k-mer count vectors
        vectorizer = CountVectorizer(analyzer="char", ngram_range=(K, K))
        encoded = vectorizer.fit_transform(seqs).toarray()
        n_features = encoded.shape[1]
        n_samples = encoded.shape[0]
        print(f"Encoded sequences into {n_features} unique {K}-mers ({n_samples} samples)")

        # Check if we have enough features for dimensionality reduction
        if n_features < 2:
            print(f"\n WARNING: Only {n_features} unique {K}-mers found.")
            print("   Sequences are too similar for clustering analysis.")
            print("   Assigning all sequences to Cluster 0.")
            return np.zeros(n_samples, dtype=int), np.column_stack([np.arange(n_samples), np.zeros(n_samples)])

        # Determine number of PCA components (must be <= min(n_samples, n_features))
        n_components = min(2, n_features, n_samples)

        # PCA embedding
        print("Computing PCA embedding...")
        pca_emb = PCA(n_components=n_components, random_state=42).fit_transform(encoded)

        # If only 1 component, add a zero column to make it 2D for visualization
        if pca_emb.shape[1] == 1:
            pca_emb = np.column_stack([pca_emb, np.zeros(n_samples)])

        # UMAP embedding
        print("Computing UMAP embedding...")
        # Ensure n_neighbors is valid (must be < number of samples)
        n_neighbors = min(30, n_samples - 1)
        if n_neighbors < 2:
            n_neighbors = 2

        # UMAP n_components must be <= n_features
        umap_components = min(2, n_features)
        umap_emb = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=0.0,
            n_components=umap_components,
            random_state=42
        ).fit_transform(encoded)

        # If only 1 component, add a zero column
        if umap_emb.shape[1] == 1:
            umap_emb = np.column_stack([umap_emb, np.zeros(n_samples)])

        # t-SNE embedding
        print("Computing t-SNE embedding...")
        perplexity = min(30, n_samples - 1)
        if perplexity < 5:
            perplexity = 5

        # t-SNE n_components must be <= n_features
        tsne_components = min(2, n_features)
        tsne_emb = TSNE(
            n_components=tsne_components,
            perplexity=perplexity,
            init='random' if n_features < 2 else 'pca',
            random_state=42
        ).fit_transform(encoded)

        # If only 1 component, add a zero column
        if tsne_emb.shape[1] == 1:
            tsne_emb = np.column_stack([tsne_emb, np.zeros(n_samples)])
        
        # Clustering on each embedding with adaptive parameters
        min_cluster_size = max(5, int(len(df)/7))
        
        print("Clustering PCA embedding...")
        pca_clusterer = hdbscan.HDBSCAN(min_samples=7, min_cluster_size=min_cluster_size)
        pca_labels = pca_clusterer.fit_predict(pca_emb)
        print(f"PCA clusters: {set(pca_labels)}")
        
        print("Clustering UMAP embedding...")
        umap_clusterer = hdbscan.HDBSCAN(min_samples=7, min_cluster_size=min_cluster_size)
        umap_labels = umap_clusterer.fit_predict(umap_emb)
        print(f"UMAP clusters: {set(umap_labels)}")
        
        print("Clustering t-SNE embedding...")
        tsne_clusterer = hdbscan.HDBSCAN(min_samples=7, min_cluster_size=min_cluster_size)
        tsne_labels = tsne_clusterer.fit_predict(tsne_emb)
        print(f"t-SNE clusters: {set(tsne_labels)}")
        
        # **NEW: Check if UMAP clustering failed (all noise)**
        if len(set(umap_labels)) == 1 and -1 in umap_labels:
            print("\n⚠️  WARNING: HDBSCAN assigned all sequences to noise (cluster -1)")
            print("   This usually happens when sequences are too similar or dataset is too small.")
            print("   Assigning all sequences to Cluster 0 instead.\n")
            umap_labels = np.zeros(len(df), dtype=int)
        
        # Create visualization with IMPROVED CLUSTER LABELING
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=["PCA (k=12)", "UMAP (k=12)", "t-SNE (k=12)"]
        )
        
        def add_panel(emb, col, labels, method_name):
            # Create color map for clusters with clear labeling
            unique_clusters = sorted(set(labels))
            cluster_colors = {c: i for i, c in enumerate(unique_clusters)}
            
            # Create hover text with cluster information
            hover_texts = []
            for i, l in enumerate(labels):
                hover_texts.append(
                    f"<b>Row {i}</b><br>"
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
        
        add_panel(pca_emb, 1, pca_labels, "PCA")
        add_panel(umap_emb, 2, umap_labels, "UMAP")
        add_panel(tsne_emb, 3, tsne_labels, "t-SNE")
        
        fig.update_layout(
            width=1500,
            height=450,
            showlegend=False,
            title=f"Clustering Analysis for {FAMILY_NAME} (k=12) - Colors indicate Cluster IDs"
        )
        
        # Save figure
        fig.write_html(DIRS['clustering'] / "clustering_visualization.html")
        print(f"Saved clustering visualization to {DIRS['clustering'] / 'clustering_visualization.html'}")
        
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
# ==================== PER-CLUSTER STATISTICS ====================
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

# ==================== GENERATE VISUALIZATION DASHBOARD ====================
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
print("\n=== STARTING PRIMER DESIGN ===")

# --- helper funcs ---
def reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]

def search_seq_chromosomal(primer, fasta=HG38_FA):
    """
    Search hg38.fa for exact matches of primer and its RC.
    Returns DataFrame of hits columns [chrom, start, stop, strand].
    """
    primer = primer.upper()
    rc = reverse_complement(primer)
    plen = len(primer)
    hits = []
    fasta_path = Path(fasta)
    print(f"Searching genome for primer {primer} (rc {rc}) in {fasta_path.name} ...")
    
    with open(fasta_path, "r") as f:
        chrom = None
        seq_buf = []
        for line in f:
            if line.startswith(">"):
                if chrom is not None:
                    seq = "".join(seq_buf).upper()
                    # forward
                    idx = seq.find(primer)
                    while idx != -1:
                        hits.append((chrom, idx+1, idx+plen, "+"))
                        idx = seq.find(primer, idx+1)
                    # reverse
                    idx = seq.find(rc)
                    while idx != -1:
                        hits.append((chrom, idx+1, idx+plen, "-"))
                        idx = seq.find(rc, idx+1)
                chrom = line[1:].strip().split()[0]
                seq_buf = []
            else:
                seq_buf.append(line.strip())
        # last chrom
        if chrom is not None:
            seq = "".join(seq_buf).upper()
            idx = seq.find(primer)
            while idx != -1:
                hits.append((chrom, idx+1, idx+plen, "+"))
                idx = seq.find(primer, idx+1)
            idx = seq.find(rc)
            while idx != -1:
                hits.append((chrom, idx+1, idx+plen, "-"))
                idx = seq.find(rc, idx+1)

    dfh = pd.DataFrame(hits, columns=["chrom","start","stop","strand"])
    print(f"  -> Found {len(dfh)} genomic hits for {primer}")
    return dfh

# ----------------- Build k-mer -> rows map -----------------
print("\nBuilding 12-mer index across sequences...")

# detect expression columns
numeric_cols = list(df2.select_dtypes(include=[np.number]).columns)
exclude = {"start", "stop", "Unnamed: 0", "chr", "Cluster"}
expr_cols = [c for c in numeric_cols if c not in exclude]
if not expr_cols:
    expr_cols = [c for c in df2.columns if any(prefix in c for prefix in 
                 ("A1_","A2_","A3_","B1_","B2_","B3_","C1_","C2_","C3_"))]
print(f"Expression columns used: {expr_cols}")

# per-row total expression
df2 = df2.reset_index(drop=True)
row_total_expr = df2[expr_cols].sum(axis=1) if expr_cols else pd.Series(0, index=df2.index)
df2["_total_expr"] = row_total_expr

kmer_to_rows = defaultdict(set)
print("Scanning sequences to collect all unique 12-mers...")
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

print(f"Total unique {K}-mers found: {len(kmer_to_rows)}")

# ----------------- compute metrics for each kmer -----------------
print("Computing coverage and total_expression per primer...")
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
    print(f"\nWARNING: No valid {K}-mers found in sequences.")
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
    for pr in selected_primers:
        df_hits = search_seq_chromosomal(pr, fasta=HG38_FA)
        primer_hits[pr] = df_hits
        fn = OUT_DIR / f"{pr}_genome_hits.csv"
        df_hits.to_csv(fn, index=False)
        print(f"Saved {fn.name}")

# ----------------- Build results DataFrame for global primers -----------------
def rows_set_to_str(rows_set):
    return ",".join(map(str, sorted(rows_set)))

if len(kmer_df) > 0:
    all_primers_df = kmer_df.copy()
    all_primers_df["rows_covered"] = all_primers_df["rows"].apply(rows_set_to_str)
    all_primers_df.sort_values(["coverage","total_expr"], ascending=[False,False])\
                  .to_csv(OUT_DIR / "all_12mer_candidates_metrics.csv", index=False)
    print(f"Saved all candidates to {OUT_DIR / 'all_12mer_candidates_metrics.csv'}")

if selected_primers:
    # Save top primers summary
    top_summary = pd.DataFrame({
        "primer": selected_primers,
        "coverage": [kmer_df.loc[kmer_df["primer"]==p,"coverage"].values[0] for p in selected_primers],
        "total_expr": [kmer_df.loc[kmer_df["primer"]==p,"total_expr"].values[0] for p in selected_primers],
        "strategy": ["cov_then_expr" if p in set(top_by_cov_expr["primer"]) else "expr_then_cov"
                     for p in selected_primers]
    })
    top_summary.to_csv(OUT_DIR / "selected_primers_summary.csv", index=False)
    print(f"Saved selected primers to {OUT_DIR / 'selected_primers_summary.csv'}")
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
        if len(s) < K:
            continue
        seen_local = set()
        for i in range(0, len(s)-K+1):
            k = s[i:i+K]
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
            primer_hits[p] = search_seq_chromosomal(p, fasta=HG38_FA)
            primer_hits[p].to_csv(OUT_DIR / f"{p}_genome_hits.csv", index=False)

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
cluster_summary_df.to_csv(OUT_DIR / "cluster_top5_primers.csv", index=False)
print(f"Saved cluster primers to {OUT_DIR / 'cluster_top5_primers.csv'}")

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

# ==================== MULTIPLE SEQUENCE ALIGNMENT ====================
print("\n=== PERFORMING MULTIPLE SEQUENCE ALIGNMENT ===")

try:
    from Bio import AlignIO
    from Bio.Align.Applications import MafftCommandline
    from Bio.Align import AlignInfo
    from Bio import SeqIO
    from io import StringIO
    import subprocess
    
    # Check if MAFFT is available
    try:
        subprocess.run(["mafft", "--version"], capture_output=True, check=True)
        mafft_available = True
    except:
        print("Warning: MAFFT not found. Attempting to install...")
        try:
            subprocess.run(["conda", "install", "-y", "-c", "bioconda", "mafft"], check=True)
            mafft_available = True
        except:
            print("Could not install MAFFT. Skipping alignment.")
            mafft_available = False
    
    if mafft_available:
        # ----------------- Global alignment -----------------
        print("\nAligning all sequences (global)...")
        input_fasta = fasta_path
        output_aligned = OUT_DIR / f"{FAMILY_NAME.lower()}_aligned.fa"
        
        # Run MAFFT alignment
        mafft_cmd = f"mafft --auto --thread -1 {input_fasta} > {output_aligned}"
        print(f"Running: {mafft_cmd}")
        result = subprocess.run(mafft_cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Global alignment saved to {output_aligned}")
            
            # Generate consensus sequence
            print("Generating global consensus sequence...")
            alignment = AlignIO.read(output_aligned, "fasta")
            summary_align = AlignInfo.SummaryInfo(alignment)
            consensus = summary_align.dumb_consensus(threshold=0.5, ambiguous='N')
            
            # Save consensus
            consensus_file = OUT_DIR / f"{FAMILY_NAME.lower()}_consensus.fa"
            with open(consensus_file, "w") as f:
                f.write(f">{FAMILY_NAME}_consensus\n")
                consensus_str = str(consensus)
                for i in range(0, len(consensus_str), 80):
                    f.write(consensus_str[i:i+80] + "\n")
            print(f"Global consensus saved to {consensus_file}")
            
            # Calculate alignment statistics
            alignment_stats_file = OUT_DIR / "alignment_stats.txt"
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
        cialign_check = subprocess.run(["CIAlign", "--version"], 
                                      capture_output=True, text=True)
        cialign_available = True
        print(f"CIAlign found: version info in stderr/stdout")
    except FileNotFoundError:
        print("CIAlign not found. Attempting installation...")
        try:
            subprocess.run(["pip", "install", "cialign"], check=True)
            cialign_available = True
            print("CIAlign installed successfully")
        except:
            print("Could not install CIAlign. Skipping CIAlign plots.")
            print("Install manually with: pip install cialign")
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
            print(f"\nGenerating CIAlign plots for {label}...")
            
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
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                   env={**os.environ, 'MPLBACKEND': 'Agg'})
            
            if result.returncode == 0:
                print(f"✓ CIAlign plots generated for {label}")
                # List generated files
                generated = list(Path(output_stem).parent.glob(f"{Path(output_stem).name}*"))
                for f in generated:
                    if f.suffix in ['.png', '.svg', '.html']:
                        print(f"  Generated: {f.name}")
                return True
            else:
                print(f"✗ CIAlign failed for {label}")
                if result.stderr:
                    print(f"  Error: {result.stderr[:200]}")
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
if len(kmer_df) > 0:
    kmer_out = kmer_df.copy()
    kmer_out["rows_covered"] = kmer_out["rows"].apply(rows_set_to_str)
    kmer_out.drop(columns=["rows"], inplace=True)
    kmer_out.to_csv(OUT_DIR / "kmer_candidate_metrics_full.csv", index=False)

if primer_hits:
    primer_hits_summary = []
    for p, dfh in primer_hits.items():
        primer_hits_summary.append({"primer": p, "genome_hits": len(dfh)})
    primer_hits_summary_df = pd.DataFrame(primer_hits_summary).sort_values("genome_hits", ascending=False)
    primer_hits_summary_df.to_csv(OUT_DIR / "primer_genome_hits_summary.csv", index=False)

print("\n" + "="*60)
print("ALL DONE! Summary:")
print("="*60)
print(f" - Family analyzed: {FAMILY_NAME}")
print(f" - All outputs in: {OUT_DIR.resolve()}")
print(f" - Number of sequences: {len(df2)}")
print(f" - Number of clusters: {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)}")
print(f" - Unique {K}-mers found: {len(kmer_to_rows) if 'kmer_to_rows' in dir() else 'N/A'}")
print(f" - Top primers selected: {len(selected_primers) if 'selected_primers' in dir() else 'N/A'}")
print("\n📊 KEY OUTPUT FILES:")
print(f" - 🎯 DASHBOARD: visualizations/index.html (OPEN THIS FIRST!)")
print(f" - 🎨 CIALIGN PLOTS: cialign_plots/index.html (Alignment visualizations)")
print(f" - 📈 Overall statistics: overall_statistics.txt")
print(f" - 📊 Cluster statistics: cluster_statistics/ directory")
print(f" - 🔬 Cluster comparison: cluster_comparison.txt")
print(f" - 🎨 Clustering visualization: clustering_visualization.html")
print(f" - 🧬 Selected primers: selected_primers_summary.csv")
print(f" - 📍 Cluster-specific primers: cluster_top5_primers.csv")
print(f" - 🧬 FASTA sequences: {FAMILY_NAME.lower()}_seqs.fa")
print(f" - 🌍 Genome hits summary: primer_genome_hits_summary.csv")
print(f" - 📐 Global alignment: {FAMILY_NAME.lower()}_aligned.fa")
print(f" - 🎯 Global consensus: {FAMILY_NAME.lower()}_consensus.fa")
print(f" - ✨ Cleaned consensus (CIAlign): cleaned_consensus/ directory")
print(f" - 📂 Cluster alignments: cluster_alignments/ directory")
print(f" - 🧬 All cluster consensuses: cluster_alignments/all_cluster_consensuses.fa")
print("="*60)

 