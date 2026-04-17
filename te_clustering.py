#!/usr/bin/env python3
"""
te_clustering.py
K-mer encoding → PCA / UMAP / t-SNE embeddings → HDBSCAN clustering.

Can be imported by query.py or called standalone for re-clustering:
    python te_clustering.py --input clustered_data.csv --kmer 18
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── helpers ────────────────────────────────────────────────────────────────

def _pp(msg):
    import datetime
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _pbar(cur, tot, prefix="Progress", length=40):
    pct = cur / tot if tot else 0
    filled = int(length * pct)
    bar = "█" * filled + "░" * (length - filled)
    print(f"\r{prefix} |{bar}| {cur}/{tot} ({pct*100:.1f}%)", end="", flush=True)
    if cur >= tot:
        print()


# ── main clustering function ────────────────────────────────────────────────

def clustering_analysis(df, kmer=18, min_cluster_size=None, out_dir=None,
                        family_name="FAMILY", debug=False):
    """Run full k-mer + UMAP/PCA/tSNE + HDBSCAN clustering on df['Seq'].

    Args:
        df:               DataFrame with a 'Seq' column.
        kmer:             K-mer length (default 18).
        min_cluster_size: HDBSCAN min_cluster_size (default N//7, min 5).
        out_dir:          Path-like; clustering visualizations saved here.
        family_name:      String used in plot titles / filenames.
        debug:            Print verbose output.

    Returns:
        df with columns 'Cluster', 'umap_x', 'umap_y', 'pca_x', 'pca_y',
        'tsne_x', 'tsne_y' added; also the UMAP cluster label array.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import umap as _umap
    import hdbscan as _hdbscan

    seqs = df["Seq"].astype(str).tolist()
    n = len(seqs)

    if n < 2:
        _pp(f"  Too few sequences ({n}) — assigning all to Cluster 0")
        df = df.copy()
        df["Cluster"] = 0
        df["umap_x"] = df["umap_y"] = df["pca_x"] = df["pca_y"] = \
            df["tsne_x"] = df["tsne_y"] = 0.0
        return df, np.zeros(n, dtype=int)

    # ── Step 1: K-mer encoding ─────────────────────────────────────────────
    _pp(f"  Step 1/5: Encoding {n} sequences as {kmer}-mer vectors...")
    vec = CountVectorizer(analyzer="char", ngram_range=(kmer, kmer))
    encoded = vec.fit_transform(seqs).toarray()
    n_feat = encoded.shape[1]
    _pp(f"    → {n_feat} unique {kmer}-mers")

    if n_feat < 2:
        _pp("    Too few k-mers — assigning all to Cluster 0")
        df = df.copy()
        df["Cluster"] = 0
        df["umap_x"] = df["umap_y"] = df["pca_x"] = df["pca_y"] = \
            df["tsne_x"] = df["tsne_y"] = 0.0
        return df, np.zeros(n, dtype=int)

    # ── Step 2: PCA ────────────────────────────────────────────────────────
    nc = min(2, n_feat, n)
    _pp(f"  Step 2/5: PCA (n_components={nc})...")
    pca_emb = PCA(n_components=nc, random_state=42).fit_transform(encoded)
    if pca_emb.shape[1] == 1:
        pca_emb = np.column_stack([pca_emb, np.zeros(n)])
    _pp("    → PCA done")

    # ── Step 3: UMAP ───────────────────────────────────────────────────────
    nn = max(2, min(30, n - 1))
    uc = min(2, n_feat)
    _pp(f"  Step 3/5: UMAP (n_neighbors={nn})...")
    try:
        umap_emb = _umap.UMAP(
            n_neighbors=nn, min_dist=0.0, n_components=uc,
            random_state=42, verbose=debug
        ).fit_transform(encoded)
        _pp("    → UMAP done")
    except Exception as e:
        _pp(f"    UMAP failed: {e} — using PCA fallback")
        umap_emb = pca_emb.copy()
    if umap_emb.shape[1] == 1:
        umap_emb = np.column_stack([umap_emb, np.zeros(n)])

    # ── Step 4: t-SNE ──────────────────────────────────────────────────────
    perp = max(5, min(30, n - 1))
    tc = min(2, n_feat)
    _pp(f"  Step 4/5: t-SNE (perplexity={perp})...")
    try:
        tsne_emb = TSNE(
            n_components=tc, perplexity=perp,
            init="random" if n_feat < 2 else "pca",
            random_state=42, verbose=1 if debug else 0
        ).fit_transform(encoded)
        _pp("    → t-SNE done")
    except Exception as e:
        _pp(f"    t-SNE failed: {e} — using PCA fallback")
        tsne_emb = pca_emb.copy()
    if tsne_emb.shape[1] == 1:
        tsne_emb = np.column_stack([tsne_emb, np.zeros(n)])

    # ── Step 5: HDBSCAN ────────────────────────────────────────────────────
    mcs = min_cluster_size if min_cluster_size else max(5, n // 7)
    _pp(f"  Step 5/5: HDBSCAN (min_cluster_size={mcs})...")

    def _cluster(emb, name):
        lbl = _hdbscan.HDBSCAN(min_samples=7, min_cluster_size=mcs).fit_predict(emb)
        nc_ = len(set(lbl)) - (1 if -1 in lbl else 0)
        _pp(f"    {name}: {nc_} clusters found")
        return lbl

    pca_labels = _cluster(pca_emb, "PCA")
    umap_labels = _cluster(umap_emb, "UMAP")
    tsne_labels = _cluster(tsne_emb, "t-SNE")

    # All-noise fallback
    if len(set(umap_labels)) == 1 and -1 in umap_labels:
        _pp("    UMAP all-noise — assigning Cluster 0")
        umap_labels = np.zeros(n, dtype=int)

    # ── Update DataFrame ───────────────────────────────────────────────────
    df = df.copy()
    df["Cluster"] = umap_labels
    df["pca_x"]  = pca_emb[:, 0];  df["pca_y"]  = pca_emb[:, 1]
    df["umap_x"] = umap_emb[:, 0]; df["umap_y"] = umap_emb[:, 1]
    df["tsne_x"] = tsne_emb[:, 0]; df["tsne_y"] = tsne_emb[:, 1]

    # ── Save visualizations ────────────────────────────────────────────────
    if out_dir is not None:
        _save_clustering_viz(
            df, pca_emb, pca_labels, umap_emb, umap_labels,
            tsne_emb, tsne_labels, out_dir, family_name, kmer
        )
        # Coordinates CSV
        coords_path = Path(out_dir) / "clustering_coordinates.csv"
        coord_cols = [c for c in ["chr","start","stop","strand","Cluster",
                                   "pca_x","pca_y","umap_x","umap_y",
                                   "tsne_x","tsne_y"] if c in df.columns]
        df[coord_cols].to_csv(coords_path, index=False)
        _pp(f"    Coordinates saved to {coords_path}")

    return df, umap_labels


def _save_clustering_viz(df, pca_emb, pca_lbl, umap_emb, umap_lbl,
                          tsne_emb, tsne_lbl, out_dir, family_name, kmer):
    """Save interactive Plotly clustering visualization."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=[f"PCA (k={kmer})", f"UMAP (k={kmer})", f"t-SNE (k={kmer})"]
        )

        def _add(emb, col, labels):
            uq = sorted(set(labels))
            cmap = {c: i for i, c in enumerate(uq)}
            htexts = []
            for i, lbl in enumerate(labels):
                row = df.iloc[i]
                coord = f"{row.get('chr','?')}:{row.get('start','?')}-{row.get('stop','?')}"
                htexts.append(
                    f"<b>Row {i}</b><br>{coord}<br>Cluster: {lbl}"
                )
            fig.add_trace(
                go.Scatter(
                    x=emb[:, 0], y=emb[:, 1], mode="markers",
                    marker=dict(size=5, color=[cmap[l] for l in labels],
                                colorscale="Viridis"),
                    text=htexts, hovertemplate="%{text}<extra></extra>"
                ),
                row=1, col=col
            )

        _add(pca_emb, 1, pca_lbl)
        _add(umap_emb, 2, umap_lbl)
        _add(tsne_emb, 3, tsne_lbl)
        fig.update_layout(
            width=1500, height=450, showlegend=False,
            title=f"{family_name} Clustering (k={kmer})"
        )
        out_path = out_dir / "clustering_visualization.html"
        fig.write_html(out_path)
        _pp(f"    Clustering visualization → {out_path}")
    except Exception as e:
        _pp(f"    WARNING: could not save clustering viz: {e}")


# ── CLI entry point ─────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Re-run clustering on an existing clustered_data.csv")
    p.add_argument("--input", required=True, help="Input CSV (must have Seq column)")
    p.add_argument("--output", default=None, help="Output CSV (default: overwrite input)")
    p.add_argument("--kmer", type=int, default=18)
    p.add_argument("--min-cluster-size", type=int, default=None)
    p.add_argument("--out-dir", default=".", help="Directory for visualization files")
    p.add_argument("--family", default="FAMILY", help="Family name for plot titles")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    df = pd.read_csv(args.input)
    if "Seq" not in df.columns:
        print("ERROR: Input CSV must have a 'Seq' column.")
        sys.exit(1)

    df_out, _ = clustering_analysis(
        df, kmer=args.kmer,
        min_cluster_size=args.min_cluster_size,
        out_dir=args.out_dir,
        family_name=args.family,
        debug=args.debug
    )
    out_path = args.output or args.input
    df_out.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
