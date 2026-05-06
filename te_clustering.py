#!/usr/bin/env python3
"""
te_clustering.py
K-mer encoding → SVD dimensionality reduction → UMAP / PCA / t-SNE → HDBSCAN.

Pipeline (optimised for biological sequences):
  Sequences
  → k-mer counts  (sparse float32, CountVectorizer)
  → TruncatedSVD  (50 dims, operates on sparse matrix)
  → UMAP          (n_jobs=-1, euclidean on SVD output)
  → t-SNE         (barnes_hut, also on SVD output)
  → HDBSCAN       (on UMAP 2-D embedding)

Key speed levers vs the old implementation:
  - k=6 default  (was 18): 4^6 = 4096 max features vs potentially 100k+ 18-mers
  - TruncatedSVD 50 dims before UMAP (was PCA 2 dims *after* UMAP ran on raw matrix)
  - Sparse matrix never densified until SVD output
  - n_jobs=-1 on UMAP and HDBSCAN
  - float32 throughout; n_epochs capped
  - Large-N (>20k): fit UMAP on random subset, transform the rest

Standalone usage:
    python te_clustering.py --input clustered_data.csv --kmer 6
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_NUMBA_CACHE = Path(tempfile.gettempdir()) / "gameca_numba_cache"
_NUMBA_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(_NUMBA_CACHE))


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

def clustering_analysis(df, kmer=6, min_cluster_size=None, out_dir=None,
                        family_name="FAMILY", debug=False,
                        pca_dims=50, n_epochs=200, random_state=42,
                        compute_tsne=True):
    """Run k-mer + SVD + UMAP/PCA/tSNE + HDBSCAN clustering on df['Seq'].

    Args:
        df:               DataFrame with a 'Seq' column.
        kmer:             K-mer length.  6 is recommended for DNA (k=18 was the
                          old default but creates >100k sparse features).
        min_cluster_size: HDBSCAN min_cluster_size (default N//7, min 5).
        out_dir:          Path-like; visualisations saved here.
        family_name:      String used in plot titles / filenames.
        debug:            Print verbose UMAP/t-SNE output.
        pca_dims:         SVD components fed into UMAP / t-SNE (default 50).
        n_epochs:         UMAP optimisation epochs (default 200).
        random_state:     Integer seed for reproducibility (default 42).
                          Pass None to enable full UMAP parallelism (n_jobs=-1
                          is silently overridden to 1 when a seed is set).
        compute_tsne:     Compute t-SNE embedding/labels. Set False for faster
                          dashboard reclustering when only UMAP is displayed.

    Returns:
        (df_with_cluster_cols, umap_label_array)
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.manifold import TSNE
    import umap as _umap
    import hdbscan as _hdbscan

    seqs = df["Seq"].astype(str).tolist()
    n = len(seqs)

    _SMALL = {"umap_x": 0.0, "umap_y": 0.0,
              "pca_x":  0.0, "pca_y":  0.0,
              "tsne_x": 0.0, "tsne_y": 0.0,
              "Cluster": 0}

    if n < 2:
        _pp(f"  Too few sequences ({n}) — assigning all to Cluster 0")
        df = df.copy()
        for k2, v in _SMALL.items():
            df[k2] = v
        return df, np.zeros(n, dtype=int)

    # ── Step 1: k-mer encoding (sparse float32) ────────────────────────────
    _pp(f"  Step 1/5: Encoding {n} sequences as {kmer}-mer counts (sparse)…")
    vec = CountVectorizer(
        analyzer="char", ngram_range=(kmer, kmer),
        dtype=np.float32,
    )
    X_sparse = vec.fit_transform(seqs)          # stays sparse
    n_feat = X_sparse.shape[1]
    _pp(f"    → {n_feat} unique {kmer}-mers  "
        f"(matrix {n}×{n_feat}, density "
        f"{X_sparse.nnz / (n * n_feat):.3%})")

    if n_feat < 2:
        _pp("    Too few k-mers — assigning all to Cluster 0")
        df = df.copy()
        for k2, v in _SMALL.items():
            df[k2] = v
        return df, np.zeros(n, dtype=int)

    # ── Step 2: TruncatedSVD (sparse-aware, replaces dense PCA) ───────────
    n_svd = min(pca_dims, n_feat - 1, n - 1)
    _pp(f"  Step 2/5: TruncatedSVD ({n_svd} components)…")
    svd = TruncatedSVD(n_components=n_svd, random_state=random_state)
    X_svd = svd.fit_transform(X_sparse).astype(np.float32)   # (n, n_svd)
    var_exp = svd.explained_variance_ratio_.sum()
    _pp(f"    → SVD done  ({var_exp*100:.1f}% variance explained)")

    # 2-D PCA-like visualisation: use the first 2 SVD components directly
    pca_emb = X_svd[:, :2].copy()
    if pca_emb.shape[1] < 2:
        pca_emb = np.column_stack([pca_emb, np.zeros(n, dtype=np.float32)])

    # ── Step 3: UMAP on SVD output ─────────────────────────────────────────
    nn = max(2, min(15, n - 1))   # 5–15 is faster and sufficient
    _pp(f"  Step 3/5: UMAP (n_neighbors={nn}, n_epochs={n_epochs}, n_jobs=-1)…")
    try:
        # Note: umap-learn overrides n_jobs=1 when random_state is set.
        # Pass random_state=None for true multi-core UMAP (non-reproducible).
        umap_model = _umap.UMAP(
            n_neighbors=nn,
            min_dist=0.1,
            n_components=2,
            metric="euclidean",
            n_epochs=n_epochs,
            n_jobs=-1,
            low_memory=(n > 10_000),
            random_state=random_state,
            verbose=debug,
        )
        if n > 20_000:
            # Fit on a random subsample, then project the rest
            rng  = np.random.default_rng(random_state)
            idx  = rng.choice(n, size=20_000, replace=False)
            mask = np.zeros(n, dtype=bool); mask[idx] = True
            _pp(f"    Large dataset — fitting UMAP on 20 000-point subsample…")
            umap_model.fit(X_svd[mask])
            umap_emb = np.zeros((n, 2), dtype=np.float32)
            umap_emb[mask]  = umap_model.transform(X_svd[mask]).astype(np.float32)
            umap_emb[~mask] = umap_model.transform(X_svd[~mask]).astype(np.float32)
        else:
            umap_emb = umap_model.fit_transform(X_svd).astype(np.float32)
        _pp("    → UMAP done")
    except Exception as e:
        _pp(f"    UMAP failed: {e} — using SVD 2-D fallback")
        umap_emb = pca_emb.copy()

    if umap_emb.shape[1] < 2:
        umap_emb = np.column_stack([umap_emb, np.zeros(n, dtype=np.float32)])

    # ── Step 4: t-SNE on SVD output ────────────────────────────────────────
    if compute_tsne:
        perp = max(5, min(30, (n - 1) // 2))
        # t-SNE init from PCA is faster and more stable (requires n_svd >= 2)
        tsne_init = "pca" if n_svd >= 2 else "random"
        _pp(f"  Step 4/5: t-SNE (perplexity={perp}, init={tsne_init!r})…")
        try:
            tsne_emb = TSNE(
                n_components=2,
                perplexity=perp,
                init=tsne_init,
                method="barnes_hut",
                random_state=random_state,
                verbose=1 if debug else 0,
            ).fit_transform(X_svd).astype(np.float32)
            _pp("    → t-SNE done")
        except Exception as e:
            _pp(f"    t-SNE failed: {e} — using SVD 2-D fallback")
            tsne_emb = pca_emb.copy()
    else:
        _pp("  Step 4/5: t-SNE skipped for fast reclustering")
        tsne_emb = pca_emb.copy()

    if tsne_emb.shape[1] < 2:
        tsne_emb = np.column_stack([tsne_emb, np.zeros(n, dtype=np.float32)])

    # ── Step 5: HDBSCAN on UMAP 2-D ───────────────────────────────────────
    mcs = min_cluster_size if min_cluster_size else max(5, n // 7)
    # min_samples: lower = fewer points required to form a cluster core
    min_s = max(1, mcs // 3)
    _pp(f"  Step 5/5: HDBSCAN (min_cluster_size={mcs}, min_samples={min_s})…")

    def _cluster(emb, name):
        lbl = _hdbscan.HDBSCAN(
            min_samples=min_s,
            min_cluster_size=mcs,
            core_dist_n_jobs=-1,
        ).fit_predict(emb)
        nc_ = len(set(lbl)) - (1 if -1 in lbl else 0)
        _pp(f"    {name}: {nc_} cluster(s) found")
        return lbl

    pca_labels  = _cluster(pca_emb,  "PCA")
    umap_labels = _cluster(umap_emb, "UMAP")
    tsne_labels = _cluster(tsne_emb, "t-SNE") if compute_tsne else pca_labels

    # All-noise fallback
    if len(set(umap_labels)) == 1 and -1 in umap_labels:
        _pp("    UMAP all-noise — assigning Cluster 0")
        umap_labels = np.zeros(n, dtype=int)

    # ── Update DataFrame ───────────────────────────────────────────────────
    df = df.copy()
    df["Cluster"] = umap_labels
    df["pca_x"]   = pca_emb[:, 0];  df["pca_y"]   = pca_emb[:, 1]
    df["umap_x"]  = umap_emb[:, 0]; df["umap_y"]  = umap_emb[:, 1]
    df["tsne_x"]  = tsne_emb[:, 0]; df["tsne_y"]  = tsne_emb[:, 1]

    # ── Save visualisations ────────────────────────────────────────────────
    if out_dir is not None:
        _save_clustering_viz(
            df, pca_emb, pca_labels, umap_emb, umap_labels,
            tsne_emb, tsne_labels, out_dir, family_name, kmer
        )
        coords_path = Path(out_dir) / "clustering_coordinates.csv"
        coord_cols = [c for c in ["chr", "start", "stop", "strand", "Cluster",
                                   "pca_x", "pca_y", "umap_x", "umap_y",
                                   "tsne_x", "tsne_y"] if c in df.columns]
        df[coord_cols].to_csv(coords_path, index=False)
        _pp(f"    Coordinates → {coords_path}")

    return df, umap_labels


def _save_clustering_viz(df, pca_emb, pca_lbl, umap_emb, umap_lbl,
                          tsne_emb, tsne_lbl, out_dir, family_name, kmer):
    """Save interactive Plotly clustering visualisation."""
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
            uq   = sorted(set(labels))
            cmap = {c: i for i, c in enumerate(uq)}
            htexts = []
            for i, lbl in enumerate(labels):
                row = df.iloc[i]
                coord = (f"{row.get('chr','?')}:"
                         f"{row.get('start','?')}-{row.get('stop','?')}")
                htexts.append(f"<b>Row {i}</b><br>{coord}<br>Cluster: {lbl}")
            fig.add_trace(
                go.Scatter(
                    x=emb[:, 0], y=emb[:, 1], mode="markers",
                    marker=dict(size=5, color=[cmap[l] for l in labels],
                                colorscale="Viridis"),
                    text=htexts, hovertemplate="%{text}<extra></extra>",
                ),
                row=1, col=col,
            )

        _add(pca_emb, 1, pca_lbl)
        _add(umap_emb, 2, umap_lbl)
        _add(tsne_emb, 3, tsne_lbl)
        fig.update_layout(
            width=1500, height=450, showlegend=False,
            title=f"{family_name} Clustering (k={kmer})",
        )
        out_path = out_dir / "clustering_visualization.html"
        fig.write_html(out_path)
        _pp(f"    Clustering visualisation → {out_path}")
    except Exception as e:
        _pp(f"    WARNING: could not save clustering viz: {e}")


# ── CLI entry point ─────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Re-run clustering on an existing *_clustered.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python te_clustering.py --input data.csv --kmer 6\n"
            "  python te_clustering.py --input data.csv --kmer 6 --pca-dims 100 --n-epochs 150\n"
        ),
    )
    p.add_argument("--input",    required=True, help="Input CSV (must have Seq column)")
    p.add_argument("--output",   default=None,  help="Output CSV (default: overwrite input)")
    p.add_argument("--kmer",     type=int, default=6,
                   help="K-mer length (default 6; was 18 in older versions)")
    p.add_argument("--pca-dims", type=int, default=50,
                   help="SVD components fed into UMAP / t-SNE (default 50)")
    p.add_argument("--n-epochs", type=int, default=200,
                   help="UMAP optimisation epochs (default 200)")
    p.add_argument("--min-cluster-size", type=int, default=None)
    p.add_argument("--out-dir",  default=".", help="Directory for visualisation files")
    p.add_argument("--family",   default="FAMILY", help="Family name for plot titles")
    p.add_argument("--random-state", type=int, default=42,
                   help="Random seed (default 42); pass 0 to disable for full parallelism")
    p.add_argument("--debug",    action="store_true")
    p.add_argument("--skip-tsne", action="store_true",
                   help="Skip t-SNE for faster UMAP-only reclustering")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    df = pd.read_csv(args.input)
    if "Seq" not in df.columns:
        print("ERROR: Input CSV must have a 'Seq' column.")
        sys.exit(1)

    rs = args.random_state if args.random_state != 0 else None
    df_out, _ = clustering_analysis(
        df,
        kmer=args.kmer,
        min_cluster_size=args.min_cluster_size,
        out_dir=args.out_dir,
        family_name=args.family,
        debug=args.debug,
        pca_dims=args.pca_dims,
        n_epochs=args.n_epochs,
        random_state=rs,
        compute_tsne=not args.skip_tsne,
    )
    out_path = args.output or args.input
    df_out.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
