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
  - k=10 default: CountVectorizer char n-grams, ngram_range=(10, 10)
  - TruncatedSVD 50 dims before UMAP (was PCA 2 dims *after* UMAP ran on raw matrix)
  - Sparse matrix never densified until SVD output
  - n_jobs=-1 on UMAP and HDBSCAN
  - float32 throughout; n_epochs capped
  - Large-N (>20k): fit UMAP on random subset, transform the rest
  - HDBSCAN defaults: min_samples=5, min_cluster_size=100 (UMAP: n_neighbors=30, min_dist=0.0)

Standalone usage:
    python te_clustering.py --input clustered_data.csv --kmer 10
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

# Ensure sibling modules (te_notify, etc.) are importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))

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

def clustering_analysis(df, kmer=10, min_cluster_size=100, out_dir=None,
                        family_name="FAMILY", debug=False,
                        pca_dims=50, n_epochs=200, random_state=42,
                        compute_tsne=True, n_neighbors=30, min_dist=0.0,
                        min_samples=5, expr_cols=None):
    """Run k-mer + SVD + UMAP/PCA/tSNE + HDBSCAN clustering on df['Seq'].

    Args:
        df:               DataFrame with a 'Seq' column.
        kmer:             K-mer length (default: 10).
        min_cluster_size: HDBSCAN min_cluster_size (default: 100).
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
        expr_cols:        Explicit list of expression column names. Only when
                          this is non-empty is the expression-sized
                          visualisation produced. Pass None to fall back to
                          numeric-column autodetection (legacy callers only).

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

    # Guard: detect degenerate all-N input before spending time on k-mer encoding
    n_degenerate = sum(1 for s in seqs if not s or set(s.upper()) <= {"N"})
    if n_degenerate > 0:
        _pp(f"  WARNING: {n_degenerate}/{n} sequences are all-N or empty (sequence fetch likely failed)")
    if n_degenerate > n * 0.5:
        raise RuntimeError(
            f"{n_degenerate}/{n} sequences are all-N or empty.\n"
            "  Sequence fetching failed upstream — check te_prep.py output for UCSC fetch warnings,\n"
            "  or re-run te_prep with --genome <local.fa> to use a local FASTA instead."
        )

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
    nn = max(2, min(n_neighbors, n - 1))
    _pp(f"  Step 3/5: UMAP (n_neighbors={nn}, min_dist={min_dist}, n_epochs={n_epochs}, n_jobs=-1)…")
    try:
        # Note: umap-learn overrides n_jobs=1 when random_state is set.
        # Pass random_state=None for true multi-core UMAP (non-reproducible).
        umap_model = _umap.UMAP(
            n_neighbors=nn,
            min_dist=min_dist,
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
    min_s = min_samples
    _pp(f"  Step 5/5: HDBSCAN (min_cluster_size={mcs}, min_samples={min_s})…")

    def _cluster(emb, name):
        # Run HDBSCAN on a 2-D embedding and return its integer cluster labels
        # (-1 = noise), using the shared min_samples / min_cluster_size settings.
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
            tsne_emb, tsne_labels, out_dir, family_name, kmer,
            expr_cols=expr_cols,
        )
        coords_path = Path(out_dir) / "clustering_coordinates.csv"
        coord_cols = [c for c in ["chr", "start", "stop", "strand", "Cluster",
                                   "pca_x", "pca_y", "umap_x", "umap_y",
                                   "tsne_x", "tsne_y"] if c in df.columns]
        df[coord_cols].to_csv(coords_path, index=False)
        _pp(f"    Coordinates → {coords_path}")

        _write_cluster_summary(df, Path(out_dir))

    return df, umap_labels


def _write_cluster_summary(df, out_dir: Path) -> None:
    """Write cluster_summary.csv with per-cluster size and strand breakdown."""
    import pandas as _pd

    strand_col = next((c for c in ("strand", "Strand") if c in df.columns), None)

    # Inputs without an explicit strand column still carry it in the te_prep
    # TE_ID ("chr|start|stop|name|div|-"). Without this fallback every cluster
    # reported n_plus=n_minus=0 / 0.00% for such inputs, which reads as "no
    # strand data" rather than "not parsed".
    strands = None
    if strand_col is not None:
        strands = df[strand_col].astype(str).str.strip()
    else:
        for col in ("TE_ID", "TE_name", "name", "Name", "id", "ID"):
            if col not in df.columns:
                continue
            extracted = (df[col].astype(str).str.strip()
                         .str.extract(r"(?:^|[|:,\s])([+-])$", expand=False))
            if extracted.notna().any():
                strands = extracted
                _pp(f"    Strand recovered from '{col}' "
                    f"({int(extracted.notna().sum())}/{len(df)} rows)")
                break

    rows = []
    for cluster_id in sorted(df["Cluster"].unique()):
        sub = df[df["Cluster"] == cluster_id]
        n = len(sub)
        if strands is not None:
            s = strands.loc[sub.index]
            n_plus  = int((s == "+").sum())
            n_minus = int((s == "-").sum())
        else:
            n_plus = n_minus = 0
        rows.append({
            "cluster":   cluster_id,
            "n_total":   n,
            "n_plus":    n_plus,
            "n_minus":   n_minus,
            "pct_plus":  round(100.0 * n_plus  / n, 2) if n else 0.0,
            "pct_minus": round(100.0 * n_minus / n, 2) if n else 0.0,
        })

    summary_path = out_dir / "cluster_summary.csv"
    _pd.DataFrame(rows).to_csv(summary_path, index=False)
    _pp(f"    Cluster summary → {summary_path}")


def _save_clustering_viz(df, pca_emb, pca_lbl, umap_emb, umap_lbl,
                          tsne_emb, tsne_lbl, out_dir, family_name, kmer,
                          expr_cols=None):
    """Save interactive Plotly clustering visualisation (+ expression-sized variant if available)."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Expression columns must be the ones the user actually designated. The
        # old behaviour — treat every numeric column as expression — sized dots
        # by RepeatMasker fields (swScore, milliDiv, genoLeft, ...) on runs that
        # had no expression data at all, producing a meaningless
        # "clustering_visualization_expr.html".
        if expr_cols is None:
            _excl = {"chr", "start", "stop", "Unnamed: 0", "Cluster", "_total_expr",
                     "pca_x", "pca_y", "umap_x", "umap_y", "tsne_x", "tsne_y"}
            expr_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                         if c not in _excl]
            if expr_cols:
                _pp(f"    (autodetected expression columns: {expr_cols})")
        expr_cols = [c for c in (expr_cols or []) if c in df.columns]
        expr_vals = None
        if not expr_cols:
            _pp("    No expression columns — skipping expression-sized visualisation")
        if expr_cols:
            raw = df[expr_cols].fillna(0).sum(axis=1).values.astype(float)
            mn, mx = raw.min(), raw.max()
            if mx > mn:
                expr_vals = 4 + 16 * (raw - mn) / (mx - mn)  # scale 4–20 px
            else:
                expr_vals = np.full(len(raw), 8.0)

        def _build_fig(marker_size):
            # Assemble the 3-panel PCA / UMAP / t-SNE scatter figure at a given
            # marker size (called twice to render both a dense and a sparse view).
            f = make_subplots(
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
                    strand = row.get('strand', '?')
                    htexts.append(
                        f"<b>Row {i}</b><br>{coord}<br>"
                        f"Strand: {strand}<br>Cluster: {lbl}"
                    )
                f.add_trace(
                    go.Scatter(
                        x=emb[:, 0], y=emb[:, 1], mode="markers",
                        marker=dict(size=marker_size, color=[cmap[l] for l in labels],
                                    colorscale="Viridis"),
                        text=htexts, hovertemplate="%{text}<extra></extra>",
                    ),
                    row=1, col=col,
                )

            _add(emb=pca_emb,  col=1, labels=pca_lbl)
            _add(emb=umap_emb, col=2, labels=umap_lbl)
            _add(emb=tsne_emb, col=3, labels=tsne_lbl)
            return f

        _layout = dict(
            autosize=True, height=500, showlegend=False,
            title=f"{family_name} Clustering (k={kmer})",
        )
        fig = _build_fig(marker_size=5)
        fig.update_layout(**_layout)
        out_path = out_dir / "clustering_visualization.html"
        fig.write_html(out_path, full_html=True, include_plotlyjs=True)
        _pp(f"    Clustering visualisation → {out_path}")

        if expr_vals is not None:
            fig_expr = _build_fig(marker_size=expr_vals.tolist())
            fig_expr.update_layout(
                **{**_layout, "title": f"{family_name} Clustering — dot size = expression (k={kmer})"}
            )
            expr_path = out_dir / "clustering_visualization_expr.html"
            fig_expr.write_html(expr_path, full_html=True, include_plotlyjs=True)
            _pp(f"    Expression-sized visualisation → {expr_path}")

    except Exception as e:
        _pp(f"    WARNING: could not save clustering viz: {e}")


# ── CLI entry point ─────────────────────────────────────────────────────────

def _parse_args():
    """Parse the standalone CLI for re-running clustering on a *_clustered.csv."""
    p = argparse.ArgumentParser(
        description="Re-run clustering on an existing *_clustered.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python te_clustering.py --input data.csv --kmer 10\n"
            "  python te_clustering.py --input data.csv --kmer 10 --pca-dims 100 --n-epochs 150\n"
        ),
    )
    p.add_argument("--input",    required=True, help="Input CSV (must have Seq column)")
    p.add_argument("--output",   default=None,  help="Output CSV (default: overwrite input)")
    p.add_argument("--kmer",     type=int, default=10,
                   help="K-mer length (default 10)")
    p.add_argument("--pca-dims", type=int, default=50,
                   help="SVD components fed into UMAP / t-SNE (default 50)")
    p.add_argument("--n-epochs", type=int, default=200,
                   help="UMAP optimisation epochs (default 200)")
    p.add_argument("--min-cluster-size", type=int, default=100)
    p.add_argument("--n-neighbors",    type=int,   default=30,  help="UMAP n_neighbors (default 30)")
    p.add_argument("--min-dist",       type=float, default=0.0, help="UMAP min_dist (default 0.0)")
    p.add_argument("--min-samples",    type=int,   default=5,   help="HDBSCAN min_samples (default 5)")
    p.add_argument("--out-dir",  default=".", help="Directory for visualisation files")
    p.add_argument("--family",   default="FAMILY", help="Family name for plot titles")
    p.add_argument("--random-state", type=int, default=42,
                   help="Random seed (default 42); pass 0 to disable for full parallelism")
    p.add_argument("--debug",    action="store_true")
    p.add_argument("--skip-tsne", action="store_true",
                   help="Skip t-SNE for faster UMAP-only reclustering")
    p.add_argument("--notify-email", default="", metavar="EMAIL",
                   help="Send a completion email to this address (requires Gmail App Password setup).")
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
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        min_samples=args.min_samples,
    )
    out_path = args.output or args.input
    df_out.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    if args.notify_email:
        from te_notify import send_completion_email
        send_completion_email(args.notify_email, "te_clustering", args.out_dir)
