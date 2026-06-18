#!/usr/bin/env python3
"""
plot_umap_consensus.py --- UMAP of TE copies, colored by consensus distance.

Builds ONE UMAP embedding of every copy (k-mer -> TruncatedSVD -> UMAP, the same
recipe as te_clustering.py) and renders a two-panel figure:

  panel A : points colored by Kimura-2P distance to the GLOBAL (family-wide) consensus
  panel B : points colored by Kimura-2P distance to each copy's OWN CLUSTER consensus

Both panels share the same embedding and the same cluster overlay: each cluster
(subfamily) is drawn as a translucent filled convex hull in a distinct color, so
the cluster structure is visible behind the distance coloring. The legend lists
every cluster with mu = its mean distance (global mu in panel A, cluster mu in
panel B); the title carries the overall mu.

Sequences come from a per-copy CSV with a Seq column; distances + cluster labels
come from consensus_distance_per_copy.csv (joined on TE_ID).

Usage:
    python plot_umap_consensus.py \
        --dist consensus_distance_per_copy.csv \
        --seqs antisense_per_copy.csv \
        --family IAP \
        --out fig_umap_consensus.pdf
"""

import argparse
import datetime
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _pp(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _get_cmap(name, n):
    try:
        return matplotlib.colormaps[name].resampled(n)
    except (AttributeError, KeyError):
        import matplotlib.cm as cm
        return cm.get_cmap(name, n)


def embed_umap(seqs, kmer=6, svd_dims=50, n_neighbors=15, min_dist=0.1, seed=42):
    """k-mer counts -> TruncatedSVD -> 2-D UMAP. Returns (N,2) float array."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import TruncatedSVD
    import umap

    n = len(seqs)
    _pp(f"  Step 1/3: {kmer}-mer counts for {n:,} sequences...")
    vec = CountVectorizer(analyzer="char", ngram_range=(kmer, kmer), dtype=np.float32)
    X = vec.fit_transform(s.upper() for s in seqs)
    _pp(f"    -> {X.shape[1]:,} unique {kmer}-mers")

    dims = int(min(svd_dims, X.shape[1] - 1, n - 1))
    _pp(f"  Step 2/3: TruncatedSVD -> {dims} dims...")
    Xs = TruncatedSVD(n_components=dims, random_state=seed).fit_transform(X)

    _pp(f"  Step 3/3: UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
    emb = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                    n_components=2, metric="euclidean",
                    random_state=seed).fit_transform(Xs)
    _pp("    -> UMAP done")
    return np.asarray(emb, dtype=float)


def hdbscan_labels(emb, min_cluster_size=150, min_samples=None):
    """HDBSCAN on the 2-D UMAP embedding. Returns integer labels (-1 = noise),
    renumbered 0,1,2,... in descending cluster-size order (noise stays -1)."""
    from sklearn.cluster import HDBSCAN
    _pp(f"  HDBSCAN on UMAP embedding (min_cluster_size={min_cluster_size}, "
        f"min_samples={min_samples})...")
    raw = HDBSCAN(min_cluster_size=min_cluster_size,
                  min_samples=min_samples, copy=True).fit_predict(emb)
    # renumber real clusters by size so '0' is the largest
    order = (pd.Series(raw[raw >= 0]).value_counts().index.tolist())
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap.get(v, -1) for v in raw])
    n_clust = len(order)
    n_noise = int((labels == -1).sum())
    _pp(f"    -> {n_clust} clusters, {n_noise:,} noise points")
    return labels


def draw_cluster_shading(ax, emb, clusters, cluster_order, cmap):
    """Soft KDE density blob per cluster, behind the scatter.

    A Gaussian KDE hugs each cluster's dense core (unlike a convex hull, which a
    few outlier copies stretch across the whole plot). One filled contour per
    cluster at a fraction of its peak density, plus a thin edge line.
    """
    from scipy.stats import gaussian_kde
    x0, x1 = emb[:, 0].min(), emb[:, 0].max()
    y0, y1 = emb[:, 1].min(), emb[:, 1].max()
    for i, c in enumerate(cluster_order):
        pts = emb[clusters.values == c]
        color = cmap(i)
        if len(pts) < 5:
            continue
        # local grid around this cluster keeps the KDE cheap and tight
        px0, px1 = pts[:, 0].min(), pts[:, 0].max()
        py0, py1 = pts[:, 1].min(), pts[:, 1].max()
        mx = 0.15 * (px1 - px0 + 1e-6); my = 0.15 * (py1 - py0 + 1e-6)
        try:
            kde = gaussian_kde(pts.T)
        except Exception:
            continue
        gx, gy = np.mgrid[px0 - mx:px1 + mx:90j, py0 - my:py1 + my:90j]
        z = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        level = z.max() * 0.30
        if not np.isfinite(level) or level <= 0:
            continue
        ax.contourf(gx, gy, z, levels=[level, z.max()],
                    colors=[color], alpha=0.20, zorder=1)
        ax.contour(gx, gy, z, levels=[level], colors=[color],
                   linewidths=0.8, alpha=0.6, zorder=1)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)


def make_panel(ax, emb, dist, clusters, cluster_order, cmap, title, cbar_label,
               vmax):
    # cluster shading behind everything
    draw_cluster_shading(ax, emb, clusters, cluster_order, cmap)

    # uncolored (no distance) copies as faint grey so the manifold shape shows
    nan_mask = ~np.isfinite(dist)
    if nan_mask.any():
        ax.scatter(emb[nan_mask, 0], emb[nan_mask, 1], s=3, c="#d9d9d9",
                   alpha=0.25, linewidths=0, zorder=2, rasterized=True,
                   label="_nolegend_")

    ok = np.isfinite(dist)
    sc = ax.scatter(emb[ok, 0], emb[ok, 1], s=10, c=dist[ok] * 100.0,
                    cmap="magma", vmin=0, vmax=vmax * 100.0, alpha=0.9,
                    linewidths=0, zorder=3, rasterized=True)
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(cbar_label, fontsize=10)

    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_xlabel("UMAP-1", fontsize=10)
    ax.set_ylabel("UMAP-2", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dist", default="consensus_distance_per_copy.csv",
                   help="CSV with TE_ID, cluster, dist_cluster, dist_global.")
    p.add_argument("--seqs", default="antisense_per_copy.csv",
                   help="Per-copy CSV with TE_ID and Seq columns.")
    p.add_argument("--family", default="IAP")
    p.add_argument("--out", default="fig_umap_consensus.pdf")
    p.add_argument("--min-cluster-size", type=int, default=150,
                   help="HDBSCAN min_cluster_size (default 150 -> ~20 clusters).")
    p.add_argument("--min-samples", type=int, default=None,
                   help="HDBSCAN min_samples (default None; lower -> less noise).")
    p.add_argument("--kmer", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    _pp("=" * 60)
    _pp(f"UMAP consensus-distance figure - {args.family}")
    _pp("=" * 60)

    dist = pd.read_csv(args.dist)
    for col in ("TE_ID", "cluster", "dist_cluster", "dist_global"):
        if col not in dist.columns:
            sys.exit(f"ERROR: {args.dist} missing column '{col}'.")
    seqs = pd.read_csv(args.seqs, usecols=lambda c: c in ("TE_ID", "Seq"))
    if "Seq" not in seqs.columns:
        sys.exit(f"ERROR: {args.seqs} has no 'Seq' column.")

    df = dist.merge(seqs.dropna(subset=["Seq"]), on="TE_ID", how="inner")
    df = df[df["Seq"].astype(str).str.len() > 0].reset_index(drop=True)
    _pp(f"  {len(df):,} copies loaded.")

    # cache the embedding keyed on the copy set so re-renders are instant
    import hashlib
    key = hashlib.md5(("|".join(df["TE_ID"]) + f"|k{args.kmer}|s{args.seed}")
                      .encode()).hexdigest()[:12]
    cache = f".umap_emb_{key}.npy"
    try:
        emb = np.load(cache)
        _pp(f"  Loaded cached embedding {cache}")
    except Exception:
        emb = embed_umap(df["Seq"].astype(str).tolist(), kmer=args.kmer, seed=args.seed)
        np.save(cache, emb)

    # clusters come from HDBSCAN on the UMAP embedding (0,1,2,...; -1 = noise),
    # NOT the RepeatMasker subfamily. Noise points are plotted but not shaded.
    labels = hdbscan_labels(emb, min_cluster_size=args.min_cluster_size,
                            min_samples=args.min_samples)
    df["cluster"] = labels.astype(str)

    # cluster order + per-cluster mean distances for the legend.
    # HDBSCAN noise (-1) is plotted but not shaded/legended.
    g = df[df["cluster"] != "-1"].groupby("cluster")
    stats = pd.DataFrame({
        "n": g.size(),
        "mu_cluster": g["dist_cluster"].mean(),
        "mu_global": g["dist_global"].mean(),
    })
    cluster_order = sorted(stats.index, key=lambda c: int(c))
    cmap = _get_cmap("tab20", max(2, len(cluster_order)))

    # shared colour scale: 99th pct of all finite distances, so outliers don't
    # wash out the gradient. Same vmax in both panels for comparability.
    alld = np.concatenate([df["dist_cluster"].to_numpy(),
                           df["dist_global"].to_numpy()])
    alld = alld[np.isfinite(alld)]
    vmax = float(np.percentile(alld, 99)) if alld.size else 0.75

    mu_g = float(np.nanmean(df["dist_global"])) * 100.0
    mu_c = float(np.nanmean(df["dist_cluster"])) * 100.0
    n_g = int(np.isfinite(df["dist_global"]).sum())
    n_c = int(np.isfinite(df["dist_cluster"]).sum())

    fig, axes = plt.subplots(1, 2, figsize=(18, 8.5))
    make_panel(axes[0], emb, df["dist_global"].to_numpy(), df["cluster"],
               cluster_order, cmap,
               f"{args.family} UMAP - distance to global consensus\n"
               f"overall mu = {mu_g:.1f}%  (N = {n_g:,} good copies)",
               "Kimura-2P to global consensus (%)", vmax)
    make_panel(axes[1], emb, df["dist_cluster"].to_numpy(), df["cluster"],
               cluster_order, cmap,
               f"{args.family} UMAP - distance to cluster consensus\n"
               f"overall mu = {mu_c:.1f}%  (N = {n_c:,} good copies)",
               "Kimura-2P to cluster consensus (%)", vmax)

    # shared cluster legend: colour swatch + per-cluster mu (global / cluster)
    def _fmt(x):
        return f"{x*100:.0f}%" if np.isfinite(x) else "--"
    handles = [
        Patch(facecolor=cmap(i), edgecolor=cmap(i), alpha=0.6,
              label=f"cluster {c}  (n={int(stats.loc[c,'n'])}, "
                    f"mu_g={_fmt(stats.loc[c,'mu_global'])}, "
                    f"mu_c={_fmt(stats.loc[c,'mu_cluster'])})")
        for i, c in enumerate(cluster_order)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5,
               frameon=False, title="HDBSCAN/UMAP cluster — mu = mean Kimura "
               "distance to global (mu_g) / cluster (mu_c) consensus",
               title_fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"{args.family} transposable-element copies — UMAP of {args.kmer}-mer "
                 f"composition, colored by consensus divergence",
                 fontweight="bold", fontsize=14, y=0.99)
    plt.tight_layout(rect=[0, 0.12, 1, 0.97])
    plt.savefig(args.out, bbox_inches="tight", dpi=200)
    plt.close()
    _pp(f"  Saved {args.out}")
    _pp("DONE")


if __name__ == "__main__":
    main()
