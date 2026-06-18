#!/usr/bin/env python3
"""
plot_umap_5clust.py --- two-panel UMAP colored by consensus distance, 5 clusters.

Run AFTER the cluster job returns consensus_distance_5clust.csv. Joins those
distances to the cached UMAP embedding (.umap_emb_tc.npy, row-aligned to
clusters5.csv) by TE_ID, so the returned CSV can be in any order.

  left  panel : points colored by distance to the global consensus
  right panel : points colored by distance to each copy's cluster consensus
Both share the 5 UMAP/HDBSCAN cluster shadings; legend gives each cluster's mu.
"""
import argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _cmap(name, n):
    try:
        return matplotlib.colormaps[name].resampled(n)
    except (AttributeError, KeyError):
        import matplotlib.cm as cm
        return cm.get_cmap(name, n)


def draw_shading(ax, emb, clusters, order, cmap):
    from scipy.stats import gaussian_kde
    x0, x1 = emb[:, 0].min(), emb[:, 0].max()
    y0, y1 = emb[:, 1].min(), emb[:, 1].max()
    for i, c in enumerate(order):
        pts = emb[clusters.values == c]
        if len(pts) < 5:
            continue
        try:
            kde = gaussian_kde(pts.T)
        except Exception:
            continue
        mx = 0.12 * (pts[:, 0].ptp() + 1e-6); my = 0.12 * (pts[:, 1].ptp() + 1e-6)
        gx, gy = np.mgrid[pts[:, 0].min()-mx:pts[:, 0].max()+mx:90j,
                          pts[:, 1].min()-my:pts[:, 1].max()+my:90j]
        z = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        lvl = z.max() * 0.30
        if np.isfinite(lvl) and lvl > 0:
            ax.contourf(gx, gy, z, levels=[lvl, z.max()], colors=[cmap(i)],
                        alpha=0.20, zorder=1)
            ax.contour(gx, gy, z, levels=[lvl], colors=[cmap(i)],
                       linewidths=0.9, alpha=0.6, zorder=1)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)


def panel(ax, emb, dist, clusters, order, cmap, title, cbar, vmax):
    draw_shading(ax, emb, clusters, order, cmap)
    nan = ~np.isfinite(dist)
    if nan.any():
        ax.scatter(emb[nan, 0], emb[nan, 1], s=3, c="#d9d9d9", alpha=0.25,
                   linewidths=0, zorder=2, rasterized=True)
    ok = np.isfinite(dist)
    sc = ax.scatter(emb[ok, 0], emb[ok, 1], s=10, c=dist[ok]*100, cmap="magma",
                    vmin=0, vmax=vmax*100, alpha=0.9, linewidths=0, zorder=3,
                    rasterized=True)
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(cbar, fontsize=10)
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_xlabel("UMAP-1", fontsize=10); ax.set_ylabel("UMAP-2", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dist", default="consensus_distance_5clust.csv")
    p.add_argument("--clusters", default="clusters5.csv")
    p.add_argument("--emb", default=".umap_emb_tc.npy")
    p.add_argument("--family", default="IAP")
    p.add_argument("--out", default="fig_umap_5clust.png")
    a = p.parse_args()

    base = pd.read_csv(a.clusters)[["TE_ID", "Cluster"]]
    emb = np.load(a.emb)
    assert len(base) == len(emb), (len(base), len(emb))
    d = pd.read_csv(a.dist)
    # prefer cluster labels from clusters5.csv (the embedding's own labels)
    keep = [c for c in ("TE_ID", "dist_cluster", "dist_global") if c in d.columns]
    base = base.merge(d[keep], on="TE_ID", how="left")
    base["cluster"] = base["Cluster"].astype(str)

    order = sorted(base["cluster"].unique(), key=lambda c: int(c))
    cmap = _cmap("tab10", max(2, len(order)))
    g = base.groupby("cluster")
    stats = pd.DataFrame({"n": g.size(), "mu_cluster": g["dist_cluster"].mean(),
                          "mu_global": g["dist_global"].mean()})

    alld = np.concatenate([base.dist_cluster.to_numpy(), base.dist_global.to_numpy()])
    alld = alld[np.isfinite(alld)]
    vmax = float(np.percentile(alld, 99)) if alld.size else 0.75
    mu_g = np.nanmean(base.dist_global)*100; mu_c = np.nanmean(base.dist_cluster)*100
    n_g = int(np.isfinite(base.dist_global).sum()); n_c = int(np.isfinite(base.dist_cluster).sum())

    fig, ax = plt.subplots(1, 2, figsize=(18, 8.5))
    panel(ax[0], emb, base.dist_global.to_numpy(), base["cluster"], order, cmap,
          f"{a.family} UMAP - distance to global consensus\n"
          f"overall mu = {mu_g:.1f}%  (N = {n_g:,}/{len(base):,} copies)",
          "Kimura-2P to global consensus (%)", vmax)
    panel(ax[1], emb, base.dist_cluster.to_numpy(), base["cluster"], order, cmap,
          f"{a.family} UMAP - distance to cluster consensus\n"
          f"overall mu = {mu_c:.1f}%  (N = {n_c:,}/{len(base):,} copies)",
          "Kimura-2P to cluster consensus (%)", vmax)

    def f(x): return f"{x*100:.0f}%" if np.isfinite(x) else "--"
    handles = [Patch(facecolor=cmap(i), edgecolor=cmap(i), alpha=0.6,
                     label=f"cluster {c}  (n={int(stats.loc[c,'n'])}, "
                           f"mu_g={f(stats.loc[c,'mu_global'])}, "
                           f"mu_c={f(stats.loc[c,'mu_cluster'])})")
               for i, c in enumerate(order)]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9, frameon=False,
               title="UMAP/HDBSCAN cluster — mu = mean Kimura distance to "
               "global (mu_g) / cluster (mu_c) consensus", title_fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{a.family} transposable-element copies — UMAP of 6-mer composition, "
                 "colored by consensus divergence", fontweight="bold", fontsize=14, y=0.99)
    plt.tight_layout(rect=[0, 0.08, 1, 0.97])
    plt.savefig(a.out, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
