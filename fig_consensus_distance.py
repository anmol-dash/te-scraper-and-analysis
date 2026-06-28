"""
Two UMAP plots colored by edit distance to consensus:
  1. All loci colored by distance to global IAP consensus (from global MSA)
  2. All loci colored by distance to their cluster's consensus (from per-cluster MSA)
Distances computed on pre-aligned sequences (proper Hamming on MSA columns).
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from scipy.spatial import ConvexHull
from pathlib import Path

HERE = Path(__file__).parent
ALN  = HERE / "alignments"
CLUS = ALN  / "cluster_alignments"

# ── FASTA readers ─────────────────────────────────────────────────────────────
def read_fasta_seq(path):
    seq = []
    with open(path) as f:
        for line in f:
            if not line.startswith(">"):
                seq.append(line.strip())
    return "".join(seq).upper()

def read_fasta_dict(path):
    """Return {header: sequence} for every record."""
    records = {}
    current, buf = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if current is not None:
                    records[current] = "".join(buf).upper()
                current, buf = line[1:], []
            else:
                buf.append(line)
    if current is not None:
        records[current] = "".join(buf).upper()
    return records

# ── aligned Hamming distance (normalised by consensus non-gap length) ─────────
def hamming_vs_consensus(aln_seq, cons):
    mismatches = 0
    total = 0
    for c, s in zip(cons, aln_seq):
        if c == "-":
            continue
        total += 1
        if s != c:
            mismatches += 1
    return mismatches / total if total > 0 else 0.0

# ── k-mer Jaccard distance (fallback for unaligned sequences) ─────────────────
def kmer_jaccard_dist(seq, ref, k=8):
    a = set(seq[i:i+k] for i in range(len(seq) - k + 1))
    b = set(ref[i:i+k] for i in range(len(ref) - k + 1))
    union = len(a | b)
    return 1.0 - len(a & b) / union if union > 0 else 1.0

# ── load clustered data ───────────────────────────────────────────────────────
df = pd.read_csv(HERE / "iap_clustered.csv")
n  = len(df)
dist_global  = np.full(n, np.nan)
dist_cluster = np.full(n, np.nan)

# ── Plot 1: global alignment ──────────────────────────────────────────────────
print("Reading global aligned FASTA …")
global_aln  = read_fasta_dict(ALN / "iap_aligned.fa")
global_cons = read_fasta_seq(ALN / "iap_consensus.fa")

print(f"Computing global distances for {n:,} loci …")
for header, seq in global_aln.items():
    m = re.match(r"row(\d+)", header)
    if m:
        idx = int(m.group(1))
        dist_global[idx] = hamming_vs_consensus(seq, global_cons)

df["dist_global"] = dist_global

# ── Plot 2: per-cluster alignments ────────────────────────────────────────────
print("Computing per-cluster distances …")
for cid in sorted(df["Cluster"].unique()):
    tag  = str(cid) if cid >= 0 else "-1"
    aln_path = CLUS / f"cluster_{tag}_aligned.fa"
    cons = read_fasta_seq(CLUS / f"cluster_{tag}_consensus.fa")

    if aln_path.stat().st_size > 0:
        # aligned MSA available — use proper Hamming
        seqs = read_fasta_dict(aln_path)
        for header, seq in seqs.items():
            m = re.match(r"row(\d+)", header)
            if m:
                dist_cluster[int(m.group(1))] = hamming_vs_consensus(seq, cons)
    else:
        # alignment missing — fall back to k-mer Jaccard on raw seqs
        print(f"  C{cid}: no alignment, using k-mer Jaccard fallback")
        seqs = read_fasta_dict(CLUS / f"cluster_{tag}_seqs.fa")
        for header, seq in seqs.items():
            m = re.match(r"row(\d+)", header)
            if m:
                dist_cluster[int(m.group(1))] = kmer_jaccard_dist(seq, cons)

df["dist_cluster"] = dist_cluster
print("Done.")

# ── shared style ──────────────────────────────────────────────────────────────
CMAP    = "plasma_r"
ALPHA   = 0.4
S       = 8
FIGSIZE = (8, 6.5)

cluster_ids    = sorted(c for c in df["Cluster"].unique() if c >= 0)
cluster_labels = {c: (f"C{c}" if c >= 0 else "Noise") for c in cluster_ids}

# one distinct color per cluster (tab10)
_tab = plt.cm.tab10(np.linspace(0, 0.9, len(cluster_ids)))
cluster_color = {c: _tab[i] for i, c in enumerate(cluster_ids)}

def draw_hulls(ax, df):
    """Draw a semi-transparent convex hull for each cluster."""
    for cid in cluster_ids:
        sub = df[df["Cluster"] == cid][["umap_x", "umap_y"]].dropna().values
        if len(sub) < 3:
            continue
        try:
            hull = ConvexHull(sub)
        except Exception:
            continue
        verts = sub[hull.vertices]
        verts = np.vstack([verts, verts[0]])   # close the polygon
        col = cluster_color[cid]
        ax.fill(verts[:, 0], verts[:, 1],
                color=col, alpha=0.15, linewidth=0, zorder=0)
        ax.plot(verts[:, 0], verts[:, 1],
                color=col, alpha=0.45, linewidth=0.8, zorder=1)

def hull_legend(ax, means):
    patches = [mpatches.Patch(color=cluster_color[c], alpha=0.6,
                              label=f"{cluster_labels[c]}  (μ={means[c]:.3f})")
               for c in cluster_ids]
    ax.legend(handles=patches, fontsize=7, framealpha=0.7,
              loc="lower left", markerscale=1.2)

def fix_cb_ticks(cb, vmin, vmax, n=6):
    ticks = np.linspace(vmin, vmax, n)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:.2f}" for t in ticks])

def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}.pdf/png")

# ── Figure 1: global consensus distance ──────────────────────────────────────
g = df["dist_global"].dropna()
g_vmin, g_vmax = np.percentile(g, 5), np.percentile(g, 95)
g_norm = mcolors.PowerNorm(gamma=0.35, vmin=g_vmin, vmax=g_vmax)
fig, ax = plt.subplots(figsize=FIGSIZE)
draw_hulls(ax, df)
sc = ax.scatter(
    df["umap_x"], df["umap_y"],
    c=df["dist_global"], cmap=CMAP, norm=g_norm,
    s=S, alpha=ALPHA, linewidths=0, rasterized=True, zorder=2
)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=g_norm, cmap=CMAP), ax=ax, fraction=0.03, pad=0.02)
cb.set_label("Fraction of positions differing from global consensus", fontsize=8)
fix_cb_ticks(cb, g_vmin, g_vmax)
g_means = {c: df[df["Cluster"]==c]["dist_global"].mean() for c in cluster_ids}
hull_legend(ax, g_means)
ax.set_title("IAP — Distance from Global Consensus")
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.set_xticks([]); ax.set_yticks([])
ax.annotate("μ = mean fraction of aligned positions differing from the global consensus",
            xy=(0.01, 0.99), xycoords="axes fraction", fontsize=6.5,
            va="top", color="gray")
plt.tight_layout()
save(fig, "fig_global_consensus_dist")

# ── Figure 2: per-cluster consensus distance ──────────────────────────────────
cl = dist_cluster[~np.isnan(dist_cluster)]
c_vmin, c_vmax = np.percentile(cl, 5), np.percentile(cl, 95)
fig, ax = plt.subplots(figsize=FIGSIZE)
norm = mcolors.PowerNorm(gamma=0.35, vmin=c_vmin, vmax=c_vmax)

draw_hulls(ax, df)
for cid in cluster_ids:
    sub = df[df["Cluster"] == cid]
    ax.scatter(
        sub["umap_x"], sub["umap_y"],
        c=sub["dist_cluster"], cmap=CMAP, norm=norm,
        s=S, alpha=ALPHA, linewidths=0, rasterized=True, zorder=2
    )

cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=ax,
                  fraction=0.03, pad=0.02)
cb.set_label("Fraction of positions differing from cluster consensus", fontsize=8)
fix_cb_ticks(cb, c_vmin, c_vmax)
c_means = {c: df[df["Cluster"]==c]["dist_cluster"].mean() for c in cluster_ids}
hull_legend(ax, c_means)
ax.set_title("IAP — Distance from Each Cluster's Consensus")
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.set_xticks([]); ax.set_yticks([])
ax.annotate("μ = mean fraction of aligned positions differing from each cluster's own consensus",
            xy=(0.01, 0.99), xycoords="axes fraction", fontsize=6.5,
            va="top", color="gray")
plt.tight_layout()
save(fig, "fig_cluster_consensus_dist")
