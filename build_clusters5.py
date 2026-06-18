#!/usr/bin/env python3
"""Assign all IAP copies to 5 UMAP/HDBSCAN clusters (te_clustering.py logic).

Reproduces te_clustering.py's embedding (k=6 -> SVD50 -> UMAP n_neighbors=30,
min_dist=0.0), runs HDBSCAN (min_cluster_size=800, min_samples=7) which yields
exactly 5 dense clusters, then folds the HDBSCAN noise points into their nearest
cluster centroid so EVERY copy gets one of 5 labels (0..4, largest first).

Writes:
  clusters5.csv          TE_ID, Seq, Cluster   (input for the MAFFT distance step)
  .umap_emb_tc.npy       2-D embedding, row-aligned to clusters5.csv
"""
import numpy as np, pandas as pd
from pathlib import Path

KMER, SVD_DIMS, SEED = 6, 50, 42
N_NEIGHBORS, MIN_DIST, N_EPOCHS = 30, 0.0, 200
MCS, MIN_SAMPLES = 800, 7
EMB_CACHE = Path(".umap_emb_tc.npy")


def main():
    d = pd.read_csv("consensus_distance_per_copy.csv")
    a = pd.read_csv("antisense_per_copy.csv", usecols=["TE_ID", "Seq"])
    df = d[["TE_ID"]].merge(a, on="TE_ID", how="inner")
    df = df[df.Seq.notna() & (df.Seq.astype(str).str.len() > 0)].reset_index(drop=True)
    seqs = df.Seq.astype(str).str.upper().tolist()
    print(f"n = {len(seqs)} copies")

    if EMB_CACHE.exists() and np.load(EMB_CACHE).shape[0] == len(df):
        emb = np.load(EMB_CACHE)
        print(f"Loaded cached embedding {EMB_CACHE}")
    else:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import TruncatedSVD
        import umap
        X = CountVectorizer(analyzer="char", ngram_range=(KMER, KMER),
                            dtype=np.float32).fit_transform(seqs)
        Xs = TruncatedSVD(n_components=SVD_DIMS, random_state=SEED).fit_transform(X)
        emb = umap.UMAP(n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST, n_components=2,
                        metric="euclidean", n_epochs=N_EPOCHS,
                        random_state=SEED).fit_transform(Xs)
        emb = np.asarray(emb, dtype=float)
        np.save(EMB_CACHE, emb)
        print(f"Computed + cached embedding -> {EMB_CACHE}")

    from sklearn.cluster import HDBSCAN
    raw = HDBSCAN(min_cluster_size=MCS, min_samples=MIN_SAMPLES,
                  copy=True).fit_predict(emb)
    ids = sorted(set(raw[raw >= 0]))
    print(f"HDBSCAN: {len(ids)} dense clusters, "
          f"{(raw==-1).sum()} noise ({100*(raw==-1).mean():.0f}%)")
    assert len(ids) == 5, f"expected 5 clusters, got {len(ids)}"

    # renumber by size (0 = largest), fold noise into nearest centroid
    centroids = {c: emb[raw == c].mean(axis=0) for c in ids}
    order = sorted(ids, key=lambda c: -(raw == c).sum())
    remap = {old: new for new, old in enumerate(order)}
    lab = np.array([remap[v] if v >= 0 else -1 for v in raw])
    noise = lab == -1
    if noise.any():
        cmat = np.vstack([centroids[order[i]] for i in range(5)])  # new-index order
        dists = np.linalg.norm(emb[noise][:, None, :] - cmat[None, :, :], axis=2)
        lab[noise] = dists.argmin(axis=1)
    df["Cluster"] = lab
    print("final cluster sizes:", df.Cluster.value_counts().sort_index().to_dict())

    df[["TE_ID", "Seq", "Cluster"]].to_csv("clusters5.csv", index=False)
    print(f"Wrote clusters5.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
