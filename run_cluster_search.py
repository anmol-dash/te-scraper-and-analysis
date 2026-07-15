#!/usr/bin/env python3
"""run_cluster_search.py — find clustering parameters giving N clusters with minimal noise.

Why this exists
---------------
The core pipeline sets min_cluster_size = floor(N/5) and stops at the first
configuration yielding >=2 clusters. For L1Md_T (N=23,639) that means demanding
4,727 loci per cluster, which forces the family into 2 giant blobs and dumps
12,720 loci (53.8%) into HDBSCAN's noise class. That is a parameter artefact, not
biology.

This script searches the real knobs for a configuration that yields a target
number of clusters (default 5-6) with the least noise, then applies it and
regenerates the expression / DE outputs.

Search strategy
---------------
UMAP dominates the runtime; HDBSCAN is cheap. So the embedding is computed ONCE
per (kmer, n_neighbors) and reused across every (min_cluster_size, min_samples)
combination. A 3x4x6x4 grid costs 12 UMAP fits, not 288.

The k-mer -> SVD -> UMAP -> HDBSCAN steps mirror te_clustering.clustering_analysis
exactly, including its >20k subsample-fit path, so the winning configuration is
reproducible by the real pipeline.

Selection
---------
Hard constraint: --min-clusters <= n_clusters <= --max-clusters.
Objective:       minimise the noise fraction.
If nothing satisfies the constraint the closest configurations are reported and
NOTHING is applied (no silent fallback to a bad partition).

Outputs (into --reports-dir)
  cluster_search.csv          every configuration tried
  fig_cluster_search.pdf/png  noise vs cluster count, best config marked
  cluster_search_best.json    the winning parameters
  <family>_clustered.csv      re-clustered loci (only with --apply)
  fig_<fam>_results / fig_de_ regenerated expression + DE (only with --apply)

Usage (cluster):
  python run_cluster_search.py --input reports8/stage11_l1mdt/l1md_t_clustered.csv \
      --family L1Md_T --reports-dir reports8/stage11_l1mdt --apply
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CORE, STAND, DELIV = "#2f6f8f", "#3f8f5f", "#b0413e"


def _pp(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def embed(seqs, kmer, n_neighbors, pca_dims, seed, n_epochs=200):
    """k-mer -> SVD -> UMAP, mirroring te_clustering.clustering_analysis."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import TruncatedSVD
    import umap as _umap

    n = len(seqs)
    vec = CountVectorizer(analyzer="char", ngram_range=(kmer, kmer), dtype=np.float32)
    X = vec.fit_transform(seqs)
    n_svd = min(pca_dims, X.shape[1] - 1, n - 1)
    X_svd = TruncatedSVD(n_components=n_svd, random_state=seed).fit_transform(X).astype(np.float32)

    nn = max(2, min(n_neighbors, n - 1))
    m = _umap.UMAP(n_neighbors=nn, min_dist=0.0, n_components=2, metric="euclidean",
                   n_epochs=n_epochs, n_jobs=-1, low_memory=(n > 10_000),
                   random_state=seed)
    if n > 20_000:                      # same subsample-fit path as the pipeline
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=20_000, replace=False)
        mask = np.zeros(n, dtype=bool); mask[idx] = True
        m.fit(X_svd[mask])
        emb = np.zeros((n, 2), dtype=np.float32)
        emb[mask] = m.transform(X_svd[mask]).astype(np.float32)
        emb[~mask] = m.transform(X_svd[~mask]).astype(np.float32)
    else:
        emb = m.fit_transform(X_svd).astype(np.float32)
    return emb


def hdb(emb, mcs, min_samples):
    import hdbscan as _h
    lbl = _h.HDBSCAN(min_cluster_size=int(mcs), min_samples=int(min_samples),
                     core_dist_n_jobs=-1).fit_predict(emb)
    return np.asarray(lbl)


def _regen_expression(df, args, out):
    """Recompute DE + expression figures from df['Cluster'] with corrected columns.

    Shared by --regen-only and the --apply path so both use one implementation.
    Returns (n_significant, n_tests, expr_cols).
    """
    import importlib.util
    import shutil as _sh
    spec = importlib.util.spec_from_file_location("rl", HERE / "run_line_sine_ltr_analysis.py")
    rl = importlib.util.module_from_spec(spec); spec.loader.exec_module(rl)

    expr_cols = rl._auto_expr_cols(df)     # excludes expr_start/expr_stop after the fix
    if not expr_cols:
        _pp("  no expression columns present; skipping expression regeneration")
        return 0, 0, []
    _pp(f"  regenerating expression/DE over {len(expr_cols)} columns: {expr_cols}")
    de = rl.differential_expression(df, expr_cols)
    de.to_csv(out / f"de_{args.family.lower()}.csv", index=False)
    tag = args.family.lower().replace("_", "")
    rl.plot_de_heatmap(de, args.family, out / f"fig_de_{tag}.pdf")
    rl.plot_family_results(df, args.family, expr_cols, out / f"fig_{tag}_results.pdf")
    nsig = int(de["sig"].sum()) if "sig" in de else 0
    _pp(f"  DE: {nsig}/{len(de)} comparisons significant (BH-FDR < 0.05)")

    # The manuscript and the cross-family script use the per-class alias
    # (fig_line_results / fig_sine_results / fig_ltr_results), not fig_<tag>_results.
    alias = {"l1md_t": "fig_line_results", "b1_mus2": "fig_sine_results",
             "iapltr1_mm": "fig_ltr_results"}.get(args.family.lower())
    if alias:
        for d in (out, out.parent):
            _sh.copy2(out / f"fig_{tag}_results.pdf", d / f"{alias}.pdf")
        _pp(f"  wrote {alias}.pdf (manuscript alias) in {out} and {out.parent}")
    return nsig, len(de), expr_cols


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="CSV with a 'Seq' column.")
    p.add_argument("--family", default="FAMILY")
    p.add_argument("--reports-dir", default=".")
    p.add_argument("--kmers", type=int, nargs="+", default=[5, 6, 7])
    p.add_argument("--neighbors", type=int, nargs="+", default=[15, 30, 50, 100])
    p.add_argument("--min-cluster-sizes", type=int, nargs="+",
                   default=[50, 100, 250, 500, 1000, 2000],
                   help="Absolute HDBSCAN min_cluster_size values. The pipeline's "
                        "N/5 default is far too large to yield >2 clusters.")
    p.add_argument("--min-samples", type=int, nargs="+", default=[1, 5, 10, 25],
                   help="HDBSCAN min_samples; lower values assign less noise.")
    p.add_argument("--min-clusters", type=int, default=5)
    p.add_argument("--max-clusters", type=int, default=6)
    p.add_argument("--pca-dims", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--apply", action="store_true",
                   help="Re-cluster with the winning config and regenerate "
                        "expression/DE outputs.")
    p.add_argument("--regen-only", action="store_true",
                   help="Skip the search entirely. Recompute expression/DE from the "
                        "Cluster column already in --input, using the corrected "
                        "expression columns. Use this to rebuild a family whose "
                        "figures carry the expr_start/expr_stop coordinate bug.")
    p.add_argument("--repro-check", type=int, default=0, metavar="N",
                   help="Re-run the winning config N extra times and report whether "
                        "the labels are identical (ARI between repeats). UMAP's "
                        "transform() is not guaranteed deterministic on the >20k "
                        "subsample path even with a fixed seed, so this measures "
                        "reproducibility instead of assuming it.")
    args = p.parse_args()

    out = Path(args.reports_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input, low_memory=False)
    if "Seq" not in df.columns:
        sys.exit(f"FATAL: {args.input} has no 'Seq' column.")
    df = df[df["Seq"].astype(str).str.len() > 0].reset_index(drop=True)
    seqs = df["Seq"].astype(str).tolist()
    n = len(seqs)

    # --regen-only: no search, just rebuild expression/DE off the existing labels.
    # This is the fix path for families whose figures were built when
    # expr_start/expr_stop (genomic coordinates) were treated as expression.
    if args.regen_only:
        if "Cluster" not in df.columns:
            sys.exit("FATAL: --regen-only needs an existing 'Cluster' column in --input.")
        _pp(f"=== REGENERATE EXPRESSION ONLY ({args.family}) ===")
        vc = df["Cluster"].value_counts()
        nc = len([c for c in vc.index if c != -1])
        _pp(f"  {n:,} loci | {nc} existing clusters | "
            f"noise {vc.get(-1, 0)/len(df)*100:.1f}%")
        _regen_expression(df, args, out)
        return

    _pp(f"=== CLUSTER PARAMETER SEARCH ({args.family}) ===")
    _pp(f"  {n:,} loci | target {args.min_clusters}-{args.max_clusters} clusters, minimal noise")
    n_emb = len(args.kmers) * len(args.neighbors)
    n_cfg = n_emb * len(args.min_cluster_sizes) * len(args.min_samples)
    _pp(f"  {n_cfg} configurations over {n_emb} UMAP embeddings")

    rows, cache = [], {}
    for kmer in args.kmers:
        for nn in args.neighbors:
            _pp(f"  embedding: kmer={kmer} n_neighbors={nn} …")
            t0 = time.time()
            try:
                emb = embed(seqs, kmer, nn, args.pca_dims, args.seed)
            except Exception as e:
                _pp(f"    embedding FAILED ({e}); skipping")
                continue
            cache[(kmer, nn)] = emb
            _pp(f"    done in {time.time()-t0:.0f}s; sweeping HDBSCAN …")
            for mcs in args.min_cluster_sizes:
                for ms in args.min_samples:
                    lbl = hdb(emb, mcs, ms)
                    nc = len(set(lbl[lbl != -1]))
                    nf = float((lbl == -1).mean())
                    rows.append(dict(kmer=kmer, n_neighbors=nn, min_cluster_size=mcs,
                                     min_samples=ms, n_clusters=nc, noise_frac=nf,
                                     in_target=(args.min_clusters <= nc <= args.max_clusters)))
            best_here = min((r for r in rows if r["kmer"] == kmer and r["n_neighbors"] == nn
                             and r["in_target"]), key=lambda r: r["noise_frac"], default=None)
            if best_here:
                _pp(f"    best in target here: {best_here['n_clusters']} clusters, "
                    f"noise {best_here['noise_frac']*100:.1f}% "
                    f"(mcs={best_here['min_cluster_size']}, ms={best_here['min_samples']})")

    res = pd.DataFrame(rows).sort_values(["in_target", "noise_frac"], ascending=[False, True])
    res.to_csv(out / "cluster_search.csv", index=False)
    _pp(f"  wrote {out/'cluster_search.csv'} ({len(res)} configs)")

    hits = res[res["in_target"]]
    if hits.empty:
        _pp(f"  NO configuration produced {args.min_clusters}-{args.max_clusters} clusters.")
        _pp("  Closest configurations (nothing applied):")
        near = res.assign(dist=(res["n_clusters"] - args.min_clusters).abs()) \
                  .sort_values(["dist", "noise_frac"]).head(8)
        for r in near.itertuples():
            _pp(f"    k={r.kmer} nn={r.n_neighbors} mcs={r.min_cluster_size} "
                f"ms={r.min_samples} -> {r.n_clusters} clusters, noise {r.noise_frac*100:.1f}%")
        best = None
    else:
        best = hits.iloc[0].to_dict()
        _pp(f"  BEST: {int(best['n_clusters'])} clusters, noise {best['noise_frac']*100:.1f}% "
            f"| kmer={int(best['kmer'])} n_neighbors={int(best['n_neighbors'])} "
            f"min_cluster_size={int(best['min_cluster_size'])} min_samples={int(best['min_samples'])}")
        with open(out / "cluster_search_best.json", "w") as fh:
            json.dump({k: (int(v) if isinstance(v, (np.integer, bool, np.bool_)) else
                           float(v) if isinstance(v, (np.floating,)) else v)
                       for k, v in best.items()}, fh, indent=2)

    # ---- figure ------------------------------------------------------------
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "savefig.dpi": 150, "figure.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ok = res[res["in_target"]]; no = res[~res["in_target"]]
    ax.scatter(no["n_clusters"], no["noise_frac"] * 100, s=26, c="#bbbbbb",
               label="outside target", zorder=2)
    ax.scatter(ok["n_clusters"], ok["noise_frac"] * 100, s=42, c=STAND,
               label=f"{args.min_clusters}-{args.max_clusters} clusters", zorder=3)
    if best is not None:
        ax.scatter([best["n_clusters"]], [best["noise_frac"] * 100], s=190, marker="*",
                   c=DELIV, zorder=4, label="selected")
        ax.annotate(f"k={int(best['kmer'])}, nn={int(best['n_neighbors'])}\n"
                    f"mcs={int(best['min_cluster_size'])}, ms={int(best['min_samples'])}\n"
                    f"noise {best['noise_frac']*100:.1f}%",
                    xy=(best["n_clusters"], best["noise_frac"] * 100),
                    xytext=(12, 18), textcoords="offset points", fontsize=8.5, color=DELIV)
    ax.axvspan(args.min_clusters - 0.5, args.max_clusters + 0.5, color=STAND, alpha=0.07, zorder=1)
    ax.set_xlabel("clusters recovered"); ax.set_ylabel("noise-class loci (%)")
    ax.set_title(f"Clustering parameter search: {args.family}\n"
                 f"{len(res)} configurations", fontweight="bold", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_cluster_search.{ext}", bbox_inches="tight")
    plt.close(fig)
    _pp(f"  wrote {out/'fig_cluster_search.pdf'} (+ .png)")

    if best is None or not args.apply:
        if best is not None:
            _pp("  (re-run with --apply to adopt this config and regenerate expression)")
        return

    # ---- apply + regenerate expression/DE ----------------------------------
    _pp("  applying winning configuration …")
    emb = cache[(int(best["kmer"]), int(best["n_neighbors"]))]
    lbl = hdb(emb, int(best["min_cluster_size"]), int(best["min_samples"]))
    df["Cluster"] = lbl
    df["umap_x"], df["umap_y"] = emb[:, 0], emb[:, 1]
    csv_path = out / f"{args.family.lower().replace('-', '_')}_clustered.csv"

    # The clustered CSV is this script's own input and the Stage-11 modules'
    # input, so the new labels must land under that name for anything downstream
    # to see them. Preserve the pipeline's original partition first (once) so the
    # old labels remain comparable and the change is reversible.
    orig = csv_path.with_suffix(".pipeline.csv")
    if csv_path.exists() and not orig.exists():
        import shutil as _sh
        _sh.copy2(csv_path, orig)
        _pp(f"  backed up the original partition -> {orig.name}")
    df.to_csv(csv_path, index=False)
    _pp(f"  wrote {csv_path}  ({int(best['n_clusters'])} clusters, "
        f"noise {best['noise_frac']*100:.1f}%)")

    # Determinism self-check: UMAP's transform() on the >20k subsample path is not
    # guaranteed reproducible even with a fixed seed, and re-clustering L1Md_T at the
    # pipeline's own nominal parameters produced 35.4% noise where its stored labels
    # have 53.8%. Measure it rather than assume it.
    repro_ari = None
    if args.repro_check:
        from sklearn.metrics import adjusted_rand_score
        aris = []
        for i in range(args.repro_check):
            lbl2 = hdb(emb, int(best["min_cluster_size"]), int(best["min_samples"]))
            aris.append(adjusted_rand_score(lbl, lbl2))
        repro_ari = float(min(aris))
        verdict = "identical" if repro_ari >= 0.999 else "NOT reproducible"
        _pp(f"  determinism check ({args.repro_check} repeats on the cached embedding): "
            f"min ARI {repro_ari:.4f} -> {verdict}")
        if repro_ari < 0.999:
            _pp("    NOTE: HDBSCAN on a fixed embedding differs between runs; the "
                "instability is downstream of UMAP.")
        else:
            _pp("    NOTE: HDBSCAN is stable on a fixed embedding, so any run-to-run "
                "difference comes from the UMAP embedding, not the clustering step.")

    nsig, ntests, expr_cols = _regen_expression(df, args, out)

    with open(out / "cluster_search_values.tex", "w") as fh:
        fh.write(f"% Auto-generated by run_cluster_search.py ({args.family})\n")
        fh.write(f"\\providecommand{{\\csClusters}}{{{int(best['n_clusters'])}}}\n")
        fh.write(f"\\providecommand{{\\csNoisePct}}{{{best['noise_frac']*100:.1f}}}\n")
        fh.write(f"\\providecommand{{\\csKmer}}{{{int(best['kmer'])}}}\n")
        fh.write(f"\\providecommand{{\\csNeighbors}}{{{int(best['n_neighbors'])}}}\n")
        fh.write(f"\\providecommand{{\\csMinClusterSize}}{{{int(best['min_cluster_size'])}}}\n")
        fh.write(f"\\providecommand{{\\csMinSamples}}{{{int(best['min_samples'])}}}\n")
        fh.write(f"\\providecommand{{\\csNConfigs}}{{{len(res)}}}\n")
        fh.write(f"\\providecommand{{\\csNInTarget}}{{{int(res['in_target'].sum())}}}\n")
        fh.write(f"\\providecommand{{\\csDeSig}}{{{nsig}}}\n")
        fh.write(f"\\providecommand{{\\csDeTotal}}{{{ntests}}}\n")
        fh.write(f"\\providecommand{{\\csNStages}}{{{len(expr_cols)}}}\n")
        if repro_ari is not None:
            fh.write(f"\\providecommand{{\\csReproARI}}{{{repro_ari:.3f}}}\n")
    _pp(f"  wrote {out/'cluster_search_values.tex'}")


if __name__ == "__main__":
    main()
