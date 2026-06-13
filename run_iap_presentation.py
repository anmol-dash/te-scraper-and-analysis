#!/usr/bin/env python3
"""
run_iap_presentation.py — end-to-end "reagent-design presentation" driver for the
IAP ultra-combo locus/expression table.

Runs DIRECTLY ON THE CLUSTER. Given `iapultracombo_tot_counts.csv` it:

  1. Loads the table, parses strand/coords out of the TE_ID field, and
     spot-checks that the recorded strand is consistent with the genomic
     coordinates on UCSC (compares the stored Seq to the UCSC plus-strand
     sequence / its reverse-complement for a random sample of loci).
  2. Clusters the per-copy sequences (k-mer -> TruncatedSVD(50) -> embeddings):
       - 2D UMAP (n_neighbors 30, falling back to 15) with an HDBSCAN
         min_cluster_size sweep N//5, N//6, N//7 ... chosen to land near 5
         clusters.
       - 2D t-SNE (same SVD input) + HDBSCAN, plotted.
       - 2D PCA (first two SVD components), cluster count not optimised.
       - 3D UMAP rendered twice: with axes and clean (no axes), coloured by the
         chosen 2D-UMAP cluster labels so all panels are comparable.
     All figures are publication-styled matplotlib PDFs/PNGs.
  3. Per-cluster MAFFT alignments + consensus sequences (+ CIAlign plots) via
     te_alignment.
  4. FRESH JASPAR motif enrichment per cluster via te_motif, using the SORTED
     bed file passed on the command line (default the mm10 sorted cache).
  5. gRNA design + coverage/off-target-style analysis via run_grna_analysis.
  6. The reagent-relevant Stage-11 modules (phylo age tree, subfamily structure,
     divergence/landscape, LTR structure, motif-gain, ORF/fold, cluster DE,
     gRNA off-target) via run_stage11_all.
  7. A presentation bundle: per-cluster value table, a cluster x developmental-
     stage expression heatmap, and a markdown explainer describing every figure
     and what it means for choosing cluster-specific reagents.

Independent heavy stages (alignment / motif / gRNA / Stage-11) run in parallel.

Example (cluster):
    python run_iap_presentation.py \
        --input iapultracombo_tot_counts.csv \
        --family IAP_Mm --assembly mm10 \
        --jaspar-bed /home/amodz/anmol/jaspar_cache/JASPAR2024_mm10.sorted.bed.gz \
        --out-dir iap_presentation
"""

import argparse
import datetime
import os
import subprocess
import sys
import textwrap
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# Headless plotting on the cluster.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_STAGES = ["twocell", "fourcell", "eightcell", "morulacell", "pronuc"]
_COMP = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N",
         "a": "t", "t": "a", "g": "c", "c": "g", "n": "n"}


# ── small helpers ────────────────────────────────────────────────────────────

def _pp(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def _rc(seq):
    return "".join(_COMP.get(b, "N") for b in reversed(str(seq)))


def _style():
    """A clean, presentation-friendly matplotlib theme."""
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })


def _palette(labels):
    """Stable colour map: noise (-1) grey, clusters from a qualitative map."""
    uniq = sorted(set(labels))
    base = plt.get_cmap("tab10").colors + plt.get_cmap("tab20").colors
    cmap = {}
    i = 0
    for lab in uniq:
        if lab == -1:
            cmap[lab] = (0.7, 0.7, 0.7)
        else:
            cmap[lab] = base[i % len(base)]
            i += 1
    return cmap


# ── data prep ────────────────────────────────────────────────────────────────

def prepare_dataframe(csv_path):
    """Load the ultra-combo CSV and derive chr/start/stop/strand/te_name.

    TE_ID looks like:  chr1|3031358|3031710|IAPLTR1a_Mm:ERVK:LTR|21|-
    The final pipe field is the strand.
    """
    df = pd.read_csv(csv_path)
    _pp(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    if "TE_ID" not in df.columns:
        raise SystemExit("ERROR: expected a 'TE_ID' column in the input CSV.")

    parts = df["TE_ID"].astype(str).str.split("|", expand=True)
    df["chr"] = parts[0]
    df["start"] = pd.to_numeric(parts[1], errors="coerce").astype("Int64")
    df["stop"] = pd.to_numeric(parts[2], errors="coerce").astype("Int64")
    df["te_name"] = parts[3]
    df["strand"] = parts[parts.shape[1] - 1].str.strip()

    # Cross-check the parsed coords against the explicit columns when present.
    for col, src in (("start", "Start"), ("stop", "Stop")):
        if src in df.columns:
            mism = (pd.to_numeric(df[src], errors="coerce") != df[col]).sum()
            if mism:
                _pp(f"  WARNING: {mism} rows where TE_ID {col} != column {src}; "
                    f"trusting explicit {src}")
                df[col] = pd.to_numeric(df[src], errors="coerce").astype("Int64")

    if "Chromosome" in df.columns:
        df["chr"] = df["Chromosome"].astype(str)

    bad_strand = (~df["strand"].isin(["+", "-"])).sum()
    if bad_strand:
        _pp(f"  WARNING: {bad_strand} rows with non +/- strand; coercing to '+'")
        df.loc[~df["strand"].isin(["+", "-"]), "strand"] = "+"

    present = [c for c in _STAGES if c in df.columns]
    for c in present:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["_total_expr"] = df[present].sum(axis=1) if present else 0.0

    n_plus = int((df["strand"] == "+").sum())
    n_minus = int((df["strand"] == "-").sum())
    _pp(f"  Strand breakdown: + {n_plus:,}  /  - {n_minus:,}")
    _pp(f"  Expression stages present: {present}")
    return df, present


# ── strand spot-check vs UCSC ────────────────────────────────────────────────

def _ucsc_seq(assembly, chrom, start0, end, timeout=20):
    """Fetch plus-strand sequence from the UCSC REST API (0-based start)."""
    import json
    import urllib.request
    url = (f"https://api.genome.ucsc.edu/getData/sequence"
           f"?genome={assembly};chrom={chrom};start={start0};end={end}")
    with urllib.request.urlopen(url) as r:
        data = json.loads(r.read().decode())
    return data.get("dna", "").upper()


def spot_check_strand(df, assembly, out_dir, n=8, seed=0):
    """Verify recorded strand against UCSC coords for a random sample.

    For a '+' locus the stored Seq should match the UCSC plus strand; for a '-'
    locus it should match the reverse-complement. We report a per-locus
    match/mismatch and an overall verdict. Network failures degrade gracefully
    (the check is reported as 'skipped', not fatal).
    """
    _pp("Spot-checking strandedness against UCSC ...")
    rng = np.random.default_rng(seed)
    rows = []
    # Sample some of each strand where possible.
    idx_minus = df.index[df["strand"] == "-"].tolist()
    idx_plus = df.index[df["strand"] == "+"].tolist()
    pick = []
    for pool in (idx_minus, idx_plus):
        if pool:
            k = min(len(pool), max(1, n // 2))
            pick += list(rng.choice(pool, size=k, replace=False))
    pick = pick[:n] if pick else list(df.index[:n])

    checked = 0
    for i in pick:
        r = df.loc[i]
        seq = str(r["Seq"]).upper()
        if not seq or set(seq) <= {"N"}:
            continue
        try:
            plus = _ucsc_seq(assembly, str(r["chr"]), int(r["start"]), int(r["stop"]))
        except Exception as e:
            rows.append({"row": int(i), "chr": r["chr"], "start": int(r["start"]),
                         "stop": int(r["stop"]), "strand": r["strand"],
                         "verdict": f"fetch_failed:{type(e).__name__}"})
            continue
        if not plus:
            rows.append({"row": int(i), "chr": r["chr"], "start": int(r["start"]),
                         "stop": int(r["stop"]), "strand": r["strand"],
                         "verdict": "empty_ucsc"})
            continue
        checked += 1
        expected = plus if r["strand"] == "+" else _rc(plus)
        # Compare on the overlapping prefix and allow tiny edge slop.
        L = min(len(expected), len(seq))
        ident = sum(a == b for a, b in zip(expected[:L], seq[:L])) / L if L else 0.0
        # Also test the *wrong* orientation so we can confirm the strand call.
        wrong = _rc(plus) if r["strand"] == "+" else plus
        Lw = min(len(wrong), len(seq))
        ident_wrong = sum(a == b for a, b in zip(wrong[:Lw], seq[:Lw])) / Lw if Lw else 0.0
        verdict = ("STRAND_OK" if ident >= 0.95 and ident > ident_wrong
                   else "MISMATCH" if ident_wrong >= 0.95
                   else f"low_ident({ident:.2f})")
        rows.append({"row": int(i), "chr": r["chr"], "start": int(r["start"]),
                     "stop": int(r["stop"]), "strand": r["strand"],
                     "ident_as_called": round(ident, 3),
                     "ident_if_flipped": round(ident_wrong, 3),
                     "verdict": verdict,
                     "ucsc_link": (f"https://genome.ucsc.edu/cgi-bin/hgTracks?db={assembly}"
                                   f"&position={r['chr']}:{int(r['start'])+1}-{int(r['stop'])}")})

    res = pd.DataFrame(rows)
    out = Path(out_dir) / "strand_spotcheck.csv"
    res.to_csv(out, index=False)
    ok = int((res.get("verdict") == "STRAND_OK").sum()) if not res.empty else 0
    bad = int((res.get("verdict") == "MISMATCH").sum()) if not res.empty else 0
    if checked == 0:
        _pp("  Strand spot-check: SKIPPED (UCSC unreachable from this node) "
            f"— links written to {out.name}")
    else:
        _pp(f"  Strand spot-check: {ok}/{checked} loci confirmed strand-correct, "
            f"{bad} flipped  → {out.name}")
    return res


# ── clustering + figures ─────────────────────────────────────────────────────

def _kmer_svd(seqs, kmer=6, pca_dims=50, seed=42):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import TruncatedSVD
    _pp(f"  k-mer encoding (k={kmer}) ...")
    vec = CountVectorizer(analyzer="char", ngram_range=(kmer, kmer), dtype=np.float32)
    X = vec.fit_transform([str(s) for s in seqs])
    n_svd = min(pca_dims, X.shape[1] - 1, X.shape[0] - 1)
    _pp(f"  TruncatedSVD({n_svd}) on {X.shape[0]}x{X.shape[1]} matrix ...")
    svd = TruncatedSVD(n_components=n_svd, random_state=seed)
    Xs = svd.fit_transform(X).astype(np.float32)
    _pp(f"  SVD variance explained: {svd.explained_variance_ratio_.sum()*100:.1f}%")
    return Xs


def _hdbscan_labels(emb, mcs, min_samples=7):
    import hdbscan
    lbl = hdbscan.HDBSCAN(min_cluster_size=int(mcs), min_samples=min_samples,
                          core_dist_n_jobs=-1).fit_predict(emb)
    nc = len(set(lbl)) - (1 if -1 in lbl else 0)
    noise = float(np.mean(lbl == -1))
    return lbl, nc, noise


def _umap_embed(Xs, n_neighbors, n_components, seed=42, n_epochs=300):
    import umap
    nn = max(2, min(n_neighbors, Xs.shape[0] - 1))
    model = umap.UMAP(n_neighbors=nn, min_dist=0.0, n_components=n_components,
                      metric="euclidean", n_epochs=n_epochs, n_jobs=-1,
                      random_state=seed, verbose=False)
    return model.fit_transform(Xs).astype(np.float32)


def _sweep_to_target(emb, n, target=5, divisors=(5, 6, 7, 8, 9, 10, 12, 15)):
    """HDBSCAN min_cluster_size = N//d for d in divisors; pick count closest to
    target (prefer >= target-1), tie-break on lower noise fraction."""
    trials = []
    for d in divisors:
        mcs = max(2, n // d)
        lbl, nc, noise = _hdbscan_labels(emb, mcs)
        trials.append({"divisor": d, "mcs": mcs, "n_clusters": nc,
                       "noise_frac": round(noise, 3), "labels": lbl})
        _pp(f"    mcs=N//{d}={mcs:>4d}  -> {nc} clusters, noise {noise:.1%}")

    def _key(t):
        return (abs(t["n_clusters"] - target), t["noise_frac"])
    best = min(trials, key=_key)
    _pp(f"  Chosen: mcs=N//{best['divisor']}={best['mcs']} "
        f"({best['n_clusters']} clusters, noise {best['noise_frac']:.1%})")
    return best, trials


def _scatter2d(ax, emb, labels, title, xlab, ylab):
    cmap = _palette(labels)
    for lab in sorted(set(labels)):
        m = labels == lab
        name = "noise" if lab == -1 else f"cluster {lab}"
        ax.scatter(emb[m, 0], emb[m, 1], s=14, alpha=0.8,
                   color=cmap[lab], label=name, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.legend(markerscale=1.5, fontsize=9, loc="best")


def run_clustering(df, present_stages, out_dir, family, kmer=6, seed=42,
                   target_clusters=5):
    """Full clustering: 2D UMAP (sweep), 2D t-SNE, 2D PCA, 3D UMAP (axes/no-axes)."""
    from sklearn.manifold import TSNE
    _style()
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    seqs = df["Seq"].astype(str).tolist()
    n = len(seqs)
    Xs = _kmer_svd(seqs, kmer=kmer, seed=seed)

    # PCA-style 2D = first two SVD components.
    pca2 = Xs[:, :2].copy()

    # 2D UMAP, try n_neighbors 30 then 15, optimise mcs toward ~5 clusters.
    chosen = None
    for nn in (30, 15):
        _pp(f"  2D UMAP (n_neighbors={nn}) ...")
        umap2 = _umap_embed(Xs, n_neighbors=nn, n_components=2, seed=seed)
        best, trials = _sweep_to_target(umap2, n, target=target_clusters)
        if chosen is None or abs(best["n_clusters"] - target_clusters) < \
                abs(chosen["best"]["n_clusters"] - target_clusters):
            chosen = {"nn": nn, "emb": umap2, "best": best, "trials": trials}
        if abs(best["n_clusters"] - target_clusters) <= 1:
            _pp(f"  n_neighbors={nn} reached ~{target_clusters} clusters; using it.")
            break

    umap2 = chosen["emb"]
    labels = np.asarray(chosen["best"]["labels"])
    nn_used = chosen["nn"]
    df = df.copy()
    df["Cluster"] = labels
    df["umap_x"], df["umap_y"] = umap2[:, 0], umap2[:, 1]
    df["pca_x"], df["pca_y"] = pca2[:, 0], pca2[:, 1]

    # 2D t-SNE.
    _pp("  2D t-SNE ...")
    perp = max(5, min(30, (n - 1) // 3))
    tsne2 = TSNE(n_components=2, perplexity=perp, init="pca",
                 random_state=seed).fit_transform(Xs).astype(np.float32)
    tsne_lbl, _, _ = _hdbscan_labels(tsne2, chosen["best"]["mcs"])
    df["tsne_x"], df["tsne_y"] = tsne2[:, 0], tsne2[:, 1]

    # 3D UMAP coloured by the chosen 2D labels.
    _pp("  3D UMAP ...")
    umap3 = _umap_embed(Xs, n_neighbors=nn_used, n_components=3, seed=seed)
    for k in range(3):
        df[f"umap3_{k}"] = umap3[:, k]

    # ── Figure 1: 2D UMAP ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    _scatter2d(ax, umap2, labels,
               f"{family} — 2D UMAP (n_neighbors={nn_used}, "
               f"min_cluster_size={chosen['best']['mcs']})",
               "UMAP-1", "UMAP-2")
    fig.savefig(fig_dir / "fig_umap_2d.pdf"); fig.savefig(fig_dir / "fig_umap_2d.png")
    plt.close(fig)

    # ── Figure 2: 2D t-SNE ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    _scatter2d(ax, tsne2, tsne_lbl, f"{family} — 2D t-SNE (perplexity={perp})",
               "t-SNE-1", "t-SNE-2")
    fig.savefig(fig_dir / "fig_tsne_2d.pdf"); fig.savefig(fig_dir / "fig_tsne_2d.png")
    plt.close(fig)

    # ── Figure 3: 2D PCA (coloured by UMAP clusters for comparability) ───────
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    _scatter2d(ax, pca2, labels, f"{family} — PCA (SVD 50d; first 2 components)",
               "PC-1", "PC-2")
    fig.savefig(fig_dir / "fig_pca_2d.pdf"); fig.savefig(fig_dir / "fig_pca_2d.png")
    plt.close(fig)

    # ── Figure 4/5: 3D UMAP with axes and without axes ──────────────────────
    cmap = _palette(labels)
    colors = [cmap[l] for l in labels]
    for variant in ("axes", "noaxes"):
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(umap3[:, 0], umap3[:, 1], umap3[:, 2],
                   c=colors, s=18, alpha=0.85, linewidths=0, depthshade=True)
        ax.view_init(elev=22, azim=-58)
        ax.set_box_aspect((1, 1, 1), zoom=1.25)
        ax.set_title(f"{family} — 3D UMAP (n_neighbors={nn_used})", y=0.98)
        if variant == "axes":
            ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2"); ax.set_zlabel("UMAP-3")
            for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
                pane.pane.set_alpha(0.04)
        else:
            ax.set_axis_off()
        # Manual legend (3D scatter has no auto legend).
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], marker="o", linestyle="", markersize=8,
                          markerfacecolor=cmap[l],
                          label=("noise" if l == -1 else f"cluster {l}"))
                   for l in sorted(set(labels))]
        ax.legend(handles=handles, fontsize=9, loc="upper left",
                  bbox_to_anchor=(0.0, 0.90))
        fig.savefig(fig_dir / f"fig_umap_3d_{variant}.pdf")
        fig.savefig(fig_dir / f"fig_umap_3d_{variant}.png")
        plt.close(fig)

    # ── Sweep diagnostics table ──────────────────────────────────────────────
    sweep_rows = []
    for nn_key in (chosen,):
        for t in nn_key["trials"]:
            sweep_rows.append({"n_neighbors": nn_key["nn"], "divisor": t["divisor"],
                               "min_cluster_size": t["mcs"],
                               "n_clusters": t["n_clusters"],
                               "noise_frac": t["noise_frac"]})
    pd.DataFrame(sweep_rows).to_csv(out_dir / "cluster_sweep.csv", index=False)

    # ── Save clustered CSV for all downstream modules ────────────────────────
    clustered = out_dir / f"{family.lower()}_clustered.csv"
    keep = (["Seq", "Cluster", "chr", "start", "stop", "strand", "te_name"]
            + present_stages
            + ["umap_x", "umap_y", "tsne_x", "tsne_y", "pca_x", "pca_y",
               "umap3_0", "umap3_1", "umap3_2"])
    df[[c for c in keep if c in df.columns]].to_csv(clustered, index=False)
    _pp(f"  Clustered table → {clustered}")

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    _pp(f"  Final: {n_clusters} clusters (n_neighbors={nn_used}, "
        f"min_cluster_size={chosen['best']['mcs']})")
    return df, clustered, {
        "n_neighbors": nn_used,
        "min_cluster_size": chosen["best"]["mcs"],
        "n_clusters": n_clusters,
        "noise_frac": chosen["best"]["noise_frac"],
    }


# ── per-cluster value table + expression heatmap ─────────────────────────────

def cluster_values(df, present_stages, out_dir, family):
    _style()
    out_dir = Path(out_dir)
    rows = []
    for cl in sorted(df["Cluster"].unique()):
        sub = df[df["Cluster"] == cl]
        rec = {
            "cluster": int(cl),
            "label": "noise" if cl == -1 else f"cluster_{cl}",
            "n_loci": len(sub),
            "n_plus": int((sub["strand"] == "+").sum()),
            "n_minus": int((sub["strand"] == "-").sum()),
            "median_len_bp": int(sub["Seq"].astype(str).str.len().median()),
            "total_expr": round(float(sub["_total_expr"].sum()), 3),
        }
        for s in present_stages:
            rec[f"mean_{s}"] = round(float(sub[s].mean()), 4)
        # dominant developmental stage by mean expression
        if present_stages:
            means = {s: float(sub[s].mean()) for s in present_stages}
            rec["dominant_stage"] = max(means, key=means.get)
        # top TE subfamily names in the cluster
        rec["top_te_name"] = (sub["te_name"].value_counts().index[0]
                              if "te_name" in sub and len(sub) else "")
        rows.append(rec)
    vals = pd.DataFrame(rows)
    vals.to_csv(out_dir / "cluster_values.csv", index=False)
    _pp(f"  Per-cluster values → cluster_values.csv")

    # Heatmap: cluster (rows) x developmental stage (cols), mean expression.
    if present_stages:
        clusters = [c for c in sorted(df["Cluster"].unique()) if c != -1] or \
                   sorted(df["Cluster"].unique())
        mat = np.array([[float(df[df["Cluster"] == c][s].mean())
                         for s in present_stages] for c in clusters])
        # row-normalise for shape comparison
        with np.errstate(invalid="ignore", divide="ignore"):
            norm = mat / np.nanmax(mat, axis=1, keepdims=True)
        norm = np.nan_to_num(norm)
        fig, ax = plt.subplots(figsize=(1.4 * len(present_stages) + 2,
                                        0.6 * len(clusters) + 2))
        im = ax.imshow(norm, aspect="auto", cmap="magma")
        ax.set_xticks(range(len(present_stages)))
        ax.set_xticklabels(present_stages, rotation=30, ha="right")
        ax.set_yticks(range(len(clusters)))
        ax.set_yticklabels([f"cluster {c}" for c in clusters])
        ax.set_title(f"{family} — mean expression by cluster\n(row-normalised)")
        for i in range(len(clusters)):
            for j in range(len(present_stages)):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="white" if norm[i, j] < 0.6 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, label="relative expression")
        ax.grid(False)
        fig.savefig(out_dir / "figures" / "fig_cluster_expression_heatmap.pdf")
        fig.savefig(out_dir / "figures" / "fig_cluster_expression_heatmap.png")
        plt.close(fig)
        _pp("  Expression heatmap → fig_cluster_expression_heatmap.pdf")
    return vals


# ── downstream stage runners (parallel) ──────────────────────────────────────

def _run(cmd, log_path):
    with open(log_path, "w", buffering=1) as lf:
        lf.write("CMD: " + " ".join(cmd) + "\n\n")
        lf.flush()
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
    return rc


def stage_alignment(py, sdir, clustered, out_dir, family):
    cmd = [py, str(sdir / "te_alignment.py"), "--input", str(clustered),
           "--out-dir", str(out_dir / "alignments"), "--family", family]
    return ("alignment", _run(cmd, out_dir / "log_alignment.txt"))


def stage_motif(py, sdir, clustered, out_dir, assembly, jaspar_bed, jaspar_dir):
    cmd = [py, str(sdir / "te_motif.py"), "--input", str(clustered),
           "--build", assembly, "--out-dir", str(out_dir / "motif"),
           "--jaspar-bed", str(jaspar_bed), "--jaspar-dir", str(jaspar_dir),
           "--force"]
    return ("motif", _run(cmd, out_dir / "log_motif.txt"))


def stage_grna(py, sdir, clustered, out_dir, family):
    rdir = out_dir / "grna"
    rdir.mkdir(parents=True, exist_ok=True)
    cmd = [py, str(sdir / "run_grna_analysis.py"), "--input", str(clustered),
           "--reports-dir", str(rdir), "--family", family, "--n-greedy", "5"]
    return ("grna", _run(cmd, out_dir / "log_grna.txt"))


def stage_stage11(py, sdir, clustered, out_dir, family, assembly, cons_fa, only):
    rdir = out_dir / "stage11"
    rdir.mkdir(parents=True, exist_ok=True)
    cmd = [py, str(sdir / "run_stage11_all.py"), "--input", str(clustered),
           "--family", family, "--assembly", assembly,
           "--reports-dir", str(rdir), "--only", only]
    if cons_fa and Path(cons_fa).exists():
        cmd += ["--consensus-fasta", str(cons_fa)]
    return ("stage11", _run(cmd, out_dir / "log_stage11.txt"))


# ── presentation explainer ───────────────────────────────────────────────────

def write_explainer(out_dir, family, cinfo, vals, strand_res):
    out_dir = Path(out_dir)
    n_ok = int((strand_res.get("verdict") == "STRAND_OK").sum()) if not strand_res.empty else 0
    n_checked = int(strand_res["verdict"].astype(str).str.startswith(("STRAND_OK", "MISMATCH", "low_ident")).sum()) if not strand_res.empty else 0
    md = textwrap.dedent(f"""\
    # {family} — reagent-design presentation bundle

    Goal: identify functionally distinct **clusters** of {family} copies and
    design **cluster-specific reagents** (gRNAs, primers) to interrogate their
    individual functions. Sequence clustering separates structurally/regulatorily
    distinct copies; alignments + consensus give the template each reagent must
    match; motif enrichment says *what each cluster may regulate*; gRNA design
    gives the actual targeting reagents; Stage-11 adds age, ORF integrity and
    off-target context that decides which cluster is worth a reagent.

    ## 0. Strand sanity check
    `strand_spotcheck.csv` — {n_ok}/{n_checked} sampled loci confirmed that the
    recorded strand matches the UCSC genomic coordinates (stored Seq equals the
    UCSC plus strand for `+` loci and its reverse-complement for `-` loci). Each
    row carries a UCSC browser link for manual confirmation.

    ## 1. Clustering ({cinfo['n_clusters']} clusters)
    Chosen UMAP `n_neighbors={cinfo['n_neighbors']}`,
    HDBSCAN `min_cluster_size={cinfo['min_cluster_size']}`
    (swept N/5, N/6, N/7 … to land near 5 clusters — see `cluster_sweep.csv`),
    noise fraction {cinfo['noise_frac']:.1%}.

    - `figures/fig_umap_2d.*` — primary view; tight, well-separated clusters mean
      genuinely distinct sequence groups → strong candidates for distinct reagents.
    - `figures/fig_tsne_2d.*` — independent neighbour-embedding; clusters that
      survive *both* UMAP and t-SNE are robust, not embedding artefacts.
    - `figures/fig_pca_2d.*` — linear SVD(50d) view; shows the coarse axes of
      variation behind the clusters (cluster count not optimised here on purpose).
    - `figures/fig_umap_3d_axes.*` / `fig_umap_3d_noaxes.*` — 3D structure; the
      no-axes version is the clean slide graphic, the axes version is for reading
      coordinates.

    ## 2. Alignments & consensus  (`alignments/`)
    Per-cluster MAFFT alignments, majority consensus FASTAs
    (`alignments/cluster_alignments/all_cluster_consensuses.fa`) and CIAlign
    plots. The consensus is the **design template** — a gRNA/primer that matches a
    cluster consensus will hit most members of that cluster and avoid the others.

    ## 3. Motif enrichment  (`motif/`)
    Fresh JASPAR (mm10, SORTED bed) overlap + Fisher enrichment per cluster
    (`motif/enrichment_results/`). Enriched TFs are the candidate **functions** of
    each cluster (e.g. a cluster enriched for a pluripotency factor is a different
    reagent target than one enriched for a repressor).

    ## 4. gRNA design  (`grna/`)
    `grna/grna_candidates.csv`, `grna/grna_greedy_set.csv`, and coverage/positional
    figures. The greedy set is the **minimal panel of guides** covering the most
    loci/expression — the actual reagents to order.

    ## 5. Stage-11 reagent context  (`stage11/`)
    Phylogeny/age, subfamily structure, divergence landscape, LTR structure,
    motif-gain, ORF/fold, and cluster differential expression — these decide which
    cluster is intact/young/expressed enough to be worth a reagent.

    ## Key values
    `cluster_values.csv` — per-cluster size, strand split, length, total &
    per-stage expression, dominant developmental stage, top subfamily.
    `figures/fig_cluster_expression_heatmap.*` — cluster × developmental stage,
    the at-a-glance "which cluster does what, when".
    """)
    (out_dir / "PRESENTATION_README.md").write_text(md)
    _pp("  Explainer → PRESENTATION_README.md")


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="IAP reagent-design presentation driver (runs on cluster).",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--input", default="iapultracombo_tot_counts.csv")
    p.add_argument("--family", default="IAP_Mm")
    p.add_argument("--assembly", default="mm10")
    p.add_argument("--jaspar-bed",
                   default="/home/amodz/anmol/jaspar_cache/JASPAR2024_mm10.sorted.bed.gz",
                   help="SORTED JASPAR bed.gz (default: mm10 sorted cache)")
    p.add_argument("--jaspar-dir", default="/home/amodz/anmol/jaspar_cache")
    p.add_argument("--out-dir", default="iap_presentation")
    p.add_argument("--kmer", type=int, default=6)
    p.add_argument("--target-clusters", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--spotcheck-n", type=int, default=8)
    p.add_argument("--stage11-only",
                   default="phylo,subfamily,divergence,ltr_struct,motif_gain,fold,expression,grna",
                   help="Comma list of reagent-relevant Stage-11 modules.")
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--skip-stage11", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sdir = Path(__file__).resolve().parent
    py = sys.executable

    print("=" * 64)
    print(f"IAP REAGENT-DESIGN PRESENTATION  —  {args.family} / {args.assembly}")
    print(f"  input : {args.input}")
    print(f"  jaspar: {args.jaspar_bed}  (SORTED)")
    print(f"  out   : {out_dir}")
    print("=" * 64)

    # 1. Load + parse.
    df, present = prepare_dataframe(args.input)

    # 2. Strand spot-check (best effort).
    strand_res = spot_check_strand(df, args.assembly, out_dir,
                                   n=args.spotcheck_n, seed=args.seed)

    # 3. Clustering + all required plots (must finish before downstream).
    df, clustered_csv, cinfo = run_clustering(
        df, present, out_dir, args.family, kmer=args.kmer,
        seed=args.seed, target_clusters=args.target_clusters)

    # 4. Per-cluster value table + expression heatmap.
    vals = cluster_values(df, present, out_dir, args.family)

    # 5. Downstream heavy stages in parallel.
    cons_fa = out_dir / "alignments" / "cluster_alignments" / "all_cluster_consensuses.fa"
    futures = {}
    results = []
    _pp(f"Launching downstream stages in parallel (max_workers={args.max_workers}) ...")
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futures[ex.submit(stage_alignment, py, sdir, clustered_csv, out_dir,
                          args.family)] = "alignment"
        futures[ex.submit(stage_motif, py, sdir, clustered_csv, out_dir,
                          args.assembly, args.jaspar_bed, args.jaspar_dir)] = "motif"
        futures[ex.submit(stage_grna, py, sdir, clustered_csv, out_dir,
                          args.family)] = "grna"
        for fut in as_completed(list(futures)):
            name, rc = fut.result()
            status = "ok" if rc == 0 else f"FAILED rc={rc}"
            _pp(f"  [{name}] {status}")
            results.append((name, rc))

    # Stage-11 after alignment (it can consume the consensus FASTA).
    if not args.skip_stage11:
        _pp("Running Stage-11 reagent-relevant modules ...")
        name, rc = stage_stage11(py, sdir, clustered_csv, out_dir, args.family,
                                 args.assembly, cons_fa, args.stage11_only)
        _pp(f"  [{name}] {'ok' if rc == 0 else f'FAILED rc={rc}'}")
        results.append((name, rc))

    # 6. Explainer.
    write_explainer(out_dir, args.family, cinfo, vals, strand_res)

    dt = time.time() - t0
    print("=" * 64)
    print("DONE")
    for name, rc in results:
        print(f"  {name:<12} {'ok' if rc == 0 else f'FAILED rc={rc}'}")
    print(f"  clusters: {cinfo['n_clusters']}  "
          f"(n_neighbors={cinfo['n_neighbors']}, mcs={cinfo['min_cluster_size']})")
    print(f"  outputs : {out_dir}")
    print(f"  elapsed : {dt/60:.1f} min")
    print("=" * 64)
    sys.exit(1 if any(rc != 0 for _, rc in results) else 0)


if __name__ == "__main__":
    main()
