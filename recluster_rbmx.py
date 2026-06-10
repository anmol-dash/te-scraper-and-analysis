#!/usr/bin/env python3
"""
recluster_rbmx.py

For one RBMX family that already has a 01_data/<fam>_with_sequences.csv (with
Ns from the failed UCSC fetch), this script:

  1. Extracts real sequences from a local hg38 FASTA using GenomeCache.
  2. Saves the corrected CSV back to 01_data/<fam>_with_sequences.csv.
  3. Runs adaptive HDBSCAN clustering:
       - Builds k-mer → SVD → UMAP embedding once per n_neighbors setting.
       - Sweeps HDBSCAN min_cluster_size = N//5, N//6, …, N//max_divisor
         on the same embedding (UMAP is reused across the HDBSCAN sweep).
       - Stops at the first configuration that yields ≥2 real clusters.
       - If n_neighbors=30 finds nothing, retries with n_neighbors=15.
  4. Saves 01_data/<fam>_clustered.csv and 03_clustering/ visualizations.
  5. Runs JASPAR motif overlap + Fisher enrichment via te_motif.py.

Usage (placed next to query.py in the project root):
    python3 recluster_rbmx.py \\
        --results-dir /home/amodz/anmol/results_rbmx_families \\
        --family THE1A \\
        --genome /home/amodz/anmol/hg38.fa \\
        [--jaspar-bed /home/amodz/anmol/JASPAR2022_hg38.bed.gz] \\
        [--jaspar-dir /home/amodz/anmol/jaspar_cache] \\
        [--max-divisor 15] \\
        [--assembly hg38]
"""

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# ── ensure project root is importable ────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from te_genome import GenomeCache, reverse_complement
from te_motif import run_motif_analysis


def _ts(msg):
    import datetime
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  SEQUENCE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_sequences(df, genome_cache):
    """Extract sequences from local genome.

    Aborts the job immediately if ANY locus cannot be fetched — Ns are never
    written to the output CSV.  Prints the first few fetched sequences so the
    caller can verify the genome is being read correctly.
    """
    import os, concurrent.futures as _cf

    rows = list(df.itertuples(index=False))
    n    = len(rows)
    _ts(f"  Extracting {n} sequences from local genome...")

    # Print these indices as spot-checks
    _spot = set([0, 1, 2]) | set(range(0, n, max(1, n // 10))) | {n - 1}

    results = [None] * n

    def _extract(i, row):
        chrom  = str(row.chr)
        start  = int(row.start)
        stop   = int(row.stop)
        strand = str(getattr(row, "strand", "+")).strip()
        seq = genome_cache.extract_sequence(chrom, start, stop)
        if seq is None:
            raise ValueError(f"genome returned None for {chrom}:{start}-{stop}")
        seq = seq.upper()
        if not seq.strip("N"):
            raise ValueError(
                f"only Ns returned for {chrom}:{start}-{stop} "
                f"(len={len(seq)}) — is --genome pointing at the right assembly?"
            )
        if strand == "-":
            seq = reverse_complement(seq)
        return i, seq.upper()

    n_workers = min(8, os.cpu_count() or 4)
    with _cf.ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_extract, i, row): i for i, row in enumerate(rows)}
        done = 0
        for fut in _cf.as_completed(futures):
            try:
                i, seq = fut.result()
            except Exception as exc:
                _ts(f"")
                _ts(f"  *** ABORT: sequence extraction failed ***")
                _ts(f"  Error: {exc}")
                _ts(f"  No output CSV will be written.")
                _ts(f"  Check: (1) --genome path is correct, (2) genome covers hg38 chroms.")
                sys.exit(1)
            results[i] = seq
            if i in _spot:
                _ts(f"    [{i:>5}] {rows[i].chr}:{rows[i].start}-{rows[i].stop}"
                    f"  strand={getattr(rows[i], 'strand', '+')}"
                    f"  seq={seq[:60]}...")
            done += 1
            if done % 1000 == 0 or done == n:
                _ts(f"    {done}/{n} sequences extracted")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2.  ADAPTIVE CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def _kmer_svd(seqs, kmer=6, pca_dims=50, random_state=42):
    """k-mer CountVectorizer → TruncatedSVD.  Returns (X_svd, pca_emb)."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import TruncatedSVD

    _ts(f"    Encoding {len(seqs)} sequences as {kmer}-mer counts...")
    vec = CountVectorizer(analyzer="char", ngram_range=(kmer, kmer), dtype=np.float32)
    X_sparse = vec.fit_transform(seqs)
    n_feat = X_sparse.shape[1]
    _ts(f"    {n_feat} unique {kmer}-mers")

    n_svd = min(pca_dims, n_feat - 1, len(seqs) - 1)
    _ts(f"    TruncatedSVD ({n_svd} components)...")
    svd = TruncatedSVD(n_components=n_svd, random_state=random_state)
    X_svd = svd.fit_transform(X_sparse).astype(np.float32)
    pca_emb = X_svd[:, :2].copy()
    if pca_emb.shape[1] < 2:
        pca_emb = np.column_stack([pca_emb, np.zeros(len(seqs), dtype=np.float32)])
    return X_svd, pca_emb


def _umap_embed(X_svd, n_neighbors=30, random_state=42, n_epochs=200, n_jobs=-1):
    """Run UMAP on the SVD-reduced matrix.  n_jobs controls NN parallelism."""
    import umap as _umap
    n = X_svd.shape[0]
    nn = max(2, min(n_neighbors, n - 1))
    _ts(f"    UMAP (n_neighbors={nn}, n_epochs={n_epochs}, n_jobs={n_jobs})...")
    model = _umap.UMAP(
        n_neighbors=nn,
        min_dist=0.0,
        n_components=2,
        metric="euclidean",
        n_epochs=n_epochs,
        n_jobs=n_jobs,
        low_memory=(n > 10_000),
        random_state=random_state,
    )
    if n > 20_000:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=20_000, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        _ts(f"    Large dataset — fitting UMAP on 20k subsample...")
        model.fit(X_svd[mask])
        emb = np.zeros((n, 2), dtype=np.float32)
        emb[mask]  = model.transform(X_svd[mask]).astype(np.float32)
        emb[~mask] = model.transform(X_svd[~mask]).astype(np.float32)
    else:
        emb = model.fit_transform(X_svd).astype(np.float32)
    _ts("    UMAP done")
    return emb


def _hdbscan_labels(emb, mcs, min_samples=7):
    import hdbscan as _hdbscan
    labels = _hdbscan.HDBSCAN(
        min_cluster_size=mcs,
        min_samples=min_samples,
        core_dist_n_jobs=-1,
    ).fit_predict(emb)
    n_real = len(set(labels) - {-1})
    return labels, n_real


def adaptive_cluster(df, kmer=6, pca_dims=50, random_state=42,
                     n_epochs=200, min_samples=7, max_divisor=15,
                     clustering_dir=None, family_name="FAMILY"):
    """
    Run k-mer + SVD + UMAP + HDBSCAN with an adaptive min_cluster_size search.

    Strategy:
      For n_neighbors in [30, 15]:
        Compute UMAP embedding once.
        Sweep min_cluster_size = N//5, N//6, …, N//max_divisor.
        Stop on the first configuration that gives ≥2 real clusters.
      If still nothing: use the last embedding with mcs=5 (smallest possible).

    Returns df with Cluster/umap_x/umap_y columns added.
    """
    seqs = df["Seq"].astype(str).tolist()
    n = len(seqs)

    # Guard degenerate input
    n_degen = sum(1 for s in seqs if not s or set(s.upper()) <= {"N"})
    if n_degen > n * 0.5:
        _ts(f"  ERROR: {n_degen}/{n} sequences are all-N — extraction failed upstream")
        df = df.copy()
        df["Cluster"] = 0
        df["umap_x"] = 0.0
        df["umap_y"] = 0.0
        df["pca_x"]  = 0.0
        df["pca_y"]  = 0.0
        return df, np.zeros(n, dtype=int), "degenerate"

    if n < 10:
        _ts(f"  Too few sequences ({n}) — assigning all to Cluster 0")
        df = df.copy()
        df["Cluster"] = 0
        df["umap_x"] = 0.0
        df["umap_y"] = 0.0
        df["pca_x"]  = 0.0
        df["pca_y"]  = 0.0
        return df, np.zeros(n, dtype=int), "too_few"

    _ts(f"  Clustering {n} sequences (k={kmer}, max_divisor={max_divisor})...")
    X_svd, pca_emb = _kmer_svd(seqs, kmer=kmer, pca_dims=pca_dims, random_state=random_state)

    # Run both n_neighbors UMAP sweeps in parallel — each gets half the CPUs.
    import os as _os, concurrent.futures as _cf
    _n_cpus   = _os.cpu_count() or 4
    _jobs_each = max(1, _n_cpus // 2)

    def _nn_sweep(n_neighbors):
        """UMAP + HDBSCAN divisor sweep for one n_neighbors value."""
        emb = _umap_embed(X_svd, n_neighbors=n_neighbors,
                          random_state=random_state, n_epochs=n_epochs,
                          n_jobs=_jobs_each)
        last = (None, None, None, n_neighbors)  # (divisor, mcs, labels, nn)
        for divisor in range(5, max_divisor + 1):
            mcs = max(5, n // divisor)
            labels, n_real = _hdbscan_labels(emb, mcs, min_samples=min_samples)
            _ts(f"    n_neighbors={n_neighbors}  N//{divisor}={mcs}  → {n_real} clusters")
            last = (divisor, mcs, labels, n_neighbors)
            if n_real >= 2:
                return emb, divisor, mcs, labels, n_neighbors, True
        divisor, mcs, labels, nn = last
        return emb, divisor, mcs, labels, nn, False

    with _cf.ThreadPoolExecutor(max_workers=2) as _ex:
        _f30 = _ex.submit(_nn_sweep, 30)
        _f15 = _ex.submit(_nn_sweep, 15)
        _r30 = _f30.result()
        _r15 = _f15.result()

    # Prefer n_neighbors=30 if it found clusters; fall back to 15; else use 30's last state
    if _r30[5]:      # nn=30 found ≥2 clusters
        best_umap, best_divisor, best_mcs, best_labels, best_nn, found = _r30
    elif _r15[5]:    # nn=15 found ≥2 clusters
        best_umap, best_divisor, best_mcs, best_labels, best_nn, found = _r15
    else:            # neither found clusters — use nn=30's last state as fallback
        best_umap, best_divisor, best_mcs, best_labels, best_nn, found = _r30

    if found:
        _ts(f"  Found ≥2 clusters: N//{best_divisor}={best_mcs} n_neighbors={best_nn}")
    else:
        _ts(f"  WARNING: no config yielded ≥2 clusters — using N//{best_divisor}={best_mcs}")
        # All-noise fallback: assign cluster 0 to everything
        if best_labels is not None and len(set(best_labels)) == 1 and -1 in set(best_labels):
            best_labels = np.zeros(n, dtype=int)

    # Annotate dataframe
    df = df.copy()
    df["Cluster"] = best_labels
    df["umap_x"]  = best_umap[:, 0]
    df["umap_y"]  = best_umap[:, 1]
    df["pca_x"]   = pca_emb[:, 0]
    df["pca_y"]   = pca_emb[:, 1]

    # Save clustering visualizations if dir provided
    if clustering_dir is not None:
        clustering_dir = Path(clustering_dir)
        clustering_dir.mkdir(parents=True, exist_ok=True)

        coord_cols = [c for c in ["chr", "start", "stop", "strand", "Cluster",
                                   "pca_x", "pca_y", "umap_x", "umap_y"] if c in df.columns]
        df[coord_cols].to_csv(clustering_dir / "clustering_coordinates.csv", index=False)

        _write_cluster_summary(df, clustering_dir, family_name)
        _save_umap_html(df, best_umap, best_labels, clustering_dir, family_name)

    status = f"N//{best_divisor}={best_mcs}_nn={best_nn}"
    return df, best_labels, status


def _write_cluster_summary(df, out_dir, family_name):
    rows = []
    strand_col = next((c for c in ("strand", "Strand") if c in df.columns), None)
    for cl in sorted(df["Cluster"].unique()):
        sub = df[df["Cluster"] == cl]
        row = {"cluster": cl, "n": len(sub), "label": "Noise" if cl == -1 else f"C{cl}"}
        if strand_col:
            row["n_plus"]  = int((sub[strand_col].astype(str).str.strip() == "+").sum())
            row["n_minus"] = int((sub[strand_col].astype(str).str.strip() == "-").sum())
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "cluster_summary.csv", index=False)


def _save_umap_html(df, umap_emb, labels, out_dir, family_name):
    try:
        import plotly.graph_objects as go
        colors = ["#cccccc" if c == -1 else f"hsl({(c * 67) % 360},70%,50%)"
                  for c in labels]
        fig = go.Figure()
        for cl in sorted(set(labels)):
            mask = labels == cl
            lbl  = "Noise" if cl == -1 else f"Cluster {cl}"
            fig.add_trace(go.Scatter(
                x=umap_emb[mask, 0], y=umap_emb[mask, 1],
                mode="markers", name=f"{lbl} (n={mask.sum()})",
                marker=dict(size=4, opacity=0.7),
            ))
        fig.update_layout(
            title=f"{family_name} UMAP clustering",
            xaxis_title="UMAP 1", yaxis_title="UMAP 2", height=600,
        )
        fig.write_html(out_dir / "clustering_visualization.html")
    except Exception:
        pass  # plotly optional


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True,
                    help="Path to results_rbmx_families/ directory")
    ap.add_argument("--family",      required=True,
                    help="Family name exactly as the directory is named (e.g. THE1A, herv9-int)")
    ap.add_argument("--genome",      required=True,
                    help="Path to hg38.fa (must be indexed or GenomeCache-loadable)")
    ap.add_argument("--jaspar-bed",  default=None,
                    help="Pre-downloaded JASPAR BED/BED.gz for hg38 (auto-resolved if omitted)")
    ap.add_argument("--jaspar-dir",  default=None,
                    help="Cache directory for JASPAR downloads")
    ap.add_argument("--max-divisor", type=int, default=15,
                    help="Largest divisor in min_cluster_size sweep (default: 15)")
    ap.add_argument("--assembly",    default="hg38")
    ap.add_argument("--kmer",        type=int, default=6)
    ap.add_argument("--pca-dims",    type=int, default=50)
    ap.add_argument("--n-epochs",    type=int, default=200)
    ap.add_argument("--min-samples", type=int, default=7)
    ap.add_argument("--p-threshold", type=float, default=0.05)
    ap.add_argument("--out-dir",      default=None,
                    help="Fresh output directory for this family (created if absent). "
                         "Defaults to --results-dir/<family> if omitted.")
    ap.add_argument("--force",       action="store_true",
                    help="Re-run motif analysis even if outputs already exist")
    args = ap.parse_args()

    results_dir  = Path(args.results_dir).resolve()
    family_lower = args.family.lower()
    src_family_dir = results_dir / family_lower

    if not src_family_dir.is_dir():
        sys.exit(f"ERROR: family directory not found: {src_family_dir}")

    # Output goes to --out-dir if given (fresh run), else same as source
    family_dir = Path(args.out_dir).resolve() if args.out_dir else src_family_dir

    data_dir       = family_dir / "01_data"
    clustering_dir = family_dir / "03_clustering"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Source CSV (loci + possibly stale Ns from prior run)
    src_csv      = src_family_dir / "01_data" / f"{family_lower}_with_sequences.csv"
    seq_csv      = data_dir / f"{family_lower}_with_sequences.csv"
    clustered_csv= data_dir / f"{family_lower}_clustered.csv"
    # If writing to a different dir, use the source CSV as input
    if family_dir != src_family_dir and not seq_csv.exists():
        import shutil as _sh
        _sh.copy2(src_csv, seq_csv)

    input_csv = seq_csv if seq_csv.exists() else src_csv
    if not input_csv.exists():
        sys.exit(f"ERROR: sequence CSV not found: {src_csv}")

    t_start = time.time()
    _ts(f"{'='*60}")
    _ts(f"Family: {args.family}  ({seq_csv})")
    _ts(f"Genome: {args.genome}")
    _ts(f"{'='*60}")

    # ── 1. Load existing loci + fill sequences ────────────────────────────
    _ts("STAGE 1: Loading loci CSV...")
    df = pd.read_csv(input_csv)
    _ts(f"  {len(df)} loci loaded from {input_csv}")
    _ts(f"  columns: {list(df.columns)}")

    # Upfront check: if every Seq value is all-Ns, we know extraction failed previously
    if "Seq" in df.columns:
        n_ns = (df["Seq"].astype(str).str.fullmatch(r"N+")).sum()
        if n_ns == len(df):
            _ts(f"  NOTE: all {len(df)} sequences are Ns from a prior failed fetch — "
                f"will re-extract from local genome now.")
        elif n_ns > 0:
            _ts(f"  NOTE: {n_ns}/{len(df)} sequences are Ns — will re-extract all.")

    for col in ("chr", "start", "stop"):
        if col not in df.columns:
            # Try capitalised variants written by older pipeline versions
            cap = col.capitalize()
            if cap in df.columns:
                df = df.rename(columns={cap: col})
            else:
                sys.exit(f"ERROR: required column '{col}' missing from {seq_csv}")

    if "strand" not in df.columns and "Strand" in df.columns:
        df = df.rename(columns={"Strand": "strand"})

    _ts("STAGE 2: Loading GenomeCache...")
    cache_dir = str(family_dir)
    gc = GenomeCache(args.genome, cache_dir=cache_dir)
    gc.load()
    if not gc.is_loaded:
        sys.exit(f"ERROR: GenomeCache failed to load {args.genome}")
    _ts(f"  Genome loaded ({len(gc._genome) if hasattr(gc, '_genome') else '?'} chromosomes)")

    _ts("STAGE 3: Extracting sequences...")
    seqs = extract_sequences(df, gc)  # aborts on any failure

    # Final sanity check before writing anything
    n_ns = sum(1 for s in seqs if not s or not s.strip("N"))
    if n_ns > 0:
        sys.exit(
            f"ERROR: {n_ns}/{len(seqs)} sequences are all-Ns after extraction — "
            f"aborting before writing CSV."
        )

    df["Seq"] = seqs
    df.to_csv(seq_csv, index=False)
    _ts(f"  All {len(df)} sequences real — saved: {seq_csv}")

    # ── 2. Adaptive clustering ────────────────────────────────────────────
    _ts("STAGE 4: Adaptive clustering...")
    jaspar_dir = args.jaspar_dir or str(family_dir / "jaspar_cache")

    df_clustered, labels, clust_status = adaptive_cluster(
        df,
        kmer=args.kmer,
        pca_dims=args.pca_dims,
        random_state=42,
        n_epochs=args.n_epochs,
        min_samples=args.min_samples,
        max_divisor=args.max_divisor,
        clustering_dir=clustering_dir,
        family_name=args.family,
    )

    n_clusters = len(set(df_clustered["Cluster"].unique()) - {-1})
    _ts(f"  Clustering done: {n_clusters} cluster(s)  [{clust_status}]")

    df_clustered.to_csv(clustered_csv, index=False)
    _ts(f"  Saved: {clustered_csv}")

    # ── 3. Motif analysis ─────────────────────────────────────────────────
    _ts("STAGE 5: JASPAR motif analysis...")
    try:
        result = run_motif_analysis(
            input_csv=str(clustered_csv),
            build=args.assembly,
            out_dir=str(family_dir),
            jaspar_bed_arg=args.jaspar_bed,
            jaspar_dir=jaspar_dir,
            p_threshold=args.p_threshold,
            force=args.force,
        )
        _ts(f"  Motif analysis done: {result.get('overlaps_path', 'see motif_analysis/')}")
    except SystemExit as exc:
        _ts(f"  WARNING: motif analysis exited with code {exc.code}")
    except Exception as exc:
        _ts(f"  WARNING: motif analysis failed: {type(exc).__name__}: {exc}")
        _ts(traceback.format_exc())

    elapsed = time.time() - t_start
    _ts(f"{'='*60}")
    _ts(f"DONE  {args.family}  [{elapsed/60:.1f} min]")
    _ts(f"  Sequences: {seq_csv}")
    _ts(f"  Clustered: {clustered_csv}")
    _ts(f"  Clusters:  {n_clusters}")
    _ts(f"  Motifs:    {family_dir}/motif_analysis/")
    _ts(f"{'='*60}")


if __name__ == "__main__":
    main()
