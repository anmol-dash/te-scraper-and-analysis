#!/usr/bin/env python3
"""run_cluster_validation.py — validate clustering and noise-class handling.

Answers Andrew's review comment C2 ("validate the clustering and noise-class
handling") with two measurements, for one family:

  1. **Noise-class fraction.** HDBSCAN labels points it cannot confidently place
     as -1. This reports what fraction of loci land there, per configuration.
  2. **Assignment stability.** The pipeline picks k and n_neighbors by an
     automatic search, so the obvious reviewer question is whether the partition
     is an artefact of that choice. This sweeps k-mer length and UMAP
     n_neighbors and scores every partition against the pipeline's accepted
     configuration with the Adjusted Rand Index (ARI) and Adjusted Mutual
     Information (AMI). ARI near 1 means the partition is reproducible; ARI near
     0 means it is parameter-dependent and should not be treated as biological.

ARI is reported two ways, because noise handling changes the answer:
  * ari_all   — noise treated as its own label (what the pipeline actually emits)
  * ari_core  — restricted to loci both runs place in a real cluster (tests the
                cluster structure independently of how much is called noise)

Reuses te_clustering.clustering_analysis so the sweep exercises the real code
path rather than a reimplementation. No data is fabricated: every row is a real
clustering run.

Outputs (into --reports-dir):
  cluster_validation.csv           one row per (k, n_neighbors) configuration
  fig_cluster_validation.pdf/.png  ARI heatmap + noise-fraction panel
  cluster_validation_values.tex    \\cv* macros for the manuscript

Usage:
  python run_cluster_validation.py --input l1md_t_clustered.csv \\
      --family L1Md_T --reports-dir reports8/stage11_l1mdt
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Shared house style so this figure matches the rest of the manuscript (C144).
CORE, STAND, DELIV = "#2f6f8f", "#3f8f5f", "#b0413e"


def _labels_for(df, kmer, n_neighbors, min_cluster_size, seed):
    """Run the real clustering path once; return the HDBSCAN label array."""
    from te_clustering import clustering_analysis
    _, labels = clustering_analysis(
        df.copy(), kmer=kmer, n_neighbors=n_neighbors,
        min_cluster_size=min_cluster_size, out_dir=None,
        family_name="validation", compute_tsne=False, random_state=seed,
    )
    return np.asarray(labels)


def _score(ref, lab):
    """ARI/AMI of `lab` vs `ref`, both with noise-as-label and core-only."""
    ari_all = adjusted_rand_score(ref, lab)
    ami_all = adjusted_mutual_info_score(ref, lab)
    core = (ref != -1) & (lab != -1)
    ari_core = adjusted_rand_score(ref[core], lab[core]) if core.sum() > 1 else float("nan")
    return ari_all, ami_all, ari_core


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="CSV with a 'Seq' column.")
    p.add_argument("--family", default="FAMILY")
    p.add_argument("--reports-dir", default=".")
    p.add_argument("--kmers", type=int, nargs="+", default=[5, 6, 7])
    p.add_argument("--neighbors", type=int, nargs="+", default=[15, 30, 50])
    p.add_argument("--ref-kmer", type=int, default=6,
                   help="k accepted by the pipeline's search (reference).")
    p.add_argument("--ref-neighbors", type=int, default=30,
                   help="n_neighbors accepted by the pipeline's search (reference).")
    p.add_argument("--divisor", type=int, default=5,
                   help="min_cluster_size = floor(N/divisor), as the pipeline does.")
    p.add_argument("--max-loci", type=int, default=None,
                   help="Subsample for a faster sweep (sampling is seeded).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.reports_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    if "Seq" not in df.columns:
        sys.exit(f"FATAL: {args.input} has no 'Seq' column. Cluster validation needs "
                 f"sequences; use the clustered CSV from the core run.")
    df = df[df["Seq"].astype(str).str.len() > 0].reset_index(drop=True)
    if args.max_loci and len(df) > args.max_loci:
        df = df.sample(args.max_loci, random_state=args.seed).reset_index(drop=True)

    n = len(df)
    mcs = max(2, n // args.divisor)
    print(f"=== CLUSTER VALIDATION ({args.family}) ===")
    print(f"  {n:,} loci | min_cluster_size = N/{args.divisor} = {mcs}")
    print(f"  reference config: k={args.ref_kmer}, n_neighbors={args.ref_neighbors}")

    # Reference partition = the configuration the pipeline's search accepted.
    ref = _labels_for(df, args.ref_kmer, args.ref_neighbors, mcs, args.seed)
    ref_noise = float((ref == -1).mean())
    print(f"  reference: {len(set(ref[ref != -1]))} clusters, "
          f"noise {ref_noise*100:.1f}%")

    rows = []
    for k in args.kmers:
        for nn in args.neighbors:
            is_ref = (k == args.ref_kmer and nn == args.ref_neighbors)
            lab = ref if is_ref else _labels_for(df, k, nn, mcs, args.seed)
            ari_all, ami_all, ari_core = _score(ref, lab)
            rows.append(dict(
                kmer=k, n_neighbors=nn, is_reference=is_ref,
                n_clusters=len(set(lab[lab != -1])),
                noise_frac=float((lab == -1).mean()),
                ari_all=ari_all, ami_all=ami_all, ari_core=ari_core,
            ))
            print(f"  k={k} nn={nn:<3} clusters={rows[-1]['n_clusters']} "
                  f"noise={rows[-1]['noise_frac']*100:5.1f}% "
                  f"ARI={ari_all:.3f} ARI_core={ari_core:.3f}")

    res = pd.DataFrame(rows)
    csv_path = out / "cluster_validation.csv"
    res.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path}")

    # ---- figure: ARI heatmap + noise fraction ------------------------------
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "savefig.dpi": 150, "figure.facecolor": "white"})
    piv = res.pivot(index="kmer", columns="n_neighbors", values="ari_all")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    im = ax1.imshow(piv.values, vmin=0, vmax=1, cmap="viridis")
    ax1.set_xticks(range(len(piv.columns)), [str(c) for c in piv.columns])
    ax1.set_yticks(range(len(piv.index)), [str(i) for i in piv.index])
    ax1.set_xlabel("UMAP n_neighbors"); ax1.set_ylabel("k-mer length")
    ax1.set_title(f"Partition stability vs accepted config\n"
                  f"(ARI, {args.family})", fontweight="bold", fontsize=11)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax1.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                     color="white" if v < 0.6 else "black")
    fig.colorbar(im, ax=ax1, label="Adjusted Rand Index")

    lbls = [f"k={r.kmer}\nnn={r.n_neighbors}" for r in res.itertuples()]
    cols = [DELIV if r.is_reference else CORE for r in res.itertuples()]
    ax2.bar(range(len(res)), res["noise_frac"] * 100, color=cols)
    ax2.set_xticks(range(len(res)), lbls, fontsize=7)
    # Keep a readable axis even when no loci are called noise (all bars zero).
    ax2.set_ylim(0, max(1.0, res["noise_frac"].max() * 100 * 1.15))
    ax2.set_ylabel("Noise-class loci (%)")
    ax2.set_title("HDBSCAN noise fraction per configuration\n"
                  "(red = accepted config)", fontweight="bold", fontsize=11)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_cluster_validation.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out/'fig_cluster_validation.pdf'} (+ .png)")

    # ---- LaTeX macros ------------------------------------------------------
    nonref = res[~res["is_reference"]]
    tex = out / "cluster_validation_values.tex"
    with open(tex, "w") as fh:
        fh.write(f"% Auto-generated by run_cluster_validation.py ({args.family})\n")
        fh.write(f"\\providecommand{{\\cvFamily}}{{{args.family.replace('_', chr(92)+'_')}}}\n")
        fh.write(f"\\providecommand{{\\cvNLoci}}{{{n:,}}}\n")
        fh.write(f"\\providecommand{{\\cvNConfigs}}{{{len(res)}}}\n")
        fh.write(f"\\providecommand{{\\cvRefKmer}}{{{args.ref_kmer}}}\n")
        fh.write(f"\\providecommand{{\\cvRefNeighbors}}{{{args.ref_neighbors}}}\n")
        fh.write(f"\\providecommand{{\\cvNoisePct}}{{{ref_noise*100:.1f}}}\n")
        fh.write(f"\\providecommand{{\\cvNoiseMinPct}}{{{res['noise_frac'].min()*100:.1f}}}\n")
        fh.write(f"\\providecommand{{\\cvNoiseMaxPct}}{{{res['noise_frac'].max()*100:.1f}}}\n")
        fh.write(f"\\providecommand{{\\cvMinARI}}{{{nonref['ari_all'].min():.2f}}}\n")
        fh.write(f"\\providecommand{{\\cvMedianARI}}{{{nonref['ari_all'].median():.2f}}}\n")
        fh.write(f"\\providecommand{{\\cvMinARIcore}}{{{nonref['ari_core'].min():.2f}}}\n")
        fh.write(f"\\providecommand{{\\cvClustersMin}}{{{res['n_clusters'].min()}}}\n")
        fh.write(f"\\providecommand{{\\cvClustersMax}}{{{res['n_clusters'].max()}}}\n")
    print(f"  wrote {tex}")
    print("\n  Paste-ready summary:")
    print(f"    {n:,} loci | {len(res)} configurations | noise "
          f"{res['noise_frac'].min()*100:.1f}-{res['noise_frac'].max()*100:.1f}% "
          f"| ARI vs accepted config: min {nonref['ari_all'].min():.2f}, "
          f"median {nonref['ari_all'].median():.2f}")


if __name__ == "__main__":
    main()
