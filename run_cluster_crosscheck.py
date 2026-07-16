#!/usr/bin/env python3
"""run_cluster_crosscheck.py — do the clusters correspond to distinct consensus sequences?

Answers the outstanding half of Andrew's C2: "k-mer clustering ... yields sub-family
boundaries that should be validated against the consensus alignments before being
treated as biological."

The clusters are built from k-mer composition. That says nothing about whether they
are *subfamilies*, i.e. whether each cluster has its own consensus sequence that its
members match better than any other cluster's. This tests exactly that, on evidence
independent of the k-mer/UMAP feature space used to define them:

  1. Build a consensus per cluster (MAFFT alignment of a sample, majority rule),
     reusing the pipeline's own run_mafft/consensus_of.
  2. Align every sampled copy against EVERY cluster consensus and score Kimura
     two-parameter divergence (the same kimura2p the phylogenetics module uses).
  3. Assign each copy to its nearest consensus and compare with its cluster label.

Reading the result:
  * high assignment accuracy => copies match their own cluster's consensus better
    than any other's, which is what a subfamily means at the sequence level.
  * chance-level accuracy => the partition does not correspond to distinct
    consensus sequences and must not be called a subfamily assignment.

Between/within separation is reported too: median divergence to the OWN consensus
versus to the NEAREST OTHER consensus. A partition can be accurate yet biologically
thin if the two are nearly equal.

The HDBSCAN noise class (-1) is excluded: it is not a cluster and has no consensus.

Outputs (into --reports-dir):
  cluster_crosscheck.csv           per-copy divergence to every cluster consensus
  cluster_crosscheck_matrix.csv    mean divergence, cluster x consensus
  fig_cluster_crosscheck.pdf/png   confusion matrix + own-vs-other separation
  cluster_crosscheck_values.tex    \\xc* macros for the manuscript

Usage:
  python run_cluster_crosscheck.py --input reports8/stage11_l1mdt/l1md_t_clustered.csv \\
      --family L1Md_T --reports-dir reports8/stage11_l1mdt --sample 150
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
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


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Clustered CSV (needs Seq + Cluster).")
    p.add_argument("--family", default="FAMILY")
    p.add_argument("--reports-dir", default=".")
    p.add_argument("--sample", type=int, default=150,
                   help="Copies sampled per cluster for consensus building and testing. "
                        "Every copy is aligned against every consensus, so cost is "
                        "quadratic in the cluster count.")
    p.add_argument("--mafft-cmd", default="mafft")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from run_phylo_analysis import run_mafft, consensus_of, kimura2p, _read_fasta_text
    globals()['_read_fasta_text'] = _read_fasta_text

    out = Path(args.reports_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input, low_memory=False)
    for col in ("Seq", "Cluster"):
        if col not in df.columns:
            sys.exit(f"FATAL: {args.input} has no '{col}' column.")
    df = df[df["Seq"].astype(str).str.len() > 0]
    # The noise class is not a cluster and gets no consensus (see run_phylo_analysis).
    n_noise = int((df["Cluster"] == -1).sum())
    df = df[df["Cluster"] != -1].reset_index(drop=True)
    cids = sorted(df["Cluster"].unique())

    _pp(f"=== CLUSTER / CONSENSUS CROSS-CHECK ({args.family}) ===")
    _pp(f"  {len(df):,} clustered loci across {len(cids)} clusters "
        f"(noise class excluded: {n_noise:,} loci)")
    if len(cids) < 2:
        sys.exit("FATAL: need >= 2 clusters to cross-check.")

    rng = np.random.default_rng(args.seed)
    samples = {}
    for c in cids:
        sub = df[df["Cluster"] == c]
        take = min(args.sample, len(sub))
        samples[c] = sub.sample(take, random_state=args.seed)["Seq"].astype(str).tolist()
        _pp(f"  cluster {c}: {len(sub):,} copies, sampling {take}")

    # ---- 1. consensus per cluster (pipeline's own MAFFT + majority rule) ----
    cons = {}
    for c in cids:
        _pp(f"  building consensus for cluster {c} ...")
        recs = [(f"c{c}_{i}", s) for i, s in enumerate(samples[c])]
        try:
            aln = run_mafft(recs, args.mafft_cmd)
            cons[c] = consensus_of(aln)
            _pp(f"    consensus length {len(cons[c])}")
        except Exception as e:
            _pp(f"    FAILED: {e}")
    cids = [c for c in cids if c in cons and cons[c]]
    if len(cids) < 2:
        sys.exit("FATAL: fewer than 2 consensuses could be built.")

    # ---- 2. every sampled copy vs every consensus ---------------------------
    # kimura2p compares column-by-column and therefore requires the copy and the
    # consensus to be ALIGNED. Comparing raw sequences of different lengths gives
    # a meaningless ~10x inflated divergence and collapses every copy onto one
    # consensus. Project the copies onto each consensus's coordinate frame with
    # `mafft --addfragments --keeplength`, exactly as run_phylo_analysis does.
    all_copies = [(c, s) for c in cids for s in samples[c]]
    div = {}                                   # (copy_index, consensus_id) -> divergence
    for c2 in cids:
        _pp(f"  projecting all {len(all_copies)} copies onto cluster {c2}'s consensus ...")
        with tempfile.TemporaryDirectory() as td:
            ref = Path(td) / "ref.fa"
            ref.write_text(f">CONS\n{cons[c2]}\n")
            frags = Path(td) / "frags.fa"
            frags.write_text("".join(f">c{i}\n{s}\n" for i, (_, s) in enumerate(all_copies)))
            r = subprocess.run([shutil.which(args.mafft_cmd) or shutil.which("mafft"),
                                "--addfragments", str(frags), "--keeplength",
                                "--thread", "-1", "--quiet", str(ref)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                _pp(f"    mafft failed ({r.returncode}); skipping consensus {c2}")
                continue
            aln = dict(_read_fasta_text(r.stdout))
        cons_row = aln.get("CONS")
        if not cons_row:
            _pp(f"    consensus row missing; skipping {c2}")
            continue
        for i in range(len(all_copies)):
            row = aln.get(f"c{i}")
            try:
                div[(i, c2)] = kimura2p(row, cons_row) if row else float("nan")
            except Exception:
                div[(i, c2)] = float("nan")

    rows = []
    for i, (c, s) in enumerate(all_copies):
            d = {c2: div.get((i, c2), float("nan")) for c2 in cids}
            vals = {k: v for k, v in d.items() if v == v}
            if not vals:
                continue
            nearest = min(vals, key=vals.get)
            own = vals.get(c, float("nan"))
            others = {k: v for k, v in vals.items() if k != c}
            rows.append(dict(true_cluster=c, nearest_consensus=nearest,
                             correct=(nearest == c), div_own=own,
                             div_nearest_other=(min(others.values()) if others else float("nan")),
                             **{f"div_to_{k}": v for k, v in d.items()}))
    res = pd.DataFrame(rows)
    res.to_csv(out / "cluster_crosscheck.csv", index=False)

    acc = float(res["correct"].mean())
    chance = 1.0 / len(cids)
    per_cluster = res.groupby("true_cluster")["correct"].mean()
    med_own = float(res["div_own"].median())
    med_oth = float(res["div_nearest_other"].median())
    sep = med_oth - med_own

    _pp("")
    _pp(f"  assignment accuracy: {acc*100:.1f}%  (chance {chance*100:.1f}%)")
    for c, v in per_cluster.items():
        _pp(f"    cluster {c}: {v*100:5.1f}%")
    _pp(f"  median divergence to OWN consensus:          {med_own:.4f}")
    _pp(f"  median divergence to NEAREST OTHER consensus:{med_oth:.4f}")
    _pp(f"  separation (other - own):                    {sep:+.4f}")

    mat = res.groupby("true_cluster")[[f"div_to_{c}" for c in cids]].mean()
    mat.to_csv(out / "cluster_crosscheck_matrix.csv")

    # ---- 3. figure ---------------------------------------------------------
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "savefig.dpi": 150, "figure.facecolor": "white"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    conf = pd.crosstab(res["true_cluster"], res["nearest_consensus"], normalize="index")
    conf = conf.reindex(index=cids, columns=cids, fill_value=0.0)
    im = ax1.imshow(conf.values, vmin=0, vmax=1, cmap="viridis")
    ax1.set_xticks(range(len(cids)), [str(c) for c in cids])
    ax1.set_yticks(range(len(cids)), [str(c) for c in cids])
    ax1.set_xlabel("nearest consensus"); ax1.set_ylabel("cluster label")
    ax1.set_title(f"Do copies match their own cluster's consensus?\n"
                  f"{args.family}: {acc*100:.0f}% correct (chance {chance*100:.0f}%)",
                  fontweight="bold", fontsize=10.5)
    for i in range(len(cids)):
        for j in range(len(cids)):
            v = conf.values[i, j]
            ax1.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                     color="white" if v < 0.6 else "black")
    fig.colorbar(im, ax=ax1, label="fraction assigned")

    ax2.hist(res["div_own"], bins=40, alpha=0.75, color=STAND, label="own consensus")
    ax2.hist(res["div_nearest_other"], bins=40, alpha=0.6, color=DELIV,
             label="nearest other consensus")
    ax2.set_xlabel("Kimura divergence"); ax2.set_ylabel("copies")
    ax2.set_title("Divergence to own vs nearest other consensus",
                  fontweight="bold", fontsize=10.5)
    ax2.legend(frameon=False, fontsize=8.5)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_cluster_crosscheck.{ext}", bbox_inches="tight")
    plt.close(fig)
    _pp(f"  wrote {out/'fig_cluster_crosscheck.pdf'} (+ .png)")

    with open(out / "cluster_crosscheck_values.tex", "w") as fh:
        fh.write(f"% Auto-generated by run_cluster_crosscheck.py ({args.family})\n")
        fh.write(f"\\providecommand{{\\xcNClusters}}{{{len(cids)}}}\n")
        fh.write(f"\\providecommand{{\\xcNTested}}{{{len(res):,}}}\n")
        fh.write(f"\\providecommand{{\\xcSample}}{{{args.sample}}}\n")
        fh.write(f"\\providecommand{{\\xcAccuracy}}{{{acc*100:.1f}}}\n")
        fh.write(f"\\providecommand{{\\xcChance}}{{{chance*100:.1f}}}\n")
        fh.write(f"\\providecommand{{\\xcWorstCluster}}{{{per_cluster.min()*100:.1f}}}\n")
        fh.write(f"\\providecommand{{\\xcDivOwn}}{{{med_own:.3f}}}\n")
        fh.write(f"\\providecommand{{\\xcDivOther}}{{{med_oth:.3f}}}\n")
        fh.write(f"\\providecommand{{\\xcSeparation}}{{{sep:.3f}}}\n")
    _pp(f"  wrote {out/'cluster_crosscheck_values.tex'}")


if __name__ == "__main__":
    main()
