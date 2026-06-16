#!/usr/bin/env python3
"""
plot_consensus_distance.py --- two consensus-distance figures for a TE family.

Given a loci/expression CSV (e.g. iapultracombo_tot_counts.csv) this builds REAL
MAFFT alignments and produces the two figures that best summarise a Stage-11
family:

  fig_cluster_consensus_distance.pdf
      Each copy's Kimura-2P distance to ITS OWN cluster's consensus.
      (How diverged is each copy from the other copies of the same subfamily.)

  fig_global_consensus_distance.pdf
      Each copy's Kimura-2P distance to the SINGLE family-wide (global) consensus.
      (How diverged is each copy from the family as a whole.)

Both figures:
  * group copies by cluster on the x-axis,
  * draw an alternating shaded band behind every cluster so the clusters are
    visually separated,
  * print the cluster's MEAN distance (and n) next to the cluster name on the
    x-axis, and draw a black mean bar across each cluster,
  * state in the title that the per-cluster mean is annotated, plus the overall
    mean and the copy count actually used.

"Good alignments" is enforced two ways:
  1. Clusters default to the RepeatMasker subfamily parsed from the TE_ID column
     (homogeneous subfamilies align cleanly) rather than 3-mer k-means, which
     mixes solo-LTRs and internals and produces saturated, meaningless distances.
  2. Copies whose Kimura distance saturates / is unalignable (K2P NaN or above
     --max-k2p) are EXCLUDED from the plotted distribution and counted in the
     title, so a bad alignment never masquerades as a real distance.

Alignment method (one MAFFT pass per batch, never a pseudo-alignment):
  per cluster -> longest copy is the seed; every copy is projected onto it with
  `mafft --addfragments --keeplength`; the consensus is the per-column majority.
  The global consensus is the majority over the MAFFT alignment of all cluster
  consensuses; every copy is then projected onto it the same way.

Usage (cluster, automatic):
    python plot_consensus_distance.py \
        --input iapultracombo_tot_counts.csv \
        --family IAP \
        --out-dir consensus_distance_figs

    # cluster by an explicit column instead of the TE_ID subfamily:
    python plot_consensus_distance.py --input loci.csv --family IAP \
        --cluster-from column --cluster-col Cluster

Requires: numpy, pandas, matplotlib, and MAFFT on PATH (same env the Stage-11
phylo module already uses).
"""

import argparse
import datetime
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _get_cmap(name: str, n: int):
    """Colormap with n discrete colors, robust across matplotlib versions."""
    try:
        return matplotlib.colormaps[name].resampled(n)   # mpl >= 3.7
    except (AttributeError, KeyError):
        import matplotlib.cm as cm
        return cm.get_cmap(name, n)                       # older mpl


# ── logging ──────────────────────────────────────────────────────────────────
def _pp(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Kimura 2-parameter distance ───────────────────────────────────────────────
_PURINES = set("AG")
_PYRIMIDINES = set("CT")


def kimura2p(a: str, b: str, min_sites: int = 20, max_k2p: float = 1.0) -> float:
    """Kimura-2P distance between two aligned, equal-length sequences.

    Returns NaN when the pair is unusable (too few comparable sites, the K2P
    formula saturates, or the result exceeds max_k2p i.e. the two sequences are
    not genuinely homologous). A NaN here is treated as a failed alignment and
    dropped from the plot rather than reported as a distance.
    """
    transitions = transversions = sites = 0
    for x, y in zip(a.upper(), b.upper()):
        if x in "-N" or y in "-N" or x not in "ACGT" or y not in "ACGT":
            continue
        sites += 1
        if x == y:
            continue
        if (x in _PURINES and y in _PURINES) or (x in _PYRIMIDINES and y in _PYRIMIDINES):
            transitions += 1
        else:
            transversions += 1
    if sites < min_sites:
        return float("nan")
    P = transitions / sites
    Q = transversions / sites
    term1 = 1.0 - 2.0 * P - Q
    term2 = 1.0 - 2.0 * Q
    if term1 <= 0 or term2 <= 0:
        return float("nan")
    k = -0.5 * np.log(term1) - 0.25 * np.log(term2)
    if not np.isfinite(k) or k > max_k2p:
        return float("nan")
    return float(k)


# ── FASTA / MAFFT helpers ─────────────────────────────────────────────────────
class AlignmentError(RuntimeError):
    """Raised when a real alignment cannot be produced. We never pseudo-align."""


def _read_fasta_text(text: str) -> list:
    records, name, buf = [], None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(buf)))
            name = line[1:].strip().split()[0] if line[1:].strip() else ""
            buf = []
        else:
            buf.append(line.strip())
    if name is not None:
        records.append((name, "".join(buf)))
    return records


def _mafft_exe(mafft_cmd: str) -> str:
    exe = shutil.which(mafft_cmd) or shutil.which("mafft")
    if not exe:
        raise AlignmentError(
            f"mafft not found (looked for {mafft_cmd!r} and 'mafft' on PATH). "
            "Install MAFFT or pass --mafft-cmd. Refusing to pseudo-align.")
    return exe


def run_mafft(records: list, mafft_cmd: str) -> list:
    """Plain MAFFT MSA of (name, seq) records -> list of (name, aligned_seq)."""
    exe = _mafft_exe(mafft_cmd)
    with tempfile.TemporaryDirectory() as td:
        fin = Path(td) / "in.fa"
        fin.write_text("".join(f">{n}\n{s}\n" for n, s in records))
        res = subprocess.run([exe, "--auto", "--quiet", str(fin)],
                             capture_output=True, text=True)
        if res.returncode != 0:
            raise AlignmentError(f"mafft failed ({res.returncode}): "
                                 f"{res.stderr.strip()[:300]}")
        out = _read_fasta_text(res.stdout)
        if not out:
            raise AlignmentError("mafft produced no alignment output.")
        return out


def _consensus_of(aln: list) -> str:
    """Per-column majority base over aligned (name, seq) records (ungapped)."""
    if not aln:
        return ""
    L = max(len(s) for _, s in aln)
    cols = []
    for i in range(L):
        counts = {}
        for _, s in aln:
            if i < len(s):
                b = s[i].upper()
                if b in "ACGT":
                    counts[b] = counts.get(b, 0) + 1
        cols.append(max(counts, key=counts.get) if counts else "-")
    return "".join(cols).replace("-", "")


def align_cluster_to_consensus(seqs: list, mafft_cmd: str):
    """Project every sequence onto a cluster consensus in ONE MAFFT pass.

    seqs: list of plain DNA strings.
    Returns (consensus_row, [aligned_row_per_input_or_None]) where every row is
    the seed length (--keeplength), directly comparable to consensus_row.
    """
    exe = _mafft_exe(mafft_cmd)
    clean = [(i, s.upper().replace("-", "")) for i, s in enumerate(seqs)]
    clean = [(i, s) for i, s in clean if s]
    if not clean:
        raise AlignmentError("no non-empty sequences in cluster.")
    seed_i, seed_seq = max(clean, key=lambda t: len(t[1]))
    with tempfile.TemporaryDirectory() as td:
        ref = Path(td) / "ref.fa"
        ref.write_text(f">SEED\n{seed_seq}\n")
        frags = Path(td) / "frags.fa"
        frags.write_text("".join(f">c{i}\n{s}\n" for i, s in clean))
        res = subprocess.run(
            [exe, "--addfragments", str(frags), "--keeplength",
             "--thread", "-1", "--quiet", str(ref)],
            capture_output=True, text=True)
        if res.returncode != 0:
            raise AlignmentError(f"mafft --addfragments failed ({res.returncode}): "
                                 f"{res.stderr.strip()[:300]}")
        aligned = dict(_read_fasta_text(res.stdout))
    rows = {i: aligned[f"c{i}"] for i, _ in clean if f"c{i}" in aligned}
    if not rows:
        raise AlignmentError("no copy rows in mafft --addfragments output.")
    L = max(len(r) for r in rows.values())
    cons_chars = []
    rlist = list(rows.values())
    for col in range(L):
        counts = {}
        for r in rlist:
            if col < len(r):
                b = r[col].upper()
                if b in "ACGT":
                    counts[b] = counts.get(b, 0) + 1
        cons_chars.append(max(counts, key=counts.get) if counts else "N")
    cons_row = "".join(cons_chars)
    return cons_row, [rows.get(i) for i in range(len(seqs))]


def align_to_reference(seqs: list, reference: str, mafft_cmd: str, batch: int = 1000):
    """Project each sequence onto a FIXED reference with --keeplength, in batches.

    Returns a list (len == len(seqs)) of aligned rows (reference length) or None.
    """
    exe = _mafft_exe(mafft_cmd)
    ref_clean = reference.upper().replace("-", "")
    out_rows = [None] * len(seqs)
    n = len(seqs)
    for start in range(0, n, batch):
        chunk = [(i, seqs[i].upper().replace("-", ""))
                 for i in range(start, min(start + batch, n))]
        chunk = [(i, s) for i, s in chunk if s]
        if not chunk:
            continue
        with tempfile.TemporaryDirectory() as td:
            ref = Path(td) / "ref.fa"
            ref.write_text(f">GLOBAL\n{ref_clean}\n")
            frags = Path(td) / "frags.fa"
            frags.write_text("".join(f">c{i}\n{s}\n" for i, s in chunk))
            res = subprocess.run(
                [exe, "--addfragments", str(frags), "--keeplength",
                 "--thread", "-1", "--quiet", str(ref)],
                capture_output=True, text=True)
            if res.returncode != 0:
                raise AlignmentError(f"global mafft --addfragments failed "
                                     f"({res.returncode}): {res.stderr.strip()[:300]}")
            aligned = dict(_read_fasta_text(res.stdout))
        for i, _ in chunk:
            out_rows[i] = aligned.get(f"c{i}")
        _pp(f"    global alignment: {min(start + batch, n):,}/{n:,} copies")
    return out_rows


# ── clustering source ─────────────────────────────────────────────────────────
def _subfamily_from_te_id(te_id: str) -> str:
    """RepeatMasker subfamily name from a TE_ID like
    'chr1|3031358|3031710|IAPLTR1a_Mm:ERVK:LTR|21|-' -> 'IAPLTR1a_Mm'."""
    try:
        field = str(te_id).split("|")[3]
        return field.split(":")[0]
    except Exception:
        return "NA"


def assign_clusters(df: pd.DataFrame, source: str, cluster_col: str) -> pd.Series:
    if source == "column":
        if cluster_col not in df.columns:
            sys.exit(f"ERROR: --cluster-from column but '{cluster_col}' not in CSV.")
        return df[cluster_col].astype(str)
    if source == "te_id":
        if "TE_ID" not in df.columns:
            sys.exit("ERROR: --cluster-from te_id but no TE_ID column in CSV.")
        return df["TE_ID"].map(_subfamily_from_te_id).astype(str)
    sys.exit(f"ERROR: unknown --cluster-from {source!r}")


# ── plotting ──────────────────────────────────────────────────────────────────
def make_distance_figure(df: pd.DataFrame, dist_col: str, cluster_col: str,
                         family: str, what: str, out_path: Path,
                         seed: int = 0):
    """Strip plot: per-copy distance, grouped+shaded by cluster, mean annotated.

    df is already filtered to rows with a finite, good distance in dist_col.
    `what` is a short phrase for titles/labels, e.g. 'per-cluster consensus'.
    """
    rng = np.random.default_rng(seed)
    # Order clusters by mean distance so the gradient of divergence is readable.
    stats = (df.groupby(cluster_col)[dist_col]
               .agg(["mean", "median", "count"])
               .sort_values("mean"))
    clusters = list(stats.index)
    pos = {c: i for i, c in enumerate(clusters)}
    cmap = _get_cmap("viridis", max(2, len(clusters)))

    fig_w = max(9.0, 0.42 * len(clusters) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, 6.2))

    # alternating shaded band behind every cluster -> clear separation
    for i, c in enumerate(clusters):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="#000000", alpha=0.045, zorder=0)

    for i, c in enumerate(clusters):
        vals = df.loc[df[cluster_col] == c, dist_col].to_numpy() * 100.0
        x = i + (rng.random(len(vals)) - 0.5) * 0.62
        ax.scatter(x, vals, s=7, alpha=0.45, linewidths=0,
                   color=cmap(i), zorder=2, rasterized=True)
        m = stats.loc[c, "mean"] * 100.0
        # black mean bar across the cluster
        ax.plot([i - 0.42, i + 0.42], [m, m], color="black", lw=2.2, zorder=4)

    overall_mean = df[dist_col].mean() * 100.0
    ax.axhline(overall_mean, color="#c0392b", ls="--", lw=1.4, zorder=3,
               label=f"overall mean {overall_mean:.1f}%")

    # x labels: cluster name + (n, mean%) — the mean printed next to the name
    labels = [f"{c}\n(n={int(stats.loc[c, 'count'])}, μ={stats.loc[c, 'mean']*100:.1f}%)"
              for c in clusters]
    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_xlim(-0.6, len(clusters) - 0.4)
    ax.set_ylim(bottom=0)

    ax.set_ylabel("Kimura-2P distance from\n" + what + " (%)", fontsize=11)
    ax.set_xlabel("cluster (subfamily)  —  per-cluster mean μ annotated", fontsize=10)
    ax.set_title(
        f"{family} — per-copy distance from {what}\n"
        f"per-cluster mean (μ) printed under each cluster;  "
        f"overall μ = {overall_mean:.1f}%,  N = {len(df):,} copies (good alignments only)",
        fontweight="bold", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", ls=":", alpha=0.4, zorder=0)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()
    _pp(f"  Saved {out_path}")


# ── args + main ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Two consensus-distance figures (cluster + global) for a TE family.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--input", required=True, help="Loci CSV with a Seq column.")
    p.add_argument("--family", default="TE")
    # --reports-dir is the alias run_stage11_all.py / query.py pass to every module.
    p.add_argument("--out-dir", "--reports-dir", dest="out_dir",
                   default="consensus_distance_figs")
    p.add_argument("--cluster-from", choices=["te_id", "column"], default="te_id",
                   help="te_id: cluster by RepeatMasker subfamily parsed from TE_ID "
                        "(default, gives clean alignments); column: use --cluster-col.")
    p.add_argument("--cluster-col", default="Cluster",
                   help="Column to cluster by when --cluster-from column.")
    p.add_argument("--min-cluster", type=int, default=10,
                   help="Drop clusters with fewer than this many copies (default 10).")
    p.add_argument("--max-k2p", type=float, default=0.75,
                   help="Distances above this (saturated/non-homologous) are dropped "
                        "as failed alignments (default 0.75).")
    p.add_argument("--global-sample", type=int, default=400,
                   help="Max cluster-consensus seqs MAFFT-aligned to build the global "
                        "consensus (default 400).")
    p.add_argument("--mafft-cmd", default="mafft")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    _pp("=" * 60)
    _pp(f"Consensus-distance figures — {args.family}")
    _pp(f"  Input:   {args.input}")
    _pp(f"  Out:     {out}")
    _pp("=" * 60)

    df = pd.read_csv(args.input, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    if "Seq" not in df.columns:
        sys.exit("ERROR: CSV must have a 'Seq' column.")
    df = df[df["Seq"].astype(str).str.len() > 0].reset_index(drop=True)
    df["__cluster"] = assign_clusters(df, args.cluster_from, args.cluster_col)

    # drop tiny clusters so a 5-copy subfamily doesn't dominate a band
    sizes = df["__cluster"].value_counts()
    keep = sizes[sizes >= args.min_cluster].index
    dropped_small = int((~df["__cluster"].isin(keep)).sum())
    df = df[df["__cluster"].isin(keep)].reset_index(drop=True)
    _pp(f"  {len(df):,} copies in {df['__cluster'].nunique()} clusters "
        f"(dropped {dropped_small} copies in clusters < {args.min_cluster}).")

    seqs = df["Seq"].astype(str).str.upper().tolist()
    cluster_dist = np.full(len(df), np.nan)
    cluster_consensuses = {}

    # 1. per-cluster consensus + distance
    _pp("Aligning each cluster to its own consensus (MAFFT --addfragments)...")
    for cid, sub in df.groupby("__cluster"):
        idx = sub.index.to_list()
        try:
            cons_row, rows = align_cluster_to_consensus([seqs[i] for i in idx],
                                                         args.mafft_cmd)
        except AlignmentError as e:
            _pp(f"  cluster {cid}: SKIP — {e}")
            continue
        cluster_consensuses[cid] = cons_row.replace("-", "")
        good = 0
        for j, i in enumerate(idx):
            row = rows[j]
            if row is not None:
                k = kimura2p(row, cons_row, max_k2p=args.max_k2p)
                cluster_dist[i] = k
                if np.isfinite(k):
                    good += 1
        _pp(f"  cluster {cid}: {len(idx)} copies, cons {len(cons_row)} bp, "
            f"{good} good distances")
    df["dist_cluster"] = cluster_dist

    # 2. global consensus (majority over MAFFT of the cluster consensuses) + distance
    _pp("Building global (family-wide) consensus...")
    cons_recs = [(str(cid), s) for cid, s in cluster_consensuses.items() if s]
    if len(cons_recs) >= 2:
        if len(cons_recs) > args.global_sample:
            idx = np.linspace(0, len(cons_recs) - 1, args.global_sample).astype(int)
            cons_recs = [cons_recs[i] for i in sorted(set(idx))]
        global_consensus = _consensus_of(run_mafft(cons_recs, args.mafft_cmd))
    elif cons_recs:
        global_consensus = cons_recs[0][1]
    else:
        sys.exit("ERROR: no cluster consensus could be built; cannot make global consensus.")
    _pp(f"  Global consensus: {len(global_consensus)} bp")

    _pp("Aligning every copy to the global consensus...")
    global_rows = align_to_reference(seqs, global_consensus, args.mafft_cmd)
    global_dist = np.full(len(df), np.nan)
    for i, row in enumerate(global_rows):
        if row is not None:
            global_dist[i] = kimura2p(row, global_consensus, max_k2p=args.max_k2p)
    df["dist_global"] = global_dist

    # save the per-copy table
    keep_cols = [c for c in ("TE_ID", "Chromosome", "Start", "Stop") if c in df.columns]
    df[keep_cols + ["__cluster", "dist_cluster", "dist_global"]].rename(
        columns={"__cluster": "cluster"}).to_csv(
        out / "consensus_distance_per_copy.csv", index=False)
    _pp(f"  Wrote {out/'consensus_distance_per_copy.csv'}")

    # 3. figures (good alignments only)
    cl = df[np.isfinite(df["dist_cluster"])].copy()
    gl = df[np.isfinite(df["dist_global"])].copy()
    _pp(f"  Cluster-consensus: {len(cl):,}/{len(df):,} copies usable "
        f"({len(df)-len(cl)} unalignable/saturated dropped).")
    _pp(f"  Global-consensus:  {len(gl):,}/{len(df):,} copies usable "
        f"({len(df)-len(gl)} unalignable/saturated dropped).")

    make_distance_figure(cl, "dist_cluster", "__cluster", args.family,
                         "per-cluster consensus",
                         out / "fig_cluster_consensus_distance.pdf")
    make_distance_figure(gl, "dist_global", "__cluster", args.family,
                         "global (family-wide) consensus",
                         out / "fig_global_consensus_distance.pdf")

    _pp("=" * 60)
    _pp("DONE")
    _pp(f"  {out/'fig_cluster_consensus_distance.pdf'}")
    _pp(f"  {out/'fig_global_consensus_distance.pdf'}")
    _pp(f"  {out/'consensus_distance_per_copy.csv'}")


if __name__ == "__main__":
    main()
