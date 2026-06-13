#!/usr/bin/env python3
"""
te_divergence.py — CpG-corrected Kimura 2-parameter divergence + repeat landscape
─────────────────────────────────────────────────────────────────────────────────
Computes per-locus substitution divergence from the cluster consensus sequence
using the Kimura 2-parameter model with CpG-site correction (ω=10, Smit et al.).

The resulting histogram — divergence frequency per %CpG-corrected divergence bin —
is the "repeat landscape" plot that is standard in TE methods papers.

Inputs  (all auto-detected from the output directory):
  - clustered CSV produced by te_clustering.py   (needs 'Seq', 'Cluster')
  - cluster_alignments/all_cluster_consensuses.fa (from te_alignment.py)
    → if absent, majority-vote consensus is built on the fly from 'Seq' column

Outputs  (written to <reports_dir>/):
  divergence_per_locus.csv     per-locus K2P distance + aligned length
  fig_repeat_landscape.pdf     CpG-corrected divergence histogram, faceted by cluster
  fig_repeat_landscape.png     same, raster
  repeat_landscape_values.tex  machine-readable LaTeX value macros for the report

Standalone:
    python te_divergence.py --input clustered.csv --reports-dir ./reports --family HERVK9

As standout module (called by query.py Stage 11):
    python run_divergence.py --input clustered.csv --reports-dir ./reports --family HERVK9
"""

import argparse
import math
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Kimura 2-parameter with CpG correction ─────────────────────────────────

_TRANSITIONS  = {("A","G"),("G","A"),("C","T"),("T","C")}
_TRANSVERSIONS = {("A","C"),("C","A"),("A","T"),("T","A"),
                  ("G","C"),("C","G"),("G","T"),("T","G")}


def _is_cpg_site(seq, i):
    """True if position i is in a CpG context (C followed by G, or G preceded by C)."""
    b = seq[i]
    if b == "C" and i + 1 < len(seq) and seq[i + 1] == "G":
        return True
    if b == "G" and i > 0 and seq[i - 1] == "C":
        return True
    return False


def kimura2p_cpg(seq_ref, seq_query, cpg_omega=10.0):
    """CpG-corrected Kimura 2-parameter divergence between two aligned sequences.

    Args:
        seq_ref:    aligned reference (consensus) — same length as seq_query
        seq_query:  aligned locus sequence
        cpg_omega:  CpG transition rate inflation factor (default 10)

    Returns:
        float divergence (0.0 = identical) or None if calculation not feasible
    """
    seq_ref   = seq_ref.upper()
    seq_query = seq_query.upper()

    n_sites  = 0
    n_trans  = 0.0
    n_transv = 0.0

    for i, (a, b) in enumerate(zip(seq_ref, seq_query)):
        if a in "-N" or b in "-N":
            continue
        n_sites += 1
        if a == b:
            continue
        pair = (a, b)
        if pair in _TRANSITIONS:
            # Downweight CpG transitions
            in_cpg = _is_cpg_site(seq_ref, i) or _is_cpg_site(seq_query, i)
            n_trans += 1.0 / cpg_omega if in_cpg else 1.0
        elif pair in _TRANSVERSIONS:
            n_transv += 1.0

    if n_sites < 4:
        return None

    p = n_trans  / n_sites
    q = n_transv / n_sites

    term1 = 1 - 2 * p - q
    term2 = 1 - 2 * q
    if term1 <= 0 or term2 <= 0:
        return None
    try:
        d = -0.5 * math.log(term1) - 0.25 * math.log(term2)
        return max(0.0, d)
    except ValueError:
        return None


# ── Pairwise alignment ──────────────────────────────────────────────────────

def _align_pair(seq_a, seq_b):
    """Locally align two DNA sequences with Biopython. Returns (aligned_a, aligned_b)
    or None if no real alignment can be produced.

    NEVER pads/pseudo-aligns: a None result makes the caller record NaN rather
    than a fabricated divergence. Local mode lets a truncated copy align to its
    homologous stretch of the consensus without end-gap penalties.
    """
    seq_a = seq_a.upper()
    seq_b = seq_b.upper()
    try:
        from Bio import Align as _BA
    except ImportError as e:
        raise RuntimeError(
            "Biopython is required for divergence alignment (pip install biopython). "
            "Refusing to pseudo-align.") from e
    aligner = _BA.PairwiseAligner()
    aligner.mode               = "local"
    aligner.match_score        = 2
    aligner.mismatch_score     = -1
    aligner.open_gap_score     = -5
    aligner.extend_gap_score   = -0.5
    try:
        aln = aligner.align(seq_a, seq_b)[0]
        a_str = str(aln[0]).upper()
        b_str = str(aln[1]).upper()
    except Exception:               # noqa: BLE001 — any failure → NaN, never pad
        return None
    if len(a_str) != len(b_str):
        return None
    return a_str, b_str


# ── Consensus building ──────────────────────────────────────────────────────

def _majority_vote_consensus(seqs, sample_n=400):
    """Build a majority-vote consensus from a REAL MAFFT alignment of the sequences.

    Subsamples up to `sample_n` representative sequences (MAFFT does not scale to
    tens of thousands), aligns them with MAFFT, and takes the per-column majority.
    Raises RuntimeError if MAFFT is unavailable or fails — we never pad/pseudo-align.
    """
    import shutil
    import subprocess
    import tempfile
    from collections import Counter

    seqs = [s.upper() for s in seqs if s and set(s.upper()) - {"N"}]
    if not seqs:
        return ""
    if len(seqs) == 1:
        return seqs[0]
    if len(seqs) > sample_n:
        idx = np.linspace(0, len(seqs) - 1, sample_n).astype(int)
        seqs = [seqs[i] for i in sorted(set(idx))]

    exe = shutil.which("mafft")
    if not exe:
        raise RuntimeError(
            "mafft not found on PATH — cannot build a real consensus and will not "
            "pseudo-align. Install MAFFT or provide --consensus-fasta.")
    with tempfile.TemporaryDirectory() as td:
        fin = Path(td) / "in.fa"
        fin.write_text("".join(f">s{i}\n{s}\n" for i, s in enumerate(seqs)))
        res = subprocess.run([exe, "--auto", "--quiet", str(fin)],
                             capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"mafft failed ({res.returncode}): {res.stderr.strip()[:200]}")

    aligned, cur = [], []
    for line in res.stdout.splitlines():
        if line.startswith(">"):
            if cur:
                aligned.append("".join(cur)); cur = []
        else:
            cur.append(line.strip().upper())
    if cur:
        aligned.append("".join(cur))
    if not aligned:
        raise RuntimeError("mafft produced no alignment output.")

    L = max(len(s) for s in aligned)
    consensus = []
    for i in range(L):
        col = [s[i] for s in aligned if i < len(s) and s[i] in "ACGT"]
        consensus.append(Counter(col).most_common(1)[0][0] if col else "N")
    return "".join(consensus).replace("-", "")


def _load_consensuses(consensus_fasta):
    """Parse cluster consensus FASTA → {cluster_id: sequence}.

    Header format: >cluster_0_consensus  or  >cluster_N
    """
    seqs = {}
    if not consensus_fasta or not Path(consensus_fasta).exists():
        return seqs
    cur_id, cur_seq = None, []
    with open(consensus_fasta) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if cur_id is not None:
                    seqs[cur_id] = "".join(cur_seq).upper()
                hdr = line[1:]
                # Extract cluster number from header like "cluster_3_consensus"
                import re
                m = re.search(r"cluster[_\s]?(\d+)", hdr, re.IGNORECASE)
                cur_id = int(m.group(1)) if m else hdr
                cur_seq = []
            else:
                cur_seq.append(line)
    if cur_id is not None:
        seqs[cur_id] = "".join(cur_seq).upper()
    return seqs


# ── Main analysis function ──────────────────────────────────────────────────

def run_divergence_analysis(input_csv, reports_dir, family_name,
                             consensus_fasta=None, cpg_omega=10.0,
                             bin_width=1.0, max_div=40.0):
    """Compute per-locus CpG-corrected K2P divergence and generate repeat landscape.

    Args:
        input_csv:       path to clustered CSV (needs 'Seq', 'Cluster')
        reports_dir:     directory to write outputs
        family_name:     used in plot titles
        consensus_fasta: path to all_cluster_consensuses.fa (auto-detected if None)
        cpg_omega:       CpG correction factor (default 10)
        bin_width:       histogram bin width in % divergence (default 1.0)
        max_div:         x-axis cap for histogram (default 40 %)

    Returns:
        DataFrame with per-locus divergence column added
    """
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== DIVERGENCE ANALYSIS ({family_name}) ===")
    df = pd.read_csv(input_csv)
    print(f"  {len(df):,} loci × {len(df.columns)} columns")

    cl_col = next((c for c in ("Cluster", "cluster") if c in df.columns), None)
    if cl_col is None or "Seq" not in df.columns:
        print("  SKIP: need 'Cluster' and 'Seq' columns")
        return df

    df[cl_col] = df[cl_col].astype(int)
    cluster_ids = sorted(c for c in df[cl_col].unique() if c >= 0)
    print(f"  Clusters: {cluster_ids}")

    # ── Locate consensus FASTA ───────────────────────────────────────────────
    if consensus_fasta is None:
        # Try standard locations relative to input CSV
        _base = Path(input_csv).parent.parent
        for _candidate in [
            _base / "cluster_alignments" / "all_cluster_consensuses.fa",
            _base / "04_alignments"       / "all_cluster_consensuses.fa",
            _base / "05_consensus"        / "all_cluster_consensuses.fa",
        ]:
            if _candidate.exists():
                consensus_fasta = str(_candidate)
                break

    consensuses = _load_consensuses(consensus_fasta)
    if consensuses:
        print(f"  Loaded {len(consensuses)} consensus sequences from {consensus_fasta}")
    else:
        print("  No consensus FASTA found — building majority-vote consensus per cluster")
        for cid in cluster_ids:
            seqs = df[df[cl_col] == cid]["Seq"].dropna().astype(str).tolist()
            consensuses[cid] = _majority_vote_consensus(seqs)
            print(f"    Cluster {cid}: consensus len={len(consensuses[cid])}")

    # ── Per-locus divergence ─────────────────────────────────────────────────
    k2p_vals  = []
    aln_lens  = []
    pct_cpg   = []

    print(f"  Computing K2P divergence (ω={cpg_omega})...")
    for i in range(len(df)):
        row    = df.iloc[i]
        cid    = int(row[cl_col])
        seq    = str(row["Seq"]).upper()
        cons   = consensuses.get(cid, "")
        if not cons or not seq or set(seq) <= {"N"}:
            k2p_vals.append(np.nan)
            aln_lens.append(0)
            pct_cpg.append(np.nan)
            continue

        pair = _align_pair(cons, seq)
        if pair is None:
            k2p_vals.append(np.nan)
            aln_lens.append(0)
            pct_cpg.append(np.nan)
            continue
        a_ref, a_qry = pair
        div = kimura2p_cpg(a_ref, a_qry, cpg_omega=cpg_omega)
        k2p_vals.append(div * 100 if div is not None else np.nan)

        n_valid = sum(1 for a, b in zip(a_ref, a_qry) if a not in "-N" and b not in "-N")
        aln_lens.append(n_valid)

        # CpG density in the locus
        n_cpg = sum(1 for j in range(len(seq) - 1) if seq[j] == "C" and seq[j+1] == "G")
        pct_cpg.append(100.0 * n_cpg / max(len(seq) - 1, 1))

        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(df)} done")

    df = df.copy()
    df["kimura_div_pct"] = k2p_vals
    df["aln_length"]     = aln_lens
    df["cpg_density_pct"] = pct_cpg

    out_csv = reports_dir / "divergence_per_locus.csv"
    df[["chr", "start", "stop", cl_col, "kimura_div_pct", "aln_length", "cpg_density_pct"]
       if all(c in df.columns for c in ["chr","start","stop"])
       else [cl_col, "kimura_div_pct", "aln_length", "cpg_density_pct"]].to_csv(out_csv, index=False)
    print(f"  Per-locus CSV → {out_csv}")

    # ── Repeat landscape histogram ───────────────────────────────────────────
    valid = df["kimura_div_pct"].dropna()
    if valid.empty:
        print("  No valid divergence values — skipping plot")
        return df

    bins = np.arange(0, max_div + bin_width, bin_width)
    palette = [
        "#4C9BE8","#E8604C","#2ECC71","#9B59B6","#F39C12",
        "#1ABC9C","#E74C3C","#3498DB","#E67E22","#8E44AD",
    ]

    fig, ax = plt.subplots(figsize=(12, 5))

    # Combined (all clusters) background
    hist_all, _ = np.histogram(valid, bins=bins)
    ax.bar(bins[:-1], hist_all, width=bin_width, color="#cccccc", alpha=0.5,
           label="All loci", zorder=1)

    # Per-cluster lines
    for i, cid in enumerate(cluster_ids):
        sub = df[df[cl_col] == cid]["kimura_div_pct"].dropna()
        if sub.empty:
            continue
        hist_cl, _ = np.histogram(sub, bins=bins)
        ax.step(bins[:-1], hist_cl, where="post",
                color=palette[i % len(palette)], linewidth=2,
                label=f"Cluster {cid} (n={len(sub):,})", zorder=2 + i)

    median_div = float(valid.median())
    ax.axvline(median_div, color="black", linestyle="--", linewidth=1.5,
               label=f"Median {median_div:.1f}%")

    ax.set_xlabel("CpG-corrected Kimura 2-parameter divergence (%)", fontsize=12)
    ax.set_ylabel("Number of loci", fontsize=12)
    ax.set_title(f"{family_name} — Repeat Landscape (CpG-corrected, ω={cpg_omega:.0f})",
                 fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max_div)

    for fmt, fname in [("pdf", "fig_repeat_landscape.pdf"),
                        ("png", "fig_repeat_landscape.png")]:
        p = reports_dir / fname
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  {fmt.upper()} → {p}")
    plt.close(fig)

    # ── Per-cluster faceted plot ─────────────────────────────────────────────
    if len(cluster_ids) > 1:
        ncols = min(3, len(cluster_ids))
        nrows = math.ceil(len(cluster_ids) / ncols)
        fig2, axes = plt.subplots(nrows, ncols,
                                   figsize=(ncols * 5, nrows * 3.5),
                                   sharey=False, sharex=True)
        axes_flat = np.array(axes).flatten() if hasattr(axes, "__len__") else [axes]
        for i, cid in enumerate(cluster_ids):
            ax2 = axes_flat[i]
            sub = df[df[cl_col] == cid]["kimura_div_pct"].dropna()
            if sub.empty:
                ax2.set_visible(False)
                continue
            hist2, _ = np.histogram(sub, bins=bins)
            ax2.bar(bins[:-1], hist2, width=bin_width,
                    color=palette[i % len(palette)], alpha=0.85)
            ax2.set_title(f"Cluster {cid}  n={len(sub):,}", fontsize=10)
            ax2.set_xlim(0, max_div)
            ax2.spines[["top","right"]].set_visible(False)
            med = float(sub.median())
            ax2.axvline(med, color="black", linestyle="--", linewidth=1, alpha=0.7)
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)
        fig2.supxlabel("Divergence (%)", fontsize=11)
        fig2.supylabel("Loci count", fontsize=11)
        fig2.suptitle(f"{family_name} — Per-cluster Repeat Landscape", fontweight="bold")
        plt.tight_layout()
        p2 = reports_dir / "fig_repeat_landscape_per_cluster.png"
        fig2.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Per-cluster PNG → {p2}")

    # ── Stats CSV ────────────────────────────────────────────────────────────
    rows = []
    for cid in cluster_ids:
        sub = df[df[cl_col] == cid]["kimura_div_pct"].dropna()
        if sub.empty:
            continue
        rows.append({
            "cluster": cid, "n": len(sub),
            "mean_div_pct": round(float(sub.mean()), 3),
            "median_div_pct": round(float(sub.median()), 3),
            "std_div_pct": round(float(sub.std()), 3),
            "p10_div_pct": round(float(np.percentile(sub, 10)), 3),
            "p90_div_pct": round(float(np.percentile(sub, 90)), 3),
        })
    all_valid = df["kimura_div_pct"].dropna()
    rows.append({
        "cluster": "all", "n": len(all_valid),
        "mean_div_pct": round(float(all_valid.mean()), 3),
        "median_div_pct": round(float(all_valid.median()), 3),
        "std_div_pct": round(float(all_valid.std()), 3),
        "p10_div_pct": round(float(np.percentile(all_valid, 10)), 3),
        "p90_div_pct": round(float(np.percentile(all_valid, 90)), 3),
    })
    stats_df = pd.DataFrame(rows)
    stats_csv = reports_dir / "divergence_stats.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"  Stats CSV → {stats_csv}")

    # ── LaTeX value macros ───────────────────────────────────────────────────
    n_loci   = len(all_valid)
    med_all  = float(all_valid.median())
    mean_all = float(all_valid.mean())
    tex = (
        "% Auto-generated by te_divergence.py\n"
        f"\\newcommand{{\\divNloci}}{{{n_loci:,}}}\n"
        f"\\newcommand{{\\divMedianAll}}{{{med_all:.1f}\\%}}\n"
        f"\\newcommand{{\\divMeanAll}}{{{mean_all:.1f}\\%}}\n"
        f"\\newcommand{{\\divCpgOmega}}{{{cpg_omega:.0f}}}\n"
        f"\\newcommand{{\\divNClusters}}{{{len(cluster_ids)}}}\n"
    )
    for row in rows[:-1]:  # skip 'all'
        cid = row["cluster"]
        tex += (
            f"\\newcommand{{\\divMedianC{cid}}}{{{row['median_div_pct']:.1f}\\%}}\n"
        )
    tex_path = reports_dir / "repeat_landscape_values.tex"
    tex_path.write_text(tex)
    print(f"  LaTeX values → {tex_path}")

    return df


# ── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="CpG-corrected Kimura 2-parameter divergence + repeat landscape")
    p.add_argument("--input",          required=True)
    p.add_argument("--reports-dir",    required=True)
    p.add_argument("--family",         default="FAMILY")
    p.add_argument("--consensus-fasta",default=None,
                   help="Path to all_cluster_consensuses.fa "
                        "(auto-detected from --input location if omitted)")
    p.add_argument("--cpg-omega",      type=float, default=10.0,
                   help="CpG transition correction factor (default 10)")
    p.add_argument("--bin-width",      type=float, default=1.0)
    p.add_argument("--max-div",        type=float, default=40.0)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_divergence_analysis(
        input_csv      = args.input,
        reports_dir    = args.reports_dir,
        family_name    = args.family,
        consensus_fasta= args.consensus_fasta,
        cpg_omega      = args.cpg_omega,
        bin_width      = args.bin_width,
        max_div        = args.max_div,
    )
