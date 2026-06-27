#!/usr/bin/env python3
"""
te_subfamily.py — Automatic subfamily resolution
─────────────────────────────────────────────────────────────────────────────
Each HDBSCAN cluster has a consensus sequence (from MAFFT alignment or
majority vote). This module:

  1. Fetches the canonical Dfam consensus for the queried family (if reachable)
  2. Computes CpG-corrected K2P divergence between every cluster consensus
     and the canonical
  3. Builds a pairwise distance matrix between cluster consensuses
  4. Outputs an UPGMA dendrogram (subfamily tree) and annotated table

No external tools needed. Tree building uses scipy.cluster.hierarchy (UPGMA).

Outputs (in <reports_dir>/):
  subfamily_table.csv          per-cluster: divergence from canonical,
                               provisional subfamily label
  fig_subfamily_tree.png       UPGMA dendrogram of cluster consensuses
  fig_subfamily_divergence.png per-cluster distance from canonical (bar chart)
  subfamily_values.tex         LaTeX macros for the report

Standalone:
    python te_subfamily.py --input clustered.csv --reports-dir ./reports --family HERVK9
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


# ── Reuse K2P from te_divergence (or inline if not importable) ─────────────

def _kimura2p_cpg(seq_a, seq_b, omega=10.0):
    """CpG-corrected Kimura 2-parameter divergence (see te_divergence.py)."""
    _TR = {("A","G"),("G","A"),("C","T"),("T","C")}
    _TV = {("A","C"),("C","A"),("A","T"),("T","A"),
           ("G","C"),("C","G"),("G","T"),("T","G")}
    n, ts, tv = 0, 0.0, 0.0
    a, b = seq_a.upper(), seq_b.upper()
    for i, (x, y) in enumerate(zip(a, b)):
        if x in "-N" or y in "-N":
            continue
        n += 1
        if x == y:
            continue
        if (x, y) in _TR:
            cpg = (b[i] == "G" and i > 0 and b[i-1] == "C") or \
                  (a[i] == "G" and i > 0 and a[i-1] == "C") or \
                  (b[i] == "C" and i+1 < len(b) and b[i+1] == "G") or \
                  (a[i] == "C" and i+1 < len(a) and a[i+1] == "G")
            ts += 1.0 / omega if cpg else 1.0
        elif (x, y) in _TV:
            tv += 1.0
    if n < 4:
        return None
    p, q = ts / n, tv / n
    t1, t2 = 1 - 2*p - q, 1 - 2*q
    if t1 <= 0 or t2 <= 0:
        return None
    try:
        return max(0.0, -0.5 * math.log(t1) - 0.25 * math.log(t2))
    except ValueError:
        return None


_SF_ALIGNER = None


def _local_align(a, b):
    """Local pairwise alignment via Biopython. Returns (aln_a, aln_b) or None.

    NEVER pads/pseudo-aligns. Raises RuntimeError if Biopython is unavailable.
    """
    global _SF_ALIGNER
    if _SF_ALIGNER is None:
        try:
            from Bio import Align as _BA
        except ImportError as e:
            raise RuntimeError(
                "Biopython is required for subfamily K2P alignment "
                "(pip install biopython). Refusing to pseudo-align.") from e
        al = _BA.PairwiseAligner()
        al.mode = "local"
        al.match_score = 2
        al.mismatch_score = -1
        al.open_gap_score = -5
        al.extend_gap_score = -0.5
        _SF_ALIGNER = al
    a, b = a.upper(), b.upper()
    if not a or not b:
        return None
    try:
        aln = _SF_ALIGNER.align(a, b)[0]
        a_str, b_str = str(aln[0]).upper(), str(aln[1]).upper()
    except Exception:               # noqa: BLE001 — any failure → NaN, never pad
        return None
    return (a_str, b_str) if len(a_str) == len(b_str) else None


def _pairwise_k2p(seqs, omega=10.0):
    """Return NxN symmetric distance matrix of CpG-corrected K2P distances (%).

    Unalignable pairs are recorded as NaN (never a fabricated default).
    """
    n = len(seqs)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            pair = _local_align(seqs[i], seqs[j])
            d = _kimura2p_cpg(*pair, omega=omega) if pair else None
            D[i, j] = D[j, i] = d * 100 if d is not None else np.nan
    return D


# ── Dfam canonical fetch ────────────────────────────────────────────────────

def _fetch_dfam_consensus(family_name, build="hg38", timeout=15):
    """Try to fetch the canonical consensus from the Dfam REST API.

    Returns consensus sequence string or None on failure.
    """
    import urllib.request, json
    # Dfam search by name
    name_q = family_name.replace(" ", "+")
    url = f"https://www.dfam.org/api/families?name={name_q}&limit=5"
    try:
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read())
        results = data.get("results", [])
        if not results:
            return None
        # Pick best match: exact name preferred
        match = next((r for r in results
                      if r.get("name","").lower() == family_name.lower()), results[0])
        accession = match.get("accession", "")
        if not accession:
            return None
        # Fetch consensus sequence via /api/families/{acc}
        fam_url = f"https://www.dfam.org/api/families/{accession}"
        with urllib.request.urlopen(fam_url) as resp2:
            fam_data = json.loads(resp2.read())
        cons = fam_data.get("consensus_sequence", "")
        if cons:
            return cons.upper().replace("\n", "").replace(" ", "")
    except Exception:
        pass
    return None


# ── Majority-vote consensus ────────────────────────────────────────────────

def _majority_vote(seqs, sample_n=400):
    """Majority-vote consensus from a REAL MAFFT alignment of (a subsample of) seqs.

    Raises RuntimeError if MAFFT is unavailable or fails — never pads/pseudo-aligns.
    """
    import shutil, subprocess, tempfile
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
            "pseudo-align. Install MAFFT or provide a consensus FASTA.")
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
    out = []
    for i in range(L):
        col = [s[i] for s in aligned if i < len(s) and s[i] in "ACGT"]
        out.append(Counter(col).most_common(1)[0][0] if col else "N")
    return "".join(out).replace("-", "")


def _load_consensus_fasta(path):
    """Read all_cluster_consensuses.fa → {cluster_id: sequence}."""
    seqs = {}
    if not path or not Path(path).exists():
        return seqs
    cur_id, cur_seq = None, []
    import re
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if cur_id is not None:
                    seqs[cur_id] = "".join(cur_seq).upper()
                m = re.search(r"cluster[_\s]?(\d+)", line, re.IGNORECASE)
                cur_id = int(m.group(1)) if m else line[1:]
                cur_seq = []
            else:
                cur_seq.append(line)
    if cur_id is not None:
        seqs[cur_id] = "".join(cur_seq).upper()
    return seqs


# ── UPGMA tree via scipy ────────────────────────────────────────────────────

def _build_upgma_tree(names, dist_matrix):
    """Build and return a scipy linkage matrix (UPGMA) for a dendrogram."""
    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage
    condensed = squareform(dist_matrix)
    linkage_matrix = linkage(condensed, method="average")
    return linkage_matrix


def _draw_dendrogram(linkage_matrix, labels, title, out_path, palette=None):
    from scipy.cluster.hierarchy import dendrogram
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.9 + 3), 5))
    dn = dendrogram(linkage_matrix, labels=labels, orientation="right",
                    leaf_font_size=10, ax=ax,
                    color_threshold=0.7 * max(linkage_matrix[:, 2]))
    ax.set_xlabel("CpG-corrected K2P divergence (%)", fontsize=11)
    ax.set_title(title, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main function ───────────────────────────────────────────────────────────

def run_subfamily_analysis(input_csv, reports_dir, family_name,
                            consensus_fasta=None, assembly="hg38",
                            cpg_omega=10.0):
    """Build subfamily table and UPGMA tree from cluster consensuses.

    Returns:
        DataFrame with per-cluster subfamily annotations
    """
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== SUBFAMILY RESOLUTION ({family_name}) ===")
    df = pd.read_csv(input_csv)
    cl_col = next((c for c in ("Cluster", "cluster") if c in df.columns), None)
    if cl_col is None or "Seq" not in df.columns:
        print("  SKIP: need 'Cluster' and 'Seq' columns")
        return df

    df[cl_col] = df[cl_col].astype(int)
    cluster_ids = sorted(c for c in df[cl_col].unique() if c >= 0)
    if len(cluster_ids) < 2:
        print("  SKIP: need ≥2 clusters for subfamily tree")
        return df

    print(f"  {len(cluster_ids)} clusters: {cluster_ids}")

    # ── Load or build consensus sequences ───────────────────────────────────
    if consensus_fasta is None:
        _base = Path(input_csv).parent.parent
        for _c in [
            _base / "cluster_alignments" / "all_cluster_consensuses.fa",
            _base / "04_alignments"       / "all_cluster_consensuses.fa",
            _base / "05_consensus"        / "all_cluster_consensuses.fa",
        ]:
            if _c.exists():
                consensus_fasta = str(_c)
                break

    consensuses = _load_consensus_fasta(consensus_fasta)
    if consensuses:
        print(f"  Loaded {len(consensuses)} consensus sequences")
    else:
        print("  Building majority-vote consensuses...")
        for cid in cluster_ids:
            seqs = df[df[cl_col] == cid]["Seq"].dropna().astype(str).tolist()
            consensuses[cid] = _majority_vote(seqs)
            print(f"    Cluster {cid}: {len(consensuses[cid])} bp")

    # ── Fetch Dfam canonical ─────────────────────────────────────────────────
    print(f"  Fetching Dfam canonical for '{family_name}'...")
    canonical = _fetch_dfam_consensus(family_name, build=assembly)
    if canonical:
        print(f"  Dfam canonical: {len(canonical)} bp")
    else:
        print("  Dfam not reachable or family not found — skipping canonical divergence")

    # ── Divergence from canonical ────────────────────────────────────────────
    canonical_divs = {}
    if canonical:
        for cid in cluster_ids:
            cons = consensuses.get(cid, "")
            if not cons:
                canonical_divs[cid] = None
                continue
            pair = _local_align(cons, canonical)
            d = _kimura2p_cpg(*pair, omega=cpg_omega) if pair else None
            canonical_divs[cid] = round(d * 100, 3) if d is not None else None

    # ── Pairwise distance matrix between cluster consensuses ─────────────────
    ordered_ids = [cid for cid in cluster_ids if cid in consensuses]
    seqs_for_tree = [consensuses[cid] for cid in ordered_ids]
    print(f"  Computing {len(ordered_ids)}×{len(ordered_ids)} pairwise K2P matrix...")
    D = _pairwise_k2p(seqs_for_tree, omega=cpg_omega)

    # ── Provisional subfamily labels ─────────────────────────────────────────
    # Clusters with divergence ≤ threshold from canonical → close to canonical
    # Simple rule: cluster within 2% of the canonical = "canonical-type"
    CLOSE_THRESH = 3.0  # % CpG-corrected divergence
    subfam_labels = {}
    for cid in cluster_ids:
        d_can = canonical_divs.get(cid)
        if d_can is None:
            # No canonical → label by cluster ID
            subfam_labels[cid] = f"{family_name}_sub{cid}"
        elif d_can <= CLOSE_THRESH:
            subfam_labels[cid] = f"{family_name}_canonical"
        else:
            subfam_labels[cid] = f"{family_name}_sub{cid}"

    # ── Subfamily table ──────────────────────────────────────────────────────
    rows = []
    for cid in cluster_ids:
        sub = df[df[cl_col] == cid]
        cons = consensuses.get(cid, "")
        d_can = canonical_divs.get(cid)
        rows.append({
            "cluster":             cid,
            "n_loci":              len(sub),
            "consensus_length_bp": len(cons),
            "div_from_canonical_pct": d_can,
            "subfamily_label":     subfam_labels.get(cid, f"{family_name}_sub{cid}"),
            # NB: must be an explicit None-check — `d_can or 999` would treat a
            # perfect 0.0% match (canonical_divs == 0.0) as falsy and mislabel
            # the most-canonical cluster as non-canonical.
            "is_canonical_type":   (d_can is not None and d_can <= CLOSE_THRESH),
        })
    sfam_df = pd.DataFrame(rows)
    sfam_csv = reports_dir / "subfamily_table.csv"
    sfam_df.to_csv(sfam_csv, index=False)
    print(f"  Subfamily table → {sfam_csv}")

    # ── UPGMA tree ────────────────────────────────────────────────────────────
    if len(ordered_ids) >= 2 and not np.isfinite(D).all():
        print("  WARNING: some cluster-consensus pairs were unalignable (NaN distance); "
              "skipping UPGMA tree rather than fabricating distances.")
    elif len(ordered_ids) >= 2:
        try:
            from scipy.cluster.hierarchy import linkage
            from scipy.spatial.distance import squareform

            condensed = squareform(D)
            Z = linkage(condensed, method="average")
            labels = [f"C{cid}\n({subfam_labels[cid].split('_')[-1]})" for cid in ordered_ids]
            tree_path = reports_dir / "fig_subfamily_tree.png"
            _draw_dendrogram(Z, labels,
                             f"{family_name} — Subfamily UPGMA Tree (CpG-corrected K2P)",
                             tree_path)
            print(f"  UPGMA tree → {tree_path}")

            # Export Newick string
            from scipy.cluster.hierarchy import to_tree
            def _to_newick(node, labels):
                if node.is_leaf():
                    return labels[node.id]
                l = _to_newick(node.left,  labels)
                r = _to_newick(node.right, labels)
                dist = node.dist
                return f"({l}:{dist/2:.4f},{r}:{dist/2:.4f})"
            root = to_tree(Z)
            newick = _to_newick(root, labels) + ";"
            (reports_dir / "subfamily_tree.nwk").write_text(newick)
            print(f"  Newick → {reports_dir / 'subfamily_tree.nwk'}")

        except Exception as e:
            print(f"  WARNING: tree building failed: {e}")

    # ── Bar chart: divergence from canonical ─────────────────────────────────
    if canonical and any(v is not None for v in canonical_divs.values()):
        palette = [
            "#4C9BE8","#E8604C","#2ECC71","#9B59B6","#F39C12",
            "#1ABC9C","#E74C3C","#3498DB","#E67E22","#8E44AD",
        ]
        valid_ids  = [cid for cid in cluster_ids if canonical_divs.get(cid) is not None]
        valid_divs = [canonical_divs[cid] for cid in valid_ids]
        colors     = [palette[i % len(palette)] for i in range(len(valid_ids))]

        fig, ax = plt.subplots(figsize=(max(6, len(valid_ids) * 1.2 + 2), 4))
        x = np.arange(len(valid_ids))
        bars = ax.bar(x, valid_divs, color=colors, alpha=0.85)
        ax.axhline(CLOSE_THRESH, color="black", linestyle="--", linewidth=1.2,
                   label=f"Canonical-type threshold ({CLOSE_THRESH}%)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Cluster {cid}\n{subfam_labels[cid].split('_')[-1]}"
                             for cid in valid_ids], fontsize=9)
        ax.set_ylabel("Divergence from Dfam canonical (%)")
        ax.set_title(f"{family_name} — Cluster Divergence from Canonical",
                     fontweight="bold")
        ax.legend(fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        div_path = reports_dir / "fig_subfamily_divergence.png"
        fig.savefig(div_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Divergence bar → {div_path}")

    # ── LaTeX macros ─────────────────────────────────────────────────────────
    n_subfams = len(set(subfam_labels.values()))
    tex = (
        "% Auto-generated by te_subfamily.py\n"
        f"\\newcommand{{\\subfamNClusters}}{{{len(cluster_ids)}}}\n"
        f"\\newcommand{{\\subfamNSubfamilies}}{{{n_subfams}}}\n"
    )
    if canonical:
        valid_divs_all = [v for v in canonical_divs.values() if v is not None]
        if valid_divs_all:
            tex += (
                f"\\newcommand{{\\subfamMinDiv}}{{{min(valid_divs_all):.1f}\\%}}\n"
                f"\\newcommand{{\\subfamMaxDiv}}{{{max(valid_divs_all):.1f}\\%}}\n"
            )
    (reports_dir / "subfamily_values.tex").write_text(tex)
    print(f"  LaTeX values → {reports_dir / 'subfamily_values.tex'}")

    # Write subfamily labels back into main DataFrame (for downstream use)
    df = df.copy()
    df["subfamily_label"] = df[cl_col].map(
        lambda c: subfam_labels.get(c, f"{family_name}_sub{c}") if c >= 0 else "noise"
    )

    return df


# ── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Subfamily resolution: K2P distance from canonical + UPGMA tree")
    p.add_argument("--input",           required=True)
    p.add_argument("--reports-dir",     required=True)
    p.add_argument("--family",          default="FAMILY")
    p.add_argument("--consensus-fasta", default=None)
    p.add_argument("--assembly",        default="hg38")
    p.add_argument("--cpg-omega",       type=float, default=10.0)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_subfamily_analysis(
        input_csv       = args.input,
        reports_dir     = args.reports_dir,
        family_name     = args.family,
        consensus_fasta = args.consensus_fasta,
        assembly        = args.assembly,
        cpg_omega       = args.cpg_omega,
    )
