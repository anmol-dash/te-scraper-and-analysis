#!/usr/bin/env python3
"""
run_phylo_analysis.py --- GAMECA Subfamily Phylogenetics, Divergence-Age & Master Elements

Answers the question every TE biologist asks but has to stitch together by hand:
which copies of a family are young/intact vs old/degraded, how they relate
phylogenetically, and which copies are the likely active "source" elements.

Pipeline:
  1. Load loci from --input CSV (Seq column required; optional Cluster column).
  2. Multiple sequence alignment of copies (MAFFT) --- graceful fallback if absent.
  3. Build a majority-rule consensus and measure each copy's Kimura 2-parameter
     divergence from it; convert to a molecular-clock age estimate.
  4. ORF-integrity scan (longest ORF fraction) → intact vs degraded annotation.
  5. Neighbour-joining tree over cluster consensuses (or representative copies)
     via Biopython --- graceful fallback if Biopython/MAFFT unavailable.
  6. Master/source-element ranking: youngest (lowest divergence) + most intact.
  7. Figures:
       fig_phylo_tree.pdf        --- NJ tree of consensuses / representatives
       fig_phylo_divergence.pdf  --- Kimura divergence + estimated-age histograms
       fig_phylo_master.pdf      --- top candidate master/source elements
  8. Write phylo_measured_values.tex + phylo_report.txt

Usage:
    python run_phylo_analysis.py \\
        --input loci_with_sequences.csv \\
        --reports-dir ./reports \\
        [--family LTR5_Hs] \\
        [--subst-rate 2.2e-9] \\
        [--clock-divisor 1] \\
        [--intact-orf-aa 100] \\
        [--max-tree-tips 60] \\
        [--mafft-cmd mafft]

Notes on the molecular clock:
  Divergence from the family consensus measures substitutions accumulated along a
  single lineage from the ancestral (consensus) sequence, so age ≈ K / rate.
  For LTR families an LTR–LTR estimate would instead divide by 2*rate; expose this
  via --clock-divisor (1 = consensus-based default, 2 = paired-LTR).
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── constants ──────────────────────────────────────────────────────────────────

_CODON_TABLE = {
    "TTT":"F","TTC":"F","TTA":"L","TTG":"L","CTT":"L","CTC":"L","CTA":"L","CTG":"L",
    "ATT":"I","ATC":"I","ATA":"I","ATG":"M","GTT":"V","GTC":"V","GTA":"V","GTG":"V",
    "TCT":"S","TCC":"S","TCA":"S","TCG":"S","CCT":"P","CCC":"P","CCA":"P","CCG":"P",
    "ACT":"T","ACC":"T","ACA":"T","ACG":"T","GCT":"A","GCC":"A","GCA":"A","GCG":"A",
    "TAT":"Y","TAC":"Y","TAA":"*","TAG":"*","CAT":"H","CAC":"H","CAA":"Q","CAG":"Q",
    "AAT":"N","AAC":"N","AAA":"K","AAG":"K","GAT":"D","GAC":"D","GAA":"E","GAG":"E",
    "TGT":"C","TGC":"C","TGA":"*","TGG":"W","CGT":"R","CGC":"R","CGA":"R","CGG":"R",
    "AGT":"S","AGC":"S","AGA":"R","AGG":"R","GGT":"G","GGC":"G","GGA":"G","GGG":"G",
}
_PURINES = set("AG")
_PYRIMIDINES = set("CT")


# ── helpers ────────────────────────────────────────────────────────────────────

def _pp(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _texesc(s) -> str:
    """Escape LaTeX special characters in free text (family/locus names)."""
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def _revcomp(seq: str) -> str:
    return seq.upper().translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def _longest_orf_aa(seq: str) -> int:
    """Length (in aa) of the longest ORF over all 6 frames."""
    seq = seq.upper().replace(" ", "")
    best = 0
    for s in (seq, _revcomp(seq)):
        for frame in range(3):
            start = None
            for i in range(frame, len(s) - 2, 3):
                aa = _CODON_TABLE.get(s[i:i+3], "X")
                if aa == "M" and start is None:
                    start = i
                elif aa == "*" and start is not None:
                    best = max(best, (i - start) // 3)
                    start = None
    return best


# ── divergence (Kimura 2-parameter) ──────────────────────────────────────────────

def kimura2p(a: str, b: str) -> float:
    """Kimura 2-parameter distance between two aligned, equal-length sequences."""
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
    if sites == 0:
        return float("nan")
    P = transitions / sites
    Q = transversions / sites
    try:
        k = -0.5 * np.log(1 - 2 * P - Q) - 0.25 * np.log(1 - 2 * Q)
    except (ValueError, FloatingPointError):
        return float("nan")
    return float(k) if np.isfinite(k) else float("nan")


# ── FASTA + alignment ─────────────────────────────────────────────────────────

def _write_fasta(records: list, path: Path):
    path.write_text("".join(f">{n}\n{s}\n" for n, s in records))


def _read_fasta(path: Path) -> list:
    records, name, buf = [], None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(buf)))
            name, buf = line[1:].strip(), []
        else:
            buf.append(line.strip())
    if name is not None:
        records.append((name, "".join(buf)))
    return records


def run_mafft(records: list, mafft_cmd: str) -> list:
    """Align records with MAFFT. Returns list of (name, aligned_seq) or [] on failure."""
    exe = shutil.which(mafft_cmd) or shutil.which("mafft")
    if not exe:
        _pp("  WARNING: mafft not found --- falling back to length-padded pseudo-alignment.")
        return _pad_align(records)
    with tempfile.TemporaryDirectory() as td:
        fin = Path(td) / "in.fa"
        _write_fasta(records, fin)
        try:
            res = subprocess.run([exe, "--auto", "--quiet", str(fin)],
                                 capture_output=True, text=True, timeout=600)
        except Exception as e:                                    # noqa: BLE001
            _pp(f"  WARNING: mafft failed ({e}); using pseudo-alignment.")
            return _pad_align(records)
        if res.returncode != 0:
            _pp(f"  WARNING: mafft returncode {res.returncode}; using pseudo-alignment.")
            return _pad_align(records)
        fout = Path(td) / "out.fa"
        fout.write_text(res.stdout)
        return _read_fasta(fout)


def _pad_align(records: list) -> list:
    """Trivial gap-pad to equal length so divergence math still works without MAFFT."""
    m = max((len(s) for _, s in records), default=0)
    return [(n, s + "-" * (m - len(s))) for n, s in records]


def consensus_of(aln: list) -> str:
    """Majority-rule consensus across aligned sequences (ties → most common base)."""
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
        cols.append(max(counts, key=counts.get) if counts else "N")
    return "".join(cols)


# ── NJ tree (Biopython) ─────────────────────────────────────────────────────────

def build_nj_tree(aln: list, out_path: Path, family: str) -> dict:
    """Neighbour-joining tree from an alignment. Returns {'ok','n_tips','newick'}."""
    try:
        from Bio.Align import MultipleSeqAlignment
        from Bio.Seq import Seq
        from Bio.SeqRecord import SeqRecord
        from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
        import Bio.Phylo as Phylo
    except ImportError:
        _pp("  WARNING: Biopython not available --- skipping tree (install biopython).")
        return {"ok": False, "n_tips": 0, "newick": ""}

    if len(aln) < 3:
        _pp("  Fewer than 3 sequences --- skipping tree.")
        return {"ok": False, "n_tips": len(aln), "newick": ""}

    L = max(len(s) for _, s in aln)
    msa = MultipleSeqAlignment(
        [SeqRecord(Seq(s.ljust(L, "-")), id=n[:40]) for n, s in aln])
    calc = DistanceCalculator("identity")
    dm = calc.get_distance(msa)
    tree = DistanceTreeConstructor().nj(dm)
    tree.ladderize()

    fig = plt.figure(figsize=(8, max(3, 0.28 * len(aln))))
    ax = fig.add_subplot(1, 1, 1)
    Phylo.draw(tree, axes=ax, do_show=False,
               label_func=lambda c: c.name if c.is_terminal() else "")
    ax.set_title(f"{family} --- neighbour-joining tree ({len(aln)} tips)",
                 fontweight="bold", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    _pp(f"  Saved {out_path}")

    import io
    sio = io.StringIO()
    Phylo.write(tree, sio, "newick")
    return {"ok": True, "n_tips": len(aln), "newick": sio.getvalue().strip()}


# ── figures ────────────────────────────────────────────────────────────────────

def fig_divergence(div: np.ndarray, ages: np.ndarray, family: str, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    d = div[np.isfinite(div)]
    axes[0].hist(d * 100, bins=min(30, max(5, len(d) // 2)),
                 color="#2980b9", edgecolor="white")
    if len(d):
        axes[0].axvline(np.median(d) * 100, color="black", ls="--",
                        label=f"median {np.median(d)*100:.2f}%")
        axes[0].legend(fontsize=8)
    axes[0].set_xlabel("Kimura divergence from consensus (%)")
    axes[0].set_ylabel("copies")
    axes[0].set_title("Divergence distribution", fontweight="bold")

    a = ages[np.isfinite(ages)] / 1e6
    axes[1].hist(a, bins=min(30, max(5, len(a) // 2)),
                 color="#27ae60", edgecolor="white")
    if len(a):
        axes[1].axvline(np.median(a), color="black", ls="--",
                        label=f"median {np.median(a):.2f} Myr")
        axes[1].legend(fontsize=8)
    axes[1].set_xlabel("Estimated age (Myr)")
    axes[1].set_ylabel("copies")
    axes[1].set_title("Molecular-clock age", fontweight="bold")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{family} --- copy age structure", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    _pp(f"  Saved {out_path}")


def fig_master(master_df: pd.DataFrame, family: str, out_path: Path):
    if master_df.empty:
        return
    m = master_df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(2.5, 0.45 * len(m))))
    colors = ["#c0392b" if intact else "#95a5a6" for intact in m["intact"]]
    ax.barh(m["label"], m["master_score"], color=colors, edgecolor="white")
    ax.set_xlabel("Master-element score  (intactness − divergence)")
    ax.set_title(f"{family} --- candidate master / source elements",
                 fontweight="bold", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    import matplotlib.patches as mpatches
    ax.legend(handles=[mpatches.Patch(color="#c0392b", label="intact ORF"),
                       mpatches.Patch(color="#95a5a6", label="degraded")],
              fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    _pp(f"  Saved {out_path}")


# ── measured values ──────────────────────────────────────────────────────────

def write_values(res: dict, reports: Path):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tex = [
        "% Auto-generated by run_phylo_analysis.py", f"% {ts}", "",
        rf"\providecommand{{\phyloFamily}}{{{_texesc(res['family'])}}}",
        rf"\providecommand{{\phyloNCopies}}{{{res['n_copies']}}}",
        rf"\providecommand{{\phyloNTreeTips}}{{{res['n_tips']}}}",
        rf"\providecommand{{\phyloMedianDiv}}{{{res['median_div_pct']:.2f}}}",
        rf"\providecommand{{\phyloMedianAgeMyr}}{{{res['median_age_myr']:.2f}}}",
        rf"\providecommand{{\phyloNIntact}}{{{res['n_intact']}}}",
        rf"\providecommand{{\phyloMasterName}}{{\texttt{{{_texesc(res['master_name'])}}}}}",
        rf"\providecommand{{\phyloMasterDiv}}{{{res['master_div_pct']:.2f}}}",
        "",
    ]
    (reports / "phylo_measured_values.tex").write_text("\n".join(tex))
    txt = [
        "=" * 60, "GAMECA Phylogenetics / Divergence-Age --- Measured Values",
        f"Generated: {ts}", "=" * 60,
        f"  Family:                {res['family']}",
        f"  Copies analysed:       {res['n_copies']}",
        f"  Tree tips:             {res['n_tips']}",
        f"  Median divergence:     {res['median_div_pct']:.2f} %",
        f"  Median est. age:       {res['median_age_myr']:.2f} Myr",
        f"  Intact (ORF≥thr):      {res['n_intact']}",
        "",
        "  ── Top master/source element ──",
        f"  Name:                  {res['master_name']}",
        f"  Divergence:            {res['master_div_pct']:.2f} %",
    ]
    (reports / "phylo_report.txt").write_text("\n".join(txt))
    _pp("  Written phylo_measured_values.tex and phylo_report.txt")


# ── args + main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="TE subfamily phylogenetics, divergence-age, and master elements",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--input", required=True, help="CSV with a Seq column")
    p.add_argument("--reports-dir", default="./reports")
    p.add_argument("--family", default="TE")
    p.add_argument("--subst-rate", type=float, default=2.2e-9,
                   help="Neutral substitution rate (subs/site/year); human≈2.2e-9")
    p.add_argument("--clock-divisor", type=float, default=1.0,
                   help="1=consensus-based age (default); 2=paired-LTR estimate")
    p.add_argument("--intact-orf-aa", type=int, default=100,
                   help="Longest-ORF aa threshold to call a copy 'intact'")
    p.add_argument("--max-tree-tips", type=int, default=60,
                   help="Cap tips on the NJ tree for readability")
    p.add_argument("--mafft-cmd", default="mafft")
    return p.parse_args()


def main():
    args = parse_args()
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)

    _pp("=" * 60)
    _pp(f"GAMECA Phylogenetics / Divergence-Age --- {args.family}")
    _pp(f"  Input:   {args.input}")
    _pp(f"  Reports: {reports}")
    _pp("=" * 60)

    df = pd.read_csv(args.input, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    if "Seq" not in df.columns:
        _pp("ERROR: CSV must have a 'Seq' column.")
        sys.exit(1)
    df = df[df["Seq"].astype(str).str.len() > 0].reset_index(drop=True)
    _pp(f"  {len(df):,} copies with sequence")

    def _label(i, row):
        for c in ("TE_name", "name", "locus"):
            if c in df.columns and pd.notna(row.get(c)):
                return str(row[c])
        chrom = row.get("chr", row.get("chromosome", "?"))
        return f"{chrom}:{row.get('start', i)}"

    records = [(_label(i, r), str(r["Seq"]).upper()) for i, r in df.iterrows()]

    # 1. align all copies
    _pp("Aligning copies (MAFFT)...")
    aln = run_mafft(records, args.mafft_cmd)
    cons = consensus_of(aln)
    _pp(f"  Consensus length: {len(cons)} bp")

    # 2. per-copy divergence + age
    _pp("Computing Kimura divergence + clock age...")
    aln_map = dict(aln)
    div, ages, intact_flags, longest_orfs = [], [], [], []
    for name, raw in records:
        a = aln_map.get(name, raw)
        k = kimura2p(a, cons)
        div.append(k)
        ages.append((k / (args.subst_rate * args.clock_divisor)) if np.isfinite(k) else float("nan"))
        orf = _longest_orf_aa(raw)
        longest_orfs.append(orf)
        intact_flags.append(orf >= args.intact_orf_aa)
    div = np.array(div)
    ages = np.array(ages)

    df_out = df.copy()
    df_out["kimura_div"] = div
    df_out["est_age_yr"] = ages
    df_out["longest_orf_aa"] = longest_orfs
    df_out["intact"] = intact_flags
    df_out["label"] = [n for n, _ in records]

    # 3. master/source score: intact copies that are youngest (least diverged)
    finite = div[np.isfinite(div)]
    max_div = finite.max() if len(finite) else 1.0
    norm_div = np.where(np.isfinite(div), div / max_div if max_div else 0.0, 1.0)
    orf_arr = np.array(longest_orfs, dtype=float)
    orf_norm = orf_arr / orf_arr.max() if orf_arr.max() else orf_arr
    df_out["master_score"] = 0.6 * orf_norm + 0.4 * (1 - norm_div)
    master_df = df_out.sort_values("master_score", ascending=False)

    df_out.to_csv(reports / "phylo_per_copy.csv", index=False)
    _pp(f"  Wrote {reports/'phylo_per_copy.csv'}")

    # 4. tree (cluster consensuses if available, else representative copies)
    _pp("Building NJ tree...")
    if "Cluster" in df.columns and df["Cluster"].nunique() > 1:
        tip_records = []
        for cl, sub in df.groupby("Cluster"):
            sub_aln = [(n, aln_map.get(n, s)) for n, s in
                       [(_label(i, r), str(r["Seq"])) for i, r in sub.iterrows()]]
            tip_records.append((f"cluster_{cl}_n{len(sub)}", consensus_of(sub_aln)))
        _pp(f"  Tree over {len(tip_records)} cluster consensuses")
    else:
        tip_records = aln[: args.max_tree_tips]
        _pp(f"  Tree over {len(tip_records)} representative copies")
    tree_aln = run_mafft([(n, s.replace("-", "")) for n, s in tip_records], args.mafft_cmd) \
        if "Cluster" in df.columns else tip_records
    tree_res = build_nj_tree(tree_aln, reports / "fig_phylo_tree.pdf", args.family)

    # 5. figures
    fig_divergence(div, ages, args.family, reports / "fig_phylo_divergence.pdf")
    fig_master(master_df, args.family, reports / "fig_phylo_master.pdf")

    # 6. measured values
    top = master_df.iloc[0] if len(master_df) else {}
    res = {
        "family": args.family,
        "n_copies": len(df_out),
        "n_tips": tree_res["n_tips"],
        "median_div_pct": float(np.nanmedian(div) * 100) if np.isfinite(div).any() else 0.0,
        "median_age_myr": float(np.nanmedian(ages) / 1e6) if np.isfinite(ages).any() else 0.0,
        "n_intact": int(np.sum(intact_flags)),
        "master_name": str(top.get("label", "---")) if len(master_df) else "---",
        "master_div_pct": float(top.get("kimura_div", float("nan")) * 100)
        if len(master_df) and np.isfinite(top.get("kimura_div", float("nan"))) else 0.0,
    }
    write_values(res, reports)

    _pp("=" * 60)
    _pp("DONE")
    print("\n" + "=" * 60)
    print("MEASURED VALUES (paste into chat):")
    print("=" * 60)
    print((reports / "phylo_report.txt").read_text())


if __name__ == "__main__":
    main()
