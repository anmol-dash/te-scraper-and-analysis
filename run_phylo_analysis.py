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

# Neutral substitution rate (subs/site/year) keyed by genome assembly. The mouse
# lineage substitutes ~2x faster than human, so using the human rate on mm10/mm39
# data overestimates ages ~2x. Values are order-of-magnitude lineage estimates.
_SUBST_RATES = {
    "hg38": 2.2e-9, "hg19": 2.2e-9, "grch38": 2.2e-9, "grch37": 2.2e-9,
    "t2t": 2.2e-9, "chm13": 2.2e-9, "hs1": 2.2e-9,            # human/primate ~2.2e-9
    "mm10": 4.5e-9, "mm39": 4.5e-9, "mm9": 4.5e-9, "grcm38": 4.5e-9,
    "grcm39": 4.5e-9,                                          # mouse ~4.5e-9
    "rn6": 4.5e-9, "rn7": 4.5e-9,                              # rat ~ mouse-like
}
_DEFAULT_SUBST_RATE = 2.2e-9

# Sanity ceiling on an estimated age: nothing in a genome can be older than the
# age of the Earth. Ages above this are a bug (e.g. a saturated divergence) and
# are recorded as NaN rather than reported.
_MAX_PLAUSIBLE_AGE_YR = 4.5e9


def rate_for_assembly(assembly: str, override: float | None) -> float:
    """Pick a substitution rate: explicit --subst-rate wins, else assembly map."""
    if override is not None:
        return override
    return _SUBST_RATES.get(str(assembly).lower().strip(), _DEFAULT_SUBST_RATE)


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

# Above this Kimura distance a pair is treated as non-homologous / unalignable
# (K2P saturates ~0.75 for random DNA; >1.0 means the two sequences are not a
# real alignment). Such values are returned as NaN rather than emitted as a
# "distance", so they never feed a molecular-clock age.
_MAX_REAL_K2P = 1.0


def kimura2p(a: str, b: str, min_sites: int = 20) -> float:
    """Kimura 2-parameter distance between two aligned, equal-length sequences.

    Returns NaN (not a number) when the pair is unusable: too few comparable
    sites, the K2P formula saturates (1-2P-Q <= 0 or 1-2Q <= 0), or the result
    exceeds _MAX_REAL_K2P (i.e. the two sequences are not genuinely homologous).
    A NaN here propagates to a NaN age — we never fabricate a divergence.
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
    if term1 <= 0 or term2 <= 0:          # saturation — distance is undefined
        return float("nan")
    k = -0.5 * np.log(term1) - 0.25 * np.log(term2)
    if not np.isfinite(k) or k > _MAX_REAL_K2P:
        return float("nan")
    return float(k)


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


class AlignmentError(RuntimeError):
    """Raised when a real alignment cannot be produced. We never pseudo-align."""


def run_mafft(records: list, mafft_cmd: str) -> list:
    """Align records with MAFFT. Returns list of (name, aligned_seq).

    Raises AlignmentError if MAFFT is missing or fails — we NEVER fall back to a
    length-padded pseudo-alignment, because that produces meaningless per-site
    comparisons (and therefore bogus divergence/age). No timeout: large inputs
    are allowed to run to completion (subsample the input instead).
    """
    exe = shutil.which(mafft_cmd) or shutil.which("mafft")
    if not exe:
        raise AlignmentError(
            f"mafft not found (looked for {mafft_cmd!r} and 'mafft' on PATH). "
            "Install MAFFT or pass --mafft-cmd. Refusing to pseudo-align.")
    with tempfile.TemporaryDirectory() as td:
        fin = Path(td) / "in.fa"
        _write_fasta(records, fin)
        res = subprocess.run([exe, "--auto", "--quiet", str(fin)],
                             capture_output=True, text=True)
        if res.returncode != 0:
            raise AlignmentError(
                f"mafft exited with code {res.returncode}: {res.stderr.strip()[:300]}")
        fout = Path(td) / "out.fa"
        fout.write_text(res.stdout)
        out = _read_fasta(fout)
        if not out:
            raise AlignmentError("mafft produced no alignment output.")
        return out


# ── align all copies to the consensus in ONE pass ──────────────────────────────
# Per-copy divergence is measured by aligning every copy onto the family consensus
# coordinate frame with `mafft --addfragments ... --keeplength` (one MAFFT call,
# not a 12k-way MSA and not 12k separate Smith-Waterman alignments — both of which
# are far too slow / what previously timed out and fell back to a pseudo-alignment).
# --keeplength projects each copy onto the consensus length, so every copy row is
# directly comparable to the consensus row column-by-column.

def align_cluster_to_consensus(records: list, mafft_cmd: str):
    """Place every copy on a common frame and derive a real consensus.

    Uses the longest copy as a compact seed reference, then `mafft --addfragments
    --keeplength` to project every copy onto the seed's coordinates in ONE pass
    (no slow N-way MSA). The consensus is then the per-column majority over the
    aligned copies. Returns (consensus_row, {pos: aligned_copy_row}).

    Raises AlignmentError on failure — never pads/pseudo-aligns.
    """
    exe = shutil.which(mafft_cmd) or shutil.which("mafft")
    if not exe:
        raise AlignmentError(
            f"mafft not found (looked for {mafft_cmd!r} and 'mafft'). "
            "Install MAFFT or pass --mafft-cmd. Refusing to pseudo-align.")
    clean = [(i, s.upper().replace("-", "")) for i, (_, s) in enumerate(records)]
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
            raise AlignmentError(
                f"mafft --addfragments failed ({res.returncode}): "
                f"{res.stderr.strip()[:300]}")
        aligned = _read_fasta_text(res.stdout)
    if not aligned:
        raise AlignmentError("mafft --addfragments produced no output.")
    name_to_seq = dict(aligned)
    copy_rows = {i: name_to_seq[f"c{i}"] for i, _ in clean if f"c{i}" in name_to_seq}
    if not copy_rows:
        raise AlignmentError("no copy rows in mafft --addfragments output.")
    # Real consensus = per-column majority base over the aligned copies (with
    # --keeplength every row is the seed length, so columns line up).
    L = max(len(r) for r in copy_rows.values())
    cons_chars = []
    rows = list(copy_rows.values())
    for col in range(L):
        counts = {}
        for r in rows:
            if col < len(r):
                b = r[col].upper()
                if b in "ACGT":
                    counts[b] = counts.get(b, 0) + 1
        cons_chars.append(max(counts, key=counts.get) if counts else "N")
    cons_row = "".join(cons_chars)
    return cons_row, copy_rows


def _read_fasta_text(text: str) -> list:
    records, name, buf = [], None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(buf)))
            name, buf = line[1:].strip().split()[0] if line[1:].strip() else "", []
        else:
            buf.append(line.strip())
    if name is not None:
        records.append((name, "".join(buf)))
    return records


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


def _seq_features(seq: str) -> np.ndarray:
    """3-mer frequency vector (64 features) for a DNA sequence."""
    seq = seq.upper()
    bases = "ACGT"
    kmers = [a + b + c for a in bases for b in bases for c in bases]
    kmap = {k: i for i, k in enumerate(kmers)}
    v = np.zeros(64, dtype=np.float32)
    n = len(seq) - 2
    if n > 0:
        for i in range(n):
            km = seq[i:i + 3]
            if km in kmap:
                v[kmap[km]] += 1
        v /= n
    return v


def auto_cluster_sequences(records: list, target_size: int = 500) -> list:
    """K-means on 3-mer composition. Returns int labels list."""
    n = len(records)
    k = max(2, int(np.ceil(n / target_size)))
    _pp(f"  Auto-clustering {n:,} sequences into {k} clusters "
        f"(target ~{target_size}/cluster)...")
    feats = np.vstack([_seq_features(s) for _, s in records])
    try:
        from sklearn.cluster import MiniBatchKMeans
        labels = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3,
                                 batch_size=min(1024, n)).fit_predict(feats).tolist()
    except ImportError:
        gc = np.array([sum(s.count(b) for b in "GC") / max(len(s), 1)
                       for _, s in records])
        order = np.argsort(gc)
        labels = [0] * n
        for rank, idx in enumerate(order):
            labels[idx] = min(k - 1, int(rank * k / n))
    return labels


def build_family_consensus(records: list, consensus_fasta, mafft_cmd: str,
                           sample_n: int) -> str:
    """Return a single family consensus sequence.

    Priority:
      1. If --consensus-fasta is given, MAFFT-align those consensus sequences
         (few, short) and take the majority — a real alignment.
      2. Otherwise MAFFT-align a representative subsample of up to `sample_n`
         copies and take the majority consensus.

    Never pads/pseudo-aligns. Raises AlignmentError on failure so the caller can
    fail-and-log instead of emitting garbage.
    """
    if consensus_fasta and Path(consensus_fasta).exists():
        recs = _read_fasta(Path(consensus_fasta))
        recs = [(n, s.replace("-", "")) for n, s in recs if s.strip()]
        if not recs:
            raise AlignmentError(f"consensus FASTA {consensus_fasta} contained no sequences.")
        if len(recs) == 1:
            return recs[0][1].upper()
        _pp(f"  Building family consensus from {len(recs)} provided consensus sequences (MAFFT)...")
        return consensus_of(run_mafft(recs, mafft_cmd))

    n = len(records)
    if n == 0:
        raise AlignmentError("no input sequences to build a consensus from.")
    if n <= sample_n:
        sample = records
    else:
        # Deterministic, even-spread subsample across the (genome-ordered) copies.
        idx = np.linspace(0, n - 1, sample_n).astype(int)
        sample = [records[i] for i in sorted(set(idx))]
    _pp(f"  Building family consensus from a {len(sample)}-copy representative MAFFT alignment "
        f"(of {n} copies)...")
    return consensus_of(run_mafft(sample, mafft_cmd))


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
        f"  Assembly / rate:       {res['assembly']}  ({res['subst_rate']:.3g} subs/site/yr, "
        f"clock divisor {res['clock_divisor']:g})",
        f"  Copies analysed:       {res['n_copies']}",
        f"  Usable divergence:     {res['n_usable_div']}  ({res['n_unalignable']} unalignable→NaN)",
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
    p.add_argument("--assembly", default="hg38",
                   help="Genome assembly; selects the neutral substitution rate "
                        "(hg38≈2.2e-9, mm10/mm39≈4.5e-9) unless --subst-rate is given")
    p.add_argument("--subst-rate", type=float, default=None,
                   help="Neutral substitution rate (subs/site/year). Overrides the "
                        "assembly-derived rate when set")
    p.add_argument("--clock-divisor", type=float, default=1.0,
                   help="1=consensus-based age (default, correct for divergence from a "
                        "consensus); 2=paired-LTR (5'/3' LTR) estimate")
    p.add_argument("--intact-orf-aa", type=int, default=100,
                   help="Longest-ORF aa threshold to call a copy 'intact'")
    p.add_argument("--consensus-fasta", default=None,
                   help="Optional consensus FASTA (e.g. all_cluster_consensuses.fa); "
                        "used to build the family consensus instead of subsampling copies")
    p.add_argument("--consensus-sample", type=int, default=400,
                   help="Max copies MAFFT-aligned to build the family consensus when no "
                        "--consensus-fasta is given (default 400)")
    p.add_argument("--max-tree-tips", type=int, default=60,
                   help="Cap tips on the NJ tree for readability")
    p.add_argument("--mafft-cmd", default="mafft")
    p.add_argument("--auto-cluster-size", type=int, default=500,
                   help="Target copies per auto-cluster when no Cluster column is present "
                        "(default 500; prevents MAFFT OOM on large families)")
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

    subst_rate = rate_for_assembly(args.assembly, args.subst_rate)
    _pp(f"  Assembly {args.assembly} → substitution rate {subst_rate:.3g} subs/site/yr "
        f"(clock divisor {args.clock_divisor:g})")

    # 1+2. Per-copy Kimura divergence. To keep the reference compact and biologically
    #      meaningful (a heterogeneous family — solo LTRs + full-length internals —
    #      has no single sensible consensus), each cluster gets its OWN consensus,
    #      built fresh from a MAFFT subsample of that cluster's real copies; its
    #      copies are then placed on that consensus with `mafft --addfragments
    #      --keeplength`. No single-family 33 kb mosaic, no pseudo-alignment.
    div_by_idx = {}
    cluster_consensus = {}      # cid -> ungapped consensus (reused for the NJ tree)
    n_unalignable = 0

    if "Cluster" in df.columns:
        cluster_col = "Cluster"
        groups = list(df.groupby(cluster_col).groups.items())
        # HDBSCAN labels loci it cannot confidently place as -1. That is not a
        # cluster: it is a bag of leftovers with no shared ancestry, and for
        # L1Md_T it held 12,720 of 23,639 loci. Collapsing it to a consensus and
        # placing it in the tree as "cluster_-1" produced a meaningless taxon and
        # let unassigned loci contaminate a consensus. Drop it.
        n_before = len(groups)
        groups = [(cid, idx) for cid, idx in groups if cid != -1]
        if len(groups) < n_before:
            n_noise = int((df[cluster_col] == -1).sum())
            _pp(f"  Excluding the HDBSCAN noise class (-1): {n_noise:,} loci "
                f"({n_noise/len(df)*100:.1f}%) are not a cluster and get no consensus")
        _pp(f"Per-cluster divergence over {len(groups)} clusters "
            f"(consensus from up to {args.consensus_sample} copies each)...")
    else:
        _pp("No Cluster column — auto-clustering by 3-mer composition...")
        auto_labels = auto_cluster_sequences(records, target_size=args.auto_cluster_size)
        df["_auto_cluster"] = auto_labels
        cluster_col = "_auto_cluster"
        groups = list(df.groupby(cluster_col).groups.items())
        _pp(f"  → {len(groups)} auto-clusters, "
            f"largest {max(len(v) for _, v in groups):,} copies")

    for cid, idx_labels in groups:
        members = [(int(i), records[df.index.get_loc(i)]) for i in idx_labels]
        recs = [(f"c{pos}", rec[1]) for pos, (i, rec) in enumerate(members)]
        try:
            cons_row, copy_rows = align_cluster_to_consensus(recs, args.mafft_cmd)
        except AlignmentError as e:
            _pp(f"  cluster {cid}: FATAL — cannot align ({e})")
            sys.exit(2)
        _pp(f"  cluster {cid}: {len(members)} copies, consensus {len(cons_row)} bp, "
            f"{len(copy_rows)} placed")
        cluster_consensus[cid] = cons_row.replace("-", "")
        for pos, (orig_i, rec) in enumerate(members):
            row = copy_rows.get(pos)
            k = kimura2p(row, cons_row) if row is not None else float("nan")
            div_by_idx[orig_i] = k

    _pp("Computing clock ages...")
    div, ages, intact_flags, longest_orfs = [], [], [], []
    for pos, (name, raw) in enumerate(records):
        orig_i = df.index[pos]
        k = div_by_idx.get(orig_i, float("nan"))
        if not np.isfinite(k):
            n_unalignable += 1
            age = float("nan")
        else:
            age = k / (subst_rate * args.clock_divisor)
            if age > _MAX_PLAUSIBLE_AGE_YR:           # impossible — treat as NaN
                age = float("nan")
        div.append(k)
        ages.append(age)
        orf = _longest_orf_aa(raw)
        longest_orfs.append(orf)
        intact_flags.append(orf >= args.intact_orf_aa)
    div = np.array(div)
    ages = np.array(ages)
    _pp(f"  {np.isfinite(div).sum():,}/{len(div):,} copies with a usable divergence "
        f"({n_unalignable:,} unalignable/saturated → NaN)")
    if not np.isfinite(div).any():
        _pp("FATAL: no copy produced a usable divergence; refusing to report a median.")
        sys.exit(2)

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

    # 4. NJ tree over the per-cluster consensuses we ALREADY built above (no extra
    #    slow MSA). If MAFFT is unavailable the tree is skipped (non-fatal) — we
    #    never pseudo-align to fake a tree.
    _pp("Building NJ tree...")
    tree_res = {"ok": False, "n_tips": 0, "newick": ""}
    try:
        tip_records = [(f"cluster_{cid}_n{int((df[cluster_col] == cid).sum())}", cons)
                       for cid, cons in cluster_consensus.items() if cons]
        if len(tip_records) >= 3:
            _pp(f"  Tree over {len(tip_records)} cluster consensuses")
            tree_aln = run_mafft(tip_records, args.mafft_cmd)
            tree_res = build_nj_tree(tree_aln, reports / "fig_phylo_tree.pdf", args.family)
        else:
            _pp(f"  Only {len(tip_records)} cluster consensuses — skipping NJ tree.")
    except AlignmentError as e:
        _pp(f"  WARNING: skipping NJ tree — {e}")

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
        "n_usable_div": int(np.isfinite(div).sum()),
        "n_unalignable": int(n_unalignable),
        "subst_rate": subst_rate,
        "clock_divisor": args.clock_divisor,
        "assembly": args.assembly,
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
