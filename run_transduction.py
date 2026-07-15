#!/usr/bin/env python3
"""
run_transduction.py --- GAMECA 3' Transduction Detection (#12b)

LINE-1 frequently mobilizes unique sequence located 3' of the element ("3'
transduction"), so copies that carry the SAME transduced tail were produced by
the same source element. Linking copies by shared 3' tails reconstructs
mobilization lineages and points back to active source elements.

Strategy (genome-free, uses the element + optional downstream flank):
  1. Load copies from --input CSV (Seq required; optional --flank-col with the
     downstream genomic flank, else the 3' terminus of Seq is used).
  2. Orient each copy by strand; take the 3'-terminal window (--tail-bp).
  3. Build the family consensus tail; the portion of each copy's tail that
     EXTENDS BEYOND / DIVERGES FROM the consensus is the candidate transduced seg.
  4. Detect the poly-A signal (AATAAA) and 3' poly-A tail (hallmark of L1).
  5. Group copies sharing transduced-tail k-mers (not present in consensus tail)
     into transduction lineages (connected components).
  6. Figures:
       fig_transduction_groups.pdf  --- lineage sizes + shared-tail heatmap
       fig_transduction_polya.pdf   --- poly-A tail length + signal prevalence
  7. Write transduction_measured_values.tex + transduction_report.txt + CSV.

Usage:
    python run_transduction.py --input copies.csv --reports-dir ./reports \\
        [--family L1HS] [--tail-bp 150] [--kmer 12] [--min-shared 3] [--flank-col flank3]
"""

import argparse
import datetime
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_POLYA_SIGNAL = "AATAAA"


def _pp(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def _texesc(s):
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def _rc(seq):
    return seq.upper().translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def consensus_of(seqs):
    if not seqs:
        return ""
    L = max(len(s) for s in seqs)
    out = []
    for i in range(L):
        c = {}
        for s in seqs:
            if i < len(s) and s[i] in "ACGT":
                c[s[i]] = c.get(s[i], 0) + 1
        out.append(max(c, key=c.get) if c else "N")
    return "".join(out)


def polya_tail_len(seq):
    """Length of the terminal poly-A (or poly-T) run at the 3' end."""
    m = re.search(r"(A+)$", seq.upper())
    return len(m.group(1)) if m else 0


def kmers(seq, k):
    seq = seq.upper()
    return {seq[i:i+k] for i in range(len(seq) - k + 1) if "N" not in seq[i:i+k]}


def is_low_complexity(km, max_base_frac):
    """True for homopolymer-ish k-mers (poly-A tail, A-rich linker).

    These carry no lineage information but survive the consensus subtraction,
    because consensus_of() is positional over unaligned tails and so rarely
    places the poly-A run at a fixed offset. Left in, a single poly-A k-mer is
    shared by nearly every copy: it fuses the family into one component and its
    posting list expands to O(n^2) pairs.
    """
    return max(km.count(b) for b in "ACGT") >= max_base_frac * len(km)


def parse_args():
    p = argparse.ArgumentParser(
        description="3' transduction detection", epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--reports-dir", default="./reports")
    p.add_argument("--family", default="TE")
    p.add_argument("--tail-bp", type=int, default=150)
    p.add_argument("--kmer", type=int, default=12)
    p.add_argument("--min-shared", type=int, default=3,
                   help="Min shared transduced k-mers to link two copies")
    p.add_argument("--max-base-frac", type=float, default=0.8,
                   help="Drop transduced k-mers where one base is >= this fraction "
                        "(poly-A tail / A-rich linker); they are not lineage markers")
    p.add_argument("--max-kmer-copies", type=int, default=0,
                   help="Explicit cap on how many copies may carry a transduced k-mer "
                        "before it is treated as family background. 0 => derive from "
                        "--max-kmer-frac / --kmer-cap-floor")
    p.add_argument("--max-kmer-frac", type=float, default=0.02,
                   help="A k-mer carried by more than this fraction of copies is family "
                        "background, not a lineage marker. This is the k-mer-based "
                        "backstop for consensus_of() being positional over unaligned "
                        "tails: variable poly-A length shifts the element body to a "
                        "different offset per copy, so body k-mers survive the consensus "
                        "subtraction and would otherwise fuse the whole family.")
    p.add_argument("--kmer-cap-floor", type=int, default=50,
                   help="Lower bound for the derived cap, so small families are not "
                        "over-filtered")
    p.add_argument("--heatmap-max", type=int, default=200,
                   help="Max lineage copies drawn in the shared-k-mer heatmap")
    p.add_argument("--flank-col", default="",
                   help="Column holding downstream genomic flank (else 3' of Seq)")
    return p.parse_args()


def main():
    args = parse_args()
    reports = Path(args.reports_dir); reports.mkdir(parents=True, exist_ok=True)
    _pp("=" * 60); _pp(f"GAMECA 3' Transduction --- {args.family}"); _pp("=" * 60)

    df = pd.read_csv(args.input, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    if "Seq" not in df.columns:
        _pp("ERROR: CSV needs a 'Seq' column."); sys.exit(1)
    df = df[df["Seq"].astype(str).str.len() > 0].reset_index(drop=True)
    strand = df["strand"] if "strand" in df.columns else pd.Series(["+"] * len(df))

    def label(i, r):
        for c in ("TE_name", "name", "locus"):
            if c in df.columns and pd.notna(r.get(c)):
                return str(r[c])
        return f"{r.get('chr','?')}:{r.get('start', i)}"

    # 3' tail per copy (oriented)
    tails, labels, polya_lens, has_signal = [], [], [], []
    for i, r in df.iterrows():
        s = str(r["Seq"]).upper()
        if args.flank_col and args.flank_col in df.columns and pd.notna(r.get(args.flank_col)):
            region = str(r[args.flank_col]).upper()
        else:
            region = s
        if str(strand.iloc[i]) == "-":
            region = _rc(region)
        tail = region[-args.tail_bp:]
        tails.append(tail)
        labels.append(label(i, r))
        polya_lens.append(polya_tail_len(region))
        has_signal.append(_POLYA_SIGNAL in region[-args.tail_bp - 30:])

    cons_tail = consensus_of(tails)
    cons_kmers = kmers(cons_tail, args.kmer)
    _pp(f"  {len(tails)} copies; consensus tail {len(cons_tail)} bp")

    # transduced k-mers = tail kmers not in family consensus tail
    trans_kmers = [kmers(t, args.kmer) - cons_kmers for t in tails]
    n_trans = [len(tk) for tk in trans_kmers]

    # link copies sharing >= min_shared transduced kmers
    n = len(tails)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    # Invert transduced k-mers to kmer -> copies. Only copies that share at
    # least one k-mer can ever reach --min-shared, so pair counting runs over
    # posting lists rather than all n^2 pairs.
    postings = defaultdict(list)
    n_lowc = 0
    for i, tk in enumerate(trans_kmers):
        for km in tk:
            if is_low_complexity(km, args.max_base_frac):
                n_lowc += 1
                continue
            postings[km].append(i)
    _pp(f"  dropped {n_lowc:,} low-complexity k-mer instances "
        f"(>= {args.max_base_frac:.0%} one base)")

    # A k-mer carried by a large share of the family is consensus-like residue,
    # not a lineage marker; its posting list is quadratic to expand and links
    # everything into one component. Drop it.
    if args.max_kmer_copies > 0:
        max_copies = args.max_kmer_copies
    else:
        max_copies = max(args.kmer_cap_floor, int(round(n * args.max_kmer_frac)))
    informative = {km: ids for km, ids in postings.items() if 2 <= len(ids) <= max_copies}
    _pp(f"  {len(postings):,} transduced k-mers; {len(informative):,} informative "
        f"(shared by 2..{max_copies:,} copies)")

    pair_counts = defaultdict(int)
    for ids in informative.values():
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                pair_counts[(ids[a], ids[b])] += 1
    _pp(f"  {len(pair_counts):,} candidate pairs share >= 1 transduced k-mer")

    linked = [p for p, c in pair_counts.items() if c >= args.min_shared]
    for i, j in linked:
        union(i, j)
    _pp(f"  {len(linked):,} pairs linked at >= {args.min_shared} shared k-mers")

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    lineages = [g for g in groups.values() if len(g) >= 2]
    lineages.sort(key=len, reverse=True)
    n_in_lineage_all = sum(len(g) for g in lineages)

    df_out = df.copy()
    df_out["label"] = labels
    df_out["polyA_tail_bp"] = polya_lens
    df_out["polyA_signal"] = has_signal
    df_out["n_transduced_kmers"] = n_trans
    grp_id = {i: -1 for i in range(n)}
    for gi, g in enumerate(lineages):
        for i in g:
            grp_id[i] = gi
    df_out["transduction_group"] = [grp_id[i] for i in range(n)]
    df_out.to_csv(reports / "transduction_per_copy.csv", index=False)
    _pp(f"  {len(lineages)} candidate transduction lineages "
        f"(>= 2 copies, >= {args.min_shared} shared k-mers)")

    # ── figures ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if lineages:
        sizes = [len(g) for g in lineages]
        axes[0].bar(range(len(sizes)), sizes, color="#8e44ad", edgecolor="white")
        axes[0].set_xlabel("transduction lineage"); axes[0].set_ylabel("copies")
        axes[0].set_title("Lineage sizes", fontweight="bold")
    else:
        axes[0].text(0.5, 0.5, "no multi-copy lineages\n(shared 3' tails)",
                     ha="center", va="center", transform=axes[0].transAxes, color="gray")
        axes[0].set_title("Lineage sizes", fontweight="bold")
    # Heatmap over lineage members only, largest lineage first. At full family
    # size an n-by-n image is neither renderable nor legible; the informative
    # block is the copies that actually got linked.
    hm_idx = [i for g in lineages for i in g][:args.heatmap_max]
    if hm_idx:
        m = len(hm_idx)
        shared_mat = np.zeros((m, m), dtype=np.int32)
        for a in range(m):
            ka = trans_kmers[hm_idx[a]]
            for b in range(a + 1, m):
                sh = len(ka & trans_kmers[hm_idx[b]])
                shared_mat[a, b] = shared_mat[b, a] = sh
        im = axes[1].imshow(shared_mat, cmap="magma", aspect="auto")
        plt.colorbar(im, ax=axes[1], shrink=0.8, label="shared k-mers")
        shown = f"{m} of {n_in_lineage_all}" if n_in_lineage_all > m else f"{m}"
        axes[1].set_title(f"Shared transduced k-mers\n({shown} lineage copies)",
                          fontweight="bold")
        axes[1].set_xlabel("copy"); axes[1].set_ylabel("copy")
    else:
        axes[1].text(0.5, 0.5, "no linked copies to compare",
                     ha="center", va="center", transform=axes[1].transAxes, color="gray")
        axes[1].set_title("Shared transduced k-mers", fontweight="bold")
    for ax in (axes[0],):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{args.family} --- 3' transduction lineages", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(reports / "fig_transduction_groups.pdf", bbox_inches="tight"); plt.close()
    _pp(f"  Saved {reports/'fig_transduction_groups.pdf'}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(polya_lens, bins=min(20, max(3, n // 2)),
                 color="#16a085", edgecolor="white")
    axes[0].set_xlabel("poly-A tail length (bp)"); axes[0].set_ylabel("copies")
    axes[0].set_title("3' poly-A tail", fontweight="bold")
    sig = int(np.sum(has_signal))
    axes[1].bar(["AATAAA\npresent", "absent"], [sig, n - sig],
                color=["#27ae60", "#bdc3c7"], edgecolor="white")
    axes[1].set_ylabel("copies"); axes[1].set_title("Poly-A signal", fontweight="bold")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{args.family} --- 3' end features", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(reports / "fig_transduction_polya.pdf", bbox_inches="tight"); plt.close()
    _pp(f"  Saved {reports/'fig_transduction_polya.pdf'}")

    # ── measured values ──
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    largest = len(lineages[0]) if lineages else 0
    n_in_lineage = n_in_lineage_all
    tex = [
        "% Auto-generated by run_transduction.py", f"% {ts}", "",
        rf"\providecommand{{\transFamily}}{{{_texesc(args.family)}}}",
        rf"\providecommand{{\transNCopies}}{{{n}}}",
        rf"\providecommand{{\transNLineages}}{{{len(lineages)}}}",
        rf"\providecommand{{\transLargestLineage}}{{{largest}}}",
        rf"\providecommand{{\transNInLineage}}{{{n_in_lineage}}}",
        rf"\providecommand{{\transNPolyaSignal}}{{{sig}}}",
        rf"\providecommand{{\transMedPolyaLen}}{{{int(np.median(polya_lens)) if polya_lens else 0}}}",
        "",
    ]
    (reports / "transduction_measured_values.tex").write_text("\n".join(tex))
    txt = [
        "=" * 60, "GAMECA 3' Transduction --- Measured Values", f"Generated: {ts}", "=" * 60,
        f"  Family:               {args.family}",
        f"  Copies:               {n}",
        f"  Transduction lineages:{len(lineages)}",
        f"  Largest lineage:      {largest} copies",
        f"  Copies in a lineage:  {n_in_lineage}",
        f"  With poly-A signal:   {sig}",
        f"  Median poly-A tail:   {int(np.median(polya_lens)) if polya_lens else 0} bp",
    ]
    (reports / "transduction_report.txt").write_text("\n".join(txt))
    _pp("  Written transduction_measured_values.tex and transduction_report.txt")

    _pp("=" * 60); _pp("DONE")
    print("\n" + "=" * 60); print("MEASURED VALUES (paste into chat):"); print("=" * 60)
    print((reports / "transduction_report.txt").read_text())


if __name__ == "__main__":
    main()
