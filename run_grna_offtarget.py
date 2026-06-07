#!/usr/bin/env python3
"""
run_grna_offtarget.py --- GAMECA Allele-Aware gRNA Off-Target & Specificity/Coverage Pareto (#3)

TEs are repetitive, so off-targets ARE the central design problem: a guide that
knocks out the whole family also fires at every paralogous copy. Generic CRISPR
tools score one locus at a time and miss this. This module scores each candidate
guide against the FULL TE copy landscape (every copy in --input) plus an optional
--background FASTA (other TE families / genomic regions), then reports the
specificity-vs-coverage Pareto frontier so you can pick guides that cover the
family with the fewest off-target sites.

Pipeline:
  1. Load copies from --input CSV (Seq column required).
  2. Build a majority-rule consensus; extract PAM-adjacent candidate guides from it.
  3. Index all family copies (both strands) → mismatch-tolerant COVERAGE.
  4. Index --background FASTA (if given) → OFF-TARGET hits. Without background,
     off-target = paralog multiplicity (extra family hits beyond one per copy).
  5. Score guides: coverage, off-target, on-target quality, specificity ratio.
  6. Compute the coverage-vs-off-target Pareto frontier.
  7. Figures:
       fig_grna_offtarget_pareto.pdf  --- coverage vs off-target scatter + frontier
       fig_grna_offtarget_top.pdf     --- top Pareto guides (coverage / off-target)
       fig_grna_offtarget_mm.pdf      --- coverage growth vs mismatch tolerance
  8. Write grna_offtarget_measured_values.tex + grna_offtarget_report.txt + CSV.

Usage:
    python run_grna_offtarget.py --input copies.csv --reports-dir ./reports \\
        [--family LTR5_Hs] [--cas SpCas9] [--grna-len 20] [--max-mm 2] \\
        [--background other_families.fa] [--top-n 25]
"""

import argparse
import datetime
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

BASES = "ACGT"
_PAM_TYPES = {
    "SpCas9": ("NGG", 3), "SaCas9": ("NNGRRT", 6),
    "Cas12a": ("TTTV", 4), "SpRY": ("NRN", 3),
}
_IUPAC = {"N": "ACGT", "R": "AG", "Y": "CT", "S": "GC", "W": "AT", "K": "GT",
          "M": "AC", "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG",
          "A": "A", "C": "C", "G": "G", "T": "T"}


def _pp(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _texesc(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def _rc(seq: str) -> str:
    return seq.upper().translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def _matches_pam(seq_p, pam_pattern):
    if len(seq_p) != len(pam_pattern):
        return False
    return all(s in _IUPAC.get(p, p) for s, p in zip(seq_p, pam_pattern))


def consensus_of(seqs: list) -> str:
    if not seqs:
        return ""
    L = max(len(s) for s in seqs)
    out = []
    for i in range(L):
        counts = {}
        for s in seqs:
            if i < len(s) and s[i] in BASES:
                counts[s[i]] = counts.get(s[i], 0) + 1
        out.append(max(counts, key=counts.get) if counts else "N")
    return "".join(out)


def extract_pam_grnas(seq, grna_len=20, cas="SpCas9"):
    """Return list of guide protospacer strings (both strands) adjacent to PAMs."""
    seq = seq.upper()
    L = len(seq)
    pam_pattern, pam_len = _PAM_TYPES.get(cas, ("NGG", 3))
    out = []
    if cas in ("SpCas9", "SpRY", "SaCas9"):
        for i in range(L - grna_len - pam_len + 1):
            pam = seq[i + grna_len: i + grna_len + pam_len]
            if _matches_pam(pam, pam_pattern):
                g = seq[i: i + grna_len]
                if "N" not in g:
                    out.append(g)
        for i in range(pam_len, L - grna_len + 1):
            if seq[i - pam_len] == "C" and seq[i - pam_len + 1] == "C":
                proto = seq[i: i + grna_len]
                if len(proto) == grna_len and "N" not in proto:
                    out.append(_rc(proto))
    elif cas == "Cas12a":
        for i in range(pam_len, L - grna_len + 1):
            if _matches_pam(seq[i - pam_len: i], pam_pattern):
                g = seq[i: i + grna_len]
                if "N" not in g:
                    out.append(g)
    return out


def on_target_score(g: str) -> float:
    n = len(g)
    s = 1.0
    gc = (g.count("G") + g.count("C")) / n
    if gc < 0.25 or gc > 0.80:   s *= 0.5
    elif gc < 0.40 or gc > 0.70: s *= 0.8
    if any(b * 4 in g for b in BASES): s *= 0.6
    if g[0] == "T": s *= 0.85
    return round(s, 4)


def build_kmer_index(seqs, grna_len=20, progress_label="copies"):
    """{kmer: set(seq_index)} over both strands."""
    idx = defaultdict(set)
    n = len(seqs)
    for si, seq in enumerate(seqs):
        if n > 400 and (si + 1) % 400 == 0:
            _pp(f"    Indexing {progress_label} {si+1}/{n}...")
        seq = seq.upper()
        seen = set()
        for i in range(len(seq) - grna_len + 1):
            km = seq[i: i + grna_len]
            if "N" in km:
                continue
            if km not in seen:
                idx[km].add(si); seen.add(km)
            kr = _rc(km)
            if kr not in seen:
                idx[kr].add(si); seen.add(kr)
    return idx


def _variants(g, d):
    """All Hamming<=d variants (d in {0,1,2}); seed (last 12) kept exact for d>=1."""
    yield g
    if d < 1:
        return
    seed = g[-12:]
    pos1 = [(i, b) for i in range(len(g) - 12) for b in BASES if b != g[i]]
    for i, b in pos1:
        yield g[:i] + b + g[i+1:]
    if d < 2:
        return
    for a in range(len(g) - 12):
        for b1 in BASES:
            if b1 == g[a]:
                continue
            g1 = g[:a] + b1 + g[a+1:]
            for c in range(a + 1, len(g) - 12):
                for b2 in BASES:
                    if b2 == g1[c]:
                        continue
                    yield g1[:c] + b2 + g1[c+1:]
    _ = seed  # seed held exact by construction (we never mutate last 12)


def hits(g, idx, max_mm):
    covered = set()
    for v in _variants(g, max_mm):
        covered |= idx.get(v, set())
    return covered


def read_fasta_seqs(path: Path) -> list:
    seqs, buf = [], []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if buf:
                seqs.append("".join(buf)); buf = []
        else:
            buf.append(line.strip())
    if buf:
        seqs.append("".join(buf))
    return seqs


def pareto_frontier(df, x="coverage", y="offtarget"):
    """Frontier maximizing x, minimizing y."""
    pts = df.sort_values([x, y], ascending=[False, True])
    frontier, best_y = [], float("inf")
    for _, r in pts.iterrows():
        if r[y] < best_y:
            frontier.append(r.name)
            best_y = r[y]
    return set(frontier)


# ── figures ──────────────────────────────────────────────────────────────────

def fig_pareto(df, frontier_idx, family, n_copies, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    on = df.index.isin(frontier_idx)
    ax.scatter(df.loc[~on, "coverage"], df.loc[~on, "offtarget"],
               s=18, c="#bdc3c7", alpha=0.6, label="candidate guides")
    fr = df.loc[df.index.isin(frontier_idx)].sort_values("coverage")
    ax.plot(fr["coverage"], fr["offtarget"], "-o", color="#c0392b",
            ms=6, lw=1.8, label="Pareto frontier")
    ax.set_xlabel(f"Family coverage (copies hit of {n_copies})")
    ax.set_ylabel("Off-target sites")
    ax.set_title(f"{family} --- gRNA specificity vs coverage", fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(out_path, bbox_inches="tight"); plt.close()
    _pp(f"  Saved {out_path}")


def fig_top(df, frontier_idx, family, out_path):
    fr = df.loc[df.index.isin(frontier_idx)].sort_values(
        ["coverage", "offtarget"], ascending=[False, True]).head(15).iloc[::-1]
    if fr.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, max(2.5, 0.45 * len(fr))))
    axes[0].barh(fr["guide"], fr["coverage"], color="#27ae60", edgecolor="white")
    axes[0].set_xlabel("copies covered"); axes[0].set_title("Coverage", fontweight="bold")
    axes[1].barh(fr["guide"], fr["offtarget"], color="#e67e22", edgecolor="white")
    axes[1].set_xlabel("off-target sites"); axes[1].set_title("Off-target", fontweight="bold")
    axes[1].set_yticklabels([])
    for ax in axes:
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{family} --- top Pareto-optimal guides", fontweight="bold", y=1.02)
    plt.tight_layout(); plt.savefig(out_path, bbox_inches="tight"); plt.close()
    _pp(f"  Saved {out_path}")


def fig_mm(cov_by_mm, family, n_copies, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for mm, covs in sorted(cov_by_mm.items()):
        if covs:
            ax.hist(covs, bins=min(20, n_copies + 1),
                    alpha=0.55, label=f"≤{mm} mismatch", edgecolor="white")
    ax.set_xlabel(f"family coverage per guide (of {n_copies})")
    ax.set_ylabel("guides")
    ax.set_title(f"{family} --- coverage growth with mismatch tolerance",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(out_path, bbox_inches="tight"); plt.close()
    _pp(f"  Saved {out_path}")


def write_values(res, reports):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tex = [
        "% Auto-generated by run_grna_offtarget.py", f"% {ts}", "",
        rf"\providecommand{{\grnaOTFamily}}{{{_texesc(res['family'])}}}",
        rf"\providecommand{{\grnaOTCas}}{{{_texesc(res['cas'])}}}",
        rf"\providecommand{{\grnaOTNCopies}}{{{res['n_copies']}}}",
        rf"\providecommand{{\grnaOTNGuides}}{{{res['n_guides']}}}",
        rf"\providecommand{{\grnaOTNFrontier}}{{{res['n_frontier']}}}",
        rf"\providecommand{{\grnaOTBestGuide}}{{\texttt{{{_texesc(res['best_guide'])}}}}}",
        rf"\providecommand{{\grnaOTBestCov}}{{{res['best_cov']}}}",
        rf"\providecommand{{\grnaOTBestOff}}{{{res['best_off']}}}",
        rf"\providecommand{{\grnaOTBestCovPct}}{{{res['best_cov_pct']:.1f}}}",
        "",
    ]
    (reports / "grna_offtarget_measured_values.tex").write_text("\n".join(tex))
    txt = [
        "=" * 60, "GAMECA gRNA Off-Target / Specificity-Coverage --- Measured Values",
        f"Generated: {ts}", "=" * 60,
        f"  Family:               {res['family']}",
        f"  Cas / PAM:            {res['cas']}",
        f"  Family copies:        {res['n_copies']}",
        f"  Candidate guides:     {res['n_guides']}",
        f"  Pareto-frontier:      {res['n_frontier']}",
        "",
        "  ── Best coverage/specificity guide ──",
        f"  Guide:                {res['best_guide']}",
        f"  Coverage:             {res['best_cov']} / {res['n_copies']} ({res['best_cov_pct']:.1f}%)",
        f"  Off-target sites:     {res['best_off']}",
    ]
    (reports / "grna_offtarget_report.txt").write_text("\n".join(txt))
    _pp("  Written grna_offtarget_measured_values.tex and grna_offtarget_report.txt")


def parse_args():
    p = argparse.ArgumentParser(
        description="Allele-aware gRNA off-target + specificity/coverage Pareto",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--input", required=True, help="CSV with a Seq column")
    p.add_argument("--reports-dir", default="./reports")
    p.add_argument("--family", default="TE")
    p.add_argument("--cas", default="SpCas9", choices=list(_PAM_TYPES))
    p.add_argument("--grna-len", type=int, default=20)
    p.add_argument("--max-mm", type=int, default=2, choices=[0, 1, 2])
    p.add_argument("--background", default="",
                   help="FASTA of off-target background (other families/genome)")
    p.add_argument("--max-guides", type=int, default=400,
                   help="Cap guides scored (taken from consensus)")
    p.add_argument("--top-n", type=int, default=25)
    return p.parse_args()


def main():
    args = parse_args()
    reports = Path(args.reports_dir); reports.mkdir(parents=True, exist_ok=True)
    _pp("=" * 60)
    _pp(f"GAMECA gRNA Off-Target --- {args.family} ({args.cas})")
    _pp("=" * 60)

    df = pd.read_csv(args.input, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    if "Seq" not in df.columns:
        _pp("ERROR: CSV must have a 'Seq' column."); sys.exit(1)
    copies = [str(s).upper() for s in df["Seq"] if isinstance(s, str) and s]
    n_copies = len(copies)
    _pp(f"  {n_copies} family copies")

    cons = consensus_of(copies)
    cand = list(dict.fromkeys(extract_pam_grnas(cons, args.grna_len, args.cas)))
    if len(cand) < 5:  # consensus too degenerate → pull from copies too
        for s in copies[:50]:
            cand.extend(extract_pam_grnas(s, args.grna_len, args.cas))
        cand = list(dict.fromkeys(cand))
    cand = cand[: args.max_guides]
    _pp(f"  {len(cand)} candidate guides from consensus/copies")
    if not cand:
        _pp("  No PAM-adjacent guides found."); sys.exit(0)

    _pp("Indexing family copy landscape...")
    fam_idx = build_kmer_index(copies, args.grna_len, "family")

    bg_idx, bg_n = None, 0
    if args.background and Path(args.background).exists():
        bg_seqs = read_fasta_seqs(Path(args.background))
        bg_n = len(bg_seqs)
        _pp(f"Indexing background ({bg_n} seqs)...")
        bg_idx = build_kmer_index(bg_seqs, args.grna_len, "background")
    else:
        if args.background:
            _pp(f"  WARNING: background '{args.background}' not found --- "
                "off-target = within-family paralog multiplicity.")
        else:
            _pp("  No --background given --- off-target = within-family paralog multiplicity.")

    _pp(f"Scoring {len(cand)} guides (max_mm={args.max_mm})...")
    rows, cov_by_mm = [], {m: [] for m in range(args.max_mm + 1)}
    for g in cand:
        cov = hits(g, fam_idx, args.max_mm)
        coverage = len(cov)
        for m in range(args.max_mm + 1):
            cov_by_mm[m].append(len(hits(g, fam_idx, m)))
        if bg_idx is not None:
            off = len(hits(g, bg_idx, args.max_mm))
        else:
            # paralog burden: total mismatch-tolerant family hits beyond exact 1/copy
            exact = len(hits(g, fam_idx, 0))
            off = max(0, coverage - exact)
        rows.append({"guide": g, "coverage": coverage, "offtarget": off,
                     "cov_pct": 100.0 * coverage / n_copies if n_copies else 0.0,
                     "on_target": on_target_score(g),
                     "specificity": coverage / (coverage + off) if (coverage + off) else 0.0})
    res_df = pd.DataFrame(rows).drop_duplicates("guide").reset_index(drop=True)
    frontier = pareto_frontier(res_df, "coverage", "offtarget")
    res_df["pareto"] = res_df.index.isin(frontier)
    res_df.sort_values(["pareto", "coverage", "offtarget"],
                       ascending=[False, False, True], inplace=True)
    res_df.to_csv(reports / "grna_offtarget_guides.csv", index=False)
    _pp(f"  Wrote {reports/'grna_offtarget_guides.csv'} "
        f"({len(frontier)} on Pareto frontier)")

    fig_pareto(res_df, frontier, args.family, n_copies,
               reports / "fig_grna_offtarget_pareto.pdf")
    fig_top(res_df, frontier, args.family, reports / "fig_grna_offtarget_top.pdf")
    fig_mm(cov_by_mm, args.family, n_copies, reports / "fig_grna_offtarget_mm.pdf")

    best = res_df.sort_values(["specificity", "coverage"],
                              ascending=[False, False]).iloc[0]
    write_values({
        "family": args.family, "cas": args.cas, "n_copies": n_copies,
        "n_guides": len(res_df), "n_frontier": len(frontier),
        "best_guide": best["guide"], "best_cov": int(best["coverage"]),
        "best_off": int(best["offtarget"]), "best_cov_pct": float(best["cov_pct"]),
    }, reports)

    _pp("=" * 60); _pp("DONE")
    print("\n" + "=" * 60)
    print("MEASURED VALUES (paste into chat):")
    print("=" * 60)
    print((reports / "grna_offtarget_report.txt").read_text())


if __name__ == "__main__":
    main()
