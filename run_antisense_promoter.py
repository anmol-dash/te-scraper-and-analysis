#!/usr/bin/env python3
"""
run_antisense_promoter.py --- GAMECA Antisense / Bidirectional Promoter Activity (#13)

The L1 5'UTR and many LTRs drive transcription from BOTH strands; antisense
promoter activity seeds chimeric transcripts and host-gene regulation. This
module scores bidirectional promoter potential per copy from sequence and,
when stranded expression is provided, from sense-vs-antisense signal.

Pipeline:
  1. Load copies from --input CSV (Seq required; strand optional).
  2. Take the 5' region (--promoter-bp) of each copy (oriented by strand).
  3. Scan BOTH strands for core promoter elements (TATA box, Initiator,
     and a CpG-richness proxy) → a bidirectionality score per copy.
  4. If stranded expression columns are present (auto-detected by *_plus/_minus,
     *_fwd/_rev, *_sense/_antisense, or *_pos/_neg suffixes), compute the
     antisense:sense expression ratio per copy.
  5. Figures:
       fig_antisense_motifs.pdf  --- sense vs antisense promoter-motif counts
       fig_antisense_expr.pdf    --- antisense:sense expression (or bidir. score)
  6. Write antisense_measured_values.tex + antisense_report.txt + CSV.

Usage:
    python run_antisense_promoter.py --input copies.csv --reports-dir ./reports \\
        [--family L1HS] [--promoter-bp 200]
"""

import argparse
import datetime
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_TATA = re.compile(r"TATA[AT]A[AT]")
_INR = re.compile(r"[CT][CT]A[ACGT][AT][CT][CT]")   # Initiator (loose Inr consensus)

_STRAND_SUFFIX_PAIRS = [
    ("_plus", "_minus"), ("_fwd", "_rev"), ("_sense", "_antisense"),
    ("_pos", "_neg"), (".plus", ".minus"), ("_p", "_m"),
]


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


def promoter_motif_counts(region):
    """(tata, inr, cpg_ratio) on the given strand orientation."""
    region = region.upper()
    tata = len(_TATA.findall(region))
    inr = len(_INR.findall(region))
    cg = region.count("CG")
    cpg_ratio = cg / max(1, len(region) - 1)
    return tata, inr, cpg_ratio


def detect_stranded_cols(cols):
    """Return list of (plus_col, minus_col) pairs sharing a base name."""
    pairs = []
    lower = {c.lower(): c for c in cols}
    for plus_sfx, minus_sfx in _STRAND_SUFFIX_PAIRS:
        for lc, orig in lower.items():
            if lc.endswith(plus_sfx):
                base = lc[: -len(plus_sfx)]
                partner = base + minus_sfx
                if partner in lower:
                    pairs.append((orig, lower[partner]))
    return pairs


def parse_args():
    p = argparse.ArgumentParser(
        description="Antisense / bidirectional promoter activity", epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--reports-dir", default="./reports")
    p.add_argument("--family", default="TE")
    p.add_argument("--promoter-bp", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    reports = Path(args.reports_dir); reports.mkdir(parents=True, exist_ok=True)
    _pp("=" * 60); _pp(f"GAMECA Antisense / Bidirectional Promoter --- {args.family}")
    _pp("=" * 60)

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

    sense_tata, sense_inr, anti_tata, anti_inr, cpg, bidir, labels = \
        [], [], [], [], [], [], []
    for i, r in df.iterrows():
        s = str(r["Seq"]).upper()
        if str(strand.iloc[i]) == "-":
            s = _rc(s)
        five = s[: args.promoter_bp]
        st, si, cg = promoter_motif_counts(five)              # sense
        at, ai, _ = promoter_motif_counts(_rc(five))          # antisense
        sense_tata.append(st); sense_inr.append(si)
        anti_tata.append(at); anti_inr.append(ai)
        cpg.append(cg)
        # bidirectionality: antisense motif strength relative to total
        tot = st + si + at + ai
        bidir.append((at + ai) / tot if tot else 0.0)
        labels.append(label(i, r))

    df_out = df.copy()
    df_out["label"] = labels
    df_out["sense_promoter_motifs"] = np.array(sense_tata) + np.array(sense_inr)
    df_out["antisense_promoter_motifs"] = np.array(anti_tata) + np.array(anti_inr)
    df_out["cpg_ratio"] = cpg
    df_out["bidirectionality"] = bidir

    # stranded expression if available
    stranded = detect_stranded_cols(df.columns)
    anti_ratio = None
    if stranded:
        _pp(f"  Stranded expression detected: {stranded}")
        plus_sum = sum(pd.to_numeric(df[p], errors="coerce").fillna(0) for p, _ in stranded)
        minus_sum = sum(pd.to_numeric(df[m], errors="coerce").fillna(0) for _, m in stranded)
        # antisense = strand opposite to element annotation
        sense_expr, antisense_expr = [], []
        for i in range(len(df)):
            if str(strand.iloc[i]) == "-":
                sense_expr.append(minus_sum.iloc[i]); antisense_expr.append(plus_sum.iloc[i])
            else:
                sense_expr.append(plus_sum.iloc[i]); antisense_expr.append(minus_sum.iloc[i])
        sense_expr = np.array(sense_expr); antisense_expr = np.array(antisense_expr)
        anti_ratio = antisense_expr / (sense_expr + antisense_expr + 1e-9)
        df_out["sense_expr"] = sense_expr
        df_out["antisense_expr"] = antisense_expr
        df_out["antisense_ratio"] = anti_ratio
    else:
        _pp("  No stranded expression columns --- using sequence bidirectionality only.")

    df_out.to_csv(reports / "antisense_per_copy.csv", index=False)
    _pp(f"  Wrote {reports/'antisense_per_copy.csv'}")

    # ── figures ──
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df_out))
    ax.bar(x - 0.2, df_out["sense_promoter_motifs"], width=0.4,
           label="sense", color="#2980b9", edgecolor="white")
    ax.bar(x + 0.2, df_out["antisense_promoter_motifs"], width=0.4,
           label="antisense", color="#e74c3c", edgecolor="white")
    ax.set_xlabel("copy"); ax.set_ylabel("promoter motifs (TATA + Inr)")
    ax.set_title(f"{args.family} --- sense vs antisense promoter motifs",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(reports / "fig_antisense_motifs.pdf", bbox_inches="tight"); plt.close()
    _pp(f"  Saved {reports/'fig_antisense_motifs.pdf'}")

    fig, ax = plt.subplots(figsize=(7, 5))
    if anti_ratio is not None:
        ax.hist(anti_ratio, bins=min(20, max(3, len(df_out) // 2)),
                color="#9b59b6", edgecolor="white")
        ax.axvline(0.5, color="black", ls="--", label="balanced (0.5)")
        ax.set_xlabel("antisense fraction of expression")
        ax.set_title(f"{args.family} --- antisense:sense expression", fontweight="bold")
        ax.legend(fontsize=9)
    else:
        ax.hist(bidir, bins=min(20, max(3, len(df_out) // 2)),
                color="#9b59b6", edgecolor="white")
        ax.set_xlabel("sequence bidirectionality score")
        ax.set_title(f"{args.family} --- promoter bidirectionality", fontweight="bold")
    ax.set_ylabel("copies"); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(reports / "fig_antisense_expr.pdf", bbox_inches="tight"); plt.close()
    _pp(f"  Saved {reports/'fig_antisense_expr.pdf'}")

    # ── measured values ──
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_bidir = int(np.sum(np.array(bidir) >= 0.4))
    median_anti = float(np.median(anti_ratio)) if anti_ratio is not None else float("nan")
    tex = [
        "% Auto-generated by run_antisense_promoter.py", f"% {ts}", "",
        rf"\providecommand{{\antiFamily}}{{{_texesc(args.family)}}}",
        rf"\providecommand{{\antiNCopies}}{{{len(df_out)}}}",
        rf"\providecommand{{\antiNBidir}}{{{n_bidir}}}",
        rf"\providecommand{{\antiMedBidir}}{{{float(np.median(bidir)):.3f}}}",
        rf"\providecommand{{\antiHasStranded}}{{{'yes' if anti_ratio is not None else 'no'}}}",
        (rf"\providecommand{{\antiMedAntisenseFrac}}{{{median_anti:.3f}}}"
         if anti_ratio is not None else r"\providecommand{\antiMedAntisenseFrac}{---}"),
        "",
    ]
    (reports / "antisense_measured_values.tex").write_text("\n".join(tex))
    txt = [
        "=" * 60, "GAMECA Antisense / Bidirectional Promoter --- Measured Values",
        f"Generated: {ts}", "=" * 60,
        f"  Family:                  {args.family}",
        f"  Copies:                  {len(df_out)}",
        f"  Bidirectional (>=0.4):   {n_bidir}",
        f"  Median bidir. score:     {float(np.median(bidir)):.3f}",
        f"  Stranded expression:     {'yes' if anti_ratio is not None else 'no'}",
    ]
    if anti_ratio is not None:
        txt.append(f"  Median antisense frac:   {median_anti:.3f}")
    (reports / "antisense_report.txt").write_text("\n".join(txt))
    _pp("  Written antisense_measured_values.tex and antisense_report.txt")

    _pp("=" * 60); _pp("DONE")
    print("\n" + "=" * 60); print("MEASURED VALUES (paste into chat):"); print("=" * 60)
    print((reports / "antisense_report.txt").read_text())


if __name__ == "__main__":
    main()
