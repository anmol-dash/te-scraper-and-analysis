#!/usr/bin/env python3
"""
run_ctcf_tad.py --- GAMECA CTCF-site & TAD-boundary TE Analysis (#12)

Many TE families donate CTCF binding sites and cluster at TAD boundaries / loop
anchors (e.g. B2 SINEs, MIRs, some ERVs) --- a 3D-genome angle most TE tools ignore.
This module flags copies that (a) overlap CTCF ChIP peaks and/or carry the CTCF
core motif, and (b) sit near TAD boundaries / loop anchors.

Inputs / degradation:
  --ctcf PEAKS.bed   CTCF ChIP-seq peaks  (authoritative; needs bedtools)
  --tads BOUND.bed   TAD boundaries / loop anchors (needs bedtools)
  If --ctcf is absent, a sequence-based CTCF core-motif scan runs instead
  (heuristic, both strands) so the analysis still produces an estimate locally.
  If --tads is absent, the TAD-distance panel is annotated as "no boundary file".

Outputs:
  fig_ctcf_overlap.pdf       CTCF peak overlap + motif presence
  fig_ctcf_tad_distance.pdf  distance to nearest TAD boundary
  ctcf_tad_measured_values.tex + ctcf_tad_report.txt + ctcf_tad_per_copy.csv

Usage:
    python run_ctcf_tad.py --input copies.csv --reports-dir ./reports \\
        [--family B2_Mm] [--ctcf ctcf_peaks.bed] [--tads tad_boundaries.bed] \\
        [--motif-mismatch 3] [--boundary-window 50000]
"""

import argparse
import datetime
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import te_overlay as ov

# CTCF core consensus (degenerate); N = wildcard. Scanned with mismatch tolerance.
_CTCF_CORE = "CCGCGNGGNGGCAG"


def _rc(seq):
    return seq.upper().translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def _core_hit(seq, mismatch):
    """True if the CTCF core motif occurs (either strand) within `mismatch` subs."""
    core = _CTCF_CORE
    L = len(core)
    for s in (seq.upper(), _rc(seq)):
        for i in range(len(s) - L + 1):
            mm = sum(1 for a, b in zip(s[i:i+L], core)
                     if b != "N" and a != b)
            if mm <= mismatch:
                return True
    return False


_CTCF_PRESETS  = ["K562", "GM12878", "HeLa-S3"]
_TAD_PRESETS   = ["K562", "GM12878", "IMR90"]


def parse_args():
    p = argparse.ArgumentParser(
        description="CTCF-site and TAD-boundary TE analysis", epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--reports-dir", default="./reports")
    p.add_argument("--family", default="TE")
    p.add_argument("--ctcf", default="", help="CTCF ChIP-seq peaks BED (explicit path)")
    p.add_argument("--ctcf-preset", choices=_CTCF_PRESETS, metavar="CELL_TYPE",
                   help=(f"Auto-download CTCF ChIP peaks from ENCODE. "
                         f"Choices: {_CTCF_PRESETS}. Ignored if --ctcf is given."))
    p.add_argument("--tads", default="", help="TAD boundaries BED (explicit path)")
    p.add_argument("--tads-preset", choices=_TAD_PRESETS, metavar="CELL_TYPE",
                   help=(f"Auto-download Rao 2014 TAD boundaries from GEO (hg19 coords). "
                         f"Choices: {_TAD_PRESETS}. Ignored if --tads is given."))
    p.add_argument("--motif-mismatch", type=int, default=3)
    p.add_argument("--boundary-window", type=int, default=50000,
                   help="bp window to call a copy 'boundary-proximal'")
    return p.parse_args()


def main():
    args = parse_args()
    reports = Path(args.reports_dir); reports.mkdir(parents=True, exist_ok=True)
    ov.pp("=" * 60); ov.pp(f"GAMECA CTCF / TAD --- {args.family}"); ov.pp("=" * 60)

    df = ov.load_loci(args.input)
    ov.pp(f"  {len(df)} loci")

    # ── resolve preset paths (only used when explicit path not given) ──
    if not args.ctcf and args.ctcf_preset:
        path = ov.fetch_ctcf_preset(args.ctcf_preset)
        if path:
            args.ctcf = str(path)
    if not args.tads and args.tads_preset:
        path = ov.fetch_tad_preset(args.tads_preset)
        if path:
            args.tads = str(path)

    # ── CTCF: ChIP overlap if provided, else motif scan ──
    chip_overlap = {}
    used_chip = False
    if args.ctcf:
        if ov.have("bedtools") and Path(args.ctcf).exists():
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                bed = Path(td) / "loci.bed"
                if ov.has_coords(df):
                    ov.write_bed(df, bed)
                    chip_overlap = ov.bedtools_intersect_count(bed, args.ctcf)
                    used_chip = True
                    ov.pp(f"  CTCF ChIP overlap via bedtools ({len(chip_overlap)} loci scored)")
                else:
                    ov.pp("  WARNING: no coordinates in CSV --- cannot overlap CTCF peaks.")
        else:
            ov.pp(f"  WARNING: --ctcf given but bedtools/file unavailable "
                  f"(bedtools={ov.have('bedtools')}, exists={Path(args.ctcf).exists()}).")

    motif_hit = []
    if "Seq" in df.columns:
        motif_hit = [bool(_core_hit(str(s), args.motif_mismatch)) if isinstance(s, str) else False
                     for s in df["Seq"]]
        ov.pp(f"  CTCF core-motif scan: {int(np.sum(motif_hit))}/{len(df)} copies hit "
              f"(<= {args.motif_mismatch} mismatch)")
    else:
        ov.pp("  No Seq column --- skipping motif scan.")
        motif_hit = [False] * len(df)

    df_out = df.copy()
    df_out["ctcf_chip_overlap"] = [chip_overlap.get(n, 0) for n in df["name"]] if used_chip else np.nan
    df_out["ctcf_motif"] = motif_hit

    # ── TAD distance ──
    tad_dist = {}
    used_tads = False
    if args.tads:
        if ov.have("bedtools") and Path(args.tads).exists() and ov.has_coords(df):
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                bed = Path(td) / "loci.bed"; ov.write_bed(df, bed)
                tad_dist = ov.bedtools_closest_distance(bed, args.tads)
                used_tads = bool(tad_dist)
                ov.pp(f"  TAD distance via bedtools ({len(tad_dist)} loci)")
        else:
            ov.pp(f"  WARNING: --tads given but bedtools/file/coords unavailable.")
    df_out["tad_distance_bp"] = [tad_dist.get(n, np.nan) for n in df["name"]] if used_tads else np.nan
    if used_tads:
        df_out["boundary_proximal"] = df_out["tad_distance_bp"] <= args.boundary_window

    df_out.to_csv(reports / "ctcf_tad_per_copy.csv", index=False)
    ov.pp(f"  Wrote {reports/'ctcf_tad_per_copy.csv'}")

    # ── figures ──
    fig, ax = plt.subplots(figsize=(7, 5))
    n = len(df)
    motif_n = int(np.sum(motif_hit))
    bars = {"CTCF motif\n(seq)": motif_n}
    if used_chip:
        bars["CTCF ChIP\noverlap"] = int(np.sum(np.array(list(chip_overlap.values())) > 0))
    ax.bar(list(bars.keys()), list(bars.values()), color="#2c3e50", edgecolor="white")
    ax.axhline(n, color="#bdc3c7", ls="--", label=f"all copies ({n})")
    ax.set_ylabel("copies"); ax.legend(fontsize=8)
    ax.set_title(f"{args.family} --- CTCF site provision", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(reports / "fig_ctcf_overlap.pdf", bbox_inches="tight"); plt.close()
    ov.pp(f"  Saved {reports/'fig_ctcf_overlap.pdf'}")

    fig, ax = plt.subplots(figsize=(7, 5))
    if used_tads:
        d = np.array([v for v in tad_dist.values()], dtype=float) / 1000.0
        ax.hist(d, bins=min(30, max(5, len(d) // 2)), color="#e67e22", edgecolor="white")
        ax.axvline(args.boundary_window / 1000, color="black", ls="--",
                   label=f"{args.boundary_window/1000:.0f} kb window")
        ax.set_xlabel("distance to nearest TAD boundary (kb)")
        ax.set_ylabel("copies"); ax.legend(fontsize=8)
        ax.set_title(f"{args.family} --- TAD-boundary proximity", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "no TAD boundary file provided\n(pass --tads boundaries.bed)",
                ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title(f"{args.family} --- TAD-boundary proximity", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(reports / "fig_ctcf_tad_distance.pdf", bbox_inches="tight"); plt.close()
    ov.pp(f"  Saved {reports/'fig_ctcf_tad_distance.pdf'}")

    # ── measured values ──
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_chip = int(np.sum(np.array(list(chip_overlap.values())) > 0)) if used_chip else 0
    n_prox = int(df_out["boundary_proximal"].sum()) if used_tads else 0
    med_dist = float(np.median(list(tad_dist.values()))) if used_tads else float("nan")
    tex = [
        "% Auto-generated by run_ctcf_tad.py", f"% {ts}", "",
        rf"\providecommand{{\ctcfFamily}}{{{ov.texesc(args.family)}}}",
        rf"\providecommand{{\ctcfNCopies}}{{{n}}}",
        rf"\providecommand{{\ctcfNMotif}}{{{motif_n}}}",
        rf"\providecommand{{\ctcfUsedChip}}{{{'yes' if used_chip else 'no'}}}",
        rf"\providecommand{{\ctcfNChip}}{{{n_chip}}}",
        rf"\providecommand{{\ctcfUsedTads}}{{{'yes' if used_tads else 'no'}}}",
        rf"\providecommand{{\ctcfNBoundaryProx}}{{{n_prox}}}",
        (rf"\providecommand{{\ctcfMedTadDistKb}}{{{med_dist/1000:.1f}}}"
         if used_tads else r"\providecommand{\ctcfMedTadDistKb}{---}"),
        "",
    ]
    (reports / "ctcf_tad_measured_values.tex").write_text("\n".join(tex))
    txt = [
        "=" * 60, "GAMECA CTCF / TAD --- Measured Values", f"Generated: {ts}", "=" * 60,
        f"  Family:               {args.family}",
        f"  Copies:               {n}",
        f"  CTCF core motif:      {motif_n}",
        f"  CTCF ChIP overlap:    {n_chip if used_chip else 'n/a (no peaks file)'}",
        f"  Boundary-proximal:    {n_prox if used_tads else 'n/a (no TAD file)'}",
        f"  Median TAD distance:  {f'{med_dist/1000:.1f} kb' if used_tads else 'n/a'}",
    ]
    (reports / "ctcf_tad_report.txt").write_text("\n".join(txt))
    ov.pp("  Written ctcf_tad_measured_values.tex and ctcf_tad_report.txt")
    ov.pp("=" * 60); ov.pp("DONE")
    print("\n" + "=" * 60); print("MEASURED VALUES (paste into chat):"); print("=" * 60)
    print((reports / "ctcf_tad_report.txt").read_text())


if __name__ == "__main__":
    main()
