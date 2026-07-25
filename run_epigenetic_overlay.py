#!/usr/bin/env python3
"""
run_epigenetic_overlay.py --- GAMECA Epigenetic / Regulatory Context Overlay (#4)

Annotate each TE copy with chromatin / regulatory marks (ENCODE/Roadmap chromatin
state, ATAC, H3K9me3, CpG methylation, etc.) to separate silenced from
derepressed/exapted copies --- directly complementing the SQuIRE expression module.

Inputs / degradation:
  --tracks NAME=track.bed ...   one or more annotation BED files (needs bedtools)
  If no tracks are given (or bedtools/files missing) the analysis reports zero
  tracks honestly and emits a figure explaining how to supply them --- it never
  fabricates overlap numbers.

Outputs:
  fig_epigenetic_overlap.pdf   % of copies overlapping each mark
  fig_epigenetic_heatmap.pdf   copy × mark overlap heatmap
  epigenetic_measured_values.tex + epigenetic_report.txt + epigenetic_per_copy.csv

Usage:
    python run_epigenetic_overlay.py --input copies.csv --reports-dir ./reports \\
        --family LTR5_Hs \\
        --tracks H3K9me3=h3k9me3.bed ATAC=atac.bed CpG=cpg_islands.bed
"""

import argparse
import datetime
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import te_overlay as ov


_PRESET_CHOICES = ["K562", "GM12878", "HeLa-S3", "IMR-90", "H1-hESC"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Epigenetic / regulatory context overlay", epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--reports-dir", default="./reports")
    p.add_argument("--family", default="TE")
    p.add_argument("--tracks", nargs="*", default=[],
                   help="NAME=path.bed annotation tracks")
    p.add_argument("--preset", choices=_PRESET_CHOICES, metavar="CELL_TYPE",
                   help=(f"Auto-download H3K9me3 + ATAC + CpG tracks from ENCODE/UCSC "
                         f"for a preset cell type. Choices: {_PRESET_CHOICES}. "
                         "Explicit --tracks take precedence."))
    return p.parse_args()


def main():
    args = parse_args()
    reports = Path(args.reports_dir); reports.mkdir(parents=True, exist_ok=True)
    ov.pp("=" * 60); ov.pp(f"GAMECA Epigenetic Overlay --- {args.family}"); ov.pp("=" * 60)

    df = ov.load_loci(args.input)
    ov.pp(f"  {len(df)} loci")

    # Auto-download preset tracks if requested; explicit --tracks override
    preset_tracks: dict = {}
    if args.preset:
        preset_tracks = ov.fetch_epigenetic_preset(args.preset)
    explicit_tracks = ov.parse_named_paths(args.tracks)
    tracks = {**preset_tracks, **explicit_tracks}   # explicit wins on name collision

    overlap_by_track = {}
    used = []
    if tracks and ov.have("bedtools") and ov.has_coords(df):
        with tempfile.TemporaryDirectory() as td:
            bed = Path(td) / "loci.bed"; ov.write_bed(df, bed)
            for name, path in tracks.items():
                if not Path(path).exists():
                    ov.pp(f"  WARNING: track '{name}' file missing: {path}")
                    continue
                counts = ov.bedtools_intersect_count(bed, path)
                if not counts:
                    # bedtools returned nothing at all: the intersect failed or the
                    # track shares no contigs with the loci. That is "unknown", not
                    # "zero overlap" — recording a 0 column would put a measured-
                    # looking 0% into the report for a track that never ran.
                    ov.pp(f"  WARNING: track '{name}' produced no rows "
                          f"(intersect failed or no shared contigs) --- reporting n/a")
                    continue
                overlap_by_track[name] = counts
                used.append(name)
                n_hit = sum(1 for v in counts.values() if v > 0)
                ov.pp(f"  {name}: {n_hit}/{len(df)} copies overlap")
    else:
        if not tracks:
            ov.pp("  No --tracks provided. Use --preset K562 (or GM12878/HeLa-S3/IMR-90/H1-hESC) "
                  "to auto-download ENCODE tracks, or pass --tracks NAME=path.bed.")
        elif not ov.have("bedtools"):
            ov.pp("  WARNING: bedtools not found --- cannot compute overlaps.")
        elif not ov.has_coords(df):
            ov.pp("  WARNING: CSV lacks chr/start/stop --- cannot compute overlaps.")

    df_out = df.copy()
    for name in used:
        df_out[f"overlap_{name}"] = [overlap_by_track[name].get(n, 0) for n in df["name"]]
    df_out.to_csv(reports / "epigenetic_per_copy.csv", index=False)
    ov.pp(f"  Wrote {reports/'epigenetic_per_copy.csv'}")

    # ── figures ──
    fig, ax = plt.subplots(figsize=(8, 5))
    if used:
        pct = [100.0 * sum(1 for v in overlap_by_track[n].values() if v > 0) / len(df)
               for n in used]
        ax.bar(used, pct, color="#16a085", edgecolor="white")
        for i, v in enumerate(pct):
            ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
        ax.set_ylabel("% copies overlapping"); ax.set_ylim(0, 105)
        ax.set_title(f"{args.family} --- epigenetic mark overlap", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "no annotation tracks supplied\n"
                "(--tracks NAME=track.bed ...)",
                ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title(f"{args.family} --- epigenetic mark overlap", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(reports / "fig_epigenetic_overlap.pdf", bbox_inches="tight"); plt.close()
    ov.pp(f"  Saved {reports/'fig_epigenetic_overlap.pdf'}")

    fig, ax = plt.subplots(figsize=(8, 6))
    if used:
        mat = np.array([[1 if overlap_by_track[n].get(nm, 0) > 0 else 0
                         for n in used] for nm in df["name"]])
        im = ax.imshow(mat, aspect="auto", cmap="Greens", interpolation="nearest")
        ax.set_xticks(range(len(used))); ax.set_xticklabels(used, rotation=45, ha="right")
        ax.set_ylabel("copy"); ax.set_title(f"{args.family} --- copy × mark", fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.7, label="overlap")
    else:
        ax.text(0.5, 0.5, "no annotation tracks supplied",
                ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title(f"{args.family} --- copy × mark", fontweight="bold")
    plt.tight_layout()
    plt.savefig(reports / "fig_epigenetic_heatmap.pdf", bbox_inches="tight"); plt.close()
    ov.pp(f"  Saved {reports/'fig_epigenetic_heatmap.pdf'}")

    # ── measured values ──
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tex = ["% Auto-generated by run_epigenetic_overlay.py", f"% {ts}", "",
           rf"\providecommand{{\epiFamily}}{{{ov.texesc(args.family)}}}",
           rf"\providecommand{{\epiNCopies}}{{{len(df)}}}",
           rf"\providecommand{{\epiNTracks}}{{{len(used)}}}",
           rf"\providecommand{{\epiTrackList}}{{{ov.texesc(', '.join(used)) if used else '---'}}}", ""]
    (reports / "epigenetic_measured_values.tex").write_text("\n".join(tex))
    txt = ["=" * 60, "GAMECA Epigenetic Overlay --- Measured Values", f"Generated: {ts}",
           "=" * 60, f"  Family:        {args.family}", f"  Copies:        {len(df)}",
           f"  Tracks used:   {len(used)}  ({', '.join(used) if used else 'none'})"]
    for n in used:
        n_hit = sum(1 for v in overlap_by_track[n].values() if v > 0)
        txt.append(f"    {n:<12} {n_hit}/{len(df)} copies ({100*n_hit/len(df):.0f}%)")
    unusable = [n for n in tracks if n not in used]
    if unusable:
        txt.append(f"  Tracks requested but unusable (reported as n/a, NOT as 0%):")
        for n in unusable:
            txt.append(f"    {n:<12} n/a")
    (reports / "epigenetic_report.txt").write_text("\n".join(txt))
    ov.pp("  Written epigenetic_measured_values.tex and epigenetic_report.txt")
    ov.pp("=" * 60); ov.pp("DONE")
    print("\n" + "=" * 60); print("MEASURED VALUES (paste into chat):"); print("=" * 60)
    print((reports / "epigenetic_report.txt").read_text())


if __name__ == "__main__":
    main()
