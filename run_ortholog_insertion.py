#!/usr/bin/env python3
"""
run_ortholog_insertion.py --- GAMECA Orthologous-Insertion Calling Across Species (#5)

Is a TE copy shared (ancestral, present before species split) or lineage-specific
(inserted after)? liftOver-based presence/absence across a set of target species
answers this per copy and reveals insertion timing.

Inputs / degradation:
  --chains SPECIES=chain.over.chain ...   UCSC liftOver chains to each target
  --liftover-cmd liftOver                 (needs the UCSC liftOver binary)
  If liftOver or the chains are missing, the analysis reports honestly that the
  ortholog call could not be made --- it never invents presence/absence.

A copy that lifts to a species → orthologous locus exists there (shared).
A copy that fails to lift  → no orthologous locus (lineage-specific in reference).

Outputs:
  fig_ortholog_presence.pdf  copy × species presence/absence matrix
  fig_ortholog_summary.pdf   shared vs lineage-specific counts per species
  ortholog_measured_values.tex + ortholog_report.txt + ortholog_per_copy.csv

Usage:
    python run_ortholog_insertion.py --input copies.csv --reports-dir ./reports \\
        --family LTR5_Hs \\
        --chains panTro6=hg38ToPanTro6.over.chain rheMac10=hg38ToRheMac10.over.chain
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


_SPECIES_PRESETS = {
    "panTro6":  "Chimpanzee",
    "gorGor6":  "Gorilla",
    "ponAbe3":  "Sumatran orangutan",
    "rheMac10": "Rhesus macaque",
    "mm39":     "Mouse (mm39)",
    "rn7":      "Rat (rn7)",
    "canFam6":  "Dog (canFam6)",
    "galGal6":  "Chicken (galGal6)",
    "bosTau9":  "Cow (bosTau9)",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Orthologous-insertion calling across species", epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--reports-dir", default="./reports")
    p.add_argument("--family", default="TE")
    p.add_argument("--source-assembly", default="hg38",
                   help=("Assembly the input coordinates are in. Chains are "
                         "fetched as <source>To<Species>.over.chain, so a mouse "
                         "family needs mm10/mm39 here or it silently lifts from "
                         "the wrong genome."))
    p.add_argument("--chains", nargs="*", default=[], help="SPECIES=chain ...")
    p.add_argument("--species", nargs="*", default=[],
                   metavar="SPECIES",
                   help=(f"Auto-download UCSC liftOver chains for named species "
                         f"(no chain files needed). "
                         f"Choices: {list(_SPECIES_PRESETS)}. "
                         "Combined with any explicit --chains."))
    p.add_argument("--liftover-cmd", default="liftOver")
    return p.parse_args()


def main():
    args = parse_args()
    reports = Path(args.reports_dir); reports.mkdir(parents=True, exist_ok=True)
    ov.pp("=" * 60); ov.pp(f"GAMECA Ortholog Insertion --- {args.family}"); ov.pp("=" * 60)

    df = ov.load_loci(args.input)
    ov.pp(f"  {len(df)} loci")

    # Auto-download chains for --species presets; merge with any explicit --chains
    auto_chains: dict = {}
    for sp in (args.species or []):
        if sp not in _SPECIES_PRESETS:
            ov.pp(f"  WARNING: unknown species '{sp}'. "
                  f"Choices: {list(_SPECIES_PRESETS)}")
            continue
        path = ov.fetch_chain(sp, src_asm=args.source_assembly)
        if path:
            auto_chains[sp] = str(path)
        else:
            ov.pp(f"  WARNING: chain download failed for {sp}; skipping")
    explicit_chains = ov.parse_named_paths(args.chains)
    chains = {**auto_chains, **explicit_chains}     # explicit wins on name collision

    presence = {}     # species -> set(mapped names)
    used = []
    if chains and ov.have(args.liftover_cmd) and ov.has_coords(df):
        with tempfile.TemporaryDirectory() as td:
            bed = Path(td) / "loci.bed"; ov.write_bed(df, bed)
            for sp, chain in chains.items():
                mapped, unmapped, ran = ov.liftover(bed, chain, args.liftover_cmd)
                if ran:
                    presence[sp] = mapped
                    used.append(sp)
                    ov.pp(f"  {sp}: {len(mapped)} shared / {len(unmapped)} lineage-specific")
                else:
                    ov.pp(f"  WARNING: liftOver to {sp} did not run (chain missing?)")
    else:
        if not chains:
            ov.pp("  No chains available. Use --species panTro6 rheMac10 mm39 "
                  "to auto-download, or pass --chains SPECIES=chain.")
        elif not ov.have(args.liftover_cmd):
            ov.pp(f"  WARNING: '{args.liftover_cmd}' not found --- install UCSC liftOver.")
        elif not ov.has_coords(df):
            ov.pp("  WARNING: CSV lacks chr/start/stop --- cannot liftOver.")

    df_out = df.copy()
    for sp in used:
        df_out[f"present_{sp}"] = [n in presence[sp] for n in df["name"]]
    if used:
        df_out["n_species_shared"] = df_out[[f"present_{sp}" for sp in used]].sum(axis=1)
        df_out["lineage_specific"] = df_out["n_species_shared"] == 0
    df_out.to_csv(reports / "ortholog_per_copy.csv", index=False)
    ov.pp(f"  Wrote {reports/'ortholog_per_copy.csv'}")

    # ── figures ──
    fig, ax = plt.subplots(figsize=(8, 6))
    if used:
        mat = np.array([[1 if n in presence[sp] else 0 for sp in used]
                        for n in df["name"]])
        im = ax.imshow(mat, aspect="auto", cmap="Blues", interpolation="nearest")
        ax.set_xticks(range(len(used))); ax.set_xticklabels(used, rotation=45, ha="right")
        ax.set_ylabel("copy"); ax.set_title(f"{args.family} --- ortholog presence",
                                            fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.7, label="present (1) / absent (0)")
    else:
        ax.text(0.5, 0.5, "no liftOver chains supplied\n(--chains SPECIES=chain ...)",
                ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title(f"{args.family} --- ortholog presence", fontweight="bold")
    plt.tight_layout()
    plt.savefig(reports / "fig_ortholog_presence.pdf", bbox_inches="tight"); plt.close()
    ov.pp(f"  Saved {reports/'fig_ortholog_presence.pdf'}")

    fig, ax = plt.subplots(figsize=(8, 5))
    if used:
        shared = [len(presence[sp]) for sp in used]
        specific = [len(df) - s for s in shared]
        x = np.arange(len(used))
        ax.bar(x, shared, label="shared (ancestral)", color="#2980b9", edgecolor="white")
        ax.bar(x, specific, bottom=shared, label="absent / lineage-specific",
               color="#e74c3c", edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(used, rotation=45, ha="right")
        ax.set_ylabel("copies"); ax.legend(fontsize=9)
        ax.set_title(f"{args.family} --- shared vs lineage-specific", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "no liftOver chains supplied",
                ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title(f"{args.family} --- shared vs lineage-specific", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(reports / "fig_ortholog_summary.pdf", bbox_inches="tight"); plt.close()
    ov.pp(f"  Saved {reports/'fig_ortholog_summary.pdf'}")

    # ── measured values ──
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_specific = int(df_out["lineage_specific"].sum()) if used else 0
    tex = ["% Auto-generated by run_ortholog_insertion.py", f"% {ts}", "",
           rf"\providecommand{{\orthoFamily}}{{{ov.texesc(args.family)}}}",
           rf"\providecommand{{\orthoNCopies}}{{{len(df)}}}",
           rf"\providecommand{{\orthoNSpecies}}{{{len(used)}}}",
           rf"\providecommand{{\orthoSpeciesList}}{{{ov.texesc(', '.join(used)) if used else '---'}}}",
           rf"\providecommand{{\orthoNLineageSpecific}}{{{n_specific if used else '---'}}}", ""]
    (reports / "ortholog_measured_values.tex").write_text("\n".join(tex))
    txt = ["=" * 60, "GAMECA Ortholog Insertion --- Measured Values", f"Generated: {ts}",
           "=" * 60, f"  Family:              {args.family}", f"  Copies:              {len(df)}",
           f"  Target species:      {len(used)}  ({', '.join(used) if used else 'none'})",
           f"  Lineage-specific:    {n_specific if used else 'n/a (no chains)'}"]
    for sp in used:
        txt.append(f"    {sp:<14} {len(presence[sp])}/{len(df)} shared")
    dropped = [sp for sp in (args.species or []) if sp not in used]
    if dropped:
        txt.append("  Requested but unavailable (reported as n/a, NOT as absent):")
        for sp in dropped:
            why = ("not a known species" if sp not in _SPECIES_PRESETS
                   else "chain download or liftOver failed")
            txt.append(f"    {sp:<14} n/a  ({why})")
    (reports / "ortholog_report.txt").write_text("\n".join(txt))
    ov.pp("  Written ortholog_measured_values.tex and ortholog_report.txt")
    ov.pp("=" * 60); ov.pp("DONE")
    print("\n" + "=" * 60); print("MEASURED VALUES (paste into chat):"); print("=" * 60)
    print((reports / "ortholog_report.txt").read_text())


if __name__ == "__main__":
    main()
