#!/usr/bin/env python3
"""
run_multiassembly_liftover.py --- GAMECA Multi-Assembly / Liftover Native (#7)

Run the same family across assemblies of the same species (e.g. hg38, T2T-CHM13,
mm10, mm39) and liftOver coordinates between them. T2T-CHM13 matters especially
because it resolves repeat-rich regions the others cannot, so a copy that maps to
T2T but not hg38 (or vice-versa) is informative about assembly quality.

Inputs / degradation:
  --chains ASM=chain.over.chain ...   liftOver chains from the source assembly
  --liftover-cmd liftOver
  If liftOver/chains are absent, mapping rates are reported as n/a (not faked).

Outputs:
  fig_multiassembly_mapping.pdf  per-assembly mapping rate
  fig_multiassembly_coords.pdf   mapped vs unmapped per assembly
  multiassembly_measured_values.tex + multiassembly_report.txt + multiassembly_per_copy.csv

Usage:
    python run_multiassembly_liftover.py --input copies.csv --reports-dir ./reports \\
        --family LTR5_Hs --source-assembly hg38 \\
        --chains t2t=hg38ToHs1.over.chain hg19=hg38ToHg19.over.chain
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


# Known UCSC target assemblies for --target-assemblies (allow-list guarding
# against typos before a chain download is attempted). Values are human-readable
# labels only; fetch_chain builds the actual mm10ToMm39-style chain name. Covers
# both human and mouse/rat so cross-species TE families (mm10/mm39) resolve
# chains instead of skipping — the chains exist at
# goldenPath/<src>/liftOver/<src>To<Target>.over.chain.gz.
_ASSEMBLY_PRESETS = {
    "hg38": "GRCh38",
    "hg19": "GRCh37",
    "t2t":  "T2T-CHM13 (hs1)",
    "hs1":  "T2T-CHM13 (hs1)",
    "mm39": "GRCm39",
    "mm10": "GRCm38",
    "rn7":  "mRatBN7.2",
    "rn6":  "RGSC 6.0",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-assembly liftover", epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--reports-dir", default="./reports")
    p.add_argument("--family", default="TE")
    p.add_argument("--source-assembly", default="hg38")
    p.add_argument("--chains", nargs="*", default=[], help="ASSEMBLY=chain ...")
    p.add_argument("--target-assemblies", nargs="*", default=[],
                   metavar="ASSEMBLY",
                   help=(f"Auto-download UCSC chains to named assemblies. "
                         f"Choices: {list(_ASSEMBLY_PRESETS)}. "
                         "Combined with any explicit --chains."))
    p.add_argument("--liftover-cmd", default="liftOver")
    return p.parse_args()


def main():
    args = parse_args()
    reports = Path(args.reports_dir); reports.mkdir(parents=True, exist_ok=True)
    ov.pp("=" * 60)
    ov.pp(f"GAMECA Multi-Assembly Liftover --- {args.family} (from {args.source_assembly})")
    ov.pp("=" * 60)

    df = ov.load_loci(args.input)
    ov.pp(f"  {len(df)} loci")

    # Auto-download chains for --target-assemblies; merge with any explicit --chains
    auto_chains: dict = {}
    for asm in (args.target_assemblies or []):
        if asm not in _ASSEMBLY_PRESETS:
            ov.pp(f"  WARNING: unknown assembly '{asm}'. "
                  f"Choices: {list(_ASSEMBLY_PRESETS)}")
            continue
        path = ov.fetch_chain(asm, src_asm=args.source_assembly)
        if path:
            # normalise alias to canonical name for display
            label = "hs1" if asm in ("t2t", "hs1") else asm
            auto_chains[label] = str(path)
        else:
            ov.pp(f"  WARNING: chain download failed for {asm}; skipping")
    explicit_chains = ov.parse_named_paths(args.chains)
    chains = {**auto_chains, **explicit_chains}

    mapped_by_asm = {}
    used = []
    if chains and ov.have(args.liftover_cmd) and ov.has_coords(df):
        with tempfile.TemporaryDirectory() as td:
            bed = Path(td) / "loci.bed"; ov.write_bed(df, bed)
            for asm, chain in chains.items():
                mapped, unmapped, ran = ov.liftover(bed, chain, args.liftover_cmd)
                if ran:
                    mapped_by_asm[asm] = mapped
                    used.append(asm)
                    rate = 100.0 * len(mapped) / len(df) if len(df) else 0.0
                    ov.pp(f"  {args.source_assembly}->{asm}: "
                          f"{len(mapped)}/{len(df)} mapped ({rate:.1f}%)")
    else:
        if not chains:
            ov.pp("  No chains available. Use --target-assemblies hg19 t2t "
                  "to auto-download, or pass --chains ASM=chain.")
        elif not ov.have(args.liftover_cmd):
            ov.pp(f"  WARNING: '{args.liftover_cmd}' not found --- install UCSC liftOver.")
        elif not ov.has_coords(df):
            ov.pp("  WARNING: CSV lacks chr/start/stop --- cannot liftOver.")

    df_out = df.copy()
    for asm in used:
        df_out[f"maps_{asm}"] = [n in mapped_by_asm[asm] for n in df["name"]]
    df_out.to_csv(reports / "multiassembly_per_copy.csv", index=False)
    ov.pp(f"  Wrote {reports/'multiassembly_per_copy.csv'}")

    # ── figures ──
    fig, ax = plt.subplots(figsize=(8, 5))
    if used:
        rates = [100.0 * len(mapped_by_asm[a]) / len(df) for a in used]
        ax.bar(used, rates, color="#34495e", edgecolor="white")
        for i, v in enumerate(rates):
            ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
        ax.set_ylim(0, 105); ax.set_ylabel("% copies mapped")
        ax.set_title(f"{args.family} --- liftOver mapping rate from {args.source_assembly}",
                     fontweight="bold")
    else:
        ax.text(0.5, 0.5, "no liftOver chains supplied\n(--chains ASM=chain ...)",
                ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title(f"{args.family} --- liftOver mapping rate", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(reports / "fig_multiassembly_mapping.pdf", bbox_inches="tight"); plt.close()
    ov.pp(f"  Saved {reports/'fig_multiassembly_mapping.pdf'}")

    fig, ax = plt.subplots(figsize=(8, 5))
    if used:
        mp = [len(mapped_by_asm[a]) for a in used]
        un = [len(df) - m for m in mp]
        x = np.arange(len(used))
        ax.bar(x, mp, label="mapped", color="#27ae60", edgecolor="white")
        ax.bar(x, un, bottom=mp, label="unmapped", color="#c0392b", edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(used)
        ax.set_ylabel("copies"); ax.legend(fontsize=9)
        ax.set_title(f"{args.family} --- mapped vs unmapped per assembly", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "no liftOver chains supplied",
                ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title(f"{args.family} --- mapped vs unmapped", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(reports / "fig_multiassembly_coords.pdf", bbox_inches="tight"); plt.close()
    ov.pp(f"  Saved {reports/'fig_multiassembly_coords.pdf'}")

    # ── measured values ──
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    best_asm = max(used, key=lambda a: len(mapped_by_asm[a])) if used else "---"
    best_rate = (100.0 * len(mapped_by_asm[best_asm]) / len(df)) if used else 0.0
    tex = ["% Auto-generated by run_multiassembly_liftover.py", f"% {ts}", "",
           rf"\providecommand{{\masmFamily}}{{{ov.texesc(args.family)}}}",
           rf"\providecommand{{\masmSource}}{{{ov.texesc(args.source_assembly)}}}",
           rf"\providecommand{{\masmNCopies}}{{{len(df)}}}",
           rf"\providecommand{{\masmNAssemblies}}{{{len(used)}}}",
           rf"\providecommand{{\masmAsmList}}{{{ov.texesc(', '.join(used)) if used else '---'}}}",
           rf"\providecommand{{\masmBestAsm}}{{{ov.texesc(best_asm)}}}",
           rf"\providecommand{{\masmBestRate}}{{{best_rate:.1f}}}", ""]
    (reports / "multiassembly_measured_values.tex").write_text("\n".join(tex))
    txt = ["=" * 60, "GAMECA Multi-Assembly Liftover --- Measured Values", f"Generated: {ts}",
           "=" * 60, f"  Family:           {args.family}",
           f"  Source assembly:  {args.source_assembly}",
           f"  Copies:           {len(df)}",
           f"  Target assemblies:{len(used)}  ({', '.join(used) if used else 'none'})"]
    for a in used:
        txt.append(f"    {a:<10} {len(mapped_by_asm[a])}/{len(df)} mapped "
                   f"({100*len(mapped_by_asm[a])/len(df):.1f}%)")
    (reports / "multiassembly_report.txt").write_text("\n".join(txt))
    ov.pp("  Written multiassembly_measured_values.tex and multiassembly_report.txt")
    ov.pp("=" * 60); ov.pp("DONE")
    print("\n" + "=" * 60); print("MEASURED VALUES (paste into chat):"); print("=" * 60)
    print((reports / "multiassembly_report.txt").read_text())


if __name__ == "__main__":
    main()
