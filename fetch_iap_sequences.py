#!/usr/bin/env python3
"""fetch_iap_sequences.py — fetch sequences for ALL IAP repNames from rmsk.

The stock te_prep.py matches a single repName exactly. "IAP" is not one
repName but a whole set (IAPLTR1_Mm, IAPLTR1a_Mm, IAPEz-int, IAPEY*, IAP-d-int,
IAPA_MM-int, ...). This driver reuses te_prep's machinery to:

  1) scan rmsk_<build>.txt.gz ONCE, keeping every row whose repName matches the
     IAP pattern (substring, case-insensitive);
  2) extract sequences strand-aware (minus-strand loci are reverse-complemented,
     exactly as te_prep does — via the same extract/fetch helpers);
  3) write one combined CSV with the same column schema te_prep produces.

Usage:
  python fetch_iap_sequences.py --build mm10 --out iap_mm10_sequences.csv
  python fetch_iap_sequences.py --build mm10 --genome-fa /path/mm10.fa --out ...
  python fetch_iap_sequences.py --build mm10 --pattern IAP --rmsk-dir ~/te_analysis/rmsk

With a local --genome-fa it extracts offline; otherwise it falls back to the
UCSC sequence API (needs outbound HTTPS).
"""
import argparse
import gzip
import os
import sys
import time
from pathlib import Path

import te_prep as tp


def parse_rmsk_pattern(rmsk_path, pattern, std_chroms=None):
    """Stream rmsk.txt.gz, keep rows whose repName contains `pattern` (ci)."""
    pat = pattern.lower()
    hits = []
    t0 = time.time()
    print(f"  Scanning {Path(rmsk_path).name} for repName containing '{pattern}'...")
    with gzip.open(rmsk_path, "rt") as f:
        for n_lines, line in enumerate(f, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 13:
                continue
            if pat not in fields[tp.COL_REPNAME].lower():
                continue
            chrom = fields[tp.COL_CHROM]
            if std_chroms and chrom not in std_chroms:
                continue
            hits.append({
                "Chromosome": chrom,
                "Start":      int(fields[tp.COL_START]),
                "Stop":       int(fields[tp.COL_END]),
                "strand":     fields[tp.COL_STRAND],
                "repName":    fields[tp.COL_REPNAME],
                "repClass":   fields[tp.COL_REPCLASS],
                "repFamily":  fields[tp.COL_REPFAM],
                "swScore":    int(fields[tp.COL_SWSCORE]),
                "milliDiv":   int(fields[tp.COL_MILLIDIV]),
            })
    print(f"  Scanned {n_lines:,} lines in {time.time()-t0:.1f}s -> {len(hits):,} hits")
    return hits


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build", default="mm10", help="Genome assembly (default: mm10)")
    p.add_argument("--pattern", default="IAP",
                   help="repName substring to match, case-insensitive (default: IAP)")
    p.add_argument("--out", default=None,
                   help="Output CSV path (default: iap_<build>_sequences.csv)")
    p.add_argument("--rmsk-dir", default=tp._DEFAULT_RMSK,
                   help="Directory holding rmsk_<build>.txt.gz (downloaded if absent)")
    p.add_argument("--genome-fa", default=None,
                   help="Local genome FASTA for offline strand-aware extraction")
    p.add_argument("--fetch-workers", type=int, default=3,
                   help="Parallel workers for UCSC API fallback (default: 3)")
    p.add_argument("--max-loci", type=int, default=None, help="Cap loci (debug)")
    p.add_argument("--no-filter-chroms", action="store_true",
                   help="Keep non-standard chromosomes/scaffolds")
    args = p.parse_args()

    build = args.build
    out = args.out or f"iap_{build}_sequences.csv"
    std_chroms = None
    if not args.no_filter_chroms:
        std_chroms = tp.STD_CHROMS_HUMAN if build.startswith("hg") else tp.STD_CHROMS_MOUSE

    # Resolve genome FASTA (optional; UCSC API is the fallback path)
    genome_fa = (args.genome_fa
                 or tp.GENOME_FA.get(build, "")
                 or os.environ.get(f"{build.upper()}_FA", ""))
    use_ucsc = False
    if genome_fa and not os.path.exists(genome_fa):
        print(f"WARNING: genome FASTA not found at {genome_fa} -> UCSC API fallback")
        genome_fa = ""
    if not genome_fa:
        print(f"  No local genome FASTA -> fetching sequences from UCSC API ({build})")
        use_ucsc = True

    rmsk_path = tp.get_rmsk_path(build, args.rmsk_dir)  # downloads if missing

    print("=" * 60)
    print("Fetch ALL IAP sequences from rmsk")
    print(f"  Build:     {build}")
    print(f"  Pattern:   repName contains '{args.pattern}' (case-insensitive)")
    print(f"  rmsk:      {rmsk_path}")
    print(f"  Genome FA: {genome_fa or '(none - UCSC API fallback)'}")
    print(f"  Output:    {Path(out).resolve()}")
    print("=" * 60)

    hits = parse_rmsk_pattern(rmsk_path, args.pattern, std_chroms)
    if not hits:
        sys.exit(f"FATAL: no loci matching '{args.pattern}' in {build} rmsk.")

    import pandas as pd
    df = pd.DataFrame(hits).sort_values(["Chromosome", "Start"]).reset_index(drop=True)

    counts = df["repName"].value_counts()
    print(f"\n  {len(df):,} loci across {len(counts)} IAP repNames, "
          f"{df['Chromosome'].nunique()} chromosomes")
    print(f"  Strand: + {int((df['strand']=='+').sum()):,}  "
          f"- {int((df['strand']=='-').sum()):,}")
    for name, cnt in counts.items():
        print(f"    {name:28s} {cnt:>8,}")

    if args.max_loci and len(df) > args.max_loci:
        print(f"  Capping to {args.max_loci} (--max-loci)")
        df = df.head(args.max_loci).reset_index(drop=True)

    # TE_ID schema identical to te_prep
    df["TE_ID"] = (
        df["Chromosome"] + "|" + df["Start"].astype(str) + "|" + df["Stop"].astype(str)
        + "|" + df["repName"] + ":" + df["repFamily"] + ":" + df["repClass"]
        + "|" + df["swScore"].astype(str) + "|" + df["strand"]
    )
    df["TE_name"] = df["TE_ID"]
    df["chr"], df["start"], df["stop"] = df["Chromosome"], df["Start"], df["Stop"]

    # Strand-aware sequence retrieval (minus strand reverse-complemented)
    if use_ucsc:
        print(f"\nExtracting via UCSC API ({build})...")
        seqs, urls = tp.fetch_sequences_ucsc(df, assembly=build,
                                             n_workers=args.fetch_workers)
        df["Seq"], df["ucsc_url"] = seqs, urls
    else:
        print(f"\nExtracting from local FASTA: {genome_fa}")
        df["Seq"] = tp.extract_sequences(genome_fa, df)
        df["ucsc_url"] = ""

    before = len(df)
    df = df[df["Seq"].str.len() > 0].reset_index(drop=True)
    if len(df) < before:
        print(f"  Dropped {before - len(df)} empty sequences")

    df["Length"] = df["Seq"].str.len()
    df["GC_Content"] = df["Seq"].apply(
        lambda s: (s.upper().count("G") + s.upper().count("C")) / len(s) if s else 0)

    keep = [c for c in ["Chromosome", "Start", "Stop", "chr", "start", "stop",
                        "Seq", "TE_ID", "TE_name", "strand",
                        "repName", "repClass", "repFamily",
                        "swScore", "milliDiv", "Length", "GC_Content", "ucsc_url"]
            if c in df.columns]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df[keep].to_csv(out, index=False)
    print(f"\nDONE: {len(df):,} IAP loci -> {out}")
    print(f"  Length mean={df['Length'].mean():.0f}  GC mean={df['GC_Content'].mean():.3f}")


if __name__ == "__main__":
    main()
