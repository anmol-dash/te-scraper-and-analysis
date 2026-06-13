#!/usr/bin/env python3
"""
te_primers.py
K-mer based primer design with genome-wide specificity checking.

Can be imported by query.py or called standalone:
    python te_primers.py --input clustered_data.csv --genome /path/to/genome.fa
"""

import argparse
import random
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path

import numpy as np
import pandas as pd

from te_genome import GenomeCache, _search_single_chrom, reverse_complement


MAJOR_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
MAX_GENOME_HITS = 10_000


def _pp(msg):
    import datetime
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _pbar(cur, tot, prefix="Progress", length=40):
    pct = cur / tot if tot else 0
    filled = int(length * pct)
    bar = "█" * filled + "░" * (length - filled)
    print(f"\r{prefix} |{bar}| {cur}/{tot} ({pct*100:.1f}%)", end="", flush=True)
    if cur >= tot:
        print()


# ── genome search helpers ───────────────────────────────────────────────────

def _search_genome_full(primer, fasta_path, stop_event=None):
    """File-based full genome search. Returns hit list or None if cancelled."""
    primer = primer.upper()
    rc = reverse_complement(primer)
    plen = len(primer)
    hits = []
    total = 0

    with open(fasta_path) as f:
        chrom = None
        buf = []
        for line in f:
            if stop_event and stop_event.is_set():
                return None
            if line.startswith(">"):
                if chrom is not None:
                    ch, total = _search_single_chrom(
                        primer, rc, plen, chrom, "".join(buf), MAX_GENOME_HITS, total
                    )
                    hits.extend(ch)
                    if total >= MAX_GENOME_HITS:
                        return hits
                chrom = line[1:].strip().split()[0]
                buf = []
            else:
                buf.append(line.strip())
        if chrom and (not stop_event or not stop_event.is_set()):
            ch, total = _search_single_chrom(
                primer, rc, plen, chrom, "".join(buf), MAX_GENOME_HITS, total
            )
            hits.extend(ch)
    return hits


def _estimate_genome_hits_by_sampling(primer, fasta_path, sample_chroms=5, seed=None):
    """Estimate genome-wide hits by sampling a few major chromosomes."""
    primer = primer.upper()
    rc = reverse_complement(primer)
    plen = len(primer)

    # Index chromosome positions
    chrom_positions = {}
    _pp("    Indexing genome chromosomes...")
    with open(fasta_path) as f:
        cur = None
        cur_start = 0
        pos = 0
        for line in f:
            if line.startswith(">"):
                if cur is not None:
                    chrom_positions[cur] = (cur_start, pos)
                cur = line[1:].strip().split()[0]
                cur_start = pos + len(line)
            pos += len(line)
        if cur is not None:
            chrom_positions[cur] = (cur_start, pos)
    _pp(f"    {len(chrom_positions)} chromosomes/contigs found")

    available = [c for c in MAJOR_CHROMS if c in chrom_positions] or \
                list(chrom_positions.keys())[:24]
    if not available:
        return pd.DataFrame(columns=["chrom", "start", "stop", "strand"]), True

    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    sampled = random.sample(available, min(sample_chroms, len(available)))
    _pp(f"    Sampling: {', '.join(sampled)}")

    hits = []
    sampled_len = 0
    with open(fasta_path) as f:
        for i, chrom in enumerate(sampled):
            st, en = chrom_positions[chrom]
            f.seek(st)
            data = f.read(en - st)
            seq = "".join(l for l in data.split("\n") if not l.startswith(">"))
            sampled_len += len(seq)
            ch, _ = _search_single_chrom(primer, rc, plen, chrom, seq)
            hits.extend(ch)
            _pp(f"    {chrom}: {len(ch)} hits ({len(seq):,} bp)")

    GENOME_SIZE = 3_100_000_000
    scaling = GENOME_SIZE / sampled_len if sampled_len > 0 else 1
    estimated = int(len(hits) * scaling)
    _pp(f"    Sampled {len(sampled)} chroms → {len(hits)} hits → ~{estimated:,} est. genome-wide")

    dfh = pd.DataFrame(hits, columns=["chrom", "start", "stop", "strand"])
    dfh["estimated_total"] = estimated
    dfh["sampling_note"] = f"Estimated from {len(sampled)} chroms (seed={seed})"
    return dfh, True


def search_seq_chromosomal(primer, fasta=None, timeout=120, genome_cache=None):
    """Search for primer + RC in genome. Returns DataFrame of hits.

    Priority:
    1. In-memory GenomeCache (fastest)
    2. File-based search with timeout + random-sampling fallback
    3. Empty DataFrame if no genome available
    """
    import time

    primer = primer.upper()

    # Fast path: in-memory cache
    if genome_cache is not None and genome_cache.is_loaded:
        t0 = time.time()
        hits = genome_cache.search_primer(primer)
        dfh = pd.DataFrame(hits, columns=["chrom", "start", "stop", "strand"]) if hits else \
              pd.DataFrame(columns=["chrom", "start", "stop", "strand"])
        _pp(f"  → {len(dfh):,} hits in {time.time()-t0:.1f}s (in-memory)")
        return dfh

    # File-based fallback
    fasta_path = Path(fasta) if fasta else None
    if fasta_path is None or not fasta_path.exists():
        return pd.DataFrame(columns=["chrom", "start", "stop", "strand"])

    rc = reverse_complement(primer)
    _pp(f"  Searching {fasta_path.name} for {primer} (rc: {rc})  [timeout={timeout}s]")

    stop_ev = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_search_genome_full, primer, fasta_path, stop_ev)
        try:
            hits = fut.result()
            if hits is None:
                raise FuturesTimeoutError()
            dfh = pd.DataFrame(hits, columns=["chrom", "start", "stop", "strand"])
            _pp(f"  → {len(dfh):,} hits (full search)")
            return dfh
        except FuturesTimeoutError:
            stop_ev.set()
            _pp(f"  TIMEOUT — falling back to random sampling...")
            dfh, _ = _estimate_genome_hits_by_sampling(primer, fasta_path)
            return dfh


# ── k-mer / primer design ───────────────────────────────────────────────────

def build_kmer_index(df, primer_k):
    """Build {kmer: set(row_indices)} mapping from df['Seq']."""
    kmer_to_rows = defaultdict(set)
    seqs = df["Seq"].astype(str)
    n = len(seqs)
    for idx, (ridx, seq) in enumerate(seqs.items()):
        _pbar(idx + 1, n, prefix="  Indexing k-mers")
        s = seq.upper()
        seen = set()
        for i in range(len(s) - primer_k + 1):
            km = s[i:i + primer_k]
            if "N" in km or km in seen:
                continue
            kmer_to_rows[km].add(ridx)
            seen.add(km)
    print()  # newline after progress bar
    return kmer_to_rows


def design_primers(df, primer_k=18, top_global=8, top_cluster=5,
                   genome_fa=None, genome_cache=None, primer_timeout=120,
                   out_dir=None, family_name="FAMILY"):
    """Design k-mer primers globally and per cluster.

    Args:
        df:            DataFrame with 'Seq', 'Cluster', and optional expression cols.
        primer_k:      K-mer size.
        top_global:    Top N global primers.
        top_cluster:   Top N primers per cluster.
        genome_fa:     Path to genome FASTA (for specificity search).
        genome_cache:  GenomeCache instance (preferred over genome_fa).
        primer_timeout: Seconds before fallback to random sampling.
        out_dir:       Output directory for CSVs.
        family_name:   Used in filenames.

    Returns:
        dict with keys 'selected_primers', 'kmer_df', 'primer_hits',
        'cluster_top_primers', 'top_by_cov_expr', 'top_by_expr_cov'
    """
    out_dir = Path(out_dir) if out_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Detect expression columns
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    exclude = {"start", "stop", "Unnamed: 0", "chr", "Cluster"}
    expr_cols = [c for c in numeric_cols if c not in exclude]

    df = df.reset_index(drop=True).copy()
    df["_total_expr"] = df[expr_cols].sum(axis=1) if expr_cols else 0
    rows_total_expr = df["_total_expr"].to_dict()

    # Build global k-mer index
    _pp(f"Building {primer_k}-mer index for {len(df)} sequences...")
    kmer_to_rows = build_kmer_index(df, primer_k)
    _pp(f"  {len(kmer_to_rows):,} unique {primer_k}-mers")

    if not kmer_to_rows:
        _pp(f"WARNING: No valid {primer_k}-mers found — skipping primer design")
        return {"selected_primers": [], "kmer_df": pd.DataFrame(),
                "primer_hits": {}, "cluster_top_primers": {}}

    # Score k-mers
    records = [
        {"primer": km, "coverage": len(rs),
         "total_expr": sum(rows_total_expr[r] for r in rs), "rows": rs}
        for km, rs in kmer_to_rows.items()
    ]
    kmer_df = pd.DataFrame(records)

    top_cov_expr  = kmer_df.sort_values(["coverage", "total_expr"], ascending=[False, False]).head(top_global).copy()
    top_expr_cov  = kmer_df.sort_values(["total_expr", "coverage"], ascending=[False, False]).head(top_global).copy()
    selected = pd.concat([top_cov_expr["primer"], top_expr_cov["primer"]]).unique().tolist()

    # Genome searches for selected primers
    primer_hits = {}
    if selected and (genome_fa or (genome_cache and genome_cache.is_loaded)):
        _pp(f"Genome search for {len(selected)} selected primers...")
        for i, pr in enumerate(selected):
            _pp(f"  [{i+1}/{len(selected)}] {pr}")
            dfh = search_seq_chromosomal(pr, fasta=genome_fa,
                                         genome_cache=genome_cache)
            primer_hits[pr] = dfh
            dfh.to_csv(out_dir / f"{pr}_genome_hits.csv", index=False)

    # Per-cluster top primers
    cluster_top_primers = {}
    for cluster, rows_idx in df.groupby("Cluster").groups.items():
        clabel = "noise" if cluster == -1 else str(cluster)
        _pp(f"  Cluster {clabel} primers...")
        km_local = defaultdict(set)
        for rid in rows_idx:
            s = str(df.at[rid, "Seq"]).upper()
            seen = set()
            for i in range(len(s) - primer_k + 1):
                km = s[i:i + primer_k]
                if "N" in km or km in seen:
                    continue
                km_local[km].add(rid)
                seen.add(km)

        if not km_local:
            continue
        recs = [{"primer": k, "coverage": len(rs),
                 "total_expr": sum(rows_total_expr[r] for r in rs),
                 "rows": rs}
                for k, rs in km_local.items()]
        df_local = pd.DataFrame(recs)
        top5 = df_local.sort_values(
            ["coverage", "total_expr"], ascending=[False, False]
        ).head(top_cluster)
        cluster_top_primers[cluster] = top5

        for p in top5["primer"].tolist():
            if p not in primer_hits and (genome_fa or (genome_cache and genome_cache.is_loaded)):
                primer_hits[p] = search_seq_chromosomal(
                    p, fasta=genome_fa, timeout=primer_timeout,
                    genome_cache=genome_cache
                )
                primer_hits[p].to_csv(out_dir / f"{p}_genome_hits.csv", index=False)

    # ── Save outputs ────────────────────────────────────────────────────────
    def _rows_str(rs):
        return ",".join(map(str, sorted(rs)))

    def _coords_str(rs):
        parts = []
        for r in sorted(rs):
            row = df.loc[r]
            parts.append(f"{row.get('chr','?')}:{row.get('start','?')}-{row.get('stop','?')}")
        return ",".join(parts)

    # All candidates
    kmer_out = kmer_df.copy()
    kmer_out["rows_covered"] = kmer_out["rows"].apply(_rows_str)
    kmer_out["rows_coordinates"] = kmer_out["rows"].apply(_coords_str)
    kmer_out.sort_values(["coverage", "total_expr"], ascending=[False, False])\
            .to_csv(out_dir / f"all_{primer_k}mer_candidates_metrics.csv", index=False)

    # Selected primers summary
    if selected:
        top_summary = pd.DataFrame({
            "primer": selected,
            "coverage":   [kmer_df.loc[kmer_df["primer"] == p, "coverage"].values[0] for p in selected],
            "total_expr": [kmer_df.loc[kmer_df["primer"] == p, "total_expr"].values[0] for p in selected],
            "rows_covered": [_rows_str(kmer_df.loc[kmer_df["primer"] == p, "rows"].values[0]) for p in selected],
            "rows_coordinates": [_coords_str(kmer_df.loc[kmer_df["primer"] == p, "rows"].values[0]) for p in selected],
            "strategy": [
                "cov_then_expr" if p in set(top_cov_expr["primer"]) else "expr_then_cov"
                for p in selected
            ],
        })
        top_summary.to_csv(out_dir / "selected_primers_summary.csv", index=False)
        _pp(f"  Saved selected_primers_summary.csv ({len(selected)} primers)")

    # Cluster top primers
    cluster_rows = []
    for cl, dfp in cluster_top_primers.items():
        for _, r in dfp.iterrows():
            cluster_rows.append({
                "cluster": cl, "primer": r["primer"],
                "coverage": r["coverage"], "total_expr": r["total_expr"],
                "rows_covered": _rows_str(r["rows"])
            })
    if cluster_rows:
        pd.DataFrame(cluster_rows).to_csv(out_dir / "cluster_top_primers.csv", index=False)
        _pp(f"  Saved cluster_top_primers.csv")

    # Genome hits summary
    if primer_hits:
        summary_rows = []
        for p, dfh in primer_hits.items():
            is_est = "estimated_total" in dfh.columns
            summary_rows.append({
                "primer": p,
                "genome_hits": len(dfh),
                "estimated": is_est,
                "estimated_total": dfh["estimated_total"].iloc[0] if is_est and len(dfh) > 0 else len(dfh)
            })
        pd.DataFrame(summary_rows).sort_values("genome_hits", ascending=False)\
          .to_csv(out_dir / "primer_genome_hits_summary.csv", index=False)
        _pp("  Saved primer_genome_hits_summary.csv")

    return {
        "selected_primers": selected,
        "kmer_df": kmer_df,
        "primer_hits": primer_hits,
        "cluster_top_primers": cluster_top_primers,
        "top_by_cov_expr": top_cov_expr,
        "top_by_expr_cov": top_expr_cov,
    }


# ── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Primer design for TE families")
    p.add_argument("--input",   required=True, help="CSV with Seq + Cluster columns")
    p.add_argument("--genome",  default=None,  help="Path to genome FASTA")
    p.add_argument("--kmer",    type=int, default=18)
    p.add_argument("--top-global",  type=int, default=8)
    p.add_argument("--top-cluster", type=int, default=5)
    p.add_argument("--timeout", type=int, default=120, help="Primer search timeout (s)")
    p.add_argument("--out-dir", default="primers", help="Output directory")
    p.add_argument("--family",  default="FAMILY")
    p.add_argument("--notify-email", default="", metavar="EMAIL",
                   help="Send a completion email to this address (requires Gmail App Password setup).")
    return p.parse_args()


if __name__ == "__main__":
    import sys
    args = _parse_args()
    df = pd.read_csv(args.input)

    gc = None
    if args.genome:
        gc = GenomeCache(args.genome)
        gc.load()

    design_primers(
        df, primer_k=args.kmer,
        top_global=args.top_global, top_cluster=args.top_cluster,
        genome_fa=args.genome, genome_cache=gc,
        primer_timeout=args.timeout, out_dir=args.out_dir,
        family_name=args.family,
    )
    if args.notify_email:
        from te_notify import send_completion_email
        send_completion_email(args.notify_email, "te_primers", args.out_dir)
