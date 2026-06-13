#!/usr/bin/env python3
"""
te_moods_scan.py — on-demand JASPAR motif scanning of TE loci with MOODS.

Replaces the genome-wide bigBed download + tabix subset + bedtools intersect
pipeline (te_motif.py) with a direct scan of TE locus sequences against the
JASPAR PWM bundle.

Rationale (why scan, not intersect):
  The query set (a few thousand TE loci, ~5-50 MB of sequence total) is tiny
  next to a genome-wide materialised JASPAR track. Scanning that query against
  ~700 vertebrate PWMs with MOODS is a seconds-to-minutes operation and needs
  only (1) an indexed reference FASTA and (2) the JASPAR PFM flat file. No
  .bb download, no .bed.gz conversion, no .tbi, no tabix, no bedtools.

Inputs:
  - reference FASTA, indexed (.fai); created on the fly if missing.
  - JASPAR PFM bundle in JASPAR "raw" format, e.g.
        >MA0004.1   Arnt
        A  [   4  19   0 ... ]
        C  [  16   0  20 ... ]
        G  [   0   1   0 ... ]
        T  [   0   0   0 ... ]

Output (in-memory): a DataFrame of locus x motif hits with genome coordinates,
matching the column schema produced by the old bedtools-intersect path so that
the downstream Fisher enrichment in te_motif.py is unchanged.

p-value threshold defaults to 1e-4 — the JASPAR / FIMO field standard.
"""

from __future__ import annotations

import os
import sys
import time
import logging
import multiprocessing as mp
from pathlib import Path

import pandas as pd

log = logging.getLogger("te_motif")

# Nucleotide order used everywhere here: A, C, G, T (JASPAR + MOODS convention).
_BASES = "ACGT"


# ── JASPAR PFM parsing ──────────────────────────────────────────────────────

def parse_jaspar_pfms(path):
    """Parse a multi-matrix JASPAR 'raw' PFM file.

    Returns a list of dicts: {"id": matrix_id, "name": tf_name, "counts": [[A..],[C..],[G..],[T..]]}.
    Each counts row is a list of floats of equal length (the motif width).
    Robust to either '>MAxxxx.y TF' or '>MAxxxx.y\tTF' headers and to rows
    written either as 'A [ 1 2 3 ]' or 'A 1 2 3'.
    """
    path = Path(path)
    pfms = []
    cur_id = cur_name = None
    rows = {}

    def _flush():
        if cur_id is None:
            return
        if set(rows) != set(_BASES):
            log.warning("MOODS: skipping malformed matrix %s (rows present: %s)",
                        cur_id, sorted(rows))
            return
        widths = {len(rows[b]) for b in _BASES}
        if len(widths) != 1:
            log.warning("MOODS: skipping %s (ragged rows, widths=%s)", cur_id, widths)
            return
        counts = [rows[b] for b in _BASES]
        pfms.append({"id": cur_id, "name": cur_name or cur_id, "counts": counts})

    with open(path) as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith(">"):
                _flush()
                hdr = line[1:].strip()
                parts = hdr.replace("\t", " ").split(None, 1)
                cur_id = parts[0]
                cur_name = parts[1].strip() if len(parts) > 1 else parts[0]
                rows = {}
                continue
            # data row: leading base letter then numbers (brackets optional)
            base = line.strip()[0].upper()
            if base not in _BASES:
                continue
            nums = (line.replace("[", " ").replace("]", " ")
                        .strip()[1:].split())
            try:
                rows[base] = [float(x) for x in nums]
            except ValueError:
                log.warning("MOODS: could not parse counts row for %s: %r", cur_id, line)
    _flush()

    if not pfms:
        raise ValueError(f"No PFMs parsed from {path}")
    return pfms


# ── MOODS scanner construction ──────────────────────────────────────────────

def build_scanner(pfms, pvalue=1e-4, bg=None, pseudocount=0.8, window=7):
    """Build a MOODS Scanner over forward + reverse-complement log-odds matrices.

    Returns (scanner, meta) where meta[i] = (motif_label, strand) aligned to the
    scanner's matrix index order: first all forward matrices, then all RC.
    motif_label is 'TFNAME (MATRIX_ID)'.
    """
    import MOODS.scan
    import MOODS.tools

    if bg is None:
        bg = MOODS.tools.flat_bg(4)  # [0.25, 0.25, 0.25, 0.25]

    lo_fwd, thr_fwd, labels = [], [], []
    for p in pfms:
        lo = MOODS.tools.log_odds(p["counts"], bg, pseudocount)
        thr = MOODS.tools.threshold_from_p(lo, bg, pvalue)
        lo_fwd.append(lo)
        thr_fwd.append(thr)
        labels.append(f'{p["name"]} ({p["id"]})')

    lo_rc = [MOODS.tools.reverse_complement(m) for m in lo_fwd]
    matrices = lo_fwd + lo_rc
    thresholds = thr_fwd + thr_fwd
    meta = [(lab, "+") for lab in labels] + [(lab, "-") for lab in labels]

    scanner = MOODS.scan.Scanner(window)
    scanner.set_motifs(matrices, bg, thresholds)
    # motif widths for converting match positions to genome coords
    widths = [len(m[0]) for m in lo_fwd]
    widths = widths + widths
    return scanner, meta, widths


# ── per-process worker (multiprocessing) ────────────────────────────────────

_W = {}  # per-worker globals: scanner, meta, widths, fasta handle


def _worker_init(genome_fa, pfm_path, pvalue, pseudocount):
    import pysam
    pfms = parse_jaspar_pfms(pfm_path)
    scanner, meta, widths = build_scanner(pfms, pvalue=pvalue, pseudocount=pseudocount)
    _W["scanner"] = scanner
    _W["meta"] = meta
    _W["widths"] = widths
    _W["fa"] = pysam.FastaFile(genome_fa)


def _scan_chunk(chunk):
    """Scan a list of (chrom, start, stop) loci. Returns list of result tuples."""
    fa = _W["fa"]; scanner = _W["scanner"]; meta = _W["meta"]; widths = _W["widths"]
    chroms = set(fa.references)
    out = []
    for chrom, start, stop in chunk:
        c = chrom if chrom in chroms else (
            chrom[3:] if chrom.startswith("chr") and chrom[3:] in chroms else
            ("chr" + chrom) if ("chr" + chrom) in chroms else None)
        if c is None:
            continue
        start = max(0, int(start)); stop = int(stop)
        try:
            seq = fa.fetch(c, start, stop)
        except (KeyError, ValueError):
            continue
        if not seq:
            continue
        results = scanner.scan(seq)
        for mi, hits in enumerate(results):
            if not hits:
                continue
            label, strand = meta[mi]
            w = widths[mi]
            for h in hits:
                g0 = start + int(h.pos)
                out.append((chrom, start, stop, c, g0, g0 + w,
                            label, round(float(h.score), 4), strand))
    return out


# ── public entry point ──────────────────────────────────────────────────────

def ensure_fai(genome_fa):
    import pysam
    fai = str(genome_fa) + ".fai"
    if not os.path.exists(fai):
        log.info("MOODS: building FASTA index (.fai) for %s ...", genome_fa)
        t0 = time.time()
        pysam.faidx(str(genome_fa))
        log.info("MOODS: .fai built in %.1fs", time.time() - t0)


def scan_loci(loci_df, genome_fa, pfm_path, chr_col, start_col, stop_col,
              pvalue=1e-4, threads=None, pseudocount=0.8, chunk_size=64):
    """Scan all loci in loci_df against the JASPAR PFMs.

    Returns a DataFrame with columns
        [chr_col, start_col, stop_col,
         Motif_chr, Motif_start, Motif_end, Motif_name, Motif_score, Motif_strand]
    i.e. the same locus x motif-hit schema the bedtools path produced.
    """
    genome_fa = str(genome_fa)
    if not os.path.exists(genome_fa):
        raise FileNotFoundError(f"Genome FASTA not found: {genome_fa}")
    ensure_fai(genome_fa)

    pfms = parse_jaspar_pfms(pfm_path)
    total_bp = sum(int(r[stop_col]) - int(r[start_col])
                   for _, r in loci_df[[start_col, stop_col]].iterrows())
    log.info("MOODS: %d PFMs, %d loci, %.2f Mb of locus sequence, p<%.0e",
             len(pfms), len(loci_df), total_bp / 1e6, pvalue)

    loci = list(zip(loci_df[chr_col].astype(str),
                    loci_df[start_col].astype(int),
                    loci_df[stop_col].astype(int)))
    chunks = [loci[i:i + chunk_size] for i in range(0, len(loci), chunk_size)]

    if threads is None:
        threads = max(1, (os.cpu_count() or 2))
    threads = min(threads, len(chunks)) or 1

    t0 = time.time()
    rows = []
    if threads == 1:
        _worker_init(genome_fa, pfm_path, pvalue, pseudocount)
        for ci, ch in enumerate(chunks):
            rows.extend(_scan_chunk(ch))
            if (ci + 1) % 20 == 0:
                log.info("MOODS: scanned %d/%d chunks (%.0fs)",
                         ci + 1, len(chunks), time.time() - t0)
    else:
        ctx = mp.get_context("spawn") if sys.platform == "darwin" else mp.get_context("fork")
        with ctx.Pool(processes=threads, initializer=_worker_init,
                      initargs=(genome_fa, pfm_path, pvalue, pseudocount)) as pool:
            done = 0
            for res in pool.imap_unordered(_scan_chunk, chunks):
                rows.extend(res)
                done += 1
                if done % 20 == 0:
                    log.info("MOODS: scanned %d/%d chunks (%.0fs)",
                             done, len(chunks), time.time() - t0)

    elapsed = time.time() - t0
    cols = [chr_col, start_col, stop_col, "Motif_chr", "Motif_start",
            "Motif_end", "Motif_name", "Motif_score", "Motif_strand"]
    df = pd.DataFrame(rows, columns=cols)
    log.info("MOODS: %d motif hits across %d loci in %.1fs (%.0f loci/s)",
             len(df), len(loci_df), elapsed,
             len(loci_df) / elapsed if elapsed else 0)
    return df


# ── standalone CLI (handy for benchmarking / smoke tests) ────────────────────

def _main():
    import argparse
    ap = argparse.ArgumentParser(description="MOODS on-demand JASPAR scan of TE loci")
    ap.add_argument("--input", required=True, help="clustered CSV or BED of TE loci")
    ap.add_argument("--genome-fa", required=True, help="indexed reference FASTA")
    ap.add_argument("--jaspar-pfm", required=True, help="JASPAR raw PFM bundle")
    ap.add_argument("--out", default="moods_overlaps.tsv")
    ap.add_argument("--pvalue", type=float, default=1e-4)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    p = Path(args.input)
    if p.suffix.lower() in (".bed", ".tsv") and "," not in p.read_text(errors="ignore")[:200]:
        df = pd.read_csv(p, sep="\t", header=None).iloc[:, :3]
        df.columns = ["chr", "start", "stop"]
    else:
        df = pd.read_csv(p)
        # detect coord columns
        def fc(cands):
            for c in cands:
                if c in df.columns:
                    return c
        cc = fc(["chr", "Chromosome", "chrom", "Chr", "#chrom"])
        sc = fc(["start", "Start", "chromStart"])
        ec = fc(["stop", "Stop", "End", "end", "chromEnd"])
        df = df.rename(columns={cc: "chr", sc: "start", ec: "stop"})

    out = scan_loci(df, args.genome_fa, args.jaspar_pfm, "chr", "start", "stop",
                    pvalue=args.pvalue, threads=args.threads)
    out.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {len(out)} hits -> {args.out}")


if __name__ == "__main__":
    _main()
