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


# ── motif consensus sequences ───────────────────────────────────────────────
# A motif table that names TFs but never shows the SEQUENCE they were matched on
# is not usable on its own: the first question asked of one is always "what does
# that TF actually bind here", and answering it otherwise means going back to
# JASPAR by hand, matrix by matrix. Every motif output in this repo is expected
# to carry a consensus column, so the code that builds one lives here -- beside
# the PFM parser both motif paths already share -- rather than in any one caller.

# IUPAC ambiguity codes, keyed by the set of bases they stand for.
IUPAC = {
    frozenset("A"): "A", frozenset("C"): "C", frozenset("G"): "G", frozenset("T"): "T",
    frozenset("AG"): "R", frozenset("CT"): "Y", frozenset("CG"): "S", frozenset("AT"): "W",
    frozenset("GT"): "K", frozenset("AC"): "M",
    frozenset("CGT"): "B", frozenset("AGT"): "D", frozenset("ACT"): "H", frozenset("ACG"): "V",
    frozenset("ACGT"): "N",
}

IUPAC_LEGEND = ("IUPAC: R=A/G Y=C/T S=C/G W=A/T K=G/T M=A/C "
                "B=not A D=not C H=not G V=not T N=any")

_COMP_IUPAC = str.maketrans("ACGTRYSWKMBDHVNacgtryswkmbdhvn",
                            "TGCAYRSWMKVHDBNtgcayrswmkvhdbn")


def revcomp_iupac(s):
    """Reverse-complement, ambiguity codes included."""
    return s.translate(_COMP_IUPAC)[::-1]


def iupac_from_freqs(freqs, cum=0.75):
    """Collapse one column of base frequencies (A,C,G,T order) to one letter.

    Bases are taken in descending frequency until they account for `cum` of the
    column, so a column dominated by one base gives that base and a split column
    gives the ambiguity code for the bases carrying it.
    """
    total = sum(freqs) or 1.0
    order = sorted(range(4), key=lambda i: -freqs[i])
    picked, acc, last = [], 0.0, 0.0
    for i in order:
        picked.append(_BASES[i])
        acc += freqs[i] / total
        last = freqs[i]
        if acc >= cum:
            break
    # Any base as frequent as the last one taken must come too, or where the cum
    # threshold happens to fall decides the answer: a uniform column would stop
    # after three equal bases and read 'V' when the honest answer is 'N'.
    for i in order[len(picked):]:
        if freqs[i] >= last:
            picked.append(_BASES[i])
    return IUPAC.get(frozenset(picked), "N")


def column_info(freqs):
    """Information content of one column, in bits (0 = uniform, 2 = fixed)."""
    import math
    total = sum(freqs) or 1.0
    ic = 2.0
    for f in freqs:
        p = f / total
        if p > 0:
            ic += p * math.log2(p)
    return ic


def consensus_from_counts(counts, cum=0.75, min_bits=1.0):
    """(consensus, core, info_bits) for one PFM's [[A..],[C..],[G..],[T..]].

    `core` is the consensus trimmed to the informative middle: flanking columns
    below min_bits are dropped, since a matrix's low-information tails are noise
    to the eye and make two views of one site look unrelated.
    """
    cols = list(zip(*counts))
    cons = "".join(iupac_from_freqs(c, cum) for c in cols)
    ics = [column_info(c) for c in cols]
    lo, hi = 0, len(cols)
    while lo < hi and ics[lo] < min_bits:
        lo += 1
    while hi > lo and ics[hi - 1] < min_bits:
        hi -= 1
    return cons, (cons[lo:hi] or cons), round(sum(ics), 2)


def pfm_consensus_map(pfm_path, cum=0.75, min_bits=1.0):
    """{label: {id,name,consensus,core,width,info_bits}} for a JASPAR bundle.

    Keyed by 'NAME (ID)' -- the same Motif_name scan_loci emits -- and also by
    the bare matrix ID and bare TF name, so callers whose motif labels came from
    a JASPAR BED track rather than from this scanner can still look one up.
    """
    out = {}
    for p in parse_jaspar_pfms(pfm_path):
        cons, core, bits = consensus_from_counts(p["counts"], cum, min_bits)
        rec = {"id": p["id"], "name": p["name"], "consensus": cons, "core": core,
               "width": len(p["counts"][0]), "info_bits": bits}
        out[f'{p["name"]} ({p["id"]})'] = rec
        out.setdefault(p["id"], rec)
        out.setdefault(p["id"].split(".")[0], rec)
        out.setdefault(p["name"], rec)
        out.setdefault(p["name"].upper(), rec)
    return out


def observed_consensus(seqs, cum=0.75, min_n=3):
    """IUPAC consensus of actual matched substrings -- what the copies REALLY have.

    Sequences must already be oriented to the motif (reverse-complement '-'
    strand matches first). Returns '' when there is too little to summarise.
    """
    seqs = [s.upper() for s in seqs if s]
    seqs = [s for s in seqs if set(s) <= set(_BASES)]
    if len(seqs) < min_n:
        return ""
    widths = {}
    for s in seqs:
        widths[len(s)] = widths.get(len(s), 0) + 1
    w = max(widths, key=widths.get)
    seqs = [s for s in seqs if len(s) == w]
    if len(seqs) < min_n:
        return ""
    cols = []
    for i in range(w):
        counts = [0.0] * 4
        for s in seqs:
            counts[_BASES.index(s[i])] += 1
        cols.append(iupac_from_freqs(counts, cum))
    return "".join(cols)


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

_W = {}  # per-worker globals: scanner, meta, widths, fasta handle (optional)


def _worker_init(genome_fa, pfm_path, pvalue, pseudocount):
    pfms = parse_jaspar_pfms(pfm_path)
    scanner, meta, widths = build_scanner(pfms, pvalue=pvalue, pseudocount=pseudocount)
    _W["scanner"] = scanner
    _W["meta"] = meta
    _W["widths"] = widths
    if genome_fa is not None:
        import pysam
        _W["fa"] = pysam.FastaFile(genome_fa)


def _scan_chunk(chunk):
    """Scan a list of (chrom, start, stop[, seq]) loci. Returns list of result tuples.

    When a 4th element (seq) is present the FASTA handle is not consulted.
    """
    fa = _W.get("fa"); scanner = _W["scanner"]; meta = _W["meta"]; widths = _W["widths"]
    chroms = set(fa.references) if fa is not None else set()
    out = []
    for item in chunk:
        if len(item) == 4:
            chrom, start, stop, seq = item
            start = max(0, int(start)); stop = int(stop)
            c = chrom
        else:
            chrom, start, stop = item
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


# ── UCSC sequence fallback ──────────────────────────────────────────────────

def _fetch_sequences_ucsc(chr_vals, start_vals, stop_vals, assembly, n_workers=4):
    """Fetch sequences from the UCSC getData/sequence API using curl.

    Uses curl via shell=True so DNS routes through /bin/sh, which works on HPC
    nodes where Python's socket resolver fails. No timeout — retries indefinitely
    with exponential back-off on rate-limits and transient errors.
    Returns a list of sequences in the same order as the input iterables.
    """
    import subprocess, json as _json, shlex, threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    loci = list(zip(chr_vals, start_vals, stop_vals))
    n = len(loci)
    seqs = [None] * n
    _lock = threading.Lock()

    def _curl_json(url):
        cmd = f"curl -s {shlex.quote(url)}"
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if r.returncode != 0:
            raise ConnectionError(f"curl exit={r.returncode}: {(r.stderr or r.stdout).strip()[:120]}")
        try:
            return _json.loads(r.stdout)
        except _json.JSONDecodeError as exc:
            raise ValueError(f"JSON parse: {exc} body={r.stdout[:80]!r}")

    def _fetch_one(i):
        chrom, start, stop = loci[i]
        url = (f"https://api.genome.ucsc.edu/getData/sequence?"
               f"genome={assembly};chrom={chrom};start={int(start)};end={int(stop)}")
        attempt = 0
        while True:
            try:
                import time as _time
                _time.sleep(0.4)
                res = _curl_json(url)
                if res.get("statusCode") == 429:
                    wait = min(120, 30 * 2 ** attempt)
                    log.warning("UCSC rate-limited [%d] — sleeping %ds", i, wait)
                    _time.sleep(wait)
                    attempt += 1
                    continue
                if "error" in res:
                    raise ValueError(res["error"])
                dna = res.get("dna", "")
                if not dna:
                    raise ValueError("empty dna field")
                return i, dna.upper()
            except Exception as exc:
                wait = min(120, 2 ** attempt)
                log.warning("UCSC fetch [%d] attempt %d failed (%s) — retry in %ds",
                            i, attempt + 1, exc, wait)
                import time as _time
                _time.sleep(wait)
                attempt += 1

    log.info("UCSC: fetching %d sequences for assembly=%s (%d workers) …", n, assembly, n_workers)
    done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as exe:
        futures = {exe.submit(_fetch_one, i): i for i in range(n)}
        for fut in as_completed(futures):
            i, seq = fut.result()
            seqs[i] = seq
            done += 1
            if done % 50 == 0 or done == n:
                log.info("UCSC: %d/%d sequences fetched", done, n)
    return seqs


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
              pvalue=1e-4, threads=None, pseudocount=0.8, chunk_size=64,
              seq_col=None, assembly=None):
    """Scan all loci in loci_df against the JASPAR PFMs.

    If seq_col is given, sequences are taken directly from that DataFrame column
    and genome_fa is not needed (pass None). Otherwise genome_fa must be an
    indexed FASTA path.

    Returns a DataFrame with columns
        [chr_col, start_col, stop_col,
         Motif_chr, Motif_start, Motif_end, Motif_name, Motif_score, Motif_strand]
    i.e. the same locus x motif-hit schema the bedtools path produced.
    """
    using_seq_col = seq_col is not None and seq_col in loci_df.columns
    if using_seq_col:
        genome_fa = None
        total_bp = loci_df[seq_col].astype(str).str.len().sum()
        log.info("MOODS: %d loci, %.2f Mb of pre-loaded sequence, p<%.0e (seq_col=%s)",
                 len(loci_df), total_bp / 1e6, pvalue, seq_col)
    elif genome_fa is not None:
        genome_fa = str(genome_fa)
        if not os.path.exists(genome_fa):
            raise FileNotFoundError(f"Genome FASTA not found: {genome_fa}")
        ensure_fai(genome_fa)
        total_bp = sum(int(r[stop_col]) - int(r[start_col])
                       for _, r in loci_df[[start_col, stop_col]].iterrows())
        log.info("MOODS: %d loci, %.2f Mb of locus sequence, p<%.0e",
                 len(loci_df), total_bp / 1e6, pvalue)
    else:
        # No FASTA, no seq col — fall back to UCSC API download
        if assembly is None:
            raise ValueError("assembly must be supplied for UCSC fallback (e.g. 'mm10', 'hg38')")
        log.info("MOODS: no genome FASTA or seq column — falling back to UCSC API (assembly=%s)", assembly)
        seqs = _fetch_sequences_ucsc(
            loci_df[chr_col].astype(str),
            loci_df[start_col].astype(int),
            loci_df[stop_col].astype(int),
            assembly=assembly,
        )
        loci_df = loci_df.copy()
        loci_df["_seq_ucsc"] = seqs
        seq_col = "_seq_ucsc"
        using_seq_col = True
        genome_fa = None

    pfms = parse_jaspar_pfms(pfm_path)
    log.info("MOODS: %d PFMs loaded", len(pfms))

    if using_seq_col:
        loci = list(zip(loci_df[chr_col].astype(str),
                        loci_df[start_col].astype(int),
                        loci_df[stop_col].astype(int),
                        loci_df[seq_col].astype(str)))
    else:
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
    ap.add_argument("--genome-fa", default=None,
                    help="indexed reference FASTA (not needed when --seq-col is used)")
    ap.add_argument("--seq-col", default=None,
                    help="CSV column containing pre-extracted sequences (e.g. Seq); "
                         "auto-detected from Seq/seq/sequence if present")
    ap.add_argument("--assembly", default=None,
                    help="UCSC assembly name (e.g. mm10, hg38) — used as fallback when "
                         "neither --genome-fa nor a sequence column is available")
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
        seq_col = None
    else:
        df = pd.read_csv(p)
        def fc(cands):
            for c in cands:
                if c in df.columns:
                    return c
        cc = fc(["chr", "Chromosome", "chrom", "Chr", "#chrom"])
        sc = fc(["start", "Start", "chromStart"])
        ec = fc(["stop", "Stop", "End", "end", "chromEnd"])
        df = df.rename(columns={cc: "chr", sc: "start", ec: "stop"})
        # auto-detect sequence column
        seq_col = args.seq_col or fc(["Seq", "seq", "sequence"])

    if seq_col is None and args.genome_fa is None and args.assembly is None:
        ap.error("Provide --genome-fa, a CSV with a Seq column, or --assembly for UCSC fallback")

    out = scan_loci(df, args.genome_fa, args.jaspar_pfm, "chr", "start", "stop",
                    pvalue=args.pvalue, threads=args.threads, seq_col=seq_col,
                    assembly=args.assembly)
    out.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {len(out)} hits -> {args.out}")


if __name__ == "__main__":
    _main()
