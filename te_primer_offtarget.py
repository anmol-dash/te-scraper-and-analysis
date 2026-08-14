#!/usr/bin/env python3
"""te_primer_offtarget.py --- genome-wide, mismatch-tolerant off-target scan for
qPCR primers.

te_qpcr_primers.py reports a genome *hit count* from an exact-match search. That
undercounts what can actually prime: a primer with a perfect 3' end and one or
two mismatches in its 5' body still extends. This enumerates those sites.

Criterion for a site (both strands):
  * the primer's 3' ANCHOR bp match exactly       -- 3' fidelity drives extension
  * the remaining 5' body has <= MAX_MM mismatches

Why an anchor at all: the search space is the whole genome, so the anchor is the
seed that makes this tractable. Every qualifying site *must* contain the 3'
anchor exactly, so seeding on it misses nothing that the criterion admits -- but
the criterion itself is only as permissive as the anchor is short. ANCHOR=8
leaves ~47k seed sites per strand per primer to extend (sub-second); ANCHOR=5
(te_qpcr_coverage.PROTECT3) leaves ~3M, which is ~15 s of extension per primer
per strand on top of a much slower seed scan. Sites needing a mismatch inside
the last ANCHOR bp are therefore NOT enumerated -- state that alongside results
rather than calling the output exhaustive.

Output: one row per site, plus a per-primer summary.

  python3 te_primer_offtarget.py --primers LTR66_top60_primers.csv \
      --genome hg38.fa --family LTR66 --out results [--anchor 8] [--max-mm 3]
"""
import argparse, csv, os, sys, time

from te_genome import GenomeCache, reverse_complement


def _rc(s):
    return reverse_complement(s)


def scan_primer(primer, genomes, anchor, max_mm):
    """All sites where primer's 3' anchor matches exactly and the 5' body has
    <= max_mm mismatches. Returns list of (chrom, start1, end1, strand, n_mm)."""
    p = primer.upper()
    L = len(p)
    if anchor > L:
        anchor = L
    body_len = L - anchor
    out = []

    # Forward orientation: the primer as written. Its 3' anchor is the suffix, so
    # a genomic site starts body_len bp before the anchor match.
    seed_f = p[L - anchor:]
    body_f = p[:body_len]
    # Minus strand: the primer anneals to the reverse complement, so search for
    # rc(primer). rc's 3' end of the primer becomes rc's PREFIX.
    rcp = _rc(p)
    seed_r = rcp[:anchor]
    body_r = rcp[anchor:]

    for chrom, seq in genomes.items():
        n = len(seq)

        idx = seq.find(seed_f)
        while idx != -1:
            start = idx - body_len
            if start >= 0:
                win = seq[start:idx]
                mm = 0
                for a, b in zip(body_f, win):
                    if a != b:
                        mm += 1
                        if mm > max_mm:
                            break
                if mm <= max_mm:
                    out.append((chrom, start + 1, start + L, "+", mm))
            idx = seq.find(seed_f, idx + 1)

        idx = seq.find(seed_r)
        while idx != -1:
            end = idx + anchor + body_len
            if end <= n:
                win = seq[idx + anchor:end]
                mm = 0
                for a, b in zip(body_r, win):
                    if a != b:
                        mm += 1
                        if mm > max_mm:
                            break
                if mm <= max_mm:
                    out.append((chrom, idx + 1, idx + L, "-", mm))
            idx = seq.find(seed_r, idx + 1)

    return out


def load_primers(path):
    """Read a top_primers/qpcr_pairs CSV -> ordered unique [(name, seq)].

    Pairs share primers, so scanning per unique sequence avoids repeat work; the
    name records every pair/role the sequence came from.
    """
    seen = {}
    order = []
    with open(path, newline="") as fh:
        rdr = csv.DictReader(fh)
        cols = rdr.fieldnames or []
        fcol = next((c for c in cols if c.lower() in ("fwd", "forward", "fwd_seq")), None)
        rcol = next((c for c in cols if c.lower() in ("rev", "reverse", "rev_seq")), None)
        if not fcol or not rcol:
            sys.exit(f"no fwd/rev columns in {path} (cols: {cols})")
        rankcol = next((c for c in cols if c.lower() in ("rank", "pair")), None)
        for i, row in enumerate(rdr):
            rank = row.get(rankcol) if rankcol else str(i)
            for role, col in (("fwd", fcol), ("rev", rcol)):
                s = (row.get(col) or "").strip().upper()
                if not s:
                    continue
                tag = f"pair{rank}_{role}"
                if s in seen:
                    seen[s].append(tag)
                else:
                    seen[s] = [tag]
                    order.append(s)
    return [(";".join(seen[s]), s) for s in order]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--primers", required=True, help="CSV with fwd/rev columns")
    ap.add_argument("--genome", required=True)
    ap.add_argument("--family", default="FAMILY")
    ap.add_argument("--out", default="results")
    ap.add_argument("--anchor", type=int, default=8,
                    help="3' bp that must match exactly (seed); shorter = more sites, much slower")
    ap.add_argument("--max-mm", type=int, default=3, help="max mismatches in the 5' body")
    ap.add_argument("--primary-only", action="store_true",
                    help="restrict to chr1..22,X,Y,M (skip alt/random/Un contigs)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    primers = load_primers(a.primers)
    if not primers:
        sys.exit(f"no primers in {a.primers}")
    print(f"[ot] {a.family}: {len(primers)} unique primers from {a.primers}")

    cache = GenomeCache(a.genome)
    cache.load()
    genomes = cache.genomes
    if a.primary_only:
        keep = {f"chr{c}" for c in list(range(1, 23)) + ["X", "Y", "M"]}
        genomes = {c: s for c, s in genomes.items() if c in keep}
    print(f"[ot] scanning {len(genomes)} contigs, anchor={a.anchor}, max-mm={a.max_mm}")

    sites_path = os.path.join(a.out, f"{a.family}_offtarget_sites.csv")
    summ_path = os.path.join(a.out, f"{a.family}_offtarget_summary.csv")

    t_all = time.time()
    with open(sites_path, "w", newline="") as sf, open(summ_path, "w", newline="") as mf:
        sw = csv.writer(sf)
        sw.writerow(["primer", "pairs", "chrom", "start", "end", "strand", "n_mismatch"])
        mw = csv.writer(mf)
        mw.writerow(["primer", "pairs", "total_sites", "perfect_0mm",
                     "mm1", "mm2", "mm3", "seconds"])
        for i, (tag, seq) in enumerate(primers, 1):
            t0 = time.time()
            hits = scan_primer(seq, genomes, a.anchor, a.max_mm)
            dt = time.time() - t0
            by_mm = {k: 0 for k in range(a.max_mm + 1)}
            for h in hits:
                by_mm[h[4]] = by_mm.get(h[4], 0) + 1
                sw.writerow([seq, tag, h[0], h[1], h[2], h[3], h[4]])
            sf.flush()
            mw.writerow([seq, tag, len(hits), by_mm.get(0, 0), by_mm.get(1, 0),
                         by_mm.get(2, 0), by_mm.get(3, 0), round(dt, 1)])
            mf.flush()
            print(f"[ot] {i}/{len(primers)} {seq} {tag}: {len(hits)} sites "
                  f"(0mm={by_mm.get(0,0)}) in {dt:.1f}s", flush=True)

    print(f"[ot] {a.family}: done in {time.time()-t_all:.0f}s")
    print(f"[ot] sites   -> {sites_path}")
    print(f"[ot] summary -> {summ_path}")


if __name__ == "__main__":
    main()
