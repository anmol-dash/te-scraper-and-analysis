#!/usr/bin/env python3
"""top_primers.py --- rank the TOP-N qPCR primer pairs for a TE family by
combined coverage (family copies + expression).

Where te_qpcr_primers.py returns a small greedy *set* that covers the family,
this enumerates a large candidate pool (primer3 across many real copies), scores
EVERY candidate by how much of the family it amplifies -- weighted by both copy
count and per-locus expression -- and writes the top N ranked pairs. Genome
specificity (hit count) is computed for the top N only, using the in-memory
GenomeCache so it's fast.

  python3 top_primers.py --input HERVK-int_gse.seqs.csv --family HERVK-int \
      --genome hg38.fa --out results --top-n 100 [--w-seq 1 --w-expr 1]
"""
import argparse, csv, os, sys

from te_qpcr_primers import read_records, design_on, genome_hits
from te_qpcr_coverage import _amplifies


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--family", default="FAMILY")
    ap.add_argument("--out", default="results")
    ap.add_argument("--genome", default=None)
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--max-reps", type=int, default=40, help="real copies to seed primer3")
    ap.add_argument("--per-copy", type=int, default=20, help="primer3 candidates per copy")
    ap.add_argument("--expr-cols", nargs="+", default=None)
    ap.add_argument("--amp-min", type=int, default=80)
    ap.add_argument("--amp-max", type=int, default=150)
    ap.add_argument("--opt-tm", type=float, default=60.0)
    ap.add_argument("--max-mm", type=int, default=3)
    ap.add_argument("--w-seq", type=float, default=1.0)
    ap.add_argument("--w-expr", type=float, default=1.0)
    a = ap.parse_args()

    seqs, _clusters, weights = read_records(a.input, a.expr_cols)
    if not seqs:
        sys.exit("no usable sequences")
    weighted = weights is not None
    w = weights if weighted else [1.0] * len(seqs)
    tot_c = float(len(seqs)); tot_e = float(sum(w)) or 1.0
    lo, hi = int(a.amp_min * 0.6), int(a.amp_max * 1.6)
    print(f"[top] {a.family}: {len(seqs)} copies, {'expression-weighted' if weighted else 'copy-count'}")

    # --- enumerate a large candidate pool: primer3 across the longest real copies
    reps = sorted(seqs, key=len, reverse=True)[:min(a.max_reps, len(seqs))]
    cand = {}
    for rep in reps:
        if len(rep) < a.amp_min:
            continue
        res = design_on(rep, a.family, a.per_copy, a.amp_min, a.amp_max, a.opt_tm)
        for i in range(int(res.get("PRIMER_PAIR_NUM_RETURNED", 0))):
            key = (res[f"PRIMER_LEFT_{i}_SEQUENCE"], res[f"PRIMER_RIGHT_{i}_SEQUENCE"])
            cand.setdefault(key, {
                "size": res[f"PRIMER_PAIR_{i}_PRODUCT_SIZE"],
                "tm_f": round(res[f"PRIMER_LEFT_{i}_TM"], 1),
                "tm_r": round(res[f"PRIMER_RIGHT_{i}_TM"], 1),
                "gc_f": round(res[f"PRIMER_LEFT_{i}_GC_PERCENT"], 1),
                "gc_r": round(res[f"PRIMER_RIGHT_{i}_GC_PERCENT"], 1)})
    print(f"[top] {len(cand)} unique candidate pairs from {len(reps)} copies; scoring coverage...")

    # --- score EVERY candidate by combined coverage
    scored = []
    for (f, r), m in cand.items():
        covered = [i for i, s in enumerate(seqs) if _amplifies(f, r, s, lo, hi, a.max_mm)]
        cov = len(covered); ecov = sum(w[i] for i in covered)
        score = a.w_seq * (cov / tot_c) + a.w_expr * (ecov / tot_e)
        scored.append((score, cov, ecov, f, r, m))
    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[:a.top_n]

    # --- genome specificity for the top N only (in-memory cache -> fast)
    cache = None
    if a.genome:
        try:
            from te_genome import GenomeCache
            cache = GenomeCache(a.genome); cache.load()
        except Exception as e:
            print(f"[top] genome cache unavailable ({e}); genome_hits = NA")
            cache = None

    os.makedirs(a.out, exist_ok=True)
    out_csv = os.path.join(a.out, f"{a.family}_top{a.top_n}_primers.csv")
    with open(out_csv, "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["rank", "fwd", "rev", "amplicon_bp", "tm_fwd", "tm_rev",
                      "gc_fwd", "gc_rev", "copies_amplified", "pct_family",
                      "expr_covered", "pct_expr", "combined_score",
                      "fwd_genome_hits", "rev_genome_hits"])
        for rank, (score, cov, ecov, f, r, m) in enumerate(top, 1):
            hf = genome_hits(f, a.genome, cache) if a.genome else "NA"
            hr = genome_hits(r, a.genome, cache) if a.genome else "NA"
            wtr.writerow([rank, f, r, m["size"], m["tm_f"], m["tm_r"], m["gc_f"], m["gc_r"],
                          cov, round(100 * cov / tot_c, 1),
                          round(ecov, 3), round(100 * ecov / tot_e, 1) if weighted else "NA",
                          round(score, 4), hf, hr])
    best = top[0]
    print(f"[top] wrote {len(top)} pairs -> {out_csv}")
    print(f"[top] #1: {best[1]}/{int(tot_c)} copies ({round(100*best[1]/tot_c,1)}%)"
          + (f", {round(100*best[2]/tot_e,1)}% expression" if weighted else ""))


if __name__ == "__main__":
    main()
