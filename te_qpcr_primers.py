#!/usr/bin/env python3
"""te_qpcr_primers.py --- true qPCR primer PAIRS for a TE family.

te_primers.py designs specificity-vetted *candidate* primer sequences but does
not pair them into amplicons. This adds that: build a majority-rule consensus
from the family copies, run primer3 for short qPCR amplicons (Tm ~60C, 80-150 bp),
then reuse the existing GAMECA genome search (te_genome.GenomeCache /
te_primers.search_seq_chromosomal) to count genome hits per primer so you can
flag off-target risk before ordering.

Input : a CSV with a 'Seq' column (e.g. a query.py family sequences CSV) OR a FASTA.
Output: <out>/<family>_qpcr_pairs.csv  (fwd/rev, Tm, amplicon size, genome hits)

  python te_qpcr_primers.py --input LTR5_Hs_sequences.csv --family LTR5_Hs \
      --genome /path/hg38.fa --out results
"""
import argparse, csv, os, sys


def read_seqs(path):
    seqs = []
    if path.lower().endswith((".fa", ".fasta", ".fna")):
        cur = []
        with open(path) as fh:
            for line in fh:
                if line.startswith(">"):
                    if cur: seqs.append("".join(cur)); cur = []
                else:
                    cur.append(line.strip())
        if cur: seqs.append("".join(cur))
    else:  # CSV/TSV with a Seq column
        with open(path, newline="") as fh:
            sniff = fh.read(2048); fh.seek(0)
            delim = "\t" if sniff.count("\t") > sniff.count(",") else ","
            rdr = csv.DictReader(fh, delimiter=delim)
            col = next((c for c in (rdr.fieldnames or []) if c.lower() == "seq"), None)
            if not col:
                sys.exit(f"no 'Seq' column in {path} (cols: {rdr.fieldnames})")
            for row in rdr:
                s = (row.get(col) or "").strip().upper()
                if s and set(s) <= set("ACGTN"):
                    seqs.append(s)
    return [s for s in seqs if len(s) >= 40]


def majority_consensus(seqs):
    """Ungapped majority consensus (same approach as run_grna_offtarget)."""
    if not seqs:
        sys.exit("no usable sequences")
    L = max(len(s) for s in seqs)
    out = []
    for i in range(L):
        cnt = {}
        for s in seqs:
            if i < len(s) and s[i] != "N":
                cnt[s[i]] = cnt.get(s[i], 0) + 1
        out.append(max(cnt, key=cnt.get) if cnt else "N")
    return "".join(out)


def genome_hits(primer, genome_fa, cache):
    """Count genome occurrences via the existing GAMECA search (best-effort)."""
    try:
        from te_primers import search_seq_chromosomal
        r = search_seq_chromosomal(primer, fasta=genome_fa, genome_cache=cache)
        # normalize a few possible return shapes to an int hit count
        if isinstance(r, (list, tuple, set)): return len(r)
        if isinstance(r, dict): return int(r.get("hits", r.get("count", 0)))
        if isinstance(r, int): return r
        return int(getattr(r, "hits", 0) or 0)
    except Exception as e:
        return f"NA({type(e).__name__})"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="family Seq CSV or FASTA")
    ap.add_argument("--family", default="FAMILY")
    ap.add_argument("--out", default="results")
    ap.add_argument("--genome", default=None, help="genome FASTA for specificity (optional)")
    ap.add_argument("--n-return", type=int, default=10, help="primer pairs to return")
    ap.add_argument("--amp-min", type=int, default=80)
    ap.add_argument("--amp-max", type=int, default=150)
    ap.add_argument("--opt-tm", type=float, default=60.0)
    a = ap.parse_args()

    seqs = read_seqs(a.input)
    cons = majority_consensus(seqs).replace("N", "")  # primer3 dislikes runs of N
    print(f"[qpcr] {a.family}: {len(seqs)} copies, consensus {len(cons)} bp")

    try:
        import primer3
    except ImportError:
        sys.exit("primer3-py not available in this environment (pip install primer3-py, "
                 "or run inside a container that has it)")

    seq_args = {"SEQUENCE_ID": a.family, "SEQUENCE_TEMPLATE": cons}
    global_args = {
        "PRIMER_OPT_SIZE": 20, "PRIMER_MIN_SIZE": 18, "PRIMER_MAX_SIZE": 25,
        "PRIMER_OPT_TM": a.opt_tm, "PRIMER_MIN_TM": a.opt_tm - 3, "PRIMER_MAX_TM": a.opt_tm + 3,
        "PRIMER_MIN_GC": 40.0, "PRIMER_MAX_GC": 60.0,
        "PRIMER_PRODUCT_SIZE_RANGE": [[a.amp_min, a.amp_max]],
        "PRIMER_NUM_RETURN": a.n_return,
    }
    # primer3-py renamed the entry point across versions; try both.
    try:
        res = primer3.bindings.design_primers(seq_args, global_args)
    except AttributeError:
        res = primer3.bindings.designPrimers(seq_args, global_args)

    n = int(res.get("PRIMER_PAIR_NUM_RETURNED", 0))
    if n == 0:
        sys.exit(f"[qpcr] primer3 returned no pairs for {a.family} "
                 "(try widening --amp-max / --opt-tm tolerance)")

    cache = None
    if a.genome:
        try:
            from te_genome import GenomeCache
            cache = GenomeCache(a.genome)
        except Exception as e:
            print(f"[qpcr] genome cache unavailable ({e}); skipping specificity")

    os.makedirs(a.out, exist_ok=True)
    out_csv = os.path.join(a.out, f"{a.family}_qpcr_pairs.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair", "fwd", "rev", "amplicon_bp", "tm_fwd", "tm_rev",
                    "gc_fwd", "gc_rev", "fwd_genome_hits", "rev_genome_hits"])
        for i in range(n):
            fwd = res[f"PRIMER_LEFT_{i}_SEQUENCE"]
            rev = res[f"PRIMER_RIGHT_{i}_SEQUENCE"]
            hf = genome_hits(fwd, a.genome, cache) if a.genome else "NA"
            hr = genome_hits(rev, a.genome, cache) if a.genome else "NA"
            w.writerow([i, fwd, rev, res[f"PRIMER_PAIR_{i}_PRODUCT_SIZE"],
                        round(res[f"PRIMER_LEFT_{i}_TM"], 1),
                        round(res[f"PRIMER_RIGHT_{i}_TM"], 1),
                        round(res[f"PRIMER_LEFT_{i}_GC_PERCENT"], 1),
                        round(res[f"PRIMER_RIGHT_{i}_GC_PERCENT"], 1), hf, hr])
    print(f"[qpcr] wrote {n} pairs -> {out_csv}")


if __name__ == "__main__":
    main()
