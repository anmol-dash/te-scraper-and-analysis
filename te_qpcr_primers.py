#!/usr/bin/env python3
"""te_qpcr_primers.py --- true qPCR primer PAIRS for a TE family.

te_primers.py designs specificity-vetted *candidate* primer sequences but does
not pair them into amplicons. This adds that: build a majority-rule consensus
from the family copies, run primer3 for short qPCR amplicons (Tm ~60C, 80-150 bp),
then reuse the existing GAMECA genome search (te_genome.GenomeCache /
te_primers.search_seq_chromosomal) to count genome hits per primer.

Divergent families (e.g. HERVK-int) are poorly covered by one consensus pair.
With --n-clusters K, design one pair per sub-cluster (reusing the 'Cluster'
labels query.py already assigned) so K pairs together cover more of the family;
te_qpcr_coverage.py then reports each pair's coverage and the union.

Input : a CSV with a 'Seq' column (+ optional 'Cluster') OR a FASTA.
Output: <out>/<family>_qpcr_pairs.csv

  python te_qpcr_primers.py --input LTR5_Hs_sequences.csv --family LTR5_Hs \
      --genome /path/hg38.fa --out results [--n-clusters 3]
"""
import argparse, csv, os, sys


def read_records(path):
    """Return (seqs, clusters): clusters is a parallel list or None (FASTA/no col)."""
    seqs, clusters = [], []
    have_cluster = False
    if path.lower().endswith((".fa", ".fasta", ".fna")):
        cur = []
        with open(path) as fh:
            for line in fh:
                if line.startswith(">"):
                    if cur: seqs.append("".join(cur)); clusters.append(None); cur = []
                else:
                    cur.append(line.strip())
        if cur: seqs.append("".join(cur)); clusters.append(None)
    else:
        with open(path, newline="") as fh:
            sniff = fh.read(2048); fh.seek(0)
            delim = "\t" if sniff.count("\t") > sniff.count(",") else ","
            rdr = csv.DictReader(fh, delimiter=delim)
            cols = rdr.fieldnames or []
            scol = next((c for c in cols if c.lower() == "seq"), None)
            ccol = next((c for c in cols if c.lower() == "cluster"), None)
            have_cluster = ccol is not None
            if not scol:
                sys.exit(f"no 'Seq' column in {path} (cols: {cols})")
            for row in rdr:
                s = (row.get(scol) or "").strip().upper()
                if s and set(s) <= set("ACGTN") and len(s) >= 40:
                    seqs.append(s)
                    clusters.append(row.get(ccol) if have_cluster else None)
    return seqs, (clusters if have_cluster else None)


def majority_consensus(seqs):
    if not seqs:
        return ""
    L = max(len(s) for s in seqs)
    out = []
    for i in range(L):
        cnt = {}
        for s in seqs:
            if i < len(s) and s[i] != "N":
                cnt[s[i]] = cnt.get(s[i], 0) + 1
        out.append(max(cnt, key=cnt.get) if cnt else "N")
    return "".join(out)


def pick_groups(seqs, clusters, k):
    """{label: [seqs]} for the k largest clusters; whole family if k<=1/no labels."""
    if k <= 1 or clusters is None:
        return {"all": seqs}
    groups = {}
    for s, c in zip(seqs, clusters):
        groups.setdefault(str(c), []).append(s)
    top = sorted(groups, key=lambda c: len(groups[c]), reverse=True)[:k]
    return {c: groups[c] for c in top}


def genome_hits(primer, genome_fa, cache):
    try:
        from te_primers import search_seq_chromosomal
        r = search_seq_chromosomal(primer, fasta=genome_fa, genome_cache=cache)
        if isinstance(r, (list, tuple, set)): return len(r)
        if isinstance(r, dict): return int(r.get("hits", r.get("count", 0)))
        if isinstance(r, int): return r
        return int(getattr(r, "hits", 0) or 0)
    except Exception as e:
        return f"NA({type(e).__name__})"


def design_on(consensus, family, n_return, amp_min, amp_max, opt_tm):
    import primer3
    seq_args = {"SEQUENCE_ID": family, "SEQUENCE_TEMPLATE": consensus}
    global_args = {
        "PRIMER_OPT_SIZE": 20, "PRIMER_MIN_SIZE": 18, "PRIMER_MAX_SIZE": 25,
        "PRIMER_OPT_TM": opt_tm, "PRIMER_MIN_TM": opt_tm - 3, "PRIMER_MAX_TM": opt_tm + 3,
        "PRIMER_MIN_GC": 40.0, "PRIMER_MAX_GC": 60.0,
        "PRIMER_PRODUCT_SIZE_RANGE": [[amp_min, amp_max]],
        "PRIMER_NUM_RETURN": n_return,
    }
    try:
        return primer3.bindings.design_primers(seq_args, global_args)
    except AttributeError:
        return primer3.bindings.designPrimers(seq_args, global_args)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--family", default="FAMILY")
    ap.add_argument("--out", default="results")
    ap.add_argument("--genome", default=None)
    ap.add_argument("--n-clusters", type=int, default=1,
                    help="design one pair per sub-cluster (K largest); default 1 = whole family")
    ap.add_argument("--pairs-per-cluster", type=int, default=1)
    ap.add_argument("--n-return", type=int, default=10, help="pairs when --n-clusters<=1")
    ap.add_argument("--amp-min", type=int, default=80)
    ap.add_argument("--amp-max", type=int, default=150)
    ap.add_argument("--opt-tm", type=float, default=60.0)
    a = ap.parse_args()

    seqs, clusters = read_records(a.input)
    if not seqs:
        sys.exit("no usable sequences")
    try:
        import primer3  # noqa: F401
    except ImportError:
        sys.exit("primer3-py not available (pip install primer3-py or run in a container that has it)")

    groups = pick_groups(seqs, clusters, a.n_clusters)
    if a.n_clusters > 1 and clusters is None:
        print(f"[qpcr] {a.family}: no Cluster column -> designing from whole family instead")
    print(f"[qpcr] {a.family}: {len(seqs)} copies, {len(groups)} group(s)")

    cache = None
    if a.genome:
        try:
            from te_genome import GenomeCache
            cache = GenomeCache(a.genome)
        except Exception as e:
            print(f"[qpcr] genome cache unavailable ({e}); skipping specificity")

    per_group = a.pairs_per_cluster if a.n_clusters > 1 else a.n_return
    rows = []
    for label, gseqs in groups.items():
        cons = majority_consensus(gseqs).replace("N", "")
        if len(cons) < a.amp_min:
            print(f"[qpcr]  cluster {label}: consensus too short ({len(cons)}bp) -> skip"); continue
        res = design_on(cons, a.family, per_group, a.amp_min, a.amp_max, a.opt_tm)
        n = int(res.get("PRIMER_PAIR_NUM_RETURNED", 0))
        if n == 0:
            print(f"[qpcr]  cluster {label} ({len(gseqs)} copies): no primer3 pair"); continue
        for i in range(min(n, per_group)):
            fwd = res[f"PRIMER_LEFT_{i}_SEQUENCE"]; rev = res[f"PRIMER_RIGHT_{i}_SEQUENCE"]
            rows.append({
                "pair": len(rows), "cluster": label, "cluster_size": len(gseqs),
                "fwd": fwd, "rev": rev,
                "amplicon_bp": res[f"PRIMER_PAIR_{i}_PRODUCT_SIZE"],
                "tm_fwd": round(res[f"PRIMER_LEFT_{i}_TM"], 1),
                "tm_rev": round(res[f"PRIMER_RIGHT_{i}_TM"], 1),
                "gc_fwd": round(res[f"PRIMER_LEFT_{i}_GC_PERCENT"], 1),
                "gc_rev": round(res[f"PRIMER_RIGHT_{i}_GC_PERCENT"], 1),
                "fwd_genome_hits": genome_hits(fwd, a.genome, cache) if a.genome else "NA",
                "rev_genome_hits": genome_hits(rev, a.genome, cache) if a.genome else "NA",
            })
    if not rows:
        sys.exit(f"[qpcr] no pairs produced for {a.family}")

    os.makedirs(a.out, exist_ok=True)
    out_csv = os.path.join(a.out, f"{a.family}_qpcr_pairs.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[qpcr] wrote {len(rows)} pairs -> {out_csv}")


if __name__ == "__main__":
    main()
