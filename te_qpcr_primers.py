#!/usr/bin/env python3
"""te_qpcr_primers.py --- true qPCR primer PAIRS for a TE family, coverage-optimized.

Consensus of divergent copies makes chimeric primers, so instead we run primer3
on real copies, then GREEDILY pick a set of pairs that maximizes the UNION of
family copies amplified (each pair's covered-copy set is computed with
te_qpcr_coverage, incl. 3'-end-protected mismatch tolerance). If per-locus
expression columns are present (--expr-cols, or auto-detected), the greedy step
maximizes covered EXPRESSION instead of copy count -- so a few pairs can capture
most of the expressed signal even in a divergent family.

Input : CSV with 'Seq' (+ optional 'Cluster' and expression columns) OR FASTA.
Output: <out>/<family>_qpcr_pairs.csv  (per pair: Tm, amplicon, genome hits,
        copies_amplified, pct_family; + cumulative union printed to stdout)

  python te_qpcr_primers.py --input LTR5_Hs_clustered.csv --family LTR5_Hs \
      --genome hg38.fa --out results --n-pairs 3 [--expr-cols TPM]
"""
import argparse, csv, os, sys

_EXPR_HINTS = ("total_expr", "expression", "tpm", "fpkm", "cpm", "expr")


def read_records(path, expr_cols=None):
    """Return (seqs, clusters, weights). clusters/weights are parallel lists or None."""
    seqs, clusters, weights = [], [], []
    if path.lower().endswith((".fa", ".fasta", ".fna")):
        cur = []
        with open(path) as fh:
            for line in fh:
                if line.startswith(">"):
                    if cur: seqs.append("".join(cur)); clusters.append(None); weights.append(1.0); cur = []
                else: cur.append(line.strip())
        if cur: seqs.append("".join(cur)); clusters.append(None); weights.append(1.0)
        return seqs, None, None
    with open(path, newline="") as fh:
        sniff = fh.read(2048); fh.seek(0)
        delim = "\t" if sniff.count("\t") > sniff.count(",") else ","
        rdr = csv.DictReader(fh, delimiter=delim)
        cols = rdr.fieldnames or []
        scol = next((c for c in cols if c.lower() == "seq"), None)
        ccol = next((c for c in cols if c.lower() == "cluster"), None)
        if not scol:
            sys.exit(f"no 'Seq' column in {path} (cols: {cols})")
        if expr_cols:
            ecols = [c for c in cols if c in expr_cols]
        else:
            ecols = [c for c in cols if c.lower() in _EXPR_HINTS]
        for row in rdr:
            s = (row.get(scol) or "").strip().upper()
            if not (s and set(s) <= set("ACGTN") and len(s) >= 40):
                continue
            seqs.append(s)
            clusters.append(row.get(ccol) if ccol else None)
            w = 0.0
            for c in ecols:
                try: w += float(row.get(c) or 0)
                except ValueError: pass
            weights.append(w if ecols else 1.0)
    if ecols and sum(weights) > 0:
        print(f"[qpcr] expression-weighted using columns: {ecols}")
    else:
        weights = None
    return seqs, (clusters if ccol else None), weights


def pick_groups(seqs, clusters, k):
    if k <= 1 or clusters is None:
        return {"all": list(range(len(seqs)))}
    groups = {}
    for i, c in enumerate(clusters):
        groups.setdefault(str(c), []).append(i)
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
    ap.add_argument("--n-pairs", type=int, default=3, help="max pairs in the greedy union set")
    ap.add_argument("--n-clusters", type=int, default=1, help="draw candidates from K largest sub-clusters")
    ap.add_argument("--expr-cols", nargs="+", default=None, help="per-locus expression columns to weight by")
    ap.add_argument("--amp-min", type=int, default=80)
    ap.add_argument("--amp-max", type=int, default=150)
    ap.add_argument("--opt-tm", type=float, default=60.0)
    ap.add_argument("--max-mm", type=int, default=3, help="5'-body mismatch tolerance (3' end exact)")
    a = ap.parse_args()

    seqs, clusters, weights = read_records(a.input, a.expr_cols)
    if not seqs:
        sys.exit("no usable sequences")
    try:
        import primer3  # noqa: F401
    except ImportError:
        sys.exit("primer3-py not available (pip install primer3-py or run in a container that has it)")
    from te_qpcr_coverage import _amplifies

    w = weights if weights else [1.0] * len(seqs)
    total_w = sum(w) or 1.0
    weighted = weights is not None
    lo, hi = int(a.amp_min * 0.6), int(a.amp_max * 1.6)
    groups = pick_groups(seqs, clusters, a.n_clusters)
    if a.n_clusters > 1 and clusters is None:
        print(f"[qpcr] {a.family}: no Cluster column -> whole family")
    print(f"[qpcr] {a.family}: {len(seqs)} copies, {len(groups)} group(s), "
          f"{'expression-weighted' if weighted else 'copy-count'}")

    cache = None
    if a.genome:
        try:
            from te_genome import GenomeCache
            cache = GenomeCache(a.genome)
        except Exception as e:
            print(f"[qpcr] genome cache unavailable ({e}); skipping specificity")

    # candidate pairs from real copies, each with its covered-copy set over the WHOLE family
    cand = {}
    for label, idxs in groups.items():
        reps = sorted((seqs[i] for i in idxs), key=len, reverse=True)[:min(5, len(idxs))]
        for rep in reps:
            if len(rep) < a.amp_min:
                continue
            res = design_on(rep, a.family, 12, a.amp_min, a.amp_max, a.opt_tm)
            for i in range(int(res.get("PRIMER_PAIR_NUM_RETURNED", 0))):
                f = res[f"PRIMER_LEFT_{i}_SEQUENCE"]; r = res[f"PRIMER_RIGHT_{i}_SEQUENCE"]
                if (f, r) in cand:
                    continue
                covered = frozenset(j for j, s in enumerate(seqs) if _amplifies(f, r, s, lo, hi, a.max_mm))
                cand[(f, r)] = {
                    "cluster": label, "covered": covered,
                    "size": res[f"PRIMER_PAIR_{i}_PRODUCT_SIZE"],
                    "tm_f": round(res[f"PRIMER_LEFT_{i}_TM"], 1),
                    "tm_r": round(res[f"PRIMER_RIGHT_{i}_TM"], 1),
                    "gc_f": round(res[f"PRIMER_LEFT_{i}_GC_PERCENT"], 1),
                    "gc_r": round(res[f"PRIMER_RIGHT_{i}_GC_PERCENT"], 1)}
    if not cand:
        sys.exit(f"[qpcr] no primer3 pairs for {a.family}")

    # greedy set-cover: add the pair with the largest marginal (weighted) gain
    covered, chosen = set(), []
    for _ in range(a.n_pairs):
        best, best_gain = None, 0.0
        for key, m in cand.items():
            if key in chosen:
                continue
            gain = sum(w[i] for i in (m["covered"] - covered))
            if gain > best_gain:
                best, best_gain = key, gain
        if best is None or best_gain <= 0:
            break
        chosen.append(best); covered |= cand[best]["covered"]

    rows = []
    for pid, key in enumerate(chosen):
        f, r = key; m = cand[key]
        cov = len(m["covered"])
        rows.append({
            "pair": pid, "cluster": m["cluster"], "fwd": f, "rev": r,
            "amplicon_bp": m["size"], "tm_fwd": m["tm_f"], "tm_rev": m["tm_r"],
            "gc_fwd": m["gc_f"], "gc_rev": m["gc_r"],
            "fwd_genome_hits": genome_hits(f, a.genome, cache) if a.genome else "NA",
            "rev_genome_hits": genome_hits(r, a.genome, cache) if a.genome else "NA",
            "copies_amplified": cov, "pct_family": round(100.0 * cov / len(seqs), 1)})

    os.makedirs(a.out, exist_ok=True)
    out_csv = os.path.join(a.out, f"{a.family}_qpcr_pairs.csv")
    with open(out_csv, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wtr.writeheader(); wtr.writerows(rows)
    uc = len(covered); uw = sum(w[i] for i in covered)
    msg = f"[qpcr] {a.family}: {len(rows)} pairs -> UNION {uc}/{len(seqs)} copies ({round(100*uc/len(seqs),1)}%)"
    if weighted:
        msg += f", {round(100*uw/total_w,1)}% of family expression"
    print(msg + f" -> {out_csv}")


if __name__ == "__main__":
    main()
