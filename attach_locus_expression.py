#!/usr/bin/env python3
"""attach_locus_expression.py --- add a per-copy 'expression' column to a TE
family sequences CSV by matching each copy's genomic coordinates to the
per-locus CPM table from submit_locus_expression.sh (--merge).

Then feed the result to te_qpcr_primers.py with --expr-cols expression so the
greedy set targets the copies that actually drive the family's expression.

  python3 attach_locus_expression.py --expr locus_expression.tsv \
      --seqs LTR5_Hs_clustered.csv --out LTR5_Hs_expr.csv [--cpm-col cpm_mean] [--tol 20]
"""
import argparse, csv, sys

CHROM_KEYS = ("chrom", "chromosome", "chr", "genoname", "seqnames")
START_KEYS = ("start", "genostart", "chromstart", "chrom_start", "genomicstart")


def _find(cols, keys):
    low = {c.lower(): c for c in cols}
    for k in keys:
        if k in low: return low[k]
    return None


def load_expr(path, cpm_col):
    with open(path, newline="") as fh:
        rdr = csv.DictReader(fh, delimiter="\t")
        cols = rdr.fieldnames or []
        if cpm_col not in cols:
            sys.exit(f"--cpm-col '{cpm_col}' not in {path} (cols: {cols})")
        by_chrom = {}
        for r in rdr:
            try:
                s = int(float(r["start"])); v = float(r[cpm_col])
            except (ValueError, KeyError):
                continue
            by_chrom.setdefault(r["chrom"], []).append((s, v))
    for c in by_chrom:
        by_chrom[c].sort()
    return by_chrom


def nearest(by_chrom, chrom, start, tol):
    import bisect
    arr = by_chrom.get(chrom)
    if not arr:
        return None
    starts = [a[0] for a in arr]
    i = bisect.bisect_left(starts, start)
    best = None
    for j in (i - 1, i, i + 1):
        if 0 <= j < len(arr):
            d = abs(arr[j][0] - start)
            if d <= tol and (best is None or d < best[0]):
                best = (d, arr[j][1])
    return best[1] if best else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expr", required=True)
    ap.add_argument("--seqs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cpm-col", default="cpm_mean")
    ap.add_argument("--tol", type=int, default=20, help="max bp between copy start and locus start")
    a = ap.parse_args()

    by_chrom = load_expr(a.expr, a.cpm_col)
    with open(a.seqs, newline="") as fh:
        sniff = fh.read(2048); fh.seek(0)
        delim = "\t" if sniff.count("\t") > sniff.count(",") else ","
        rdr = csv.DictReader(fh, delimiter=delim)
        cols = rdr.fieldnames or []
        cc = _find(cols, CHROM_KEYS); sc = _find(cols, START_KEYS)
        if not cc or not sc:
            sys.exit(f"could not find chrom/start columns in {a.seqs} (cols: {cols})")
        rows = list(rdr)

    matched = 0
    for r in rows:
        chrom = (r.get(cc) or "").strip()
        try:
            start = int(float(r.get(sc)))
        except (TypeError, ValueError):
            r["expression"] = 0.0; continue
        v = nearest(by_chrom, chrom, start, a.tol)
        r["expression"] = round(v, 4) if v is not None else 0.0
        matched += v is not None

    fields = cols + (["expression"] if "expression" not in cols else [])
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"[attach] {a.seqs}: matched {matched}/{len(rows)} copies to a locus "
          f"(chrom='{cc}', start='{sc}', tol={a.tol}) -> {a.out}")
    if matched == 0:
        print("[attach] WARNING: 0 matches -- check coordinate columns / genome build / tol")


if __name__ == "__main__":
    main()
