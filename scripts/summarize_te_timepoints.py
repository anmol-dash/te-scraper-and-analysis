#!/usr/bin/env python3
"""summarize_te_timepoints.py --- subfamily-level TE expression per sample group.

Reads the TEcount .cntTable files produced by submit_locus_expression.sh
(ALSO_TECOUNT=1) or submit_hervk_ccle_requant.sh, CPM-normalizes each library
(raw counts are not comparable across libraries), groups samples by the
manifest's group column, and reports mean CPM per group plus log2 fold change
and a Welch t-test for the subfamilies of interest.

Stdlib only -- no numpy/scipy -- so it runs on the login node or in any of the
tool containers.

  python3 summarize_te_timepoints.py --counts $WORK/counts \
      --manifest scripts/srp090091_manifest.tsv --families LTR66 LTR10G \
      --group-col 2 --sample-col 1 --out $WORK/te_timepoint_summary.tsv
"""
import argparse
import csv
import glob
import math
import os
import sys


def read_cnt_table(path):
    """TEcount .cntTable -> {feature: count}. Feature is 'name:family:class' for TEs."""
    counts = {}
    with open(path) as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if len(row) < 2:
                continue
            try:
                counts[row[0]] = float(row[1])
            except ValueError:
                continue  # header line
    return counts


def _betacf(a, b, x, itmax=200, eps=3e-16):
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + b * math.log1p(-x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


def welch_t(a, b):
    """Welch's unequal-variance t-test. Returns (t, two-sided p, df).

    Uses the Welch-Satterthwaite df and the exact Student-t survival function
    (via the regularized incomplete beta) -- a normal approximation would report
    wildly overconfident p-values at these sample sizes (n=7 per group).
    """
    na, nb = len(a), len(b)
    nan = float("nan")
    if na < 2 or nb < 2:
        return nan, nan, nan
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    se2 = va / na + vb / nb
    if se2 <= 0:
        return nan, nan, nan
    t = (ma - mb) / math.sqrt(se2)
    denom = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    if denom <= 0:
        return t, nan, nan
    df = se2 ** 2 / denom
    p = betainc(df / 2.0, 0.5, df / (df + t * t))   # two-sided
    return t, p, df


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", required=True, help="dir of <sample>.cntTable")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--families", nargs="+", required=True,
                    help="subfamily repNames, e.g. LTR66 LTR10G")
    ap.add_argument("--sample-col", type=int, default=1, help="1-based manifest column with the sample label")
    ap.add_argument("--group-col", type=int, default=2, help="1-based manifest column with the group label")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    groups, order = {}, []
    with open(a.manifest) as fh:
        rows = [ln.rstrip("\n").split("\t") for ln in fh if ln.strip()]
    for r in rows[1:]:
        if len(r) < max(a.sample_col, a.group_col):
            continue
        sample, group = r[a.sample_col - 1], r[a.group_col - 1]
        groups[sample] = group
        if group not in order:
            order.append(group)
    order.sort()

    # CPM per sample; library size = all assigned counts (genes + TEs)
    cpm, missing = {}, []
    for sample in sorted(groups):
        path = os.path.join(a.counts, f"{sample}.cntTable")
        if not os.path.exists(path):
            missing.append(sample)
            continue
        counts = read_cnt_table(path)
        lib = sum(counts.values())
        if lib <= 0:
            missing.append(sample)
            continue
        cpm[sample] = {k: v / lib * 1e6 for k, v in counts.items()}
    if missing:
        sys.stderr.write(f"[summary] WARNING: no usable counts for: {', '.join(missing)}\n")
    if not cpm:
        sys.exit(f"[summary] no .cntTable files found under {a.counts}")

    any_sample = next(iter(cpm.values()))
    feature_of = {}
    for fam in a.families:
        # TEcount names TE features '<repName>:<repFamily>:<repClass>'
        hit = next((f for f in any_sample if f.split(":")[0] == fam), None)
        feature_of[fam] = hit
        if hit is None:
            sys.stderr.write(
                f"[summary] WARNING: '{fam}' absent from the count tables "
                "(check the repName spelling / that it survived the TE GTF filter)\n")

    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        header = ["subfamily", "feature"]
        for g in order:
            header += [f"mean_cpm_{g}", f"sd_cpm_{g}", f"n_{g}"]
        if len(order) == 2:
            header += ["log2FC", "welch_t", "welch_df", "p_value"]
        header += ["per_sample_cpm"]
        w.writerow(header)

        for fam in a.families:
            feat = feature_of[fam]
            row, vals_by_group, per_sample = [fam, feat or "ABSENT"], {}, []
            for g in order:
                vals = [cpm[s].get(feat, 0.0) for s in sorted(cpm) if groups[s] == g] if feat else []
                vals_by_group[g] = vals
                n = len(vals)
                mean = sum(vals) / n if n else 0.0
                sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
                row += [f"{mean:.4f}", f"{sd:.4f}", str(n)]
            if len(order) == 2:
                ga, gb = order
                va, vb = vals_by_group[ga], vals_by_group[gb]
                ma = sum(va) / len(va) if va else 0.0
                mb = sum(vb) / len(vb) if vb else 0.0
                # gb vs ga; pseudocount keeps the ratio finite at zero counts
                lfc = math.log2((mb + 1e-6) / (ma + 1e-6))
                t, p, df = welch_t(vb, va)
                row += [f"{lfc:.4f}",
                        "NA" if math.isnan(t) else f"{t:.3f}",
                        "NA" if math.isnan(df) else f"{df:.2f}",
                        "NA" if math.isnan(p) else f"{p:.4g}"]
            for s in sorted(cpm):
                per_sample.append(f"{s}={cpm[s].get(feat, 0.0):.3f}" if feat else f"{s}=NA")
            row.append(";".join(per_sample))
            w.writerow(row)

    print(f"[summary] {len(cpm)} samples, groups: {', '.join(f'{g}(n={sum(1 for s in cpm if groups[s]==g)})' for g in order)}")
    print(f"[summary] wrote {a.out}\n")
    with open(a.out) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            print("  " + "\t".join(f[:-1]))


if __name__ == "__main__":
    main()
