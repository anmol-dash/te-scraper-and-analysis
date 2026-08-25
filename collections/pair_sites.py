#!/usr/bin/env python
"""Pair off-target priming SITES into predicted off-target AMPLICONS.

A priming site is not a product. A product needs two sites on opposite strands,
pointing at each other, within a workable distance. This enumerates those, so
the specificity number reported is "products this pair can make" rather than
"places a primer can land".

Counted per pair, for every primer combination that can close a product:
fwd+rev (the intended one), fwd+fwd and rev+rev (real PCR artefacts --- one
primer can prime both ends if the genome supplies an inverted copy).

Two windows:
  designed  0.6x - 1.6x the designed amplicon, i.e. a product that would be
            indistinguishable from the real one on a melt curve / gel
  broad     50 - 1000 bp, anything a qPCR reaction could plausibly make
"""
import sys
from bisect import bisect_left, bisect_right
from collections import defaultdict

import pandas as pd

BROAD_LO, BROAD_HI = 50, 1000

FAMS = {
    "Harlequin-int": "/Users/anmol/guides/Harlequin-int/harlequin-int_expr.csv",
    "MER21C":        "/Users/anmol/guides/MER21C/mer21c_expr.csv",
}

all_amp = []
summary = []

for fam, seq_csv in FAMS.items():
    d = pd.read_csv(seq_csv)
    fam_iv = defaultdict(list)
    for c, s, e in zip(d["Chromosome"], d["Start"], d["Stop"]):
        fam_iv[c].append((int(s), int(e)))
    for c in fam_iv:
        fam_iv[c].sort()

    def on_family(chrom, start, end):
        iv = fam_iv.get(chrom)
        if not iv:
            return False
        i = bisect_right(iv, (end, float("inf")))
        for a, b in iv[max(0, i - 40):i]:
            if not (end < a or start > b):
                return True
        return False

    sites = pd.read_csv(f"/Users/anmol/qpcr/{fam}/{fam}_offtarget_sites.csv")
    pairs = pd.read_csv(f"/Users/anmol/qpcr/{fam}/{fam}_qpcr_pairs.csv")

    # index: (primer, chrom, strand) -> sorted list of (start, end, n_mismatch)
    idx = defaultdict(list)
    for r in sites.itertuples():
        idx[(r.primer, r.chrom, r.strand)].append((int(r.start), int(r.end), int(r.n_mismatch)))
    for k in idx:
        idx[k].sort()

    for r in pairs.itertuples():
        fwd, rev = r.fwd.upper(), r.rev.upper()
        dlo, dhi = int(r.amplicon_bp * 0.6), int(r.amplicon_bp * 1.6)
        found = []
        combos = [(fwd, rev, "fwd+rev"), (fwd, fwd, "fwd+fwd"), (rev, rev, "rev+rev")]
        for p_plus, p_minus, kind in combos:
            chroms = {c for (p, c, s) in idx if p == p_plus}
            for chrom in chroms:
                plus = idx.get((p_plus, chrom, "+"), [])
                minus = idx.get((p_minus, chrom, "-"), [])
                if not plus or not minus:
                    continue
                mstarts = [m[0] for m in minus]
                for ps, pe, pmm in plus:
                    # product runs from the + site's start to the - site's end
                    lo_i = bisect_left(mstarts, ps)
                    hi_i = bisect_right(mstarts, ps + BROAD_HI)
                    for ms, me, mmm in minus[lo_i:hi_i]:
                        plen = me - ps + 1
                        if not (BROAD_LO <= plen <= BROAD_HI):
                            continue
                        found.append({
                            "family": fam, "pair": f"{fam}_pair{r.pair}",
                            "kind": kind, "chrom": chrom,
                            "start": ps, "end": me, "product_bp": plen,
                            "mm_plus": pmm, "mm_minus": mmm,
                            "mm_total": pmm + mmm,
                            "in_designed_window": dlo <= plen <= dhi,
                            "on_family": on_family(chrom, ps, me),
                        })
        f = pd.DataFrame(found)
        if len(f):
            f = f.drop_duplicates(["chrom", "start", "end", "kind"])
        all_amp.append(f)
        off = f[~f.on_family] if len(f) else f
        summary.append({
            "family": fam, "pair": f"{fam}_pair{r.pair}", "cluster": r.cluster,
            "amplicon_bp": r.amplicon_bp,
            "amplicons_total": len(f),
            "amplicons_on_family": int(f.on_family.sum()) if len(f) else 0,
            "amplicons_off_family": len(off),
            "off_family_designed_window": int(off.in_designed_window.sum()) if len(off) else 0,
            "off_family_le2mm": int((off.mm_total <= 2).sum()) if len(off) else 0,
            "off_family_0mm": int((off.mm_total == 0).sum()) if len(off) else 0,
            "off_family_fwd_rev": int((off.kind == "fwd+rev").sum()) if len(off) else 0,
        })

amp = pd.concat([a for a in all_amp if len(a)], ignore_index=True)
amp.to_csv("/Users/anmol/qpcr/QPCR_OFFTARGET_AMPLICONS.csv", index=False)
s = pd.DataFrame(summary)
s.to_csv("/Users/anmol/qpcr/QPCR_OFFTARGET_AMPLICON_SUMMARY.csv", index=False)
print(f"{len(amp)} predicted amplicons total\n")
print(s.to_string(index=False))
