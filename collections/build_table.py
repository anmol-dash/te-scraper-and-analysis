#!/usr/bin/env python
"""One CSV of every qPCR pair with the evidence behind its coverage claim.

Per pair we recompute (not copy) coverage from the actual copy sequences, so
the copies/expression numbers in the table are reproducible from the inputs,
and we attach the genome-wide off-target counts already scanned, split into
sites that land inside a copy of the family (expected for a multi-copy assay)
and sites that do not (the ones that can produce a spurious product).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/Users/anmol/Documents/utility_result")
from te_qpcr_coverage import _amplifies                     # noqa: E402

AMP_LO_F, AMP_HI_F = 0.6, 1.6          # same tolerance te_qpcr_coverage uses
MAX_MM = 3

FAMS = {
    "Harlequin-int": "/Users/anmol/guides/Harlequin-int/harlequin-int_expr.csv",
    "MER21C":        "/Users/anmol/guides/MER21C/mer21c_expr.csv",
}

rows = []
for fam, seq_csv in FAMS.items():
    d = pd.read_csv(seq_csv)
    d = d[d["Seq"].astype(str).str.len() > 0].reset_index(drop=True)
    seqs = d["Seq"].astype(str).str.upper().tolist()
    expr = d["expression"].astype(float).values
    tot_cop, tot_expr = len(seqs), float(expr.sum())

    pairs = pd.read_csv(f"/Users/anmol/qpcr/{fam}/{fam}_qpcr_pairs.csv")
    sites = pd.read_csv(f"/Users/anmol/qpcr/{fam}/{fam}_offtarget_sites.csv")

    # family footprint on the genome, for the on/off-family split
    fam_iv = {}
    for c, s, e in zip(d["Chromosome"], d["Start"], d["Stop"]):
        fam_iv.setdefault(c, []).append((int(s), int(e)))
    for c in fam_iv:
        fam_iv[c].sort()

    def on_family(chrom, start, end):
        for a, b in fam_iv.get(chrom, ()):
            if not (end < a or start > b):
                return True
        return False

    sites["on_family"] = [on_family(c, s, e) for c, s, e
                          in zip(sites["chrom"], sites["start"], sites["end"])]

    ot = {}
    for p, g in sites.groupby("primer"):
        ot[p] = {
            "sites": len(g),
            "on_fam": int(g["on_family"].sum()),
            "off_fam": int((~g["on_family"]).sum()),
            "off_fam_0mm": int(((~g["on_family"]) & (g["n_mismatch"] == 0)).sum()),
            "off_fam_le1mm": int(((~g["on_family"]) & (g["n_mismatch"] <= 1)).sum()),
        }

    cum = set()
    for r in pairs.itertuples():
        lo = int(r.amplicon_bp * AMP_LO_F)
        hi = int(r.amplicon_bp * AMP_HI_F)
        hit = {i for i, s in enumerate(seqs)
               if _amplifies(r.fwd.upper(), r.rev.upper(), s, lo, hi, MAX_MM)}
        cum |= hit
        f, v = ot.get(r.fwd, {}), ot.get(r.rev, {})
        rows.append({
            "family": fam,
            "pair": f"{fam}_pair{r.pair}",
            "cluster": r.cluster,
            "fwd": r.fwd,
            "rev": r.rev,
            "amplicon_bp": r.amplicon_bp,
            "tm_fwd": r.tm_fwd, "tm_rev": r.tm_rev,
            "gc_fwd": r.gc_fwd, "gc_rev": r.gc_rev,
            # ---- coverage evidence ----
            "copies_amplified": len(hit),
            "family_copies": tot_cop,
            "pct_copies": round(100 * len(hit) / tot_cop, 2),
            "expr_amplified_cpm": round(float(expr[sorted(hit)].sum()), 2),
            "family_expr_cpm": round(tot_expr, 2),
            "pct_expression": round(100 * float(expr[sorted(hit)].sum()) / tot_expr, 2),
            "cum_copies": len(cum),
            "cum_pct_copies": round(100 * len(cum) / tot_cop, 2),
            "cum_pct_expression": round(100 * float(expr[sorted(cum)].sum()) / tot_expr, 2),
            # ---- specificity evidence ----
            "fwd_sites_total": f.get("sites", 0),
            "fwd_sites_on_family": f.get("on_fam", 0),
            "fwd_sites_off_family": f.get("off_fam", 0),
            "fwd_off_family_0mm": f.get("off_fam_0mm", 0),
            "fwd_off_family_le1mm": f.get("off_fam_le1mm", 0),
            "rev_sites_total": v.get("sites", 0),
            "rev_sites_on_family": v.get("on_fam", 0),
            "rev_sites_off_family": v.get("off_fam", 0),
            "rev_off_family_0mm": v.get("off_fam_0mm", 0),
            "rev_off_family_le1mm": v.get("off_fam_le1mm", 0),
        })

out = pd.DataFrame(rows)
out["specificity_flag"] = [
    "clean" if (a + b) <= 100 else ("watch" if (a + b) <= 400 else "dirty")
    for a, b in zip(out.fwd_sites_off_family, out.rev_sites_off_family)]
p = "/Users/anmol/qpcr/QPCR_PAIRS_WITH_EVIDENCE.csv"
out.to_csv(p, index=False)
print(f"wrote {p}  ({len(out)} pairs, {len(out.columns)} columns)")
print(out[["pair", "cluster", "amplicon_bp", "copies_amplified", "pct_copies",
           "pct_expression", "cum_pct_expression", "fwd_sites_off_family",
           "rev_sites_off_family", "specificity_flag"]].to_string(index=False))
