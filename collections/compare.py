#!/usr/bin/env python
"""Side-by-side motif ranking: dinucleotide-shuffle vs genomic background."""
import sys
import pandas as pd

TOPN = 100
FAMS = ["LTR66", "MER21C", "Harlequin-int"]
KEEP = ["motif", "consensus", "observed_consensus", "pct_copies",
        "pct_copies_bg", "fold_prevalence", "hits_per_kb", "fdr_q"]
rows = []
for fam in FAMS:
    d = pd.read_csv(f"/Users/anmol/motifcmp/dinuc/motifs_{fam}_enrichment.tsv", sep="\t")
    g = pd.read_csv(f"/Users/anmol/motifcmp/genomic/motifs_{fam}_enrichment.tsv", sep="\t")
    for f in (d, g):
        f.sort_values(["fdr_q", "fold_prevalence", "pct_copies"],
                      ascending=[True, False, False], inplace=True)
        f.reset_index(drop=True, inplace=True)
        f["rank"] = f.index + 1
    m = d[KEEP + ["rank"]].merge(g[KEEP + ["rank"]], on=["motif", "consensus"],
                                 suffixes=("_dinuc", "_genomic"), how="outer")
    m.insert(0, "family", fam)
    # motif sequence columns: matrix core + what these copies actually carry
    m["observed_consensus"] = m["observed_consensus_dinuc"].fillna(
        m["observed_consensus_genomic"])
    m["pct_copies"] = m["pct_copies_dinuc"].fillna(m["pct_copies_genomic"])
    m["rank_shift"] = m["rank_genomic"] - m["rank_dinuc"]
    m["sig_dinuc"] = m["fdr_q_dinuc"] < 0.05
    m["sig_genomic"] = m["fdr_q_genomic"] < 0.05
    m["agreement"] = m.apply(
        lambda r: "both" if r.sig_dinuc and r.sig_genomic
        else ("dinuc only" if r.sig_dinuc
              else ("genomic only" if r.sig_genomic else "neither")), axis=1)
    rows.append(m)

out = pd.concat(rows, ignore_index=True)
cols = ["family", "motif", "consensus", "observed_consensus", "pct_copies",
        "pct_copies_bg_dinuc", "fold_prevalence_dinuc", "fdr_q_dinuc", "rank_dinuc",
        "pct_copies_bg_genomic", "fold_prevalence_genomic", "fdr_q_genomic",
        "rank_genomic", "rank_shift", "agreement"]
out = out[cols].sort_values(["family", "rank_dinuc"])
out.to_csv("/Users/anmol/motifcmp/MOTIF_BACKGROUND_COMPARISON.tsv", sep="\t", index=False)

with open("/Users/anmol/motifcmp/COMPARISON_SUMMARY.txt", "w") as fh:
    fh.write("MOTIF ENRICHMENT: dinucleotide-shuffle vs genomic background\n")
    fh.write("JASPAR2024 CORE vertebrates (879 matrices), MOODS p<1e-4, 25x background\n")
    fh.write("consensus        = the matrix's IUPAC core (what the TF binds)\n")
    fh.write("observed_consensus = the bases these copies actually carry at the hits\n\n")
    for fam in FAMS:
        s = out[out.family == fam]
        fh.write(f"=== {fam} ===\n")
        fh.write(f"  significant (FDR<0.05): dinuc {int(s.sig_count_d) if False else (s.fdr_q_dinuc<0.05).sum()}"
                 f"   genomic {(s.fdr_q_genomic<0.05).sum()}"
                 f"   both {(s.agreement=='both').sum()}"
                 f"   dinuc-only {(s.agreement=='dinuc only').sum()}"
                 f"   genomic-only {(s.agreement=='genomic only').sum()}\n")
        top = s.nsmallest(TOPN, "rank_dinuc")
        fh.write(f"  {'motif':<24}{'observed seq':<22}{'%cop':>6}"
                 f"{'foldD':>7}{'foldG':>7}{'rkD':>5}{'rkG':>5}{'shift':>7}\n")
        for r in top.itertuples():
            fh.write(f"  {r.motif:<24}{str(r.observed_consensus)[:20]:<22}"
                     f"{r.pct_copies:>6.1f}{r.fold_prevalence_dinuc:>7.1f}"
                     f"{r.fold_prevalence_genomic:>7.1f}"
                     f"{r.rank_dinuc:>5.0f}{r.rank_genomic:>5.0f}"
                     f"{r.rank_shift:>+7.0f}\n")
        fh.write("\n")
print(open("/Users/anmol/motifcmp/COMPARISON_SUMMARY.txt").read())
