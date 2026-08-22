#!/usr/bin/env python3
"""
te_motif_groups.py -- collapse a motif enrichment table into non-redundant groups.

The problem this exists to solve: JASPAR is redundant. An enrichment table with
60 significant motifs is not 60 findings. Many of those matrices describe the
same binding preference, so they fire on the SAME bases of the SAME copies, and
reading them as independent hits inflates the apparent answer.

Two things are produced from an already-finished run (no rescan, no MOODS):

  1. The motif sequence for every TF, as an IUPAC consensus read off the JASPAR
     PFM that was scanned. This is what the matrix is looking for -- 'what
     sequence did ESR2 match' now has a printable answer.

  2. Groups of motifs that bind the same sequence. Grouping is by CO-OCCURRENCE
     IN THIS DATA, not by name or by matrix similarity: two motifs join a group
     when their hits land on top of each other in the same copies. That is the
     operational definition of "these matrices are describing one site" for this
     locus set, and it needs no external motif-similarity database.

     Concretely: hits A and B are the same site when they overlap by at least
     --min-overlap of the shorter motif's width. Motifs A and B are linked when
     (explained A hits + explained B hits) / (all A + all B) >= --min-shared.
     Groups are the connected components (single linkage).

  3. Per group, the UNION of copies carrying any member motif. This is the
     number to quote: if 12 matrices all mark one site in 48% of copies, the
     group prevalence is ~48%, not 12 x 48%.

Inputs are what the enrichment stage already wrote to the bucket:
  motifs_<FAM>_enrichment.tsv, motifs_<FAM>_hits.tsv.gz, and the JASPAR PFM
  bundle that was scanned (results/motifs/ and cache/jaspar_pfms.txt).

If the hits file carries a matched_seq column (te_motif_enrichment.py emits one
as of this commit), the OBSERVED consensus -- the actual genomic bases matched,
column by column -- is reported next to the PFM consensus. Older hits files
lack it; everything else still works.

Caveat kept in view: co-occurrence groups matrices by where they fire, so it
also merges genuinely different TFs whose sites coincide in these elements
(composite elements, half-site overlaps). The group is a SITE, not a TF.
"""
from __future__ import annotations

import argparse
import gzip
import logging
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("motif-groups")

_BASES = "ACGT"

# Consensus construction is shared with te_moods_scan so that every motif output
# in the repo spells sequences the same way; this module only arranges them.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from te_moods_scan import (parse_jaspar_pfms, consensus_from_counts,  # noqa: E402
                           observed_consensus, revcomp_iupac, IUPAC_LEGEND)


def parse_pfms(path):
    """{label: {id,name,consensus,core,width,ic}} keyed by 'NAME (ID)'."""
    out = {}
    for p in parse_jaspar_pfms(path):
        cons, core, bits = consensus_from_counts(p["counts"])
        out[f'{p["name"]} ({p["id"]})'] = {
            "id": p["id"], "name": p["name"], "consensus": cons, "core": core,
            "width": len(p["counts"][0]), "ic": bits,
        }
    if not out:
        raise ValueError(f"no PFMs parsed from {path}")
    return out


# ── grouping ─────────────────────────────────────────────────────────────────

class Union:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def cooccurrence(hits, min_overlap):
    """Count, per motif pair, how many hits of each are explained by the other.

    hits: list of (element, start, end, motif). Returns
      pair_expl[(a,b)] -> (set of a-hit indices, set of b-hit indices)
      n_hits[motif]     -> total hits
    A sweep per element: hits are sorted by start, and each hit is compared only
    against still-open earlier hits, so this is linear in overlapping pairs
    rather than quadratic in hits.
    """
    by_elem = defaultdict(list)
    for idx, (el, s, e, m) in enumerate(hits):
        by_elem[el].append((s, e, m, idx))
    pair = defaultdict(lambda: (set(), set()))
    n_hits = defaultdict(int)
    for _, hl in by_elem.items():
        hl.sort()
        active = []
        for s, e, m, idx in hl:
            n_hits[m] += 1
            active = [h for h in active if h[1] > s]
            for s2, e2, m2, idx2 in active:
                if m2 == m:
                    continue
                ov = min(e, e2) - max(s, s2)
                if ov <= 0:
                    continue
                if ov < min_overlap * min(e - s, e2 - s2):
                    continue
                a, b = (m, m2) if m < m2 else (m2, m)
                ia, ib = (idx, idx2) if m < m2 else (idx2, idx)
                sa, sb = pair[(a, b)]
                sa.add(ia)
                sb.add(ib)
            active.append((s, e, m, idx))
    return pair, n_hits


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hits", required=True, help="motifs_<FAM>_hits.tsv.gz")
    ap.add_argument("--enrichment", required=True, help="motifs_<FAM>_enrichment.tsv")
    ap.add_argument("--jaspar-pfm", required=True, help="the PFM bundle that was scanned")
    ap.add_argument("--family", default=None, help="label for the report (default: from --hits)")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--q", type=float, default=0.05, help="FDR cutoff for motifs to group")
    ap.add_argument("--min-overlap", type=float, default=0.5,
                    help="two hits are the same site at this fraction of the shorter motif")
    ap.add_argument("--min-shared", type=float, default=0.5,
                    help="link two motifs when this fraction of their hits is shared")
    ap.add_argument("--top", type=int, default=20, help="groups printed in the readable report")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import pandas as pd

    fam = args.family
    if fam is None:
        m = re.search(r"motifs_(.+?)_hits", Path(args.hits).name)
        fam = m.group(1) if m else Path(args.hits).stem

    pfms = parse_pfms(args.jaspar_pfm)
    log.info("%s: %d PFM consensuses parsed", fam, len(pfms))

    enr = pd.read_csv(args.enrichment, sep="\t", float_precision="round_trip")
    sig = enr[enr["fdr_q"] < args.q].copy() if "fdr_q" in enr.columns else enr.copy()
    keep = set(sig["motif"])
    log.info("%s: %d motifs at FDR<%g of %d", fam, len(keep), args.q, len(enr))
    if not keep:
        log.warning("%s: nothing significant -- nothing to group", fam)
        return

    op = gzip.open if str(args.hits).endswith(".gz") else open
    with op(args.hits, "rt") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        ci = {c: i for i, c in enumerate(hdr)}
        need = ("element", "elem_pos_start", "elem_pos_end", "Motif_name")
        if any(c not in ci for c in need):
            sys.exit(f"hits file missing columns {need}: has {hdr}")
        has_seq = "matched_seq" in ci
        rows, copies, matched = [], defaultdict(set), defaultdict(list)
        for line in fh:
            f = line.rstrip("\n").split("\t")
            motif = f[ci["Motif_name"]]
            if motif not in keep:
                continue
            el = f[ci["element"]]
            rows.append((el, int(f[ci["elem_pos_start"]]), int(f[ci["elem_pos_end"]]), motif))
            copies[motif].add(el)
            if has_seq:
                seq = f[ci["matched_seq"]].upper()
                # a '-' strand hit matched the reverse complement; normalise so
                # the observed consensus is comparable to the PFM consensus
                if "Motif_strand" in ci and f[ci["Motif_strand"]] == "-":
                    seq = revcomp_iupac(seq)
                matched[motif].append(seq)
    log.info("%s: %d hits over %d significant motifs%s", fam, len(rows), len(keep),
             " (with matched sequence)" if has_seq else "")

    pair, n_hits = cooccurrence(rows, args.min_overlap)
    uf = Union(sorted(keep))
    links = []
    for (a, b), (sa, sb) in pair.items():
        denom = n_hits[a] + n_hits[b]
        shared = (len(sa) + len(sb)) / denom if denom else 0.0
        if shared >= args.min_shared:
            uf.union(a, b)
            links.append((a, b, round(shared, 3)))
    log.info("%s: %d motif pairs linked at >=%.0f%% shared hits",
             fam, len(links), 100 * args.min_shared)

    stats = sig.set_index("motif").to_dict("index")
    n_copies = int(sig["n_copies"].iloc[0]) if "n_copies" in sig.columns and len(sig) else 0

    members = defaultdict(list)
    for m in keep:
        members[uf.find(m)].append(m)

    groups = []
    for root, mem in members.items():
        # representative = the member with the strongest enrichment
        mem = sorted(mem, key=lambda m: (stats.get(m, {}).get("fdr_q", 1.0),
                                         -stats.get(m, {}).get("pct_copies", 0.0)))
        rep = mem[0]
        union_copies = set()
        for m in mem:
            union_copies |= copies[m]
        groups.append({
            "group_rep": rep,
            "n_motifs": len(mem),
            "rep_consensus": pfms.get(rep, {}).get("core", ""),
            "rep_consensus_full": pfms.get(rep, {}).get("consensus", ""),
            "rep_observed": observed_consensus(matched.get(rep, [])) if has_seq else "",
            "copies_with_group": len(union_copies),
            "n_copies": n_copies,
            "pct_copies_group": round(100.0 * len(union_copies) / n_copies, 2) if n_copies else 0.0,
            "rep_pct_copies": stats.get(rep, {}).get("pct_copies", ""),
            "rep_fold": stats.get(rep, {}).get("fold_prevalence", ""),
            "rep_fdr_q": stats.get(rep, {}).get("fdr_q", ""),
            "total_hits": sum(n_hits[m] for m in mem),
            "members": ";".join(mem),
        })
    groups.sort(key=lambda g: (-g["copies_with_group"], g["group_rep"]))
    for i, g in enumerate(groups, 1):
        g["group"] = f"G{i:02d}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gcols = ["group", "n_motifs", "group_rep", "rep_consensus", "rep_observed",
             "copies_with_group", "n_copies", "pct_copies_group", "rep_pct_copies",
             "rep_fold", "rep_fdr_q", "total_hits", "rep_consensus_full", "members"]
    gpath = out_dir / f"motif_groups_{fam}.tsv"
    pd.DataFrame(groups)[gcols].to_csv(gpath, sep="\t", index=False)

    # per-motif table: the consensus sequence for every significant TF
    gof = {m: g["group"] for g in groups for m in g["members"].split(";")}
    per = []
    for m in keep:
        s = stats.get(m, {})
        per.append({
            "motif": m,
            "group": gof.get(m, ""),
            "consensus": pfms.get(m, {}).get("core", ""),
            "consensus_full": pfms.get(m, {}).get("consensus", ""),
            "observed_consensus": observed_consensus(matched.get(m, [])) if has_seq else "",
            "width": pfms.get(m, {}).get("width", ""),
            "info_bits": pfms.get(m, {}).get("ic", ""),
            "pct_copies": s.get("pct_copies", ""),
            "fold_prevalence": s.get("fold_prevalence", ""),
            "hits": n_hits.get(m, 0),
            "fdr_q": s.get("fdr_q", ""),
        })
    per.sort(key=lambda r: (r["group"], r["fdr_q"] if r["fdr_q"] != "" else 1.0))
    ppath = out_dir / f"motif_consensus_{fam}.tsv"
    pd.DataFrame(per).to_csv(ppath, sep="\t", index=False)

    lines = [
        f"MOTIF GROUPS -- {fam}\n",
        "=" * (16 + len(fam)) + "\n",
        f"{len(keep)} motifs at FDR<{args.q:g} collapse into {len(groups)} groups.\n",
        "Motifs are grouped when their hits land on the SAME bases of the SAME\n"
        "copies (>= %.0f%% overlap, >= %.0f%% of hits shared), i.e. they are one\n"
        "site described by several JASPAR matrices.\n" % (100 * args.min_overlap,
                                                          100 * args.min_shared),
        "%%copies is the UNION over the group -- the fraction of copies carrying\n"
        "the site at all, which is the number to quote, not the per-matrix one.\n",
        "Consensus is IUPAC, trimmed to the informative core of the matrix.\n",
        "\nA group is a SITE, not a TF: matrices merge here because they fire in\n"
        "the same place, which also merges distinct TFs with coincident sites.\n\n",
        f"  {'grp':<5}{'consensus':<20}{'n':>3}  {'%cop':>6}{'fold':>7}{'q':>10}  representative\n",
    ]
    for g in groups[:args.top]:
        q = g["rep_fdr_q"]
        lines.append(
            f"  {g['group']:<5}{(g['rep_consensus'] or '-')[:19]:<20}{g['n_motifs']:>3}  "
            f"{g['pct_copies_group']:>6.1f}{float(g['rep_fold']):>7.1f}"
            f"{float(q):>10.1e}  {g['group_rep']}\n")
        if g["n_motifs"] > 1:
            others = [m for m in g["members"].split(";") if m != g["group_rep"]]
            wrapped, cur = [], "         also: "
            for o in others:
                if len(cur) + len(o) > 92:
                    wrapped.append(cur)
                    cur = "               "
                cur += o + ", "
            wrapped.append(cur.rstrip(", "))
            lines.extend(w + "\n" for w in wrapped)
    lines.append(f"\nfull tables: {gpath.name}, {ppath.name}\n")
    if not has_seq:
        lines.append(
            "\nNote: this hits file predates the matched_seq column, so the observed\n"
            "consensus (the actual bases matched) is blank; the PFM consensus is what\n"
            "is shown. Re-run the motifs stage to populate it.\n")
    txt = "".join(lines)
    (out_dir / f"MOTIF_GROUPS_{fam}.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
