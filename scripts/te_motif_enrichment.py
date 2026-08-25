#!/usr/bin/env python3
"""
te_motif_enrichment.py -- which TF motifs are enriched inside a TE family.

Answers "what are the top motifs within those elements?" for a set of repeat
families, over the SAME locus set the expression pipeline counts (the rmsk-
derived TE GTF), so motif calls and expression refer to identical copies.

Method
  1. Pull every copy of each target family out of the TE GTF.
  2. Extract its sequence from the reference FASTA, reverse-complementing
     minus-strand copies so positions are relative to the ELEMENT, not the
     chromosome. (MOODS scans both strands regardless, so this changes
     coordinates and orientation, never which motifs are found.)
  3. Scan copies against the JASPAR CORE PFM bundle with MOODS (p < 1e-4),
     reusing te_moods_scan.scan_loci.
  4. Scan a background. Two are available (--background):
       dinuc   (default) a DINUCLEOTIDE-PRESERVING shuffle of each copy
       genomic length-matched random intervals sampled off-family
     They answer different questions: the shuffle asks whether a motif is
     explained by this family's own base composition; the genomic one asks
     whether it is more common here than in average genomic sequence, and
     does NOT control composition. Report which one produced a number.
     On the default:
     This is the step that makes the answer mean anything. Raw motif counts in
     a GC-rich LTR are dominated by whichever matrices happen to like GC, and a
     mononucleotide shuffle does not control for that (CpG especially). The
     shuffle preserves each copy's length AND dinucleotide composition, so what
     is left to explain a hit is sequence ORDER -- i.e. real motif structure.
  5. Per motif: prevalence across copies, hits/kb, fold vs background, Fisher
     exact p, Benjamini-Hochberg q.

Outputs into --out-dir:
  MOTIF_SEQUENCES.tsv           EVERY motif and its sequence, all families, one
                                row per motif x family -- the lookup table
  motifs_<FAM>_enrichment.tsv   ranked motif table, per family
  motifs_<FAM>_hits.tsv.gz      every copy x motif hit, with the matched bases
  TOP_MOTIFS.txt                human-readable summary across all families

Every table naming a motif also carries its SEQUENCE: `consensus` (IUPAC, the
matrix's informative core) and `observed_consensus` (built from the bases these
copies actually carry). A TF name alone cannot be acted on.

Limits this reports rather than hides:
  * JASPAR is redundant. Many matrices describe one TF family, so the top of
    the table reads as clusters of near-identical motifs, not N independent
    findings.
  * Fragmented families (LTR66's median copy is a couple of hundred bp) depress
    per-copy prevalence, which is why hits/kb is reported next to it.
  * A PWM match is a SEQUENCE match. It is not evidence of binding, and says
    nothing on its own about this tissue.
"""
from __future__ import annotations

import argparse
import gzip
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("motifs")

# ── reference FASTA: index + random access, without pysam ────────────────────
# Deliberately dependency-free. pysam is an extra wheel to install on the VM and
# has already cost this project a day (it SIGABRTs under FIPS in the cluster
# container). A .fai is a trivial format; building one is a single linear read.

def load_or_build_fai(fa_path):
    """Return {chrom: (seq_offset, line_bases, line_width, length)}."""
    fai_path = str(fa_path) + ".fai"
    idx = {}
    if os.path.exists(fai_path):
        with open(fai_path) as fh:
            for line in fh:
                f = line.rstrip("\n").split("\t")
                if len(f) >= 5:
                    idx[f[0]] = (int(f[2]), int(f[3]), int(f[4]), int(f[1]))
        if idx:
            log.info("FASTA index: reused %s (%d sequences)", fai_path, len(idx))
            return idx

    log.info("FASTA index: building for %s (one linear read) ...", fa_path)
    t0 = time.time()
    name = None
    offset = length = lb = lw = 0
    pos = 0
    with open(fa_path, "rb") as fh:
        for line in fh:
            n = len(line)
            if line.startswith(b">"):
                if name is not None:
                    idx[name] = (offset, lb, lw, length)
                name = line[1:].split()[0].decode()
                offset, length, lb, lw = pos + n, 0, 0, 0
            else:
                sl = len(line.rstrip(b"\r\n"))
                if lb == 0:
                    lb, lw = sl, n
                length += sl
            pos += n
    if name is not None:
        idx[name] = (offset, lb, lw, length)
    log.info("FASTA index: %d sequences in %.0fs", len(idx), time.time() - t0)
    try:  # persist so a re-run skips the scan
        with open(fai_path, "w") as out:
            for c, (o, b, w, ln) in idx.items():
                out.write(f"{c}\t{ln}\t{o}\t{b}\t{w}\n")
    except OSError:
        pass
    return idx


def fetch_seq(fh, idx, chrom, start1, end1):
    """1-based inclusive fetch. Returns '' for unknown//empty ranges."""
    rec = idx.get(chrom)
    if rec is None:
        return ""
    offset, lb, lw, length = rec
    if lb <= 0:
        return ""
    s0 = max(0, int(start1) - 1)
    e0 = min(int(length), int(end1))
    if e0 <= s0:
        return ""

    def byte_at(p):
        return offset + (p // lb) * lw + (p % lb)

    fh.seek(byte_at(s0))
    raw = fh.read(byte_at(e0 - 1) - byte_at(s0) + 1)
    return raw.replace(b"\n", b"").replace(b"\r", b"").decode("ascii", "replace").upper()


_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s):
    return s.translate(_COMP)[::-1]


# ── dinucleotide-preserving shuffle (Altschul-Erikson) ───────────────────────

def dinuc_shuffle(s, rng):
    """Shuffle s preserving its exact dinucleotide composition.

    Euler-path method: the last edge out of every vertex is reserved so that it
    forms a tree rooted at the final character; every other edge is permuted
    freely. Traversing the result yields a sequence with identical mono- AND
    dinucleotide counts.
    """
    if len(s) < 4:
        return s
    edges = defaultdict(list)
    for a, b in zip(s, s[1:]):
        edges[a].append(b)
    first, last = s[0], s[-1]
    verts = list(edges.keys())

    for _ in range(100):                       # retry until the tree is valid
        last_edge = {v: rng.choice(edges[v]) for v in verts if v != last}
        ok = True
        for v in verts:
            if v == last:
                continue
            seen, u = set(), v
            while u != last:
                if u in seen or u not in last_edge:
                    ok = False
                    break
                seen.add(u)
                u = last_edge[u]
            if not ok:
                break
        if ok:
            break
    else:
        return s                                # pathological; leave unshuffled

    out_edges = {}
    for v in verts:
        e = list(edges[v])
        if v != last:
            e.remove(last_edge[v])
            rng.shuffle(e)
            e.append(last_edge[v])
        else:
            rng.shuffle(e)
        out_edges[v] = e

    res = [first]
    cur = first
    used = defaultdict(int)
    for _ in range(len(s) - 1):
        e = out_edges.get(cur)
        if not e or used[cur] >= len(e):
            break
        nxt = e[used[cur]]
        used[cur] += 1
        res.append(nxt)
        cur = nxt
    return "".join(res)


# ── genomic background ───────────────────────────────────────────────────────

_PRIMARY = tuple("chr%s" % c for c in list(range(1, 23)) + ["X", "Y"])


def sample_genomic_background(fh, idx, lens, n_each, rng, exclude,
                              max_n_frac=0.2, max_tries=60):
    """Length-matched random genomic intervals, one set of n_each per real copy.

    The other background a reader can ask for: instead of asking "is this motif
    explained by THIS sequence's own composition" (the dinucleotide shuffle),
    it asks "is this motif more common here than in an average piece of the
    genome". Both are legitimate and they answer different questions --- the
    genomic one does NOT control for the family's base composition, so an
    AT-rich family will show AT-rich matrices enriched by composition alone.
    That is exactly why it is worth reporting next to the shuffle, not instead.

    Intervals overlapping a copy of the family itself are rejected, as are
    intervals that are mostly assembly gap.
    """
    chroms = [c for c in _PRIMARY if c in idx]
    if not chroms:
        chroms = list(idx.keys())
    weights = [idx[c][3] for c in chroms]          # (offset, lb, lw, length)
    out_ids, out_seqs = [], []
    for i, L in enumerate(lens):
        got = 0
        for _ in range(n_each * max_tries):
            if got >= n_each:
                break
            c = rng.choices(chroms, weights=weights, k=1)[0]
            clen = idx[c][3]
            if clen <= L:
                continue
            s0 = rng.randrange(1, clen - L + 1)
            e0 = s0 + L - 1
            if any(not (e0 < a or s0 > b) for a, b in exclude.get(c, ())):
                continue
            seq = fetch_seq(fh, idx, c, s0, e0)
            if len(seq) < L or seq.count("N") / max(len(seq), 1) > max_n_frac:
                continue
            out_seqs.append(revcomp(seq) if rng.random() < 0.5 else seq)
            out_ids.append(f"bg{got}|{c}:{s0}-{e0}")
            got += 1
    return out_ids, out_seqs


# ── TE GTF ───────────────────────────────────────────────────────────────────

def load_family_loci(gtf_path, families):
    """Return {family: [(element_id, chrom, start1, end1, strand), ...]}."""
    tags = {f: 'gene_id "%s";' % f for f in families}
    out = {f: [] for f in families}
    with open(gtf_path) as fh:
        for line in fh:
            for fam, tag in tags.items():
                if tag in line:
                    c = line.split("\t")
                    if len(c) < 9:
                        break
                    eid = fam
                    k = c[8].find('transcript_id "')
                    if k >= 0:
                        j = c[8].find('"', k + 15)
                        eid = c[8][k + 15:j]
                    out[fam].append((eid, c[0], int(c[3]), int(c[4]), c[6]))
                    break
    return out


# ── statistics ───────────────────────────────────────────────────────────────

def benjamini_hochberg(pvals):
    n = len(pvals)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvals[i])
    q = [1.0] * n
    prev = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        prev = min(prev, pvals[i] * n / rank)
        q[i] = prev
    return q


def write_motif_sequences(out_dir, families, cons_map, pd):
    """One file with every motif and its sequence, across all families.

    The per-family tables answer "what is enriched in THIS family"; this answers
    "what is motif X, and where did it turn up" in a single place, which is the
    lookup anyone actually reaches for. Long format, one row per motif x family,
    and it covers EVERY matrix in the scanned bundle -- including the ones with
    no hit anywhere, since "we looked and it is not there" is also an answer and
    a table missing those silently looks like they were never tested.

    Built by reading the per-family enrichment TSVs back, so it can be rebuilt
    from a finished run without rescanning (--combine-only).
    """
    out_dir = Path(out_dir)
    per_fam = {}
    for fam in families:
        p = out_dir / f"motifs_{fam}_enrichment.tsv"
        if p.exists():
            # round_trip: pandas' default CSV float parser is fast but up to a
            # couple of ULP off, so q-values re-read here would not be bit-exact
            # against the table they came from. Nothing here turns on the last
            # bit of a 1e-90 q-value, but two files disagreeing on the same
            # number is the kind of thing that costs an afternoon later.
            df = pd.read_csv(p, sep="\t", float_precision="round_trip")
            per_fam[fam] = {r["motif"]: r for _, r in df.iterrows()}
        else:
            log.warning("combined table: %s missing, skipping %s", p.name, fam)
    if not per_fam:
        return None

    labels = sorted(k for k in cons_map if k.endswith(")"))
    rows = []
    for fam, table in per_fam.items():
        for label in labels:
            cm = cons_map[label]
            r = table.get(label)
            rows.append({
                "motif": label,
                "tf_name": cm["name"],
                "matrix_id": cm["id"],
                "consensus": cm["core"],
                "consensus_full": cm["consensus"],
                "motif_width": cm["width"],
                "info_bits": cm["info_bits"],
                "family": fam,
                "observed_consensus": (r["observed_consensus"] if r is not None
                                       and isinstance(r["observed_consensus"], str) else ""),
                "detected": "yes" if r is not None else "no",
                "hits": int(r["hits"]) if r is not None else 0,
                "copies_with_hit": int(r["copies_with_hit"]) if r is not None else 0,
                "n_copies": int(r["n_copies"]) if r is not None else "",
                "pct_copies": r["pct_copies"] if r is not None else 0.0,
                "fold_prevalence": r["fold_prevalence"] if r is not None else "",
                "fdr_q": r["fdr_q"] if r is not None else "",
                "significant": ("yes" if (r is not None and float(r["fdr_q"]) < 0.05)
                                else "no"),
            })
    df = pd.DataFrame(rows).sort_values(
        ["family", "significant", "pct_copies"], ascending=[True, False, False])
    path = out_dir / "MOTIF_SEQUENCES.tsv"
    df.to_csv(path, sep="\t", index=False)
    n_sig = int((df["significant"] == "yes").sum())
    log.info("combined table: %s -- %d matrices x %d families = %d rows (%d significant)",
             path.name, len(labels), len(per_fam), len(df), n_sig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gtf", help="rmsk-derived TE GTF")
    ap.add_argument("--genome-fa", help="reference FASTA")
    ap.add_argument("--jaspar-pfm", required=True, help="JASPAR raw PFM bundle")
    ap.add_argument("--families", nargs="+", required=True)
    ap.add_argument("--out-dir", default="motifs")
    ap.add_argument("--pvalue", type=float, default=1e-4)
    ap.add_argument("--background", choices=["dinuc", "genomic"], default="dinuc",
                    help="dinuc = per-copy dinucleotide-preserving shuffle "
                         "(controls the family's own base composition); "
                         "genomic = length-matched random intervals off-family "
                         "(does NOT control composition)")
    ap.add_argument("--nshuffle", type=int, default=50,
                    help="dinuc-shuffled background copies per element")
    ap.add_argument("--min-len", type=int, default=50,
                    help="skip copies shorter than this")
    ap.add_argument("--max-n-frac", type=float, default=0.2)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--top", type=int, default=25, help="rows per family in TOP_MOTIFS.txt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--combine-only", action="store_true",
                    help="rebuild MOTIF_SEQUENCES.tsv from the per-family tables "
                         "already in --out-dir; no genome, no GTF, no rescan")
    args = ap.parse_args()
    if not args.combine_only and not (args.gtf and args.genome_fa):
        ap.error("--gtf and --genome-fa are required (unless --combine-only)")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import pandas as pd
    from scipy import stats as sstats
    from te_moods_scan import (scan_loci, pfm_consensus_map, observed_consensus,
                               revcomp_iupac, IUPAC_LEGEND)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # The consensus sequence for every matrix, up front. A motif table that
    # names TFs without showing the sequence they matched cannot be read on its
    # own, so this is not optional output -- it goes in the main table.
    cons_map = pfm_consensus_map(args.jaspar_pfm)
    log.info("consensus sequences derived for %d matrices",
             sum(1 for k in cons_map if k.endswith(")")))

    if args.combine_only:
        p = write_motif_sequences(out_dir, args.families, cons_map, pd)
        print(f"wrote {p}" if p else "nothing to combine")
        return

    log.info("loading loci for %s from %s", ", ".join(args.families), args.gtf)
    fam_loci = load_family_loci(args.gtf, args.families)
    idx = load_or_build_fai(args.genome_fa)

    summary_lines = []
    with open(args.genome_fa, "rb") as fh:
        for fam in args.families:
            loci = fam_loci.get(fam, [])
            log.info("%s: %d copies annotated", fam, len(loci))
            if not loci:
                summary_lines.append(f"\n=== {fam} ===\n  no copies found in the GTF -- check the family name\n")
                continue

            ids, seqs, skipped = [], [], 0
            for eid, chrom, s1, e1, strand in loci:
                seq = fetch_seq(fh, idx, chrom, s1, e1)
                if len(seq) < args.min_len:
                    skipped += 1
                    continue
                if seq.count("N") / len(seq) > args.max_n_frac:
                    skipped += 1
                    continue
                seqs.append(revcomp(seq) if strand == "-" else seq)
                ids.append(f"{eid}|{chrom}:{s1}-{e1}({strand})")
            if not seqs:
                summary_lines.append(f"\n=== {fam} ===\n  every copy filtered out (min-len {args.min_len})\n")
                continue

            lens = sorted(len(s) for s in seqs)
            total_kb = sum(lens) / 1e3
            med = lens[len(lens) // 2]
            log.info("%s: %d copies kept (%d skipped), median %d bp, %.1f kb total",
                     fam, len(seqs), skipped, med, total_kb)

            # real
            real_df = pd.DataFrame({"chr": ids, "start": [0] * len(seqs),
                                    "stop": [len(s) for s in seqs], "Seq": seqs})
            hits = scan_loci(real_df, None, args.jaspar_pfm, "chr", "start", "stop",
                             pvalue=args.pvalue, threads=args.threads, seq_col="Seq")

            if args.background == "genomic":
                log.info("%s: sampling %dx length-matched genomic background "
                         "(%.1f kb), family loci excluded",
                         fam, args.nshuffle, total_kb * args.nshuffle)
                excl = defaultdict(list)
                for _eid, chrom, s1, e1, _st in loci:
                    excl[chrom].append((s1, e1))
                bg_ids, bg_seqs = sample_genomic_background(
                    fh, idx, [len(s) for s in seqs], args.nshuffle, rng,
                    excl, max_n_frac=args.max_n_frac)
            else:
                log.info("%s: building %dx dinucleotide-shuffled background (%.1f kb)",
                         fam, args.nshuffle, total_kb * args.nshuffle)
                bg_ids, bg_seqs = [], []
                for i, s in enumerate(seqs):
                    for k in range(args.nshuffle):
                        bg_seqs.append(dinuc_shuffle(s, rng))
                        bg_ids.append(f"bg{k}|{ids[i]}")
            bg_df = pd.DataFrame({"chr": bg_ids, "start": [0] * len(bg_seqs),
                                  "stop": [len(s) for s in bg_seqs], "Seq": bg_seqs})
            bg_hits = scan_loci(bg_df, None, args.jaspar_pfm, "chr", "start", "stop",
                                pvalue=args.pvalue, threads=args.threads, seq_col="Seq")

            n_real, n_bg = len(seqs), len(bg_seqs)
            # measured, not assumed: genomic sampling can fall short of
            # nshuffle x when a copy's length cannot be placed off-family
            kb_real, kb_bg = total_kb, sum(len(s) for s in bg_seqs) / 1e3

            obs_cop = hits.groupby("Motif_name")["chr"].nunique() if len(hits) else {}
            obs_n = hits["Motif_name"].value_counts() if len(hits) else {}
            bg_cop = bg_hits.groupby("Motif_name")["chr"].nunique() if len(bg_hits) else {}
            bg_n = bg_hits["Motif_name"].value_counts() if len(bg_hits) else {}

            # The bases every match actually landed on, per motif. '-' strand
            # matches are reverse-complemented so the observed consensus is
            # directly comparable to the matrix consensus rather than being its
            # mirror image for half the hits.
            by_id = {eid: s for eid, s in zip(ids, seqs)}
            matched = defaultdict(list)
            if len(hits):
                for el, a0, b0, mname, mstrand in zip(
                        hits["chr"], hits["Motif_start"], hits["Motif_end"],
                        hits["Motif_name"], hits["Motif_strand"]):
                    sub = by_id.get(el, "")[int(a0):int(b0)]
                    matched[mname].append(revcomp_iupac(sub) if mstrand == "-" else sub)

            rows = []
            for motif in sorted(set(list(obs_cop.keys() if hasattr(obs_cop, "keys") else []))):
                a = int(obs_cop.get(motif, 0))
                b = n_real - a
                c = int(bg_cop.get(motif, 0))
                d = n_bg - c
                prev_o = a / n_real
                prev_b = c / n_bg if n_bg else 0.0
                try:
                    _, p = sstats.fisher_exact([[a, b], [c, d]], alternative="greater")
                except Exception:
                    p = 1.0
                cm = cons_map.get(motif, {})
                rows.append({
                    "motif": motif,
                    # what the matrix binds, and what these copies actually have
                    "consensus": cm.get("core", ""),
                    "observed_consensus": observed_consensus(matched.get(motif, [])),
                    "consensus_full": cm.get("consensus", ""),
                    "motif_width": cm.get("width", ""),
                    "copies_with_hit": a,
                    "n_copies": n_real,
                    "pct_copies": round(100.0 * prev_o, 2),
                    "pct_copies_bg": round(100.0 * prev_b, 3),
                    "fold_prevalence": round(prev_o / prev_b, 2) if prev_b > 0 else float("inf"),
                    "hits": int(obs_n.get(motif, 0)),
                    "hits_per_kb": round(int(obs_n.get(motif, 0)) / kb_real, 2) if kb_real else 0.0,
                    "hits_per_kb_bg": round(int(bg_n.get(motif, 0)) / kb_bg, 3) if kb_bg else 0.0,
                    "best_score": round(float(hits.loc[hits["Motif_name"] == motif, "Motif_score"].max()), 3) if len(hits) else 0.0,
                    "fisher_p": p,
                })
            res = pd.DataFrame(rows)
            if len(res):
                res["fdr_q"] = benjamini_hochberg(res["fisher_p"].tolist())
                res = res.sort_values(["fdr_q", "fold_prevalence", "pct_copies"],
                                      ascending=[True, False, False])
            enr_path = out_dir / f"motifs_{fam}_enrichment.tsv"
            res.to_csv(enr_path, sep="\t", index=False)

            hits_path = out_dir / f"motifs_{fam}_hits.tsv.gz"
            hits_out = hits.rename(columns={"chr": "element", "Motif_start": "elem_pos_start",
                                            "Motif_end": "elem_pos_end"}).drop(
                columns=[c for c in ("start", "stop", "Motif_chr") if c in hits.columns])
            # The bases each match actually landed on. Free here -- the element
            # sequences are already in memory and positions are element-relative
            # -- and without it the only way to see what a TF matched is to
            # re-extract from hg38, which needs the 3 GB FASTA. A '-' strand hit
            # matched the reverse complement of the motif; the substring is
            # written as it appears on the element, so revcomp it to compare
            # against the matrix consensus (te_motif_groups.py does this).
            if len(hits_out):
                by_id = {eid: s for eid, s in zip(ids, seqs)}
                hits_out["matched_seq"] = [
                    by_id.get(e, "")[int(a):int(b)]
                    for e, a, b in zip(hits_out["element"],
                                       hits_out["elem_pos_start"],
                                       hits_out["elem_pos_end"])]
            hits_out.to_csv(hits_path, sep="\t", index=False, compression="gzip")

            sig = res[res["fdr_q"] < 0.05] if len(res) else res
            summary_lines.append(
                f"\n=== {fam} ===\n"
                f"  {n_real} copies scanned ({skipped} skipped), median {med} bp, {total_kb:.1f} kb\n"
                f"  background: {args.nshuffle}x "
                + ("length-matched random genomic intervals, family excluded"
                   if args.background == "genomic"
                   else "dinucleotide-preserving shuffle")
                + f" ({n_bg} sequences, {kb_bg:.0f} kb)\n"
                f"  {len(sig)} motifs at FDR < 0.05 of {len(res)} detected\n"
                f"  full table: {enr_path.name}\n")
            head = (sig if len(sig) else res).head(args.top)
            if len(head):
                # 'binds' is the matrix consensus, 'observed' the bases these
                # copies actually carry. Both are in the printed table because
                # the TF name alone is the one thing nobody can act on.
                summary_lines.append(
                    f"  {'motif':<26}{'binds':<20}{'observed':<20}"
                    f"{'%copies':>8}{'fold':>7}{'q':>10}\n")
                for _, r in head.iterrows():
                    summary_lines.append(
                        f"  {str(r['motif'])[:25]:<26}{str(r['consensus'])[:19]:<20}"
                        f"{str(r['observed_consensus'])[:19]:<20}"
                        f"{r['pct_copies']:>8.1f}{r['fold_prevalence']:>7.1f}"
                        f"{r['fdr_q']:>10.1e}\n")

    banner = (
        "TOP MOTIFS BY FAMILY\n"
        "====================\n"
        f"JASPAR CORE vertebrates, MOODS p < {args.pvalue:g}, both strands.\n"
        "Enrichment is against a dinucleotide-preserving shuffle of the same\n"
        "copies, so GC/CpG content is controlled for and what remains is order.\n"
        "\n'binds' is the JASPAR matrix consensus, trimmed to its informative\n"
        "core; 'observed' is built from the bases these copies actually carry.\n"
        f"{IUPAC_LEGEND}\n"
        "\nRead with three caveats:\n"
        "  * JASPAR is redundant -- adjacent rows are often one TF family, not\n"
        "    independent hits.\n"
        "  * A PWM match is a sequence match, NOT evidence of binding, and is\n"
        "    not tissue-specific on its own.\n"
        "  * Fragmented families depress %copies; read hits/kb alongside it.\n")

    combined = write_motif_sequences(out_dir, args.families, cons_map, pd)
    if combined:
        summary_lines.append(
            f"\nEvery motif and its sequence, all families in one table:\n"
            f"  {combined.name}\n")

    (Path(args.out_dir) / "TOP_MOTIFS.txt").write_text(banner + "".join(summary_lines))
    print(banner + "".join(summary_lines))


if __name__ == "__main__":
    main()
