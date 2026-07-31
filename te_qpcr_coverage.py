#!/usr/bin/env python3
"""te_qpcr_coverage.py --- annotate qPCR primer pairs with FAMILY COVERAGE.

te_qpcr_primers.py reports each pair's Tm/amplicon/genome-specificity but not
"what fraction of the family this pair actually amplifies". This adds that: for
each pair it scans every family copy and counts those where BOTH primers match
(<= --max-mm mismatches, to tolerate family divergence) in the right orientation
and at a plausible amplicon distance. Pure stdlib -> runs in seconds with plain
python3, no container / numpy needed, and does NOT recompute primers or genome
hits.

  python3 te_qpcr_coverage.py --pairs LTR5_Hs/LTR5_Hs_qpcr_pairs.csv \
      --seqs LTR5_Hs/<family_sequences>.csv         # rewrites the pairs CSV in place
"""
import argparse, csv, os, sys

_COMP = str.maketrans("ACGTN", "TGCAN")
def _rc(s): return s.translate(_COMP)[::-1]


def read_seqs(path):
    seqs = []
    if path.lower().endswith((".fa", ".fasta", ".fna")):
        cur = []
        for line in open(path):
            if line.startswith(">"):
                if cur: seqs.append("".join(cur)); cur = []
            else: cur.append(line.strip())
        if cur: seqs.append("".join(cur))
    else:
        with open(path, newline="") as fh:
            sniff = fh.read(2048); fh.seek(0)
            delim = "\t" if sniff.count("\t") > sniff.count(",") else ","
            rdr = csv.DictReader(fh, delimiter=delim)
            col = next((c for c in (rdr.fieldnames or []) if c.lower() == "seq"), None)
            if not col: sys.exit(f"no 'Seq' column in {path}")
            for row in rdr:
                s = (row.get(col) or "").strip().upper()
                if s and set(s) <= set("ACGTN"): seqs.append(s)
    return [s for s in seqs if len(s) >= 40]


def _match_positions(primer, seq, max_mm):
    """Start indices where primer aligns to seq with <= max_mm mismatches."""
    L = len(primer); pos = []
    for i in range(len(seq) - L + 1):
        mm = 0
        for a, b in zip(primer, seq[i:i + L]):
            if a != b:
                mm += 1
                if mm > max_mm: break
        else:
            pos.append(i)
    return pos


def pair_coverage(fwd, rev, seqs, amp_min, amp_max, max_mm):
    """# family copies amplified by (fwd, rev): both primers, right orient+distance."""
    revc = _rc(rev)
    lo, hi = int(amp_min * 0.6), int(amp_max * 1.6)   # tolerance around designed size
    n = 0
    for s in seqs:
        done = False
        for strand in (s, _rc(s)):
            F = _match_positions(fwd, strand, max_mm)
            if not F: continue
            R = _match_positions(revc, strand, max_mm)
            if not R: continue
            for f in F:
                for r in R:
                    size = r + len(revc) - f
                    if r >= f and lo <= size <= hi:
                        n += 1; done = True; break
                if done: break
            if done: break
    return n


def annotate(pairs_csv, seqs, max_mm, amp_min, amp_max, out=None):
    with open(pairs_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows: sys.exit(f"no rows in {pairs_csv}")
    tot = len(seqs)
    for row in rows:
        cov = pair_coverage(row["fwd"].upper(), row["rev"].upper(), seqs,
                            amp_min, amp_max, max_mm)
        row["copies_amplified"] = cov
        row["family_total"] = tot
        row["pct_family"] = round(100.0 * cov / tot, 1) if tot else 0.0
    fields = list(rows[0].keys())
    out = out or pairs_csv
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    best = max(rows, key=lambda r: r["copies_amplified"])
    print(f"[cov] {os.path.basename(pairs_csv)}: {len(rows)} pairs, family={tot} copies; "
          f"best pair {best['pair']} covers {best['copies_amplified']} ({best['pct_family']}%)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True, help="<family>_qpcr_pairs.csv to annotate")
    ap.add_argument("--seqs", required=True, help="family sequences CSV/FASTA (Seq column)")
    ap.add_argument("--out", default=None, help="output (default: overwrite --pairs)")
    ap.add_argument("--max-mm", type=int, default=2)
    ap.add_argument("--amp-min", type=int, default=80)
    ap.add_argument("--amp-max", type=int, default=150)
    a = ap.parse_args()
    seqs = read_seqs(a.seqs)
    annotate(a.pairs, seqs, a.max_mm, a.amp_min, a.amp_max, a.out)


if __name__ == "__main__":
    main()
