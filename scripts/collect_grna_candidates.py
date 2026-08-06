#!/usr/bin/env python3
"""collect_grna_candidates.py --- build a run_grna_combos.py guide sheet from the
guides that submit_reagent_design.sh already designed.

run_grna_combos.py wants a table with a 'family' and a 'guide' column, and it
enumerates every subset of the guides it is given -- so the input has to be a
short shortlist, not the full candidate pool.

Per family this takes, from <reagent_dir>/<family>/:
  grna_greedy_set.csv   the greedy set (always kept -- it is the recommendation)
  grna_candidates.csv   top-N remaining candidates by PAM-aware copy coverage

and writes one combined CSV. Guides are de-duplicated, and any guide whose
length disagrees with --grna-len is dropped: run_grna_analysis.py defaults to
18 nt while run_grna_combos.py assumes 20, and mixing them silently corrupts
the PAM-required coverage.

Stdlib only.

  python3 collect_grna_candidates.py --reagent-dir ~/endometrium_reagents \
      --families LTR66 LTR10G --top-n 12 --grna-len 20 \
      --out ~/endometrium_reagents/combos/combo_input_guides.csv
"""
import argparse
import csv
import os
import sys


def _read(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _guide_col(row):
    for c in row:
        if c.lower() in ("grna", "guide", "protospacer", "sequence", "seq"):
            return c
    return None


def _num(row, *names):
    """First parseable numeric field among *names* (0.0 if none)."""
    for n in names:
        if n in row:
            try:
                return float(row[n])
            except (TypeError, ValueError):
                pass
    return 0.0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reagent-dir", required=True)
    ap.add_argument("--families", nargs="+", required=True)
    ap.add_argument("--top-n", type=int, default=12,
                    help="guides per family (2^N subsets are enumerated; keep it small)")
    ap.add_argument("--grna-len", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out_rows, report = [], []
    for fam in a.families:
        fdir = os.path.join(a.reagent_dir, fam)
        greedy = _read(os.path.join(fdir, "grna_greedy_set.csv"))
        cands = _read(os.path.join(fdir, "grna_candidates.csv"))
        if not greedy and not cands:
            report.append(f"  {fam}: NO gRNA output under {fdir} -- skipped")
            continue

        picked, seen, wrong_len = [], set(), 0

        def take(rows, source, limit=None):
            nonlocal wrong_len
            n = 0
            for r in rows:
                gc = _guide_col(r)
                if not gc:
                    continue
                g = (r.get(gc) or "").strip().upper()
                if not g or set(g) - set("ACGT"):
                    continue
                if len(g) != a.grna_len:
                    wrong_len += 1
                    continue
                if g in seen:
                    continue
                seen.add(g)
                picked.append({
                    "family": fam, "guide": g, "source": source,
                    "pct_seq_cov": _num(r, "pct_loci", "pct_seq_cov"),
                    "pct_expr_cov": _num(r, "pct_expr", "pct_expr_cov"),
                    "on_target_score": _num(r, "on_target_score"),
                })
                n += 1
                if limit is not None and n >= limit:
                    break

        take(greedy, "greedy")
        # best remaining candidates by PAM-aware copy coverage, then on-target score
        cands.sort(key=lambda r: (_num(r, "pct_seq_cov", "pct_loci"),
                                  _num(r, "on_target_score")), reverse=True)
        take(cands, "candidate", limit=max(0, a.top_n - len(picked)))

        for i, p in enumerate(picked, 1):
            p["rank"] = i
        out_rows.extend(picked)
        note = f" ({wrong_len} dropped for length != {a.grna_len})" if wrong_len else ""
        report.append(f"  {fam}: {len(picked)} guides "
                      f"[{sum(1 for p in picked if p['source'] == 'greedy')} greedy] "
                      f"-> 2^{len(picked)} subsets{note}")

    if not out_rows:
        sys.exit("[collect] no guides found -- did the reagent design stage run?")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    fields = ["family", "rank", "guide", "source", "pct_seq_cov",
              "pct_expr_cov", "on_target_score"]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in fields})

    print("[collect] guide shortlist for run_grna_combos.py:")
    for line in report:
        print(line)
    print(f"[collect] wrote {a.out}")


if __name__ == "__main__":
    main()
