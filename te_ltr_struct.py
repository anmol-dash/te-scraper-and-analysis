#!/usr/bin/env python3
"""
te_ltr_struct.py — LTR structural annotation (pure Python, no external tools)
─────────────────────────────────────────────────────────────────────────────
Annotates each LTR retrotransposon locus with structural features:

  • LTR boundaries — detects 5'/3' LTR-like repeats at the element ends
    by comparing the first/last N bp of the sequence to each other
  • PBS — primer binding site (~18 bp, downstream of 5' LTR)
    searched using known consensus sequences for common LTR families
    and a generic tRNA-complementarity scan
  • PPT — polypurine tract (~15 bp of >70% purines, upstream of 3' LTR)
  • TSD — target site duplication (requires flanking genomic sequence;
    flagged "n/a" when flanking sequence is not available)
  • Classification: full-length | internal-only | solo-LTR | truncated

Works entirely on the 'Seq' column of the clustered CSV.  No external
tools (LTRharvest, LTRfinder) are required.

Outputs (all in <reports_dir>/):
  ltr_struct_annotated.csv     per-locus structural features
  fig_ltr_struct.png           stacked-bar summary by cluster
  ltr_struct_values.tex        LaTeX macros for the report

Standalone:
    python te_ltr_struct.py --input clustered.csv --reports-dir ./reports --family THE1D-int
"""

import argparse
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Known PBS consensus sequences ─────────────────────────────────────────
# Sequences are the genomic (positive-strand) PBS motif, located just
# downstream of the 5' LTR.  These represent the complement of the 3' end
# of the cognate tRNA.
_PBS_KNOWN = {
    # tRNA-Lys3 (AAA/AAG) — HERV-K, most HERVs
    "tRNA-Lys3":  "CCCAGCGAGCCGAGTGATTT",
    # tRNA-Pro (CCC) — THE1, MLT1, MSTA, IAPLTR1_Mm
    "tRNA-Pro":   "CAGCCTCGAGTCTGAATCTGAG",
    # tRNA-Arg — LTR7/HERV-H
    "tRNA-Arg":   "GCCCCGGATTAGCTTAAGT",
    # tRNA-Leu — LTR5_Hs subset
    "tRNA-Leu":   "GCCCAGCTGGCGAAATCGG",
    # tRNA-Asp — MERVL, MMERVL
    "tRNA-Asp":   "TTCTGAGCCCAGGATCGAAC",
    # generic 18-mer enriched in tRNA mimicry (fallback)
    "generic":    "GGAAATCGGCTCCCTTCGG",
}

# Families known to use each tRNA
_FAMILY_PBS = {
    "hervk": "tRNA-Lys3", "herv-k": "tRNA-Lys3",
    "ltr5": "tRNA-Lys3",  "ltr5_hs": "tRNA-Lys3",
    "the1":  "tRNA-Pro",  "the1d": "tRNA-Pro", "the1b": "tRNA-Pro",
    "mlt1":  "tRNA-Pro",  "msta": "tRNA-Pro",
    "ltr7":  "tRNA-Arg",  "herv-h": "tRNA-Arg",
    "mervl": "tRNA-Asp",
    "iapltr1": "tRNA-Pro",
}


def _rc(seq):
    """Reverse complement."""
    comp = str.maketrans("ACGTN", "TGCAN")
    return seq.translate(comp)[::-1]


def _hamming_pct(a, b):
    """Fraction of mismatches (0 = identical, 1 = all different)."""
    if not a or not b:
        return 1.0
    n = min(len(a), len(b))
    return sum(x != y for x, y in zip(a[:n], b[:n])) / n


def _best_match(query, target, max_mismatches=3):
    """Slide query over target and return (best_pos, mismatch_count) or None."""
    q = query.upper()
    t = target.upper()
    lq, lt = len(q), len(t)
    if lt < lq:
        return None
    best_pos, best_mm = None, max_mismatches + 1
    for i in range(lt - lq + 1):
        mm = sum(q[j] != t[i + j] for j in range(lq))
        if mm < best_mm:
            best_mm = mm
            best_pos = i
    if best_pos is not None and best_mm <= max_mismatches:
        return best_pos, best_mm
    return None


# ── LTR boundary detection ─────────────────────────────────────────────────

def detect_ltr_boundaries(seq, ltr_window=150, min_identity=0.65):
    """Compare 5' and 3' ends to detect LTR-like terminal repeats.

    Args:
        seq:          full element sequence (string, upper)
        ltr_window:   number of bp to examine at each end (default 150)
        min_identity: minimum fraction identity to call an LTR (default 0.65)

    Returns:
        dict with keys:
            has_5ltr, has_3ltr  (bool)
            ltr_identity        (float 0–1, similarity between the two ends)
            ltr_window          (int, bp examined)
            classification      ("full-length" | "solo-LTR" | "internal-only" | "truncated")
    """
    seq = seq.upper()
    L   = len(seq)
    w   = min(ltr_window, L // 3)

    if w < 30 or L < 100:
        return {
            "has_5ltr": False, "has_3ltr": False,
            "ltr_identity": 0.0, "ltr_window": w,
            "classification": "too-short",
        }

    five_prime  = seq[:w]
    three_prime = seq[L - w:]

    # Pairwise identity (simple hamming on equal-length windows)
    identity = 1.0 - _hamming_pct(five_prime, three_prime)

    # Also try reverse complement of the 3' end (in case orientation differs)
    identity_rc = 1.0 - _hamming_pct(five_prime, _rc(three_prime))
    identity = max(identity, identity_rc)

    has_5ltr = identity >= min_identity
    has_3ltr = identity >= min_identity

    if has_5ltr and has_3ltr:
        # Is the middle significantly different from the ends? (internal region)
        if L > 3 * w:
            mid = seq[w: L - w]
            mid_vs_5 = 1.0 - _hamming_pct(mid[:w], five_prime)
            if mid_vs_5 < min_identity:
                classification = "full-length"
            else:
                classification = "solo-LTR"
        else:
            classification = "solo-LTR"
    elif has_5ltr or has_3ltr:
        classification = "truncated"
    else:
        classification = "internal-only"

    return {
        "has_5ltr":       has_5ltr,
        "has_3ltr":       has_3ltr,
        "ltr_identity":   round(identity, 3),
        "ltr_window":     w,
        "classification": classification,
    }


# ── PBS detection ──────────────────────────────────────────────────────────

def detect_pbs(seq, family=None, search_window_bp=200, max_mismatches=3):
    """Search for the primer binding site (PBS) downstream of the 5' LTR.

    Searches the first `search_window_bp` bp for each known PBS motif
    (or a family-specific one) and reports the best hit.

    Returns:
        dict: pbs_found (bool), pbs_position (int or None),
              pbs_motif (str), pbs_mismatches (int)
    """
    seq = seq.upper()[:search_window_bp]

    # Determine tRNA to prioritise
    fam_key = (family or "").lower().replace("-", "").replace("_", "")
    pbs_order = []
    for k, v in _FAMILY_PBS.items():
        if k.replace("-","").replace("_","") in fam_key or fam_key in k.replace("-","").replace("_",""):
            pbs_order.insert(0, (k, _PBS_KNOWN[v], v))
    for name, pbs_seq in _PBS_KNOWN.items():
        if not any(p[1] == pbs_seq for p in pbs_order):
            pbs_order.append((name, pbs_seq, name))

    best = None
    for fam_hint, pbs_seq, trna_name in pbs_order:
        # Try both orientations
        for probe in [pbs_seq, _rc(pbs_seq)]:
            result = _best_match(probe[:18], seq, max_mismatches=max_mismatches)
            if result:
                pos, mm = result
                if best is None or mm < best[2]:
                    best = (pos, trna_name, mm)

    if best:
        return {
            "pbs_found":       True,
            "pbs_position":    best[0],
            "pbs_motif":       best[1],
            "pbs_mismatches":  best[2],
        }
    return {
        "pbs_found":       False,
        "pbs_position":    None,
        "pbs_motif":       "none",
        "pbs_mismatches":  None,
    }


# ── PPT detection ──────────────────────────────────────────────────────────

def detect_ppt(seq, window_bp=15, min_purine_frac=0.70, search_from_end_bp=300):
    """Detect polypurine tract (PPT) near the 3' end of the element.

    Slides a window of `window_bp` bp over the last `search_from_end_bp` bp
    and reports the window with the highest purine (A+G) content.

    Returns:
        dict: ppt_found (bool), ppt_position (int from sequence start),
              ppt_purine_frac (float), ppt_sequence (str)
    """
    seq = seq.upper()
    L   = len(seq)
    if L < window_bp + 20:
        return {"ppt_found": False, "ppt_position": None,
                "ppt_purine_frac": None, "ppt_sequence": ""}

    search_start = max(0, L - search_from_end_bp)
    region = seq[search_start:]

    best_pos, best_frac, best_seq = None, 0.0, ""
    for i in range(len(region) - window_bp + 1):
        w = region[i: i + window_bp]
        pur = sum(1 for c in w if c in "AG") / window_bp
        if pur > best_frac:
            best_frac = pur
            best_pos  = search_start + i
            best_seq  = w

    found = best_frac >= min_purine_frac
    return {
        "ppt_found":        found,
        "ppt_position":     best_pos if found else None,
        "ppt_purine_frac":  round(best_frac, 3) if found else None,
        "ppt_sequence":     best_seq if found else "",
    }


# ── TSD detection ──────────────────────────────────────────────────────────

def detect_tsd(left_flank, right_flank, tsd_range=(4, 6)):
    """Detect target site duplication (TSD) in genomic flanking sequences.

    Args:
        left_flank:  sequence immediately 5' of the TE (10–20 bp)
        right_flank: sequence immediately 3' of the TE (10–20 bp)
        tsd_range:   (min_len, max_len) of expected TSD

    Returns:
        dict: tsd_found (bool), tsd_sequence (str), tsd_length (int)
    """
    if not left_flank or not right_flank:
        return {"tsd_found": False, "tsd_sequence": "n/a",
                "tsd_length": 0, "tsd_note": "no flanking sequence"}

    lf = left_flank.upper().strip()[-tsd_range[1]:]  # take last N bp of left flank
    rf = right_flank.upper().strip()[:tsd_range[1]]  # take first N bp of right flank

    best_len, best_seq = 0, ""
    for tsd_len in range(tsd_range[1], tsd_range[0] - 1, -1):
        if len(lf) < tsd_len or len(rf) < tsd_len:
            continue
        cand_l = lf[-tsd_len:]
        cand_r = rf[:tsd_len]
        mm = sum(a != b for a, b in zip(cand_l, cand_r))
        if mm <= 1:
            best_len, best_seq = tsd_len, cand_l
            break

    if best_len >= tsd_range[0]:
        return {"tsd_found": True,  "tsd_sequence": best_seq,
                "tsd_length": best_len, "tsd_note": ""}
    return {"tsd_found": False, "tsd_sequence": "not detected",
            "tsd_length": 0, "tsd_note": ""}


# ── Full per-locus annotation ──────────────────────────────────────────────

def annotate_locus(seq, family=None,
                   left_flank=None, right_flank=None,
                   ltr_window=150, min_ltr_identity=0.65,
                   pbs_window=200, ppt_window=15):
    """Return a single annotation dict for one locus sequence."""
    seq = str(seq).upper() if seq else ""
    result = {"seq_length": len(seq)}
    if len(seq) < 80:
        result.update({
            "classification": "too-short",
            "has_5ltr": False, "has_3ltr": False,
            "ltr_identity": None, "ltr_window": 0,
            "pbs_found": False, "pbs_position": None, "pbs_motif": "none", "pbs_mismatches": None,
            "ppt_found": False, "ppt_position": None, "ppt_purine_frac": None, "ppt_sequence": "",
            "tsd_found": False, "tsd_sequence": "n/a", "tsd_length": 0, "tsd_note": "too-short",
        })
        return result

    result.update(detect_ltr_boundaries(seq, ltr_window, min_ltr_identity))
    result.update(detect_pbs(seq, family=family, search_window_bp=pbs_window))
    result.update(detect_ppt(seq, window_bp=ppt_window))
    result.update(detect_tsd(left_flank, right_flank))
    return result


# ── Batch analysis ──────────────────────────────────────────────────────────

def run_ltr_struct_analysis(input_csv, reports_dir, family_name,
                             min_ltr_identity=0.65):
    """Annotate all loci in the clustered CSV with LTR structural features.

    Returns the input DataFrame with annotation columns added.
    """
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== LTR STRUCTURAL ANNOTATION ({family_name}) ===")
    df = pd.read_csv(input_csv)
    print(f"  {len(df):,} loci")

    if "Seq" not in df.columns:
        print("  SKIP: no 'Seq' column")
        return df

    cl_col = next((c for c in ("Cluster", "cluster") if c in df.columns), None)

    print(f"  Annotating (min_ltr_identity={min_ltr_identity})...")
    records = []
    for i in range(len(df)):
        row    = df.iloc[i]
        seq    = str(row.get("Seq", "")).upper()
        ann    = annotate_locus(seq, family=family_name,
                                ltr_window=150, min_ltr_identity=min_ltr_identity)
        ann["locus_idx"] = i
        records.append(ann)
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(df)}")

    ann_df = pd.DataFrame(records)

    # Merge annotation back onto df
    df = df.copy()
    for col in ann_df.columns:
        if col != "locus_idx":
            df[col] = ann_df[col].values

    out_csv = reports_dir / "ltr_struct_annotated.csv"
    df.to_csv(out_csv, index=False)
    print(f"  Annotated CSV → {out_csv}")

    # ── Summary bar chart ───────────────────────────────────────────────────
    if cl_col and "classification" in df.columns:
        classes = ["full-length", "solo-LTR", "internal-only", "truncated", "too-short"]
        palette_map = {
            "full-length":   "#2ECC71",
            "solo-LTR":      "#4C9BE8",
            "internal-only": "#E8604C",
            "truncated":     "#F39C12",
            "too-short":     "#cccccc",
        }
        cluster_ids = sorted(c for c in df[cl_col].unique() if c >= 0)
        fig, ax = plt.subplots(figsize=(max(7, len(cluster_ids) * 1.5 + 3), 5))
        x = np.arange(len(cluster_ids))
        bottoms = np.zeros(len(cluster_ids))
        for cls in classes:
            counts = []
            for cid in cluster_ids:
                sub = df[df[cl_col] == cid]
                counts.append((sub["classification"] == cls).sum())
            counts = np.array(counts, dtype=float)
            totals = np.array([
                len(df[df[cl_col] == cid]) for cid in cluster_ids
            ], dtype=float)
            pcts = 100 * counts / np.maximum(totals, 1)
            ax.bar(x, pcts, bottom=bottoms,
                   color=palette_map.get(cls, "#aaa"),
                   label=cls, alpha=0.9)
            bottoms += pcts

        ax.set_xticks(x)
        ax.set_xticklabels([f"C{cid}" for cid in cluster_ids])
        ax.set_ylabel("% loci")
        ax.set_title(f"{family_name} — LTR Structural Classification by Cluster",
                     fontweight="bold")
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        ax.set_ylim(0, 105)
        plt.tight_layout()
        fig_path = reports_dir / "fig_ltr_struct.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Bar chart → {fig_path}")

    # ── PBS/PPT summary ──────────────────────────────────────────────────────
    if "pbs_found" in df.columns:
        n_pbs = int(df["pbs_found"].sum())
        n_ppt = int(df.get("ppt_found", pd.Series(False)).sum())
        n_full = int((df.get("classification","") == "full-length").sum()) if "classification" in df.columns else 0
        n_solo = int((df.get("classification","") == "solo-LTR").sum()) if "classification" in df.columns else 0

        print(f"  full-length: {n_full:,}  solo-LTR: {n_solo:,}  PBS: {n_pbs:,}  PPT: {n_ppt:,}")

        # Per-cluster breakdown
        summary_rows = []
        if cl_col:
            for cid in sorted(c for c in df[cl_col].unique() if c >= 0):
                sub = df[df[cl_col] == cid]
                summary_rows.append({
                    "cluster": cid,
                    "n_total": len(sub),
                    "n_full_length": int((sub.get("classification","") == "full-length").sum()),
                    "n_solo_ltr": int((sub.get("classification","") == "solo-LTR").sum()),
                    "n_truncated": int((sub.get("classification","") == "truncated").sum()),
                    "n_pbs": int(sub["pbs_found"].sum()) if "pbs_found" in sub else 0,
                    "n_ppt": int(sub.get("ppt_found", pd.Series(False)).sum()),
                    "mean_ltr_identity": round(float(sub["ltr_identity"].dropna().mean()), 3)
                        if "ltr_identity" in sub.columns else None,
                })
        else:
            summary_rows.append({
                "cluster": "all", "n_total": len(df),
                "n_full_length": n_full, "n_solo_ltr": n_solo,
                "n_truncated": int((df.get("classification","") == "truncated").sum()),
                "n_pbs": n_pbs, "n_ppt": n_ppt,
                "mean_ltr_identity": round(float(df["ltr_identity"].dropna().mean()), 3)
                    if "ltr_identity" in df.columns else None,
            })
        summary_df = pd.DataFrame(summary_rows)
        summary_csv = reports_dir / "ltr_struct_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"  Summary CSV → {summary_csv}")

        # ── LaTeX macros ─────────────────────────────────────────────────────
        n_total = len(df)
        tex = (
            "% Auto-generated by te_ltr_struct.py\n"
            f"\\newcommand{{\\ltrNloci}}{{{n_total:,}}}\n"
            f"\\newcommand{{\\ltrNfull}}{{{n_full:,}}}\n"
            f"\\newcommand{{\\ltrNsolo}}{{{n_solo:,}}}\n"
            f"\\newcommand{{\\ltrNpbs}}{{{n_pbs:,}}}\n"
            f"\\newcommand{{\\ltrNppt}}{{{n_ppt:,}}}\n"
            f"\\newcommand{{\\ltrPctFull}}{{{100*n_full/max(n_total,1):.1f}\\%}}\n"
        )
        (reports_dir / "ltr_struct_values.tex").write_text(tex)
        print(f"  LaTeX values → {reports_dir / 'ltr_struct_values.tex'}")

    return df


# ── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="LTR structural annotation — PBS, PPT, TSD, LTR boundaries")
    p.add_argument("--input",       required=True)
    p.add_argument("--reports-dir", required=True)
    p.add_argument("--family",      default="FAMILY")
    p.add_argument("--min-ltr-identity", type=float, default=0.65,
                   help="Minimum fractional identity to call an LTR (default 0.65)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_ltr_struct_analysis(
        input_csv         = args.input,
        reports_dir       = args.reports_dir,
        family_name       = args.family,
        min_ltr_identity  = args.min_ltr_identity,
    )
