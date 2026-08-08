#!/usr/bin/env python3
"""
run_grna_analysis.py --- GAMECA gRNA Design Analysis
Expression-weighted CRISPR gRNA design for TE families.

Implements PAM-aware candidate generation, exact and mismatch-tolerant coverage,
expression-weighted coverage maximisation (EWMC), on-target quality scoring,
and a full visualisation suite.

Input CSV must have columns: Seq, Cluster, and one or more numeric expression columns.
Default example: mt2_mm_ultracombo_countsv4.csv

Usage:
    python run_grna_analysis.py \
        --input /path/to/mt2_mm_ultracombo_countsv4.csv \
        --reports-dir ~/gameca_reports \
        [--family MT2_Mm] \
        [--grna-len 20] \
        [--cas SpCas9]

Output figures (saved as PDFs to --reports-dir):
    fig_grna_coverage.pdf       4-panel: distribution / seq-vs-expr scatter /
                                mismatch curve / per-cluster heatmap
    fig_grna_positional.pdf     PAM-site position density in TE sequences
    fig_grna_candidates.pdf     Top gRNA candidates ranked side-by-side

Output text:
    grna_measured_values.tex    LaTeX macros for the report
    grna_report.txt             Human-readable numbers to paste
    grna_candidates.csv         Full ranked candidate table
"""

import argparse
import datetime
import os
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── constants ─────────────────────────────────────────────────────────────────

BASES = "ACGT"

_PAM_TYPES = {
    "SpCas9":    ("NGG",  3),   # 3' PAM, 3 bp
    "SaCas9":    ("NNGRRT", 6), # 3' PAM, 6 bp  (R=A/G, T=T)
    "Cas12a":    ("TTTV",  4),  # 5' PAM, 4 bp  (V=A/C/G)
    "SpRY":      ("NRN",  3),   # relaxed PAM
}

_NON_EXPR = {
    "unnamed: 0","chromosome","start","stop","seq","te_id","te_name",
    "cluster","pca_x","pca_y","umap_x","umap_y","tsne_x","tsne_y",
    "chr","length","gc_content","_total_expr",
}

_PALETTE = [
    "#4C9BE8","#E8604C","#2ECC71","#9B59B6","#F39C12",
    "#1ABC9C","#E74C3C","#3498DB","#E67E22","#8E44AD",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _pp(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _rc(seq):
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(comp)[::-1]


def _auto_expr_cols(df):
    out = []
    for c in df.columns:
        if c.lower().strip() in _NON_EXPR:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


# ── PAM matching ──────────────────────────────────────────────────────────────

def _matches_pam(seq3, pam_pattern):
    """Return True if seq3 matches the IUPAC PAM pattern."""
    _IUPAC = {
        "N": "ACGT", "R": "AG", "Y": "CT", "S": "GC", "W": "AT",
        "K": "GT",   "M": "AC", "B": "CGT","D": "AGT","H": "ACT",
        "V": "ACG",  "A": "A",  "C": "C",  "G": "G",  "T": "T",
    }
    if len(seq3) != len(pam_pattern):
        return False
    return all(s in _IUPAC.get(p, p) for s, p in zip(seq3, pam_pattern))


def extract_pam_grnas(seq, grna_len=20, cas="SpCas9"):
    """Extract all gRNA candidates adjacent to valid PAM sites on both strands.

    Returns list of (grna_seq, strand, start_pos) tuples, where start_pos is
    the position of the protospacer start in the original sequence.

    SpCas9 (NGG, 3' PAM):
        + strand: 5'-[protospacer]-[NGG]-3' → gRNA = protospacer
        - strand: [CCN]-[protospacer_rc] on + strand → gRNA = RC(protospacer_rc)
    """
    seq = seq.upper()
    L = len(seq)
    pam_pattern, pam_len = _PAM_TYPES.get(cas, ("NGG", 3))
    results = []

    if cas in ("SpCas9", "SpRY"):
        # 3' PAM: protospacer then PAM
        # + strand
        for i in range(L - grna_len - pam_len + 1):
            pam = seq[i + grna_len : i + grna_len + pam_len]
            if len(pam) == pam_len and _matches_pam(pam, pam_pattern):
                g = seq[i : i + grna_len]
                if "N" not in g:
                    results.append((g, "+", i))
        # - strand: [CCN] on + strand means NGG on - strand
        # PAM complement = last pam_len bp before the protospacer rc on + strand
        # fwd: 5'-...[RC-PAM-complement][protospacer_complement]...-3'
        rc_pam = _rc(pam_pattern)   # complement (not full RC) for PAM on - strand
        for i in range(pam_len, L - grna_len + 1):
            pam_c = seq[i - pam_len : i]          # should equal RC of PAM pattern
            # Direct check: is seq[i-pam_len:i] the reverse complement of any NGG?
            # For NGG: RC is CCN, reversed = NCC. Actually: RC("NGG") = "CCN" reversed = "NCC"
            # But the PAM on the - strand reads 5'→3' as NGG, which on + strand is CCN reversed
            # On + strand (5'→3') the PAM complement is: seq[j:j+pam_len] = complement(pam reversed)
            # For NGG: the reverse complement is CCN, so we look for seq[j:j+pam_len] ~ CCN
            # i.e., seq[j]=='C' and seq[j+1]=='C' for NGG
            if pam_len == 3 and seq[i-3] == "C" and seq[i-2] == "C":
                proto_c = seq[i : i + grna_len]
                if len(proto_c) == grna_len and "N" not in proto_c:
                    results.append((_rc(proto_c), "-", i))

    elif cas == "Cas12a":
        # 5' PAM: [TTTV]-[protospacer] on + strand; gRNA=protospacer
        # + strand
        for i in range(pam_len, L - grna_len + 1):
            pam = seq[i - pam_len : i]
            if len(pam) == pam_len and _matches_pam(pam, pam_pattern):
                g = seq[i : i + grna_len]
                if "N" not in g:
                    results.append((g, "+", i))
        # - strand
        for i in range(L - grna_len - pam_len + 1):
            pam_c = seq[i + grna_len : i + grna_len + pam_len]
            # TTTV on - strand = BAAA on + strand (reversed complement)
            if _matches_pam(_rc(pam_c)[::-1], pam_pattern):
                g = _rc(seq[i : i + grna_len])
                if "N" not in g:
                    results.append((g, "-", i))

    return results


def _extract_chunk(args):
    """Worker: extract PAM gRNAs for a chunk of (ridx, seq) pairs."""
    ridx_list, seq_list, grna_len, cas = args
    out = []
    for ridx, seq in zip(ridx_list, seq_list):
        for grna, strand, pos in extract_pam_grnas(seq, grna_len=grna_len, cas=cas):
            out.append((grna, ridx, strand, pos))
    return out


# ── on-target quality scoring ─────────────────────────────────────────────────

def on_target_score(grna: str) -> float:
    """Quick on-target quality estimate (0–1) based on known gRNA features.

    Based on published guidelines from Doench et al. 2016 and Moreno-Mateos 2015:
      - GC content 40–70%: preferred
      - No runs of 4+ same nucleotide (homopolymers)
      - Seed region (last 12 bp before PAM): moderate GC preferred
      - No T at the first position (transcriptional termination risk)
      - G at position 20 (immediately 5' of PAM) slightly preferred
    """
    g = grna.upper()
    n = len(g)
    score = 1.0

    # GC content
    gc = (g.count("G") + g.count("C")) / n
    if gc < 0.25 or gc > 0.80:
        score *= 0.5
    elif gc < 0.40 or gc > 0.70:
        score *= 0.8

    # Homopolymer runs
    for base in "ACGT":
        if base * 4 in g:
            score *= 0.6
            break

    # No T at position 0 (U in gRNA → PolIII termination)
    if g[0] == "T":
        score *= 0.85

    # Seed region GC (last 12 bp adjacent to PAM, regardless of guide length)
    seed = g[-12:] if n >= 12 else g
    seed_gc = (seed.count("G") + seed.count("C")) / len(seed)
    if seed_gc < 0.30 or seed_gc > 0.80:
        score *= 0.75

    return round(score, 4)


# ── k-mer index ───────────────────────────────────────────────────────────────

# Above this many sequences, a global all-20-mer index (both strands) of
# full-length elements holds >10^8 distinct k-mers and OOM-kills the job
# (historical SIGKILL/-9). Exact coverage — which drives candidate ranking, the
# greedy set and the UMAP/t-SNE coverage figure — does NOT use this index, so we
# skip it for large inputs and report mm1 columns as not-computed.
GLOBAL_MM_INDEX_MAX_SEQS = 4000


def build_kmer_index_20(df, grna_len=20):
    """Build {kmer: set(row_indices)} for all 20-mers (both strands) in df['Seq'].

    Used for mismatch-tolerant coverage look-ups.
    """
    kmer_to_rows = defaultdict(set)
    seqs = df["Seq"].astype(str).str.upper()
    n = len(seqs)
    for idx, (ridx, seq) in enumerate(seqs.items()):
        if (idx + 1) % 500 == 0:
            _pp(f"    Indexing {idx+1}/{n}...")
        L = len(seq)
        seen = set()
        for i in range(L - grna_len + 1):
            km = seq[i : i + grna_len]
            if "N" not in km and km not in seen:
                kmer_to_rows[km].add(ridx)
                seen.add(km)
            # Also reverse complement
            km_rc = _rc(km)
            if "N" not in km_rc and km_rc not in seen:
                kmer_to_rows[km_rc].add(ridx)
                seen.add(km_rc)
    return kmer_to_rows


# ── mismatch-tolerant coverage ────────────────────────────────────────────────

def _1mm_variants(grna: str) -> list:
    """Return all single-mismatch (Hamming-1) variants of grna."""
    out = []
    for i, c in enumerate(grna):
        for b in BASES:
            if b != c:
                out.append(grna[:i] + b + grna[i+1:])
    return out


def coverage_with_mismatches(grna: str, kmer_to_rows: dict,
                              max_mm: int = 1) -> set:
    """Return set of locus indices covered by grna within max_mm mismatches.

    Uses variant enumeration:
      d=0: O(1) dict look-up
      d=1: 60 single-mismatch variants × O(1) = O(60)
      d=2: 1710 two-mismatch variants × O(1) = O(1710)
    Also enforces a seed-region constraint: the final 12 bp (seed region,
    adjacent to PAM) must match exactly to avoid spurious off-target counting.
    """
    covered = set(kmer_to_rows.get(grna, set()))

    if max_mm >= 1:
        seed = grna[-12:]   # last 12 bp = seed region (closest to PAM)
        for v in _1mm_variants(grna):
            if v[-12:] == seed:   # seed must match exactly
                covered |= kmer_to_rows.get(v, set())

    if max_mm >= 2:
        for v1 in _1mm_variants(grna):
            if v1[-12:] != grna[-12:]:
                continue  # skip seed-region mismatches
            for v2 in _1mm_variants(v1):
                if v2[-12:] == grna[-12:]:
                    covered |= kmer_to_rows.get(v2, set())

    return covered


# ── candidate scoring ─────────────────────────────────────────────────────────

def score_candidates(df, grna_len=20, cas="SpCas9", kmer_index=None,
                     expr_col="_total_expr", n_workers=1) -> pd.DataFrame:
    """For each unique PAM-adjacent gRNA candidate, compute coverage metrics.

    Returns DataFrame with columns:
        grna, gc, on_target_score,
        exact_cov, exact_expr_cov,
        mm1_cov, mm1_expr_cov,
        pct_seq_cov, pct_expr_cov,
        rows_exact (frozenset of locus indices)
    """
    _pp(f"  Extracting PAM-adjacent gRNA candidates ({cas}, len={grna_len}, workers={n_workers})...")
    seqs = df["Seq"].astype(str).str.upper().tolist()
    total_expr = df[expr_col].values

    # Collect all unique candidates + their initial exact coverage from index
    cand_to_exact_rows = defaultdict(set)  # grna → set of row indices (exact)
    cand_to_positions = defaultdict(list)  # grna → list of (row_idx, strand, pos)

    n = len(seqs)
    ridx_list = list(range(n))
    chunk_size = max(1, (n + n_workers - 1) // n_workers)
    chunks = [
        (ridx_list[i:i+chunk_size], seqs[i:i+chunk_size], grna_len, cas)
        for i in range(0, n, chunk_size)
    ]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for chunk_result in ex.map(_extract_chunk, chunks):
            for grna, ridx, strand, pos in chunk_result:
                cand_to_positions[grna].append((ridx, strand, pos))
    _pp(f"    extraction complete ({n} sequences across {len(chunks)} chunks)")

    _pp(f"  {len(cand_to_positions):,} unique PAM-adjacent gRNA candidates")

    # Exact coverage from the per-candidate occurrence sets
    for grna, hits in cand_to_positions.items():
        cand_to_exact_rows[grna] = frozenset(ridx for ridx, _, _ in hits)

    n_loci = len(df)
    total_family_expr = float(total_expr.sum())

    records = []
    for grna, exact_rows in cand_to_exact_rows.items():
        gc = (grna.count("G") + grna.count("C")) / len(grna)
        ots = on_target_score(grna)
        e_cov = int(len(exact_rows))
        e_expr = float(sum(total_expr[r] for r in exact_rows))

        # 1-mismatch coverage from global k-mer index
        mm1_rows = set()
        if kmer_index is not None:
            mm1_rows = coverage_with_mismatches(grna, kmer_index, max_mm=1)
        mm1_cov  = len(mm1_rows)
        mm1_expr = float(sum(total_expr[r] for r in mm1_rows))

        records.append({
            "grna":           grna,
            "gc":             round(gc, 3),
            "on_target_score": ots,
            "exact_cov":      e_cov,
            "exact_expr_cov": round(e_expr, 2),
            "mm1_cov":        mm1_cov,
            "mm1_expr_cov":   round(mm1_expr, 2),
            "pct_seq_cov":    round(100 * e_cov / max(n_loci, 1), 2),
            "pct_expr_cov":   round(100 * e_expr / max(total_family_expr, 1e-9), 2),
            "rows_exact":     exact_rows,
        })

    df_cands = pd.DataFrame(records)
    if not df_cands.empty:
        df_cands.sort_values("exact_expr_cov", ascending=False, inplace=True)
        df_cands.reset_index(drop=True, inplace=True)

    _pp(f"  Scoring done: {len(df_cands):,} candidates")
    return df_cands


# ── greedy multi-gRNA set cover ───────────────────────────────────────────────

def greedy_grna_set(df_cands: pd.DataFrame, df: pd.DataFrame,
                    n_guides: int = 5,
                    mode: str = "expression") -> pd.DataFrame:
    """Iterative greedy maximum-coverage gRNA selection.

    Greedily adds gRNAs that maximise the marginal coverage of uncovered loci.
    mode='expression' → maximise uncovered expression
    mode='sequence'   → maximise uncovered locus count

    Returns DataFrame of selected gRNAs with cumulative coverage stats.
    """
    expr_col = "_total_expr"
    total_expr = df[expr_col].values if expr_col in df.columns else np.ones(len(df))

    uncovered = set(range(len(df)))
    selected = []
    cumulative_loci = 0
    cumulative_expr = 0.0

    for _ in range(n_guides):
        best_grna = None
        best_score = -1.0
        best_rows = set()

        for _, row in df_cands.iterrows():
            marginal = set(row["rows_exact"]) & uncovered
            if not marginal:
                continue
            if mode == "expression":
                score = sum(total_expr[r] for r in marginal)
            else:
                score = len(marginal)
            if score > best_score:
                best_score = score
                best_grna = row["grna"]
                best_rows = marginal

        if best_grna is None:
            break

        marginal_loci = len(best_rows)
        marginal_expr = float(sum(total_expr[r] for r in best_rows))
        cumulative_loci += marginal_loci
        cumulative_expr += marginal_expr
        uncovered -= best_rows
        selected.append({
            "rank":            len(selected) + 1,
            "grna":            best_grna,
            "marginal_loci":   marginal_loci,
            "marginal_expr":   round(marginal_expr, 2),
            "cumulative_loci": cumulative_loci,
            "cumulative_expr": round(cumulative_expr, 2),
            "pct_loci":        round(100 * cumulative_loci / len(df), 2),
            "pct_expr":        round(100 * cumulative_expr / max(total_expr.sum(), 1e-9), 2),
        })

    return pd.DataFrame(selected)


# ── figures ───────────────────────────────────────────────────────────────────

def fig_grna_coverage(df_cands: pd.DataFrame, df: pd.DataFrame,
                      top_grna_seq: str, top_grna_expr: str,
                      family: str, out_path: Path,
                      grna_len: int = 20, has_expression: bool = True):
    """Main results figure.

    With expression → 2×2 panels:
        (a) sequence-coverage distribution, (b) sequence-vs-expression scatter,
        (c) mismatch tolerance of the best guide, (d) per-cluster heatmap.
    Without expression → only the two panels that don't depend on expression,
        (a) mismatch tolerance and (b) per-cluster heatmap. Panels comparing
        sequence-vs-expression coverage are omitted (they'd be degenerate).

    Returns (corr, mm_df); corr is NaN when there is no expression.
    """
    cluster_col = "Cluster" if "Cluster" in df.columns else "cluster"

    # ── Panel: sequence-coverage distribution (expression case only) ──────────
    def _panel_seq_dist(ax):
        ax.hist(df_cands["pct_seq_cov"], bins=60, color="#4C9BE8",
                edgecolor="white", linewidth=0.4, alpha=0.85)
        top_val = df_cands["pct_seq_cov"].max()
        ax.axvline(top_val, color="#E8604C", linestyle="--", linewidth=1.5,
                   label=f"Best: {top_val:.1f}%")
        ax.set_xlabel("Sequence coverage (%)")
        ax.set_ylabel("Number of gRNA candidates")
        ax.set_title("(a) Sequence coverage distribution", fontweight="bold")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    # ── Panel: sequence vs expression scatter (expression case only) ──────────
    def _panel_seq_vs_expr(ax):
        x = df_cands["pct_seq_cov"].values
        y = df_cands["pct_expr_cov"].values
        ax.scatter(x, y, s=3, alpha=0.3, color="#9B59B6", rasterized=True)
        best_seq  = df_cands.loc[df_cands["pct_seq_cov"].idxmax()]
        best_expr = df_cands.loc[df_cands["pct_expr_cov"].idxmax()]
        ax.scatter(best_seq["pct_seq_cov"], best_seq["pct_expr_cov"],
                   s=80, color="#E8604C", zorder=5, label="Best by seq. cov.")
        ax.scatter(best_expr["pct_seq_cov"], best_expr["pct_expr_cov"],
                   s=80, color="#2ECC71", zorder=5, label="Best by expr. cov.")
        c = float(np.corrcoef(x, y)[0, 1])
        ax.set_xlabel("Sequence coverage (%)")
        ax.set_ylabel("Expression coverage (%)")
        ax.set_title(f"(b) Seq. vs expr. coverage  (r = {c:.3f})", fontweight="bold")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        return c

    # ── Panel: mismatch tolerance of the best guide ───────────────────────────
    def _panel_mismatch(ax, letter):
        _pp("  Computing mismatch tolerance curve (may take ~30s)...")
        # Index k-mers at the ACTUAL guide length (was hard-coded to 20, which
        # returned an empty curve for any grna_len != 20).
        kmer_idx = build_kmer_index_20(df.head(min(len(df), 1000)), grna_len=grna_len)
        rank_col = "pct_expr_cov" if has_expression else "pct_seq_cov"
        top_g = df_cands.loc[df_cands[rank_col].idxmax(), "grna"]
        total_expr_sum = df["_total_expr"].sum()
        recs = []
        for mm in range(4):
            rows = coverage_with_mismatches(top_g, kmer_idx, max_mm=mm)
            cov_pct = 100 * len(rows) / max(len(df), 1)
            expr_sum = df["_total_expr"].iloc[list(rows)].sum() if rows else 0.0
            expr_pct = 100 * expr_sum / max(total_expr_sum, 1e-9)
            recs.append({"mm": mm, "cov_pct": cov_pct, "expr_pct": expr_pct})
        mmdf = pd.DataFrame(recs)
        ax.plot(mmdf["mm"], mmdf["cov_pct"], "o-", color="#4C9BE8", linewidth=2,
                markersize=7, label=("Seq. coverage" if has_expression else "Coverage"))
        if has_expression:
            ax.plot(mmdf["mm"], mmdf["expr_pct"], "s-", color="#E8604C",
                    linewidth=2, markersize=7, label="Expr. coverage")
        ax.set_xlabel("Allowed mismatches (non-seed region)")
        ax.set_ylabel("Coverage (%)")
        ttl = "best EWC gRNA" if has_expression else "best gRNA"
        ax.set_title(f"({letter}) Mismatch tolerance --- {ttl}", fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xticks([0, 1, 2, 3])
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.35)
        return mmdf

    # ── Panel: per-cluster coverage heatmap ───────────────────────────────────
    def _panel_cluster_heat(ax, letter):
        clusters = sorted(c for c in df[cluster_col].unique() if c >= 0)
        top5 = df_cands.head(5)["grna"].tolist()
        heat = np.zeros((len(top5), len(clusters)))
        for gi, g in enumerate(top5):
            rows_g = set(df_cands.loc[df_cands["grna"] == g, "rows_exact"].values[0])
            for ci, cl in enumerate(clusters):
                cl_rows = set(df[df[cluster_col] == cl].index.tolist())
                if cl_rows:
                    heat[gi, ci] = 100 * len(rows_g & cl_rows) / len(cl_rows)
        im = ax.imshow(heat, aspect="auto", cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(len(clusters)))
        ax.set_xticklabels([f"C{c}" for c in clusters], fontsize=9)
        ax.set_yticks(range(len(top5)))
        ax.set_yticklabels([f"{g[:12]}…" for g in top5], fontsize=8,
                           fontfamily="monospace")
        plt.colorbar(im, ax=ax, label="% cluster loci covered", shrink=0.85)
        ax.set_title(f"({letter}) Top-5 gRNAs × cluster coverage (%)", fontweight="bold")
        for gi in range(len(top5)):
            for ci in range(len(clusters)):
                ax.text(ci, gi, f"{heat[gi, ci]:.0f}", ha="center", va="center",
                        fontsize=7, color="black" if heat[gi, ci] < 60 else "white")

    if has_expression:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"{family} --- gRNA Coverage Analysis", fontsize=14, fontweight="bold")
        _panel_seq_dist(axes[0, 0])
        corr = _panel_seq_vs_expr(axes[0, 1])
        mm_df = _panel_mismatch(axes[1, 0], "c")
        _panel_cluster_heat(axes[1, 1], "d")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"{family} --- gRNA Coverage Analysis", fontsize=14, fontweight="bold")
        corr = float("nan")
        mm_df = _panel_mismatch(axes[0], "a")
        _panel_cluster_heat(axes[1], "b")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    _pp(f"  Saved {out_path}")
    return corr, mm_df


def fig_grna_positional(df: pd.DataFrame, grna_len=20, cas="SpCas9",
                        family="FAMILY", out_path: Path = None):
    """PAM-site density across the sequence length of a TE family."""
    seqs = df["Seq"].astype(str).str.upper().tolist()
    # Normalise positions to [0, 1) relative to sequence length
    positions_fwd, positions_rev = [], []
    for seq in seqs:
        L = len(seq)
        if L < grna_len + 3:
            continue
        for grna, strand, pos in extract_pam_grnas(seq, grna_len=grna_len, cas=cas):
            frac = pos / L
            if strand == "+":
                positions_fwd.append(frac)
            else:
                positions_rev.append(frac)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f"{family} --- PAM Site Positional Density ({cas})",
                 fontsize=13, fontweight="bold")

    bins = np.linspace(0, 1, 51)
    for ax, pos, label, color in [
        (axes[0], positions_fwd, "+ strand gRNAs", "#4C9BE8"),
        (axes[1], positions_rev, "- strand gRNAs", "#E8604C"),
    ]:
        if pos:
            ax.hist(pos, bins=bins, color=color, edgecolor="white",
                    linewidth=0.4, alpha=0.85, density=True)
        ax.set_ylabel("PAM site density")
        ax.set_title(label, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

    axes[1].set_xlabel("Relative position in sequence (0 = 5′, 1 = 3′)")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    _pp(f"  Saved {out_path}")
    return len(positions_fwd) + len(positions_rev)


def fig_grna_candidates(df_cands: pd.DataFrame, family: str, out_path: Path,
                        has_expression: bool = True):
    """Top gRNA candidates ranked. With expression: two panels (by sequence
    coverage and by expression coverage). Without expression: a single panel
    ranked by sequence coverage (the expression ranking would be identical)."""
    top_seq = df_cands.nlargest(10, "pct_seq_cov")

    if has_expression:
        top_expr = df_cands.nlargest(10, "pct_expr_cov")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        panels = [
            (axes[0], top_seq,  "pct_seq_cov",  "#4C9BE8",
             "(a) Ranked by sequence coverage", "Sequence coverage (%)"),
            (axes[1], top_expr, "pct_expr_cov", "#E8604C",
             "(b) Ranked by expression coverage", "Expression coverage (%)"),
        ]
    else:
        fig, ax0 = plt.subplots(1, 1, figsize=(9, 6))
        panels = [
            (ax0, top_seq, "pct_seq_cov", "#4C9BE8",
             "Ranked by sequence coverage", "Sequence coverage (%)"),
        ]

    fig.suptitle(f"{family} --- Top gRNA Candidates", fontsize=14, fontweight="bold")

    for ax, df_top, metric, color, title, col in panels:
        seqs_short = [f"{g[:14]}…" for g in df_top["grna"]]
        vals = df_top[metric].values
        bars = ax.barh(seqs_short[::-1], vals[::-1], color=color, alpha=0.85,
                       edgecolor="white")
        for bar, ots in zip(bars, df_top["on_target_score"].values[::-1]):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                    f"OTS={ots:.2f}", va="center", fontsize=7.5)
        ax.set_xlabel(col)
        ax.set_title(title, fontweight="bold")
        ax.set_xlim(0, max(vals.max() * 1.2, 5))
        ax.spines[["top","right"]].set_visible(False)
        ax.xaxis.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_yticklabels([s for s in seqs_short[::-1]],
                           fontfamily="monospace")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    _pp(f"  Saved {out_path}")


def _grna_rows(df_cands, grna):
    """Exact-coverage locus indices for one guide (frozenset), or empty set."""
    hit = df_cands.loc[df_cands["grna"] == grna, "rows_exact"]
    return set(hit.values[0]) if len(hit) else set()


def fig_grna_embedding_coverage(df, df_cands, greedy_df, family, out_path,
                                covered_csv=None, embeddings=None):
    """Highlight, on the requested embedding(s), every locus that the optimal
    multi-guide panel (greedy set) can hit by exact protospacer match.

    Two top panels: the full greedy combo, with each locus coloured by the
    FIRST guide (in greedy rank order) that covers it; uncovered loci stay grey.
    Two bottom panels: the single best (rank-1) guide alone — covered vs not.
    This is the slide that says "here are the actual loci our reagents reach".

    Coverage is EXACT protospacer match (the honest, PAM-aware metric); nothing
    here is mismatch-extrapolated.

    ``embeddings`` selects which projections to draw as columns, any of
    ("umap", "tsne", "pca"). Default (None) keeps the historical UMAP+t-SNE view.
    """
    _EMB_SPECS = [("umap", "UMAP", "umap_x", "umap_y"),
                  ("tsne", "t-SNE", "tsne_x", "tsne_y"),
                  ("pca", "PCA", "pca_x", "pca_y")]
    want = tuple(embeddings) if embeddings else ("umap", "tsne")
    available = {key: (name, xc, yc) for key, name, xc, yc in _EMB_SPECS
                 if {xc, yc}.issubset(df.columns)}
    selected = [k for k in want if k in available]
    if not selected:
        _pp("  [embedding-coverage] no requested embedding columns "
            f"({', '.join(want)}) in input — skipping embedding coverage figure")
        return None
    if greedy_df is None or len(greedy_df) == 0:
        _pp("  [embedding-coverage] empty greedy set — skipping")
        return None

    n_loci = len(df)
    guides = list(greedy_df["grna"])

    # First-covering-guide label per locus (greedy/marginal assignment), plus a
    # boolean "covered by any guide in the combo".
    first_guide = np.full(n_loci, -1, dtype=int)   # -1 = uncovered
    any_covered = np.zeros(n_loci, dtype=bool)
    guide_total_hits = []
    for gi, g in enumerate(guides):
        rows = _grna_rows(df_cands, g)
        rows = {r for r in rows if 0 <= r < n_loci}
        guide_total_hits.append(len(rows))
        for r in rows:
            any_covered[r] = True
            if first_guide[r] == -1:
                first_guide[r] = gi

    rank1_rows = {r for r in _grna_rows(df_cands, guides[0]) if 0 <= r < n_loci}
    rank1_mask = np.zeros(n_loci, dtype=bool)
    for r in rank1_rows:
        rank1_mask[r] = True

    # Optional: write the per-locus covering-guide table for reproducibility.
    if covered_csv is not None:
        cov = pd.DataFrame({
            "locus_index": np.arange(n_loci),
            "covered_any": any_covered,
            "first_guide_rank": np.where(first_guide >= 0, first_guide + 1, 0),
            "first_guide_seq": [guides[i] if i >= 0 else "" for i in first_guide],
        })
        for c in ("Cluster", "cluster", "chr", "start", "stop", "strand", "te_name"):
            if c in df.columns:
                cov[c] = df[c].values
        cov.to_csv(covered_csv, index=False)
        _pp(f"  Saved {covered_csv}")

    embeds = []
    for key in selected:
        name, xc, yc = available[key]
        embeds.append((name, df[xc].values, df[yc].values))

    n_cov = int(any_covered.sum())
    pct_cov = 100.0 * n_cov / max(n_loci, 1)
    expr = df["_total_expr"].values if "_total_expr" in df.columns else np.ones(n_loci)
    pct_expr = 100.0 * float(expr[any_covered].sum()) / max(float(expr.sum()), 1e-9)

    fig, axes = plt.subplots(2, len(embeds), figsize=(7.2 * len(embeds), 12),
                             squeeze=False)
    fig.suptitle(
        f"{family} — gRNA reagent coverage on the embeddings\n"
        f"optimal {len(guides)}-guide panel hits {n_cov:,}/{n_loci:,} loci "
        f"({pct_cov:.1f}%), {pct_expr:.1f}% of expression — exact protospacer match",
        fontsize=14, fontweight="bold")

    grey = (0.82, 0.82, 0.82)

    # ── Row 0: full optimal combo, coloured by first covering guide ───────────
    for col, (name, ex, ey) in enumerate(embeds):
        ax = axes[0][col]
        unc = first_guide < 0
        ax.scatter(ex[unc], ey[unc], s=8, color=grey, alpha=0.55,
                   linewidths=0, rasterized=True, label="not covered")
        for gi, g in enumerate(guides):
            m = first_guide == gi
            if not m.any():
                continue
            ax.scatter(ex[m], ey[m], s=14, color=_PALETTE[gi % len(_PALETTE)],
                       alpha=0.9, linewidths=0, rasterized=True,
                       label=f"#{gi+1} {g[:10]}…  (+{int(m.sum())} loci, "
                             f"{guide_total_hits[gi]} total)")
        ax.set_title(f"({chr(97+col)}) Optimal combo on {name}", fontweight="bold")
        ax.set_xlabel(f"{name}-1"); ax.set_ylabel(f"{name}-2")
        ax.spines[["top", "right"]].set_visible(False)
        if col == 0:
            ax.legend(fontsize=7.5, loc="best", markerscale=1.6,
                      title="covering guide (rank)")

    # ── Row 1: best single guide, covered vs not ──────────────────────────────
    for col, (name, ex, ey) in enumerate(embeds):
        ax = axes[1][col]
        ax.scatter(ex[~rank1_mask], ey[~rank1_mask], s=8, color=grey,
                   alpha=0.55, linewidths=0, rasterized=True, label="not hit")
        ax.scatter(ex[rank1_mask], ey[rank1_mask], s=16, color=_PALETTE[0],
                   alpha=0.9, linewidths=0, rasterized=True,
                   label=f"hit by #1 ({int(rank1_mask.sum())} loci)")
        ax.set_title(f"({chr(97+len(embeds)+col)}) Best single guide on {name}\n"
                     f"{guides[0][:20]}", fontweight="bold", fontsize=11)
        ax.set_xlabel(f"{name}-1"); ax.set_ylabel(f"{name}-2")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, loc="best", markerscale=1.4)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    _pp(f"  Saved {out_path}")
    return {"n_covered": n_cov, "pct_loci": pct_cov, "pct_expr": pct_expr}


# ── measured values writer ────────────────────────────────────────────────────

def write_grna_values(results: dict, reports_dir: Path):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tex = [
        "% Auto-generated by run_grna_analysis.py",
        f"% {ts}",
        "",
        rf"\providecommand{{\grnaNCandidates}}{{{results['n_candidates']:,}}}",
        rf"\providecommand{{\grnaNPamSites}}{{{results['n_pam_sites']:,}}}",
        rf"\providecommand{{\grnaTopSeqCovPct}}{{{results['top_seq_cov_pct']:.1f}}}",
        rf"\providecommand{{\grnaTopExprCovPct}}{{{results['top_expr_cov_pct']:.1f}}}",
        rf"\providecommand{{\grnaSeqExprCorr}}{{{results['seq_expr_corr']:.3f}}}",
        rf"\providecommand{{\grnaBestSeqGuide}}{{\texttt{{{results['best_seq_grna']}}}}}",
        rf"\providecommand{{\grnaBestExprGuide}}{{\texttt{{{results['best_expr_grna']}}}}}",
        rf"\providecommand{{\grnaGreedy5Loci}}{{{results['greedy5_loci_pct']:.1f}}}",
        rf"\providecommand{{\grnaGreedy5Expr}}{{{results['greedy5_expr_pct']:.1f}}}",
        rf"\providecommand{{\grnaNLoci}}{{{results['n_loci']:,}}}",
        "",
    ]
    (reports_dir / "grna_measured_values.tex").write_text("\n".join(tex))

    txt = [
        "=" * 60,
        "GAMECA gRNA Design --- Measured Values",
        f"Generated: {ts}",
        "=" * 60,
        f"  Family:                   {results['family']}",
        f"  Total loci:               {results['n_loci']:,}",
        f"  PAM-adjacent candidates:  {results['n_candidates']:,}",
        f"  PAM sites total:          {results['n_pam_sites']:,}",
        "",
        "  ── Top gRNAs ─────────────────────────────",
        f"  Best by seq. coverage:    {results['best_seq_grna']}",
        f"    Seq. coverage:          {results['top_seq_cov_pct']:.1f}%",
        f"  Best by expr. coverage:   {results['best_expr_grna']}",
        f"    Expr. coverage:         {results['top_expr_cov_pct']:.1f}%",
        f"  Seq/expr coverage corr.:  r = {results['seq_expr_corr']:.3f}",
        "",
        "  ── Greedy 5-guide set ────────────────────",
        f"  Cumulative loci covered:  {results['greedy5_loci_pct']:.1f}%",
        f"  Cumulative expr covered:  {results['greedy5_expr_pct']:.1f}%",
    ]
    (reports_dir / "grna_report.txt").write_text("\n".join(txt))
    _pp(f"  Written grna_measured_values.tex and grna_report.txt")


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Expression-weighted gRNA design for TE families",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input",       required=True,
                   help="Path to input CSV (Seq + expression columns)")
    p.add_argument("--reports-dir", default=os.path.expanduser("~/gameca_reports"),
                   help="Output directory for figures and measured_values.tex")
    p.add_argument("--family",      default="MT2_Mm",
                   help="Family name for plot titles")
    p.add_argument("--grna-len",    type=int, default=20,
                   help="gRNA spacer length (default: 20, the standard SpCas9 "
                        "spacer; use 18 for truncated 'tru-gRNAs')")
    p.add_argument("--cas",         default="SpCas9",
                   choices=list(_PAM_TYPES.keys()),
                   help="Cas protein / PAM type (default: SpCas9)")
    p.add_argument("--max-candidates", type=int, default=None,
                   help="Cap number of candidates for speed (default: all)")
    p.add_argument("--n-greedy",    type=int, default=5,
                   help="Number of gRNAs in greedy set-cover solution")
    p.add_argument("--global-mm-index", choices=["auto", "on", "off"],
                   default="auto",
                   help="Build the global both-strand 20-mer index for mm1 "
                        "coverage columns. 'auto' (default) skips it above "
                        f"{GLOBAL_MM_INDEX_MAX_SEQS} sequences to avoid OOM; "
                        "exact coverage (used everywhere downstream) is "
                        "unaffected.")
    p.add_argument("--n-workers", type=int, default=4,
                   help="Parallel workers for PAM extraction (default: 4)")
    return p.parse_args()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    _pp("=" * 60)
    _pp(f"GAMECA gRNA Design --- {args.family}")
    _pp(f"  Input:    {args.input}")
    _pp(f"  Cas:      {args.cas}  PAM: {_PAM_TYPES[args.cas][0]}")
    _pp(f"  gRNA len: {args.grna_len} bp")
    _pp(f"  Workers:  {args.n_workers}")
    _pp(f"  Reports:  {reports_dir}")
    _pp("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    _pp("Loading CSV...")
    df = pd.read_csv(args.input)
    _pp(f"  {len(df):,} rows, {len(df.columns)} columns")

    expr_cols = _auto_expr_cols(df)
    has_expression = bool(expr_cols)
    _pp(f"  Expression columns ({len(expr_cols)}): {expr_cols}")
    # No expression → weight every locus equally so coverage == sequence coverage
    # (previously 0.0, which made every expression metric degenerate).
    df["_total_expr"] = df[expr_cols].sum(axis=1) if has_expression else 1.0

    # ── Build global 20-mer index (mm1 columns only; OOM-prone, so optional) ──
    want_index = (args.global_mm_index == "on" or
                  (args.global_mm_index == "auto" and
                   len(df) <= GLOBAL_MM_INDEX_MAX_SEQS))
    if want_index:
        _pp("Building global 20-mer index (both strands)...")
        kmer_index = build_kmer_index_20(df, grna_len=args.grna_len)
        _pp(f"  {len(kmer_index):,} unique 20-mers indexed")
    else:
        kmer_index = None
        _pp(f"Skipping global 20-mer index ({len(df):,} seqs > "
            f"{GLOBAL_MM_INDEX_MAX_SEQS} or --global-mm-index=off): "
            "mm1_* columns will be 0; exact coverage drives all downstream steps.")

    # ── Score PAM-adjacent candidates ─────────────────────────────────────────
    _pp("Scoring gRNA candidates...")
    df_cands = score_candidates(
        df, grna_len=args.grna_len, cas=args.cas,
        kmer_index=kmer_index, expr_col="_total_expr",
        n_workers=args.n_workers,
    )

    if df_cands.empty:
        _pp("FATAL: no PAM-adjacent candidates found --- check sequences")
        sys.exit(1)

    n_pam_sites = int(df_cands["exact_cov"].sum())  # total PAM site occurrences

    if args.max_candidates:
        df_cands = df_cands.head(args.max_candidates)

    # Save full candidate table. Drop the expression columns when there is no
    # expression data, so the table doesn't carry meaningless duplicate columns.
    cand_out = df_cands.drop(columns=["rows_exact"]).copy()
    if not has_expression:
        cand_out = cand_out.drop(columns=[c for c in
                    ("exact_expr_cov", "mm1_expr_cov", "pct_expr_cov")
                    if c in cand_out.columns])
    cand_out.to_csv(reports_dir / "grna_candidates.csv", index=False)
    _pp(f"  Saved grna_candidates.csv ({len(cand_out):,} candidates)")

    # ── Top candidates ────────────────────────────────────────────────────────
    best_seq  = df_cands.loc[df_cands["pct_seq_cov"].idxmax()]
    best_expr = df_cands.loc[df_cands["pct_expr_cov"].idxmax()]
    _pp(f"  Best by sequence coverage:   {best_seq['grna']}  ({best_seq['pct_seq_cov']:.1f}%)")
    _pp(f"  Best by expression coverage: {best_expr['grna']}  ({best_expr['pct_expr_cov']:.1f}%)")

    # ── Greedy set-cover ──────────────────────────────────────────────────────
    greedy_mode = "expression" if has_expression else "sequence"
    _pp(f"Running greedy {args.n_greedy}-guide set-cover ({greedy_mode} mode)...")
    greedy_df = greedy_grna_set(df_cands, df, n_guides=args.n_greedy, mode=greedy_mode)
    greedy_df.to_csv(reports_dir / "grna_greedy_set.csv", index=False)
    _pp(f"  Greedy {len(greedy_df)} guides:")
    for _, row in greedy_df.iterrows():
        _pp(f"    #{row['rank']} {row['grna']}  +{row['marginal_loci']} loci "
            f"  cumulative: {row['pct_loci']:.1f}% loci / {row['pct_expr']:.1f}% expr")

    greedy5_loci = greedy_df["pct_loci"].max() if len(greedy_df) > 0 else 0.0
    greedy5_expr = greedy_df["pct_expr"].max() if len(greedy_df) > 0 else 0.0

    # ── Figures ───────────────────────────────────────────────────────────────
    _pp("Generating figures...")

    corr, mm_df = fig_grna_coverage(
        df_cands, df,
        top_grna_seq=best_seq["grna"],
        top_grna_expr=best_expr["grna"],
        family=args.family,
        out_path=reports_dir / "fig_grna_coverage.pdf",
        grna_len=args.grna_len, has_expression=has_expression,
    )

    n_pam = fig_grna_positional(
        df, grna_len=args.grna_len, cas=args.cas,
        family=args.family,
        out_path=reports_dir / "fig_grna_positional.pdf",
    )

    fig_grna_candidates(df_cands, args.family,
                        out_path=reports_dir / "fig_grna_candidates.pdf",
                        has_expression=has_expression)

    # The reagent-design centrepiece: where the optimal guide panel lands on the
    # UMAP / t-SNE embeddings (requires umap_x/umap_y / tsne_x/tsne_y in input).
    emb_cov = fig_grna_embedding_coverage(
        df, df_cands, greedy_df, args.family,
        out_path=reports_dir / "fig_grna_umap_tsne_coverage.pdf",
        covered_csv=reports_dir / "grna_covered_loci.csv",
    )
    if emb_cov:
        _pp(f"  Embedding coverage: optimal panel hits {emb_cov['pct_loci']:.1f}% "
            f"of loci / {emb_cov['pct_expr']:.1f}% of expression")

    # ── Write measured values ─────────────────────────────────────────────────
    results = {
        "family":           args.family,
        "n_loci":           len(df),
        "n_candidates":     len(df_cands),
        "n_pam_sites":      n_pam,
        "top_seq_cov_pct":  float(best_seq["pct_seq_cov"]),
        "top_expr_cov_pct": float(best_expr["pct_expr_cov"]),
        "best_seq_grna":    best_seq["grna"],
        "best_expr_grna":   best_expr["grna"],
        "seq_expr_corr":    float(corr),
        "greedy5_loci_pct": greedy5_loci,
        "greedy5_expr_pct": greedy5_expr,
    }
    write_grna_values(results, reports_dir)

    _pp("=" * 60)
    _pp("DONE")
    _pp(f"  Figures:    {reports_dir}")
    _pp(f"  Candidates: {reports_dir / 'grna_candidates.csv'}")
    _pp(f"  Report:     {reports_dir / 'grna_report.txt'}")
    _pp("=" * 60)

    print("\n" + "=" * 60)
    print("MEASURED VALUES (paste into chat):")
    print((reports_dir / "grna_report.txt").read_text())


if __name__ == "__main__":
    main()
