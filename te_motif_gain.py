#!/usr/bin/env python3
"""
te_motif_gain.py — TF motif gain/loss annotation for GAMECA
─────────────────────────────────────────────────────────────────────────────
For each TE copy, compares its JASPAR TF binding sites (from all_overlaps.tsv)
against the Dfam canonical sequence to identify:

  GAINED — motif present in this copy but the corresponding k-mer is absent
            from the canonical (≥3 mismatches to every canonical window of the
            same length), meaning the binding site arose from copy-specific
            substitutions.

  LOST   — motif found in ≥LOST_FRAC of all copies (and confirmed in the
            canonical by reverse lookup) but absent from this copy.

The canonical comparison uses a mismatch-tolerant sliding-window scan
(Hamming distance ≤ max_mm, default 2) so that synonymous or near-neutral
single-nucleotide changes in the copy do not spuriously trigger "gained".

Inputs (auto-detected from --input path):
  motif_analysis/all_overlaps.tsv   produced by te_motif.py (Stage 8)
  cluster_alignments/all_cluster_consensuses.fa  (fallback canonical)

External:
  Dfam REST API (optional, falls back to cluster consensus if unreachable)

Outputs (in --reports-dir):
  motif_gain_per_copy.csv     per-locus gain/loss summary
  motif_gain_summary.csv      per-motif frequency across copies
  fig_motif_gains_bar.png     top-20 gained motifs barplot
  fig_motif_gains_heatmap.png cluster × top-motif heatmap
  motif_gain_values.tex       LaTeX macros

Standalone:
    python te_motif_gain.py --input clustered.csv --reports-dir ./reports \\
        --family LTR5_Hs --assembly hg38
"""

import argparse
import datetime
import json
import math
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── constants ────────────────────────────────────────────────────────────────

_DEFAULT_MAX_MM  = 2      # mismatches allowed when checking canonical
_LOST_FRAC       = 0.50   # motif must appear in ≥50% of copies to count as "lost"
_MIN_KMER_LEN    = 6
_MAX_KMER_LEN    = 35
_TOP_N_PLOT      = 20


# ── helpers ──────────────────────────────────────────────────────────────────

def _pp(msg: str) -> None:
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def _texesc(s: str) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def _revcomp(seq: str) -> str:
    comp = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq[::-1].translate(comp)


# ── Dfam canonical fetch (mirrors te_subfamily.py) ───────────────────────────

def _fetch_dfam_consensus(family_name: str, build: str = "hg38",
                          timeout: int = 15) -> str | None:
    """Fetch the Dfam canonical consensus via REST API. Returns sequence string or None."""
    base = "https://www.dfam.org/api"
    for name_try in [family_name, family_name.split("-")[0], family_name.split("_")[0]]:
        try:
            url = f"{base}/families?name={urllib.request.quote(name_try)}&format=json&limit=1"
            req = urllib.request.Request(url, headers={"User-Agent": "GAMECA/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
            hits = data.get("results", [])
            if not hits:
                continue
            acc = hits[0].get("accession", "")
            if not acc:
                continue
            seq_url = f"{base}/families/{acc}/sequence?format=json"
            req2 = urllib.request.Request(seq_url, headers={"User-Agent": "GAMECA/1.0"})
            with urllib.request.urlopen(req2, timeout=timeout) as r2:
                seq_data = json.loads(r2.read())
            seq = (seq_data.get("sequence") or seq_data.get("cons_seq") or "").strip().upper()
            if len(seq) >= 20:
                _pp(f"  Dfam canonical fetched: {acc}  ({len(seq)} bp)")
                return seq
        except Exception:
            pass
    return None


def _consensus_from_fasta(fa_path: Path) -> str | None:
    """Return the first sequence from a FASTA file."""
    try:
        seq_lines = []
        with open(fa_path) as fh:
            in_seq = False
            for line in fh:
                line = line.strip()
                if line.startswith(">"):
                    if seq_lines:
                        break
                    in_seq = True
                elif in_seq:
                    seq_lines.append(line.upper())
        seq = "".join(seq_lines)
        if len(seq) >= 20:
            return seq
    except Exception:
        pass
    return None


# ── canonical window cache ────────────────────────────────────────────────────

def _build_canonical_cache(canonical: str) -> dict[int, list[str]]:
    """Build {length: [windows]} for all window lengths likely seen in JASPAR motifs."""
    can = canonical.upper()
    cache: dict[int, list[str]] = {}
    for L in range(_MIN_KMER_LEN, min(_MAX_KMER_LEN + 1, len(can) + 1)):
        cache[L] = [can[i:i+L] for i in range(len(can) - L + 1)]
    return cache


# Result cache: kmer → bool (is kmer in canonical within max_mm?)
_gain_cache: dict[tuple[str, int], bool] = {}


def _kmer_in_canonical(kmer: str, cache: dict[int, list[str]], max_mm: int) -> bool:
    """Return True if kmer (or its revcomp) matches any canonical window ≤ max_mm mismatches."""
    L = len(kmer)
    key = (kmer, max_mm)
    if key in _gain_cache:
        return _gain_cache[key]
    km  = kmer.upper()
    km_rc = _revcomp(km)
    result = False
    for w in cache.get(L, []):
        if sum(a != b for a, b in zip(km, w)) <= max_mm:
            result = True
            break
        if sum(a != b for a, b in zip(km_rc, w)) <= max_mm:
            result = True
            break
    _gain_cache[key] = result
    return result


# ── overlap TSV loader ────────────────────────────────────────────────────────

def _load_overlaps(overlaps_path: Path) -> pd.DataFrame | None:
    """Load all_overlaps.tsv, normalise to expected columns."""
    if not overlaps_path.exists():
        return None
    try:
        df = pd.read_csv(overlaps_path, sep="\t")
    except Exception as exc:
        _pp(f"  WARNING: could not load {overlaps_path}: {exc}")
        return None
    df.columns = [str(c).strip() for c in df.columns]

    # Normalise coordinate columns
    for cand in ("chr", "chrom", "chromosome", "#chrom", "genoName"):
        if cand in df.columns:
            df.rename(columns={cand: "te_chr"}, inplace=True); break
    for cand in ("start", "chromStart", "genoStart"):
        if cand in df.columns:
            df.rename(columns={cand: "te_start"}, inplace=True); break
    for cand in ("stop", "end", "chromEnd", "genoEnd"):
        if cand in df.columns:
            df.rename(columns={cand: "te_stop"}, inplace=True); break

    needed = {"te_chr", "te_start", "te_stop", "Motif_name"}
    if not needed.issubset(df.columns):
        _pp(f"  WARNING: overlaps missing columns {needed - set(df.columns)}")
        return None

    for col in ("te_start", "te_stop"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("Motif_start", "Motif_end"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ── main analysis ─────────────────────────────────────────────────────────────

def run_motif_gain_analysis(
    input_csv:    str | Path,
    reports_dir:  str | Path,
    family_name:  str,
    assembly:     str  = "hg38",
    overlaps_tsv: str | Path | None = None,
    max_mm:       int  = _DEFAULT_MAX_MM,
    consensus_fasta: str | Path | None = None,
) -> None:
    """
    Main entry point. All output is written to reports_dir.
    """
    _gain_cache.clear()

    rdir   = Path(reports_dir);    rdir.mkdir(parents=True, exist_ok=True)
    incsv  = Path(input_csv)
    ts     = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    _pp("=" * 60)
    _pp(f"GAMECA TF Motif Gain/Loss  ---  {family_name}")
    _pp("=" * 60)

    # ── 1. Load clustered CSV ──────────────────────────────────────────────
    try:
        df_copies = pd.read_csv(incsv, sep=None, engine="python")
    except Exception as exc:
        _pp(f"  ERROR: cannot load {incsv}: {exc}")
        return
    df_copies.columns = [str(c).strip() for c in df_copies.columns]

    # Normalise coordinate columns
    ren = {}
    for cand in ("chr", "chrom", "chromosome", "#chrom", "genoName"):
        if cand in df_copies.columns: ren[cand] = "chr"; break
    for cand in ("start", "chromStart", "genoStart"):
        if cand in df_copies.columns: ren[cand] = "start"; break
    for cand in ("stop", "end", "chromEnd", "genoEnd"):
        if cand in df_copies.columns: ren[cand] = "stop"; break
    df_copies.rename(columns=ren, inplace=True)

    if "Seq" not in df_copies.columns:
        _pp("  ERROR: 'Seq' column not found in input CSV. Cannot extract motif windows.")
        _write_empty(rdir, family_name, ts, reason="No Seq column")
        return

    for col in ("start", "stop"):
        if col in df_copies.columns:
            df_copies[col] = pd.to_numeric(df_copies[col], errors="coerce")

    cl_col = next((c for c in ("Cluster", "cluster", "CLUSTER") if c in df_copies.columns), None)
    _pp(f"  {len(df_copies)} loci loaded  |  cluster col: {cl_col}")

    # ── 2. Locate all_overlaps.tsv ────────────────────────────────────────
    if overlaps_tsv is None:
        motif_dir = incsv.parent.parent / "motif_analysis"
        overlaps_path = motif_dir / "all_overlaps.tsv"
    else:
        overlaps_path = Path(overlaps_tsv)

    df_ov = _load_overlaps(overlaps_path)
    if df_ov is None or df_ov.empty:
        _pp(f"  WARNING: {overlaps_path} not found or empty — motif gain requires Stage 8 output.")
        _write_empty(rdir, family_name, ts,
                     reason=f"all_overlaps.tsv not found at {overlaps_path}")
        return
    _pp(f"  {len(df_ov)} overlap rows loaded from {overlaps_path}")

    # ── 3. Fetch / build canonical sequence ───────────────────────────────
    canonical = None

    # Try Dfam first
    canonical = _fetch_dfam_consensus(family_name, build=assembly)

    # Fallback: consensus FASTA from Stage 7 (cluster_alignments/)
    if canonical is None:
        if consensus_fasta is not None:
            canonical = _consensus_from_fasta(Path(consensus_fasta))
        if canonical is None:
            for fa in [
                incsv.parent.parent / "cluster_alignments" / "all_cluster_consensuses.fa",
                incsv.parent / "all_cluster_consensuses.fa",
            ]:
                canonical = _consensus_from_fasta(fa)
                if canonical:
                    _pp(f"  Canonical from consensus FASTA: {fa}")
                    break

    if canonical is None:
        _pp("  WARNING: Cannot fetch canonical sequence (Dfam unreachable, no consensus FASTA).")
        _pp("  Falling back to majority-vote consensus from copy sequences.")
        seqs = [s for s in df_copies["Seq"].dropna() if isinstance(s, str) and len(s) >= 20]
        if seqs:
            max_len = max(len(s) for s in seqs)
            canonical = "".join(
                max("ACGT", key=lambda b: sum(1 for s in seqs
                                              if i < len(s) and s[i].upper() == b))
                for i in range(max_len)
            )
            _pp(f"  Majority-vote canonical: {len(canonical)} bp from {len(seqs)} copies")
        else:
            _pp("  ERROR: No valid sequences in Seq column. Cannot proceed.")
            _write_empty(rdir, family_name, ts, reason="No valid canonical")
            return

    canonical_len = len(canonical)
    _pp(f"  Canonical length: {canonical_len} bp  |  max_mm={max_mm}")

    # ── 4. Build canonical window cache ───────────────────────────────────
    can_cache = _build_canonical_cache(canonical)
    _pp(f"  Canonical cache built: {len(can_cache)} length bins")

    # ── 5. Build locus → (Seq, Cluster) lookup ────────────────────────────
    if "chr" in df_copies.columns and "start" in df_copies.columns:
        locus_seq     = {}
        locus_cluster = {}
        for _, row in df_copies.iterrows():
            key = (str(row.get("chr", "")), int(row["start"]) if pd.notna(row.get("start")) else -1)
            locus_seq[key]     = str(row.get("Seq", "")) if pd.notna(row.get("Seq")) else ""
            locus_cluster[key] = row.get(cl_col, -1) if cl_col else -1
    else:
        _pp("  ERROR: chr/start columns required to match overlaps to sequences.")
        _write_empty(rdir, family_name, ts, reason="No chr/start columns")
        return

    # ── 6. Per-(locus, motif) gain/loss classification ────────────────────
    _pp("  Classifying motif gains and losses …")

    records = []  # {locus_key, motif_name, is_gained}
    total_pairs = 0

    has_mstart = "Motif_start" in df_ov.columns and "Motif_end" in df_ov.columns

    for _, row in df_ov.iterrows():
        te_chr   = str(row["te_chr"])
        te_start = int(row["te_start"]) if pd.notna(row["te_start"]) else -1
        motif_nm = str(row["Motif_name"])

        lkey = (te_chr, te_start)
        seq  = locus_seq.get(lkey, "")
        if not seq or len(seq) < 6:
            continue

        total_pairs += 1

        # Extract k-mer at motif position (if coordinates available)
        kmer = ""
        if has_mstart and pd.notna(row.get("Motif_start")) and pd.notna(row.get("Motif_end")):
            m_start = int(row["Motif_start"])
            m_end   = int(row["Motif_end"])
            offset  = m_start - te_start
            mlen    = m_end - m_start
            if _MIN_KMER_LEN <= mlen <= _MAX_KMER_LEN and offset >= 0:
                end_idx = offset + mlen
                if end_idx <= len(seq):
                    kmer = seq[offset:end_idx].upper()

        # If we can't extract a specific k-mer, skip (we can't do the comparison)
        if not kmer or len(kmer) < _MIN_KMER_LEN:
            continue

        # Check gain: is this k-mer absent from the canonical?
        in_can = _kmer_in_canonical(kmer, can_cache, max_mm)
        records.append({
            "te_chr":     te_chr,
            "te_start":   te_start,
            "te_stop":    int(row.get("te_stop", te_start)) if pd.notna(row.get("te_stop")) else te_start,
            "cluster":    locus_cluster.get(lkey, -1),
            "motif_name": motif_nm,
            "kmer":       kmer,
            "in_canonical": in_can,
            "is_gained":  not in_can,
        })

    _pp(f"  {total_pairs} (locus, motif) pairs evaluated  |  {len(records)} with extractable k-mers")

    if not records:
        _pp("  No k-mers could be extracted — likely Motif_start/Motif_end missing from overlaps TSV.")
        _write_empty(rdir, family_name, ts, reason="No extractable k-mers from overlaps")
        return

    df_pairs = pd.DataFrame(records)
    n_gained = int(df_pairs["is_gained"].sum())
    _pp(f"  {n_gained} gained events  |  {len(df_pairs) - n_gained} canonical-level events")

    # ── 7. Per-copy summary ────────────────────────────────────────────────
    def _join_names(group):
        return "|".join(sorted(set(group)))

    gained_df = df_pairs[df_pairs["is_gained"]]
    per_copy_gained = (
        gained_df.groupby(["te_chr", "te_start", "te_stop", "cluster"])["motif_name"]
        .agg(_join_names).reset_index()
        .rename(columns={"motif_name": "gained_motifs"})
    )
    per_copy_gained["n_gained"] = per_copy_gained["gained_motifs"].str.count(r"\|") + 1

    # Merge with full copy list so non-gained copies show up with 0
    merge_cols = ["chr", "start", "stop"] + ([cl_col] if cl_col else [])
    df_base = df_copies[merge_cols].copy().rename(columns={
        "chr": "te_chr", "start": "te_start", "stop": "te_stop",
        **({cl_col: "cluster"} if cl_col else {})
    })
    df_per_copy = df_base.merge(per_copy_gained, on=["te_chr", "te_start", "te_stop"],
                                 how="left", suffixes=("", "_dup"))
    if "cluster_dup" in df_per_copy.columns:
        df_per_copy.drop(columns=["cluster_dup"], inplace=True)
    df_per_copy["gained_motifs"] = df_per_copy["gained_motifs"].fillna("")
    df_per_copy["n_gained"] = df_per_copy["n_gained"].fillna(0).astype(int)
    df_per_copy.to_csv(rdir / "motif_gain_per_copy.csv", index=False)
    _pp(f"  Wrote {rdir / 'motif_gain_per_copy.csv'}")

    # ── 8. Per-motif summary ───────────────────────────────────────────────
    n_loci = len(df_copies)
    motif_summary = (
        df_pairs.groupby("motif_name")
        .agg(n_gained=("is_gained", "sum"),
             n_total=("is_gained", "count"))
        .reset_index()
    )
    motif_summary["pct_gained"] = (
        100.0 * motif_summary["n_gained"] / motif_summary["n_total"]
    ).round(1)
    motif_summary["copies_with_gain"] = (
        gained_df.groupby("motif_name")["te_start"].nunique().reindex(
            motif_summary["motif_name"], fill_value=0).values
    )
    motif_summary["pct_copies"] = (
        100.0 * motif_summary["copies_with_gain"] / max(n_loci, 1)
    ).round(1)

    # Enriched clusters per motif
    if "cluster" in gained_df.columns:
        def _cluster_list(grp):
            return "|".join(str(int(c)) for c in sorted(grp.unique()) if pd.notna(c))
        enriched = gained_df.groupby("motif_name")["cluster"].apply(_cluster_list)
        motif_summary = motif_summary.merge(
            enriched.rename("enriched_clusters"), on="motif_name", how="left")
    motif_summary.sort_values("n_gained", ascending=False, inplace=True)
    motif_summary.to_csv(rdir / "motif_gain_summary.csv", index=False)
    _pp(f"  Wrote {rdir / 'motif_gain_summary.csv'}")

    top_motif    = motif_summary.iloc[0]["motif_name"] if len(motif_summary) else "—"
    n_copies_any = int((df_per_copy["n_gained"] > 0).sum())
    n_unique_mot = int((motif_summary["n_gained"] > 0).sum())

    # ── 9. Figures ─────────────────────────────────────────────────────────

    # Bar chart: top-N gained motifs
    top = motif_summary[motif_summary["n_gained"] > 0].head(_TOP_N_PLOT)
    fig, ax = plt.subplots(figsize=(10, 5))
    if not top.empty:
        bars = ax.barh(top["motif_name"][::-1], top["n_gained"][::-1],
                       color="#8e44ad", edgecolor="white")
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{int(w)}", va="center", fontsize=7)
        ax.set_xlabel("Number of copies with gained binding site")
        ax.set_title(f"{_texesc(family_name)} — Top gained TF motifs "
                     f"(absent from canonical, max_mm={max_mm})", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No gained motifs detected", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_title(f"{_texesc(family_name)} — TF motif gain analysis", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(rdir / "fig_motif_gains_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    _pp(f"  Saved {rdir / 'fig_motif_gains_bar.png'}")

    # Heatmap: cluster × top-N motifs (fraction of copies with gain)
    if cl_col and "cluster" in df_per_copy.columns and not gained_df.empty:
        top_motifs = motif_summary[motif_summary["n_gained"] > 0].head(min(30, _TOP_N_PLOT))
        motif_list = top_motifs["motif_name"].tolist()
        clusters   = sorted(df_per_copy["cluster"].dropna().unique())

        heat_data  = np.zeros((len(clusters), len(motif_list)))
        for mi, mot in enumerate(motif_list):
            sub = gained_df[gained_df["motif_name"] == mot]
            for ci, cl in enumerate(clusters):
                n_cl    = int((df_per_copy["cluster"] == cl).sum())
                n_gain  = int(((sub["cluster"] == cl)).sum()) if "cluster" in sub.columns else 0
                heat_data[ci, mi] = (n_gain / n_cl) if n_cl > 0 else 0.0

        fig, ax = plt.subplots(figsize=(max(8, len(motif_list) * 0.5), max(4, len(clusters) * 0.7)))
        im = ax.imshow(heat_data, aspect="auto", cmap="Purples", vmin=0, vmax=1,
                       interpolation="nearest")
        ax.set_xticks(range(len(motif_list)))
        ax.set_xticklabels(motif_list, rotation=75, ha="right", fontsize=7)
        ax.set_yticks(range(len(clusters)))
        ax.set_yticklabels([f"Cl {int(c)}" for c in clusters], fontsize=8)
        ax.set_title(f"{_texesc(family_name)} — fraction of copies per cluster "
                     "with gained TF motif", fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.7, label="fraction of copies")
        plt.tight_layout()
        plt.savefig(rdir / "fig_motif_gains_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()
        _pp(f"  Saved {rdir / 'fig_motif_gains_heatmap.png'}")

    # ── 10. LaTeX macros ────────────────────────────────────────────────────
    tex = [
        "% Auto-generated by te_motif_gain.py", f"% {ts}", "",
        rf"\providecommand{{\motifGainFamily}}{{{_texesc(family_name)}}}",
        rf"\providecommand{{\motifGainTotal}}{{{n_gained}}}",
        rf"\providecommand{{\motifGainNCopies}}{{{n_copies_any}}}",
        rf"\providecommand{{\motifGainNFamily}}{{{n_unique_mot}}}",
        rf"\providecommand{{\motifGainTopName}}{{{_texesc(top_motif)}}}",
        rf"\providecommand{{\motifGainCanonLen}}{{{canonical_len}}}",
        rf"\providecommand{{\motifGainMaxMM}}{{{max_mm}}}",
        "",
    ]
    (rdir / "motif_gain_values.tex").write_text("\n".join(tex))
    _pp(f"  Wrote {rdir / 'motif_gain_values.tex'}")

    _pp("=" * 60)
    _pp(f"DONE  |  gained events={n_gained}  |  copies with gain={n_copies_any}"
        f"  |  unique gained motifs={n_unique_mot}")
    print("\n" + "=" * 60)
    print("MEASURED VALUES (paste into chat):")
    print("=" * 60)
    print(f"  Family:              {family_name}")
    print(f"  Loci analysed:       {n_loci}")
    print(f"  Gained events:       {n_gained}")
    print(f"  Copies with gain:    {n_copies_any}")
    print(f"  Unique gained TFs:   {n_unique_mot}")
    print(f"  Top gained motif:    {top_motif}")
    print(f"  Canonical length:    {canonical_len} bp")
    print(f"  Max mismatches:      {max_mm}")


def _write_empty(rdir: Path, family_name: str, ts: str, reason: str) -> None:
    """Write placeholder outputs when analysis cannot run."""
    pd.DataFrame(columns=["te_chr", "te_start", "te_stop", "cluster",
                           "gained_motifs", "n_gained"]).to_csv(
        rdir / "motif_gain_per_copy.csv", index=False)
    pd.DataFrame(columns=["motif_name", "n_gained", "n_total",
                           "pct_gained", "copies_with_gain"]).to_csv(
        rdir / "motif_gain_summary.csv", index=False)
    for fig_name in ("fig_motif_gains_bar.png", "fig_motif_gains_heatmap.png"):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, f"TF motif gain analysis unavailable\n({reason})",
                ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title(f"{family_name} — TF motif gain", fontweight="bold")
        plt.tight_layout()
        plt.savefig(rdir / fig_name, dpi=120, bbox_inches="tight")
        plt.close()
    tex = [
        "% Auto-generated by te_motif_gain.py (no data)", f"% {ts}", "",
        rf"\providecommand{{\motifGainFamily}}{{{_texesc(family_name)}}}",
        rf"\providecommand{{\motifGainTotal}}{{0}}",
        rf"\providecommand{{\motifGainNCopies}}{{0}}",
        rf"\providecommand{{\motifGainNFamily}}{{0}}",
        rf"\providecommand{{\motifGainTopName}}{{---}}",
        rf"\providecommand{{\motifGainCanonLen}}{{0}}",
        rf"\providecommand{{\motifGainMaxMM}}{{{_DEFAULT_MAX_MM}}}",
        "",
    ]
    (rdir / "motif_gain_values.tex").write_text("\n".join(tex))
    _pp(f"  Wrote placeholder outputs  ({reason})")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="GAMECA TF motif gain/loss annotation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--input",        required=True, help="Clustered copies CSV")
    p.add_argument("--reports-dir",  default="./reports")
    p.add_argument("--family",       default="TE")
    p.add_argument("--assembly",     default="hg38")
    p.add_argument("--overlaps-tsv", default=None,
                   help="Path to motif_analysis/all_overlaps.tsv (auto-detected if omitted)")
    p.add_argument("--consensus-fasta", default=None)
    p.add_argument("--max-mm", type=int, default=_DEFAULT_MAX_MM,
                   help="Hamming mismatches allowed when matching copy k-mer to canonical "
                        "(default: 2; higher = fewer gains called)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_motif_gain_analysis(
        input_csv       = args.input,
        reports_dir     = args.reports_dir,
        family_name     = args.family,
        assembly        = args.assembly,
        overlaps_tsv    = args.overlaps_tsv,
        max_mm          = args.max_mm,
        consensus_fasta = args.consensus_fasta,
    )
