#!/usr/bin/env python3
"""
te_go.py  —  GAMECA step G: Gene / GO Annotation
─────────────────────────────────────────────────────────────────────────────
Looks up gene function and GO terms for TF motifs identified as enriched by
te_motif.py, using the mygene.info REST API.

When enrichment results are absent, te_motif.py (JASPAR + Fisher) runs
automatically first.  Pass --homer to also run HOMER in the same chain.

Input:   clustered CSV from te_clustering.py  (--input)
      OR enrichment_results/ directory from te_motif.py (--enrichment-dir)

Output:
  <out_dir>/go_annotations/
    gene_functions.csv                   all gene-level GO data
    cluster_N_enrichment_annotated.csv   enrichment + GO columns
    strand_plots/
      clusterN_sense_go_bp.png           GO-BP bar chart (+ strand)
      clusterN_antisense_go_bp.png       GO-BP bar chart (– strand)

Usage:
    # Simplest — pass the clustered CSV; te_motif runs automatically if needed
    python te_go.py --input clustered.csv --build hg38

    # Family + assembly shortcut — locates the CSV in standard output paths
    python te_go.py --family AluY  --assembly hg38  --out-dir ./results
    python te_go.py --family B1_Mm --assembly mm10  --out-dir ./results --homer

    # Manual — point at existing enrichment results (legacy / direct)
    python te_go.py \\
        --enrichment-dir ./results/enrichment_results \\
        --clustered-csv  ./results/clustered.csv \\
        --build hg38 --out-dir ./results
"""

import argparse
import os
import re
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests

_DEFAULT_BASE = os.environ.get("TE_BASE_DIR",   str(Path.home() / "te_analysis"))
_DEFAULT_JASP = os.environ.get("TE_JASPAR_DIR", str(Path(_DEFAULT_BASE) / "jaspar"))

SPECIES_IDS = {"hg38": "9606", "hg19": "9606", "mm10": "10090", "mm39": "10090"}

PALETTE = [
    "#4C9BE8","#E8604C","#2ECC71","#9B59B6","#F39C12",
    "#1ABC9C","#E74C3C","#3498DB","#E67E22","#8E44AD",
]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="GAMECA step G: GO annotation for enriched TF motifs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Primary inputs (one or more required) ────────────────────────────────
    p.add_argument("--enrichment-dir", default=None,
                   help="Directory with cluster_N_enrichment.csv files (from te_motif.py). "
                        "Auto-resolved from --out-dir/enrichment_results if omitted; "
                        "te_motif.py is run first when results are absent.")
    p.add_argument("--input", default=None,
                   help="Clustered CSV (from te_clustering.py). Triggers automatic "
                        "te_motif.py run when enrichment results are absent.")
    p.add_argument("--family", default=None,
                   help="TE family name (e.g. AluY). Used to locate the clustered CSV "
                        "under --out-dir when --input is not specified.")

    # ── Genome / assembly ────────────────────────────────────────────────────
    p.add_argument("--build", "--assembly", default="hg38",
                   help="Genome build: hg38/hg19/mm10/mm39 (default: hg38). "
                        "--assembly is an accepted synonym.")

    # ── Output ───────────────────────────────────────────────────────────────
    p.add_argument("--out-dir", default=".",
                   help="Output directory (go_annotations/ created inside)")
    p.add_argument("--clustered-csv", default=None,
                   help="Clustered CSV for strand GO plots (inferred from --input when possible)")

    # ── GO options ───────────────────────────────────────────────────────────
    p.add_argument("--p-threshold", type=float, default=0.05,
                   help="Fisher p-value cutoff for GO query (default: 0.05)")
    p.add_argument("--top-motifs", type=int, default=30,
                   help="Number of top motifs per cluster to annotate (default: 30)")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if gene_functions.csv already exists")

    # ── JASPAR options (forwarded to te_motif when auto-chaining) ────────────
    p.add_argument("--jaspar-bed", default=None,
                   help="Path to JASPAR BED (auto-downloaded if omitted)")
    p.add_argument("--jaspar-dir", default=_DEFAULT_JASP,
                   help=f"Cache directory for JASPAR BED files (default: {_DEFAULT_JASP})")

    # ── HOMER (forwarded to te_motif when auto-chaining) ─────────────────────
    p.add_argument("--homer", action="store_true",
                   help="Also run HOMER findMotifsGenome.pl when te_motif is auto-chained")
    p.add_argument("--homer-genome", default=None,
                   help="HOMER genome name or FASTA path (defaults to --build value)")
    p.add_argument("--homer-size",    default="200",
                   help="HOMER -size parameter (default: 200)")
    p.add_argument("--homer-threads", type=int, default=4,
                   help="HOMER -p threads per cluster (default: 4)")
    p.add_argument("--notify-email", default="", metavar="EMAIL",
                   help="Send a completion email to this address (requires Gmail App Password setup).")

    args = p.parse_args()
    if not args.enrichment_dir and not args.input and not args.family:
        p.error(
            "provide at least one of: --enrichment-dir, --input (clustered CSV), "
            "or --family (to locate the clustered CSV automatically)"
        )
    return args


# ── mygene.info helpers ───────────────────────────────────────────────────────

def query_gene(name, species_id):
    clean  = re.sub(r"[^A-Za-z0-9]",  "", name)
    hyphen = re.sub(r"[^A-Za-z0-9-]", "", name)
    for q in [f"symbol:{hyphen}", f"symbol:{clean}", hyphen, clean]:
        try:
            r = requests.get(
                f"https://mygene.info/v3/query?q={q}&species={species_id}"
                f"&fields=symbol,name,summary,go&size=1",
                timeout=15,
            )
            if r.status_code == 200:
                hits = r.json().get("hits", [])
                if hits:
                    return hits[0]
        except Exception:
            pass
    return None


def go_terms(go_data, key, n=5):
    if not isinstance(go_data, dict):
        return ""
    t = go_data.get(key, [])
    return "; ".join(x.get("term", "") for x in t[:n]) if isinstance(t, list) else ""


# ── main ──────────────────────────────────────────────────────────────────────

def run_go_annotation(enrichment_dir, build, out_dir, clustered_csv=None,
                      p_threshold=0.05, top_motifs=30, force=False):
    """
    Annotate enriched TF motifs with GO terms via mygene.info.

    Parameters
    ----------
    enrichment_dir : path to directory with cluster_N_enrichment.csv files
    build          : genome build string
    out_dir        : root output directory
    clustered_csv  : path to clustered CSV for strand information (optional)
    p_threshold    : Fisher p-value threshold used when loading enrichment CSVs
    top_motifs     : max motifs per cluster to send to mygene.info
    force          : re-run even if gene_functions.csv already exists

    Returns
    -------
    dict with keys:
        gene_functions_path  – path to gene_functions.csv
        n_genes              – number of genes annotated
    """
    enrichment_dir = Path(enrichment_dir)
    out_dir        = Path(out_dir)
    go_dir         = out_dir / "go_annotations"
    strand_dir     = go_dir  / "strand_plots"
    go_dir.mkdir(parents=True, exist_ok=True)
    strand_dir.mkdir(parents=True, exist_ok=True)

    species_id = SPECIES_IDS.get(build, "9606")
    gf_path    = go_dir / "gene_functions.csv"

    # ── Collect significant motifs from enrichment CSVs ───────────────────────
    enrich_files = sorted(enrichment_dir.glob("cluster_*_enrichment.csv"))
    if not enrich_files:
        print(f"FATAL: No cluster_*_enrichment.csv files found in {enrichment_dir}")
        sys.exit(1)

    significant_tfs = {}
    all_sig_motifs  = set()
    cluster_ids     = []

    for fp in enrich_files:
        # Extract cluster id from filename
        try:
            cid = int(re.search(r"cluster_(-?\d+)_enrichment", fp.name).group(1))
        except Exception:
            continue
        rdf = pd.read_csv(fp)
        sig = rdf[rdf["P_Value"] < p_threshold]
        significant_tfs[cid] = sig
        cluster_ids.append(cid)
        all_sig_motifs.update(sig["Motif"].head(top_motifs).tolist())

    cluster_ids = sorted(cluster_ids)
    print(f"  {len(cluster_ids)} clusters, {len(all_sig_motifs)} unique sig motifs")

    # ── Load or build gene_functions ──────────────────────────────────────────
    if gf_path.exists() and not force:
        print(f"\n  [SKIP] gene_functions.csv exists — loading cached results")
        gf_df = pd.read_csv(gf_path)
    else:
        print(f"\n  Querying mygene.info for {len(all_sig_motifs)} motifs (species={species_id})...")
        gene_functions = {}
        for i, motif in enumerate(sorted(all_sig_motifs), 1):
            gn  = motif.split("::")[0] if "::" in motif else motif
            key = re.sub(r"[^A-Za-z0-9-]", "", gn)
            if not key or key in gene_functions:
                continue
            info = query_gene(gn, species_id)
            if info:
                gene_functions[key] = info
            if i % 20 == 0:
                print(f"    {i}/{len(all_sig_motifs)}, {len(gene_functions)} hits so far")
            time.sleep(0.1)

        rows = []
        for gn, info in gene_functions.items():
            gd = info.get("go", {})
            rows.append({
                "Gene":    gn,
                "Symbol":  info.get("symbol", ""),
                "Name":    info.get("name", ""),
                "Summary": info.get("summary", ""),
                "GO_BP":   go_terms(gd, "BP"),
                "GO_MF":   go_terms(gd, "MF"),
                "GO_CC":   go_terms(gd, "CC"),
            })
        gf_df = pd.DataFrame(rows)
        gf_df.to_csv(gf_path, index=False)
        print(f"  {len(gf_df)} genes saved → {gf_path.name}")

    # Build lookup index
    gf_idx = gf_df.set_index("Gene") if len(gf_df) > 0 else pd.DataFrame()

    def _lookup(motif, col):
        g = re.sub(r"[^A-Za-z0-9-]", "",
                   motif.split("::")[0] if "::" in motif else motif)
        if len(gf_idx) and g in gf_idx.index:
            return str(gf_idx.loc[g].get(col, ""))
        return ""

    # ── Annotate enrichment CSVs ──────────────────────────────────────────────
    print("\n  Annotating enrichment CSVs...")
    annotated_frames = []
    for cid in cluster_ids:
        sig = significant_tfs.get(cid, pd.DataFrame())
        if not len(sig):
            continue
        sig = sig.copy()
        sig["Cluster"] = cid
        sig["Gene_Name"] = sig["Motif"].apply(lambda m: _lookup(m, "Name"))
        sig["Summary"]   = sig["Motif"].apply(lambda m: (_lookup(m, "Summary") or "")[:200])
        sig["GO_BP"]     = sig["Motif"].apply(lambda m: _lookup(m, "GO_BP"))
        out_ann = enrichment_dir / f"cluster_{cid}_enrichment_annotated.csv"
        sig.to_csv(out_ann, index=False)
        annotated_frames.append(sig)
        print(f"  Cluster {cid}: {len(sig)} annotated → {out_ann.name}")

    _general_go_plots(gf_df, annotated_frames, go_dir)

    # ── Strand GO plots (requires clustered CSV + overlap data) ──────────────
    if clustered_csv:
        _strand_go_plots(clustered_csv, out_dir, cluster_ids,
                         gf_idx, go_dir, strand_dir)

    print(f"\n  GO annotation complete → {go_dir}")
    return {"gene_functions_path": str(gf_path), "n_genes": len(gf_df)}


def _split_go_terms(value):
    terms = []
    for item in str(value or "").split(";"):
        item = item.strip()
        if item:
            terms.append(item)
    return terms


def _general_go_plots(gf_df, annotated_frames, go_dir):
    """Generate non-strand GO summary plots so GO has visible outputs."""
    go_dir = Path(go_dir)
    summary_rows = []

    if len(gf_df) and "GO_BP" in gf_df.columns:
        for terms in gf_df["GO_BP"].dropna():
            for term in _split_go_terms(terms):
                summary_rows.append({"Source": "all_genes", "GO_Term": term})

    for frame in annotated_frames:
        if "GO_BP" not in frame.columns or "Cluster" not in frame.columns:
            continue
        for _, row in frame.iterrows():
            for term in _split_go_terms(row.get("GO_BP", "")):
                summary_rows.append({"Source": f"cluster_{row['Cluster']}", "GO_Term": term})

    if not summary_rows:
        (go_dir / "go_plot_status.txt").write_text(
            "No GO Biological Process terms were available for plotting. "
            "This can happen when no motifs pass the p-value cutoff or mygene.info "
            "returns no GO annotations.\n"
        )
        return

    counts = pd.DataFrame(summary_rows)
    overall = counts["GO_Term"].value_counts().head(20).reset_index()
    overall.columns = ["GO_Term", "Count"]
    overall.to_csv(go_dir / "go_bp_top_terms.csv", index=False)

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(overall))))
        ax.barh(overall["GO_Term"][::-1], overall["Count"][::-1], color="#4C9BE8")
        ax.set_xlabel("Count")
        ax.set_title("Top GO Biological Process Terms", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(go_dir / "go_bp_top_terms.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved go_bp_top_terms.png")
    except Exception as e:
        print(f"  [WARN] GO BP plot failed: {e}")


def _strand_go_plots(clustered_csv, out_dir, cluster_ids,
                     gf_idx, go_dir, strand_dir):
    """Generate per-cluster, per-strand GO BP bar charts."""
    overlaps_path = Path(out_dir) / "motif_analysis" / "all_overlaps.tsv"
    if not overlaps_path.exists():
        print("  [SKIP] Strand GO plots: all_overlaps.tsv not found (run te_motif.py first)")
        return

    try:
        ov = pd.read_csv(overlaps_path, sep="\t")
    except Exception as e:
        print(f"  [WARN] Strand GO plots: could not load overlaps: {e}")
        return

    # Normalise cluster col name
    for col in ("Cluster", "cluster"):
        if col in ov.columns:
            ov["_cluster"] = ov[col].astype(int)
            break
    if "_cluster" not in ov.columns:
        print("  [SKIP] Strand GO plots: no cluster column in overlaps")
        return

    if "strand" not in ov.columns:
        print("  [SKIP] Strand GO plots: no strand column in overlaps")
        return

    def _lookup(motif, col):
        g = re.sub(r"[^A-Za-z0-9-]", "",
                   motif.split("::")[0] if "::" in motif else motif)
        if len(gf_idx) and g in gf_idx.index:
            return str(gf_idx.loc[g].get(col, ""))
        return ""

    for i, cid in enumerate(cluster_ids):
        color = PALETTE[i % len(PALETTE)]
        for strand, label in [("+", "sense"), ("-", "antisense")]:
            sub = ov[(ov["_cluster"] == cid) & (ov["strand"] == strand)]
            if not len(sub):
                continue
            counts = sub["Motif_name"].value_counts().reset_index()
            counts.columns = ["Motif", "Count"]
            counts["Gene_Name"] = counts["Motif"].apply(lambda m: _lookup(m, "Name"))
            counts["GO_BP"]     = counts["Motif"].apply(lambda m: _lookup(m, "GO_BP"))
            counts.to_csv(go_dir / f"cluster{cid}_{label}_go_annotated.csv", index=False)

            bp_terms = []
            for b in counts["GO_BP"].dropna():
                bp_terms += [t.strip() for t in b.split(";") if t.strip()]
            if not bp_terms:
                continue

            tc = pd.DataFrame(Counter(bp_terms).most_common(15), columns=["GO_Term", "Count"])
            fig, ax = plt.subplots(figsize=(11, max(5, len(tc) * 0.5)))
            ax.barh(tc["GO_Term"], tc["Count"], color=color, alpha=0.85)
            ax.invert_yaxis()
            ax.set_xlabel("Frequency")
            strand_str = "Sense (+)" if strand == "+" else "Antisense (-)"
            ax.set_title(f"Cluster {cid} — {strand_str}\nGO Biological Process",
                         fontweight="bold")
            ax.spines[["top","right"]].set_visible(False)
            ax.xaxis.grid(True, linestyle="--", alpha=0.4)
            ax.set_axisbelow(True)
            plt.tight_layout()
            fname = strand_dir / f"cluster{cid}_{label}_go_bp.png"
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved {fname.name}")


def main():
    args    = parse_args()
    build   = args.build
    out_dir = Path(args.out_dir)

    # ── Resolve clustered CSV ─────────────────────────────────────────────────
    input_csv = args.input
    if not input_csv and args.family:
        family_lc = args.family.lower()
        candidates = [
            out_dir / f"{family_lc}_clustered.csv",
            out_dir / "01_data" / f"{family_lc}_clustered.csv",
            out_dir / family_lc / "01_data" / f"{family_lc}_clustered.csv",
        ]
        for c in candidates:
            if c.exists():
                input_csv = str(c)
                print(f"  Found clustered CSV: {c}")
                break
        if not input_csv:
            print(
                f"  [INFO] Could not find {family_lc}_clustered.csv under {out_dir}.\n"
                f"  Run te_clustering.py first, or pass --input explicitly."
            )

    # ── Resolve enrichment directory ──────────────────────────────────────────
    enrich_dir = Path(args.enrichment_dir) if args.enrichment_dir else (out_dir / "enrichment_results")

    # ── Auto-chain te_motif.py when enrichment results are absent ─────────────
    needs_motif = (
        not enrich_dir.exists()
        or not any(enrich_dir.glob("cluster_*_enrichment.csv"))
    )
    if needs_motif:
        if not input_csv:
            print(
                "FATAL: No enrichment results found in:\n"
                f"  {enrich_dir}\n"
                "Provide --input (clustered CSV), --enrichment-dir, or --family to\n"
                "locate the clustered CSV automatically under --out-dir."
            )
            sys.exit(1)
        print(f"\n[AUTO-CHAIN] Enrichment results not found — running te_motif.py first")
        try:
            from te_motif import run_motif_analysis, run_homer
        except ImportError:
            print("FATAL: te_motif.py not found on PYTHONPATH; cannot auto-chain.")
            sys.exit(1)
        run_motif_analysis(
            input_csv      = input_csv,
            build          = build,
            out_dir        = str(out_dir),
            jaspar_bed_arg = args.jaspar_bed,
            jaspar_dir     = args.jaspar_dir,
            p_threshold    = args.p_threshold,
            force          = args.force,
        )
        if args.homer:
            print("\n[AUTO-CHAIN] Running HOMER (--homer)")
            run_homer(
                input_csv = input_csv,
                build     = build,
                out_dir   = str(out_dir),
                genome    = args.homer_genome,
                size      = args.homer_size,
                threads   = args.homer_threads,
                force     = args.force,
            )

    # ── GO annotation ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("GAMECA — GO Annotation")
    print(f"  Enrichment dir: {enrich_dir}")
    print(f"  Build:          {build}")
    print(f"  OutDir:         {out_dir}")
    print("=" * 60)
    run_go_annotation(
        enrichment_dir = str(enrich_dir),
        build          = build,
        out_dir        = str(out_dir),
        clustered_csv  = input_csv or args.clustered_csv,
        p_threshold    = args.p_threshold,
        top_motifs     = args.top_motifs,
        force          = args.force,
    )
    if args.notify_email:
        from te_notify import send_completion_email
        send_completion_email(args.notify_email, "te_go", str(out_dir))


if __name__ == "__main__":
    main()
