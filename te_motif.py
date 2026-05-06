#!/usr/bin/env python3
"""
te_motif.py  —  GAMECA step M: Motif Analysis
─────────────────────────────────────────────────────────────────────────────
Overlaps TE loci against JASPAR TFBS predictions (via bedtools intersect)
then runs a per-cluster Fisher's exact test to find enriched TF motifs.

Input:   clustered CSV produced by te_clustering.py
         (needs columns: chr/Chromosome, start/Start, stop/Stop/End, Cluster)

Output:
  <out_dir>/motif_analysis/
    all_overlaps.tsv              raw bedtools overlap output
    overall_motif_counts.csv      motif frequency across all loci
    overall_top_motifs.png        top-20 bar chart
  <out_dir>/enrichment_results/
    cluster_N_enrichment.csv      Fisher p-values per cluster
    enrichment_heatmap.png        -log10(p) heatmap across clusters

JASPAR BED resolution order
  1. --jaspar-bed FILE
  2. TE_JASPAR_<BUILD> environment variable
  3. <jaspar-dir>/JASPAR2024_<build>.sorted.bed.gz  (cached)
  4. Auto-download from jaspar.elixir.no

Usage:
    python te_motif.py \\
        --input ./results/clustered.csv \\
        --build hg38 \\
        --out-dir ./results

    python te_motif.py --input data.csv --build mm10 \\
        --jaspar-bed /path/JASPAR2024_mm10.bed.gz
"""

import argparse
import gzip
import os
import sys
import time
import tempfile
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ── JASPAR download URLs ──────────────────────────────────────────────────────

JASPAR_URLS = {
    "hg38": "https://jaspar.elixir.no/static/data/beds/JASPAR2024_hg38.bed.gz",
    "hg19": "https://jaspar.elixir.no/static/data/beds/JASPAR2024_hg19.bed.gz",
    "mm10": "https://jaspar.elixir.no/static/data/beds/JASPAR2024_mm10.bed.gz",
    "mm39": "https://jaspar.elixir.no/static/data/beds/JASPAR2024_mm39.bed.gz",
}

_DEFAULT_BASE = os.environ.get("TE_BASE_DIR",   str(Path.home() / "te_analysis"))
_DEFAULT_JASP = os.environ.get("TE_JASPAR_DIR", str(Path(_DEFAULT_BASE) / "jaspar"))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="GAMECA step M: JASPAR motif overlap + Fisher enrichment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input",      required=True, help="Clustered CSV (from te_clustering.py)")
    p.add_argument("--build",      default="hg38", help="Genome build: hg38/hg19/mm10/mm39")
    p.add_argument("--out-dir",    default=".", help="Output directory")
    p.add_argument("--jaspar-bed", default=None,
                   help="Path to JASPAR BED (auto-downloaded if omitted)")
    p.add_argument("--jaspar-dir", default=_DEFAULT_JASP,
                   help=f"Cache directory for JASPAR BED files (default: {_DEFAULT_JASP})")
    p.add_argument("--p-threshold", type=float, default=0.05,
                   help="Fisher p-value significance threshold (default: 0.05)")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if overlap file already exists")
    p.add_argument("--homer", action="store_true",
                   help="Also run HOMER findMotifsGenome.pl on each cluster")
    p.add_argument("--homer-genome", default=None,
                   help="HOMER genome name or FASTA path (e.g. hg38, mm10); "
                        "defaults to --build value")
    p.add_argument("--homer-size", default="200",
                   help="HOMER -size parameter (default: 200)")
    p.add_argument("--homer-threads", type=int, default=4,
                   help="HOMER -p threads per cluster (default: 4)")
    return p.parse_args()


# ── JASPAR BED helpers ────────────────────────────────────────────────────────

def resolve_jaspar_bed(build, jaspar_bed_arg, jaspar_dir):
    """Return path to a valid JASPAR BED, downloading if necessary."""
    jaspar_dir = Path(jaspar_dir)
    jaspar_dir.mkdir(parents=True, exist_ok=True)

    if jaspar_bed_arg and Path(jaspar_bed_arg).exists():
        return jaspar_bed_arg

    env_path = os.environ.get(f"TE_JASPAR_{build.upper()}", "")
    if env_path and Path(env_path).exists():
        print(f"  Using JASPAR BED from env: {env_path}")
        return env_path

    local_path = jaspar_dir / f"JASPAR2024_{build}.sorted.bed.gz"
    if local_path.exists():
        print(f"  Using cached JASPAR BED: {local_path}")
        return str(local_path)

    url = JASPAR_URLS.get(build)
    if not url:
        print(f"\nFATAL: No JASPAR URL for build '{build}'.")
        print(f"  Provide --jaspar-bed manually or set TE_JASPAR_{build.upper()}.")
        sys.exit(1)

    print(f"\n  JASPAR BED not found — downloading from JASPAR portal...")
    print(f"  URL: {url}")
    print(f"  Destination: {local_path}")
    print(f"  (hg38 ~1–2 GB; this may take several minutes)")
    print(f"  Tip: set TE_JASPAR_{build.upper()}=/path/to/file to skip future downloads\n")

    import subprocess
    for tool in [
        ["wget", "-q", "--show-progress", "-O", str(local_path), url],
        ["curl", "-L", "-o", str(local_path), url],
    ]:
        if subprocess.run(tool).returncode == 0:
            print(f"  Downloaded {local_path.stat().st_size / 1e6:.0f} MB → {local_path}")
            return str(local_path)

    print("  FATAL: JASPAR BED download failed.")
    print(f"  Manual: wget -O {local_path} '{url}'")
    sys.exit(1)


def validate_jaspar_bed(bed_path, sample=500):
    opener = gzip.open if str(bed_path).endswith(".gz") else open
    col_counts = []
    try:
        with opener(str(bed_path), "rt") as fh:
            for i, line in enumerate(fh):
                if i >= sample:
                    break
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("track"):
                    continue
                col_counts.append(len(line.split("\t")))
    except Exception as e:
        print(f"  WARN: Cannot validate BED: {e}")
        return
    if col_counts:
        modal = Counter(col_counts).most_common(1)[0][0]
        print(f"  JASPAR BED validated: modal_cols={modal}, sampled {len(col_counts)} lines")


def _normalise_bed(src, dst, n_cols=6):
    opener_r = gzip.open if str(src).endswith(".gz") else open
    opener_w = gzip.open if str(dst).endswith(".gz") else open
    n = 0
    with opener_r(str(src), "rt") as fin, opener_w(str(dst), "wt") as fout:
        for line in fin:
            s = line.rstrip("\n")
            if not s or s.startswith("#") or s.startswith("track"):
                fout.write(s + "\n"); continue
            parts = s.split("\t")
            if len(parts) >= n_cols:
                fout.write("\t".join(parts[:n_cols]) + "\n")
                n += 1
    print(f"  Normalised BED → {dst} ({n:,} lines)")


def bedtools_intersect_safe(v_bed, jaspar_bed):
    try:
        import pybedtools
    except ImportError:
        print("FATAL: pybedtools not installed.  conda install -c bioconda pybedtools")
        sys.exit(1)

    v_bt = pybedtools.BedTool(str(v_bed))
    m_bt = pybedtools.BedTool(str(jaspar_bed))
    print("  Running bedtools intersect...")
    try:
        overlaps = v_bt.intersect(m_bt, wa=True, wb=True)
        print(f"  {len(overlaps):,} overlaps")
        return overlaps
    except Exception as e:
        if "fields" in str(e).lower():
            print("  Column mismatch — normalising JASPAR BED to 6 cols...")
            norm = str(jaspar_bed).replace(".bed.gz", ".norm6.bed.gz").replace(".bed", ".norm6.bed")
            _normalise_bed(str(jaspar_bed), norm, 6)
            overlaps = v_bt.intersect(pybedtools.BedTool(norm), wa=True, wb=True)
            print(f"  Retry: {len(overlaps):,} overlaps")
            return overlaps
        raise


# ── coordinate column detection ───────────────────────────────────────────────

def _detect_coords(df):
    def _fc(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None
    return (
        _fc(["chr", "Chromosome", "chrom", "Chr", "#chrom"]),
        _fc(["start", "Start", "chromStart"]),
        _fc(["stop", "Stop", "End", "end", "chromEnd"]),
    )


def _cluster_col(df):
    for c in ("Cluster", "cluster"):
        if c in df.columns:
            return c
    return None


# ── main ──────────────────────────────────────────────────────────────────────

def run_motif_analysis(input_csv, build, out_dir, jaspar_bed_arg,
                       jaspar_dir, p_threshold=0.05, force=False):
    """
    Run JASPAR motif overlap and Fisher enrichment.

    Parameters
    ----------
    input_csv    : path to clustered CSV (output of te_clustering.py)
    build        : genome build string, e.g. "hg38"
    out_dir      : root output directory
    jaspar_bed_arg : explicit JASPAR BED path or None
    jaspar_dir   : directory for caching downloaded JASPAR files
    p_threshold  : Fisher p-value cutoff for reporting significance
    force        : re-run even if outputs already exist

    Returns
    -------
    dict with keys:
        overlaps_path      – path to all_overlaps.tsv
        enrichment_dir     – path to directory with per-cluster CSVs
        significant_tfs    – {cluster_id: DataFrame of significant TFs}
    """
    out_dir = Path(out_dir)
    motif_dir   = out_dir / "motif_analysis"
    enrich_dir  = out_dir / "enrichment_results"
    motif_dir.mkdir(parents=True, exist_ok=True)
    enrich_dir.mkdir(parents=True, exist_ok=True)

    overlaps_path = motif_dir / "all_overlaps.tsv"

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n  Loading {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"  {len(df):,} rows")

    chr_col, start_col, stop_col = _detect_coords(df)
    if not all([chr_col, start_col, stop_col]):
        print("FATAL: Cannot find coordinate columns (chr/start/stop).")
        sys.exit(1)

    cl_col = _cluster_col(df)
    if cl_col is None:
        print("FATAL: No Cluster column found. Run te_clustering.py first.")
        sys.exit(1)

    df[cl_col] = df[cl_col].astype(int)
    cluster_ids = sorted([c for c in df[cl_col].unique() if c >= 0])
    has_strand  = "strand" in df.columns

    print(f"  {len(cluster_ids)} clusters: {cluster_ids}")

    # ── Resolve JASPAR BED ────────────────────────────────────────────────────
    jaspar_bed = resolve_jaspar_bed(build, jaspar_bed_arg, jaspar_dir)
    validate_jaspar_bed(jaspar_bed)

    # ── Bedtools intersect ────────────────────────────────────────────────────
    if overlaps_path.exists() and not force:
        print(f"\n  [SKIP] Overlaps file exists: {overlaps_path}")
        df_ov = pd.read_csv(overlaps_path, sep="\t")
    else:
        t0 = time.time()
        scratch = os.environ.get("TMPDIR", str(out_dir / "tmp"))
        Path(scratch).mkdir(parents=True, exist_ok=True)
        try:
            import pybedtools
            pybedtools.set_tempdir(scratch)
        except ImportError:
            pass
        tempfile.tempdir = scratch

        v_bed = motif_dir / "te_loci.bed"
        df[[chr_col, start_col, stop_col]].to_csv(v_bed, sep="\t", header=False, index=False)

        overlaps = bedtools_intersect_safe(str(v_bed), jaspar_bed)
        if len(overlaps) == 0:
            print("FATAL: Zero overlaps. Check genome build and JASPAR BED.")
            sys.exit(1)

        # Parse column layout from first line
        first = str(overlaps[0]).strip().split("\t")
        n_mc  = len(first) - 3
        cols_v = [chr_col, start_col, stop_col]
        if   n_mc == 4: cols_m = ["Motif_chr","Motif_start","Motif_end","Motif_name"]
        elif n_mc == 5: cols_m = ["Motif_chr","Motif_start","Motif_end","Motif_name","Motif_score"]
        elif n_mc == 6: cols_m = ["Motif_chr","Motif_start","Motif_end","Motif_name","Motif_score","Motif_strand"]
        elif n_mc == 7: cols_m = ["Motif_chr","Motif_start","Motif_end","Motif_ID","Motif_score","Motif_strand","Motif_name"]
        else:           cols_m = [f"motif_col_{i}" for i in range(n_mc)]

        df_ov = overlaps.to_dataframe(names=cols_v + cols_m, header=None)
        if "Motif_name" not in df_ov.columns:
            nc = [c for c in df_ov.columns if "name" in c.lower() or "id" in c.lower()]
            df_ov["Motif_name"] = df_ov[nc[0]] if nc else df_ov.iloc[:, 3]

        # Merge cluster info
        mcols = [chr_col, start_col, stop_col, cl_col] + (["strand"] if has_strand else [])
        w = df[mcols].copy()
        for c in [start_col, stop_col]:
            df_ov[c] = pd.to_numeric(df_ov[c], errors="coerce")
            w[c]     = pd.to_numeric(w[c], errors="coerce")
        df_ov[chr_col] = df_ov[chr_col].astype(str)
        w[chr_col]     = w[chr_col].astype(str)
        df_ov = df_ov.merge(w, on=[chr_col, start_col, stop_col], how="left",
                            suffixes=("", "_new"))
        for col in [cl_col, "strand"]:
            if f"{col}_new" in df_ov.columns:
                df_ov[col] = df_ov[f"{col}_new"].combine_first(
                    df_ov.get(col, pd.Series(dtype=object)))
        df_ov.drop(columns=[c for c in df_ov.columns if c.endswith("_new")], inplace=True)

        df_ov.to_csv(overlaps_path, sep="\t", index=False)
        print(f"  Saved {overlaps_path.name} ({len(df_ov):,} rows) [{time.time()-t0:.1f}s]")

        # Overall motif counts
        overall = df_ov["Motif_name"].value_counts()
        overall.to_csv(motif_dir / "overall_motif_counts.csv")
        print(f"  {len(overall)} unique motifs. Top 5:\n{overall.head(5)}")

        # Top-20 bar chart
        try:
            tc = overall.head(20).reset_index()
            tc.columns = ["Motif", "Count"]
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(range(len(tc)), tc["Count"], color="#3498DB", alpha=0.85)
            ax.set_xticks(range(len(tc)))
            ax.set_xticklabels(tc["Motif"], rotation=45, ha="right", fontsize=9)
            ax.set_ylabel("Count")
            ax.set_title(f"Top 20 Motifs ({build})", fontweight="bold")
            ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            plt.savefig(motif_dir / "overall_top_motifs.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("  Saved overall_top_motifs.png")
        except Exception as e:
            print(f"  [WARN] bar chart: {e}")

    # ── Fisher's exact test ───────────────────────────────────────────────────
    print(f"\n  Running Fisher's exact test ({len(cluster_ids)} clusters)...")
    t0 = time.time()
    all_motifs = df_ov["Motif_name"].unique()
    total_n    = len(df)
    significant_tfs = {}

    for cid in cluster_ids:
        cn = int((df[cl_col] == cid).sum())
        results = []
        for motif in all_motifs:
            mi   = len(df_ov[(df_ov[cl_col]==cid) & (df_ov["Motif_name"]==motif)]
                        [[chr_col, start_col, stop_col]].drop_duplicates())
            mt   = len(df_ov[df_ov["Motif_name"]==motif]
                        [[chr_col, start_col, stop_col]].drop_duplicates())
            nmc  = cn - mi
            mnc  = mt - mi
            nmnc = max((total_n - cn) - mnc, 0)
            try:
                odds, pv = stats.fisher_exact([[mi, mnc],[nmc, nmnc]], alternative="greater")
            except Exception:
                odds, pv = 1.0, 1.0
            results.append({
                "Motif": motif, "In_Cluster": mi, "Total": mt,
                "Cluster_Size": cn, "Odds_Ratio": round(odds, 4), "P_Value": pv,
            })

        rdf = pd.DataFrame(results).sort_values("P_Value")
        out_csv = enrich_dir / f"cluster_{cid}_enrichment.csv"
        rdf.to_csv(out_csv, index=False)
        sig = rdf[rdf["P_Value"] < p_threshold]
        significant_tfs[cid] = sig
        print(f"  Cluster {cid} (n={cn:,}): {len(sig)} significant TFs → {out_csv.name}")

    # Enrichment heatmap
    try:
        all_sig = []
        for cid, sdf in significant_tfs.items():
            if len(sdf):
                t = sdf.head(10).copy(); t["cluster"] = cid; all_sig.append(t)
        if all_sig:
            comb = pd.concat(all_sig)
            comb["nlp"] = -np.log10(comb["P_Value"].clip(lower=1e-300))
            piv = comb.pivot_table(index="Motif", columns="cluster",
                                   values="nlp", aggfunc="max").fillna(0)
            top_m = piv.max(axis=1).sort_values(ascending=False).head(30).index
            piv   = piv.loc[top_m]
            fig, ax = plt.subplots(figsize=(8, max(6, len(top_m) * 0.4)))
            im = ax.imshow(piv.values, aspect="auto", cmap="Reds")
            plt.colorbar(im, ax=ax, label="-log10(p)")
            ax.set_xticks(range(len(piv.columns)))
            ax.set_xticklabels([f"Cl {c}" for c in piv.columns])
            ax.set_yticks(range(len(top_m))); ax.set_yticklabels(top_m, fontsize=8)
            ax.set_title("Motif Enrichment per Cluster", fontweight="bold")
            plt.tight_layout()
            plt.savefig(enrich_dir / "enrichment_heatmap.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("  Saved enrichment_heatmap.png")
    except Exception as e:
        print(f"  [WARN] heatmap: {e}")

    print(f"  Fisher step [{time.time()-t0:.1f}s]")
    total_sig = sum(len(s) for s in significant_tfs.values())
    print(f"  Total significant TF hits (p<{p_threshold}): {total_sig}")

    return {
        "overlaps_path":   str(overlaps_path),
        "enrichment_dir":  str(enrich_dir),
        "significant_tfs": significant_tfs,
    }


def run_homer(input_csv, build, out_dir, genome=None, size="200",
              threads=4, force=False):
    """
    Run HOMER findMotifsGenome.pl on each cluster's loci.

    Parameters
    ----------
    input_csv : path to clustered CSV (output of te_clustering.py)
    build     : genome build string used as fallback genome name
    out_dir   : root output directory; results go in <out_dir>/homer_results/
    genome    : HOMER genome name or FASTA path (defaults to build)
    size      : HOMER -size value, e.g. "200" or "given"
    threads   : HOMER -p (parallel threads per cluster run)
    force     : re-run even if per-cluster output already exists

    Returns
    -------
    dict mapping cluster_id -> path to knownResults.txt (or None if failed)
    """
    import shutil, subprocess

    if not shutil.which("findMotifsGenome.pl"):
        print("FATAL: HOMER not found. Install via http://homer.ucsd.edu/homer/introduction/install.html")
        sys.exit(1)

    genome = genome or build
    out_dir = Path(out_dir)
    homer_root = out_dir / "homer_results"
    homer_root.mkdir(parents=True, exist_ok=True)

    print(f"\n  Loading {input_csv} for HOMER...")
    df = pd.read_csv(input_csv)
    chr_col, start_col, stop_col = _detect_coords(df)
    if not all([chr_col, start_col, stop_col]):
        print("FATAL: Cannot find coordinate columns.")
        sys.exit(1)
    cl_col = _cluster_col(df)
    if cl_col is None:
        print("FATAL: No Cluster column. Run te_clustering.py first.")
        sys.exit(1)

    df[cl_col] = df[cl_col].astype(int)
    cluster_ids = sorted([c for c in df[cl_col].unique() if c >= 0])
    print(f"  {len(cluster_ids)} clusters: {cluster_ids}")

    results = {}
    for cid in cluster_ids:
        cluster_dir = homer_root / f"cluster_{cid}"
        known_txt   = cluster_dir / "knownResults.txt"

        if known_txt.exists() and not force:
            print(f"  [SKIP] Cluster {cid}: {known_txt}")
            results[cid] = str(known_txt)
            continue

        cluster_dir.mkdir(parents=True, exist_ok=True)

        # Write BED for this cluster (HOMER wants: name chr start end strand)
        sub = df[df[cl_col] == cid][[chr_col, start_col, stop_col]].copy()
        sub.insert(0, "name", [f"locus_{i}" for i in range(len(sub))])
        sub["strand"] = "+"
        bed_path = cluster_dir / f"cluster_{cid}.bed"
        sub[["name", chr_col, start_col, stop_col, "strand"]].to_csv(
            bed_path, sep="\t", header=False, index=False
        )

        cmd = [
            "findMotifsGenome.pl",
            str(bed_path),
            genome,
            str(cluster_dir),
            "-size", str(size),
            "-p", str(threads),
            "-nomotif",       # skip de-novo to keep runtime reasonable
        ]
        print(f"  Cluster {cid} (n={len(sub):,}): running HOMER...")
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - t0

        if proc.returncode != 0:
            print(f"  [WARN] Cluster {cid} HOMER failed ({elapsed:.1f}s):\n{proc.stderr[-500:]}")
            results[cid] = None
            continue

        if not known_txt.exists():
            print(f"  [WARN] Cluster {cid}: knownResults.txt not produced")
            results[cid] = None
            continue

        # Parse and summarise top 10 known motifs
        try:
            kr = pd.read_csv(known_txt, sep="\t")
            kr.columns = [c.strip() for c in kr.columns]
            p_col = next((c for c in kr.columns if "p-value" in c.lower() or "pvalue" in c.lower()), None)
            name_col = kr.columns[0]
            if p_col:
                kr = kr.sort_values(p_col)
                top = kr[[name_col, p_col]].head(10).to_string(index=False)
            else:
                top = kr.head(10).to_string(index=False)
            print(f"  Cluster {cid}: top known motifs [{elapsed:.1f}s]\n{top}")
        except Exception as e:
            print(f"  Cluster {cid}: HOMER done [{elapsed:.1f}s] (parse warning: {e})")

        # Save a summary CSV alongside HOMER output
        try:
            kr.to_csv(cluster_dir / "knownResults_summary.csv", index=False)
        except Exception:
            pass

        results[cid] = str(known_txt)

    n_ok = sum(1 for v in results.values() if v)
    print(f"\n  HOMER complete: {n_ok}/{len(cluster_ids)} clusters succeeded → {homer_root}")
    return results


def main():
    args = parse_args()
    print("=" * 60)
    print("GAMECA — Motif Analysis")
    print(f"  Input:  {args.input}")
    print(f"  Build:  {args.build}")
    print(f"  OutDir: {args.out_dir}")
    print("=" * 60)
    run_motif_analysis(
        input_csv      = args.input,
        build          = args.build,
        out_dir        = args.out_dir,
        jaspar_bed_arg = args.jaspar_bed,
        jaspar_dir     = args.jaspar_dir,
        p_threshold    = args.p_threshold,
        force          = args.force,
    )
    if args.homer:
        run_homer(
            input_csv = args.input,
            build     = args.build,
            out_dir   = args.out_dir,
            genome    = args.homer_genome,
            size      = args.homer_size,
            threads   = args.homer_threads,
            force     = args.force,
        )


if __name__ == "__main__":
    main()
