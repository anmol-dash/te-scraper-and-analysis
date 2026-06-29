#!/usr/bin/env python3
"""
te_alignment.py
Multiple sequence alignment (MAFFT) + CIAlign visualization + consensus generation.

Can be imported by query.py or called standalone:
    python te_alignment.py --input clustered_data.csv --family HERVK9
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _pp(msg):
    import datetime
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ── FASTA helpers ───────────────────────────────────────────────────────────

def write_fasta(df, fasta_path, family_name=""):
    """Write df['Seq'] to a FASTA file."""
    fasta_path = Path(fasta_path)
    with open(fasta_path, "w") as fh:
        for rid, row in df.iterrows():
            te_name = str(row.get("TE_name", ""))[:50] or str(row.get("repName", ""))[:50]
            cluster = row.get("Cluster", "")
            header = f">row{rid}|{te_name}|cluster{cluster}"
            seq = str(row["Seq"]).strip().upper()
            fh.write(header + "\n")
            for i in range(0, len(seq), 80):
                fh.write(seq[i:i + 80] + "\n")
    return fasta_path


def check_mafft():
    """Return True if MAFFT is in PATH."""
    try:
        subprocess.run(["mafft", "--version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def check_cialign():
    """Return True if CIAlign is in PATH."""
    try:
        subprocess.run(["CIAlign", "--version"], capture_output=True)
        return True
    except FileNotFoundError:
        return False


def _count_fasta_seqs(fasta_path):
    """Count '>' header lines in a FASTA file."""
    n = 0
    try:
        with open(fasta_path) as f:
            for line in f:
                if line.startswith('>'):
                    n += 1
    except Exception:
        pass
    return n


def run_mafft(input_fasta, output_fasta, threads=-1, timeout=14400):
    """Run MAFFT alignment. Skips if output already exists and is non-empty.
    Selects --parttree / --retree / --auto based on sequence count.
    Returns True on success."""
    input_fasta = Path(input_fasta)
    output_fasta = Path(output_fasta)

    # Skip if output already exists with content
    if output_fasta.exists() and output_fasta.stat().st_size > 0:
        _pp(f"  MAFFT: {output_fasta.name} already exists — skipping")
        return True

    n_seqs = _count_fasta_seqs(input_fasta)
    if n_seqs > 50000:
        mode = "--parttree"
        _pp(f"  MAFFT parttree mode ({n_seqs:,} seqs, timeout={timeout}s)")
    elif n_seqs > 10000:
        mode = "--retree 2 --maxiterate 0"
        _pp(f"  MAFFT fast mode ({n_seqs:,} seqs, timeout={timeout}s)")
    else:
        mode = "--auto"
        _pp(f"  MAFFT auto mode ({n_seqs:,} seqs, timeout={timeout}s)")

    cmd = f"mafft {mode} --thread {threads} {input_fasta} > {output_fasta}"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                timeout=timeout)
    except subprocess.TimeoutExpired:
        _pp(f"  MAFFT TIMEOUT after {timeout}s for {input_fasta.name} — skipping alignment")
        if output_fasta.exists():
            output_fasta.unlink()
        return False
    if result.returncode != 0:
        _pp(f"  MAFFT failed: {result.stderr[:300]}")
        return False
    return True


# ── consensus generation ────────────────────────────────────────────────────

def consensus_from_alignment(aligned_fasta):
    """Read aligned FASTA and return majority-vote consensus string."""
    from Bio import AlignIO
    aln = AlignIO.read(str(aligned_fasta), "fasta")
    aln_len = aln.get_alignment_length()
    chars = []
    for i in range(aln_len):
        col = aln[:, i]
        counts = {}
        for ch in col.upper():
            if ch != "-":
                counts[ch] = counts.get(ch, 0) + 1
        total = sum(counts.values())
        if total == 0:
            chars.append("N")
        else:
            best = max(counts, key=counts.get)
            chars.append(best if counts[best] / total >= 0.5 else "N")
    return "".join(chars)


def save_consensus(consensus_str, out_path, label):
    """Write consensus FASTA."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f">{label}\n")
        for i in range(0, len(consensus_str), 80):
            f.write(consensus_str[i:i + 80] + "\n")


def consensus_from_sequences(seqs):
    """Return a majority consensus directly from unaligned sequences."""
    clean = [
        "".join(ch for ch in str(seq).upper() if ch in "ACGTN")
        for seq in seqs
        if isinstance(seq, str) and str(seq).strip()
    ]
    if not clean:
        return ""
    max_len = max(len(seq) for seq in clean)
    chars = []
    for i in range(max_len):
        counts = {}
        for seq in clean:
            if i < len(seq):
                ch = seq[i]
                counts[ch] = counts.get(ch, 0) + 1
        chars.append(max(counts, key=counts.get) if counts else "N")
    return "".join(chars)


def alignment_stats(aligned_fasta):
    """Return dict of alignment statistics."""
    from Bio import AlignIO
    aln = AlignIO.read(str(aligned_fasta), "fasta")
    aln_len = aln.get_alignment_length()
    gap_counts = [aln[:, i].count("-") for i in range(aln_len)]
    consensus = consensus_from_alignment(aligned_fasta)
    return {
        "n_sequences": len(aln),
        "alignment_length": aln_len,
        "consensus_length_no_gaps": len(consensus.replace("-", "")),
        "avg_gaps_per_col": float(np.mean(gap_counts)),
        "cols_no_gaps": int(sum(1 for g in gap_counts if g == 0)),
        "cols_majority_gap": int(sum(1 for g in gap_counts if g > len(aln) / 2)),
    }


# ── CIAlign helper ──────────────────────────────────────────────────────────

MAX_CIALIGN_SEQS = 25000   # subsample alignment to this many seqs before CIAlign
CIALIGN_TIMEOUT  = 3600    # seconds; CIAlign hangs forever on huge inputs otherwise


def _subsample_fasta(input_fasta, output_fasta, max_seqs):
    """Write every Nth record of input_fasta so output has at most max_seqs seqs."""
    records = []
    buf = []
    with open(input_fasta) as f:
        for line in f:
            if line.startswith('>') and buf:
                records.append(''.join(buf))
                buf = []
            buf.append(line)
    if buf:
        records.append(''.join(buf))

    if len(records) <= max_seqs:
        return input_fasta   # no subsampling needed

    step = len(records) // max_seqs
    sampled = records[::step][:max_seqs]
    with open(output_fasta, 'w') as f:
        for rec in sampled:
            f.write(rec)
    _pp(f"    CIAlign: subsampled {len(records):,} → {len(sampled):,} seqs for plotting")
    return output_fasta


def run_cialign(input_fasta, output_stem, label="", timeout=CIALIGN_TIMEOUT):
    """Run CIAlign cleaning + plotting on *input_fasta*.
    Subsamples automatically if the alignment is larger than MAX_CIALIGN_SEQS."""
    import os
    input_fasta = Path(input_fasta)
    output_stem = Path(output_stem)

    n_seqs = _count_fasta_seqs(input_fasta)
    run_fasta = input_fasta
    if n_seqs > MAX_CIALIGN_SEQS:
        subsample_fa = output_stem.parent / (output_stem.name + '_cialign_input.fa')
        run_fasta = Path(_subsample_fasta(input_fasta, subsample_fa, MAX_CIALIGN_SEQS))

    cmd = [
        "CIAlign",
        "--infile",       str(run_fasta),
        "--outfile_stem", str(output_stem),
        "--remove_insertions",
        "--remove_divergent",
        "--remove_short",
        "--plot_input",
        "--plot_output",
        "--plot_markup",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "MPLBACKEND": "Agg"}
        )
    except subprocess.TimeoutExpired:
        _pp(f"    CIAlign TIMEOUT after {timeout}s for {label} — skipping plots")
        return False
    if result.returncode == 0:
        pngs = list(output_stem.parent.glob(f"{output_stem.name}*.png"))
        _pp(f"    CIAlign {label}: {len(pngs)} plots ({n_seqs:,} seqs)")
        return True
    else:
        _pp(f"    CIAlign failed for {label}: {result.stderr[:200]}")
        return False


def _cialign_index_html(cialign_dir, family_name, df):
    """Generate an index.html for CIAlign plots."""
    cialign_dir = Path(cialign_dir)
    clusters = sorted(set(
        int(f.name.split("_")[1])
        for f in cialign_dir.glob("cluster_*_alignment*.png")
    ))

    def _plot_cards(pattern):
        html = ""
        for pf in sorted(cialign_dir.glob(pattern)):
            if pf.suffix in (".png", ".svg"):
                name = pf.stem.replace("_", " ").title()
                html += (
                    f'<div class="plot-card"><h3>{name}</h3>'
                    f'<img src="{pf.name}" alt="{name}">'
                    f'<a href="{pf.name}" download>Download</a></div>\n'
                )
        return html or "<p>No plots generated.</p>"

    cluster_sections = ""
    for cl in clusters:
        cluster_sections += (
            f"<h3>Cluster {cl}</h3>"
            f'<div class="plot-grid">{_plot_cards(f"cluster_{cl}_alignment*")}</div>\n'
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
  <title>CIAlign — {family_name}</title>
  <style>
    body {{font-family:Arial,sans-serif;margin:20px;background:#f5f5f5;}}
    h1,h2,h3 {{color:#333;}}
    .plot-grid {{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:16px;margin:12px 0;}}
    .plot-card {{background:#fff;padding:16px;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,.1);}}
    .plot-card img {{width:100%;border:1px solid #ddd;border-radius:4px;}}
    .plot-card a {{display:inline-block;margin-top:8px;padding:6px 14px;background:#4CAF50;
                   color:#fff;text-decoration:none;border-radius:4px;}}
  </style>
</head>
<body>
  <h1>CIAlign — {family_name}</h1>
  <h2>Global Alignment</h2>
  <div class="plot-grid">{_plot_cards("global_alignment*")}</div>
  <h2>Per-Cluster Alignments</h2>
  {cluster_sections}
  <p style="margin-top:40px;font-size:12px;color:#888;">
    Generated with <a href="https://github.com/KatyBrown/CIAlign">CIAlign</a>
  </p>
</body>
</html>"""
    idx = cialign_dir / "index.html"
    idx.write_text(html)
    _pp(f"    CIAlign index → {idx}")


# ── main entry point ────────────────────────────────────────────────────────

def run_alignment_pipeline(df, out_dir, family_name="FAMILY"):
    """Run global + per-cluster MAFFT alignment, CIAlign, and consensus generation.

    Args:
        df:          DataFrame with 'Seq' and 'Cluster' columns.
        out_dir:     Root output directory.
        family_name: Used in filenames and FASTA headers.

    Returns:
        dict with 'mafft_available', 'cialign_available',
        'global_alignment', 'cluster_consensus_summary'
    """
    out_dir = Path(out_dir)
    mafft_ok = check_mafft()
    cialign_ok = check_cialign()

    family_lower = family_name.lower()
    consensus_dir = out_dir / "05_consensus"
    consensus_dir.mkdir(parents=True, exist_ok=True)

    # ── Write global FASTA ─────────────────────────────────────────────────
    global_fasta = out_dir / f"{family_lower}_seqs.fa"
    write_fasta(df, global_fasta, family_name)
    _pp(f"  Written {len(df)} sequences → {global_fasta.name}")

    # ── Global alignment ───────────────────────────────────────────────────
    global_aligned = out_dir / f"{family_lower}_aligned.fa"
    if mafft_ok:
        _pp(f"  Running global MAFFT alignment ({len(df)} sequences)...")
        aln_ok = run_mafft(global_fasta, global_aligned)
    else:
        _pp("  MAFFT not found — writing unaligned majority consensuses")
        aln_ok = False

    global_consensus = None
    if aln_ok:
        try:
            consensus = consensus_from_alignment(global_aligned)
            cons_path = out_dir / f"{family_lower}_consensus.fa"
            save_consensus(consensus, cons_path, f"{family_name}_consensus")
            save_consensus(consensus, consensus_dir / f"{family_lower}_consensus.fa",
                           f"{family_name}_consensus")
            _pp(f"  Global consensus → {cons_path.name}")
            global_consensus = cons_path

            # Alignment stats
            stats = alignment_stats(global_aligned)
            stats_dir = out_dir / "04_alignments"
            stats_dir.mkdir(parents=True, exist_ok=True)
            stats_file = stats_dir / "alignment_stats.txt"
            with open(stats_file, "w") as f:
                f.write("=== GLOBAL ALIGNMENT STATISTICS ===\n\n")
                for k, v in stats.items():
                    f.write(f"{k}: {v}\n")
            _pp(f"  Alignment stats → {stats_file.name}")
        except Exception as e:
            _pp(f"  WARNING: consensus generation failed: {e}")
    else:
        consensus = consensus_from_sequences(df["Seq"].tolist())
        if consensus:
            cons_path = out_dir / f"{family_lower}_consensus.fa"
            save_consensus(consensus, cons_path, f"{family_name}_consensus")
            save_consensus(consensus, consensus_dir / f"{family_lower}_consensus.fa",
                           f"{family_name}_consensus")
            _pp(f"  Global fallback consensus → {cons_path.name}")
            global_consensus = cons_path

    # ── Per-cluster alignments ─────────────────────────────────────────────
    cluster_dir = out_dir / "cluster_alignments"
    cluster_dir.mkdir(exist_ok=True)
    cluster_summaries = []

    for cluster in sorted(df["Cluster"].unique()):
        c_df = df[df["Cluster"] == cluster]
        clabel = f"noise_{abs(cluster)}" if cluster == -1 else str(cluster)
        c_fasta = cluster_dir / f"cluster_{cluster}_seqs.fa"
        c_aligned = cluster_dir / f"cluster_{cluster}_aligned.fa"
        write_fasta(c_df, c_fasta, family_name)
        if mafft_ok and len(c_df) >= 2:
            _pp(f"  Cluster {clabel} alignment ({len(c_df)} seqs)...")
            ok = run_mafft(c_fasta, c_aligned)
        else:
            ok = False
            reason = "single sequence" if len(c_df) < 2 else "MAFFT unavailable"
            _pp(f"  Cluster {clabel}: writing fallback consensus ({reason})")

        try:
            con = consensus_from_alignment(c_aligned) if ok else consensus_from_sequences(c_df["Seq"].tolist())
            if not con:
                continue
            con_path = cluster_dir / f"cluster_{cluster}_consensus.fa"
            save_consensus(con, con_path, f"{family_name}_cluster{cluster}_consensus")
            save_consensus(con, consensus_dir / f"cluster_{cluster}_consensus.fa",
                           f"{family_name}_cluster{cluster}_consensus")
            stats = alignment_stats(c_aligned) if ok else {
                "n_sequences": len(c_df),
                "alignment_length": len(con),
                "consensus_length_no_gaps": len(con),
                "avg_gaps_per_col": 0.0,
                "cols_no_gaps": len(con),
                "cols_majority_gap": 0,
            }
            cluster_summaries.append({
                "cluster": cluster, "size": len(c_df),
                "consensus_length": len(con.replace("-", "")),
                "fallback": "" if ok else "unaligned_majority",
                **stats,
            })
        except Exception as e:
            _pp(f"    WARNING cluster {clabel} consensus failed: {e}")

    # Combined cluster consensuses FASTA
    if cluster_summaries:
        all_cons = cluster_dir / "all_cluster_consensuses.fa"
        with open(all_cons, "w") as outf:
            for cl in sorted(df["Cluster"].unique()):
                cf = cluster_dir / f"cluster_{cl}_consensus.fa"
                if cf.exists():
                    outf.write(cf.read_text())
        _pp(f"  All cluster consensuses → {all_cons.name}")
        pd.DataFrame(cluster_summaries).to_csv(
            cluster_dir / "cluster_consensus_summary.csv", index=False
        )
        pd.DataFrame(cluster_summaries).to_csv(
            consensus_dir / "cluster_consensus_summary.csv", index=False
        )

    # ── CIAlign ────────────────────────────────────────────────────────────
    cialign_dir = out_dir / "cialign_plots"
    cleaned_dir = out_dir / "cleaned_consensus"
    cialign_dir.mkdir(exist_ok=True)
    cleaned_dir.mkdir(exist_ok=True)

    if cialign_ok:
        import os
        os.environ["MPLBACKEND"] = "Agg"
        _pp("  Running CIAlign...")

        # Global
        if global_aligned.exists() and global_aligned.stat().st_size > 0 and aln_ok:
            run_cialign(global_aligned, cialign_dir / "global_alignment", "global")
            gc_cleaned = cialign_dir / "global_alignment_cleaned.fasta"
            if gc_cleaned.exists():
                try:
                    con = consensus_from_alignment(gc_cleaned)
                    save_consensus(con, cleaned_dir / f"{family_lower}_cleaned_consensus.fa",
                                   f"{family_name}_cleaned_consensus")
                except Exception:
                    pass

        # Per-cluster
        all_cleaned = []
        for cl in sorted(df["Cluster"].unique()):
            ca = cluster_dir / f"cluster_{cl}_aligned.fa"
            if ca.exists() and ca.stat().st_size > 0:
                run_cialign(ca, cialign_dir / f"cluster_{cl}_alignment", f"cluster {cl}")
                cl_cleaned = cialign_dir / f"cluster_{cl}_alignment_cleaned.fasta"
                if cl_cleaned.exists():
                    try:
                        con = consensus_from_alignment(cl_cleaned)
                        cp = cleaned_dir / f"cluster_{cl}_cleaned_consensus.fa"
                        save_consensus(con, cp, f"{family_name}_cluster{cl}_cleaned_consensus")
                        all_cleaned.append({"cluster": cl, "consensus": con})
                    except Exception:
                        pass

        # Combined cleaned FASTA
        if all_cleaned:
            with open(cleaned_dir / "all_clusters_cleaned_consensus.fa", "w") as f:
                for item in all_cleaned:
                    cl = item["cluster"]
                    con = item["consensus"]
                    f.write(f">{family_name}_cluster{cl}_cleaned_consensus\n")
                    for i in range(0, len(con), 80):
                        f.write(con[i:i + 80] + "\n")

        # CIAlign index page
        _cialign_index_html(cialign_dir, family_name, df)
    else:
        _pp("  CIAlign not found — skipping alignment visualization")

    return {
        "mafft_available": mafft_ok,
        "cialign_available": cialign_ok,
        "global_alignment": str(global_aligned) if global_aligned.exists() else None,
        "cluster_consensus_summary": cluster_summaries,
    }


# ── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="MAFFT alignment + CIAlign visualization")
    p.add_argument("--input",  required=True, help="CSV with Seq + Cluster columns")
    p.add_argument("--out-dir", default=".", help="Output directory")
    p.add_argument("--family", default="FAMILY")
    p.add_argument("--notify-email", default="", metavar="EMAIL",
                   help="Send a completion email to this address (requires Gmail App Password setup).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    df = pd.read_csv(args.input)
    if "Seq" not in df.columns:
        print("ERROR: Input CSV must have a 'Seq' column.")
        sys.exit(1)
    if "Cluster" not in df.columns:
        df["Cluster"] = 0
    run_alignment_pipeline(df, args.out_dir, args.family)
    if args.notify_email:
        from te_notify import send_completion_email
        send_completion_email(args.notify_email, "te_alignment", args.out_dir)
