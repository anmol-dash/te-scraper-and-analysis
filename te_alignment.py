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


def _cialign_version():
    """CIAlign's reported version, or 'unknown'.

    Recorded alongside every failure: CIAlign crashes are version-specific, and
    a traceback without the version it came from is close to unusable when the
    cluster and the workstation have different builds.
    """
    try:
        r = subprocess.run(["CIAlign", "--version"], capture_output=True, text=True)
        return (r.stdout or r.stderr or "").strip() or "unknown"
    except (FileNotFoundError, OSError):
        return "unknown"


def check_cialign():
    """Return True if CIAlign is in PATH."""
    try:
        subprocess.run(["CIAlign", "--version"], capture_output=True)
        _pp(f"  CIAlign version: {_cialign_version()}")
        return True
    except FileNotFoundError:
        return False


def check_trimal():
    """Return True if trimAl is in PATH."""
    try:
        subprocess.run(["trimal", "--version"], capture_output=True)
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
        if result.returncode != 0:
            # bioconda 7.525 segfaults in pairlocalalign; --6merpair uses a
            # different pairwise path that avoids the crash (verified vs 7.525).
            _pp("  MAFFT failed — retrying with --6merpair (segfault-safe path)")
            cmd = f"mafft --6merpair --retree 2 --thread {threads} {input_fasta} > {output_fasta}"
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

MAX_CIALIGN_SEQS      = 25000   # subsample alignment to this many seqs before CIAlign
CIALIGN_TIMEOUT       = 3600    # seconds; CIAlign hangs forever on huge inputs otherwise
TRIMAL_TIMEOUT        = 1800    # seconds; trimAl fallback cleaning
CIALIGN_FALLBACK_SEQS = 1000    # seqs to display when retrying after a trimAl fallback


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


def run_trimal(input_fasta, output_fasta, timeout=TRIMAL_TIMEOUT):
    """Clean an alignment with trimAl (-automated1 heuristic).

    Used as the CIAlign fallback: trimAl's column-trimming is far cheaper
    than CIAlign's own cleaning + plotting pass, so it can shrink a huge or
    gappy alignment down to something CIAlign can actually finish on retry.
    """
    input_fasta, output_fasta = Path(input_fasta), Path(output_fasta)
    cmd = ["trimal", "-in", str(input_fasta), "-out", str(output_fasta), "-automated1"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        _pp(f"    trimAl TIMEOUT after {timeout}s for {input_fasta.name}")
        return False
    except FileNotFoundError:
        _pp("    trimAl not found — cannot fall back")
        return False
    if result.returncode == 0 and output_fasta.exists():
        _pp(f"    trimAl cleaned {input_fasta.name} → {output_fasta.name}")
        return True
    _err = ((result.stderr or "") + (result.stdout or "")).strip()
    _tail = [ln for ln in _err.splitlines() if ln.strip()][-3:] or ["(no output)"]
    _pp(f"    trimAl failed for {input_fasta.name} (exit {result.returncode}):")
    for _ln in _tail:
        _pp(f"      {_ln}")
    return False


def run_cialign(input_fasta, output_stem, label="", timeout=CIALIGN_TIMEOUT,
                 max_seqs=MAX_CIALIGN_SEQS, _fallback_attempted=False):
    """Run CIAlign cleaning + plotting on *input_fasta*.

    Subsamples automatically if the alignment is larger than *max_seqs*.

    Design choice: if CIAlign fails outright or times out (>1h on huge
    alignments), don't just give up — clean the alignment with trimAl
    (-automated1, much cheaper than CIAlign's own cleaning pass) and retry
    CIAlign once on the trimAl-cleaned alignment, capped at
    CIALIGN_FALLBACK_SEQS (1000) sequences for display. This keeps massive
    families (tens of thousands of loci) from silently ending up with no
    alignment visualization at all.
    """
    import os
    input_fasta = Path(input_fasta)
    output_stem = Path(output_stem)

    n_seqs = _count_fasta_seqs(input_fasta)
    run_fasta = input_fasta
    if n_seqs > max_seqs:
        subsample_fa = output_stem.parent / (output_stem.name + '_cialign_input.fa')
        run_fasta = Path(_subsample_fasta(input_fasta, subsample_fa, max_seqs))

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
    err_path = output_stem.parent / (output_stem.name + "_cialign_error.txt")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "MPLBACKEND": "Agg"}
        )
        failed = result.returncode != 0
        full_err = (result.stderr or "") + (result.stdout or "")
        reason = f"exit {result.returncode}"
    except subprocess.TimeoutExpired:
        failed, full_err, reason = True, "", f"TIMEOUT after {timeout}s"

    if not failed:
        pngs = list(output_stem.parent.glob(f"{output_stem.name}*.png"))
        _pp(f"    CIAlign {label}: {len(pngs)} plots ({n_seqs:,} seqs)")
        return True

    # The old code kept only stderr[:200], which cut every traceback off inside
    # the site-packages path and left the actual exception unknowable. Write the
    # whole thing to a file next to the plots and echo the tail (the exception
    # line lives at the END of a traceback, not the start).
    if full_err:
        try:
            err_path.write_text(
                f"CMD: {' '.join(cmd)}\n"
                f"CIAlign: {_cialign_version()}\n"
                f"input: {input_fasta}  ({n_seqs} seqs)\n"
                f"{'=' * 60}\n{full_err}"
            )
        except OSError:
            pass
        tail = [ln for ln in full_err.strip().splitlines() if ln.strip()][-3:]
        _pp(f"    CIAlign failed for {label} ({reason}):")
        for ln in tail:
            _pp(f"      {ln}")
        _pp(f"      full output → {err_path}")
    else:
        _pp(f"    CIAlign failed for {label}: {reason}")

    if _fallback_attempted:
        _pp(f"    CIAlign fallback (trimAl) also failed for {label} — skipping plots")
        return False

    trimmed_fa = output_stem.parent / (output_stem.name + '_trimal_cleaned.fa')
    if not run_trimal(input_fasta, trimmed_fa):
        return False

    _pp(f"    Retrying CIAlign on trimAl-cleaned alignment ({label}), "
        f"capped at {CIALIGN_FALLBACK_SEQS:,} seqs")
    return run_cialign(
        trimmed_fa, output_stem, label=f"{label} [trimAl fallback]",
        timeout=timeout, max_seqs=CIALIGN_FALLBACK_SEQS, _fallback_attempted=True,
    )


def _alignment_index_html(align_dir, family_name, groups):
    """Write 04_alignments/index.html — one section per group.

    *groups* is the ordered list of group names ("whole_family", "cluster_0",
    ...). Each section shows the CIAlign plots for that group side by side and
    links its raw + cleaned FASTAs, so the uncleaned and CIAlign-cleaned
    versions of the same alignment are always presented together.
    """
    align_dir = Path(align_dir)
    images_dir = align_dir / "images"
    fasta_dir = align_dir / "fasta"

    def _fasta_links(group):
        rows = [
            ("Sequences (unaligned)",        f"{group}_seqs.fa"),
            ("Alignment — not cleaned",      f"{group}_aligned.fa"),
            ("Alignment — CIAlign cleaned",  f"{group}_aligned_cleaned.fa"),
            ("Consensus — not cleaned",      f"{group}_consensus.fa"),
            ("Consensus — CIAlign cleaned",  f"{group}_consensus_cleaned.fa"),
        ]
        out = ""
        for label, fn in rows:
            if (fasta_dir / fn).exists():
                kb = (fasta_dir / fn).stat().st_size / 1e3
                out += (f'<tr><td>{label}</td>'
                        f'<td><a href="fasta/{fn}">{fn}</a></td>'
                        f'<td class="num">{kb:,.1f} KB</td></tr>')
            else:
                out += (f'<tr class="missing"><td>{label}</td>'
                        f'<td colspan="2">not produced</td></tr>')
        return out

    def _plot_cards(group):
        html = ""
        for pf in sorted(images_dir.glob(f"{group}_alignment*")):
            if pf.suffix.lower() not in (".png", ".svg"):
                continue
            tag = pf.stem[len(group) + 1:].replace("_", " ").strip() or "plot"
            kind = ("CIAlign cleaned (output)" if "output" in pf.stem
                    else "not cleaned (input)" if "input" in pf.stem else tag)
            html += (
                f'<div class="plot-card"><h4>{tag}</h4>'
                f'<p class="kind">{kind}</p>'
                f'<img src="images/{pf.name}" alt="{tag}" loading="lazy">'
                f'<a href="images/{pf.name}" download>Download</a></div>\n'
            )
        return html or '<p class="none">No CIAlign plots generated for this group.</p>'

    sections = ""
    for g in groups:
        title = "Whole family" if g == "whole_family" else g.replace("_", " ").title()
        sections += f"""
  <section>
    <h2>{title}</h2>
    <table class="files">
      <tr><th>File</th><th>Name</th><th class="num">Size</th></tr>
      {_fasta_links(g)}
    </table>
    <div class="plot-grid">{_plot_cards(g)}</div>
  </section>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Alignments &amp; Consensus — {family_name}</title>
  <style>
    body {{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;
           margin:0;padding:24px;background:#f5f5f5;color:#222;}}
    h1 {{margin:0 0 4px;}}
    .sub {{color:#666;margin:0 0 24px;font-size:14px;}}
    section {{background:#fff;padding:20px;border-radius:8px;margin-bottom:20px;
              box-shadow:0 2px 4px rgba(0,0,0,.08);}}
    h2 {{margin:0 0 14px;padding-bottom:8px;border-bottom:2px solid #eee;}}
    table.files {{border-collapse:collapse;width:100%;max-width:760px;margin-bottom:18px;
                  font-size:14px;}}
    table.files th, table.files td {{text-align:left;padding:6px 10px;border-bottom:1px solid #eee;}}
    table.files th {{background:#fafafa;font-weight:600;}}
    td.num, th.num {{text-align:right;white-space:nowrap;}}
    tr.missing td {{color:#aaa;font-style:italic;}}
    .plot-grid {{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:16px;}}
    .plot-card {{background:#fcfcfc;padding:14px;border:1px solid #eee;border-radius:6px;}}
    .plot-card h4 {{margin:0 0 2px;font-size:14px;}}
    .plot-card .kind {{margin:0 0 8px;font-size:12px;color:#777;}}
    .plot-card img {{width:100%;border:1px solid #ddd;border-radius:4px;background:#fff;}}
    .plot-card a {{display:inline-block;margin-top:8px;padding:5px 12px;background:#4CAF50;
                   color:#fff;text-decoration:none;border-radius:4px;font-size:13px;}}
    .none {{color:#999;font-style:italic;}}
    footer {{font-size:12px;color:#888;margin-top:8px;}}
  </style>
</head>
<body>
  <h1>Alignments &amp; Consensus — {family_name}</h1>
  <p class="sub">
    Every group below (the whole family and each cluster) has both a
    <strong>not cleaned</strong> and a <strong>CIAlign-cleaned</strong> alignment
    and consensus. All FASTAs are in <code>fasta/</code>; all images are in
    <code>images/</code>.
  </p>
  {sections}
  <footer>
    Alignment: <a href="https://mafft.cbrc.jp/alignment/software/">MAFFT</a> &middot;
    cleaning/plots: <a href="https://github.com/KatyBrown/CIAlign">CIAlign</a>
  </footer>
</body>
</html>"""
    idx = align_dir / "index.html"
    idx.write_text(html)
    _pp(f"  Alignment index → {idx}")


# ── main entry point ────────────────────────────────────────────────────────

def _group_frames(df):
    """Yield (group_name, sub_df) for the whole family and then each cluster.

    The whole family is treated as just another group so it gets exactly the
    same treatment as the clusters — its own alignment, consensus, CIAlign
    cleaning and plots — rather than being a separate special case whose
    outputs landed in a different directory.
    """
    yield "whole_family", df
    if "Cluster" not in df.columns:
        return
    for cl in sorted(df["Cluster"].unique()):
        name = "cluster_noise" if cl == -1 else f"cluster_{cl}"
        yield name, df[df["Cluster"] == cl]


def run_alignment_pipeline(df, out_dir, family_name="FAMILY"):
    """Run MAFFT alignment, CIAlign cleaning and consensus for every group.

    A "group" is the whole family plus each cluster. Each group gets, in a
    single output tree:

        04_alignments/fasta/<group>_seqs.fa               unaligned input
        04_alignments/fasta/<group>_aligned.fa            alignment, not cleaned
        04_alignments/fasta/<group>_aligned_cleaned.fa    alignment, CIAlign-cleaned
        04_alignments/fasta/<group>_consensus.fa          consensus, not cleaned
        04_alignments/fasta/<group>_consensus_cleaned.fa  consensus, CIAlign-cleaned
        04_alignments/images/<group>_alignment*.png       CIAlign plots
        04_alignments/logs/                               CIAlign logs / side files

    plus combined FASTAs, consensus_summary.csv, alignment_stats.txt and an
    index.html. All FASTAs live in one directory and all images in another —
    consensus, alignments and CIAlign output are no longer split across
    05_consensus/, cluster_alignments/, cialign_plots/ and cleaned_consensus/.

    Args:
        df:          DataFrame with 'Seq' and 'Cluster' columns.
        out_dir:     Root output directory.
        family_name: Used in filenames and FASTA headers.

    Returns:
        dict with 'mafft_available', 'cialign_available', 'trimal_available',
        'alignment_dir', 'groups' and 'cluster_consensus_summary'
    """
    import os
    import shutil as _shutil

    out_dir = Path(out_dir)
    mafft_ok = check_mafft()
    cialign_ok = check_cialign()
    trimal_ok = check_trimal()
    if cialign_ok and not trimal_ok:
        _pp("  Note: trimAl not found — CIAlign failures/timeouts will not have a fallback")
    if not cialign_ok:
        _pp("  CIAlign not found — cleaned alignments/consensuses and plots will be skipped")

    # One folder for everything alignment-related.
    align_dir = out_dir / "04_alignments"
    fasta_dir = align_dir / "fasta"
    images_dir = align_dir / "images"
    logs_dir = align_dir / "logs"
    for d in (fasta_dir, images_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    if cialign_ok:
        os.environ["MPLBACKEND"] = "Agg"

    summaries = []
    group_names = []
    cialign_failures = []   # groups whose plots/cleaned alignment never appeared

    for gname, g_df in _group_frames(df):
        n = len(g_df)
        if n == 0:
            continue
        group_names.append(gname)
        pretty = "whole family" if gname == "whole_family" else gname.replace("_", " ")
        _pp(f"  ── {pretty}  ({n:,} sequences) ──")

        seqs_fa    = fasta_dir / f"{gname}_seqs.fa"
        aligned_fa = fasta_dir / f"{gname}_aligned.fa"
        write_fasta(g_df, seqs_fa, family_name)

        # ── MAFFT ──────────────────────────────────────────────────────────
        if mafft_ok and n >= 2:
            aln_ok = run_mafft(seqs_fa, aligned_fa)
        else:
            aln_ok = False
            reason = "single sequence" if n < 2 else "MAFFT unavailable"
            _pp(f"    no alignment ({reason}) — using unaligned majority consensus")

        row = {"group": gname,
               "cluster": ("whole_family" if gname == "whole_family"
                           else ("noise" if gname == "cluster_noise"
                                 else gname.rsplit("_", 1)[-1])),
               "size": n,
               "aligned": bool(aln_ok),
               "fallback": "" if aln_ok else "unaligned_majority"}

        # ── Consensus from the NOT-cleaned alignment ───────────────────────
        con = None
        try:
            con = (consensus_from_alignment(aligned_fa) if aln_ok
                   else consensus_from_sequences(g_df["Seq"].tolist()))
            if con:
                save_consensus(con, fasta_dir / f"{gname}_consensus.fa",
                               f"{family_name}_{gname}_consensus")
                row["consensus_length"] = len(con.replace("-", ""))
        except Exception as e:
            _pp(f"    WARNING: consensus failed for {pretty}: {e}")

        if aln_ok:
            try:
                row.update(alignment_stats(aligned_fa))
            except Exception as e:
                _pp(f"    WARNING: alignment stats failed for {pretty}: {e}")
        elif con:
            row.update({"n_sequences": n, "alignment_length": len(con),
                        "consensus_length_no_gaps": len(con),
                        "avg_gaps_per_col": 0.0, "cols_no_gaps": len(con),
                        "cols_majority_gap": 0})

        # ── CIAlign: cleaned alignment + plots + cleaned consensus ─────────
        row["cialign_cleaned"] = False
        if cialign_ok and aln_ok and aligned_fa.exists() and aligned_fa.stat().st_size > 0:
            stem = images_dir / f"{gname}_alignment"
            if not run_cialign(aligned_fa, stem, pretty):
                cialign_failures.append(gname)

            # CIAlign drops its cleaned alignment next to the plots; move it
            # into the FASTA folder so every .fa lives in exactly one place.
            produced = images_dir / f"{gname}_alignment_cleaned.fasta"
            cleaned_fa = fasta_dir / f"{gname}_aligned_cleaned.fa"
            if produced.exists() and produced.stat().st_size > 0:
                _shutil.move(str(produced), str(cleaned_fa))
                row["cialign_cleaned"] = True
                try:
                    ccon = consensus_from_alignment(cleaned_fa)
                    if ccon:
                        save_consensus(
                            ccon, fasta_dir / f"{gname}_consensus_cleaned.fa",
                            f"{family_name}_{gname}_cleaned_consensus")
                        row["cleaned_consensus_length"] = len(ccon.replace("-", ""))
                    row.update({f"cleaned_{k}": v
                                for k, v in alignment_stats(cleaned_fa).items()})
                except Exception as e:
                    _pp(f"    WARNING: cleaned consensus failed for {pretty}: {e}")
            else:
                _pp(f"    CIAlign produced no cleaned alignment for {pretty}")

            # Keep images/ images-only: sweep CIAlign's logs and side files out.
            for side in list(images_dir.glob(f"{gname}_alignment*")):
                if side.suffix.lower() in (".png", ".svg"):
                    continue
                try:
                    _shutil.move(str(side), str(logs_dir / side.name))
                except Exception:
                    pass

        summaries.append(row)

    # ── Combined FASTAs ────────────────────────────────────────────────────
    def _concat(dest, names, suffix):
        wrote = 0
        with open(dest, "w") as outf:
            for g in names:
                src = fasta_dir / f"{g}_{suffix}"
                if src.exists():
                    outf.write(src.read_text())
                    wrote += 1
        if wrote:
            _pp(f"  {dest.name}: {wrote} sequences")
        else:
            dest.unlink(missing_ok=True)
        return wrote

    cluster_only = [g for g in group_names if g != "whole_family"]
    # Everything, including the whole family.
    _concat(fasta_dir / "all_consensuses.fa", group_names, "consensus.fa")
    _concat(fasta_dir / "all_consensuses_cleaned.fa", group_names, "consensus_cleaned.fa")
    # Clusters-only, under the historical names: downstream modules
    # (te_divergence, te_subfamily, te_motif_gain, run_phylo_analysis) parse
    # these into {cluster_id: seq} and would choke on a whole-family record.
    _concat(fasta_dir / "all_cluster_consensuses.fa", cluster_only, "consensus.fa")
    _concat(fasta_dir / "all_clusters_cleaned_consensus.fa",
            cluster_only, "consensus_cleaned.fa")

    # ── Summary table + stats ──────────────────────────────────────────────
    if summaries:
        sum_df = pd.DataFrame(summaries)
        sum_df.to_csv(align_dir / "consensus_summary.csv", index=False)
        # Historical filename, same content, for the report generators.
        sum_df.to_csv(align_dir / "cluster_consensus_summary.csv", index=False)
        _pp(f"  Consensus summary → {align_dir / 'consensus_summary.csv'}")

        with open(align_dir / "alignment_stats.txt", "w") as f:
            f.write(f"=== ALIGNMENT STATISTICS — {family_name} ===\n")
            f.write("Groups: the whole family and each cluster.\n")
            f.write("'cleaned_*' rows come from the CIAlign-cleaned alignment.\n\n")
            for row in summaries:
                head = ("WHOLE FAMILY" if row["group"] == "whole_family"
                        else row["group"].replace("_", " ").upper())
                f.write(f"--- {head} ---\n")
                for k, v in row.items():
                    if k == "group":
                        continue
                    f.write(f"{k}: {v}\n")
                f.write("\n")
        _pp(f"  Alignment stats → {align_dir / 'alignment_stats.txt'}")

    _alignment_index_html(align_dir, family_name, group_names)

    # Surface CIAlign losses instead of letting the stage read as clean: an
    # 11/11 wipe-out previously showed up only as absent PNGs, with the stage
    # still logging "Alignment OK".
    if cialign_failures:
        _pp(f"  WARNING: CIAlign failed for {len(cialign_failures)}/"
            f"{len(group_names)} groups: {', '.join(cialign_failures)}")
        _pp(f"           No cleaned alignment or plots for those groups. "
            f"Tracebacks: {logs_dir}/<group>_alignment_cialign_error.txt")

    whole = fasta_dir / "whole_family_aligned.fa"
    return {
        "mafft_available": mafft_ok,
        "cialign_available": cialign_ok,
        "trimal_available": trimal_ok,
        "alignment_dir": str(align_dir),
        "groups": group_names,
        "cialign_failures": cialign_failures,
        "global_alignment": str(whole) if whole.exists() else None,
        "cluster_consensus_summary": summaries,
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
