#!/usr/bin/env python3
"""
compute_consensus5_distances.py --- CIAlign-cleaned consensus distances for ALL
copies, grouped by the 5 UMAP/HDBSCAN clusters in clusters5.csv.

Why CIAlign: a raw MAFFT consensus built on the longest copy's frame is bloated
(thousands of insertion columns present in only a handful of copies -> a
multi-kb "consensus" that is mostly gaps). CIAlign removes insertions, crops
poorly aligned ends, and drops divergent sequences, leaving a short core
consensus that actually represents the cluster. Distances are then measured to
that cleaned consensus.

Per cluster:
  1. MAFFT --auto MSA of a sample of the cluster (--msa-sample copies).
  2. CIAlign cleans the MSA  -> cleaned alignment + cleaned (core) consensus.
  3. EVERY copy is projected onto the cleaned consensus with
     `mafft --addfragments --keeplength` -> Kimura-2P dist_cluster.

Global:
  the 5 cleaned cluster consensuses are MSA'd + CIAlign-cleaned -> cleaned global
  consensus; each copy's dist_global is read through a cheap cleaned-cluster ->
  cleaned-global position map (no 12k-copy family-wide MAFFT).

Outputs:
  --out                 consensus_distance_5clust.csv  (TE_ID, cluster,
                        dist_cluster, dist_global)
  --aln-dir/            cluster_<c>_cleaned.fasta, cluster_<c>_consensus.fasta,
                        global_cleaned.fasta, global_consensus.fasta

Requires: numpy, pandas, MAFFT and CIAlign on PATH. No matplotlib / sklearn.

Usage:
    python compute_consensus5_distances.py \
        --input clusters5.csv \
        --out   consensus_distance_5clust.csv \
        --aln-dir cleaned_aln \
        --threads ${SLURM_CPUS_PER_TASK:-8}
"""
import argparse
import datetime
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def _pp(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Kimura 2-parameter distance ───────────────────────────────────────────────
_PUR, _PYR = set("AG"), set("CT")


def kimura2p(a, b, min_sites=20, max_k2p=5.0):
    transitions = transversions = sites = 0
    for x, y in zip(a.upper(), b.upper()):
        if x in "-N" or y in "-N" or x not in "ACGT" or y not in "ACGT":
            continue
        sites += 1
        if x == y:
            continue
        if (x in _PUR and y in _PUR) or (x in _PYR and y in _PYR):
            transitions += 1
        else:
            transversions += 1
    if sites < min_sites:
        return float("nan")
    P, Q = transitions / sites, transversions / sites
    t1, t2 = 1.0 - 2.0 * P - Q, 1.0 - 2.0 * Q
    if t1 <= 0 or t2 <= 0:
        return float("nan")
    k = -0.5 * np.log(t1) - 0.25 * np.log(t2)
    if not np.isfinite(k) or k > max_k2p:
        return float("nan")
    return float(k)


# ── FASTA / MAFFT / CIAlign helpers ───────────────────────────────────────────
class AlignmentError(RuntimeError):
    pass


def _exe(cmd, alt=None):
    e = shutil.which(cmd) or (shutil.which(alt) if alt else None)
    if not e:
        raise AlignmentError(f"{cmd!r} not found on PATH"
                             + (f" (nor {alt!r})" if alt else "") + ".")
    return e


def _read_fasta_text(text):
    records, name, buf = [], None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(buf)))
            name = line[1:].strip().split()[0] if line[1:].strip() else ""
            buf = []
        else:
            buf.append(line.strip())
    if name is not None:
        records.append((name, "".join(buf)))
    return records


def _ungap(s):
    return "".join(c for c in s.upper() if c in "ACGT")


def run_mafft_msa(records, cmd, threads):
    """Plain MAFFT --auto MSA of (name, seq) -> list of (name, aligned_seq)."""
    exe = _exe(cmd, "mafft")
    with tempfile.TemporaryDirectory() as td:
        fin = Path(td) / "in.fa"
        fin.write_text("".join(f">{n}\n{s}\n" for n, s in records))
        res = subprocess.run([exe, "--auto", "--thread", str(threads),
                              "--quiet", str(fin)], capture_output=True, text=True)
        if res.returncode != 0:
            raise AlignmentError(f"mafft failed ({res.returncode}): "
                                 f"{res.stderr.strip()[:300]}")
        out = _read_fasta_text(res.stdout)
        if not out:
            raise AlignmentError("mafft produced no alignment output.")
        return out


def run_cialign(aln_records, stem, aln_dir, cmd, keep_consensus_in_aln=False):
    """Clean a MAFFT MSA with CIAlign. Writes <stem>_cleaned.fasta and
    <stem>_consensus.fasta into aln_dir; returns the cleaned (core) consensus
    string (gaps stripped)."""
    exe = _exe(cmd, "cialign")
    aln_dir = Path(aln_dir)
    aln_dir.mkdir(parents=True, exist_ok=True)
    out_stem = aln_dir / stem
    with tempfile.TemporaryDirectory() as td:
        fin = Path(td) / "msa.fa"
        fin.write_text("".join(f">{n}\n{s}\n" for n, s in aln_records))
        cmd_v = [exe, "--infile", str(fin), "--outfile_stem", str(out_stem),
                 "--remove_insertions", "--crop_ends", "--remove_divergent",
                 "--make_consensus"]
        res = subprocess.run(cmd_v, capture_output=True, text=True)
        if res.returncode != 0:
            raise AlignmentError(f"CIAlign failed ({res.returncode}): "
                                 f"{(res.stderr or res.stdout).strip()[:300]}")
    cons_fa = Path(f"{out_stem}_consensus.fasta")
    if not cons_fa.exists():
        raise AlignmentError(f"CIAlign produced no consensus ({cons_fa}).")
    recs = _read_fasta_text(cons_fa.read_text())
    # pick the record named like a consensus, else the longest
    cand = [r for r in recs if "consensus" in r[0].lower()] or recs
    cons = max(cand, key=lambda r: len(_ungap(r[1])))[1]
    return _ungap(cons)


def project_onto_reference(seqs, ref, cmd, threads):
    """Project every sequence onto a given reference (the cleaned consensus) in
    ONE MAFFT pass. Returns [aligned_row_or_None] of len == len(seqs), each row
    in ref-frame coordinates (== len(ref)) thanks to --keeplength."""
    exe = _exe(cmd, "mafft")
    clean = [(i, _ungap(s)) for i, s in enumerate(seqs)]
    clean = [(i, s) for i, s in clean if s]
    if not clean:
        raise AlignmentError("no non-empty sequences to project.")
    with tempfile.TemporaryDirectory() as td:
        refp = Path(td) / "ref.fa"
        refp.write_text(f">REF\n{ref}\n")
        frags = Path(td) / "frags.fa"
        frags.write_text("".join(f">c{i}\n{s}\n" for i, s in clean))
        res = subprocess.run(
            [exe, "--addfragments", str(frags), "--keeplength",
             "--thread", str(threads), "--quiet", str(refp)],
            capture_output=True, text=True)
        if res.returncode != 0:
            raise AlignmentError(f"mafft --addfragments failed ({res.returncode}): "
                                 f"{res.stderr.strip()[:300]}")
        aligned = dict(_read_fasta_text(res.stdout))
    rows = {i: aligned[f"c{i}"] for i, _ in clean if f"c{i}" in aligned}
    return [rows.get(i) for i in range(len(seqs))]


def map_consensus(src_cons, dst_cons, cmd, threads):
    """Pairwise-align src consensus to dst consensus; return, per src position,
    the aligned dst base ('ACGT') or None."""
    aln = dict(run_mafft_msa([("D", dst_cons), ("S", src_cons)], cmd, threads))
    aD, aS = aln.get("D", ""), aln.get("S", "")
    mapping = [None] * len(src_cons)
    pos = 0
    for col in range(min(len(aD), len(aS))):
        if aS[col] != "-":
            if pos < len(mapping):
                b = aD[col].upper()
                mapping[pos] = b if b in "ACGT" else None
            pos += 1
    return mapping


# ── main ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="clusters5.csv",
                   help="CSV with TE_ID, Seq, Cluster columns.")
    p.add_argument("--out", default="consensus_distance_5clust.csv")
    p.add_argument("--aln-dir", default="cleaned_aln",
                   help="Directory for CIAlign cleaned alignments + consensuses.")
    p.add_argument("--cluster-col", default="Cluster")
    p.add_argument("--msa-sample", type=int, default=250,
                   help="Max copies per cluster MSA'd before CIAlign (default 250). "
                        "ALL copies are still scored against the cleaned consensus.")
    p.add_argument("--max-k2p", type=float, default=5.0,
                   help="Drop distances above this (default 5.0 = keep everything).")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--mafft-cmd", default="mafft")
    p.add_argument("--cialign-cmd", default="CIAlign")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    _pp("=" * 60)
    _pp("5-cluster CIAlign-cleaned consensus distances")
    _pp(f"  input={args.input}  out={args.out}  aln-dir={args.aln_dir}")
    _pp(f"  msa-sample={args.msa_sample}  threads={args.threads}")
    _pp("=" * 60)
    # fail fast if tools are missing
    _exe(args.mafft_cmd, "mafft"); _exe(args.cialign_cmd, "cialign")

    df = pd.read_csv(args.input)
    df.columns = [str(c).strip() for c in df.columns]
    for col in ("TE_ID", "Seq", args.cluster_col):
        if col not in df.columns:
            sys.exit(f"ERROR: input missing column '{col}'.")
    df = df[df["Seq"].astype(str).str.len() > 0].reset_index(drop=True)
    df["__cluster"] = df[args.cluster_col].astype(str)
    seqs = df["Seq"].astype(str).str.upper().tolist()
    _pp(f"  {len(df):,} copies in {df['__cluster'].nunique()} clusters.")

    cluster_dist = np.full(len(df), np.nan)
    cluster_cons, cluster_rows = {}, {}

    for cid, sub in df.groupby("__cluster"):
        idx = sub.index.to_list()
        # 1. MSA of a sample, 2. CIAlign clean -> core consensus
        if len(idx) > args.msa_sample:
            sidx = rng.choice(idx, size=args.msa_sample, replace=False)
        else:
            sidx = idx
        sample_recs = [(f"s{i}", seqs[i]) for i in sidx]
        try:
            raw = run_mafft_msa(sample_recs, args.mafft_cmd, args.threads)
            cons = run_cialign(raw, f"cluster_{cid}", args.aln_dir, args.cialign_cmd)
        except AlignmentError as e:
            _pp(f"  cluster {cid}: SKIP — {e}")
            continue
        if len(cons) < 20:
            _pp(f"  cluster {cid}: SKIP — cleaned consensus too short ({len(cons)} bp)")
            continue
        cluster_cons[cid] = cons
        # 3. project ALL copies onto the cleaned consensus
        rows = project_onto_reference([seqs[i] for i in idx], cons,
                                      args.mafft_cmd, args.threads)
        cluster_rows[cid] = {idx[j]: rows[j] for j in range(len(idx))}
        good = 0
        for j, i in enumerate(idx):
            if rows[j] is not None:
                k = kimura2p(rows[j], cons, max_k2p=args.max_k2p)
                cluster_dist[i] = k
                good += int(np.isfinite(k))
        _pp(f"  cluster {cid}: {len(idx)} copies, cleaned cons {len(cons)} bp, "
            f"{good} distances")
    df["dist_cluster"] = cluster_dist

    # global: MSA + CIAlign of the 5 cleaned cluster consensuses
    _pp("Building cleaned global consensus from the cluster consensuses...")
    cons_recs = [(f"cluster_{cid}", s) for cid, s in cluster_cons.items() if s]
    if len(cons_recs) >= 2:
        raw_g = run_mafft_msa(cons_recs, args.mafft_cmd, args.threads)
        global_cons = run_cialign(raw_g, "global", args.aln_dir, args.cialign_cmd)
        if len(global_cons) < 20:  # CIAlign can over-trim 5 short seqs
            global_cons = _ungap(max((s for _, s in cons_recs), key=len))
            _pp("  (CIAlign over-trimmed; using longest cluster consensus as global)")
    elif cons_recs:
        global_cons = cons_recs[0][1]
    else:
        sys.exit("ERROR: no cluster consensus; cannot build global consensus.")
    _pp(f"  Cleaned global consensus: {len(global_cons)} bp")

    _pp("Projecting copies onto the cleaned global consensus...")
    global_dist = np.full(len(df), np.nan)
    for cid, cons in cluster_cons.items():
        mapping = map_consensus(cons, global_cons, args.mafft_cmd, args.threads)
        mapped = [(p, gb) for p, gb in enumerate(mapping) if gb is not None]
        gstr = "".join(gb for _, gb in mapped)
        positions = [p for p, _ in mapped]
        for i, row in cluster_rows[cid].items():
            if row is None:
                continue
            cstr = "".join(row[p] if p < len(row) else "-" for p in positions)
            global_dist[i] = kimura2p(cstr, gstr, max_k2p=args.max_k2p)
    df["dist_global"] = global_dist

    out_cols = ["TE_ID", "__cluster", "dist_cluster", "dist_global"]
    df[out_cols].rename(columns={"__cluster": "cluster"}).to_csv(args.out, index=False)

    n = len(df)
    nc = int(np.isfinite(df["dist_cluster"]).sum())
    ng = int(np.isfinite(df["dist_global"]).sum())
    _pp("=" * 60)
    _pp(f"DONE -> {args.out}")
    _pp(f"  cleaned alignments + consensuses in: {args.aln_dir}/")
    _pp(f"  cluster-distance coverage: {nc:,}/{n:,} ({100*nc/n:.1f}%)")
    _pp(f"  global-distance  coverage: {ng:,}/{n:,} ({100*ng/n:.1f}%)")


if __name__ == "__main__":
    main()
