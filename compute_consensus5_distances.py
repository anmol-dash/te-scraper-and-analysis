#!/usr/bin/env python3
"""
compute_consensus5_distances.py --- MAFFT consensus distances for ALL copies,
grouped by the 5 UMAP/HDBSCAN clusters in clusters5.csv.

For every copy it computes:
  dist_cluster : Kimura-2P distance to its OWN cluster's consensus (1 of 5)
  dist_global  : Kimura-2P distance to the family-wide (global) consensus

Method (one MAFFT pass per cluster, never a pseudo-alignment):
  per cluster -> longest copy is the seed; every copy is projected onto it with
  `mafft --addfragments --keeplength`; the consensus is the per-column majority.
  The global consensus is the majority over the MAFFT alignment of the 5 cluster
  consensuses; each copy's global distance is read through a cheap
  consensus->consensus position map (no 12k-copy family-wide MAFFT).

Unlike the per-subfamily script, NOTHING is dropped for saturation: --max-k2p
defaults high so every alignable copy keeps a real distance. Copies MAFFT cannot
place at all are reported in the coverage summary and left blank.

Requires: numpy, pandas, and MAFFT on PATH. No matplotlib / sklearn / umap.

Usage:
    python compute_consensus5_distances.py \
        --input clusters5.csv \
        --out consensus_distance_5clust.csv \
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


# ── MAFFT helpers ─────────────────────────────────────────────────────────────
class AlignmentError(RuntimeError):
    pass


def _mafft_exe(cmd):
    exe = shutil.which(cmd) or shutil.which("mafft")
    if not exe:
        raise AlignmentError(f"mafft not found (looked for {cmd!r} and 'mafft').")
    return exe


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


def run_mafft(records, cmd, threads):
    exe = _mafft_exe(cmd)
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


def _consensus_of(aln):
    if not aln:
        return ""
    L = max(len(s) for _, s in aln)
    cols = []
    for i in range(L):
        counts = {}
        for _, s in aln:
            if i < len(s):
                b = s[i].upper()
                if b in "ACGT":
                    counts[b] = counts.get(b, 0) + 1
        cols.append(max(counts, key=counts.get) if counts else "-")
    return "".join(cols).replace("-", "")


def align_cluster_to_consensus(seqs, cmd, threads):
    """Project every sequence onto a cluster consensus in ONE MAFFT pass."""
    exe = _mafft_exe(cmd)
    clean = [(i, s.upper().replace("-", "")) for i, s in enumerate(seqs)]
    clean = [(i, s) for i, s in clean if s]
    if not clean:
        raise AlignmentError("no non-empty sequences in cluster.")
    seed_i, seed_seq = max(clean, key=lambda t: len(t[1]))
    with tempfile.TemporaryDirectory() as td:
        ref = Path(td) / "ref.fa"
        ref.write_text(f">SEED\n{seed_seq}\n")
        frags = Path(td) / "frags.fa"
        frags.write_text("".join(f">c{i}\n{s}\n" for i, s in clean))
        res = subprocess.run(
            [exe, "--addfragments", str(frags), "--keeplength",
             "--thread", str(threads), "--quiet", str(ref)],
            capture_output=True, text=True)
        if res.returncode != 0:
            raise AlignmentError(f"mafft --addfragments failed ({res.returncode}): "
                                 f"{res.stderr.strip()[:300]}")
        aligned = dict(_read_fasta_text(res.stdout))
    rows = {i: aligned[f"c{i}"] for i, _ in clean if f"c{i}" in aligned}
    if not rows:
        raise AlignmentError("no copy rows in mafft --addfragments output.")
    L = max(len(r) for r in rows.values())
    cons = []
    rlist = list(rows.values())
    for col in range(L):
        counts = {}
        for r in rlist:
            if col < len(r):
                b = r[col].upper()
                if b in "ACGT":
                    counts[b] = counts.get(b, 0) + 1
        cons.append(max(counts, key=counts.get) if counts else "N")
    return "".join(cons), [rows.get(i) for i in range(len(seqs))]


def map_cluster_to_global(cluster_cons, global_cons, cmd, threads):
    aln = dict(run_mafft([("G", global_cons), ("C", cluster_cons)], cmd, threads))
    aG, aC = aln.get("G", ""), aln.get("C", "")
    mapping = [None] * len(cluster_cons)
    c_pos = 0
    for col in range(min(len(aG), len(aC))):
        if aC[col] != "-":
            if c_pos < len(mapping):
                gb = aG[col].upper()
                mapping[c_pos] = gb if gb in "ACGT" else None
            c_pos += 1
    return mapping


def global_distance_for_cluster(cons_row, member_rows, global_cons, cmd,
                                threads, max_k2p):
    mapping = map_cluster_to_global(cons_row, global_cons, cmd, threads)
    mapped = [(p, gb) for p, gb in enumerate(mapping) if gb is not None]
    gstr = "".join(gb for _, gb in mapped)
    positions = [p for p, _ in mapped]
    out = {}
    for i, row in member_rows.items():
        if row is None:
            out[i] = float("nan")
            continue
        cstr = "".join(row[p] if p < len(row) else "-" for p in positions)
        out[i] = kimura2p(cstr, gstr, max_k2p=max_k2p)
    return out


# ── main ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="clusters5.csv",
                   help="CSV with TE_ID, Seq, Cluster columns.")
    p.add_argument("--out", default="consensus_distance_5clust.csv")
    p.add_argument("--cluster-col", default="Cluster")
    p.add_argument("--max-k2p", type=float, default=5.0,
                   help="Drop distances above this (default 5.0 = keep everything).")
    p.add_argument("--global-sample", type=int, default=400,
                   help="Max cluster consensuses MAFFT-aligned for the global consensus.")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--mafft-cmd", default="mafft")
    return p.parse_args()


def main():
    args = parse_args()
    _pp("=" * 60)
    _pp("5-cluster consensus distances (MAFFT)")
    _pp(f"  input={args.input}  out={args.out}  threads={args.threads}")
    _pp("=" * 60)

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

    _pp("Aligning each cluster to its consensus (MAFFT --addfragments)...")
    for cid, sub in df.groupby("__cluster"):
        idx = sub.index.to_list()
        try:
            cons_row, rows = align_cluster_to_consensus(
                [seqs[i] for i in idx], args.mafft_cmd, args.threads)
        except AlignmentError as e:
            _pp(f"  cluster {cid}: SKIP — {e}")
            continue
        cluster_cons[cid] = cons_row
        cluster_rows[cid] = {idx[j]: rows[j] for j in range(len(idx))}
        good = 0
        for j, i in enumerate(idx):
            if rows[j] is not None:
                k = kimura2p(rows[j], cons_row, max_k2p=args.max_k2p)
                cluster_dist[i] = k
                good += int(np.isfinite(k))
        _pp(f"  cluster {cid}: {len(idx)} copies, cons {len(cons_row)} bp, "
            f"{good} distances")
    df["dist_cluster"] = cluster_dist

    _pp("Building global consensus from the 5 cluster consensuses...")
    cons_recs = [(str(cid), s.replace("-", "")) for cid, s in cluster_cons.items() if s]
    if len(cons_recs) >= 2:
        if len(cons_recs) > args.global_sample:
            sidx = np.linspace(0, len(cons_recs) - 1, args.global_sample).astype(int)
            cons_recs = [cons_recs[i] for i in sorted(set(sidx))]
        global_cons = _consensus_of(run_mafft(cons_recs, args.mafft_cmd, args.threads))
    elif cons_recs:
        global_cons = cons_recs[0][1]
    else:
        sys.exit("ERROR: no cluster consensus; cannot build global consensus.")
    _pp(f"  Global consensus: {len(global_cons)} bp")

    _pp("Projecting copies onto the global consensus...")
    global_dist = np.full(len(df), np.nan)
    for cid, cons_row in cluster_cons.items():
        gd = global_distance_for_cluster(cons_row, cluster_rows[cid], global_cons,
                                         args.mafft_cmd, args.threads, args.max_k2p)
        for i, k in gd.items():
            global_dist[i] = k
    df["dist_global"] = global_dist

    out_cols = ["TE_ID", "__cluster", "dist_cluster", "dist_global"]
    df[out_cols].rename(columns={"__cluster": "cluster"}).to_csv(args.out, index=False)

    n = len(df)
    nc = int(np.isfinite(df["dist_cluster"]).sum())
    ng = int(np.isfinite(df["dist_global"]).sum())
    _pp("=" * 60)
    _pp(f"DONE -> {args.out}")
    _pp(f"  cluster-distance coverage: {nc:,}/{n:,} ({100*nc/n:.1f}%)")
    _pp(f"  global-distance  coverage: {ng:,}/{n:,} ({100*ng/n:.1f}%)")


if __name__ == "__main__":
    main()
