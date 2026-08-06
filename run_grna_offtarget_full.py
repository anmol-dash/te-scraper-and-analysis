#!/usr/bin/env python3
"""
run_grna_offtarget_full.py --- GAMECA EXHAUSTIVE genome-wide gRNA off-target search

This is the comprehensive version of the scan in run_grna_combos.py. That one
searches canonical NGG PAMs, mismatches only, on the primary chromosomes --- fast
enough for a laptop, but it cannot see three whole classes of off-target:

  1. BULGES. Cas9 tolerates a bulged base on either side of the duplex:
       - DNA bulge --- an unpaired base in the target (target is 1+ nt longer)
       - RNA bulge --- an unpaired base in the guide (target is 1+ nt shorter)
     A bulged site is a near-perfect duplex that a mismatch-only search scores
     as ~15 mismatches and throws away.
  2. NON-CANONICAL PAMs. SpCas9 also cuts at NAG and NGA, at reduced but real
     efficiency. An NAG site with 0 mismatches can out-cut an NGG site with 4.
  3. EVERYTHING OUTSIDE chr1-22,X,Y. Alt haplotypes, unplaced and unlocalized
     contigs are real sequence in the cells you transfect.

Every alignment is enumerated: for each guide, each PAM in the PAM set, each
strand, no-bulge plus every DNA/RNA bulge position and size, over every contig.
Nothing is seeded, sampled or heuristically pruned.

That costs roughly 40-120x the fast scan --- hours of CPU --- so the search
REFUSES to run outside a scheduler job unless you pass --force-local. Use
scripts/submit_grna_offtarget_full.sh. Run --estimate first to size the job.

Sites are written in the same schema run_grna_combos.py consumes, so:

    python run_grna_offtarget_full.py ... --out DIR      # comprehensive sites
    python run_grna_combos.py ... --out DIR --reuse-sites  # combination tables

re-derives every combination/coverage table on top of the comprehensive site
list without rescanning.

Reused GAMECA modules:
    run_grna_combos.py    : load_rmsk/load_refgene/annotate_sites, read_guides,
                            _pam_starts, _rc_pam, download_refgene, STD_CHROMS
    run_grna_offtarget.py : _rc, _PAM_TYPES, on_target_score

Usage:
    python run_grna_offtarget_full.py --guides guides.xlsx --genome hg38.fa \\
        --out ./ot_full --ot-mm 4 --bulge 1 --pam-set NGG,NAG,NGA --workers 16
    python run_grna_offtarget_full.py --guides g.xlsx --genome hg38.fa --estimate
    python run_grna_offtarget_full.py --self-test
"""

import argparse
import datetime
import os
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_grna_offtarget import _rc, on_target_score          # noqa: E402
from run_grna_combos import (                                 # noqa: E402
    STD_CHROMS, _pam_starts, _rc_pam, annotate_sites, download_refgene,
    load_refgene, load_rmsk, read_guides,
)

# SpCas9 cuts NGG well and NAG/NGA measurably; keep them labelled separately so
# a non-canonical hit is never silently ranked next to a canonical one.
CANONICAL_PAM = "NGG"
DEFAULT_PAMS = ("NGG", "NAG", "NGA")

_SCHED_VARS = ("LSB_JOBID", "SLURM_JOB_ID", "PBS_JOBID", "JOB_ID",
               "SGE_TASK_ID", "FLUX_JOB_ID")

_COMP = np.arange(256, dtype=np.uint8)
for _a, _b in zip(b"ACGTacgt", b"TGCAtgca"):
    _COMP[_a] = _b


def _pp(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# FASTA: byte-offset index so workers read only their own contig
# ═══════════════════════════════════════════════════════════════════════════

def fasta_offsets(path):
    """{contig: (byte_start, byte_end)} of the sequence body of each record."""
    idx, name, start, pos = {}, None, None, 0
    with open(path, "rb") as fh:
        for line in fh:
            if line.startswith(b">"):
                if name is not None:
                    idx[name] = (start, pos)
                name = line[1:].split()[0].decode()
                start = pos + len(line)
            pos += len(line)
    if name is not None:
        idx[name] = (start, pos)
    return idx


def read_contig(path, span):
    with open(path, "rb") as fh:
        fh.seek(span[0])
        raw = fh.read(span[1] - span[0])
    return raw.replace(b"\n", b"").replace(b"\r", b"").upper()


# ═══════════════════════════════════════════════════════════════════════════
# Alignment configurations: no-bulge + every DNA/RNA bulge
# ═══════════════════════════════════════════════════════════════════════════

def build_configs(glen, max_bulge, distal_only=False):
    """Every guide<->target alignment to test.

    Each config is (bulge_type, bulge_size, dna_len, segments) where segments is
    a list of (guide_start, guide_stop, dna_start): guide[gs:ge] pairs with
    dna[ds : ds+(ge-gs)]. Guide index 0 is the PAM-DISTAL end, glen-1 sits
    against the PAM.

      DNA bulge size b at p: target carries b unpaired bases, dna_len = glen+b
      RNA bulge size b at p: guide carries b unpaired bases, dna_len = glen-b
    """
    cfgs = [("none", 0, glen, [(0, glen, 0)])]
    hi = (glen // 2) if distal_only else (glen - 1)
    for b in range(1, max_bulge + 1):
        for p in range(1, hi + 1):
            # DNA bulge: guide[:p] -> dna[:p]; guide[p:] -> dna[p+b:]
            cfgs.append(("DNA", b, glen + b, [(0, p, 0), (p, glen, p + b)]))
            # RNA bulge: guide[:p] -> dna[:p]; guide[p+b:] -> dna[p:]
            if p + b < glen:
                cfgs.append(("RNA", b, glen - b, [(0, p, 0), (p + b, glen, p)]))
    return cfgs


def _apply_config(win, W, pat, cfg):
    """Mismatch count of one alignment against a window matrix."""
    _, _, dlen, segs = cfg
    base = W - dlen
    mm = None
    for gs, ge, ds in segs:
        if ge <= gs:
            continue
        c0 = base + ds
        sub = win[:, c0: c0 + (ge - gs)]
        cnt = np.count_nonzero(sub != pat[gs:ge], axis=1)
        mm = cnt if mm is None else mm + cnt
    return mm


# ═══════════════════════════════════════════════════════════════════════════
# The scan (one contig per worker)
# ═══════════════════════════════════════════════════════════════════════════

def scan_contig(job):
    """Scan one contig for every guide x PAM x strand x alignment."""
    (path, contig, span, guides, glen, max_mm, pams, max_bulge,
     distal_only, chunk, bulge_mm) = job
    raw = read_contig(path, span)
    a = np.frombuffer(raw, dtype=np.uint8)
    n = a.size
    W = glen + max_bulge
    if n < W + 8:
        return contig, []

    cfgs = build_configs(glen, max_bulge, distal_only)
    pats = [np.frombuffer(g.encode("ascii"), dtype=np.uint8) for g in guides]
    arW = np.arange(W, dtype=np.int64)
    best = {}          # (gi, strand, pam_start) -> (n_mm, btype, bsize, dlen, pam)

    for pam in pams:
        plen = len(pam)
        for strand in "+-":
            pattern = pam if strand == "+" else _rc_pam(pam)
            pos = _pam_starts(a, pattern)
            if pos.size == 0:
                continue
            if strand == "+":
                ws = pos - W                       # window ends at the PAM
                keep = ws >= 0
            else:
                ws = pos + plen                    # window starts after rc(PAM)
                keep = ws + W <= n
            pos, ws = pos[keep], ws[keep]
            if pos.size == 0:
                continue

            for off in range(0, ws.size, chunk):
                wsc, posc = ws[off: off + chunk], pos[off: off + chunk]
                win = a[wsc[:, None] + arW]
                if strand == "-":
                    # reverse-complement the window so the guide->DNA index map
                    # is identical to the plus strand
                    win = np.ascontiguousarray(_COMP[win][:, ::-1])
                for gi, pat in enumerate(pats):
                    for cfg in cfgs:
                        btype, bsize, dlen, _ = cfg
                        # A bulged alignment has far more freedom to match by
                        # chance (it drops a position and multiplies the number
                        # of alignments ~37x), so it gets its own, tighter
                        # mismatch budget. Without this the output is dominated
                        # by background: at <=4 mismatches a bulged alignment
                        # yields ~800 random hits per guide per 47 Mb.
                        lim = max_mm if btype == "none" else min(max_mm, bulge_mm)
                        mm = _apply_config(win, W, pat, cfg)
                        hits = np.flatnonzero(mm <= lim)
                        if hits.size == 0:
                            continue
                        for h in hits:
                            key = (gi, strand, int(posc[h]))
                            cand = (int(mm[h]), btype, bsize, dlen, pam)
                            prev = best.get(key)
                            # prefer no bulge, then fewer mismatches
                            if prev is None or (cand[0] + (cand[1] != "none"),) \
                                    < (prev[0] + (prev[1] != "none"),):
                                best[key] = cand

    rows = []
    for (gi, strand, pam_start), (mm, btype, bsize, dlen, pam) in best.items():
        plen = len(pam)
        if strand == "+":
            s0, s1 = pam_start - dlen, pam_start
            site = raw[s0:s1].decode()
            pam_seq = raw[pam_start: pam_start + plen].decode()
            cut = s1 - 3
        else:
            s0, s1 = pam_start + plen, pam_start + plen + dlen
            site = _rc(raw[s0:s1].decode())
            pam_seq = _rc(raw[pam_start: pam_start + plen].decode())
            cut = s0 + 3
        rows.append((guides[gi], contig, s0, s1, strand, mm, site, pam_seq, cut,
                     btype, bsize, pam))
    return contig, rows


SITE_COLS = ["guide", "chrom", "start", "end", "strand", "n_mm", "site", "pam",
             "cut", "bulge_type", "bulge_size", "pam_class"]


def scan_genome_full(genome, guides, max_mm, pams, max_bulge, contigs=None,
                     workers=1, distal_only=False, chunk=250_000, glen=20,
                     bulge_mm=2):
    offs = fasta_offsets(genome)
    todo = [(c, s) for c, s in offs.items() if not contigs or c in contigs]
    todo.sort(key=lambda t: -(t[1][1] - t[1][0]))          # big contigs first
    _pp(f"  {len(todo)} contigs, {sum(s[1]-s[0] for _, s in todo)/1e9:.2f} Gb (raw)")
    jobs = [(genome, c, s, guides, glen, max_mm, pams, max_bulge, distal_only,
             chunk, bulge_mm) for c, s in todo]

    rows, done = [], 0
    t0 = time.time()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(scan_contig, j): j[1] for j in jobs}
            for fut in as_completed(futs):
                contig, r = fut.result()
                rows.extend(r)
                done += 1
                _pp(f"    [{done}/{len(jobs)}] {contig}: {len(r):,} sites "
                    f"({time.time()-t0:.0f}s elapsed)")
    else:
        for j in jobs:
            contig, r = scan_contig(j)
            rows.extend(r)
            done += 1
            _pp(f"    [{done}/{len(jobs)}] {contig}: {len(r):,} sites "
                f"({time.time()-t0:.0f}s elapsed)")
    return pd.DataFrame(rows, columns=SITE_COLS)


# ═══════════════════════════════════════════════════════════════════════════
# Risk tiers --- explicit rules, not a published scoring matrix
# ═══════════════════════════════════════════════════════════════════════════

def risk_tier(row):
    """Ordering heuristic over hard evidence (PAM class, mismatches, bulge).

    Deliberately NOT a CFD/MIT score: no published scoring matrix is embedded
    here, so nothing is implied beyond what was actually measured.
    """
    canon = row["pam_class"] == CANONICAL_PAM
    bulged = row["bulge_type"] != "none"
    mm = row["n_mm"]
    if canon and not bulged and mm <= 2:
        return "1_cut_likely"
    if canon and not bulged:
        return "2_possible"
    if canon and bulged and mm <= 2:
        return "2_possible"
    if not canon and not bulged and mm <= 1:
        return "2_possible"
    return "3_unlikely"


def seed_mismatches(row, seed=12):
    """Mismatches falling in the PAM-proximal seed --- 0 means an intact seed."""
    g, s = str(row["guide"]), str(row["site"])
    if row["bulge_type"] != "none" or len(g) != len(s):
        return -1                      # undefined under a bulged alignment
    return sum(1 for a, b in zip(g[-seed:], s[-seed:]) if a != b)


# ═══════════════════════════════════════════════════════════════════════════
# Guard: this is a cluster job
# ═══════════════════════════════════════════════════════════════════════════

def on_scheduler():
    return any(os.environ.get(v) for v in _SCHED_VARS)


def require_cluster(force, est_hours):
    if on_scheduler():
        _pp(f"  scheduler job detected "
            f"({', '.join(v for v in _SCHED_VARS if os.environ.get(v))})")
        return
    if force:
        _pp("  WARNING: --force-local set; running the comprehensive scan off-cluster")
        return
    print(f"""
REFUSING to run the comprehensive scan outside a scheduler job.

  Estimated cost: {est_hours:.1f} core-hours, plus a ~3.3 GB genome resident.
  This is not a laptop workload --- it enumerates every PAM, every strand,
  every bulge position and every contig.

  Submit it instead:
      bash scripts/submit_grna_offtarget_full.sh

  Size the job first:
      python run_grna_offtarget_full.py --guides ... --genome ... --estimate

  If you really mean to run it here, re-run with --force-local.
""".rstrip(), file=sys.stderr)
    sys.exit(3)


def calibrate(glen=20, max_bulge=1, distal_only=False, n=250_000):
    """Measured throughput in position-alignments/s, through the real code path.

    Calibrating on a single contiguous no-bulge compare flatters the estimate:
    bulge alignments split into two shorter slices and the minus strand pays a
    complement+reverse per chunk. Time the actual mix instead.
    """
    W = glen + max_bulge
    cfgs = build_configs(glen, max_bulge, distal_only)
    rng = np.random.default_rng(0)
    win = rng.integers(65, 85, size=(n, W), dtype=np.uint8)
    pat = rng.integers(65, 85, size=glen, dtype=np.uint8)
    t0 = time.time()
    winm = np.ascontiguousarray(_COMP[win][:, ::-1])       # minus-strand cost
    for cfg in cfgs:
        _apply_config(win, W, pat, cfg)
        _apply_config(winm, W, pat, cfg)
    dt = max(1e-6, time.time() - t0)
    return (n * len(cfgs) * 2) / dt


def expected_by_chance(bp, pams, glen, max_bulge, distal_only, max_mm, bulge_mm):
    """Sites per guide expected from random sequence alone (uniform-base null).

    Pure combinatorics, no fitted parameters: it says how much of the output is
    background, which matters because widening the search space (more PAMs, more
    alignments, more mismatches) inflates the hit count on its own.
    """
    from math import comb
    pos = bp * len(pams) * 2 / 16.0

    def p(L, mm):
        return sum(comb(L, k) * 3 ** k for k in range(mm + 1)) / 4 ** L

    nb = pos * p(glen, max_mm)
    bl = 0.0
    for (bt, bs, dlen, segs) in build_configs(glen, max_bulge, distal_only):
        if bt == "none":
            continue
        L = sum(ge - gs for gs, ge, _ in segs)
        bl += pos * p(L, min(bulge_mm, max_mm))
    return nb, bl


def estimate(genome, n_guides, pams, max_bulge, glen, contigs, distal_only):
    offs = fasta_offsets(genome)
    bp = sum(e - s for c, (s, e) in offs.items() if not contigs or c in contigs)
    n_cfg = len(build_configs(glen, max_bulge, distal_only))
    # each PAM pattern constrains 2 fixed bases -> ~1/16 of positions, x2 strands
    pos = bp * len(pams) * 2 / 16.0
    rate = calibrate(glen, max_bulge, distal_only)
    core_s = pos * n_cfg * n_guides / rate
    return {"bp": bp, "n_cfg": n_cfg, "positions": pos, "rate": rate,
            "core_hours": core_s / 3600.0}


# ═══════════════════════════════════════════════════════════════════════════
# Self-test: plant bulged / non-canonical / high-mismatch sites and find them
# ═══════════════════════════════════════════════════════════════════════════

def self_test():
    import random
    import tempfile
    rng = random.Random(17)
    ok = True

    def chk(label, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {label}: got {got}, expected {want}")

    print("=" * 70)
    print("SELF-TEST --- comprehensive scan (bulges, non-canonical PAMs, strands)")
    print("=" * 70)

    g = "GACCTGAGTCGATTACGCAT"          # non-palindromic
    glen = len(g)
    F = "TTT"                            # left flank: never creates a CCN
    bg = "".join(rng.choice("ACGT") for _ in range(3000))

    # segment maths, independent of any scanning
    cfgs = build_configs(glen, 1)
    chk("config count = 1 + DNA(19) + RNA(18)", len(cfgs), 1 + 19 + 18)
    dna_p5 = next(c for c in cfgs if c[0] == "DNA" and c[3][0][1] == 5)
    chk("DNA bulge target is 1 nt longer", dna_p5[2], glen + 1)
    rna_p5 = next(c for c in cfgs if c[0] == "RNA" and c[3][0][1] == 5)
    chk("RNA bulge target is 1 nt shorter", rna_p5[2], glen - 1)

    seq, pos = "", {}

    def add(s, tag=None):
        nonlocal seq
        if tag:
            pos[tag] = len(seq)
        seq += s

    # 1. perfect canonical site
    add(bg[:400] + F)
    add(g + "CGG", "perfect")
    # 2. DNA bulge: an extra base inserted into the target at position 5
    dna_bulge = g[:5] + "A" + g[5:]
    add(bg[400:800] + F)
    add(dna_bulge + "AGG", "dna_bulge")
    # 3. RNA bulge: target is missing guide base 5
    rna_bulge = g[:5] + g[6:]
    add(bg[800:1200] + F)
    add(rna_bulge + "TGG", "rna_bulge")
    # 4. non-canonical NAG PAM, perfect protospacer
    add(bg[1200:1600] + F)
    add(g + "AAG", "nag")
    # 5. 4 mismatches, canonical PAM
    mm4 = "".join(("T" if c != "T" else "A") if i in (0, 2, 4, 6) else c
                  for i, c in enumerate(g))
    add(bg[1600:2000] + F)
    add(mm4 + "GGG", "mm4")
    # 6. minus strand, perfect
    add(bg[2000:2400] + F + "CCT")
    add(_rc(g), "minus")
    # 7. no PAM at all -> must never appear
    add(bg[2400:2800] + F)
    add(g + "AAT", "nopam")
    add(bg[2800:3000])

    with tempfile.TemporaryDirectory() as td:
        fa = Path(td) / "mini.fa"
        fa.write_text(">chrT\n" + "\n".join(seq[i:i + 60]
                                            for i in range(0, len(seq), 60)) + "\n")

        # canonical-only, no bulges: the baseline search
        base = scan_genome_full(str(fa), [g], 4, ["NGG"], 0, {"chrT"}, workers=1)
        found = {int(s) for s in base["start"]}
        chk("no-bulge/NGG search finds the perfect site",
            pos["perfect"] in found, True)
        chk("no-bulge search MISSES the DNA-bulge site",
            pos["dna_bulge"] in found, False)
        chk("no-bulge search MISSES the RNA-bulge site",
            pos["rna_bulge"] in found, False)
        chk("NGG-only search MISSES the NAG site", pos["nag"] in found, False)

        # comprehensive
        full = scan_genome_full(str(fa), [g], 4, list(DEFAULT_PAMS), 1,
                                {"chrT"}, workers=1)
        f2 = {int(s): r for s, r in zip(full["start"], full.itertuples(index=False))}
        chk("comprehensive finds the perfect site", pos["perfect"] in f2, True)
        chk("comprehensive finds the DNA-bulge site", pos["dna_bulge"] in f2, True)
        chk("comprehensive finds the RNA-bulge site", pos["rna_bulge"] in f2, True)
        chk("comprehensive finds the NAG site", pos["nag"] in f2, True)
        chk("comprehensive finds the 4-mismatch site", pos["mm4"] in f2, True)
        chk("comprehensive finds the minus-strand site", pos["minus"] in f2, True)
        chk("PAM-less site still absent", pos["nopam"] in f2, False)

        r = f2[pos["dna_bulge"]]
        chk("DNA-bulge call typed correctly", r.bulge_type, "DNA")
        chk("DNA-bulge site spans glen+1", int(r.end) - int(r.start), glen + 1)
        chk("DNA-bulge alignment has 0 mismatches", int(r.n_mm), 0)
        r = f2[pos["rna_bulge"]]
        chk("RNA-bulge call typed correctly", r.bulge_type, "RNA")
        chk("RNA-bulge site spans glen-1", int(r.end) - int(r.start), glen - 1)
        chk("RNA-bulge alignment has 0 mismatches", int(r.n_mm), 0)
        # a bulge carrying 3 extra mismatches must be gated by --bulge-mm
        loose = scan_genome_full(str(fa), [g], 4, list(DEFAULT_PAMS), 1,
                                 {"chrT"}, workers=1, bulge_mm=4)
        tight = scan_genome_full(str(fa), [g], 4, list(DEFAULT_PAMS), 1,
                                 {"chrT"}, workers=1, bulge_mm=2)
        lb = (loose.bulge_type != "none").sum()
        tb = (tight.bulge_type != "none").sum()
        chk("bulge-mm=2 keeps the clean bulged sites", tb >= 2, True)
        chk("bulge-mm=2 admits no more than bulge-mm=4", tb <= lb, True)
        chk("bulge-mm=2 drops high-mismatch bulged noise",
            int((tight.bulge_type != "none").sum()) ==
            int(((loose.bulge_type != "none") & (loose.n_mm <= 2)).sum()), True)

        r = f2[pos["nag"]]
        chk("NAG site labelled non-canonical", r.pam_class, "NAG")
        chk("NAG site protospacer is perfect", int(r.n_mm), 0)
        r = f2[pos["minus"]]
        chk("minus-strand site reads as the guide", r.site, g)
        chk("minus-strand site has 0 mismatches", int(r.n_mm), 0)
        chk("4-mismatch site counted as 4", int(f2[pos["mm4"]].n_mm), 4)

        tiers = {pos["perfect"]: "1_cut_likely", pos["mm4"]: "2_possible",
                 pos["nag"]: "2_possible", pos["dna_bulge"]: "2_possible"}
        for p, want in tiers.items():
            row = full[full.start == p].iloc[0]
            chk(f"risk tier at {p}", risk_tier(row), want)

    print("=" * 70)
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    print("=" * 70)
    return 0 if ok else 1


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Exhaustive genome-wide gRNA off-target search "
                    "(bulges + non-canonical PAMs + all contigs)",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--guides", help=".xlsx/.csv of guides (all sheets scanned)")
    p.add_argument("--genome", default="", help="Genome FASTA")
    p.add_argument("--out", default="./ot_full")
    p.add_argument("--assembly", default="hg38")
    p.add_argument("--rmsk-dir", default=str(Path.home() / "te_analysis" / "rmsk"))
    p.add_argument("--rmsk", default="")
    p.add_argument("--refgene", default="")
    p.add_argument("--ot-mm", type=int, default=4,
                   help="Mismatches allowed at any position (default 4)")
    p.add_argument("--bulge", type=int, default=1,
                   help="Max DNA/RNA bulge size, 0 disables (default 1)")
    p.add_argument("--bulge-mm", type=int, default=2,
                   help="Mismatches allowed IN ADDITION to a bulge (default 2). "
                        "Kept below --ot-mm on purpose: a bulged alignment drops a "
                        "position and multiplies alignments ~37x, so at 4 mismatches "
                        "it returns ~800 random sites per guide per 47 Mb")
    p.add_argument("--pam-set", default=",".join(DEFAULT_PAMS),
                   help="Comma-separated PAMs (default NGG,NAG,NGA)")
    p.add_argument("--grna-len", type=int, default=20)
    p.add_argument("--bulge-distal-only", action="store_true",
                   help="Only bulge the PAM-distal half (halves the cost)")
    p.add_argument("--primary-only", action="store_true",
                   help="chr1-22,X,Y,M only (default: every contig)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--chunk", type=int, default=250_000,
                   help="PAM positions scored per batch (cache-sized; ~8%% faster "
                        "than 1M and a smaller window matrix per worker)")
    p.add_argument("--estimate", action="store_true", help="Size the job and exit")
    p.add_argument("--force-local", action="store_true",
                   help="Run off-cluster anyway (slow; see --estimate)")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        sys.exit(self_test())
    if not args.guides or not args.genome:
        sys.exit("ERROR: --guides and --genome are required (or --self-test)")
    if not Path(args.genome).exists():
        sys.exit(f"ERROR: genome not found: {args.genome}")

    grows = read_guides(args.guides)
    guides = sorted({r["guide"] for r in grows})
    fam_of = {}
    for r in grows:
        fam_of.setdefault(r["guide"], r["family"])
    pams = [p.strip().upper() for p in args.pam_set.split(",") if p.strip()]
    contigs = STD_CHROMS if args.primary_only else None

    est = estimate(args.genome, len(guides), pams, args.bulge, args.grna_len,
                   contigs, args.bulge_distal_only)
    _pp("=" * 66)
    _pp("GAMECA comprehensive off-target search")
    _pp("=" * 66)
    _pp(f"  guides:      {len(guides)} from {Path(args.guides).name}")
    _pp(f"  PAMs:        {', '.join(pams)}  (canonical = {CANONICAL_PAM})")
    _pp(f"  mismatches:  <= {args.ot_mm}, at any position")
    _pp(f"  bulges:      up to {args.bulge} nt, DNA and RNA, "
        f"<= {args.bulge_mm} extra mismatches"
        + (" (PAM-distal half only)" if args.bulge_distal_only else "")
        if args.bulge else "  bulges:      disabled")
    _pp(f"  contigs:     {'chr1-22,X,Y,M' if args.primary_only else 'ALL'} "
        f"({est['bp']/1e9:.2f} Gb)")
    _pp(f"  alignments:  {est['n_cfg']} per guide per PAM per strand")
    _pp(f"  candidates:  {est['positions']/1e9:.2f} G PAM positions")
    _pp(f"  estimate:    {est['core_hours']:.1f} core-hours "
        f"(~{est['core_hours']/max(1,args.workers):.1f} h on {args.workers} workers)")
    if args.estimate:
        _pp("  --estimate given; not scanning.")
        return

    require_cluster(args.force_local, est["core_hours"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sites = scan_genome_full(args.genome, guides, args.ot_mm, pams, args.bulge,
                             contigs, args.workers, args.bulge_distal_only,
                             args.chunk, args.grna_len, args.bulge_mm)
    _pp(f"  {len(sites):,} sites in {(time.time()-t0)/60:.1f} min")
    if sites.empty:
        _pp("  no sites found"); return

    # annotation
    rmsk_path = args.rmsk or None
    if not rmsk_path:
        try:
            from te_prep import get_rmsk_path
            rmsk_path = get_rmsk_path(args.assembly, args.rmsk_dir)
        except Exception as e:
            _pp(f"  WARNING: rmsk unavailable ({e})")
            rmsk_path = None
    rmsk_ix = load_rmsk(rmsk_path, contigs) if rmsk_path and Path(rmsk_path).exists() else None
    rg = args.refgene or download_refgene(args.assembly, args.rmsk_dir)
    gene_ix = load_refgene(rg, contigs) if rg and Path(rg).exists() else None
    _pp("  Annotating (repeats + genes)...")
    sites = annotate_sites(sites, rmsk_ix, gene_ix)

    sites["family"] = sites["guide"].map(fam_of)
    sites["risk_tier"] = sites.apply(risk_tier, axis=1)
    sites["seed_mm"] = sites.apply(seed_mismatches, axis=1)
    sites["primary_contig"] = sites["chrom"].isin(STD_CHROMS)
    sites = sites.sort_values(["guide", "risk_tier", "n_mm"])
    sites.to_csv(out / "offtarget_sites_all.csv", index=False)
    _pp(f"  Wrote {out/'offtarget_sites_all.csv'}")

    # per-guide summary + report
    target_fams = set()
    if rmsk_ix is not None:
        fams = {f.lower() for f in fam_of.values()}
        for (nm, cl, fm) in rmsk_ix.payload:
            if nm.lower() in fams:
                target_fams.add(fm)

    rep = ["=" * 78, "GAMECA --- comprehensive off-target search",
           f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}", "=" * 78, "",
           f"Search space: PAMs {'/'.join(pams)}; mismatches <= {args.ot_mm} at any "
           f"position; bulges up to {args.bulge} nt (DNA and RNA); "
           f"{'primary chromosomes' if args.primary_only else 'ALL contigs'}; "
           f"both strands; {est['n_cfg']} alignments per guide/PAM/strand.",
           f"On-family = repFamily {'/'.join(sorted(target_fams)) or '(rmsk unavailable)'}"
           " --- related TEs, expected and acceptable.",
           "Risk tiers are rule-based (PAM class + mismatches + bulge). No published",
           "scoring matrix (CFD/MIT) is embedded, so no cutting frequency is implied.",
           ""]
    summ = []
    for g in guides:
        sub = sites[sites.guide == g]
        onf = sub.repFamily.isin(target_fams) & (sub.repFamily != "")
        off = sub[~onf]
        row = {
            "guide": g, "family": fam_of.get(g, ""),
            "on_target_score": on_target_score(g),
            "sites_total": len(sub), "sites_onfamily": int(onf.sum()),
            "sites_offfamily": len(off),
            "off_tier1_cut_likely": int((off.risk_tier == "1_cut_likely").sum()),
            "off_perfect_canonical": int(((off.n_mm == 0) & (off.bulge_type == "none")
                                          & (off.pam_class == CANONICAL_PAM)).sum()),
            "off_bulged": int((off.bulge_type != "none").sum()),
            "off_noncanonical_pam": int((off.pam_class != CANONICAL_PAM).sum()),
            "off_nonrepeat": int((off.repName == "").sum()),
            "off_coding_exon": int(off.hits_coding_exon.sum()),
            "off_alt_contig": int((~off.primary_contig).sum()),
            "off_exon_genes": ";".join(sorted(
                {x for s in off[off.hits_coding_exon]["genes"] if s
                 for x in str(s).split(",") if x})),
        }
        summ.append(row)
    summ = pd.DataFrame(summ).sort_values("off_tier1_cut_likely", ascending=False)
    summ.to_csv(out / "offtarget_by_guide_full.csv", index=False)

    rep.append("-- per guide: off-family burden " + "-" * 44)
    rep.append(f"{'guide':<22} {'fam':<10} {'all':>7} {'onFam':>7} {'offFam':>7} "
               f"{'tier1':>6} {'0mm':>5} {'bulge':>6} {'nonPAM':>7} {'exon':>5} {'alt':>5}")
    for _, r in summ.iterrows():
        rep.append(f"{r['guide']:<22} {str(r['family'])[:10]:<10} {r['sites_total']:>7,} "
                   f"{r['sites_onfamily']:>7,} {r['sites_offfamily']:>7,} "
                   f"{r['off_tier1_cut_likely']:>6} {r['off_perfect_canonical']:>5} "
                   f"{r['off_bulged']:>6} {r['off_noncanonical_pam']:>7} "
                   f"{r['off_coding_exon']:>5} {r['off_alt_contig']:>5}")
    rep += ["  tier1 = canonical PAM, no bulge, <=2 mismatches (the ones most likely",
            "  to actually cut); 0mm = perfect canonical off-family matches;",
            "  bulge/nonPAM = sites only a comprehensive search can see;",
            "  exon = coding-exon hits outside the family; alt = on non-primary contigs.", ""]

    # what only the comprehensive search found, against the random background
    onf_all = sites.repFamily.isin(target_fams) & (sites.repFamily != "")
    extra_mask = ((sites.bulge_type != "none") | (sites.pam_class != CANONICAL_PAM)
                  | (~sites.primary_contig))
    extra = sites[extra_mask]
    real = sites[extra_mask & sites.risk_tier.isin(["1_cut_likely", "2_possible"])]
    exp_nb, exp_bl = expected_by_chance(est["bp"], pams, args.grna_len, args.bulge,
                                        args.bulge_distal_only, args.ot_mm,
                                        args.bulge_mm)
    rep.append("-- sites a canonical mismatch-only scan would have MISSED " + "-" * 19)
    rep.append(f"   {len(extra):,} of {len(sites):,} total "
               f"({100.0*len(extra)/max(1,len(sites)):.1f}%), of which "
               f"{len(real):,} are tier 1 or 2")
    for label, mask in (("bulged alignments", sites.bulge_type != "none"),
                        ("non-canonical PAM", sites.pam_class != CANONICAL_PAM),
                        ("non-primary contig", ~sites.primary_contig)):
        m = sites[mask]
        rep.append(f"   {len(m):>9,}  {label}   "
                   f"(off-family: {len(m[~onf_all.reindex(m.index, fill_value=False)]):,})")
    rep += ["",
            "   Background check (uniform-base null, exact combinatorics): random",
            "   sequence alone is expected to yield about",
            f"     {exp_nb:,.0f} unbulged and {exp_bl:,.0f} bulged sites PER GUIDE "
            f"over {est['bp']/1e6:,.0f} Mb.",
            "   Counts near those numbers are background, not off-targets. Widening",
            "   the search (more PAMs, more alignments, more mismatches) inflates the",
            "   raw hit count by itself --- rank by risk tier, not by total sites.",
            f"   Bulged alignments are held to <= {args.bulge_mm} mismatches for this",
            f"   reason (unbulged: <= {args.ot_mm}).", ""]

    worst = sites[sites.hits_coding_exon & ~onf_all].sort_values(
        ["risk_tier", "n_mm"])
    rep.append("-- coding-exon hits outside the target family " + "-" * 30)
    if worst.empty:
        rep.append("   none")
    else:
        rep.append(f"   {len(worst)} site(s); highest-risk first")
        for r in worst.head(40).itertuples(index=False):
            b = f"{r.bulge_type}{r.bulge_size}" if r.bulge_type != "none" else "-"
            rep.append(f"   [{r.risk_tier}] {r.chrom}:{r.start+1}-{r.end} {r.strand} "
                       f"{r.n_mm}mm PAM={r.pam}({r.pam_class}) bulge={b} "
                       f"{r.gene_context} {r.genes} [{r.repName or 'non-repeat'}] {r.guide}")
    rep.append("")
    txt = "\n".join(rep)
    (out / "offtarget_full_report.txt").write_text(txt)
    _pp(f"  Wrote {out/'offtarget_full_report.txt'}")
    print("\n" + txt)
    _pp("Next: python run_grna_combos.py --guides ... --out "
        f"{out} --reuse-sites   (combination tables on these sites)")


if __name__ == "__main__":
    main()
