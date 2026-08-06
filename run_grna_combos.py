#!/usr/bin/env python3
"""
run_grna_combos.py --- GAMECA guide-COMBINATION coverage + genome-wide off-target/gene safety

Answers two questions about a set of guides that have already been chosen
(e.g. the 10 guides ordered per family), reusing the GAMECA modules:

  1. COMBINATIONS --- which combination of the ordered guides covers the most
     copies of the family?  Coverage of a guide SET = union of the copies each
     member can actually cut: a PAM must be present at the copy (this is what
     makes a site cuttable) and mismatches are allowed only in the PAM-distal
     positions, using the same seed-protected model as run_grna_offtarget.py
     (_variants()).  EVERY subset of the input guides is enumerated, so the
     "best pair" and the "smallest set reaching X% coverage" are exact answers,
     not greedy approximations.

  2. OFF-TARGET / GENE SAFETY --- every guide is scanned against the whole
     genome EXHAUSTIVELY (mismatches allowed at any position, PAM required,
     both strands, no seed heuristic) and every site found is annotated with
       * RepeatMasker  -> is this just another copy of a related TE?  (fine)
       * RefSeq genes  -> is it in a CDS / UTR / intron / promoter?    (not fine)
     Hitting LTR5A instead of LTR5_Hs is expected; cutting a protein-coding
     exon is the thing to avoid, so that is what gets flagged.

Both families are scanned in ONE pass over the genome, so adding families is
nearly free.

Reused GAMECA modules:
    run_grna_offtarget.py : extract_pam_grnas(), _variants(), _rc(),
                            on_target_score(), _PAM_TYPES
    te_prep.py            : download_rmsk(), get_rmsk_path(), rmsk column layout

Usage:
    python run_grna_combos.py \
        --guides "HERVK-int_LTR5_Hs_guide candidates.xlsx" \
        --seqs-dir ~/hervk_reagents \
        --genome ~/hervk_ccle/ref/hg38.fa \
        --out ~/hervk_reagents/combos \
        --target-cov 0.5

    # coverage/combination half only (no genome needed):
    python run_grna_combos.py --guides guides.xlsx --seqs-dir . --skip-offtarget

    # correctness controls on synthetic data (no genome/annotation needed):
    python run_grna_combos.py --self-test
"""

import argparse
import datetime
import gzip
import itertools
import os
import subprocess
import shlex
import sys
import warnings
from array import array
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ── reuse the existing GAMECA gRNA model ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_grna_offtarget import (  # noqa: E402
    _PAM_TYPES, _IUPAC, _rc, extract_pam_grnas, _variants, on_target_score,
)

# PAM geometry: True = PAM is 3' of the protospacer (Cas9-like)
_PAM_3PRIME = {"SpCas9": True, "SaCas9": True, "SpRY": True, "Cas12a": False}

_IUPAC_COMP = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N", "R": "Y",
               "Y": "R", "S": "S", "W": "W", "K": "M", "M": "K", "B": "V",
               "V": "B", "D": "H", "H": "D"}

STD_CHROMS = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY", "chrM"}

REFGENE_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/{build}/database/refGene.txt.gz"

# refGene.txt column layout (UCSC, with leading bin column)
RG_NAME, RG_CHROM, RG_STRAND, RG_TXS, RG_TXE = 1, 2, 3, 4, 5
RG_CDSS, RG_CDSE, RG_NEX, RG_EXS, RG_EXE, RG_NAME2 = 6, 7, 8, 9, 10, 12

# rmsk.txt column layout (mirrors te_prep.py)
RM_SW, RM_MILLIDIV, RM_CHROM, RM_START, RM_END, RM_STRAND = 1, 2, 5, 6, 7, 9
RM_NAME, RM_CLASS, RM_FAMILY = 10, 11, 12

PROMOTER_BP = 2000          # window upstream of a TSS treated as promoter
# severity of a gene context, worst first
_CTX_RANK = {"CDS": 5, "UTR": 4, "ncRNA_exon": 3, "promoter": 2, "intron": 1,
             "intergenic": 0}


def _pp(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# 1. INPUT: guides (.xlsx without openpyxl, or .csv) and family copies
# ═══════════════════════════════════════════════════════════════════════════

def _xlsx_sheets(path):
    """{sheet_name: [[cell,...],...]} from a .xlsx using only the stdlib.

    openpyxl is not guaranteed inside the container, and an .xlsx is just a zip
    of XML, so parse it directly rather than adding a dependency.
    """
    import zipfile
    from xml.etree import ElementTree as ET

    NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    RNS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"

    def col_idx(ref):
        n = 0
        for ch in ref:
            if ch.isalpha():
                n = n * 26 + (ord(ch.upper()) - 64)
            else:
                break
        return n - 1

    out = {}
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        shared = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall(f"{NS}si"):
                shared.append("".join(t.text or "" for t in si.iter(f"{NS}t")))

        rels = {}
        if "xl/_rels/workbook.xml.rels" in names:
            for rel in ET.fromstring(z.read("xl/_rels/workbook.xml.rels")):
                tgt = rel.get("Target", "")
                tgt = tgt if tgt.startswith("xl/") else "xl/" + tgt.lstrip("./")
                rels[rel.get("Id")] = tgt

        wb = ET.fromstring(z.read("xl/workbook.xml"))
        for i, sh in enumerate(wb.iter(f"{NS}sheet"), 1):
            target = rels.get(sh.get(f"{RNS}id"), f"xl/worksheets/sheet{i}.xml")
            if target not in names:
                continue
            grid = []
            for row in ET.fromstring(z.read(target)).iter(f"{NS}row"):
                cells = {}
                for ci, c in enumerate(row.findall(f"{NS}c")):
                    ref = c.get("r") or ""
                    j = col_idx(ref) if ref and ref[0].isalpha() else ci
                    t = c.get("t")
                    if t == "inlineStr":
                        v = "".join(x.text or "" for x in c.iter(f"{NS}t"))
                    else:
                        node = c.find(f"{NS}v")
                        v = node.text if node is not None else ""
                        if t == "s" and v not in (None, ""):
                            v = shared[int(v)] if int(v) < len(shared) else ""
                    cells[j] = (v or "").strip()
                grid.append([cells.get(j, "") for j in range(max(cells) + 1)] if cells else [])
            out[sh.get("name") or f"Sheet{i}"] = grid
    return out


def _is_guide(s, lo=16, hi=25):
    s = str(s).strip().upper()
    return lo <= len(s) <= hi and set(s) <= set("ACGT")


def _family_from_sheet(name):
    """'HERVK-int guides' -> 'HERVK-int'."""
    toks = str(name).split()
    while toks and toks[-1].lower() in ("guide", "guides", "grna", "grnas", "candidates"):
        toks.pop()
    return " ".join(toks).strip() or str(name).strip()


def read_guides(path):
    """[{family, rank, guide, src_coverage, src_offtarget}] from .xlsx or .csv."""
    path = Path(path)
    rows = []
    if path.suffix.lower() in (".csv", ".tsv", ".txt"):
        df = pd.read_csv(path, sep=None, engine="python")
        df.columns = [str(c).strip() for c in df.columns]
        gcol = next((c for c in df.columns if c.lower().startswith("guide")
                     or c.lower() in ("protospacer", "sequence", "seq")), None)
        fcol = next((c for c in df.columns if c.lower() in ("family", "fam", "target")), None)
        if gcol is None:
            sys.exit(f"ERROR: no guide column in {path}")
        for i, r in df.iterrows():
            g = str(r[gcol]).strip().upper()
            if not _is_guide(g):
                continue
            rows.append({"family": str(r[fcol]).strip() if fcol else "TE",
                         "rank": i + 1, "guide": g,
                         "src_coverage": np.nan, "src_offtarget": np.nan})
        return rows

    sheets = _xlsx_sheets(path)
    for sheet, grid in sheets.items():
        fam = _family_from_sheet(sheet)
        # locate the header row (the one naming the guide column)
        hdr_i, gcol, ccol, ocol = None, None, None, None
        for i, row in enumerate(grid):
            low = [str(c).strip().lower() for c in row]
            cand = next((j for j, c in enumerate(low)
                         if c.startswith("guide") or c == "protospacer"), None)
            if cand is not None:
                hdr_i, gcol = i, cand
                ccol = next((j for j, c in enumerate(low) if c.startswith("coverage (")), None)
                ocol = next((j for j, c in enumerate(low) if c.startswith("off-target")), None)
                break
        body = grid[hdr_i + 1:] if hdr_i is not None else grid
        rank = 0
        for row in body:
            if gcol is not None and gcol < len(row):
                g = str(row[gcol]).strip().upper()
            else:
                g = next((str(c).strip().upper() for c in row if _is_guide(c)), "")
            if not _is_guide(g):
                continue
            rank += 1

            def _num(j):
                if j is None or j >= len(row):
                    return np.nan
                try:
                    return float(str(row[j]).replace(",", ""))
                except ValueError:
                    return np.nan

            rows.append({"family": fam, "rank": rank, "guide": g,
                         "src_coverage": _num(ccol), "src_offtarget": _num(ocol)})
    if not rows:
        sys.exit(f"ERROR: no guide sequences found in {path}")
    return rows


_SEQ_COLS = ("Seq", "seq", "sequence", "Sequence", "SEQ")
_CHR_COLS = ("Chromosome", "chromosome", "chrom", "chr", "Chr", "genoName")
_START_COLS = ("Start", "start", "genoStart", "chromStart")
_STOP_COLS = ("Stop", "stop", "End", "end", "genoEnd", "chromEnd")


def _pick(cols, options):
    return next((c for c in options if c in cols), None)


def find_seqs_csv(root, family):
    """Locate the family copies CSV (a CSV carrying a sequence column)."""
    root = Path(root)
    fam_l = family.lower()
    roots = [root / family, root / fam_l, root / fam_l.replace("-", "_"), root]
    pats = ["*_with_sequences.csv", "*sequences*.csv", "*.csv"]
    seen = []
    for r in roots:
        if not r.is_dir():
            continue
        for pat in pats:
            for p in sorted(r.rglob(pat)):
                if p in seen:
                    continue
                seen.append(p)
                try:
                    head = pd.read_csv(p, nrows=0)
                except Exception:
                    continue
                if _pick(list(head.columns), _SEQ_COLS) is None:
                    continue
                if r is root and fam_l.replace("-", "_") not in str(p).lower().replace("-", "_"):
                    continue          # only accept a loose match if it names the family
                return p
    return None


def load_copies(csv_path):
    """-> (list[str] sequences, DataFrame of coordinates or None)."""
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    scol = _pick(list(df.columns), _SEQ_COLS)
    if scol is None:
        sys.exit(f"ERROR: no sequence column in {csv_path}")
    keep = df[scol].apply(lambda s: isinstance(s, str) and len(s) > 0)
    df = df[keep].reset_index(drop=True)
    seqs = [str(s).upper() for s in df[scol]]
    ccol, s0, s1 = (_pick(list(df.columns), _CHR_COLS),
                    _pick(list(df.columns), _START_COLS),
                    _pick(list(df.columns), _STOP_COLS))
    coords = df[[ccol, s0, s1]].rename(
        columns={ccol: "chrom", s0: "start", s1: "stop"}) if all((ccol, s0, s1)) else None
    return seqs, coords


# ═══════════════════════════════════════════════════════════════════════════
# 2. COVERAGE of single guides over the family copies
# ═══════════════════════════════════════════════════════════════════════════

def guide_variant_map(guides, max_mm):
    """{variant_seq: {guide_index: n_mismatch}} using run_grna_offtarget._variants.

    _variants() holds the PAM-proximal seed (last 12 nt) exact and mutates only
    the PAM-distal positions --- the same tolerance model the shipped candidate
    list was built with.
    """
    vm = {}
    for gi, g in enumerate(guides):
        prev = set()
        for d in range(max_mm + 1):
            cur = set(_variants(g, d))
            for v in cur - prev:
                slot = vm.setdefault(v, {})
                if slot.get(gi, 99) > d:
                    slot[gi] = d
            prev = cur
    return vm


def coverage_maps(seqs, guides, max_mm, cas="SpCas9", glen=20):
    """Per guide: {copy_index: fewest mismatches}, with and without a PAM requirement.

    Returns (pam_hits, nopam_hits) --- lists indexed by guide.
    """
    vm = guide_variant_map(guides, max_mm)
    pam_hits = [dict() for _ in guides]
    nopam_hits = [dict() for _ in guides]
    n = len(seqs)
    for ci, seq in enumerate(seqs):
        if n > 200 and (ci + 1) % 200 == 0:
            _pp(f"    scoring copy {ci+1}/{n}...")
        for p in set(extract_pam_grnas(seq, glen, cas)):
            for gi, d in vm.get(p, {}).items():
                if pam_hits[gi].get(ci, 99) > d:
                    pam_hits[gi][ci] = d
        rseq = _rc(seq)
        for s in (seq, rseq):
            for i in range(len(s) - glen + 1):
                for gi, d in vm.get(s[i:i + glen], {}).items():
                    if nopam_hits[gi].get(ci, 99) > d:
                        nopam_hits[gi][ci] = d
    return pam_hits, nopam_hits


def cov_set(hits, max_mm):
    return {ci for ci, d in hits.items() if d <= max_mm}


def pam_proto_counts(seqs, cas="SpCas9", glen=20):
    """{protospacer: number of copies where it sits next to a real PAM}."""
    cnt = defaultdict(int)
    for s in seqs:
        for p in set(extract_pam_grnas(s, glen, cas)):
            cnt[p] += 1
    return cnt


def rescue_guides(guide, proto_counts, glen=20, kmin=15, top=3):
    """PAM-valid guides overlapping the same site as a guide that has no PAM.

    A guide whose protospacer is real but whose PAM is not NGG in the actual
    copies cannot be cut; the fix is almost always the same site shifted by a
    base or two, or the opposite strand. Anything sharing a *kmin*-mer with the
    dead guide (either orientation) is a candidate.
    """
    kmers = [guide[i:i + kmin] for i in range(glen - kmin + 1)]
    kmers += [_rc(k) for k in kmers]
    out = [(p, c) for p, c in proto_counts.items() if any(k in p for k in kmers)]
    out.sort(key=lambda t: -t[1])
    return out[:top]


# ═══════════════════════════════════════════════════════════════════════════
# 3. COMBINATIONS
# ═══════════════════════════════════════════════════════════════════════════

def enumerate_combos(cov_sets, n_copies, max_size=None, cap=400_000):
    """Union coverage of every guide subset (exhaustive), largest first per size.

    Falls back to greedy for sizes whose combination count would exceed *cap*.
    """
    n = len(cov_sets)
    max_size = min(max_size or n, n)
    out = []
    for k in range(1, max_size + 1):
        n_comb = 1
        for i in range(k):
            n_comb = n_comb * (n - i) // (i + 1)
        if n_comb > cap:
            out.append(_greedy_combo(cov_sets, k))
            continue
        for combo in itertools.combinations(range(n), k):
            u = set()
            for gi in combo:
                u |= cov_sets[gi]
            out.append({"guides": combo, "n_guides": k, "covered": len(u),
                        "cov_pct": 100.0 * len(u) / n_copies if n_copies else 0.0,
                        "_set": u})
    return out


def _greedy_combo(cov_sets, k):
    chosen, u = [], set()
    for _ in range(k):
        best, best_gain = None, -1
        for gi, s in enumerate(cov_sets):
            if gi in chosen:
                continue
            gain = len(s - u)
            if gain > best_gain:
                best, best_gain = gi, gain
        if best is None:
            break
        chosen.append(best)
        u |= cov_sets[best]
    return {"guides": tuple(sorted(chosen)), "n_guides": len(chosen), "covered": len(u),
            "cov_pct": 0.0, "_set": u, "greedy": True}


# ═══════════════════════════════════════════════════════════════════════════
# 4. GENOME-WIDE OFF-TARGET SCAN (exhaustive, PAM-aware, both strands)
# ═══════════════════════════════════════════════════════════════════════════

def iter_fasta(path):
    name, buf = None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(buf).upper()
                name = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if name is not None:
        yield name, "".join(buf).upper()


def _rc_pam(pattern):
    return "".join(_IUPAC_COMP.get(c, "N") for c in pattern)[::-1]


def _pam_starts(a, pattern):
    """Indices where *pattern* (IUPAC) matches the uint8 sequence *a*."""
    L = len(pattern)
    if a.size < L:
        return np.empty(0, dtype=np.int64)
    ok = np.ones(a.size - L + 1, dtype=bool)
    for off, ch in enumerate(pattern):
        allowed = _IUPAC.get(ch, "ACGT")
        if allowed == "ACGT":
            continue
        sub = a[off: off + ok.size]
        m = np.zeros(ok.size, dtype=bool)
        for b in allowed:
            m |= (sub == ord(b))
        ok &= m
    return np.flatnonzero(ok)


def scan_genome(genome_path, guides, max_mm, cas="SpCas9", glen=20,
                chroms=None, chunk=2_000_000, progress=True):
    """Exhaustive genome-wide guide search --- mismatches allowed ANYWHERE.

    A site is reported when the PAM matches and the protospacer is within
    *max_mm* mismatches of the guide. No seed heuristic is applied, so this
    finds the off-targets a seed-anchored search would miss.

    -> DataFrame [guide, chrom, start, end, strand, n_mm, site, pam, cut]
    """
    pattern, plen = _PAM_TYPES.get(cas, ("NGG", 3))
    three_prime = _PAM_3PRIME.get(cas, True)
    rc_pattern = _rc_pam(pattern)
    fwd_pat = [np.frombuffer(g.encode("ascii"), dtype=np.uint8) for g in guides]
    rev_pat = [np.frombuffer(_rc(g).encode("ascii"), dtype=np.uint8) for g in guides]
    ar = np.arange(glen, dtype=np.int64)
    rows = []

    for chrom, seq in iter_fasta(genome_path):
        if chroms and chrom not in chroms:
            continue
        a = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
        n = a.size
        if n < glen + plen:
            continue

        # + strand: guide reads along the plus strand
        p = _pam_starts(a, pattern)
        s_f = (p - glen) if three_prime else (p + plen)
        s_f = s_f[(s_f >= 0) & (s_f + glen <= n)]
        # - strand: plus-strand text holds rc(PAM) and rc(protospacer)
        q = _pam_starts(a, rc_pattern)
        s_r = (q + plen) if three_prime else (q - glen)
        s_r = s_r[(s_r >= 0) & (s_r + glen <= n)]

        if progress:
            _pp(f"    {chrom}: {n/1e6:.0f} Mb, {s_f.size + s_r.size:,} PAM sites")

        for starts, pats, strand in ((s_f, fwd_pat, "+"), (s_r, rev_pat, "-")):
            for off in range(0, starts.size, chunk):
                st = starts[off: off + chunk]
                win = a[st[:, None] + ar]
                for gi, pat in enumerate(pats):
                    mm = np.count_nonzero(win != pat, axis=1)
                    for h in np.flatnonzero(mm <= max_mm):
                        s0 = int(st[h])
                        if strand == "+":
                            site = seq[s0:s0 + glen]
                            pam = (seq[s0 + glen:s0 + glen + plen] if three_prime
                                   else seq[s0 - plen:s0])
                            cut = s0 + glen - 3 if three_prime else s0 + glen
                        else:
                            site = _rc(seq[s0:s0 + glen])
                            pam = (_rc(seq[s0 - plen:s0]) if three_prime
                                   else _rc(seq[s0 + glen:s0 + glen + plen]))
                            cut = s0 + 3 if three_prime else s0
                        rows.append((guides[gi], chrom, s0, s0 + glen, strand,
                                     int(mm[h]), site, pam, cut))
    return pd.DataFrame(rows, columns=["guide", "chrom", "start", "end", "strand",
                                       "n_mm", "site", "pam", "cut"])


# ═══════════════════════════════════════════════════════════════════════════
# 5. ANNOTATION --- RepeatMasker + RefSeq genes
# ═══════════════════════════════════════════════════════════════════════════

class _IntervalIndex:
    """Sorted per-chromosome intervals with a prefix-max of ends for stabbing."""

    def __init__(self):
        self._raw = defaultdict(lambda: (array("l"), array("l"), array("l")))
        self.payload = []
        self.idx = {}

    def add(self, chrom, start, end, payload_id):
        s, e, p = self._raw[chrom]
        s.append(start); e.append(end); p.append(payload_id)

    def finalize(self):
        for chrom, (s, e, p) in self._raw.items():
            starts = np.frombuffer(s, dtype=np.int64) if s.itemsize == 8 else np.array(s, dtype=np.int64)
            ends = np.frombuffer(e, dtype=np.int64) if e.itemsize == 8 else np.array(e, dtype=np.int64)
            pids = np.frombuffer(p, dtype=np.int64) if p.itemsize == 8 else np.array(p, dtype=np.int64)
            o = np.argsort(starts, kind="stable")
            starts, ends, pids = starts[o], ends[o], pids[o]
            self.idx[chrom] = (starts, ends, pids, np.maximum.accumulate(ends))
        self._raw = None
        return self

    def hits(self, chrom, start, end):
        ent = self.idx.get(chrom)
        if ent is None:
            return []
        starts, ends, pids, maxend = ent
        i = int(np.searchsorted(starts, end, side="right")) - 1
        out = []
        while i >= 0 and maxend[i] > start:
            if ends[i] > start and starts[i] < end:
                out.append((int(pids[i]), min(end, int(ends[i])) - max(start, int(starts[i]))))
            i -= 1
        return out


def load_rmsk(rmsk_path, chroms=None):
    """RepeatMasker interval index -> (_IntervalIndex, payload list of (name, cls, fam))."""
    _pp(f"  Indexing RepeatMasker: {Path(rmsk_path).name}")
    ix = _IntervalIndex()
    key = {}
    n = 0
    with gzip.open(rmsk_path, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) <= RM_FAMILY:
                continue
            chrom = f[RM_CHROM]
            if chroms and chrom not in chroms:
                continue
            k = (f[RM_NAME], f[RM_CLASS], f[RM_FAMILY])
            pid = key.get(k)
            if pid is None:
                pid = key[k] = len(key)
                ix.payload.append(k)
            ix.add(chrom, int(f[RM_START]), int(f[RM_END]), pid)
            n += 1
            if n % 1_000_000 == 0:
                _pp(f"    {n:,} repeats...")
    _pp(f"  {n:,} repeat intervals, {len(key):,} distinct repNames")
    return ix.finalize()


def load_refgene(path, chroms=None):
    """RefSeq transcript index -> (_IntervalIndex, payload of transcript dicts).

    Intervals are padded upstream by PROMOTER_BP so promoter-proximal cuts are
    still found; the exact context is decided in classify_gene().
    """
    _pp(f"  Indexing RefSeq genes: {Path(path).name}")
    ix = _IntervalIndex()
    op = gzip.open if str(path).endswith(".gz") else open
    n = 0
    with op(path, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) <= RG_NAME2:
                continue
            chrom = f[RG_CHROM]
            if chroms and chrom not in chroms:
                continue
            try:
                txs, txe = int(f[RG_TXS]), int(f[RG_TXE])
                cdss, cdse = int(f[RG_CDSS]), int(f[RG_CDSE])
                exs = [int(x) for x in f[RG_EXS].rstrip(",").split(",") if x != ""]
                exe = [int(x) for x in f[RG_EXE].rstrip(",").split(",") if x != ""]
            except ValueError:
                continue
            tx = {"name": f[RG_NAME], "gene": f[RG_NAME2], "strand": f[RG_STRAND],
                  "txs": txs, "txe": txe, "cdss": cdss, "cdse": cdse,
                  "exs": exs, "exe": exe,
                  "coding": cdss < cdse and f[RG_NAME].startswith(("NM_", "XM_")),
                  "curated": f[RG_NAME].startswith(("NM_", "NR_"))}
            ix.payload.append(tx)
            ix.add(chrom, max(0, txs - PROMOTER_BP), txe + PROMOTER_BP, len(ix.payload) - 1)
            n += 1
    _pp(f"  {n:,} RefSeq transcripts")
    return ix.finalize()


def classify_gene(tx, start, end):
    """Where does [start,end) fall relative to transcript *tx*?"""
    if end <= tx["txs"] or start >= tx["txe"]:
        if tx["strand"] == "+":
            lo, hi = tx["txs"] - PROMOTER_BP, tx["txs"]
        else:
            lo, hi = tx["txe"], tx["txe"] + PROMOTER_BP
        return "promoter" if (end > lo and start < hi) else "intergenic"
    in_exon = any(end > es and start < ee for es, ee in zip(tx["exs"], tx["exe"]))
    if not in_exon:
        return "intron"
    if not tx["coding"]:
        return "ncRNA_exon"
    return "CDS" if (end > tx["cdss"] and start < tx["cdse"]) else "UTR"


def annotate_sites(sites, rmsk_ix, gene_ix):
    """Add repeat + gene context columns to the off-target site table."""
    rep_name, rep_class, rep_fam = [], [], []
    ctx, genes, coding_gene = [], [], []
    for chrom, s, e in zip(sites["chrom"], sites["start"], sites["end"]):
        # repeat: longest overlap wins
        best = ("", "", "")
        if rmsk_ix is not None:
            hits = rmsk_ix.hits(chrom, int(s), int(e))
            if hits:
                pid = max(hits, key=lambda t: t[1])[0]
                best = rmsk_ix.payload[pid]
        rep_name.append(best[0]); rep_class.append(best[1]); rep_fam.append(best[2])

        worst, gset, cflag = "intergenic", set(), False
        if gene_ix is not None:
            for pid, _ in gene_ix.hits(chrom, int(s), int(e)):
                tx = gene_ix.payload[pid]
                c = classify_gene(tx, int(s), int(e))
                if c == "intergenic":
                    continue
                gset.add(tx["gene"])
                if _CTX_RANK[c] > _CTX_RANK[worst]:
                    worst = c
                if c in ("CDS", "UTR") and tx["coding"]:
                    cflag = True
        ctx.append(worst)
        genes.append(",".join(sorted(gset)))
        coding_gene.append(cflag)

    out = sites.copy()
    out["repName"] = rep_name
    out["repClass"] = rep_class
    out["repFamily"] = rep_fam
    out["gene_context"] = ctx
    out["genes"] = genes
    out["hits_coding_exon"] = coding_gene
    return out


def download_refgene(build, ann_dir):
    """Fetch refGene.txt.gz from UCSC (same shell-based fetch as te_prep.py)."""
    ann_dir = Path(ann_dir)
    ann_dir.mkdir(parents=True, exist_ok=True)
    out = ann_dir / f"refGene_{build}.txt.gz"
    if out.exists() and out.stat().st_size > 0:
        return str(out)
    url = REFGENE_URL.format(build=build)
    _pp(f"  Downloading {url}")
    for cmd in (f"curl -4 -L -s -o {shlex.quote(str(out))} {shlex.quote(url)}",
                f"curl -L -s -o {shlex.quote(str(out))} {shlex.quote(url)}",
                f"wget -q -O {shlex.quote(str(out))} {shlex.quote(url)}"):
        try:
            if subprocess.run(cmd, shell=True).returncode == 0 and out.exists() \
                    and out.stat().st_size > 0:
                _pp(f"  refGene -> {out} ({out.stat().st_size/1e6:.1f} MB)")
                return str(out)
        except Exception:
            pass
        if out.exists():
            out.unlink()
    _pp("  WARNING: refGene download failed --- gene context will be unavailable")
    return None


# ═══════════════════════════════════════════════════════════════════════════
# 6. REPORTING
# ═══════════════════════════════════════════════════════════════════════════

def _combo_ot(combo_guides, per_guide_sites, target_fams):
    """Aggregate off-target counts over the guides of one combination.

    Sites are de-duplicated by locus, so two guides hitting the same place are
    counted once.
    """
    loci, off_fam, exon, exon_off_fam, uniq, gene_names = set(), set(), set(), set(), set(), set()
    perfect_off = set()
    off_rep = defaultdict(set)
    for g in combo_guides:
        df = per_guide_sites.get(g)
        if df is None or df.empty:
            continue
        for r in df.itertuples(index=False):
            key = (r.chrom, r.start, r.strand)
            loci.add(key)
            related = r.repFamily in target_fams and r.repFamily != ""
            if not related:
                off_fam.add(key)
                if r.repName:
                    off_rep[r.repName].add(key)
                if r.n_mm == 0:
                    perfect_off.add(key)
            if not r.repName:
                uniq.add(key)
            if r.hits_coding_exon:
                exon.add(key)
                if r.genes:
                    gene_names.update(r.genes.split(","))
                if not related:
                    exon_off_fam.add(key)
    top_rep, top_n = "", 0
    if off_rep:
        top_rep = max(off_rep, key=lambda k: len(off_rep[k]))
        top_n = len(off_rep[top_rep])
    return {"ot_sites": len(loci), "ot_offfamily": len(off_fam),
            "ot_offfamily_perfect": len(perfect_off),
            "ot_top_offfamily_rep": top_rep, "ot_top_offfamily_n": top_n,
            "ot_nonrepeat": len(uniq), "ot_coding_exon": len(exon),
            "ot_coding_exon_offfamily": len(exon_off_fam),
            "ot_genes": ";".join(sorted(gene_names))}


def _fmt_combo(guides, idxs):
    return " + ".join(guides[i] for i in idxs)


# ═══════════════════════════════════════════════════════════════════════════
# 7. SELF-TEST (controls on synthetic data --- no genome/annotation needed)
# ═══════════════════════════════════════════════════════════════════════════

def self_test():
    """Plant known sites in a synthetic genome and check every layer finds them."""
    import random
    import tempfile
    rng = random.Random(7)
    ok = True

    def chk(label, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {label}: got {got}, expected {want}")

    print("=" * 68)
    print("SELF-TEST --- synthetic controls")
    print("=" * 68)

    # non-palindromic guides, and every planted site gets a TTT left flank so no
    # accidental CCN (minus-strand PAM) is created next to it
    g1 = "GACCTGAGTCGATTACGCAT"
    g2 = "TTGCATTGCTTAGCATTGCA"
    assert _rc(g1) != g1 and _rc(g2) != g2
    bg = "".join(rng.choice("ACGT") for _ in range(4000))
    F = "TTT"

    # --- coverage: PAM required, mismatches only PAM-distal ------------------
    # copy A: g1 + AGG PAM (perfect, cuttable)
    # copy B: g1 with 1 PAM-distal mismatch + TGG PAM (cuttable, 1 mm)
    # copy C: g1 perfect but PAM is AAA (NOT cuttable) -> the PAM test
    # copy D: g1 with a mismatch inside the PAM-proximal seed (not covered)
    # copy E: g2 on the reverse strand with a CCA PAM upstream (strand test)
    flip = lambda b: "T" if b != "T" else "A"
    mm_distal = flip(g1[0]) + g1[1:]                      # position 0  = PAM-distal
    mm_seed = g1[:17] + flip(g1[17]) + g1[18:]            # position 17 = inside the seed
    copies = [
        bg[:200] + F + g1 + "AGG" + bg[200:400],
        bg[:200] + F + mm_distal + "TGG" + bg[200:400],
        bg[:200] + F + g1 + "AAA" + bg[200:400],
        bg[:200] + F + mm_seed + "AGG" + bg[200:400],
        bg[:200] + F + "CCA" + _rc(g2) + bg[200:400],
    ]
    pam_hits, nopam_hits = coverage_maps(copies, [g1, g2], max_mm=2)
    chk("g1 covers copies A,B (PAM-aware, mm<=2)", sorted(cov_set(pam_hits[0], 2)), [0, 1])
    chk("g1 exact-match coverage is copy A only", sorted(cov_set(pam_hits[0], 0)), [0])
    chk("g1 without a PAM requirement also sees C,D", sorted(cov_set(nopam_hits[0], 2)), [0, 1, 2])
    chk("seed mismatch is not covered", 3 in cov_set(pam_hits[0], 2), False)
    chk("g2 found on the reverse strand", sorted(cov_set(pam_hits[1], 0)), [4])

    # --- combinations --------------------------------------------------------
    cov_sets = [{0, 1, 2}, {2, 3}, {0, 1}]
    combos = enumerate_combos(cov_sets, n_copies=4)
    pairs = {c["guides"]: c["covered"] for c in combos if c["n_guides"] == 2}
    chk("union of disjoint-ish pair (0,1)", pairs[(0, 1)], 4)
    chk("redundant pair (0,2) adds nothing", pairs[(0, 2)], 3)
    best2 = max((c for c in combos if c["n_guides"] == 2), key=lambda c: c["covered"])
    chk("best pair is (0,1)", best2["guides"], (0, 1))

    # --- genome scan ---------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        chrom1, pos = "", {}

        def add(s, tag=None):
            nonlocal chrom1
            if tag:
                pos[tag] = len(chrom1)
            chrom1 += s

        off1 = g1[:2] + flip(g1[2]) + g1[3:]             # 1 mismatch at position 2
        add(bg[:1000] + F)
        add(g1 + "CGG", "fwd0")                          # + strand, 0 mm
        add(bg[1000:2000] + F)
        add(off1 + "TGG", "fwd1")                        # + strand, 1 mm
        add(bg[2000:3000] + F + "CCT")                   # minus-strand PAM
        add(_rc(g1), "rev0")                             # - strand, 0 mm
        add(bg[3000:3500] + F)
        add(g1 + "AAT", "nopam")                         # no PAM -> must NOT be found
        add(bg[3500:4000])
        fa = td / "mini.fa"
        fa.write_text(">chr1\n" + "\n".join(chrom1[i:i + 60]
                                            for i in range(0, len(chrom1), 60)) + "\n")
        sites = scan_genome(str(fa), [g1], max_mm=1, chroms={"chr1"}, progress=False)
        chk("genome scan finds 3 sites (2 fwd, 1 rev), skips the PAM-less one",
            len(sites), 3)
        chk("strands found", sorted(sites["strand"]), ["+", "+", "-"])
        chk("mismatch counts", sorted(sites["n_mm"]), [0, 0, 1])
        chk("PAM-less planted site is absent",
            pos["nopam"] in set(sites["start"]), False)
        got0 = int(sites[(sites.strand == "+") & (sites.n_mm == 0)]["start"].iloc[0])
        chk("plus-strand site coordinate", got0, pos["fwd0"])
        rev = sites[sites.strand == "-"].iloc[0]
        chk("minus-strand site coordinate", int(rev["start"]), pos["rev0"])
        chk("minus-strand site sequence is the guide", rev["site"], g1)
        chk("minus-strand PAM reads as NGG", rev["pam"][1:], "GG")

        # --- annotation ------------------------------------------------------
        # a repeat covering the fwd0 site; a gene whose single CDS exon covers fwd1
        rmsk = td / "rmsk.txt.gz"
        with gzip.open(rmsk, "wt") as fh:
            row = ["0"] * 17
            row[RM_CHROM] = "chr1"
            row[RM_START], row[RM_END] = str(pos["fwd0"] - 10), str(pos["fwd0"] + 30)
            row[RM_NAME], row[RM_CLASS], row[RM_FAMILY] = "LTR5_Hs", "LTR", "ERVK"
            fh.write("\t".join(row) + "\n")
        rg = td / "refGene.txt.gz"
        txs, txe = pos["fwd1"] - 100, pos["fwd1"] + 100
        with gzip.open(rg, "wt") as fh:
            row = ["0"] * 16
            row[RG_NAME], row[RG_CHROM], row[RG_STRAND] = "NM_TEST", "chr1", "+"
            row[RG_TXS], row[RG_TXE] = str(txs), str(txe)
            row[RG_CDSS], row[RG_CDSE] = str(txs), str(txe)
            row[RG_NEX], row[RG_EXS], row[RG_EXE] = "1", f"{txs},", f"{txe},"
            row[RG_NAME2] = "TESTGENE"
            fh.write("\t".join(row) + "\n")
        ann = annotate_sites(sites, load_rmsk(str(rmsk)), load_refgene(str(rg)))
        ann = ann.sort_values("start").reset_index(drop=True)
        chk("fwd0 annotated as LTR5_Hs/ERVK", ann.iloc[0]["repName"], "LTR5_Hs")
        chk("fwd0 has no repeat-free label", ann.iloc[0]["repFamily"], "ERVK")
        chk("exactly one coding-exon hit", int(ann["hits_coding_exon"].sum()), 1)
        chk("that site names the gene",
            sorted(set(ann[ann.hits_coding_exon]["genes"]))[0], "TESTGENE")
        chk("gene context is CDS",
            sorted(set(ann[ann.hits_coding_exon]["gene_context"]))[0], "CDS")
        chk("fwd0 is within 2 kb upstream of the TSS -> promoter",
            ann.iloc[0]["gene_context"], "promoter")

    print("=" * 68)
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    print("=" * 68)
    return 0 if ok else 1


# ═══════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Guide-combination coverage + genome-wide off-target/gene safety",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--guides", help=".xlsx or .csv of the guides to combine")
    p.add_argument("--seqs-dir", default=".",
                   help="Directory holding <family>/.../*_with_sequences.csv")
    p.add_argument("--seqs", action="append", default=[], metavar="FAM=PATH",
                   help="Explicit copies CSV for a family (repeatable)")
    p.add_argument("--family", action="append", default=[],
                   help="Restrict to these families (default: all sheets in --guides)")
    p.add_argument("--out", default="./grna_combos")
    p.add_argument("--genome", default="", help="Genome FASTA for off-target scan")
    p.add_argument("--assembly", default="hg38")
    p.add_argument("--rmsk-dir", default=str(Path.home() / "te_analysis" / "rmsk"),
                   help="Cache dir for rmsk / refGene")
    p.add_argument("--rmsk", default="", help="Explicit rmsk.txt.gz")
    p.add_argument("--refgene", default="", help="Explicit refGene.txt.gz")
    p.add_argument("--cas", default="SpCas9", choices=list(_PAM_TYPES))
    p.add_argument("--grna-len", type=int, default=20)
    p.add_argument("--cov-mm", type=int, default=2, choices=[0, 1, 2],
                   help="Mismatches allowed for on-family coverage (PAM-distal only)")
    p.add_argument("--ot-mm", type=int, default=3,
                   help="Mismatches allowed for genome-wide off-target (any position)")
    p.add_argument("--dead-pct", type=float, default=5.0,
                   help="Flag guides covering less than this %% as un-cuttable")
    p.add_argument("--target-cov", type=float, default=0.5,
                   help="Coverage fraction the guide set must reach (default 0.5)")
    p.add_argument("--max-combo-size", type=int, default=0,
                   help="Largest combination to enumerate (default: all guides)")
    p.add_argument("--all-contigs", action="store_true",
                   help="Scan alts/scaffolds too (default: chr1-22,X,Y,M)")
    p.add_argument("--skip-offtarget", action="store_true")
    p.add_argument("--reuse-sites", action="store_true",
                   help="Reuse offtarget_sites_all.csv in --out instead of rescanning "
                        "the genome (only valid if the guide set is unchanged)")
    p.add_argument("--self-test", action="store_true",
                   help="Run synthetic controls and exit")
    return p.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        sys.exit(self_test())
    if not args.guides:
        sys.exit("ERROR: --guides is required (or use --self-test)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    glen, cas = args.grna_len, args.cas

    _pp("=" * 66)
    _pp(f"GAMECA guide combinations + off-target safety ({cas})")
    _pp("=" * 66)

    # ── guides ──────────────────────────────────────────────────────────────
    grows = read_guides(args.guides)
    fams = {}
    for r in grows:
        fams.setdefault(r["family"], []).append(r)
    if args.family:
        want = {f.lower() for f in args.family}
        fams = {k: v for k, v in fams.items() if k.lower() in want}
    if not fams:
        sys.exit("ERROR: no families to analyse")
    for f, rs in fams.items():
        _pp(f"  {f}: {len(rs)} guides from {Path(args.guides).name}")

    explicit = {}
    for spec in args.seqs:
        if "=" not in spec:
            sys.exit(f"ERROR: --seqs expects FAM=PATH, got '{spec}'")
        k, v = spec.split("=", 1)
        explicit[k.strip().lower()] = v.strip()

    # ── per-family copy coverage ────────────────────────────────────────────
    data = {}
    for fam, rs in fams.items():
        guides = [r["guide"] for r in rs]
        csv_path = explicit.get(fam.lower()) or find_seqs_csv(args.seqs_dir, fam)
        if not csv_path or not Path(csv_path).exists():
            _pp(f"  ERROR: no copies CSV for {fam} under {args.seqs_dir} "
                f"(pass --seqs {fam}=/path/to/*_with_sequences.csv)")
            sys.exit(2)
        _pp(f"[{fam}] copies: {csv_path}")
        seqs, coords = load_copies(csv_path)
        _pp(f"[{fam}] {len(seqs)} copies --- scoring {len(guides)} guides "
            f"(PAM required, <={args.cov_mm} PAM-distal mismatches)")
        pam_hits, nopam_hits = coverage_maps(seqs, guides, args.cov_mm, cas, glen)
        data[fam] = {"rows": rs, "guides": guides, "seqs": seqs, "coords": coords,
                     "csv": str(csv_path), "pam": pam_hits, "nopam": nopam_hits}

    # ── genome-wide off-target (one pass, all families) ─────────────────────
    sites = pd.DataFrame()
    rmsk_ix = gene_ix = None
    cached = out / "offtarget_sites_all.csv"
    if args.reuse_sites and cached.exists():
        sites = pd.read_csv(cached).fillna({"repName": "", "repClass": "",
                                            "repFamily": "", "genes": ""})
        _pp(f"Reusing {cached} ({len(sites):,} annotated sites) --- no genome rescan")
    elif not args.skip_offtarget and args.genome and Path(args.genome).exists():
        all_guides = sorted({g for d in data.values() for g in d["guides"]})
        chroms = None if args.all_contigs else STD_CHROMS
        _pp(f"Genome-wide scan: {len(all_guides)} guides vs {Path(args.genome).name} "
            f"(exhaustive, <={args.ot_mm} mismatches anywhere, {cas} PAM)")
        sites = scan_genome(args.genome, all_guides, args.ot_mm, cas, glen, chroms)
        _pp(f"  {len(sites):,} raw sites")

        rmsk_path = args.rmsk or None
        if not rmsk_path:
            try:
                from te_prep import get_rmsk_path
                rmsk_path = get_rmsk_path(args.assembly, args.rmsk_dir)
            except SystemExit:
                rmsk_path = None
            except Exception as e:
                _pp(f"  WARNING: rmsk unavailable ({e})")
                rmsk_path = None
        if rmsk_path and Path(rmsk_path).exists():
            rmsk_ix = load_rmsk(rmsk_path, chroms)
        else:
            _pp("  WARNING: no rmsk --- repeat context will be blank")

        rg_path = args.refgene or download_refgene(args.assembly, args.rmsk_dir)
        if rg_path and Path(rg_path).exists():
            gene_ix = load_refgene(rg_path, chroms)
        else:
            _pp("  WARNING: no refGene --- gene context will be blank")

        _pp("  Annotating sites (repeats + genes)...")
        sites = annotate_sites(sites, rmsk_ix, gene_ix)
        sites.to_csv(out / "offtarget_sites_all.csv", index=False)
        _pp(f"  Wrote {out/'offtarget_sites_all.csv'}")
    elif not args.skip_offtarget:
        _pp(f"  WARNING: genome '{args.genome}' not found --- skipping off-target scan")

    per_guide_sites = {g: d for g, d in sites.groupby("guide")} if len(sites) else {}

    # ── per-family analysis + report ────────────────────────────────────────
    answer = ["=" * 78,
              "GAMECA --- guide combinations and off-target safety",
              f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
              "=" * 78, ""]

    for fam, d in data.items():
        guides, seqs = d["guides"], d["seqs"]
        n_copies = len(seqs)
        cov_sets = [cov_set(h, args.cov_mm) for h in d["pam"]]

        # target repFamily (e.g. ERVK) --- hits to any member are acceptable
        target_fams = set()
        if rmsk_ix is not None:
            for (nm, cl, fm) in rmsk_ix.payload:
                if nm.lower() == fam.lower():
                    target_fams.add(fm)
        elif len(sites):        # --reuse-sites: recover it from the annotated table
            m = sites[sites.repName.str.lower() == fam.lower()]
            target_fams = set(m["repFamily"].unique()) - {""}

        # per-guide table
        gt = []
        for gi, r in enumerate(d["rows"]):
            g = r["guide"]
            row = {"family": fam, "rank": r["rank"], "guide": g,
                   "on_target_score": on_target_score(g),
                   "cov_exact": len(cov_set(d["pam"][gi], 0)),
                   "cov_mm1": len(cov_set(d["pam"][gi], 1)),
                   "cov_mm2": len(cov_set(d["pam"][gi], 2)),
                   "cov_pct": 100.0 * len(cov_sets[gi]) / n_copies if n_copies else 0.0,
                   "cov_noPAM_mm2": len(cov_set(d["nopam"][gi], args.cov_mm)),
                   "src_coverage": r["src_coverage"]}
            row.update(_combo_ot([g], per_guide_sites, target_fams))
            gt.append(row)
        gt = pd.DataFrame(gt).sort_values("cov_pct", ascending=False)
        gt.to_csv(out / f"guide_coverage_{fam}.csv", index=False)

        # combinations
        combos = enumerate_combos(cov_sets, n_copies,
                                  args.max_combo_size or len(guides))
        crows, combo_idx = [], {}
        for c in combos:
            row = {"family": fam, "n_guides": c["n_guides"],
                   "guides": _fmt_combo(guides, c["guides"]),
                   "ranks": "+".join(str(d["rows"][i]["rank"]) for i in c["guides"]),
                   "covered": c["covered"], "cov_pct": c["cov_pct"],
                   "cov_exact": len(set().union(*[cov_set(d["pam"][i], 0)
                                                  for i in c["guides"]])),
                   "min_on_target": min(on_target_score(guides[i])
                                        for i in c["guides"])}
            row.update(_combo_ot([guides[i] for i in c["guides"]],
                                 per_guide_sites, target_fams))
            combo_idx[row["ranks"]] = c["guides"]
            crows.append(row)
        cdf = pd.DataFrame(crows).sort_values(
            ["n_guides", "covered"], ascending=[True, False])
        cdf.to_csv(out / f"combos_{fam}.csv", index=False)

        # drop_duplicates (not groupby.first, which mixes columns across rows)
        best_by_size = cdf.sort_values(["n_guides", "covered", "ot_coding_exon_offfamily"],
                                       ascending=[True, False, True]) \
                          .drop_duplicates("n_guides")
        best_by_size.to_csv(out / f"best_by_size_{fam}.csv", index=False)

        need = best_by_size[best_by_size["cov_pct"] >= args.target_cov * 100]
        pairs = cdf[cdf.n_guides == 2].sort_values(
            ["covered", "ot_coding_exon_offfamily"], ascending=[False, True])

        # ── narrative ────────────────────────────────────────────────────────
        answer += [f"{'='*78}", f"{fam}  ---  {n_copies} copies  "
                   f"({Path(d['csv']).name})", f"{'='*78}", ""]
        answer.append(f"Coverage model: {cas} PAM required at the copy, "
                      f"<={args.cov_mm} mismatches allowed PAM-distal only.")
        if len(sites):
            answer.append(f"Off-target model: exhaustive genome scan, <={args.ot_mm} "
                          f"mismatches at ANY position, PAM required.")
            if target_fams:
                answer.append(f"Hits to repFamily {'/'.join(sorted(target_fams))} "
                              "(the target's own family: LTR5A/B, HERVK-int, ...) "
                              "are counted as 'on-family' and are expected/acceptable.")
        answer.append("")

        answer.append("-- single guides (coverage of this family) " + "-" * 34)
        answer.append(f"{'rank':>4} {'guide':<22} {'exact':>6} {'<=1mm':>6} {'<=2mm':>6} "
                      f"{'%cov':>6} {'OT':>6} {'exonic':>7}")
        for _, r in gt.iterrows():
            answer.append(f"{int(r['rank']):>4} {r['guide']:<22} {int(r['cov_exact']):>6} "
                          f"{int(r['cov_mm1']):>6} {int(r['cov_mm2']):>6} "
                          f"{r['cov_pct']:>5.1f}% {int(r['ot_sites']):>6} "
                          f"{int(r['ot_coding_exon_offfamily']):>7}")
        answer.append("  (OT = genome-wide sites; exonic = sites in a coding exon that are"
                      " NOT part of the target repeat family)")
        answer.append("")

        dead = gt[gt.cov_pct < args.dead_pct]
        if len(dead):
            pc = pam_proto_counts(seqs, cas, glen)
            answer.append("-- !! guides with no cuttable site " + "-" * 42)
            for _, r in dead.iterrows():
                answer.append(
                    f"   rank {int(r['rank'])} {r['guide']}: the protospacer is present in "
                    f"{int(r['cov_noPAM_mm2'])} copies, but it is never followed by an "
                    f"{cas} PAM in any copy, so Cas9 cannot cut it. The coverage quoted for "
                    "it came from a PAM-free sequence match.")
                res = rescue_guides(r["guide"], pc, glen)
                if res:
                    answer.append("      PAM-valid guides at the same site (exact-match copies):")
                    for p, c in res:
                        answer.append(f"        {p}  {c:>5} copies "
                                      f"({100.0*c/n_copies if n_copies else 0:.1f}%)")
            answer.append("")

        answer.append("-- best 2-guide combinations " + "-" * 47)
        answer.append(f"{'ranks':>7} {'copies':>7} {'%cov':>7} {'gain':>6} {'minOTS':>7} "
                      f"{'OT':>6} {'offFam':>7} {'0mm':>5} {'exon':>5}  "
                      f"{'biggest off-family hit':<26} guides")
        singles = dict(zip(gt["guide"], gt["cov_pct"]))
        for _, r in pairs.head(10).iterrows():
            g1, g2 = r["guides"].split(" + ")
            gain = r["cov_pct"] - max(singles.get(g1, 0), singles.get(g2, 0))
            top = (f"{r['ot_top_offfamily_rep']} x{int(r['ot_top_offfamily_n'])}"
                   if r["ot_top_offfamily_rep"] else "-")
            answer.append(f"{r['ranks']:>7} {int(r['covered']):>7} {r['cov_pct']:>6.1f}% "
                          f"{gain:>+5.1f} {r['min_on_target']:>7.2f} {int(r['ot_sites']):>6} "
                          f"{int(r['ot_offfamily']):>7} {int(r['ot_offfamily_perfect']):>5} "
                          f"{int(r['ot_coding_exon_offfamily']):>5}  {top:<26} {r['guides']}")
        answer.append("  (gain = percentage points added over the better single guide;")
        answer.append("   minOTS = worst on-target quality score in the pair --- a low value")
        answer.append("   means one member is AT-rich / homopolymeric and may cut poorly;")
        answer.append("   offFam = genome sites outside the target repeat family, 0mm = how")
        answer.append("   many of those are PERFECT matches and so will actually be cut)")
        # The top-coverage pair is often not the one to build: rank it by safety
        # first (no perfect-match off-family cuts, no coding-exon hits, decent
        # on-target quality) and report what that costs in coverage.
        tiers = [("no perfect-match off-family cuts, no coding-exon hits, "
                  "good on-target quality",
                  lambda d: (d.min_on_target >= 0.8) & (d.ot_offfamily_perfect == 0)
                  & (d.ot_coding_exon_offfamily == 0)),
                 ("no perfect-match off-family cuts",
                  lambda d: d.ot_offfamily_perfect == 0),
                 ("good on-target quality", lambda d: d.min_on_target >= 0.8)]
        if len(pairs):
            for label, filt in (tiers if len(sites) else tiers[2:]):
                sub = pairs[filt(pairs)]
                if not len(sub):
                    continue
                c0 = sub.iloc[0]
                cost = pairs.iloc[0]["cov_pct"] - c0["cov_pct"]
                same = c0["ranks"] == pairs.iloc[0]["ranks"]
                answer.append(f"  >> RECOMMENDED pair: ranks {c0['ranks']} = "
                              f"{int(c0['covered'])}/{n_copies} ({c0['cov_pct']:.1f}%) --- "
                              f"{label}" + (" (also the top-coverage pair)" if same else
                                            f"; costs {cost:.1f} pp vs the top-coverage pair"))
                answer.append(f"     {c0['guides']}")
                break
        answer.append("")

        answer.append("-- coverage vs number of guides (best set at each size) " + "-" * 21)
        answer.append(f"{'n':>2} {'copies':>7} {'%cov':>7} {'OT':>6} {'offFam':>7} "
                      f"{'0mm':>5} {'exon':>5}  ranks")
        for _, r in best_by_size.iterrows():
            answer.append(f"{int(r['n_guides']):>2} {int(r['covered']):>7} "
                          f"{r['cov_pct']:>6.1f}% {int(r['ot_sites']):>6} "
                          f"{int(r['ot_offfamily']):>7} {int(r['ot_offfamily_perfect']):>5} "
                          f"{int(r['ot_coding_exon_offfamily']):>5}  {r['ranks']}")
        answer.append("")

        # Are the copies a guide set misses just short fragments? rmsk splits
        # old elements into many small pieces that cannot contain the target
        # site at all, which drags the whole-family percentage down.
        ref_row = need.iloc[0] if len(need) else best_by_size.iloc[-1]
        sel = combo_idx.get(ref_row["ranks"], tuple(range(len(guides))))
        sel_cov = set().union(*[cov_sets[i] for i in sel]) if sel else set()
        Lc = np.array([len(s) for s in seqs])
        ref_len = float(np.percentile(Lc, 95)) if len(Lc) else 0.0
        if ref_len > 0:
            answer.append(f"-- coverage by copy length (set {ref_row['ranks']}; "
                          f"full length ~= {ref_len:,.0f} bp) " + "-" * 12)
            answer.append(f"{'copy length':>16} {'copies':>7} {'covered':>8} {'%':>7}")
            edges = [(0.0, 0.10), (0.10, 0.25), (0.25, 0.50), (0.50, 0.90), (0.90, 9.9)]
            labels = ["<10% (fragment)", "10-25%", "25-50%", "50-90%", ">=90% (full)"]
            for (lo, hi), lab in zip(edges, labels):
                idx = [i for i in range(len(seqs)) if lo <= Lc[i] / ref_len < hi]
                if not idx:
                    continue
                cv = sum(1 for i in idx if i in sel_cov)
                answer.append(f"{lab:>16} {len(idx):>7} {cv:>8} "
                              f"{100.0*cv/len(idx):>6.1f}%")
            full_idx = [i for i in range(len(seqs)) if Lc[i] / ref_len >= 0.90]
            if full_idx:
                fc = sum(1 for i in full_idx if i in sel_cov)
                answer.append(f"  >> near-full-length elements: {fc}/{len(full_idx)} "
                              f"({100.0*fc/len(full_idx):.1f}%) covered. These are the "
                              "copies that can still be transcriptionally/coding active;")
                answer.append("     the short fragments cannot contain the target site "
                              "at all, so they cap the whole-family percentage.")
            answer.append("")

        tgt_pct = args.target_cov * 100
        if len(need):
            row = need.iloc[0]
            answer.append(f">> {int(row['n_guides'])} guides reach {tgt_pct:.0f}% coverage: "
                          f"ranks {row['ranks']} = {int(row['covered'])}/{n_copies} "
                          f"({row['cov_pct']:.1f}%)")
            answer.append(f"   {row['guides']}")
        else:
            top = best_by_size.iloc[-1]
            answer.append(f">> {tgt_pct:.0f}% coverage is NOT reachable with these "
                          f"{len(guides)} guides. All {int(top['n_guides'])} together "
                          f"cover {int(top['covered'])}/{n_copies} ({top['cov_pct']:.1f}%).")
        answer.append("")

        # off-target detail for the recommended sets
        if len(sites):
            rec = []
            if len(pairs):
                rec.append(("best pair", pairs.iloc[0]))
            if len(need):
                rec.append((f"smallest set >={tgt_pct:.0f}%", need.iloc[0]))
            for label, row in rec:
                gl = row["guides"].split(" + ")
                sub = sites[sites.guide.isin(gl)]
                answer.append(f"-- off-target profile: {label} (ranks {row['ranks']}) " + "-" * 20)
                if sub.empty:
                    answer.append("   no sites found")
                    continue
                by_rep = sub.assign(
                    bucket=np.where(sub.repFamily.isin(target_fams) & (sub.repFamily != ""),
                                    "on-family (" + sub.repFamily + ")",
                                    np.where(sub.repName == "", "unique / non-repeat",
                                             "other repeat: " + sub.repClass))
                ).groupby("bucket").size().sort_values(ascending=False)
                for k, v in by_rep.items():
                    answer.append(f"   {v:>7,}  {k}")
                top_rep = sub[sub.repName != ""].groupby("repName").size() \
                             .sort_values(ascending=False).head(8)
                if len(top_rep):
                    answer.append("   most-hit elements: " +
                                  ", ".join(f"{k}={int(v)}" for k, v in top_rep.items()))
                ctx = sub.groupby("gene_context").size().reindex(
                    ["CDS", "UTR", "ncRNA_exon", "promoter", "intron", "intergenic"]).dropna()
                answer.append("   gene context: " +
                              ", ".join(f"{k}={int(v)}" for k, v in ctx.items()))

                on_fam = sub.repFamily.isin(target_fams) & (sub.repFamily != "")
                exonic = sub[sub.hits_coding_exon]
                bad = sub[sub.hits_coding_exon & ~on_fam]
                if bad.empty:
                    answer.append("   >> NO coding-exon hits outside the target repeat "
                                  "family: nothing lands in a gene's coding sequence "
                                  "except via the TE family itself.")
                else:
                    answer.append(f"   >> {len(bad)} coding-exon hit(s) OUTSIDE the family "
                                  "--- these are the ones to check:")
                if len(exonic):
                    answer.append(f"   all {len(exonic)} coding-exon/UTR hit(s) "
                                  "(on-family ones are TE-derived exons):")
                    for r in exonic.sort_values(["n_mm"]).head(30).itertuples(index=False):
                        tag = "on-family" if (r.repFamily in target_fams and r.repFamily) \
                              else "OFF-FAMILY"
                        answer.append(f"      [{tag}] {r.chrom}:{r.start+1}-{r.end} {r.strand} "
                                      f"{r.n_mm}mm {r.gene_context} {r.genes} "
                                      f"[{r.repName or 'non-repeat'}] {r.guide}")
                nr = sub[(sub.repName == "") & (sub.n_mm <= 2)]
                answer.append(f"   unique (non-repeat) sites with <=2 mismatches: {len(nr)}"
                              + (" --- the classic off-target class" if len(nr) else ""))
                for r in nr.sort_values("n_mm").head(15).itertuples(index=False):
                    answer.append(f"      {r.chrom}:{r.start+1}-{r.end} {r.strand} "
                                  f"{r.n_mm}mm {r.gene_context} {r.genes or '-'} {r.guide}")
                answer.append("")

            # control: genome-wide on-family sites vs copy-CSV coverage
            fam_sites = sites[sites.repName.str.lower() == fam.lower()]
            if len(fam_sites):
                per = fam_sites.groupby("guide").size()
                answer.append("-- control: genome sites inside rmsk " + fam +
                              " intervals vs copies covered from the CSV")
                for _, r in gt.iterrows():
                    answer.append(f"   {r['guide']}  genome={int(per.get(r['guide'], 0)):>5}"
                                  f"   copies={int(r['cov_mm2']):>5}")
                answer.append("   (genome >= copies is expected: the genome scan allows "
                              "mismatches anywhere, the copy scan only PAM-distal ones)")
                answer.append("")

    txt = "\n".join(answer)
    (out / "ANSWER.txt").write_text(txt)
    _pp(f"Wrote {out/'ANSWER.txt'}")
    print("\n" + txt)


if __name__ == "__main__":
    main()
