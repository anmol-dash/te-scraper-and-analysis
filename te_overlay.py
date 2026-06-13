#!/usr/bin/env python3
"""
te_overlay.py --- GAMECA shared genomic-overlay helpers for annotation/liftover modules.

Common BED I/O + bedtools / liftOver wrappers used by the external-data analyses
(epigenetic overlay #4, ortholog insertion #5, multi-assembly liftover #7,
CTCF/TAD #12). Every helper degrades gracefully: if a tool or file is missing it
returns an empty/honest result rather than raising, so the analysis still emits a
valid (and truthful) report noting that the annotation was unavailable.
"""

from __future__ import annotations

import datetime
import gzip
import json
import os
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


def pp(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def texesc(s) -> str:
    """Escape LaTeX special characters in a free-text string (e.g. family names)."""
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


def load_loci(csv_path: str) -> pd.DataFrame:
    """Load a copies CSV and normalise chr/start/stop/strand/name columns."""
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    ren = {}
    for cand in ("chr", "chrom", "chromosome", "#chrom", "te_chromosome", "genoName"):
        if cand in df.columns:
            ren[cand] = "chr"; break
    for cand in ("start", "chromStart", "genoStart", "te_start"):
        if cand in df.columns:
            ren[cand] = "start"; break
    for cand in ("stop", "end", "chromEnd", "genoEnd", "te_end"):
        if cand in df.columns:
            ren[cand] = "stop"; break
    df = df.rename(columns=ren)
    if "strand" not in df.columns:
        df["strand"] = "+"
    if "name" not in df.columns:
        for c in ("TE_name", "locus", "repName"):
            if c in df.columns:
                df["name"] = df[c]; break
        else:
            df["name"] = [f"locus_{i}" for i in range(len(df))]
    return df


def has_coords(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in ("chr", "start", "stop")) and len(df) > 0


def write_bed(df: pd.DataFrame, path: Path):
    cols = df[["chr", "start", "stop", "name", "strand"]].copy()
    cols.insert(3, "score", 0)
    cols.columns = ["chr", "start", "stop", "name", "score", "strand"]
    cols.to_csv(path, sep="\t", header=False, index=False)


def bedtools_intersect_count(loci_bed: Path, ann_bed: str) -> dict:
    """Return {locus_name: overlap_count} for loci overlapping the annotation BED."""
    if not have("bedtools") or not Path(ann_bed).exists():
        return {}
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out.txt"
        try:
            res = subprocess.run(
                ["bedtools", "intersect", "-a", str(loci_bed), "-b", ann_bed, "-c"],
                capture_output=True, text=True)
        except Exception as e:                                   # noqa: BLE001
            pp(f"    bedtools intersect failed: {e}")
            return {}
        if res.returncode != 0:
            pp(f"    bedtools intersect returncode {res.returncode}: {res.stderr[:160]}")
            return {}
        counts = {}
        for line in res.stdout.splitlines():
            f = line.split("\t")
            if len(f) >= 7:
                counts[f[3]] = int(f[-1])
        return counts


def bedtools_closest_distance(loci_bed: Path, ref_bed: str) -> dict:
    """Return {locus_name: distance_to_nearest_feature} using bedtools closest -d."""
    if not have("bedtools") or not Path(ref_bed).exists():
        return {}
    with tempfile.TemporaryDirectory() as td:
        sorted_loci = Path(td) / "loci.sorted.bed"
        sorted_ref = Path(td) / "ref.sorted.bed"
        try:
            subprocess.run(f"sort -k1,1 -k2,2n {loci_bed} > {sorted_loci}",
                           shell=True, check=True)
            subprocess.run(f"sort -k1,1 -k2,2n {ref_bed} > {sorted_ref}",
                           shell=True, check=True)
            res = subprocess.run(
                ["bedtools", "closest", "-d", "-a", str(sorted_loci), "-b", str(sorted_ref)],
                capture_output=True, text=True)
        except Exception as e:                                   # noqa: BLE001
            pp(f"    bedtools closest failed: {e}")
            return {}
        if res.returncode != 0:
            return {}
        dist = {}
        for line in res.stdout.splitlines():
            f = line.split("\t")
            if len(f) >= 2 and f[-1].lstrip("-").isdigit():
                dist[f[3]] = int(f[-1])
        return dist


def liftover(loci_bed: Path, chain: str, liftover_cmd: str = "liftOver") -> tuple:
    """Run UCSC liftOver. Returns (mapped_names:set, unmapped_names:set, ran:bool)."""
    if not have(liftover_cmd) or not Path(chain).exists():
        return set(), set(), False
    with tempfile.TemporaryDirectory() as td:
        mapped = Path(td) / "mapped.bed"
        unmapped = Path(td) / "unmapped.bed"
        try:
            res = subprocess.run(
                [liftover_cmd, str(loci_bed), chain, str(mapped), str(unmapped)],
                capture_output=True, text=True)
        except Exception as e:                                   # noqa: BLE001
            pp(f"    liftOver failed: {e}")
            return set(), set(), False
        _ = res
        m = {ln.split("\t")[3] for ln in mapped.read_text().splitlines()
             if len(ln.split("\t")) >= 4} if mapped.exists() else set()
        u = {ln.split("\t")[3] for ln in unmapped.read_text().splitlines()
             if not ln.startswith("#") and len(ln.split("\t")) >= 4} \
            if unmapped.exists() else set()
        return m, u, True


def parse_named_paths(items: list) -> dict:
    """Parse ['name=path', ...] into {name: path}; bare paths get basename keys."""
    out = {}
    for it in items or []:
        if "=" in it:
            k, v = it.split("=", 1)
            out[k.strip()] = v.strip()
        else:
            out[Path(it).stem] = it
    return out


# ── download / preset helpers ─────────────────────────────────────────────────

_CACHE_DIR = Path(os.environ.get("GAMECA_CACHE", str(Path.home() / ".gameca_cache")))


def _cache(subdir: str = "") -> Path:
    d = _CACHE_DIR / subdir if subdir else _CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_if_absent(url: str, dest: Path, label: str,
                       gunzip: bool | None = None) -> bool:
    """Download url -> dest, gunzipping if inferred or requested. Returns success."""
    if dest.exists() and dest.stat().st_size > 0:
        pp(f"  (cached) {label}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    do_gz = (url.endswith(".gz") and not str(dest).endswith(".gz")) \
        if gunzip is None else gunzip
    tmp = dest.with_name(dest.name + ".dl")
    try:
        pp(f"  Downloading {label} ...")
        urllib.request.urlretrieve(url, tmp)
        if do_gz:
            with gzip.open(tmp, "rb") as fin, open(dest, "wb") as fout:
                shutil.copyfileobj(fin, fout)
            tmp.unlink()
        else:
            tmp.rename(dest)
        pp(f"    -> {dest.name} ({dest.stat().st_size // 1024:,} KB)")
        return True
    except Exception as e:
        pp(f"  WARNING: download failed for {label}: {e}")
        for p in (tmp, dest):
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
        return False


def encode_search_url(assay_title: str, biosample: str,
                      target_label: str = "", assembly: str = "GRCh38") -> str | None:
    """Query ENCODE REST API; return download URL of best released BED or None."""
    params: dict = {
        "type": "File",
        "assay_title": assay_title,
        "biosample_term_name": biosample,
        "file_format": "bed",
        "output_type": "peaks",
        "assembly": assembly,
        "status": "released",
        "limit": "1",
        "format": "json",
    }
    if target_label:
        params["target.label"] = target_label
    api_url = "https://www.encodeproject.org/search/?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(api_url, headers={
            "Accept": "application/json",
            "User-Agent": "GAMECA/0.4 genomics-research-tool",
        })
        with urllib.request.urlopen(req) as r:
            data = json.loads(r.read())
        hits = data.get("@graph", [])
        if not hits:
            return None
        acc = hits[0].get("accession", "")
        if not acc:
            return None
        return f"https://www.encodeproject.org/files/{acc}/@@download/{acc}.bed.gz"
    except Exception as e:
        pp(f"  WARNING: ENCODE API query failed: {e}")
        return None


def ucsc_cpg_to_bed(gz_path: Path, bed_path: Path) -> bool:
    """Convert UCSC cpgIslandExt.txt.gz -> 4-col BED (handles optional bin column)."""
    try:
        with gzip.open(gz_path, "rt") as fin, open(bed_path, "w") as fout:
            for line in fin:
                f = line.strip().split("\t")
                if not f or f[0].startswith("#"):
                    continue
                if f[0].lstrip("-").isdigit():      # has leading bin column
                    chrom, s, e = f[1], f[2], f[3]
                    name = f[4] if len(f) > 4 else "CpGisland"
                else:
                    chrom, s, e = f[0], f[1], f[2]
                    name = f[3] if len(f) > 3 else "CpGisland"
                fout.write(f"{chrom}\t{s}\t{e}\t{name}\n")
        return True
    except Exception as e:
        pp(f"  WARNING: CpG island conversion failed: {e}")
        return False


def rao2014_to_bed(gz_path: Path, bed_path: Path) -> bool:
    """Convert Rao 2014 Arrowhead domain list -> TAD-boundary BED (1 kb at each edge)."""
    try:
        with gzip.open(gz_path, "rt") as fin, open(bed_path, "w") as fout:
            for i, line in enumerate(fin):
                if i == 0:
                    continue                         # header row
                f = line.strip().split("\t")
                if len(f) < 3:
                    continue
                chrom, x1_s, x2_s = f[0], f[1], f[2]
                if not chrom.startswith("chr"):
                    chrom = f"chr{chrom}"
                try:
                    x1, x2 = int(x1_s), int(x2_s)
                except ValueError:
                    continue
                fout.write(f"{chrom}\t{x1}\t{min(x1 + 1000, x2)}\tL\t0\t.\n")
                fout.write(f"{chrom}\t{max(x2 - 1000, x1)}\t{x2}\tR\t0\t.\n")
        return True
    except Exception as e:
        pp(f"  WARNING: Rao 2014 conversion failed: {e}")
        return False


_ASSEMBLY_ALIASES: dict = {
    "t2t": "hs1", "T2T": "hs1", "chm13": "hs1", "CHM13": "hs1",
}

_TAD_URLS: dict = {
    "K562": ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/"
             "GSE63525_K562_Arrowhead_domainlist.txt.gz"),
    "GM12878": ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/"
                "GSE63525_GM12878_primary%2Breplicate_Arrowhead_domainlist.txt.gz"),
    "IMR90": ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/"
              "GSE63525_IMR90_Arrowhead_domainlist.txt.gz"),
}


def fetch_chain(target: str, src_asm: str = "hg38") -> Path | None:
    """Download + cache UCSC liftOver chain; returns unzipped path or None."""
    target = _ASSEMBLY_ALIASES.get(target, target)
    ucsc_target = target[0].upper() + target[1:]
    chain_name = f"{src_asm}To{ucsc_target}.over.chain"
    dest = _cache("chains") / chain_name
    if dest.exists() and dest.stat().st_size > 0:
        pp(f"  (cached) chain {target}")
        return dest
    gz_dest = dest.with_name(chain_name + ".gz")
    url = (f"https://hgdownload.soe.ucsc.edu/goldenPath/{src_asm}/"
           f"liftOver/{chain_name}.gz")
    if not download_if_absent(url, gz_dest, f"chain {src_asm}->{target}", gunzip=False):
        return None
    try:
        with gzip.open(gz_dest, "rb") as fin, open(dest, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        return dest
    except Exception as e:
        pp(f"  WARNING: could not gunzip chain: {e}")
        return None


def fetch_epigenetic_preset(cell_type: str) -> dict:
    """Auto-download H3K9me3 + ATAC peaks + CpG islands for cell_type. Returns {name: path}."""
    pp(f"  Fetching epigenetic preset: {cell_type}")
    cd = _cache(f"epigenetic/{cell_type}")
    tracks: dict = {}

    for mark, assay, target in [
        ("H3K9me3", "Histone ChIP-seq", "H3K9me3"),
        ("ATAC",    "ATAC-seq",          ""),
    ]:
        dest = cd / f"{mark}_peaks.bed"
        if not (dest.exists() and dest.stat().st_size > 0):
            url = encode_search_url(assay, cell_type, target_label=target)
            if url:
                download_if_absent(url, dest, f"{mark} {cell_type}")
            else:
                pp(f"  WARNING: ENCODE returned no {mark} file for {cell_type}")
        if dest.exists() and dest.stat().st_size > 0:
            tracks[mark] = str(dest)

    cpg_gz  = _cache("ucsc") / "cpgIslandExt.txt.gz"
    cpg_bed = _cache("ucsc") / "cpgIslandExt.bed"
    if not (cpg_bed.exists() and cpg_bed.stat().st_size > 0):
        ok = download_if_absent(
            "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cpgIslandExt.txt.gz",
            cpg_gz, "CpG islands (UCSC hg38)", gunzip=False)
        if ok:
            ucsc_cpg_to_bed(cpg_gz, cpg_bed)
    if cpg_bed.exists() and cpg_bed.stat().st_size > 0:
        tracks["CpG"] = str(cpg_bed)

    return tracks


def fetch_ctcf_preset(cell_type: str) -> Path | None:
    """Download CTCF ChIP-seq narrowPeak BED for cell_type from ENCODE. Returns path or None."""
    pp(f"  Fetching CTCF ChIP preset: {cell_type}")
    dest = _cache(f"ctcf/{cell_type}") / "CTCF_peaks.bed"
    if dest.exists() and dest.stat().st_size > 0:
        pp(f"  (cached) CTCF {cell_type}")
        return dest
    url = encode_search_url("TF ChIP-seq", cell_type, target_label="CTCF")
    if url and download_if_absent(url, dest, f"CTCF {cell_type}"):
        return dest
    pp(f"  WARNING: could not fetch CTCF peaks for {cell_type}")
    return None


def fetch_tad_preset(cell_type: str) -> Path | None:
    """Download + convert Rao 2014 TAD domains -> boundary BED (hg19). Returns path or None."""
    if cell_type not in _TAD_URLS:
        pp(f"  WARNING: no TAD preset for '{cell_type}'. Choices: {list(_TAD_URLS)}")
        return None
    cd  = _cache(f"tads/{cell_type}")
    gz  = cd / f"rao2014_{cell_type}.txt.gz"
    bed = cd / f"rao2014_{cell_type}_boundaries.bed"
    if not (gz.exists() and gz.stat().st_size > 0):
        download_if_absent(_TAD_URLS[cell_type], gz,
                           f"Rao 2014 TADs {cell_type}", gunzip=False)
    if gz.exists() and not (bed.exists() and bed.stat().st_size > 0):
        pp(f"  Converting {cell_type} TAD domains -> boundary BED (hg19 coords)...")
        rao2014_to_bed(gz, bed)
    if bed.exists() and bed.stat().st_size > 0:
        pp("  NOTE: Rao 2014 TADs are hg19 coordinates; "
           "for exact results supply --tads with an hg38-lifted file.")
        return bed
    return None
