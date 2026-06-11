#!/usr/bin/env python3
"""
bb_to_sorted_bed.py  —  Convert a bigBed (.bb) into a sorted, bgzipped BED.

    python3 bb_to_sorted_bed.py INPUT.bb --out-dir DIR [--output NAME.bed.gz]

Output:  <out-dir>/<NAME>           (default NAME = <input basename>.sorted.bed.gz)

Pipeline (parallel where possible):
  1. Extract bigBed → BED, one chromosome per worker (parallel) using
     `bigBedToBed -chrom=...`, with a pybigtools fallback (no binary / glibc
     needed) and a whole-file extraction as a last resort.
  2. Sort by chrom,start with GNU `sort --parallel=N -S <mem>`.
  3. Compress with `bgzip -@ N` (falls back to gzip).
  4. Optionally tabix-index the result (--tabix).

Requires one of: `bigBedToBed` on PATH / --bigbedtobed, or `pip install pybigtools`.
`bigBedInfo` is used when present to list chromosomes for parallel extraction.
"""

import argparse
import concurrent.futures as cf
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _which(name, override=""):
    if override:
        return shutil.which(override) or (override if Path(override).exists() else None)
    return shutil.which(name)


def _chroms_via_bigbedinfo(bigbedinfo, bb):
    """Return a list of chromosome names via `bigBedInfo -chromList`, or []."""
    if not bigbedinfo:
        return []
    try:
        r = subprocess.run([bigbedinfo, "-chromList", str(bb)],
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            return []
        chroms = []
        for line in r.stdout.splitlines():
            line = line.strip()
            # rows look like "chr1 0 248956422"; skip the header lines
            if not line or line.startswith(("chromCount", "Chrom", "#")):
                continue
            tok = line.split()
            if tok and not tok[0].endswith(":"):
                chroms.append(tok[0])
        return chroms
    except Exception:
        return []


def _extract_one_chrom(bigbedtobed, bb, chrom, dest):
    """Extract a single chromosome to `dest` BED via bigBedToBed. Returns (chrom, ok, err)."""
    cmd = [bigbedtobed, str(bb), str(dest), f"-chrom={chrom}"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return chrom, False, (r.stderr or "")[:300]
        return chrom, True, ""
    except Exception as exc:
        return chrom, False, str(exc)


def _extract_with_binary(bigbedtobed, bigbedinfo, bb, work, threads):
    """Extract the bigBed to per-chromosome BED files in `work`. Returns list of paths."""
    chroms = _chroms_via_bigbedinfo(bigbedinfo, bb)

    # Whole-file extraction when we can't enumerate chromosomes (still works).
    if not chroms:
        print("  No chromosome list available — extracting whole file (single stream).")
        dest = work / "all.bed"
        r = subprocess.run([bigbedtobed, str(bb), str(dest)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"bigBedToBed failed: {(r.stderr or '')[:400]}")
        return [dest]

    print(f"  Extracting {len(chroms)} chromosomes with {threads} parallel workers ...")
    out_paths, errors = [], []
    with cf.ThreadPoolExecutor(max_workers=threads) as ex:
        futs = {ex.submit(_extract_one_chrom, bigbedtobed, bb, c, work / f"{c}.bed"): c
                for c in chroms}
        for fut in cf.as_completed(futs):
            chrom, ok, err = fut.result()
            dest = work / f"{chrom}.bed"
            if ok and dest.exists() and dest.stat().st_size > 0:
                out_paths.append(dest)
            elif ok:
                pass  # empty chromosome — nothing to keep
            else:
                errors.append(f"{chrom}: {err}")
    if errors and not out_paths:
        raise RuntimeError("bigBedToBed failed for all chromosomes; first error:\n  "
                           + errors[0])
    if errors:
        print(f"  WARNING: {len(errors)} chromosome(s) failed (continuing): "
              f"{errors[0][:160]}")
    return out_paths


def _extract_with_pybigtools(bb, work, threads):
    """Fallback extraction using pybigtools (pure Rust — no binary / glibc needed)."""
    import pybigtools  # noqa: PLC0415

    bw = pybigtools.open(str(bb), "r")
    chroms = list(bw.chroms().items())  # {chrom: length}
    print(f"  pybigtools: extracting {len(chroms)} chromosomes with {threads} workers ...")

    def _one(item):
        chrom, length = item
        dest = work / f"{chrom}.bed"
        n = 0
        try:
            bb_local = pybigtools.open(str(bb), "r")  # one handle per thread
            with open(dest, "w") as fh:
                # .records() yields (start, end, *rest) for bigBed; rest fields
                # (name, score, strand, …) are split by whitespace.
                for rec in bb_local.records(chrom, 0, int(length)):
                    s, e = rec[0], rec[1]
                    rest = "\t".join(str(x) for x in rec[2:])
                    fh.write(f"{chrom}\t{s}\t{e}" + (f"\t{rest}" if rest else "") + "\n")
                    n += 1
            return dest if n else None
        except Exception as exc:
            print(f"  WARNING: pybigtools failed on {chrom}: {exc}")
            return None

    out = []
    with cf.ThreadPoolExecutor(max_workers=threads) as ex:
        for dest in ex.map(_one, chroms):
            if dest:
                out.append(dest)
    if not out:
        raise RuntimeError("pybigtools produced no records.")
    return out


def _advise_no_backend():
    """Print actionable guidance when neither bigBedToBed nor pybigtools works."""
    sys.exit(
        "\nERROR: no usable bigBed backend on this machine.\n"
        "  • bigBedToBed needs glibc >= 2.29 (absent on old HPC nodes), and\n"
        "  • pybigtools is not installed (its prebuilt wheel needs glibc >= 2.28,\n"
        "    and building from source needs gcc >= 4.9 + Rust).\n"
        "\nFix one of:\n"
        "  A) On the cluster, load a modern compiler then build pybigtools:\n"
        "       module load gcc/9.3.0   # any gcc >= 4.9\n"
        "       export CC=$(which gcc) CXX=$(which g++)\n"
        "       pip install pybigtools\n"
        "  B) Run this script on a machine with a modern toolchain (e.g. your\n"
        "     laptop: `pip install pybigtools`), then copy the .sorted.bed.gz back.\n"
        "  C) Skip the bigBed: use fetch_jaspar.py (per-motif CMMT download — no\n"
        "     binary/compiler needed) to produce the same sorted.bed.gz.\n"
    )


def main():
    ap = argparse.ArgumentParser(description="Convert a bigBed (.bb) to a sorted .bed.gz")
    ap.add_argument("input", help="Path to the input .bb (bigBed) file.")
    ap.add_argument("--out-dir", required=True, help="Directory for the output file.")
    ap.add_argument("--output", default=None,
                    help="Output filename (default: <input basename>.sorted.bed.gz).")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4,
                    help="Parallelism for extraction / sort / bgzip (default: all cores).")
    ap.add_argument("--sort-mem", default="2G",
                    help="Memory buffer for sort -S (default: 2G).")
    ap.add_argument("--bigbedtobed", default="",
                    help="Path to bigBedToBed (else found on PATH).")
    ap.add_argument("--bigbedinfo", default="",
                    help="Path to bigBedInfo (else found on PATH).")
    ap.add_argument("--tabix", action="store_true", help="Also create a .tbi tabix index.")
    ap.add_argument("--keep-tmp", action="store_true", help="Keep intermediate BED files.")
    args = ap.parse_args()

    bb = Path(args.input)
    if not bb.exists():
        sys.exit(f"ERROR: input not found: {bb}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.output or (bb.stem + ".sorted.bed.gz")
    if not name.endswith(".gz"):
        name += ".gz"
    out_path = out_dir / name

    threads = max(1, args.threads)
    bigbedtobed = _which("bigBedToBed", args.bigbedtobed)
    bigbedinfo = _which("bigBedInfo", args.bigbedinfo)

    print("=" * 64)
    print(f"bigBed → sorted.bed.gz")
    print(f"  input   : {bb}  ({bb.stat().st_size/1e6:.1f} MB)")
    print(f"  output  : {out_path}")
    print(f"  threads : {threads}   sort-mem : {args.sort_mem}")
    print(f"  tools   : bigBedToBed={bigbedtobed or 'none'}  bigBedInfo={bigbedinfo or 'none'}")
    print("=" * 64)

    t0 = time.time()
    work = Path(tempfile.mkdtemp(prefix="bb2bed_", dir=out_dir))
    try:
        # ── 1. Extract bigBed → per-chrom BEDs (parallel) ────────────────────
        def _pybigtools_or_advise():
            try:
                return _extract_with_pybigtools(bb, work, threads)
            except ImportError:
                _advise_no_backend()

        if bigbedtobed:
            try:
                bed_parts = _extract_with_binary(bigbedtobed, bigbedinfo, bb, work, threads)
            except Exception as exc:
                print(f"  bigBedToBed extraction failed ({exc}) — trying pybigtools ...")
                bed_parts = _pybigtools_or_advise()
        else:
            print("  bigBedToBed not found — trying pybigtools fallback.")
            bed_parts = _pybigtools_or_advise()

        print(f"  Extracted {len(bed_parts)} BED part(s) in {time.time()-t0:.1f}s")

        # ── 2 + 3. Sort (parallel) and bgzip (parallel) in one stream ────────
        sort_bin = shutil.which("sort")
        bgzip = shutil.which("bgzip")
        if not sort_bin:
            sys.exit("ERROR: `sort` not found on PATH.")

        sort_cmd = [sort_bin, "-k1,1", "-k2,2n",
                    f"--parallel={threads}", "-S", args.sort_mem, "-T", str(work)]
        env = dict(os.environ, LC_ALL="C")  # byte-wise, fast, deterministic

        print(f"  Sorting (parallel={threads}) and compressing "
              f"({'bgzip -@'+str(threads) if bgzip else 'gzip'}) ...")
        with open(out_path, "wb") as out_fh:
            # cat parts → sort → bgzip/gzip
            cat = subprocess.Popen(["cat", *map(str, bed_parts)], stdout=subprocess.PIPE)
            srt = subprocess.Popen(sort_cmd, stdin=cat.stdout,
                                   stdout=subprocess.PIPE, env=env)
            cat.stdout.close()
            if bgzip:
                comp = subprocess.Popen([bgzip, "-@", str(threads), "-c"],
                                        stdin=srt.stdout, stdout=out_fh)
            else:
                comp = subprocess.Popen(["gzip", "-c"], stdin=srt.stdout, stdout=out_fh)
            srt.stdout.close()
            comp.communicate()
            srt.wait()
            cat.wait()
            for proc, label in ((cat, "cat"), (srt, "sort"), (comp, "compress")):
                if proc.returncode not in (0, None):
                    sys.exit(f"ERROR: {label} stage failed (rc={proc.returncode}).")

        # ── 4. Optional tabix index ──────────────────────────────────────────
        if args.tabix:
            tabix = shutil.which("tabix")
            if not tabix:
                print("  WARNING: --tabix requested but tabix not found; skipping index.")
            elif not bgzip:
                print("  WARNING: tabix needs a bgzip-compressed file; gzip was used — skipping.")
            else:
                r = subprocess.run([tabix, "-p", "bed", str(out_path)],
                                   capture_output=True, text=True)
                if r.returncode == 0:
                    print(f"  Tabix index: {out_path}.tbi")
                else:
                    print(f"  WARNING: tabix failed: {(r.stderr or '')[:200]}")
    finally:
        if not args.keep_tmp:
            shutil.rmtree(work, ignore_errors=True)
        else:
            print(f"  (kept intermediates in {work})")

    size_mb = out_path.stat().st_size / 1e6
    print("=" * 64)
    print(f"DONE in {time.time()-t0:.1f}s → {out_path}  ({size_mb:.1f} MB)")
    print("=" * 64)


if __name__ == "__main__":
    main()
