#!/usr/bin/env python3
"""
fetch_jaspar.py  —  Pre-fetch & cache genome-wide JASPAR 2022 sorted BED files.

Downloads the full per-motif CMMT JASPAR 2022 track for one or more genome
builds and writes a single sorted, cached BED per build:

    <cache-dir>/JASPAR2022_<build>.sorted.bed.gz

These caches are GENOME-WIDE (no TE-loci filter), so every future pipeline run
— for any family — reuses the same file with a single read. Point the pipeline
at the cache with  --jaspar-dir <cache-dir>  (or env TE_JASPAR_<BUILD>=<file>).

NO TIME LIMIT: the download runs to completion however long it takes (the CMMT
per-motif fallback in te_motif has no overall cap).

Parallelism:
  • --parallel-builds N   run N builds at once (default 1; the submit script
                          runs one build per LSF job, i.e. four jobs in parallel)
  • --workers N           parallel HTTP workers WITHIN each build (default 4)

Download elsewhere:
  • --cache-dir PATH      put the .bed.gz caches anywhere (shared scratch, a
                          project dir, a different filesystem, your laptop…)
  • --source-base-url URL pull from an alternate JASPAR/CMMT mirror

Examples:
    # one build, on a node/laptop with network
    python3 fetch_jaspar.py --build hg38 --cache-dir /shared/jaspar_cache

    # all four builds, 2 at a time, 4 HTTP workers each
    python3 fetch_jaspar.py --builds hg38 hg19 mm10 mm39 \\
        --parallel-builds 2 --workers 4 --cache-dir /shared/jaspar_cache
"""

import argparse
import concurrent.futures as cf
import os
import sys
import time
from pathlib import Path

# The four standard mammalian builds the pipeline uses.
DEFAULT_BUILDS = ["hg38", "hg19", "mm10", "mm39"]


def _fetch_one(build, cache_dir, workers):
    """Download (or reuse) the genome-wide JASPAR sorted BED for one build."""
    import te_motif  # heavy imports (pandas/matplotlib) — done lazily per call

    os.environ["TE_CMMT_WORKERS"] = str(workers)
    cache_dir = Path(cache_dir)
    out = cache_dir / f"JASPAR2022_{build}.sorted.bed.gz"

    # Cache hit → single read, no re-download.
    if out.exists():
        try:
            if te_motif.validate_jaspar_bed(str(out)):
                size_mb = out.stat().st_size / 1e6
                print(f"[{build}] CACHED ({size_mb:.1f} MB) — valid, skipping: {out}",
                      flush=True)
                return build, str(out), True
            print(f"[{build}] cached file invalid — re-downloading: {out}", flush=True)
            out.unlink(missing_ok=True)
        except Exception as exc:
            print(f"[{build}] cache validation errored ({exc}); re-downloading", flush=True)

    t0 = time.time()
    print(f"[{build}] downloading genome-wide JASPAR 2022 (workers={workers}) → {out}",
          flush=True)
    try:
        path = te_motif._download_cmmt_per_motif(build, str(cache_dir), loci_bed=None)
    except Exception as exc:
        print(f"[{build}] FAILED with exception: {exc}", flush=True)
        return build, None, False

    ok = bool(path) and Path(path).exists()
    dt = time.time() - t0
    if ok:
        print(f"[{build}] DONE in {dt/60:.1f} min → {path} "
              f"({Path(path).stat().st_size/1e6:.1f} MB)", flush=True)
    else:
        print(f"[{build}] FAILED after {dt/60:.1f} min", flush=True)
    return build, path, ok


def main():
    ap = argparse.ArgumentParser(
        description="Fetch & cache genome-wide JASPAR 2022 sorted BED files.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--build", help="Single build (hg38/hg19/mm10/mm39/…).")
    g.add_argument("--builds", nargs="+", default=None,
                   help=f"Builds to fetch (default: {' '.join(DEFAULT_BUILDS)}).")
    ap.add_argument("--cache-dir", required=True,
                    help="Directory to write JASPAR2022_<build>.sorted.bed.gz into. "
                         "Can be any location (shared scratch, project dir, laptop).")
    ap.add_argument("--parallel-builds", type=int, default=1,
                    help="How many builds to download concurrently (default 1).")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel HTTP workers within each build (default 4).")
    ap.add_argument("--source-base-url", default=None,
                    help="Override the CMMT base URL to use an alternate mirror.")
    args = ap.parse_args()

    builds = [args.build] if args.build else (args.builds or DEFAULT_BUILDS)

    if args.source_base_url:
        import te_motif
        te_motif.CMMT_BASE_URL = args.source_base_url.rstrip("/")
        print(f"Source mirror overridden → {te_motif.CMMT_BASE_URL}", flush=True)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64, flush=True)
    print(f"JASPAR pre-fetch  |  builds={builds}  cache={cache_dir}", flush=True)
    print(f"  parallel-builds={args.parallel_builds}  workers/build={args.workers}",
          flush=True)
    print("  NO TIME LIMIT — runs to completion.", flush=True)
    print("=" * 64, flush=True)

    results = []
    if args.parallel_builds <= 1 or len(builds) == 1:
        for b in builds:
            results.append(_fetch_one(b, cache_dir, args.workers))
    else:
        with cf.ThreadPoolExecutor(max_workers=args.parallel_builds) as ex:
            futs = [ex.submit(_fetch_one, b, cache_dir, args.workers) for b in builds]
            for fut in cf.as_completed(futs):
                results.append(fut.result())

    print("\n" + "=" * 64, flush=True)
    print("SUMMARY", flush=True)
    for b, path, ok in sorted(results):
        print(f"  {b:8s} {'OK   ' + str(path) if ok else 'FAILED'}", flush=True)
    print("=" * 64, flush=True)

    sys.exit(0 if all(ok for _, _, ok in results) else 1)


if __name__ == "__main__":
    main()
