#!/usr/bin/env python3
"""
run_gameca_cluster_test.py — comprehensive multi-family / multi-assembly GAMECA
integration test, designed to run on the HPC cluster.

What it exercises (end to end, for every (family, assembly) in the matrix):
  1. UCSC pull (retrieval) — te_prep.py downloads the rmsk track and fetches
                        per-locus sequences straight from the UCSC browser
                        (api.genome.ucsc.edu), so the live fetch path is tested.
                        Timed separately from everything below.
  2. Clustering       — te_clustering.py assigns real clusters; if that yields
                        <2 clusters (small families) we fall back to deterministic
                        synthetic clusters so cluster-aware modules still run.
  3. Synthetic expr   — we inject SYNTHETIC expression columns with deliberately
                        varied, non-developmental sample names and mixed encodings
                        (integer counts, numeric strings, SQUIRE-style list cells)
                        to prove te_expression.py tolerates arbitrary sample names
                        — NOT just "two cell / four cell" style stage names.
                        These numbers are fabricated test fixtures ONLY; they are
                        never reported as real measurements.
  4. Stage 11         — every standout-analysis module runs on the prepared CSV.
                        Nextflow + Singularity are the baseline for this test,
                        same as every regular query.py job hpc_client submits
                        (hpc_client._pipeline_cli_args() always passes
                        --post-alignment-analyses-nextflow, and CONTAINER_SIF
                        is auto-provisioned at connect time). --stage11-engine
                        defaults to 'nextflow' (nextflow/post_alignment_analyses.nf,
                        real per-module parallelism) and --container-sif
                        auto-detects gameca.sif from $GAMECA_SIF / ./gameca.sif /
                        ~/gameca/gameca.sif — the same cache path hpc_client
                        uses — so this test doesn't run against a different
                        environment than the regular pipeline. nextflow itself
                        is self-installed into ~/gameca/bin if not already on
                        PATH (mirrors hpc_client._nextflow_setup_block).
                        Pass --stage11-engine loop to use the in-process
                        subprocess loop instead (query.py's own fallback path
                        when nextflow is unavailable).

Timing: retrieval (UCSC pull), clustering, synthetic-expression injection, and
Stage 11 (broken into per-module seconds, with a "Stage 11 excl. fold" total)
are all captured separately per family and written to the master log + JSON
summary — not just the retrieval step. ColabFold (--include-fold) is excluded
from the "excl. fold" total since it is heavy/GPU and off by default.

A master log + JSON summary capture, per family and per module, OK/FAIL/SKIP and
timing, so a single output file tells you whether the whole program is healthy
across assemblies and families.

Usage (cluster):
    python run_gameca_cluster_test.py \
        --out-dir ~/gameca_cluster_test \
        --notify-email you@example.com
    # (nextflow + singularity are on by default; nothing extra to pass)

    # custom matrix / caps:
    python run_gameca_cluster_test.py --out-dir OUT \
        --matrix "L1HS:hg38,MT2_Mm:mm10" --max-loci 200 --include-fold

    # fall back to the in-process loop engine / no container, for comparison:
    python run_gameca_cluster_test.py --out-dir OUT --stage11-engine loop
"""

import argparse
import datetime
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Default test matrix: spans TE classes (LINE/SINE/LTR) and two assemblies. ──
# Kept small + youngish so UCSC fetch + MAFFT stay tractable under --max-loci.
DEFAULT_MATRIX = [
    ("L1HS",    "hg38"),   # LINE-1, human
    ("AluYa5",  "hg38"),   # SINE/Alu, human
    ("LTR5_Hs", "hg38"),   # LTR (HERV-K), human
    ("MT2_Mm",  "mm10"),   # LTR (ERVL-MaLR), mouse — canonical family
    ("B2_Mm2",  "mm10"),   # SINE/B2, mouse
]


def _run(cmd, log, cwd=None, env=None):
    """Run a subprocess, streaming combined output into open file *log*."""
    log.write(f"\n$ {' '.join(str(c) for c in cmd)}\n"); log.flush()
    t0 = time.time()
    rc = subprocess.run([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT,
                        cwd=cwd, env=env).returncode
    dt = time.time() - t0
    log.write(f"[exit {rc}  {dt:.0f}s]\n"); log.flush()
    return rc, dt


def _default_container_sif():
    """Auto-detect gameca.sif the same way hpc_client._ensure_container_sif does
    on a normal query.py run, so this test doesn't diverge from the regular
    pipeline's container path. Checked in order: $GAMECA_SIF env var, a local
    ./gameca.sif, then the conventional ~/gameca/gameca.sif cache location."""
    env = os.environ.get("GAMECA_SIF", "").strip()
    if env and Path(env).is_file():
        return env
    for candidate in (Path("gameca.sif"), Path.home() / "gameca" / "gameca.sif"):
        if candidate.is_file():
            return str(candidate)
    return None


def _java_major_version(java_bin):
    """Return the major version number of *java_bin*, or None if unusable."""
    try:
        out = subprocess.run([java_bin, "-version"], capture_output=True,
                             text=True, timeout=10).stderr
    except Exception:
        return None
    m = re.search(r'version "(\d+)(?:\.(\d+))?', out)
    if not m:
        return None
    major = int(m.group(1))
    return int(m.group(2)) if major == 1 and m.group(2) else major


def _ensure_java(min_major=17):
    """Find (or install) a Java >= min_major and return its JAVA_HOME.

    Mirrors query.py's _ensure_java(): HPC nodes commonly pin JAVA_HOME to an
    old conda-bundled JVM (Java 11) that modern Nextflow refuses to run
    under, even when a newer JVM exists elsewhere. Search common JVM
    locations first; fall back to a portable Eclipse Temurin JRE download
    into ~/gameca/jdk. Returns a JAVA_HOME path, or None if unavailable.
    """
    cached_jdk = Path.home() / "gameca" / "jdk"
    candidates = []
    java_home_env = os.environ.get("JAVA_HOME", "").strip()
    if java_home_env:
        candidates.append(str(Path(java_home_env) / "bin" / "java"))
    which_java = shutil.which("java")
    if which_java:
        candidates.append(which_java)
    for pattern in ("/usr/lib/jvm/*/bin/java", "/usr/local/lib/jvm/*/bin/java",
                    "/opt/*/bin/java", "/opt/*/*/bin/java",
                    str(cached_jdk / "*" / "bin" / "java")):
        candidates += glob.glob(pattern)

    for cand in dict.fromkeys(candidates):
        if not Path(cand).is_file():
            continue
        v = _java_major_version(cand)
        if v is not None and v >= min_major:
            return str(Path(cand).resolve().parent.parent)

    print(f"  [nextflow] no Java >= {min_major} found — installing a portable "
          "Temurin JRE ...")
    cached_jdk.mkdir(parents=True, exist_ok=True)
    tarball = cached_jdk / "jre.tar.gz"
    url = ("https://api.adoptium.net/v3/binary/latest/21/ga/linux/x64/jre/"
           "hotspot/normal/eclipse?project=jdk")
    rc = subprocess.run(["curl", "-sL", "-o", str(tarball), url]).returncode
    if rc != 0 or not tarball.exists() or tarball.stat().st_size == 0:
        print("  [nextflow] WARNING: portable JRE download failed")
        return None
    subprocess.run(["tar", "xzf", str(tarball), "-C", str(cached_jdk)])
    tarball.unlink(missing_ok=True)
    extracted = next((p for p in cached_jdk.iterdir()
                      if p.is_dir() and (p / "bin" / "java").is_file()), None)
    if extracted:
        return str(extracted)
    print("  [nextflow] WARNING: portable JRE extraction failed")
    return None


def _ensure_nextflow():
    """Self-install nextflow into ~/gameca/bin if it isn't already on PATH,
    and make sure a Java >= 17 is what nextflow will actually pick up.

    Mirrors hpc_client._nextflow_setup_block()/query.py's _ensure_nextflow()
    so a plain `nextflow not on PATH` (or wrong-Java) failure here doesn't
    mean this test exercises a different environment than the regular
    query.py --post-alignment-analyses-nextflow jobs hpc_client submits.
    JAVA_HOME/PATH are overridden even when nextflow was already found,
    since a stale conda-pinned JAVA_HOME is the actual failure mode this
    guards against. Returns the resolved nextflow path, or None if
    unavailable.
    """
    nf_dir = Path.home() / "gameca" / "bin"
    cached = nf_dir / "nextflow"
    nf = shutil.which("nextflow")
    if not nf and cached.is_file():
        os.environ["PATH"] = f"{nf_dir}:{os.environ.get('PATH', '')}"
        nf = str(cached)
    if not nf:
        nf_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [nextflow] not on PATH — installing self-contained launcher into {nf_dir} ...")
        rc = subprocess.run("curl -s https://get.nextflow.io | bash", shell=True,
                            cwd=str(nf_dir)).returncode
        if rc == 0 and cached.is_file():
            cached.chmod(0o755)
            os.environ["PATH"] = f"{nf_dir}:{os.environ.get('PATH', '')}"
            print(f"  [nextflow] installed at {cached}")
            nf = str(cached)
        else:
            print("  [nextflow] WARNING: self-install failed — nextflow engine unavailable")
            return None

    java_home = _ensure_java()
    if not java_home:
        print("  [nextflow] WARNING: no usable Java (>=17) found or installable")
        return None
    os.environ["JAVA_HOME"] = java_home
    os.environ["PATH"] = f"{java_home}/bin:{os.environ.get('PATH', '')}"
    return nf


def _singularity_wrap(cmd, container_sif):
    """Prefix *cmd* with `singularity exec <binds> <sif>` when a .sif is set.

    Binds the paths any argument in cmd points at (so relative-to-host paths
    resolve identically inside the container) plus the repo dir itself.
    """
    if not container_sif:
        return cmd
    runtime = shutil.which("singularity") or shutil.which("apptainer") or "singularity"
    binds = {str(Path(__file__).resolve().parent)}
    for c in cmd:
        p = Path(str(c))
        if p.is_absolute() and (p.exists() or p.parent.exists()):
            binds.add(str(p.parent if not p.exists() or p.is_file() else p))
    bind_args = []
    for b in sorted(binds):
        bind_args += ["-B", f"{b}:{b}"]
    return [runtime, "exec"] + bind_args + [container_sif] + cmd


def _synthetic_clusters(df, k=3):
    """Deterministic clusters from GC tertiles (falls back to row modulo).

    Ties cluster membership to a real sequence property so differential-
    expression tests have non-trivial structure even on synthetic input.
    """
    n = len(df)
    if "GC_Content" in df.columns and df["GC_Content"].nunique() >= k:
        try:
            return pd.qcut(df["GC_Content"].rank(method="first"), k,
                           labels=False).astype(int).to_numpy()
        except Exception:
            pass
    return (np.arange(n) % k).astype(int)


def _inject_synthetic_expression(df, seed):
    """Add SYNTHETIC expression columns with varied sample names + encodings.

    The point is to stress te_expression.py's sample-name tolerance, so the
    names are intentionally NOT developmental stages and the encodings vary:
      - integer counts            (K562_rep1, K562_rep2)
      - numeric strings           (Liver_donorA)
      - float values              (siCTRL_24h, patient_07_tumor)
      - SQUIRE-style list cells   (GM12878.scRNA)  → "['AAAC..','CCGT..']"
    A weak cluster-dependent shift is added so DE has something to find.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    cl = df["Cluster"].to_numpy() if "Cluster" in df.columns else np.zeros(n, int)
    shift = (cl - cl.mean())  # cluster-correlated component

    df = df.copy()
    df["K562_rep1"] = np.clip(rng.poisson(5, n) + (shift * 2).astype(int), 0, None)
    df["K562_rep2"] = np.clip(rng.poisson(5, n) + (shift * 2).astype(int), 0, None)
    df["Liver_donorA"] = [str(int(v)) for v in
                          np.clip(rng.poisson(8, n) - shift.astype(int), 0, None)]
    df["siCTRL_24h"] = np.round(np.clip(rng.gamma(2.0, 2.0, n) + shift, 0, None), 3)
    df["patient_07_tumor"] = np.round(np.clip(rng.gamma(1.5, 3.0, n) - shift, 0, None), 3)

    bc = lambda: "".join(rng.choice(list("ACGT"), 16))
    cells = []
    for i in range(n):
        m = int(np.clip(rng.poisson(3) + int(shift[i]), 0, None))
        cells.append("[" + ", ".join(f"'{bc()}'" for _ in range(m)) + "]")
    df["GM12878.scRNA"] = cells
    return df, ["K562_rep1", "K562_rep2", "Liver_donorA",
                "siCTRL_24h", "patient_07_tumor", "GM12878.scRNA"]


def prepare_family(family, assembly, work_root, sdir, py, args, mlog):
    """Pull from UCSC, cluster, inject synthetic expression.

    Returns (prepared_csv_path, timings) on success, or (None, {}) on failure.
    timings is {"retrieval_s": ..., "clustering_s": ..., "synexpr_s": ...} —
    kept separate from Stage 11 timing so the "only sequence fetching is
    timed" gap is closed: every stage that runs after retrieval gets its own
    measured wall-clock number too.
    """
    tag = f"{family}_{assembly}"
    work = work_root / tag
    work.mkdir(parents=True, exist_ok=True)
    fam_log = work / "prep.log"
    print(f"  [{tag}] preparing (UCSC pull → cluster → synthetic expr)...",
          flush=True)
    timings = {}

    with open(fam_log, "w", buffering=1) as lf:
        # 1. UCSC pull via te_prep.py (absolute base-dir → predictable output).
        prep_cmd = [py, str(sdir / "te_prep.py"), family, assembly,
                    "--base-dir", str(work), "--max-loci", str(args.max_loci)]
        if args.rmsk_dir:
            prep_cmd += ["--rmsk-dir", args.rmsk_dir]
        if args.fetch_workers:
            prep_cmd += ["--fetch-workers", str(args.fetch_workers)]
        rc, dt = _run(prep_cmd, lf)
        timings["retrieval_s"] = round(dt, 1)
        clustered = work / "clustered_data.csv"
        if rc != 0 or not clustered.exists():
            mlog.write(f"  [{tag}] PREP FAILED (te_prep rc={rc}, "
                       f"retrieval={dt:.0f}s)\n")
            print(f"  [{tag}] PREP FAILED — see {fam_log}")
            return None, timings

        df = pd.read_csv(clustered)
        if "Seq" not in df.columns or df.empty:
            mlog.write(f"  [{tag}] PREP FAILED (no Seq / empty)\n")
            return None, timings
        n_loci = len(df)

        # 2. Cluster via te_clustering.py; fall back to synthetic clusters.
        clu_out = work / "clustered_with_labels.csv"
        clu_cmd = [py, str(sdir / "te_clustering.py"), "--input", str(clustered),
                   "--output", str(clu_out)]
        rc, dt = _run(clu_cmd, lf)
        timings["clustering_s"] = round(dt, 1)
        if rc == 0 and clu_out.exists():
            df = pd.read_csv(clu_out)
        n_real = (df["Cluster"].nunique() if "Cluster" in df.columns else 0)
        if "Cluster" not in df.columns or n_real < 2:
            df["Cluster"] = _synthetic_clusters(df)
            lf.write(f"[clusters] using synthetic clusters "
                     f"(real n_clusters={n_real})\n")

        # 3. Inject synthetic expression with varied sample names.
        _t0 = time.time()
        df, expr_cols = _inject_synthetic_expression(
            df, seed=abs(hash(tag)) % (2**32))
        prepared = work / f"{tag}_prepared.csv"
        df.to_csv(prepared, index=False)
        timings["synexpr_s"] = round(time.time() - _t0, 1)
        lf.write(f"[prepared] {prepared}  rows={len(df)} "
                 f"clusters={df['Cluster'].nunique()} expr_cols={expr_cols}\n")

        # Also stage the clustered CSV as <family_lower>_clustered.csv under
        # 01_data/ so --stage11-engine nextflow can point --results straight at
        # `work` (post_alignment_analyses.nf reads that exact path).
        data_dir = work / "01_data"
        data_dir.mkdir(exist_ok=True)
        df.to_csv(data_dir / f"{family.lower()}_clustered.csv", index=False)

    print(f"  [{tag}] prepared: {n_loci} loci, "
          f"{df['Cluster'].nunique()} clusters, {len(expr_cols)} synthetic samples  "
          f"(retrieval={timings['retrieval_s']:.0f}s  "
          f"clustering={timings['clustering_s']:.0f}s  "
          f"synexpr={timings['synexpr_s']:.1f}s)")
    mlog.write(f"  [{tag}] prepared OK: loci={n_loci} "
               f"clusters={df['Cluster'].nunique()} expr_cols={len(expr_cols)}  "
               f"retrieval={timings['retrieval_s']:.0f}s "
               f"clustering={timings['clustering_s']:.0f}s "
               f"synexpr={timings['synexpr_s']:.1f}s\n")
    mlog.flush()
    return prepared, timings


def run_family_stage11(family, assembly, prepared, work, out_root, sdir, py,
                       args, mlog):
    """Run every Stage 11 module for one prepared family. Returns result dict.

    engine="loop" (default) invokes run_stage11_all.py's in-process subprocess
    loop directly; engine="nextflow" hands the same module registry to
    nextflow/post_alignment_analyses.nf so it's exercised for real (not just
    the fallback path). Either engine runs inside gameca.sif when
    --container-sif is set. Per-module elapsed seconds are always captured
    (not just the retrieval step) — from run_stage11_all.py's own summary
    lines in loop mode, or from Nextflow's trace file in nextflow mode.
    """
    tag = f"{family}_{assembly}"
    reports = out_root / tag / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    log = out_root / tag / "stage11.log"
    skip_fold = not args.include_fold

    if args.stage11_engine == "nextflow":
        rc, dt, modules = _run_stage11_nextflow(
            family, assembly, work, reports, log, sdir, skip_fold, args, mlog)
    else:
        rc, dt, modules = _run_stage11_loop(
            family, assembly, prepared, reports, log, sdir, py, skip_fold, args)

    n_ok = sum(1 for _, s in modules.values() if s == "OK")
    n_fail = sum(1 for _, s in modules.values() if s.startswith("FAILED"))
    n_skip = sum(1 for _, s in modules.values() if s == "SKIP")
    # Fallback/cross-check for loop mode: the 'Done: N ok / N failed / N
    # skipped' line is authoritative. If the per-module table couldn't be
    # scraped (e.g. format drift), trust the Done line so the master summary
    # never reports a false zero for a run that actually executed modules.
    if args.stage11_engine != "nextflow":
        done = _parse_stage11_done(log)
        if done and (n_ok, n_fail, n_skip) != done:
            if not modules:
                n_ok, n_fail, n_skip = done
            else:
                mlog.write(f"  [{tag}] WARNING: module table "
                           f"({n_ok}/{n_fail}/{n_skip}) disagrees with Done line "
                           f"({done[0]}/{done[1]}/{done[2]})\n")

    # Stage 11 total, excluding fold (the default; fold is skipped unless
    # --include-fold), so this stays comparable across --include-fold runs.
    excl_fold_s = round(sum(el for name, (el, _) in modules.items()
                            if name != "fold" and el is not None), 1)
    fold_s = next((el for name, (el, _) in modules.items() if name == "fold"
                   and el is not None), None)

    print(f"  [{tag}] Stage 11 ({args.stage11_engine}) done rc={rc} "
          f"({dt:.0f}s total, {excl_fold_s:.0f}s excl. fold) — "
          f"ok={n_ok} failed={n_fail} skipped={n_skip}")
    mlog.write(f"  [{tag}] STAGE11[{args.stage11_engine}] rc={rc} "
               f"total={dt:.0f}s excl_fold={excl_fold_s:.0f}s "
               f"ok={n_ok} failed={n_fail} skipped={n_skip}\n")
    for m, (el, st) in modules.items():
        mlog.write(f"        {m:<20} {st:<18} "
                   f"{'' if el is None else f'{el:.0f}s'}\n")
    mlog.flush()
    return {"tag": tag, "family": family, "assembly": assembly, "rc": rc,
            "engine": args.stage11_engine,
            "elapsed_s": round(dt, 1), "elapsed_excl_fold_s": excl_fold_s,
            "fold_s": fold_s,
            "modules": {m: st for m, (el, st) in modules.items()},
            "module_seconds": {m: el for m, (el, st) in modules.items()},
            "n_ok": n_ok, "n_fail": n_fail, "n_skip": n_skip,
            "log": str(log), "reports": str(reports)}


def _run_stage11_loop(family, assembly, prepared, reports, log, sdir, py,
                      skip_fold, args):
    """engine="loop": run_stage11_all.py's in-process subprocess loop."""
    tag = f"{family}_{assembly}"
    cmd = [py, str(sdir / "run_stage11_all.py"),
           "--input", str(prepared),
           "--family", family,
           "--assembly", assembly,
           "--reports-dir", str(reports),
           "--expr-stage-cols", "K562_rep1", "K562_rep2", "Liver_donorA",
           "siCTRL_24h", "patient_07_tumor", "GM12878.scRNA"]
    if skip_fold:
        cmd += ["--skip", "fold"]
    cmd = _singularity_wrap(cmd, args.container_sif)
    print(f"  [{tag}] running Stage 11 (loop{', singularity' if args.container_sif else ''}"
          f"{', skipping fold' if skip_fold else ''})...", flush=True)

    t0 = time.time()
    with open(log, "w", buffering=1) as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
    dt = time.time() - t0
    return rc, dt, _parse_stage11_summary(log)


def _run_stage11_nextflow(family, assembly, work, reports, log, sdir,
                          skip_fold, args, mlog):
    """engine="nextflow": nextflow/post_alignment_analyses.nf, scattering the
    same module registry across real Nextflow processes. `work` must contain
    01_data/<family_lower>_clustered.csv (staged by prepare_family)."""
    tag = f"{family}_{assembly}"
    nf = _ensure_nextflow()
    if not nf:
        mlog.write(f"  [{tag}] STAGE11 FAILED: nextflow not on PATH and self-install failed\n")
        print(f"  [{tag}] Stage 11 (nextflow) FAILED: nextflow not on PATH and self-install failed")
        return -1, 0.0, {}

    trace = reports / "nextflow_trace.txt"
    cmd = [nf, "run", str(sdir / "nextflow" / "post_alignment_analyses.nf"),
           "--results", str(work), "--family", family, "--assembly", assembly,
           "--gameca_home", str(sdir), "-with-trace", str(trace), "-resume"]
    if skip_fold:
        cmd += ["--skip_modules", "fold"]
    profiles = (["singularity"] if args.container_sif else []) + \
               ([p for p in (args.nextflow_profile or "").split(",") if p])
    if profiles:
        cmd += ["-profile", ",".join(dict.fromkeys(profiles))]
    if args.container_sif:
        cmd += ["--container_sif", args.container_sif]
    print(f"  [{tag}] running Stage 11 (nextflow"
          f"{', singularity' if args.container_sif else ''}"
          f"{', skipping fold' if skip_fold else ''})...", flush=True)

    t0 = time.time()
    with open(log, "w", buffering=1) as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                            cwd=str(sdir)).returncode
    dt = time.time() - t0
    return rc, dt, _parse_nextflow_trace(trace, log)


def _parse_stage11_summary(log_path):
    """Pull the module-status + per-module elapsed-seconds table written by
    run_stage11_all.py.

    run_stage11_all prints the table to stdout under a bare ``SUMMARY`` header
    (the ``STAGE 11 SUMMARY`` header only appears in the file it writes to its
    own --reports-dir log, which we don't read). Match either so the harness
    counts modules regardless of which sink it scrapes.

    Returns {name: (elapsed_seconds_or_None, status_str)}.
    """
    out = {}
    try:
        lines = Path(log_path).read_text(errors="replace").splitlines()
    except Exception:
        return out
    in_summary = False
    for ln in lines:
        if ln.strip() == "SUMMARY" or "STAGE 11 SUMMARY" in ln:
            in_summary = True
            continue
        if in_summary:
            s = ln.strip()
            if not s or s.startswith("Done:") or s.startswith("="):
                if s.startswith("Done:"):
                    break
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            name = parts[0]
            elapsed = None
            if parts[-1].endswith("s") and parts[-1][:-1].replace(".", "", 1).isdigit():
                elapsed = float(parts[-1][:-1])
                status_tokens = parts[1:-1]
            else:
                status_tokens = parts[1:]
            status = status_tokens[0] if status_tokens[0] in ("OK", "SKIP") \
                else " ".join(status_tokens)
            out[name] = (elapsed, status)
    return out


def _parse_nextflow_trace(trace_path, log_path):
    """Parse a Nextflow -with-trace TSV into {module_name: (seconds, status)}.

    The default trace has no separate `tag` column — the GAMECA_STANDOUT tag
    directive (`"${meta.id}:${mod.name}"`) is embedded in `name`, e.g.
    "STANDOUT:GAMECA_STANDOUT (MT2_Mm:phylo)". `realtime` is a human string
    like "1.2s"/"3m 20s"/"1h 5m 10s". Falls back to an empty dict (elapsed
    stays None) if the trace is missing or Nextflow changes its column
    layout — the caller still has rc/dt for the aggregate total either way.
    """
    import re
    out = {}
    try:
        lines = Path(trace_path).read_text(errors="replace").splitlines()
    except Exception:
        return out
    if not lines:
        return out
    header = lines[0].split("\t")
    try:
        i_name = header.index("name")
        i_status = header.index("status")
        i_realtime = header.index("realtime")
    except ValueError:
        return out
    for ln in lines[1:]:
        cols = ln.split("\t")
        if len(cols) <= max(i_name, i_status, i_realtime):
            continue
        name = cols[i_name]
        m = re.search(r"\(([^:()]+):([^:()]+)\)\s*$", name)
        mod_name = m.group(2) if m else name
        raw_status = cols[i_status]
        status = "OK" if raw_status == "COMPLETED" else \
                 "SKIP" if raw_status == "CACHED" else \
                 (raw_status if raw_status.startswith("FAILED") else f"FAILED {raw_status}")
        secs = _realtime_to_seconds(cols[i_realtime])
        out[mod_name] = (secs, status)
    return out


def _realtime_to_seconds(s):
    """Convert Nextflow's 'realtime' string ("1h 5m 10s", "3m 20s", "450ms")
    to float seconds, or None if unparseable."""
    import re
    s = s.strip()
    if not s or s in ("-", "n/a"):
        return None
    if s.endswith("ms"):
        try:
            return float(s[:-2]) / 1000.0
        except ValueError:
            return None
    total = 0.0
    matched = False
    for value, unit in re.findall(r"([\d.]+)\s*([hms])", s):
        matched = True
        total += float(value) * {"h": 3600, "m": 60, "s": 1}[unit]
    return total if matched else None


def _parse_stage11_done(log_path):
    """Return (n_ok, n_fail, n_skip) from the 'Done: N ok / N failed / N skipped'
    line run_stage11_all prints, or None if not found."""
    import re
    try:
        text = Path(log_path).read_text(errors="replace")
    except Exception:
        return None
    m = re.search(r"Done:\s*(\d+)\s*ok\s*/\s*(\d+)\s*failed\s*/\s*(\d+)\s*skipped",
                  text)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def main():
    p = argparse.ArgumentParser(
        description="Comprehensive multi-family/assembly GAMECA cluster test.")
    p.add_argument("--out-dir", required=True,
                   help="Top-level output dir (e.g. ~/gameca_cluster_test).")
    p.add_argument("--matrix", default=None,
                   help="Comma-separated FAMILY:ASSEMBLY pairs "
                        "(default: built-in LINE/SINE/LTR × hg38/mm10 matrix).")
    p.add_argument("--max-loci", type=int, default=300,
                   help="Cap loci per family (keeps UCSC fetch + MAFFT tractable).")
    p.add_argument("--rmsk-dir", default=None,
                   help="Directory holding rmsk_<build>.txt.gz (passed to te_prep).")
    p.add_argument("--fetch-workers", type=int, default=3,
                   help="UCSC parallel fetch workers (te_prep, capped at 3).")
    p.add_argument("--include-fold", action="store_true",
                   help="Also run the ColabFold module (heavy/GPU; off by default).")
    p.add_argument("--stop-on-fail", action="store_true",
                   help="Abort the whole test on the first family failure "
                        "(default: keep going through the rest of the matrix).")
    p.add_argument("--stage11-engine", choices=["loop", "nextflow"], default="nextflow",
                   help="How to run the Stage 11 modules: 'nextflow' (default — "
                        "nextflow/post_alignment_analyses.nf, real per-module parallelism, "
                        "matching query.py --post-alignment-analyses-nextflow which "
                        "hpc_client now enables on every job) or 'loop' (run_stage11_all.py's "
                        "in-process subprocess loop, the same fallback query.py uses if "
                        "nextflow is unavailable).")
    p.add_argument("--container-sif", default=None,
                   help="Path to a prebuilt gameca.sif. Auto-detected by default from "
                        "$GAMECA_SIF, ./gameca.sif, or ~/gameca/gameca.sif (the same cache "
                        "path hpc_client._ensure_container_sif uses) so this test runs "
                        "against the same container every regular pipeline job uses. "
                        "loop engine: wraps the Stage 11 subprocess in `singularity exec`. "
                        "nextflow engine: adds -profile singularity --container_sif <path>.")
    p.add_argument("--nextflow-profile", default=None,
                   help="Extra comma-separated Nextflow profiles to combine with "
                        "'singularity' when --container-sif is set (e.g. 'lsf').")
    p.add_argument("--notify-email", default="",
                   help="Email address for a completion summary (Resend API).")
    args = p.parse_args()

    if not args.container_sif:
        args.container_sif = _default_container_sif()

    if args.matrix:
        matrix = []
        for pair in args.matrix.split(","):
            fam, _, asm = pair.strip().partition(":")
            if fam and asm:
                matrix.append((fam, asm))
        if not matrix:
            sys.exit("ERROR: --matrix parsed to nothing; use FAMILY:ASSEMBLY,...")
    else:
        matrix = DEFAULT_MATRIX

    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    work_root = out_root / "_work"
    work_root.mkdir(parents=True, exist_ok=True)

    sdir = Path(__file__).resolve().parent
    py = shutil.which("python3") or shutil.which("python") or sys.executable
    master_log = out_root / "gameca_cluster_test.log"
    summary_json = out_root / "gameca_cluster_test_summary.json"

    started = datetime.datetime.now()
    print("=" * 64)
    print("GAMECA COMPREHENSIVE CLUSTER TEST")
    print("=" * 64)
    print(f"  Output:   {out_root}")
    print(f"  Matrix:   {', '.join(f'{f}:{a}' for f, a in matrix)}")
    print(f"  Max loci: {args.max_loci}   Fold: {args.include_fold}")
    print(f"  Stage 11: {args.stage11_engine}"
          f"{'  (container: ' + args.container_sif + ')' if args.container_sif else ''}")
    print(f"  Master log: {master_log}")
    print("=" * 64)

    results = []
    with open(master_log, "w", buffering=1) as mlog:
        mlog.write("GAMECA Comprehensive Cluster Test\n")
        mlog.write(f"Started:  {started:%Y-%m-%d %H:%M:%S}\n")
        mlog.write(f"Output:   {out_root}\n")
        mlog.write(f"Matrix:   {', '.join(f'{f}:{a}' for f, a in matrix)}\n")
        mlog.write(f"Max loci: {args.max_loci}  Fold: {args.include_fold}\n")
        mlog.write(f"Stage 11 engine: {args.stage11_engine}  "
                   f"Container: {args.container_sif or '(none)'}\n")
        mlog.write(f"Python:   {py}\n")
        mlog.write("NOTE: expression numbers are SYNTHETIC test fixtures.\n")
        mlog.write("=" * 64 + "\n")

        for family, assembly in matrix:
            tag = f"{family}_{assembly}"
            work = work_root / tag
            mlog.write(f"\n{'='*64}\n[{tag}]\n{'='*64}\n"); mlog.flush()
            try:
                prepared, timings = prepare_family(family, assembly, work_root,
                                                   sdir, py, args, mlog)
                if prepared is None:
                    results.append({"tag": tag, "family": family,
                                    "assembly": assembly, "rc": -1,
                                    "error": "prep_failed", "modules": {},
                                    "n_ok": 0, "n_fail": 0, "n_skip": 0,
                                    **timings})
                    if args.stop_on_fail:
                        break
                    continue
                res = run_family_stage11(family, assembly, prepared, work,
                                         out_root, sdir, py, args, mlog)
                res.update(timings)
                results.append(res)
            except Exception as e:
                mlog.write(f"  [{tag}] EXCEPTION: {e}\n")
                print(f"  [{tag}] EXCEPTION: {e}")
                results.append({"tag": tag, "family": family,
                                "assembly": assembly, "rc": -2,
                                "error": str(e), "modules": {},
                                "n_ok": 0, "n_fail": 0, "n_skip": 0})
                if args.stop_on_fail:
                    break

        # ── Master summary ────────────────────────────────────────────────────
        fam_ok = sum(1 for r in results if r.get("rc") == 0)
        fam_bad = len(results) - fam_ok
        total_ok = sum(r.get("n_ok", 0) for r in results)
        total_fail = sum(r.get("n_fail", 0) for r in results)
        total_skip = sum(r.get("n_skip", 0) for r in results)

        mlog.write(f"\n{'='*64}\nMASTER SUMMARY\n{'='*64}\n")
        for r in results:
            mlog.write(f"  {r['tag']:<22} rc={r.get('rc')}  "
                       f"ok={r.get('n_ok',0)} failed={r.get('n_fail',0)} "
                       f"skipped={r.get('n_skip',0)}"
                       f"{'  ['+r['error']+']' if r.get('error') else ''}\n")
        mlog.write(f"\nFamilies: {fam_ok} ok / {fam_bad} problem (of {len(results)})\n")
        mlog.write(f"Modules:  {total_ok} ok / {total_fail} failed / "
                   f"{total_skip} skipped (across all families)\n")

        # ── Timing breakdown ────────────────────────────────────────────────
        # Retrieval (UCSC pull) was previously the only step with visible
        # timing; every stage after it is now measured too (per family, plus
        # a per-module Stage 11 breakdown, always excluding fold from the
        # "excl. fold" total since it's off by default).
        mlog.write(f"\n{'-'*64}\nTIMING (seconds)\n{'-'*64}\n")
        mlog.write(f"  {'family':<22} {'retrieval':>10} {'cluster':>9} "
                   f"{'synexpr':>8} {'stage11':>9} {'excl_fold':>10} {'fold':>8}\n")
        for r in results:
            mlog.write(
                f"  {r['tag']:<22} "
                f"{r.get('retrieval_s', float('nan')):>10.1f} "
                f"{r.get('clustering_s', float('nan')):>9.1f} "
                f"{r.get('synexpr_s', float('nan')):>8.1f} "
                f"{r.get('elapsed_s', float('nan')):>9.1f} "
                f"{r.get('elapsed_excl_fold_s', float('nan')):>10.1f} "
                f"{r.get('fold_s') if r.get('fold_s') is not None else '-':>8}\n"
            )
        mlog.write("Finished: {:%Y-%m-%d %H:%M:%S}\n".format(datetime.datetime.now()))

    summary = {
        "started": started.isoformat(),
        "finished": datetime.datetime.now().isoformat(),
        "out_dir": str(out_root),
        "matrix": [f"{f}:{a}" for f, a in matrix],
        "max_loci": args.max_loci,
        "include_fold": args.include_fold,
        "stage11_engine": args.stage11_engine,
        "container_sif": args.container_sif,
        "families_ok": fam_ok,
        "families_problem": fam_bad,
        "modules_ok": total_ok,
        "modules_failed": total_fail,
        "modules_skipped": total_skip,
        "results": results,
        "note": "expression values are synthetic test fixtures, not measurements",
    }
    summary_json.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 64)
    print("MASTER SUMMARY")
    print("=" * 64)
    for r in results:
        print(f"  {r['tag']:<22} rc={r.get('rc')}  ok={r.get('n_ok',0)} "
              f"failed={r.get('n_fail',0)} skipped={r.get('n_skip',0)}"
              f"{'  ['+r['error']+']' if r.get('error') else ''}")
    print(f"\n  Families: {fam_ok} ok / {fam_bad} problem (of {len(results)})")
    print(f"  Modules:  {total_ok} ok / {total_fail} failed / {total_skip} skipped")
    print(f"\n  Timing (s):  {'family':<22} {'retrieval':>9} {'cluster':>8} "
          f"{'synexpr':>8} {'stage11':>8} {'excl_fold':>10}")
    for r in results:
        print(f"               {r['tag']:<22} "
              f"{r.get('retrieval_s', float('nan')):>9.1f} "
              f"{r.get('clustering_s', float('nan')):>8.1f} "
              f"{r.get('synexpr_s', float('nan')):>8.1f} "
              f"{r.get('elapsed_s', float('nan')):>8.1f} "
              f"{r.get('elapsed_excl_fold_s', float('nan')):>10.1f}")
    print(f"  Master log:  {master_log}")
    print(f"  JSON summary: {summary_json}")

    if args.notify_email:
        try:
            from te_notify import send_completion_email
            lines = [f"{r['tag']:<22} rc={r.get('rc')}  ok={r.get('n_ok',0)} "
                     f"failed={r.get('n_fail',0)} skipped={r.get('n_skip',0)}"
                     for r in results]
            lines.append(f"\nFamilies: {fam_ok} ok / {fam_bad} problem")
            lines.append(f"Modules:  {total_ok} ok / {total_fail} failed / "
                         f"{total_skip} skipped")
            send_completion_email(
                args.notify_email,
                "GAMECA comprehensive cluster test",
                str(out_root),
                summary="\n".join(lines),
            )
            print(f"  Notification sent to {args.notify_email}")
        except Exception as e:
            print(f"  WARNING: email notification failed: {e}")

    sys.exit(1 if fam_bad else 0)


if __name__ == "__main__":
    main()
