#!/usr/bin/env python3
"""
run_gameca_cluster_test.py — comprehensive multi-family / multi-assembly GAMECA
integration test, designed to run on the HPC cluster.

What it exercises (end to end, for every (family, assembly) in the matrix):
  1. UCSC pull        — te_prep.py downloads the rmsk track and fetches per-locus
                        sequences straight from the UCSC browser
                        (api.genome.ucsc.edu), so the live fetch path is tested.
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
  4. Stage 11         — run_stage11_all.py runs every standout-analysis module on
                        the prepared CSV.

A master log + JSON summary capture, per family and per module, OK/FAIL/SKIP and
timing, so a single output file tells you whether the whole program is healthy
across assemblies and families.

Usage (cluster):
    python run_gameca_cluster_test.py \
        --out-dir /home/amodz/anmol/gamecatestv624 \
        --notify-email anmoldash@gmail.com

    # custom matrix / caps:
    python run_gameca_cluster_test.py --out-dir OUT \
        --matrix "L1HS:hg38,MT2_Mm:mm10" --max-loci 200 --include-fold
"""

import argparse
import datetime
import json
import os
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
    """Pull from UCSC, cluster, inject synthetic expression. Returns prepared CSV
    path or None on failure."""
    tag = f"{family}_{assembly}"
    work = work_root / tag
    work.mkdir(parents=True, exist_ok=True)
    fam_log = work / "prep.log"
    print(f"  [{tag}] preparing (UCSC pull → cluster → synthetic expr)...",
          flush=True)

    with open(fam_log, "w", buffering=1) as lf:
        # 1. UCSC pull via te_prep.py (absolute base-dir → predictable output).
        prep_cmd = [py, str(sdir / "te_prep.py"), family, assembly,
                    "--base-dir", str(work), "--max-loci", str(args.max_loci)]
        if args.rmsk_dir:
            prep_cmd += ["--rmsk-dir", args.rmsk_dir]
        if args.fetch_workers:
            prep_cmd += ["--fetch-workers", str(args.fetch_workers)]
        rc, _ = _run(prep_cmd, lf)
        clustered = work / "clustered_data.csv"
        if rc != 0 or not clustered.exists():
            mlog.write(f"  [{tag}] PREP FAILED (te_prep rc={rc})\n")
            print(f"  [{tag}] PREP FAILED — see {fam_log}")
            return None

        df = pd.read_csv(clustered)
        if "Seq" not in df.columns or df.empty:
            mlog.write(f"  [{tag}] PREP FAILED (no Seq / empty)\n")
            return None
        n_loci = len(df)

        # 2. Cluster via te_clustering.py; fall back to synthetic clusters.
        clu_out = work / "clustered_with_labels.csv"
        clu_cmd = [py, str(sdir / "te_clustering.py"), "--input", str(clustered),
                   "--output", str(clu_out)]
        rc, _ = _run(clu_cmd, lf)
        if rc == 0 and clu_out.exists():
            df = pd.read_csv(clu_out)
        n_real = (df["Cluster"].nunique() if "Cluster" in df.columns else 0)
        if "Cluster" not in df.columns or n_real < 2:
            df["Cluster"] = _synthetic_clusters(df)
            lf.write(f"[clusters] using synthetic clusters "
                     f"(real n_clusters={n_real})\n")

        # 3. Inject synthetic expression with varied sample names.
        df, expr_cols = _inject_synthetic_expression(
            df, seed=abs(hash(tag)) % (2**32))
        prepared = work / f"{tag}_prepared.csv"
        df.to_csv(prepared, index=False)
        lf.write(f"[prepared] {prepared}  rows={len(df)} "
                 f"clusters={df['Cluster'].nunique()} expr_cols={expr_cols}\n")

    print(f"  [{tag}] prepared: {n_loci} loci, "
          f"{df['Cluster'].nunique()} clusters, {len(expr_cols)} synthetic samples")
    mlog.write(f"  [{tag}] prepared OK: loci={n_loci} "
               f"clusters={df['Cluster'].nunique()} expr_cols={len(expr_cols)}\n")
    mlog.flush()
    return prepared


def run_family_stage11(family, assembly, prepared, out_root, sdir, py, args, mlog):
    """Run run_stage11_all.py for one prepared family. Returns result dict."""
    tag = f"{family}_{assembly}"
    reports = out_root / tag / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    log = out_root / tag / "stage11.log"

    cmd = [py, str(sdir / "run_stage11_all.py"),
           "--input", str(prepared),
           "--family", family,
           "--assembly", assembly,
           "--reports-dir", str(reports),
           "--expr-stage-cols", "K562_rep1", "K562_rep2", "Liver_donorA",
           "siCTRL_24h", "patient_07_tumor", "GM12878.scRNA"]
    if not args.include_fold:
        cmd += ["--skip", "fold"]
    print(f"  [{tag}] running Stage 11"
          f"{' (skipping fold)' if not args.include_fold else ''}...", flush=True)

    t0 = time.time()
    with open(log, "w", buffering=1) as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
    dt = time.time() - t0

    # Parse the per-module summary that run_stage11_all wrote into the log.
    modules = _parse_stage11_summary(log)
    n_ok = sum(1 for s in modules.values() if s == "OK")
    n_fail = sum(1 for s in modules.values() if s.startswith("FAILED"))
    n_skip = sum(1 for s in modules.values() if s == "SKIP")
    print(f"  [{tag}] Stage 11 done rc={rc} ({dt:.0f}s) — "
          f"ok={n_ok} failed={n_fail} skipped={n_skip}")
    mlog.write(f"  [{tag}] STAGE11 rc={rc} {dt:.0f}s ok={n_ok} "
               f"failed={n_fail} skipped={n_skip}\n")
    for m, s in modules.items():
        mlog.write(f"        {m:<20} {s}\n")
    mlog.flush()
    return {"tag": tag, "family": family, "assembly": assembly, "rc": rc,
            "elapsed_s": round(dt, 1), "modules": modules,
            "n_ok": n_ok, "n_fail": n_fail, "n_skip": n_skip,
            "log": str(log), "reports": str(reports)}


def _parse_stage11_summary(log_path):
    """Pull the 'STAGE 11 SUMMARY' table written by run_stage11_all.py."""
    out = {}
    try:
        lines = Path(log_path).read_text(errors="replace").splitlines()
    except Exception:
        return out
    in_summary = False
    for ln in lines:
        if "STAGE 11 SUMMARY" in ln:
            in_summary = True
            continue
        if in_summary:
            s = ln.strip()
            if not s or s.startswith("Done:") or s.startswith("="):
                if s.startswith("Done:"):
                    break
                continue
            parts = s.split()
            if len(parts) >= 2:
                out[parts[0]] = parts[1] if parts[1] in ("OK", "SKIP") \
                    else " ".join(parts[1:3])
    return out


def main():
    p = argparse.ArgumentParser(
        description="Comprehensive multi-family/assembly GAMECA cluster test.")
    p.add_argument("--out-dir", required=True,
                   help="Top-level output dir (e.g. /home/amodz/anmol/gamecatestv624).")
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
    p.add_argument("--notify-email", default="",
                   help="Email address for a completion summary (Resend API).")
    args = p.parse_args()

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
    print(f"  Master log: {master_log}")
    print("=" * 64)

    results = []
    with open(master_log, "w", buffering=1) as mlog:
        mlog.write("GAMECA Comprehensive Cluster Test\n")
        mlog.write(f"Started:  {started:%Y-%m-%d %H:%M:%S}\n")
        mlog.write(f"Output:   {out_root}\n")
        mlog.write(f"Matrix:   {', '.join(f'{f}:{a}' for f, a in matrix)}\n")
        mlog.write(f"Max loci: {args.max_loci}  Fold: {args.include_fold}\n")
        mlog.write(f"Python:   {py}\n")
        mlog.write("NOTE: expression numbers are SYNTHETIC test fixtures.\n")
        mlog.write("=" * 64 + "\n")

        for family, assembly in matrix:
            tag = f"{family}_{assembly}"
            mlog.write(f"\n{'='*64}\n[{tag}]\n{'='*64}\n"); mlog.flush()
            try:
                prepared = prepare_family(family, assembly, work_root, sdir,
                                          py, args, mlog)
                if prepared is None:
                    results.append({"tag": tag, "family": family,
                                    "assembly": assembly, "rc": -1,
                                    "error": "prep_failed", "modules": {},
                                    "n_ok": 0, "n_fail": 0, "n_skip": 0})
                    if args.stop_on_fail:
                        break
                    continue
                res = run_family_stage11(family, assembly, prepared, out_root,
                                         sdir, py, args, mlog)
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
        mlog.write(f"Finished: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")

    summary = {
        "started": started.isoformat(),
        "finished": datetime.datetime.now().isoformat(),
        "out_dir": str(out_root),
        "matrix": [f"{f}:{a}" for f, a in matrix],
        "max_loci": args.max_loci,
        "include_fold": args.include_fold,
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
