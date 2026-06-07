#!/usr/bin/env python3
"""
gameca_pipeline.py --- GAMECA declarative YAML + DAG pipeline runner (#8)

Compose primer / gRNA / expression / phylogenetics / fold / overlay stages in one
reproducible config instead of remembering a dozen CLIs and flag combinations.
Stages form a DAG via `depends_on`; the runner groups them into parallel waves,
skips stages already completed (resume via te_provenance checkpoints, #6), and
records a provenance manifest for every run.

Config format (gameca.yaml):

    family: LTR5_Hs
    input: loci_with_sequences.csv
    reports_dir: ./reports
    parallel: true        # run independent stages concurrently (default: false)
    stages:
      - name: phylo
        script: run_phylo_analysis.py
        args: {family: LTR5_Hs, intact-orf-aa: 100}
      - name: grna_offtarget
        script: run_grna_offtarget.py
        args: {cas: SpCas9, max-mm: 2}
        depends_on: [phylo]

Top-level `family`, `input`, `reports_dir` are injected into every stage unless
the stage's own args override them.

Usage:
    python gameca_pipeline.py --config gameca.yaml [--dry-run] [--force] \\
        [--parallel] [--workers N] [--only phylo,grna_offtarget] [--python python3]
"""

import argparse
import concurrent.futures
import subprocess
import sys
import threading
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml required.  pip install pyyaml", file=sys.stderr)
    sys.exit(1)

import te_provenance as prov

# guards provenance.json writes when multiple stage threads run concurrently
_prov_lock = threading.Lock()


def _level_sort(stages: list) -> list:
    """Kahn by level: return stages grouped into parallel waves."""
    by_name = {s["name"]: s for s in stages}
    indeg   = {s["name"]: 0 for s in stages}
    adj     = {s["name"]: [] for s in stages}
    for s in stages:
        for dep in s.get("depends_on", []) or []:
            if dep not in by_name:
                raise ValueError(
                    f"Stage '{s['name']}' depends on unknown stage '{dep}'")
            adj[dep].append(s["name"])
            indeg[s["name"]] += 1
    remaining = dict(indeg)
    levels = []
    while remaining:
        wave = [n for n, d in remaining.items() if d == 0]
        if not wave:
            raise ValueError("Cycle detected in stage dependency graph")
        levels.append([by_name[n] for n in wave])
        for n in wave:
            del remaining[n]
            for m in adj[n]:
                remaining[m] -= 1
    return levels


def _build_cmd(python, script_dir, stage, glob, reports_dir):
    args = dict(stage.get("args", {}) or {})
    args.setdefault("input",       glob["input"])
    args.setdefault("reports-dir", reports_dir)
    args.setdefault("family",      glob.get("family", "TE"))
    cmd = [python, str(Path(script_dir) / stage["script"])]
    for k, v in args.items():
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        elif isinstance(v, (list, tuple)):
            cmd.append(flag)
            cmd.extend(str(x) for x in v)
        else:
            cmd.extend([flag, str(v)])
    return cmd


def _run_stage(python, script_dir, s, glob, reports_dir, force,
               log_path=None) -> tuple:
    """Execute one stage. Returns (name, returncode, skipped:bool)."""
    name = s["name"]
    cmd  = _build_cmd(python, script_dir, s, glob, reports_dir)
    stage_args = dict(s.get("args", {}) or {})

    if not force and prov.is_done(reports_dir, name):
        return name, 0, True

    with _prov_lock:
        prov.record_stage(reports_dir, name, stage_args,
                          [glob["input"]], status="running")

    if log_path:
        with open(log_path, "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    else:
        rc = subprocess.run(cmd).returncode

    status = "completed" if rc == 0 else f"failed(rc={rc})"
    if rc == 0:
        prov.mark_done(reports_dir, name)
    with _prov_lock:
        prov.record_stage(reports_dir, name, stage_args,
                          [glob["input"]], status=status)

    return name, rc, False


def main():
    ap = argparse.ArgumentParser(
        description="GAMECA YAML + DAG pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--config",   required=True)
    ap.add_argument("--dry-run",  action="store_true",
                    help="Print the plan; run nothing")
    ap.add_argument("--force",    action="store_true",
                    help="Ignore checkpoints; re-run all stages")
    ap.add_argument("--only",     default="",
                    help="Comma-separated subset of stage names to run")
    ap.add_argument("--parallel", action="store_true",
                    help="Run independent stages in each wave concurrently "
                         "(also enabled by `parallel: true` in config)")
    ap.add_argument("--workers",  type=int, default=0,
                    help="Max parallel workers per wave (0 = one per stage)")
    ap.add_argument("--python",   default=sys.executable or "python3")
    args = ap.parse_args()

    cfg         = yaml.safe_load(Path(args.config).read_text())
    glob        = {"input": cfg["input"], "family": cfg.get("family", "TE")}
    reports_dir = cfg.get("reports_dir", "./reports")
    script_dir  = Path(__file__).resolve().parent
    parallel    = args.parallel or bool(cfg.get("parallel", False))

    Path(reports_dir).mkdir(parents=True, exist_ok=True)

    # build level DAG and optionally filter to --only subset
    levels = _level_sort(cfg["stages"])
    only   = {s.strip() for s in args.only.split(",") if s.strip()}
    if only:
        levels = [[s for s in lv if s["name"] in only] for lv in levels]
        levels = [lv for lv in levels if lv]

    all_stages = [s for lv in levels for s in lv]
    prov.init(reports_dir, repo_root=str(script_dir))

    print(f"GAMECA pipeline: {len(all_stages)} stage(s) | "
          f"family={glob['family']} | "
          f"parallel={'yes' if parallel else 'no'}")
    print("Waves:", " | ".join(
        f"[{' '.join(s['name'] for s in lv)}]" for lv in levels))

    log_dir = Path(reports_dir) / ".stage_logs"
    if parallel:
        log_dir.mkdir(parents=True, exist_ok=True)

    failed = []

    for wave_i, level in enumerate(levels, start=1):
        if not level:
            continue

        if args.dry_run:
            for s in level:
                cmd = _build_cmd(args.python, script_dir, s, glob, reports_dir)
                print(f"  [plan] {s['name']}: {' '.join(cmd)}")
            continue

        if parallel and len(level) > 1:
            workers = args.workers or len(level)
            names   = ", ".join(s["name"] for s in level)
            print(f"\n  -- wave {wave_i}: {len(level)} stages in parallel "
                  f"(workers={workers}): {names}")

            def _task(s=None):
                lp = log_dir / f"{s['name']}.log"
                return _run_stage(args.python, script_dir, s, glob,
                                  reports_dir, args.force, log_path=lp)

            wave_failed = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_task, s): s for s in level}
                for fut in concurrent.futures.as_completed(futs):
                    s = futs[fut]
                    name, rc, skipped = fut.result()
                    if skipped:
                        print(f"  [skip] {name} (checkpoint exists)")
                    elif rc == 0:
                        print(f"  [ ok ] {name}  (log: {log_dir/f'{name}.log'})")
                    else:
                        print(f"  [FAIL] {name} (exit {rc})"
                              f"  (log: {log_dir/f'{name}.log'})")
                        wave_failed.append(name)

            failed.extend(wave_failed)
            hard_fails = [
                n for n in wave_failed
                if not next(s for s in level if s["name"] == n
                            ).get("continue_on_error", False)
            ]
            if hard_fails:
                print(f"  Hard failure(s) in wave {wave_i}: {hard_fails}. Stopping.")
                break

        else:
            # serial execution of this wave (single stage, or parallel=False)
            for s in level:
                name = s["name"]
                cmd  = _build_cmd(args.python, script_dir, s, glob, reports_dir)

                if not args.force and prov.is_done(reports_dir, name):
                    print(f"  [skip] {name} (checkpoint exists; use --force to re-run)")
                    continue

                print(f"  [run ] {name}: {' '.join(cmd)}")
                prov.record_stage(reports_dir, name,
                                  dict(s.get("args", {}) or {}),
                                  [glob["input"]], status="running")
                rc = subprocess.run(cmd).returncode
                if rc == 0:
                    prov.mark_done(reports_dir, name)
                    prov.record_stage(reports_dir, name,
                                      dict(s.get("args", {}) or {}),
                                      [glob["input"]], status="completed")
                    print(f"  [ ok ] {name}")
                else:
                    prov.record_stage(reports_dir, name,
                                      dict(s.get("args", {}) or {}),
                                      [glob["input"]], status=f"failed(rc={rc})")
                    print(f"  [FAIL] {name} (exit {rc})")
                    failed.append(name)
                    if not s.get("continue_on_error", False):
                        break

    if not args.dry_run:
        prov.to_tex(reports_dir)
        print("\nProvenance manifest:", Path(reports_dir) / "provenance.json")
    if failed:
        print("Failed stages:", ", ".join(failed))
        sys.exit(1)
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
