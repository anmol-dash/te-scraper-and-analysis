#!/usr/bin/env python3
"""
te_provenance.py --- GAMECA provenance manifest + resume checkpoints (#6)

Reviewers and reproducibility demand a machine-readable record of exactly how a
result was produced, and long HPC jobs must resume rather than restart. This
module provides both:

  • A JSON provenance manifest (reports/provenance.json) capturing git SHA,
    platform, Python, third-party tool versions, per-stage parameters, input
    file SHA-256 hashes, timestamps, and status.
  • Stage checkpoints (reports/.checkpoints/<stage>.done) so a re-run skips
    completed stages unless --force is given.
  • A LaTeX export (provenance_measured_values.tex) to embed in every report.

CLI:
    python te_provenance.py --reports-dir ./reports --tex      # write .tex
    python te_provenance.py --reports-dir ./reports --show     # print manifest
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

_TOOLS = ["mafft", "bedtools", "liftOver", "colabfold_batch", "samtools",
          "FastTree", "iqtree2", "python3"]


def _git_sha(repo: Path) -> str:
    """Return the repo's current commit SHA, or "unknown" if git is unavailable."""
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _tool_version(tool: str) -> str:
    """Best-effort version string for an external tool.

    Returns "not found" if it isn't on PATH; otherwise tries the common version
    flags and returns the first output line (truncated), falling back to the
    resolved executable path.
    """
    exe = shutil.which(tool)
    if not exe:
        return "not found"
    for flag in ("--version", "-version", "version", "-v"):
        try:
            r = subprocess.run([exe, flag], capture_output=True, text=True)
            out = (r.stdout or r.stderr).strip().splitlines()
            if out:
                return out[0][:80]
        except Exception:
            continue
    return exe


def file_sha256(path: str, limit_mb: int = 200) -> str:
    """SHA-256 of a file, streamed in chunks.

    Returns "missing" for absent files and a "skipped(...)" marker for files
    larger than limit_mb (hashing multi-GB genomes/BEDs isn't worth the cost).
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        return "missing"
    if p.stat().st_size > limit_mb * 1024 * 1024:
        return f"skipped(>{limit_mb}MB):size={p.stat().st_size}"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_path(reports_dir) -> Path:
    return Path(reports_dir) / "provenance.json"


def load(reports_dir) -> dict:
    """Load the provenance manifest dict, or {} if none exists / is unreadable."""
    mp = _manifest_path(reports_dir)
    if mp.exists():
        try:
            return json.loads(mp.read_text())
        except Exception:
            pass
    return {}


def _save(reports_dir, manifest: dict):
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    _manifest_path(reports_dir).write_text(json.dumps(manifest, indent=2))


def init(reports_dir, repo_root: str = ".") -> dict:
    """Initialise (or load) the manifest with environment + tool versions."""
    m = load(reports_dir)
    if "environment" not in m:
        m["environment"] = {
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "git_sha": _git_sha(Path(repo_root).resolve()),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "tools": {t: _tool_version(t) for t in _TOOLS},
        }
        m.setdefault("stages", {})
        _save(reports_dir, m)
    return m


def record_stage(reports_dir, stage: str, params: dict, inputs: list,
                 status: str = "completed"):
    """Record a stage's params, input hashes, status, and timestamp."""
    m = init(reports_dir)
    m.setdefault("stages", {})[stage] = {
        "status": status,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "params": params,
        "inputs": {i: file_sha256(i) for i in inputs if i},
    }
    _save(reports_dir, m)


# ── checkpoints / resume ───────────────────────────────────────────────────────

def _ckpt_dir(reports_dir) -> Path:
    d = Path(reports_dir) / ".checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def mark_done(reports_dir, stage: str):
    """Write a <stage>.done checkpoint so a re-run can skip this stage."""
    (_ckpt_dir(reports_dir) / f"{stage}.done").write_text(
        datetime.datetime.now().isoformat(timespec="seconds"))


def is_done(reports_dir, stage: str) -> bool:
    """True if the given stage has a completed checkpoint."""
    return (_ckpt_dir(reports_dir) / f"{stage}.done").exists()


def clear(reports_dir, stage: str = ""):
    """Delete one stage's checkpoint, or all of them when stage is empty (--force)."""
    cd = _ckpt_dir(reports_dir)
    if stage:
        (cd / f"{stage}.done").unlink(missing_ok=True)
    else:
        for f in cd.glob("*.done"):
            f.unlink()


# ── LaTeX export ───────────────────────────────────────────────────────────────

def to_tex(reports_dir) -> str:
    """Export environment/tool provenance as \\provXxx LaTeX macros.

    Writes provenance_measured_values.tex (git SHA, platform, Python, tool
    versions, stage count) for \\input into the report, and returns its path.
    """
    m = init(reports_dir)
    env = m.get("environment", {})
    tools = env.get("tools", {})
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_stages = len(m.get("stages", {}))

    def esc(s):
        return str(s).replace("_", r"\_").replace("&", r"\&").replace("#", r"\#")

    lines = [
        "% Auto-generated by te_provenance.py", f"% {ts}", "",
        rf"\providecommand{{\provGitSha}}{{\texttt{{{esc(env.get('git_sha','?')[:12])}}}}}",
        rf"\providecommand{{\provPlatform}}{{{esc(env.get('platform','?'))}}}",
        rf"\providecommand{{\provPython}}{{{esc(env.get('python','?'))}}}",
        rf"\providecommand{{\provNStages}}{{{n_stages}}}",
        rf"\providecommand{{\provMafftVer}}{{{esc(tools.get('mafft','?'))}}}",
        rf"\providecommand{{\provBedtoolsVer}}{{{esc(tools.get('bedtools','?'))}}}",
        rf"\providecommand{{\provCreated}}{{{esc(env.get('created','?'))}}}",
        "",
    ]
    out = Path(reports_dir) / "provenance_measured_values.tex"
    out.write_text("\n".join(lines))
    return str(out)


def main():
    """CLI entry point: init the manifest, then --show / --tex / --clear-checkpoints."""
    ap = argparse.ArgumentParser(description="GAMECA provenance manifest tool")
    ap.add_argument("--reports-dir", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--tex", action="store_true", help="write provenance_measured_values.tex")
    ap.add_argument("--show", action="store_true", help="print the manifest")
    ap.add_argument("--clear-checkpoints", action="store_true")
    args = ap.parse_args()

    init(args.reports_dir, args.repo_root)
    if args.clear_checkpoints:
        clear(args.reports_dir)
        print("Cleared checkpoints.")
    if args.tex:
        print("Wrote", to_tex(args.reports_dir))
    if args.show or not (args.tex or args.clear_checkpoints):
        print(json.dumps(load(args.reports_dir), indent=2))


if __name__ == "__main__":
    main()
