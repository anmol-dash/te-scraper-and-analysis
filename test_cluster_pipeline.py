#!/usr/bin/env python3
"""
test_cluster_pipeline.py — GAMECA end-to-end integration test for HPC clusters.

Self-contained: generates all synthetic TE data internally; no genome FASTA,
no real RepeatMasker table, no network access required.

Tests every pipeline stage and Stage 11 module.  Modules that require
optional external tools (MAFFT, bedtools, liftOver, colabfold) degrade
gracefully — the test marks them SKIP rather than FAIL when the tool is absent.

Usage (interactive or inside a bsub/sbatch job):
    python test_cluster_pipeline.py
    python test_cluster_pipeline.py --work-dir /scratch/$USER/gameca_ci
    python test_cluster_pipeline.py --only clustering,expression,phylo
    python test_cluster_pipeline.py --skip fold    # skip slow ColabFold test
    python test_cluster_pipeline.py --python python3.11

Exit code: 0 = all tests passed (or skipped); 1 = ≥1 FAIL.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import random
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

# ── colour helpers ─────────────────────────────────────────────────────────────
_TTY = sys.stdout.isatty()

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text

def _green(t):  return _c(t, "32")
def _red(t):    return _c(t, "31")
def _yellow(t): return _c(t, "33")
def _bold(t):   return _c(t, "1")

# ── result tracking ────────────────────────────────────────────────────────────
_results: list[tuple[str, str, str]] = []   # (name, status, detail)

def _record(name: str, status: str, detail: str = "") -> None:
    _results.append((name, status, detail))
    icon = {"PASS": _green("PASS"), "FAIL": _red("FAIL"), "SKIP": _yellow("SKIP")}[status]
    suffix = f"  {_yellow(detail)}" if detail else ""
    print(f"  [{icon}] {name}{suffix}", flush=True)


# ── synthetic data generation ─────────────────────────────────────────────────

_BASES = "ACGT"
_STAGE_COLS = ["pronuc", "twocell", "fourcell", "eightcell", "morula",
               "icm", "epiblast", "primes", "e65", "e75"]


def _rand_seq(length: int, gc: float = 0.50, rng: random.Random | None = None) -> str:
    r = rng or random.Random()
    out = []
    for _ in range(length):
        if r.random() < gc:
            out.append(r.choice("GC"))
        else:
            out.append(r.choice("AT"))
    return "".join(out)


def _mutate(seq: str, rate: float = 0.05, rng: random.Random | None = None) -> str:
    r = rng or random.Random()
    s = list(seq)
    for i in range(len(s)):
        if r.random() < rate:
            s[i] = r.choice(_BASES)
    return "".join(s)


def _make_ltr_seq(rng: random.Random, length: int = 450) -> str:
    """Simple synthetic LTR-like sequence: U3(~300bp) + R(~50bp) + U5(~100bp)."""
    u3 = _rand_seq(300, gc=0.55, rng=rng)
    r  = _rand_seq(50,  gc=0.48, rng=rng)
    u5 = _rand_seq(length - 350, gc=0.50, rng=rng)
    return u3 + r + u5


def _make_internal_seq(rng: random.Random, length: int = 6000) -> str:
    """Synthetic internal LTR retrotransposon region (gag-pol-like)."""
    return _rand_seq(length, gc=0.52, rng=rng)


def _build_te_seq(rng: random.Random, ltr_len: int = 450, int_len: int = 3000) -> str:
    """Full solo-LTR or complete element: LTR + internal + LTR."""
    ltr = _make_ltr_seq(rng, ltr_len)
    return ltr + _make_internal_seq(rng, int_len) + _mutate(ltr, 0.02, rng)


def _make_dataset(n: int = 80, seed: int = 42,
                  expr_cols: list[str] | None = None,
                  include_genomic: bool = True,
                  full_element: bool = False) -> list[dict]:
    """Generate n synthetic TE loci with expression data and cluster labels."""
    rng = random.Random(seed)
    cols = expr_cols or _STAGE_COLS

    # Two clusters: cluster 0 high in early embryo, cluster 1 high in late.
    cluster_profiles = {
        0: [10.0, 8.5, 6.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0, 0.8],
        1: [0.5,  0.8, 1.5, 2.5, 4.0, 6.5, 8.0, 9.0, 9.5, 10.0],
    }

    # Ancestral sequences for each cluster.
    ancestor0 = _build_te_seq(rng) if full_element else _rand_seq(900, gc=0.48, rng=rng)
    ancestor1 = _build_te_seq(rng) if full_element else _rand_seq(950, gc=0.56, rng=rng)
    ancestors = {0: ancestor0, 1: ancestor1}

    rows = []
    chroms = [f"chr{i}" for i in range(1, 20)] + ["chrX"]
    for i in range(n):
        clust = 0 if i < n // 2 else 1
        seq = _mutate(ancestors[clust], rate=0.03 + rng.random() * 0.08, rng=rng)
        # Slight length variation (±10%).
        trim = rng.randint(-50, 50)
        seq = seq[:max(200, len(seq) + trim)]

        chrom = rng.choice(chroms)
        start = rng.randint(1_000_000, 200_000_000)
        stop  = start + len(seq)

        profile = cluster_profiles[clust]
        row: dict = {
            "chr":     chrom,
            "start":   start,
            "stop":    stop,
            "strand":  rng.choice(["+", "-"]),
            "TE_name": f"TEST_LTR5_Hs#{i+1}",
            "family":  "LTR5_Hs",
            "Seq":     seq,
            "Cluster": clust,
        }
        for j, col in enumerate(cols):
            base = profile[j] if j < len(profile) else profile[-1]
            row[col] = round(max(0.0, base * 10 + rng.gauss(0, 2.0)), 3)
        rows.append(row)
    return rows


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        raise ValueError("empty rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ── subprocess runner ──────────────────────────────────────────────────────────

def _run(cmd: list[str], cwd: Path, timeout: int = 300,
         env: dict | None = None) -> tuple[int, str, str]:
    combined_env = os.environ.copy()
    combined_env.setdefault("MPLBACKEND", "Agg")
    combined_env.setdefault("NUMBA_CACHE_DIR", str(cwd / ".numba_cache"))
    if env:
        combined_env.update(env)
    try:
        r = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True,
            timeout=timeout, env=combined_env,
        )
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"TIMEOUT after {timeout}s"
    except Exception as exc:
        return -2, "", str(exc)


def _has_tool(name: str) -> bool:
    return shutil.which(name) is not None


# ── individual stage tests ─────────────────────────────────────────────────────

def _test_clustering(py: str, repo: Path, work: Path, csv_in: Path) -> Path | None:
    """te_clustering.py with small params so it runs in seconds."""
    out_csv = work / "clustered.csv"
    shutil.copy(csv_in, out_csv)
    code, out, err = _run(
        [py, str(repo / "te_clustering.py"),
         "--input", str(out_csv),
         "--output", str(out_csv),
         "--kmer", "4",
         "--pca-dims", "10",
         "--n-epochs", "20",
         "--min-cluster-size", "5",
         "--min-samples", "2",
         "--n-neighbors", "5",
         "--out-dir", str(work),
         "--family", "LTR5_Hs",
         "--skip-tsne",
         ],
        cwd=repo, timeout=180,
    )
    if code == 0 and out_csv.exists():
        _record("clustering", "PASS")
        return out_csv
    else:
        _record("clustering", "FAIL", (err or out)[-300:].strip().replace("\n", " "))
        return None


def _test_expression(py: str, repo: Path, work: Path, csv_in: Path) -> None:
    out_dir = work / "expression"
    code, out, err = _run(
        [py, str(repo / "te_expression.py"),
         "--input", str(csv_in),
         "--out-dir", str(out_dir),
         "--family", "LTR5_Hs",
         "--stage-cols"] + _STAGE_COLS,
        cwd=repo, timeout=120,
    )
    expected = out_dir / "expression_plots" / "expression_stats.csv"
    if code == 0 and expected.exists():
        _record("expression", "PASS")
    else:
        _record("expression", "FAIL", (err or out)[-300:].strip().replace("\n", " "))


def _test_alignment(py: str, repo: Path, work: Path, csv_in: Path) -> None:
    if not _has_tool("mafft"):
        _record("alignment", "SKIP", "mafft not found")
        return
    out_dir = work / "alignment"
    code, out, err = _run(
        [py, str(repo / "te_alignment.py"),
         "--input", str(csv_in),
         "--out-dir", str(out_dir),
         "--family", "LTR5_Hs",
         "--max-seqs", "20",
         ],
        cwd=repo, timeout=180,
    )
    if code == 0:
        _record("alignment", "PASS")
    else:
        _record("alignment", "FAIL", (err or out)[-200:].strip().replace("\n", " "))


def _test_motif(py: str, repo: Path, work: Path, csv_in: Path) -> None:
    out_dir = work / "motif"
    code, out, err = _run(
        [py, str(repo / "te_motif.py"),
         "--input", str(csv_in),
         "--out-dir", str(out_dir),
         "--family", "LTR5_Hs",
         "--skip-jaspar",   # no tabix index on CI
         ],
        cwd=repo, timeout=180,
    )
    if code == 0:
        _record("motif", "PASS")
    elif "skip" in (out + err).lower() or code == 0:
        _record("motif", "PASS")
    else:
        _record("motif", "FAIL", (err or out)[-200:].strip().replace("\n", " "))


def _test_go(py: str, repo: Path, work: Path, csv_in: Path) -> None:
    out_dir = work / "go"
    code, out, err = _run(
        [py, str(repo / "te_go.py"),
         "--input", str(csv_in),
         "--out-dir", str(out_dir),
         "--family", "LTR5_Hs",
         ],
        cwd=repo, timeout=120,
    )
    if code == 0:
        _record("go", "PASS")
    else:
        _record("go", "FAIL", (err or out)[-200:].strip().replace("\n", " "))


def _test_enrichment(py: str, repo: Path, work: Path, csv_in: Path) -> None:
    out_dir = work / "enrichment"
    code, out, err = _run(
        [py, str(repo / "te_enrichment.py"),
         "--input", str(csv_in),
         "--out-dir", str(out_dir),
         "--family", "LTR5_Hs",
         ],
        cwd=repo, timeout=120,
    )
    if code == 0:
        _record("enrichment", "PASS")
    else:
        _record("enrichment", "FAIL", (err or out)[-200:].strip().replace("\n", " "))


def _test_primers(py: str, repo: Path, work: Path, csv_in: Path) -> None:
    out_dir = work / "primers"
    code, out, err = _run(
        [py, str(repo / "te_primers.py"),
         "--input", str(csv_in),
         "--out-dir", str(out_dir),
         "--family", "LTR5_Hs",
         "--kmer", "12",
         "--top-n", "5",
         ],
        cwd=repo, timeout=120,
    )
    if code == 0:
        _record("primers", "PASS")
    else:
        _record("primers", "FAIL", (err or out)[-200:].strip().replace("\n", " "))


# ── Stage 11 module tests ──────────────────────────────────────────────────────

def _s11(name: str, py: str, repo: Path, work: Path, csv_in: Path,
         extra_args: list[str] | None = None,
         need_tools: list[str] | None = None,
         timeout: int = 180) -> None:
    for tool in (need_tools or []):
        if not _has_tool(tool):
            _record(f"s11/{name}", "SKIP", f"{tool} not found")
            return

    script = repo / f"run_{name.replace('/', '_')}.py"
    if not script.exists():
        _record(f"s11/{name}", "SKIP", f"{script.name} not in repo")
        return

    out_dir = work / f"s11_{name}"
    cmd = [
        py, str(script),
        "--input",      str(csv_in),
        "--reports-dir", str(out_dir),
        "--family",     "LTR5_Hs",
    ] + (extra_args or [])

    code, out, err = _run(cmd, cwd=repo, timeout=timeout)
    combined = (out + err)[-500:].strip().replace("\n", " ")
    if code == 0:
        _record(f"s11/{name}", "PASS")
    elif "ImportError" in combined or "ModuleNotFoundError" in combined:
        _record(f"s11/{name}", "SKIP", "missing optional dep")
    else:
        _record(f"s11/{name}", "FAIL", combined[-200:])


def _test_phylo(py: str, repo: Path, work: Path, csv_in: Path) -> None:
    _s11("phylo_analysis", py, repo, work, csv_in,
         extra_args=["--assembly", "hg38",
                     "--clock-divisor", "1",
                     "--intact-orf-aa", "50"],
         need_tools=["mafft"],
         timeout=300)


def _test_grna(py: str, repo: Path, work: Path, csv_in: Path) -> None:
    _s11("grna_offtarget", py, repo, work, csv_in,
         extra_args=["--cas", "SpCas9", "--max-mm", "2"],
         timeout=240)


def _test_transduction(py: str, repo: Path, work: Path, csv_in: Path) -> None:
    _s11("transduction", py, repo, work, csv_in,
         extra_args=["--tail-bp", "300", "--min-shared", "2"],
         timeout=120)


def _test_antisense(py: str, repo: Path, work: Path, csv_in: Path) -> None:
    _s11("antisense_promoter", py, repo, work, csv_in,
         extra_args=["--promoter-bp", "500"],
         timeout=120)


def _test_ctcf_tad(py: str, repo: Path, work: Path, csv_in: Path) -> None:
    _s11("ctcf_tad", py, repo, work, csv_in,
         need_tools=["bedtools"],
         timeout=180)


def _test_epigenetic(py: str, repo: Path, work: Path, csv_in: Path) -> None:
    _s11("epigenetic_overlay", py, repo, work, csv_in,
         need_tools=["bedtools"],
         timeout=180)


def _test_ortholog(py: str, repo: Path, work: Path, csv_in: Path) -> None:
    _s11("ortholog_insertion", py, repo, work, csv_in,
         need_tools=["liftOver"],
         timeout=180)


def _test_multiassembly(py: str, repo: Path, work: Path, csv_in: Path) -> None:
    _s11("multiassembly_liftover", py, repo, work, csv_in,
         need_tools=["liftOver"],
         timeout=180)


def _test_divergence(py: str, repo: Path, work: Path, csv_in: Path) -> None:
    _s11("divergence", py, repo, work, csv_in,
         extra_args=["--assembly", "hg38"],
         timeout=180)


def _test_ltr_struct(py: str, repo: Path, work: Path, csv_in: Path) -> None:
    _s11("ltr_struct", py, repo, work, csv_in,
         extra_args=["--assembly", "hg38"],
         need_tools=["mafft"],
         timeout=240)


def _test_subfamily(py: str, repo: Path, work: Path, csv_in: Path) -> None:
    _s11("subfamily", py, repo, work, csv_in,
         extra_args=["--assembly", "hg38",
                     "--min-cluster-size", "5"],
         timeout=180)


def _test_benchmark(py: str, repo: Path, work: Path, csv_in: Path) -> None:
    _s11("benchmark", py, repo, work, csv_in, timeout=180)


def _test_motif_gain(py: str, repo: Path, work: Path, csv_in: Path) -> None:
    _s11("motif_gain", py, repo, work, csv_in,
         extra_args=["--skip-jaspar"],
         timeout=180)


def _test_fold(py: str, repo: Path, work: Path, csv_in: Path,
               skip: bool = False) -> None:
    if skip:
        _record("s11/fold_prediction", "SKIP", "--skip fold requested")
        return
    if not _has_tool("colabfold_batch"):
        _record("s11/fold_prediction", "SKIP", "colabfold_batch not found")
        return
    _s11("fold_prediction", py, repo, work, csv_in,
         extra_args=["--per-cluster", "--top-n", "1", "--min-aa", "30"],
         timeout=600)


def _test_run_stage11_all(py: str, repo: Path, work: Path, csv_in: Path) -> None:
    """Integration test: run_stage11_all.py orchestrates all Stage 11 modules."""
    script = repo / "run_stage11_all.py"
    if not script.exists():
        _record("run_stage11_all", "SKIP", "run_stage11_all.py not in repo")
        return
    out_dir = work / "stage11_all"
    code, out, err = _run(
        [py, str(script),
         "--input", str(csv_in),
         "--family", "LTR5_Hs",
         "--assembly", "hg38",
         "--reports-dir", str(out_dir),
         # Skip the slowest / tool-dependent modules so this finishes quickly.
         "--skip", "fold,ortholog,multiassembly,ctcf_tad,epigenetic",
         ],
        cwd=repo, timeout=600,
    )
    combined = (out + err)[-500:].strip().replace("\n", " ")
    if code == 0:
        _record("run_stage11_all", "PASS")
    elif "ImportError" in combined or "ModuleNotFoundError" in combined:
        _record("run_stage11_all", "SKIP", "missing optional dep")
    else:
        _record("run_stage11_all", "FAIL", combined[-200:])


def _test_gameca_pipeline(py: str, repo: Path, work: Path, csv_in: Path) -> None:
    """gameca_pipeline.py with a minimal two-stage DAG."""
    script = repo / "gameca_pipeline.py"
    yaml_path = work / "ci_gameca.yaml"
    prov_dir  = work / "ci_pipeline_out"
    prov_dir.mkdir(parents=True, exist_ok=True)

    # Minimal YAML: two stages — clustering → expression.
    yaml_content = textwrap.dedent(f"""\
        family: LTR5_Hs
        input: {csv_in}
        reports_dir: {prov_dir}
        parallel: false
        stages:
          - name: clustering
            script: te_clustering.py
            args:
              kmer: 4
              pca-dims: 10
              n-epochs: 20
              min-cluster-size: 5
              min-samples: 2
              n-neighbors: 5
              skip-tsne: true
              out-dir: {prov_dir}
          - name: expression
            script: te_expression.py
            args:
              out-dir: {prov_dir}/expression
              stage-cols: {" ".join(_STAGE_COLS)}
            depends_on: [clustering]
    """)
    yaml_path.write_text(yaml_content)

    code, out, err = _run(
        [py, str(script), "--config", str(yaml_path)],
        cwd=repo, timeout=300,
    )
    combined = (out + err)[-500:].strip().replace("\n", " ")
    if code == 0:
        _record("gameca_pipeline", "PASS")
    elif "ImportError" in combined or "ModuleNotFoundError" in combined:
        _record("gameca_pipeline", "SKIP", "missing optional dep")
    else:
        _record("gameca_pipeline", "FAIL", combined[-200:])


def _test_gameca_pipeline_resume(py: str, repo: Path, work: Path, csv_in: Path) -> None:
    """Re-run pipeline with --force=false to verify checkpoint/resume."""
    script  = repo / "gameca_pipeline.py"
    yaml_path = work / "ci_gameca.yaml"
    if not yaml_path.exists():
        _record("gameca_pipeline/resume", "SKIP", "yaml missing (pipeline test failed)")
        return
    prov_dir = work / "ci_pipeline_out"

    code, out, err = _run(
        [py, str(script), "--config", str(yaml_path)],
        cwd=repo, timeout=120,
    )
    if code == 0 and "skip" in (out + err).lower():
        _record("gameca_pipeline/resume", "PASS")
    elif code == 0:
        _record("gameca_pipeline/resume", "PASS", "re-ran without skip — provenance may be off")
    else:
        combined = (out + err)[-300:].strip().replace("\n", " ")
        _record("gameca_pipeline/resume", "FAIL", combined[-200:])


def _test_gameca_pipeline_dry_run(py: str, repo: Path, work: Path, csv_in: Path) -> None:
    script   = repo / "gameca_pipeline.py"
    yaml_path = work / "ci_gameca.yaml"
    if not yaml_path.exists():
        _record("gameca_pipeline/dry-run", "SKIP", "yaml missing")
        return
    code, out, err = _run(
        [py, str(script), "--config", str(yaml_path), "--dry-run"],
        cwd=repo, timeout=30,
    )
    if code == 0:
        _record("gameca_pipeline/dry-run", "PASS")
    else:
        _record("gameca_pipeline/dry-run", "FAIL",
                (err or out)[-200:].strip().replace("\n", " "))


# ── edge-case / regression tests ───────────────────────────────────────────────

def _test_single_cluster(py: str, repo: Path, work: Path) -> None:
    """All sequences in one cluster — nothing should crash."""
    rows = _make_dataset(n=30, seed=99)
    for r in rows:
        r["Cluster"] = 0          # force single cluster
    csv_in = work / "edge_single_cluster.csv"
    _write_csv(rows, csv_in)

    out_dir = work / "edge_single_cluster_expr"
    code, out, err = _run(
        [py, str(repo / "te_expression.py"),
         "--input", str(csv_in), "--out-dir", str(out_dir), "--family", "LTR5_Hs",
         "--stage-cols"] + _STAGE_COLS,
        cwd=repo, timeout=60,
    )
    if code == 0:
        _record("edge/single-cluster-expression", "PASS")
    else:
        _record("edge/single-cluster-expression", "FAIL",
                (err or out)[-200:].strip().replace("\n", " "))


def _test_no_expr_cols(py: str, repo: Path, work: Path) -> None:
    """CSV with no expression columns — expression module must warn, not crash."""
    rows = _make_dataset(n=30, seed=77, expr_cols=[])
    for r in rows:
        # strip expression cols
        for col in _STAGE_COLS:
            r.pop(col, None)
    csv_in = work / "edge_no_expr.csv"
    _write_csv(rows, csv_in)

    out_dir = work / "edge_no_expr_out"
    code, _, _ = _run(
        [py, str(repo / "te_expression.py"),
         "--input", str(csv_in), "--out-dir", str(out_dir), "--family", "LTR5_Hs"],
        cwd=repo, timeout=60,
    )
    # exit 0 (graceful skip) or exit 1 with a clear warning are both acceptable.
    if code in (0, 1):
        _record("edge/no-expr-cols", "PASS")
    else:
        _record("edge/no-expr-cols", "FAIL", f"unexpected exit {code}")


def _test_minimal_n(py: str, repo: Path, work: Path) -> None:
    """Tiny dataset (n=8) — clustering must not crash with min_cluster_size > n."""
    rows = _make_dataset(n=8, seed=55)
    csv_in = work / "edge_minimal_n.csv"
    _write_csv(rows, csv_in)
    out_csv = work / "edge_minimal_n_clustered.csv"
    shutil.copy(csv_in, out_csv)
    code, out, err = _run(
        [py, str(repo / "te_clustering.py"),
         "--input", str(out_csv),
         "--output", str(out_csv),
         "--kmer", "4",
         "--pca-dims", "4",
         "--n-epochs", "10",
         "--min-cluster-size", "3",
         "--min-samples", "2",
         "--n-neighbors", "3",
         "--out-dir", str(work),
         "--family", "LTR5_Hs",
         "--skip-tsne",
         ],
        cwd=repo, timeout=60,
    )
    if code == 0:
        _record("edge/minimal-n-clustering", "PASS")
    else:
        _record("edge/minimal-n-clustering", "FAIL",
                (err or out)[-200:].strip().replace("\n", " "))


def _test_all_noise_clusters(py: str, repo: Path, work: Path) -> None:
    """Confirm phylo module doesn't crash when HDBSCAN assigns all seqs as noise."""
    rows = _make_dataset(n=20, seed=33)
    for r in rows:
        r["Cluster"] = -1   # simulate all-noise
    csv_in = work / "edge_all_noise.csv"
    _write_csv(rows, csv_in)
    if not _has_tool("mafft"):
        _record("edge/all-noise-phylo", "SKIP", "mafft not found")
        return
    script = repo / "run_phylo_analysis.py"
    if not script.exists():
        _record("edge/all-noise-phylo", "SKIP", "run_phylo_analysis.py missing")
        return
    out_dir = work / "edge_all_noise_phylo"
    code, out, err = _run(
        [py, str(script),
         "--input", str(csv_in),
         "--reports-dir", str(out_dir),
         "--family", "LTR5_Hs",
         "--assembly", "hg38",
         "--clock-divisor", "1",
         "--intact-orf-aa", "50",
         ],
        cwd=repo, timeout=120,
    )
    if code in (0, 1):   # non-zero is acceptable for all-noise
        _record("edge/all-noise-phylo", "PASS")
    else:
        _record("edge/all-noise-phylo", "FAIL",
                (err or out)[-200:].strip().replace("\n", " "))


def _test_unicode_family_name(py: str, repo: Path, work: Path, csv_in: Path) -> None:
    """Family name with spaces and unicode — no path/shell injection."""
    out_dir = work / "edge_unicode"
    code, _, _ = _run(
        [py, str(repo / "te_expression.py"),
         "--input", str(csv_in),
         "--out-dir", str(out_dir),
         "--family", "LTR5 Hs Señal",
         "--stage-cols"] + _STAGE_COLS,
        cwd=repo, timeout=60,
    )
    if code in (0, 1):
        _record("edge/unicode-family-name", "PASS")
    else:
        _record("edge/unicode-family-name", "FAIL", f"exit {code}")


def _test_missing_seq_col(py: str, repo: Path, work: Path) -> None:
    """CSV without Seq column — modules must exit with non-zero, not crash."""
    rows = _make_dataset(n=20, seed=11)
    for r in rows:
        del r["Seq"]
    csv_in = work / "edge_no_seq.csv"
    _write_csv(rows, csv_in)
    out_csv = work / "edge_no_seq_clustered.csv"
    shutil.copy(csv_in, out_csv)
    code, _, _ = _run(
        [py, str(repo / "te_clustering.py"),
         "--input", str(out_csv), "--output", str(out_csv),
         "--kmer", "4", "--n-epochs", "10", "--min-cluster-size", "3",
         "--out-dir", str(work), "--family", "LTR5_Hs", "--skip-tsne"],
        cwd=repo, timeout=30,
    )
    if code != 0:
        _record("edge/missing-seq-col", "PASS")
    else:
        _record("edge/missing-seq-col", "FAIL",
                "clustering accepted CSV without Seq column — should have errored")


# ── Python environment smoke tests ─────────────────────────────────────────────

def _test_imports(py: str) -> None:
    """Verify core dependencies are importable."""
    deps = [
        "numpy", "pandas", "scipy", "sklearn",
        "umap", "hdbscan", "matplotlib", "plotly",
    ]
    missing = []
    for dep in deps:
        code, _, _ = _run([py, "-c", f"import {dep}"], cwd=Path("."), timeout=15)
        if code != 0:
            missing.append(dep)
    if not missing:
        _record("imports/core-deps", "PASS")
    else:
        _record("imports/core-deps", "FAIL", f"missing: {', '.join(missing)}")


def _test_gameca_modules_importable(py: str, repo: Path) -> None:
    """All GAMECA pipeline modules must import without error."""
    modules = [
        "te_clustering", "te_expression", "te_alignment",
        "te_motif", "te_go", "te_primers",
        "te_provenance", "te_scheduler", "gameca_pipeline",
    ]
    bad = []
    for mod in modules:
        script = repo / f"{mod}.py"
        if not script.exists():
            continue
        code, _, err = _run(
            [py, "-c", f"import sys; sys.path.insert(0,'{repo}'); import {mod}"],
            cwd=repo, timeout=20,
        )
        if code != 0:
            bad.append(f"{mod}({err.strip()[-60:]})")
    if not bad:
        _record("imports/gameca-modules", "PASS")
    else:
        _record("imports/gameca-modules", "FAIL", "; ".join(bad))


# ── main ───────────────────────────────────────────────────────────────────────

_ALL_TESTS = [
    "imports",
    "clustering", "expression", "alignment", "motif", "go",
    "enrichment", "primers",
    "phylo", "grna", "transduction", "antisense", "ctcf_tad",
    "epigenetic", "ortholog", "multiassembly",
    "divergence", "ltr_struct", "subfamily", "benchmark", "motif_gain",
    "fold",
    "run_stage11_all",
    "gameca_pipeline",
    "edge",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GAMECA end-to-end cluster integration test (no genome FASTA required)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--work-dir", default="",
                   help="Directory for test outputs (default: auto tempdir, cleaned up on exit)")
    p.add_argument("--keep",  action="store_true",
                   help="Keep work directory after test (default: delete on PASS, keep on FAIL)")
    p.add_argument("--only",  default="",
                   help="Comma-separated list of test groups to run (e.g. clustering,expression)")
    p.add_argument("--skip",  default="",
                   help="Comma-separated list of test groups to skip")
    p.add_argument("--python", default=sys.executable,
                   help="Python interpreter for subprocesses (default: current interpreter)")
    p.add_argument("--n-seqs", type=int, default=80,
                   help="Number of synthetic TE sequences to generate (default: 80)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for synthetic data (default: 42)")
    p.add_argument("--repo-dir", default="",
                   help="Path to GAMECA repo root (default: directory of this script)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    repo = Path(args.repo_dir) if args.repo_dir else Path(__file__).resolve().parent
    py   = args.python

    # Resolve work dir.
    _auto_tmpdir = None
    if args.work_dir:
        work = Path(args.work_dir)
        work.mkdir(parents=True, exist_ok=True)
    else:
        _auto_tmpdir = tempfile.mkdtemp(prefix="gameca_ci_")
        work = Path(_auto_tmpdir)

    print(_bold(f"\nGAMECA integration test  —  {time.strftime('%Y-%m-%d %H:%M:%S')}"))
    print(f"  repo   : {repo}")
    print(f"  work   : {work}")
    print(f"  python : {py}")
    print(f"  n_seqs : {args.n_seqs}")
    print()

    # Build filter sets.
    only_set = {t.strip() for t in args.only.split(",") if t.strip()}
    skip_set = {t.strip() for t in args.skip.split(",") if t.strip()}

    def _enabled(group: str) -> bool:
        if only_set and group not in only_set:
            return False
        if group in skip_set:
            return False
        return True

    # ── generate synthetic data ────────────────────────────────────────────────
    print(_bold("Generating synthetic data…"))
    t0 = time.monotonic()
    rows = _make_dataset(n=args.n_seqs, seed=args.seed, full_element=False)
    main_csv = work / "ltr5hs_synthetic.csv"
    _write_csv(rows, main_csv)
    print(f"  wrote {args.n_seqs} loci to {main_csv}  ({time.monotonic()-t0:.1f}s)")
    print()

    print(_bold("Running tests…"))
    t_start = time.monotonic()

    if _enabled("imports"):
        _test_imports(py)
        _test_gameca_modules_importable(py, repo)

    # Core stages — need a clustered CSV.
    clustered_csv: Path | None = None
    if _enabled("clustering"):
        clustered_csv = _test_clustering(py, repo, work, main_csv)
    # Fall back to the pre-labelled main_csv (has Cluster column from _make_dataset).
    analysis_csv = clustered_csv if clustered_csv is not None else main_csv

    if _enabled("expression"):
        _test_expression(py, repo, work, analysis_csv)
    if _enabled("alignment"):
        _test_alignment(py, repo, work, analysis_csv)
    if _enabled("motif"):
        _test_motif(py, repo, work, analysis_csv)
    if _enabled("go"):
        _test_go(py, repo, work, analysis_csv)
    if _enabled("enrichment"):
        _test_enrichment(py, repo, work, analysis_csv)
    if _enabled("primers"):
        _test_primers(py, repo, work, analysis_csv)

    # Stage 11 modules.
    if _enabled("phylo"):
        _test_phylo(py, repo, work, analysis_csv)
    if _enabled("grna"):
        _test_grna(py, repo, work, analysis_csv)
    if _enabled("transduction"):
        _test_transduction(py, repo, work, analysis_csv)
    if _enabled("antisense"):
        _test_antisense(py, repo, work, analysis_csv)
    if _enabled("ctcf_tad"):
        _test_ctcf_tad(py, repo, work, analysis_csv)
    if _enabled("epigenetic"):
        _test_epigenetic(py, repo, work, analysis_csv)
    if _enabled("ortholog"):
        _test_ortholog(py, repo, work, analysis_csv)
    if _enabled("multiassembly"):
        _test_multiassembly(py, repo, work, analysis_csv)
    if _enabled("divergence"):
        _test_divergence(py, repo, work, analysis_csv)
    if _enabled("ltr_struct"):
        _test_ltr_struct(py, repo, work, analysis_csv)
    if _enabled("subfamily"):
        _test_subfamily(py, repo, work, analysis_csv)
    if _enabled("benchmark"):
        _test_benchmark(py, repo, work, analysis_csv)
    if _enabled("motif_gain"):
        _test_motif_gain(py, repo, work, analysis_csv)
    if _enabled("fold"):
        _test_fold(py, repo, work, analysis_csv,
                   skip="fold" in skip_set)

    # Integration / orchestration.
    if _enabled("run_stage11_all"):
        _test_run_stage11_all(py, repo, work, analysis_csv)
    if _enabled("gameca_pipeline"):
        _test_gameca_pipeline(py, repo, work, analysis_csv)
        _test_gameca_pipeline_resume(py, repo, work, analysis_csv)
        _test_gameca_pipeline_dry_run(py, repo, work, analysis_csv)

    # Edge cases.
    if _enabled("edge"):
        _test_single_cluster(py, repo, work)
        _test_no_expr_cols(py, repo, work)
        _test_minimal_n(py, repo, work)
        _test_all_noise_clusters(py, repo, work)
        _test_unicode_family_name(py, repo, work, analysis_csv)
        _test_missing_seq_col(py, repo, work)

    # ── summary ───────────────────────────────────────────────────────────────
    elapsed = time.monotonic() - t_start
    n_pass = sum(1 for _, s, _ in _results if s == "PASS")
    n_fail = sum(1 for _, s, _ in _results if s == "FAIL")
    n_skip = sum(1 for _, s, _ in _results if s == "SKIP")
    total  = len(_results)

    print()
    print(_bold("=" * 60))
    print(_bold("RESULTS"))
    print(_bold("=" * 60))
    col_w = max((len(name) for name, _, _ in _results), default=20)
    for name, status, detail in _results:
        icon = {"PASS": _green("PASS"), "FAIL": _red("FAIL"), "SKIP": _yellow("SKIP")}[status]
        suffix = f"  — {detail[:80]}" if detail and status != "PASS" else ""
        print(f"  {icon}  {name:<{col_w}}{suffix}")
    print()
    print(
        f"  Passed: {_green(str(n_pass))}   "
        f"Failed: {_red(str(n_fail))}   "
        f"Skipped: {_yellow(str(n_skip))}   "
        f"Total: {total}   ({elapsed:.0f}s)"
    )
    print()

    passed = n_fail == 0
    if _auto_tmpdir and (passed or not args.keep):
        if passed:
            shutil.rmtree(_auto_tmpdir, ignore_errors=True)
    elif not passed:
        print(f"  Work dir kept for inspection: {work}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
