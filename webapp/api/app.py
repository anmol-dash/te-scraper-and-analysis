#!/usr/bin/env python3
"""
TERRA Web API
==============
A thin HTTP layer over the *existing* TERRA pipeline. It does not reimplement
any science: it imports and calls the repository's own functions unchanged —

  clustering  → te_clustering.clustering_analysis()
  guides      → run_grna_analysis.{score_candidates, greedy_grna_set, fig_*}
  family pull → te_prep.py  (subprocess, exactly as the CLI runs it)

The matplotlib / plotly writers in those modules pick their format from the
output file's extension, so we simply hand them ``.png`` / ``.html`` paths and
get web-ready artefacts with zero changes to the pipeline code.

Everything runs as a background job (thread) with a live captured log, so the
static GitHub-Pages front-end can start a run, poll status, and then render the
figures + tables the pipeline produced.

Run locally:
    pip install -r requirements.txt
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import traceback
import uuid
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

# ── Make the repository's pipeline modules importable ────────────────────────
# webapp/api/app.py  →  repo root is two levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

# Pipeline modules (imported, never modified) --------------------------------
import te_clustering  # noqa: E402
import run_grna_analysis as grna  # noqa: E402

# ── Config ───────────────────────────────────────────────────────────────────
JOBS_ROOT = Path(os.environ.get("TERRA_JOBS_DIR",
                                str(Path(tempfile.gettempdir())))) / "terra_webapp_jobs"
JOBS_ROOT.mkdir(parents=True, exist_ok=True)

# Persistent cache of *downloaded sequences* for the family+assembly path, keyed
# by (family, assembly, source, max_loci) — the inputs that determine what gets
# fetched. Clustering parameters (kmer, min_cluster_size, UMAP knobs) are NOT in
# the key, so re-clustering the same family with different parameters is a cache
# hit and never re-downloads from UCSC/Dfam.
CACHE_ROOT = Path(os.environ.get("TERRA_CACHE_DIR", str(JOBS_ROOT / "_seqcache")))
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
_CACHE_LOCKS: dict[str, threading.Lock] = {}
_CACHE_LOCKS_GUARD = threading.Lock()

MAX_UPLOAD_MB = int(os.environ.get("TERRA_MAX_UPLOAD_MB", "50"))
ALLOWED_ASSEMBLIES = ["hg38", "hg19", "mm10", "mm39", "hs1", "chm13"]

app = FastAPI(title="TERRA Web API", version="1.0.0")

# Allow the GitHub Pages origin (and anything else) to call the API. Override
# with TERRA_CORS_ORIGINS="https://user.github.io,https://foo" to lock down.
_origins_env = os.environ.get("TERRA_CORS_ORIGINS", "*")
_origins = ["*"] if _origins_env.strip() == "*" else [
    o.strip() for o in _origins_env.split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═════════════════════════════════════════════════════════════════════════════
# Job manager
# ═════════════════════════════════════════════════════════════════════════════

class _LogWriter(io.TextIOBase):
    """File-like object that appends everything written to a job's log list."""

    def __init__(self, job: "Job"):
        self._job = job
        self._buf = ""

    def write(self, s: str) -> int:  # noqa: D401
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._job.log.append(line)
        # keep the live tail from unbounded growth
        if len(self._job.log) > 5000:
            del self._job.log[:-5000]
        return len(s)

    def flush(self):  # noqa: D401
        if self._buf:
            self._job.log.append(self._buf)
            self._buf = ""


class Job:
    def __init__(self, kind: str):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind                      # "cluster" | "guides"
        self.status = "queued"                # queued|running|done|error
        self.stage = ""
        self.error = None
        self.result: dict | None = None
        self.log: list[str] = []
        self.created = datetime.utcnow().isoformat() + "Z"
        self.dir = JOBS_ROOT / self.id
        self.dir.mkdir(parents=True, exist_ok=True)

    def set_stage(self, stage: str):
        self.stage = stage
        self.log.append(f"[stage] {stage}")

    def to_public(self) -> dict:
        return {
            "job_id": self.id,
            "kind": self.kind,
            "status": self.status,
            "stage": self.stage,
            "error": self.error,
            "result": self.result,
            "log_tail": self.log[-40:],
            "log_len": len(self.log),
            "created": self.created,
        }


JOBS: dict[str, Job] = {}


def _json_safe(obj):
    """Recursively replace non-finite floats (NaN/Inf) with None so the result is
    valid JSON — Starlette serialises with allow_nan=False and would 500 otherwise
    (e.g. the seq-vs-expr correlation is NaN when there is no expression)."""
    import math
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _run_in_thread(job: Job, target):
    def _wrap():
        job.status = "running"
        writer = _LogWriter(job)
        try:
            with redirect_stdout(writer), redirect_stderr(writer):
                job.result = _json_safe(target(job))
            writer.flush()
            job.status = "done"
            job.set_stage("done")
        except Exception as exc:  # noqa: BLE001
            writer.flush()
            job.error = f"{type(exc).__name__}: {exc}"
            job.log.append("TRACEBACK:\n" + traceback.format_exc())
            job.status = "error"
    t = threading.Thread(target=_wrap, daemon=True)
    t.start()


# ═════════════════════════════════════════════════════════════════════════════
# Data helpers
# ═════════════════════════════════════════════════════════════════════════════

def _read_table(path: Path) -> pd.DataFrame:
    """Read a CSV or XLSX into a DataFrame."""
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def _guess_roles(columns: list[str]) -> dict:
    """Heuristically guess which column is Sequence / Name / Start / Stop / etc."""
    low = {c: c.lower().strip() for c in columns}

    def find(cands, contains=False):
        for c in columns:
            lc = low[c]
            if contains:
                if any(k in lc for k in cands):
                    return c
            elif lc in cands:
                return c
        return None

    seq = find({"seq", "sequence", "dna", "seqs"}) or find(["seq", "sequence"], contains=True)
    name = (find({"name", "te_name", "te_id", "id", "locus"})
            or find(["name", "_id", "locus"], contains=True))
    start = find({"start"}) or find(["start"], contains=True)
    stop = find({"stop", "end"}) or find(["stop", "end"], contains=True)
    strand = find({"strand"})
    chrom = find({"chr", "chrom", "chromosome"})
    cluster = find({"cluster"})

    taken = {c for c in (seq, name, start, stop, strand, chrom, cluster) if c}
    reserved = grna._NON_EXPR  # canonical non-expression names used by the pipeline
    expression = []
    for c in columns:
        if c in taken:
            continue
        if low[c] in reserved:
            continue
        # numeric-looking columns are candidate expression tracks
        expression.append(c)
    return {
        "sequence": seq, "name": name, "start": start, "stop": stop,
        "strand": strand, "chr": chrom, "cluster": cluster,
        "expression": expression,
    }


def _apply_mapping(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Return a normalized frame the pipeline understands (a 'Seq' column, plus
    optional chr/start/stop/strand/Cluster and the chosen expression columns)."""
    seq_col = mapping.get("sequence")
    if not seq_col or seq_col not in df.columns:
        raise HTTPException(400, f"Sequence column '{seq_col}' not found in upload.")

    out = pd.DataFrame()
    out["Seq"] = df[seq_col].astype(str).str.strip()

    def carry(src_key, dst):
        col = mapping.get(src_key)
        if col and col in df.columns:
            out[dst] = df[col].values

    carry("name", "TE_name")
    carry("chr", "chr")
    carry("start", "start")
    carry("stop", "stop")
    carry("strand", "strand")
    carry("cluster", "Cluster")

    # Pass through any embedding coords already present (so the Guides tab can
    # draw the reagent-coverage-on-embedding figure without re-clustering).
    for c in ("umap_x", "umap_y", "tsne_x", "tsne_y", "pca_x", "pca_y"):
        if c in df.columns:
            out[c] = df[c].values

    # Expression columns keep their original names → auto-detected downstream.
    for c in mapping.get("expression", []) or []:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Drop empty sequences.
    out = out[out["Seq"].str.len() > 0].reset_index(drop=True)
    if len(out) == 0:
        raise HTTPException(400, "No non-empty sequences after applying the mapping.")
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Endpoints
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
def health():
    return {"ok": True, "service": "terra-web-api", "version": app.version,
            "assemblies": ALLOWED_ASSEMBLIES}


@app.post("/api/inspect")
async def inspect(file: UploadFile = File(...)):
    """Upload a CSV/XLSX; return its columns, a preview, and role guesses so the
    user can confirm which column is Sequence / Name / Start / Stop / expression."""
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB} MB limit.")
    tmp = JOBS_ROOT / f"_inspect_{uuid.uuid4().hex[:8]}{Path(file.filename or 'x.csv').suffix}"
    tmp.write_bytes(raw)
    try:
        df = _read_table(tmp)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Could not parse file: {exc}")
    finally:
        tmp.unlink(missing_ok=True)

    cols = [str(c) for c in df.columns]
    numeric = [str(c) for c in df.columns
               if pd.api.types.is_numeric_dtype(df[c])]
    preview = df.head(5).astype(str).where(pd.notnull(df.head(5)), "").to_dict(orient="records")
    return {
        "filename": file.filename,
        "n_rows": int(len(df)),
        "columns": cols,
        "numeric_columns": numeric,
        "preview": preview,
        "guesses": _guess_roles(cols),
    }


def _save_upload(file: UploadFile, dest_dir: Path) -> Path:
    dest = dest_dir / (file.filename or "upload.csv")
    dest.write_bytes(file.file.read())
    return dest


# ── Clustering ───────────────────────────────────────────────────────────────

@app.post("/api/cluster")
async def cluster(
    mode: str = Form(...),                       # "upload" | "family"
    file: UploadFile | None = File(None),
    mapping: str = Form("{}"),
    family: str = Form(""),
    assembly: str = Form("hg38"),
    source: str = Form("rmsk"),                   # rmsk | dfam
    max_loci: int = Form(0),
    kmer: int = Form(10),
    min_cluster_size: int = Form(100),
    n_neighbors: int = Form(30),
    min_dist: float = Form(0.0),
    min_samples: int = Form(5),
    compute_tsne: bool = Form(True),
):
    job = Job("cluster")
    JOBS[job.id] = job

    if mode == "upload":
        if file is None:
            raise HTTPException(400, "mode='upload' requires a file.")
        upload_path = _save_upload(file, job.dir)
        mp = json.loads(mapping or "{}")

        def work(j: Job):
            j.set_stage("Reading upload")
            df_raw = _read_table(upload_path)
            print(f"Loaded {len(df_raw):,} rows, {len(df_raw.columns)} columns")
            df = _apply_mapping(df_raw, mp)
            print(f"{len(df):,} usable sequences after mapping")
            return _do_cluster(j, df, family or "UPLOAD", kmer, min_cluster_size,
                               n_neighbors, min_dist, min_samples, compute_tsne)

        _run_in_thread(job, work)

    elif mode == "family":
        if not family.strip():
            raise HTTPException(400, "mode='family' requires a family name.")
        if assembly not in ALLOWED_ASSEMBLIES:
            raise HTTPException(400, f"assembly must be one of {ALLOWED_ASSEMBLIES}")

        def work(j: Job):
            df, from_cache = _fetch_family(j, family.strip(), assembly, source, max_loci)
            # The family path has no expression data — te_prep emits only genomic
            # metadata (swScore, milliDiv, Length, coordinates). Keep just the
            # columns clustering needs so none of that metadata is mistaken for
            # an expression track (no bogus "size by expression" view, and any
            # downstream guide design stays on uniform weights).
            keep = [c for c in ("Seq", "chr", "start", "stop", "strand", "TE_name")
                    if c in df.columns]
            df = df[keep]
            res = _do_cluster(j, df, family.strip(), kmer, min_cluster_size,
                              n_neighbors, min_dist, min_samples, compute_tsne)
            res["sequences_cached"] = from_cache
            return res

        _run_in_thread(job, work)
    else:
        raise HTTPException(400, "mode must be 'upload' or 'family'.")

    return {"job_id": job.id}


def _cache_key(family: str, assembly: str, source: str, max_loci: int) -> str:
    raw = f"{family.strip().lower()}\x1f{assembly}\x1f{source}\x1f{int(max_loci or 0)}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _cache_lock(key: str) -> threading.Lock:
    """One lock per cache key so concurrent identical requests fetch only once."""
    with _CACHE_LOCKS_GUARD:
        lk = _CACHE_LOCKS.get(key)
        if lk is None:
            lk = _CACHE_LOCKS[key] = threading.Lock()
        return lk


def _run_te_prep(job: Job, family: str, assembly: str, source: str,
                 max_loci: int, base_dir: Path) -> Path:
    """Shell the existing te_prep.py CLI to pull loci + sequences into base_dir."""
    import subprocess
    cmd = [sys.executable, str(REPO_ROOT / "te_prep.py"),
           family, assembly, "--base-dir", str(base_dir), "--source", source]
    if max_loci and max_loci > 0:
        cmd += ["--max-loci", str(max_loci)]
    print("RUN: " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, cwd=str(REPO_ROOT))
    for line in proc.stdout:  # stream te_prep output into the job log
        print(line.rstrip())
    proc.wait()
    clustered = base_dir / "clustered_data.csv"
    if proc.returncode != 0 or not clustered.exists():
        raise RuntimeError(
            f"te_prep.py failed (exit {proc.returncode}). "
            "Check the family name (case-sensitive) / assembly, or try source='dfam'."
        )
    return clustered


def _fetch_family(job: Job, family: str, assembly: str, source: str,
                  max_loci: int):
    """Return (df, from_cache). Sequences are cached on disk keyed by
    (family, assembly, source, max_loci); re-clustering with different clustering
    parameters reuses the cached download instead of hitting UCSC/Dfam again."""
    key = _cache_key(family, assembly, source, max_loci)
    cache_dir = CACHE_ROOT / key
    cached_csv = cache_dir / "clustered_data.csv"

    with _cache_lock(key):
        if cached_csv.exists() and cached_csv.stat().st_size > 0:
            job.set_stage(f"Reusing cached sequences for {family} ({assembly}, {source})")
            df = pd.read_csv(cached_csv)
            print(f"CACHE HIT [{key}]: {len(df):,} loci reused — no re-download")
            return df, True

        job.set_stage(f"Fetching {family} ({assembly}, {source}) via te_prep.py")
        tmp_dir = CACHE_ROOT / f"{key}.partial"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            clustered = _run_te_prep(job, family, assembly, source, max_loci, tmp_dir)
            df = pd.read_csv(clustered)
            # Commit to the cache atomically only on success.
            shutil.rmtree(cache_dir, ignore_errors=True)
            os.replace(tmp_dir, cache_dir)
            (cache_dir / "meta.json").write_text(json.dumps({
                "family": family, "assembly": assembly, "source": source,
                "max_loci": int(max_loci or 0), "n_loci": int(len(df)),
                "fetched": datetime.utcnow().isoformat() + "Z", "key": key,
            }, indent=2))
            print(f"CACHED [{key}]: {len(df):,} loci with sequences")
            return df, False
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _do_cluster(job: Job, df: pd.DataFrame, family: str, kmer, min_cluster_size,
                n_neighbors, min_dist, min_samples, compute_tsne) -> dict:
    job.set_stage(f"Clustering {len(df):,} sequences (k={kmer})")
    df_out, labels = te_clustering.clustering_analysis(
        df,
        kmer=kmer,
        min_cluster_size=min_cluster_size,
        out_dir=job.dir,
        family_name=family,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        min_samples=min_samples,
        compute_tsne=compute_tsne,
    )
    # Full frame (Seq + Cluster + coords + expression) → consumed by Guides tab.
    clustered_csv = job.dir / "clustered_data.csv"
    df_out.to_csv(clustered_csv, index=False)

    n_clusters = int(len(set(int(x) for x in labels)) - (1 if -1 in set(labels) else 0))
    summary = []
    summ_path = job.dir / "cluster_summary.csv"
    if summ_path.exists():
        summary = pd.read_csv(summ_path).to_dict(orient="records")

    files = {"clustered_csv": "clustered_data.csv"}
    for fname, key in [("clustering_visualization.html", "viz_html"),
                       ("clustering_visualization_expr.html", "viz_html_expr"),
                       ("clustering_coordinates.csv", "coords_csv"),
                       ("cluster_summary.csv", "summary_csv")]:
        if (job.dir / fname).exists():
            files[key] = fname

    return {
        "family": family,
        "n_loci": int(len(df_out)),
        "n_clusters": n_clusters,
        "has_expression": bool(_detect_expr_cols(df_out)),
        "cluster_summary": summary,
        "files": files,
    }


def _detect_expr_cols(df: pd.DataFrame) -> list[str]:
    return grna._auto_expr_cols(df)


# ── Guides ───────────────────────────────────────────────────────────────────

@app.post("/api/guides")
async def guides(
    mode: str = Form(...),                       # "upload" | "job"
    file: UploadFile | None = File(None),
    mapping: str = Form("{}"),
    source_job_id: str = Form(""),
    family: str = Form("FAMILY"),
    cas: str = Form("SpCas9"),
    grna_len: int = Form(20),
    n_greedy: int = Form(5),
    n_workers: int = Form(2),
    global_mm_index: str = Form("auto"),
):
    if cas not in grna._PAM_TYPES:
        raise HTTPException(400, f"cas must be one of {list(grna._PAM_TYPES)}")

    job = Job("guides")
    JOBS[job.id] = job

    if mode == "job":
        src = JOBS.get(source_job_id)
        if src is None:
            raise HTTPException(404, f"source_job_id '{source_job_id}' not found.")
        src_csv = src.dir / "clustered_data.csv"
        if not src_csv.exists():
            raise HTTPException(400, "Source job has no clustered_data.csv yet.")

        def work(j: Job):
            df = pd.read_csv(src_csv)
            print(f"Loaded clustered result from job {source_job_id}: {len(df):,} rows")
            return _do_guides(j, df, family, cas, grna_len, n_greedy, n_workers,
                              global_mm_index)

        _run_in_thread(job, work)

    elif mode == "upload":
        if file is None:
            raise HTTPException(400, "mode='upload' requires a file.")
        upload_path = _save_upload(file, job.dir)
        mp = json.loads(mapping or "{}")

        def work(j: Job):
            df_raw = _read_table(upload_path)
            print(f"Loaded {len(df_raw):,} rows, {len(df_raw.columns)} columns")
            df = _apply_mapping(df_raw, mp)
            return _do_guides(j, df, family, cas, grna_len, n_greedy, n_workers,
                              global_mm_index)

        _run_in_thread(job, work)
    else:
        raise HTTPException(400, "mode must be 'upload' or 'job'.")

    return {"job_id": job.id}


def _do_guides(job: Job, df: pd.DataFrame, family: str, cas: str, grna_len: int,
               n_greedy: int, n_workers: int, global_mm_index: str) -> dict:
    reports = job.dir

    # Cluster column is required by the coverage figure; default to a single
    # cluster when the input was not clustered.
    if "Cluster" not in df.columns and "cluster" not in df.columns:
        df["Cluster"] = 0

    # Expression → total weight. If none provided, weight every locus equally
    # (sequence-coverage == expression-coverage) and switch greedy to sequence
    # mode so the ranking is honest about what it optimises.
    expr_cols = grna._auto_expr_cols(df)
    if expr_cols:
        df["_total_expr"] = df[expr_cols].sum(axis=1)
        greedy_mode = "expression"
        print(f"Expression columns ({len(expr_cols)}): {expr_cols}")
    else:
        df["_total_expr"] = 1.0
        greedy_mode = "sequence"
        print("No expression columns detected → uniform weights, greedy=sequence")

    # Optional global 20-mer index (mismatch coverage) — same guard as the CLI.
    want_index = (global_mm_index == "on" or
                  (global_mm_index == "auto" and
                   len(df) <= grna.GLOBAL_MM_INDEX_MAX_SEQS))
    kmer_index = None
    if want_index:
        job.set_stage("Building 20-mer mismatch index")
        kmer_index = grna.build_kmer_index_20(df, grna_len=grna_len)

    job.set_stage(f"Scoring PAM-adjacent candidates ({cas}, len={grna_len})")
    df_cands = grna.score_candidates(
        df, grna_len=grna_len, cas=cas, kmer_index=kmer_index,
        expr_col="_total_expr", n_workers=max(1, n_workers),
    )
    if df_cands.empty:
        raise RuntimeError("No PAM-adjacent gRNA candidates found — check sequences / Cas type.")

    n_pam_sites = int(df_cands["exact_cov"].sum())
    cand_csv = df_cands.drop(columns=["rows_exact"])
    if not expr_cols:  # no expression → don't carry meaningless expression columns
        cand_csv = cand_csv.drop(columns=[c for c in
                    ("exact_expr_cov", "mm1_expr_cov", "pct_expr_cov")
                    if c in cand_csv.columns])
    cand_csv.to_csv(reports / "grna_candidates.csv", index=False)

    best_seq = df_cands.loc[df_cands["pct_seq_cov"].idxmax()]
    best_expr = df_cands.loc[df_cands["pct_expr_cov"].idxmax()]

    job.set_stage(f"Greedy {n_greedy}-guide set-cover ({greedy_mode})")
    greedy_df = grna.greedy_grna_set(df_cands, df, n_guides=n_greedy, mode=greedy_mode)
    greedy_df.to_csv(reports / "grna_greedy_set.csv", index=False)

    greedy5_loci = float(greedy_df["pct_loci"].max()) if len(greedy_df) else 0.0
    greedy5_expr = float(greedy_df["pct_expr"].max()) if len(greedy_df) else 0.0

    has_expression = bool(expr_cols)
    job.set_stage("Rendering figures")
    corr, _mm_df = grna.fig_grna_coverage(
        df_cands, df, top_grna_seq=best_seq["grna"], top_grna_expr=best_expr["grna"],
        family=family, out_path=reports / "fig_grna_coverage.png",
        grna_len=grna_len, has_expression=has_expression,
    )
    grna.fig_grna_positional(
        df, grna_len=grna_len, cas=cas, family=family,
        out_path=reports / "fig_grna_positional.png",
    )
    grna.fig_grna_candidates(df_cands, family, out_path=reports / "fig_grna_candidates.png",
                             has_expression=has_expression)

    # Reagent coverage projected onto each available embedding (UMAP / t-SNE /
    # PCA), one figure each so the UI can switch between them. Coverage numbers
    # are identical across projections (exact match), so keep the first as emb_cov.
    emb_cov = None
    embedding_figs = {}
    for emb in ("umap", "tsne", "pca"):
        out = reports / f"fig_grna_{emb}_coverage.png"
        cov = grna.fig_grna_embedding_coverage(
            df, df_cands, greedy_df, family, out_path=out,
            covered_csv=(reports / "grna_covered_loci.csv" if emb_cov is None else None),
            embeddings=(emb,),
        )
        if cov is not None and out.exists():
            embedding_figs[emb] = out.name
            if emb_cov is None:
                emb_cov = cov

    results = {
        "family": family, "cas": cas, "grna_len": grna_len,
        "n_loci": int(len(df)), "n_candidates": int(len(df_cands)),
        "n_pam_sites": n_pam_sites,
        "greedy_mode": greedy_mode,
        "has_expression": bool(expr_cols),
        "top_seq_cov_pct": float(best_seq["pct_seq_cov"]),
        "top_expr_cov_pct": float(best_expr["pct_expr_cov"]),
        "best_seq_grna": str(best_seq["grna"]),
        "best_expr_grna": str(best_expr["grna"]),
        "seq_expr_corr": float(corr),
        "greedy5_loci_pct": greedy5_loci,
        "greedy5_expr_pct": greedy5_expr,
    }
    grna.write_grna_values(results, reports)  # writes .tex + .txt (faithful to CLI)

    # ── Per-cluster coverage for the tables ───────────────────────────────────
    # For each guide, the % of every cluster's loci it hits by exact match — the
    # same quantity as the heatmap in fig_grna_coverage panel (d), but for every
    # displayed guide. rows_exact are positional indices aligned with df.index.
    cluster_col = "Cluster" if "Cluster" in df.columns else "cluster"
    cluster_ids = sorted(int(c) for c in df[cluster_col].unique() if c >= 0)
    cluster_rows = {c: set(df.index[df[cluster_col] == c].tolist()) for c in cluster_ids}

    def _cluster_pcts(rows_exact):
        rows = set(rows_exact)
        return {str(c): (round(100 * len(rows & cluster_rows[c]) / len(cluster_rows[c]), 1)
                         if cluster_rows[c] else 0.0) for c in cluster_ids}

    # Structured tables for the UI (with per-cluster coverage attached).
    top_cols = ["grna", "gc", "on_target_score", "exact_cov", "exact_expr_cov",
                "pct_seq_cov", "pct_expr_cov"]
    top_candidates = []
    for _, row in df_cands.head(25).iterrows():
        rec = {c: row[c] for c in top_cols}
        rec["grna_rc"] = grna._rc(str(row["grna"]))   # reverse-complement (bottom strand)
        rec["clusters"] = _cluster_pcts(row["rows_exact"])
        top_candidates.append(rec)

    greedy_rows = []
    for _, row in greedy_df.iterrows():
        rec = row.to_dict()
        rec["grna_rc"] = grna._rc(str(row["grna"]))
        rec["clusters"] = _cluster_pcts(grna._grna_rows(df_cands, row["grna"]))
        greedy_rows.append(rec)

    figures = {}
    for fname, key in [("fig_grna_coverage.png", "coverage"),
                       ("fig_grna_positional.png", "positional"),
                       ("fig_grna_candidates.png", "candidates")]:
        if (reports / fname).exists():
            figures[key] = fname
    if embedding_figs:
        figures["embedding"] = embedding_figs   # {emb: filename} — UI switches between them

    files = {"candidates_csv": "grna_candidates.csv",
             "greedy_csv": "grna_greedy_set.csv"}
    if (reports / "grna_covered_loci.csv").exists():
        files["covered_csv"] = "grna_covered_loci.csv"

    return {
        "results": results,
        "cluster_ids": cluster_ids,
        "greedy": greedy_rows,
        "top_candidates": top_candidates,
        "embedding_coverage": emb_cov,
        "figures": figures,
        "files": files,
    }


# ── Status + files ───────────────────────────────────────────────────────────

@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job.to_public()


@app.get("/api/jobs/{job_id}/log")
def job_log(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return JSONResponse({"log": job.log})


_MEDIA = {".png": "image/png", ".html": "text/html", ".csv": "text/csv",
          ".txt": "text/plain", ".tex": "text/plain", ".pdf": "application/pdf",
          ".svg": "image/svg+xml"}
# Types the browser should render in place (iframe/img), not download.
_INLINE_TYPES = {".png", ".html", ".svg", ".jpg", ".jpeg", ".gif"}


@app.get("/api/files/{job_id}/{name}")
def job_file(job_id: str, name: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    # prevent path traversal
    safe = Path(name).name
    path = job.dir / safe
    if not path.exists():
        raise HTTPException(404, f"file '{safe}' not found")
    suffix = path.suffix.lower()
    media = _MEDIA.get(suffix, "application/octet-stream")
    # Set the disposition ourselves: 'inline' lets figures/plots render inside an
    # <iframe>/<img>; 'attachment' makes CSV/text downloads keep their filename
    # (the anchor's download= attribute is ignored cross-origin, so this matters).
    disposition = "inline" if suffix in _INLINE_TYPES else "attachment"
    resp = FileResponse(path, media_type=media)
    resp.headers["Content-Disposition"] = f'{disposition}; filename="{safe}"'
    return resp


# ── Serve the static front-end from the same origin (single-container deploy) ─
# webapp/api/app.py → webapp/frontend is one level up + /frontend. When present
# (Docker image copies it in, or a full checkout), the UI is served at "/" and
# the front-end talks to /api/* on the same origin — no API-URL config needed.
# When absent (API-only deploy) we fall back to a JSON service descriptor.
_FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

if _FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
else:
    @app.get("/")
    def root():
        return {"service": "TERRA Web API", "docs": "/docs",
                "endpoints": ["/api/health", "/api/inspect", "/api/cluster",
                              "/api/guides", "/api/jobs/{id}", "/api/files/{id}/{name}"]}
