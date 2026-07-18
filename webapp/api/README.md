# TERRA Web API

A thin FastAPI layer over the **existing** TERRA pipeline. It does not
reimplement any science — it imports and calls the repository's own functions:

| Route | Calls (unchanged) |
|-------|-------------------|
| `POST /api/cluster` (upload) | `te_clustering.clustering_analysis()` |
| `POST /api/cluster` (family) | `te_prep.py` CLI → then `clustering_analysis()` |
| `POST /api/guides` | `run_grna_analysis.{score_candidates, greedy_grna_set, fig_*}` |

The matplotlib/plotly writers in those modules choose their format from the
output file extension, so the API hands them `.png`/`.html` paths and serves the
resulting web-ready artefacts back to the front-end. **Nothing in the pipeline
is modified.**

## Why a separate backend?

GitHub Pages only serves static files. The clustering step uses
UMAP + HDBSCAN + scikit-learn and the family/assembly path fetches from UCSC —
none of which can run in a browser. So the static site (on GitHub Pages) talks
to this API, which runs the real pipeline. Point the front-end's **API URL**
setting at wherever you deploy this.

## Run locally

```bash
cd webapp/api
pip install -r requirements.txt          # (repo's scientific stack must also be importable)
uvicorn app:app --host 0.0.0.0 --port 8000 --app-dir .
# open http://localhost:8000/docs
```

The API imports `te_clustering.py`, `run_grna_analysis.py` and `te_prep.py` from
the repository root (resolved automatically as two levels up from `app.py`), so
run it from a checkout of the whole repo.

## Deploy (Docker)

The build context must be the **repository root** (the image copies the pipeline
modules):

```bash
docker build -f webapp/api/Dockerfile -t terra-api .
docker run -p 8000:8000 terra-api
```

### Render.com
- New **Web Service** → build from this repo.
- Environment: **Docker**; Dockerfile path `webapp/api/Dockerfile`; **Docker build context `.`** (repo root).
- Render injects `$PORT` (the Dockerfile honours it).

### Fly.io
```bash
fly launch --dockerfile webapp/api/Dockerfile   # accept the generated fly.toml, internal_port 8000
fly deploy
```

### Hugging Face Spaces (Docker SDK)
Point the Space at this repo with `app_port: 8000` in the Space README front-matter.

## Configuration (env vars)

| Var | Default | Purpose |
|-----|---------|---------|
| `TERRA_CORS_ORIGINS` | `*` | Comma-separated allowed origins. Set to your Pages origin, e.g. `https://anmol-dash.github.io`, to lock it down. |
| `TERRA_JOBS_DIR` | system temp | Where per-job working dirs (figures, CSVs) are written. |
| `TERRA_CACHE_DIR` | `<jobs>/_seqcache` | Where downloaded family sequences are cached (see below). |
| `TERRA_MAX_UPLOAD_MB` | `50` | Upload size cap. |
| `PORT` | `8000` | Listen port (set by most PaaS). |

## Sequence caching (family + assembly)

Downloaded sequences are cached on disk under `TERRA_CACHE_DIR`, keyed by
`(family, assembly, source, max_loci)` — the inputs that determine *what* is
fetched. Clustering parameters (`kmer`, `min_cluster_size`, UMAP knobs) are **not**
in the key, so **re-clustering the same family with different parameters reuses
the cached download and never re-hits UCSC/Dfam.** The UI shows a "sequences
reused from cache" note when this happens. Changing the family, assembly, source,
or max-loci fetches fresh. The cache commit is atomic (a `.partial` dir is
`os.replace`d into place only on success), so an interrupted fetch never poisons
the cache. Delete a `TERRA_CACHE_DIR/<key>/` folder to force a re-download.

## Notes for hosted use

- The **family + assembly** path fetches from UCSC (`api.genome.ucsc.edu`) or
  Dfam, which are rate-limited. Use the **Max loci** field in the UI to keep
  hosted runs quick, and prefer `source=dfam` to avoid the one-time ~150 MB
  `rmsk.txt.gz` download on ephemeral hosts. Downloads are cached (above), so the
  rate-limit cost is paid once per family/assembly/source/max-loci.
- Jobs are held in memory + a temp dir; a container restart clears them. Fine
  for interactive use, not a durable store.
