# TERRA web app

A two-tab website over the existing TERRA pipeline:

- **Cluster** — upload a CSV/XLSX (identify the Sequence / Name / Start / Stop /
  expression columns after upload) *or* enter a family name + assembly; runs
  `te_clustering.clustering_analysis()` (k-mer → SVD → UMAP/t-SNE → HDBSCAN).
- **Guides** — takes the clustered sequences and runs `run_grna_analysis.py`
  (PAM-aware candidates, on-target scoring, expression-weighted greedy set-cover)
  and shows figures explaining *why* each guide is optimal.

## Two pieces

| Piece | Where it lives | Where it runs |
|-------|----------------|---------------|
| `frontend/` | static HTML/CSS/JS, no build step | **GitHub Pages** |
| `api/` | FastAPI wrapper importing the repo's pipeline modules | a Python host you choose (Render / Fly / HF Spaces) |

GitHub Pages can only serve static files, so the heavy pipeline (UMAP/HDBSCAN,
UCSC fetch) can't run there — it runs in `api/`. The two are wired together at
runtime: open the site, click **⚙︎**, and paste your API URL.

## Quick start (local)

```bash
# 1. backend
cd webapp/api
pip install -r requirements.txt
uvicorn app:app --port 8000 --app-dir .

# 2. frontend (any static server; here Python's)
cd ../frontend
python3 -m http.server 5173
# open http://localhost:5173  → it auto-targets http://localhost:8000
```

## Deploy

- **Frontend → GitHub Pages:** in the repo settings set *Pages → Source →
  GitHub Actions*. Pushing to `main` runs `.github/workflows/pages.yml`, which
  publishes `webapp/frontend`. Site URL: `https://<user>.github.io/<repo>/`.
- **Backend:** see [`api/README.md`](api/README.md) (Docker + Render/Fly/HF).
  Then open the Pages site, click **⚙︎**, and enter the backend URL. Set
  `TERRA_CORS_ORIGINS` on the backend to your Pages origin to lock down CORS.
