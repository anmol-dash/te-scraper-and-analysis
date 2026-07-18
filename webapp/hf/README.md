---
title: TERRA
emoji: 🧬
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: TE clustering + gRNA guide design pipeline
---

# TERRA

Transposable-element clustering and CRISPR gRNA guide design, served as a single
Docker Space: a FastAPI backend (the repository's own pipeline, unchanged) plus a
static front-end on the same origin.

This README's YAML front-matter is what Hugging Face reads to configure the Space
(`sdk: docker`, `app_port: 7860`). The container is built from the `Dockerfile`
at the root of the Space repo — see `webapp/api/Dockerfile` in the source repo.
