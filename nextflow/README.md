# GAMECA — Nextflow orchestration

This directory turns the GAMECA TE-analysis pipeline into a **Nextflow DSL2**
workflow. It wraps the existing engine (`query.py`) and the ~15 post-alignment analyses
`run_*.py` modules so the pipeline can be used two ways:

1. **Internally** — every pipeline stage (genome load, data load, sequences,
   stats, clustering, dashboard, alignment, motif/TFBS/GO, primers, plus the
   15 post-alignment modules) is its own Nextflow process with real
   parallelism, `-resume`, and per-process LSF resource requests — replacing
   `query.py`'s sequential in-process loop end to end, not just the
   post-alignment step.
2. **As a subworkflow inside other pipelines** — `include { GAMECA }` (nf-core
   meta convention) and drop it into any DSL2 pipeline.

Nothing in the Python code was removed: `query.py` still runs the whole
pipeline standalone in one process too (`python query.py --family ...` with no
`--nextflow` flag). The Nextflow layer calls the same engine one stage at a
time via `query.py --run-stage <stage>`.

## Layout

```
nextflow/
  main.nf                       full pipeline entrypoint (samplesheet → report)
  post_alignment_analyses.nf    post-alignment-only entrypoint (what query.py's
                                 narrower --post-alignment-analyses-nextflow calls)
  nextflow.config               params, profiles (lsf/singularity/docker/test)
  conf/
    base.config                 process-label → resources (+ resourceLimits ceilings)
    modules.config              per-process publishDir + pass-through ext.args
    test.config                 tiny stub smoke-test params
  modules/local/
    genome.nf      GAMECA_GENOME       Stage 1  (query.py --run-stage genome)
    data.nf        GAMECA_DATA         Stage 2  (--run-stage data)
    sequences.nf   GAMECA_SEQUENCES    Stage 3  (--run-stage sequences)
    stats.nf       GAMECA_STATS        Stage 4  (--run-stage stats)
    clustering.nf  GAMECA_CLUSTERING   Stage 5  (--run-stage clustering)
    dashboard.nf   GAMECA_DASHBOARD    Stage 6  (--run-stage dashboard)
    alignment.nf   GAMECA_ALIGNMENT    Stage 7  (--run-stage alignment)
    motif.nf       GAMECA_MOTIF        Stage 9  (--run-stage motif)
    primers.nf     GAMECA_PRIMERS      Stage 10 (--run-stage primers)
    merge.nf       GAMECA_MERGE        folds the 4 parallel post-clustering branches back together
    standout.nf    GAMECA_STANDOUT     one post-alignment analyses run_*.py module per task
    report.nf      GAMECA_REPORT       merge module reports + generate_report.py
  subworkflows/local/
    standout.nf   STANDOUT   registry + scatter/gather of post-alignment analyses modules
    gameca.nf     GAMECA     GENOME→DATA→SEQUENCES→STATS→CLUSTERING→{DASHBOARD,ALIGNMENT,MOTIF,PRIMERS}→MERGE→STANDOUT→REPORT
  assets/
    samplesheet.csv            example input
    schema_input.json          samplesheet schema
```

`GAMECA_DASHBOARD`/`GAMECA_ALIGNMENT`/`GAMECA_MOTIF`/`GAMECA_PRIMERS` only need
the clustered CSV, so they fan out in parallel off `GAMECA_CLUSTERING`'s
output; `GAMECA_MERGE` folds those four branches' new files back into one
family folder before `STANDOUT` (which needs the alignment consensus FASTA
alongside the clustered CSV). `GAMECA_GENOME` and `GAMECA_SEQUENCES`/
`GAMECA_PRIMERS` reload the on-disk genome pickle `GAMECA_GENOME` wrote
instead of re-parsing the FASTA each time (see `te_genome.GenomeCache`).

`subworkflows/local/standout.nf::buildRegistry()` is the single source of truth
for the post-alignment analyses module list and mirrors `query.py`'s `_MODULES`.

## Run the full pipeline

```bash
# samplesheet: columns family,assembly,input (input optional)
nextflow run nextflow/main.nf --input nextflow/assets/samplesheet.csv \
    --outdir results -profile lsf,singularity

# single family
nextflow run nextflow/main.nf --family HERVK9 --assembly hg38 \
    --outdir results -profile lsf,singularity

# smoke test — validates config + full DAG via process stubs (no genome/net/deps).
# The test profile sets stubRun=true, so -stub-run is not needed.
nextflow run nextflow/main.nf -profile test
```

Each family fans out to 9 core-stage tasks + 15 standout tasks + 1 report, all
with their own resource request. Add `-resume` to reuse completed tasks.

Key params (see `nextflow.config` for the full list):

| param | default | meaning |
|-------|---------|---------|
| `--input` | – | samplesheet CSV (`family,assembly,input`) |
| `--family` / `--assembly` | – / `hg38` | single-run shortcut |
| `--genome_fasta` | – | local reference FASTA on the exec host |
| `--report_pdf` | `false` | also render the LaTeX/PDF report |
| `--skip_standout` | `false` | core stages only (Stages 1-10), skip the 15 post-alignment modules |
| `--core_args` | – | extra flags passed straight to `query.py` |
| `--container_sif` | – | path to `gameca.sif` (with `-profile singularity`) |
| `--max_cpus`/`--max_memory`/`--max_time` | 16 / 128.GB / 48.h | resource ceilings |

post-alignment analyses knobs (`--clock_divisor`, `--grna_cas`, `--tail_bp`, `--ortholog_species`,
…) mirror the matching `query.py` flags and default to each module's own default.

## Containers

`-profile singularity` (or `docker`) runs every process through the GAMECA image
— the same one `CONTAINER_SIF`/`Dockerfile` produce — so the glibc/import issues
on old HPC nodes disappear. Point `--container_sif /path/gameca.sif` at your
prebuilt image, or let it pull `docker://ghcr.io/anmol-dash/gameca:latest`
(override with `--container_image`).
ColabFold fold prediction still runs from its separate image (as before).

## Use inside another pipeline

```groovy
include { GAMECA } from './nextflow/subworkflows/local/gameca.nf'

workflow {
    // tuple( meta, input_csv )  — meta = [ id: family, assembly: 'hg38' ]
    ch = Channel.of( tuple([id:'HERVK9', assembly:'hg38'], []) )
    GAMECA( ch )
    GAMECA.out.reports.view()
}
```

Or pull in just the pieces: any single stage module under `modules/local/`, or
the `STANDOUT` subworkflow (post-alignment analyses against an existing family
folder).

## From query.py (internal use)

`query.py` can hand the **entire** pipeline to Nextflow instead of running it
in-process:

```bash
python query.py --family HERVK9 --assembly hg38 --genome /path/hg38.fa \
    --nextflow --nextflow-profile lsf,singularity
```

This runs `nextflow/main.nf`, which invokes `query.py --run-stage <stage>` once
per pipeline stage per family. If `nextflow` isn't on `PATH` (or the run
fails), `query.py` transparently falls back to running the full pipeline
in-process, one stage after another in the same Python process.

A narrower flag still exists for handing off just the post-alignment step
after Stages 1-10 have already run in-process:

```bash
python query.py --family HERVK9 --assembly hg38 \
    --post-alignment-analyses-nextflow --nextflow-profile lsf,singularity
```

This invokes `nextflow/post_alignment_analyses.nf` directly on the finished
family folder so the 15 modules run concurrently, without touching Stages
1-10 at all.
