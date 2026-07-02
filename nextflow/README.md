# GAMECA — Nextflow orchestration

This directory turns the GAMECA TE-analysis pipeline into a **Nextflow DSL2**
workflow. It wraps the existing engine (`query.py`) and the ~15 post-alignment analyses
`run_*.py` modules so the pipeline can be used two ways:

1. **Internally** — real per-module parallelism, `-resume`, and per-process LSF
   resource requests, replacing `query.py`'s sequential subprocess loop for
   post-alignment analyses.
2. **As a subworkflow inside other pipelines** — `include { GAMECA }` (nf-core
   meta convention) and drop it into any DSL2 pipeline.

Nothing in the Python code was removed: `query.py` still runs standalone. The
Nextflow layer just calls it.

## Layout

```
nextflow/
  main.nf                       full pipeline entrypoint (samplesheet → report)
  post_alignment_analyses.nf    post-alignment-only entrypoint (what query.py calls)
  nextflow.config               params, profiles (lsf/singularity/docker/test)
  conf/
    base.config                 process-label → resources (+ resourceLimits ceilings)
    modules.config              per-process publishDir + pass-through ext.args
    test.config                 tiny stub smoke-test params
  modules/local/
    core.nf       GAMECA_CORE       query.py Stages 1-10 (--skip-standout)
    standout.nf   GAMECA_STANDOUT   one post-alignment analyses run_*.py module per task
    report.nf     GAMECA_REPORT     merge module reports + generate_report.py
  subworkflows/local/
    standout.nf   STANDOUT   registry + scatter/gather of post-alignment analyses modules
    gameca.nf     GAMECA     CORE → STANDOUT (the composable end-to-end unit)
  assets/
    samplesheet.csv            example input
    schema_input.json          samplesheet schema
```

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

Each family fans out to 1 core + 15 standout tasks + 1 report, all with their
own resource request. Add `-resume` to reuse completed tasks.

Key params (see `nextflow.config` for the full list):

| param | default | meaning |
|-------|---------|---------|
| `--input` | – | samplesheet CSV (`family,assembly,input`) |
| `--family` / `--assembly` | – / `hg38` | single-run shortcut |
| `--genome_fasta` | – | local reference FASTA on the exec host |
| `--report_pdf` | `false` | also render the LaTeX/PDF report |
| `--skip_standout` | `false` | core stages only (Stages 1-10) |
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

Or pull in just the pieces: `GAMECA_CORE` (Stages 1-10) or the `STANDOUT`
subworkflow (post-alignment analyses against an existing family folder).

## From query.py (internal use)

`query.py` can hand post-alignment analyses to Nextflow instead of running the modules
sequentially:

```bash
python query.py --family HERVK9 --assembly hg38 \
    --post-alignment-analyses-nextflow --nextflow-profile lsf,singularity
```

`query.py` runs Stages 1-10 in-process, then invokes `nextflow/post_alignment_analyses.nf` on
the finished family folder so the 15 modules run concurrently. If `nextflow`
isn't on `PATH`, it transparently falls back to the in-process loop.
