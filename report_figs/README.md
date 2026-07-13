# GAMECA report figures

This folder holds the schematic figures for `gameca_report.tex` and is the
documented home of the figure pipeline. Every figure the manuscript references is
built by `make_report_figures.py` — schematics directly, data figures by invoking
the real analysis scripts. **No figure is ever fabricated:** a data figure that
lacks real inputs is reported `MISSING` in the manifest, never invented
(see the no-fabrication policy below).

## Prototype TE: IAPLTR1_Mm (mm10)

Per the review, the paper follows **one named TE as a prototype throughout**:
`IAPLTR1_Mm`, an LTR/endogenous-retrovirus family. It is the LTR family
already carried in the cross-family results, so its real data is reused rather than
regenerated. The three-family LINE/SINE/LTR set (L1Md_T, B1_Mus2, IAPLTR1_Mm)
is intended as **supplementary** breadth; the main text should walk the reader
through the single prototype from database annotation to ordered reagents. The
prototype's measured Stage-11 panels are written to `reports8/stage11_iapltr1/` by
`run_stage11_all.py`.

## Deliverables come first

The reagents are the payoff and should appear early and prominently:
**family/locus-specific primers** and **allele-aware CRISPRa/i guides**. The
CRISPR module is what moves the tool from "useful" to "necessary" for the field —
`fig_prototype_journey` and `fig_crispr_deliverable` exist to make that case.

## Brand / house style (keep every figure consistent)

All figures share one look. Match these when producing data figures too:

| Role | Hex |
|------|-----|
| Input / neutral | `#555555` |
| Core stage (query.py) | `#2f6f8f` |
| Stage-11 standout module | `#3f8f5f` |
| Gather / report | `#8f6f2f` |
| Execution environment | `#8f3f6f` |
| **Deliverables (primers + CRISPR)** | `#b0413e` |

Font: DejaVu Sans. Rounded boxes, thin arrows, PNG at 150 dpi + vector PDF.
`apply_house_style()` sets these as matplotlib rcParams for the schematics.

## Figures in this folder (schematics — always generated)

- **fig_pipeline_flowchart** — end-to-end flow: input to core Stages 1-7 to
  Stage 11 to report, inside the Nextflow/Singularity execution band.
  *Section: Design/architecture.*
- **fig_architecture** — three-layer software architecture (desktop/CLI to HPC
  client to cluster). *Section: Design and architecture (Figure 1).*
- **fig_stage11_analyses** — grid of the 16 Stage-11 copy-resolved modules by
  theme. *Section: Copy-resolved analysis modules.*
- **fig_nextflow_dag** — the Nextflow DSL2 scatter/gather DAG. *Section: HPC.*
- **fig_prototype_journey** — NEW. The IAPLTR1_Mm storyline: RMSK/Dfam
  loci to clustering to expression filter (meaningful vs inert) to consensus+motif
  to the highlighted **primers + CRISPRa/i deliverables**. Schematic spine of the
  results narrative. *Should open the demonstration/results section.*
- **fig_locus_filtering** — NEW. Why GAMECA resolves copies instead of lumping
  them: lumping averages signal across meaningful, low-function and inert loci,
  whereas cluster+expression resolution separates them; also expressed-consensus
  vs common-consensus. Concept diagram (no numbers). *Frame this as a conceptual
  contribution.*
- **fig_crispr_deliverable** — NEW. Concept diagram of allele-aware CRISPRa/i
  guide design for repetitive targets (seed-anchored, mismatch-tolerant coverage
  vs off-target; Pareto trade-off). The measured version is
  `fig_grna_offtarget_pareto`. *Section: deliverables / CRISPR.*

## Prototype data panels (generated on the cluster into `reports8/stage11_iapltr1/`)

Produced by `python make_report_figures.py --data --input <IAPLTR1_Mm loci
csv> --iapltr1-expr <expr csv> ...`, which drives `run_stage11_all.py`. These are
the *real* per-family outputs (the kind sent to collaborators):

- **fig_phylo_tree / fig_phylo_divergence / fig_phylo_master** — subfamily
  phylogeny, per-copy molecular-clock age, and the combined age/intactness master
  panel.
- **fig_grna_offtarget_pareto / fig_grna_offtarget_top** — measured guide
  coverage-vs-off-target frontier and the top guides.
- **fig_transduction_groups** — 3' transduction lineage groups.
- **fig_antisense_motifs / fig_antisense_expr** — bidirectional-promoter core
  motifs and antisense expression.

## C33 coverage checklist — the collaborator-style outputs

AJM (comment C33) asked for the real procedural graphs sent to Katie/Claire/Diego —
clusters, expression, consensus, alignment, motif, primers, guide off-targets — for
the prototype. Every one is produced by a genuine generator and registered in the
manifest (`MADE` once real inputs run; `MISSING` otherwise — never fabricated):

| C33 item | Figure(s) | Generator |
|----------|-----------|-----------|
| Clusters (UMAP/HDBSCAN) | `reports8/fig_ltr_results`, `reports8/fig_line_sine_ltr_clustering` | `run_line_sine_ltr_analysis.py` |
| Expression (per-cluster DE) | `reports8/fig_de_iapltr1mm`, `reports8/expression_plots/stage_profile`, `.../chromosomal_heatmap` | cross-family + `te_expression.py` |
| New consensus sequences | `.../fig_cluster_consensus_distance`, `.../fig_global_consensus_distance`, `.../fig_phylo_tree` | `plot_consensus_distance.py`, `run_phylo_analysis.py` |
| Motif analysis (turnover) | `.../fig_motif_gains_bar`, `.../fig_motif_gains_heatmap` | `run_motif_gain.py` |
| Guide off-targets (CRISPR) | `.../fig_grna_offtarget_pareto`, `.../fig_grna_offtarget_top` | `run_grna_offtarget.py` |
| LTR structure (prototype is an LTR) | `.../fig_ltr_struct`, `.../fig_subfamily_tree` | `run_ltr_struct.py`, `run_subfamily.py` |
| Regulation (antisense, CTCF) | `.../fig_antisense_motifs`, `.../fig_antisense_expr`, `.../fig_ctcf_overlap` | `run_antisense_promoter.py`, `run_ctcf_tad.py` |
| 3' transduction | `.../fig_transduction_groups` | `run_transduction.py` |

Three C33 items are deliberately **not** matplotlib figures — flagged here so they are
not silently assumed:

- **Alignments** — the multiple alignment is cleaned/visualised by **CIAlign** (its own
  PNG in the alignment output dir); the alignment-derived *figure* in the report is the
  phylogeny/consensus panel (`fig_phylo_tree`).
- **JASPAR motif *enrichment*** (`overall_top_motifs.png`, `enrichment_heatmap.png`) is a
  **core `query.py` Stage-4** output, not produced by the two generators driven here; the
  registered motif figure is the evolutionary motif-**turnover** (`fig_motif_gains_*`).
  Include the enrichment panel from a core run or the collaborator report if the paper
  needs it.
- **Primers** are a CSV table (`06_primers/selected_primers_summary.csv`); the primer
  *figure* is `expression_plots/primer_expression` (primers × expression).

## Regenerate

```bash
# schematics + README only (no cluster data needed)
python make_report_figures.py

# everything, prototype-first (real inputs; nothing fabricated)
python make_report_figures.py --data \
    --input IAPLTR1_Mm_loci.csv --iapltr1-expr IAPLTR1_Mm_ultracombo.csv \
    --genome-fa ~/te_analysis/mm10.fa --build mm10
```

The run prints a **manifest** marking each figure `MADE`, `MISSING` (needs real
inputs — not fabricated), or `MANUAL` (no automated generator).

## No-fabrication policy

Data figures require genuine loci/expression/genome inputs. When an input or a
real generator is missing, the figure is reported `MISSING` with the reason and
left unbuilt. Numbers, plots and measured panels are never invented.
