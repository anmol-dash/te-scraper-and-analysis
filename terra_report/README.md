# TERRA report — rewritten draft

Self-contained rewrite of the manuscript against Andrew's review of `GAMECA_Report_AJM.docx`.
See **`AJM_COMMENTS.md`** for all 22 comments and how each one is addressed.

```
terra_report/
  terra_report.tex     the rewritten manuscript
  terra_report.pdf     compiled output (15 pages)
  references.bib       bibliography (+4 new entries for AJM's C101)
  figs/                the 7 real schematic figures
  AJM_COMMENTS.md      the compiled comment list + status
  README.md            this file
```

Build with:

```bash
pdflatex terra_report && bibtex terra_report && pdflatex terra_report && pdflatex terra_report
```

## What changed

- **Title/name:** renamed **TERRA** with **CRISPR in the title** (C0, C157). The name is
  provisional and marked with a red TODO; Andrew asked to crowdsource it.
- **Results, not methods** (his main point): §2 is now a single-family walkthrough of the
  **L1Md_T** prototype from RepeatMasker annotation to ordered reagents. Implementation detail
  moved *after* the results into §5.
- **Deliverables first** (C159, C131): the abstract and intro lead with primers and allele-aware
  CRISPRa/i guides; guides get their own results subsection.
- **One TE throughout** (C33, C158): L1Md_T. The LINE/SINE/LTR set is demoted to §7 plus
  Supplementary Files.
- **Lumping vs resolving** (C149, C107) promoted to a headline conceptual contribution.
- **Comparison broadened** (C101): TEtranscripts/TElocal, Telescope, scTE, ERVmap added with real
  citations, plus TE-Seq (citation verified via Crossref).
- **No em-dashes** (C15): zero in the prose, verified programmatically.

## Prototype family: why L1Md_T

Andrew asked for one TE carried throughout. **L1Md_T** is used because it is the family the
copy-resolved modules were actually run on, so every section cites a real measured number:
23,639 loci · 2 clusters · 7,490 expressed loci · 3.5% median Kimura divergence · 8 significant
DE comparisons · 9,388 intact copies · master element **chr1:95511396** · 400 guides with 17
non-dominated.

> IAPLTR1_Mm was considered as the prototype, but its copy-resolved module run has never
> completed, so those numbers do not exist. Rather than invent them, the prototype is the family
> with real data. IAPLTR1_Mm and B1_Mus2 appear in §7 with their real cross-family numbers.

## Figure status (important)

The 7 **schematics in `figs/` are real and render now**:

| File | Used as |
|------|---------|
| `fig_prototype_journey` | Fig 1, the annotation-to-reagents arc |
| `fig_locus_filtering` | Fig 2, lumping vs resolving (C149) |
| `fig_crispr_deliverable` | Fig 7, guide-design logic (C131/C157) |
| `fig_architecture` | three-layer architecture |
| `fig_pipeline_flowchart` | end-to-end flow |
| `fig_nextflow_dag` | DSL2 execution graph |
| `fig_stage11_analyses` | copy-resolved module overview |

The **data figures are not yet generated**. `reports8/` is empty on this machine, and the last
cluster manifest reported them MISSING there too, so they cannot be shipped. They render as
labelled placeholder boxes naming the exact missing file. Nothing is fabricated. Pending:

```
fig_line_results              L1Md_T UMAP + expression + divergence   (Fig 3)
fig_de_l1mdt                  cluster DE heatmap                       (Fig 4)
fig_phylo_tree                subfamily phylogeny                      (Fig 5)
fig_phylo_master              master-element nomination                (Fig 6)
fig_grna_offtarget_pareto     measured coverage/off-target frontier    (Fig 8)
fig_transduction_groups       3' transduction lineages                 (Fig 9)
fig_antisense_motifs          bidirectional promoter content           (Fig 10)
```

### To populate them

1. Generate them on the cluster (the pysam/FIPS crash is fixed via `TE_NO_PYSAM=1`):

   ```bash
   cd ~/anmol/te-scraper-and-analysis && git pull
   bsub -J terra_figs -M 32000 -n 4 -o terra_figs.log \
    "cd ~/anmol/te-scraper-and-analysis && singularity exec --env TE_NO_PYSAM=1 gameca.sif \
     python make_report_figures.py --data \
       --l1mdt-expr /home/amodz/anmol/L1Md_T_ultracombo.csv \
       --b1mus2-expr /home/amodz/anmol/B1_Mus2_ultracombo.csv \
       --iapltr1-expr /home/amodz/anmol/IAPLTR1_Mm_ultracombo.csv \
       --build mm10"
   ```

2. Copy the results next to this `.tex` (the `\graphicspath` already looks in `reports8/` and
   `reports8/stage11_l1mdt/`, so no edits are needed):

   ```bash
   rsync -av amodz@hpclogin1:~/anmol/te-scraper-and-analysis/reports8/ ./reports8/
   ```

3. Rebuild. The placeholder boxes become the real figures.

### How the numbers work (you do not have to remember anything)

One rule: **a generated file wins if it exists; otherwise the fallback in the preamble is used.**

The preamble's `\protoLoci`, `\protoGuides` etc. are the real L1Md_T values from the run Andrew
reviewed. `\inputL` looks for generated macro files **only** in `reports8/stage11_l1mdt/`, so
when a fresh L1Md_T run lands you just drop `reports8/` next to the `.tex` and rebuild: the
numbers update themselves. The prose never contains a literal number, only macros, so there is
nothing to hand-edit.

The search path is restricted on purpose. The `*_measured_values.tex` files sitting in the **repo
root** are left over from an older **IAP** run (they say `\grnaOTFamily{IAP}`, 12,577 copies, 25
lineages). If they were picked up they would silently overwrite the L1Md_T numbers with a
different family's results. Because `\inputL` never looks in the root, that cannot happen.

## Figure brand (C144)

All schematics share one style via `apply_house_style()` in `make_report_figures.py`:
DejaVu Sans, vector PDF + 150 dpi PNG, and one palette:

| Role | Hex |
|------|-----|
| Input / neutral | `#555555` |
| Core stage | `#2f6f8f` |
| Copy-resolved module | `#3f8f5f` |
| Gather / report | `#8f6f2f` |
| Execution environment | `#8f3f6f` |
| **Deliverables (primers + CRISPR)** | `#b0413e` |

Match these when producing the data figures so the set reads as one system.
