# GAMECA Demo Walkthrough — LTR5_Hs (human ERVK LTR)

A shot-by-shot guide for a demo video showcasing the new standout modules on the
**LTR5_Hs** family (the LTR of HERV-K (HML-2), the youngest, most active human ERV —
a great demo subject because it has intact ORFs, polymorphic insertions, and known
regulatory activity).

> **Data note (read first):** the analyses need an LTR5_Hs loci table with a `Seq`
> column (and ideally `chr/start/stop/strand` for the genomic-overlay modules).
> None ship in the repo. Generate it one of two ways:
> ```bash
> # A) GAMECA local mode (needs hg38.fa available locally)
> python query.py --local --family LTR5_Hs --assembly hg38
> # B) From the UCSC RepeatMasker track: filter repName == "LTR5_Hs", then
> #    extract sequences → a CSV with columns chr,start,stop,strand,Seq,...
> ```
> Save it as `ltr5hs_loci.csv` and point `gameca.yaml: input:` at it. Until then
> every figure renders a labelled placeholder and every macro shows "(pending run)"
> — nothing is fabricated.

---

## One-command run (the headline shot)

`gameca.yaml` is preconfigured for LTR5_Hs. The entire DAG runs, resumes, and
records provenance from a single command:

```bash
python gameca_pipeline.py --config gameca.yaml            # full DAG
python gameca_pipeline.py --config gameca.yaml --dry-run  # show the plan first
scripts/run-container.sh gameca.yaml                       # same, containerized (HPC)
```

Show the `--dry-run` topological plan first, then the real run streaming
`[run] / [ok]` per stage, then re-run to show `[skip]` (resume).

---

## Files to point out, by feature

| Feature | Script(s) | What it produces | On-screen talking point |
|---|---|---|---|
| **#1/#2/#14 Phylogenetics, divergence-age, master elements** | `run_phylo_analysis.py` (+`submit_phylo_analysis.py`) | `fig_phylo_tree.pdf`, `fig_phylo_divergence.pdf`, `fig_phylo_master.pdf`, `phylo_per_copy.csv` | "MAFFT → consensus → Kimura divergence → molecular-clock age; NJ tree; ORF-integrity intact/degraded; ranks the youngest intact 'source' copies." |
| **#3 Allele-aware gRNA off-target** | `run_grna_offtarget.py` | `fig_grna_offtarget_pareto.pdf`, `..._top.pdf`, `..._mm.pdf`, `grna_offtarget_guides.csv` | "Scores guides against the *whole* copy landscape — the real CRISPR problem for repeats — and shows the specificity-vs-coverage Pareto frontier." |
| **#12b 3′ transduction** | `run_transduction.py` | `fig_transduction_groups.pdf`, `fig_transduction_polya.pdf`, `transduction_per_copy.csv` | "Groups copies sharing a transduced 3′ tail into mobilization lineages; detects poly-A signal/tail." |
| **#13 Antisense / bidirectional promoter** | `run_antisense_promoter.py` | `fig_antisense_motifs.pdf`, `fig_antisense_expr.pdf`, `antisense_per_copy.csv` | "Both-strand promoter-motif scan; antisense:sense ratio when stranded expression is provided." |
| **#12 CTCF / TAD boundaries** | `run_ctcf_tad.py` | `fig_ctcf_overlap.pdf`, `fig_ctcf_tad_distance.pdf`, `ctcf_tad_per_copy.csv` | "CTCF core-motif scan (sequence) + ChIP-peak overlap and TAD-boundary distance via bedtools." |
| **#4 Epigenetic overlay** | `run_epigenetic_overlay.py` | `fig_epigenetic_overlap.pdf`, `fig_epigenetic_heatmap.pdf` | "Annotate copies with ENCODE/Roadmap tracks (H3K9me3, ATAC, CpG) — silenced vs derepressed." |
| **#5 Orthologous insertion** | `run_ortholog_insertion.py` | `fig_ortholog_presence.pdf`, `fig_ortholog_summary.pdf` | "liftOver across species → shared (ancestral) vs lineage-specific insertions." |
| **#7 Multi-assembly liftover** | `run_multiassembly_liftover.py` | `fig_multiassembly_mapping.pdf`, `fig_multiassembly_coords.pdf` | "Same family across hg38 / T2T-CHM13 / mm10 / mm39 — T2T resolves repeat loci others miss." |
| **#11 Consensus protein structure** | `run_fold_prediction.py --per-cluster` | `fig_fold_orf_map.pdf`, `fig_fold_plddt.pdf`, `fig_fold_summary.pdf` | "Folds per-cluster consensus ORFs with ColabFold; per-residue pLDDT + PAE. `--per-cluster` calls `_majority_consensus()` per Cluster to build a nucleotide consensus, then `consensus_orfs()` extracts the longest ORF ≥ `--min-aa` per consensus. Consensus ORFs fold first; locus ORFs (from 6-frame `find_orfs()` over the top `--source-seqs` loci) top up to `--top-n`. ColabFold runs via `colabfold_batch` (auto-installed if missing via `pip install colabfold[alphafold]`, then localColabFold installer fallback). On HPC with old GCC, pass `--singularity-image /path/to/colabfold.sif`: the pipeline wraps the call in `singularity exec [--nv]` and skips the host-side `alphafold`-importability check entirely. `--use-mafft` pre-aligns ORFs with MAFFT and passes per-query `.a3m` files with `--msa-mode custom` instead of the MMseqs2 API. Results: PDB Cα backbone + per-residue pLDDT from ColabFold JSON; mean pLDDT + max PAE summary. Run via `run_stage11_all.py` (standalone orchestrator that mirrors the `query.py` `_MODULES` registry and runs every Stage 11 module against any loci CSV)." |
| **#6 Provenance & resume** | `te_provenance.py` | `provenance.json`, `.checkpoints/`, `provenance_measured_values.tex` | "git SHA, platform, tool versions, per-stage params + input hashes; resume skips done stages." |
| **#8 YAML + DAG pipeline** | `gameca_pipeline.py`, `gameca.yaml` | streamed stage log | "One declarative config composes everything; topological order; `--dry-run`, `--only`, `--force`." |
| **#10 Scheduler abstraction** | `te_scheduler.py` + every `submit_*.py` | submitted job | "Same submitter targets LSF / Slurm / local with Duo-aware auth and auto-detect." |
| **#9 Containerization** | `gameca.def`, `scripts/run-container.sh`, `Dockerfile` | `gameca.sif` | "Apptainer/Singularity for install-free HPC; one command builds + runs." |

### Stage 11 — how the code works (updated)

**Orchestrator:** `run_stage11_all.py` mirrors `query.py`'s `_MODULES` registry exactly and runs all Stage 11 modules (phylo, grna, transduction, antisense, ctcf_tad, epigenetic, ortholog, multiassembly, fold, divergence, ltr_struct, subfamily, benchmark, motif_gain, expression) against any loci CSV outside the main pipeline. Each module is a sibling `run_*.py` script invoked as a subprocess; logs stream to `standout_analysis.log`. Pass `--only fold,phylo` or `--skip expression` to target specific modules.

**Expression (#15):** `te_expression.py` no longer picks expression-column
order on its own when run through `query.py` — the pipeline requires the user
to designate expression column names and their exact figure order up front
(`--expr-cols`/`--expr-labels` on `query.py`, resolved once at data-load time
and persisted to `expr_cols.json`), then passes that resolved list to
`te_expression.py` as `--stage-cols`/`--stage-labels`. This is also what
enables the `stage_profile.png`/`primer_expression.png` figures, which are
otherwise skipped without an explicit column order. Standalone invocations of
`te_expression.py` (outside `query.py`) still fall back to auto-detection
unless `--stage-cols` is passed directly.

**Fold prediction (#11):** `--per-cluster` builds a position-wise majority consensus (`_majority_consensus()`) for each `Cluster` group, then `consensus_orfs()` finds the longest ORF ≥ `--min-aa` aa per consensus and deduplicates by protein sequence. Consensus ORFs fold first; the top `--source-seqs` loci (ranked by total expression or sequence length) are then scanned in all 6 frames (`find_orfs()`) and deduplicated unique locus ORFs top up the list to `--top-n`. ColabFold auto-installation: tries `pip install colabfold[alphafold]` first (≈500 MB), then falls back to the localColabFold bash installer (~20 GB). On HPC with an old host GCC, pass `--singularity-image /path/to/colabfold.sif`: this wraps every `colabfold_batch` call in `singularity exec [--nv] <sif>` and skips the host-side `alphafold`-module importability check entirely (the module lives inside the container). Add `--no-gpu` to drop `--nv` for CPU-only runs. `--use-mafft` pre-aligns ORF proteins with MAFFT and generates per-query `.a3m` files passed to ColabFold with `--msa-mode custom` instead of the default MMseqs2 online API.

**Motif scanning (te_moods_scan.py — new):** the old pipeline downloaded a genome-wide JASPAR bigBed, subsetted it with bigBedToBed + tabix, then ran bedtools intersect. The new `te_moods_scan.py` replaces the download + intersect leg with a direct MOODS-based scan of the TE locus sequences against the JASPAR PFM bundle:

1. `parse_jaspar_pfms()` reads the JASPAR raw PFM flat file (`>MAxxxx.y TFname` header + `A/C/G/T [ … ]` rows).
2. `build_scanner()` converts counts to log-odds via `MOODS.tools.log_odds()`, computes p-value thresholds, adds reverse-complement matrices, and builds a single `MOODS.scan.Scanner` covering all forward + RC matrices.
3. `scan_loci()` dispatches loci in chunks of 64 across a multiprocessing pool (fork on Linux, spawn on macOS). Each worker (`_scan_chunk`) pulls sequences from: (a) a pre-loaded `Seq` column — no genome FASTA needed; (b) a pysam-indexed FASTA; or (c) a UCSC `getData/sequence` API fallback that routes through `curl` (not Python sockets) so it works on HPC nodes with a blocked socket stack.
4. Output is a DataFrame with columns `[chr, start, stop, Motif_chr, Motif_start, Motif_end, Motif_name, Motif_score, Motif_strand]` — the same schema the old bedtools path produced, so the downstream Fisher enrichment in `te_motif.py` is unchanged. Default p-value threshold: 1e-4 (JASPAR/FIMO field standard).

The old `te_motif.py` JASPAR BED resolution order (provided file → env var → CMMT 2022 cache → pybigtools bigBed query → bigBedToBed binary → bulk JASPAR 2024 BED → per-motif TSV.gz streaming) is still used for the Fisher enrichment step on the full loci set.

### Report
- `gameca_report.tex` → new **§ "Standout analysis modules"** stitches every
  figure + measured-value macro into the paper. Build with `tectonic gameca_report.tex`
  (needs `references.bib`). All `*_measured_values.tex` are auto-written by the runs.

---

## Suggested 3-minute storyboard
1. **Hook** — `gameca_pipeline.py --config gameca.yaml --dry-run` → show the DAG.
2. **Run** — real run streaming stages; cut to `fig_phylo_tree.pdf` + master-element bar.
3. **CRISPR** — `fig_grna_offtarget_pareto.pdf`; explain why repeats break normal tools.
4. **Biology breadth** — quick montage: transduction lineages, antisense motifs, CTCF/TAD.
5. **3D / cross-species** — multi-assembly mapping (T2T), ortholog presence matrix.
6. **Structure** — pLDDT figure of the consensus ORF.
7. **Trust** — open `provenance.json`; re-run to show `[skip]` resume; mention the container.
8. **Close** — the compiled `gameca_report.pdf` scrolling through the new section.
