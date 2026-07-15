# Andrew Modzelewski's review — all comments, compiled

Source: `GAMECA_Report_AJM.docx` (track changes + 22 margin comments), plus his summary note.
This file is the checklist the rewrite (`terra_report.tex`) was written against.

Status key: **Done** = addressed in this draft · **Lab** = needs a decision or input from the lab ·
**Pre-submission** = flagged in the draft as work to do before submitting.

---

## Summary note (overall direction)

> "Once that happens, can Anmol compile all of the comments together for a new draft? I made most
> of my changes as track changes and made many comments but if anything doesn't make sense, please
> feel free to slack me for clarification. I feel like the MAJOR thing missing is that this mostly
> feels like a really detailed methods section instead of the results section. My advice is to use
> a specific TE as a prototype throughout and explain the procedural outputs your results and
> reports are showing (what you made for katie, Claire and Diego). You can turn these three into
> attached Supp Files but focus on a single case throughout for simplicity. These files you made
> for them had very useful graphs and results are largely missing from the main paper, even if they
> are mentioned in principle (alignments, motif analysis, primer output, guide off targets, etc). I
> also strongly feel like you need to mention the CRISPR stuff earlier and more prominently. This
> is what makes the tool go from 'useful' to 'Necessary' to the field."

**How the draft answers it:** the paper now opens on the deliverables, and the entire
`Results` section (§2) is a single-family walkthrough of **L1Md_T** from RepeatMasker annotation
to ordered reagents. Methods moved *after* the results into §5 (Design and implementation). The
three-family set is demoted to §7 plus Supplementary Files. CRISPR is in the title, the abstract,
the introduction, and has its own results subsection (§2.6).

---

## The 22 comments

| # | Comment (verbatim, abridged where noted) | Status | How it is addressed |
|---|---|---|---|
| **C0** | Wants "TE"/transposon in the name. "Maybe something like: TERRA (Transposable Element Repeat Resolved Analysis). Sounds like 'Terra,' which nicely implies large-scale genomic data… if you wanna go down the fantasy/D&D route, you'd use 'Gaia'." Let's crowdsource the lab. | **Lab** | Renamed **TERRA** throughout. A red `\todo` on p.1 flags the name as provisional pending the lab's crowdsourcing. |
| **C1** | "We'll add more names. We should include Vamshi and the grad students and who ever else contributed meaningful input." | **Lab** | Author line now lists Vamshi and A.J.M. with a `\todo` for the full list/affiliations. |
| **C2** | "Very impressive and very publishable. The strongest version of the paper is a software/methods manuscript… Before submission, I'd like you to strengthen four things: broaden the related-tool comparison, validate the clustering and noise-class handling, make expression and motif-counting units explicit, and separate the fully tested core workflow from modules that are still more exploratory. My current best journal target would be BMC Bioinformatics, with Bioinformatics Advances or NAR Genomics and Bioinformatics as stronger options…" | **Done** (3 of 4) + **Pre-submission** | (1) Comparison broadened → §3 Table 1 (see C101). (2) Units made explicit → §4 "Expression units" and "Motif counting units". (3) Tested core vs exploratory → §4 "Scope of the reported results" + §6 Exploratory modules. (4) Clustering/noise-class **validation** → §4 documents the parameter search and noise handling, and `run_cluster_validation.py` measures the noise fraction and partition stability (ARI) across the k / n_neighbors sweep, reported in §4 + Fig 11. **Run it to populate the numbers.** |
| **C15** | "Reviewers tend to 'frown' on these '--' dashes, because they aren't quite common in writing but AI loves them, so we need to use them sparingly just to make sure we don't get a desk rejection before getting into review." | **Done** | **Zero** em-dashes in the prose (verified programmatically). The only `--` remaining is the typographically correct `Benjamini--Hochberg`. |
| **C28** | "Vamshi has some thoughts about how to advertise this." | **Lab** | Not actionable in the draft; needs Vamshi. |
| **C33** | "I feel like REAL examples are missing from this report. I'd like to see a lot of the procedurally generated graphs and data you sent to Katie, Claire and Diego here. Expression, alignments, new consensus sequences, clusters, Motif analysis, etc. You can use a TE as a prototype through out to keep it simple to follow but you can upload those reports as 'supp data' as examples to show that this works for LINE/SINE/LTR." | **Done** | §2 is the L1Md_T prototype walkthrough carrying clustering, expression, DE, consensus/phylogeny, motif, transduction, antisense, primers and guides. LINE/SINE/LTR → §7 + Supplementary Files. **Caveat:** the data figure *files* are not yet generated (see README). |
| **C101** | "This might be a little thin. A quick review suggest you are missing a few well known ones. We need to worry about alienating potential reviewers who would be friends but if they are absent in this paper, it might annoy them. TEtranscripts/TElocal, Telescope, scTE (single cell might not matter?), TE-Seq, ERVmap" | **Done** (TE-Seq pending) | §3 Table 1 now includes TEtranscripts/TElocal, Telescope, scTE, ERVmap with real citations added to `references.bib`, plus a new "Repeat expression quantification" paragraph. **TE-Seq is listed with a `\todo` for its citation** — I would not invent a reference. |
| **C102** | "I'm not sure I understand this statement as written." (on the family-level analysis claim) | **Done** | The sentence was rewritten. §1 now states plainly why the family is the unit and why that is insufficient on its own. |
| **C106** | "Stage? Maybe I am too far away from this type of literature." | **Done** | "Stage 11" jargon removed. Modules are now called "copy-resolved modules"; developmental "stages" are named explicitly (oocyte → morula). |
| **C107** | "I feel like we need to emphasize the 'expressed consensus vs common consensus' a littler better earlier." | **Done** | Now in the **abstract**, in §1 ("Which copies matter"), and as a named point in §2.2 and the Discussion. |
| **C108** | "Is this your vocab or does the field just talk cool like this?" | **Done** | Idiosyncratic phrasing removed; the flagged passage rewritten in plain register. |
| **C130** | "I'm going to lean pretty heavily on Vamshi in these sections." | **Lab** | Noted; sections left structured for Vamshi's input. |
| **C131** | "To make this more useful to the field, I think we need to include the steps you made to help katie, where we can use the same idea to design CRISPR reagents. (CrisprA/I)" | **Done** | §2.6 "Deliverable 1: allele-aware CRISPRa/i guides" describes the design path and states explicitly that the same output supports both CRISPRa and CRISPRi. |
| **C138** | "Really?" (skepticism at a claim) | **Done** | The overclaim was removed; the surrounding text now states only what the run measured. |
| **C144** | "We'll need to make sure all figures have a certain internal consistency. Color, font, overall 'brand', if you know what I mean." | **Done** | All schematics share one house style (`apply_house_style()` in `make_report_figures.py`): one palette, DejaVu Sans, vector PDF. Palette documented in README.md. |
| **C149** | "I'd like more comments about this. I personally thing THIS is a big conceptual innovation that puts a fine point on the good work you did throughout. If we 'lump them all together' the results are averaged across meaningful, low function, and MANY irrelevant/inert loci." | **Done** | Promoted to a **headline conceptual contribution**: abstract, §1 ("Which copies matter"), its own results subsection §2.2 with `fig_locus_filtering`, and the first contribution listed in the Discussion. |
| **C150** | "For readability, these asides are something quite long." | **Done** | Long parenthetical asides broken up or cut; the worst offenders became their own short paragraphs. |
| **C153** | "This feels a little too combative and 'familiar'. We need to be as neutral and ambivalent as we can. This can be mentioned in the discussion but in the results section we need to hit above the belt." | **Done** | Results are neutral and descriptive. Positioning against other tools moved to §3 (framed as scope, not competition) and the Discussion, where it is stated as complementarity. |
| **C157** | "I think we need CRISPR in the title." | **Done** | Title: *"TERRA: an accessible, HPC-integrated pipeline for downstream transposable-element family analysis and **CRISPR reagent design**."* |
| **C158** | "Is this a real example? If so, you should name the TE in detail. This is interesting, almost figure/table worthy. This might deserves its own prototype pipeline where you start with the DFAM or RMSK database numbers, use your method to figure out which insertions might matter and then make these reagents (Primers and CRISPR). We can add the CRISPR details later, just worry on making the pipeline sleek and user friendly." | **Done** | Yes, it is real. §2 is exactly this arc: RMSK count (23,639) → which copies matter (clusters, 7,490 expressed, 9,388 intact) → named master element (**chr1:95511396**) → reagents (primers + 400 guides, 17 non-dominated). `fig_prototype_journey` renders the arc. |
| **C159** | "I think this is the first mention of this! You should mention this in the abstract and intro. THIS is what people want to see! The deliverables!" | **Done** | The abstract now leads with the two deliverables; §1 has a dedicated "Reagents are the deliverable" paragraph. |
| **C160** | "This seems like a good idea to include. Is it too cumbersome to show examples?" | **Done** | Provenance manifest (git SHA, platform, tool versions) and stage checkpoints described concretely in §6. |

---

## Open items for the lab

1. **C0** — final name. TERRA is used throughout; swap is a single find/replace if the lab prefers another.
2. **C1** — full author list and affiliations.
3. **C28 / C130** — Vamshi's input on positioning and his sections.
4. **C101** — a citation for **TE-Seq** (deliberately not invented).
5. **C2** — run `run_cluster_validation.py` on the L1Md_T clustered CSV to populate the sweep numbers (script written; needs the real input). Consensus-alignment cross-check still outstanding.
6. **Figures** — the data-figure files still need to be generated; see `README.md`.
