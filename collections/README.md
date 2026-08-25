# collections

Result sets produced on 2026-08-25, kept together so a number can be traced
back to the run that produced it.

## qpcr_primers/

qPCR primer pairs for MER21C and Harlequin-int, with the evidence behind both
the coverage claim and the specificity claim.

| file | what it holds |
|---|---|
| `QPCR_PAIRS_WITH_EVIDENCE.csv` | **start here.** 12 pairs x 37 columns: sequences, Tm/GC/amplicon, copies + expression covered (per pair and cumulative), off-target sites and off-target *amplicons*, and a `specificity_flag` |
| `QPCR_OFFTARGET_AMPLICONS.csv` | every predicted off-target product: coordinates, size, mismatches per end, on/off-family |
| `QPCR_OFFTARGET_AMPLICON_SUMMARY.csv` | the same, one row per pair |
| `<FAM>_qpcr_pairs.csv` | raw design output from `te_qpcr_primers.py` |
| `<FAM>_offtarget_sites.csv` | every priming site, genome-wide |
| `<FAM>_offtarget_summary.csv` | per-primer site counts by mismatch |

How it was made:
1. `te_qpcr_primers.py` --- primer3 on real copies (never on a consensus, which
   makes chimeric primers in a divergent family), then greedy selection of the
   pair set maximising covered EXPRESSION. 6 pairs, candidates from the 4
   largest sub-clusters, 25 copies seeded per group.
2. `te_primer_offtarget.py` --- genome-wide, hg38, 455 contigs, both strands,
   3' anchor 8 exact + <= 3 mismatches in the 5' body.
3. `pair_sites.py` --- pairs those sites into predicted amplicons. A priming
   site is not a product; a product needs two sites facing each other at a
   workable distance. Counts fwd+rev plus fwd+fwd and rev+rev.
4. `build_table.py` --- recomputes coverage from the copy sequences (rather
   than trusting the design run) and joins the specificity evidence.

Headline: Harlequin-int pair4 covers 18.5% of copies / 54.2% of expression with
1 off-family product. Pair0 + pair4 together reach ~96% of expression. MER21C
has no pair above 1.9% of family expression --- consistent with the guide work,
and structural, not a pipeline artefact.

Limits, stated rather than hidden:
* Sites needing a mismatch inside the last 8 bases are NOT enumerated --- the
  anchor is the search seed, so the criterion is only as permissive as the
  anchor is short.
* A predicted amplicon is sequence-level. It is not a measured product.

## motif_background_comparison/

The same three families (LTR66, MER21C, Harlequin-int) scanned twice, against
two different backgrounds, so the background's effect on the ranking is visible
rather than assumed.

| arm | background | what it asks |
|---|---|---|
| `dinuc/` | per-copy dinucleotide-preserving shuffle, 25x | is this motif explained by this family's own base composition? |
| `genomic/` | length-matched random intervals off-family, 25x | is this motif more common here than in average genomic sequence? |

Everything else is identical between arms: JASPAR2024 CORE vertebrates (879
matrices), MOODS p < 1e-4. Copies scanned: LTR66 405, MER21C 5590,
Harlequin-int 523.

| file | what it holds |
|---|---|
| `MOTIF_BACKGROUND_COMPARISON.tsv` | **start here.** every motif x family, both arms side by side, with `rank_shift` and an `agreement` column |
| `COMPARISON_SUMMARY.txt` | readable, top 100 per family |
| `<arm>/motifs_<FAM>_enrichment.tsv` | full ranked table for one arm |
| `<arm>/MOTIF_SEQUENCES.tsv` | every motif and its sequence |
| `<arm>/TOP_MOTIFS.txt` | per-arm human-readable summary |

Every table carries the motif SEQUENCE, not just a TF name: `consensus` is the
matrix's IUPAC core, `observed_consensus` is built from the bases these copies
actually carry.

Significant at FDR < 0.05:

| family | dinuc | genomic | both | dinuc only | genomic only |
|---|---|---|---|---|---|
| LTR66 | 159 | 148 | 145 | 14 | 3 |
| MER21C | 423 | 344 | 285 | 138 | 59 |
| Harlequin-int | 280 | 250 | 206 | 74 | 44 |

What the comparison shows: the motifs that collapse under the genomic
background are the composition-driven ones --- GC-repeat matrices (LTR66
ZNF213 rank 3 -> 102, FEZF2 5 -> 104, KLF9 12 -> 55) and AT-rich homeodomain
matrices (MER21C POU3F2 14 -> 287, POU4F2 10 -> 190). Forkhead does not behave
like them: FOXN3 in LTR66 is fold 15.1 against the composition-matched shuffle
and 15.3 against genomic. That is the evidence that the FOX signal in LTR66 is
not a base-composition artefact, despite the family being AT-rich (40.5% GC).

Caveat that applies to both arms: a PWM match is a sequence match. It is not
evidence of binding, and says nothing on its own about endometrium.
