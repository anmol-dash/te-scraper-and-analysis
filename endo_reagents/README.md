# Endometrium LTR66 / LTR10G qPCR reagents

Expression-weighted qPCR primers for LTR66 and LTR10G, plus a genome-wide
mismatch-tolerant off-target scan of the candidates.

## Provenance

Per-locus expression was recomputed from SRP090091 (14 samples, 7 proliferative
+ 7 secretory) with STAR + featureCounts on a GCP VM, 2026-08-13:
`bash scripts/gcp_expression.sh up`. 14/14 samples counted, 0 failed, 638 loci.

`te_timepoint_summary.tsv` is that run's family-level result:

| subfamily | TP1 proliferative | TP2 secretory | log2FC | p (Welch) |
|-----------|-------------------|---------------|--------|-----------|
| LTR66     | 83.5 +/- 14.0 CPM | 419.7 +/- 96.4 CPM | 2.33 | 7.7e-5 |
| LTR10G    | 50.3 +/- 6.7 CPM  | 309.6 +/- 108.4 CPM | 2.62 | 7.1e-4 |

Primers were then redesigned against those numbers (`scripts/rerun_qpcr_multipair.sh`
with `EXPR_TSV=locus_expression.tsv`), which attaches per-copy expression and
makes the greedy set optimise expression coverage rather than copy count:

    LTR66   3 pairs -> union 141/374 copies (37.7%), 93.1% of family expression
    LTR10G  3 pairs -> union  59/107 copies (55.1%), 93.6% of family expression

Coordinate matching was complete for both families (386/386 and 112/112 copies
joined to a locus at 20 bp tolerance).

## Files

| file | contents |
|------|----------|
| `<fam>_endo_qpcr_pairs.csv`   | the 3-pair greedy set -- these are the ones to order |
| `<fam>_endo_top60_primers.csv`| top 60 ranked pairs by combined copy + expression coverage |
| `<fam>_endo_offtarget_{sites,summary}.csv` | off-target scan of the top-60 pool (65 / 69 unique primers) |
| `<fam>_endo_greedy_offtarget_{sites,summary}.csv` | off-target scan of the 12 ordered primers |

`_sites` is one row per genomic site (chrom, start, end, strand, n_mismatch);
`_summary` is one row per primer with the 0/1/2/3-mismatch breakdown.

## Off-target scan: what it does and does not cover

`te_primer_offtarget.py`, all 455 contigs, both strands. A site qualifies when
the primer's **3' 8 bp match exactly** and the **5' body carries <= 3 mismatches**.

The 8 bp anchor is the seed that makes a genome-wide search tractable, and it is
also the criterion's limit: **sites needing a mismatch inside the last 8 bp are
not enumerated.** Dropping the anchor to 5 (`te_qpcr_coverage.PROTECT3`) raises
seed sites from ~47k to ~3M per primer per strand. Use `--anchor` to relax it.

Validation: the 0-mismatch subset reproduces the independent exact-match counts
from `GenomeCache.search_primer` (e.g. `CACCAGACTGGGAAGCAACA` -> 41 both ways).

Totals, top-60 pools:

| | LTR66 | LTR10G |
|---|---|---|
| unique primers | 65 | 69 |
| total sites | 16,121 | 8,670 |
| perfect (0 mm) | 579 | 307 |
| on primary chromosomes | 94.6% | 94.8% |

Two cautions when reading these:

* Counts include alt/random/Un contigs (~5% of sites). Filter on `chrom` for
  primary chromosomes only.
* A per-primer site count is not an amplicon. qPCR specificity is set by the
  **pair** -- two sites must be in the right orientation and within a plausible
  distance. Neither this scan nor `te_qpcr_primers.py` enumerates paired
  off-target amplicons; that is still an open gap.
