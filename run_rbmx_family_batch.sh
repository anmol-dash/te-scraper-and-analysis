#!/usr/bin/env bash
#
# Batch consensus + CIAlign + primer design + Stage-11 standout analysis for
# the RBMX-enrichment family list.
#
# For each family, runs the full GAMECA pipeline in --local mode (auto-downloads
# RepeatMasker coordinates + fetches sequences — no pre-existing CSV/genome
# needed): clusters, builds per-cluster consensus sequences + CIAlign alignment
# plots, designs primers, and then runs Stage 11 (phylo/age, gRNA off-target,
# transduction, antisense promoter, CTCF/TAD, epigenetic overlay, ortholog +
# multi-assembly liftover, fold prediction, divergence landscape, LTR structure,
# subfamily resolution, benchmarking, motif gain, expression). Motif/TFBS/GO
# (the *core*-pipeline JASPAR stage, not Stage 11) is skipped — it wasn't part
# of your ask and would roughly double the per-family runtime across 33 families.
#
# Usage:
#   chmod +x run_rbmx_family_batch.sh
#   ./run_rbmx_family_batch.sh
#
# Re-running is safe: each family writes to its own results_rbmx_families/<family>/
# folder, and failed families can be re-run individually (see the summary file).

set -uo pipefail

ASSEMBLY="hg38"
OUT_DIR="results_rbmx_families"
MAX_LOCI=3000        # cap loci per family for speed/reliability — raise this
                     # (or comment out the --max-loci line below) for a full run
LOG_DIR="$OUT_DIR/_logs"
mkdir -p "$LOG_DIR"

# ── Stage 11 options — chosen for *this* family list ───────────────────────
# All families are human/hg38, so the human defaults for --subst-rate,
# --clock-divisor, --intact-orf-aa, --min-ltr-identity, --cpg-omega etc. are
# already appropriate and left unset. The flags below are the ones that
# actually change what gets analysed for these particular families:
#
#   --target-assemblies t2t
#     Several of these groups (HERV9*, MER52*, LTR12C, and especially the
#     satellites GSAT*/HSAT4/HSAT5) are concentrated in pericentromeric /
#     heterochromatic regions that hg38 leaves unresolved or collapsed —
#     T2T-CHM13 is exactly the comparison that surfaces copies hg38 misses.
#
#   --ortholog-species panTro6 rheMac10
#     HERV9*, LTR12C, MER52*, MER66D, LTR1D/LTR1D1, LTR2752 are primate-
#     lineage LTR/ERV elements — chimp (panTro6) and rhesus macaque (rheMac10)
#     bracket "predates human/chimp split (~6-7 Mya)" vs "predates Old-World-
#     monkey split (~25-30 Mya)" vs "human/great-ape lineage-specific", which
#     is the natural age question for this kind of family.
#
#   --epigenetic-preset K562
#     ENCODE's most deeply profiled line (broadest track panel) — a sensible
#     general-purpose choice for "which of these copies sit in active
#     regulatory chromatin" across such a broad, mixed family list.
#
#   --ctcf-preset GM12878 --tads-preset GM12878
#     GM12878 is the best-characterised ENCODE line for both CTCF ChIP-seq
#     and Hi-C/TAD maps — relevant here because LTR/ERV-derived CTCF sites
#     are a well-documented mechanism by which these element types reshape
#     genome architecture.
#
# NOTE: --target-assemblies / --ortholog-species require the UCSC `liftOver`
# binary; --colabfold-cmd requires `colabfold_batch`. Neither is currently on
# this machine's PATH (only `mafft` is) — those modules will report
# "not found" until you install them locally or point --liftover-cmd /
# --colabfold-cmd at an install on your HPC cluster. Everything else below
# runs with no extra dependencies.

TARGET_ASSEMBLIES="t2t"
ORTHOLOG_SPECIES="panTro6 rheMac10"
EPIGENETIC_PRESET="K562"
CTCF_PRESET="GM12878"
TADS_PRESET="GM12878"

# ── Family list, grouped exactly as you described ──────────────────────────
FAMILIES=(
  # THE1A group (THE1A + THE1B/C/D and their -int regions)
  THE1A THE1A-int THE1B THE1B-int THE1C THE1C-int THE1D THE1D-int

  # MSTA1 group (MSTA1, MSTA1-int, MSTB-int, MST-int)
  MSTA1 MSTA1-int MSTB-int MST-int

  # HERV9-int group (HERV9-int, HERVK9-int, HERV9N-int, HERV9NC-int)
  HERV9-int HERVK9-int HERV9N-int HERV9NC-int

  # LTR1D group (LTR1D, LTR1D1)
  LTR1D LTR1D1

  # LTR12C group (LTR12C itself, plus LTR12F, LTR12E, LTR12D — confirmed; the
  # earlier "LTR12-" placeholder was just LTR12C, already listed here)
  LTR12C LTR12F LTR12E LTR12D

  # Standalone element
  LTR2752

  # MER52A group (MER52A, MER52C, MER52D, MER52-int)
  MER52A MER52C MER52D MER52-int

  # Standalone element
  MER66D

  # Satellite repeats — NOTE: these are tandem-repeat satellites, structurally
  # different from the dispersed TEs this pipeline is built around. RMSK
  # annotates them as very large numbers of short fragments, so clustering /
  # consensus / primer design may behave differently (and be slower) here.
  # "HSTA5" in your message is assumed to be a typo for HSAT5 — edit if wrong.
  GSAT GSATII GSATX
  HSAT4 HSAT5
)

SUMMARY="$OUT_DIR/_summary.tsv"
mkdir -p "$OUT_DIR"
echo -e "family\tstatus\tlog" > "$SUMMARY"

for FAM in "${FAMILIES[@]}"; do
  LOG="$LOG_DIR/${FAM}.log"
  echo "============================================================"
  echo " $FAM  ->  $LOG"
  echo "============================================================"

  if python3 query.py --local --family "$FAM" --assembly "$ASSEMBLY" \
        --max-loci "$MAX_LOCI" \
        --skip-motif --skip-tsne \
        --target-assemblies $TARGET_ASSEMBLIES \
        --ortholog-species $ORTHOLOG_SPECIES \
        --epigenetic-preset "$EPIGENETIC_PRESET" \
        --ctcf-preset "$CTCF_PRESET" \
        --tads-preset "$TADS_PRESET" \
        --output "$OUT_DIR" \
        > "$LOG" 2>&1
  then
    echo -e "${FAM}\tOK\t${LOG}"     >> "$SUMMARY"
    echo "  OK"
  else
    echo -e "${FAM}\tFAILED\t${LOG}" >> "$SUMMARY"
    echo "  FAILED -- see $LOG"
  fi
done

echo
echo "============================================================"
echo " Done. Summary: $SUMMARY"
echo "============================================================"
echo
echo "Per-family results live in: $OUT_DIR/<family-name-lowercased>/"
echo "  Consensus sequences : 05_consensus/"
echo "                        cluster_alignments/all_cluster_consensuses.fa"
echo "  CIAlign plots       : cialign_plots/index.html   (open in a browser)"
echo "  Primers             : 06_primers/                (global + per-cluster CSVs)"
echo "  Stage 11 reports    : reports/  (phylo, gRNA, transduction, antisense,"
echo "                        CTCF/TAD, epigenetic overlay, ortholog + multi-"
echo "                        assembly liftover, fold prediction, divergence"
echo "                        landscape, LTR structure, subfamily tree, ...)"
echo "  Stage 11 run log    : standout_analysis.log"
echo
echo "To re-run just the families that failed:"
echo "  awk -F'\\t' '\$2==\"FAILED\"{print \$1}' $SUMMARY"
