#!/usr/bin/env bash
# run_endometrium_ltr.sh --- ONE command, whole study: LTR66 + LTR10G in human
# endometrium (SRA study SRP090091 / PRJNA342633), through to bench reagents.
#
# WHAT THE DATA IS (verified against ENA, 2026-08-05):
#   SRP090091 "Comprehensive RNA sequencing of healthy human endometrium at two
#   time points of the menstrual cycle". 40 runs total:
#     14 total-RNA  (RNA-Seq, PAIRED)  from endometrial TISSUE  <-- used here
#     14 small-RNA  (ncRNA-Seq, PAIRED) from endometrial tissue     not used
#     12 small-RNA  (ncRNA-Seq, SINGLE) from endometrial STROMAL CELLS  not used
#   Groups among the 14 total-RNA samples: 7 TP1 (proliferative) vs
#   7 TP2 (secretory, 7-9 days post-ovulation).
#   NOTE: the cultured stromal-cell samples are small-RNA only, so LTR66/LTR10G
#   expression can only come from the TISSUE total-RNA arm. The stromal
#   fibroblast model is where the reagents below get TESTED, not where the
#   expression evidence comes from.
#
# WHAT IT PRODUCES:
#   $WORK/te_timepoint_summary.tsv  LTR66/LTR10G CPM, TP1 vs TP2, log2FC
#   $WORK/locus_expression.tsv      per-COPY CPM for every LTR66/LTR10G locus
#   $REAGENT_WORK/<fam>/<fam>_endo_qpcr_pairs.csv   qPCR pairs, expression-weighted
#   $REAGENT_WORK/<fam>/<fam>_endo_top*_primers.csv ranked primer candidates
#   $REAGENT_WORK/<fam>/grna_candidates.csv         CRISPR guides (a/i/KO)
#   $REAGENT_WORK/<fam>/grna_offtarget_report.txt   coverage vs off-target
#
# HOW IT RUNS: stage 1 (references/containers) runs here on the login node;
# stages 2-6 are LSF jobs chained with -w dependencies, so this script submits
# and exits. Everything is idempotent -- re-run it after a failure and completed
# work is skipped.
#
#   bash scripts/run_endometrium_ltr.sh              # go
#   bash scripts/run_endometrium_ltr.sh --dry-run    # print the plan, submit nothing
#   bash scripts/run_endometrium_ltr.sh --status     # where is it up to
#
# First run downloads ~65 GB of FASTQ and (if not already present) ~35 GB of
# references + STAR index. Run it under tmux/nohup: stage 1 can take ~30-60 min.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
SCRIPTS="$REPO/scripts"

# --- knobs ------------------------------------------------------------------
WORK=${WORK:-$HOME/endometrium_ltr}          # fastq + counts for THIS study
# References and containers are big and study-independent: reuse the HERVK ones
# if they are already built, otherwise they get built inside $WORK.
REF=${REF:-$HOME/hervk_ccle/ref}
CONT=${CONT:-$HOME/hervk_ccle/containers}
REAGENT_WORK=${REAGENT_WORK:-$HOME/endometrium_reagents}
MANIFEST=${MANIFEST:-$SCRIPTS/srp090091_manifest.tsv}
FAMILIES=${FAMILIES:-"LTR66 LTR10G"}
TAG=${TAG:-endo}                              # output suffix, keeps datasets apart
SIF=${SIF:-$REPO/gameca.sif}
GENOME="$REF/hg38.fa"
QUEUE=${QUEUE:-rhel9}
THREADS=${THREADS:-12}
MEM_MB=${MEM_MB:-45000}
CAS=${CAS:-SpCas9}
# Only loci of these subfamilies are kept in the merged per-locus table.
# transcript_id in the TE GTF is "<repName>_dup<N>", so this anchors exactly.
FILTER=${FILTER:-'^(LTR66|LTR10G)_dup'}
# Strandedness: SRP090091's library prep is not recorded in the SRA metadata,
# so both counters run unstranded (each tool's own default). If you establish
# the libraries are dUTP/TruSeq-stranded, re-run with STRANDED=reverse FC_STRAND=2.
STRANDED=${STRANDED:-no}
FC_STRAND=${FC_STRAND:-0}
# HDBSCAN min_cluster_size per family. query.py's default is 100, which is wrong
# for both of these: in hg38 rmsk (standard chromosomes only) LTR66 has 386
# copies and LTR10G has 112, so a floor of 100 would yield one cluster plus
# noise and the per-sub-cluster primer design would have nothing to work with.
MCS_LTR66=${MCS_LTR66:-25}
MCS_LTR10G=${MCS_LTR10G:-10}
NPAIRS=${NPAIRS:-3}
NCLUST=${NCLUST:-3}
TOP_N=${TOP_N:-100}
# LTR66 is heavily fragmented in hg38 -- median copy is 195 bp and only ~59% of
# copies reach 150 bp -- so a copy shorter than AMP_MIN simply cannot be
# amplified and caps the reportable coverage. 70-140 bp is still a normal qPCR
# amplicon and lets more of the short copies participate. MAX_REPS widens the
# primer3 seed pool, which these families (386 / 112 copies) can support.
AMP_MIN=${AMP_MIN:-70}
AMP_MAX=${AMP_MAX:-140}
MAX_REPS=${MAX_REPS:-30}

# per-study LSF job names so this never collides with the HERVK jobs
J_DL=${J_DL:-endo_dl}
J_ALIGN=${J_ALIGN:-endo_align}
J_MERGE=${J_MERGE:-endo_merge}
J_REAGENT=${J_REAGENT:-endo_reagents}
J_QPCR=${J_QPCR:-endo_qpcr}

DRY=0; STATUS_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    --status)  STATUS_ONLY=1 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg"; exit 2 ;;
  esac
done

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
run() { if [ "$DRY" = 1 ]; then echo "DRY: $*"; else "$@"; fi; }

N_SAMPLES=$(($(grep -c . "$MANIFEST") - 1))

# --- status -----------------------------------------------------------------
show_status() {
  say "status"
  printf '  %-26s %s\n' "manifest"      "$MANIFEST ($N_SAMPLES runs)"
  printf '  %-26s %s\n' "work dir"      "$WORK"
  printf '  %-26s %s\n' "reference dir" "$REF"
  printf '  %-26s %s\n' "reagent dir"   "$REAGENT_WORK"
  local have_fq=0
  while IFS=$'\t' read -r _ _ run _; do
    [ -z "${run:-}" ] && continue
    [ -s "$WORK/fastq/${run}_1.fastq.gz" ] && [ -s "$WORK/fastq/${run}_2.fastq.gz" ] \
      && have_fq=$((have_fq + 1))
  done < <(tail -n +2 "$MANIFEST")
  printf '  %-26s %s\n' "hg38.fa"        "$([ -s "$GENOME" ] && echo present || echo MISSING)"
  printf '  %-26s %s\n' "STAR index"     "$([ -s "$REF/star_hg38/SAindex" ] && echo built || echo 'not built')"
  printf '  %-26s %s\n' "fastq pairs"    "$have_fq / $N_SAMPLES"
  printf '  %-26s %s\n' "cntTable files" "$(ls "$WORK"/counts/*.cntTable 2>/dev/null | wc -l | tr -d ' ') / $N_SAMPLES"
  printf '  %-26s %s\n' "featureCounts"  "$(ls "$WORK"/locus/*.featureCounts.txt 2>/dev/null | wc -l | tr -d ' ') / $N_SAMPLES"
  printf '  %-26s %s\n' "locus table"    "$([ -s "$WORK/locus_expression.tsv" ] && echo present || echo 'not built')"
  printf '  %-26s %s\n' "timepoint table" "$([ -s "$WORK/te_timepoint_summary.tsv" ] && echo present || echo 'not built')"
  for fam in $FAMILIES; do
    printf '  %-26s %s\n' "reagents $fam" \
      "$([ -s "$REAGENT_WORK/$fam/${fam}_${TAG}_qpcr_pairs.csv" ] && echo done || echo pending)"
  done
  echo
  bjobs -w 2>/dev/null | grep -E "$J_DL|$J_ALIGN|$J_MERGE|$J_REAGENT|$J_QPCR|star_index" || echo "  (no matching jobs in the queue)"
}

if [ "$STATUS_ONLY" = 1 ]; then show_status; exit 0; fi

# --- preflight --------------------------------------------------------------
say "preflight"
[ -s "$MANIFEST" ] || { echo "FATAL: manifest not found: $MANIFEST"; exit 1; }
[ -s "$SIF" ] || { echo "FATAL: gameca.sif not found at $SIF (build_sif.sh, or set SIF=)"; exit 1; }
command -v bsub >/dev/null || { echo "FATAL: bsub not on PATH -- run this on the LSF cluster"; exit 1; }
command -v singularity >/dev/null || { echo "FATAL: singularity not on PATH"; exit 1; }
echo "  manifest: $MANIFEST ($N_SAMPLES paired total-RNA runs)"
echo "  families: $FAMILIES   (min_cluster_size: LTR66=$MCS_LTR66 LTR10G=$MCS_LTR10G)"
echo "  strandedness: TEcount=$STRANDED featureCounts=-s $FC_STRAND"

# disk: ~65 GB fastq + transient BAMs; references add ~35 GB if not already built
avail_gb=$(df -Pk "$(dirname "$WORK")" 2>/dev/null | awk 'NR==2{printf "%d", $4/1048576}')
need_gb=120
[ -s "$REF/star_hg38/SAindex" ] || need_gb=$((need_gb + 40))
echo "  disk free at $(dirname "$WORK"): ${avail_gb:-?} GB (need ~${need_gb} GB)"
if [ -n "${avail_gb:-}" ] && [ "$avail_gb" -lt "$need_gb" ]; then
  echo "  WARNING: that is tight. Set WORK= to a larger filesystem if the run dies mid-way."
fi
mkdir -p "$WORK" "$REAGENT_WORK"

# ============================================================================
# STAGE 1 (login node): containers, hg38, GTFs, STAR index, FASTQ download job
# ============================================================================
say "stage 1/6: references, containers, FASTQ download  (login node)"
if [ "$DRY" = 1 ]; then
  echo "DRY: WORK=$WORK REF=$REF CONT=$CONT MANIFEST=$MANIFEST JOB=$J_DL bash $SCRIPTS/setup_hervk_ccle.sh"
else
  # setup_hervk_ccle.sh is study-agnostic: it builds refs/containers, submits the
  # STAR index build if missing, and hands the manifest to the ENA download job.
  # derive the GTF sanity-check regex from FAMILIES so it tracks any override
  fam_grep="^($(printf '%s' "$FAMILIES" | tr -s ' ' '|'))$"
  env WORK="$WORK" REF="$REF" CONT="$CONT" MANIFEST="$MANIFEST" JOB="$J_DL" \
      QUEUE="$QUEUE" THREADS="$THREADS" MEM_MB="$MEM_MB" \
      FAM_GREP="$fam_grep" \
      bash "$SCRIPTS/setup_hervk_ccle.sh"
fi

# Which upstream jobs must stage 2 wait for? Decide from the FILESYSTEM, not from
# `bjobs` exit codes: a dependency on a job that does not exist leaves the whole
# chain PEND forever, and `bjobs -J` also stops matching once a job is DONE.
# bjobs is used only to warn about a dependency that looks unsatisfiable.
n_fq=0
while IFS=$'\t' read -r _ _ run _; do
  [ -z "${run:-}" ] && continue
  [ -s "$WORK/fastq/${run}_1.fastq.gz" ] && [ -s "$WORK/fastq/${run}_2.fastq.gz" ] \
    && n_fq=$((n_fq + 1))
done < <(tail -n +2 "$MANIFEST")
deps=()
if [ "$DRY" = 1 ]; then
  deps=( "done($J_DL)" "done(star_index)" )
else
  [ "$n_fq" -lt "$N_SAMPLES" ] && deps+=( "done($J_DL)" )
  [ -s "$REF/star_hg38/SAindex" ] || deps+=( "done(star_index)" )
  for d in ${deps[@]+"${deps[@]}"}; do
    jn=${d#done(}; jn=${jn%)}
    bjobs -J "$jn" >/dev/null 2>&1 || {
      echo "  WARNING: waiting on '$jn' but no such job is queued."
      echo "           Stage 2 would PEND forever -- check stage 1's output above,"
      echo "           then re-run this script once '$jn' is submitted."
    }
  done
fi
echo "  fastq already on disk: $n_fq / $N_SAMPLES"
dep_expr=""
if [ ${#deps[@]} -gt 0 ]; then
  dep_expr=$(printf ' && %s' "${deps[@]}"); dep_expr=${dep_expr:4}
  echo "  stage 2 will wait on: $dep_expr"
else
  echo "  no upstream jobs to wait on (references built, FASTQ already present)"
fi

# ============================================================================
# STAGE 2: one STAR pass per sample -> per-locus featureCounts + TEcount table
# ============================================================================
say "stage 2/6: align + count  ($N_SAMPLES-element array, job '$J_ALIGN')"
run env WORK="$WORK" REF="$REF" CONT="$CONT" MANIFEST="$MANIFEST" \
        JOB="$J_ALIGN" QUEUE="$QUEUE" THREADS="$THREADS" MEM_MB="$MEM_MB" \
        FILTER="$FILTER" ALSO_TECOUNT=1 STRANDED="$STRANDED" FC_STRAND="$FC_STRAND" \
        REQUIRE_INPUTS=0 BSUB_DEP="$dep_expr" \
        bash "$SCRIPTS/submit_locus_expression.sh"

# ============================================================================
# STAGE 3+4: merge per-locus CPM, and summarize LTR66/LTR10G by time point
# ============================================================================
say "stage 3/6 + 4/6: merge per-locus table + TP1-vs-TP2 summary  (job '$J_MERGE')"
MERGE_SH="$WORK/merge_and_summarize.sh"
if [ "$DRY" = 0 ]; then
  cat > "$MERGE_SH" <<EOF
#!/usr/bin/env bash
set -euo pipefail
env WORK="$WORK" MANIFEST="$MANIFEST" FILTER='$FILTER' CONT="$CONT" \\
    bash "$SCRIPTS/submit_locus_expression.sh" --merge
python3 "$SCRIPTS/summarize_te_timepoints.py" \\
    --counts "$WORK/counts" --manifest "$MANIFEST" \\
    --families $FAMILIES --sample-col 1 --group-col 2 \\
    --out "$WORK/te_timepoint_summary.tsv"
echo "locus table : $WORK/locus_expression.tsv"
echo "summary     : $WORK/te_timepoint_summary.tsv"
EOF
  chmod +x "$MERGE_SH"
fi
run bsub -q "$QUEUE" -n 1 -M 8000 -R "rusage[mem=8000]" -W 2:00 \
     -w "done($J_ALIGN)" -J "$J_MERGE" \
     -o "$WORK/${J_MERGE}.%J.log" -e "$WORK/${J_MERGE}.%J.log" \
     bash "$MERGE_SH"

# ============================================================================
# STAGE 5: reagent design -- runs in PARALLEL with stages 2-4 (needs only hg38)
# ============================================================================
say "stage 5/6: qPCR primers + CRISPR guides  (job '$J_REAGENT', parallel)"
run env SIF="$SIF" REF="$REF" GENOME="$GENOME" \
        REAGENT_WORK="$REAGENT_WORK" REAGENT_JOB="$J_REAGENT" \
        FAMILIES="$FAMILIES" CAS="$CAS" QUEUE="$QUEUE" \
        MCS_LTR66="$MCS_LTR66" MCS_LTR10G="$MCS_LTR10G" \
        bash "$SCRIPTS/submit_reagent_design.sh"

# ============================================================================
# STAGE 6: redesign the qPCR pairs weighted by which copies are actually expressed
# ============================================================================
say "stage 6/6: expression-weighted qPCR redesign + top-$TOP_N ranking  (job '$J_QPCR')"
run env SIF="$SIF" REF="$REF" GENOME="$GENOME" REAGENT_WORK="$REAGENT_WORK" \
        FAMILIES="$FAMILIES" NCLUST="$NCLUST" NPAIRS="$NPAIRS" TOP_N="$TOP_N" \
        AMP_MIN="$AMP_MIN" AMP_MAX="$AMP_MAX" MAX_REPS="$MAX_REPS" \
        EXPR_TSV="$WORK/locus_expression.tsv" EXPR_COLS="expression" \
        REAGENT_TAG="$TAG" JOB="$J_QPCR" QUEUE="$QUEUE" \
        BSUB_DEP="done($J_MERGE) && done($J_REAGENT)" \
        bash "$SCRIPTS/rerun_qpcr_multipair.sh"

say "submitted"
cat <<EOF
  pipeline : $J_DL -> $J_ALIGN -> $J_MERGE -> $J_QPCR
             $J_REAGENT ------------------------^
  watch    : bjobs -w        |  bash scripts/run_endometrium_ltr.sh --status
  logs     : $WORK/*.log  and  $REAGENT_WORK/*.log

  when it finishes, the two files to look at first:
    $WORK/te_timepoint_summary.tsv        is LTR66/LTR10G expressed, and does it move TP1->TP2
    $REAGENT_WORK/<family>/<family>_${TAG}_qpcr_pairs.csv   the primers to order
EOF
