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
#   bash scripts/run_endometrium_ltr.sh --collect    # tar up the results to scp home
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
# gRNA. run_grna_offtarget.py scores coverage WITHOUT requiring a PAM, which
# overstates what is actually cuttable (measured Aug 2026 on the ordered HERVK
# guides: two of them have no NGG in any copy). Stage 7 therefore re-scores the
# shortlist with run_grna_combos.py, which requires a PAM and also checks the
# whole genome for gene hits. GRNA_LEN=20 is now every tool's default too
# (they used to disagree 18/20/20); kept explicit so the study is self-documenting.
GRNA_LEN=${GRNA_LEN:-20}
TOP_GUIDES=${TOP_GUIDES:-12}   # per family; combos enumerates 2^N subsets
COV_MM=${COV_MM:-2}            # on-family coverage: PAM-distal mismatches allowed
OT_MM=${OT_MM:-3}              # off-target scan: mismatches anywhere
TARGET_COV=${TARGET_COV:-0.5}
RMSK_DIR=${RMSK_DIR:-$HOME/te_analysis/rmsk}
COMBO_OUT="$REAGENT_WORK/combos"

# per-study LSF job names so this never collides with the HERVK jobs
J_DL=${J_DL:-endo_dl}
J_ALIGN=${J_ALIGN:-endo_align}
J_MERGE=${J_MERGE:-endo_merge}
J_REAGENT=${J_REAGENT:-endo_reagents}
J_QPCR=${J_QPCR:-endo_qpcr}
J_COMBO=${J_COMBO:-endo_combos}

DRY=0; STATUS_ONLY=0; CLEAN=0; GO=0; COLLECT=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    --status)  STATUS_ONLY=1 ;;
    --collect) COLLECT=1 ;;    # bundle results into one tarball to scp
    --clean)   CLEAN=1 ;;      # bkill this study's jobs (then exit unless --go)
    --go)      GO=1 ;;         # with --clean: clean and submit in one step
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
  bjobs -w 2>/dev/null | grep -E "$J_DL|$J_ALIGN|$J_MERGE|$J_REAGENT|$J_QPCR|$J_COMBO|star_index" \
    || echo "  (no matching jobs in the queue)"
  # A job PENDing on a dependency that can never be met looks exactly like a job
  # waiting for a busy queue, so spell the difference out.
  # NB: no pipe into `grep -q` here -- grep exits on first match, bjobs takes
  # SIGPIPE, and `set -o pipefail` then reports the whole test as false.
  local pend; pend=$(bjobs -p 2>/dev/null || true)
  case "$pend" in *"never satisfied"*)
    echo
    echo "  ** Some jobs are PEND on a dependency that can NEVER be satisfied."
    echo "     An upstream job failed (done() requires exit 0). These will wait forever."
    echo "     Fix the upstream failure, then:  bash $0 --clean --go"
    ;;
  esac
  # name the runs whose FASTQ are incomplete -- that is what fails the download job
  local missing=""
  while IFS=$'\t' read -r s _ run _; do
    [ -z "${run:-}" ] && continue
    { [ -s "$WORK/fastq/${run}_1.fastq.gz" ] && [ -s "$WORK/fastq/${run}_2.fastq.gz" ]; } \
      || missing="$missing $s($run)"
  done < <(tail -n +2 "$MANIFEST")
  [ -n "$missing" ] && { echo; echo "  incomplete FASTQ:$missing"; }
}

if [ "$STATUS_ONLY" = 1 ]; then show_status; exit 0; fi

# --- collect: bundle the deliverables into one tarball for scp ---------------
# Results live in two trees ($WORK and $REAGENT_WORK) next to ~70 GB of FASTQ and
# BAMs. This gathers only the small result files -- a few MB -- so pulling them
# is one scp instead of a careful rsync with the right excludes.
if [ "$COLLECT" = 1 ]; then
  say "collect"
  stamp=$(date +%Y%m%d)
  bundle="$HOME/endometrium_results_${stamp}"
  rm -rf "$bundle"; mkdir -p "$bundle/expression" "$bundle/reagents"
  n=0
  add() {  # add() <src> <destdir>
    [ -e "$1" ] || return 0
    cp -R "$1" "$2/" 2>/dev/null && n=$((n + 1))
  }
  # expression side
  add "$WORK/te_timepoint_summary.tsv" "$bundle/expression"
  add "$WORK/locus_expression.tsv"     "$bundle/expression"
  for f in "$WORK"/counts/*.cntTable;            do add "$f" "$bundle/expression"; done
  for f in "$WORK"/locus/*.featureCounts.txt.summary; do add "$f" "$bundle/expression"; done
  # reagent side: CSVs, reports and the combos answer -- never the genome/BAMs
  for fam in $FAMILIES; do
    [ -d "$REAGENT_WORK/$fam" ] || continue
    mkdir -p "$bundle/reagents/$fam"
    for f in "$REAGENT_WORK/$fam"/*.csv "$REAGENT_WORK/$fam"/*.txt; do
      add "$f" "$bundle/reagents/$fam"
    done
  done
  [ -d "$COMBO_OUT" ] && { mkdir -p "$bundle/reagents/combos"
    for f in "$COMBO_OUT"/*.csv "$COMBO_OUT"/*.txt; do add "$f" "$bundle/reagents/combos"; done; }
  # a short provenance note so the numbers are interpretable later
  {
    echo "SRP090091 / PRJNA342633 -- LTR66 + LTR10G in human endometrium"
    echo "collected: $(date)  host: $(hostname -s)"
    echo "manifest : $MANIFEST ($N_SAMPLES paired total-RNA runs, 7 TP1 vs 7 TP2)"
    echo "families : $FAMILIES   min_cluster_size: LTR66=$MCS_LTR66 LTR10G=$MCS_LTR10G"
    echo "strand   : TEcount=$STRANDED featureCounts=-s $FC_STRAND (library prep unrecorded in SRA)"
    echo "guides   : $CAS, spacer ${GRNA_LEN} nt; coverage in combos/ requires a PAM,"
    echo "           the Pareto numbers in <fam>/grna_offtarget_report.txt do NOT."
    echo "caveat   : expression is from endometrial TISSUE; the study's stromal-cell"
    echo "           samples are small-RNA only and cannot quantify these LTRs."
    echo "git      : $(cd "$REPO" && git rev-parse --short HEAD 2>/dev/null || echo n/a)"
  } > "$bundle/README.txt"
  tar czf "${bundle}.tar.gz" -C "$(dirname "$bundle")" "$(basename "$bundle")"
  rm -rf "$bundle"
  echo "  $n files -> ${bundle}.tar.gz  ($(du -h "${bundle}.tar.gz" | cut -f1))"
  echo
  echo "  Pull it from your laptop:"
  echo "      scp $(whoami)@$(hostname -f 2>/dev/null || hostname):${bundle}.tar.gz ."
  echo "      tar xzf $(basename "${bundle}.tar.gz")"
  exit 0
fi

# --- our jobs already in the queue? ------------------------------------------
# Re-submitting while the previous attempt is still queued creates TWO jobs with
# the same name, and `done(<name>)` then means whichever LSF picks -- so the new
# chain can wait on the old, already-doomed job. Refuse, or clean up on request.
# Print the NAMES of our jobs that are currently queued. Uses `bjobs -J <name>`
# per name: portable across LSF versions, and it avoids an awk dynamic regex
# (awk -v strips one backslash level, so "\\[" arrives as "[" and the pattern
# dies as an unterminated character class -- which fails OPEN, silently).
our_jobs() {
  local j out
  for j in "$J_DL" "$J_ALIGN" "$J_MERGE" "$J_REAGENT" "$J_QPCR" "$J_COMBO"; do
    out=$(bjobs -J "$j" 2>/dev/null || true)
    case "$out" in *PEND*|*RUN*|*PSUSP*|*USUSP*) printf '%s\n' "$j" ;; esac
  done
}
if [ "$CLEAN" = 1 ]; then
  say "clean"
  names=$(our_jobs)
  if [ -n "$names" ]; then
    while read -r j; do
      [ -z "$j" ] && continue
      echo "  bkill -J $j"
      bkill -J "$j" 2>&1 | sed 's/^/    /' || true
    done <<< "$names"
    # bkill returns before mbatchd has dropped the jobs; wait for them to clear
    # rather than tripping the duplicate-name guard on our own cleanup.
    for _ in $(seq 1 15); do
      [ -z "$(our_jobs)" ] && break
      sleep 2
    done
    left=$(our_jobs)
    if [ -n "$left" ]; then
      echo "  still queued after 30s: $(echo $left)"
      echo "  (they may be in the process of being killed -- re-run in a minute)"
    else
      echo "  queue is clear"
    fi
  else
    echo "  nothing of ours in the queue"
  fi
fi
if [ "$CLEAN" = 1 ] && [ "$GO" = 0 ]; then
  echo "  cleaned. Re-run without --clean to submit, or use --clean --go next time."
  exit 0
fi
stale=$(our_jobs)
if [ -n "$stale" ] && [ "$DRY" = 0 ]; then
  echo "FATAL: these jobs of ours are still queued:"
  printf '  %s\n' $stale
  echo "  Submitting now would create duplicate job NAMES, and the -w dependencies"
  echo "  would become ambiguous. Clear them first:"
  echo "      bash $0 --clean        # bkill the above, then re-run"
  echo "  or:  bash $0 --clean --go  # clean and submit in one step"
  exit 1
fi

# --- preflight --------------------------------------------------------------
say "preflight"
[ -s "$MANIFEST" ] || { echo "FATAL: manifest not found: $MANIFEST"; exit 1; }
[ -s "$SIF" ] || { echo "FATAL: gameca.sif not found at $SIF (build_sif.sh, or set SIF=)"; exit 1; }
command -v bsub >/dev/null || { echo "FATAL: bsub not on PATH -- run this on the LSF cluster"; exit 1; }
# singularity OR apptainer -- pennhpc compute nodes may only have the latter
# shellcheck source=/dev/null
. "$SCRIPTS/lib_container.sh"
USE_CONTAINER=${USE_CONTAINER:-auto}
TOOLS=${TOOLS:-$HOME/tools}
PYBIN=${PYBIN:-$TOOLS/venv/bin/python}
container_module_load
RUNMODE=""
if [ "$USE_CONTAINER" != "0" ] && container_init 2>/dev/null && container_probe "$SIF"; then
  RUNMODE=container
  echo "  runmode: container${CONTAINER_EXEC_FLAGS:+ ($CONTAINER_EXEC_FLAGS)}"
elif [ -x "$TOOLS/bin/STAR" ] && [ -x "$PYBIN" ]; then
  RUNMODE=host; USE_CONTAINER=0
  echo "  runmode: host tools ($TOOLS) -- container unusable here"
else
  echo "FATAL: neither path is available on this cluster."
  echo "  container: cannot start (apptainer FIPS panic on every node)"
  echo "  host tools: not installed"
  echo
  echo "  Install them once (5 min, no root, no conda):"
  echo "      bash scripts/setup_host_tools.sh"
  exit 1
fi
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

# rmsk + refGene for the stage-7 off-target annotation. Fetched here, on the
# login node, so the scan job never depends on outbound network.
if [ "$DRY" = 0 ]; then
  mkdir -p "$RMSK_DIR"
  { [ "$RUNMODE" = host ] && "$PYBIN" -c "
import sys; sys.path.insert(0, '$REPO')
from te_prep import get_rmsk_path
from run_grna_combos import download_refgene
print('  rmsk:   ', get_rmsk_path('hg38', '$RMSK_DIR'))
print('  refGene:', download_refgene('hg38', '$RMSK_DIR'))
"; } || { [ "$RUNMODE" = container ] && "$GAMECA_RT" exec $CONTAINER_EXEC_FLAGS -B "$HOME" "$SIF" python -c "
import sys; sys.path.insert(0, '$REPO')
from te_prep import get_rmsk_path
from run_grna_combos import download_refgene
print('  rmsk:   ', get_rmsk_path('hg38', '$RMSK_DIR'))
print('  refGene:', download_refgene('hg38', '$RMSK_DIR'))
"; } || echo "  WARNING: annotation prefetch failed; stage 7 will retry on the node"
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
        REQUIRE_INPUTS=0 BSUB_DEP="$dep_expr" EXCLUDE_HOSTS="${EXCLUDE_HOSTS:-}" \
        USE_CONTAINER="$USE_CONTAINER" TOOLS="$TOOLS" \
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
        MCS_LTR66="$MCS_LTR66" MCS_LTR10G="$MCS_LTR10G" GRNA_LEN="$GRNA_LEN" \
        USE_CONTAINER="$USE_CONTAINER" TOOLS="$TOOLS" PYBIN="$PYBIN" \
        bash "$SCRIPTS/submit_reagent_design.sh"

# ============================================================================
# STAGE 6: redesign the qPCR pairs weighted by which copies are actually expressed
# ============================================================================
say "stage 6/6: expression-weighted qPCR redesign + top-$TOP_N ranking  (job '$J_QPCR')"
run env SIF="$SIF" REF="$REF" GENOME="$GENOME" REAGENT_WORK="$REAGENT_WORK" \
        FAMILIES="$FAMILIES" NCLUST="$NCLUST" NPAIRS="$NPAIRS" TOP_N="$TOP_N" \
        AMP_MIN="$AMP_MIN" AMP_MAX="$AMP_MAX" MAX_REPS="$MAX_REPS" \
        USE_CONTAINER="$USE_CONTAINER" TOOLS="$TOOLS" PYBIN="$PYBIN" \
        EXPR_TSV="$WORK/locus_expression.tsv" EXPR_COLS="expression" \
        REAGENT_TAG="$TAG" JOB="$J_QPCR" QUEUE="$QUEUE" \
        BSUB_DEP="done($J_MERGE) && done($J_REAGENT)" \
        bash "$SCRIPTS/rerun_qpcr_multipair.sh"

# ============================================================================
# STAGE 7: PAM-REQUIRED guide verification + genome-wide gene safety
# The Pareto coverage from stage 5 does not require a PAM, so it overstates
# what can actually be cut. Nothing should be ordered on stage 5's numbers.
# ============================================================================
say "stage 7/7: PAM-required guide combinations + gene safety  (job '$J_COMBO')"
COMBO_SH="$WORK/grna_combos.sh"
if [ "$DRY" = 0 ]; then
  mkdir -p "$COMBO_OUT"
  # explicit --seqs per family: the combos tool otherwise guesses at a
  # *_with_sequences.csv name that this layout does not necessarily use
  seqs_args=""
  for fam in $FAMILIES; do
    seqs_args="$seqs_args --family $fam"
  done
  cat > "$COMBO_SH" <<EOF
#!/usr/bin/env bash
set -euo pipefail
. "$SCRIPTS/lib_container.sh"
container_module_load
container_init || exit 1
container_probe "$SIF" || true
sing() { "\$GAMECA_RT" exec \$CONTAINER_EXEC_FLAGS -B "\$HOME" "$SIF" "\$@"; }

python3 "$SCRIPTS/collect_grna_candidates.py" \\
    --reagent-dir "$REAGENT_WORK" --families $FAMILIES \\
    --top-n $TOP_GUIDES --grna-len $GRNA_LEN \\
    --out "$COMBO_OUT/combo_input_guides.csv"

# point each family at the copies CSV the reagent stage actually produced
seqargs=()
for fam in $FAMILIES; do
  csv=\$(find "$REAGENT_WORK/\$fam" -name '*.csv' 2>/dev/null | while read -r f; do
      h=\$(head -1 "\$f" 2>/dev/null)
      echo "\$h" | grep -qiw Seq || continue
      if echo "\$h" | grep -qiw Cluster; then echo "2 \$f"; else echo "1 \$f"; fi
    done | sort -rn | head -1 | cut -d' ' -f2-)
  [ -n "\$csv" ] || { echo "[\$fam] no copies CSV -- skipping"; continue; }
  echo "[\$fam] copies: \$csv"
  seqargs+=( --seqs "\$fam=\$csv" )
done

[ \${#seqargs[@]} -gt 0 ] || { echo "no copies CSV for any family -- aborting"; exit 1; }
sing python "$REPO/run_grna_combos.py" \\
    --guides "$COMBO_OUT/combo_input_guides.csv" \\
    $seqs_args \\
    "\${seqargs[@]}" \\
    --genome "$GENOME" --assembly hg38 --rmsk-dir "$RMSK_DIR" \\
    --cas "$CAS" --grna-len $GRNA_LEN \\
    --cov-mm $COV_MM --ot-mm $OT_MM --target-cov $TARGET_COV \\
    --out "$COMBO_OUT"
echo "ANSWER: $COMBO_OUT/ANSWER.txt"
EOF
  chmod +x "$COMBO_SH"
fi
run bsub -q "$QUEUE" -n 4 -M 24000 -R "rusage[mem=24000]" -W 8:00 \
     -w "done($J_REAGENT)" -J "$J_COMBO" \
     -o "$COMBO_OUT/${J_COMBO}.%J.log" -e "$COMBO_OUT/${J_COMBO}.%J.log" \
     bash "$COMBO_SH"

say "submitted"
cat <<EOF
  pipeline : $J_DL -> $J_ALIGN -> $J_MERGE -> $J_QPCR
             $J_REAGENT ------------------------^
             $J_REAGENT -> $J_COMBO
  watch    : bjobs -w        |  bash scripts/run_endometrium_ltr.sh --status
  logs     : $WORK/*.log  and  $REAGENT_WORK/*.log

  when it finishes, the files to look at first:
    $WORK/te_timepoint_summary.tsv        is LTR66/LTR10G expressed, and does it move TP1->TP2
    $REAGENT_WORK/<family>/<family>_${TAG}_qpcr_pairs.csv   the primers to order
    $COMBO_OUT/ANSWER.txt                 which guide COMBINATION to order (PAM required)

  Do not order guides off stage 5's Pareto numbers -- that coverage does not
  require a PAM. $COMBO_OUT/ANSWER.txt is the one that does.
EOF
