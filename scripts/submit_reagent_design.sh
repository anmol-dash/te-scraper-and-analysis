#!/usr/bin/env bash
# submit_reagent_design.sh
# Generate qPCR primers + CRISPR (a/i/KO) gRNA reagents for HERVK subfamilies,
# using the existing GAMECA scripts, one LSF array element per family.
#
# Per family it runs (all inside gameca.sif, reusing the cluster hg38.fa):
#   1. query.py --stop-after primers  -> family sequences CSV + specific primers
#   2. run_grna_offtarget.py          -> coverage-vs-off-target Pareto guide set
#   3. run_grna_analysis.py           -> PAM-aware greedy guide set (SpCas9)
#   4. te_qpcr_primers.py             -> primer3 fwd/rev qPCR pairs + specificity
#
# Reagent mapping (from the CCLE HERVK expression):
#   LTR5_Hs, LTR5  = 5' LTR / promoter  -> CRISPRa (dCas9-VPR) or CRISPRi (dCas9-KRAB)
#   HERVK-int      = internal/coding    -> knockout guides + qPCR of transcript body
# (same guides serve CRISPRa and CRISPRi; only the dCas9 effector differs.)
#
# Run on login (query.py --local fetches rmsk, needs internet; compute nodes here
# also have internet):  bash scripts/submit_reagent_design.sh
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
SIF=${SIF:-$REPO/gameca.sif}
REF=${REF:-$HOME/hervk_ccle/ref}          # reuse the downloaded hg38.fa
GENOME=${GENOME:-$REF/hg38.fa}
# Dedicated names (NOT generic WORK/JOB) so stale exports from the requant
# pipeline can't redirect reagent outputs or clash with its job-array name.
WORK=${REAGENT_WORK:-$HOME/hervk_reagents}
JOB=${REAGENT_JOB:-reagents}
CAS=${CAS:-SpCas9}
FAMILIES=${FAMILIES:-"LTR5_Hs LTR5 HERVK-int"}   # add LTR5A LTR5B for full variant set
# HDBSCAN min_cluster_size for query.py. Its argparse default is 100, which is
# fine for the ~1000-copy HERVK families but silently collapses a small family
# (everything below 100 becomes one cluster + noise). Set MCS for all families,
# or MCS_<FAMILY> per family (non-alphanumerics -> '_', e.g. MCS_HERVK_int).
MCS=${MCS:-}
# Guide length. The two gRNA scripts disagree by default -- run_grna_analysis.py
# uses 18 and run_grna_offtarget.py 20 -- and run_grna_combos.py assumes 20.
# Empty keeps each tool's own default (existing HERVK behaviour); set GRNA_LEN=20
# for standard SpCas9 spacers that are consistent across all three.
GRNA_LEN=${GRNA_LEN:-}
QUEUE=${QUEUE:-rhel9}
WALL=${WALL:-8:00}
PYLIB="$WORK/pylib"                              # primer3-py installed here if the sif lacks it
mkdir -p "$WORK"
read -r -a FAM_ARR <<< "$FAMILIES"
# The sif binds $HOME, so python inside picks up host ~/.local site-packages
# (numpy 2.4 there breaks the container's numba/umap). Make every container
# python ignore ~/.local; explicit PYTHONPATH (e.g. PYLIB) is still honored.
# shellcheck source=/dev/null
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_container.sh"
container_module_load
container_init || exit 1
# this script's sing() bakes in $SIF (call sites pass only the command), so it
# overrides the generic two-argument helper from the library
sing() { "$GAMECA_RT" exec -B "$HOME" "$SIF" "$@"; }

ensure_primer3() {   # sif has numpy/pandas but not primer3; install into PYLIB once
  sing python -c "import primer3" 2>/dev/null && return 0
  SINGULARITYENV_PYTHONPATH="$PYLIB" sing python -c "import primer3" 2>/dev/null && return 0
  echo "  primer3 not in sif -> installing primer3-py into $PYLIB"
  sing pip install --quiet --target="$PYLIB" primer3-py 2>&1 | tail -2 || true
  SINGULARITYENV_PYTHONPATH="$PYLIB" sing python -c "import primer3; print('  primer3 OK via PYLIB')" \
    2>/dev/null || echo "  primer3-py install FAILED (qPCR step will be skipped)"
}

design_one() {
  local fam="$1"
  local out="$WORK/$fam"       # separate line: $fam must be set before it's used (set -u)
  mkdir -p "$out"
  # per-family MCS_<FAM> wins over the global MCS; unset = query.py's own default
  local famkey mcs mcsarg=()
  famkey="MCS_$(printf '%s' "$fam" | tr -c '[:alnum:]' '_')"
  mcs="${!famkey:-$MCS}"
  [ -n "$mcs" ] && mcsarg=( --min-cluster-size "$mcs" )
  echo "== [$fam] 1/4 query.py primers + sequences ${mcs:+(min_cluster_size=$mcs)} =="
  # ${a[@]+"${a[@]}"}: expanding an empty array unquoted-guarded is the portable
  # form; a bare "${a[@]}" is an 'unbound variable' error on bash < 4.4 + set -u
  sing python "$REPO/query.py" --local --family "$fam" --assembly hg38 \
      --genome "$GENOME" --output "$out" --stop-after primers \
      ${mcsarg[@]+"${mcsarg[@]}"}

  # locate the family sequences CSV (the one with a 'Seq' column)
  local seqcsv
  seqcsv=$(find "$out" -name '*.csv' -print0 | xargs -0 -I{} sh -c \
      'head -1 "{}" | grep -qiw Seq && echo "{}"' | head -1 || true)
  [ -z "$seqcsv" ] && { echo "[$fam] no sequences CSV with a Seq column under $out"; return 1; }
  echo "[$fam] sequences CSV: $seqcsv"

  local lenarg=(); [ -n "$GRNA_LEN" ] && lenarg=( --grna-len "$GRNA_LEN" )

  # NOTE: run_grna_offtarget.py's coverage is PAM-FREE and overstates what is
  # actually cuttable -- treat its numbers as a Pareto screen only, and confirm
  # with run_grna_combos.py (PAM required) before ordering anything.
  echo "== [$fam] 2/4 gRNA off-target/coverage Pareto ($CAS) =="
  sing python "$REPO/run_grna_offtarget.py" --input "$seqcsv" --family "$fam" \
      --cas "$CAS" --reports-dir "$out" ${lenarg[@]+"${lenarg[@]}"} \
      || echo "[$fam] offtarget step failed (see log)"

  echo "== [$fam] 3/4 gRNA greedy set ($CAS) =="
  sing python "$REPO/run_grna_analysis.py" --input "$seqcsv" --family "$fam" \
      --cas "$CAS" --reports-dir "$out" ${lenarg[@]+"${lenarg[@]}"} \
      || echo "[$fam] grna_analysis step failed (see log)"

  echo "== [$fam] 4/4 primer3 qPCR pairs =="
  SINGULARITYENV_PYTHONPATH="$PYLIB" sing python "$REPO/te_qpcr_primers.py" \
      --input "$seqcsv" --family "$fam" --genome "$GENOME" --out "$out" \
      || echo "[$fam] qpcr step failed (see log)"

  # annotate each pair with family coverage (copies_amplified / %family)
  if [ -s "$out/${fam}_qpcr_pairs.csv" ]; then
    sing python "$REPO/te_qpcr_coverage.py" \
        --pairs "$out/${fam}_qpcr_pairs.csv" --seqs "$seqcsv" \
        || echo "[$fam] coverage annotation failed"
  fi

  echo "[$fam] done -> $out"
}

preflight() {
  echo "== preflight =="
  [ -s "$SIF" ]    && echo "  sif OK: $SIF"       || { echo "  MISSING sif: $SIF"; exit 1; }
  [ -s "$GENOME" ] && echo "  genome OK: $GENOME" || { echo "  MISSING genome: $GENOME"; exit 1; }
  echo "  families: ${FAM_ARR[*]}"
  echo "  python deps in sif:"
  for m in numpy pandas sklearn; do
    sing python -c "import $m" 2>/dev/null && echo "    $m: OK" \
      || echo "    $m: MISSING (pipeline dep)"
  done
  ensure_primer3
  echo "  outputs -> $WORK   (job: $JOB)"
}

if [ "${1:-}" = "--run-one" ]; then
  idx="${LSB_JOBINDEX:?--run-one needs an array index}"
  design_one "${FAM_ARR[$((idx-1))]}"; exit 0
fi

preflight
N=${#FAM_ARR[@]}
bsub_args=( -n 4 -M 20000 -R "rusage[mem=20000]" -W "$WALL" -q "$QUEUE" )
# BSUB_DEP: LSF dependency expression, e.g. 'done(other_job)'. Passed to bsub as
# a single -w argument so multi-term expressions survive intact.
dep_args=(); [ -n "${BSUB_DEP:-}" ] && dep_args=( -w "$BSUB_DEP" )
# forward every MCS / MCS_<FAM> so the compute node clusters as configured
mcs_env=( MCS="$MCS" )
for v in $(compgen -v | grep '^MCS_' || true); do mcs_env+=( "$v=${!v}" ); done
LOG="$WORK/${JOB}.%J_%I.log"
bsub "${bsub_args[@]}" ${dep_args[@]+"${dep_args[@]}"} \
     -J "${JOB}[1-$N]" -o "$LOG" -e "$LOG" \
     env SIF="$SIF" REF="$REF" GENOME="$GENOME" REAGENT_WORK="$WORK" REAGENT_JOB="$JOB" \
         CAS="$CAS" FAMILIES="$FAMILIES" GRNA_LEN="$GRNA_LEN" "${mcs_env[@]}" \
     bash "$REPO/scripts/$(basename "$0")" --run-one
echo "submitted ${JOB}[1-$N] for: ${FAM_ARR[*]}"
echo "  watch: bjobs -J $JOB ; logs: $WORK/${JOB}.*_*.log ; outputs under $WORK/<family>/"
