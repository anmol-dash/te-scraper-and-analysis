#!/usr/bin/env bash
#
# Submit one ColabFold LSF job per fold-ready person from an existing people_runs tree.
#
# A fold-ready run needs:
#   <run>/01_data/*_with_sequences.csv
# and ideally one of:
#   <run>/cluster_alignments/all_cluster_consensuses.fa
#   <run>/cleaned_consensus/all_clusters_cleaned_consensus.fa
#   <run>/05_consensus/*_consensus.fa
#
# Defaults match the HPC paths used by submit_people_analysis.sh.

set -uo pipefail

BASE_DIR="${BASE_DIR:-/home/amodz/anmol/people_runs}"
REPO_DIR="${REPO_DIR:-/home/amodz/anmol/te-scraper-and-analysis}"
PYTHON="${PYTHON:-python3}"

PEOPLE="${PEOPLE:-Claire Diego Katie}"
COLABFOLD_CMD="${COLABFOLD_CMD:-colabfold_batch}"
SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-}"
SINGULARITY_SOURCE="${SINGULARITY_SOURCE:-$REPO_DIR/colabfold.def}"
USE_MAFFT="${USE_MAFFT:-1}"

QUEUE="${QUEUE:-normal}"
N_CORES="${N_CORES:-8}"
MEM_MB="${MEM_MB:-64000}"
WALL="${WALL:-24:00}"
LSF_RESOURCE="${LSF_RESOURCE:-rusage[mem=$MEM_MB] span[hosts=1]}"

MIN_AA="${MIN_AA:-100}"
TOP_N="${TOP_N:-5}"
SOURCE_SEQS="${SOURCE_SEQS:-100}"
NUM_RECYCLES="${NUM_RECYCLES:-3}"
NUM_MODELS="${NUM_MODELS:-1}"
FORCE="${FORCE:-0}"

JOB_DIR="${JOB_DIR:-$BASE_DIR/_colabfold_jobs}"
SUMMARY="${SUMMARY:-$JOB_DIR/people_colabfold_once_submissions.tsv}"

shopt -s nullglob

if [[ ! -f "$REPO_DIR/run_fold_prediction.py" ]]; then
  echo "ERROR: run_fold_prediction.py not found under REPO_DIR=$REPO_DIR" >&2
  exit 2
fi

mkdir -p "$JOB_DIR"

find_consensus_fasta() {
  local run_dir="$1"
  local fasta=""

  for fasta in \
    "$run_dir/cluster_alignments/all_cluster_consensuses.fa" \
    "$run_dir/cleaned_consensus/all_clusters_cleaned_consensus.fa" \
    "$run_dir/05_consensus/"*"_consensus.fa"
  do
    [[ -s "$fasta" ]] && { printf '%s\n' "$fasta"; return 0; }
  done

  return 1
}

select_run_for_person() {
  local person="$1"
  local person_dir="$BASE_DIR/$person"
  local csv run_dir

  [[ -d "$person_dir" ]] || return 1

  # Prefer runs that reached Stage 11, then any run with sequence data.
  for csv in "$person_dir"/*/*/01_data/*_with_sequences.csv; do
    run_dir="$(cd "$(dirname "$csv")/.." && pwd)"
    [[ -s "$csv" && -f "$run_dir/CHECKPOINT_STAGE11_STANDOUT.txt" ]] && {
      printf '%s\t%s\n' "$run_dir" "$csv"
      return 0
    }
  done

  for csv in "$person_dir"/*/*/01_data/*_with_sequences.csv; do
    run_dir="$(cd "$(dirname "$csv")/.." && pwd)"
    [[ -s "$csv" ]] && {
      printf '%s\t%s\n' "$run_dir" "$csv"
      return 0
    }
  done

  return 1
}

load_fold_modules() {
  # Best-effort module setup. These are harmless on systems without environment modules.
  module load mafft 2>/dev/null || module load MAFFT 2>/dev/null || true
  module load colabfold 2>/dev/null || true
  module load singularity 2>/dev/null || module load apptainer 2>/dev/null || true
}

run_one_person() {
  local person="$1"
  local selected run_dir input_csv family reports_dir consensus_fasta rc
  local cmd

  selected="$(select_run_for_person "$person")"
  if [[ -z "$selected" ]]; then
    echo "[$(date)] $person: missing fold-ready *_with_sequences.csv"
    echo -e "$person\tMISSING_INPUT\t\t\t" > "$JOB_DIR/${person}.status.tsv"
    return 2
  fi

  run_dir="${selected%%$'\t'*}"
  input_csv="${selected#*$'\t'}"
  family="$(basename "$run_dir")"
  reports_dir="$run_dir/reports"

  mkdir -p "$reports_dir"
  load_fold_modules

  cmd=(
    "$PYTHON" -u "$REPO_DIR/run_fold_prediction.py"
    --input "$input_csv"
    --reports-dir "$reports_dir"
    --family "$family"
    --min-aa "$MIN_AA"
    --top-n "$TOP_N"
    --source-seqs "$SOURCE_SEQS"
    --num-recycles "$NUM_RECYCLES"
    --num-models "$NUM_MODELS"
  )

  if consensus_fasta="$(find_consensus_fasta "$run_dir")"; then
    cmd+=(--consensus-fasta "$consensus_fasta" --per-cluster)
  else
    cmd+=(--per-cluster)
  fi

  [[ -n "$COLABFOLD_CMD" ]] && cmd+=(--colabfold-cmd "$COLABFOLD_CMD")
  [[ "$USE_MAFFT" == "1" ]] && cmd+=(--use-mafft)
  [[ "$FORCE" == "1" ]] && cmd+=(--force)
  if [[ -n "$SINGULARITY_IMAGE" ]]; then
    cmd+=(--singularity-image "$SINGULARITY_IMAGE" --singularity-source "$SINGULARITY_SOURCE")
  fi

  echo "[$(date)] $person: folding $family"
  echo "  input:   $input_csv"
  echo "  reports: $reports_dir"

  (
    cd "$REPO_DIR" || exit 2
    "${cmd[@]}"
  )
  rc=$?

  if [[ "$rc" -eq 0 ]]; then
    echo -e "$person\tOK\t$run_dir\t$input_csv\t$reports_dir" > "$JOB_DIR/${person}.status.tsv"
  else
    echo "[$(date)] $person: FAILED rc=$rc" >&2
    echo -e "$person\tFAILED_$rc\t$run_dir\t$input_csv\t$reports_dir" > "$JOB_DIR/${person}.status.tsv"
  fi

  return "$rc"
}

submit_person_job() {
  local person="$1"
  local selected run_dir input_csv family reports_dir job_script job_out job_err submit_out job_id

  selected="$(select_run_for_person "$person")"
  if [[ -z "$selected" ]]; then
    echo "[$(date)] $person: missing fold-ready *_with_sequences.csv"
    echo -e "$person\tMISSING_INPUT\t\t\t\t\t" >> "$SUMMARY"
    return 2
  fi

  run_dir="${selected%%$'\t'*}"
  input_csv="${selected#*$'\t'}"
  family="$(basename "$run_dir")"
  reports_dir="$run_dir/reports"
  job_script="$JOB_DIR/${person}_colabfold.sh"
  job_out="$JOB_DIR/${person}_colabfold.%J.out"
  job_err="$JOB_DIR/${person}_colabfold.%J.err"

  {
    echo "#!/usr/bin/env bash"
    echo "set -uo pipefail"
    printf 'export BASE_DIR=%q\n' "$BASE_DIR"
    printf 'export REPO_DIR=%q\n' "$REPO_DIR"
    printf 'export PYTHON=%q\n' "$PYTHON"
    printf 'export COLABFOLD_CMD=%q\n' "$COLABFOLD_CMD"
    printf 'export SINGULARITY_IMAGE=%q\n' "$SINGULARITY_IMAGE"
    printf 'export SINGULARITY_SOURCE=%q\n' "$SINGULARITY_SOURCE"
    printf 'export USE_MAFFT=%q\n' "$USE_MAFFT"
    printf 'export MIN_AA=%q\n' "$MIN_AA"
    printf 'export TOP_N=%q\n' "$TOP_N"
    printf 'export SOURCE_SEQS=%q\n' "$SOURCE_SEQS"
    printf 'export NUM_RECYCLES=%q\n' "$NUM_RECYCLES"
    printf 'export NUM_MODELS=%q\n' "$NUM_MODELS"
    printf 'export FORCE=%q\n' "$FORCE"
    printf 'export JOB_DIR=%q\n' "$JOB_DIR"
    printf 'cd %q\n' "$REPO_DIR"
    printf 'exec bash %q --run-one %q\n' "$REPO_DIR/run_people_colabfold_once.sh" "$person"
  } > "$job_script"
  chmod +x "$job_script"

  submit_out="$(
    bsub -J "cf_${person}" \
      -q "$QUEUE" \
      -n "$N_CORES" \
      -M "$MEM_MB" \
      -R "$LSF_RESOURCE" \
      -W "$WALL" \
      -o "$job_out" \
      -e "$job_err" \
      bash "$job_script"
  )"
  rc=$?
  if [[ "$rc" -ne 0 ]]; then
    echo "$submit_out" >&2
    echo -e "$person\tSUBMIT_FAILED_$rc\t\t$run_dir\t$input_csv\t$reports_dir\t$job_out\t$job_err" >> "$SUMMARY"
    return "$rc"
  fi

  if [[ "$submit_out" =~ \<([0-9]+)\> ]]; then
    job_id="${BASH_REMATCH[1]}"
  else
    job_id="$submit_out"
  fi

  echo "$submit_out"
  echo -e "$person\tSUBMITTED\t$job_id\t$run_dir\t$input_csv\t$reports_dir\t$job_out\t$job_err" >> "$SUMMARY"
}

submit_all_jobs() {
  local person submitted=0 missing=0 rc=0

  echo -e "person\tstatus\tjob_id\trun_dir\tinput_csv\treports_dir\tstdout\tstderr" > "$SUMMARY"
  echo "Submitting one ColabFold job per fold-ready person"
  echo "  Base dir: $BASE_DIR"
  echo "  Repo dir: $REPO_DIR"
  echo "  Job dir:  $JOB_DIR"
  echo "  LSF:      -q $QUEUE -n $N_CORES -M $MEM_MB -R \"$LSF_RESOURCE\" -W $WALL"
  echo

  for person in $PEOPLE; do
    if submit_person_job "$person"; then
      submitted=$((submitted + 1))
    else
      rc=$?
      if [[ "$rc" -eq 2 ]]; then
        missing=$((missing + 1))
      else
        return "$rc"
      fi
    fi
  done

  echo
  echo "Submitted jobs: $submitted"
  echo "Missing inputs: $missing"
  echo "Summary: $SUMMARY"

  [[ "$submitted" -gt 0 ]]
}

case "${1:-}" in
  --run-one)
    if [[ -z "${2:-}" ]]; then
      echo "ERROR: --run-one requires a person name" >&2
      exit 2
    fi
    run_one_person "$2"
    ;;
  ""|--submit)
    submit_all_jobs
    ;;
  *)
    echo "Usage: $0 [--submit | --run-one PERSON]" >&2
    exit 2
    ;;
esac
