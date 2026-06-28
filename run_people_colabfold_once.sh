#!/usr/bin/env bash
#
# Run ColabFold once per person from an existing people_runs tree.
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

PEOPLE="${PEOPLE:-Claire Diego Katie Zach}"
COLABFOLD_CMD="${COLABFOLD_CMD:-colabfold_batch}"
SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-}"
SINGULARITY_SOURCE="${SINGULARITY_SOURCE:-$REPO_DIR/colabfold.def}"
USE_MAFFT="${USE_MAFFT:-1}"

MIN_AA="${MIN_AA:-100}"
TOP_N="${TOP_N:-5}"
SOURCE_SEQS="${SOURCE_SEQS:-100}"
NUM_RECYCLES="${NUM_RECYCLES:-3}"
NUM_MODELS="${NUM_MODELS:-1}"
FORCE="${FORCE:-0}"

SUMMARY="${SUMMARY:-$BASE_DIR/people_colabfold_once_summary.tsv}"

shopt -s nullglob

if [[ ! -f "$REPO_DIR/run_fold_prediction.py" ]]; then
  echo "ERROR: run_fold_prediction.py not found under REPO_DIR=$REPO_DIR" >&2
  exit 2
fi

mkdir -p "$(dirname "$SUMMARY")"
echo -e "person\tstatus\trun_dir\tinput_csv\treports_dir\tlog" > "$SUMMARY"

# Best-effort module setup. These are harmless on systems without environment modules.
module load mafft 2>/dev/null || module load MAFFT 2>/dev/null || true
module load colabfold 2>/dev/null || true
module load singularity 2>/dev/null || module load apptainer 2>/dev/null || true

failures=0
missing=0

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

for person in $PEOPLE; do
  selected="$(select_run_for_person "$person")"
  if [[ -z "$selected" ]]; then
    echo "[$(date)] $person: missing fold-ready *_with_sequences.csv"
    echo -e "$person\tMISSING_INPUT\t\t\t\t" >> "$SUMMARY"
    missing=$((missing + 1))
    continue
  fi

  run_dir="${selected%%$'\t'*}"
  input_csv="${selected#*$'\t'}"
  family="$(basename "$run_dir")"
  reports_dir="$run_dir/reports"
  log="$run_dir/colabfold_once.log"

  mkdir -p "$reports_dir"

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
  echo "  log:     $log"

  (
    cd "$REPO_DIR" || exit 2
    "${cmd[@]}"
  ) > "$log" 2>&1
  rc=$?

  if [[ "$rc" -eq 0 ]]; then
    echo -e "$person\tOK\t$run_dir\t$input_csv\t$reports_dir\t$log" >> "$SUMMARY"
  else
    echo "[$(date)] $person: FAILED rc=$rc; see $log" >&2
    echo -e "$person\tFAILED_$rc\t$run_dir\t$input_csv\t$reports_dir\t$log" >> "$SUMMARY"
    failures=$((failures + 1))
  fi
done

echo
echo "Summary: $SUMMARY"

if [[ "$failures" -gt 0 ]]; then
  exit 1
fi
if [[ "$missing" -gt 0 ]]; then
  exit 2
fi
