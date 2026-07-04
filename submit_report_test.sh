#!/usr/bin/env bash
#
# submit_report_test.sh — validate the report changes (expression chapter,
# site-vs-insertion counts, per-motif site-multiplicity table + JASPAR matrix
# IDs) on an EXISTING MT2_Mm analysis folder, as a single LSF job.
#
# It resumes just the TFBS stage (regenerates 05_tfbs/, including the new
# motif_site_multiplicity.csv) and then rebuilds the LaTeX/PDF report, and
# finally asserts that the new artifacts are actually present. Cheap: it reuses
# the existing motif_analysis/ and expression_plots/ — no genome re-read, no
# UCSC fetch, no clustering.
#
# Usage:
#   chmod +x submit_report_test.sh
#   FAMILY_DIR=/abs/path/to/MT2_Mm_results ./submit_report_test.sh        # batch
#   FAMILY_DIR=/abs/path/to/MT2_Mm_results ./submit_report_test.sh -Is    # interactive
#
# The existing folder must already contain motif_analysis/all_overlaps.tsv
# (from a prior full run) and, for the expression chapter, expression_plots/.

set -uo pipefail

# ── Config ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAMILY="${FAMILY:-MT2_Mm}"
ASSEMBLY="${ASSEMBLY:-mm10}"
FAMILY_DIR="${FAMILY_DIR:?Set FAMILY_DIR to the existing MT2_Mm results folder}"
OUT_DIR="${OUT_DIR:-$FAMILY_DIR/report_test_logs}"    # where bsub logs land
PYTHON="${PYTHON:-python3}"
QUEUE="${QUEUE:-normal}"
N_CORES="${N_CORES:-4}"
MEM_MB="${MEM_MB:-16000}"
WALL="${WALL:-04:00}"                                 # HH:MM

if [ ! -d "$FAMILY_DIR" ]; then
  echo "ERROR: FAMILY_DIR does not exist: $FAMILY_DIR" >&2
  exit 2
fi
mkdir -p "$OUT_DIR"

# ── Build the job script ───────────────────────────────────────────────────────
JOB_SH="$OUT_DIR/_run_report_test.sh"
{
  echo "#!/usr/bin/env bash"
  echo "set -uo pipefail"
  echo "export CONDA_SOLVER=classic"          # silence non-fatal libmamba/ICU error
  printf 'cd %q\n' "$SCRIPT_DIR"
  echo 'echo "Host: $(hostname)   Started: $(date)"'
  echo 'rc=0'
  # 1. Regenerate 05_tfbs (incl. motif_site_multiplicity.csv) from real overlaps.
  printf '%s\n' "$PYTHON query.py --local --family $FAMILY --assembly $ASSEMBLY --output $(printf '%q' "$FAMILY_DIR") --resume-from tfbs || rc=1"
  # 2. Rebuild the LaTeX/PDF report (writes report.tex + report.pdf under reports/).
  printf '%s\n' "$PYTHON generate_latex_report.py $(printf '%q' "$FAMILY_DIR") || rc=1"
  # 3. Assert the new artifacts exist — makes PASS/FAIL obvious in the .out file.
  cat <<'CHECK'
echo "──────── verifying new report artifacts ────────"
mult="$FAMILY_DIR_ESC/05_tfbs/motif_site_multiplicity.csv"
tex=$(find "$FAMILY_DIR_ESC" -name report.tex | head -1)
if [ -s "$mult" ]; then echo "PASS: motif_site_multiplicity.csv present ($(wc -l < "$mult") rows)"; else echo "FAIL: motif_site_multiplicity.csv missing"; rc=1; fi
if [ -n "$tex" ]; then
  grep -q "Gene Expression Analysis" "$tex" && echo "PASS: expression chapter in report" || { echo "FAIL: expression chapter missing"; rc=1; }
  grep -q "n_loci_multisite" "$tex" && echo "PASS: site-multiplicity table in report" || { echo "FAIL: multiplicity table missing"; rc=1; }
else
  echo "FAIL: report.tex not found"; rc=1
fi
CHECK
  echo 'echo "Finished: $(date)   exit=$rc"'
  echo 'exit $rc'
} > "$JOB_SH"
# Inject the resolved FAMILY_DIR into the heredoc'd check block.
sed -i.bak "s#\$FAMILY_DIR_ESC#$FAMILY_DIR#g" "$JOB_SH" && rm -f "$JOB_SH.bak"
chmod +x "$JOB_SH"

# ── Submit ─────────────────────────────────────────────────────────────────────
BSUB_OUT="$OUT_DIR/report_test.%J.out"
BSUB_ERR="$OUT_DIR/report_test.%J.err"
BSUB_ARGS=(-J "report_test"
           -q "$QUEUE"
           -n "$N_CORES"
           -M "$MEM_MB" -R "rusage[mem=$MEM_MB]"
           -W "$WALL"
           -o "$BSUB_OUT" -e "$BSUB_ERR")

echo "============================================================"
echo "GAMECA report-change test submission"
echo "  Family dir: $FAMILY_DIR"
echo "  Logs:       $OUT_DIR   (report_test.<JOBID>.out / .err)"
echo "  Resources:  -q $QUEUE -n $N_CORES -M $MEM_MB -W $WALL"
echo "  Job script: $JOB_SH"
echo "============================================================"

if [ "${1:-}" = "-Is" ]; then
  echo "Submitting INTERACTIVE (streams output)…"
  exec bsub "${BSUB_ARGS[@]}" -Is bash "$JOB_SH"
else
  echo "Submitting BATCH job…"
  bsub "${BSUB_ARGS[@]}" bash "$JOB_SH"
  echo
  echo "Track with:  bjobs -J report_test"
  echo "Stdout:      $OUT_DIR/report_test.<JOBID>.out"
  echo "Stderr:      $OUT_DIR/report_test.<JOBID>.err"
fi
