#!/usr/bin/env bash
#
# submit_scaling_test.sh — end-to-end GAMECA test that runs the WHOLE pipeline
# from scratch (UCSC pull → clustering → alignment → motif/TFBS → expression →
# Stage-11 → LaTeX report) for a family, at several input sizes, and reports how
# long each stage takes — and how much memory it needs — as a function of size.
#
# You only give it a family name. It sweeps --max-loci over SIZES (default
# 100/500/1000/2000/all), runs the full pipeline from scratch for each, reads the
# per-stage timings and peak memory that query.py writes to stage_times.json, and
# prints + saves a consolidated timing + memory table (scaling_timings.csv),
# including a suggested LSF -M value per size.
#
# Usage:
#   chmod +x submit_scaling_test.sh
#   FAMILY=MT2_Mm ASSEMBLY=mm10 ./submit_scaling_test.sh            # batch
#   FAMILY=L1HS   ASSEMBLY=hg38 ./submit_scaling_test.sh -Is        # interactive
#   SIZES="200 1000 5000 all" FAMILY=MT2_Mm ./submit_scaling_test.sh
#   # cross-family sweep (each pair run at full size):
#   FAMILIES="AluY:hg38,MT2_Mm:mm10,L1HS:hg38" ./submit_scaling_test.sh
#
# NB: from-scratch runs fetch loci + sequences from UCSC. Compute nodes often
# need an HTTPS proxy (forwarded below if your login shell sets one) or use -Is
# to run on an internet-capable interactive node.

set -uo pipefail

# ── Config ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAMILY="${FAMILY:-MT2_Mm}"
ASSEMBLY="${ASSEMBLY:-mm10}"
SIZES="${SIZES:-100 500 1000 2000 all}"    # --max-loci caps; "all"/"0" = no cap
FAMILIES="${FAMILIES:-}"                    # optional "fam:asm,fam:asm" cross-family sweep
OUT_DIR="${OUT_DIR:-/home/amodz/anmol/gameca_scaling_test}"
SOURCE="${SOURCE:-rmsk}"                    # rmsk | dfam
RMSK_DIR="${RMSK_DIR:-}"                    # optional pre-downloaded rmsk_<build>.txt.gz dir
PYTHON="${PYTHON:-python3}"
QUEUE="${QUEUE:-normal}"
N_CORES="${N_CORES:-4}"
MEM_MB="${MEM_MB:-24000}"
WALL="${WALL:-24:00}"                       # HH:MM — whole sweep runs in one job

mkdir -p "$OUT_DIR"

# Forward an HTTPS proxy into the job so the UCSC fetch works from compute nodes.
PROXY_PREFIX=""
if [ -n "${https_proxy:-${HTTPS_PROXY:-}}" ]; then
  P="${https_proxy:-$HTTPS_PROXY}"
  PROXY_PREFIX="export https_proxy='$P' http_proxy='$P' HTTPS_PROXY='$P' HTTP_PROXY='$P'; "
  echo "Forwarding HTTPS proxy into job: $P"
fi

# ── Build the job list: either a size sweep on one family, or one full run per
#    family in FAMILIES. Each entry is "family assembly maxloci tag".
RUNS=()
if [ -n "$FAMILIES" ]; then
  IFS=',' read -ra PAIRS <<< "$FAMILIES"
  for pair in "${PAIRS[@]}"; do
    fam="${pair%%:*}"; asm="${pair##*:}"
    RUNS+=("$fam $asm all ${fam}_${asm}_all")
  done
else
  for sz in $SIZES; do
    RUNS+=("$FAMILY $ASSEMBLY $sz ${FAMILY}_${ASSEMBLY}_${sz}")
  done
fi

# ── Generate the job script ────────────────────────────────────────────────────
JOB_SH="$OUT_DIR/_run_scaling_test.sh"
{
  echo "#!/usr/bin/env bash"
  echo "set -uo pipefail"
  echo "export CONDA_SOLVER=classic"          # silence non-fatal libmamba/ICU error
  printf 'cd %q\n' "$SCRIPT_DIR"
  [ -n "$PROXY_PREFIX" ] && echo "$PROXY_PREFIX"
  echo 'echo "Host: $(hostname)   Started: $(date)"'
  printf 'OUT_DIR=%q\n' "$OUT_DIR"
  for entry in "${RUNS[@]}"; do
    set -- $entry; fam="$1"; asm="$2"; sz="$3"; tag="$4"
    run_dir="$OUT_DIR/run_$tag"
    cap=""
    [ "$sz" != "all" ] && [ "$sz" != "0" ] && cap="--max-loci $sz"
    src="--source $SOURCE"
    [ -n "$RMSK_DIR" ] && src="$src --rmsk-dir $RMSK_DIR"
    echo "echo; echo '==================== $tag ===================='"
    echo "SECONDS=0"
    echo "$PYTHON query.py --local --family $fam --assembly $asm --output $(printf '%q' "$run_dir") $cap $src || echo \"  RUN FAILED: $tag (rc=\$?)\""
    # Report is part of 'all the tests'; PDF is optional (graceful if no TeX).
    echo "$PYTHON generate_latex_report.py $(printf '%q' "$run_dir") || echo \"  report build skipped/failed: $tag\""
    echo "echo \"  wall for $tag: \${SECONDS}s\""
  done

  # ── Aggregate every stage_times.json into one table ──────────────────────────
  cat <<'PY'
echo; echo "==================== TIMING SUMMARY ===================="
python3 - "$OUT_DIR" <<'PYEOF'
import json, sys, glob, os, csv
root = sys.argv[1]
files = sorted(glob.glob(os.path.join(root, "run_*", "stage_times.json")))
if not files:
    print("No stage_times.json found — did any run complete?"); sys.exit(0)
recs, stages = [], []
for f in files:
    try: d = json.load(open(f))
    except Exception as e: print(f"  skip {f}: {e}"); continue
    recs.append(d)
    for s in d.get("stage_seconds", {}):
        if s not in stages: stages.append(s)
cols = ["family", "assembly", "n_sequences", "n_clusters", "max_loci",
        "total_seconds", "peak_rss_mb"] + stages
out_csv = os.path.join(root, "scaling_timings.csv")
with open(out_csv, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(cols)
    for d in recs:
        w.writerow([d.get("family"), d.get("assembly"), d.get("n_sequences"),
                    d.get("n_clusters"), d.get("max_loci"), d.get("total_seconds"),
                    d.get("peak_rss_mb")]
                   + [d.get("stage_seconds", {}).get(s, "") for s in stages])
# pretty print: sort by n_sequences so the scaling trend is visible
recs.sort(key=lambda d: (d.get("n_sequences") or 0))
print("### TIMING (seconds) ###")
hdr = f'{"family":<12}{"n_seq":>7}{"clust":>6}{"total_s":>9}   ' + "".join(f"{s[:11]:>12}" for s in stages)
print(hdr); print("-" * len(hdr))
for d in recs:
    row = f'{str(d.get("family"))[:11]:<12}{d.get("n_sequences") or 0:>7}{d.get("n_clusters") or 0:>6}{d.get("total_seconds") or 0:>9.1f}   '
    row += "".join(f'{d.get("stage_seconds",{}).get(s,0):>12.1f}' for s in stages)
    print(row)

# ── Memory requirements section ──────────────────────────────────────────────
print("\n### MEMORY REQUIREMENTS ###")
print("Peak resident memory per run, and a suggested LSF -M (peak x 1.3 headroom).")
mhdr = f'{"family":<12}{"n_seq":>7}{"peak_MB":>10}{"peak_GB":>9}{"suggest_-M(MB)":>16}'
print(mhdr); print("-" * len(mhdr))
have_mem = False
for d in recs:
    pk = d.get("peak_rss_mb")
    if pk is None:
        print(f'{str(d.get("family"))[:11]:<12}{d.get("n_sequences") or 0:>7}{"n/a":>10}{"n/a":>9}{"n/a":>16}')
        continue
    have_mem = True
    sug = int(-(-pk * 1.3 // 100) * 100)   # round up to nearest 100 MB
    print(f'{str(d.get("family"))[:11]:<12}{d.get("n_sequences") or 0:>7}{pk:>10.1f}{pk/1024:>9.2f}{sug:>16}')
if have_mem:
    worst = max((d["peak_rss_mb"] for d in recs if d.get("peak_rss_mb")), default=0)
    print(f'\nProvision at least {int(-(-worst*1.3//100)*100)} MB '
          f'(-M {int(-(-worst*1.3//100)*100)}) for the largest size tested.')
else:
    print("peak_rss_mb unavailable (resource module missing?) — memory not captured.")
print(f"\nSaved: {out_csv}")
PYEOF
PY
  echo 'echo "Finished: $(date)"'
} > "$JOB_SH"
chmod +x "$JOB_SH"

# ── Submit ─────────────────────────────────────────────────────────────────────
BSUB_OUT="$OUT_DIR/scaling_test.%J.out"
BSUB_ERR="$OUT_DIR/scaling_test.%J.err"
BSUB_ARGS=(-J "gameca_scaling"
           -q "$QUEUE"
           -n "$N_CORES"
           -M "$MEM_MB" -R "rusage[mem=$MEM_MB]"
           -W "$WALL"
           -o "$BSUB_OUT" -e "$BSUB_ERR")

echo "============================================================"
echo "GAMECA scaling / timing test submission"
if [ -n "$FAMILIES" ]; then
  echo "  Families: $FAMILIES  (each at full size)"
else
  echo "  Family:   $FAMILY ($ASSEMBLY)   Sizes: $SIZES"
fi
echo "  Out dir:  $OUT_DIR"
echo "  Logs:     scaling_test.<JOBID>.out / .err"
echo "  Table:    $OUT_DIR/scaling_timings.csv (written at end)"
echo "  Resources: -q $QUEUE -n $N_CORES -M $MEM_MB -W $WALL"
echo "  Job script: $JOB_SH"
echo "============================================================"

if [ "${1:-}" = "-Is" ]; then
  echo "Submitting INTERACTIVE (streams output; internet-capable node)…"
  exec bsub "${BSUB_ARGS[@]}" -Is bash "$JOB_SH"
else
  echo "Submitting BATCH job…"
  bsub "${BSUB_ARGS[@]}" bash "$JOB_SH"
  echo
  echo "Track with:  bjobs -J gameca_scaling"
  echo "Stdout:      $OUT_DIR/scaling_test.<JOBID>.out   (timing table at the end)"
  echo "Stderr:      $OUT_DIR/scaling_test.<JOBID>.err"
fi
