#!/usr/bin/env bash
#
# test_gameca.sh — single entry point for every GAMECA HPC test scenario.
#
# Consolidates what used to be five separate submit_*.sh scripts (cluster,
# clustering-defaults, nextflow, report-change, scaling) plus the stray
# generated _run_cluster_test.sh into one script with a MODE selector.
# run_gameca_cluster_test.py remains a separate file — it's the Python test
# engine MODE=cluster shells out to (analogous to query.py needing its own
# module files), not a duplicate test entry point.
#
# Usage:
#   chmod +x test_gameca.sh
#   ./test_gameca.sh <mode> [-Is]
#
# Modes:
#   cluster     (default) Multi-family/assembly LINE/SINE/LTR × hg38/mm10 matrix
#               test: UCSC pull → cluster → synthetic expression → every Stage 11
#               module, via run_gameca_cluster_test.py.
#   clustering  From-scratch single-family run validating the clustering defaults
#               (kmer=10, UMAP n_neighbors=15/min_dist=0.0, HDBSCAN
#               min_samples=5/min_cluster_size=100): UCSC fetch → clustering →
#               full pipeline → LaTeX report.
#   nextflow    Smoke-test the Nextflow pipeline (stub profile by default; set
#               FAMILY for a real single-family run) — builds/reuses gameca.sif
#               and runs `nextflow run nextflow/main.nf`. Must run on a LOGIN
#               NODE (Nextflow submits child bsub jobs itself).
#   report      Validates report-generation changes on an EXISTING analysis
#               folder (set FAMILY_DIR): resumes the TFBS stage, rebuilds the
#               LaTeX/PDF report, and asserts the new artifacts are present.
#               Cheap — no genome re-read, no UCSC fetch, no clustering.
#   scaling     Runs the whole pipeline from scratch at several --max-loci sizes
#               (or one full run per family in FAMILIES) and reports per-stage
#               timing, peak memory, and disk usage as a function of size.
#
# -Is submits interactively (streams output; needed if only login nodes reach
# UCSC). Omit it for a batch bsub job.
#
# Every mode honors the config env vars documented in its own block below, plus
# these shared ones: QUEUE, N_CORES, MEM_MB, WALL, PYTHON.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-cluster}"
[ "${1:-}" = "-Is" ] && MODE="cluster"   # `./test_gameca.sh -Is` = cluster mode, interactive
INTERACTIVE=0
for a in "$@"; do [ "$a" = "-Is" ] && INTERACTIVE=1; done

QUEUE="${QUEUE:-normal}"
N_CORES="${N_CORES:-4}"
MEM_MB="${MEM_MB:-24000}"
WALL="${WALL:-24:00}"
PYTHON="${PYTHON:-python3}"

# Forward an HTTPS proxy into the job so UCSC fetches work from compute nodes
# that only allow proxied outbound HTTPS (used by cluster/clustering/scaling).
_proxy_prefix() {
  if [ -n "${https_proxy:-${HTTPS_PROXY:-}}" ]; then
    local p="${https_proxy:-$HTTPS_PROXY}"
    echo "Forwarding HTTPS proxy into job: $p" >&2
    printf "export https_proxy=%q http_proxy=%q HTTPS_PROXY=%q HTTP_PROXY=%q; " "$p" "$p" "$p" "$p"
  fi
}

_submit() {
  # _submit <job_name> <mem_mb> <wall> <out_log> <err_log> <job_script...>
  local job_name="$1" mem_mb="$2" wall="$3" out="$4" err="$5"; shift 5
  local bsub_args=(-J "$job_name" -q "$QUEUE" -n "$N_CORES"
                    -M "$mem_mb" -R "rusage[mem=$mem_mb]" -W "$wall"
                    -o "$out" -e "$err")
  if [ "$INTERACTIVE" = "1" ]; then
    echo "Submitting INTERACTIVE (streams output; internet-capable node)…"
    exec bsub "${bsub_args[@]}" -Is "$@"
  else
    echo "Submitting BATCH job…"
    bsub "${bsub_args[@]}" "$@"
    echo
    echo "Track with:  bjobs -J $job_name"
  fi
}

case "$MODE" in

# ─────────────────────────────────────────────────────────────────────────────
cluster)
  # ── Config ───────────────────────────────────────────────────────────────
  OUT_DIR="${OUT_DIR:-/home/amodz/anmol/gamecatestv624}"
  EMAIL="${EMAIL:-anmoldash@gmail.com}"
  MATRIX="${MATRIX:-}"                 # empty → built-in matrix; or "L1HS:hg38,MT2_Mm:mm10"
  MAX_LOCI="${MAX_LOCI:-300}"          # cap loci/family for tractable UCSC + MAFFT
  RMSK_DIR="${RMSK_DIR:-}"             # optional pre-downloaded rmsk_<build>.txt.gz dir
  INCLUDE_FOLD="${INCLUDE_FOLD:-0}"    # 1 → also run ColabFold (heavy/GPU)
  MEM_MB="${MEM_MB:-24000}"            # peak observed ~19 GB (L1HS); 16 GB overran

  mkdir -p "$OUT_DIR"

  PYCMD=("$PYTHON" "$SCRIPT_DIR/run_gameca_cluster_test.py"
         --out-dir "$OUT_DIR" --max-loci "$MAX_LOCI" --notify-email "$EMAIL")
  [ -n "$MATRIX" ]   && PYCMD+=(--matrix "$MATRIX")
  [ -n "$RMSK_DIR" ] && PYCMD+=(--rmsk-dir "$RMSK_DIR")
  [ "$INCLUDE_FOLD" = "1" ] && PYCMD+=(--include-fold)

  PROXY_PREFIX="$(_proxy_prefix)"
  JOB_SH="$OUT_DIR/_run_cluster_test.sh"
  PYCMD_STR="$(printf '%q ' "${PYCMD[@]}")"
  {
    echo "#!/usr/bin/env bash"
    echo "set -uo pipefail"
    echo "export CONDA_SOLVER=classic"   # silence non-fatal libmamba/ICU load error
    printf 'cd %q\n' "$SCRIPT_DIR"
    [ -n "$PROXY_PREFIX" ] && echo "$PROXY_PREFIX"
    echo 'echo "Host: $(hostname)   Started: $(date)"'
    echo "$PYCMD_STR"
    echo 'echo "Finished: $(date)"'
  } > "$JOB_SH"
  chmod +x "$JOB_SH"

  echo "============================================================"
  echo "GAMECA cluster test [mode=cluster]"
  echo "  Out dir:  $OUT_DIR"
  echo "  Matrix:   ${MATRIX:-<built-in LINE/SINE/LTR x hg38/mm10>}"
  echo "  Max loci: $MAX_LOCI   Fold: $INCLUDE_FOLD"
  echo "  Resources: -q $QUEUE -n $N_CORES -M $MEM_MB -W $WALL"
  echo "  Job script: $JOB_SH"
  echo "============================================================"
  _submit "gamecatest" "$MEM_MB" "$WALL" "$OUT_DIR/bsub.%J.out" "$OUT_DIR/bsub.%J.err" bash "$JOB_SH"
  [ "$INTERACTIVE" = "1" ] || {
    echo "Live log:    tail -f $OUT_DIR/gameca_cluster_test.log"
    echo "Summary:     $OUT_DIR/gameca_cluster_test_summary.json"
  }
  ;;

# ─────────────────────────────────────────────────────────────────────────────
clustering)
  # ── Config ───────────────────────────────────────────────────────────────
  FAMILY="${FAMILY:-MT2_Mm}"
  ASSEMBLY="${ASSEMBLY:-mm10}"
  OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/results_clustering_test}"
  WALL="${WALL:-06:00}"

  mkdir -p "$OUT_DIR"
  PROXY_PREFIX="$(_proxy_prefix)"

  JOB_SH="$OUT_DIR/_run_clustering_test.sh"
  {
    echo "#!/usr/bin/env bash"
    echo "set -uo pipefail"
    echo "export CONDA_SOLVER=classic"
    printf 'cd %q\n' "$SCRIPT_DIR"
    [ -n "$PROXY_PREFIX" ] && echo "$PROXY_PREFIX"
    echo 'echo "Host: $(hostname)   Started: $(date)"'
    echo 'rc=0'
    printf '%s\n' "$PYTHON query.py --local --family $FAMILY --assembly $ASSEMBLY --output $(printf '%q' "$OUT_DIR") --force || rc=1"
    cat <<CHECK
echo "──────── verifying clustering output ────────"
coords=\$(find $(printf '%q' "$OUT_DIR") -name clustering_coordinates.csv | head -1)
if [ -s "\$coords" ]; then
  n_clusters=\$(tail -n +2 "\$coords" | cut -d, -f5 | sort -u | wc -l)
  echo "PASS: clustering_coordinates.csv present (\$(( \$(wc -l < "\$coords") - 1 )) rows, \$n_clusters distinct cluster labels)"
else
  echo "FAIL: clustering_coordinates.csv missing"; rc=1
fi
CHECK
    echo 'echo "Finished: $(date)   exit=$rc"'
    echo 'exit $rc'
  } > "$JOB_SH"
  chmod +x "$JOB_SH"

  echo "============================================================"
  echo "GAMECA clustering-defaults test [mode=clustering]"
  echo "  Family:    $FAMILY ($ASSEMBLY)"
  echo "  Out dir:   $OUT_DIR"
  echo "  Resources: -q $QUEUE -n $N_CORES -M $MEM_MB -W $WALL"
  echo "  Job script: $JOB_SH"
  echo "============================================================"
  _submit "gameca_clustering_test" "$MEM_MB" "$WALL" \
          "$OUT_DIR/clustering_test.%J.out" "$OUT_DIR/clustering_test.%J.err" bash "$JOB_SH"
  ;;

# ─────────────────────────────────────────────────────────────────────────────
nextflow)
  # ── Config ───────────────────────────────────────────────────────────────
  # IMPORTANT: Nextflow is the *driver* — it submits child bsub jobs itself, so
  # this mode runs directly on the login node, never wrapped in its own bsub.
  IMAGE="${IMAGE:-ghcr.io/anmol-dash/gameca:latest}"
  SIF="${SIF:-$SCRIPT_DIR/gameca.sif}"
  OUTDIR="${OUTDIR:-$SCRIPT_DIR/nextflow_test_results}"
  WORKDIR="${WORKDIR:-$SCRIPT_DIR/nf_work}"
  FAMILY="${FAMILY:-}"
  ASSEMBLY="${ASSEMBLY:-hg38}"
  GENOME="${GENOME:-}"

  if [ -n "$FAMILY" ]; then
    PROFILE="${PROFILE:-lsf,singularity}"
  else
    PROFILE="${PROFILE:-test,lsf,singularity}"
  fi

  module load nextflow 2>/dev/null || true
  module load java 2>/dev/null || module load openjdk 2>/dev/null || module load Java 2>/dev/null || true
  if ! command -v nextflow >/dev/null 2>&1; then
    echo "[nf-test] nextflow not found — installing to \$HOME/.local/bin ..."
    command -v java >/dev/null 2>&1 || {
      echo "ERROR: Java 11+ required for Nextflow. 'module load java' (or openjdk) first." >&2
      exit 1
    }
    mkdir -p "$HOME/.local/bin"
    ( cd "$HOME/.local/bin" && curl -s https://get.nextflow.io | bash )
    export PATH="$HOME/.local/bin:$PATH"
  fi
  command -v nextflow >/dev/null 2>&1 || { echo "ERROR: nextflow still not on PATH" >&2; exit 1; }
  echo "[nf-test] nextflow: $(command -v nextflow)  ($(nextflow -version 2>&1 | awk '/version/{print $3; exit}'))"

  RUNTIME="$(command -v singularity || command -v apptainer || echo singularity)"
  if [ ! -f "$SIF" ]; then
    echo "[nf-test] Building $SIF from docker://$IMAGE via $RUNTIME ..."
    "$RUNTIME" build "$SIF" "docker://$IMAGE" \
      || { echo "ERROR: failed to build $SIF (is $IMAGE pushed + public?)" >&2; exit 1; }
  else
    echo "[nf-test] Reusing existing $SIF"
  fi

  mkdir -p "$WORKDIR"
  QCFG="$WORKDIR/_queue.config"
  echo "process.queue = '$QUEUE'" > "$QCFG"

  NF_ARGS=(run "$SCRIPT_DIR/nextflow/main.nf" -profile "$PROFILE" -c "$QCFG"
           --container_sif "$SIF" --outdir "$OUTDIR" -work-dir "$WORKDIR" -resume)
  if [ -n "$FAMILY" ]; then
    NF_ARGS+=(--family "$FAMILY" --assembly "$ASSEMBLY")
    [ -n "$GENOME" ] && NF_ARGS+=(--genome_fasta "$GENOME")
  fi

  echo "============================================================"
  echo "GAMECA Nextflow test [mode=nextflow]"
  echo "  Image:   $IMAGE"
  echo "  SIF:     $SIF"
  echo "  Profile: $PROFILE"
  echo "  Mode:    ${FAMILY:+real run — family=$FAMILY assembly=$ASSEMBLY}${FAMILY:-stub smoke test}"
  echo "  Outdir:  $OUTDIR"
  echo "  Queue:   $QUEUE  (child jobs)"
  echo "  Cmd:     nextflow ${NF_ARGS[*]}"
  echo "============================================================"
  exec nextflow "${NF_ARGS[@]}"
  ;;

# ─────────────────────────────────────────────────────────────────────────────
report)
  # ── Config ───────────────────────────────────────────────────────────────
  FAMILY="${FAMILY:-MT2_Mm}"
  ASSEMBLY="${ASSEMBLY:-mm10}"
  FAMILY_DIR="${FAMILY_DIR:?Set FAMILY_DIR to the existing analysis results folder}"
  OUT_DIR="${OUT_DIR:-$FAMILY_DIR/report_test_logs}"
  MEM_MB="${MEM_MB:-16000}"
  WALL="${WALL:-04:00}"

  if [ ! -d "$FAMILY_DIR" ]; then
    echo "ERROR: FAMILY_DIR does not exist: $FAMILY_DIR" >&2
    exit 2
  fi
  mkdir -p "$OUT_DIR"

  JOB_SH="$OUT_DIR/_run_report_test.sh"
  {
    echo "#!/usr/bin/env bash"
    echo "set -uo pipefail"
    echo "export CONDA_SOLVER=classic"
    printf 'cd %q\n' "$SCRIPT_DIR"
    echo 'echo "Host: $(hostname)   Started: $(date)"'
    echo 'rc=0'
    # 1. Regenerate 05_tfbs (incl. motif_site_multiplicity.csv) from real overlaps.
    printf '%s\n' "$PYTHON query.py --local --family $FAMILY --assembly $ASSEMBLY --output $(printf '%q' "$FAMILY_DIR") --resume-from tfbs || rc=1"
    # 2. Rebuild the LaTeX/PDF report.
    printf '%s\n' "$PYTHON generate_latex_report.py $(printf '%q' "$FAMILY_DIR") || rc=1"
    # 3. Assert the new artifacts exist.
    cat <<CHECK
echo "──────── verifying new report artifacts ────────"
mult=$(printf '%q' "$FAMILY_DIR")/05_tfbs/motif_site_multiplicity.csv
tex=\$(find $(printf '%q' "$FAMILY_DIR") -name report.tex | head -1)
if [ -s "\$mult" ]; then echo "PASS: motif_site_multiplicity.csv present (\$(wc -l < "\$mult") rows)"; else echo "FAIL: motif_site_multiplicity.csv missing"; rc=1; fi
if [ -n "\$tex" ]; then
  grep -q "Gene Expression Analysis" "\$tex" && echo "PASS: expression chapter in report" || { echo "FAIL: expression chapter missing"; rc=1; }
  grep -q "n_loci_multisite" "\$tex" && echo "PASS: site-multiplicity table in report" || { echo "FAIL: multiplicity table missing"; rc=1; }
else
  echo "FAIL: report.tex not found"; rc=1
fi
CHECK
    echo 'echo "Finished: $(date)   exit=$rc"'
    echo 'exit $rc'
  } > "$JOB_SH"
  chmod +x "$JOB_SH"

  echo "============================================================"
  echo "GAMECA report-change test [mode=report]"
  echo "  Family dir: $FAMILY_DIR"
  echo "  Logs:       $OUT_DIR"
  echo "  Resources:  -q $QUEUE -n $N_CORES -M $MEM_MB -W $WALL"
  echo "  Job script: $JOB_SH"
  echo "============================================================"
  _submit "report_test" "$MEM_MB" "$WALL" \
          "$OUT_DIR/report_test.%J.out" "$OUT_DIR/report_test.%J.err" bash "$JOB_SH"
  ;;

# ─────────────────────────────────────────────────────────────────────────────
scaling)
  # ── Config ───────────────────────────────────────────────────────────────
  FAMILY="${FAMILY:-MT2_Mm}"
  ASSEMBLY="${ASSEMBLY:-mm10}"
  SIZES="${SIZES:-100 500 1000 2000 all}"
  FAMILIES="${FAMILIES:-}"
  OUT_DIR="${OUT_DIR:-/home/amodz/anmol/gameca_scaling_test}"
  SOURCE="${SOURCE:-rmsk}"
  RMSK_DIR="${RMSK_DIR:-}"
  USE_NEXTFLOW="${USE_NEXTFLOW:-0}"
  NEXTFLOW_PROFILE="${NEXTFLOW_PROFILE:-lsf,singularity}"

  mkdir -p "$OUT_DIR"

  NEXTFLOW_FLAGS=""
  if [ "$USE_NEXTFLOW" = "1" ]; then
    if ! command -v nextflow >/dev/null 2>&1; then
      echo "ERROR: USE_NEXTFLOW=1 but 'nextflow' is not on PATH." >&2
      echo "       Install it or drop USE_NEXTFLOW to scale-test the in-process pipeline instead." >&2
      exit 1
    fi
    NEXTFLOW_FLAGS="--nextflow --nextflow-profile $NEXTFLOW_PROFILE"
    echo "Nextflow detected — each size will run via: query.py --nextflow --nextflow-profile $NEXTFLOW_PROFILE"
  fi

  PROXY_PREFIX="$(_proxy_prefix)"

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

  JOB_SH="$OUT_DIR/_run_scaling_test.sh"
  {
    echo "#!/usr/bin/env bash"
    echo "set -uo pipefail"
    echo "export CONDA_SOLVER=classic"
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
      echo "$PYTHON query.py --local --family $fam --assembly $asm --output $(printf '%q' "$run_dir") $cap $src $NEXTFLOW_FLAGS || echo \"  RUN FAILED: $tag (rc=\$?)\""
      echo "$PYTHON generate_latex_report.py $(printf '%q' "$run_dir") || echo \"  report build skipped/failed: $tag\""
      echo "echo \"  wall for $tag: \${SECONDS}s\""
      echo "du -sb $(printf '%q' "$run_dir") 2>/dev/null | cut -f1 > $(printf '%q' "$run_dir")/disk_usage_bytes.txt"
    done

    cat <<'PY'
echo; echo "==================== TIMING SUMMARY ===================="
python3 - "$OUT_DIR" <<'PYEOF'
import json, sys, glob, os, csv
root = sys.argv[1]
files = sorted(glob.glob(os.path.join(root, "run_*", "*", "stage_times.json")))
if not files:
    print("No stage_times.json found — did any run complete?"); sys.exit(0)
recs, stages = [], []
for f in files:
    try: d = json.load(open(f))
    except Exception as e: print(f"  skip {f}: {e}"); continue
    run_dir = os.path.dirname(os.path.dirname(f))
    disk_bytes = None
    du_file = os.path.join(run_dir, "disk_usage_bytes.txt")
    if os.path.exists(du_file):
        try:
            disk_bytes = int(open(du_file).read().strip())
        except Exception:
            disk_bytes = None
    d["disk_bytes"] = disk_bytes
    recs.append(d)
    for s in d.get("stage_seconds", {}):
        if s not in stages: stages.append(s)
cols = ["family", "assembly", "n_sequences", "n_clusters", "max_loci",
        "total_seconds", "peak_rss_mb", "disk_bytes"] + stages
out_csv = os.path.join(root, "scaling_timings.csv")
with open(out_csv, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(cols)
    for d in recs:
        w.writerow([d.get("family"), d.get("assembly"), d.get("n_sequences"),
                    d.get("n_clusters"), d.get("max_loci"), d.get("total_seconds"),
                    d.get("peak_rss_mb"), d.get("disk_bytes")]
                   + [d.get("stage_seconds", {}).get(s, "") for s in stages])
recs.sort(key=lambda d: (d.get("n_sequences") or 0))
print("### TIMING (seconds) ###")
hdr = f'{"family":<12}{"n_seq":>7}{"clust":>6}{"total_s":>9}   ' + "".join(f"{s[:11]:>12}" for s in stages)
print(hdr); print("-" * len(hdr))
for d in recs:
    row = f'{str(d.get("family"))[:11]:<12}{d.get("n_sequences") or 0:>7}{d.get("n_clusters") or 0:>6}{d.get("total_seconds") or 0:>9.1f}   '
    row += "".join(f'{d.get("stage_seconds",{}).get(s,0):>12.1f}' for s in stages)
    print(row)

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
    sug = int(-(-pk * 1.3 // 100) * 100)
    print(f'{str(d.get("family"))[:11]:<12}{d.get("n_sequences") or 0:>7}{pk:>10.1f}{pk/1024:>9.2f}{sug:>16}')
if have_mem:
    worst = max((d["peak_rss_mb"] for d in recs if d.get("peak_rss_mb")), default=0)
    print(f'\nProvision at least {int(-(-worst*1.3//100)*100)} MB '
          f'(-M {int(-(-worst*1.3//100)*100)}) for the largest size tested.')
else:
    print("peak_rss_mb unavailable (resource module missing?) — memory not captured.")

print("\n### DISK SPACE ###")
print("Total output directory size per run (du -sb; includes all_overlaps.tsv, JASPAR BED, CIAlign PNGs, etc).")
shdr = f'{"family":<12}{"n_seq":>7}{"size_MB":>10}{"size_GB":>9}'
print(shdr); print("-" * len(shdr))
have_disk = False
for d in recs:
    db = d.get("disk_bytes")
    if db is None:
        print(f'{str(d.get("family"))[:11]:<12}{d.get("n_sequences") or 0:>7}{"n/a":>10}{"n/a":>9}')
        continue
    have_disk = True
    mb = db / (1024 * 1024)
    print(f'{str(d.get("family"))[:11]:<12}{d.get("n_sequences") or 0:>7}{mb:>10.1f}{mb/1024:>9.2f}')
if have_disk:
    worst_db = max((d["disk_bytes"] for d in recs if d.get("disk_bytes")), default=0)
    print(f'\nLargest run used {worst_db/(1024*1024):.1f} MB ({worst_db/(1024**3):.2f} GB) on disk — '
          f'plan scratch/output storage accordingly per family run.')
else:
    print("disk_usage_bytes.txt unavailable — disk usage not captured.")
print(f"\nSaved: {out_csv}")
PYEOF
PY
    echo 'echo "Finished: $(date)"'
  } > "$JOB_SH"
  chmod +x "$JOB_SH"

  echo "============================================================"
  echo "GAMECA scaling / timing test [mode=scaling]"
  if [ -n "$FAMILIES" ]; then
    echo "  Families: $FAMILIES  (each at full size)"
  else
    echo "  Family:   $FAMILY ($ASSEMBLY)   Sizes: $SIZES"
  fi
  echo "  Out dir:  $OUT_DIR"
  echo "  Table:    $OUT_DIR/scaling_timings.csv (written at end)"
  echo "  Resources: -q $QUEUE -n $N_CORES -M $MEM_MB -W $WALL"
  echo "  Job script: $JOB_SH"
  echo "============================================================"
  _submit "gameca_scaling" "$MEM_MB" "$WALL" \
          "$OUT_DIR/scaling_test.%J.out" "$OUT_DIR/scaling_test.%J.err" bash "$JOB_SH"
  ;;

*)
  echo "ERROR: unknown mode '$MODE'. Choices: cluster | clustering | nextflow | report | scaling" >&2
  exit 1
  ;;
esac
