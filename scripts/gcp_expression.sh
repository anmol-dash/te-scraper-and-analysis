#!/usr/bin/env bash
# gcp_expression.sh --- recompute locus_expression.tsv on a throwaway GCP VM.
#
# WHY: the PennHPC cluster is unavailable and the OCI box is gone. STAR needs
# ~32 GB RAM for hg38, which neither a laptop (18 GB) nor free-tier OCI can
# give. GCP is x86 Linux, so this runs STAR + featureCounts exactly as the
# cluster did -- the numbers stay methodologically comparable.
#
# COST CONTROL IS THE POINT OF THIS SCRIPT. Four independent stops, so no
# single failure can leave something running:
#
#   1. --instance-termination-action=DELETE (if the VM ends for any reason --
#      preemption, the deadline -- it is deleted, never left stopped and billing)
#   2. --max-run-duration=$MAX_HOURS  -- GCP itself deletes the VM at the
#      deadline. This is enforced server-side and survives ANY failure of the
#      guest, the network, or this laptop.
#   3. the startup script deletes its own instance when the work finishes
#   4. `down` deletes everything on demand, and is safe to run at any time
#      (including repeatedly, or when nothing exists)
#
# Expected spend: well under $5 of the $300 free trial (see `estimate`).
#
#   bash scripts/gcp_expression.sh estimate   # what it will cost, no changes
#   bash scripts/gcp_expression.sh up         # provision + start (asks first)
#   bash scripts/gcp_expression.sh status     # where is it up to
#   bash scripts/gcp_expression.sh logs       # tail the VM's progress
#   bash scripts/gcp_expression.sh fetch      # pull results down
#   bash scripts/gcp_expression.sh down       # DELETE EVERYTHING (do this!)
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
PROJECT=${PROJECT:-$(gcloud config get-value project 2>/dev/null)}
# Capacity for large instances is PER-ZONE and moves around by the hour:
# us-central1-a returned ZONE_RESOURCE_POOL_EXHAUSTED for n2-standard-16 on
# 2026-08-13, and that same contention is almost certainly why the SPOT VM there
# was reclaimed 5 min into the previous run. So try a list, not one zone.
# The bucket is US multi-region, so any US zone reads the 31 GB index cache at
# full speed with no egress charge -- zone choice costs nothing here.
ZONE=${ZONE:-us-central1-a}
ZONES=${ZONES:-"$ZONE us-central1-b us-central1-c us-central1-f
                us-east1-b us-east1-c us-east1-d us-east4-a us-east4-b
                us-west1-a us-west1-b us-west1-c"}
VM=${VM:-endo-expression}
# n2-standard-16: 16 vCPU / 64 GB.
# 32 GB is NOT enough: genomeGenerate got 51 min in and was OOM-killed during
# "generating Suffix Array index" (STAR's ~32 GB figure is for ALIGNMENT; the
# index build with --sjdbGTFfile annotations needs appreciably more). 64 GB
# removes that risk, and the extra threads roughly halve alignment time, so the
# job finishes sooner for about the same total spend.
MACHINE=${MACHINE:-n2-standard-16}
# SPOT is ~4x cheaper but GCP can reclaim the VM at any moment with ~30 s notice
# -- the 2026-08-12 run was preempted 5 min 14 s after boot, 54 s into aligning.
# A 10h+ job is a lot of exposure, and on the $300 credit the difference is a few
# dollars. STANDARD removes preemption entirely; --max-run-duration still caps
# the spend either way.
PROVISIONING=${PROVISIONING:-STANDARD}
# Machine-type fallbacks, tried within each zone. Every one of these has >= 64 GB
# RAM and that is not negotiable: STAR loads the ~31 GB hg38 index into memory to
# ALIGN, so a 32 GB machine OOMs. (64 GB was originally chosen for genomeGenerate,
# which is now cached and skipped -- but alignment still needs the index
# resident.) Fewer vCPUs just means slower, so highmem-8 is an acceptable last
# resort; less memory is not.
MACHINES=${MACHINES:-"$MACHINE n2d-standard-16 e2-standard-16 c3-standard-22 n2-highmem-8"}
DISK_GB=${DISK_GB:-200}          # hg38 3 + STAR index 30 + fastq 65 + slack
# Hard server-side deletion deadline. 18, not 10: 14 samples x ~40 min is ~10 h,
# so a 10 h cap was set exactly at the expected runtime and would have killed the
# run just before it finished. Overrun is now recoverable (finished samples are
# restored from the bucket on the next 'up'), but there is no reason to cut it
# fine at ~$0.80/h against a $300 credit.
MAX_HOURS=${MAX_HOURS:-18}
BUCKET=${BUCKET:-gs://${PROJECT}-endo-expression}
MANIFEST=${MANIFEST:-$REPO/scripts/srp090091_manifest.tsv}
FAMILIES=${FAMILIES:-"LTR66 LTR10G"}
FILTER=${FILTER:-'^(LTR66|LTR10G)_dup'}
LOCAL_OUT=${LOCAL_OUT:-$HOME/endo/expression}

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
die() { echo "FATAL: $*" >&2; exit 1; }
[ -n "$PROJECT" ] || die "no GCP project set (gcloud config set project ...)"

# The VM is not necessarily in $ZONE any more: 'up' falls through a list of zones
# looking for capacity. Ask where it actually landed instead of assuming -- above
# all so 'down' can never report "nothing to delete" while a VM bills away in
# another zone.
vm_zone() {
  local z
  z=$(gcloud compute instances list --project "$PROJECT" --filter="name=$VM" \
      --format="value(zone.basename())" 2>/dev/null | head -1)
  printf '%s' "${z:-$ZONE}"
}

cmd=${1:-help}

# ---------------------------------------------------------------- estimate --
if [ "$cmd" = "estimate" ]; then
  say "cost estimate (nothing is created)"
  python3 - "$MACHINE" "$DISK_GB" "$MAX_HOURS" "$PROVISIONING" <<'PY'
import sys
machine, disk, maxh = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
prov = sys.argv[4]
# us-central1 list prices, Aug 2026 (approximate, for sizing only)
spot = {"n2-standard-8": 0.097, "n2-standard-16": 0.194, "n2-highmem-8": 0.131}
ondm = {"n2-standard-8": 0.389, "n2-standard-16": 0.777, "n2-highmem-8": 0.525}
rate = (spot if prov == "SPOT" else ondm).get(machine, 0.10 if prov == "SPOT" else 0.40)
disk_hr = disk * 0.10 / 730          # pd-balanced ~$0.10/GB-month
# 14 samples x ~40 min (download + align + 2x featureCounts over a 4.8M-feature
# GTF). The old 4 h figure predated measuring a real sample.
run = 10.0
print(f"  machine {machine} ({prov.lower()})  ${rate:.3f}/h")
print(f"  disk    {disk} GB pd-balanced  ${disk_hr:.3f}/h")
print(f"  ingress from ENA               $0 (inbound is free)")
print(f"  egress  results ~2 MB          ~$0.00")
print()
print(f"  expected run ~{run:.0f} h  ->  ${(rate + disk_hr) * run:.2f}")
print(f"  worst case  {maxh:.0f} h  ->  ${(rate + disk_hr) * maxh:.2f}  (hard cap: GCP deletes the VM)")
print()
print(f"  Free trial credit is $300, so the ceiling above is ~{(rate + disk_hr) * maxh / 300 * 100:.0f}% of it.")
PY
  echo
  echo "  Stops in force once running:"
  echo "    * GCP deletes the VM after ${MAX_HOURS}h no matter what (--max-run-duration)"
  echo "    * the VM deletes itself when the pipeline finishes"
  echo "    * any termination deletes rather than stops it (never left billing)"
  echo "    * finished samples are in the bucket, so a restart resumes, not repeats"
  echo "    * bash scripts/gcp_expression.sh down   -- any time"
  exit 0
fi

# -------------------------------------------------------------------- down --
# Deliberately first among the action verbs, and safe to run at any time.
if [ "$cmd" = "down" ] || [ "$cmd" = "stop" ]; then
  say "down -- deleting everything"
  DZ=$(vm_zone)
  echo "  project: $PROJECT   zone: $DZ"
  if gcloud compute instances describe "$VM" --zone "$DZ" --project "$PROJECT" >/dev/null 2>&1; then
    echo "  deleting VM $VM (and its boot disk) ..."
    gcloud compute instances delete "$VM" --zone "$DZ" --project "$PROJECT" --quiet \
      && echo "  VM deleted"
  else
    echo "  VM $VM: not present in any zone (nothing to delete)"
  fi
  # Orphaned disks, if the VM was created with a non-auto-delete disk. Delete each
  # in ITS OWN zone -- a zone-fallback run can leave a disk somewhere other than
  # $ZONE, and a disk deleted nowhere is a disk that keeps billing.
  gcloud compute disks list --project "$PROJECT" --filter="name~^${VM}" \
      --format="value(name,zone.basename())" 2>/dev/null \
  | while read -r d dz; do
      [ -z "${d:-}" ] && continue
      echo "  deleting orphaned disk $d in ${dz:-$DZ} ..."
      gcloud compute disks delete "$d" --zone "${dz:-$DZ}" --project "$PROJECT" --quiet || true
    done
  if [ "${2:-}" = "--all" ]; then
    echo "  deleting bucket $BUCKET (results included) ..."
    gcloud storage rm -r "$BUCKET" --project "$PROJECT" 2>/dev/null && echo "  bucket deleted" \
      || echo "  bucket: not present"
  else
    echo "  bucket $BUCKET kept (holds your results; 'down --all' removes it)"
    echo "  storage cost of a few MB is effectively zero."
  fi
  echo
  echo "  verify nothing is billable:"
  echo "      gcloud compute instances list --project $PROJECT"
  exit 0
fi

# ------------------------------------------------------------------ status --
if [ "$cmd" = "status" ]; then
  # A read-only report must never abort partway and leave you guessing whether
  # it finished or died. Under `set -e` + pipefail a $(gcloud ...) that returns
  # non-zero because an object simply does not exist kills the script mid-output
  # -- which is exactly what truncated this report.
  set +e +o pipefail
  say "status"
  echo "  project $PROJECT   vm $VM"
  # No --zone filter, and zone is a column: 'up' may have placed the VM in any of
  # $ZONES, and a status that only looked in one zone could report "nothing
  # running" while it runs somewhere else.
  vms=$(gcloud compute instances list --project "$PROJECT" --filter="name=$VM" \
    --format="table(name,zone.basename(),status,machineType.basename(),scheduling.provisioningModel,lastStartTimestamp)" 2>/dev/null)
  if [ -n "$vms" ]; then echo "$vms" | sed 's/^/  /'
  else echo "  VM: not present (nothing running, nothing billing)"; fi
  echo
  echo "  progress markers in $BUCKET:"
  marks=$(gcloud storage ls "$BUCKET/progress/" 2>/dev/null | sed 's|.*/|    |')
  if [ -n "$marks" ]; then echo "$marks"; else echo "    (none yet -- has 'up' been run?)"; fi
  echo
  # Per-sample progress: the align stage is ~90% of the runtime and the stage
  # markers cannot show movement inside it. Counted samples land in the bucket
  # as they finish, so this is the real progress bar.
  ndone=$(gcloud storage ls "$BUCKET/results/locus/" 2>/dev/null | grep -c featureCounts)
  ntot=$(( $(grep -c . "$MANIFEST" 2>/dev/null || echo 1) - 1 ))
  echo "  samples counted: ${ndone:-0} / $ntot   (these survive a restart)"
  echo
  # Preemption is not a pipeline failure and must not read like one. It also
  # leaves NO other trace, so without this the report is just "markers stop
  # partway, no VM" -- which tells you nothing about why.
  if gcloud storage ls "$BUCKET/progress/PREEMPTED" >/dev/null 2>&1; then
    echo "  *** PREEMPTED (spot capacity reclaimed by GCP -- not a pipeline error) ***"
    gcloud storage cat "$BUCKET/progress/PREEMPTED" 2>/dev/null | sed 's/^/    /'
    echo "    'up' resumes from the ${ndone:-0} sample(s) above."
    echo "    PROVISIONING=STANDARD bash scripts/gcp_expression.sh up   # no preemption"
  fi
  # FAILED must be reported before results: an empty file can still exist
  if gcloud storage ls "$BUCKET/progress/FAILED" >/dev/null 2>&1; then
    echo "  *** RUN FAILED ***"
    gcloud storage cat "$BUCKET/progress/FAILED" 2>/dev/null | sed 's/^/    /'
    echo "    full log: bash scripts/gcp_expression.sh logfile"
  fi
  echo "  results:"
  if gcloud storage ls "$BUCKET/results/RUN_SUMMARY.txt" >/dev/null 2>&1; then
    echo -n "    "; gcloud storage cat "$BUCKET/results/RUN_SUMMARY.txt" 2>/dev/null
  fi
  for f in locus_expression.tsv te_timepoint_summary.tsv; do
    sz=$(gcloud storage ls -l "$BUCKET/results/$f" 2>/dev/null | head -1 | awk "{print \$1}" || true)
    if [ -n "${sz:-}" ] && [ "${sz:-0}" -gt 200 ] 2>/dev/null; then echo "    $f: READY ($sz bytes)"
    elif [ -n "${sz:-}" ]; then echo "    $f: PRESENT BUT SUSPICIOUSLY SMALL ($sz bytes)"
    else echo "    $f: not yet"; fi
  done
  exit 0
fi

# -------------------------------------------------------------------- logs --
if [ "$cmd" = "logfile" ]; then
  gcloud storage cat "$BUCKET/progress/endo.log" 2>/dev/null || echo "no log uploaded yet"
  exit 0
fi

if [ "$cmd" = "logs" ]; then
  gcloud compute instances get-serial-port-output "$VM" --zone "$(vm_zone)" --project "$PROJECT" 2>/dev/null \
    | grep -aE 'startup-script|\[endo\]' | tail -"${2:-40}" \
    || echo "no serial output (VM may be gone -- check 'status')"
  exit 0
fi

# ------------------------------------------------------------------- fetch --
if [ "$cmd" = "fetch" ]; then
  say "fetch"
  mkdir -p "$LOCAL_OUT"
  gcloud storage cp -r "$BUCKET/results/*" "$LOCAL_OUT/" 2>/dev/null \
    || die "no results in $BUCKET/results (check 'status')"
  echo "  -> $LOCAL_OUT"
  ls -la "$LOCAL_OUT" | sed 's/^/    /'
  echo
  echo "  Remember to release the VM if it is still up:"
  echo "      bash scripts/gcp_expression.sh down"
  exit 0
fi

# ---------------------------------------------------------------------- up --
if [ "$cmd" != "up" ]; then
  sed -n '2,30p' "$0"; exit 0
fi

say "up"
[ -s "$MANIFEST" ] || die "manifest not found: $MANIFEST"

# --- preflight: fail with something actionable, not a raw API error ---------
if ! gcloud billing projects describe "$PROJECT" --format='value(billingEnabled)' 2>/dev/null | grep -qi true; then
  die "billing is not enabled on $PROJECT. The free trial still requires a linked
  billing account (it just draws on the \$300 credit).
  Link one at: https://console.cloud.google.com/billing/linkedaccount?project=$PROJECT"
fi
for api in compute.googleapis.com storage.googleapis.com; do
  if ! gcloud services list --enabled --project "$PROJECT" --format='value(config.name)' 2>/dev/null \
       | grep -qx "$api"; then
    echo "  enabling $api (one-time, free) ..."
    gcloud services enable "$api" --project "$PROJECT" --quiet \
      || die "could not enable $api. Enable it once with:
  gcloud services enable $api --project $PROJECT"
  fi
done
echo "  APIs OK (compute, storage); billing linked"
N=$(($(grep -c . "$MANIFEST") - 1))
echo "  project  : $PROJECT"
echo "  zones    : $(echo $ZONES | tr '\n' ' ' | awk '{print $1" (preferred), then "NF-1" more"}')"
echo "  machine  : $MACHINE ($PROVISIONING)  disk ${DISK_GB} GB"
[ "$PROVISIONING" = "SPOT" ] && echo "             ^ can be reclaimed by GCP mid-run (PROVISIONING=STANDARD to avoid)"
echo "  samples  : $N   families: $FAMILIES"
echo "  hard stop: ${MAX_HOURS}h, enforced by GCP (--max-run-duration)"
echo "  bucket   : $BUCKET"
echo
# Search every zone, not just $ZONE -- a previous run may have landed elsewhere,
# and creating a second VM because we looked in the wrong place would double the
# spend silently.
if [ -n "$(gcloud compute instances list --project "$PROJECT" --filter="name=$VM" \
           --format='value(name)' 2>/dev/null)" ]; then
  die "$VM already exists (in $(vm_zone)). 'status' to inspect, 'down' to remove it first."
fi
if [ "${2:-}" != "--yes" ]; then
  read -r -p "  Create this VM now? [y/N] " ans
  case "$ans" in y|Y|yes) ;; *) echo "  aborted; nothing created."; exit 0 ;; esac
fi

echo "  clearing stale progress markers from any previous run ..."
gcloud storage rm -r "$BUCKET/progress" --project "$PROJECT" 2>/dev/null || true
echo "  ensuring bucket ..."
gcloud storage buckets describe "$BUCKET" --project "$PROJECT" >/dev/null 2>&1 \
  || gcloud storage buckets create "$BUCKET" --project "$PROJECT" --location=US --quiet
echo "  uploading manifest + helper ..."
gcloud storage cp "$MANIFEST" "$BUCKET/input/manifest.tsv" --quiet
gcloud storage cp "$REPO/scripts/summarize_te_timepoints.py" "$BUCKET/input/" --quiet

STARTUP=$(mktemp)
cat > "$STARTUP" <<STARTUP_EOF
#!/usr/bin/env bash
# Runs as root on the VM. Every path here is self-contained: no container, no
# conda, static/prebuilt binaries only.
exec > >(tee /var/log/endo.log | logger -t endo -s 2>/dev/console) 2>&1
set -uo pipefail
echo "[endo] boot \$(date)"

BUCKET="$BUCKET"
FILTER='$FILTER'
FAMILIES="$FAMILIES"
MAX_HOURS=$MAX_HOURS
THREADS=\$(nproc)
WORK=/mnt/work

# ---- stop #3/#4: independent watchdogs inside the guest ----
# Even if the pipeline hangs, power off well before the GCP deadline. Belt and
# braces: --max-run-duration already guarantees deletion server-side.
shutdown -P +\$(( MAX_HOURS * 60 - 15 )) "endo watchdog" || true

finish() {   # self-delete on the way out, whatever the outcome
  local rc=\$?
  echo "[endo] finishing rc=\$rc \$(date)"
  gcloud storage cp /var/log/endo.log "\$BUCKET/progress/endo.log" --quiet 2>/dev/null || true
  echo "[endo] deleting self"
  gcloud --quiet compute instances delete "\$(hostname)" \
    --zone="\$(curl -s -H Metadata-Flavor:Google \
      http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print \$NF}')" \
    || poweroff
}
trap finish EXIT

# A spot preemption HARD-terminates the VM: the EXIT trap never runs, nothing is
# uploaded, and the bucket simply stops updating -- indistinguishable from a hang
# or a crash, which is exactly how the 2026-08-12 run looked (markers up to
# 'align', no FAILED, no DONE, no endo.log). GCP flips this metadata key ~30 s
# before pulling the plug; that is enough time to leave a note saying so.
( while :; do
    if [ "\$(curl -s -H Metadata-Flavor:Google \
         http://metadata.google.internal/computeMetadata/v1/instance/preempted 2>/dev/null)" = "TRUE" ]; then
      echo "preempted \$(date -Is) -- GCP reclaimed this SPOT VM; not a pipeline error.
Finished samples are already in \$BUCKET/results/, and re-running 'up' resumes
from them. PROVISIONING=STANDARD avoids preemption altogether." \
        | gcloud storage cp - "\$BUCKET/progress/PREEMPTED" --quiet 2>/dev/null
      gcloud storage cp /var/log/endo.log "\$BUCKET/progress/endo.log" --quiet 2>/dev/null
      break
    fi
    sleep 10
  done ) &

mark() { echo "[endo] STAGE \$1"; echo "\$1 \$(date -Is)" | gcloud storage cp - "\$BUCKET/progress/\$1" --quiet 2>/dev/null || true; }

# A stage that cannot succeed must STOP the run, not let it march on to DONE.
# The first version of this script had no such guard: UCSC was unreachable, the
# genome never downloaded, all 14 alignments failed instantly, and an EMPTY
# locus_expression.tsv was uploaded and marked DONE. Loud failure beats a
# plausible-looking empty result.
fail() {
  echo "[endo] FATAL: \$*"
  echo "\$*" | gcloud storage cp - "\$BUCKET/progress/FAILED" --quiet 2>/dev/null || true
  exit 1
}
require() {   # require <file> <min-bytes> <what>
  local f="\$1" min="\$2" what="\$3" sz
  # wc -c, not stat: stat's size flag differs between GNU (-c%s) and BSD (-f%z),
  # so this stays testable off-VM as well as correct on it
  sz=\$(wc -c < "\$f" 2>/dev/null | tr -d ' ' || echo 0); sz=\${sz:-0}
  [ "\$sz" -ge "\$min" ] || fail "\$what missing or truncated: \$f (\$sz bytes, need >= \$min)"
  echo "[endo] ok \$what: \$f (\$sz bytes)"
}
# UCSC times out from GCP often enough to matter; try every mirror, with retries.
fetch_ucsc() {   # fetch_ucsc <path-under-goldenPath> <outfile>
  local path="\$1" out="\$2" h
  for h in hgdownload.soe.ucsc.edu hgdownload.gi.ucsc.edu hgdownload2.soe.ucsc.edu; do
    echo "[endo] trying \$h/\$path"
    if curl -fsSL --retry 5 --retry-all-errors --retry-delay 10 \
            --connect-timeout 30 --max-time 3600 -o "\$out" "https://\$h/goldenPath/\$path"; then
      [ -s "\$out" ] && { echo "[endo] got \$out from \$h"; return 0; }
    fi
    echo "[endo] \$h failed for \$path"
  done
  return 1
}

# ---- disk ----
mkfs.ext4 -F /dev/disk/by-id/google-persistent-disk-1 2>/dev/null || true
mkdir -p \$WORK && mount /dev/disk/by-id/google-persistent-disk-1 \$WORK 2>/dev/null || WORK=/var/endo
mkdir -p \$WORK/{fastq,bam,locus,counts,ref,bin}
cd \$WORK
mark boot

# ---- tools: static STAR + precompiled subread, no package manager ----
mark tools
curl -fsSL -o bin/STAR https://raw.githubusercontent.com/alexdobin/STAR/2.7.11b/bin/Linux_x86_64_static/STAR
chmod +x bin/STAR
# featureCounts: use Debian's package, NOT the SourceForge tarball. That binary
# is "statically linked, for GNU/Linux 2.6.18" -- an ancient glibc build that
# segfaults instantly on Debian 12, which boots with vsyscall=none. It crashed
# on \`-v\` at this stage and then once per sample, wasting an alignment each time.
export DEBIAN_FRONTEND=noninteractive
# aria2   -- resumable multi-stream download (the FASTQ fix, see fetch_pair)
# sra-toolkit -- fasterq-dump, for the AWS SRA mirror fallback
# pigz    -- parallel gzip, to recompress that fallback's output
# All three are optional: each has a curl/gzip path behind it, so a failed apt
# cannot sink the run. Only subread is load-bearing, and it is gated below.
apt-get update -qq && apt-get install -y -qq subread aria2 sra-toolkit pigz || true
export PATH=\$WORK/bin:\$PATH
if ! command -v featureCounts >/dev/null 2>&1; then
  echo "[endo] apt subread unavailable; falling back to the SourceForge build"
  curl -fsSL -o subread.tar.gz https://downloads.sourceforge.net/project/subread/subread-2.0.6/subread-2.0.6-Linux-x86_64.tar.gz
  tar xzf subread.tar.gz && cp subread-2.0.6-Linux-x86_64/bin/featureCounts bin/ && chmod +x bin/featureCounts
fi

# GATE the tools before spending an hour on an index and 14 alignments. The
# previous run printed STAR's version, printed NOTHING for featureCounts, and
# carried on regardless -- every sample then died in the counting step.
STAR --version >/dev/null 2>&1 || fail "STAR will not run on this VM"
echo "[endo] STAR \$(STAR --version)"
FC_VER=\$(featureCounts -v 2>&1 | tr -d '\\0' | grep -i featureCounts | head -1)
[ -n "\$FC_VER" ] || fail "featureCounts will not run on this VM (segfault or missing). \\
Debian's 'subread' package is the supported source here; the SourceForge static \\
build targets glibc 2.6.18 and dies under vsyscall=none."
echo "[endo] \$FC_VER  (\$(command -v featureCounts))"

# ---- references (cached in the bucket: a re-run skips the slow parts) ----
mark refs
cd \$WORK/ref
if gcloud storage cp "\$BUCKET/cache/hg38.fa" hg38.fa --quiet 2>/dev/null; then
  echo "[endo] hg38.fa from bucket cache"
else
  fetch_ucsc hg38/bigZips/hg38.fa.gz hg38.fa.gz || fail "hg38.fa.gz: every UCSC mirror failed"
  gunzip -f hg38.fa.gz
  require hg38.fa 3000000000 "hg38 genome"
  gcloud storage cp hg38.fa "\$BUCKET/cache/hg38.fa" --quiet 2>/dev/null || true
fi
require hg38.fa 3000000000 "hg38 genome"

if gcloud storage cp "\$BUCKET/cache/rmsk.txt.gz" rmsk.txt.gz --quiet 2>/dev/null; then
  echo "[endo] rmsk from bucket cache"
else
  fetch_ucsc hg38/database/rmsk.txt.gz rmsk.txt.gz || fail "rmsk.txt.gz: every UCSC mirror failed"
  gcloud storage cp rmsk.txt.gz "\$BUCKET/cache/rmsk.txt.gz" --quiet 2>/dev/null || true
fi
require rmsk.txt.gz 100000000 "RepeatMasker table"

# knownGene only feeds STAR's --sjdbGTFfile, which is optional; do not abort on it
SJDB=""
if fetch_ucsc hg38/bigZips/genes/hg38.knownGene.gtf.gz hg38.knownGene.gtf.gz; then
  gunzip -f hg38.knownGene.gtf.gz && SJDB="--sjdbGTFfile \$WORK/ref/hg38.knownGene.gtf --sjdbOverhang 100"
else
  echo "[endo] WARNING: knownGene GTF unavailable; building the index without splice junctions"
fi
# identical TE GTF construction to scripts/setup_hervk_ccle.sh
zcat rmsk.txt.gz | awk -F'\t' '
  \$12 ~ /Simple_repeat|Low_complexity|Satellite|^RNA\$|rRNA|scRNA|snRNA|srpRNA|tRNA|Unknown|\?/ {next}
  { n[\$11]++;
    printf "%s\trmsk\texon\t%d\t%d\t.\t%s\t.\tgene_id \"%s\"; transcript_id \"%s_dup%d\"; family_id \"%s\"; class_id \"%s\";\n",
           \$6,\$7+1,\$8,\$10,\$11,\$11,n[\$11],\$13,\$12 }' > hg38_rmsk_TE.gtf
echo "[endo] TE GTF \$(wc -l < hg38_rmsk_TE.gtf) lines; target families present:"
awk -F'gene_id "' '{split(\$2,a,"\"");print a[1]}' hg38_rmsk_TE.gtf | grep -cE "^(\${FAMILIES// /|})\$" || true

echo "[endo] TE GTF families of interest:"
awk -F'gene_id "' '{split(\$2,a,"\"");print a[1]}' hg38_rmsk_TE.gtf \
  | grep -cE "^(\${FAMILIES// /|})\$" | xargs echo "[endo]   target-family GTF rows:"

mark star_index
mkdir -p star_hg38
# the index build is ~40 min of CPU; cache it so a re-run costs minutes
if gcloud storage cp "\$BUCKET/cache/star_hg38.tar" - --quiet 2>/dev/null | tar xf - -C \$WORK/ref 2>/dev/null \
   && [ -s star_hg38/genomeParameters.txt ]; then
  echo "[endo] STAR index from bucket cache"
else
  # STAR caps itself at ~31 GB by default, below what this build needs. Size the
  # limit from the machine, leaving 6 GB for the OS.
  RAMB=\$(( \$(awk '/MemTotal/{print \$2}' /proc/meminfo) * 1024 - 6000000000 ))
  echo "[endo] genomeGenerate --limitGenomeGenerateRAM \$RAMB"
  STAR --runMode genomeGenerate --runThreadN \$THREADS --genomeDir \$WORK/ref/star_hg38 \
       --genomeFastaFiles \$WORK/ref/hg38.fa \$SJDB --outFileNamePrefix \$WORK/ref/idx_ \
       --limitGenomeGenerateRAM \$RAMB \
    || { dmesg 2>/dev/null | grep -i "killed process" | tail -2;
         fail "STAR genomeGenerate failed. A 'Killed' line above means the VM ran out of RAM -- needs >=64 GB (MACHINE=n2-standard-16)."; }
  require star_hg38/genomeParameters.txt 100 "STAR index"
  tar cf - -C \$WORK/ref star_hg38 | gcloud storage cp - "\$BUCKET/cache/star_hg38.tar" --quiet 2>/dev/null || true
fi
require star_hg38/genomeParameters.txt 100 "STAR index"
require star_hg38/SA 1000000 "STAR suffix array"

# ---- FASTQ acquisition -----------------------------------------------------
# The previous run lost all 28 FASTQ files to "transfer closed with N bytes
# remaining", then to hard 403s. Three separate mistakes, all fixed here:
#
#   1. \`curl --retry\` RESTARTS a dead transfer from byte 0. ENA drops long TLS
#      transfers, so 11 attempts x 2.3 GB each moved ~3 MB and got nowhere --
#      and the hammering is what earned us the 403s on the later samples.
#      Resume is mandatory: aria2 -c, or curl -C -.
#   2. curl's exit status was never checked, so a 3 MB stub was handed to STAR
#      ("unexpected end of file" -> "quality string length is not equal to
#      sequence length"). ENA publishes fastq_bytes and fastq_md5; verify.
#   3. Both mates were fetched CONCURRENTLY, doubling the per-IP connection
#      pressure that triggered the throttling. One at a time.
#
# This mirrors scripts/download_fastq_ena.sh, which already pulls much larger
# CCLE runs off ENA reliably using exactly this resume-and-verify loop.
fsize() { wc -c < "\$1" 2>/dev/null | tr -d ' ' || echo 0; }

ena_lookup() {   # <run> -> "urls<TAB>bytes<TAB>md5s" (';'-separated pairs)
  local run="\$1" info attempt
  for attempt in 1 2 3 4 5; do
    info=\$(curl -sS --fail --connect-timeout 20 --max-time 120 \
      "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=\${run}&result=read_run&fields=fastq_ftp,fastq_bytes,fastq_md5&format=tsv" \
      2>/dev/null | awk -F'\t' 'NR==2{print \$2"\t"\$3"\t"\$4}')
    [ -n "\$(printf '%s' "\$info" | cut -f1)" ] && { printf '%s' "\$info"; return 0; }
    echo "[endo]   ENA lookup \$run failed (\$attempt/5); retrying" >&2
    sleep \$(( attempt * 10 ))
  done
  return 1
}

get_one() {   # <url> <out> <want_bytes> <md5> -- resumable + verified
  local url="\$1" out="\$2" want="\$3" md5="\$4" have prev=-1 stalled=0 attempt got
  want=\${want:-0}
  for attempt in \$(seq 1 40); do
    have=\$(fsize "\$out")
    [ "\$want" -gt 0 ] && [ "\$have" -ge "\$want" ] && break
    # A run of attempts that move zero bytes means blocked or throttled, not
    # flaky. Retrying then just reprints the same error 40 times (the lesson
    # already learned in download_fastq_ena.sh).
    if [ "\$have" -le "\$prev" ]; then stalled=\$(( stalled + 1 )); else stalled=0; fi
    if [ "\$stalled" -ge 5 ]; then
      echo "[endo]   stalled: 5 attempts moved no bytes (\$have/\$want)"; break
    fi
    prev=\$have
    echo "[endo]   [\$attempt] \$(basename "\$out"): \$have / \$want bytes"
    if command -v aria2c >/dev/null 2>&1; then
      # -x4 not -x16: modest parallelism resumes fast without re-tripping the
      # per-IP limit that blocked the last run.
      # --allow-overwrite=true is REQUIRED alongside -c: with =false and no
      # .aria2 control file (the usual case after a killed transfer) aria2c
      # refuses the existing partial outright instead of resuming it. Same
      # combination download_fastq_ena.sh already uses against ENA.
      aria2c -x4 -s4 -c --max-tries=3 --retry-wait=15 --timeout=60 \
             --summary-interval=0 --console-log-level=warn \
             --allow-overwrite=true --auto-file-renaming=false \
             -d "\$(dirname "\$out")" -o "\$(basename "\$out")" "\$url" || true
    else
      curl -fsSL -C - -o "\$out" "\$url" --connect-timeout 30 \
           --speed-limit 2000 --speed-time 120 \
           --retry 3 --retry-delay 15 --retry-all-errors || true
    fi
    sleep 5
  done
  have=\$(fsize "\$out")
  if [ "\$want" -gt 0 ] && [ "\$have" -lt "\$want" ]; then
    echo "[endo]   SHORT \$(basename "\$out"): \$have/\$want bytes"; return 1
  fi
  if [ -n "\$md5" ]; then
    got=\$(md5sum "\$out" 2>/dev/null | awk '{print \$1}')
    [ "\$got" = "\$md5" ] || { echo "[endo]   MD5 MISMATCH \$(basename "\$out")"; return 1; }
    echo "[endo]   md5 ok \$(basename "\$out") (\$have bytes)"
  else
    # No md5 published: at minimum prove the gzip stream is complete, which is
    # the specific corruption that reached STAR last time.
    gzip -t "\$out" 2>/dev/null || { echo "[endo]   gzip -t FAILED \$(basename "\$out")"; return 1; }
    echo "[endo]   gzip ok \$(basename "\$out") (\$have bytes)"
  fi
  return 0
}

get_from_sra() {   # <run> -- AWS SRA Open Data mirror; a DIFFERENT provider, so
                   # an ENA-side block or throttle cannot take it out too.
  local run="\$1"
  command -v fasterq-dump >/dev/null 2>&1 || { echo "[endo]   no fasterq-dump; cannot use the SRA mirror"; return 1; }
  echo "[endo]   falling back to the SRA Open Data mirror (AWS, public, no credentials)"
  curl -fsSL -C - -o "fastq/\${run}.sra" \
       "https://sra-pub-run-odp.s3.amazonaws.com/sra/\${run}/\${run}" \
       --connect-timeout 30 --speed-limit 2000 --speed-time 120 \
       --retry 10 --retry-delay 15 --retry-all-errors || { echo "[endo]   SRA mirror download failed"; return 1; }
  fasterq-dump --split-files --threads \$THREADS --outdir fastq \
               --temp fastq "fastq/\${run}.sra" || { echo "[endo]   fasterq-dump failed"; return 1; }
  rm -f "fastq/\${run}.sra"
  [ -s "fastq/\${run}_1.fastq" ] && [ -s "fastq/\${run}_2.fastq" ] || return 1
  # leave them uncompressed; STAR reads them with cat and they are deleted after
  mv "fastq/\${run}_1.fastq" fastq/r1.fq && mv "fastq/\${run}_2.fastq" fastq/r2.fq
  R1=fastq/r1.fq; R2=fastq/r2.fq; RDCMD=cat
  return 0
}

fetch_pair() {   # <run> -- sets R1/R2/RDCMD on success
  local run="\$1" info urls bytes md5s u1 u2 b1 b2 m1 m2
  R1=""; R2=""; RDCMD=cat
  # A previous run may have parked verified copies here; free recovery path.
  if gcloud storage cp "\$BUCKET/cache/fastq/\${run}_1.fastq.gz" fastq/r1.gz --quiet 2>/dev/null \
  && gcloud storage cp "\$BUCKET/cache/fastq/\${run}_2.fastq.gz" fastq/r2.gz --quiet 2>/dev/null \
  && gzip -t fastq/r1.gz 2>/dev/null && gzip -t fastq/r2.gz 2>/dev/null; then
    echo "[endo]   FASTQ pair from bucket cache"
    R1=fastq/r1.gz; R2=fastq/r2.gz; RDCMD=zcat; return 0
  fi
  rm -f fastq/r1.gz fastq/r2.gz
  if info=\$(ena_lookup "\$run"); then
    urls=\$(printf '%s' "\$info" | cut -f1)
    bytes=\$(printf '%s' "\$info" | cut -f2)
    md5s=\$(printf '%s' "\$info" | cut -f3)
    u1=\${urls%%;*}; u2=\${urls##*;}
    # A single-file (unpaired) run makes both of these the SAME url -- which
    # would silently align one mate against itself. Only accept a real pair.
    if [ -n "\$u1" ] && [ "\$u1" != "\$u2" ]; then
      b1=\${bytes%%;*}; b2=\${bytes##*;}
      m1=\${md5s%%;*};  m2=\${md5s##*;}
      if get_one "https://\$u1" fastq/r1.gz "\$b1" "\$m1" \
      && get_one "https://\$u2" fastq/r2.gz "\$b2" "\$m2"; then
        R1=fastq/r1.gz; R2=fastq/r2.gz; RDCMD=zcat; return 0
      fi
      echo "[endo]   ENA transfer did not verify"
    else
      echo "[endo]   ENA reports no paired FASTQ for \$run"
    fi
  else
    echo "[endo]   ENA metadata lookup failed for \$run"
  fi
  rm -f fastq/r1.gz fastq/r2.gz
  get_from_sra "\$run"
}

# ---- manifest ----
cd \$WORK
gcloud storage cp "\$BUCKET/input/manifest.tsv" manifest.tsv --quiet
gcloud storage cp "\$BUCKET/input/summarize_te_timepoints.py" . --quiet

# ---- per sample: stream FASTQ from ENA, align, count, delete ----
mark align
# Preemption, the max-run-duration cap, or any crash must not throw away samples
# that already finished. Per-sample counts are uploaded the moment they complete,
# so pull them back onto a fresh VM and let the skip check below treat them as
# done. Without this, being preempted at sample 13 costs all 13 -- and a 14-sample
# run is ~10 h, which is a long time to gamble on nothing going wrong.
gcloud storage cp "\$BUCKET/results/locus/*.featureCounts.txt" locus/ --quiet 2>/dev/null || true
gcloud storage cp "\$BUCKET/results/counts/*.cntTable" counts/ --quiet 2>/dev/null || true
NRESUME=\$(ls locus/*.featureCounts.txt 2>/dev/null | wc -l | tr -d ' ')
if [ "\${NRESUME:-0}" -gt 0 ]; then
  echo "[endo] resuming: \$NRESUME sample(s) already counted, restored from the bucket"
fi
: > failed_samples.txt
DL_FAILS=0
# NOTE: process substitution, not "tail | while". A pipeline puts the loop in a
# subshell, where DL_FAILS cannot persist and fail() cannot stop the run.
while IFS=\$'\t' read -r sample tp run rest; do
  [ -z "\${run:-}" ] && continue
  [ -s "locus/\${sample}.featureCounts.txt" ] && { echo "[endo] \$sample done"; continue; }
  echo "[endo] === \$sample (\$run) ==="
  rm -f fastq/r1.gz fastq/r2.gz fastq/r1.fq fastq/r2.fq "fastq/\${run}"*

  if ! fetch_pair "\$run"; then
    echo "[endo] \$sample: FASTQ unavailable from ENA AND the SRA mirror"
    echo "\$sample download" >> failed_samples.txt
    DL_FAILS=\$(( DL_FAILS + 1 ))
    # Do not spend 40 minutes per sample rediscovering the same block. This is
    # what turned one network problem into a 5-hour, 14-sample, zero-output run.
    [ "\$DL_FAILS" -ge 3 ] && fail "3 consecutive samples failed to download from both
  ENA and the AWS SRA mirror. That is a network/throttling problem, not a
  per-sample one, and the remaining samples would fail the same way.
  Re-run later (references and the STAR index are cached, so it restarts in
  minutes), or seed \$BUCKET/cache/fastq/ with <RUN>_1.fastq.gz / <RUN>_2.fastq.gz."
    continue
  fi
  DL_FAILS=0

  # STAR's exit status is authoritative. Last run ignored it and counted the
  # 8 KB partial BAM that STAR leaves behind when it dies mid-write.
  if ! STAR --runThreadN \$THREADS --genomeDir ref/star_hg38 \
       --readFilesIn "\$R1" "\$R2" --readFilesCommand \$RDCMD \
       --outSAMtype BAM Unsorted --outFileNamePrefix bam/\${sample}. \
       --outFilterMultimapNmax 100 --winAnchorMultimapNmax 100 --outSAMprimaryFlag AllBestScore; then
    echo "[endo] \$sample: STAR exited non-zero -- refusing to count a partial BAM"
    echo "\$sample star" >> failed_samples.txt
    rm -f bam/\${sample}.Aligned.out.bam "\$R1" "\$R2"; continue
  fi
  BAM=bam/\${sample}.Aligned.out.bam
  # STAR can also exit 0 having read almost nothing (a truncated gz that happens
  # to end on a record boundary). Three samples did exactly that: "finished
  # mapping" in under a second, then featureCounts found no read pairs.
  NIN=\$(awk -F'\t' '/Number of input reads/{gsub(/[ \t]/,"",\$2); print \$2}' \
        bam/\${sample}.Log.final.out 2>/dev/null); NIN=\${NIN:-0}
  if [ ! -s "\$BAM" ] || [ "\$NIN" -lt 100000 ]; then
    echo "[endo] \$sample: STAR read only \$NIN input reads -- input was truncated, skipping"
    echo "\$sample truncated(\$NIN reads)" >> failed_samples.txt
    rm -f "\$BAM" "\$R1" "\$R2"; continue
  fi
  echo "[endo] \$sample: STAR input reads \$NIN"

  # Both featureCounts runs are load-bearing; a failure must not reach "OK".
  # Delete the stub on failure so it cannot be mistaken for a completed sample
  # by the resume check above or counted in NOK below.
  if ! featureCounts -a ref/hg38_rmsk_TE.gtf -o locus/\${sample}.featureCounts.txt \
      -M --fraction -O -T \$THREADS -p --countReadPairs -s 0 -t exon -g transcript_id "\$BAM"; then
    echo "[endo] \$sample: locus featureCounts FAILED"
    rm -f locus/\${sample}.featureCounts.txt
    echo "\$sample featurecounts-locus" >> failed_samples.txt
    rm -f "\$BAM" "\$R1" "\$R2"; continue
  fi
  if ! featureCounts -a ref/hg38_rmsk_TE.gtf -o counts/\${sample}.subfam.txt \
      -M --fraction -O -T \$THREADS -p --countReadPairs -s 0 -t exon -g gene_id "\$BAM"; then
    echo "[endo] \$sample: subfamily featureCounts FAILED"
    rm -f locus/\${sample}.featureCounts.txt counts/\${sample}.subfam.txt
    echo "\$sample featurecounts-subfam" >> failed_samples.txt
    rm -f "\$BAM" "\$R1" "\$R2"; continue
  fi
  awk 'BEGIN{FS=OFS="\t"} /^#/{next} \$1=="Geneid"{print "gene/TE","count";next} NF>1{printf "%s\t%.0f\n",\$1,\$NF}' \
      counts/\${sample}.subfam.txt > counts/\${sample}.cntTable
  rm -f "\$BAM" "\$R1" "\$R2"
  gcloud storage cp locus/\${sample}.featureCounts.txt "\$BUCKET/results/locus/" --quiet || true
  [ -s counts/\${sample}.cntTable ] && gcloud storage cp counts/\${sample}.cntTable "\$BUCKET/results/counts/" --quiet || true
  echo "[endo] \$sample OK (\$NIN reads counted)"
done < <(tail -n +2 manifest.tsv)

NOK=\$(ls locus/*.featureCounts.txt 2>/dev/null | wc -l)
# wc -l, not \`grep -c . || echo 0\`: grep -c prints "0" AND exits 1 on an empty
# file, so the || fires too and the variable becomes "0\n0", which then blows up
# every arithmetic test that touches it.
NFAIL=\$(wc -l < failed_samples.txt 2>/dev/null | tr -d ' '); NFAIL=\${NFAIL:-0}
echo "[endo] samples with counts: \$NOK   failed: \$NFAIL"
if [ "\$NFAIL" -gt 0 ]; then
  echo "[endo] per-sample failures:"; sed 's/^/[endo]   /' failed_samples.txt
fi
[ "\$NOK" -gt 0 ] || fail "no sample produced counts -- nothing to merge (per-sample reasons are in the log)"

# ---- merge + summarise ----
mark merge
python3 - "\$WORK/locus" "\$WORK/manifest.tsv" "\$FILTER" > locus_expression.tsv <<'PYEOF'
import sys, os, csv, re
locdir, manifest, filt = sys.argv[1], sys.argv[2], re.compile(sys.argv[3])
cells = [l.split("\t")[0] for l in open(manifest).read().splitlines()[1:] if l.strip()]
per, meta = {}, {}
for cell in cells:
    fc = os.path.join(locdir, f"{cell}.featureCounts.txt")
    if not os.path.exists(fc):
        sys.stderr.write(f"missing {fc}\n"); continue
    rows = [r for r in csv.reader(open(fc), delimiter="\t") if r and not r[0].startswith("#")]
    hdr = rows[0]; ci = len(hdr) - 1
    counts = {}
    for r in rows[1:]:
        try: c = float(r[ci])
        except (ValueError, IndexError): continue
        counts[r[0]] = c
        meta.setdefault(r[0], (r[1].split(";")[0], r[2].split(";")[0], r[3].split(";")[-1]))
    lib = sum(counts.values()) or 1.0
    for gid, c in counts.items():
        if filt.search(gid): per.setdefault(gid, {})[cell] = c / lib * 1e6
out = csv.writer(sys.stdout, delimiter="\t")
out.writerow(["locus","chrom","start","end"] + [f"cpm_{c}" for c in cells] + ["cpm_mean"])
for gid in sorted(per):
    ch, st, en = meta.get(gid, ("","",""))
    vals = [per[gid].get(c, 0.0) for c in cells]
    out.writerow([gid, ch, st, en] + [f"{v:.4f}" for v in vals] + [f"{sum(vals)/len(vals):.4f}"])
sys.stderr.write(f"[endo] merged {len(per)} loci x {len(cells)} samples\n")
PYEOF

python3 summarize_te_timepoints.py --counts \$WORK/counts --manifest manifest.tsv \
    --families \$FAMILIES --sample-col 1 --group-col 2 \
    --out te_timepoint_summary.tsv || true

# Gate on real content: a header-only file is a failure, not a result.
LOCI=\$(wc -l < locus_expression.tsv 2>/dev/null | tr -d ' '); LOCI=\$(( \${LOCI:-1} - 1 ))
echo "[endo] locus_expression.tsv data rows: \$LOCI"
[ "\$LOCI" -gt 0 ] || fail "locus_expression.tsv has no data rows (samples counted: \$NOK)"

mark upload
gcloud storage cp locus_expression.tsv "\$BUCKET/results/" --quiet
[ -s te_timepoint_summary.tsv ] && gcloud storage cp te_timepoint_summary.tsv "\$BUCKET/results/" --quiet
NTOT=\$(wc -l < manifest.tsv | tr -d ' '); NTOT=\$(( \${NTOT:-1} - 1 ))
{ echo "\$LOCI loci from \$NOK/\$NTOT samples"
  [ "\$NFAIL" -gt 0 ] && { echo "\$NFAIL sample(s) failed:"; sed 's/^/  /' failed_samples.txt; }
} | gcloud storage cp - "\$BUCKET/results/RUN_SUMMARY.txt" --quiet || true
mark DONE
echo "[endo] complete \$(date)"
# trap 'finish' deletes the instance here
STARTUP_EOF

# Walk machine types x zones until one has capacity. A single-zone create turns a
# transient, purely local shortage into "run cancelled" -- which is what
# ZONE_RESOURCE_POOL_EXHAUSTED did on 2026-08-13.
# Machine-major, NOT zone-major: the preferred type is tried in every zone before
# falling back to a smaller one. Zone is free to change (the bucket is US
# multi-region), but dropping 16 vCPU -> 8 doubles the alignment time, so never
# trade cores away just to stay in us-central1-a.
echo "  creating VM (trying zones until one has capacity) ..."
CREATED=""
for m in $MACHINES; do
  for z in $ZONES; do
    printf '    %-16s %-18s ... ' "$z" "$m"
    if out=$(gcloud compute instances create "$VM" \
        --project="$PROJECT" --zone="$z" --machine-type="$m" \
        --provisioning-model="$PROVISIONING" --instance-termination-action=DELETE \
        --max-run-duration="${MAX_HOURS}h" \
        --image-family=debian-12 --image-project=debian-cloud \
        --boot-disk-size=50GB --boot-disk-type=pd-balanced \
        --create-disk="name=${VM}-work,size=${DISK_GB}GB,type=pd-balanced,auto-delete=yes" \
        --scopes=cloud-platform \
        --metadata-from-file=startup-script="$STARTUP" \
        --labels=purpose=endo-expression 2>&1); then
      echo "CREATED"; CREATED="yes"; ZONE="$z"; MACHINE="$m"; break 2
    fi
    case "$out" in
      # Quota is a project limit, not a shortage. Other zones in the SAME region
      # will fail identically, so name it rather than hiding it behind "no
      # capacity" -- a trial project can easily be capped below 16 vCPU.
      *QUOTA_EXCEEDED*|*"Quota "*exceeded*)
        echo "QUOTA (project limit, not capacity)"; QUOTA_HIT=1 ;;
      *ZONE_RESOURCE_POOL_EXHAUSTED*|*"does not have enough resources"*|*"was not found"*|*"not available in"*)
        echo "no capacity" ;;
      *"already exists"*)
        echo; die "$VM already exists (in some zone). 'status' to inspect, 'down' to remove it." ;;
      *)
        echo "failed"; echo "$out" | sed 's/^/      /' | tail -6 ;;
    esac
  done
done
rm -f "$STARTUP"
if [ -z "$CREATED" ]; then
  if [ -n "${QUOTA_HIT:-}" ]; then
    die "blocked by QUOTA, not capacity -- more zones will not help.
  Check the CPU quota for this project and raise it (free on a trial account):
      https://console.cloud.google.com/iam-admin/quotas?project=$PROJECT
  Filter for 'CPUs' in the regions above. n2-standard-16 needs 16 vCPU in one
  region. Nothing was created and nothing is billing."
  fi
  die "no zone had capacity for any of: $MACHINES
  Tried: $(echo $ZONES | tr '\n' ' ')
  Nothing was created and nothing is billing. Options:
    * wait and re-run -- capacity frees up, often within the hour
    * widen the search:  ZONES='...' MACHINES='...' bash $0 up
  Any replacement machine type MUST have >= 64 GB RAM: STAR holds the ~31 GB
  hg38 index in memory while aligning, so a 32 GB instance will OOM."
fi
echo "  landed in $ZONE on $MACHINE"

cat <<EOF

  VM is up and working. It will delete itself when finished.

  watch     : bash scripts/gcp_expression.sh status
  live log  : bash scripts/gcp_expression.sh logs
  results   : bash scripts/gcp_expression.sh fetch     (when status says READY)

  >>> STOP IT ANY TIME:  bash scripts/gcp_expression.sh down <<<

  Guaranteed stops, in order of independence:
    1. GCP deletes this VM after ${MAX_HOURS}h  (--max-run-duration, server-side)
    2. the VM deletes itself when the pipeline ends
    3. an in-guest watchdog powers it off 15 min before the deadline
    4. 'down', any time, safe to repeat
EOF
