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
#   1. SPOT instance with --instance-termination-action=DELETE
#      (cheapest tier; if preempted the VM is deleted, not left stopped)
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
ZONE=${ZONE:-us-central1-a}
VM=${VM:-endo-expression}
# n2-standard-8: 8 vCPU / 32 GB. STAR needs ~30 GB resident for hg38, so 32 GB
# is the smallest shape that works; the 8 vCPUs are what make the 14 alignments
# finish in hours rather than a day.
MACHINE=${MACHINE:-n2-standard-8}
DISK_GB=${DISK_GB:-200}          # hg38 3 + STAR index 30 + fastq 65 + slack
MAX_HOURS=${MAX_HOURS:-10}       # hard server-side deletion deadline
BUCKET=${BUCKET:-gs://${PROJECT}-endo-expression}
MANIFEST=${MANIFEST:-$REPO/scripts/srp090091_manifest.tsv}
FAMILIES=${FAMILIES:-"LTR66 LTR10G"}
FILTER=${FILTER:-'^(LTR66|LTR10G)_dup'}
LOCAL_OUT=${LOCAL_OUT:-$HOME/endo/expression}

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
die() { echo "FATAL: $*" >&2; exit 1; }
[ -n "$PROJECT" ] || die "no GCP project set (gcloud config set project ...)"

cmd=${1:-help}

# ---------------------------------------------------------------- estimate --
if [ "$cmd" = "estimate" ]; then
  say "cost estimate (nothing is created)"
  python3 - "$MACHINE" "$DISK_GB" "$MAX_HOURS" <<'PY'
import sys
machine, disk, maxh = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
# us-central1 spot list prices, Aug 2026 (approximate, for sizing only)
spot = {"n2-standard-8": 0.097, "n2-standard-16": 0.194, "e2-standard-8": 0.080}
rate = spot.get(machine, 0.10)
disk_hr = disk * 0.10 / 730          # pd-balanced ~$0.10/GB-month
run = 6.0                             # realistic runtime
print(f"  machine {machine} (spot)      ${rate:.3f}/h")
print(f"  disk    {disk} GB pd-balanced  ${disk_hr:.3f}/h")
print(f"  ingress from ENA               $0 (inbound is free)")
print(f"  egress  results ~2 MB          ~$0.00")
print()
print(f"  expected run ~{run:.0f} h  ->  ${(rate + disk_hr) * run:.2f}")
print(f"  worst case  {maxh:.0f} h  ->  ${(rate + disk_hr) * maxh:.2f}  (hard cap: GCP deletes the VM)")
print()
print("  Free trial credit is $300, so the ceiling above is ~1% of it.")
PY
  echo
  echo "  Stops in force once running:"
  echo "    * GCP deletes the VM after ${MAX_HOURS}h no matter what (--max-run-duration)"
  echo "    * the VM deletes itself when the pipeline finishes"
  echo "    * spot preemption deletes rather than stops it"
  echo "    * bash scripts/gcp_expression.sh down   -- any time"
  exit 0
fi

# -------------------------------------------------------------------- down --
# Deliberately first among the action verbs, and safe to run at any time.
if [ "$cmd" = "down" ] || [ "$cmd" = "stop" ]; then
  say "down -- deleting everything"
  echo "  project: $PROJECT   zone: $ZONE"
  if gcloud compute instances describe "$VM" --zone "$ZONE" --project "$PROJECT" >/dev/null 2>&1; then
    echo "  deleting VM $VM (and its boot disk) ..."
    gcloud compute instances delete "$VM" --zone "$ZONE" --project "$PROJECT" --quiet \
      && echo "  VM deleted"
  else
    echo "  VM $VM: not present (nothing to delete)"
  fi
  # orphaned disks, if the VM was created with a non-auto-delete disk
  for d in $(gcloud compute disks list --project "$PROJECT" --filter="name~^${VM}" \
             --format="value(name)" 2>/dev/null || true); do
    echo "  deleting orphaned disk $d ..."
    gcloud compute disks delete "$d" --zone "$ZONE" --project "$PROJECT" --quiet || true
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
  say "status"
  echo "  project $PROJECT   zone $ZONE   vm $VM"
  gcloud compute instances list --project "$PROJECT" --filter="name=$VM" \
    --format="table(name,status,machineType.basename(),scheduling.provisioningModel,lastStartTimestamp)" \
    2>/dev/null || echo "  (no such VM)"
  echo
  echo "  progress markers in $BUCKET:"
  gcloud storage ls "$BUCKET/progress/" 2>/dev/null | sed 's|.*/|    |' || echo "    (none yet)"
  echo
  echo "  results present?"
  for f in locus_expression.tsv te_timepoint_summary.tsv; do
    if gcloud storage ls "$BUCKET/results/$f" >/dev/null 2>&1; then echo "    $f: READY"
    else echo "    $f: not yet"; fi
  done
  exit 0
fi

# -------------------------------------------------------------------- logs --
if [ "$cmd" = "logs" ]; then
  gcloud compute instances get-serial-port-output "$VM" --zone "$ZONE" --project "$PROJECT" 2>/dev/null \
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
echo "  zone     : $ZONE"
echo "  machine  : $MACHINE (SPOT)  disk ${DISK_GB} GB"
echo "  samples  : $N   families: $FAMILIES"
echo "  hard stop: ${MAX_HOURS}h, enforced by GCP (--max-run-duration)"
echo "  bucket   : $BUCKET"
echo
if gcloud compute instances describe "$VM" --zone "$ZONE" --project "$PROJECT" >/dev/null 2>&1; then
  die "$VM already exists. 'status' to inspect, 'down' to remove it first."
fi
if [ "${2:-}" != "--yes" ]; then
  read -r -p "  Create this VM now? [y/N] " ans
  case "$ans" in y|Y|yes) ;; *) echo "  aborted; nothing created."; exit 0 ;; esac
fi

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

mark() { echo "[endo] STAGE \$1"; echo "\$1 \$(date -Is)" | gcloud storage cp - "\$BUCKET/progress/\$1" --quiet 2>/dev/null || true; }

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
curl -fsSL -o subread.tar.gz https://downloads.sourceforge.net/project/subread/subread-2.0.6/subread-2.0.6-Linux-x86_64.tar.gz
tar xzf subread.tar.gz && cp subread-2.0.6-Linux-x86_64/bin/featureCounts bin/ && chmod +x bin/featureCounts
export PATH=\$WORK/bin:\$PATH
STAR --version; featureCounts -v 2>&1 | head -1

# ---- references ----
mark refs
cd \$WORK/ref
curl -fsSL -O https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz && gunzip -f hg38.fa.gz
curl -fsSL -O https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/genes/hg38.knownGene.gtf.gz && gunzip -f hg38.knownGene.gtf.gz
curl -fsSL -O https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/rmsk.txt.gz
# identical TE GTF construction to scripts/setup_hervk_ccle.sh
zcat rmsk.txt.gz | awk -F'\t' '
  \$12 ~ /Simple_repeat|Low_complexity|Satellite|^RNA\$|rRNA|scRNA|snRNA|srpRNA|tRNA|Unknown|\?/ {next}
  { n[\$11]++;
    printf "%s\trmsk\texon\t%d\t%d\t.\t%s\t.\tgene_id \"%s\"; transcript_id \"%s_dup%d\"; family_id \"%s\"; class_id \"%s\";\n",
           \$6,\$7+1,\$8,\$10,\$11,\$11,n[\$11],\$13,\$12 }' > hg38_rmsk_TE.gtf
echo "[endo] TE GTF \$(wc -l < hg38_rmsk_TE.gtf) lines; target families present:"
awk -F'gene_id "' '{split(\$2,a,"\"");print a[1]}' hg38_rmsk_TE.gtf | grep -cE "^(\${FAMILIES// /|})\$" || true

mark star_index
mkdir -p star_hg38
STAR --runMode genomeGenerate --runThreadN \$THREADS --genomeDir \$WORK/ref/star_hg38 \
     --genomeFastaFiles \$WORK/ref/hg38.fa --sjdbGTFfile \$WORK/ref/hg38.knownGene.gtf \
     --sjdbOverhang 100 --outFileNamePrefix \$WORK/ref/idx_

# ---- manifest ----
cd \$WORK
gcloud storage cp "\$BUCKET/input/manifest.tsv" manifest.tsv --quiet
gcloud storage cp "\$BUCKET/input/summarize_te_timepoints.py" . --quiet

# ---- per sample: stream FASTQ from ENA, align, count, delete ----
mark align
tail -n +2 manifest.tsv | while IFS=\$'\t' read -r sample tp run rest; do
  [ -z "\${run:-}" ] && continue
  [ -s "locus/\${sample}.featureCounts.txt" ] && { echo "[endo] \$sample done"; continue; }
  echo "[endo] === \$sample (\$run) ==="
  info=\$(curl -sS --retry 5 "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=\${run}&result=read_run&fields=fastq_ftp&format=tsv" | awk 'NR==2{print \$2}')
  [ -z "\$info" ] && { echo "[endo] \$sample: no ENA metadata, skipping"; continue; }
  u1=\${info%%;*}; u2=\${info##*;}
  curl -fsSL --retry 10 --retry-all-errors -o fastq/r1.gz "https://\$u1" &
  curl -fsSL --retry 10 --retry-all-errors -o fastq/r2.gz "https://\$u2" &
  wait
  STAR --runThreadN \$THREADS --genomeDir ref/star_hg38 \
       --readFilesIn fastq/r1.gz fastq/r2.gz --readFilesCommand zcat \
       --outSAMtype BAM Unsorted --outFileNamePrefix bam/\${sample}. \
       --outFilterMultimapNmax 100 --winAnchorMultimapNmax 100 --outSAMprimaryFlag AllBestScore
  BAM=bam/\${sample}.Aligned.out.bam
  featureCounts -a ref/hg38_rmsk_TE.gtf -o locus/\${sample}.featureCounts.txt \
      -M --fraction -O -T \$THREADS -p --countReadPairs -s 0 -t exon -g transcript_id "\$BAM"
  featureCounts -a ref/hg38_rmsk_TE.gtf -o counts/\${sample}.subfam.txt \
      -M --fraction -O -T \$THREADS -p --countReadPairs -s 0 -t exon -g gene_id "\$BAM"
  awk 'BEGIN{FS=OFS="\t"} /^#/{next} \$1=="Geneid"{print "gene/TE","count";next} NF>1{printf "%s\t%.0f\n",\$1,\$NF}' \
      counts/\${sample}.subfam.txt > counts/\${sample}.cntTable
  rm -f "\$BAM" fastq/r1.gz fastq/r2.gz
  gcloud storage cp locus/\${sample}.featureCounts.txt "\$BUCKET/results/locus/" --quiet || true
  gcloud storage cp counts/\${sample}.cntTable "\$BUCKET/results/counts/" --quiet || true
  echo "[endo] \$sample OK"
done

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

mark upload
gcloud storage cp locus_expression.tsv te_timepoint_summary.tsv "\$BUCKET/results/" --quiet
mark DONE
echo "[endo] complete \$(date)"
# trap 'finish' deletes the instance here
STARTUP_EOF

echo "  creating VM ..."
gcloud compute instances create "$VM" \
  --project="$PROJECT" --zone="$ZONE" --machine-type="$MACHINE" \
  --provisioning-model=SPOT --instance-termination-action=DELETE \
  --max-run-duration="${MAX_HOURS}h" \
  --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=50GB --boot-disk-type=pd-balanced \
  --create-disk="name=${VM}-work,size=${DISK_GB}GB,type=pd-balanced,auto-delete=yes" \
  --scopes=cloud-platform \
  --metadata-from-file=startup-script="$STARTUP" \
  --labels=purpose=endo-expression
rm -f "$STARTUP"

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
