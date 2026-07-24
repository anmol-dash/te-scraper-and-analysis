#!/usr/bin/env bash
# download_fastq_ena.sh  --  resilient FASTQ download from EBI ENA.
# The big CCLE runs (PC-3 ~87 GB, PANC-1 ~64 GB gz) download slowly and ENA
# drops the TLS connection mid-transfer (SSL_ERROR_SYSCALL). This resumes
# (curl -C -) and retries until the local file reaches ENA's reported size,
# so it finishes unattended across any number of drops.
#
# Run detached on the login node so a disconnect doesn't kill it:
#   nohup bash scripts/download_fastq_ena.sh > $HOME/hervk_ccle/download.log 2>&1 &
#   tail -f $HOME/hervk_ccle/download.log
set -uo pipefail   # NB: no -e; the retry loop handles curl's non-zero exits

WORK=${WORK:-$HOME/hervk_ccle}
MANIFEST=$(cd "$(dirname "$0")" && pwd)/hervk_ccle_manifest.tsv
mkdir -p "$WORK/fastq"

fetch() {  # $1=url  $2=out  $3=expected_bytes
  local url="$1" out="$2" want="$3" have attempt
  for attempt in $(seq 1 200); do
    have=0; [ -s "$out" ] && have=$(stat -c%s "$out" 2>/dev/null || echo 0)
    if [ -n "$want" ] && [ "$want" -gt 0 ] && [ "$have" -ge "$want" ]; then
      echo "  OK $(basename "$out") ($have bytes)"; return 0
    fi
    echo "  [$attempt] $(basename "$out"): have $have / want ${want:-?}"
    # --speed-* aborts a stalled socket so we retry-resume instead of hanging;
    # --retry-all-errors covers the ENA TLS resets.
    curl -fSL -C - -o "$out" "$url" \
         --retry 15 --retry-delay 5 --retry-all-errors \
         --connect-timeout 30 --speed-time 120 --speed-limit 2000 && true
  done
  have=0; [ -s "$out" ] && have=$(stat -c%s "$out" 2>/dev/null || echo 0)
  if [ -n "$want" ] && [ "$want" -gt 0 ] && [ "$have" -lt "$want" ]; then
    echo "  FAILED $(basename "$out") after 200 attempts ($have/$want)"; return 1
  fi
}

rc=0
tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r cl smp run rest; do
  [ -z "${run:-}" ] && continue
  info=$(curl -s "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=${run}&result=read_run&fields=fastq_ftp,fastq_bytes&format=tsv" | awk 'NR==2')
  urls=$(echo "$info" | cut -f2); bytes=$(echo "$info" | cut -f3)
  [ -z "$urls" ] && { echo "no ENA FASTQ for $run"; exit 1; }
  IFS=';' read -r u1 u2 <<< "$urls"
  IFS=';' read -r b1 b2 <<< "$bytes"
  echo "== $cl $run =="
  fetch "https://${u1}" "$WORK/fastq/$(basename "$u1")" "${b1:-0}" || rc=1
  fetch "https://${u2}" "$WORK/fastq/$(basename "$u2")" "${b2:-0}" || rc=1
done

echo "== final sizes =="
ls -lh "$WORK/fastq/" 2>/dev/null || true
echo "download_fastq_ena done (rc via subshell; verify sizes above match ENA)"
