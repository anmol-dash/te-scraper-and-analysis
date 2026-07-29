#!/usr/bin/env bash
# hervk_normalize_plot.sh
# Read the three TEcount tables, CPM-normalize (raw counts are NOT comparable:
# the libraries differ ~35x in depth), and write:
#   $WORK/hervk_cpm.tsv        wide: raw + CPM per subfamily per line
#   $WORK/hervk_cpm_long.tsv   long: subfamily, cell_line, raw, cpm  (for ggplot)
#   $WORK/hervk_cpm.png        grouped bar chart (log10 CPM) of the key subfamilies
# All ERVK-family features (HERVK*-int, LTR5/LTR5A/LTR5B/LTR5_Hs) are included.
#
# Runs inside the tetranscripts container (Python + R/ggplot2 both present).
#   bash scripts/hervk_normalize_plot.sh        (runs locally on login node; no bsub needed)
set -euo pipefail

WORK=${WORK:-$HOME/hervk_ccle}
CONT=${CONT:-$WORK/containers}   # override to reuse another study's containers
TE_SIF=$CONT/tetranscripts.sif
COUNTS=$WORK/counts
sing() { singularity exec -B "$WORK" "$TE_SIF" "$@"; }

# ---- 1. normalize (pure-Python stdlib; the numbers are the real deliverable) ----
sing python - "$COUNTS" "$WORK" <<'PY'
import sys, glob, os, csv
counts_dir, work = sys.argv[1], sys.argv[2]
# cell line -> {feature: count}, and library size = sum of ALL feature counts
libs, libsize = {}, {}
for f in sorted(glob.glob(os.path.join(counts_dir, "*.cntTable"))):
    cl = os.path.basename(f).split(".")[0]
    d = {}
    tot = 0.0
    for row in csv.reader(open(f), delimiter="\t"):
        if len(row) < 2: continue
        try: v = float(row[1])
        except ValueError: continue   # header
        d[row[0]] = v; tot += v
    libs[cl] = d; libsize[cl] = tot
cells = sorted(libs)   # MCF-7, PANC-1, PC-3

# ERVK-family subfamilies (feature = "name:family:class")
ervk = set()
for cl in cells:
    for feat in libs[cl]:
        p = feat.split(":")
        if len(p) == 3 and p[1] == "ERVK":
            ervk.add(feat)
def sub(feat): return feat.split(":")[0]
ervk = sorted(ervk, key=sub)

cpm = lambda cl, feat: libs[cl].get(feat, 0.0) / libsize[cl] * 1e6

# wide table
with open(os.path.join(work, "hervk_cpm.tsv"), "w") as out:
    hdr = ["subfamily"] + [f"{c}_raw" for c in cells] + [f"{c}_CPM" for c in cells]
    out.write("\t".join(hdr) + "\n")
    for feat in ervk:
        raw = [f"{libs[c].get(feat,0):.0f}" for c in cells]
        cp  = [f"{cpm(c,feat):.3f}" for c in cells]
        out.write("\t".join([sub(feat)] + raw + cp) + "\n")
    # HERVK-int-only family aggregate + full ERVK aggregate
    for label, keep in [("ALL_ERVK", lambda f: True),
                        ("HERVK_internal", lambda f: sub(f).startswith("HERVK")),
                        ("LTR5_group", lambda f: sub(f).startswith("LTR5"))]:
        sel = [f for f in ervk if keep(f)]
        raw = [f"{sum(libs[c].get(f,0) for f in sel):.0f}" for c in cells]
        cp  = [f"{sum(cpm(c,f) for f in sel):.3f}" for c in cells]
        out.write("\t".join([label] + raw + cp) + "\n")

# long table for ggplot
with open(os.path.join(work, "hervk_cpm_long.tsv"), "w") as out:
    out.write("subfamily\tcell_line\traw\tcpm\n")
    for feat in ervk:
        for c in cells:
            out.write(f"{sub(feat)}\t{c}\t{libs[c].get(feat,0):.0f}\t{cpm(c,feat):.4f}\n")

print("library sizes (total assigned counts):")
for c in cells: print(f"  {c:8s} {libsize[c]:,.0f}")
print("\nHERVK subfamily CPM (reads per million assigned):")
key = ["HERVK-int","HERVK9-int","HERVK11-int","HERVK22-int","LTR5_Hs","LTR5A","LTR5B","LTR5"]
print("  " + "subfamily".ljust(12) + "".join(c.rjust(12) for c in cells))
for k in key:
    feat = next((f for f in ervk if sub(f)==k), None)
    if not feat: continue
    print("  " + k.ljust(12) + "".join(f"{cpm(c,feat):12.2f}" for c in cells))
print("\nwrote hervk_cpm.tsv, hervk_cpm_long.tsv")
PY

# ---- 2. figure (R/ggplot2 inside the container; skip gracefully if absent) ----
if sing Rscript --version >/dev/null 2>&1; then
  sing Rscript - "$WORK" <<'RS'
args <- commandArgs(trailingOnly=TRUE); work <- args[1]
suppressMessages(library(ggplot2))
d <- read.delim(file.path(work,"hervk_cpm_long.tsv"))
keep <- c("HERVK-int","HERVK9-int","HERVK11-int","HERVK22-int","LTR5_Hs","LTR5A","LTR5B","LTR5")
d <- d[d$subfamily %in% keep,]
d$subfamily <- factor(d$subfamily, levels=keep)
d$cpm_floor <- pmax(d$cpm, 0.05)          # floor so log10 shows near-zero bars
ncell <- length(unique(d$cell_line))
if (ncell <= 5) {                          # few samples -> grouped bars
  p <- ggplot(d, aes(subfamily, cpm_floor, fill=cell_line)) +
    geom_col(position=position_dodge(width=0.8), width=0.72) +
    scale_y_log10() +
    labs(title="HERVK subfamily expression", x=NULL, y="CPM (log10)", fill="sample") +
    theme_minimal(base_size=13) +
    theme(axis.text.x=element_text(angle=35, hjust=1),
          panel.grid.major.x=element_blank(), legend.position="top")
  w <- 10; h <- 6
} else {                                   # many samples -> heatmap
  p <- ggplot(d, aes(cell_line, subfamily, fill=log10(cpm_floor))) +
    geom_tile(color="grey90") +
    scale_fill_viridis_c(option="magma") +
    labs(title="HERVK subfamily expression (log10 CPM)", x=NULL, y=NULL, fill="log10 CPM") +
    theme_minimal(base_size=12) +
    theme(axis.text.x=element_text(angle=45, hjust=1))
  w <- max(8, ncell*0.5); h <- 6
}
ggsave(file.path(work,"hervk_cpm.png"), p, width=w, height=h, dpi=150)
cat("wrote", file.path(work,"hervk_cpm.png"), "(", ncell, "samples )\n")
RS
else
  echo "Rscript not in container -> skipped PNG; hervk_cpm.tsv has the numbers."
fi
echo "done. Inspect: $WORK/hervk_cpm.tsv  and  $WORK/hervk_cpm.png"
