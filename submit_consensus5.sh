#!/bin/bash
#BSUB -J iap_consensus5
#BSUB -o iap_consensus5.%J.out
#BSUB -e iap_consensus5.%J.err
#BSUB -n 16
#BSUB -R "rusage[mem=32G] span[hosts=1]"
#BSUB -W 8:00
## --- adjust for your cluster if needed ---
## #BSUB -q <your_queue>
## #BSUB -P <your_project>
set -euo pipefail

# Make MAFFT available. Try the common module names, else fall back to conda.
module load mafft 2>/dev/null || module load MAFFT 2>/dev/null \
  || module load mafft/7.490 2>/dev/null || true
command -v mafft >/dev/null 2>&1 || { conda activate "${CONDA_ENV:-base}" 2>/dev/null || true; }
command -v mafft >/dev/null 2>&1 || { echo "ERROR: mafft not on PATH — load/install it"; exit 1; }
echo "mafft: $(command -v mafft)"

# CIAlign (pip package; CLI command is 'CIAlign'). Install to user site if absent.
command -v CIAlign >/dev/null 2>&1 || python -m pip install --user --quiet cialign || true
command -v CIAlign >/dev/null 2>&1 || { echo "ERROR: CIAlign not on PATH — pip install cialign"; exit 1; }
echo "CIAlign: $(command -v CIAlign)"

python compute_consensus5_distances.py \
    --input   clusters5.csv \
    --out     consensus_distance_5clust.csv \
    --aln-dir cleaned_aln \
    --threads "${LSB_DJOB_NUMPROC:-8}"
