#!/bin/bash
#SBATCH --job-name=iap_consensus5
#SBATCH --output=iap_consensus5.%j.out
#SBATCH --error=iap_consensus5.%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
## --- adjust for your cluster if needed ---
## #SBATCH --partition=<your_partition>
## #SBATCH --account=<your_account>
set -euo pipefail

# Make MAFFT available. Try the common module names, else fall back to conda.
module load mafft 2>/dev/null || module load MAFFT 2>/dev/null \
  || module load mafft/7.490 2>/dev/null || true
command -v mafft >/dev/null 2>&1 || { conda activate "${CONDA_ENV:-base}" 2>/dev/null || true; }
command -v mafft >/dev/null 2>&1 || { echo "ERROR: mafft not on PATH — load/install it"; exit 1; }
echo "mafft: $(command -v mafft)"

python compute_consensus5_distances.py \
    --input  clusters5.csv \
    --out    consensus_distance_5clust.csv \
    --threads "${SLURM_CPUS_PER_TASK:-8}"
