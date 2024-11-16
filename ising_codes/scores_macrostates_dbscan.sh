#!/bin/bash
#SBATCH -t 1:00:00
#SBATCH --ntasks=30 
#SBATCH --mem-per-cpu=100000
#SBATCH --job-name="clust"

. "/scratch/zt1/project/energybio/user/kinjal14/miniforge/etc/profile.d/conda.sh"
conda activate GNN
python3 scores_macrostates_dbscan.py
end
