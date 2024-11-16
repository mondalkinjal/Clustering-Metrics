#!/bin/bash
#SBATCH -t 3:00:00
#SBATCH --ntasks=50 
#SBATCH --mem-per-cpu=100000
#SBATCH --job-name="clust"

. "/scratch/zt1/project/energybio/user/kinjal14/miniforge/etc/profile.d/conda.sh"
conda activate GNN
python3 scores_microstates_dbscan.py
end
