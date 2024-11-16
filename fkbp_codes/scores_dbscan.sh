#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH --ntasks=30 
#SBATCH --mem-per-cpu=10000
#SBATCH --job-name="clust"

. "/scratch/zt1/project/energybio/user/kinjal14/miniconda/etc/profile.d/conda.sh"
conda activate ML
python3 scores_dbscan.py
end
