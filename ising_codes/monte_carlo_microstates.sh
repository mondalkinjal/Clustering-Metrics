#!/bin/bash
#SBATCH -t 6:00:00
#SBATCH -n 200
#SBATCH --job-name="clust"

. "/scratch/zt1/project/energybio/user/kinjal14/miniconda/etc/profile.d/conda.sh"
conda activate ML
python3 monte_carlo_microstates.py
end
