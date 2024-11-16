#!/bin/bash
#SBATCH -t 8:00:00
#SBATCH --ntasks=40 
#SBATCH --mem-per-cpu=100000
#SBATCH --job-name="clust"

. "/scratch/zt1/project/energybio/user/kinjal14/miniconda/etc/profile.d/conda.sh"
conda activate ML
python3 main_code_microstates.py
end
