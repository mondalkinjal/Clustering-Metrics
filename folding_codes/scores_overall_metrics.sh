#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH --ntasks=40
#SBATCH --mem-per-cpu=4096

. "/scratch/zt1/project/energybio/user/kinjal14/miniconda/etc/profile.d/conda.sh"
conda activate ML
python3 scores_overall_metrics.py
#python3 encoded_macrostates.py
end
