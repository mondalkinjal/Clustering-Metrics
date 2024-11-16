#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH --ntasks=40
#SBATCH --mem-per-cpu=81920

. "/scratch/zt1/project/energybio/user/kinjal14/miniconda/etc/profile.d/conda.sh"
conda activate ML
#python3 encoded_microstates.py
python3 encoded_macrostates.py
end