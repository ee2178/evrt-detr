#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=Download_Unzip_Gen1_Dataset
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Activate Conda environment
source ~/.bashrc                   # Ensure conda is available
conda activate evrt-detr      # Replace with your actual env name

cd ~/scratch/ee2178/evrt_detr
# download gen1 dataset
# wget https://kdrive.infomaniak.com/2/app/975517/share/87a11380-2464-4ee3-be18-c847a1956b3b/files/41/download -O gen1test.7z

# 7z x gen1test.7z

python3 scripts/data/psee_to_frames.py detection_dataset_duration_60s_ratio_1.0 data/gen1 -n 10 --mangling rvt-gen1 -d 50ms --workers 16

