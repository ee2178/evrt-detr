#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=Downloading_Gen4_Dataset
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Activate Conda environment
source ~/.bashrc                   # Ensure conda is available
conda activate evrt-detr      # Replace with your actual env name

cd ~/scratch/ee2178/evrt_detr

wget -c "https://kdrive.infomaniak.com/2/app/975517/share/613e58d5-3c60-4c52-a57f-a5871c00167f/archive/7e14df86-8bcb-4c27-9671-3f858f2af42e/download" -O gen4.7z

7z x gen4.7z
