#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=Unzipping_Gen4_Dataset
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Activate Conda environment
source ~/.bashrc                   # Ensure conda is available
conda activate evrt-detr      # Replace with your actual env name

cd ~/scratch/ee2178/evrt_detr/gen4_dataset

7z x "*.7z"
