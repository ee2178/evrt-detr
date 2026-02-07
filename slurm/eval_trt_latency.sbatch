#!/bin/bash
#SBATCH --account=torch_pr_89_general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --job-name=gen1_evrt-detr-int8-latency
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/evrt_detr_int8_latency.out
#SBATCH --error=logs/evrt_detr_int8_latency.err

# Activate Conda environment
source ~/.bashrc                   # Ensure conda is available
conda activate trt124      # Replace with your actual env name

cd ~/scratch/ee2178/evrt_detr

python eval_latency_trt.py models/gen1_backup/video_evrtdetr_presnet50/   --engine-dir trt_engines_int8   --split test --steps 50 --data-name video --data-path gen1/gen1_preproc_npz/ --hotswap-int8
