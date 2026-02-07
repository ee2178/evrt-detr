#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --job-name=gen1_evrt-detr-presnet50-w8a8-eval
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/evrt_detr_presnet50_w8a8.out
#SBATCH --error=logs/evrt_detr_presnet50_w8a8.err

# Activate Conda environment
source ~/.bashrc                   # Ensure conda is available
conda activate evrt-detr      # Replace with your actual env name

cd ~/scratch/ee2178/evrt_detr

# No longer need to quantize model, we did that already
# Quantize w8 model
# python3 quantize.py models/gen1/video_evrtdetr_presnet50_w2 --num-bits 2


# Try with 8 bit activation quantization
python3 scripts/eval_model_video.py models/gen1/video_evrtdetr_presnet50_w8 --data-name video
