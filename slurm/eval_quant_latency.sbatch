#!/bin/bash
#SBATCH --account=torch_pr_89_general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --job-name=gen1_evrt-detr-int8-lrd-latency
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/evrt_detr_int8_latency.out
#SBATCH --error=logs/evrt_detr_int8_latency.err

# Activate Conda environment
source ~/.bashrc                   # Ensure conda is available
conda activate trt124      # Replace with your actual env name

cd ~/scratch/ee2178/evrt_detr

PYTHONPATH=$PWD python3 scripts/quant_trt/model_quant_test_trtexec.py      models/gen1_lrd/gen1_r0.5_s2      --device cuda      --batch-size 1      --input-shape 20,256,320      --enable-profiling      --use-cuda-graph
mkdir quant_logs_r05
mv model_* quant_logs_r05

PYTHONPATH=$PWD python3 scripts/quant_trt/model_quant_test_trtexec.py      models/gen1_lrd/gen1_r0.6_s2      --device cuda      --batch-size 1      --input-shape 20,256,320      --enable-profiling      --use-cuda-graph
mkdir quant_logs_r06
mv model_* quant_logs_r06

PYTHONPATH=$PWD python3 scripts/quant_trt/model_quant_test_trtexec.py      models/gen1_lrd/gen1_r0.7_s2      --device cuda      --batch-size 1      --input-shape 20,256,320      --enable-profiling      --use-cuda-graph
mkdir quant_logs_r07
mv model_* quant_logs_r07

PYTHONPATH=$PWD python3 scripts/quant_trt/model_quant_test_trtexec.py      models/gen1_lrd/gen1_r0.8_s2      --device cuda      --batch-size 1      --input-shape 20,256,320      --enable-profiling      --use-cuda-graph
mkdir quant_logs_r08
mv model_* quant_logs_r08

PYTHONPATH=$PWD python3 scripts/quant_trt/model_quant_test_trtexec.py      models/gen1_lrd/gen1_r0.9_s2      --device cuda      --batch-size 1      --input-shape 20,256,320      --enable-profiling      --use-cuda-graph
mkdir quant_logs_r09
mv model_* quant_logs_r09
