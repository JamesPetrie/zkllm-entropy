#!/bin/bash
#SBATCH --job-name=zk-calibrate
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/calibrate-%j.out
#SBATCH --error=logs/calibrate-%j.err

set -e
mkdir -p logs

source /mnt/sharefs/user50/miniconda3/etc/profile.d/conda.sh
conda activate zkllm-env

cd /mnt/sharefs/user50/zk/zkllm-ccs2024

python python/calibrate_sigma.py \
    --model-size 7 \
    --seq-len 512 \
    --n-runs 3 \
    --verbose
