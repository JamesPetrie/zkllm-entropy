#!/bin/bash
#SBATCH --job-name=zkllm-proofs
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/sharefs/user50/zk/logs/proofs-%j.out
#SBATCH --error=/mnt/sharefs/user50/zk/logs/proofs-%j.err

set -e

export CONDA_PREFIX=/mnt/sharefs/user50/miniconda3/envs/zkllm-env
export PATH=$CONDA_PREFIX/bin:$PATH

cd /mnt/sharefs/user50/zk/zkllm-ccs2024

mkdir -p /mnt/sharefs/user50/zk/logs

# seq_len >= 1024 required: zkSoftmax needs seq_len^2 >= 2^20
python python/run_proofs.py --model_size 7 --seq_len 1024 --num_layers 32
