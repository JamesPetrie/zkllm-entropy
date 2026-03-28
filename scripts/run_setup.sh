#!/bin/bash
#SBATCH --job-name=zkllm-setup
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/sharefs/user50/zk/logs/setup-%j.out
#SBATCH --error=/mnt/sharefs/user50/zk/logs/setup-%j.err

set -e

export CONDA_PREFIX=/mnt/sharefs/user50/miniconda3/envs/zkllm-env
export PATH=$CONDA_PREFIX/bin:$PATH

cd /mnt/sharefs/user50/zk/zkllm-ccs2024

mkdir -p /mnt/sharefs/user50/zk/logs

echo "=== Step 1: Generate public parameters (llama-ppgen.py 7) ==="
date
python python/llama-ppgen.py 7

echo "=== Step 2: Commit model weights (llama-commit.py 7 16) ==="
date
python python/llama-commit.py 7 16

echo "=== Setup complete ==="
date
echo "Workdir contents:"
ls -lh zkllm-workdir/Llama-2-7b/ | head -20
echo "..."
ls -lh zkllm-workdir/Llama-2-7b/ | wc -l
echo "total files"
