#!/bin/bash
#SBATCH --job-name=zk-test-entropy
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/test-entropy-%j.out
#SBATCH --error=logs/test-entropy-%j.err

set -e
mkdir -p logs

source /mnt/sharefs/user50/miniconda3/etc/profile.d/conda.sh
conda activate zkllm-env

cd /mnt/sharefs/user50/zk/zkllm-ccs2024

make test_zkentropy
./test_zkentropy
