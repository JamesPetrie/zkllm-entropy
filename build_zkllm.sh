#!/bin/bash
#SBATCH --job-name=build-zkllm
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/sharefs/user50/zk/build_zkllm.log
#SBATCH --error=/mnt/sharefs/user50/zk/build_zkllm.log

export CONDA_PREFIX=/mnt/sharefs/user50/miniconda3/envs/zkllm-env
export PATH=$CONDA_PREFIX/bin:$PATH

cd /mnt/sharefs/user50/zk/zkllm-ccs2024

echo "=== nvcc version ==="
nvcc --version

echo "=== GPU info ==="
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

echo "=== Building ==="
make -j16 all 2>&1

echo "=== Build complete ==="
ls -la main ppgen commit-param self-attn ffn rmsnorm skip-connection \
    zkllm_entropy test_zkargmax test_zklog test_zknormalcdf test_zkentropy 2>/dev/null || echo "Some targets missing"
