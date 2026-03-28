#!/bin/bash
#SBATCH --job-name=zk-e2e
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/e2e-%j.out
#SBATCH --error=logs/e2e-%j.err

set -e
mkdir -p logs

source /mnt/sharefs/user50/miniconda3/etc/profile.d/conda.sh
conda activate zkllm-env

cd /mnt/sharefs/user50/zk/zkllm-ccs2024

WORKDIR=./zkllm-workdir/Llama-2-7b
SIGMA_EFF=5223   # calibrated from job 2062
BIT_WIDTH=32     # must cover max logit gap: scale(65536) * real_gap; 2^32/65536=65536 float units (safe upper bound)

echo "=============================="
echo "Step 1: commit final weights"
echo "=============================="
python python/commit_final_layers.py --model-size 7 --log-scale 16

echo ""
echo "=============================="
echo "Step 2: generate entropy inputs"
echo "=============================="
python python/gen_entropy_inputs.py --model-size 7 --seq-len 1024

echo ""
echo "=============================="
echo "Step 3: rebuild zkllm_entropy"
echo "=============================="
make zkllm_entropy

echo ""
echo "=============================="
echo "Step 4: run entropy prover"
echo "=============================="
./zkllm_entropy \
    $WORKDIR \
    $WORKDIR/tokens.txt \
    $WORKDIR/entropy-proof.bin \
    $SIGMA_EFF \
    1024 4096 32000 \
    $BIT_WIDTH

echo ""
echo "=============================="
echo "Step 5: verify proof"
echo "=============================="
python python/verify_entropy.py $WORKDIR/entropy-proof.bin

echo ""
echo "====== END-TO-END COMPLETE ======"
