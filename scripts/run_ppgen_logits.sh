#!/bin/bash
#SBATCH --job-name=zk-ppgen-logits
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ppgen-logits-%j.out
#SBATCH --error=logs/ppgen-logits-%j.err

set -e
mkdir -p logs

source /mnt/sharefs/user50/miniconda3/etc/profile.d/conda.sh
conda activate zkllm-env

cd /mnt/sharefs/user50/zk/zkllm-ccs2024

WORKDIR=./zkllm-workdir/Llama-2-7b
PP_FILE=$WORKDIR/lm_head-pp.bin
LOGITS_DIR=$WORKDIR/logits
VOCAB_SIZE=32768   # next power-of-2 >= 32000 (Llama-2 vocab)

echo "=== Step 1: ppgen ==="
if [ -f "$PP_FILE" ]; then
    echo "pp file already exists, skipping ppgen."
else
    echo "Generating $VOCAB_SIZE public-parameter generators -> $PP_FILE"
    ./ppgen $VOCAB_SIZE $PP_FILE
    echo "ppgen done."
fi

echo ""
echo "=== Step 2: gen_logits ==="
echo "Applying lm_head to layer-31 hidden state, saving logit tensors + commitments"
python python/gen_logits.py \
    --model_size 7 \
    --seq_len 1024 \
    --generators $PP_FILE \
    --output_dir $LOGITS_DIR
echo "gen_logits done."

echo ""
echo "=== All done ==="
echo "Logit tensors:  $LOGITS_DIR/logits_*.bin"
echo "Commitments:    $LOGITS_DIR/logits_*-commitment.bin"
echo "Token sequence: $LOGITS_DIR/tokens.txt"
