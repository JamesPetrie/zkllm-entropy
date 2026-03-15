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

# Config
MODEL_SIZE=7
SEQ_LEN=1024   # minimum for zkSoftmax proof (seq_len^2 must be >= 2^20)
NUM_LAYERS=32  # Llama-2-7b has 32 layers
WORKDIR=./zkllm-workdir/Llama-2-${MODEL_SIZE}b

INITIAL_INPUT=${WORKDIR}/layer-0-input.bin
EXPECTED_BYTES=$(( SEQ_LEN * 4096 * 4 ))  # seq_len * embed_dim * sizeof(int32)

# Regenerate if missing or wrong size (e.g. from a previous run with different seq_len)
ACTUAL_BYTES=$(stat -c%s "$INITIAL_INPUT" 2>/dev/null || echo 0)
if [ "$ACTUAL_BYTES" -ne "$EXPECTED_BYTES" ]; then
    echo "=== Generating initial input (embedding layer output, seq_len=$SEQ_LEN) ==="
    python gen_initial_input.py $SEQ_LEN
fi

echo "=== Starting layer-by-layer proofs ==="
echo "Model: Llama-2-${MODEL_SIZE}b  |  Layers: $NUM_LAYERS  |  SeqLen: $SEQ_LEN"
date

input=$INITIAL_INPUT

for layer in $(seq 0 $((NUM_LAYERS - 1))); do
    echo ""
    echo "--- Layer $layer ---"
    date

    attn_input=${WORKDIR}/layer-${layer}-attn_input.bin
    attn_output=${WORKDIR}/layer-${layer}-attn_output.bin
    post_attn_norm_input=${WORKDIR}/layer-${layer}-post_attn_norm_input.bin
    ffn_input=${WORKDIR}/layer-${layer}-ffn_input.bin
    ffn_output=${WORKDIR}/layer-${layer}-ffn_output.bin
    output=${WORKDIR}/layer-${layer}-output.bin

    python llama-rmsnorm.py $MODEL_SIZE $layer input $SEQ_LEN \
        --input_file $input --output_file $attn_input

    python llama-self-attn.py $MODEL_SIZE $layer $SEQ_LEN \
        --input_file $attn_input --output_file $attn_output

    python llama-skip-connection.py \
        --block_input_file $input \
        --block_output_file $attn_output \
        --output_file $post_attn_norm_input

    python llama-rmsnorm.py $MODEL_SIZE $layer post_attention $SEQ_LEN \
        --input_file $post_attn_norm_input --output_file $ffn_input

    python llama-ffn.py $MODEL_SIZE $layer $SEQ_LEN \
        --input_file $ffn_input --output_file $ffn_output

    python llama-skip-connection.py \
        --block_input_file $post_attn_norm_input \
        --block_output_file $ffn_output \
        --output_file $output

    echo "Layer $layer done."
    input=$output  # feed output into next layer
done

echo ""
echo "=== All $NUM_LAYERS layers complete ==="
date
