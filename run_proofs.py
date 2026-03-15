"""
Single-process proof runner for Llama-2 zkLLM.

Loads the model once and runs all 32 layers, replacing the per-layer calls to
llama-rmsnorm.py / llama-self-attn.py / llama-skip-connection.py / llama-ffn.py.

Usage:
    python run_proofs.py [--model_size 7] [--seq_len 1024] [--num_layers 32]
                        [--start_layer 0] [--initial_input PATH]
"""

import os, sys, math, argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from fileio_utils import *

VALUE_LOGSF = 16
ACCU_LOGSF  = 20

parser = argparse.ArgumentParser()
parser.add_argument('--model_size',    type=int, default=7, choices=[7, 13])
parser.add_argument('--seq_len',       type=int, default=1024,
                    help='Sequence length. Must satisfy seq_len^2 >= 2^20 for zkSoftmax.')
parser.add_argument('--num_layers',    type=int, default=32)
parser.add_argument('--start_layer',   type=int, default=0,
                    help='Resume from this layer (0-indexed).')
parser.add_argument('--initial_input', type=str, default=None,
                    help='Path to layer-0-input.bin. Auto-generated if absent or wrong size.')
args = parser.parse_args()

MODEL_CARD = f"meta-llama/Llama-2-{args.model_size}b-hf"
WORKDIR    = f"./zkllm-workdir/Llama-2-{args.model_size}b"
CACHE_DIR  = "./model-storage"
PROMPT     = "Hello, world! This is a zero-knowledge proof of a large language model."

# ---------------------------------------------------------------------------
# Load model (once)
# ---------------------------------------------------------------------------
print(f"Loading model {MODEL_CARD}...", flush=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_CARD, local_files_only=True, cache_dir=CACHE_DIR)

embed_dim  = model.config.hidden_size
num_heads  = model.config.num_attention_heads
head_dim   = embed_dim // num_heads
hidden_dim = model.model.layers[0].mlp.up_proj.out_features

print(f"embed_dim={embed_dim}, num_heads={num_heads}, head_dim={head_dim}, hidden_dim={hidden_dim}")

# ---------------------------------------------------------------------------
# Precompute rotary embeddings (same position ids for every layer)
# ---------------------------------------------------------------------------
model.model.rotary_emb.to(0)
cos, sin = model.model.rotary_emb(
    torch.randn(1, args.seq_len, embed_dim, device=0),
    torch.arange(args.seq_len, device=0).unsqueeze(0)
)

# ---------------------------------------------------------------------------
# Precompute SwiGLU lookup table (written once, deleted at end)
# ---------------------------------------------------------------------------
def prepare_swiglu(in_range_num_bit=10, in_prec_num_bit=12, out_prec_num_bit=16):
    Xs = torch.arange(-(1 << (in_range_num_bit - 1)), 1 << (in_range_num_bit - 1),
                      step=1 / (1 << in_prec_num_bit), device=0)
    Ys = Xs * torch.sigmoid(Xs)
    save_int(Ys, out_prec_num_bit, 'swiglu-table.bin')

prepare_swiglu()

# ---------------------------------------------------------------------------
# Initial input: embedding layer output
# ---------------------------------------------------------------------------
initial_input_path = args.initial_input or f"{WORKDIR}/layer-0-input.bin"
expected_bytes = args.seq_len * embed_dim * 4  # int32

actual_bytes = os.path.getsize(initial_input_path) if os.path.isfile(initial_input_path) else 0
if actual_bytes != expected_bytes:
    from transformers import AutoTokenizer
    print(f"Generating initial input (seq_len={args.seq_len})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CARD, local_files_only=True, cache_dir=CACHE_DIR)
    tokens = tokenizer(PROMPT, return_tensors="pt").input_ids[0]
    if len(tokens) >= args.seq_len:
        tokens = tokens[:args.seq_len]
    else:
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        tokens = torch.cat([tokens, torch.full((args.seq_len - len(tokens),), pad_id)])
    with torch.no_grad():
        embeddings = model.model.embed_tokens(tokens.unsqueeze(0)).squeeze(0).float()
    os.makedirs(WORKDIR, exist_ok=True)
    torch.round(embeddings * (1 << VALUE_LOGSF)).to(torch.int32).cpu().numpy().astype(np.int32).tofile(initial_input_path)
    print(f"Saved to {initial_input_path}", flush=True)

# ---------------------------------------------------------------------------
# Causal mask (shared across layers)
# ---------------------------------------------------------------------------
mask = torch.triu(torch.ones(args.seq_len, args.seq_len, device=0, dtype=bool), diagonal=1)

# ---------------------------------------------------------------------------
# Layer loop
# ---------------------------------------------------------------------------
print(f"\n=== Starting layer-by-layer proofs ===")
print(f"Model: Llama-2-{args.model_size}b  |  Layers: {args.num_layers}  |  SeqLen: {args.seq_len}")

import datetime
print(datetime.datetime.utcnow().strftime("Started: %Y-%m-%d %H:%M:%S UTC"), flush=True)

layer_input = initial_input_path
for layer_idx in range(args.start_layer, args.num_layers):
    layer        = model.model.layers[layer_idx]
    layer_prefix = f"layer-{layer_idx}"

    attn_input  = f"{WORKDIR}/{layer_prefix}-attn_input.bin"
    attn_output = f"{WORKDIR}/{layer_prefix}-attn_output.bin"
    post_attn   = f"{WORKDIR}/{layer_prefix}-post_attn_norm_input.bin"
    ffn_input   = f"{WORKDIR}/{layer_prefix}-ffn_input.bin"
    ffn_output  = f"{WORKDIR}/{layer_prefix}-ffn_output.bin"
    layer_output= f"{WORKDIR}/{layer_prefix}-output.bin"

    print(f"\n--- Layer {layer_idx} ---", flush=True)
    print(datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"), flush=True)

    # -----------------------------------------------------------------------
    # 1. RMSNorm (input layernorm)
    # -----------------------------------------------------------------------
    variance_epsilon = layer.input_layernorm.variance_epsilon
    X = torch.tensor(np.fromfile(layer_input, dtype=np.int32).reshape(args.seq_len, embed_dim),
                     device=0, dtype=torch.float64) / (1 << VALUE_LOGSF)
    rms_inv = 1 / torch.sqrt(torch.mean(X.float() ** 2, dim=1) + variance_epsilon)
    save_int(rms_inv, 1 << VALUE_LOGSF, 'rms_inv_temp.bin')
    os.system(f'./rmsnorm input {layer_input} {args.seq_len} {embed_dim} {WORKDIR} {layer_prefix} {attn_input}')
    os.remove('rms_inv_temp.bin')

    # -----------------------------------------------------------------------
    # 2. Self-attention
    # -----------------------------------------------------------------------
    os.system(f'./self-attn linear {attn_input} {args.seq_len} {embed_dim} {WORKDIR} {layer_prefix} {attn_output}')

    Q = load_int('temp_Q.bin').reshape(args.seq_len, embed_dim) / (1 << VALUE_LOGSF)
    K = load_int('temp_K.bin').reshape(args.seq_len, embed_dim) / (1 << VALUE_LOGSF)
    V = load_int('temp_V.bin').reshape(args.seq_len, embed_dim) / (1 << VALUE_LOGSF)

    Q = Q.view(args.seq_len, num_heads, head_dim).transpose(0, 1)
    K = K.view(args.seq_len, num_heads, head_dim).transpose(0, 1)
    V = V.view(args.seq_len, num_heads, head_dim).transpose(0, 1)

    Q, K = layer.self_attn.rotary_fn(Q, K, cos, sin)
    Q, K = Q.squeeze(0).to(torch.float64), K.squeeze(0).to(torch.float64)

    A = to_int64(Q @ K.transpose(-2, -1), ACCU_LOGSF)
    A -= torch.max(A * ~mask, dim=-1, keepdim=True).values
    shift = math.sqrt(head_dim) * torch.log(
        (torch.exp(to_float(A, ACCU_LOGSF) / math.sqrt(head_dim)) * ~mask).sum(dim=-1, keepdim=True)
    )
    A -= to_int64(shift, ACCU_LOGSF)
    attn_weights = (torch.exp(to_float(A, ACCU_LOGSF, torch.float64) / math.sqrt(head_dim)).float()) * ~mask

    attn_out = fromto_int64(attn_weights @ V, VALUE_LOGSF)
    attn_out = attn_out.transpose(0, 1).contiguous().view(args.seq_len, embed_dim)
    attn_out = attn_out.transpose(0, 1).reshape(args.seq_len, embed_dim)
    save_int(attn_out, 1 << VALUE_LOGSF, attn_output)

    os.system(f'./self-attn attn {attn_input} {args.seq_len} {embed_dim} {WORKDIR} {layer_prefix} {attn_output}')
    os.system('rm -f ./temp_Q.bin ./temp_K.bin ./temp_V.bin ./temp_head_Y.bin ./temp_head_out.bin')

    # -----------------------------------------------------------------------
    # 3. Skip connection (residual: layer_input + attn_output)
    # -----------------------------------------------------------------------
    os.system(f'./skip-connection {layer_input} {attn_output} {post_attn}')

    # -----------------------------------------------------------------------
    # 4. RMSNorm (post-attention layernorm)
    # -----------------------------------------------------------------------
    variance_epsilon2 = layer.post_attention_layernorm.variance_epsilon
    X2 = torch.tensor(np.fromfile(post_attn, dtype=np.int32).reshape(args.seq_len, embed_dim),
                      device=0, dtype=torch.float64) / (1 << VALUE_LOGSF)
    rms_inv2 = 1 / torch.sqrt(torch.mean(X2.float() ** 2, dim=1) + variance_epsilon2)
    save_int(rms_inv2, 1 << VALUE_LOGSF, 'rms_inv_temp.bin')
    os.system(f'./rmsnorm post_attention {post_attn} {args.seq_len} {embed_dim} {WORKDIR} {layer_prefix} {ffn_input}')
    os.remove('rms_inv_temp.bin')

    # -----------------------------------------------------------------------
    # 5. FFN
    # -----------------------------------------------------------------------
    os.system(f'./ffn {ffn_input} {args.seq_len} {embed_dim} {hidden_dim} {WORKDIR} {layer_prefix} {ffn_output}')

    # -----------------------------------------------------------------------
    # 6. Skip connection (residual: post_attn + ffn_output)
    # -----------------------------------------------------------------------
    os.system(f'./skip-connection {post_attn} {ffn_output} {layer_output}')

    print(f"Layer {layer_idx} done.", flush=True)
    layer_input = layer_output

os.remove('swiglu-table.bin')

print(f"\n=== All {args.num_layers} layers complete ===")
print(datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))
