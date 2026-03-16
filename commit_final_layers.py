#!/usr/bin/env python3
"""
commit_final_layers.py — quantise and commit lm_head + final RMSNorm weights.

These two weights are not handled by llama-commit.py (which only loops over
the per-transformer-layer parameters).  They must be committed before running
zkllm_entropy.

Output files written to <workdir>:
    lm_head-weight-int.bin            int32 quantised weight (hidden × vocab)
    lm_head-weight-commitment.bin     G1 commitment using lm_head-pp.bin
    final_norm.weight-int.bin         int32 quantised norm weight (hidden,)
    final_norm.weight-commitment.bin  G1 commitment using input_layernorm.weight-pp.bin

Usage:
    python commit_final_layers.py [--model-size 7] [--log-scale 16]
"""

import os, sys, argparse
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model-size', type=int, default=7, choices=[7, 13])
parser.add_argument('--log-scale', type=int, default=16,
                    help='Log2 of fixed-point scale (default 16 → scale=65536)')
args = parser.parse_args()

WORKDIR    = f'./zkllm-workdir/Llama-2-{args.model_size}b'
MODEL_CARD = f'meta-llama/Llama-2-{args.model_size}b-hf'
SCALE      = 1 << args.log_scale

# ── Compile commit-param if needed ───────────────────────────────────────────
if os.system('make commit-param') != 0:
    print('ERROR: make commit-param failed', file=sys.stderr)
    sys.exit(1)

# ── Load model ────────────────────────────────────────────────────────────────
print(f'Loading {MODEL_CARD}...', flush=True)
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    MODEL_CARD, local_files_only=True, cache_dir='./model-storage',
    torch_dtype=torch.float32)
model.eval()

def save_and_commit(w_int: np.ndarray, int_path: str, com_path: str,
                    pp_path: str, rows: int, cols: int):
    w_int.tofile(int_path)
    ret = os.system(f'./commit-param {pp_path} {int_path} {com_path} {rows} {cols}')
    if ret != 0:
        print(f'ERROR: commit-param failed for {int_path}', file=sys.stderr)
        sys.exit(1)
    print(f'  committed {int_path}  →  {com_path}')


# ── lm_head weight ────────────────────────────────────────────────────────────
# PyTorch shape: (vocab_size=32000, hidden_size=4096)
# After .T for zkFC convention: (in_dim=hidden_size=4096, out_dim=vocab_size=32000)
lm_head_w = model.lm_head.weight.float().T  # (4096, 32000)
lm_int    = torch.round(lm_head_w * SCALE).to(torch.int32).cpu().numpy()
in_dim, out_dim = lm_int.shape  # 4096, 32000

pp_path  = f'{WORKDIR}/lm_head-pp.bin'
int_path = f'{WORKDIR}/lm_head-weight-int.bin'
com_path = f'{WORKDIR}/lm_head-weight-commitment.bin'

if not os.path.isfile(pp_path):
    print(f'ERROR: {pp_path} not found. Run: ./ppgen 32768 {pp_path}', file=sys.stderr)
    sys.exit(1)

print(f'Committing lm_head weight ({in_dim}×{out_dim})...', flush=True)
save_and_commit(lm_int, int_path, com_path, pp_path, in_dim, out_dim)


# ── final RMSNorm weight ──────────────────────────────────────────────────────
# model.model.norm.weight shape: (hidden_size=4096,)
# In zkFC convention: treated as (in_dim=1, out_dim=hidden_size=4096)
norm_w   = model.model.norm.weight.float()  # (4096,)
norm_int = torch.round(norm_w * SCALE).to(torch.int32).cpu().numpy()
(out_dim_norm,) = norm_int.shape  # 4096

pp_path_norm  = f'{WORKDIR}/input_layernorm.weight-pp.bin'
int_path_norm = f'{WORKDIR}/final_norm.weight-int.bin'
com_path_norm = f'{WORKDIR}/final_norm.weight-commitment.bin'

if not os.path.isfile(pp_path_norm):
    print(f'ERROR: {pp_path_norm} not found. Run llama-commit.py first.', file=sys.stderr)
    sys.exit(1)

print(f'Committing final_norm weight (1×{out_dim_norm})...', flush=True)
save_and_commit(norm_int, int_path_norm, com_path_norm,
                pp_path_norm, 1, out_dim_norm)

print('Done.')
