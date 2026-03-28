#!/usr/bin/env python3
"""
gen_entropy_inputs.py — generate the two files needed by zkllm_entropy:
    final_norm-rms_inv.bin   seq_len int32 values (1/rms(hidden[t]) * scale)
    tokens.txt               greedy token ids, one per line

The final hidden state (layer-31-output.bin) must already exist from the
zkLLM layer proofs (run run_proofs.py first).

Usage:
    python gen_entropy_inputs.py [--model-size 7] [--seq-len 1024]
        [--workdir ./zkllm-workdir/Llama-2-7b]
        [--log-scale 16]
"""

import os, sys, argparse
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model-size', type=int, default=7, choices=[7, 13])
parser.add_argument('--seq-len', type=int, default=1024)
parser.add_argument('--workdir', type=str, default='')
parser.add_argument('--log-scale', type=int, default=16,
                    help='Log2 of fixed-point scale used in zkLLM (default 16)')
args = parser.parse_args()

SCALE      = 1 << args.log_scale
MODEL_CARD = f'meta-llama/Llama-2-{args.model_size}b-hf'
WORKDIR    = args.workdir or f'./zkllm-workdir/Llama-2-{args.model_size}b'

# ── Load hidden state ─────────────────────────────────────────────────────────
hidden_path = f'{WORKDIR}/layer-31-output.bin'
if not os.path.isfile(hidden_path):
    print(f'ERROR: {hidden_path} not found. Run run_proofs.py first.', file=sys.stderr)
    sys.exit(1)

print(f'Loading hidden state from {hidden_path}...', flush=True)
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    MODEL_CARD, local_files_only=True, cache_dir='./model-storage',
    torch_dtype=torch.float32)
model.eval()

hidden_size = model.config.hidden_size  # 4096
vocab_size  = model.config.vocab_size   # 32000
eps         = model.model.norm.variance_epsilon  # 1e-5

hidden_int = np.fromfile(hidden_path, dtype=np.int32)
hidden_f   = torch.tensor(
    hidden_int.reshape(args.seq_len, hidden_size), dtype=torch.float32
) / SCALE  # (seq_len, hidden_size), float

print(f'Hidden state: {hidden_f.shape}', flush=True)

# ── Compute per-position 1/rms ────────────────────────────────────────────────
# rms_inv[t] = 1 / sqrt(mean(hidden[t]^2) + eps)
rms_inv = 1.0 / torch.sqrt(hidden_f.pow(2).mean(dim=1) + eps)  # (seq_len,)
rms_inv_int = torch.round(rms_inv * SCALE).to(torch.int32).cpu().numpy()

out_rms = f'{WORKDIR}/final_norm-rms_inv.bin'
rms_inv_int.tofile(out_rms)
print(f'Saved rms_inv → {out_rms}  (shape {rms_inv_int.shape})', flush=True)

# ── Compute greedy tokens ─────────────────────────────────────────────────────
print('Applying final norm + lm_head to get greedy tokens...', flush=True)
with torch.no_grad():
    hidden_normed = model.model.norm(hidden_f.unsqueeze(0)).squeeze(0)
    logits = model.lm_head(hidden_normed)  # (seq_len, vocab_size)

greedy_tokens = logits.argmax(dim=-1).cpu().numpy()
tokens_path = f'{WORKDIR}/tokens.txt'
np.savetxt(tokens_path, greedy_tokens.astype(np.int64), fmt='%d')
print(f'Saved greedy tokens → {tokens_path}', flush=True)

print('Done.')
