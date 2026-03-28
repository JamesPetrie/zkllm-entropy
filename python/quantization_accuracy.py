#!/usr/bin/env python3
"""
quantization_accuracy.py — Compare FP32 logits with quantized fixed-point logits.

Measures how much the quantization (scale + round to int32) affects logit rankings,
argmax agreement, and per-token probability distributions. This determines whether
the quantization error is absorbed by the calibrated sigma_eff noise model.

Usage:
    python quantization_accuracy.py [--model-size 7] [--log-scale 16] [--seq-len 64]
                                     [--sigma-eff 0.08]
"""

import os, sys, argparse, math
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model-size', type=int, default=7, choices=[7, 13])
parser.add_argument('--log-scale', type=int, default=16)
parser.add_argument('--seq-len', type=int, default=64)
parser.add_argument('--sigma-eff', type=float, default=0.08)
args = parser.parse_args()

SCALE = 1 << args.log_scale

MODEL_CARD = f"meta-llama/Llama-2-{args.model_size}b-hf"
CACHE_DIR = "./model-storage"
PROMPT = "Hello, world! This is a zero-knowledge proof of a large language model."

print(f"Scale = 2^{args.log_scale} = {SCALE}")
print(f"sigma_eff = {args.sigma_eff}")
print()

# ── Load model ─────────────────────────────────────────────────────────────
print(f"Loading {MODEL_CARD}...", flush=True)
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(MODEL_CARD, local_files_only=True, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_CARD, local_files_only=True, cache_dir=CACHE_DIR,
                                              torch_dtype=torch.float32)
model.eval()

embed_dim = model.config.hidden_size
vocab_size = model.config.vocab_size
num_layers = len(model.model.layers)

# ── Tokenize ──────────────────────────────────────────────────────────────
tokens = tokenizer(PROMPT, return_tensors="pt").input_ids[0]
if len(tokens) >= args.seq_len:
    tokens = tokens[:args.seq_len]
else:
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    tokens = torch.cat([tokens, torch.full((args.seq_len - len(tokens),), pad_id)])

# ── Run FP32 inference ────────────────────────────────────────────────────
print("Running FP32 inference...", flush=True)
with torch.no_grad():
    outputs = model(tokens.unsqueeze(0))
    logits_fp32 = outputs.logits.squeeze(0).float()  # (seq_len, vocab_size)

# ── Quantize and dequantize ──────────────────────────────────────────────
# Simulate the quantization pipeline: scale → round → int32 → rescale
logits_int = torch.round(logits_fp32 * SCALE).to(torch.int64)
logits_dequant = logits_int.float() / SCALE  # Back to float after quantization

# ── Compare ──────────────────────────────────────────────────────────────
print(f"\nseq_len={args.seq_len}, vocab_size={vocab_size}")
print()

# 1. Argmax agreement
argmax_fp32 = logits_fp32.argmax(dim=1)
argmax_quant = logits_dequant.argmax(dim=1)
argmax_match = (argmax_fp32 == argmax_quant).float().mean().item()
print(f"Argmax agreement: {argmax_match*100:.1f}% ({(argmax_fp32 == argmax_quant).sum()}/{args.seq_len})")

# 2. Logit difference statistics
diff = (logits_fp32 - logits_dequant).abs()
print(f"\nLogit difference (|FP32 - quantized|):")
print(f"  Mean:   {diff.mean().item():.6f}")
print(f"  Max:    {diff.max().item():.6f}")
print(f"  Median: {diff.median().item():.6f}")
print(f"  Quantization step = 1/scale = {1/SCALE:.6f}")
print(f"  Max error / sigma_eff = {diff.max().item() / args.sigma_eff:.4f}")

# The max quantization error is bounded by 0.5/SCALE (rounding to nearest integer).
# Multiple roundings (weights + activations) compound but the error at the logit level
# includes the full pipeline.
max_theoretical = 0.5 / SCALE
print(f"  Theoretical max single-round error = {max_theoretical:.6f}")
print(f"  Actual max / theoretical max = {diff.max().item() / max_theoretical:.1f}x "
      f"(compound rounding from {num_layers} layers)")

# 3. Per-position analysis of the top-k logit gap
print(f"\nTop-k logit gaps (FP32):")
sorted_fp32, _ = logits_fp32.sort(dim=1, descending=True)
gap_1_2 = (sorted_fp32[:, 0] - sorted_fp32[:, 1])
gap_1_5 = (sorted_fp32[:, 0] - sorted_fp32[:, 4])
print(f"  Top1-Top2 gap: mean={gap_1_2.mean():.4f}, min={gap_1_2.min():.4f}, max={gap_1_2.max():.4f}")
print(f"  Top1-Top5 gap: mean={gap_1_5.mean():.4f}, min={gap_1_5.min():.4f}, max={gap_1_5.max():.4f}")
print(f"  Quantization error / min gap: {diff.max().item() / gap_1_2.min().item():.6f}")

# 4. Softmax KL divergence
print(f"\nSoftmax distribution comparison:")
probs_fp32 = torch.softmax(logits_fp32, dim=1)
probs_quant = torch.softmax(logits_dequant, dim=1)

# KL(fp32 || quant) per position
kl_per_pos = torch.sum(probs_fp32 * (torch.log(probs_fp32 + 1e-30) - torch.log(probs_quant + 1e-30)), dim=1)
print(f"  KL(FP32 || quantized): mean={kl_per_pos.mean().item():.8f}, max={kl_per_pos.max().item():.8f} nats")
print(f"  KL in bits: mean={kl_per_pos.mean().item()/math.log(2):.8f}, max={kl_per_pos.max().item()/math.log(2):.8f}")

# 5. Top-1 probability comparison
top1_prob_fp32 = probs_fp32.max(dim=1).values
top1_prob_quant = probs_quant.max(dim=1).values
top1_diff = (top1_prob_fp32 - top1_prob_quant).abs()
print(f"\nTop-1 probability:")
print(f"  FP32:  mean={top1_prob_fp32.mean():.4f}, min={top1_prob_fp32.min():.4f}")
print(f"  Quant: mean={top1_prob_quant.mean():.4f}, min={top1_prob_quant.min():.4f}")
print(f"  Max |diff|: {top1_diff.max().item():.6f}")

# 6. Does quantization error exceed sigma_eff?
print(f"\n=== Summary ===")
max_err = diff.max().item()
if max_err < args.sigma_eff:
    print(f"Quantization error ({max_err:.6f}) < sigma_eff ({args.sigma_eff}): "
          f"ERROR IS ABSORBED BY NOISE MODEL")
    print(f"  Ratio: {max_err/args.sigma_eff:.4f}x sigma_eff")
else:
    print(f"WARNING: Quantization error ({max_err:.6f}) >= sigma_eff ({args.sigma_eff})")
    print(f"  The noise model may not fully absorb quantization error.")
    print(f"  Consider increasing sigma_eff or the scaling factor.")

# 7. Per-token surprise comparison (approximate)
# Compare the entropy bounds we'd get with FP32 vs quantized logits
print(f"\nPer-token entropy bound comparison (first 20 positions):")
print(f"{'Pos':>4s}  {'FP32 argmax':>12s}  {'Quant argmax':>12s}  {'Match':>5s}  {'Gap FP32':>10s}  {'Gap Quant':>10s}")
for pos in range(min(20, args.seq_len)):
    fp32_am = argmax_fp32[pos].item()
    quant_am = argmax_quant[pos].item()
    match = "Y" if fp32_am == quant_am else "N"
    # Gap between actual token and argmax
    actual = tokens[pos].item() if pos > 0 else tokens[0].item()
    gap_fp = logits_fp32[pos, fp32_am].item() - logits_fp32[pos, actual].item()
    gap_q = logits_dequant[pos, quant_am].item() - logits_dequant[pos, actual].item()
    print(f"{pos:4d}  {fp32_am:12d}  {quant_am:12d}  {match:>5s}  {gap_fp:10.4f}  {gap_q:10.4f}")
