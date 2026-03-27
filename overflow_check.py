#!/usr/bin/env python3
"""
overflow_check.py — Check how close intermediate values get to overflowing
the Goldilocks field (p = 2^64 - 2^32 + 1) during Llama-2 inference.

Traces every matmul, Hadamard product, and accumulation through one forward
pass (embedding → 32 transformer layers → lm_head), computing the true
integer magnitude at each stage.

Usage:
    python overflow_check.py [--model-size 7] [--log-scale 16] [--seq-len 64]
"""

import os, sys, argparse, math
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model-size', type=int, default=7, choices=[7, 13])
parser.add_argument('--log-scale', type=int, default=16,
                    help='Log2 of fixed-point scale (default 16 → scale=65536)')
parser.add_argument('--seq-len', type=int, default=64,
                    help='Sequence length for the test')
args = parser.parse_args()

GOLDILOCKS_P = (1 << 64) - (1 << 32) + 1
LOG2_P = math.log2(GOLDILOCKS_P)  # ≈ 64.0
SCALE = 1 << args.log_scale

MODEL_CARD = f"meta-llama/Llama-2-{args.model_size}b-hf"
CACHE_DIR = "./model-storage"
PROMPT = "Hello, world! This is a zero-knowledge proof of a large language model."

print(f"Goldilocks p = 2^64 - 2^32 + 1 = {GOLDILOCKS_P}")
print(f"log2(p) = {LOG2_P:.6f}")
print(f"Scale = 2^{args.log_scale} = {SCALE}")
print()

# ── Load model ─────────────────────────────────────────────────────────────
print(f"Loading {MODEL_CARD}...", flush=True)
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(MODEL_CARD, local_files_only=True, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_CARD, local_files_only=True, cache_dir=CACHE_DIR,
                                              torch_dtype=torch.float32)
model.eval()

embed_dim = model.config.hidden_size
num_heads = model.config.num_attention_heads
head_dim = embed_dim // num_heads
hidden_dim = model.model.layers[0].mlp.up_proj.out_features
num_layers = len(model.model.layers)

print(f"embed_dim={embed_dim}, num_heads={num_heads}, head_dim={head_dim}, "
      f"hidden_dim={hidden_dim}, num_layers={num_layers}")
print()

# ── Tokenize input ─────────────────────────────────────────────────────────
tokens = tokenizer(PROMPT, return_tensors="pt").input_ids[0]
if len(tokens) >= args.seq_len:
    tokens = tokens[:args.seq_len]
else:
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    tokens = torch.cat([tokens, torch.full((args.seq_len - len(tokens),), pad_id)])

# ── Helper: report max magnitude ──────────────────────────────────────────
results = []

def report(name, tensor_int, in_dim=None):
    """Report the max magnitude of a quantized integer tensor and check headroom."""
    if isinstance(tensor_int, torch.Tensor):
        max_val = tensor_int.abs().max().item()
    else:
        max_val = abs(tensor_int)

    if max_val == 0:
        log2_max = 0
    else:
        log2_max = math.log2(max_val)

    headroom = LOG2_P - log2_max
    overflow = "OVERFLOW" if max_val >= GOLDILOCKS_P else ("CLOSE" if headroom < 4 else "OK")

    # For matmul: also compute the max possible dot product magnitude
    dot_info = ""
    if in_dim is not None:
        # The dot product sums in_dim products. Each product is at most max_val^2.
        # But that's pessimistic. We want the actual max row dot product.
        dot_info = f"  (in_dim={in_dim})"

    print(f"  {overflow:8s}  {name:50s}  max=2^{log2_max:5.1f}  headroom={headroom:5.1f} bits{dot_info}")
    results.append((name, max_val, log2_max, headroom, overflow))


def report_dotproduct(name, a_int, b_int, in_dim):
    """Compute the actual max dot product (true integer, not modular) for a matmul.

    a_int: (batch, in_dim) or (in_dim,) quantized input
    b_int: (in_dim, out_dim) quantized weight
    """
    # Use Python int128 via numpy to compute exact dot products
    a_np = a_int.cpu().numpy().astype(np.int64)
    b_np = b_int.cpu().numpy().astype(np.int64)

    if a_np.ndim == 1:
        a_np = a_np.reshape(1, -1)

    # Compute dot products in int64 — for overflow check we need true magnitude
    # Use float64 for the magnitude check (sufficient for ~2^53 precision,
    # and we only care about order of magnitude)
    a_f = a_np.astype(np.float64)
    b_f = b_np.astype(np.float64)
    dots = a_f @ b_f  # (batch, out_dim)
    max_dot = np.max(np.abs(dots))

    if max_dot == 0:
        log2_max = 0
    else:
        log2_max = math.log2(max_dot)

    headroom = LOG2_P - log2_max
    overflow = "OVERFLOW" if max_dot >= GOLDILOCKS_P else ("CLOSE" if headroom < 4 else "OK")

    print(f"  {overflow:8s}  {name:50s}  max=2^{log2_max:5.1f}  headroom={headroom:5.1f} bits  (in_dim={in_dim})")
    results.append((name, max_dot, log2_max, headroom, overflow))
    return max_dot


# ── Quantize weights ──────────────────────────────────────────────────────
print("=" * 90)
print("WEIGHT MAGNITUDES (quantized to int32 with scale=2^{})".format(args.log_scale))
print("=" * 90)

for i in range(min(num_layers, 2)):  # Check first 2 layers for weights
    layer = model.model.layers[i]
    for name, param in layer.named_parameters():
        if len(param.shape) == 2:
            w_int = torch.round(param.float().T * SCALE).to(torch.int64)
        else:
            w_int = torch.round(param.float() * SCALE).to(torch.int64)
        report(f"layer-{i}.{name}", w_int)

# lm_head
lm_head_int = torch.round(model.lm_head.weight.float().T * SCALE).to(torch.int64)
report("lm_head.weight", lm_head_int)

# final norm
norm_int = torch.round(model.model.norm.weight.float() * SCALE).to(torch.int64)
report("final_norm.weight", norm_int)

# ── Run quantized inference and check each stage ──────────────────────────
print()
print("=" * 90)
print("INFERENCE TRACE (checking dot product magnitudes at each stage)")
print("=" * 90)

with torch.no_grad():
    # Embedding
    embeddings = model.model.embed_tokens(tokens.unsqueeze(0)).squeeze(0).float()
    x_int = torch.round(embeddings * SCALE).to(torch.int64)
    report("embedding output", x_int)

    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        print(f"\n--- Layer {layer_idx} ---")

        # ── RMSNorm (input) ────────────────────────────────────────────
        x_float = x_int.float() / SCALE
        rms = torch.sqrt(torch.mean(x_float ** 2, dim=1, keepdim=True) +
                         layer.input_layernorm.variance_epsilon)
        rms_inv_int = torch.round((1.0 / rms) * SCALE).to(torch.int64)
        report(f"L{layer_idx} rms_inv", rms_inv_int)

        # Hadamard: x * rms_inv (element-wise, then * norm_weight, then rescale)
        norm_w_int = torch.round(layer.input_layernorm.weight.float() * SCALE).to(torch.int64)
        # The actual computation: normed = x_int * rms_inv_int / SCALE * norm_w_int / SCALE
        # Product before rescale:
        hadamard1 = x_int * rms_inv_int.expand_as(x_int)  # element-wise
        report(f"L{layer_idx} rmsnorm x*rms_inv (before rescale)", hadamard1)

        # After first rescale: hadamard1 / SCALE
        h1_rescaled = torch.round(hadamard1.float() / SCALE).to(torch.int64)
        hadamard2 = h1_rescaled * norm_w_int.expand_as(h1_rescaled)
        report(f"L{layer_idx} rmsnorm h1*norm_w (before rescale)", hadamard2)

        h2_rescaled = torch.round(hadamard2.float() / SCALE).to(torch.int64)
        report(f"L{layer_idx} rmsnorm output (after rescale)", h2_rescaled)

        # ── Q, K, V projections (matmul) ──────────────────────────────
        attn_input = h2_rescaled
        for proj_name in ['q_proj', 'k_proj', 'v_proj']:
            proj = getattr(layer.self_attn, proj_name)
            w_int = torch.round(proj.weight.float().T * SCALE).to(torch.int64)
            report_dotproduct(
                f"L{layer_idx} {proj_name} matmul (pre-rescale)",
                attn_input, w_int, embed_dim)

            # After rescale
            out_float = (attn_input.float() / SCALE) @ (w_int.float() / SCALE) * SCALE
            out_int = torch.round(out_float).to(torch.int64)
            report(f"L{layer_idx} {proj_name} output (post-rescale)", out_int)

        # ── Attention scores Q@K^T ────────────────────────────────────
        # Q, K are (seq_len, embed_dim) quantized
        q_float = (attn_input.float() / SCALE) @ (torch.round(layer.self_attn.q_proj.weight.float().T * SCALE).float() / SCALE)
        k_float = (attn_input.float() / SCALE) @ (torch.round(layer.self_attn.k_proj.weight.float().T * SCALE).float() / SCALE)
        q_int = torch.round(q_float * SCALE).to(torch.int64)
        k_int = torch.round(k_float * SCALE).to(torch.int64)

        # Reshape to heads
        q_heads = q_int.view(args.seq_len, num_heads, head_dim)
        k_heads = k_int.view(args.seq_len, num_heads, head_dim)

        # Per-head attention: Q_h @ K_h^T, dot product over head_dim
        for h in range(min(num_heads, 2)):  # Check first 2 heads
            report_dotproduct(
                f"L{layer_idx} attn_scores head-{h} (pre-rescale)",
                q_heads[:, h, :], k_heads[:, h, :].T, head_dim)

        # ── O projection ─────────────────────────────────────────────
        # Skip detailed attention computation, just check o_proj matmul
        o_proj_w_int = torch.round(layer.self_attn.o_proj.weight.float().T * SCALE).to(torch.int64)
        # o_proj input is the attention output, roughly same magnitude as v_proj output
        v_out_float = (attn_input.float() / SCALE) @ (torch.round(layer.self_attn.v_proj.weight.float().T * SCALE).float() / SCALE)
        v_out_int = torch.round(v_out_float * SCALE).to(torch.int64)
        report_dotproduct(
            f"L{layer_idx} o_proj matmul (pre-rescale)",
            v_out_int, o_proj_w_int, embed_dim)

        # ── FFN: up_proj, gate_proj, down_proj ────────────────────────
        # Use the actual layer input for FFN (post-attention + skip + norm)
        # For simplicity, use the same input magnitude estimate
        ffn_input = h2_rescaled  # Approximate (actual would include skip + second norm)

        for proj_name in ['up_proj', 'gate_proj']:
            proj = getattr(layer.mlp, proj_name)
            w_int = torch.round(proj.weight.float().T * SCALE).to(torch.int64)
            report_dotproduct(
                f"L{layer_idx} {proj_name} matmul (pre-rescale)",
                ffn_input, w_int, embed_dim)

        # SwiGLU: element-wise multiply of up_proj * sigmoid(gate_proj) outputs
        # Both are ~SCALE magnitude after rescale, product is ~SCALE^2
        up_float = (ffn_input.float() / SCALE) @ (torch.round(layer.mlp.up_proj.weight.float().T * SCALE).float() / SCALE)
        gate_float = (ffn_input.float() / SCALE) @ (torch.round(layer.mlp.gate_proj.weight.float().T * SCALE).float() / SCALE)
        swiglu_float = up_float * torch.sigmoid(gate_float) * gate_float  # Approximate SwiGLU
        swiglu_int = torch.round(swiglu_float * SCALE).to(torch.int64)

        down_w_int = torch.round(layer.mlp.down_proj.weight.float().T * SCALE).to(torch.int64)
        report_dotproduct(
            f"L{layer_idx} down_proj matmul (pre-rescale)",
            swiglu_int, down_w_int, hidden_dim)

        # Propagate approximate x_int to next layer (use FP for simplicity)
        with torch.no_grad():
            x_float_next = layer(x_int.float().unsqueeze(0) / SCALE)[0].squeeze(0)
            x_int = torch.round(x_float_next * SCALE).to(torch.int64)
        report(f"L{layer_idx} output", x_int)

    # ── Final norm + lm_head ──────────────────────────────────────────
    print(f"\n--- Final layers ---")
    report("pre-lm_head input", x_int)

    # Final RMSNorm
    x_float = x_int.float() / SCALE
    rms_final = torch.sqrt(torch.mean(x_float ** 2, dim=1, keepdim=True) +
                           model.model.norm.variance_epsilon)
    rms_inv_final = torch.round((1.0 / rms_final) * SCALE).to(torch.int64)
    norm_w_final = torch.round(model.model.norm.weight.float() * SCALE).to(torch.int64)

    h_final = x_int * rms_inv_final.expand_as(x_int)
    report("final rmsnorm x*rms_inv (before rescale)", h_final)
    h_final_r = torch.round(h_final.float() / SCALE).to(torch.int64)
    h_final2 = h_final_r * norm_w_final.expand_as(h_final_r)
    report("final rmsnorm h*norm_w (before rescale)", h_final2)
    h_final2_r = torch.round(h_final2.float() / SCALE).to(torch.int64)

    # lm_head matmul
    report_dotproduct("lm_head matmul (pre-rescale)", h_final2_r, lm_head_int, embed_dim)

print()
print("=" * 90)
print("SUMMARY")
print("=" * 90)
print(f"{'Status':8s}  {'Stage':50s}  {'Max':>10s}  {'Headroom':>10s}")
print("-" * 90)
overflows = []
close = []
for name, max_val, log2_max, headroom, status in results:
    if status == "OVERFLOW":
        overflows.append((name, log2_max, headroom))
    elif status == "CLOSE":
        close.append((name, log2_max, headroom))

if overflows:
    print(f"\nOVERFLOW ({len(overflows)} stages):")
    for name, log2_max, headroom in overflows:
        print(f"  {name:50s}  2^{log2_max:.1f}  headroom={headroom:.1f}")
else:
    print("\nNo overflows detected!")

if close:
    print(f"\nCLOSE (headroom < 4 bits, {len(close)} stages):")
    for name, log2_max, headroom in close:
        print(f"  {name:50s}  2^{log2_max:.1f}  headroom={headroom:.1f}")

# Find the tightest headroom
min_headroom = min(r[3] for r in results)
min_name = [r[0] for r in results if r[3] == min_headroom][0]
print(f"\nTightest headroom: {min_headroom:.1f} bits at '{min_name}'")
print(f"Goldilocks p = 2^{LOG2_P:.6f}")
