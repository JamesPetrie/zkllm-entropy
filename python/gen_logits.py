"""
gen_logits.py — apply lm_head to the final hidden state, save per-position
logit FrTensors and (optionally) their polynomial commitments.

Usage:
    python gen_logits.py --model_size 7 --seq_len 1024
        [--generators PATH]    # lm_head-pp.bin from ppgen; enables commitment
        [--output_dir PATH]    # default: ./zkllm-workdir/Llama-2-{size}b/logits
        [--logit_scale N]      # fixed-point scale (default: 65536 = 1<<16)

Output files (one per output position 0..seq_len-1):
    <output_dir>/logits_<t>.bin              — FrTensor (vocab_size Fr_t elements)
    <output_dir>/logits_<t>-commitment.bin   — G1TensorJacobian (if --generators given)
    <output_dir>/tokens.txt                  — greedy token ids, one per line
"""

import os, sys, argparse
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model_size', type=int, default=7, choices=[7, 13])
parser.add_argument('--seq_len',    type=int, default=1024)
parser.add_argument('--generators', type=str, default='',
                    help='Path to lm_head-pp.bin for commitment generation.')
parser.add_argument('--output_dir', type=str, default='',
                    help='Directory to write logit files.')
parser.add_argument('--logit_scale', type=int, default=65536,
                    help='Fixed-point scale for quantising logits (default 1<<16).')
args = parser.parse_args()

MODEL_CARD = f"meta-llama/Llama-2-{args.model_size}b-hf"
WORKDIR    = f"./zkllm-workdir/Llama-2-{args.model_size}b"
CACHE_DIR  = "./model-storage"
OUTPUT_DIR = args.output_dir or os.path.join(WORKDIR, "logits")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print(f"Loading model {MODEL_CARD}...", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_CARD, local_files_only=True, cache_dir=CACHE_DIR,
    torch_dtype=torch.float32
)
model.eval()

lm_head = model.lm_head  # Linear(hidden_size, vocab_size, bias=False)
vocab_size  = model.config.vocab_size          # e.g. 32000
hidden_size = model.config.hidden_size         # e.g. 4096
print(f"vocab_size={vocab_size}, hidden_size={hidden_size}", flush=True)

# ---------------------------------------------------------------------------
# Load final hidden state (layer-{N}-output.bin)
# ---------------------------------------------------------------------------
last_layer = 31  # 0-indexed; 32 layers total for 7b
hidden_path = os.path.join(WORKDIR, f"layer-{last_layer}-output.bin")
if not os.path.isfile(hidden_path):
    print(f"ERROR: {hidden_path} not found. Run run_proofs.py first.", file=sys.stderr)
    sys.exit(1)

VALUE_LOGSF = 16
hidden = torch.tensor(
    np.fromfile(hidden_path, dtype=np.int32).reshape(args.seq_len, hidden_size),
    dtype=torch.float32
) / (1 << VALUE_LOGSF)
print(f"Loaded hidden state: {hidden.shape}", flush=True)

# ---------------------------------------------------------------------------
# Apply lm_head and final norm (if present)
# ---------------------------------------------------------------------------
with torch.no_grad():
    # Apply final RMSNorm before lm_head
    norm = model.model.norm
    hidden_normed = norm(hidden.unsqueeze(0)).squeeze(0)  # (seq_len, hidden_size)
    # Apply lm_head: (seq_len, hidden_size) -> (seq_len, vocab_size)
    logits_float = lm_head(hidden_normed)                 # (seq_len, vocab_size)

print(f"Logits shape: {logits_float.shape}", flush=True)

# Greedy tokens (argmax at each position)
greedy_tokens = logits_float.argmax(dim=-1).cpu().numpy()
tokens_path = os.path.join(OUTPUT_DIR, "tokens.txt")
np.savetxt(tokens_path, greedy_tokens.astype(np.int64), fmt='%d')
print(f"Saved greedy tokens to {tokens_path}", flush=True)

# ---------------------------------------------------------------------------
# Quantise and save per-position logit tensors as raw int32 binary
# ---------------------------------------------------------------------------
# Each logit is a signed 32-bit integer (scaled by logit_scale).
# The C++ code loads these via FrTensor::from_int_bin() which reads int32
# values and converts to field elements on the GPU — works for any field.
logits_scaled = (logits_float * args.logit_scale).round().to(torch.int32).cpu().numpy()
# Shape: (seq_len, vocab_size)

print(f"Saving {args.seq_len} per-position logit tensors to {OUTPUT_DIR}/...", flush=True)

for t in range(args.seq_len):
    row = logits_scaled[t]  # shape (vocab_size,) int32
    out_path = os.path.join(OUTPUT_DIR, f"logits_{t}.bin")
    row.tofile(out_path)

print(f"Saved logit tensors.", flush=True)

# ---------------------------------------------------------------------------
# Commitment generation (optional, requires generators file)
# ---------------------------------------------------------------------------
if args.generators:
    if not os.path.isfile(args.generators):
        print(f"ERROR: generators file not found: {args.generators}", file=sys.stderr)
        sys.exit(1)

    ZKLLM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    commit_bin = os.path.join(ZKLLM_DIR, "commit_logits")
    if not os.path.isfile(commit_bin):
        print("ERROR: ./commit_logits binary not found. Run 'make commit_logits' first.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Generating commitments using {args.generators}...", flush=True)
    for t in range(args.seq_len):
        logits_path = os.path.join(OUTPUT_DIR, f"logits_{t}.bin")
        com_path    = os.path.join(OUTPUT_DIR, f"logits_{t}-commitment.bin")
        cmd = f"{commit_bin} {args.generators} {logits_path} {com_path}"
        ret = os.system(cmd)
        if ret != 0:
            print(f"ERROR: commit_logits failed at position {t}", file=sys.stderr)
            sys.exit(1)
        if (t + 1) % 100 == 0:
            print(f"  Committed {t+1}/{args.seq_len}", flush=True)

    print(f"All commitments saved to {OUTPUT_DIR}/", flush=True)
else:
    print("No --generators specified; skipping commitment generation.", flush=True)
    print("To generate commitments later:")
    print(f"  ./ppgen 32768 {WORKDIR}/lm_head-pp.bin")
    print(f"  python gen_logits.py --generators {WORKDIR}/lm_head-pp.bin "
          f"--output_dir {OUTPUT_DIR}")

print("Done.", flush=True)
