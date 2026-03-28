"""
Generate layer-0-input.bin: embedding output for a short prompt.
Usage: python gen_initial_input.py [seq_len]
Default seq_len=64 for a fast demo (README uses 2048 for full runs).
"""
import os, sys
import torch
import numpy as np

SEQ_LEN = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
MODEL_CARD = "meta-llama/Llama-2-7b-hf"
CACHE_DIR = "./model-storage"
OUT_PATH = "./zkllm-workdir/Llama-2-7b/layer-0-input.bin"
SCALE = 1 << 16
PROMPT = "Hello, world! This is a zero-knowledge proof of a large language model."

from transformers import AutoTokenizer, AutoModelForCausalLM

print(f"Loading tokenizer and embedding layer (seq_len={SEQ_LEN})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CARD, local_files_only=True, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_CARD, local_files_only=True, cache_dir=CACHE_DIR)

tokens = tokenizer(PROMPT, return_tensors="pt").input_ids[0]
# Truncate or pad to SEQ_LEN
if len(tokens) >= SEQ_LEN:
    tokens = tokens[:SEQ_LEN]
else:
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    tokens = torch.cat([tokens, torch.full((SEQ_LEN - len(tokens),), pad_id)])

print(f"Token sequence length: {SEQ_LEN}  (prompt tokens: {len(tokenizer(PROMPT).input_ids)})")

with torch.no_grad():
    embeddings = model.model.embed_tokens(tokens.unsqueeze(0)).squeeze(0).float()  # (seq_len, embed_dim)

print(f"Embedding shape: {tuple(embeddings.shape)}")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
int_embeddings = torch.round(embeddings * SCALE).to(torch.int32)
int_embeddings.cpu().numpy().astype(np.int32).tofile(OUT_PATH)

print(f"Saved to {OUT_PATH}")
