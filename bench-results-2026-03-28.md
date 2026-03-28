# GPU Latency Reduction Results (2026-03-28)

## Optimizations Implemented

### 1. gold_mul Optimization (17% matmul speedup)
- Eliminated second `UMUL64HI` intrinsic using algebraic identity
- Matmul kernel: **409ms → 340ms** per 4096×4096 multiply
- Further simplified to Plonky2-style reduction (cleaner code, same perf)

### 2. CPU Skip-Connection (40× faster)
- Skip-connection is just int32 element-wise addition, no proofs needed
- CPU: **0.1s** vs GPU: **3.9s** (3.5s was CUDA init)
- Output is bit-identical

### 3. Combined Binaries (2.1× per-layer)
- `gold_rmsnorm_linear`: RMSNorm + self-attn linear in one process
- `gold_post_attn`: attn + skip + RMSNorm + FFN + skip in one process
- Both compute rms_inv internally (no Python/torch dependency for this step)
- Per-layer: **14.7s** vs **31.1s** (2 CUDA inits vs 5)

### 4. Persistent CUDA Server (3.7× per-layer)
- `gold_layer_server`: single process, reads commands from stdin
- Initializes CUDA once for all 32 layers
- Per-layer amortized: **8.4s** vs **31.1s**

## Benchmark Summary

| Configuration | Per-layer | 32-layer projected | Speedup |
|---|---|---|---|
| Original (5 GPU binaries) | 31.1s | ~995s | 1.0× |
| Combined (2 GPU binaries) | 14.7s | ~470s | 2.1× |
| Server (1 CUDA init) | 8.4s | ~272s | 3.7× |
| Previously measured (with Python overhead) | ~12s | 383s | - |

### Per-layer time breakdown (server mode, 8.4s amortized)
- CUDA init: ~0.1s (amortized)
- Weight loading: ~0.5s (7 weights, ~900MB total)
- Matmul kernels: ~2.5s (7 matmuls × 340ms)
- Sumcheck/commitment: ~2.0s
- Softmax proof: ~1.5s
- Other (rescaling, etc.): ~1.8s

## Key Insight
CUDA context initialization was **77% of per-layer wall time** (20.4s of 26.3s sys time). 
The actual GPU compute was only 6.2s/layer. Eliminating process spawning was the single 
highest-impact optimization.

## Files Changed
- `goldilocks.cuh` — Plonky2-style gold_mul reduction
- `rmsnorm_linear.cu` — Combined RMSNorm + self-attn linear
- `post_attn.cu` — Combined attn + skip + RMSNorm + FFN + skip  
- `layer_server.cu` — Persistent CUDA server (stdin/stdout IPC)
- `generate_swiglu_table.py` — Pure Python SwiGLU table generator
- `run_proofs.py` — Added `--goldilocks` and `--server` flags
- `Makefile` — New targets: gold_rmsnorm_linear, gold_post_attn, gold_layer_server
