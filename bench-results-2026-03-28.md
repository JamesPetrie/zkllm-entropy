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

| Configuration | Per-layer | 32-layer | Speedup |
|---|---|---|---|
| Original (5 GPU binaries) | 31.1s | ~995s | 1.0× |
| Combined (2 GPU binaries) | 14.7s | ~470s | 2.1× |
| Server (1 CUDA init, projected) | 8.4s | ~272s | 3.7× |
| **Server (measured end-to-end)** | **7.2s** | **230s** | **4.3×** |
| Previously measured (with Python overhead) | ~12s | 383s | 1.7× vs server |

## Profiling: Where the Time Goes

Profiled on H100 PCIe using `nsys`, one full layer via `gold_layer_server`.

### GPU kernel time (~4.4s, 61% of wall time)

| Kernel | Time | Count | What | Bottleneck |
|---|---|---|---|---|
| `matrixMultiplyOptimized` | **~3,920ms** | 10 | All weight matmuls | **Instruction latency** |
| `Fr_me_step` | ~190ms | ~30k | Sumcheck multilinear eval | Compute |
| `tlookup_inv_kernel` | ~90ms | 24 | SwiGLU lookup table | Memory (random access) |
| `zksoftmax_shift` | ~5ms | 1 | Softmax shift | Compute |
| Other proof kernels | ~150ms | ~5k | Rescaling, poly eval, etc. | Mixed |

#### Matmul breakdown per layer

| Matmul | Dimensions (M×K × K×N) | Time | Count |
|---|---|---|---|
| FFN up/gate/down proj | 1024×4096 × 4096×11008 | ~910ms each | 3 |
| QKV proj | 1024×4096 × 4096×4096 | ~340ms each | 3 |
| Q@K^T (attention) | 1024×128 × 128×1024 (×32 heads) | ~86ms | 1 |
| Y@V (attention) | 1024×1024 × 1024×128 (×32 heads) | ~85ms | 1 |
| RMSNorm weights | 1024×1 × 1×4096 | ~1.4ms each | 2 |

### CPU/driver overhead (~2.8s, 39% of wall time)

| Component | Time | What | Bottleneck |
|---|---|---|---|
| **GPU-CPU serialization** | ~1.4s | 1,915 `cudaDeviceSynchronize` calls | Latency (sync barriers) |
| `cudaMalloc` + `cudaFree` | ~525ms | ~14,000 alloc/dealloc calls | Latency (driver roundtrip) |
| `cudaMemcpy` | ~320ms | 302MB H2D + 109MB D2H + 3.5GB D2D | Bandwidth |
| `cudaLaunchKernel` | ~190ms | 21,607 kernel launches | Latency (driver) |
| File I/O | ~250ms | ~900MB weight data from NVMe | Bandwidth (NVMe) |
| CPU compute | ~100ms | rms_inv, random vectors | CPU compute |

## Analysis: The Latency-Bound Matmul

The matmul kernel accounts for **54% of total wall time** and runs at only **~4% of the H100's INT32 peak throughput**. It is not compute-bound or memory-bound — it is **instruction-latency-bound**.

### Why field multiplication is slow on GPUs

Goldilocks field multiplication (`gold_mul`) requires computing a 128-bit product and reducing modulo p = 2^64 - 2^32 + 1. On GPU this compiles to:

```
MUL.WIDE.U64  lo, a, b       // low 64 bits (multi-cycle, uses INT32 pipes)
UMUL64HI      hi, a, b       // high 64 bits (multi-cycle, INT32 pipes)
SHR.U64       hi_hi, hi, 32  // extract top 32 bits
AND.U64       hi_lo, hi, 0xFFFFFFFF
SUB.U64       t0, lo, hi_hi  // may borrow
SETP.LT.U64  borrow, lo, hi_hi
@borrow SUB   t0, t0, eps    // conditional correction
MUL.U64       t1, hi_lo, eps // 32x32 multiply
ADD.U64       r, t0, t1      // may overflow
SETP.LT.U64  overflow, r, t0
@overflow ADD r, r, eps       // conditional correction
SETP.GE.U64  ge_p, r, p
@ge_p SUB     r, r, p        // final reduction
```

This is **~12-15 dependent instructions per multiply-add**, with `UMUL64HI` alone taking multiple cycles (it synthesizes 64×64→64 from 32-bit integer pipes). Each thread in the dot product must complete the full chain before the next accumulation step.

### Comparison to standard matrix multiply

| | FP16 Tensor Core | Goldilocks field |
|---|---|---|
| Op per instruction | 4×4×4 = 64 FMAs | 1 multiply-add |
| Cycles per op | 1 (fused) | ~15 (dependency chain) |
| H100 peak | 1513 TFLOPS | ~101 Gfield-ops/s (measured) |
| Utilization | 50-80% typical | ~4% of INT32 peak |

The ~960× throughput gap between Tensor Core FP16 and Goldilocks field arithmetic is fundamental to the field's 64-bit integer representation. No GPU kernel optimization can bridge this gap — it would require hardware support for modular arithmetic or a smaller field.

### Why more parallelism doesn't help

We tested multiple kernel variants:
- **Register tiling** (2×2, 4×4 output elements per thread): No improvement. More register pressure, same instruction latency per element.
- **Larger tiles** (32×32, 64×64): No improvement. Shared memory isn't the bottleneck.
- **Double buffering**: No improvement. Memory latency is already hidden by the many warps.
- **Loop unrolling**: Already done by the compiler at `-O3`.

The kernel is already saturating the pipeline with warps (each SM runs many concurrent warps), but the dependency chain within each warp cannot be shortened.

### What could help

1. **PTX assembly with `mad.wide.u64`**: Could potentially shave 1-2 instructions from the dependency chain by fusing multiply and add, but unlikely to exceed ~20% improvement.
2. **Smaller field (e.g., Mersenne-31)**: A 31-bit field would use native 32-bit ops, roughly 4× faster per multiply. Plonky3/Binius use this approach. Would require rewriting the entire proof system.
3. **Custom hardware**: FPGAs or ASICs with modular multiplication units could achieve much higher throughput.

## Scaling to Larger Models

### GPU-CPU serialization diminishes with scale

The sumcheck protocol requires O(log n) rounds of GPU-CPU communication, where n is the tensor size. Matmul compute scales as O(d_in × d_out × seq_len). As model dimensions grow:

| Model | embed_dim | hidden_dim | Matmul time/layer | Sumcheck rounds | Serialization |
|---|---|---|---|---|---|
| Llama-2-7B | 4,096 | 11,008 | ~3.9s | ~22 per proof | ~1.4s (36%) |
| Llama-2-13B | 5,120 | 13,824 | ~7.6s | ~23 per proof | ~1.5s (20%) |
| Llama-2-70B | 8,192 | 28,672 | ~32s | ~25 per proof | ~1.7s (5%) |

The serialization overhead grows logarithmically while compute grows quadratically in model dimension, so **it becomes a negligible fraction at 70B+ scale**.

### Memory allocation overhead is addressable

The ~525ms spent on 14,000 `cudaMalloc`/`cudaFree` calls could be eliminated with a CUDA memory pool (pre-allocate a large buffer, sub-allocate from it). This is a software optimization that would save ~7% of per-layer time at 7B scale.

### Matmul dominance increases with scale

At 70B scale, the matmul kernel would consume ~85%+ of wall time (vs 54% at 7B). The proof overhead (sumcheck, commitments, lookups) grows much more slowly than the matmul itself. This means the bottleneck becomes increasingly pure arithmetic throughput — confirming that faster field arithmetic (smaller fields, hardware acceleration) is the long-term lever.

## Files Changed
- `goldilocks.cuh` — Plonky2-style gold_mul reduction
- `rmsnorm_linear.cu` — Combined RMSNorm + self-attn linear
- `post_attn.cu` — Combined attn + skip + RMSNorm + FFN + skip
- `layer_server.cu` — Persistent CUDA server (stdin/stdout IPC)
- `generate_swiglu_table.py` — Pure Python SwiGLU table generator
- `run_proofs.py` — Added `--goldilocks` and `--server` flags
- `Makefile` — New targets: gold_rmsnorm_linear, gold_post_attn, gold_layer_server
