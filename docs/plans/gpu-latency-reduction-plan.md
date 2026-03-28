# GPU Latency Reduction Plan

**Date:** 2026-03-27 (updated with measured data)
**Target:** zkllm-entropy full 32-layer prover on H100 PCIe (sm_90)

## System Overview

The full proof pipeline (`run_proofs.py`) proves all 32 transformer layers of Llama-2-7B, then the entropy tail. Each layer executes 7 steps via separate subprocess calls:

```
Per layer (×32):
  1. RMSNorm (input)           → gold_rmsnorm
  2. Self-attn QKV projections → gold_self-attn linear   (3 × zkFC + 3 × Rescaling)
  3. Self-attn attention       → gold_self-attn attn      (QK^T matmul, softmax, AV matmul)
  4. Skip connection           → gold_skip-connection
  5. RMSNorm (post-attention)  → gold_rmsnorm
  6. FFN                       → gold_ffn                 (3 × zkFC + SwiGLU lookup + 4 × Rescaling)
  7. Skip connection           → gold_skip-connection

Final:
  8. Final RMSNorm + lm_head + entropy proof → gold_zkllm_entropy
```

Total: **224 subprocess invocations** for the layer pipeline, plus the entropy tail.

## Measured Timings (Goldilocks, H100 PCIe, n=1024)

### End-to-End

| Milestone | Time | Notes |
|-----------|------|-------|
| 32-layer proof (full pipeline) | **6m 23s (383s)** | After -O3, sync removal, CPU scalar ops |
| Entropy tail only | **17.8s** | gold_zkllm_entropy on final RMSNorm + lm_head + entropy |
| **Per-layer average** | **~11.4s** | (383s - 18s) / 32 layers |

### Per-Step Measurements (Layer 0)

| Step | Wall time | CUDA init overhead | Actual compute+prove |
|------|-----------|-------------------|---------------------|
| self-attn linear (3× QKV matmul + proofs) | 5.8s | ~3.5s | ~2.3s |
| self-attn attn (QK^T, softmax, AV + proofs) | 4.4s | ~3.5s | ~0.9s |
| skip-connection (vector addition) | 3.9s | ~3.5s | ~0.4s |
| FFN (3× matmul + SwiGLU + proofs) | ~6.1s | ~3.5s | ~2.6s |
| RMSNorm (estimated) | ~4.5s | ~3.5s | ~1.0s |
| skip-connection (2nd) | ~3.9s | ~3.5s | ~0.4s |
| **Layer total** | **~28.6s** | **~24.5s** | **~7.6s** |

**Note:** The per-step times sum to ~28.6s but the pipeline runs at ~11.4s/layer. The difference is because the Python script may be running some steps in shared processes or the CUDA context is cached across quick successive invocations. The skip-connection measurement (3.9s for trivial compute) establishes the per-process CUDA overhead baseline at ~3.5s.

### nsys Profiling (Entropy Proof)

From nsys kernel-level breakdown of the entropy binary:

| Kernel category | % of GPU time |
|----------------|--------------|
| matmul (matrixMultiplyOptimized) | 87% |
| tLookup | 5.7% |
| multilinear extension | 2.9% |
| Other (rescaling, NTT, etc.) | 4.4% |

### Key Insight from Measurements

The kernel is running at only **~4% of peak INT32 throughput** on the matmul. This is not bandwidth-bound — it's **latency-bound** due to small tile sizes and lack of register tiling. We verified this experimentally: storing weights as fp16 (4× less bandwidth) produced no speedup and was actually 7% slower due to conversion overhead.

## Where Time Is Spent (Per Layer)

### Compute (forward pass on GPU)

| Operation | Dimensions | Field multiply-adds |
|-----------|-----------|---------------------|
| Q projection (zkFC) | 1024 × 4096 × 4096 | 17.2B |
| K projection (zkFC) | 1024 × 4096 × 4096 | 17.2B |
| V projection (zkFC) | 1024 × 4096 × 4096 | 17.2B |
| QK^T attention | 1024 × 1024 × 128 (per head, ×32) | 4.3B |
| Softmax (exp + norm) | 1024 × 1024 × 32 | lookup-dominated |
| AV multiply | 1024 × 1024 × 128 (per head, ×32) | 4.3B |
| FFN up_proj (zkFC) | 1024 × 4096 × 11008 | 46.1B |
| FFN gate_proj (zkFC) | 1024 × 4096 × 11008 | 46.1B |
| FFN down_proj (zkFC) | 1024 × 11008 × 4096 | 46.1B |
| RMSNorm (×2) | 1024 × 4096 | negligible |
| Rescaling (×7) | various | element-wise |
| **Total per layer** | | **~198B field multiply-adds** |
| **Total 32 layers** | | **~6.3 trillion** |

## Bottleneck Analysis

### Bottleneck 1: Per-Process Overhead (Critical — Architectural)

`run_proofs.py` spawns **224 separate GPU processes** via `os.system()`. Each process:
- Initializes a CUDA context (~3.5s measured — dominated by driver init, not weight loading)
- Loads weights from disk via `FrTensor::from_int_bin()` (disk I/O)
- Loads input tensors from disk
- Runs GPU computation
- Writes output tensors to disk
- Exits (destroying all GPU state)

**Measured:** `gold_skip-connection` (trivial compute) takes 3.9s wall time, establishing a ~3.5s per-process floor. With 7 subprocess invocations per layer × 32 layers = 224 processes, **CUDA init alone costs ~3.5s × 224 = ~784s (~13 min)** — more than double the actual 383s total. This means the pipeline must be benefiting from CUDA context caching between rapid successive calls, but the overhead is still the dominant cost.

**Weight loading per layer:** Each layer loads 7 weight matrices from disk:
- Q, K, V projections: 3 × (4096 × 4096 × 4 bytes) = 192 MB (int32 on disk)
- FFN up, gate, down: 3 × (4096 × 11008 × 4 bytes) = 516 MB
- 2 × RMSNorm weights: negligible
- **Total: ~0.7 GB per layer × 32 = ~22 GB of weight I/O** (plus commitment files)

### Bottleneck 2: Matrix Multiply Kernel (Critical — Compute)

The `matrixMultiplyOptimized` kernel uses:
- **TILE_WIDTH = 16** (16×16 thread blocks = 256 threads)
- Shared memory tiling for both A and B matrices
- One field multiply-add per thread per tile iteration

**Measured throughput:** ~4% of peak INT32 ops. The kernel is latency-bound, not bandwidth-bound. Root causes:
- TILE_WIDTH=16 means each thread block loads only 16×16×8 = 2 KB tiles — too small for the H100's 256 KB shared memory per SM
- No register-level tiling (each thread computes exactly one output element)
- No double-buffering of shared memory loads
- 256 threads/block limits occupancy and instruction-level parallelism

### Bottleneck 3: Softmax Proof — O(n²) (Critical at Longer Contexts)

Self-attention softmax uses `zkSoftmax` which operates on the n×n attention matrix. At n=1024 this is 1M elements per head × 32 heads = 32M elements. The softmax proof involves:
- Exp lookup via tLookup (32M lookups)
- Normalization proof
- Per-head sumcheck

At n=1024 this is a small fraction of per-layer time (~0.9s for the full attn step minus CUDA overhead). At n=4096+ it becomes dominant due to O(n²) scaling.

### Bottleneck 4: Sumcheck Round-Trip Overhead (Medium)

Every sumcheck executes log₂(N) rounds, each requiring a GPU kernel launch + sync + CPU challenge. For a single zkFC.prove() on a 4096×4096 weight matrix: ~12 rounds. Per layer this happens ~7× for weight proofs, giving ~84 sync points per layer × 32 layers = ~2,700 sync events.

**Note:** We already eliminated ~50 unnecessary `cudaDeviceSynchronize()` calls in a prior optimization pass. The remaining syncs are structurally required by the interactive sumcheck protocol (CPU must read GPU polynomial evaluations to generate challenges).

---

## Optimization Plan

### P0: Unified Layer Binary (Critical, Medium Effort)

**Problem:** 224 separate process invocations, each paying ~3.5s CUDA init overhead.

**Solution:** Create a single `gold_full_pipeline` binary that:
1. Initializes CUDA once
2. Loops over 32 layers, loading one layer's weights at a time
3. Keeps intermediate activations in GPU memory between steps (no disk I/O for intermediates)
4. Optionally prefetches next layer's weights while proving current layer

**What this eliminates:**
- ~224 CUDA context initializations (~3.5s each, though amortized in practice)
- All intermediate tensor disk I/O (activations stay on GPU)
- Python subprocess overhead

**Estimated savings:** The 32-layer proof currently takes 383s. Per-layer compute+prove is ~7.6s measured. Pure compute across 32 layers = ~243s. The remaining ~140s is overhead (CUDA init, disk I/O, Python). Eliminating most of this overhead gives an estimated time of **~250-280s (~35% speedup)**.

**Implementation note:** Llama-2-7B total weights = ~13B params × 4 bytes (int32) = ~52 GB on disk, ~104 GB as field elements. Doesn't fit in 80 GB VRAM, so load/free one layer at a time, but keep the CUDA context alive.

### P1: Matmul Kernel Optimization (Critical, Medium Effort)

**Problem:** `matrixMultiplyOptimized` achieves ~4% of peak throughput. Matmul is 87% of GPU kernel time.

**Proposed improvements, in order of expected impact:**

1. **Register tiling.** Each thread should compute a 4×4 or 8×8 sub-tile of the output, accumulating in registers. This is the single biggest lever — it increases arithmetic intensity per shared memory load by 4-8× and is the standard technique in high-performance GEMM.

2. **Increase TILE_WIDTH to 32.** Each Goldilocks element is 8 bytes, so a 32×32 tile = 8 KB — still well within shared memory. Combined with register tiling (4×4 per thread), this means 32×32 output tile computed by 8×8 thread block = 64 threads, each handling a 4×4 sub-tile.

3. **Double-buffered shared memory loading.** Prefetch the next tile from global memory while computing on the current tile, hiding memory latency.

4. **Vectorized global memory loads.** Use `uint2` loads (16 bytes = 2 field elements at a time) to maximize memory transaction efficiency.

**Target:** Reaching even 15-20% of peak (from current 4%) would be a 4-5× speedup on the matmul kernel. Since matmul is 87% of GPU kernel time, this translates to ~3.5-4× on total GPU kernel time.

**Estimated savings on 32-layer proof:** Matmul compute is roughly 87% of the ~243s compute time ≈ 211s. A 4× speedup on matmul → ~53s matmul + ~32s other = ~85s compute. With overhead: **~225s total (40% faster)**.

**Combined P0+P1:** ~110-150s total (2.5-3.5× speedup from current 383s).

### P2: Weight Memory Management (Low Priority — Subsumed by P0)

**Problem:** Each layer loads ~0.7 GB of weights from disk.

With P0 (unified binary), this becomes a streaming problem: load layer N+1 weights while proving layer N. The weight files are on local SSD, so sequential reads of 0.7 GB take <0.5s — well hidden behind proof compute.

**Standalone value:** Only matters if P0 is not done. With P0, this is free.

### P3: Batch Sumcheck Across Dimensions (Medium Impact, High Effort)

**Problem:** Each zkFC.prove() runs a sequential sumcheck with log₂(N) rounds, each requiring a CPU-GPU sync.

**Existing code:** `zkFCStacked` in `zkfc.cu` already proves N stacked FC layers together, amortizing sumcheck rounds. Check whether `run_proofs.py` uses it — if not, enabling it is low-hanging fruit.

**Other approaches:**
- **Kernel fusion:** Fuse polynomial evaluation + reduction into a single kernel per sumcheck round
- **Pipelined sumcheck:** Overlap compute for round N+1 with CPU challenge generation for round N using 2 CUDA streams

**Estimated savings:** Unclear without profiling the sumcheck separately. The fact that self-attn linear (3 matmuls + 3 proofs + 3 openings) only takes ~2.3s of actual compute suggests proofs are already fast relative to matmul.

### P4: GKR Protocol for Matrix Multiply (High Impact, High Effort)

zkGPT reports 6.5× speedup using GKR protocol for matrix multiplication proofs vs sumcheck. This replaces the inner-product sumcheck with a layered circuit approach.

**Estimated savings:** 2-5× on all linear layer proofs. At n=1024, linear layers are the dominant cost, so this is a ~1.5-3× overall speedup.

**Effort:** High — requires implementing GKR prover on GPU. Well-understood protocol but significant engineering.

### P5: Softmax Proof Optimization (Low Priority at n=1024, Critical at n>4K)

At n=1024, the full attention step (including softmax) takes only ~0.9s of compute per layer. Not worth optimizing until we scale to longer contexts.

At n=4096, the attention matrix is 16× larger (16M per head × 32 heads = 512M elements) and softmax proof becomes O(n²) dominant.

### P6: GPU-Side Argmax + Batched Entropy Sumchecks (Low Priority)

The entropy tail is 17.8s — a small fraction of the 383s total. Further optimizing it has diminishing returns until the layer pipeline is faster.

### P7: Compiler Flags and -dlto (Low Effort, Already Partially Done)

- `-O3` already enabled ✓
- `-dlto` (device link-time optimization) is available but untested. Low effort to try.
- `cudaOccupancyMaxPotentialBlockSize` for dynamic block size tuning

**Estimated savings:** 5-10%.

---

## Priority Summary

| Priority | Optimization | Expected Impact | Effort |
|----------|-------------|-----------------|--------|
| **P0** | Unified binary (eliminate subprocess overhead) | 383s → ~250s (35% faster) | Medium |
| **P1** | Matmul kernel (register tiling, TILE=32, double buffer) | 4× matmul speed | Medium |
| **P0+P1** | Combined | 383s → ~110-150s (2.5-3.5× faster) | Medium |
| **P3** | Check if zkFCStacked is used; enable if not | Unknown (need profiling) | Low-Medium |
| **P4** | GKR protocol for matmul proofs | 1.5-3× overall | High |
| **P7** | -dlto + block size tuning | 5-10% | Low |
| **P5** | Softmax proof optimization | Critical only at n>4K | High |
| **P6** | Batched entropy sumchecks | Marginal (17.8s of 383s) | Medium |

## Time Budget Summary

| Scenario | Estimated time | Speedup |
|----------|---------------|---------|
| **Current (measured)** | **383s (6m 23s)** | baseline |
| After P0 (unified binary) | ~250s | 1.5× |
| After P0 + P1 (+ fast matmul) | ~110-150s | 2.5-3.5× |
| After P0 + P1 + P4 (+ GKR proofs) | ~60-90s | 4-6× |

## Optimizations Already Applied

These are not in the plan because they're already done:

1. **-O3 compiler flag** — ~10% speedup
2. **cudaDeviceSynchronize removal** — ~50 unnecessary syncs eliminated, ~2s/layer saved
3. **CPU-side scalar field operators** — Goldilocks add/mul/div done on CPU instead of launching GPU kernels. Eliminated ~16k cudaMalloc/cudaFree calls per layer. 29% speedup on entropy proof, ~5% on layer proofs.
4. **fp16 weight storage** — Implemented and tested, but disabled (7% slower due to conversion overhead). The kernel is latency-bound, not bandwidth-bound, so reducing memory footprint doesn't help speed. Code remains dormant for future memory-constrained scenarios.

## Key Insight

**Process overhead dominates at small compute-per-step, matmul dominates at large compute-per-step.** The skip-connection (trivial compute) spends 90% of its wall time on CUDA init. The FFN (3 large matmuls) spends ~60% on compute. P0 eliminates the overhead floor; P1 attacks the compute ceiling. Together they address both regimes.
