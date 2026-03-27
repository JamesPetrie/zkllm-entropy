# GPU Latency Reduction Plan

**Date:** 2026-03-27
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

## Where Time Is Spent (Per Layer)

Each layer has these GPU-intensive operations:

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

### Prove (sumcheck protocols)

Each zkFC.prove() runs an inner-product sumcheck over the input and weight tensors. Each Rescaling.prove() runs a simpler sumcheck. The softmax proof involves exp/log lookups via tLookup.

At the current matmul throughput of ~13 Gop/s (from bench-goldilocks-results.md), the compute-only matmul time per layer is roughly:

```
198B ops × 2 (mul+add) / 13 Gop/s ≈ 30 seconds per layer (compute only)
```

The prove phase adds additional sumcheck rounds on top. With 32 layers, **matmul compute alone is ~960 seconds**, before any proof overhead.

## Bottleneck Analysis

### Bottleneck 1: Per-Process Overhead (Critical — Architectural)

`run_proofs.py` spawns **224 separate GPU processes** via `os.system()`. Each process:
- Initializes a CUDA context (~0.5-2s)
- Loads weights from disk via `FrTensor::from_int_bin()` (disk I/O)
- Loads input tensors from disk
- Runs GPU computation
- Writes output tensors to disk
- Exits (destroying all GPU state)

**Estimated overhead:** 1-2 seconds per subprocess × 224 = **~4-7 minutes of pure overhead** just from process spawning and CUDA initialization, plus all the disk I/O for intermediate tensors.

**Weight loading per layer:** Each layer loads 7 weight matrices from disk:
- Q, K, V projections: 3 × (4096 × 4096 × 8 bytes) = 384 MB
- FFN up, gate, down: (4096 × 11008 + 4096 × 11008 + 11008 × 4096) × 8 bytes = 1.03 GB
- 2 × RMSNorm weights: negligible
- **Total: ~1.4 GB per layer × 32 = ~45 GB of weight I/O**

Plus all weight commitments loaded alongside.

### Bottleneck 2: Matrix Multiply Kernel (Critical — Compute)

The `matrixMultiplyOptimized` kernel (`fr-tensor.cu:892`) uses:
- **TILE_WIDTH = 16** (16×16 thread blocks = 256 threads)
- Shared memory tiling for both A and B matrices
- One field multiply-add per thread per tile iteration

At 13 Gop/s achieved throughput vs 83 Gop/s bandwidth ceiling (from int32-throughput-analysis.md), the kernel achieves only **16% of memory bandwidth**. Root causes:
- TILE_WIDTH=16 means each thread block loads only 16×16×8 = 2 KB tiles — too small for the H100's 256 KB shared memory per SM
- No register-level tiling (each thread computes exactly one output element)
- No double-buffering of shared memory loads
- 256 threads/block may limit occupancy depending on register pressure

### Bottleneck 3: Softmax Proof — O(n²) (Critical at Longer Contexts)

Self-attention softmax (`self-attn.cu:96-122`) uses `zkSoftmax` which operates on the n×n attention matrix. At n=1024 this is 1M elements per head × 32 heads = 32M elements. The softmax proof involves:
- Exp lookup via tLookup (32M lookups)
- Normalization proof
- Per-head sumcheck

From `improvement-opportunities.md`: "Self-attention softmax proof: O(n² × d) — accounts for ~99% of proving time at 1M context." At n=1024 it's manageable but still significant.

### Bottleneck 4: Sumcheck Round-Trip Overhead (Medium)

Every sumcheck (inner product for zkFC, binary for argmax, LogUp for lookups) executes log₂(N) rounds, each requiring:
1. GPU kernel launch for polynomial evaluation
2. `cudaDeviceSynchronize()` to read the result
3. CPU-side challenge generation (Fiat-Shamir or random)
4. Next round

For a single zkFC.prove() on a 4096×4096 weight matrix:
- N = 4096 → 12 rounds
- Each round: 1 kernel launch + 1 sync + 1 reduction

Per layer this happens 7× (3 QKV + 3 FFN + output projection in attention), giving ~84 sync points per layer × 32 layers = ~2,700 sync events for weight proofs alone.

### Bottleneck 5: Entropy Proof Tail (Lower Priority)

The entropy phase (634s at 1024 tokens with BLS12-381 — should be ~65s with Goldilocks) includes per-token argmax, CDF, and log proofs. This is documented in bench-results-2026-03-27.md. With Goldilocks it becomes a smaller fraction of total time compared to the 32-layer pipeline.

---

## Optimization Plan

### P0: Unified Layer Binary (Critical, Medium Effort)

**Problem:** 224 separate process invocations with full CUDA init + disk I/O each time.

**Solution:** Create a single `gold_full_pipeline` binary that:
1. Initializes CUDA once
2. Loads all 32 layers' weights into GPU memory at startup (Llama-2-7B total weights ~14B params × 8 bytes = ~112 GB — won't fit in 80 GB H100 VRAM, so load 1-2 layers at a time but keep the CUDA context alive)
3. Keeps intermediate activations in GPU memory between steps (no disk I/O for intermediates)
4. Streams weight loading for the next layer while proving the current layer

**What this eliminates:**
- ~224 CUDA context initializations (~4-7 min)
- ~45 GB of intermediate tensor disk I/O
- Python subprocess overhead

**Variant (lower effort):** Keep the Python orchestrator but use a long-lived GPU daemon process that accepts commands over a socket/pipe, avoiding context re-initialization. Or restructure as a single C++ binary with a layer loop (similar to how `zkllm_entropy.cu` already handles the entropy tail).

**Estimated savings:** 20-40% of total wall time, depending on how much is currently disk I/O vs compute.

### P1: Matmul Kernel Optimization (Critical, Medium Effort)

**Problem:** `matrixMultiplyOptimized` achieves 16% of memory bandwidth.

**Proposed improvements, in order of expected impact:**

1. **Increase TILE_WIDTH to 32 or 64.** Each Goldilocks element is 8 bytes, so a 32×32 tile = 8 KB — still well within shared memory. The H100 has 256 KB shared memory per SM. Larger tiles reduce shared memory traffic and improve arithmetic intensity.

2. **Register tiling.** Each thread should compute a 4×4 or 8×8 sub-tile of the output, accumulating in registers. This increases arithmetic intensity per byte loaded from shared memory by 4-8×.

3. **Double-buffered shared memory loading.** Prefetch the next tile from global memory while computing on the current tile, hiding memory latency.

4. **Vectorized global memory loads.** Use `uint4` or `longlong2` loads (16 bytes at a time) to maximize memory transaction efficiency.

**Target:** Reach 50-70% of memory bandwidth ceiling (~40-58 Gop/s), a 3-4× improvement over the current 13 Gop/s. This would reduce per-layer matmul time from ~30s to ~8-10s, saving **~700 seconds across 32 layers**.

**Alternative:** If the matmul is the same operation repeated many times (it is — zkFC with different weights), consider using cuBLAS-like approaches adapted for 64-bit field arithmetic via INT32 decomposition.

### P2: Weight Memory Management (Medium, Medium Effort)

**Problem:** Each layer loads ~1.4 GB of weights from disk (or from host memory), even though weights are static.

**Solution:**
- Pre-compute weight commitments once and cache them
- For the unified binary (P0), keep a 2-layer weight buffer on GPU: load layer N+1 weights while proving layer N
- Use pinned (page-locked) host memory for weight staging to enable async DMA transfers
- Use CUDA streams to overlap weight transfer with compute

**Estimated savings:** Eliminates weight loading latency for all but the first layer — could save 5-10s per layer if I/O bound.

### P3: Batch Sumcheck Across Dimensions (High, High Effort)

**Problem:** Each zkFC.prove() runs a sequential sumcheck with log₂(N) rounds, each round requiring a CPU-GPU sync.

**Proposed approaches:**

**a) Kernel fusion within a sumcheck:** Fuse the polynomial evaluation kernel + reduction into a single kernel that produces all 3 polynomial coefficients in one pass, avoiding intermediate global memory writes.

**b) Batched proving:** When proving multiple zkFC layers with the same structure (e.g., all 32 layers' Q projections have the same dimensions), batch them into a single `zkFCStacked` proof. The code already has `zkFCStacked` in `zkfc.cu:258-298` — it proves N stacked FC layers together, amortizing sumcheck rounds. **This is already implemented but may not be used in the layer pipeline.**

**c) Pipelined sumcheck:** While one sumcheck round computes on GPU, prepare the next round's challenge on CPU. Use 2 CUDA streams to overlap.

**Estimated savings:** 2-3× on proof time if `zkFCStacked` can be applied across layers.

### P4: GKR Protocol for Matrix Multiply (High Impact, High Effort)

**From `improvement-opportunities.md`:** zkGPT reports 6.5× speedup using GKR protocol for matrix multiplication proofs vs the sumcheck approach used here.

**Estimated savings:** 2-5× on all linear layer proofs.

**At n=1024:** Linear layers are ~60% of proving cost, so this is a ~1.5-3× overall speedup.

**Effort:** High — requires implementing GKR prover on GPU, but well-understood protocol with reference implementations.

### P5: Softmax Proof Optimization (Medium Impact at n=1024, Critical for Longer Contexts)

**Problem:** Softmax proof is O(n² × d) due to the full attention matrix.

**Possible approaches:**
- FlashAttention-style chunking to reduce peak memory
- Sparse attention patterns that avoid materializing the full n×n matrix
- At n=1024, this is less critical than matmul optimization; at n=4096+ it becomes dominant

### P6: GPU-Side Argmax + Batched Entropy Sumchecks (Medium)

**Problem:** Argmax in `zkargmax.cu:34-43` copies full 32K logit vector to CPU per token (2 GB total for 1024 tokens). Per-token sumchecks create ~46,000 CPU-GPU sync events.

**Solution:** GPU parallel reduction for argmax (only return index). Batch sumcheck rounds across all 1024 tokens.

**Estimated savings:** 5-10× on entropy prove phase, but this phase is now a smaller fraction of total time when the 32-layer pipeline is included.

### P7: Kernel Launch Tuning and -dlto (Low Effort)

- Enable `-dlto` in Makefile for release builds (currently commented out)
- Tune block sizes via `cudaOccupancyMaxPotentialBlockSize`
- Use shared memory for NTT butterflies

**Estimated savings:** 5-15% on kernel execution time.

---

## Priority Summary

| Priority | Optimization | Target | Expected Impact | Effort |
|----------|-------------|--------|-----------------|--------|
| **P0** | Unified binary (eliminate 224 subprocess spawns + disk I/O) | Architecture | 20-40% wall time | Medium |
| **P1** | Matmul kernel (TILE=32+, register tiling, double buffer) | Compute | 3-4× matmul speed → ~700s saved | Medium |
| **P2** | Weight memory management (async loading, pinned memory) | I/O | 5-10s per layer | Medium |
| **P3** | Batched/stacked FC proving (use existing zkFCStacked) | Prove | 2-3× proof time | Medium-High |
| **P4** | GKR protocol for matmul proofs | Prove | 2-5× linear proof | High |
| **P5** | Softmax proof optimization | Prove (attention) | Critical at n>4K | High |
| **P6** | GPU argmax + batched entropy sumchecks | Entropy tail | 5-10× entropy prove | Medium |
| **P7** | Kernel tuning + -dlto | All kernels | 5-15% | Low |

## Estimated Total Time Budget (32 Layers, n=1024, Goldilocks)

### Current (estimated, before any optimization)

| Component | Per Layer | × 32 | Notes |
|-----------|----------|------|-------|
| Process spawn + CUDA init | ~2s | ~64s | 2 subprocesses have most of the cost |
| Weight loading from disk | ~3-5s | ~96-160s | ~1.4 GB per layer |
| Matmul compute (all FC layers) | ~30s | ~960s | 7 matmuls per layer at 13 Gop/s |
| Sumcheck proofs (FC + rescaling) | ~15-30s | ~480-960s | Multiple sumchecks per layer |
| Softmax compute + proof | ~5-10s | ~160-320s | Attention only |
| Intermediate disk I/O | ~2-3s | ~64-96s | Tensors written between steps |
| **Layer subtotal** | **~57-80s** | **~1,824-2,560s** | |
| Entropy tail (Goldilocks) | — | ~65-100s | From BLS12-381 timing / 9.8× |
| **Total estimate** | | **~1,900-2,700s** | **~30-45 minutes** |

### After P0 + P1 (most impactful, medium effort)

| Component | Per Layer | × 32 | Change |
|-----------|----------|------|--------|
| Process spawn + CUDA init | 0s | 0s | Eliminated by P0 |
| Weight loading | ~1-2s | ~32-64s | Still needed, but no disk I/O for intermediates |
| Matmul compute | ~8-10s | ~256-320s | 3-4× faster kernel (P1) |
| Sumcheck proofs | ~15-30s | ~480-960s | Unchanged |
| Softmax compute + proof | ~5-10s | ~160-320s | Unchanged |
| Intermediate disk I/O | 0s | 0s | Eliminated by P0 |
| **Layer subtotal** | **~29-52s** | **~928-1,664s** | |
| Entropy tail | — | ~65-100s | |
| **Total estimate** | | **~1,000-1,760s** | **~1.5-2× speedup** |

### After P0 + P1 + P3 + P4

With batched proving (P3) and GKR matmul proofs (P4) reducing proof time by 2-5×:

| **Total estimate** | | **~500-900s** | **~3-5× speedup** |
|---------------------|---|---------------|---------------------|

## Key Insight

**The 32 transformer layers dominate total proving time, not the entropy tail.** The entropy proof (634s with BLS12-381, ~65s with Goldilocks) is important but is dwarfed by 32 layers × 7 matmuls/layer × sumcheck proofs per matmul. The most impactful optimizations target the per-layer overhead (P0), matmul kernel performance (P1), and proof protocol efficiency (P3/P4).
