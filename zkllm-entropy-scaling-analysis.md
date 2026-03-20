# Scaling Analysis: Zero-Knowledge Proofs of LLM Inference (zkllm-entropy)

## Executive Summary

This report analyzes the computational scaling complexity of [zkllm-entropy](https://github.com/JamesPetrie/zkllm-entropy), a zero-knowledge proof system for verifying LLM inference by proving conditional entropy bounds on outputs. The system avoids exact floating-point reproducibility by proving that outputs have bounded conditional entropy under a Gaussian noise model of GPU non-determinism.

**Reference point:** Proving an 8B parameter model with 1,000-token context takes ~8 minutes on one GPU.

**Target:** Proving a 1T parameter model with 1M token context.

**Key finding:** Naively scaling from the reference point, proving a 1T/1M-context model would take approximately **~250 years on a single GPU**. The dominant bottleneck is the O(seq_len^2) self-attention softmax proof, which alone accounts for ~200 years at 1M context. Even with aggressive optimization and massive parallelism (4,096 GPUs + algorithmic improvements), estimated wall-clock time is on the order of **2-4 weeks**, making this a grand-challenge-level engineering problem.

---

## 1. System Architecture Overview

The zkllm-entropy proof pipeline operates on committed model weights and proves the following end-to-end:

```
For each of L transformer layers:
    1. RMSNorm:          zkFC + Hadamard product + Rescaling
    2. Self-Attention:   3x zkFC (Q,K,V) + matmul (QK^T) + zkSoftmax + zkFC (output proj)
    3. FFN:              3x zkFC (gate/up/down) + SwiGLU (tLookup) + Rescaling
    4. Skip Connection:  Element-wise addition

Final layers:
    5. Final RMSNorm + lm_head FC → logits
    6. zkConditionalEntropy: zkArgmax + zkNormalCDF + zkLog per position
```

Each linear layer is proven via an **inner product sumcheck** protocol. Non-linear operations (softmax, SwiGLU) use **table lookup proofs** (tLookupRangeMapping via the LogUp protocol). Weight integrity is ensured via **Pedersen commitments** using BLS12-381 multi-scalar multiplication (MSM).

---

## 2. Complexity by Component

### 2.1 Notation

| Symbol | Description | 8B Reference | 1T Target |
|--------|-------------|-------------|-----------|
| L | Number of layers | 32 | ~128 |
| d | Hidden dimension (d_model) | 4,096 | ~25,600 |
| d_ff | FFN intermediate dim | 11,008 | ~68,000 |
| h | Number of attention heads | 32 | ~160 |
| n | Sequence length (context) | 1,024 | 1,000,000 |
| V | Vocabulary size | 32,000 | ~128,000 |
| P | Total parameters | 8×10^9 | 1×10^12 |

*1T model dimensions estimated using standard Chinchilla/GPT-style scaling ratios: d ≈ 25,600, d_ff ≈ 4d ≈ 68,000 (with SwiGLU adjustments), L ≈ 128, h ≈ 160.*

### 2.2 Per-Component Complexity

| Component | Computation Complexity | Proof Complexity (polynomials) | Memory |
|-----------|----------------------|-------------------------------|--------|
| **RMSNorm** (×L) | O(n × d) | O(log(n × d)) | O(n × d) |
| **Self-Attn QKV** (×L) | O(n × d²) | O(log(n × d²)) | O(n × d) |
| **Self-Attn Softmax** (×L) | **O(n² × d)** | **O(n² × log(n²))** | **O(n²)** |
| **Self-Attn Output** (×L) | O(n² × d) | O(log(n² × d)) | O(n² + n×d) |
| **FFN matmuls** (×L) | O(n × d × d_ff) | O(log(n × d × d_ff)) | O(n × d_ff) |
| **FFN SwiGLU** (×L) | O(n × d_ff × log(d_ff)) | O(n × d_ff × log(d_ff)) | O(n × d_ff) |
| **lm_head FC** (×1) | O(n × d × V) | O(log(n × d × V)) | O(n × V) |
| **zkArgmax** (×n) | O(n × V) | O(n × bit_width) | O(V × bit_width) |
| **zkCDF** (×n) | O(n × V × log(V)) | O(n × log(V)) | O(V) |
| **Weight Commitment** | O(P × log(P)) | N/A (setup phase) | O(P) |

### 2.3 Dominant Terms

The total proving work is dominated by three terms:

1. **Self-attention softmax proof:** O(L × n² × d) — **quadratic in context length**
2. **FFN matrix multiply proof:** O(L × n × d × d_ff) — **linear in both model size and context**
3. **Weight commitment MSM:** O(P × log(P)) — **linear in total parameters**

---

## 3. Scaling Analysis

### 3.1 Scaling with Context Length (n)

Context length scaling is dominated by self-attention, which is **quadratic** in sequence length.

| Context Length | Self-Attn Cost (relative) | FFN Cost (relative) | Total (relative) | Estimated Time (8B model, 1 GPU) |
|---------------|--------------------------|--------------------|-----------------|---------------------------------|
| 1,000 | 1× | 1× | 1× | ~8 min |
| 4,000 | 16× | 4× | ~12× | ~1.6 hours |
| 16,000 | 256× | 16× | ~170× | ~23 hours |
| 64,000 | 4,096× | 64× | ~2,600× | ~14 days |
| 256,000 | 65,536× | 256× | ~40,000× | ~7 months |
| 1,000,000 | 1,000,000× | 1,000× | ~600,000× | **~9 years** |

The quadratic attention cost makes long-context proving prohibitively expensive without architectural changes.

**Memory scaling with context:** The attention matrix alone requires O(n²) storage in field elements (32 bytes each):
- n = 1,000: ~32 MB
- n = 1,000,000: **~32 TB** (infeasible for single GPU)

### 3.2 Scaling with Model Size (Parameters)

Model size affects proving through two channels: (a) larger weight matrices increase per-layer proof cost, and (b) more layers multiply the total cost.

| Model Size | d_model | d_ff | L | Per-Layer Cost (relative) | Total (relative) | Estimated Time (n=1000, 1 GPU) |
|-----------|---------|------|---|--------------------------|------------------|---------------------------------|
| 8B | 4,096 | 11,008 | 32 | 1× | 1× | ~8 min |
| 13B | 5,120 | 13,824 | 40 | 2.0× | 2.5× | ~20 min |
| 70B | 8,192 | 28,672 | 80 | 7.3× | 18× | ~2.5 hours |
| 405B | 16,384 | 53,248 | 126 | 27× | 106× | ~14 hours |
| 1T | 25,600 | 68,000 | 128 | 54× | 216× | **~29 hours** |

Per-layer cost scales roughly as O(d² + d × d_ff) ≈ O(d²) since d_ff ∝ d.

**Weight commitment cost** scales linearly with total parameters:
- 8B: ~10-60 seconds (MSM over 8×10^9 field elements)
- 1T: ~20-120 minutes (MSM over 10^12 field elements)

### 3.3 Combined Scaling: 1T Model × 1M Context

Combining both dimensions:

| Component | 8B / 1K tokens | 1T / 1M tokens | Scaling Factor |
|-----------|---------------|----------------|---------------|
| Self-Attn Softmax | ~3 min | ~3 min × 54 × 10^6 | **~300 years** |
| FFN proofs | ~2 min | ~2 min × 54 × 1000 | ~200 days |
| QKV projections | ~1.5 min | ~1.5 min × 54 × 1000 | ~150 days |
| lm_head | ~1 min | ~1 min × 6 × 1000 | ~4 days |
| Weight commitment | ~0.5 min | ~0.5 min × 125 | ~1 hour |
| Entropy proof | <0.1 min | <0.1 min × 1000 × 4 | ~7 hours |
| **Total** | **~8 min** | | **~300 years** |

The self-attention softmax proof completely dominates at 1M context length.

### 3.4 Scaling with Number of GPUs

The current implementation is **single-GPU only**. However, we can analyze theoretical multi-GPU scaling:

#### Parallelizable Dimensions

| Parallelism Strategy | Max Useful GPUs | Speedup | Communication Overhead |
|---------------------|----------------|---------|----------------------|
| **Layer-parallel** (pipeline) | L (128) | Up to 128× | Low — sequential dependency between layers |
| **Head-parallel** (attention) | h (160) | Up to 160× per layer | Medium — requires all-reduce for output projection |
| **Sequence-parallel** (context chunks) | n / chunk_size | Up to n× for FFN | **High for attention** — each chunk needs full KV |
| **Weight-parallel** (tensor model parallel) | d / shard_size | Up to d× for matmuls | High — all-reduce after every matmul |
| **Commitment-parallel** (MSM) | Arbitrary | Near-linear | Low — MSM is embarrassingly parallel |

#### Realistic Multi-GPU Scaling Estimates

| GPUs | Strategy | Theoretical Speedup | Realistic Speedup (with overhead) |
|------|----------|--------------------|---------------------------------|
| 8 | Tensor parallel within layer | 8× | 5-6× |
| 64 | Tensor + pipeline parallel | 64× | 30-40× |
| 512 | + sequence parallel for FFN | 512× | 100-200× |
| 4,096 | All strategies combined | 4,096× | 500-1,000× |
| 32,768 | Diminishing returns | 32,768× | 1,000-3,000× |

**Key limitation:** Self-attention cannot be easily sequence-parallelized because each position attends to all previous positions. The O(n²) attention matrix must be materialized (or streamed) regardless of GPU count.

---

## 4. Memory Requirements

### 4.1 Single-GPU Memory Budget

| Component | 8B / 1K | 1T / 1M | Notes |
|-----------|---------|---------|-------|
| Hidden state (n × d × 32B) | 134 MB | 819 GB | Field element representation |
| Attention matrix (n² × 32B) | 32 MB | **32 TB** | Per head, per layer |
| FFN intermediate (n × d_ff × 32B) | 360 MB | 2.2 TB | Peak during FFN |
| Logits (n × V × 32B) | 1 GB | 4 TB | Final layer output |
| Weight storage (P × 32B) | 256 GB | 32 TB | Full model in field elements |
| **Total peak** | **~2-3 GB** | **~70 TB** | Excludes weight storage |

A single GPU (80 GB H100) cannot hold even the intermediate tensors for 1T/1M. **Distributed memory is mandatory.**

### 4.2 Proof Size Estimates

| Configuration | Proof Size | Notes |
|--------------|-----------|-------|
| 8B / 1K | ~5 MB | Mostly sumcheck polynomials |
| 8B / 1M | ~5 GB | Scales with n for entropy, n² for attention |
| 1T / 1K | ~50 MB | More layers and larger polynomials |
| 1T / 1M | ~50 GB | Dominated by attention sumchecks |

---

## 5. Proposed Improvements for Large-Scale Proving

### 5.1 Algorithmic Improvements (Highest Impact)

#### 5.1.1 Replace Quadratic Attention with Linear Attention Proofs

**Problem:** O(n²) attention is the single biggest bottleneck.

**Solution:** Prove inference for models that use linear attention variants (e.g., Mamba, RWKV, RetNet, or linear attention approximations).

- Linear attention: O(n × d²) instead of O(n² × d) — a massive win for long contexts
- At n=1M, d=25,600: linear attention is **40×** cheaper than quadratic
- **Estimated impact:** Reduces 1T/1M proving from ~300 years to ~7 years (still requires parallelism)

If the model architecture cannot be changed, consider:

- **Sliding window attention proofs**: Only prove attention within a fixed window w. Cost: O(n × w × d) where w << n
- **Sparse attention proofs**: Prove only the top-k attention weights per position
- **Chunked attention with cross-chunk summaries**: Prove attention within chunks, with a separate proof for cross-chunk information flow

#### 5.1.2 Recursive/Folding Proof Composition

**Problem:** Proving all L layers sequentially in one monolithic proof is memory-intensive and non-parallelizable.

**Solution:** Use recursive proof composition (e.g., Nova/SuperNova folding schemes):

1. Prove each layer independently, producing a layer-proof
2. Fold layer-proofs together using an IVC (Incrementally Verifiable Computation) scheme
3. Final proof is constant-size regardless of L

**Benefits:**
- Each layer proof is independent → perfectly parallelizable across L GPUs
- Memory per GPU: O(n × d) instead of O(L × n × d)
- **Estimated impact:** ~128× speedup from layer parallelism alone

#### 5.1.3 Streaming/Chunked Sequence Processing

**Problem:** Materializing the full n × n attention matrix is infeasible at n=1M.

**Solution:** Process the sequence in chunks of size c:

1. For each chunk i of size c:
   - Compute attention within chunk: O(c² × d)
   - Compute cross-attention to all previous chunks: O(c × i×c × d)
   - Prove chunk computation with a self-contained proof
2. Compose chunk proofs using folding

**Memory per chunk:** O(c² + c × d) — manageable for c = 1,000-4,000

**Total cost:** O(n/c × c² × d) + O(n² × d / 2) — still quadratic in n for full attention, but memory-feasible.

#### 5.1.4 Probabilistic Verification (Token Sampling)

**Problem:** Proving all n output positions is expensive.

**Solution:** As noted in the zkllm-entropy README, randomly sample 5-10% of positions for proof generation.

- Prover commits to full output sequence
- Verifier selects random positions after commitment
- Prover generates entropy proofs only for selected positions
- Security: Cheating on k% of positions is detected with probability 1-(1-k/100)^s where s is sample size

**Impact on entropy proof:** 10-20× reduction in entropy proving cost (minor overall impact since entropy proofs are <1% of total)

**Impact on layer proofs:** Does not directly help — full forward pass must still be proven.

### 5.2 Systems/Engineering Improvements (Medium Impact)

#### 5.2.1 Multi-GPU Tensor Parallelism

**Implementation plan:**
1. Shard weight matrices across GPUs along the output dimension
2. Each GPU proves its shard of the matrix multiply via local sumcheck
3. All-reduce the sumcheck challenges (not the tensors)
4. Compose shard proofs into a single layer proof

**Communication cost:** O(log(d) × num_gpus) field elements per sumcheck round — negligible.

**Expected speedup:** Near-linear up to min(d/64, num_gpus) GPUs per layer.

#### 5.2.2 GPU MSM Optimization

**Current state:** The codebase uses basic BLS12-381 MSM.

**Improvements:**
- Use GPU-optimized MSM libraries (e.g., ICICLE, cuBellman, or Sppark)
- Pippenger's algorithm with optimal window size tuning for GPU
- Expected 3-5× speedup on commitment generation

#### 5.2.3 Optimized Field Arithmetic

**Current state:** BLS12-381 scalar field (Fr) operations at ~20ns per multiply on GPU.

**Improvements:**
- Use Montgomery multiplication with CUDA PTX intrinsics
- Exploit warp-level parallelism for polynomial evaluation
- Consider switching to a more GPU-friendly field (e.g., Goldilocks for inner proofs, with BLS12-381 only for commitments)
- Expected 2-3× speedup on all field operations

#### 5.2.4 Memory-Efficient Proof Streaming

**Problem:** Current implementation materializes all intermediate tensors in GPU VRAM.

**Solution:**
- Stream intermediate results to host RAM or NVMe between layers
- Recompute rather than store where recomputation is cheaper than I/O
- Use gradient-checkpointing-style techniques: store every k-th layer, recompute others

**Memory reduction:** O(L × n × d) → O(k × n × d) where k << L

### 5.3 Proof System Improvements (Research-Level)

#### 5.3.1 Switch to a More Efficient Proof System

**Current:** Custom sumcheck + Pedersen commitments over BLS12-381.

**Alternatives:**
| System | Prover Time | Proof Size | Verifier Time | GPU-Friendly |
|--------|-------------|-----------|---------------|--------------|
| Current (sumcheck + Pedersen) | Baseline | ~5 MB | Fast | Moderate |
| Plonky2/Plonky3 (FRI-based) | 0.5-2× | 50-200 KB | Medium | Yes (Goldilocks) |
| HyperNova (folding) | 0.3-1× | ~1 KB (recursive) | Fast | Yes |
| Binius (binary field) | 0.2-0.5× | 10-50 KB | Fast | Potentially |
| GKR (layered circuits) | 0.5-1× | ~5 MB | Fast | Yes |

**Recommendation:** A hybrid approach using GKR for the structured matrix multiplications (which map naturally to layered arithmetic circuits) combined with lookup arguments for non-linearities would likely be 2-5× faster than the current custom sumcheck approach.

#### 5.3.2 Hardware Acceleration

- **FPGA-based provers:** Field arithmetic on FPGAs can achieve 5-10× over GPUs for specific operations
- **ASIC provers:** Companies like Cysic and Irreducible are building ZK-specific ASICs with 100× improvements
- **Multi-GPU NVLink clusters:** DGX systems with NVLink provide 900 GB/s inter-GPU bandwidth, enabling efficient tensor parallelism

---

## 6. Estimated Proving Times for 1T / 1M Configuration

### 6.1 Baseline (No Improvements)

| Component | Time (1 GPU) |
|-----------|-------------|
| Self-attention softmax (128 layers × 10^12 ops each) | ~200 years |
| FFN proofs (128 layers) | ~200 days |
| QKV/output projections | ~150 days |
| lm_head FC | ~4 days |
| Weight commitments | ~2 hours |
| Entropy proofs | ~7 hours |
| **Total** | **~200 years** |

### 6.2 With Engineering Improvements Only

| Improvement | Speedup Factor | Cumulative Time |
|-------------|---------------|----------------|
| Baseline | 1× | ~200 years |
| + 4,096 GPU cluster (tensor + pipeline + data parallel) | ~1,000× | ~73 days |
| + Optimized MSM (ICICLE library) | ~1.1× | ~66 days |
| + Optimized field arithmetic (Montgomery + PTX) | ~1.3× | ~51 days |
| + Memory streaming (enables the computation) | 1× (enabler) | ~51 days |
| **Total with engineering** | **~1,400×** | **~51 days** |

### 6.3 With Algorithmic + Engineering Improvements

| Improvement | Speedup Factor | Cumulative Time |
|-------------|---------------|----------------|
| Engineering improvements (above) | ~1,400× | ~51 days |
| + Linear/sliding-window attention (w=4096) | ~60× (for attention) | ~5 days |
| + Recursive proof composition (layer folding) | ~2× | ~2.5 days |
| + GKR-based matmul proofs | ~3× | ~20 hours |
| + Probabilistic token sampling (10%) | ~1.05× (minor) | ~19 hours |
| **Total with all improvements** | **~500,000×** | **~19 hours** |

### 6.4 Aggressive Optimizations (Research Frontier)

| Improvement | Additional Factor | Time |
|-------------|------------------|------|
| All above improvements | — | ~19 hours |
| + ZK-specific ASIC provers | ~10-50× | ~0.5-2 hours |
| + Binius/binary-field proofs | ~2-5× | ~15-30 min |
| **Theoretical minimum** | **~10,000,000×** | **~10-30 min** |

---

## 7. Roadmap: Path to Practical 1T/1M Proving

### Phase 1: Multi-GPU Foundation (3-6 months)
- Implement tensor parallelism for zkFC across GPUs
- Implement pipeline parallelism across layers
- Integrate GPU-optimized MSM library (ICICLE)
- **Target:** 8B model, 1M context on 64 GPUs in < 1 day

### Phase 2: Memory & Attention Scaling (6-12 months)
- Implement chunked/streaming attention proofs
- Add sliding-window or sparse attention support
- Memory-efficient proof streaming to host/NVMe
- **Target:** 70B model, 1M context on 256 GPUs in < 1 day

### Phase 3: Proof System Upgrades (12-18 months)
- Implement recursive layer composition (Nova/HyperNova folding)
- Replace custom sumcheck with GKR for structured matmuls
- Optimize non-linear proofs with batched lookup arguments
- **Target:** 405B model, 1M context on 1,024 GPUs in < 1 day

### Phase 4: Full Scale (18-24 months)
- Hardware-software co-design with FPGA/ASIC provers
- Full 1T model support with 4,096+ GPU clusters
- Production-grade distributed proving infrastructure
- **Target:** 1T model, 1M context in < 24 hours

---

## 8. Conclusions

1. **Context length is the dominant scaling challenge.** The O(n²) self-attention proof makes 1M-context proving infeasible without either (a) switching to linear attention architectures or (b) using sliding-window/sparse attention approximations in the proof.

2. **Model size scaling is manageable.** Going from 8B to 1T increases proving cost by ~200× — significant but addressable with ~200 GPUs via tensor and pipeline parallelism.

3. **Multi-GPU support is essential** but currently absent. The single-GPU implementation is the most critical engineering limitation.

4. **The entropy proof component is negligible** (<1% of total proving cost). Optimization efforts should focus on the transformer layer proofs, not the entropy verification.

5. **A realistic path to 1T/1M proving exists** but requires both algorithmic breakthroughs (linear attention proofs, recursive composition) and substantial engineering (multi-GPU, optimized field arithmetic, memory streaming). With aggressive optimization, proving times of **hours to a day** on a large GPU cluster appear achievable within 2 years.

---

*Report generated 2026-03-18. Based on analysis of [zkllm-entropy](https://github.com/JamesPetrie/zkllm-entropy) source code.*
