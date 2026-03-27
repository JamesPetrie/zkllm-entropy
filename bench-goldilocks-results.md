# Goldilocks + FRI Benchmark Results

**Date:** 2026-03-27
**Hardware:** NVIDIA H100 PCIe (80 GB), sm_90
**Model:** Llama-2-7b (hidden_size=4096, vocab_size=32000)
**Software:** zkllm-entropy, branch `goldilocks-fri`

## Field Arithmetic Microbenchmark

Element-wise operations on N = 33,554,432 (2^25) random field elements.
Each number is the average of 10 repetitions, run sequentially (no GPU contention).

| Operation | Goldilocks (64-bit) | BLS12-381 (256-bit) | Speedup |
|-----------|--------------------:|--------------------:|--------:|
| Add       | 2.22 ms             | 18.37 ms            | **8.3×** |
| Multiply  | 2.58 ms             | 25.36 ms            | **9.8×** |
| Sum       | 8.13 ms             | 24.30 ms            | **3.0×** |
| Add throughput | 15.1 Gop/s     | 1.83 Gop/s          | 8.3×    |
| Mul throughput | 13.0 Gop/s     | 1.32 Gop/s          | 9.8×    |

**Key takeaway:** Goldilocks multiply is ~10× faster than BLS12-381 on the H100. Since field multiplication dominates the sumcheck protocol (which accounts for >90% of proving time), this translates directly to end-to-end speedup.

The sum (reduction) speedup is lower (3×) because reductions are memory-bandwidth-bound rather than compute-bound — both fields read data at the same memory bandwidth, so the 4× smaller element size (8 vs 32 bytes) provides the speedup.

## End-to-End Pipeline: BLS12-381 (1024 tokens, timed)

Per-phase breakdown from `zkllm_entropy_timed` running on Llama-2-7b with 1024 tokens, σ_eff = 3.0.

| Phase              | Time (s) | % of total |
|--------------------|----------:|-----------:|
| Load data + weights | 4.35    | 0.6%       |
| RMSNorm compute    | 0.03     | 0.0%       |
| lm_head compute    | 30.08    | 4.4%       |
| Entropy compute    | 3.60     | 0.5%       |
| **Entropy prove**  | **634.28** | **92.6%** |
| lm_head prove      | 7.88     | 1.2%       |
| RMSNorm prove      | 4.08     | 0.6%       |
| Serialize proof    | 0.34     | 0.0%       |
| **TOTAL**          | **684.80** | 100%     |

**Observation:** Entropy prove dominates at 92.6% of total time. This phase is almost entirely sumcheck (field multiply + add), making it the primary beneficiary of the Goldilocks speedup.

## End-to-End Pipeline: Goldilocks (1024 tokens)

The Goldilocks build (`gold_zkllm_entropy`) completed the full 1024-token pipeline successfully.

- **Total wall time:** ~466 s (7 min 46 s)
- **Entropy bound:** 34.2947 bits total (0.0335 bits/token)
- **Proof file:** 73,768 bytes (6,144 polynomials)
- **Speedup vs BLS:** ~1.47× (684.8 / 466)

**Note:** This run used the non-timed binary, so per-phase breakdown is not available at 1024 tokens. The modest 1.47× overall speedup (vs the expected ~10× from field benchmarks) suggests that the current Goldilocks proving pipeline has bottlenecks beyond raw field arithmetic — likely in:
1. Memory allocation patterns (FrTensor temporaries)
2. Kernel launch overhead (many small kernels in the sumcheck)
3. Hash-based commitment overhead (Merkle tree construction in FRI PCS)
4. The `zkConditionalEntropy::prove` loop which processes positions sequentially

This indicates significant optimization headroom in Phase 7.

## Proof Size Comparison

| Build | Tokens | Polynomials | Proof size |
|-------|-------:|------------:|-----------:|
| BLS12-381 | 1024 | 6,144 | 221,224 bytes (216 KB) |
| Goldilocks | 1024 | 6,144 | 73,768 bytes (72 KB) |
| Goldilocks | 64 | 384 | 4,648 bytes (4.5 KB) |

Goldilocks proofs are **3.0× smaller** than BLS12-381 proofs for the same computation, because each field element is 8 bytes instead of 32 bytes. The polynomial count is identical (same proof structure).

## Correctness Validation

- Both builds produce the same entropy bound for the same input (34.2947 bits at 1024 tokens, σ_eff = 3.0)
- Both builds generate 6,144 proof polynomials for 1024 tokens
- The Goldilocks build completes all proof phases (entropy prove, lm_head prove, RMSNorm prove) and serializes successfully
- Unit test suite: 92/92 tests pass (field, tensor, NTT, Merkle, FRI, FRI PCS)

## Summary

| Metric | BLS12-381 | Goldilocks | Ratio |
|--------|-----------|------------|-------|
| Field multiply | 25.4 ms / 33M ops | 2.6 ms / 33M ops | **9.8× faster** |
| End-to-end (1024 tok) | 684.8 s | ~466 s | **1.47× faster** |
| Proof size (1024 tok) | 216 KB | 72 KB | **3.0× smaller** |
| Security | 255-bit (classical) | 64-bit (per query) | Different model |
| Post-quantum | No (EC-based) | Yes (hash-based) | Goldilocks advantage |

The 10× field-level speedup does not yet translate to 10× end-to-end speedup, indicating that the current pipeline is not fully compute-bound on field arithmetic. Phase 7 optimizations (NTT tuning, Poseidon2 hashing, kernel fusion) are expected to close this gap.
