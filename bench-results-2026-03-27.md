# Benchmark Results: Field Arithmetic and Proof Timing (2026-03-27)

Hardware: NVIDIA H100 PCIe (sm_90), driver 590, CUDA 13.1.

## Experiment 1: Field Arithmetic Throughput

**Goal.** Measure raw multiply throughput for three field sizes to validate the claimed 10–25× speedup from switching to smaller proof fields.

**Method.** Standalone CUDA kernel (`bench_field_arith.cu`) performing 400 million field multiplications (2M elements × 200 iterations) per field. Each kernel runs a tight multiply-accumulate loop. Warmup pass before timing. Timed with `cudaEvent`.

| Field | Size | Throughput (billion muls/s) | Speedup vs BLS12-381 |
|---|---|---|---|
| BLS12-381 (current) | 255-bit | 5.6 | 1× |
| Goldilocks | 64-bit | 86.0 | 15.4× |
| Mersenne31 | 31-bit | 323.8 | 58.1× |

**Interpretation.** The Goldilocks speedup (15.4×) falls squarely within the paper's 10–20× estimate. The Mersenne31 raw speedup (58×) exceeds the paper's 12–25× estimate — the paper's conservative figure accounts for overflow handling (wider accumulators, range checks) that would reduce the net gain in a full proof system. These are raw multiply throughput numbers; the actual end-to-end speedup would be lower due to memory bandwidth, non-multiply operations, and any overflow mitigation, but they confirm the fundamental arithmetic advantage of smaller fields.

## Experiment 2: Proof Timing Breakdown

**Goal.** Identify where proving time is spent to guide optimization priorities, and provide a baseline for future scaling experiments.

**Method.** Instrumented version of `zkllm_entropy` (`zkllm_entropy_timed.cu`) with `Timer` checkpoints around each phase. Ran on Llama-2-7B with 1024-token context, sigma_eff=5223. The binary takes the precomputed layer-31 hidden state as input and proves the final RMSNorm, lm_head linear layer, and per-token conditional entropy.

| Phase | Time (s) | % of Total |
|---|---|---|
| Load data + weights | 4.3 | 0.6% |
| RMSNorm compute | 0.03 | 0.0% |
| lm_head compute | 30.1 | 4.4% |
| Entropy compute | 3.6 | 0.5% |
| **Entropy prove** | **634.3** | **92.6%** |
| lm_head prove | 7.9 | 1.2% |
| RMSNorm prove | 4.1 | 0.6% |
| Serialize proof | 0.3 | 0.0% |
| **TOTAL** | **684.8** | **100%** |

**Interpretation.** The entropy proof dominates at 92.6% of total time. This phase runs per-token argmax, normal CDF lookup, and log lookup proofs — each involving sumcheck rounds and table lookup range proofs over the 32,000-element vocabulary. The lm_head matmul (compute + prove combined) accounts for only ~5.5%, and RMSNorm is negligible.

This has two implications:

1. **Field size optimization would help.** The entropy prove phase is sumcheck-heavy, so its runtime scales roughly with field multiply cost. A 15× speedup from Goldilocks would reduce this phase from ~634s to ~42s, bringing total time from ~685s to ~93s.

2. **The per-token lookup proofs are the bottleneck, not the linear algebra.** Future optimization should focus on the argmax/CDF/log proof pipeline. Breaking down the entropy prove phase further (into argmax, CDF, and log sub-timings) would identify which sub-component dominates.

## Experiment 3: Pedersen vs SHA-256 Merkle Tree Commitment

**Goal.** Compare the current Pedersen commitment scheme (elliptic curve multi-scalar multiplication, not post-quantum) against a hash-based Merkle tree (SHA-256, post-quantum). This is relevant to the design goal of post-quantum security: if hash-based commitments are fast enough, switching from Pedersen eliminates the only non-post-quantum component.

**Method.** Benchmark (`bench_commitment.cu`) commits to the same vector of random field elements using both schemes. Pedersen uses the existing `commit_int()` (bit-by-bit EC scalar multiplication with 31-bit signed integers). SHA-256 Merkle tree hashes each 32-byte field element as a leaf, then binary-tree reduces with SHA-256 pair hashing. Three trials, average reported. Warmup pass before timing.

### Results at 131K elements (4 MB, ~one small layer)

| Scheme | Avg Time (s) | Throughput (M elements/s) | Speedup |
|---|---|---|---|
| Pedersen (EC scalar mul) | 0.099 | 1.3 | 1× |
| SHA-256 Merkle tree | 0.0012 | 110.4 | **84×** |

### Results at 4M elements (128 MB, ~one large weight matrix)

| Scheme | Avg Time (s) | Throughput (M elements/s) | Speedup |
|---|---|---|---|
| Pedersen (EC scalar mul) | 2.95 | 1.4 | 1× |
| SHA-256 Merkle tree | 0.0093 | 450.7 | **318×** |

**Interpretation.** SHA-256 Merkle tree commitment is 84–318× faster than Pedersen on GPU, with the gap widening at larger sizes (the Merkle tree benefits more from memory bandwidth at scale). Pedersen throughput is roughly constant at ~1.4 M elements/s regardless of size, while SHA-256 scales from 110 to 451 M elements/s as parallelism increases.

This has several implications:

1. **Commitment is not currently the bottleneck** for the entropy proof tail (Experiment 2 showed the entropy prove phase dominates). But for the full 32-layer proof pipeline, Pedersen commitment of all model weights (~7B parameters = ~7 billion field elements) would take ~5,000 seconds (~83 minutes) at 1.4 M/s. SHA-256 Merkle trees would do the same in ~16 seconds.

2. **Post-quantum security is essentially free for commitments.** The hash-based scheme is not only post-quantum but also dramatically faster. The cost of post-quantum commitments is in the proof system, not the commitment step.

3. **Other hash functions would be even faster.** SHA-256 was chosen as a conservative baseline. Blake3 (used by SP1 and Risc0) and Poseidon/Poseidon2 (used by Plonky2/Plonky3, designed for algebraic circuits) are alternatives worth benchmarking. For committing *outside* the proof (as done here), Blake3 would likely be fastest. For commitments that must be *opened inside* a proof, Poseidon over Goldilocks would be the natural pairing if the proof field is switched to Goldilocks.

4. **Caveat: commitment opening inside the proof.** These benchmarks measure commitment *construction* only. The commitment scheme also affects the *opening* protocol used during proof verification. Pedersen commitments have a natural algebraic opening that composes directly with the sumcheck protocol — the prover opens the commitment at a random evaluation point using a multiexponentiation, and the verifier checks it with a single pairing or inner product. Hash-based commitments (SHA-256 Merkle trees) require a fundamentally different opening strategy, such as FRI-based polynomial commitment (used in STARKs, Plonky2/3) or direct Merkle path verification. FRI-based opening adds its own proving cost (Reed-Solomon encoding, multiple query rounds), which could partially offset the commitment speed advantage. The real question is total system cost — commitment construction plus opening inside the proof — not commitment speed alone. Nevertheless, the 2–3 order-of-magnitude gap in commitment construction, combined with the post-quantum requirement, strongly suggests hash-based commitments are the right direction.

5. **Interactive proofs dramatically reduce FRI query cost.** In a non-interactive (Fiat-Shamir) setting, FRI over a 64-bit field would need ~50 queries per opening to reach 100-bit security (each query adds ~2 bits of soundness). But in an interactive setting — where the output is committed before the verifier sends challenges — a cheating prover is caught with probability 1 − 2⁻⁶⁴ (~1 − 10⁻¹⁹) on a *single* query, because they cannot grind offline. This reduces FRI to **1 query per opening**, cutting prover work by ~50× and making the opening phase essentially free. The dominant FRI cost becomes just the NTT and Merkle tree construction during commitment (~0.2s for 4M elements over Goldilocks, vs 2.95s for Pedersen). This advantage compounds across the whole proof: each sumcheck round ends with a polynomial opening, so 50× fewer queries per opening multiplied across hundreds of openings is a major saving.

## Next Steps

- **Sub-breakdown of entropy prove:** Instrument `zkConditionalEntropy::prove()` to separate argmax, CDF, and log proving times.
- **Context length scaling:** Run at seq_len = 64, 128, 256, 512, 1024 to verify O(n) scaling of the entropy proof (which is per-token, so should be linear) vs O(n²) for self-attention layers.
- **Poseidon and Blake3 benchmarks:** Compare against SHA-256 for both raw commitment throughput and in-proof verification cost.
