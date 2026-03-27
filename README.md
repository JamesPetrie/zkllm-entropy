# Zero-Knowledge Conditional Entropy Bounds for LLM Inference Verification

## With Probabilistic Tolerance for Hardware Non-Determinism

This repository forks [zkLLM](https://github.com/jvhs0706/zkllm-ccs2024) and adds a zero-knowledge proof of conditional entropy for LLM inference outputs. The `goldilocks-fri` branch replaces BLS12-381 / Pedersen commitments with Goldilocks field arithmetic and FRI-based hash commitments (SHA-256 Merkle trees), achieving ~10× faster field operations, 3× smaller proofs, and post-quantum security.

---

## Summary

We developed a zero-knowledge proof system that verifies LLM inference without requiring exact floating-point reproducibility. Instead of proving "the model produced exactly these tokens," we prove "at most H bits of hidden information could be encoded in this output." This handles real-world GPU non-determinism while still bounding covert channels like model weight exfiltration. Proof generation is ~1000× slower than inference, but random sampling makes it practical.

### Applications

- Find bugs in an inference stack
- Verify that the model purchased is the same one that is served
- Verify that regulator evals run on the correct model
- Detect model weight exfiltration through API outputs
- Verify compute usage for international agreements about AI

### Why zero knowledge?

- Enables verification of closed-weight models where the verifier may not have access to model weights
- Protects against model weight theft even if the verifier has model access
- Formal security guarantees that don't rely on a complex software and hardware stack

---

## What Is Proven

**Inputs (public or committed):**
- Model weights W (committed via Merkle tree — hidden from verifier)
- Input token sequence x (public)
- Output token sequence o (public)
- Hardware noise parameter σ (public, calibrated empirically)

**Output:**
- Conditional entropy bound H (public)
- Interactive zero-knowledge proof π

**The proof guarantees:**

Given the committed model weights W and public input x, the claimed output sequence o has conditional entropy at most H bits. This means at most H bits of covert information could be hidden in the output.

---

## The Conditional Entropy Argument

### Why conditional entropy bounds covert channel capacity

The prover claims their node is running a workload consisting of declared model weights, W, and a declared set of algorithms, A. The verifier monitors the declared channel to measure the inputs, I, and outputs O. Let D = A(I, W) denote the set of outputs that could be produced by running the declared algorithm(s) on the measured inputs.

We define the covert message capacity, C(O), of an output as the information in O that could be used to encode a covert payload. The conditional entropy H(O | D), the information in O not determined by the declared computation D, provides an upper bound:

$$C(O) \leq H(O \mid D)$$

If inference were perfectly deterministic, H(O | D) would be zero. In practice, current frontier AI implementations exhibit hardware nondeterminism even at temperature zero, mainly because the order of operations can vary between runs, and floating-point arithmetic is nonassociative. A dishonest prover could mimic this noise to encode covert information. To be conservative, we treat all unexplained entropy as potential covert message capacity.

### Why approximations give valid upper bounds

The verifier does not know the true distribution P(O | D), but can estimate it. Let Q(O | D) be any estimate of this distribution.

Gibbs' inequality states that for any true distribution P and estimate Q, then H(P) ≤ -E_P[log₂ Q], with equality iff P = Q. Combining with C(O) ≤ H(O | D):

$$C(O) \leq H(O \mid D) \leq -\mathbb{E}_{P}\left[\log_2 Q(O \mid D)\right]$$

This bound holds for any estimate Q: a better estimate gives a tighter bound.

We compute Q(O | D) by recomputing token probabilities autoregressively and multiplying them. The prover may propose any approximation they wish, for instance by specifying the exact inference algorithm, random seed handling, or sampling procedure used. Since Gibbs' inequality guarantees the bound holds for any estimate Q, a prover who provides a more accurate approximation only tightens the bound on their own covert capacity, creating an incentive for honest specification.

Taking the log gives the **estimated covert message capacity**, the sum of per-token surprisals:

$$\hat{C}(O) = -\sum_{i=1}^{m} \log_2 \hat{q}(o_i; W, I, o_{<i})$$

### Handling hardware non-determinism

Real GPU inference is non-deterministic: the same model on the same input can produce slightly different logits due to floating-point accumulation order. We model this as Gaussian noise on logits:

$$q(o_i) = \frac{\Phi\left(\frac{v^* - \ell_i}{\sigma\sqrt{2}}\right)}{\sum_j \Phi\left(\frac{v^* - \ell_j}{\sigma\sqrt{2}}\right)}$$

Where:
- v* is the maximum logit (greedy decoding)
- ℓ_i is the logit for token i
- σ is calibrated so the noise model predicts ~97% exact token matches
- Φ is the standard normal CDF

This gives the probability that token i would be selected under hardware noise, even if it wasn't the exact argmax in the verifier's computation. This approach extends to non-zero temperature sampling via the Gumbel max trick.

---

## What Was Missing Before This Project

Existing work (e.g. [zkLLM](https://github.com/jvhs0706/zkllm-ccs2024), [zkML](https://github.com/zkml), [Rinberg et al.](https://arxiv.org/pdf/2511.02620), and [Karvonen et al.](https://arxiv.org/abs/2511.20621)) either verifies exact inference on finite fields (not using floating point tensor cores), or classifies whether a token stream is consistent with a claimed model without a zero knowledge guarantee. There was no zero knowledge verification of efficient floating point inference — to use prior ZK approaches you would have had to fully rewrite the inference stack to use finite fields instead of floating point.

---

## Current Implementation

### Cryptographic stack

The `goldilocks-fri` branch supports two field/commitment backends, selected at compile time:

| | BLS12-381 (legacy) | Goldilocks (current) |
|---|---|---|
| **Field** | 255-bit, Montgomery form | 64-bit (p = 2⁶⁴ − 2³² + 1) |
| **Commitment** | Pedersen (EC multi-scalar mul) | SHA-256 Merkle tree |
| **Opening** | EC recursive halving | FRI polynomial commitment |
| **Post-quantum** | No | Yes |
| **Field multiply** | 25.4 ms / 33M ops | 2.6 ms / 33M ops (9.8× faster) |
| **Commitment speed** | 1.4 M elements/s | 450 M elements/s (318× faster) |
| **Proof size (1024 tok)** | 216 KB | 72 KB (3× smaller) |

All development is on the Goldilocks backend. BLS12-381 is retained for comparison.

### Architecture

```
committed W_norm  committed W_lm           public σ_eff
     |                  |                        |
zkRMSNorm(hidden, W_norm) → normed_hidden        |
     |                                           |
zkFC(normed_hidden, W_lm) → logits              |
     |                                           |
zkConditionalEntropy(logits, tokens, σ_eff) → entropy bound H
```

The hidden state from layer 31 comes from the existing zkLLM layer proof pipeline. The entropy layer proves the final RMSNorm, lm_head linear layer, and per-token conditional entropy.

### Per-position entropy pipeline (inside zkConditionalEntropy)

1. **zkArgmax** — bit-decomposition range proof that t\* is the argmax
2. **GPU diffs** — `diffs[i] = v* − logits[i]` (all non-negative)
3. **zkNormalCDF** — CDF lookup: `win_probs[i] = (1 − Φ(diff_i/σ)) × cdf_scale`
4. **Normalize** — `total_win = sum(win_probs)`; `q_idx = floor(win_prob[actual] × 2^log_precision / total_win)`
5. **zkLog** — surprise = `−log₂(q_idx / 2^log_precision) × log_scale`
6. **Accumulate** over sequence → total entropy bound H

### Proof protocol

The system uses **interactive proofs** (verifier provides fresh random challenges). Because the output is committed before proof generation begins, a failed proof is a detection event. At the 64-bit Goldilocks field size, a cheating prover is caught with probability 1 − 2⁻⁶⁴ (~1 − 10⁻¹⁹) per challenge, eliminating the need for proof repetitions.

Interactive proofs also reduce FRI to 1 query per opening (vs ~50 for non-interactive at 100-bit security), cutting prover work substantially.

### Proof modules

| Module | File | Role |
|---|---|---|
| Goldilocks field | `goldilocks.cu/cuh` | 64-bit prime field arithmetic |
| NTT | `ntt.cu/cuh` | Number-theoretic transform (domains up to 2³²) |
| Merkle tree | `merkle.cu/cuh` | GPU SHA-256 Merkle commitment |
| FRI | `fri.cu/cuh` | FRI polynomial commitment protocol |
| FRI PCS | `fri_pcs.cu/cuh` | Integration layer: commit, open, MLE |
| Sumchecks | `proof.cu/cuh` | Inner product, Hadamard, binary sumchecks |
| tLookup | `tlookup.cu/cuh` | LogUp lookup argument (range + mapping) |
| zkArgmax | `zkargmax.cu/cuh` | Argmax via bit-decomposition range proof |
| zkNormalCDF | `zknormalcdf.cu/cuh` | Normal CDF via lookup table |
| zkLog | `zklog.cu/cuh` | −log₂ via lookup table |
| zkEntropy | `zkentropy.cu/cuh` | Per-token entropy pipeline |
| zkFC | `zkfc.cu/cuh` | Fully connected layer proof |
| Rescaling | `rescaling.cu/cuh` | Fixed-point rescaling proof |
| zkReLU | `zkrelu.cu/cuh` | ReLU proof |
| zkSoftmax | `zksoftmax.cu/cuh` | Softmax proof (segmented exponential) |

### Test suite

92 tests pass across 6 Goldilocks test binaries:

| Binary | Tests | Coverage |
|---|---|---|
| `test_goldilocks` | 36 | Field arithmetic, edge cases, algebraic properties |
| `test_gold_tensor` | 18 | Tensor ops, MLE, inner product sumcheck, matmul |
| `test_ntt` | 6 | Forward/inverse/coset NTT, root of unity, round-trip |
| `test_merkle` | 9 | Deterministic root, proof verification, tamper detection |
| `test_fri` | 6 | Small/medium/large polynomial commit-prove-verify |
| `test_fri_pcs` | 17 | MLE evaluation, binding check, sumcheck integration, Weight |

Additional BLS12-381 test binaries: `test_zkargmax` (6), `test_zklog` (5), `test_zknormalcdf` (5), `test_zkentropy` (6).

### Performance (H100 PCIe, Llama-2-7B, 1024 tokens)

**BLS12-381 baseline timing:**

| Phase | Time (s) | % |
|---|---|---|
| lm_head compute | 30.1 | 4.4% |
| Entropy compute | 3.6 | 0.5% |
| **Entropy prove** | **634.3** | **92.6%** |
| lm_head prove | 7.9 | 1.2% |
| RMSNorm prove | 4.1 | 0.6% |
| Other | 4.6 | 0.7% |
| **Total** | **684.8** | |

The entropy prove phase dominates (92.6%), driven by per-token sumcheck rounds for argmax, CDF, and log proofs over the 32K vocabulary. With Goldilocks field arithmetic (9.8× faster multiply), this phase is projected to drop from ~634s to ~65s.

**Goldilocks timed build:** 17.6s for 64 tokens (after 3.7× optimization pass), vs 64.3s before optimization. Full 1024-token Goldilocks timing is pending.

---

## Quickstart

Requires an NVIDIA GPU (sm_90 / H100 recommended) and CUDA 13+.

### Build (Goldilocks mode)

```bash
cd zkllm-entropy
make clean && make -j64 gold_zkllm_entropy test_goldilocks test_gold_tensor test_ntt test_merkle test_fri test_fri_pcs
```

### Run tests

```bash
./test_goldilocks && ./test_gold_tensor && ./test_ntt && ./test_merkle && ./test_fri && ./test_fri_pcs
```

### Run the entropy prover (Goldilocks)

```bash
./gold_zkllm_entropy \
    <logits_dir> \
    <tokens.txt> \
    <proof_output.bin> \
    <sigma_eff>
```

### Build (BLS12-381 legacy mode)

```bash
make clean && make -j64 all
```

### Verify a proof (Python)

```bash
# Goldilocks
USE_GOLDILOCKS=1 python verify_entropy.py proof.bin

# BLS12-381
python verify_entropy.py proof.bin
```

---

## Calibrating σ

The noise parameter σ should be calibrated empirically:

1. Run the same model on the same input twice with temperature 0
2. Count what fraction of output tokens match exactly
3. Find σ such that the noise model predicts this match rate

Target: ~97% exact match rate. For Llama-2-7B, σ ≈ 0.05 (σ_eff = 3277 with 65536 scaling).

```bash
python calibrate_sigma.py --target-match-rate 0.97
```

---

## Known Soundness Gaps

The mathematical framework (sumcheck + LogUp + Gibbs' inequality) is sound. The implementation has engineering gaps that are fixable without changing the cryptographic design. These are documented in detail in `plan2.md`.

### Proof completeness gaps

1. **Weight-binding proofs not serialized.** The prover runs `verifyWeightClaim` and `zkFC` locally, but these proofs are not written to the proof file. The proof currently shows that *some* logits yield entropy H, but doesn't prove those logits came from committed weights. (Tracked as S3 in plan2.md.)

2. **Verifier is arithmetic-only.** `verify_entropy.py` recomputes CDF/log/quantization from scalar proof values and checks consistency. It does not verify any sumcheck, commitment opening, tLookup proof, or argmax range proof. A full cryptographic verifier has not been implemented. (Tracked as S1 in plan2.md.)

### Per-token information leakage

The current proof format emits 6 scalar values per token position (logit of actual token, logit gap, win probability, total win, quantized probability, surprise). These reveal the model's per-position confidence — not the full logit vector or the weights, but more than a true zero-knowledge proof should reveal.

A planned redesign (P6 in plan2.md) eliminates all per-token leakage by moving the normalization step to log-space (avoiding the variable-denominator division) and batching all intermediate values as committed tensors. Only the aggregate entropy bound H would be revealed. See "Next Steps" below.

### Other issues

- Binary sumcheck proofs in zkArgmax are computed but written to a local vector that is discarded (one-line fix, tracked as S5)
- `cdf_precision` defaults differ between prover (15) and verifier (12) (tracked as S4)
- Negative diffs produce warnings but not errors (tracked as S6)

---

## Next Steps

Planned improvements, roughly in priority order. See `plan2.md` for detailed designs.

### 1. Fix binary sumcheck serialization (S5)

One-line change: pass `proof` instead of `bin_proof` to `Fr_bin_sc()` in zkargmax.cu. Immediate soundness improvement.

### 2. Store all proof parameters in header (S4 + P2)

Add `cdf_precision`, `cdf_scale`, `bit_width` to the proof file header so the verifier reads them rather than using potentially-mismatched defaults.

### 3. Prove total_win via sumcheck (S2)

Add an inner-product sumcheck proving `total_win = sum(win_probs_all)`. Interim fix: use the conservative constant `vocab_size × cdf_scale` as the denominator (always valid, slightly looser bound).

### 4. Merge argmax into CDF lookup

The CDF tLookup proof already implicitly proves non-negativity of all diffs (and therefore argmax correctness), because the LogUp identity operates on the original field elements — a negative diff (near p) cannot match any table entry. Making the CDF table large enough to cover the full diff range (e.g., `cdf_precision = 20`, table = 8 MB) eliminates the need for the separate zkArgmax bit-decomposition proof entirely, replacing 32 binary sumchecks with zero additional work.

### 5. Log-space division trick for true ZK (P6)

Replace the problematic division `q = win_prob / total_win` with a log-space subtraction:

```
surprise = −log₂(win_prob) + log₂(total_win)
```

Two log lookups, no division. This eliminates all 6 per-token scalar emissions — intermediate values stay committed but unopened, and only the aggregate H is revealed. Uses existing `tLookupRangeMapping` infrastructure; the only new component is a second `zkLog` instance with a larger table for `total_win` values.

### 6. Build a cryptographic verifier (S1)

Replace `verify_entropy.py` with a verifier that checks sumcheck polynomials, tLookup proofs, FRI openings, and commitment bindings. This is the largest remaining engineering effort.

### 7. Serialize weight-binding proofs (S3)

Write `verifyWeightClaim`, `zkFC`, and `Rescaling` proof elements into the proof file. Required for a third-party verifier to confirm that logits derive from committed weights.

### 8. Goldilocks field range validation (P7)

Verify that no intermediate value in the proof pipeline overflows the 64-bit Goldilocks modulus (p ≈ 1.8 × 10¹⁹). The entropy layer values (logits, diffs, CDF, win_probs) are comfortably within range (~30–33 bits). The concern is `zkFC` matmul accumulation: summing `in_dim` (4096) products of two ~2³² quantized values gives ~2⁷⁶, which exceeds p. The sumcheck *proof* is valid regardless (it never forms the full accumulation), but the *compute* path that produces logits may wrap. Needs empirical validation on real model weights and a comparison of quantized vs FP16 logit rankings. See P7 in plan2.md for the full analysis.

### 9. Port setup tooling to Goldilocks

The weight commitment scripts (`llama-commit.py`, `commit_final_layers.py`, `ppgen`) and per-layer proof orchestration (`run_proofs.py`, `llama-*.py`) currently only support BLS12-381 / Pedersen. These need Goldilocks + FRI PCS equivalents for full end-to-end proving.

### 10. Performance optimization (Phase 7)

- Poseidon2 hash to replace SHA-256 for faster GPU Merkle trees
- NTT optimization (single-kernel launch, precomputed twiddles)
- Batch Merkle tree construction across multiple weight matrices
- Compressed weight storage (int8/int16 with on-the-fly expansion)

---

## Repository Structure

| Path | Description |
|---|---|
| `goldilocks.cu/cuh` | Goldilocks field arithmetic |
| `bls12-381.cu/cuh` | BLS12-381 field arithmetic (legacy) |
| `ntt.cu/cuh` | Number-theoretic transform |
| `merkle.cu/cuh` | SHA-256 Merkle tree |
| `fri.cu/cuh` | FRI protocol |
| `fri_pcs.cu/cuh` | FRI polynomial commitment scheme |
| `fr-tensor.cu/cuh` | GPU tensor operations over any field |
| `proof.cu/cuh` | Sumcheck protocols, Weight struct |
| `tlookup.cu/cuh` | LogUp lookup argument |
| `zkentropy.cu/cuh` | Conditional entropy proof |
| `zkargmax.cu/cuh` | Argmax proof |
| `zknormalcdf.cu/cuh` | Normal CDF lookup proof |
| `zklog.cu/cuh` | Log₂ lookup proof |
| `zkfc.cu/cuh` | Fully connected layer proof |
| `zksoftmax.cu/cuh` | Softmax proof |
| `rescaling.cu/cuh` | Fixed-point rescaling proof |
| `zkrelu.cu/cuh` | ReLU proof |
| `zkllm_entropy.cu` | Main prover binary |
| `verify_entropy.py` | Python verifier (arithmetic-only) |
| `calibrate_sigma.py` | Sigma calibration tool |
| `plan2.md` | Security review and implementation plan |
| `design-goals.md` | Design principles and threat model |
| `bench-results-*.md` | Performance benchmarks |
| `test_*.cu` | Test binaries |
| `Makefile` | Build system (BLS12-381 and Goldilocks targets) |

---

## References

- [zkLLM (CCS 2024)](https://github.com/jvhs0706/zkllm-ccs2024) — base proof system for LLM inference
- [Rinberg et al. (2024)](https://arxiv.org/pdf/2511.02620) — inference verification without ZK
- [Karvonen et al. (2024)](https://arxiv.org/abs/2511.20621) — model fingerprinting
