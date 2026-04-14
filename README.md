# Zero-Knowledge Conditional Entropy Bounds for LLM Inference Verification

## With Probabilistic Tolerance for Hardware Non-Determinism

This repository forks [zkLLM](https://github.com/jvhs0706/zkllm-ccs2024) and adds a zero-knowledge proof of conditional entropy for LLM inference outputs. `main` uses the original BLS12-381 field and Pedersen (Hyrax-style) commitments — the same cryptographic stack as the upstream zkLLM paper. The `pq-goldilocks` branch ports the pipeline to the Goldilocks field with FRI / Merkle commitments for post-quantum security and faster GPU field arithmetic.

---

## Summary

We developed a zero-knowledge proof system that verifies LLM inference without requiring exact floating-point reproducibility. Instead of proving "the model produced exactly these tokens," we prove "at most H bits of hidden information could be encoded in this output." This handles real-world GPU non-determinism while still bounding covert channels like model weight exfiltration.

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
- Model weights W (committed via Pedersen commitment — hidden from verifier)
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

$$\hat{C}(O) = -\sum_{i=1}^{m} \log_2 \hat{q}(o_i; W, I, o_{\lt i})$$

### Handling hardware non-determinism

Real GPU inference is non-deterministic: the same model on the same input can produce slightly different logits due to floating-point accumulation order. We model this as Gaussian noise on logits:

$$q(o_i) = \frac{\Phi\left(\frac{v^{*} - \ell_i}{\sigma\sqrt{2}}\right)}{\sum_j \Phi\left(\frac{v^{*} - \ell_j}{\sigma\sqrt{2}}\right)}$$

Where:
- v\* is the maximum logit (greedy decoding)
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

`main` uses the upstream zkLLM cryptographic stack:

- **Field:** BLS12-381 scalar field (255-bit, Montgomery form)
- **Commitment:** Pedersen (EC multi-scalar multiplication), Hyrax-style with trusted-setup generators
- **Opening:** EC recursive halving
- **Sumchecks:** Inner-product, Hadamard-product, and Boolean sumchecks built on top of the Pedersen commitment layer

The `pq-goldilocks` branch swaps BLS12-381 for the 64-bit Goldilocks field and replaces Pedersen with FRI / SHA-256 Merkle commitments, eliminating the trusted setup and providing post-quantum security. See that branch's README for performance comparisons.

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

### Batched entropy proof pipeline (inside zkConditionalEntropy)

All T positions are processed as a single T×V tensor. Only the aggregate entropy H is revealed — no per-token scalars leak.

1. **zkArgmax** — per-position bit-decomposition range proof that t\* is the argmax
2. **GPU diffs** — `diffs[t,i] = v*[t] − logits[t,i]` (T×V tensor, all non-negative)
3. **zkNormalCDF** — CDF tLookup: `win_probs[t,i] = (1 − Φ(diff/σ)) × cdf_scale` (one proof over T×V)
4. **Row-sum sumcheck** — proves `total_win[t] = Σ_i win_probs[t,i]` via partial MLE + inner product sumcheck
5. **Indicator extraction** — proves `wp[t] = win_probs[t, token[t]]` via inner product with indicator tensor
6. **Quotient-remainder** — proves `q[t] = floor(wp[t] × 2^p / total_win[t])` via division relation `q·tw + r = wp·2^p` with bit-decomposition range proofs on q, r, and (tw−r−1)
7. **Surprise lookup** — `surprise[t] = −log₂(q[t] / 2^p) × log_scale` via tLookup
8. **Accumulate** — `H = Σ_t surprise[t]`

A slack-free redesign of this pipeline (sandwich-based argmax and surprise, no quotient-remainder) is described in `docs/analysis/entropy-redesign-plan.md`.

### Proof protocol

The system uses **interactive proofs** (verifier provides fresh random challenges). Because the output is committed before proof generation begins, a failed proof is a detection event. Soundness error per challenge is negligible in the BLS12-381 scalar field.

### Proof modules

| Module | File | Role |
|---|---|---|
| BLS12-381 field | `src/field/bls12-381.cu/cuh` | 255-bit prime field arithmetic (Montgomery form) |
| Commitment | `src/commit/commitment.cu/cuh` | Pedersen commitment (MSM) |
| Sumchecks | `src/proof/proof.cu/cuh` | Inner product, Hadamard, binary sumchecks |
| Polynomial | `src/poly/polynomial.cu/cuh` | Low-degree polynomial manipulation for sumcheck rounds |
| tLookup | `src/zknn/tlookup.cu/cuh` | LogUp lookup argument (range + mapping) |
| zkArgmax | `src/zknn/zkargmax.cu/cuh` | Argmax via bit-decomposition range proof |
| zkNormalCDF | `src/zknn/zknormalcdf.cu/cuh` | Normal CDF via lookup table |
| zkLog | `src/zknn/zklog.cu/cuh` | −log₂ via lookup table |
| zkEntropy | `src/entropy/zkentropy.cu/cuh` | Batched conditional entropy proof |
| zkFC | `src/zknn/zkfc.cu/cuh` | Fully connected layer proof |
| Rescaling | `src/zknn/rescaling.cu/cuh` | Fixed-point rescaling proof |
| zkReLU | `src/zknn/zkrelu.cu/cuh` | ReLU proof |
| zkSoftmax | `src/zknn/zksoftmax.cu/cuh` | Softmax proof (segmented exponential) |

### Test suite

| Binary | Coverage |
|---|---|
| `test_zkargmax` | Argmax via bit-decomposition range proof |
| `test_zklog` | `−log₂` lookup table proof |
| `test_zknormalcdf` | Normal CDF lookup table proof |
| `test_zkentropy` | Batched entropy: argmax, surprise, proof generation, consistency |

---

## Quickstart

Requires an NVIDIA GPU (sm_90 / H100 recommended) and CUDA 13+.

### Build

```bash
cd zkllm-entropy
make clean && make -j64 all
```

### Run tests

```bash
./test_zkargmax && ./test_zklog && ./test_zknormalcdf && ./test_zkentropy
```

### Run the entropy prover

```bash
./zkllm_entropy \
    <logits_dir> \
    <tokens.txt> \
    <proof_output.bin> \
    <sigma_eff>
```

### Verify a proof (Python)

```bash
python python/verify_entropy.py proof.bin
```

---

## Calibrating σ

The noise parameter σ should be calibrated empirically:

1. Run the same model on the same input twice with temperature 0
2. Count what fraction of output tokens match exactly
3. Find σ such that the noise model predicts this match rate

Target: ~97% exact match rate. For Llama-2-7B, σ ≈ 0.05 (σ_eff = 3277 with 65536 scaling).

```bash
python python/calibrate_sigma.py --target-match-rate 0.97
```

---

## Known Soundness Gaps

The mathematical framework (sumcheck + LogUp + Gibbs' inequality) is sound. The implementation has engineering gaps that are fixable without changing the cryptographic design. See `docs/analysis/prover-determinism.md` for a detailed analysis of residual prover freedom in zkLLM-style protocols and the sandwich construction that eliminates it.

### Proof completeness gaps

1. **Weight-binding proofs not serialized.** The prover runs `verifyWeightClaim` and `zkFC` locally, but these proofs are not written to the proof file. The proof currently shows that *some* logits yield entropy H, but doesn't prove those logits came from committed weights.

2. **Verifier is arithmetic-only.** `verify_entropy.py` recomputes CDF/log/quantization from scalar proof values and checks consistency. It does not verify any sumcheck, commitment opening, tLookup proof, or argmax range proof. A full cryptographic verifier has not been implemented.

### Other issues

- Binary sumcheck proofs in zkArgmax are computed but written to a local vector that is discarded (one-line fix)
- `cdf_precision` defaults differ between prover (15) and verifier (12)
- Negative diffs produce warnings but not errors

---

## Next Steps

Planned improvements, roughly in priority order.

### 1. Execute the slack-free entropy redesign

Replace the bit-decomposition zkArgmax and the quotient-remainder surprise with the sandwich construction described in `docs/analysis/entropy-redesign-plan.md`. This removes prover freedom around rounding and eliminates ~32 binary sumchecks per position.

### 2. Fix binary sumcheck serialization

One-line change: pass `proof` instead of `bin_proof` to `Fr_bin_sc()` in zkargmax.cu. Immediate soundness improvement.

### 3. Store all proof parameters in header

Add `cdf_precision`, `cdf_scale`, `bit_width` to the proof file header so the verifier reads them rather than using potentially-mismatched defaults.

### 4. Build a cryptographic verifier

Replace `verify_entropy.py` with a verifier that checks sumcheck polynomials, tLookup proofs, Pedersen openings, and commitment bindings. This is the largest remaining engineering effort.

### 5. Serialize weight-binding proofs

Write `verifyWeightClaim`, `zkFC`, and `Rescaling` proof elements into the proof file. Required for a third-party verifier to confirm that logits derive from committed weights.

---

## Repository Structure

```
src/
  field/          bls12-381.cu/cuh — field arithmetic
  tensor/         fr-tensor.cu/cuh, g1-tensor.cu/cuh — GPU tensor operations
  poly/           polynomial.cu/cuh — polynomials
  commit/         commitment.cu/cuh — Pedersen commitment
  proof/          proof.cu/cuh — sumcheck protocols, Weight struct
  zknn/           tlookup, zkargmax, zknormalcdf, zklog, zkfc,
                  zksoftmax, rescaling, zkrelu — proof modules
  entropy/        zkentropy.cu/cuh — conditional entropy proof
  llm/            rmsnorm, ffn, self-attn, etc. — LLM layer proofs
  util/           ioutils, timer — I/O and timing helpers
test/
  test_*.cu       Test binaries
docs/analysis/
  prover-determinism.md           Sources of prover freedom and the sandwich fix
  entropy-redesign-plan.md        Slack-free redesign of the entropy stage
python/
  verify_entropy.py       Python verifier (arithmetic-only)
  calibrate_sigma.py      Sigma calibration tool
  commit_final_layers.py  Weight commitment scripts
  run_proofs.py           Per-layer proof orchestration
  llama-*.py              Llama-2 layer proof runners
Makefile                  Build system (BLS12-381 / Pedersen)
```

The Goldilocks/FRI port, post-quantum commitment scheme, NTT, SHA-256 Merkle trees, and CPU verifier all live on the `pq-goldilocks` branch.

---

## References

- [zkLLM (CCS 2024)](https://github.com/jvhs0706/zkllm-ccs2024) — base proof system for LLM inference
- [Rinberg et al. (2024)](https://arxiv.org/pdf/2511.02620) — inference verification without ZK
- [Karvonen et al. (2024)](https://arxiv.org/abs/2511.20621) — model fingerprinting
