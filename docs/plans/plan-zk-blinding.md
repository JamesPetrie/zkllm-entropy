# Plan: Adding Zero-Knowledge to the Sumcheck + FRI Pipeline

**Date:** 2026-04-01
**Status:** Partially implemented (see Implementation Status below)
**Prerequisite:** Current FRI + Goldilocks implementation (goldilocks-fri branch)

---

## Problem

The current system does **not achieve zero-knowledge** (documented in `docs/analysis/security-review.md`):

1. **FRI itself is not ZK** — query responses reveal information about committed polynomials
2. **Per-position intermediate values** (logit gaps, win probabilities, surprises) are revealed in the clear during sumcheck rounds
3. **Cross-layer junctions** expose random evaluations of intermediate polynomials when one proof step's output feeds into the next

## Background: How Plonky2 Handles This

Plonky2 (Polygon Zero, 2022) uses three techniques to add ZK to a FRI-based proof system:

1. **Blind trace polynomials before committing** — add random rows to the trace so that the committed polynomial is already randomized when it enters FRI
2. **Evaluate on a coset of H** — FRI evaluations are on a shifted domain, so no evaluation directly reveals a witness value
3. **Salt Merkle leaves** — each leaf hashes `(evaluation || random_salt)`, making the Merkle tree a hiding commitment

These techniques are independent of PLONK's constraint system and apply equally to sumcheck-based proofs. Plonky2 notes in footnote 6 that "FRI lacks the zero-knowledge property of a PCS opening" — their ZK comes from blinding _before_ FRI, not from FRI itself.

Reference: Plonky2 whitepaper, Section 3.6 (https://docs.rs/crate/plonky2/latest/source/plonky2.pdf)

## Why Not Switch to PLONK or VOLE-Based ZK?

**PLONK** would replace sumcheck with a general-purpose constraint system. This is worse for our use case because sumcheck over multilinear extensions exploits the algebraic structure of matrix multiplication (O(MKN) prover time vs O(MKN log(MKN)) for PLONK's NTT over a flattened trace). The ZK techniques are orthogonal to the proof system choice.

**VOLE-based ZK** (emp-zk / QuickSilver / Mystique) achieves ZK inherently — all wire values are secret-shared. However, it requires O(n) interactive communication (1 field element per multiplication gate), making it impractical at LLM scale. Mystique proved ResNet-101 (42.5M params) with 1.98 GB communication on CPU. Extrapolating to LLaMA-7B would require ~330 GB of real-time interactive communication.

## Estimated Overhead

Adding ZK to the current pipeline costs < 0.1% of proving time. The dominant cost is field multiplications in the matmul kernel (54% of wall time at ~3.9s/layer). ZK adds only field additions and a small number of extra Merkle commitments.

| ZK technique | Added cost per layer | Relative overhead |
|---|---|---|
| Blinding polynomials (field additions) | ~2ms | 0.03% |
| Coset evaluation | ~0ms (already done) or ~3ms | 0-0.04% |
| Salted Merkle leaves | ~0ms (same SHA-256 block count) | 0% |
| Extra masking polynomial commits (~6) | ~5ms | 0.07% |
| Extra FRI openings for blinding polys | ~3ms | 0.04% |
| **Total** | **~10ms** | **< 0.15%** |

The reason: ZK adds operations in the commitment/opening layer (additions, hashing), but the bottleneck is in the proof computation layer (field multiplications). These are decoupled.

---

## Design

### Proof Pipeline (Simplified Single-Layer)

```
hidden → zkRMSNorm → normed → zkFC(W_lm) → logits → zkEntropy → H_avg
```

Each arrow is a junction where blinding must be maintained.

### Technique 1: Masked Sumcheck (Per-Step ZK)

For each sumcheck in the pipeline, the prover masks the polynomial being summed.

**Example: zkFC matmul sumcheck**

Without ZK, the sumcheck proves:

```
logits̃(r) = Σ_{c ∈ {0,1}^k} W̃(r, c) · x̃(c)
```

In each round j, the prover sends a degree-2 univariate s_j(X) that is a partial sum over half the remaining hypercube. These s_j reveal weighted sums of W and x entries.

**With ZK:**

1. Prover samples a random masking function ρ: {0,1}^k → F with the constraint Σ_c ρ(c) = 0 (so the mask doesn't change the claimed sum). Concretely: sample ρ(c) for all c except one, set the last to make the sum zero.

2. Prover commits to the MLE of ρ (one additional Merkle tree).

3. Run the sumcheck on h(c) = W̃(r,c) · x̃(c) + ρ(c) instead of W̃(r,c) · x̃(c).

4. Each round message s_j(X) is now uniformly random subject to s_j(0) + s_j(1) = previous claim. A simulator can produce this distribution without knowing W or x — just pick a random degree-2 polynomial satisfying the sum constraint.

**Cost per sumcheck:** One extra polynomial commitment for ρ (~1ms for NTT + Merkle over the sumcheck domain).

### Technique 2: Blinded FRI Commitments

Before committing any witness polynomial via FRI:

1. Sample a random polynomial r_f of the same degree as witness polynomial f
2. Commit to f + r_f (the blinded version)
3. Salt each Merkle leaf: hash `(evaluation || 8-byte random salt)` — still fits in one SHA-256 block (64 bytes), so no extra hash calls

When the prover needs to open f at a point z:
- Open the blinded commitment at z → reveals (f + r_f)(z)
- Open r_f at z separately (from its own commitment, or from a batched commitment of all blinding polynomials)
- Verifier computes f(z) = (f + r_f)(z) - r_f(z)

The FRI query responses now reveal blinded values, which look uniformly random.

### Technique 3: Cross-Layer Blinding

This is the hard part. Two approaches, in order of implementation complexity:

#### Approach A: Commitment Chaining (Recommended First)

At each junction, re-commit the intermediate value with fresh blinding and prove the two commitments are consistent.

```
Step A output: Commit(logits̃ + r_A) = C_A
Step B input:  Commit(logits̃ + r_B) = C_B  (fresh randomness)

Prove C_A and C_B commit to the same polynomial:
  1. Verifier sends random challenge ζ
  2. Prover opens both at ζ:
     - C_A(ζ) = logits̃(ζ) + r_A(ζ)
     - C_B(ζ) = logits̃(ζ) + r_B(ζ)
  3. Prover also opens r_A(ζ) and r_B(ζ)
  4. Verifier checks: C_A(ζ) - r_A(ζ) == C_B(ζ) - r_B(ζ)
```

**Information leaked:** The verifier learns logits̃(ζ) at one random point per junction. This is one random linear combination of all V=32000 logit values — negligible information about the actual logit distribution.

**Advantage:** Each proof step remains self-contained. ZK can be added incrementally to individual sub-proofs without refactoring the pipeline.

**Disadvantage:** The verifier learns O(1) random evaluations per junction. For k junctions, the verifier learns k scalar equations in ~32000·k unknowns — information-theoretically negligible but formally not perfect ZK.

#### Approach B: Batched Deferred Openings (Stronger ZK)

Never open any polynomial individually. Accumulate all opening claims and verify in one batched FRI at the end.

**Protocol:**

```
PROVER                                    VERIFIER

--- Phase 1: Commitments ---
Commit all blinded polynomials:
  C_W = Commit(W̃ + r_W)
  C_h = Commit(hiddeñ + r_h)
  C_ρ₁ = Commit(ρ̃₁)  (mask for RMSNorm)
  ...                                     (Merkle roots)

--- Phase 2: All sumchecks ---
For each sub-proof (RMSNorm, FC, CDF,
  row-sum, quotient, log, accumulate):
                                           ← random challenges
  Run masked sumcheck (technique 1)
  Sumcheck produces final claim:
    "fᵢ(zᵢ) = vᵢ"
  → DEFER claim, don't open yet            (claim added to batch)

Only H_avg (final entropy) is revealed
unblinded — it's the public output.

--- Phase 3: Batch verification ---
Prover has N deferred claims:
  (f₁, z₁, v₁), ..., (fₙ, zₙ, vₙ)
                                           ← batch challenge α
Prover constructs batched quotient:
  B(x) = Σᵢ αⁱ · (fᵢ(x) - vᵢ) / (x - zᵢ)

If all claims correct, B(x) is a
polynomial (divisions are exact).

Commit(B) and run single FRI on B.        ↔ FRI query/response
                                           Verify B passes FRI.
                                           ACCEPT / REJECT
```

**What the verifier learns at each step:**

| Step | Verifier sees | Information |
|---|---|---|
| Commitments | Merkle roots | Nothing (hiding) |
| Sumcheck rounds | Masked sⱼ(X) | Nothing (random due to ρ) |
| Deferred claims vᵢ | vᵢ = real_value + ρᵢ(uᵢ) | Nothing (masked) |
| Batch FRI queries | Blinded evaluations | Nothing (blinded by rᵢ) |
| **H_avg** | **Entropy bound** | **The one public output** |

**Cross-layer consistency is automatic:** Both steps A and B reference the same committed polynomial (same Merkle root). The batch FRI verifies that all claims against this commitment are consistent. The verifier never sees any individual intermediate evaluation.

**Advantage:** Formal honest-verifier zero-knowledge. No intermediate values revealed. The batch FRI may also be cheaper than multiple individual FRIs (amortized setup cost).

**Disadvantage:** Requires refactoring the pipeline to accumulate claims rather than verify inline. The prover must hold all deferred polynomials in memory until the batch phase.

---

## Implementation Plan

### Phase 1: Per-Sumcheck Masking (Approach A junctions)

1. **Add masking polynomial generation to sumcheck**
   - In `src/proof/proof.cu`: before each sumcheck, generate random ρ with Σρ = 0
   - Add ρ's contribution to each round's partial evaluation
   - Commit ρ via existing Merkle infrastructure

2. **Blind FRI commitments**
   - In `src/commit/fri_pcs.cu`: before `commit()`, add random polynomial r_f
   - Store r_f alongside f for later opening
   - Salt Merkle leaves with per-leaf randomness in `src/commit/merkle.cu`

3. **Add junction equality proofs**
   - At each pipeline junction, implement the Approach A consistency check
   - Open both commitments at a verifier-chosen point, verify equality

4. **Update verifier**
   - Extend the Python verifier to handle masked sumcheck transcripts
   - Verify junction equality proofs
   - Check that only H_avg is revealed unblinded

### Phase 2: Batched Deferred Openings (Approach B)

5. **Add claim accumulator**
   - New data structure to collect (polynomial, point, value) triples
   - Each sumcheck appends its final claim instead of opening immediately

6. **Implement batched quotient polynomial**
   - Construct B(x) = Σᵢ αⁱ(fᵢ(x) - vᵢ)/(x - zᵢ) on GPU
   - This is a batch polynomial division — can be done via NTT

7. **Single batched FRI**
   - Replace per-opening FRI calls with one FRI on B(x)
   - Update proof serialization format

8. **Remove junction equality proofs** (no longer needed — batch handles consistency)

---

## Current PCS Architecture — Gap Analysis

**Critical:** The current `FriPcs` implementation (`src/commit/fri_pcs.cu`) does **not** use FRI for opening proofs. The verifier recomputes MLE evaluations directly from the raw data. The code comments note:

> "For a succinct third-party verifier (who doesn't have the data), a quotient polynomial evaluation argument would be needed on top of FRI. This is left for future work."

**This architecture cannot serve the target specification.** The spec requires the verifier to have only commitments and hashes — W, x, o, K are all private. The verifier cannot recompute MLE evaluations because it doesn't have the underlying data. A real FRI opening protocol (quotient polynomial argument) is **mandatory**, not optional.

This means:
- The batched FRI (Approach B) is a **requirement**, not an optimization. Without it, the verifier has no way to check polynomial evaluations against commitments.
- ZK matters for the full proof transcript — sumcheck round messages, final claims, and FRI query responses all must hide private inputs.
- The current `FriPcs::open()` must be replaced with a proper quotient-based opening proof.

---

## FRI vs Pedersen: Full Prover Cost Comparison

### Why the "318× commitment speedup" is misleading

The README reports that FRI+Goldilocks commits at 450M elements/s vs Pedersen+BLS12-381 at 1.4M elements/s (318× ratio). This compares only the **commitment** step — building a Merkle tree vs computing an MSM.

This is not a fair comparison because the two schemes have fundamentally different cost structures:

- **Pedersen/KZG:** Expensive commit (MSM), cheap open (homomorphic — one evaluation proof per claim)
- **FRI:** Cheap commit (Merkle hash), expensive open (NTT over blown-up domain + FRI folding + Merkle trees at every layer)

For the target spec, where the verifier doesn't have the data, both commit AND open costs matter.

### FRI+Goldilocks: Full commit+open cost

For a polynomial of degree n, FRI commit+open requires:

**Commit:**
1. NTT to evaluate on coset domain of size 2n (blowup factor 2): O(2n log(2n)) Goldilocks multiplies
2. Merkle tree over 2n leaves: O(2n) SHA-256 hashes

**Open (prove f(z) = v):**
1. Compute quotient q(x) = (f(x) - v) / (x - z): O(2n) field ops
2. FRI on q(x): log(n) folding rounds, each halving the domain
   - Each round: half-size NTT + Merkle tree
   - Total across all rounds: ~2 × 2n field ops (geometric sum) + ~2 × 2n hashes

**Concrete numbers for W_lm (degree 2^27, blowup 2, on H100):**

| Operation | Field ops | Hashes | GPU time (est.) |
|---|---|---|---|
| NTT (commit) | 28 × 2^28 ≈ 7.5B | — | ~580ms |
| Merkle tree (commit) | — | 2^28 ≈ 268M | ~540ms |
| Quotient computation | 2^28 | — | ~20ms |
| FRI folding NTTs (all rounds) | ~2 × 2^28 ≈ 537M | — | ~40ms |
| FRI Merkle trees (all rounds) | — | ~2 × 2^28 ≈ 537M | ~1070ms |
| **Total** | **~8.3B** | **~1.1B** | **~2.3s** |

Throughput assumptions: Goldilocks multiply at 13 Gops/s, SHA-256 Merkle at ~500M hashes/s (H100).

### Pedersen+BLS12-381: Full commit+open cost

| Operation | GPU time (est.) |
|---|---|
| Commit (MSM, 2^27 points) | ~92s |
| Open (quotient MSM, 2^27 points) | ~92s |
| **Total** | **~184s** |

MSM throughput: ~1.4M elements/s on H100 (measured in this codebase).

### Honest comparison

| Metric | FRI+Goldilocks | Pedersen+BLS12-381 | Ratio |
|---|---|---|---|
| Commit only | ~1.1s | ~92s | 84× |
| Open only | ~1.1s | ~92s | 84× |
| **Commit + open** | **~2.3s** | **~184s** | **~80×** |
| ZK overhead | ~10ms (blinding, salting) | 0 (inherent) | — |
| Post-quantum | Yes | No | — |
| Proof size | O(log²n) field elements | O(1) group elements | KZG wins |

**The real speedup is ~80×, not 318×.** The 318× figure counts only the Merkle hash vs MSM commit — it ignores FRI's opening cost entirely. With the full opening protocol required by the spec, FRI+Goldilocks is still much faster, but the gap narrows from 318× to ~80×.

### Why FRI still wins despite opening costs

1. **Goldilocks arithmetic is ~10× faster** than BLS12-381 per multiply on H100 (64-bit vs 381-bit)
2. **NTT parallelizes near-perfectly** on GPU (independent butterfly operations) — unlike MSM's bucket reduction which has cross-thread synchronization
3. **SHA-256 hashing is trivially parallel** (independent leaves), achieving near-peak GPU throughput
4. **FRI folding rounds are geometrically shrinking** — round k operates on 2^(27-k) elements, so the total work across all rounds equals ~2× the first round (not 27× as a naive sum would suggest)

### Batched opening amortization

With the batched quotient approach (Approach B), multiple opening claims share a single FRI proof. If the prover has N claims against polynomials of max degree 2^27:

- **Without batching:** N × 2.3s = N individual FRI proofs
- **With batching:** One FRI proof over B(x) of degree 2^27, cost ~2.3s regardless of N

This amortization is critical: a single inference layer may produce ~10 opening claims (W, logits, win_probs, normed_hidden, masking polynomials, etc.). Batching reduces the opening cost from ~23s to ~2.3s.

For Pedersen, opening is already cheap per-claim (homomorphic), so batching matters less. But the commit cost dominates and is unavoidable.

### Academic context

- **NP Labs PCS benchmarks** (CPU, single-threaded, n=2^20): Ligero (hash-based) total 513ms vs KZG total 1080ms vs Hyrax (Pedersen) total 583ms. At this scale, hash-based and EC-based are roughly comparable. The GPU advantage tips the balance decisively toward hash-based schemes.
- **DeepFold** (USENIX Security 2025): Reports 3.6× faster than HyperPlonk+mKZG for multilinear evaluation commitments, using a FRI-like folding approach.
- **pc-bench**: FRI "10× faster" than MSM on single CPU core but "struggles with parallelization" — this does not apply to GPU where both NTT and Merkle hashing parallelize well.

---

## Merkle Tree Sizing for Batched FRI (Approach B)

### Committed polynomial sizes

For the entropy layer proof (lm_head + entropy), the committed polynomials are:

| Polynomial | Tensor shape | Elements | Padded to power of 2 |
|---|---|---|---|
| W_lm (lm_head weights) | 4096 × 32000 | 131,072,000 | 2^27 = 134,217,728 |
| logits | 1024 × 32000 | 32,768,000 | 2^25 = 33,554,432 |
| win_probs | 1024 × 32000 | 32,768,000 | 2^25 |
| normed_hidden | 1024 × 4096 | 4,194,304 | 2^22 |
| Per-token vectors (surprise, q, etc.) | 1024 | 1,024 | 2^10 |
| Lookup tables (CDF, log) | 32,768 | 32,768 | 2^15 |
| Masking polynomials ρᵢ (~6) | ≤ 1024 × 32000 | ≤ 32M each | ≤ 2^25 |

### B(x) degree and FRI domain

The batched quotient polynomial B(x) = Σᵢ αⁱ · (fᵢ(x) - vᵢ) / (x - zᵢ) has degree max(deg(fᵢ)) - 1.

With the default blowup factor of 2 (`FRI_DEFAULT_PARAMS` in `fri.cuh`), the FRI evaluation domain is 2× the polynomial degree.

**If all polynomials are batched together (including W_lm):**

| | Value |
|---|---|
| B(x) degree | 2^27 - 1 |
| FRI domain size | 2^28 = 268M points |
| Domain in bytes (Goldilocks) | 2^28 × 8 = 2 GB |

**If W_lm is excluded from the batch (verified separately):**

| | Value |
|---|---|
| B(x) degree | 2^25 - 1 |
| FRI domain size | 2^26 = 67M points |
| Domain in bytes | 2^26 × 8 = 512 MB |

### FRI Merkle tree memory

Each FRI folding layer requires a Merkle tree. Each internal node is 32 bytes (SHA-256). A Merkle tree over n leaves has 2n - 1 nodes.

**FRI commits are streaming** — each layer's tree can be freed after extracting the root and generating query proofs. Peak memory is dominated by the first (largest) layer.

**Including W_lm (B(x) degree 2^27):**

| | Size |
|---|---|
| Layer 0 Merkle tree (peak) | 2 × 2^28 × 32 B = **16 GB** |
| Polynomial evaluations | 2^28 × 8 B = 2 GB |
| **Peak FRI memory** | **~18 GB** |
| Sum across all layers | ~32 GB (but streaming, not simultaneous) |

**Excluding W_lm (B(x) degree 2^25):**

| | Size |
|---|---|
| Layer 0 Merkle tree (peak) | 2 × 2^26 × 32 B = **4 GB** |
| Polynomial evaluations | 2^26 × 8 B = 512 MB |
| **Peak FRI memory** | **~4.5 GB** |

### Separating the weight commitment

W_lm is static — it doesn't change between inferences. This suggests:

1. The weight commitment C_W is verified **once** (e.g., via a one-time FRI opening proof, or the verifier obtains the weights through a secure channel and checks the Merkle root).
2. Per-inference proofs batch only the **dynamic** polynomials (logits, entropy intermediates, masking polynomials).
3. The matmul sumcheck ends with a claim "W̃(r,u) · normed̃(u) = v" — the W̃(r,u) component is verified against C_W separately, while normed̃(u) and v enter the per-inference batch.

This reduces the per-inference FRI from 2^28 domain to 2^26 domain — a 4× reduction in memory and compute.

**Note:** The exact mechanism for "verify W̃(r,u) against C_W separately" needs design. Options include:
- A standalone FRI opening proof for W̃ at the challenge point (one FRI per inference, but only over the weight polynomial)
- The verifier caches the weight data and recomputes W̃(r,u) directly (current approach, not succinct)
- A batch of all weight-related claims across multiple inferences into a single periodic FRI proof

---

## Open Questions

1. **MLE-to-univariate conversion for FRI.** The committed data are flat vectors, treated as multilinear extensions in the sumcheck. For FRI, these need to be treated as evaluations of a univariate polynomial. The natural mapping (vector of n elements → unique polynomial of degree n-1 interpolating those values) works, but the relationship between MLE evaluation points and FRI evaluation points needs to be made precise. The current `FriPcs::open` computes MLE evaluations directly — a batched FRI approach would need to reconcile the sumcheck's MLE evaluation claims with FRI's univariate low-degree test.

2. **Interaction with interactive proof model.** Our design uses interactive proofs (verifier supplies fresh challenges). This simplifies ZK significantly — the verifier can't grind on challenges, and honest-verifier ZK is sufficient. If we ever move to Fiat-Shamir (non-interactive), the blinding requirements become stricter.

3. **Masking polynomial degree.** For a degree-d sumcheck polynomial, the mask ρ should have degree d+1 per variable to ensure the round messages are uniformly random. For our inner product sumchecks (d=2), this means degree-3 round messages — 4 field elements per round instead of 3. Negligible communication increase.

4. **Compatibility with tLookup.** The LogUp-based lookup arguments (CDF, log) have their own sumcheck structure. The masking technique applies but needs to preserve the LogUp identity Σ 1/(table - witness) = 0. Need to verify that adding a mask to the LogUp sumcheck doesn't break the lookup relation.

5. **Formal security proof.** The honest-verifier ZK property should be formally argued for the composed protocol — specifically that the simulator can produce transcripts for the full pipeline (not just individual sumchecks) without knowing any private inputs.

6. **Weight opening mechanism.** If W_lm is excluded from the per-inference batch, the matmul sumcheck's final claim still requires verifying W̃(r,u) against C_W. The current approach (verifier has all weight data) works but is not succinct. A succinct approach needs a separate FRI opening proof for the weight polynomial, or a periodic batched proof covering multiple inferences.

7. **NTT cost for B(x).** Computing B(x) on the FRI domain requires an NTT of size equal to the domain. At 2^26 (excluding W_lm), this is ~2^26 × 26 ≈ 1.7B field operations. At the measured Goldilocks multiply throughput of ~13 Gops/s on H100, this is roughly ~130ms. Not negligible but within budget. Needs measurement.

---

## Implementation Status (2026-04-03)

The per-sumcheck masking (Technique 1) has been partially implemented on branch `zk-masking-implementation`. The approach chosen differs from the random-polynomial-with-zero-sum masking described above: instead, it uses **vanishing polynomial masking** + **XZZ+19 transcript masking** (documented in `appendix-pqzk-feasibility.md`).

### What's implemented

| Component | Status | Files |
|-----------|--------|-------|
| Vanishing polynomial masking types and helpers | Done | `src/proof/zk_mask.cuh`, `src/proof/zk_mask.cu` |
| Transcript masking (g+ρ·p) | Done | `src/proof/zk_mask.cu` |
| Degree-4 sumcheck kernel (`zkip_zk`) | Done | `src/proof/zk_sumcheck.cu` |
| Stacked sumcheck kernel (`zkip_stacked_zk`) | Done | `src/proof/zk_sumcheck.cu` |
| Flat-vector format (`inner_product_sumcheck_zk`) | Done | `src/proof/zk_sumcheck.cu` |
| Lagrange interpolation (`Polynomial::from_evaluations`) | Done | `src/poly/polynomial.cu` |
| CPU verifier for flat ZK proofs | Done | `verifier/sumcheck_verifier.h` |
| CPU verifier for stacked ZK proofs | Not started | — |
| Integration: zkFC | Done | `src/zknn/zkfc.cu` |
| Integration: zkAttn/zkAttnStacked | Done | `src/zknn/zksoftmax.cu` |
| Integration: zkEntropy (4 sumcheck sites) | Done | `src/entropy/zkentropy.cu` |
| Integration: multi_hadamard_sumchecks | Deferred (variable-degree) | — |
| Verifier negative tests | Not started | — |
| Fiat-Shamir / interactive ρ | Not started (ρ is prover-generated) | — |
| FRI blinding (Technique 2) | Not started | — |
| Cross-layer blinding (Technique 3) | Not started | — |

### Key design decisions

1. **Vanishing polynomial masking** (Z_f(X) = f(X) + Σ c_i·X_i(1-X_i)) was chosen over random-polynomial-with-zero-sum because it requires no additional polynomial commitment — the vanishing terms are zero on {0,1}^k, so the committed data on the hypercube is unchanged.

2. **Degree increase**: The product of two masked degree-2 polynomials is degree-4, requiring 5 evaluations per round instead of 3. This is a ~1.67x overhead per sumcheck round.

3. **Variable ordering**: `zkip_zk` processes variables in reverse order (variable b-1-j at round j), matching the existing `zkip` convention. The transcript mask round polynomial handles arbitrary variable binding order.

### Known gaps (from independent audit, 2026-04-03)

- **No verifier negative tests**: `verify_zk_ip_sumcheck` has no tests that verify rejection of tampered proofs.
- **ρ generated by prover**: In production, ρ must be a verifier challenge (Fiat-Shamir or interactive). Currently all call sites do `Fr_t rho = random_vec(1)[0]`.
- **No stacked ZK verifier**: `zkip_stacked_zk` proofs (used by zkFC stacked and zkAttn stacked) cannot be verified yet.
- **GPU tests pending**: Build succeeds but GPU tests blocked by GPU reset needed on H100.
