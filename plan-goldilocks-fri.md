# Implementation Plan: Goldilocks Field + FRI Polynomial Commitment

## Motivation

The current prototype uses BLS12-381 (255-bit field) with Pedersen commitments (elliptic curve multi-scalar multiplication). Benchmarks (see `bench-results-2026-03-27.md`) show:

- **Goldilocks field arithmetic is 15.4× faster** than BLS12-381 on H100.
- **SHA-256 Merkle tree commitment is 84–318× faster** than Pedersen.
- **Entropy proving (92.6% of total time) is sumcheck-heavy**, so it benefits directly from faster field arithmetic.
- **Interactive proofs reduce FRI to 1 query per opening** (64-bit field gives 2⁻⁶⁴ soundness per attempt), eliminating the multi-query overhead that makes FRI expensive in non-interactive settings.

Switching to Goldilocks + FRI (hash-based polynomial commitment) achieves three design goals simultaneously: faster proving, post-quantum security, and elimination of elliptic curve dependencies.

## Architecture Overview

```
Current:    Fr_t (BLS12-381 255-bit)  +  Pedersen (EC multi-scalar mul)  +  me_open()
Proposed:   Gold_t (Goldilocks 64-bit) +  FRI (NTT + Merkle tree)        +  fri_open()
```

The sumcheck protocol itself is field-agnostic — it only needs field add, multiply, and inverse. The commitment scheme is the only component that changes structurally (from algebraic EC opening to FRI-based opening).

## Implementation Status

### Phase 1: Goldilocks Field Arithmetic — COMPLETE ✓

Replaced the BLS12-381 scalar field with Goldilocks (p = 2⁶⁴ − 2³² + 1).

**Files created:**
- `goldilocks.cuh` — `Gold_t` type (single `uint64_t`), all device functions: `gold_add`, `gold_sub`, `gold_mul`, `gold_sqr`, `gold_inverse`, `gold_pow`, `gold_mont`/`gold_unmont` (identity for Goldilocks). Compatibility `#define` macros mapping `blstrs__scalar__Scalar_*` to `gold_*`. Initializer macros `FR_ZERO`, `FR_ONE`, `FR_LITERAL`.
- `goldilocks.cu` — device constant definitions (`gold_ZERO`, `gold_ONE`, `gold_P`).
- `test_goldilocks.cu` — 36 unit tests for all field operations.

**Files modified (via `#ifdef USE_GOLDILOCKS` guards):**
- `fr-tensor.cuh` — conditional include of `goldilocks.cuh` vs `bls12-381.cuh`, guard EC types behind `#ifndef USE_GOLDILOCKS`
- `fr-tensor.cu` — Goldilocks-compatible: `operator<<`, `random_vec`, `random_int_kernel`, `random_kernel`, all `_to_scalar`/`scalar_to_` conversions, `modular_inverse`
- `proof.cuh` — guard `g1-tensor.cuh`, `commitment.cuh` includes and `verifyWeightClaim` declaration
- `proof.cu` — guard `verifyWeightClaim`, fix `operator==` for single-field Gold_t, fix Fr_t literal initializers
- `polynomial.cu` — replace 8-element BLS initializers with `blstrs__scalar__Scalar_ZERO`/`ONE`, guard zero-checks in `operator/` and `inv()`

**Build system:**
- Makefile: `GOLD_FLAG := -DUSE_GOLDILOCKS`, `gold_%.o` pattern rule, standalone Goldilocks targets

**Test results:** 36/36 field tests + 18/18 tensor tests (creation, negatives, add, Hadamard mul, sum, MLE, random, inner product sumcheck, matmul) all pass on H100.

### Phase 2: NTT over Goldilocks — COMPLETE ✓

NTT (Number Theoretic Transform) for polynomial evaluation on multiplicative subgroups.

**Files created:**
- `ntt.cuh` — interface: `get_root_of_unity()`, `ntt_forward()`, `ntt_inverse()`, `ntt_coset_forward()`, `ntt_coset_inverse()`
- `ntt.cu` — Cooley-Tukey radix-2 DIT with bit-reversal permutation. Host-side precomputation of twiddle factors, GPU butterfly kernel. Coset variants multiply by powers of a shift element before/after NTT.
- `test_ntt.cu` — 6 tests: root of unity properties (log_n=1..20), NTT round-trip (n=8), polynomial evaluation at roots of unity (n=4), coset NTT round-trip (n=8), large round-trip (n=2^16), very large round-trip (n=2^20).

**Design notes:**
- Goldilocks has 2³² roots of unity (p−1 = 2³² × (2³²−1)), supporting NTT domains up to 2³².
- Generator g=7 for the full multiplicative group; ω_k = 7^((p−1)/2^k).
- Twiddle factors precomputed on host, uploaded per butterfly stage. This is simple but not optimal — a production version could precompute all twiddles once and reuse.

**Test results:** 6/6 tests pass on H100, including 1M-element round-trip.

### Phase 3: Merkle Tree Commitment — COMPLETE ✓

SHA-256 hash-based Merkle tree replacing Pedersen EC commitment.

**Files created:**
- `merkle.cuh` — `Hash256` type, `MerkleProof` struct, `MerkleTree` class with `root()`, `prove()`, `verify()`
- `merkle.cu` — GPU SHA-256 implementation (compress, leaf hash, pair hash), both device and host versions. `MerkleTree` builds tree on GPU, copies to host for proof generation. Supports both Goldilocks (8-byte) and BLS12-381 (32-byte) leaf formats via `#ifdef`.
- `test_merkle.cu` — 9 tests: deterministic root, different data → different root, all-leaf proof verification (n=8), proof path length, wrong value fails, wrong index fails, large tree proofs (n=2^16), path length verification.

**Design notes:**
- Commitment = 32-byte Merkle root (vs 48-byte compressed EC point for Pedersen).
- Proof = log₂(n) × 32-byte sibling hashes.
- SHA-256 chosen for initial implementation. Can swap to Poseidon2 or Blake3 later without structural changes.

**Test results:** 9/9 tests pass on H100.

### Phase 4: FRI Polynomial Commitment — COMPLETE ✓

Full FRI (Fast Reed-Solomon IOP) protocol for polynomial commitment.

**Files created:**
- `fri.cuh` — `FriParams`, `FriCommitment`, `FriQueryRound`, `FriProof` structs, `FriProver` and `FriVerifier` classes
- `fri.cu` — FRI implementation:
  - **Commit**: coset NTT evaluation + Merkle tree
  - **Fold**: GPU kernel `fri_fold_kernel` implements `f_new[i] = (f[i] + f[i+half])/2 + α(f[i] − f[i+half])/(2x_i)`
  - **Query**: opens positions across all layers with Merkle proofs for both paired indices
  - **Verify**: checks folding consistency at queried positions, verifies Merkle proofs, checks remainder
- `test_fri.cu` — 6 tests: small polynomial (degree 3), multiple queries (degree 7, 3 queries), larger polynomial (degree 1023)

**Design notes:**
- Blowup factor = 2 (evaluation domain = 2× polynomial degree).
- 1 query per opening in the interactive setting (2⁻⁶⁴ soundness).
- Coset offset = (2N)-th root of unity, ensuring evaluation domain doesn't include 0.
- Domain offset squares at each folding round: offset' = offset².

**Test results:** 6/6 tests pass on H100 (after fixing resize+push_back bug and position tracking in verifier).

### Phase 5: Integration Layer — COMPLETE ✓

FRI PCS (Polynomial Commitment Scheme) bridging FRI to the sumcheck-based proof system.

**Files created:**
- `fri_pcs.cuh` — `FriPcsCommitment`, `FriPcsOpeningProof` structs, `FriPcs` class with `commit()`, `open()`, `verify()`, `multilinear_eval_host()`
- `fri_pcs.cu` — Implementation:
  - **Commit**: Merkle tree over raw data (commitment = root hash)
  - **Open**: compute multilinear evaluation via iterative folding, then FRI-commit polynomial and generate proof
  - **Verify**: verify FRI proof (polynomial is low-degree and matches commitment)
  - Fiat-Shamir challenge derivation (simple hash mixing — placeholder for proper transcript)
- `test_fri_pcs.cu` — 19 tests:
  - Commit + open + verify at specific points (n=8)
  - Multilinear evaluation correctness (all corners + midpoint of 4-element MLE)
  - Large vector (n=1024) open + verify
  - FrTensor MLE matches FriPcs MLE (cross-validation)
  - **End-to-end: inner product sumcheck + FRI PCS** — verifies that sumcheck final evaluations a(u), b(u) match FRI PCS openings

**Test results:** 19/19 tests pass on H100. The sumcheck+FRI PCS integration test confirms the two systems produce consistent results.

## Remaining Work: Production Integration

The core infrastructure (Phases 1–5) is complete and tested. The remaining work is to wire the new FRI PCS into the production proof pipeline, replacing the Pedersen commitment path.

### Phase 6: Replace Pedersen Commitment in Proof Pipeline

**Goal:** Make `zkllm_entropy` (and the full 32-layer pipeline) use Goldilocks + FRI instead of BLS12-381 + Pedersen.

**6a. Replace `Weight` struct and `verifyWeightClaim()`**

The current `Weight` struct (`commitment.cuh:28-36`) bundles:
- `Commitment generator` (EC generators for multi-scalar multiplication)
- `FrTensor weight` (quantized weight values)
- `G1TensorJacobian com` (precomputed Pedersen commitment)

Replace with:
```cpp
struct Weight {
    FrTensor weight;
    FriPcsCommitment com;  // Merkle root of weight data
    uint in_dim, out_dim;
};
```

Replace `verifyWeightClaim()` (`proof.cu:4-11`):
```cpp
// Current: opens Pedersen commitment via EC multi-exp
void verifyWeightClaim(const Weight& w, const Claim& c) {
    auto opening = w.generator.open(w_padded, w.com, u_cat);
    if (opening != c.claim) throw ...;
}

// New: opens FRI PCS commitment
void verifyWeightClaim(const Weight& w, const Claim& c) {
    auto proof = FriPcs::open(w.weight.gpu_data, w.weight.size, u_cat);
    if (proof.claimed_value != c.claim) throw ...;
    if (!FriPcs::verify(w.com, u_cat, proof)) throw ...;
}
```

**Files to modify:** `commitment.cuh`, `proof.cuh`, `proof.cu`

**6b. Adapt Goldilocks guards in remaining source files**

These files contain BLS12-381 8-element initializer patterns (`{val, 0, 0, 0, 0, 0, 0, 0}`) and EC-specific code that need `#ifdef USE_GOLDILOCKS` guards:

| File | What needs changing |
|------|-------------------|
| `zkentropy.cu` | Fr_t literal initializers, commitment usage |
| `zkargmax.cu` | Fr_t literal initializers |
| `zklog.cu` | Fr_t literal initializers |
| `zknormalcdf.cu` | Fr_t literal initializers |
| `zksoftmax.cu` | Fr_t literal initializers |
| `zkfc.cu` | Fr_t initializers, Weight/Claim interaction |
| `zkrelu.cu` | Fr_t initializers |
| `tlookup.cu` | Fr_t initializers |
| `rescaling.cu` | Fr_t initializers |

Pattern: replace `{val, 0, 0, 0, 0, 0, 0, 0}` with `FR_LITERAL(val)` or use `blstrs__scalar__Scalar_ZERO`/`ONE` in device code, and explicit `Fr_t fr_zero = {0ULL}` under `#ifdef USE_GOLDILOCKS` in host code.

**6c. Adapt data loading for Goldilocks serialization**

The Weight loading path (`create_weight` in `commitment.cu`) reads binary files with 32-byte BLS12-381 elements. For Goldilocks, elements are 8 bytes. Need:
- New `create_weight_gold()` function or conditional loading in `create_weight()`
- Update `FrTensor::from_int_bin()` if the format changes
- Update Python scripts that generate weight files (`llama-commit.py`, `commit_final_layers.py`)

**6d. Update Python verification and data preparation**

| Script | Change |
|--------|--------|
| `llama-commit.py` | Generate Merkle commitments instead of Pedersen |
| `commit_final_layers.py` | Same |
| `llama-ppgen.py` | Remove (no longer needed — no EC generators to precompute) |
| `verify_entropy.py` | Update proof format parsing for FRI proofs |
| `run_e2e_local.sh` | Update binary names, remove ppgen step |

### Phase 7: Performance Optimization

After correctness is established, optimize for throughput:

**7a. NTT optimization**
- Current implementation: per-stage twiddle upload + kernel launch. Inefficient for large transforms.
- Target: single kernel launch with precomputed twiddle table in constant/shared memory, similar to the existing `blstrs__scalar__Scalar_radix_fft` kernel.
- Expected: 2–5× NTT speedup.

**7b. Batch FRI commitments**
- The 32-layer pipeline commits ~100+ weight matrices. Currently each is committed independently.
- Batch the Merkle tree construction to amortize kernel launch overhead.
- Precompute all NTTs in a pipeline.

**7c. Poseidon2 hash function**
- SHA-256 requires multiple rounds of 32-bit operations per hash.
- Poseidon2 over Goldilocks uses native 64-bit field arithmetic (which we already have) — potentially 3–10× faster than SHA-256 on GPU.
- Drop-in replacement in `merkle.cu` (change leaf/pair hash functions).

**7d. Fiat-Shamir transcript**
- Current: simple hash mixing (placeholder).
- Target: proper Merlin-style transcript or Poseidon2-based sponge for non-interactive proofs (needed if we want to generate proofs offline).

### Phase 8: End-to-End Validation

1. **Correctness**: Run full entropy proof (`run_e2e_local.sh`) with Goldilocks + FRI on the same model/input as the BLS12-381 baseline. Verify identical entropy bound.
2. **Performance**: Benchmark total proving time against the BLS12-381 baseline (684.8s for 1024 tokens). Target: ~45s (15× speedup).
3. **Proof size**: Measure total proof size. Expected: ~775 KB for full model (well within design goal of "up to input size").
4. **Memory**: Profile GPU memory usage. FRI needs additional memory for NTT buffers and Merkle trees vs Pedersen.

## Test Suite Summary

| Test binary | Tests | Status |
|-------------|-------|--------|
| `test_goldilocks` | 36 field arithmetic tests | ✓ Pass |
| `test_gold_tensor` | 18 tensor + sumcheck + matmul tests | ✓ Pass |
| `test_ntt` | 6 NTT tests (up to 2^20 elements) | ✓ Pass |
| `test_merkle` | 9 Merkle tree tests (up to 2^16) | ✓ Pass |
| `test_fri` | 6 FRI tests (up to degree 1023) | ✓ Pass |
| `test_fri_pcs` | 19 integration tests (sumcheck + FRI PCS) | ✓ Pass |
| **Total** | **94 tests** | **All pass** |

## Expected Performance

| Component | BLS12-381 + Pedersen | Goldilocks + FRI | Speedup |
|---|---|---|---|
| Field multiply | 5.6 B/s | 86 B/s | 15.4× |
| Entropy prove (634s) | 634s | ~41s | ~15× |
| lm_head prove (7.9s) | 7.9s | ~0.5s | ~15× |
| Weight commitment (7B params) | ~83 min | ~16s (Merkle) + ~12s (NTT) | ~170× |
| Commitment opening | EC multi-exp | 1-query FRI | cheaper |
| **Total for entropy tail** | **685s** | **~45s** | **~15×** |

## Risk and Open Questions

1. **Overflow in Goldilocks accumulation.** Each multiplication produces a reduced result, so accumulation of n products needs at most 64 + log₂(n) bits. For n = 32,000 (vocab size) this is 79 bits, safely within 128-bit intermediates. Verified in the existing kernels.

2. **NTT domain size.** For a weight matrix of size 4096 × 32000 ≈ 131M elements, NTT domain = 256M elements × 8 bytes = ~2 GB. With blowup factor 2: ~4 GB. H100 has 80 GB — sufficient for a 7B model but may need streaming for larger models.

3. **FRI proof size.** With 1 query per opening: log₂(n) Merkle paths × log₂(ρn) depth. For n = 131M, ρ = 2: ~896 bytes per opening. Total for full model: ~775 KB. Well within design goals.

4. **Poseidon2 vs SHA-256.** SHA-256 is working. Poseidon2 would be faster (native field arithmetic) and enable algebraic in-circuit verification. Can swap later — the FRI/Merkle structure is hash-agnostic.

5. **Fiat-Shamir security.** Current challenge derivation is a placeholder. For production, need a proper transcript (Merlin or Poseidon-based sponge). Only matters for non-interactive proofs.

6. **Compatibility with existing Python pipeline.** Weight files need re-serialization (32 → 8 bytes per element). Commitment files change format (EC point → Merkle root). Scripts need updates but the changes are mechanical.
