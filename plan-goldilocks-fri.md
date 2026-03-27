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

## Implementation Phases

### Phase 1: Goldilocks Field Arithmetic (~1 week)

Replace the BLS12-381 scalar field with Goldilocks (p = 2⁶⁴ − 2³² + 1).

**Files to create:**
- `goldilocks.cuh` — field element type, device functions for add, sub, mul, inverse, Montgomery form
- `goldilocks.cu` — host-side utilities, constants

**Design decisions:**
- `Gold_t` is a single `uint64_t` (vs 8 × `uint32_t` for BLS12-381). This changes the `Fr_t` typedef and every kernel that touches field elements.
- Montgomery multiplication: Goldilocks has a fast reduction because p = 2⁶⁴ − 2³² + 1 allows reduction with shifts and adds instead of general modular reduction. The `bench_field_arith.cu` benchmark already has a working device implementation.
- Batch inverse: implement Montgomery's trick (one inversion + 3(n−1) multiplications) since sumcheck needs inverses.

**Key concern: overflow.** The sumcheck accumulates dot products of field elements. For a dot product of n terms, the intermediate sum can reach n × p². In Goldilocks (64-bit), a single product is 128 bits; accumulating n products needs log₂(n) extra bits. For n = 32,000 (vocabulary size), this needs ~143 bits — safely under 2 × 64 = 128 bits only if we reduce after each multiply-add. The existing `bench_field_arith.cu` Goldilocks multiply already returns a reduced result, so accumulation is safe as long as we reduce per step (not batched). For the self-attention dot product (n up to 1M context length), same approach applies — reduce after each multiply-add.

**Validation:** Run existing test cases (`test_zkargmax`, `test_zklog`, `test_zknormalcdf`, `test_zkentropy`) with Goldilocks field and verify identical proof structure (different field values, same polynomial degrees and claim counts).

### Phase 2: NTT over Goldilocks (~1 week)

FRI requires evaluating polynomials on multiplicative subgroups, which uses the Number Theoretic Transform (NTT).

**Files to create:**
- `ntt.cuh` / `ntt.cu` — forward and inverse NTT over Goldilocks

**Design decisions:**
- Goldilocks p = 2⁶⁴ − 2³² + 1 has multiplicative group order p − 1 = 2³² × (2³² − 1). The 2³² factor means NTT domains up to size 2³² are supported natively (root of unity exists). This is sufficient for any realistic model — 2³² = 4 billion elements covers even 1T-parameter models.
- Standard Cooley-Tukey radix-2 butterfly, parallelized across GPU threads.
- For FRI, we need NTT on coset domains (shifted by a generator), not just the standard domain. Implement as `ntt(data, shift)` where `shift` is the coset offset.

**Validation:** Verify NTT(INTT(x)) = x, and that polynomial evaluation via NTT matches naive evaluation at random points.

### Phase 3: Merkle Tree Commitment (~3 days)

Replace Pedersen commitment with a hash-based Merkle tree.

**Files to create:**
- `merkle.cuh` / `merkle.cu` — Merkle tree construction and opening

**Design decisions:**
- Hash function: SHA-256 initially (already benchmarked). Can swap to Poseidon2 over Goldilocks or Blake3 later.
- Tree structure: binary Merkle tree over the NTT evaluation domain (not the coefficient domain). Store all internal nodes on GPU for fast path extraction.
- Commitment = Merkle root (32 bytes).
- Opening at a point = Merkle authentication path (log₂(n) × 32 bytes).

**Files to modify:**
- `commitment.cuh` — replace `Commitment` class (currently extends `G1TensorJacobian`) with `MerkleCommitment` class that stores root hash and internal nodes.
- `commitment.cu` — replace `commit()`, `commit_int()`, `open()`, `me_open()` with Merkle-based equivalents.

**What gets deleted:**
- `g1-tensor.cuh` / `g1-tensor.cu` — all elliptic curve point operations (no longer needed).
- EC-specific parts of `bls12-381.cuh` / `bls12-381.cu` — the `Fp_t`, `G1Affine_t`, `G1Jacobian_t` types and all curve arithmetic.

### Phase 4: FRI Polynomial Commitment (~2 weeks)

Implement FRI as the polynomial commitment opening protocol, replacing the Pedersen `me_open()`.

**Files to create:**
- `fri.cuh` / `fri.cu` — FRI commit, open, and verify

**FRI commit (done once per committed polynomial):**
1. Compute low-degree extension: evaluate the multilinear polynomial on a domain of size ρ × n (blowup factor ρ, typically 2 or 4).
2. Build Merkle tree over the extended evaluations.
3. Output: Merkle root (the commitment).

**FRI open (done per sumcheck opening):**
1. Verifier sends random challenge α.
2. Prover folds the polynomial: f'(x) = f_even(x) + α · f_odd(x), halving the degree.
3. Prover commits to f' with a new Merkle tree.
4. Repeat for log₂(n) rounds until polynomial is constant.
5. Prover sends the final constant.
6. **Interactive setting: verifier sends 1 query position** (not 50+). Prover responds with Merkle authentication paths at the query position across all rounds. Verifier checks consistency.

**Key design decision: 1 query per opening.** In the interactive setting with 64-bit field, a single query gives soundness error 2⁻⁶⁴. The prover commits output before seeing challenges, so grinding is impossible. This makes FRI openings extremely cheap:
- Prover work per opening: log₂(n) Merkle paths = log₂(n) × log₂(ρn) hashes to extract.
- Verifier work per opening: log₂(n) hash checks + log₂(n) field operations for folding consistency.

**Files to modify:**
- `proof.cuh` / `proof.cu` — replace `verifyWeightClaim()` to use FRI opening instead of `me_open()`.
- The `Weight` struct in `commitment.cuh` changes: remove `Commitment generator` and `G1TensorJacobian com`, replace with `MerkleCommitment com` (just a root hash + cached tree).

**Integration with sumcheck:** Currently, each sumcheck ends with a call to `verifyWeightClaim()` which opens the Pedersen commitment at the random evaluation point produced by the sumcheck. With FRI, the same evaluation point is used — the sumcheck protocol is unchanged, only the opening mechanism at the end differs. The call sites in `zkllm_entropy.cu:180,188`, `self-attn.cu:65-67`, `ffn.cu:84,90,93`, and `rmsnorm.cu:49` all go through `verifyWeightClaim()`, so they only need the signature to change.

### Phase 5: Integration and Testing (~1 week)

**Files to modify (call sites — minimal changes):**
- `zkllm_entropy.cu` / `zkllm_entropy_timed.cu` — update `Weight` loading, remove EC generators
- `self-attn.cu`, `ffn.cu`, `rmsnorm.cu`, `skip-connection.cu` — same pattern
- `zkfc.cu` — `zkFC::prove()` returns claims, no change needed if `Claim` struct stays the same
- Python scripts (`llama-commit.py`, `commit_final_layers.py`, `llama-ppgen.py`) — replace Pedersen commitment generation with Merkle tree construction

**What stays unchanged:**
- All sumcheck code (`proof.cu`: `Fr_ip_sc`, `Fr_hp_sc`, `Fr_bin_sc`) — field-agnostic, just needs the new `Fr_t` typedef.
- All entropy proof code (`zkentropy.cu`, `zkargmax.cu`, `zklog.cu`, `zknormalcdf.cu`) — operates on `Fr_t` and `FrTensor`, field-agnostic.
- Table lookup (`tlookup.cu`) — field-agnostic.
- Rescaling (`rescaling.cu`) — field-agnostic.
- Polynomial representation (`polynomial.cu`) — field-agnostic.

**Validation:**
1. Run full end-to-end proof (`run_e2e_local.sh`) with Goldilocks + FRI.
2. Verify entropy bound matches BLS12-381 version (same model, same input, same tokens).
3. Run `verify_entropy.py` (updated for new proof format).
4. Benchmark: compare total proving time against BLS12-381 baseline (684.8s for 1024 tokens).

## Expected Performance

Based on benchmarks:

| Component | BLS12-381 + Pedersen | Goldilocks + FRI | Speedup |
|---|---|---|---|
| Field multiply | 5.6 B/s | 86 B/s | 15.4× |
| Entropy prove (634s) | 634s | ~41s | ~15× |
| lm_head prove (7.9s) | 7.9s | ~0.5s | ~15× |
| Weight commitment (7B params) | ~83 min | ~16s (Merkle) + ~12s (NTT) | ~170× |
| Commitment opening | EC multi-exp | 1-query FRI | cheaper |
| **Total for entropy tail** | **685s** | **~45s** | **~15×** |

The 15× speedup on the entropy prove phase (which is 92.6% of current time) dominates. The commitment and opening changes are secondary for the entropy tail but significant for the full 32-layer proof pipeline where weight commitments are a larger fraction of total work.

## Risk and Open Questions

1. **Overflow in Goldilocks accumulation.** Needs careful analysis per kernel. The sumcheck accumulates products of field elements — each multiplication produces a full field element (already reduced), so accumulation of n reduced products needs at most 64 + log₂(n) bits before the final reduction. For n = 32,000 this is 79 bits, which fits in 128-bit intermediate. Need to verify all kernels use 128-bit intermediates for accumulation.

2. **NTT domain size.** FRI needs the committed polynomial evaluated on a multiplicative subgroup. For a weight matrix of size d_in × d_out (e.g., 4096 × 32000 ≈ 131M), we need an NTT domain of size ≥ ρ × 131M. With ρ = 2 and padding to power-of-2: NTT of size 256M. This fits in Goldilocks (max domain 2³²) but requires significant GPU memory (~2 GB per committed matrix at 8 bytes/element, ~4 GB with blowup). The H100 has 80 GB, so this is fine for a 7B model but may need streaming for larger models.

3. **FRI proof size.** With 1 query per opening: each opening produces log₂(n) Merkle paths, each of depth log₂(ρn). For n = 131M, ρ = 2: path depth = 28, path size = 28 × 32 = 896 bytes. With ~27 openings per layer × 32 layers ≈ 864 openings total: ~775 KB of Merkle paths. This is well within the "proofs up to the size of the input" design goal.

4. **Poseidon vs SHA-256.** SHA-256 is a safe starting point, but Poseidon2 over Goldilocks would be algebraically compatible with the field, potentially enabling more efficient in-circuit verification if needed. Can be swapped later without changing the FRI structure.

5. **Compatibility with existing Python scripts.** The Python verification and data preparation scripts will need updates to work with Goldilocks serialization (8 bytes per element vs 32 bytes). This is straightforward but touches many files.
