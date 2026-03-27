# Entropy Proof Redesign: Eliminating Per-Token Leakage

**Date:** 2026-03-27
**Branch:** goldilocks-fri
**Depends on:** Kernel fusion work (in progress on another agent) may overlap with Steps 3-4.

## Goal

Replace the current 6-scalars-per-position proof structure with a tensor-level batched
proof that reveals only the aggregate entropy bound H. This fixes:
- Per-token information leakage (diff_actual, win_prob, total_win, surprise all revealed)
- Self-reported total_win (soundness gap S2)
- Missing CDF/log tLookup proofs (no cryptographic binding)
- Fiat-Shamir not implemented (challenges are prover-chosen)

## Current Flow (per-position loop, `zkentropy.cu:133-189`)

```
for each position t:
    t_star = argmax(logits[t])                    # GPU→CPU copy, sequential scan
    v_star = logits[t][t_star]
    diffs[i] = v_star - logits[t][i]              # GPU kernel, size V=32000
    argmax_prover.prove(logits, t_star, v_star)   # bit-decomp sumcheck
    cdf_vals = cdf_prover.compute(diffs)          # tLookup compute (NOT proven)
    win_probs = cdf_scale - cdf_vals              # GPU kernel
    total_win = sum(win_probs)                    # reduction
    q_idx = win_prob[actual] * 2^p / total_win    # CPU integer division
    surprise = log_table[q_idx]                   # tLookup compute (NOT proven)
    EMIT(diff_actual, win_prob, total_win, q_idx, surprise)  ← LEAKS
```

## Proposed Flow (batched tensors, no per-position scalars)

### Phase 1: Compute (all T positions at once)

```
logits_all:     T × V matrix  (already computed by lm_head)
t_star_vec:     T vector       argmax per row (GPU reduction)
v_star_vec:     T vector       logits_all[t][t_star[t]]
diffs_all:      T × V matrix   v_star_vec[t] - logits_all[t][i]
cdf_all:        T × V matrix   cdf_table[diffs_all[t][i]]
win_probs_all:  T × V matrix   cdf_scale - cdf_all
total_win_vec:  T vector       sum over V dimension
actual_wp_vec:  T vector       win_probs_all[t][tokens[t]]

# Log-space subtraction (Option A):
log_tw_vec:     T vector       log_table_ranged(total_win_vec)
log_wp_vec:     T vector       log_table(actual_wp_vec)
surprise_vec:   T vector       log_tw_vec - log_wp_vec
H:              scalar         sum(surprise_vec)   ← ONLY THIS IS REVEALED
```

### Phase 2: Prove (batched sumchecks over tensors)

```
1. Batched argmax proof          (T × V tensor, one set of sumchecks)
2. CDF lookup proof              (T×V → T×V via tLookup.prove, one proof)
3. total_win sum proof            (T×V → T via sumcheck)
4. Actual-token extraction proof  (T×V → T via inner product with indicator)
5. Log lookup proofs              (T → T, two tLookup.prove calls)
6. Subtraction check              (linear relation, trivial)
7. Final sum                      (T → scalar, reduction)
```

## Implementation Steps

### Step 0: Fiat-Shamir Transcript (prerequisite for all steps)

**Files:** new `transcript.cuh/cu`, modify `fr-tensor.cu` (`random_vec`)

Currently all challenges come from `std::random_device` (`fr-tensor.cu:22-36`).
For the non-interactive proof file to be sound, challenges must be hash-derived
from the proof transcript so far.

**Implementation:**
- Create a `Transcript` class wrapping an incremental SHA-256 hash state
- `transcript.append(data, len)` — absorb field elements or proof polynomials
- `transcript.challenge()` — squeeze one Fr_t challenge from the hash state
- Thread a `Transcript&` through all prove functions instead of calling `random_vec()`

**Overlap with kernel fusion:** None — this is a protocol-level change, not a kernel change.

**Effort:** Medium. Mechanical refactoring to thread Transcript through the call stack.
SHA-256 is already implemented in `merkle.cu` (GPU side) but the transcript hashing
should be CPU-side (it hashes proof polynomials, not large tensors).

---

### Step 1: GPU-Side Argmax (standalone, no dependencies)

**Files:** `zkargmax.cu` (modify `compute`), `fr-tensor.cu` (add reduction kernel)

Replace the CPU argmax (`zkargmax.cu:34-43`) with a GPU parallel reduction that
returns only the index. Pattern already exists in `Fr_sum_reduction`.

```cpp
// New kernel: parallel argmax reduction
__global__ void fr_argmax_reduce(const Fr_t* data, uint N, uint* result_idx, Fr_t* result_val);

uint zkArgmax::compute(const FrTensor& logits) {
    uint result_idx;
    Fr_t result_val;
    // ... launch reduction kernel, copy back only idx + val (16 bytes vs 256 KB)
    return result_idx;
}
```

**Overlap with kernel fusion:** Low. This is a new kernel, not a modification of existing
kernels. If the fusion work adds a fused argmax+diff kernel, this step provides the
standalone argmax reduction that it would build on.

**Effort:** Low. Straightforward GPU reduction.

---

### Step 2: Reshape Per-Position Loop into Tensor Operations

**Files:** `zkentropy.cu` (major rewrite of `compute` and `prove`)

Replace the `for (pos = 0; pos < T; pos++)` loop with batch tensor operations.

**2a: Batched argmax** — compute argmax for all T rows simultaneously.

The logits are already a flat T×V tensor (`logits_batch_` in `zkllm_entropy.cu`).
Add a batched argmax that processes T rows:

```cpp
// Returns t_star_vec (T,) and v_star_vec (T,)
pair<vector<uint>, FrTensor> batchedArgmax(const FrTensor& logits_all, uint T, uint V);
```

**2b: Batched diffs** — single kernel computing diffs_all[t][i] = v_star_vec[t] - logits_all[t][i]:

```cpp
__global__ void batched_diffs_kernel(const Fr_t* logits, const Fr_t* v_star_vec,
                                      Fr_t* diffs, uint T, uint V);
```

**2c: Batched CDF and win_probs** — the existing `tLookupRangeMapping::operator()(diffs_all)`
already handles arbitrary-size tensors. Call it once with the T×V tensor instead of T times
with V-sized tensors.

**2d: Row sums for total_win** — reduce each row of win_probs_all to get total_win_vec (T,):

```cpp
__global__ void batched_row_sum_kernel(const Fr_t* data, Fr_t* sums, uint T, uint V);
```

**2e: Actual-token extraction** — extract win_probs_all[t][tokens[t]] into actual_wp_vec (T,):

```cpp
__global__ void extract_by_index_kernel(const Fr_t* data, const uint* indices,
                                         Fr_t* out, uint T, uint V);
```

**Overlap with kernel fusion:** MEDIUM. Steps 2b-2c-2d could be targets for kernel fusion
(e.g., fusing diffs + CDF lookup + win_prob subtraction into a single pass). Coordinate:
this step should define the tensor shapes and interfaces; the fusion work can optimize
the kernel implementations.

**Effort:** Medium. New kernels for 2a, 2b, 2d, 2e; existing tLookup handles 2c.

---

### Step 3: Log-Space Subtraction with Range Reduction

**Files:** new `zklog_ranged.cu/cuh`, modify `zkentropy.cu`

Implement the range-reduced log lookup for total_win_vec values in [1, V×cdf_scale ≈ 2^31].

**3a: Bit-decomposition of total_win_vec**

Reuse the existing `zkargmax_bit_extract_kernel` on the T-length total_win_vec. This
produces bit planes bits_b (T,) for b = 0..30. The bit-decomposition infrastructure
from zkArgmax works directly — just on a T-length vector instead of V-length.

**3b: Highest-bit extraction**

From the bit decomposition, find the exponent e[t] = position of highest set bit.
This can be computed as a weighted sum: e = sum_b(b × bits_b) where bits_b is 1 only
for the highest bit. But finding the *highest* bit specifically requires either:
- A dedicated kernel that scans bits top-down (simple, O(1) per element with 31 comparisons)
- A parallel prefix approach

Simpler: compute e[t] = floor(log2(total_win[t])) directly on GPU via `__clzll()` (count
leading zeros). This is a single-element operation, no sumcheck needed — e is a small
public integer per position (or if we want ZK for e, include it in the bit-decomp proof).

Actually, for zero-knowledge we should NOT reveal e[t] per position (it leaks ~5 bits of
info about total_win). Instead:

- The prover commits to the full bit-decomposition of total_win_vec (32 bit-plane tensors,
  each of length T). This is proven sound via the existing batched binary check.
- The mantissa is extracted as a linear combination of bit planes (provable via sumcheck).
- The exponent is a linear combination of bit planes (provable via sumcheck).
- Neither e[t] nor mantissa[t] is revealed — only the final log value contributes to H.

**3c: Mantissa extraction and log lookup**

```
mantissa_idx[t] = (total_win[t] >> (e[t] - k)) & ((1 << k) - 1)
```

For a fixed k=16, this is a shift-and-mask on the bit decomposition. Given the bit
planes bits_b (b=0..30), the mantissa is:

```
mantissa_idx[t] = sum_{j=0}^{k-1} bits_{e[t]-k+j}[t] * 2^j
```

This is a polynomial in the bit planes, provable via sumcheck. Then:

```
log2(total_win[t]) ≈ (e[t] - k) * log_scale + log_table_k[mantissa_idx[t]]
```

The log_table_k lookup uses a 2^k = 2^16 table (same size as existing log table).
Prove via tLookup.prove().

**3d: Subtraction and sum**

```
surprise_vec[t] = log_tw_vec[t] - log_wp_vec[t]
H = sum(surprise_vec)
```

Both are linear operations. The subtraction is verified by checking:
```
surprise_vec(u) = log_tw_vec(u) - log_wp_vec(u)
```
at random challenge u (Schwartz-Zippel). The sum is a standard reduction.

**Overlap with kernel fusion:** LOW for 3a-3b (bit operations). MEDIUM for 3c (the log
lookup could be fused with other lookups if the fusion work generalizes tLookup).

**Effort:** High. This is the core new algorithm. But each sub-step reuses existing
primitives (bit-extract, tLookup, sumcheck).

**Alternative (simpler, slightly less tight):** Skip range reduction entirely. Use Option C
from the analysis — bound surprise using only win_prob with the lower bound
total_win ≥ cdf_scale/2. This eliminates Step 3 entirely at the cost of ~1 bit/token
looseness. If acceptable, skip to Step 4.

---

### Step 4: Batched Proofs

**Files:** `zkentropy.cu` (prove method), `zkargmax.cu`, `tlookup.cu`

Wire up the batched tensor operations from Step 2 with proper sumcheck proofs.

**4a: Batched argmax proof**

The current `zkArgmax::prove()` operates on a single V-length logit vector. Extend it
to operate on a T×V matrix, proving all T argmax claims simultaneously:

```cpp
Fr_t zkArgmax::proveBatch(const FrTensor& logits_all, uint T, uint V,
                           const vector<uint>& t_star_vec,
                           const FrTensor& v_star_vec,
                           const vector<Fr_t>& u_T,    // challenge over T dimension
                           const vector<Fr_t>& u_V,    // challenge over V dimension
                           vector<Polynomial>& proof);
```

The bit-decomposition becomes T×V tensors (bit planes of size T×V). The batched binary
check operates on the full T×V tensor. The reconstruction check is at a random point
(u_T, u_V) in the T×V space.

This replaces T separate argmax proofs with one batched proof.

**4b: CDF tLookup proof (T×V → T×V)**

Call `tLookupRangeMapping::prove()` once on the T×V diffs_all tensor. This is the
existing interface — tLookup already handles arbitrary-size tensors. The only constraint
is D % N == 0 where D = T×V (padded) and N = 2^cdf_precision.

For T=1024, V=32000: D pads to 32768×1024 = 33,554,432. N = 2^15 = 32768.
D/N = 1024. This satisfies the divisibility constraint.

This single tLookup.prove() cryptographically binds all CDF values to the lookup table,
fixing the missing-proof problem and enabling total_win to be proven.

**4c: total_win sum proof**

Prove total_win_vec[t] = sum_i(win_probs_all[t][i]) for all t. This is a batched
inner-product sumcheck: for each row, take inner product with an all-ones vector.

The existing `zkip_stacked` (zkfc.cu:206-219) proves stacked inner products —
this is the same operation with the "weight" matrix being all-ones.

Alternatively, use a custom sumcheck that reduces the T×V matrix to a T vector
via row sums, verifiable in log2(V) rounds.

**4d: Actual-token extraction proof**

Prove actual_wp_vec[t] = win_probs_all[t][tokens[t]]. This is an inner product of
each row with an indicator vector (1 at position tokens[t], 0 elsewhere).

The indicator vectors are different per position, so this is a batched inner-product
sumcheck with position-dependent "weights." Use the existing `zkip_stacked` with
the indicator matrix (T×V, sparse).

**4e: Log lookup proofs**

Two tLookup.prove() calls:
- `log_prover.prove(actual_wp_vec, log_wp_vec, ...)` — T elements, table size 2^16
- `log_prover_ranged.prove(total_win_vec, log_tw_vec, ...)` — T elements, via Step 3

For T=1024, table size 2^16: D/N = 1024/65536 — D < N, doesn't satisfy D % N == 0.
**Fix:** Pad actual_wp_vec to 2^16 elements (pad with dummy value 1, whose log is a
known constant). Then D = N = 2^16, D/N = 1. This works.

Or batch both log lookups into a single 2T-element vector if they use the same table.

**4f: Subtraction and final sum**

The relation surprise_vec = log_tw_vec - log_wp_vec is a linear constraint verifiable
at a random evaluation point. H = sum(surprise_vec) is a standard reduction. Both can
be checked by the verifier given commitments to the three vectors.

**Overlap with kernel fusion:** HIGH for 4b-4c. The batched CDF tLookup proof and the
row-sum sumcheck are the most computationally intensive parts. If the fusion work
optimizes tLookup kernels (fusing the inv_kernel + poly_kernel + reduce_kernel), that
directly benefits 4b. Coordinate on the tLookup interface — ensure fused kernels
maintain the same input/output contract.

**Effort:** High. This is the main integration step.

---

### Step 5: Proof Serialization and Verifier

**Files:** `zkllm_entropy.cu` (serialization), `verify_entropy.py` (rewrite)

**5a: New proof format**

The proof now contains:
- Header: H, seq_len, vocab_size, sigma_eff, cdf_precision, log_precision, scales
- Argmax proof polynomials (batched)
- CDF tLookup proof polynomials (one set for T×V)
- total_win sumcheck polynomials
- Actual-token extraction sumcheck polynomials
- Log lookup proof polynomials (two sets)
- Subtraction verification data
- Weight-binding proofs (zkFC + Rescaling, from S3)

Use a tagged section format so the verifier can parse each component independently.

**5b: Verifier rewrite**

The Python verifier (`verify_entropy.py`) must:
1. Reconstruct the Fiat-Shamir transcript to re-derive all challenges
2. Verify each sumcheck polynomial chain
3. Verify tLookup proofs (check polynomial identities)
4. Check the final claim: H matches the committed value

This is the most effort-intensive step but is independent of the prover changes.

**Overlap with kernel fusion:** None.

**Effort:** High (especially the verifier).

---

## Dependency Graph

```
Step 0 (Fiat-Shamir) ──────────────┐
                                    │
Step 1 (GPU argmax)                 │
    │                               │
    ▼                               ▼
Step 2 (Tensor reshape) ──────► Step 4 (Batched proofs) ──► Step 5 (Serialize + Verify)
                                    ▲
Step 3 (Range-reduced log) ────────┘
```

Steps 0, 1, 2, 3 can proceed in parallel. Step 4 depends on 2 and 3. Step 5 depends on 4.

## Overlap with Kernel Fusion Work

| Step | Overlap | Coordination needed |
|------|---------|-------------------|
| 0 (Fiat-Shamir) | None | — |
| 1 (GPU argmax) | Low | If fusion adds argmax+diff fused kernel, build on Step 1's reduction |
| 2 (Tensor reshape) | Medium | Define tensor shapes/interfaces first; fusion optimizes kernels |
| 3 (Range-reduced log) | Low | New bit-decomp kernels, unlikely to conflict |
| 4 (Batched proofs) | **High** | tLookup kernel fusion directly affects 4b performance. Agree on tLookup::prove() interface before either side modifies it |
| 5 (Serialize/verify) | None | — |

**Recommended coordination:** Start with Steps 0, 1, 3 (no overlap). Let the fusion work
proceed on kernel internals. Then integrate at Step 4 once both sides have stabilized.

## What Gets Dropped

The current per-position structure in `zkentropy.cu:133-189` (the for loop emitting 6
constant polynomials) is entirely replaced. The following functions become obsolete:
- `zkConditionalEntropy::computePosition()` — replaced by batched compute
- The `fr_to_ull()` / `fr_is_large()` helpers — the integer division is eliminated
- The q_idx clamping logic (lines 83-85) — no more integer division

The following are preserved but modified:
- `zkArgmax::prove()` — extended with a batch variant
- `zkNormalCDF` / `zkLog` — compute() still used; prove() now actually called
- `tLookupRangeMapping::prove()` — called on larger tensors, interface unchanged

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| tLookup D%N constraint fails for some tensor size | Pad tensors; verify D/N ratios for all cases up front |
| Range-reduced log introduces rounding that violates upper bound | Always round log(total_win) upward; prove the rounding direction |
| Batched argmax changes proof structure, breaking compatibility | This is a new proof version — old proofs won't verify with new verifier anyway |
| Fiat-Shamir transcript ordering matters for soundness | Follow standard practice: append each proof polynomial before deriving next challenge |
| Memory: T×V = 33M elements × 8 bytes = 256 MB per tensor, multiple tensors needed | H100 has 80 GB; budget ~10 T×V tensors = 2.5 GB. Fine. |
