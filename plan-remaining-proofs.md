# Plan: Implement Remaining Entropy Proof Placeholders

Two placeholder proofs in `src/entropy/zkentropy.cu` need real implementations:
1. **Row-sum sumcheck** (proving `total_win` is correct)
2. **Quotient-remainder proof** (replacing `range_reduced_log`)

---

## A. Row-sum Sumcheck for `total_win`

**Goal:** Prove that `total_win_vec[t] = sum_{i=0}^{V-1} win_probs_all[t*V + i]` for all t in [0,T).

**Current state:** The placeholder at ~line 197 in `zkentropy.cu` just skips the proof with a `// TODO` comment.

**Approach:** Use `partial_me` to reduce the V dimension, then `inner_product_sumcheck` to bind the T dimension.

### Steps

1. **Generate challenge vectors:**
   - `u_v` of length `log2(V)` — challenges for the vocabulary (inner) dimension
   - `u_t` of length `log2(T)` — challenges for the position (outer) dimension
   - These come from the verifier (use `random_vec()`)

2. **Evaluate `win_probs_all` partial MLE:**
   - `win_probs_all` is a flat tensor of size T*V, viewed as a T x V matrix
   - Call `win_probs_all.partial_me(u_v, V)` — this fixes the V dimension at challenge `u_v`, producing a tensor of size T
   - Result: `wp_partial[t] = MLE(win_probs_all[t*V .. t*V+V-1])(u_v)`

3. **Construct the all-ones vector for the sum:**
   - The row-sum identity is: `total_win[t] = <win_probs_all[t, :], ones_V>`
   - After partial MLE on the V dimension: `total_win(u_t) = <wp_partial, eq(u_t, .)>` ... but actually the partial_me already handles this
   - More precisely: evaluating the 2D MLE at `(u_t, u_v)` should equal `total_win.partial_me(u_t)` when summing over V
   - Use `inner_product_sumcheck(wp_partial, ones_T, u_t)` where `ones_T` is a length-T tensor of ones? No — the claim is about the *sum* over V.

   **Correction — simpler approach:** The relation to prove is:
   ```
   total_win_vec(u_t) = win_probs_all.partial_me(u_v, V)(u_t)
                        when u_v is set to make partial_me compute the sum
   ```
   But `partial_me` evaluates the MLE at a point, not a sum. The sum over V is `partial_me` evaluated at the point where `eq(u_v, i) = 1` for all i, which doesn't exist for a random point.

   **Correct approach using sumcheck directly:**
   - The relation is: for each t, `total_win[t] = sum_{i=0}^{V-1} win_probs_all[t*V + i]`
   - Equivalently: `sum_i (total_win_replicated[t*V+i] / V - win_probs_all[t*V+i]) = 0` but this is awkward.

   **Best approach — Hadamard + inner product:**
   - View `win_probs_all` as T*V tensor, `total_win_vec` as T tensor
   - The verifier wants to check: `total_win_vec[t] = sum_v win_probs_all[t*V + v]` for all t
   - At random challenge `u_t` (length log2(T)): `total_win_vec(u_t) = sum_v (sum_t eq(u_t, t) * win_probs_all[t*V + v])`
   - The RHS is `sum_v partial_wp[v]` where `partial_wp = win_probs_all.partial_me(u_t, V)` (fixing the T dimension, keeping V)

   Wait — `partial_me(u, window_size)` evaluates the *first* `len(u)` dimensions. For a T*V tensor where T is the outer (first) dimension and V is the inner (last) dimension, calling `partial_me(u_t, V)` with `u_t` of length `log2(T)` will fix the T dimension and produce a V-length result.

   Then `sum(partial_wp)` should equal `total_win_vec(u_t)`.

   - Compute `partial_wp = win_probs_all.partial_me(u_t, V)` — this gives a V-length tensor
   - Compute `total_win_claim = total_win_vec(u_t)` — scalar
   - Check that `partial_wp.sum() == total_win_claim`
   - To make this a proper proof, use `inner_product_sumcheck(partial_wp, ones_V, u_v)` which proves `<partial_wp, ones_V> = partial_wp.sum()` and binds to challenge `u_v`
   - The sumcheck returns a claim on `partial_wp(u_v)`, which equals `win_probs_all(u_t || u_v)` — this chains to the upstream proof of `win_probs_all`

### Implementation (in `prove()`)

```cpp
// Row-sum sumcheck: total_win_vec[t] = sum_v win_probs_all[t*V + v]
auto u_t = random_vec(ceilLog2(T));

// Fix T dimension, get V-length tensor
FrTensor wp_partial = win_probs_all.partial_me(u_t, V);

// Claim: sum of wp_partial == total_win_vec(u_t)
Fr_t total_win_claim = total_win_vec(u_t);
Fr_t wp_partial_sum = wp_partial.sum();
// Sanity check
if (total_win_claim != wp_partial_sum)
    throw std::runtime_error("row-sum mismatch");

// Prove the sum via inner product with ones
FrTensor ones_V(V);
// ... fill with 1s ...
auto u_v = random_vec(ceilLog2(V));
Fr_t ip_claim = inner_product_sumcheck(wp_partial, ones_V, u_v, proof);
// ip_claim is now wp_partial(u_v) = win_probs_all(u_t || u_v)
```

**Lines of code:** ~15-20 lines replacing the placeholder.

**Dependencies:** `partial_me`, `inner_product_sumcheck`, `random_vec` — all exist.

---

## B. Quotient-Remainder Proof (replacing `range_reduced_log`)

**Goal:** Prove that `surprise[t] = log_table(floor(wp[t] * 2^p / tw[t]))` without leaking per-position values.

**Current state:** `range_reduced_log()` is a CPU helper that computes the log and returns the value. The proof is a placeholder. This function handles `total_win` values that exceed the tLookup table range.

**Approach:** Integer division proof via quotient-remainder decomposition, reusing zkArgmax's bit decomposition infrastructure.

### Background

- `wp[t]` = win_prob for position t (known from CDF lookup)
- `tw[t]` = total_win for position t (proven correct by row-sum sumcheck above)
- `2^p` = scaling factor (p = log_precision, e.g. 16)
- We want `q[t] = floor(wp[t] * 2^p / tw[t])`, then `surprise[t] = log_table(q[t])`
- The division relation: `q[t] * tw[t] + r[t] = wp[t] * 2^p` where `0 <= r[t] < tw[t]`

### What needs to be proven

For each position t (batched as tensors of length T):

1. **Division relation at random point:** `q(u) * tw(u) + r(u) = wp_scaled(u)` where `wp_scaled = wp * 2^p`
   - Single Schwartz-Zippel check at random `u`

2. **q is in table range:** `q[t] in [0, 2^p)` — bit decomposition of q into p bits
   - Reuse `zkargmax_bit_extract_kernel` for bit extraction
   - Batched binary check (same pattern as zkArgmax lines 172-187)
   - Reconstruction check: `sum_b 2^b * q_bits[b](u) == q(u)`

3. **r >= 0:** `r[t] in [0, 2^B)` for some bound B — bit decomposition of r into B bits
   - B = bit_width of the field elements (same as tw's range)
   - Same kernel and check pattern

4. **r < tw:** Prove `tw[t] - r[t] - 1 >= 0` — bit decomposition of `(tw - r - 1)` into B bits
   - Compute `gap[t] = tw[t] - r[t] - 1` on GPU
   - Bit decompose gap, same pattern

5. **Surprise lookup:** `surprise[t] = log_table(q[t])` via `tLookup.prove(q_tensor, surprise_tensor, ...)`
   - q values are in [0, 2^p), log_table maps this range
   - Padding: T must be a multiple of table size (2^p) — pad if needed

### Implementation Plan

#### Step 1: Add quotient-remainder compute

In `compute()`, replace `range_reduced_log()` calls:

```cpp
// For each position t:
// wp_scaled[t] = win_prob[t] * 2^p  (just shift, or multiply by scalar)
// q[t] = wp_scaled[t] / tw[t]       (integer division)
// r[t] = wp_scaled[t] % tw[t]       (remainder)
// surprise[t] = log_table[q[t]]
```

This is done on CPU after copying wp, tw to host. Or add a GPU kernel for the division.

**Decision:** Do it on CPU (T is small, ~1024 max). Copy wp and tw to host, compute q and r, look up log table, copy surprise back to GPU.

#### Step 2: Add GPU kernels

Two new kernels (or reuse existing ones):

```cpp
// Kernel: wp_scaled[t] = wp[t] * (2^p)  — just scalar multiply on GPU
// Kernel: gap[t] = tw[t] - r[t] - 1     — element-wise subtract on GPU
```

Both are trivial; could use existing FrTensor arithmetic (`wp * Fr_from_int(1 << p)` and `tw - r - ones`).

#### Step 3: Bit decomposition (reuse from zkArgmax)

Extract the bit decomposition pattern from `zkArgmax::prove()` lines 131-187 into a reusable helper:

```cpp
// Option A: Copy-paste the pattern (3 uses: q, r, gap)
// Option B: Extract a helper function

// Helper signature:
// void prove_nonneg(const FrTensor& vals, uint num_bits, const vector<Fr_t>& u,
//                   vector<Polynomial>& proof, vector<FrTensor>& all_bit_planes,
//                   vector<Fr_t>& batch_coeffs);
```

**Decision:** Extract a small helper since we use it 3 times. Place it in `zkentropy.cu` as a static function (not in zkArgmax, to avoid modifying that file).

The helper:
1. Launches `zkargmax_bit_extract_kernel` for each bit (already declared in `zkargmax.cuh`)
2. Checks reconstruction at challenge u
3. Accumulates binary check terms into a running `combined_error` tensor with random coefficients

#### Step 4: Proof assembly in `prove()`

Replace the `range_reduced_log` placeholder (~line 230-250) with:

```cpp
// 1. Compute wp_scaled = win_prob_vec * 2^p
FrTensor wp_scaled = win_prob_vec * Fr_from_int(1ULL << log_precision);

// 2. Compute q, r on CPU
Fr_t* cpu_wp = new Fr_t[T]; // copy wp_scaled to host
Fr_t* cpu_tw = new Fr_t[T]; // copy total_win_vec to host
Fr_t* cpu_q  = new Fr_t[T];
Fr_t* cpu_r  = new Fr_t[T];
for (uint t = 0; t < T; t++) {
    long wp_val = scalar_to_long(cpu_wp[t]);
    long tw_val = scalar_to_long(cpu_tw[t]);
    cpu_q[t] = long_to_scalar(wp_val / tw_val);
    cpu_r[t] = long_to_scalar(wp_val % tw_val);
}
FrTensor q_tensor(T, cpu_q);
FrTensor r_tensor(T, cpu_r);

// 3. Division relation at random u_t:
//    q(u_t) * tw(u_t) + r(u_t) == wp_scaled(u_t)
auto u_t = random_vec(ceilLog2(T));
Fr_t q_u = q_tensor(u_t), tw_u = total_win_vec(u_t);
Fr_t r_u = r_tensor(u_t), wp_u = wp_scaled(u_t);
if (q_u * tw_u + r_u != wp_u)
    throw std::runtime_error("division relation failed");

// 4. Non-negativity proofs via bit decomposition:
//    a) q in [0, 2^p): decompose q into p bits
//    b) r >= 0: decompose r into B bits
//    c) r < tw: decompose (tw - r - 1) into B bits
FrTensor gap = total_win_vec - r_tensor - FrTensor::ones(T);

uint B = bit_width; // from constructor, e.g. 60
prove_nonneg(q_tensor, log_precision, u_t, proof, ...);
prove_nonneg(r_tensor, B, u_t, proof, ...);
prove_nonneg(gap, B, u_t, proof, ...);

// 5. Surprise lookup: surprise[t] = log_table(q[t])
//    Use log_prover (tLookupRangeMapping) on q_tensor
auto [surprise_tensor, _mults] = log_prover(q_tensor);
// Pad T to multiple of table size if needed
Fr_t surprise_claim = log_prover.prove(q_tensor, surprise_tensor, u_log, proof);

// 6. Final entropy: H = sum(surprise) / T
Fr_t entropy_claim = surprise_tensor.sum();
```

#### Step 5: Wire up the proof chain

The proof produces claims that chain together:
- Row-sum sumcheck yields a claim on `win_probs_all(u_t, u_v)` — chains to CDF tLookup
- Division relation binds `q`, `r`, `tw`, `wp` at the same challenge `u_t`
- Bit decomposition proofs bind bit planes at `u_t`
- Surprise tLookup binds `q` and `surprise` at `u_log`

The verifier checks:
1. `total_win_vec(u_t) == row_sum_claim` (from row-sum sumcheck)
2. `q(u_t) * tw(u_t) + r(u_t) == wp(u_t) * 2^p` (division)
3. Bit decomposition reconstruction checks (3 of them)
4. Batched binary check == 0 (one combined check for all bit planes)
5. `surprise(u_log) == log_table_claim` (from tLookup)
6. `sum(surprise) == claimed_entropy * T` (final check)

### Files to modify

| File | Change |
|------|--------|
| `src/entropy/zkentropy.cu` | Replace `range_reduced_log()` and both placeholder blocks. Add `prove_nonneg` helper. |
| `src/entropy/zkentropy.cuh` | Add `log_precision` member to `zkEntropy` class (or pass as parameter). |
| `test_zkentropy.cu` | Add test for quotient-remainder correctness; test that proof doesn't throw. |

### Files NOT modified

| File | Reason |
|------|--------|
| `src/zknn/zkargmax.cu` | Reuse its kernels via `#include`, don't modify |
| `src/proof/proof.cu` | Use existing `inner_product_sumcheck` as-is |
| `src/tensor/fr-tensor.cu` | Use existing `partial_me`, `sum`, arithmetic ops as-is |

### Risks and considerations

1. **FrTensor size constraints:** `q_tensor`, `r_tensor`, `gap` are all size T. Bit planes are also size T. T is small (~1024), so memory is not a concern.

2. **tLookup D%N==0 constraint:** T (number of positions) must be a multiple of the log table size (2^p). If T < 2^p, we need to pad q_tensor. The existing tLookup auto-padding handles non-power-of-2 D, but D%N==0 must hold after padding.
   - If p=16, table has 65536 entries. T=1024 means D=1024, N=65536 → D < N, so D%N != 0.
   - **Fix:** Pad q_tensor to size N (65536) with valid table indices (e.g., 0). This makes D=N, so D%N==0.

3. **Log table range:** The table maps [0, 2^p) → log values. q values are in [0, 2^p) by construction (since `wp <= tw`, so `wp * 2^p / tw <= 2^p`). Edge case: if `wp == tw`, then `q = 2^p` exactly, which is out of range. Handle by clamping q to `2^p - 1` (or noting that `wp <= tw` implies `q <= 2^p`, and if `q == 2^p` then `r == 0` and we can set `q = 2^p - 1, r = tw`... but this changes the semantics). Best: table has 2^p + 1 entries, or clamp.

4. **Kernel availability:** `zkargmax_bit_extract_kernel` is declared with `KERNEL` (i.e., `__global__`). It's defined in `zkargmax.cu`. To call it from `zkentropy.cu`, either:
   - Declare it `extern` in a header, or
   - Duplicate the kernel (it's 5 lines), or
   - Move it to a shared utility header

   **Decision:** Declare `extern` in `zkargmax.cuh` (one line addition).

### Order of implementation

1. Implement row-sum sumcheck first (simpler, fewer dependencies)
2. Add `prove_nonneg` helper
3. Implement quotient-remainder proof
4. Add/update tests
5. Build and verify on H100

### Estimated scope

- ~80 lines of new proof code in `zkentropy.cu`
- ~5 lines in `zkentropy.cuh`
- ~1 line in `zkargmax.cuh` (extern kernel declaration)
- ~30 lines of new tests
- Total: ~120 lines of code
