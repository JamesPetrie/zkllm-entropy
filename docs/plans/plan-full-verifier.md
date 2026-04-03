# Plan: Full Cryptographic Verifier for zkllm-entropy

## Goal

Replace the current arithmetic-only `verify_entropy.py` with a complete cryptographic verifier that checks all proof components end-to-end, closing soundness gaps S1, S2, S3, and establishing a proper interactive challenge protocol.

## Current State

`verify_entropy.py` reads 8 constant polynomials per token position and checks arithmetic consistency (CDF table lookups, log lookups, quantization, entropy sum). It does **not** verify:
- Argmax bit-decomposition sumcheck proofs (S1)
- That `total_win` equals the true sum of win probabilities (S2)
- That logits derive from committed weights via zkFC/Rescaling (S3)
- Any challenge binding (prover currently generates its own challenges)

## Architecture Decision: Interactive Protocol

Per `design-goals.md`, the system favors **interactive proofs** over Fiat-Shamir:
- Output is committed before proof is requested (failed proof = detection event)
- 64-bit Goldilocks field gives 2^(-64) soundness error per attempt
- Fiat-Shamir would enable offline grinding (prover tries many outputs)

The verifier will be a **two-round interactive protocol**:
1. Prover sends commitments (Merkle roots for all tensors)
2. Verifier sends all challenges
3. Prover sends proof (evaluations, sumcheck polynomials, Merkle opening proofs)
4. Verifier checks everything

This matches the existing code structure where all `prove()` functions accept challenges as parameters.

## Implementation Language

**C++ verifier** (not Python), because:
- Needs Goldilocks field arithmetic (already implemented in `goldilocks.cuh`)
- Needs SHA-256 for Merkle verification (already implemented in `merkle.cu`)
- Needs polynomial evaluation (already implemented in `polynomial.cu`)
- Can reuse existing host-side code directly
- Python would require reimplementing all field arithmetic or FFI bindings

The verifier runs on **CPU only** (no GPU required, no CUDA dependency). Verification is inherently lightweight and sequential — the whole point of a succinct proof system is that the verifier does far less work than the prover:

- **Sumcheck verification**: O(log n) rounds, ~15 field ops per round. Microseconds on CPU.
- **Merkle verification**: ~20 SHA-256 pairs per opening. Microseconds.
- **Total field ops**: O(thousands), not O(millions). No parallelism to exploit.
- **Table reconstruction**: CDF/log tables (~32K entries) are trivially fast on CPU.

A GPU would add launch overhead exceeding the actual computation. A simple single-threaded C++ binary with no CUDA dependency is also easier to audit and deploy independently — important for a trust-critical component.

---

## Phase 0: Proof Format Redesign

### Current format

```
[v2 header: magic, entropy_val, T, vocab_size, sigma_eff, log_scale,
            cdf_precision, log_precision, cdf_scale]
[n_polys: uint32]
[polynomial_0: n_coeffs + coeffs]
...
[polynomial_{n_polys-1}]
```

Flat list of polynomials with implicit position-based indexing (8 per position). No structure, no room for additional proof components.

### New format: tagged sections

```
[v3 header]
  magic: 0x5A4B454E54524F50 (8 bytes)
  version: uint32 = 3
  entropy_val: uint64
  T: uint32
  vocab_size: uint32
  sigma_eff: double
  log_scale: uint32
  cdf_precision: uint32
  log_precision: uint32
  cdf_scale: uint32
  bit_width: uint32            (NEW: for argmax proof)
  hidden_size: uint32          (NEW: for weight-binding)

[section: COMMITMENTS]
  tag: "COMM" (4 bytes)
  n_commitments: uint32
  For each commitment:
    name_len: uint32
    name: char[name_len]       (e.g., "logits_pos_0", "lm_weight", "norm_weight")
    root: Hash256 (32 bytes)   (Merkle root)

[section: CHALLENGES]
  tag: "CHAL" (4 bytes)
  (empty in prover's initial message; filled by verifier in round 2)

[section: ENTROPY_PROOF]
  tag: "ENTR" (4 bytes)
  For each position (T times):
    [constants: 8 Fr_t values]
      ind_sum, ind_dot, logit_act, diff_actual,
      win_prob, total_win, q_fr, surprise

    [argmax_proof]
      n_sumcheck_rounds: uint32
      For each round:
        3 Fr_t values (univariate polynomial at 0, 1, challenge)
      combined_error_eval: Fr_t   (batched binary check at u)
      reconstruction_eval: Fr_t   (sum of 2^b * bits_b(u))
      diffs_eval: Fr_t            (diffs(u))

    [total_win_sumcheck]         (NEW: proves sum(win_probs) = total_win)
      n_rounds: uint32
      For each round:
        3 Fr_t values
      final_a: Fr_t, final_b: Fr_t

[section: WEIGHT_BINDING]
  tag: "WBND" (4 bytes)

  [lm_head_proof]
    [fc_sumcheck: zkip polynomials]
      n_rounds: uint32
      For each round: Polynomial (n_coeffs + coeffs)
    claim_W: Fr_t               (weight MLE evaluation)
    u_batch, u_input, u_output: vectors of Fr_t

    [rescaling_proof: tLookup polynomials]
      n_phase1_rounds: uint32
      For each: Polynomial
      n_phase2_rounds: uint32
      For each: Polynomial

    [merkle_openings]
      n_openings: uint32
      For each:
        leaf_index: uint32
        value: Fr_t
        path: n_levels * Hash256

  [norm_proof]
    (same structure as lm_head_proof)

  [hadamard_proof]
    n_rounds: uint32
    For each round: Polynomial
```

### Changes to prover (zkllm_entropy.cu)

The main change is collecting all proof data into the structured format instead of a flat polynomial list:

1. **Commitment phase** (new): Before proving, commit all tensors and output Merkle roots
2. **Entropy prove**: Same as now, but also serialize argmax sumcheck polys and total_win sumcheck
3. **Weight-binding prove**: Same computation as now, but serialize the proofs instead of discarding

### Estimated effort: 2-3 days

The proof generation code already exists — this is serialization plumbing. Main work:
- Define binary format structs/write functions (~100 lines)
- Modify `zkConditionalEntropy::prove()` to append argmax sumcheck data (~30 lines)
- Add `Fr_ip_sc(win_probs_all, ones_vector, u_tw)` call and serialization (~20 lines)
- Modify `zkllm_entropy.cu` to serialize weight-binding proofs (~50 lines)

---

## Phase 1: Verifier Core — Field Arithmetic and Utilities

### Files to create

**`verifier.cpp`** — main verifier entry point
**`verifier_utils.h`** — proof parsing, field arithmetic (host-side)

### What to implement

The verifier needs host-side Goldilocks arithmetic. Most already exists:

| Function | Existing? | Location | Notes |
|----------|-----------|----------|-------|
| `gold_add`, `gold_sub`, `gold_mul` | Yes | `goldilocks.cuh` (DEVICE functions) | Need `__host__` copies. Already have `host_mul` etc. in `ntt.cu`. |
| `gold_inverse`, `gold_pow` | Yes | `goldilocks.cuh` | Need host versions |
| SHA-256 | Yes | `merkle.cu` (`host_sha256_*`) | Already host-side |
| Merkle verify | Yes | `merkle.cu` (`MerkleTree::verify_host`) | Already host-side |
| Polynomial eval | Yes | `polynomial.cu` | Host-side operators exist |
| MLE evaluation | Partial | `fr-tensor.cu` | Need host-side `operator()(vector<Fr_t> u)` |

**New code needed:**
- Proof file parser (~150 lines)
- Host-side MLE evaluation (~30 lines)
- Sumcheck verifier (~80 lines)
- tLookup verifier (~100 lines)

### Estimated effort: 3-4 days

Most arithmetic already exists. Main work is the proof parser and wiring.

---

## Phase 2: Sumcheck Verifier

The core verification primitive. All higher-level checks reduce to sumcheck verification.

### Algorithm

Given a claimed sum `C = sum_{x in {0,1}^n} f(x)` and a sequence of univariate polynomials `p_1, ..., p_n`:

```
verify_sumcheck(claim, rounds, challenges):
  current_claim = claim
  for i in 1..n:
    p_i = rounds[i]  // degree-2 univariate
    // Check: p_i(0) + p_i(1) == current_claim
    if p_i.eval(0) + p_i.eval(1) != current_claim:
      return FAIL
    // Reduce: evaluate at challenge
    current_claim = p_i.eval(challenges[i])
  // After all rounds: current_claim should equal f(challenges)
  return current_claim  // caller checks against oracle query
```

### Variants needed

1. **Inner product sumcheck** (`Fr_ip_sc` verifier): Verify `<a, b> = C`
   - After reduction: check final claim equals `a(u) * b(u)` where `a(u), b(u)` are provided as final proof elements
   - Verify `a(u)` and `b(u)` against commitments (Merkle opening or FRI)

2. **Hadamard sumcheck** (`Fr_hp_sc` verifier): Verify `sum(a_i * b_i * eq(i, v)) = C`
   - Two challenge vectors (u, v)
   - After reduction: check `a(u) * b(u) * eq(u, v)` against commitments

3. **Multi-Hadamard sumcheck** (`multi_hadamard_sumchecks` verifier): Verify product of multiple tensors
   - Recursive reduction with polynomial proof elements
   - Each round: verify degree bound and sum constraint

### Implementation

```cpp
// Returns the final reduced claim after sumcheck verification
// Caller must check this against committed evaluations
Fr_t verify_sumcheck(
    Fr_t claim,
    const vector<array<Fr_t, 3>>& round_polys,  // p_i(0), p_i(1), p_i(challenge_i)
    const vector<Fr_t>& challenges
) {
    Fr_t current = claim;
    for (size_t i = 0; i < round_polys.size(); i++) {
        Fr_t p0 = round_polys[i][0];
        Fr_t p1 = round_polys[i][1];
        if (gold_add(p0, p1) != current) return ERROR;
        current = round_polys[i][2];  // p_i(challenge_i)
    }
    return current;
}
```

### Estimated effort: 2-3 days

The sumcheck verifier is simple — it's just checking polynomial evaluations. The complexity is in parsing the proof format and routing to the right verifier variant.

---

## Phase 3: Entropy Layer Verification (S1 + S2)

### 3a. Argmax Verification (S1)

The argmax proof consists of:
1. **Indicator constraints**: `ind_sum = 1`, `ind_dot = 0` (already checked)
2. **Bit-decomposition reconstruction**: `sum(2^b * bits_b(u)) = diffs(u)` at challenge u
3. **Batched binary check**: `combined_error(u) = 0` where `combined_error = sum_k r_k * (bits_k^2 - bits_k) + r_{bw} * (ind^2 - ind)`

**Verifier checks:**
```
1. Parse argmax proof: reconstruction_eval, combined_error_eval, diffs_eval
2. Verify ind_sum == 1, ind_dot == 0 (already done)
3. Verify combined_error_eval == 0 (batched binary check passed)
4. Verify reconstruction_eval == diffs_eval (bits reconstruct to diffs)
5. Verify diffs_eval == v_star - logits(u) (diffs consistent with claimed argmax)
```

**What the prover must serialize (new):**
- `combined_error(u)`: 1 Fr_t (should be 0)
- `reconstruction_eval`: 1 Fr_t (sum of 2^b * bits_b(u))
- `diffs(u)`: 1 Fr_t
- `logits(u)`: 1 Fr_t (MLE of logits at challenge u)
- The random vector `r` used for batching (bit_width + 1 Fr_t values)
- Optionally: per-bit evaluations `bits_b(u)` for independent verification

**Prover changes:** In `zkargmax.cu::prove()`, append the above values to the proof vector after the existing `ind_sum` and `ind_dot` constants. ~20 lines.

**Verifier checks:**
```cpp
// Check 1: combined binary error is zero
if (combined_error_eval != Fr_t{0}) return FAIL;

// Check 2: bits reconstruct to diffs
if (reconstruction_eval != diffs_eval) return FAIL;

// Check 3: diffs consistent with argmax
// diffs(u) should equal v_star - logits(u)
Fr_t expected_diffs = gold_sub(v_star, logits_eval_at_u);
if (diffs_eval != expected_diffs) return FAIL;
```

**Note on binding:** The verifier must also verify that `logits(u)` is consistent with the committed logits tensor. This requires a Merkle opening proof at evaluation point u. This is handled in Phase 5 (commitment verification).

### 3b. total_win Verification (S2)

Prove that `total_win = sum(win_probs_all)` via inner product sumcheck.

**Prover changes:** In `zkentropy.cu::prove()`, after computing `total_win`:
```cpp
// Prove total_win = sum(win_probs_all)
FrTensor ones(vocab_size, FR_ONE);  // all-ones vector
auto u_tw = random_vec(ceilLog2(vocab_size));  // or from verifier
vector<Fr_t> tw_proof;
Fr_ip_sc(win_probs_all, ones, u_tw.begin(), u_tw.end(), tw_proof);
// Append tw_proof to main proof vector
```

This appends `3 * ceilLog2(vocab_size) + 2` Fr_t values (~47 values for vocab_size=32000).

**Verifier checks:**
```cpp
// 1. Run sumcheck verifier on total_win claim
Fr_t reduced_claim = verify_sumcheck(total_win, tw_round_polys, challenges_tw);

// 2. After reduction: claim should equal win_probs(u_tw) * ones(u_tw)
//    ones(u_tw) = product((1 - u_tw_i) + u_tw_i) = 1 for all u_tw
//    So reduced_claim should equal win_probs(u_tw)
// 3. Verify win_probs(u_tw) against commitment (Merkle opening)
```

**Alternatively (simpler, less tight):** Use the conservative bound `total_win = vocab_size * cdf_scale` as a public constant. This requires 0 proof data but loosens the entropy bound by ~1 bit/token. Can be a compile-time option.

### Estimated effort: 1-2 weeks

- Prover serialization changes: 2-3 days
- Verifier argmax check: 2-3 days
- Verifier total_win sumcheck: 2-3 days
- Testing: 2-3 days

---

## Phase 4: tLookup / LogUp Verifier

Both CDF and log lookups use `tLookupRangeMapping::prove()`, which delegates to the LogUp argument. The verifier must check this.

### LogUp Verification Algorithm

The LogUp argument proves: every element of witness S appears in table T, with multiplicity vector m.

**Prover's claim:**
```
sum_{i in [D]} 1/(S[i] + beta) = sum_{j in [N]} m[j]/(T[j] + beta)
```

The prover reduces this to a sumcheck via the `alpha` challenge:
```
claim = alpha + alpha^2
```

The sumcheck then reduces through two phases:
- **Phase 1** (while D > N): Fold the witness side
- **Phase 2** (D == N): Fold both sides together

**Verifier must check:**

```cpp
// 1. Reconstruct initial claim
Fr_t alpha_sq = gold_mul(alpha, alpha);
Fr_t initial_claim = gold_add(alpha, alpha_sq);

// 2. Verify phase 1 sumcheck rounds
Fr_t claim = initial_claim;
for (round in phase1_rounds) {
    Fr_t p0 = round.eval_at_0;
    Fr_t p1 = round.eval_at_1;
    if (gold_add(p0, p1) != claim) return FAIL;
    claim = round.eval_at_challenge;
}

// 3. Verify phase 2 sumcheck rounds
for (round in phase2_rounds) {
    // Same pattern
}

// 4. After final reduction: verify against committed evaluations
// A(u) = 1/(S(u) + beta), B(v) = 1/(T(v) + beta)
// Check: reduced_claim == alpha * A(u) * eq(u, v) - alpha_sq * B(v) * m(v) / (N/D)
```

### What the prover must serialize

For each tLookupRangeMapping proof:
- Phase 1 polynomials: `ceilLog2(D/N)` Polynomial objects
- Phase 2 polynomials: `ceilLog2(N)` Polynomial objects
- Final evaluations: A(u), B(v), S(u), T(v), m(v) — 5 Fr_t values

Currently `tLookupRangeMapping::prove()` already appends polynomials to a `vector<Polynomial>& proof`. The main work is serializing this to the file format and parsing it in the verifier.

### Where tLookup is used

| Caller | Table | Input | Size |
|--------|-------|-------|------|
| `zkNormalCDF::prove()` | CDF table (2^cdf_precision entries) | diffs (vocab_size per position) | D=vocab_size, N=2^15 |
| `zkLog::prove()` | Log table (2^log_precision entries) | q_idx (1 per position) | D=1, N=2^15 |
| `Rescaling::prove()` | Remainder table (2*scaling entries) | remainders (tensor_size) | Variable |

### Estimated effort: 1-2 weeks

- Parse tLookup proof from file: 2-3 days
- Implement LogUp verifier (phase 1 + phase 2 sumcheck): 3-4 days
- Handle range mapping (combined S_in + S_out via r challenge): 1-2 days
- Testing: 2-3 days

---

## Phase 5: Commitment Verification

The verifier must check that claimed evaluations (logits(u), weights(u), etc.) are consistent with committed data. Two commitment schemes are used:

### 5a. FRI-PCS (Goldilocks)

For Goldilocks field, polynomial commitments use FRI-PCS (Merkle-based):

**Commit:** Build Merkle tree over evaluations → root hash
**Open:** Provide leaf value + Merkle path at queried position
**Evaluate:** Multilinear evaluation at challenge u (can be verified by FRI low-degree test or by opening all leaves)

**Verifier checks:**
```cpp
// 1. Verify Merkle proof: leaf at index i has value v
bool ok = merkle_verify(root, leaf_index, value, path);

// 2. For multilinear evaluation at u:
//    Need to open enough positions to reconstruct f(u)
//    OR: accept evaluation as prover's claim, verify via FRI
```

**Key question:** Does the verifier need full FRI verification (folding + queries), or just Merkle opening verification?

- **For interactive proofs:** Merkle openings suffice. The verifier picks random positions after seeing the commitment, and checks evaluations there. With enough queries (even 1 for 64-bit field), soundness is 2^(-64).
- **FRI low-degree test** adds assurance that the committed data is close to a low-degree polynomial. Important for polynomial commitment schemes but may not be needed if we only use Merkle-based multilinear commitments.

**Recommendation:** Use Merkle-based multilinear commitments for simplicity. The verifier:
1. Receives Merkle root (commitment)
2. Sends random evaluation point u (challenge)
3. Receives claimed evaluation f(u) + Merkle opening proofs for the positions needed to reconstruct f(u)
4. Verifies Merkle proofs
5. Reconstructs f(u) from opened values and checks against claimed evaluation

### 5b. Pedersen (BLS12-381, legacy)

For BLS12-381, commitments use Pedersen over G1:
- `me_open()` produces 3 G1 curve points per round
- Verifier needs elliptic curve arithmetic (pairing-based verification)
- **Not needed for Goldilocks version** — skip for now

### Estimated effort: 1 week

- Merkle verification: already implemented (`MerkleTree::verify_host`)
- Multilinear reconstruction from opened leaves: ~50 lines
- Integration with sumcheck final claims: ~30 lines
- Testing: 2-3 days

---

## Phase 6: Weight-Binding Verification (S3)

### What the prover currently does (but doesn't serialize)

```
1. normed_hidden = RMSNorm(hidden_state, norm_weights)
   Proved by: zkFC + Rescaling + Hadamard sumcheck

2. logits = lm_head(normed_hidden)
   Proved by: zkFC + Rescaling

3. For each, verifyWeightClaim checks commitment opening
```

### Verification chain

The verifier must check this chain backwards:

```
                         committed                    committed
                         lm_head_W                    norm_W
                            |                            |
logits_batch_ ← rescale ← lm_fc(normed_) ← normed_ ← rescale ← ... ← norm_fc(rms_inv)
      |              |          |                |           |                |
   entropy      tLookup     zkip             Hadamard    tLookup          zkip
   proof        proof       sumcheck          sumcheck    proof          sumcheck
```

**Step 1: lm_head FC proof**
```
claim: logits_batch(u_batch, u_output) = sum_j X(u_batch, j) * W(j, u_output)
proof: zkip sumcheck polynomials (log(inputSize) rounds)
verify: sumcheck → final claim → check X(u) and W(u) against commitments
```

**Step 2: lm_head Rescaling proof**
```
claim: logits_batch_(u) = floor(logits_batch(u) / scaling_factor)
proof: tLookup on remainders
verify: LogUp verifier on remainder table
```

**Step 3: Hadamard product proof**
```
claim: normed(u, v) = g_inv_rms(u) * hidden(v)
proof: Hadamard sumcheck polynomials
verify: sumcheck → final claims on g_inv_rms and hidden
```

**Step 4: norm FC proof** (same structure as Step 1)

**Step 5: norm Rescaling proofs** (same structure as Step 2, x2)

### What the prover must serialize (new)

For each of the above steps:
- Sumcheck polynomials (already computed, just need to be written)
- Final evaluation claims (already computed)
- Challenge vectors used (u_batch, u_input, u_output, etc.)
- Merkle opening proofs for committed tensors at the evaluation points

### Estimated effort: 2-3 weeks

This is the largest single component:
- Prover serialization: 3-4 days (code already exists, just needs I/O)
- Verifier: zkFC check (3-4 days), Rescaling/tLookup check (reuses Phase 4), Hadamard check (2-3 days)
- Commitment opening verification (reuses Phase 5)
- Integration testing: 3-4 days

---

## Phase 7: Interactive Protocol

### Protocol Design

```
ROUND 1: Prover → Verifier
  [v3 header]
  [COMMITMENTS section]
    - Merkle root for logits tensor (per position, or batched)
    - Merkle root for lm_head weights
    - Merkle root for norm weights
    - Merkle root for hidden state
  [CONSTANTS section]
    - Per-position: ind_sum, ind_dot, logit_act, diff_actual,
                    win_prob, total_win, q_fr, surprise
    - entropy_val (claimed aggregate)

ROUND 2: Verifier → Prover
  [CHALLENGES section]
    - u_arg: ceilLog2(vocab_size) Fr_t values (per position, or batched)
    - r_batch: (bit_width + 1) Fr_t values (for argmax binary check)
    - u_tw: ceilLog2(vocab_size) Fr_t values (for total_win sumcheck)
    - u_batch, u_input, u_output: for FC sumcheck
    - alpha, beta, u_lookup, v_lookup: for each tLookup
    - query_positions: for Merkle openings

ROUND 3: Prover → Verifier
  [ENTROPY_PROOF section]
    - Per-position argmax proof data
    - total_win sumcheck polynomials
  [WEIGHT_BINDING section]
    - FC sumcheck polynomials
    - Rescaling tLookup polynomials
    - Hadamard sumcheck polynomials
  [OPENINGS section]
    - Merkle opening proofs at queried positions
```

### Implementation

Two executables:
- **`zkllm_entropy_prover`**: Runs rounds 1 and 3
- **`zkllm_entropy_verifier`**: Runs rounds 2 and 4 (verification)

Communication via files (simplest) or TCP sockets (more practical):

```
# File-based (for testing):
./zkllm_entropy_prover --round1 --out commitment.bin
./zkllm_entropy_verifier --challenges commitment.bin --out challenges.bin
./zkllm_entropy_prover --round3 --challenges challenges.bin --out proof.bin
./zkllm_entropy_verifier --verify proof.bin

# Socket-based (for production):
./zkllm_entropy_prover --connect verifier:8080
./zkllm_entropy_verifier --listen 8080
```

### Challenge generation (verifier side)

The verifier must generate cryptographically secure random challenges:

```cpp
// Use /dev/urandom or OS CSPRNG
vector<Fr_t> generate_challenges(uint n) {
    vector<Fr_t> out(n);
    // Read from /dev/urandom
    int fd = open("/dev/urandom", O_RDONLY);
    for (uint i = 0; i < n; i++) {
        uint64_t raw;
        read(fd, &raw, 8);
        out[i] = {raw % GOLDILOCKS_P};
    }
    close(fd);
    return out;
}
```

### Estimated effort: 1 week

- Protocol framing (file I/O or socket): 2-3 days
- Prover refactoring (split into round 1 + round 3): 2-3 days
- Verifier challenge generation: 1 day
- Integration: 1-2 days

---

## Phase 8: End-to-End Testing

### Test strategy

1. **Unit tests for verifier components:**
   - Sumcheck verifier: test with known-good proofs from prover
   - tLookup verifier: test with known-good lookup proofs
   - Merkle verifier: already tested (`test_merkle.cu`)
   - Field arithmetic: already tested (`test_goldilocks.cu`)

2. **Integration test: honest prover**
   - Run full prover → verifier pipeline
   - Verify that honest proofs pass

3. **Negative tests: cheating prover**
   - **S1 test:** Modify prover to claim wrong v_star. Verify rejection.
   - **S2 test:** Modify prover to deflate total_win. Verify rejection.
   - **S3 test:** Modify prover to use fabricated logits. Verify rejection.
   - **Challenge test:** Modify prover to reuse challenges. Verify rejection.
   - **Merkle test:** Modify prover to alter a leaf value. Verify rejection.

4. **Regression test:**
   - Verify existing 92 unit tests still pass
   - Verify end-to-end entropy proof still produces correct H value

### Estimated effort: 1-2 weeks

---

## Summary: Complete Roadmap

| Phase | What | Effort | Depends On | Closes |
|-------|------|--------|------------|--------|
| 0 | Proof format redesign | 2-3 days | — | — |
| 1 | Verifier core (field arith, parsing) | 3-4 days | Phase 0 | — |
| 2 | Sumcheck verifier | 2-3 days | Phase 1 | — |
| 3 | Entropy layer (argmax + total_win) | 1-2 weeks | Phases 1-2 | S1, S2 |
| 4 | tLookup / LogUp verifier | 1-2 weeks | Phases 1-2 | — |
| 5 | Commitment verification | 1 week | Phases 1, 4 | — |
| 6 | Weight-binding verification | 2-3 weeks | Phases 2-5 | S3 |
| 7 | Interactive protocol | 1 week | Phase 0 | Challenge binding |
| 8 | End-to-end testing | 1-2 weeks | All | — |

**Total estimated effort: 8-12 weeks**

### Critical path

```
Phase 0 (format) → Phase 1 (core) → Phase 2 (sumcheck) → Phase 3 (S1+S2)
                                                        ↘ Phase 4 (tLookup) → Phase 5 (commitment) → Phase 6 (S3)
Phase 7 (interactive) can proceed in parallel after Phase 0
Phase 8 (testing) runs continuously alongside development
```

### Minimum viable verifier (4-5 weeks)

Phases 0-3 + 7 produce a verifier that:
- Closes S1 (argmax verification)
- Closes S2 (total_win sumcheck)
- Uses interactive challenges (no prover self-challenges)
- Does NOT yet verify weight-binding (S3 still open)

This is the recommended first milestone — it closes the two highest-severity gaps with the least effort.

---

## Files to Create/Modify

### New files
- `verifier.cpp` — main verifier entry point (~500 lines)
- `verifier_utils.h` — proof parsing, host field arithmetic — **CREATED** (`verifier/verifier_utils.h`)
- `sumcheck_verifier.h` — sumcheck verification — **CREATED** (`verifier/sumcheck_verifier.h`, includes both standard and ZK variants)
- `tlookup_verifier.h` — LogUp verification (~150 lines)
- `test_verifier.cu` — verifier unit tests (~300 lines)
- `test_zk_verifier.cpp` — ZK verifier tests — **CREATED** (`verifier/test_zk_verifier.cpp`, 12 tests passing)

### Modified files
- `zkllm_entropy.cu` — split into round 1 (commit) + round 3 (prove) (~100 lines changed)
- `zkentropy.cu` — serialize argmax proof + total_win sumcheck (~50 lines)
- `zkargmax.cu` — serialize combined_error, reconstruction, diffs evaluations (~20 lines)
- `Makefile` — add verifier build target (~10 lines)

### Unchanged files (reused as-is)
- `goldilocks.cuh` — field arithmetic (host-compatible via `__host__ __device__`)
- `merkle.cu/cuh` — Merkle verification (already has host functions)
- `polynomial.cu/cuh` — polynomial evaluation (host-side operators exist)
- All other proof-generation code — unchanged, just serialize what they already compute

---

## Open Design Questions

1. **Batch vs. per-position argmax challenges:** Should the verifier send one challenge vector per position (T * ceilLog2(V) values) or a single batch challenge? Per-position is more standard but larger. Batch with position-dependent derivation is more efficient.

2. **tLookup table verification:** The lookup tables (CDF, log) are deterministic from public parameters. Should the verifier recompute them (current approach in verify_entropy.py) or should the prover commit to them and the verifier check the commitment? Recomputation is simpler and sufficient.

3. **Merkle opening strategy:** For multilinear evaluation f(u), the verifier needs to reconstruct f(u) from opened leaves. Should we:
   - (a) Open all 2^n leaves (defeats commitment purpose), or
   - (b) Use FRI to prove the evaluation without opening all leaves, or
   - (c) Use the sumcheck-to-evaluation reduction (standard approach: sumcheck reduces to a single point query, which is opened via Merkle)?

   Option (c) is standard and already how the prover works.

4. **Proof size budget:** The interactive protocol sends commitments (small) and then proofs (larger). For T=1024, vocab_size=32000:
   - Argmax proof: ~50 Fr_t per position × 1024 = ~400 KB
   - total_win sumcheck: ~47 Fr_t per position × 1024 = ~380 KB
   - Weight-binding: ~200 Fr_t total = ~1.6 KB
   - Merkle openings: ~32 hashes per opening × n_openings
   - **Total estimate: ~1-2 MB** (well within the "proofs up to size of input" budget from design goals)
