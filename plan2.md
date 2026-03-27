# plan2.md — Security Review & Improvement Plan

## Threat Model

The entropy proof claims: *"Given committed weights W and public input x, the output
sequence o has conditional entropy at most H."*

A **malicious prover** wants to produce a low H for a sequence that actually has high
entropy (i.e., hide the fact that tokens were steganographically chosen). The proof
must make this impossible.

A **verifier** has: the proof file, the committed weight parameters (sigma_eff, vocab_size,
etc.), and access to the public CDF/log lookup tables. It does NOT have the logits or
hidden states.

---

## Current Architecture

The proof has two layers:

1. **Weight-binding layer** (zkllm_entropy.cu lines 241–252): proves that logits derive
   from committed weights via `zkFC + Rescaling + verifyWeightClaim`. These proofs are
   executed locally by the prover (self-checked) but **not serialised** into the proof file.

2. **Entropy layer** (zkentropy.cu): for each position, records 6 constant polynomials
   (logit_act, diff_actual, win_prob, total_win, q_fr, surprise). The Python verifier
   checks arithmetic consistency of these 6 values against public lookup tables.

The argmax proof (zkargmax.cu) runs bit-decomposition + binary sumcheck, but its proof
polynomials go into the same `vector<Polynomial>` and are serialised alongside the
entropy constants. **However, the Python verifier ignores them** — it only reads the
6-per-position constant polynomials.

---

## Issues Found (Severity Order)

### S1. Verifier does not check argmax or bit-decomposition proofs
**Severity: HIGH — soundness gap**

The prover runs `argmax_prover.prove()` which appends sumcheck polynomials to `proof`,
and `Fr_bin_sc()` which writes into a local `bin_proof` vector that is **discarded**
(zkargmax.cu:143–144). Neither set of polynomials is read by `verify_entropy.py`.

A malicious prover can claim any `v_star` (not the true argmax). If the claimed v_star
is higher than the true max, diffs become larger, CDF values increase, win_probs
decrease, and the entropy bound drops — the prover cheats by inflating the apparent
logit gap.

**Fix:** Either (a) make verify_entropy.py parse and verify the sumcheck polynomials,
or (b) change the proof format so the argmax proof is self-contained and verified.
In either case, the binary sumcheck proofs from `Fr_bin_sc` must be appended to the
main `proof` vector (currently discarded).

### S2. total_win is self-reported, not proven
**Severity: HIGH — soundness gap**

The prover records `total_win = sum(win_probs_all)` as a constant polynomial
(zkentropy.cu:160). The verifier checks `win_prob <= total_win` (verify_entropy.py:183)
but has no way to verify that `total_win` is the true sum over all vocab_size tokens.

A malicious prover can inflate `total_win` to make `q_idx = win_prob * 2^lp / total_win`
smaller, increasing surprise. But more dangerously, a prover can **deflate total_win**
(claim it equals win_prob, making q_idx = 2^lp, surprise = 0) to hide entropy.

The only constraint is `win_prob <= total_win`, which a cheating prover trivially satisfies.

**Fix:** The prover must commit to the full win_probs_all tensor and prove that total_win
equals its sum. Options:
- (a) Record a sum-check proof that `win_probs_all.sum() = total_win` using the existing
  `Fr_ip_sc` infrastructure (inner product with a ones-vector).
- (b) Prove the CDF lookups over the full vocab via `tLookupRangeMapping::prove()`,
  which implicitly commits to all values. This is the "right" approach but hits the
  D%N divisibility constraint (see S4).
- (c) As a simpler interim fix: use the conservative upper bound
  `total_win = vocab_size * cdf_scale` (a public constant). This is always valid and
  requires no proof. It makes the entropy bound looser but is unconditionally sound.

### S3. Proof file does not include weight-binding proofs
**Severity: MEDIUM — completeness gap**

`rs_lm.prove()`, `verifyWeightClaim()`, `rs_norm2.prove()`, etc. (zkllm_entropy.cu:243–252)
run locally but their outputs are not in the proof file. The proof file only shows that
*some* set of logits yields entropy H, but doesn't prove those logits came from committed
weights.

An adversary who controls the prover binary can supply fabricated logit tensors that
produce low entropy.

**Fix:** Serialise the weight-binding proofs (FC sumcheck polynomials, rescaling lookup
proofs, commitment opening proofs) into the proof file, and extend the verifier to check
them. This is a significant engineering effort since it requires the verifier to handle
elliptic-curve commitment openings.

### S4. verify_entropy.py cdf_precision default (12) disagrees with prover default (15)
**Severity: MEDIUM — verification will reject honest proofs with wrong flags**

The prover now defaults to `cdf_precision=15` (zkllm_entropy.cu:89), but
`verify_entropy.py` defaults to `--cdf-precision 12` (line 63). The job 2142 verification
passed because `diff_actual` values happened to be < 4096 for all 1024 greedy tokens
(greedy token has diff=0, so this works). But for non-greedy tokens, `diff_actual` could
exceed 4095, and the verifier would use `d_clamped = min(diff_actual, 4095)` while the
prover used `min(diff_actual, 32767)`, giving different CDF values.

**Fix:** Store `cdf_precision` in the proof file header (alongside `log_precision`) and
have the verifier read it from there rather than using a command-line default. This
prevents any parameter mismatch.

### S5. Binary sumcheck proofs are computed but discarded
**Severity: MEDIUM — wasted work, incomplete proof**

In zkargmax.cu:141–145, `Fr_bin_sc()` writes into a local `bin_proof` that goes out of
scope. The binary sumcheck proves each bit tensor contains only {0,1} values, which is
essential for the bit-decomposition range proof. Without it, a prover can use non-binary
"bits" that reconstruct to the correct diff at challenge point u but represent a different
(larger) value elsewhere.

**Fix:** Append `bin_proof` contents to the main `proof` vector:
```cpp
Fr_bin_sc(bits_vecs[b], u.begin(), u.end(), v.begin(), v.end(), proof);
//                                                              ^^^^^ not bin_proof
```
This is a one-line change. (The proof vector already accepts `Fr_t` elements from the
sumcheck.)

### S6. Negative diffs produce warnings but not errors
**Severity: LOW — defence in depth**

zkargmax.cu:94–103 prints to stderr when negative diffs are detected but does not throw.
The subsequent reconstruction check (line 137) will likely catch this, but not guaranteed
for all challenge points.

**Fix:** Throw `std::runtime_error` if `n_negative > 0`. An honest prover should never
produce negative diffs.

### S7. Diagnostic code in zkargmax.cu should be removed or gated
**Severity: LOW — performance**

The diagnostic block (zkargmax.cu:70–112) copies all N diffs and logits to CPU for every
position — 32000 * 32 bytes * 2 = ~2 MB per position, 1024 positions = ~2 GB of
unnecessary memcpy. This was added during debugging and should be removed or put behind
a compile-time flag.

**Fix:** Remove the diagnostic block or guard it with `#ifdef ZKARGMAX_DEBUG`.

---

## Simplification Opportunities

### P1. Use conservative total_win to eliminate the hardest soundness gap

Replace the actual `win_probs_all.sum()` with the public upper bound
`vocab_size * cdf_scale`. This eliminates the need to prove total_win (S2) at the cost
of a looser entropy bound.

Impact on tightness: For greedy tokens (diff=0, win_prob≈cdf_scale/2), the conservative
`q_idx = (cdf_scale/2 * 2^lp) / (32000 * cdf_scale) = 2^lp / 64000 ≈ 0.51` gives
surprise ≈ 1 bit. The current approach gives surprise ≈ 0 bits. So the bound becomes
~1 bit/token looser for greedy tokens, but is unconditionally sound.

If tighter bounds are needed, the full tLookupRangeMapping proof path should be used (P3).

### P2. Store all proof parameters in the proof file header

Extend the header to include: `cdf_precision`, `cdf_scale`, `bit_width`. Currently only
`sigma_eff`, `log_scale`, `seq_len`, `vocab_size` are stored. The verifier should read
all parameters from the proof file to avoid mismatches.

### P3. Prove CDF lookups over full vocab (future, requires tLookup work)

The "right" way to handle total_win: run `tLookupRangeMapping::prove()` over the full
`diffs_all` tensor (size vocab_size=32000) and the CDF table (size 2^15=32768). This
cryptographically binds the CDF values to the lookup table.

The current obstacle: `tLookup::prove()` requires `D % N == 0` and `D` must be a power
of 2. For vocab_size=32000, D pads to 32768 = N, making D/N=1 and v1 empty, which works.
But the divisibility check `D != 1 << ceilLog2(D)` would pass (32768 is a power of 2).
This path should be tested.

### P4. Simplify argmax proof by removing diagnostic code

After S5, S6, and S7, the argmax prover becomes:
1. Compute diffs on GPU
2. Bit-decompose
3. Verify reconstruction at u (throw on mismatch)
4. Binary sumcheck for each bit vector (appended to proof)
5. Return MLE claim

This is clean and matches the standard bit-decomposition range proof from the literature.

### P5. Unify the proof format

Currently the proof file interleaves argmax sumcheck polynomials with entropy constant
polynomials in a flat list. The verifier skips the sumcheck polys by assuming exactly
6 per position. If argmax bit_width or vocab_size changes, the polynomial count changes.

Better: use a tagged format with section headers, or separate the argmax proofs from the
entropy claims so each can be verified independently.

---

## P6. Log-Space Division Trick (Enables True ZK)

The current entropy pipeline computes per-token surprise as:
```
q = floor(win_prob * 2^p / total_win)
surprise = -log2(q / 2^p)
```

This requires proving an integer division by a variable denominator (`total_win`), which
is the hardest unsolved cryptographic sub-problem (see S2). The softmax module avoids an
analogous division using a log-space subtraction trick — the same idea applies here.

### Core observation

Since `-log2(win_prob / total_win) = -log2(win_prob) + log2(total_win)`, the division
becomes a subtraction of two log lookups:

```
surprise = log_table_tw[total_win] - log_table_wp[win_prob]
```

No division, no quantization step, no `q_idx`. The two log values are looked up
independently via `tLookupRangeMapping`, and their difference is a linear relation
provable with a single sumcheck.

### Lookup table sizing

- **`win_prob`** is in `[1, cdf_scale]` (typically 2^15). A `zkLog` table of size 2^15
  handles this directly — same as the existing log table.
- **`total_win`** is in `[1, vocab_size × cdf_scale]` (up to ~2^30 for 32K vocab).
  A direct table of size 2^25 (~256 MB on GPU) covers values up to ~33M, which exceeds
  `32768 × 1024 ≈ 33.5M` for sequences up to 1024 tokens. For longer sequences or
  larger vocabularies, a 2^27 table (~1 GB) provides ample headroom. Both are feasible
  on H100 (80 GB).

### Per-position proof structure (replaces current 6-constant approach)

For each position, the prover holds intermediate tensors but never reveals them:

1. **`zkArgmax.prove(logits)`** — proves `t_star` is the argmax and `v_star` is its value.
   Existing module, just needs binary sumcheck fix (S5) and serialisation.

2. **`cdf_prover.prove(diffs_all)`** — proves CDF lookup over full vocabulary via
   `tLookupRangeMapping::prove()`. Existing module. With `D = vocab_size` padded to 32768
   and `N = 2^cdf_precision`, the `D % N == 0` constraint is satisfiable.

3. **Sumcheck: `total_win = sum(win_probs_all)`** — standard inner-product sumcheck with
   an all-ones vector. Uses existing `Fr_ip_sc`.

4. **`log_prover_wp.prove(win_prob_vec)`** — log lookup on per-position win_prob values
   batched across all T positions. Uses existing `tLookupRangeMapping` with table size
   2^15. Requires `T` padded to a multiple of table size.

5. **`log_prover_tw.prove(total_win_vec)`** — log lookup on per-position total_win values
   batched across all T positions. New `zkLog` instance with larger precision (2^25).

6. **Subtraction sumcheck: `surprise_vec = log_tw_vec - log_wp_vec`** — linear relation,
   provable as a single-round sumcheck or direct MLE equality check.

7. **Sum: `H = sum(surprise_vec)`** — only H (or H/T) is revealed.

### What this achieves

- **Eliminates the variable-denominator division** — the hardest unsolved sub-problem.
- **Eliminates all 6 per-position constant polynomials** — no `logit_act`, `diff_actual`,
  `win_prob`, `total_win`, `q_fr`, or `surprise` values are revealed. All intermediate
  values remain committed but unopened.
- **Reveals only the aggregate entropy bound H** — achieving true zero knowledge (in the
  cryptographic sense) for the entropy layer.
- **Reuses existing modules** — `zkArgmax`, `tLookupRangeMapping`, `Fr_ip_sc` all work
  as-is. The only new component is a second `zkLog` instance with a larger table.

### Precision considerations

The current pipeline quantizes `win_prob / total_win` to a `log_precision`-bit integer
before the log lookup, introducing quantization error. The log-space approach applies the
log directly to `win_prob` and `total_win` separately, which changes the error profile:

- Each log lookup has ±0.5 ULP rounding error relative to `log_scale`.
- The subtraction doubles this to ±1 ULP.
- This is comparable to the current quantization error and can be made arbitrarily small
  by increasing `log_scale`.

The bound remains valid (always an upper bound on true entropy) because the CDF
approximation via Gibbs' inequality is the dominant source of looseness, not the log
precision.

### Relationship to other issues

- **Supersedes S2** (total_win unproven): total_win is now proven via sumcheck (step 3)
  and its log is proven via lookup (step 5).
- **Supersedes P1** (conservative total_win): no longer needed — the actual total_win is
  used with full tightness.
- **Complements S1** (argmax verification): the argmax proof is still needed and must be
  serialised and verified.
- **Complements S3** (weight binding): the weight-binding layer is orthogonal and still
  needs serialisation for full security.
- **Resolves the ZK leakage concern**: the current proof format leaks per-token logit
  gaps, probabilities, and surprise values. This approach reveals only H.

### Implementation effort

| Component | Existing? | Effort |
|-----------|-----------|--------|
| Batch CDF lookup (`tLookup.prove`) | Yes | Low |
| `total_win` sum proof (`Fr_ip_sc`) | Yes | Low |
| Log lookup on `win_prob` batch | Yes (`zkLog`) | Low |
| Log lookup on `total_win` batch | New `zkLog` instance, larger table | Low–Medium |
| Subtraction sumcheck | Trivial linear check | Low |
| Final sum proof | Trivial | Low |
| Remove scalar emissions, commit intermediates | Refactoring | Medium |
| Build matching cryptographic verifier | No verifier exists yet | High |

Total: moderate effort for the prover-side changes. The verifier is the dominant cost,
but that work is needed regardless of this change (see S1, S3).

---

## Recommended Implementation Order

1. **S5** — Fix binary sumcheck (one-line change, immediate soundness improvement)
2. **S6** — Throw on negative diffs (one-line change)
3. **S7** — Remove/gate diagnostic code (cleanup)
4. **S4 + P2** — Store cdf_precision in proof header, fix verifier defaults
5. **S2 via P1** — Use conservative total_win as interim fix (eliminates hardest soundness
   gap with minimal code change)
6. **S1** — Extend verifier to check argmax sumcheck polynomials (moderate effort)
7. **P6** — Log-space division trick (replaces P1 with tight bound, eliminates per-token
   leakage, achieves true ZK for the entropy layer)
8. **S3** — Serialise weight-binding proofs (large effort, needed for full security)

---

## P7. Goldilocks Field Range Validation: FP16 Accuracy and Overflow Analysis

The Goldilocks field has modulus p = 2⁶⁴ − 2³² + 1 ≈ 1.8 × 10¹⁹. Field arithmetic wraps
silently at p. If any intermediate value in the proof pipeline exceeds p, the computation
produces a valid field element that bears no relation to the intended integer — a silent
correctness failure, not a crash. This section plans a systematic check.

### Where overflow could occur

The proof pipeline converts FP16/FP32 model outputs to quantized integers, then operates
on them in the Goldilocks field. The critical chain is:

1. **Logit quantization:** `logit_int = round(logit_fp × scaling_factor)`. With FP16
   logits in [−65504, 65504] and typical scaling_factor = 65536, the max value is
   ~4.3 × 10⁹ (~32 bits). Safe.

2. **Logit diffs:** `diff = v_star − logits[i]`. Max = 2 × max_logit ≈ 8.6 × 10⁹
   (~33 bits). Safe.

3. **CDF values:** `cdf_scale` (typically 2¹⁵ = 32768). Safe.

4. **Win probability sum:** `total_win = sum(win_probs)` where each win_prob ≤ cdf_scale.
   Max = vocab_size × cdf_scale = 32000 × 32768 ≈ 10⁹ (~30 bits). Safe.

5. **Rescaling products:** `zkFC` computes `output = input × weight`, summed over
   `in_dim`. If input and weight are both ~2³² and in_dim = 4096, the sum is
   ~2³² × 2³² × 2¹² = 2⁷⁶. **This exceeds p ≈ 2⁶⁴.** The Rescaling step divides
   by scaling_factor after the multiply, but the intermediate accumulation happens in
   the field and would wrap.

6. **RMSNorm inverse:** `rms_inv = round(1/rms × scaling_factor)`. Bounded by
   scaling_factor. Safe.

7. **Hadamard products in RMSNorm:** element-wise multiply of two ~2³² values = ~2⁶⁴.
   Borderline — depends on actual magnitude.

8. **tLookup inverse:** `A[i] = 1/(S[i] + beta)` — field inverse, always valid.

9. **Sumcheck intermediate products:** `a(u) × b(u)` where both are MLE evaluations.
   MLE values are linear combinations of field elements with coefficients in [0,1], so
   they stay within the range of the input elements. Products of two ~2³² values = ~2⁶⁴.
   Borderline.

### Risk assessment

| Location | Max magnitude | Goldilocks headroom | Risk |
|----------|--------------|---------------------|------|
| Logits, diffs | ~2³³ | 2³¹ headroom | None |
| CDF, win_prob | ~2³⁰ | 2³⁴ headroom | None |
| total_win | ~2³⁰ | 2³⁴ headroom | None |
| zkFC accumulation | ~2⁷⁶ | **Overflows** | **HIGH** |
| Hadamard (RMSNorm) | ~2⁶⁴ | Borderline | Medium |
| Sumcheck products | ~2⁶⁴ | Borderline | Medium |

### The zkFC overflow question

The `zkFC` matmul accumulates `in_dim` products of quantized values. In BLS12-381
(255-bit), this never overflows. In Goldilocks (64-bit), it can.

However, this is the **intended** behavior for sumcheck-based proofs: the matmul is
verified via a sumcheck protocol that reduces the inner product to a single evaluation,
never forming the full accumulation in the field. The sumcheck polynomial at each round
is degree 2 (product of two multilinear functions), and the evaluation point is a random
field element, so the intermediate value is a random element of F_p — it does not
correspond to any integer accumulation.

The question is whether the **compute** path (which forms the actual matmul to produce
logits) overflows before the result is rescaled. If the compute path uses GPU field
arithmetic (accumulating in F_p), the accumulation wraps modulo p, giving wrong logits.
If it uses host-side integer arithmetic or floating point, it may be fine.

**Action items:**

1. **Trace the zkFC compute path.** Determine whether `FrTensor::matmul` accumulates in
   the field (wraps at p) or uses wider arithmetic. If it accumulates in F_p, verify that
   the Rescaling step accounts for wrap-around or that values are small enough to avoid it.

2. **Empirical range check.** Run the pipeline on real Llama-2-7B weights and log the
   maximum intermediate value at each stage. Compare against p.
   ```python
   # Pseudocode for instrumented run:
   for each layer output:
       max_val = max(abs(scalar_to_long(elem)) for elem in tensor)
       log2_max = log2(max_val)
       print(f"Stage: {name}, max magnitude: 2^{log2_max:.1f}, headroom: {64 - log2_max:.1f} bits")
   ```

3. **Compare quantized vs FP16 outputs.** Run inference in FP16 and in quantized
   fixed-point (with the same scaling factor). Compare per-token logit rankings and
   argmax agreement. Metrics:
   - Fraction of positions where argmax matches
   - Mean/max absolute difference in logits (after rescaling back to float)
   - Per-position KL divergence between softmax distributions
   - Whether the quantization error is smaller than the calibrated σ (if so, it's
     absorbed by the noise model and doesn't affect the entropy bound)

4. **Determine minimum safe scaling factor.** The scaling factor controls precision
   (larger = more precise quantization) but also overflow risk (larger = bigger
   intermediate values). Find the sweet spot where quantization error < σ but
   accumulations stay within p.

5. **Consider mixed-precision strategy.** If overflow is a problem in zkFC, options:
   - Reduce scaling_factor for weight quantization (trades precision for range)
   - Split the matmul into blocks and rescale between blocks
   - Use extension field (Goldilocks quadratic extension, 128-bit) for accumulations
   - Accept the overflow and handle it modularly (the sumcheck proof is valid
     regardless of overflow — the question is only whether the *proved* computation
     matches the *intended* computation)

---

## What Is Already Sound

- The CDF and log lookup *values* are publicly deterministic: given diff_actual and the
  public parameters, anyone can recompute win_prob and surprise. The verifier does this.
- The entropy sum is verified: claimed entropy = sum of per-position surprises.
- The win_prob ≤ total_win consistency check catches obviously impossible claims.
- The Rescaling proofs for weight-binding (rs_lm, rs_norm) now work correctly after the
  Bug 4 fix (is_neg removal).
- The argmax computation (fr_gt) is correct after the Bug 1 fix.
- The tLookup bounds clamping prevents GPU memory corruption.
