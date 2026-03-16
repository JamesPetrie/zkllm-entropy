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

## Recommended Implementation Order

1. **S5** — Fix binary sumcheck (one-line change, immediate soundness improvement)
2. **S6** — Throw on negative diffs (one-line change)
3. **S7** — Remove/gate diagnostic code (cleanup)
4. **S4 + P2** — Store cdf_precision in proof header, fix verifier defaults
5. **S2 via P1** — Use conservative total_win (eliminates hardest soundness gap with
   minimal code change; re-evaluate tightness to decide if P3 is needed)
6. **S1** — Extend verifier to check argmax sumcheck polynomials (moderate effort)
7. **S3** — Serialise weight-binding proofs (large effort, needed for full security)

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
