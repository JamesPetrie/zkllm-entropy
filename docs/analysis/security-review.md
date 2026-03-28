# Security Review: zkllm-entropy (FRI + Goldilocks Version)

## Correctness, Soundness, and Zero-Knowledge Properties

**Date:** 2026-03-27
**Scope:** Full codebase review of the FRI + Goldilocks implementation

---

## Executive Summary

The zkllm-entropy project implements zero-knowledge proofs of conditional entropy for LLM inference verification. The goal is to prove that a model's output has bounded covert channel capacity (at most H bits of hidden information) without revealing the model weights.

**Bottom line:** The low-level cryptographic primitives (Goldilocks field arithmetic, NTT, Merkle trees, FRI folding) are **correctly implemented**. However, the system has **critical soundness gaps** at the integration layer — particularly between the prover and verifier — that would allow a malicious prover to forge false (low) entropy bounds. The system does **not achieve zero-knowledge** in its current form.

| Property | Status | Summary |
|----------|--------|---------|
| **Correctness** | Mostly achieved | Field arithmetic, NTT, FRI folding, Merkle trees, polynomial operations all correct. Several bugs found and fixed (argmax comparison, CDF table sizing, rescaling). |
| **Soundness** | Not achieved | Two critical gaps (S1, S2) allow a cheating prover to produce false entropy bounds. Weight-binding proofs not serialized (S3). |
| **Zero-Knowledge** | Not achieved | No blinding/masking in FRI. Per-position values (logit gaps, win probabilities, surprises) are revealed in the clear. |

---

## 1. Correctness

### 1.1 Goldilocks Field Arithmetic — CORRECT

The Goldilocks prime p = 2^64 - 2^32 + 1 = `0xFFFFFFFF00000001` is correctly defined (`goldilocks.cuh:31`).

All field operations exploit the identity 2^64 = 2^32 - 1 (mod p):

- **Addition** (`gold_add`): Correctly handles 64-bit overflow via `(sum < a.val)` check, single conditional subtraction of p sufficient since max sum is 2p-2.
- **Subtraction** (`gold_sub`): Correctly handles underflow by adding p when a < b.
- **Multiplication** (`gold_mul`): 5-step reduction algorithm using `UMUL64HI` for 128-bit products. Traced bounds confirm no intermediate overflow: hi*epsilon fits in 96 bits, carry propagation is bounded, final value < 2p before canonical reduction.
- **Inverse** (`gold_inverse`): Fermat's little theorem a^(p-2) via binary exponentiation. Correct.
- **Montgomery form**: Identity function (documented and intentional — Goldilocks doesn't need Montgomery representation since reduction is already cheap).

**Test coverage:** 36 tests in `test_goldilocks.cu` verify all operations including the critical property 2^32 * 2^32 = 2^32 - 1 (mod p).

### 1.2 NTT — CORRECT

- Root of unity computation: g=7 as generator, omega_k = g^((p-1)/2^k). Supports NTTs up to size 2^32.
- Forward NTT: Standard Cooley-Tukey DIT with bit-reversal permutation. Butterfly kernel correctly implements (u,v) -> (u+wv, u-wv).
- Inverse NTT: Uses omega^(-1) and scales by 1/n.
- Coset NTT: Correctly multiplies by powers of shift before/after transform.
- **Test coverage:** 6 tests verify root properties (omega^n=1, omega^(n/2)=-1), round-trip identity for sizes up to 2^20.

### 1.3 Merkle Tree — CORRECT

- SHA-256 based, built bottom-up on GPU.
- Leaf hashing uses proper single-block SHA-256 padding for 8-byte (Goldilocks) or 32-byte (BLS12-381) field elements.
- Proof generation/verification uses XOR-1 for sibling index, `rev > idx` guard to prevent double-swap.
- **Test coverage:** 9 tests including deterministic root, all-leaf proof verification, wrong-leaf rejection, large tree (2^16) round-trip.

### 1.4 FRI Protocol — MOSTLY CORRECT

The FRI folding formula is correctly implemented:
```
f_new[i] = (f[i] + f[i+half])/2 + alpha * (f[i] - f[i+half]) / (2*x_i)
```

- Domain offset squaring after each fold: correct.
- Precomputed `inv_two_domain[i] = 1/(2*x_i)`: correct.
- Layer Merkle commitments: correctly built per fold.

**Issues found in FRI:**
1. **Verifier round_offset tracking** (`fri.cu:327-410`): The verifier's domain point computation across rounds is unclear — `round_offset` is initialized but the squaring progression may not match the prover's.
2. **Remainder verification** (`fri.cu:398-405`): Out-of-bounds check on `pos_in_half < commitment.remainder.size()` silently passes on failure rather than returning false.
3. **Position consistency check** (`fri.cu:376-382`): Verifier accepts if *either* of the two proofs matches the expected next-round position, giving the prover unnecessary flexibility.

### 1.5 Polynomial Operations — CORRECT

- Addition/subtraction handle different degrees.
- Multiplication implements standard convolution.
- Evaluation uses Horner's method.
- eq() polynomial: `eq_u(x) = (1-x)(1-u) + xu` — correct for sumcheck.

### 1.6 Entropy Pipeline — CORRECT (for honest provers)

The per-token entropy computation pipeline is mathematically sound:

1. **Argmax**: Finds winning token t_star with value v_star
2. **Diffs**: d_i = v_star - logits[i] (non-negative for the argmax)
3. **CDF lookup**: Phi(d_i / sigma_eff) via table with precision 2^15
4. **Win probability**: win_prob_i = (1 - Phi(d_i/sigma)) * scale
5. **Normalization**: q_idx = (win_prob * 2^p) / total_win
6. **Log lookup**: surprise = -log2(q_idx / 2^p) * scale
7. **Accumulation**: H = sum(surprise)

Several important bugs were found and fixed during development:
- `fr_gt` comparison was inverted for "negative" field elements (values > p/2)
- CDF table with precision=12 was too small (needed precision=15 for 6-sigma coverage)
- Out-of-range index clamping in tLookup incorrectly mapped valid negative rescaling values to index 0

### 1.7 Sumcheck Protocols — CORRECT FORMULA, WRONG RANDOMNESS

The sumcheck reduction kernels (inner product, Hadamard, binary) implement correct multilinear polynomial reductions. The three-evaluation-per-round pattern is standard: evaluate at {0, 1, and challenge point}, check p(0)+p(1) = claim.

**However**: Challenges (u, v vectors) are passed as function parameters, not derived from a Fiat-Shamir transcript. This is architecturally intentional — the system is designed for **interactive proofs** where the verifier supplies challenges (see design-goals.md). But the current Python verifier does not supply or check these challenges, creating a gap.

---

## 2. Soundness

### 2.1 Threat Model

From `design-goals.md`: The prover is untrusted and runs on their own hardware. The verifier must detect any attempt to forge a lower entropy bound. The system favors interactive proofs (verifier provides challenges), not Fiat-Shamir.

### 2.2 Critical Soundness Gap S1: Argmax Proofs Not Verified — HIGH

**Location:** `zkargmax.cu:66-195` (prover), `verify_entropy.py:204-209` (verifier)

The prover generates an argmax proof consisting of:
- Bit-decomposition of diffs (proving all diffs are non-negative integers fitting in `bit_width` bits)
- Indicator constraints: sum(ind) = 1 and <ind, diffs> = 0 (proving v_star is actually in the logits tensor)
- Binary check: all bit vectors are in {0,1} via random linear combination at challenge point u
- Reconstruction: sum(2^b * bits_b(u)) = diffs(u) at challenge point

The verifier reads 8 polynomials per position and checks only `ind_sum == 1` and `ind_dot == 0` as raw constants extracted from polynomial coefficients. **It does not verify the sumcheck polynomials** that would prove the bit-decomposition and binary constraints.

**Attack:** A malicious prover claims a false v_star (higher than the true max logit). This makes all diffs artificially large, which increases CDF values, decreases win probabilities, and decreases entropy. The indicator constraints `ind_sum=1, ind_dot=0` can be trivially satisfied by choosing ind to point at any logit equal to the claimed v_star. Since the bit-decomposition sumcheck is not verified, the prover can set arbitrary diffs.

**Impact:** Prover can forge an arbitrarily low entropy bound.

### 2.3 Critical Soundness Gap S2: total_win Is Self-Reported — HIGH

**Location:** `zkentropy.cu:75-77` (prover), `verify_entropy.py:222-229` (verifier)

The prover computes `total_win = sum(win_probs_all)` over all vocab_size tokens and writes it as a constant polynomial. The verifier can only check weak bounds:
- `total_win >= win_prob` (lower bound)
- `total_win <= vocab_size * cdf_scale` (upper bound)

**Attack:** A malicious prover deflates total_win to be close to win_prob. This inflates q_idx = (win_prob * 2^p) / total_win toward 2^p, making surprise = -log2(q_idx/2^p) approach 0.

**Concrete example:**
- Honest: win_prob=1000, total_win=50000, q_idx=655, surprise=4.93*scale
- Attack: win_prob=1000, total_win=1000, q_idx=32768, surprise=0
- Verifier accepts: 1000 >= 1000 and 1000 <= 32000*65536

**Impact:** Prover can hide entropy completely.

### 2.4 Soundness Gap S3: Weight-Binding Proofs Not Serialized — MEDIUM

**Location:** `zkllm_entropy.cu:196-207` (prover generates but doesn't serialize), `verify_entropy.py` (no weight verification)

The prover runs zkFC sumcheck, Rescaling lookups, and commitment opening proofs to verify that logits derive from committed weights. But these proofs are **never added to the proof vector** — only the entropy polynomials are serialized.

**Impact:** A malicious prover can fabricate logits that produce any desired entropy bound, since the verifier cannot check the link to committed weights.

### 2.5 Soundness Gap S4: Parameter Mismatch — FIXED

Prover and verifier previously used different default cdf_precision values (15 vs 12). Fixed by storing parameters in the v2 proof header. The verifier now reads cdf_precision, log_precision, and cdf_scale from the header when present.

**Note:** `zkllm_entropy_timed.cu` still writes a v1 header (missing the v2 fields), so proofs from the timed version may still trigger the mismatch.

### 2.6 Soundness Gap S5: Binary Sumcheck Proofs Discarded — PARTIALLY FIXED

Originally, `Fr_bin_sc()` output was lost. Fixed by using a batched random linear combination check (commit 957f804). However, the Python verifier still does not check these polynomials.

### 2.7 tLookup / LogUp Issues

Several issues in the lookup argument infrastructure:

| Issue | Severity | Location | Description |
|-------|----------|----------|-------------|
| Out-of-range clamping | HIGH | `tlookup.cu:424` | Values exceeding table_bound silently map to last entry. Safe for CDF (monotone), but not enforced by the proof. |
| Padding adds wrong index | HIGH | `tlookup.cu:505-512` | When D isn't a power of 2, padding uses `table[0]` and increments `m[0]`, potentially introducing false lookups. |
| No multiplicity validation | MEDIUM | `tlookup.cu:321` | No check that sum(m) == D or that multiplicities match actual counts. |
| Alpha not transcript-derived | MEDIUM | `tlookup.cu:339` | Alpha is a function parameter, not derived from Hash(table, S, m). In an interactive setting this is OK if the verifier supplies it, but the current Python verifier doesn't. |

### 2.8 FRI-Specific Soundness Issues

| Issue | Severity | Description |
|-------|----------|-------------|
| No Fiat-Shamir | CRITICAL (for non-interactive) | Challenges are passed as parameters. By design, the system uses interactive proofs where the verifier supplies challenges. Tests use hardcoded values (`i*37+13`). If deployed non-interactively, this breaks soundness. |
| Position consistency check too permissive | MEDIUM | Verifier accepts if either value or sibling matches expected position (`fri.cu:376-382`). |
| Remainder bounds check silent | MEDIUM | Failed bounds check at `fri.cu:398` doesn't error. |
| No blowup_factor validation | LOW | No check that blowup_factor >= 2 for adequate soundness. |

### 2.9 Sumcheck/Challenge Generation

The sumcheck protocols accept challenge vectors (u, v) as inputs. This is correct for interactive proofs where the verifier generates random challenges. However:

- The Python verifier (`verify_entropy.py`) does not generate or verify challenges
- The prover generates its own challenges via `curand` with a seed (not cryptographically secure)
- No transcript binding between commitment and challenges

In the **interactive model** described in design-goals.md, the verifier would supply challenges after seeing the commitment, which resolves this. But the current implementation lacks this interactive protocol — it runs as a single-shot prover outputting a file.

---

## 3. Zero-Knowledge

### 3.1 FRI Has No Blinding — NOT ZERO-KNOWLEDGE

The FRI implementation commits to polynomial evaluations directly via Merkle trees with no randomization:
- No random blinding polynomial added before commitment
- No masking of evaluations during folding rounds
- Remainder polynomial stored in the clear
- All queried evaluation points are revealed

This means the FRI commitment reveals evaluation values at queried positions, violating zero-knowledge.

### 3.2 Per-Position Values Leaked — NOT ZERO-KNOWLEDGE

The proof file contains, for each of the 1024 token positions:
- `logit_act`: the actual token's logit (reveals model confidence)
- `diff_actual`: gap between max logit and actual token's logit
- `win_prob`: probability of the actual token under the noise model
- `total_win`: sum of all win probabilities (reveals distribution spread)
- `q_fr`: quantized probability index
- `surprise`: per-token entropy contribution

These values reveal the **per-token probability distribution** of the model, far more than the aggregate entropy bound. An adversary with access to the proof file could:
- Reconstruct token-level confidence scores
- Distinguish between "certain" tokens (low surprise) and "uncertain" tokens (high surprise)
- Infer information about the model's behavior on specific inputs

### 3.3 What Would Be Needed for ZK

1. **FRI blinding**: Add a random polynomial of appropriate degree before committing. Mask folded evaluations to hide the original polynomial's values.

2. **Aggregate-only proofs**: The `plan2.md` document proposes a log-space division trick (P6) that would reveal only the aggregate entropy H, not per-position values:
   - Replace `q_idx = win_prob * 2^p / total_win` with `surprise = log(total_win) - log(win_prob)`
   - Use homomorphic sumcheck to prove the total without revealing per-position surprises
   - Status: Not yet started

3. **Commitment hiding**: The current Merkle-based commitment is binding but not hiding (anyone who knows the data can verify). For zero-knowledge, use a hiding commitment (e.g., blinded Pedersen or blinded FRI).

### 3.4 Design Goals vs. Reality

The design goals document states: "The verifier learns nothing beyond whether the prover is compliant. Model weights remain hidden behind a cryptographic commitment."

The current implementation partially achieves this:
- Weights: Hidden behind commitment (correct)
- Per-token distributions: **Revealed** in the proof file (violates ZK)
- Architecture: Must be revealed (acknowledged in design goals)

---

## 4. Architecture Assessment

### 4.1 The Two-Layer Gap

The proof system has two layers that are currently **decoupled**:

```
Layer 1 (Weight-Binding):     committed_weights -> logits
  Proved by: zkFC, Rescaling, commitment openings
  Verified by: Nobody (proofs not serialized)

Layer 2 (Entropy):            logits -> entropy bound H
  Proved by: zkEntropy (argmax + CDF + log + accumulation)
  Verified by: verify_entropy.py (arithmetic only, no cryptographic checks)
```

The Python verifier checks Layer 2 arithmetic in isolation. It cannot verify:
- That logits came from committed weights (Layer 1)
- That the argmax is correct (Layer 2 sumcheck proofs)
- That total_win is the actual sum (Layer 2 missing proof)

### 4.2 Interactive vs. Non-Interactive

The design explicitly favors interactive proofs (design-goals.md). In an interactive setting:
- Verifier sends challenges after seeing the commitment
- Even with 64-bit Goldilocks field, soundness error is 2^(-64) per attempt
- No need for Fiat-Shamir (which would enable offline grinding)

However, the **current implementation operates non-interactively**: the prover writes a proof file, the verifier reads it. There is no challenge-response protocol. The challenges are generated by the prover itself (via `curand`), which is equivalent to the prover choosing its own challenges — destroying soundness.

**Resolution needed:** Either implement the interactive protocol (verifier sends challenges after commitment), or add proper Fiat-Shamir (hash commitments to derive challenges).

### 4.3 What Works Well

Despite the gaps, several components are well-implemented:

1. **Field arithmetic**: The Goldilocks implementation is clean, efficient, and correct.
2. **NTT**: Standard Cooley-Tukey with proper root-of-unity computation.
3. **Merkle trees**: Correct SHA-256 implementation with good test coverage.
4. **FRI folding**: Mathematically correct folding with proper domain tracking.
5. **tLookup/LogUp**: Functional lookup argument for range-mapped table lookups.
6. **Entropy formula**: The noise-model approach (Gibbs inequality + normal CDF) is mathematically sound for bounding entropy.
7. **Bug fixes**: Critical bugs (fr_gt comparison, CDF table sizing, rescaling clamping) were found and fixed.
8. **Test suite**: 92 tests covering unit-level components, all passing.

---

## 5. Summary of Findings

### Critical (Must Fix for Any Security Guarantee)

| ID | Issue | Component | Can Cheat? |
|----|-------|-----------|------------|
| S1 | Argmax proofs not verified | verify_entropy.py | Yes — forge low entropy by claiming wrong max logit |
| S2 | total_win self-reported | verify_entropy.py | Yes — deflate total_win to hide entropy |
| S3 | Weight-binding proofs not serialized | zkllm_entropy.cu | Yes — fabricate logits entirely |
| -- | No interactive challenge protocol | Architecture | Yes — prover chooses own challenges |

### High (Should Fix)

| ID | Issue | Component |
|----|-------|-----------|
| -- | FRI: no Fiat-Shamir or interactive protocol | fri.cu |
| -- | tLookup: out-of-range clamping not proven | tlookup.cu |
| -- | tLookup: padding at index 0 | tlookup.cu |
| -- | zkllm_entropy_timed.cu: writes v1 header (missing params) | zkllm_entropy_timed.cu |

### Medium

| ID | Issue | Component |
|----|-------|-----------|
| -- | tLookup: no multiplicity validation | tlookup.cu |
| -- | FRI: permissive position check | fri.cu |
| -- | FRI: silent remainder bounds failure | fri.cu |
| -- | fr-tensor: weak curand-based RNG | fr-tensor.cu |
| -- | Commitment: no transcript hash binding | commitment.cu |

### Zero-Knowledge Failures

| Issue | Component |
|-------|-----------|
| No FRI blinding | fri.cu |
| Per-position values in proof file | zkentropy.cu / verify_entropy.py |
| Proof reveals token-level model confidence | Architecture |

---

## 6. Difficulty Assessment and Recommendations

### Phase 1: Quick Wins (days)

These are already done or trivial:

| Fix | Effort | Status |
|-----|--------|--------|
| S4 — Store params in v2 header | Done | Fixed (commit ca9b8bb) |
| S5 — Batch binary check | Done | Fixed (commit 957f804) |
| S7 — Gate diagnostic code | Done | Fixed (commit 957f804) |
| S6 — Explicit negative diff check | ~5 lines | Implicit catch exists via reconstruction; add explicit throw |
| zkllm_entropy_timed.cu v1 header bug | ~10 lines | Add missing cdf_precision/log_precision/cdf_scale writes |

### Phase 2: Core Soundness Fixes (1-3 weeks)

**S2 — total_win (LOW effort for interim fix, HIGH impact)**

The quickest soundness win. Three options:

| Option | Effort | Tradeoff |
|--------|--------|----------|
| Conservative bound: `total_win = vocab_size * cdf_scale` | 1 line in zkentropy.cu | Loosens entropy ~1 bit/token. Was tried then reverted as too loose. |
| Sumcheck proof of sum: `Fr_ip_sc(win_probs_all, ones, u)` | ~30 lines prover + ~50 lines verifier | Proves the sum cryptographically. Requires verifier to parse sumcheck polys. Intermediate step — proves the sum but individual win_prob values still trusted. |
| Full P6 log-space trick | See Phase 3 | Complete solution, eliminates division entirely. |

**Recommended interim:** Option 2 (sumcheck of sum). The `Fr_ip_sc` infrastructure already exists in `proof.cu:58`. Prover appends the sumcheck polynomials; verifier reconstructs and checks `p(0)+p(1) = total_win`. ~1 week including testing.

**S1 — Argmax proof verification (MEDIUM effort)**

The prover already computes the argmax proof correctly (`zkargmax.cu:66-195`). What's missing is serialization and verification:

- **Prover side (~30 lines):** Append the `combined_error` polynomial (batched binary check result) and reconstruction claim to the proof vector. Currently these are verified locally and discarded.
- **Verifier side (~80 lines):** Parse the additional polynomials. Reconstruct the combined_error check: verify that `sum_k r_k * (bits_k^2 - bits_k) + r_{bw} * (ind^2 - ind)` evaluates to 0 at challenge u. Verify reconstruction `sum(2^b * bits_b(u)) = diffs(u)`.
- **Blocker:** The verifier needs to know the random vector `r` used for batching. Either serialize `r` in the proof file, or derive it from a transcript hash (Fiat-Shamir on argmax commitment). The latter is better for soundness.
- **Estimated time:** ~2 weeks including proof format changes and testing.

### Phase 3: Full Security (1-3 months)

**Interactive challenge protocol or Fiat-Shamir (MEDIUM effort, architectural)**

This is the foundational fix that makes all other soundness guarantees meaningful. Without it, the prover chooses its own challenges.

- **Option A — Interactive protocol:** Build a verifier service that sends challenges after receiving commitments. Requires network protocol, but conceptually simple. The prover already accepts challenges as parameters. ~2-3 weeks.
- **Option B — Fiat-Shamir:** Hash each commitment layer's Merkle root into the transcript to derive the next challenge. Requires adding a transcript object that accumulates hashes. Changes to `fri.cu`, `proof.cu`, `tlookup.cu`, `zkargmax.cu`, `zkentropy.cu`. ~2-3 weeks for the transform, but design-goals.md explicitly prefers interactive over Fiat-Shamir (Fiat-Shamir enables offline grinding).

**Recommended:** Option A (interactive), consistent with design goals.

**S3 — Weight-binding proof serialization (HIGH effort)**

This is the hardest fix because it spans the full proof stack:

- **Prover side (~100 lines):** Collect already-computed proofs from `zkFC.prove()`, `Rescaling.prove()`, `verifyWeightClaim()` and append to the proof file. The proofs are already generated — they just aren't serialized. Moderate effort.
- **Verifier side (VERY HIGH):** The Python verifier would need to:
  - Parse FC sumcheck polynomials and verify the multilinear reduction
  - Parse Rescaling tLookup proofs and verify LogUp arguments
  - Verify FRI-PCS commitment openings (or Pedersen openings for BLS12-381)
  - This requires implementing field arithmetic, polynomial evaluation, and commitment verification in Python (or calling into a C++ library)
- **Estimated time:** 4-8 weeks. The prover side is straightforward; the verifier is a major engineering project.

**P6 — Log-space division trick (MEDIUM prover + HIGH verifier)**

The elegant long-term solution that also achieves zero-knowledge:

- **Core idea:** Replace `q_idx = (win_prob * 2^p) / total_win` with `surprise = log_table[total_win] - log_table[win_prob]`. This eliminates variable-denominator division and enables aggregate-only proofs.
- **Prover (~200 lines):** Refactor `computePosition` and `prove` in zkentropy.cu. Uses existing zkLog and tLookupRangeMapping infrastructure. Main challenge: table sizing for total_win (range up to ~2^31 at default cdf_scale=65536; options: reduce cdf_scale, use 2-stage log, or accept practical bounds).
- **Verifier (HIGH):** Requires cryptographic verification of lookup proofs — same infrastructure needed for S1 and S3.
- **Estimated time:** 3-6 weeks. Depends heavily on whether the cryptographic verifier from S3 already exists.

**FRI blinding for zero-knowledge (MEDIUM effort)**

- Add random polynomial of degree <= blowup_factor before committing
- Mask remainder polynomial
- Adjust folding to maintain blinding invariant
- ~2 weeks, well-understood technique from FRI literature

### Summary: Effort vs. Impact

```
                          LOW EFFORT ──────────────────> HIGH EFFORT
                    ┌─────────────────────────────────────────────────┐
HIGH IMPACT         │  S2 interim    S1 verify    Interactive    S3   │
(soundness)         │  (1 week)      (2 weeks)    protocol      (4-8 │
                    │                              (2-3 weeks)   wks) │
                    ├─────────────────────────────────────────────────┤
MEDIUM IMPACT       │  S6 check     tLookup       P6 log-space       │
                    │  (1 day)      padding fix    (3-6 weeks)        │
                    │               (1 week)                          │
                    ├─────────────────────────────────────────────────┤
LOW IMPACT          │  timed.cu     FRI position   FRI blinding       │
(defense-in-depth)  │  header fix   check fix      (2 weeks)          │
                    │  (1 day)      (1 day)                           │
                    └─────────────────────────────────────────────────┘
```

### Key Insight: The Verifier Is the Bottleneck

The prover infrastructure (sumchecks, lookups, FRI, commitment openings) **already exists and works**. The fundamental gap is that `verify_entropy.py` is an **arithmetic transcript checker**, not a **cryptographic proof verifier**. Fixes S1, S2, S3, and P6 all converge on the same requirement: a verifier that can parse and validate sumcheck polynomials, tLookup proofs, and commitment openings.

Building this cryptographic verifier — whether in Python with C extensions, or as a standalone C++ verifier — is the single highest-leverage investment. Once it exists, all of S1, S2 (via sumcheck), S3, and P6 become primarily prover-side plumbing (appending already-computed proofs to the output file).

### Recommended Roadmap

1. **Now:** S6 explicit check + timed.cu header fix (1 day)
2. **Week 1-2:** S2 interim fix via `Fr_ip_sc` sumcheck of total_win
3. **Week 2-4:** S1 argmax proof serialization + verification
4. **Week 3-6:** Interactive challenge protocol
5. **Week 4-10:** Cryptographic verifier (enables S3 and P6)
6. **Week 8-14:** P6 log-space trick (true ZK for entropy layer)
7. **Week 10-16:** S3 weight-binding serialization + FRI blinding
