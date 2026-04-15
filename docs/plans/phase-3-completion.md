# Phase 3 Completion Plan — Paper-Faithful Cleanup, Wiring, Multi-Hadamard, E2E Tests

**Date:** 2026-04-15
**Status:** Revised after Hyrax paper review (2026-04-15)
**Branch:** `phase-3-zk-sumcheck` (continue; merge to `main` when §5 lands)
**Estimated effort:** ~5 working days

## Context

`phase-3-zk-sumcheck.md` lists five subcomponents:

1. Hyrax §A.1 Σ-protocols — **DONE** (commit `e7536a4`).
2. Round-polynomial commitment helpers — **DONE** (commit `2d34a94`).
3. ZK sumcheck driver (IP + binary + degree-2 HP) — **DONE** (`2d34a94` +
   `2eadcc2`), but with paper deviations flagged in §0 below.
4. Wiring into call sites — **OPEN**, this plan.
5. End-to-end tests + distinguisher — **OPEN**, this plan.

One residual design item is also scoped here: the degree-K multi-Hadamard
extension of the HP driver, needed by softmax.

## Paper review (2026-04-15) — deviations from Hyrax §4 Protocol 3

After re-reading Wahby et al. 2018 (eprint 2017/1132) §4 and §A.1
alongside our driver, two deviations surfaced. Both are fixed in §0
below.

**G1 — Missing per-coefficient proof-of-opening.** §4 (p. 7):

> "In round j of the sum-check, P commits to s_j(t) = c_{0,j} + c_{1,j}·t
> + c_{2,j}·t² + c_{3,j}·t³, via δc_{0,j}←Com(c_{0,j}), …, and **P and V
> execute proof-of-opening for each one.**"

Our `ZKSumcheckRound` carries only a single `eq_proof`, no per-
coefficient openings. `proof-of-equality` binds message equality but
does not prove knowledge of either commitment's opening individually —
that's `proof-of-opening`'s job. Without it the extractor can't lift
round-polynomial coefficients. Knowledge-soundness deviation, not
completeness, but paper-faithful means we add them.

**G2 — Top-level `T_0` redundancy.** §4 ("Step 1"):

> "V computes **C_0 = Com(Ṽ_y(q',q); 0)**."

Paper has the *verifier* construct the initial claim commitment from the
public sum `S` with blinding 0 — no prover commitment, no proof-of-
opening. Our current design has the prover commit `T_0 = Com(S; ρ_0)`
and send a `SigmaOpeningProof`; this is (a) redundant crypto work, and
(b) weak-binding (a prover can commit to `S' ≠ S` and still produce a
valid opening for `S'`). Rewriting to match the paper deletes code and
closes the weak-binding hole simultaneously.

**G3 — Eq-factored HP is not from Hyrax §4.** The Libra-style eq-
factored round identity `(1−u)·h(0) + u·h(1) = h_prev(v_prev)` used by
our HP driver is from Xie et al. 2019 Appendix A, not Hyrax. This is
already cited correctly in `zk_sumcheck.cuh`; no code change — noted
for completeness. Still uses §A.1 `proof-of-equality` for the round-to-
round tie, just with weights `α = (1, u, u)` instead of `(2, 1, 1)`.

## Inventory — real call sites

Grep over `src/`, `bin/` for the four plain sumcheck variants yields **8
call sites**. Binary sumcheck is declared but unused outside `proof.cu`.
`zkfc.cu` / `zkargmax.cu` go through `verifyWeightClaimZK` (Phase 2 CG1)
— not sumcheck-level.

| # | File | Line | Variant | Notes |
|---|------|------|---------|-------|
| 1 | `src/entropy/zkentropy.cu` | 177 | `inner_product_sumcheck` | bit-plane check, looped over bit `b` |
| 2 | `src/entropy/zkentropy.cu` | 478 | `inner_product_sumcheck` | `wp_partial · ones_V` row-sum |
| 3 | `src/entropy/zkentropy.cu` | 499 | `inner_product_sumcheck` | `win_probs_all · indicator` extract |
| 4 | `src/entropy/zkentropy.cu` | 581 | `inner_product_sumcheck` | `surprise_sum_input · ones_T` |
| 5 | `src/llm/rmsnorm.cu` | 48 | `hadamard_product_sumcheck` | degree-2 HP |
| 6 | `bin/zkllm_entropy.cu` | 187 | `hadamard_product_sumcheck` | degree-2 HP |
| 7 | `bin/zkllm_entropy_timed.cu` | 182 | `hadamard_product_sumcheck` | degree-2 HP (benchmark) |
| 8 | `src/zknn/zksoftmax.cu` | 183 | `multi_hadamard_sumchecks` | degree-K, K = `Y_segments.size()` |

## Step 0 — Cleanup pass (paper-faithfulness + uniform recursion)

**Three changes, done together because they touch the same file:**

### 0a. Drop `T_0` / `T0_open` (G2)

- Delete `ZKTopLevel`, `commit_and_open_top`, `verify_top_open`.
- Delete `T0`, `T0_open` fields from `ZKSumcheckProof`.
- Delete the top-level entry from `sigma_challenges` (size drops from
  n+1 to n).
- First round's `T_prev_eval` is `S·U`, computed verifier-side via the
  existing size-1 commitment helper (`commit_mU_rH(U, H, S, 0)` or
  equivalent).
- Prover's handoff loses `rho0`; `rho_prev_eval` for round 0 is 0.
- Test update: `test_zk_sumcheck.cu` drops `ZKTopLevel`-specific
  negatives; re-enables the "wrong claimed S" sub-test (now rejected by
  round-0 equality proof against `S·U`, not by a binding check).

### 0b. Add per-coefficient proof-of-opening (G1)

- Extend `ZKSumcheckRound`:
  ```cpp
  struct ZKSumcheckRound {
      std::vector<G1Jacobian_t>        T;
      std::vector<SigmaOpeningProof>   T_open;   // NEW: size d+1
      SigmaEqualityProof               eq_proof;
  };
  ```
- In `emit_zk_round`: loop over `T` and call `prove_opening(U, H, T[k],
  coeffs[k], rc.rho[k], e_open_k)`.
- In `verify_zk_round`: loop over `round.T` and call `verify_opening(U,
  H, round.T[k], round.T_open[k], e_open_k)`.
- Challenge layout: per round consumes d+2 sigma challenges — one per
  opening, plus one for the equality. Total sigma challenges = n·(d+2).
- Redocument sigma_challenges sizing in all three driver headers.

### 0c. Retrofit HP driver to back-to-front (Q2 resolution)

- All four ZK drivers consume challenges back-to-front, matching plain
  `multi_hadamard_sumchecks`.
- Extract a shared `zk_sc_recurse_back_to_front<EmitFn, FoldFn>` helper
  that handles the generic glue (commit round poly → per-coef openings
  → equality proof → fold tensors → recurse). Variant-specific kernels
  (`Fr_ip_sc_step`, `Fr_hp_sc_step`, `Fr_bin_sc_step`) plug in as
  function arguments.
- Net effect: IP, binary, HP drivers all become ~30 LoC thin wrappers
  around the shared helper. Multi-HP in Step 1 becomes a similar thin
  wrapper.

**Effort:** ~1 working day for all of 0a + 0b + 0c, including test
updates. Existing positive/negative tests in `test_zk_sumcheck.cu`
catch regressions.

## Step 1 — Multi-Hadamard degree-K driver

After Step 0's shared helper, this is mostly plumbing.

**Starting point.** `multi_hadamard_sumchecks` in `src/proof/proof.cu:220`
recursively peels one variable per call:

- consumes `u.back()` and `v.back()`;
- splits each `X ∈ Xs` via `hadamard_split_kernel` into `X0` (low half)
  and `X1 - X0` (high minus low);
- builds degree-(K+1) round polynomial `p(X) = eq(X, u.back()) · Π_k
  h_k(X)` where each `h_k` is linear in `X`;
- identity check: `claim == p(0) + p(1)`;
- recurses with `new_claim = p(v.back())` and `Xs` reduced to low-half.

**ZK layer.** Round emits K+2 coefficient commitments. Round identity
at commitment level uses weights `α = (1, u_j, u_j, …, u_j)` of length
K+2 (one `1` plus K+1 copies of `u_j`) — same `combine_commitments_weighted`
primitive.

**Final-claim.** `T_final` commits to `Π_k X_k(v) · eq(v, u_reversed)`.
Softmax's caller discharges per-segment scalars via proof-of-product
chained K-ary (or equivalently by emitting the contracted scalars and
letting `verifyWeightClaimZK` handle each).

**New API:**

```cpp
ZKSumcheckProof prove_zk_multi_hadamard(
    G1Jacobian_t U, G1Jacobian_t H,
    Fr_t         claimed_S,
    const std::vector<FrTensor>& Xs,
    const std::vector<Fr_t>& u_challenges,     // size n
    const std::vector<Fr_t>& v_challenges,     // size n
    const std::vector<Fr_t>& sigma_challenges, // size n·(K+3)
    std::vector<Fr_t>&       final_Xs_out,     // size K
    ZKSumcheckProverHandoff& handoff_out);

bool verify_zk_multi_hadamard(
    G1Jacobian_t U, G1Jacobian_t H,
    Fr_t         claimed_S,
    const ZKSumcheckProof& proof,
    uint         K,
    const std::vector<Fr_t>& u_challenges,
    const std::vector<Fr_t>& v_challenges,
    const std::vector<Fr_t>& sigma_challenges);
```

**Test.** `negatives_multi_hadamard(K)` for K ∈ {2, 3, 4} in
`test_zk_sumcheck.cu`. Positive: round-trip + verify `T_final` opens to
`Π_k X_k(v) · eq(v, u_reversed)`.

**Effort:** ~2 hours once Step 0's helper is in place.

## Step 2 — Wire the seven mechanical sites

Per-site pattern: replace plain sumcheck call with ZK equivalent, thread
`(U, H)`, pass fresh `sigma_challenges` (size n·(d+2) for IP/binary,
n·(d+2) for degree-2 HP), feed `T_final` into `verifyWeightClaimZK`.

**Challenge provenance.** Phase 3 non-goal: interactive, caller-supplied
oracle. Same `random_vec(...)` pattern Phase 2 uses.

**Per-site breakdown:**

- **Sites 1–4 (`zkentropy.cu`):** IP-only. Plain code returns `[a(u),
  b(u)]` inline; ZK returns `T_final` + the two scalars via the handoff
  struct. Thread `ρ_final` into the surrounding `Weight`. ~1 day total.
- **Site 5 (`rmsnorm.cu:48`):** Plain call discards return value.
  Mechanical swap. ~2 hours.
- **Sites 6–7 (`bin/zkllm_entropy*.cu`):** Top-level driver; `_timed`
  keeps instrumentation symmetry. ~2 hours.
- **Site 8 (softmax):** Blocks on Step 1. Swap for
  `prove_zk_multi_hadamard`; softmax's downstream `claim_Y_segs` becomes
  commitments + contracted scalars. ~half day.

**Effort:** ~2 days for sites 1–7; softmax stacks on Step 1.

## Step 3 — End-to-end pipeline test

**File:** `test/test_entropy_zk_pipeline.cu` (new).

Run the entropy pipeline end-to-end in ZK mode. Assertions:
- prover runs to completion without throwing;
- verifier accepts;
- every `T_final` chains into a successful `verifyWeightClaimZK` /
  proof-of-product;
- zero plain-sumcheck leaks in the transcript (assert on proof-struct
  types).

Negatives: one per verifier check-point. Reuse `negatives_*` helpers
from `test_zk_sumcheck.cu`.

**Effort:** ~1 day.

## Step 4 — χ² distinguisher test

**File:** `test/test_zk_distinguisher.cu` (new).

Same shape as Phase 2's `test_opening_distinguisher`. Two witnesses,
same claim, bucketed-LSB χ² on pooled transcript field elements. Per
variant: IP, degree-2 HP, binary, multi-HP (K=3).

**Effort:** ~1 day.

## Recommended ordering

0. **Cleanup pass** (G1 + G2 + direction retrofit). Touches one file;
   preconditions everything else; makes Step 1 trivial. **~1 day.**
1. Multi-Hadamard driver. **~2 hours.**
2. Seven mechanical call sites. **~2 days.**
3. Softmax (after Step 1). **~half day.**
4. E2E pipeline test. **~1 day.**
5. χ² distinguisher. **~1 day.**

**Total: ~5 working days.**

## Open questions / deferred to user

- **Q3** (challenge provenance at call sites): `random_vec(n·(d+2))`
  alongside existing fold `random_vec(n)`, same model as Phase 2. **Default yes** unless user objects.
- **Q4** (distinguisher threshold): p ≥ 0.01, N ≈ 10k samples (same as
  Phase 2). **Default yes.**
- **Q5** (E2E scope): synthetic fixture for CI, real `zkllm_entropy`
  gated on env var. **Default yes.**

Q1 is resolved by Step 0a. Q2 is resolved by Step 0c.

## Non-goals (carried from kickoff)

- Fiat-Shamir — Phase 5+.
- O(log n) opening — Phase 2 future work.
- Removing plain-sumcheck variants from `proof.cu` — keep until last
  caller migrated; clean-up commit at the end.
- GPU-side Σ-protocol kernels — still CPU-side.

## Files to touch

- `src/proof/zk_sumcheck.cuh`, `src/proof/zk_sumcheck.cu` — Step 0 +
  multi-HP.
- `src/proof/zk_round_commit.*` — unchanged; existing helpers suffice.
- `src/entropy/zkentropy.cu` — sites 1–4.
- `src/llm/rmsnorm.cu` — site 5.
- `bin/zkllm_entropy.cu`, `bin/zkllm_entropy_timed.cu` — sites 6–7.
- `src/zknn/zksoftmax.cu` — site 8.
- `test/test_zk_sumcheck.cu` — Step 0 test updates, multi-HP tests.
- `test/test_entropy_zk_pipeline.cu` (new) — Step 3.
- `test/test_zk_distinguisher.cu` (new) — Step 4.
- `Makefile` — add the two new test targets.

## Exit criteria

- `ZKTopLevel`, `commit_and_open_top`, `verify_top_open` gone.
- Every `ZKSumcheckRound` carries d+1 proof-of-openings + 1 equality
  proof.
- All four ZK drivers (IP, binary, degree-2 HP, multi-HP) share one
  back-to-front recursion helper.
- All 8 call sites under ZK path.
- `test_zk_sumcheck`, `test_entropy_zk_pipeline`, `test_zk_distinguisher`
  pass on H100 server.
- `grep -rn "inner_product_sumcheck\|hadamard_product_sumcheck\|multi_hadamard_sumchecks" src/ bin/`
  returns no matches outside `src/proof/proof.cu`.
- Branch merged to `main`.
