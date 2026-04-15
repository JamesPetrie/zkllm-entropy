# Phase 3 — ZK Sumcheck (Hyrax §4 / §A.1 / §A.2)

**Date:** 2026-04-15
**Status:** Draft / kickoff
**Branch:** `phase-3-zk-sumcheck` (to be created)
**Parent plan:** `docs/plans/plan-production-readiness.md`
**Estimated effort:** ~2 weeks (revised down from parent plan's 3 weeks;
see "Effort breakdown" below).

## Source-material note

This phase implements the **Hyrax ZK sumcheck** — the blinded variant of
the Thaler-style multilinear sumcheck used throughout the entropy
pipeline. The authoritative construction is **Wahby et al. 2018, "Doubly-
efficient zkSNARKs without trusted setup" (eprint 2017/1132)**:

- **§4, Protocol 3 (p. 11–13) — ZK sumcheck.** Per round `j`, the
  prover Pedersen-commits the round polynomial `g_j(X)` rather than
  sending coefficients in the clear, and ties consecutive rounds
  together via the Σ-protocols of §A.1.
- **§A.1 Figure 5 (p. 17) — proof-of-opening / proof-of-equality /
  proof-of-product.** Three Σ-protocols over Pedersen commitments.
  Phase 3's per-round glue.
- **§A.2 Figure 6 (p. 18) — proof-of-dot-prod.** Already shipped in
  Phase 2. Reused at the *end* of Phase 3 to discharge the final
  dot-product claim the sumcheck reduces to.

The plain sumcheck variants currently in `src/proof/proof.cu`
(`inner_product_sumcheck`, `hadamard_product_sumcheck`,
`binary_sumcheck`, `multi_hadamard_sumchecks`) emit round polynomials
— degree 1, 2, 2, K respectively — as scalars `a(0), b(0), out0, out1,
out2, …`. Every one of these values leaks a linear function of the
witness. Phase 3 replaces the scalars with Pedersen commitments and
bolts on per-round Σ-protocols so nothing beyond the final evaluation
is revealed.

> "In this study, Hyrax [54], a variant of the Pedersen commitment
> [43] that does not require a trusted setup, is used as an instantiation
> of the polynomial commitment scheme."
> — Sun, Li, Zhang 2024, *zkLLM: Zero Knowledge Proofs for Large
> Language Models*, §3.4.

## Goal

Replace the four plain-sumcheck variants in `src/proof/proof.cu` with a
ZK sumcheck whose transcript consists entirely of Pedersen commitments
and Σ-protocol responses. By the end of Phase 3, running
`zkllm_entropy` emits a proof where:

- No sumcheck round polynomial is transmitted in the clear.
- Every commitment-to-commitment transition across rounds is backed by
  a Figure 5 proof-of-opening / -equality / -product.
- The final round's dot-product claim is discharged via the Phase 2
  Figure 6 primitive — no second ZK argument to implement.

## Non-goals

- **O(log n) opening.** Phase 3 finalizes with Phase 2's Figure 6
  (O(n)), not Figure 7+8. Pivoting is tracked as Phase 2 future work.
- **Fiat-Shamir.** Interactive. Per-round challenges come from the same
  caller-supplied oracle model Phase 2 uses. Phase 5+ wraps this in a
  hash.
- **Plonky2-style masking.** The `zk-masking-implementation` branch's
  vanishing-polynomial + XZZ+19 approach is **reference only** (per
  parent plan 2026-04-15 decision). Phase 3 is Hyrax-faithful.
- **GPU-side Σ-protocol kernels.** The Σ-protocol work is tiny (a
  handful of group ops per round). Keep it CPU-side or use the existing
  MSM kernels; no new CUDA kernels specifically for §A.1.
- **Verifier binary.** Phase 5 ships the standalone verifier. Phase 3
  only provides an in-process `verify_*` entry point callable from
  tests so negative cases run end-to-end.
- **Removal of plain sumcheck.** Keep the existing plain variants
  compiling until the last caller is migrated, same playbook as Phase 2
  with `me_open` vs `open_zk`.

## Construction (Hyrax §4 Protocol 3)

Notation: paper is multiplicative, our code is additive. Throughout
this doc `Com(m; r) = m·U + r·H` denotes a scalar-over-generator
Pedersen commitment (Phase 2's `τ`), and `Com_{g⃗}(v⃗; r) = Σ vᵢ·Gᵢ + r·H`
a vector Pedersen commitment (Phase 2's `ξ`, `δ`). Both use the pp
already in place since Phase 1.5: generators `{Gᵢ}`, `H`, `U` are
hash-to-curve-derived, so Σ-protocols compose cleanly.

### Prover inputs (same as plain sumcheck)

A claim of the form

    S = Σ_{x∈{0,1}ⁿ} f(x)

where `f` is a product of multilinear polynomials (e.g. `f = a · b` for
inner-product sumcheck). The prover knows `f` as a tensor and is given
a challenge sequence `u = (u_1, …, u_n)` that in the current plain
codebase is consumed at the *end* of each round. In the ZK variant the
challenge pattern stays the same — the change is what the prover sends.

### Round `j` (plain → ZK transformation)

**Plain (today, `src/proof/proof.cu`):**
1. Prover sends round polynomial `g_j(X)` as its `d+1` evaluations on
   `{0, 1, …, d}` (for degree-`d` rounds) — these are the
   `out0, out1, out2, …` scalars emitted per round.
2. Verifier checks `g_{j-1}(r_{j-1}) = g_j(0) + g_j(1)` by literal
   subtraction.
3. Verifier sends `r_j` (current code: `u[j]` or `v[j]`).
4. Prover commits to sending `g_j(r_j)` as next round's claim.

**ZK (Phase 3, Protocol 3 of §4):**
1. Prover samples fresh blindings `ρ_j^{(0)}, …, ρ_j^{(d)}` and sends
   Pedersen commitments `T_j^{(k)} = Com(g_j(k); ρ_j^{(k)})` for
   `k = 0, …, d`. (One commitment per coefficient/evaluation.)
2. Prover proves `T_{j-1}(r_{j-1}) = T_j^{(0)} + T_j^{(1)}` via a
   Figure 5 **proof-of-equality** — the previous-round-evaluation
   commitment equals the sum of this round's `X=0` and `X=1`
   commitments (the sumcheck identity, now at commitment level).
3. Verifier sends `r_j`.
4. Prover locally computes `T_j(r_j) = Com(g_j(r_j); ρ_j(r_j))` by
   homomorphic linear combination of `{T_j^{(k)}}` with the public
   Lagrange weights — no new commitment, no new transcript element.
   `ρ_j(r_j)` is derived identically by the prover.
5. Loop: `T_j(r_j)` is the input commitment for round `j+1`.

At the top, the prover publishes `T_0 = Com(S; ρ_0)` with the claimed
sum. At the bottom (round `n`), the last-round commitment
`T_n(r_n) = Com(g_n(r_n); ρ_n(r_n))` commits the final evaluation.
Under the sumcheck identity this equals `Com(f(r); ρ_final)`.

### Final reduction (§A.2 Figure 6, Phase 2 primitive)

After `n` rounds the sumcheck collapses to a claim "the committed
scalar at `T_n(r_n)` equals `f(r)`". For the sumcheck shapes we use
— inner-product (`f = a·b`), Hadamard (`f = a·b·eq`), multi-hadamard
(`f = ∏ Xᵢ`) — `f(r)` is a single multilinear evaluation (or a small
number of them) at the public challenge `r`, which is exactly the claim
shape Phase 2's `open_zk` / `verify_zk` already handles.

So the final round discharges to one `verify_zk` call per operand
tensor. **No new opening protocol.** The Phase 2 Figure 6 transcript is
reused verbatim.

### Degree-2 and degree-K variants

- Inner-product (`d = 1`): `a(X)·b(X)` where `a, b` are multilinear in
  the round variable → `g_j` is degree 2. Per round: `T_j^{(0)}, T_j^{(1)},
  T_j^{(2)}`. Identical machinery.
- Hadamard (`d = 2`): same emission count as inner-product.
- Binary (`d = 2`): `a(X)·(a(X)-1)` → degree 2.
- Multi-hadamard of `K` factors (`d = K`): per round `T_j^{(0)}, …, T_j^{(K)}`.

All handled by a single generic driver parameterized by `d`.

## Σ-protocols (Hyrax §A.1 Figure 5)

Three protocols, each a few lines of code. All are Schnorr-style — two
moves on top of Pedersen commitments.

### proof-of-opening

*"Given commitment `C`, prover knows `(m, r)` such that
`C = m·U + r·H`."*

**P → V:** `A = s_m·U + s_r·H` for fresh `(s_m, s_r)`.
**V → P:** challenge `e`.
**P → V:** `z_m = s_m + e·m`, `z_r = s_r + e·r`.
**V check:** `z_m·U + z_r·H =? A + e·C`.

### proof-of-equality

*"Given `C_1 = m·U + r_1·H`, `C_2 = m·U + r_2·H`, prover knows `m, r_1,
r_2` (same `m`)."*

Equivalent to proof-of-opening for `C_1 - C_2 = (r_1 - r_2)·H`: a
Schnorr of the discrete log in base `H`.

### proof-of-product

*"Given `C_a = a·U + r_a·H`, `C_b = b·U + r_b·H`, `C_c = c·U + r_c·H`
with `c = a·b`, prover knows `a, b, r_a, r_b, r_c`."*

Paper's construction (Figure 5, third sub-protocol): prover sends
`E = b·B + r_E·H` for auxiliary `B = s_a·U + s_r·H`; responds with
linear combinations that force `c = a·b` under the Σ-identity.
~20 lines in code; protocol is stock textbook (Camenisch-Stadler).

### Where each is used in Phase 3

| Σ-protocol | Used for |
|---|---|
| proof-of-opening | Top-level `T_0 = Com(S; ρ_0)` pinning the claimed sum `S`. |
| proof-of-equality | Per-round sumcheck identity `T_{j-1}(r_{j-1}) = T_j^{(0)} + T_j^{(1)}` (trivially homomorphic — the paper's "sum-of-two-openings" is a single equality check in the exponent). |
| proof-of-product | Not needed if every round is linear-in-commitments (which is the case for the sumcheck recursion — round-to-round is a linear combination with public weights). **See "Open question" below.** |

**Open question.** The Hyrax paper cites proof-of-product primarily for
the product-of-multilinears case that arises in Phase 4+ serialization,
not for the sumcheck recursion itself. For Phase 3 we may be able to
get away with **just** proof-of-opening (top) + proof-of-equality
(per round). Flagging this for the A3 audit to settle definitively
before we commit to the full Figure-5 triple in the implementation.
See "Risks" §1.

## Subcomponent breakdown (build order)

### (1) `src/proof/hyrax_sigma.{cu,cuh}` — Σ-protocols from Figure 5

- Structs: `ProofOfOpening`, `ProofOfEquality`, (maybe) `ProofOfProduct`.
- Entry points: `prove_opening`, `verify_opening` (and equality).
- Pure-CPU; calls the existing `G1Jacobian_t` group ops from
  `src/tensor/g1-tensor`.
- **~300 LOC** including tests.
- **Tests:** honest accept, tampered `A`/`z_m`/`z_r` each reject,
  wrong-commitment rejects, wrong-witness rejects, distinguisher across
  challenges.
- **Effort: ~3 days.**

### (2) `src/proof/zk_round_commit.{cu,cuh}` — commitment-to-round-poly

- Given a degree-`d` polynomial as `d+1` evaluations, commit each as
  `Com(y_k; ρ_k)` with fresh blindings; return a `RoundCommitment`
  bundle `{T^{(0)}, …, T^{(d)}}` plus retained `{ρ_k}`.
- Helper for homomorphic evaluation `eval_at(r)` — folds the
  commitments and blindings via the Lagrange weights. Verifier does the
  same fold from `{T^{(k)}}` alone.
- Linearity check: `fold({Com(y_k; ρ_k)}, r) = Com(p(r); Σ Lag_k(r)·ρ_k)`.
- **~200 LOC** including tests.
- **Tests:** fold correctness on random polys; commitment-level
  fold matches scalar-level fold; tampered coefficient commitment
  propagates.
- **Effort: ~2 days.**

### (3) ZK sumcheck driver — `src/proof/zk_sumcheck.{cu,cuh}`

Replaces the four plain variants. Generic signature parameterized by
the round polynomial's degree and its coefficient-evaluation kernel.

The driver reuses the *kernels* from `src/proof/proof.cu`
(`Fr_ip_sc_step`, `Fr_hp_sc_step`, `Fr_bin_sc_step`) unchanged — they
already compute `out0, out1, out2` scalars per round. The ZK wrapper:

1. For each round, run the existing kernel to get `{out_k}`.
2. Commit each `out_k` via subcomponent (2).
3. Emit the previous-round proof-of-equality tying last round's
   `T_{j-1}(r_{j-1})` to `T_j^{(0)} + T_j^{(1)}`.
4. Accept `r_j` from the challenger (same API as plain sumcheck).
5. Fold `{T_j^{(k)}}` into `T_j(r_j)` for the next round.

**Transcript struct:** `ZKSumcheckProof { T0_open; per_round[]; final_fold; }`
carrying the top-level opening proof, `n` round bundles, and the handoff
to Phase 2's `OpeningProof`.

Variants to cover: `inner_product`, `hadamard_product`, `binary`,
`multi_hadamard` (degree `d = 1, 2, 2, K`). One generic driver plus
four degree-specific thin wrappers.

- **~400 LOC** including the four wrappers.
- **Tests:**
  - Positive: each variant at `N = 4, 16, 64, 256`; honest transcript
    verifies end-to-end.
  - Composition: chained `inner_product → hadamard_product` (mimicking
    zkFC + rescale patterns) still verifies.
  - Negative: tamper each of (`T_j^{(k)}`, Σ-response `z_m`, top-level
    opening, final Figure-6 transcript); verifier rejects with a
    specific error message so the test can distinguish which check
    failed.
- **Effort: ~1 week.**

### (4) Wiring into call sites

Every plain-sumcheck call site switches to the ZK driver:

| Call site | File:line | Variant |
|---|---|---|
| Softmax segment proofs | `src/zknn/zksoftmax.cu:183` | multi_hadamard |
| RMSNorm Hadamard (layer) | `src/llm/rmsnorm.cu:48` | hadamard |
| RMSNorm Hadamard (entropy) | `bin/zkllm_entropy.cu:187` | hadamard |
| RMSNorm Hadamard (timed) | `bin/zkllm_entropy_timed.cu:182` | hadamard |
| Entropy bit-plane binary | `src/entropy/zkentropy.cu:177` | inner_product (×B) |
| Entropy row-sum | `src/entropy/zkentropy.cu:478` | inner_product |
| Entropy indicator extract | `src/entropy/zkentropy.cu:499` | inner_product |
| Surprise accumulation | `src/entropy/zkentropy.cu:581` | inner_product |
| zkArgmax binary check | `src/zknn/zkargmax.cuh` (declaration) | binary |
| zkFC prove | `src/zknn/zkfc.cu` (via `verifyWeightClaim` path) | inner_product |

Most sites just switch function names + return types. The thing to
watch is that `serialize_ip_sumcheck` in `zkentropy.cu` assumes the
plain scalar transcript; a ZK-aware serializer replaces it (defers the
final on-disk format to Phase 4 — in Phase 3 we just pass through an
in-memory `ZKSumcheckProof`).

- **~300–400 LOC** of mechanical changes.
- **Effort: ~3–4 days.**

### (5) End-to-end tests

- `test_zk_sumcheck`: per-variant positive + negative.
- `test_entropy_zk_pipeline`: `zkllm_entropy` runs produce proofs where
  every round polynomial is a commitment (not a plaintext poly).
- Distinguisher: two runs with the same inputs + different challenges
  produce transcripts whose commitment+response distributions pass a
  χ² gate (same style as Phase 2's `test_opening_distinguisher`).
- Regression: Phase 1/1.5/2 suites unchanged.
- **Effort: ~2–3 days.**

## Effort breakdown (honest)

| Item | Estimate |
|---|---|
| (1) Σ-protocols | 3 days |
| (2) Round commit | 2 days |
| (3) Driver | ~1 week |
| (4) Wiring | 3–4 days |
| (5) Tests | 2–3 days |
| **Total** | **~2 weeks** |

Revised down from the parent plan's 3-week estimate because:
- Phase 2 already shipped the Figure 6 primitive this phase relies on.
- Phase 1.5 already shipped independent hash-to-curve generators, so
  the Σ-protocols compose without a `(G_i, H)` dlog-known caveat.
- The sumcheck *kernels* (`Fr_ip_sc_step` et al.) are reused unchanged;
  only the surrounding transcript wrapping is new.

If Σ-protocol audit (A3) surfaces the proof-of-product requirement,
add ~1 day. If the composition tests surface a cross-variant bug, add
~2–3 days for debug.

## Tests

### Positive

1. `test_sigma_opening_roundtrip`, `test_sigma_equality_roundtrip`
   — honest Σ-protocol accept.
2. `test_round_commit_fold` — commitment-level fold matches
   scalar-level fold for random degree-`d` polys.
3. `test_zk_inner_product_sumcheck` at `N = 4, 16, 64, 256`.
4. `test_zk_hadamard_sumcheck` ditto.
5. `test_zk_binary_sumcheck` ditto.
6. `test_zk_multi_hadamard_sumcheck` for `K = 2, 3, 4`.
7. `test_zk_sumcheck_composition` — chained `inner → hadamard` (zkFC
   shape).
8. `test_entropy_zk_pipeline` — `zkllm_entropy` end-to-end produces
   valid `ZKSumcheckProof` at every sumcheck site, all verify.

### Negative

9. `test_sigma_opening_tampered_A` — alter Σ-protocol's `A`; rejects.
10. `test_sigma_opening_tampered_z` — alter `z_m` or `z_r`; rejects.
11. `test_zk_sumcheck_tampered_T` — alter one of `T_j^{(k)}`;
    proof-of-equality chain breaks at round `j`.
12. `test_zk_sumcheck_tampered_final` — alter the Figure 6 transcript
    at the end; rejects there.
13. `test_zk_sumcheck_wrong_top` — prover publishes wrong top-level `S`;
    proof-of-opening for `T_0` rejects.
14. `test_zk_sumcheck_wrong_r` — challenger sends `r' ≠ r` at verify;
    proof-of-equality at that round fails.

### Hiding (statistical)

15. `test_zk_sumcheck_distinguisher` — 5 000 runs at the same inputs
    with fresh challenges. Each emitted group element (`T_j^{(k)}`
    across all rounds) projected to an F_r hash passes a χ² uniformity
    gate. Catches blinding-reuse regressions across rounds or across
    calls.
16. `test_sigma_distinguisher` — same discipline, for Σ-protocol `A`
    across many runs with the same `C, m, r`.

### Regression

17. All Phase 1 / 1.5 / 2 suites unchanged.
18. `test_zkargmax`, `test_zklog`, `test_zknormalcdf`, `test_zkentropy`
    regression-pass against the ZK driver (their assertions are about
    the claim, not the transcript shape).

## Public-parameter changes

**None.** Phase 1.5's v2 pp (embedded DST + `{Gᵢ}, H, U`) carries
everything Phase 3 needs. The Σ-protocols use only `H` and `U`.

Phase 3 does *not* introduce a new DST or bump the pp version. If a
refactor ever changes how the Σ-protocol's auxiliary commitment `A` is
derived from generators, *that* would not affect the pp (it's prover-
side randomness, not parameters).

## Risks

1. **proof-of-product requirement unclear.** The §A.1 Figure 5 triple
   includes proof-of-product, but the sumcheck recursion itself is
   linear-in-commitments. If audit A3 confirms proof-of-opening +
   proof-of-equality suffice for Phase 3, we save ~1 day. If the audit
   flags a product check (e.g. for multi-hadamard's intra-round
   products), we add it. **Mitigation:** spawn A3 with an explicit
   question "does the sumcheck recursion need proof-of-product, or
   only -opening and -equality?"
2. **Per-round blinding bookkeeping.** Each round emits `d+1`
   blindings; the round-to-round fold derives new blindings
   linearly from public challenge weights. A single sign error or off-
   by-one in the fold silently breaks the commitment chain without
   triggering a verifier reject until the end (where it manifests as a
   Figure 6 failure at the final reduction — hard to localize).
   **Mitigation:** the round-commit component (2) has its own
   end-to-end roundtrip test independent of the sumcheck driver, and
   the driver test asserts the per-round proof-of-equality accepts
   *before* reaching the final Figure 6 step.
3. **Composition with Phase 2 `verify_zk`.** The final round's claim
   feeds directly into `open_zk` / `verify_zk`. The commitment carried
   in from the sumcheck must match the commitment that `verify_zk`
   expects as `τ`. Transcript-splice bugs here are the #2 footgun.
   **Mitigation:** dedicated composition test (#7 above) that treats
   the sumcheck output as an input to `verify_zk`.
4. **Transcript-size inflation.** Per round grows from `d+1` scalars
   to `d+1` group elements + one Σ-protocol response. At ~48 bytes per
   G1 element + 32 bytes per Σ-response (3 scalars) vs 32 bytes per
   scalar, the per-round size roughly quadruples. At `n = 15` rounds
   the sumcheck transcript grows from ~480 bytes/poly to ~2.5 KB/poly.
   Still small vs the Figure 6 finalization (~1 MB). **Mitigation:**
   track total proof-size delta as an acceptance criterion; flag if it
   crosses 2× the Phase 2 baseline.
5. **Kernel reuse risk.** `Fr_ip_sc_step` et al. were written for the
   plain path and assume the caller doesn't care about intermediate
   state past the emitted `out_k`. Phase 3 additionally needs the
   *blindings* for each `out_k`, but those are prover-local, not GPU-
   resident. No kernel change expected. **Mitigation:** confirm during
   subcomponent (3) — if a kernel forces a refactor, flag early.

## Audit checklist for A3 (fresh opus agent, no context)

1. For every commitment the prover sends (`T_0, {T_j^{(k)}}_{j,k},
   final τ`), confirm the verifier references it and every check
   equation involves it at least once.
2. For every blinding scalar (`ρ_j^{(k)}, s_m, s_r, …`), confirm a
   fresh `FrTensor::random` / equivalent draw per invocation. No
   across-round reuse. No across-call reuse.
3. For the per-round proof-of-equality, confirm the verifier
   recomputes `T_j^{(0)} + T_j^{(1)}` *from the prover's commitments*,
   not from a prover-claimed sum — otherwise the soundness check is
   vacuous.
4. Confirm the final-round handoff to `verify_zk` matches the
   commitment the sumcheck chain produced — no re-commitment, no
   freshness reset. (Phase 2's `τ` must equal the sumcheck's
   `T_n(r_n)`.)
5. **Settle proof-of-product:** is it needed for any of the four
   sumcheck variants, or do proof-of-opening + proof-of-equality
   suffice? If needed, name the variant and the specific check.
6. Confirm the degree-K multi-hadamard driver emits exactly `K+1`
   commitments per round (not `K` or `K+2`) and the Lagrange fold uses
   the matching weight count.
7. Confirm no plain-sumcheck function is still reachable from any
   migrated call site (grep for `inner_product_sumcheck(`,
   `hadamard_product_sumcheck(`, etc. outside the kept-for-reference
   plain definitions).
8. Confirm the simulator argument for the ZK sumcheck: Protocol 3's
   per-round transcript is simulatable given the challenges, by the
   same reduction as Figure 6 (uniform responses + forced
   commitments). Check that the implementation doesn't accidentally
   emit anything the simulator cannot produce (e.g. a prover-side
   random-oracle tag not under challenge control).

## What this does NOT unblock

- **Phase 4 (proof serialization):** blocks on Phase 3 output, but the
  on-disk format is cut in Phase 4, not here.
- **Phase 5 (verifier binary):** Phase 3 ships in-process verification
  entry points only; the standalone verifier is Phase 5's job.

## What this unblocks

- **Phase 4 (proof serialization):** can now finalize the
  `ZKSumcheckProof` wire format. The wire layout is a Phase 4 decision;
  Phase 3 only fixes the in-memory transcript struct.
- **Phase 5 (crypto verifier):** the CPU verifier needs to check the
  same Σ-protocol and Figure-6 equations Phase 3 establishes.
- Paper writing can now honestly say the entropy pipeline produces a
  fully-ZK transcript modulo Fiat-Shamir.

## Future work

- **Fiat-Shamir (post-Phase 5).** Replace the caller-supplied challenger
  with a transcript hash. Straightforward once the proof format is
  finalized.
- **Figure 7+8 log-n opening (Phase 2 future work).** Cuts the final-
  reduction cost; orthogonal to the Phase 3 construction.
- **Batched Σ-protocols.** All per-round proof-of-equalities share the
  same generator pair `(U, H)`; a batched variant reduces the `n`
  Σ-protocols to one at the cost of slightly larger responses. Worth
  considering once we see the Phase 3 transcript size in practice.

## References

- `docs/plans/phase-2-blinded-opening.md` — Phase 2 kickoff; ships the
  Figure 6 primitive Phase 3 finalizes with.
- `docs/plans/phase-1.5-hash-to-curve.md` — hash-to-curve generators
  that eliminate the known-dlog concern for Σ-protocol composition.
- `docs/plans/plan-production-readiness.md` — parent plan, Phase 3 row.
- **Wahby et al. 2018, "Doubly-efficient zkSNARKs without trusted setup"
  (Hyrax) — eprint 2017/1132.** Primary protocol reference:
  - §4 Protocol 3 (p. 11–13) — ZK sumcheck.
  - §A.1 Figure 5 (p. 17) — proof-of-opening / -equality / -product.
  - §A.2 Figure 6 (p. 18), Theorem 11 — the final-reduction primitive,
    shipped in Phase 2.
- **zkLLM §3.4** (Sun, Li, Zhang 2024) — names Hyrax as the PCS.
- `src/proof/proof.cu` — the plain-sumcheck variants Phase 3 replaces.
