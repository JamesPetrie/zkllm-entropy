# Phase 2 — Hiding (Blinded) Polynomial Opening

**Date:** 2026-04-15
**Status:** Draft / kickoff
**Branch:** `phase-2-blinded-opening` (to be created)
**Parent plan:** `docs/plans/plan-production-readiness.md`
**Estimated effort:** 1.5 weeks

## Source-material note

This phase implements the **Hyrax evaluation-proof protocol** — the ZK
polynomial-opening argument that tops the Pedersen commitment scheme
Phase 1 shipped. zkLLM (§3.4, p. 5) anchors its commitment scheme
directly to Hyrax:

> "In this study, Hyrax [54], a variant of the Pedersen commitment [43]
> that does not require a trusted setup, is used as an instantiation of
> the polynomial commitment scheme."
> — Sun, Li, Zhang 2024, *zkLLM: Zero Knowledge Proofs for Large
> Language Models*, §3.4.

The authoritative construction is **Wahby et al. 2018, "Doubly-efficient
zkSNARKs without trusted setup" (eprint 2017/1132)**. For a
multilinear-polynomial opening like ours, the relevant building blocks
are:

- **§6.1 — matrix-layout multilinear commitment.** Reduces a multilinear
  polynomial opening to a single dot-product claim against the verifier-
  computable vector `d⃗` derived from the challenge point `u⃗`.
- **§A.1 Figure 4 — Pedersen commitment scheme.** The group-element
  commitment primitive Phase 1 shipped.
- **§A.2 Figure 6 — `proof-of-dot-prod`.** The O(n) ZK dot-product
  argument. Theorem 11 (p. 18): *"The protocol of Figure 6 is complete,
  honest-verifier perfect zero-knowledge, and special sound under the
  discrete log assumption."* **This is the Phase 2 transcript.**
- **§A.3 Figures 7 + 8 — `proof_log-of-dot-prod`.** O(log n) variant
  obtained by wrapping Bulletproofs recursive reduction around Figure 6.
  *Not* implemented in Phase 2. Paper (§A.3, p. 19): *"The dot-product
  argument of Appendix A.2 has communication `4 + n` elements for a
  vector of length `n`. By adapting the Bulletproof recursive reduction
  of Bünz et al. [31], we reduce this to `4 + 2 log n`. Figures 7 and 8
  detail this protocol."* — Figures 7+8 are the future optimization
  path; see "Future work" at the bottom of this doc.

The Phase 2 target is the composition of §6.1 (row-wise commitments →
single ξ via the public `ẽq`-MLE weights) with §A.2 Figure 6 (O(n) ZK
dot-product). We deliberately pick Figure 6 over Figure 7+8:

- **Consistency.** Phase 3 (ZK sumcheck) also finalizes its per-round
  commitments via a dot-product proof. Using the same Figure 6
  transcript for opening and sumcheck finalization keeps the verifier
  and the simulator argument uniform.
- **Simplicity.** Figure 6 is one round of Σ-protocol over a size-n
  vector. No recursive folding, no inversion-per-round, no sign-
  convention traps in `bullet-reduce`. The implementation risk is
  dominated by blinding-freshness bugs, which we can test.
- **Transcript size is acceptable.** At `n = out_dim = 32000` for
  lm_head a single row opening emits ~32 k field elements — ~1 MB per
  row. Weight proofs already run into the tens-of-megabytes range; this
  is within the current operating budget. If proof size becomes the
  binding constraint we pivot to Figure 7+8 in a follow-up (see "Future
  work").

## Goal

Replace `Commitment::open` / `Commitment::me_open`
(`src/commit/commitment.cu:261-293`) — currently a non-ZK Bulletproofs-style
recursive halving that ends by sending the final scalar `t(0)` in the
clear — with a **blinded** evaluation proof that reveals nothing about
the committed polynomial beyond the public evaluation value `v = f̃(u)`.

The target construction is Hyrax's **O(n) ZK dot-product argument**
(§A.2 Figure 6), applied after the §6.1 matrix-layout reduction that
collapses a multi-row polynomial opening to a single dot-product claim.
Three ingredients:

1. **Pedersen blinding** on every commitment the prover emits — the
   per-row `C_i` from Phase 1, the scalar-target commitment
   `τ = g^v ⊙ h^{r_τ}`, and the mask commitments `δ, β` — using the
   hiding generator `h` that Phase 1 added.
2. **Mask vector `d⃗` of size n** sampled uniformly per opening, with
   fresh blindings `r_δ, r_β`. This is the single source of ZK in the
   transcript: every response the prover emits is `d⃗` plus a
   challenge-weighted witness contribution, statistically indistinguish-
   able from a fresh uniform draw.
3. **Σ-protocol responses** (Figure 6 step 3, checked by eqs 13 & 14):
   prover sends `z⃗ = c·x̂ + d⃗`, `z_δ = c·r_ξ + r_δ`, `z_β = c·r_τ + r_β`;
   verifier checks `ξ^c ⊙ δ = Com(z⃗; z_δ)` and
   `τ^c ⊙ β = Com(⟨z⃗, â⟩; z_β)`.

The new generator `g` (Figure 6 statement) — the one that carries the
scalar inner-product target `v` inside `τ = g^v ⊙ h^{r_τ}` — is **not**
in the Phase 1 pp. Phase 2 extends the pp format in a backwards-
compatible way to include it, shipped as a `.u` sidecar.

By the end of Phase 2, the prover's opening transcript is
zero-knowledge against a single verifier query, assuming discrete log
in BLS12-381 G1 is hard — matching the security story zkLLM asserts.

## Non-goals

- **ZK sumcheck.** The per-round sumcheck polynomial is still sent in
  the clear. Phase 3 addresses that (Hyrax §4 per-round commitments
  + §A.2 Figure 6 finalization, reusing the exact primitive this phase
  ships). Phase 2 only fixes the *opening* of already-sent commitments.
- **Square-root matrix reduction.** The existing code in this repo
  commits weights row-by-row (`create_weight` produces `in_dim` rows
  each of width `out_dim`); that is already a square-root-style layout
  matching Hyrax §6.1. Phase 2 does not rewrite that decomposition —
  it only blinds each row-opening and the post-reduction inner-product
  argument that follows it.
- **O(log n) transcript.** Figure 7+8 (`bullet-reduce` +
  `proof_log-of-dot-prod`) is deferred. See "Future work" below.
- **Hash-to-curve for U.** Same treatment as H in Phase 1: U comes from
  the pp sampler and trusts the ppgen operator. Production hardening is
  out of scope, tracked alongside H's hash-to-curve follow-up.
- **Fiat-Shamir.** Interactive. Verifier challenges come from a caller-
  supplied oracle; no hash-based derivation yet.
- **Verifier binary.** Phase 5 builds the standalone verifier. Phase 2
  only ships a `verify` helper callable from the test harness so that
  negative tests can run end-to-end.

## Construction (Hyrax §6.1 reduction + §A.2 Figure 6)

Notation below follows the Hyrax paper. For clarity this doc writes the
group operation multiplicatively (`⊙`) and exponentiation as group
scalar-multiplication, matching the paper. In our CUDA code the group
is additive (BLS12-381 G1 Jacobian), so `h^{r}` in the paper becomes
`r·H` in code, and `Π gᵢ^{xᵢ}` becomes `Σ xᵢ·Gᵢ`.

### Public parameters

Required generators, all in `G1`:

- `g⃗ = (g_0, …, g_{N-1})` — witness generators (Phase 1, in the pp
  file).
- `h` — hiding generator (Phase 1, in the `.h` sidecar).
- `g` — scalar generator for the inner-product-target commitment `τ`
  (Figure 6 statement). **New in Phase 2; shipped as `.u` sidecar.**

In code we keep the existing `hiding_generator` name for `h` and add
`u_generator` for `g`. (Calling it `u_generator` avoids collision with
the `G_i` generator-vector variable names and with the `g` symbol that
`Commitment::random` already uses as the base point when sampling.)

### Prover inputs

- Witness `x⃗ ∈ F^N` (the committed polynomial's coefficient vector,
  padded to `N = 2^n`).
- Row-wise blindings `{r_{ξ,i}}` from Phase 1 (the `.r` sidecars of the
  row commitments).
- Public evaluation point `u⃗ ∈ F^n`, public value `v = f̃(u⃗)`.

### Step 1 — §6.1 matrix-layout reduction (pre-Figure-6)

Hyrax's multilinear PCS (§6.1) commits the witness as a `ℓ × h` matrix
with per-row Pedersen commitments `C_0, …, C_{h-1}` — this matches what
Phase 1 ships today (one row per `in_dim` slot of a weight).

For challenge `u⃗ = (u_L ‖ u_R)` with `|u_L| = log ℓ`, `|u_R| = log h`,
the verifier can homomorphically compute the reduced commitment

    ξ  ←  Σ_i  ẽq(bits(i), u_R) · C_i           (verifier-computable)

where `ẽq(bits(i), u_R)` is the standard equality-MLE weight. The
prover's matching witness is the folded vector `x̂ = M_x · e⃗` where
`e⃗_i = ẽq(bits(i), u_R)`. The row-blinding also folds homomorphically:
`r_ξ = Σ_i ẽq(bits(i), u_R) · r_{ξ,i}` (the per-row blindings from
Phase 1). The dot-product claim to discharge is

    ⟨x̂, â⟩ = v    with    â_j = ẽq(bits(j), u_L)

— `â` is also verifier-computable. This reduces the multilinear-
polynomial opening to a single ZK-dot-product instance `(ξ, τ, â)` with
`τ = g^v ⊙ h^{r_τ}` for fresh prover blinding `r_τ`.

### Step 2 — §A.2 Figure 6 `proof-of-dot-prod(ξ, τ, â)`

Inputs: `ξ = Com_{g⃗}(x̂; r_ξ)`, `τ = Com(v; r_τ)`, vector `â`.
Prover knows `x̂, r_ξ, r_τ` and the value `v = ⟨x̂, â⟩`.

**Step (1) of Figure 6 — P → V (one round, two group elements).**
Prover samples `d⃗ ∈ F^n` and `r_δ, r_β ∈ F` uniformly and sends

    δ  ←  Com_{g⃗}(d⃗; r_δ)     =  h^{r_δ} ⊙ Π gᵢ^{dᵢ}          (eq 11)
    β  ←  Com(⟨â, d⃗⟩; r_β)    =  g^{⟨â, d⃗⟩} ⊙ h^{r_β}        (eq 12)

*(Hyrax paper Figure 6, step 1, eqs 11 & 12, p. 18.)*

**Step (2) of Figure 6 — V → P.** Verifier sends challenge `c ∈ F`.

**Step (3) of Figure 6 — P → V (n + 2 field elements).** Prover sends

    z⃗    ←  c · x̂ + d⃗                 ∈ F^n
    z_δ  ←  c · r_ξ + r_δ              ∈ F
    z_β  ←  c · r_τ + r_β              ∈ F

*(Hyrax paper Figure 6, step 3.)*

**Step (4) of Figure 6 — V check, eqs 13 & 14, p. 18.**

    ξ^c ⊙ δ   =?   Com_{g⃗}(z⃗; z_δ)   =   h^{z_δ} ⊙ Π gᵢ^{zᵢ}      (eq 13)
    τ^c ⊙ β   =?   Com(⟨z⃗, â⟩; z_β)  =   g^{⟨z⃗, â⟩} ⊙ h^{z_β}    (eq 14)

If both equations hold (and the verifier has correctly computed `ξ`
from `{C_i}` and `â` from `u⃗`), the opening is accepted.

### Communication summary

Per opening Figure 6 emits 2 group elements (`δ, β`) plus `n + 2` field
elements (`z⃗, z_δ, z_β`), matching the paper's O(n) claim (§A.3, p. 19:
*"has communication `4 + n` elements for a vector of length `n`"*).

For our lm_head case (`N = 2^15 = 32768`, so `n = 15` after §6.1
reduction — the row-reduction collapses the `in_dim` axis into a single
folded row of length `out_dim`, and `out_dim = 32000 ≤ 2^15 = 32768`):

- **After §6.1 reduction**, `n = log₂(32768) = 15` is the *log-size of
  the folded row*, not the vector length going into Figure 6.
- **The Figure-6 vector is length `2^n = 32768`** — the full folded row
  `x̂`. So the transcript is 2 group + 32 770 field elements.

At ~32 bytes/field on BLS12-381 Fr, that's ~1 MB per opening. There are
six `verifyWeightClaim` call sites in the entropy pipeline, so the
weight-opening contribution to proof size is ~6 MB. Small vs the Phase 3
sumcheck proof that dominates total size; not small enough to ignore.

## Public-parameter file-format change

Phase 1 shipped:
- `<pp>`: `G₀, …, G_{size-1}` (legacy `G1TensorJacobian::save` format).
- `<pp>.h`: `H` (1 × `G1Jacobian_t`).

Phase 2 adds:
- `<pp>.u`: `U` (1 × `G1Jacobian_t`).

`Commitment::load_hiding` learns the `.u` sidecar. Load path semantics:

| `.h` present | `.u` present | Result |
|---|---|---|
| no | no | legacy non-hiding (unchanged) |
| yes | no | **rejected during Phase 2 open**; Phase 1 commit ops still work |
| yes | yes | hiding + openable |

The `.u`-missing-but-`.h`-present state is a transitional artefact from
Phase 1 pps generated before Phase 2 lands. Phase 2's `open` throws
loudly rather than silently degrading.

## Code changes

### Files to modify

1. **`src/commit/commitment.cuh`** — add `G1Jacobian_t u_generator` to
   `Commitment`; add `is_openable()` (returns `is_hiding() && u_generator`
   non-identity); declare new `save_openable(path)` /
   `load_openable(path)` or fold into `save_hiding` / `load_hiding` with
   the `.u` sidecar treated as optional-on-save, required-on-open.

2. **`src/commit/commitment.cu`** — sample `U` inside
   `hiding_random()`; write/read `.u` sidecar. Add the new opening
   primitives (see "New primitives" below). `me_open` / `open` remain
   as **legacy** entry points (untouched); new entry points are
   `open_zk` / `verify_zk`.

3. **`bin/ppgen.cu`** — ensure `U` is written. The current ppgen calls
   `Commitment::hiding_random` and `save_hiding`; touching those puts
   `U` in the pp automatically.

4. **`src/proof/proof.cu` — `verifyWeightClaim`** — new blinded overload
   consuming the new `OpeningProof` struct. The existing overload stays
   in place as the unblinded legacy path until Phase 3 retires it.

5. **Call sites of `verifyWeightClaim`** (6 sites surveyed in Phase 1):
   - `src/llm/rmsnorm.cu:49`
   - `src/llm/self-attn.cu:57-59`
   - `src/llm/ffn.cu:76-85`
   - `bin/zkllm_entropy.cu:174, 182`
   - `bin/zkllm_entropy_timed.cu:169, 180`
   Each passes `w.r` (already wired in Phase 1 step 3) and now also
   receives and stores/checks the new `OpeningProof`.

### New primitives

```cpp
// src/commit/commitment.cuh
//
// Field names map to Hyrax 2017/1132 Figure 6 verbatim — this
// collapses a whole class of transcription bugs at code-review time.
struct OpeningProof {
    // Figure 6 step 1 (P → V), eqs 11 & 12: two group elements.
    G1Jacobian_t delta;       // δ = h^{r_δ} ⊙ Π gᵢ^{dᵢ}
    G1Jacobian_t beta;        // β = g^{⟨â,d⃗⟩} ⊙ h^{r_β}

    // Figure 6 step 3 (P → V): n + 2 field elements.
    FrTensor     z;           // z⃗ = c·x̂ + d⃗   (length n)
    Fr_t         z_delta;     // z_δ = c·r_ξ + r_δ
    Fr_t         z_beta;      // z_β = c·r_τ + r_β

    // Prover also emits the commitment τ (binds v with fresh r_τ) once
    // per opening; ξ is verifier-computable from {C_i} via eq-MLE.
    G1Jacobian_t tau;         // τ = g^v ⊙ h^{r_τ}
};

struct OpeningChallenge {
    Fr_t c;                   // Figure 6 step 2
};

// Prover side — produces transcript. Consumes w.r blindings from Phase 1.
OpeningProof Commitment::open_zk(
    const FrTensor& t,                    // padded polynomial
    const FrTensor& row_blindings,        // r_{ξ,i}  (w.r from Phase 1)
    const G1TensorJacobian& com,          // row commitments C_0 … C_{h-1}
    const std::vector<Fr_t>& u,
    const OpeningChallenge& chal          // verifier-supplied challenge
) const;

// Verifier side — standalone check. Returns true iff eqs 13 & 14 both hold.
bool Commitment::verify_zk(
    const G1TensorJacobian& com,
    const std::vector<Fr_t>& u,
    Fr_t v,
    const OpeningProof& proof,
    const OpeningChallenge& chal
) const;
```

The challenge-injection model matches the rest of the codebase: the
caller (test, higher-level prover) drives challenges. Fiat-Shamir
derivation is Phase 5+.

### Why not rewrite `me_open` in place

`me_open`'s per-round emission is the unblinded ancestor of a
Bulletproofs-style recursive halving — a shape Phase 2 is moving *away
from* toward a single-round O(n) Σ-protocol. Ripping the new primitive
into the old recursion's shape would fight the code's natural flow.
Cleaner to introduce `open_zk` alongside, migrate Weight openings over
one call site at a time, then delete `me_open` once the last legacy
caller is gone.

### Call-site inventory (authoritative)

| Site | File:line | Change |
|---|---|---|
| `verifyWeightClaim` — legacy path | `src/proof/proof.cu:12` (as of Phase 1 state) | Add ZK overload; keep legacy for now |
| `verifyWeightClaim` — rmsnorm | `src/llm/rmsnorm.cu:49` | Switch to ZK path after `OpeningProof` wired |
| `verifyWeightClaim` — self-attn Q/K/V | `src/llm/self-attn.cu:57-59` | Ditto |
| `verifyWeightClaim` — ffn up/gate/down | `src/llm/ffn.cu:76-85` | Ditto |
| `verifyWeightClaim` — final_norm, lm_head | `bin/zkllm_entropy.cu:174,182`, `bin/zkllm_entropy_timed.cu:169,180` | Ditto |
| `ppgen` | `bin/ppgen.cu` | Shipped via `hiding_random` change; no caller edit |
| `load_hiding` sites | `bin/commit-param.cu:30`, `bin/commit_logits.cu:33`, tests | Rejects pp lacking `.u` when `open_zk` is called; commit-only paths unchanged |

Total: **~8 call sites**, 5 files for core wiring + 2 binaries.

## Tests

### Positive

1. **`test_opening_roundtrip`**: commit a random tensor (hiding);
   produce an `OpeningProof` at a random `u`; verify passes; claimed
   `v` matches `f̃(u)`.
2. **`test_opening_small`**: 4-element tensor (n = 2); inspect all
   transcript elements are present and lengths match (`|z| = 2^n = 4`).
3. **`test_opening_multi_row`**: multi-row weight (`in_dim > 1`,
   `out_dim = 4`); every row's opening verifies. Exercises the §6.1
   row-reduction `ξ = Σ ẽq(bits(i), u_R)·C_i` path and the homomorphic
   blinding fold `r_ξ = Σ ẽq · r_{ξ,i}`.
4. **`test_opening_challenges_deterministic`**: given fixed
   `(t, u, c)`, proof is byte-identical across runs *modulo* the fresh
   blinding scalars (`r_τ, d⃗, r_δ, r_β`). Catches accidental
   derandomization of the blinding path.
5. **`test_opening_pp_roundtrip`**: save pp with `.u`, reload, confirm
   `u_generator` byte-exactly round-trips.

### Negative

6. **`test_opening_tampered_delta_rejects`**: alter `δ`; verifier
   rejects at eq (13).
7. **`test_opening_tampered_beta_rejects`**: alter `β`; verifier
   rejects at eq (14).
8. **`test_opening_tampered_z_rejects`**: flip one entry of `z⃗`;
   verifier rejects (both eq (13) and eq (14) become inconsistent).
9. **`test_opening_tampered_z_delta_rejects`**: alter `z_δ`; verifier
   rejects at eq (13).
10. **`test_opening_tampered_z_beta_rejects`**: alter `z_β`; verifier
    rejects at eq (14).
11. **`test_opening_tampered_tau_rejects`**: alter `τ`; verifier rejects
    (eq 14's `τ^c ⊙ β` side no longer matches the response side).
12. **`test_opening_wrong_v_rejects`**: prover claims `v' ≠ f̃(u)`;
    verifier rejects (τ commits wrong value, eq 14 fails).
13. **`test_opening_replayed_blinding_rejects`**: prover reuses the
    same `d⃗` (and same `r_δ, r_β`) in two separate openings of the
    same commitment; target check: distinguisher test (#15) trips,
    *not* the verifier (verifier has no way to know — this is a hiding
    break, not a soundness break).
14. **`test_opening_missing_u_throws`**: load a Phase-1 pp that has
    `.h` but lacks `.u`; calling `open_zk` throws with a clear message.

### Hiding property (statistical)

15. **`test_opening_distinguisher`**: fix a pp and a commitment to a
    known `t`; run 5 000 openings at the *same* `u` with fresh
    challenges (and thus fresh blindings). Every transcript element
    (`τ, δ, β, z⃗, z_δ, z_β`) should have distribution independent of
    `t`. Apply a χ² gate projected to a fixed `F_r` hash of each group
    element and a sample of `z⃗` entries, flagging any RNG regression
    that collapses blinding.

This is the robustness test that would catch a future refactor that
accidentally zeroes out `d⃗, r_δ, r_β, r_τ` — soundness-preserving
(verifier still accepts) but hiding-destroying.

### Regression

16. All Phase 1 hiding-Pedersen tests still pass with the `.u` sidecar
    added.
17. `test_zkargmax`, `test_zklog`, `test_zknormalcdf`, `test_zkentropy`:
    unchanged (they don't open via `open_zk` yet; Phase 2 keeps the
    legacy path until the last caller is migrated).
18. End-to-end `zkllm_entropy` run: all six `verifyWeightClaim` calls
    produce valid `OpeningProof`s that round-trip through verify.

## Simulator argument (informal)

Theorem 11 (Hyrax paper, §A.2 p. 18):

> "The protocol of Figure 6 is complete, honest-verifier perfect
> zero-knowledge, and special sound under the discrete log assumption."
> — Wahby et al. 2017/1132, Theorem 11, p. 18.

The paper's simulator sketch (p. 18, just before Figure 6) —
reproduced here against our code's field names:

1. Sample `z⃗', z_δ', z_β' ∈ F` uniformly.
2. Given the verifier's challenge `c`, compute `δ` and `β` by inverting
   the check equations (13) and (14):

       δ   =   (h^{z_δ'} ⊙ Π gᵢ^{zᵢ'}) ⊙ ξ^{-c}              (eq 15)
       β   =   (g^{⟨z⃗', â⟩} ⊙ h^{z_β'}) ⊙ τ^{-c}             (eq 16)

3. Also sample `r_τ ∈ F` uniformly and set `τ = g^v ⊙ h^{r_τ}` using
   the public `v`.
4. Emit the transcript `(τ, δ, β, c, z⃗', z_δ', z_β')`.

The simulator's output is distributed identically to the honest
prover's transcript: in the honest run, `z⃗, z_δ, z_β` are the sum of
the prover's fresh mask `(d⃗, r_δ, r_β)` and a challenge-weighted
witness contribution, and the mask terms make each of them a uniform
draw; `τ` is uniform because `r_τ` is; `δ` and `β` are determined by
eqs (13) and (14) in both honest and simulated runs. Every transcript
element is either sampled freely or forced by eqs (13) & (14), and the
simulator matches that same constraint pattern.

Making this a malicious-verifier (rewinding-free) result requires
Fiat-Shamir or a straight-line simulator, both out of scope for
Phase 2.

## Risks

1. **Transcript emission mismatch.** Prover and verifier disagree on
   the order of `δ, β, z⃗, z_δ, z_β`. Mitigation: single `OpeningProof`
   struct with fixed field order; serialization is struct-level, not
   field-by-field.
2. **Subtle blinding reuse.** Sampling `d⃗` once per `Commitment`
   instance instead of per `open_zk` call is the #1 footgun — a single
   reused `d⃗` across two openings at the same `(ξ, â, c)` reveals
   `x̂`. Mitigation: distinguisher test (#15) gates merges; auditor
   specifically looks for "is every blinding scalar a fresh draw per
   invocation?" (see audit checklist).
3. **Homomorphic row-blinding fold.** Under the §6.1 reduction the
   opening uses `r_ξ = Σ ẽq(bits(i), u_R) · r_{ξ,i}`. A bug in the
   fold (wrong `ẽq` weights, wrong combination order) silently breaks
   eq (13) when the verifier recomputes `ξ`. Mitigation: dedicated
   multi-row positive test (#3) plus a stand-alone unit test on the
   fold formula.
4. **Transcript size at large n.** O(n) communication means a
   32 k-vector opening is ~1 MB per call. Mitigation: acceptance
   criterion on total proof size is tracked; if it exceeds the budget
   the fallback is Figure 7+8 (see "Future work"). Not expected to
   bind in Phase 2.
5. **GPU memory for `d⃗` and `z⃗`.** Both are FrTensors of length
   `n = out_dim`. Fits comfortably on H100.

## Acceptance criteria

- [x] Cross-reference the protocol against Wahby et al. eprint 2017/1132
      (done 2026-04-15; target is §A.2 Figure 6).
- [ ] All positive tests (#1–5, #16, #18) pass.
- [ ] All negative tests (#6–14) reject.
- [ ] Distinguisher test (#15) passes.
- [ ] `zkllm_entropy` end-to-end run produces a proof transcript where
      all six `verifyWeightClaim` sites carry a valid `OpeningProof`.
- [ ] Prover time at `in_dim × out_dim = 4096 × 32000` (lm_head) is
      within 2× the current non-ZK `verifyWeightClaim` wall time.
      Rationale: Figure 6 is a single multi-scalar-multiplication of
      length `n` (for `δ`) plus two dot-products (for `β` and `z⃗`),
      comparable to what the current unblinded path already does.
- [ ] Total proof-size delta at the entropy pipeline is within budget
      (~10 MB additional across all six openings). If this is
      exceeded, file a Phase 7 ticket to pivot to Figure 7+8.
- [ ] Independent audit A2 finds no gaps on transcript completeness,
      blinding-freshness, and simulator-argument consistency.

## Audit checklist for A2 (fresh opus agent, no context)

1. For every prover emission in `open_zk` (`τ, δ, β, z⃗, z_δ, z_β`),
   confirm (a) the verifier references it and (b) it matches Hyrax
   Figure 6 step 1 (eqs 11, 12) and step 3. Walk line-by-line.
2. For every blinding scalar (`r_τ, d⃗, r_δ, r_β`), confirm a fresh
   `FrTensor::random` draw per `open_zk` call and no reuse across
   invocations or call sites. `d⃗` is length-n; confirm each coordinate
   is drawn fresh (not one scalar broadcast).
3. Confirm the verifier recomputes `ξ` as
   `Σ_i ẽq(bits(i), u_R) · C_i` from the committed `{C_i}` (Phase 1
   commitments), not taken on the prover's word. `â` is likewise
   recomputed from `u⃗` alone.
4. Confirm the verifier checks both eq (13) *and* eq (14). A
   plausible bug is checking only one; eq (13) alone does not bind
   the value `v`.
5. Confirm `.u` sidecar is mandatory for `open_zk`; a Phase-1-only pp
   is rejected at load, not silently zeroed.
6. Confirm the simulator argument in this doc matches what the code
   actually emits — if the code uses a different variable order or
   computes `β` differently from eq (16), the simulator argument is
   wrong.
7. Confirm `z⃗ = c·x̂ + d⃗` (not `d⃗ + c·x̂` with some sign swap, and
   not `c·d⃗ + x̂`). The paper is explicit (Figure 6 step 3); the
   symmetry makes it easy to mis-transcribe.

## What unblocks after Phase 2

- Phase 3 (Hyrax §4 ZK sumcheck): the per-round sumcheck commitments
  are finalized via the same Figure 6 primitive this phase ships.
  Direct reuse — no new opening protocol.
- Phase 4 (proof serialization): `OpeningProof` layout must be
  finalized before the on-disk proof format is cut.
- Phase 5 (crypto verifier): implements `verify_zk` in the C++/Rust
  standalone verifier binary; depends on this doc's acceptance criteria
  being settled.

## Future work — Figure 7+8 (log-n opening)

If proof size becomes the binding constraint, the next step is Hyrax
§A.3 Figure 7 + Figure 8 (`proof_log-of-dot-prod` + `bullet-reduce`),
which cuts opening communication from `n + 4` elements to `2 log n + 4`.
At `n = 32768` that's 30 group elements + 2 field elements per opening
vs the current 2 + 32 770 — a ~1000× reduction in transcript size.

Lemma 12 (p. 20): *"The protocol of Figures 7–8 is complete, honest-
verifier perfect ZK, and generalized special sound under the discrete
log assumption."* — i.e. the same security properties as Figure 6.

Costs of the pivot: recursive folding with per-round blinding
accumulation, a separate final Σ-protocol (δ, β, z_1, z_2), sign-
convention care in `bullet-reduce` step 4. Worth doing if/when proof
size is the bottleneck; not worth doing pre-emptively when Figure 6
gets us a simpler transcript and a matching Phase 3 finalizer.

## References

- `docs/plans/phase-1-hiding-pedersen.md` — previous phase (the
  commitment Phase 2 opens).
- `docs/plans/plan-production-readiness.md` — parent plan; Phase 2 row.
- **Wahby et al. 2018, "Doubly-efficient zkSNARKs without trusted
  setup" (Hyrax) — eprint 2017/1132.** Primary protocol reference.
  Relevant sections: §6.1 (matrix-layout multilinear commitment, p. 9);
  §A.1 Figure 4 (Pedersen, p. 17); §A.2 Figure 6 (`proof-of-dot-prod`,
  O(n), p. 18) — **the transcript this phase implements**; Theorem 11
  (p. 18) — completeness/ZK/soundness of Figure 6; §A.3 Figures 7 + 8
  (`proof_log-of-dot-prod` + `bullet-reduce`, pp. 19–20) and Lemma 12
  (p. 20) — deferred log-n optimization, see "Future work".
- **zkLLM §3.4** (Sun, Li, Zhang 2024) — names Hyrax as the PCS
  instantiation, motivating the protocol choice here.
- Bünz et al. 2018, "Bulletproofs" [31 in Hyrax's bibliography] —
  §4.2 recursive halving. The historical ancestor of Hyrax's
  Figure 7+8 optimization path; not used in Phase 2 itself.
