# Phase 1 — Hiding Pedersen Commitment

**Date:** 2026-04-15
**Status:** In progress (kickoff)
**Branch:** `phase-1-hiding-pedersen`
**Parent plan:** `docs/plans/plan-production-readiness.md`
**Estimated effort:** 2 weeks

## Goal

Replace every Pedersen commitment on the BLS12-381 path from
`C = Σᵢ tᵢ · Gᵢ` (binding only) with
`C = Σᵢ tᵢ · Gᵢ + r · H` (binding + hiding), where:

> **Hyrax §3.1 Commitment schemes (p. 4):** "Informally, a commitment
> scheme allows a sender to produce a message C = Com(m) that hides m
> from a receiver but binds the sender to the value m. The sender
> reveals m, or equivalently may be convinced that this was indeed
> the sender's original value. We say that Com_pp(m; r) is a
> commitment to the message m with randomness r, and sometimes do
> the same for the opening, e.g., Com(m, r)."

> **Hyrax §3.1 Perfect hiding definition (p. 4):** "Perfect hiding:
> For any m_0, m_1 ∈ {0, 1}^λ and m_0 ≠ m_1: {Com(m_0; r)}_{r←R} and
> {Com(m_1; r)}_{r←R} are identically distributed."

> **Hyrax §3.1 (p. 4):** "We define only the computational variant of
> binding and the perfect variant of hiding because the commitment
> schemes used in our implementation satisfy these properties."

Our construction instantiates `Com` with Pedersen (§A.1 Fig. 5,
Pedersen 1991 [85] in Hyrax's bibliography). The phase-1 commit
function matches `Com(t; r) = Σᵢ tᵢ Gᵢ + r H` with `r` sampled uniformly
per commitment.

- `H ∈ G1` is an independent generator (no known discrete log
  relative to any `Gᵢ`),
- `r ∈ F_r` is sampled uniformly at random per commitment,
- `(commitment, r)` is returned from `commit` and threaded through every
  downstream consumer (storage of weights, opening proofs, claim
  verification).

By the end of Phase 1 the codebase still uses *plain* (non-ZK) sumcheck
and *unblinded* openings — hiding at the commitment level alone does
**not** give ZK. Phase 1 is a *prerequisite* for Phase 2 (blinded
opening) and Phase 3 (Hyrax §4 ZK sumcheck), which together deliver the
full hiding property.

## Anchor quote for the construction

The construction below implements Pedersen commitments as used by
Hyrax's Appendix A.1 Σ-protocols:

> **Hyrax §A.1 proof-of-opening (p. 20), Figure 5:** The prover holds
> `x, r` such that `C = g^x · h^r` (multiplicative notation; our
> additive form is `C = x · G + r · H`). The prover sends a "blinded"
> commitment `A = g^{x'} · h^{r'}` to random `x', r'`; the verifier
> replies with a challenge `c`; the prover opens `z_1 = x' + c · x`,
> `z_2 = r' + c · r`; and the verifier checks
> `g^{z_1} · h^{z_2} = A · C^c`.

Phase 1 only introduces the commitment; Phases 2 and 3 will build the
Σ-protocols on top of it that Hyrax §A describes.

## Non-goals for this phase

- No changes to sumcheck (still non-ZK).
- No changes to `me_open` / `open` (still non-hiding; callers pass `r`
  through but the opening transcript doesn't yet blind round outputs).
- No verifier changes. `verify_entropy.py` remains arithmetic-only; it
  isn't touched. The C++ verifier doesn't exist yet.

## Generator `H` derivation

**Approach: one more random-scalar generator, written into pp.**

`H = s_H · G` where `s_H ← F_r` is sampled once inside `ppgen`
alongside the `Gᵢ = sᵢ · G` generators that `Commitment::random`
already samples. `H` is saved into the pp file next to the `Gᵢ`.

**Trust model.** This matches the existing trust model of the
codebase: whoever runs `ppgen` knows all pairwise discrete logs
among `{Gᵢ}`, and now also knows `dlog_G(H)`. That knowledge does
**not** affect the hiding property (which is against the verifier,
who never sees any scalar). It does affect binding: an adversary
who runs `ppgen` themselves could equivocate on commitments they
produce. Binding against a prover who also ran `ppgen` is therefore
assumed at the setup level — same assumption as the existing
zkLLM paper (§3.4 "trusted public parameters"), carried over
unchanged.

**Independence.** `H` is independent of `{Gᵢ}` in the sense that
matters for perfect hiding: `s_H` is an independent fresh uniform
draw from `F_r`. `H` may land in the linear span of `{Gᵢ}` with
probability `N/|F_r|` ≈ 2^(-245), which is a cryptographic
non-event on BLS12-381.

**Sanity check.** After generation, verify `H` is not the identity
(defensive, cheap).

**Alternative considered and rejected:** RFC 9380 hash-to-curve
(SHA-256 XMD + SSWU + isogeny) would give a publicly verifiable `H`
independent of any pp ceremony. That's the right answer for a
production deployment that wants to remove the trusted setup. It's
out of scope for Phase 1 because (a) no hash-to-curve exists in
this codebase (zkLLM's BLS12-381 implementation is custom, not
`blst`/`blstrs`), (b) implementing SSWU+isogeny is ~400 lines of
new crypto code plus RFC-vector test harness, and (c) it wouldn't
materially change the security story while the `Gᵢ` themselves
stay trusted-setup-generated. Flag for future work alongside a
full setup-ceremony redesign.

## Public-parameter file format change

Current `ppgen` format: a `Commitment` (i.e. `G1TensorJacobian`) of
`size` generators, saved via `G1TensorJacobian::save`. No header.

**New format, version 1:**

```
magic     : 8 bytes  = "ZKEPP\x00v1"
version   : uint32   = 1
flags     : uint32   = bit 0 set ⇒ hiding (contains H)
size      : uint32   = number of G_i generators
G_i       : size * sizeof(G1Jacobian_t)
H         : sizeof(G1Jacobian_t)
```

Backwards-compat read path: if the first 8 bytes don't match the new
magic, fall back to the legacy `G1TensorJacobian::load` path and treat
the pp as **non-hiding** (emits a warning; Phase 1 tests reject this
mode). This lets us keep existing test fixtures working while we
migrate.

## Code changes

### Files to modify

1. **`src/commit/commitment.cuh`**
   - Add `G1Jacobian_t hiding_generator` field to `Commitment`.
   - Change `commit` / `commit_int` / `commit_int_multi` return type to a
     new struct `HidingCommitment { G1TensorJacobian com; FrTensor r; }`.
   - Extend `Weight` to carry `FrTensor r` alongside `com`.
   - Add `Commitment::hiding_random(uint size)` factory.
   - Add save/load helpers for the new pp format.

2. **`src/commit/commitment.cu`**
   - Implement `commit` returning `(C, r)`: sample `r ← F_r` via
     `FrTensor::random`, compute `C = rowwise_sum(...) + r · H`
     (broadcasting the appropriate `r` per committed row).
   - Same treatment for `commit_int` and `commit_int_multi`.
   - `me_open` and `open` signatures unchanged in Phase 1. They take
     the extra `r` via a new optional parameter (defaulted for compat
     during migration) and fold it into the final-round check. Full
     blinding of the transcript is Phase 2.

3. **`src/proof/proof.cu`**
   - `verifyWeightClaim` reads `w.r` and passes it to `open`.

4. **`bin/ppgen.cu`**
   - Emit the new pp format. Derive `H` via hash-to-curve on the DST.
   - Add `--legacy` flag for regression testing.

5. **`bin/commit-param.cu`**
   - `commit_int` now returns `(com, r)`. Save both:
     `<output>.com.bin` and `<output>.r.bin`.
   - Update sibling Python scripts (`python/commit_final_layers.py`
     etc.) to read/copy both files.

6. **`bin/commit_logits.cu`**
   - Same as above: save `logits.com.bin` + `logits.r.bin`.

7. **`src/commit/commitment.cu` → `create_weight`**
   - New signature:
     `create_weight(gen_file, weight_file, com_file, r_file, in_dim, out_dim)`.
   - Backwards-compat shim: if `r_file` is missing, load zero-blinding
     (emits a warning; disallowed when pp has hiding bit set).

8. **`bin/zkllm_entropy.cu`, `bin/zkllm_entropy_timed.cu`**
   - Update the two `create_weight` sites per file (final_norm, lm_head)
     to pass the new r-file path. Paths:
     `final_norm.weight-blinding.bin`, `lm_head-weight-blinding.bin`.

### Call-site inventory (authoritative)

| Site | File:line | Change |
|---|---|---|
| `Commitment::random` → `hiding_random` | `bin/ppgen.cu:8` | Add H derivation |
| `generator.commit_int` | `bin/commit-param.cu:24` | Capture + save `r` |
| `generators.commit` | `bin/commit_logits.cu:54` | Capture + save `r` |
| `create_weight` — final_norm | `bin/zkllm_entropy.cu:100`, `_timed.cu:88` | Add r-file arg |
| `create_weight` — lm_head | `bin/zkllm_entropy.cu:107`, `_timed.cu:94` | Add r-file arg |
| `create_weight` — rmsnorm | `src/llm/rmsnorm.cu:19` | Add r-file arg |
| `create_weight` — self-attn Q/K/V | `src/llm/self-attn.cu:21,26,31` | Add r-file arg |
| `create_weight` — ffn up/gate/down | `src/llm/ffn.cu:20,25,30` | Add r-file arg |
| `verifyWeightClaim` (6 sites) | `rmsnorm.cu:49`, `self-attn.cu:57-59`, `ffn.cu:76-85`, `zkllm_entropy.cu:174,182`, `_timed.cu:169,180` | No direct change; gets `r` via `w.r` |
| `Commitment::open` | `src/proof/proof.cu:12` | Thread `r` through |
| `Commitment::me_open` | `src/commit/commitment.cu:109-128` | Thread `r` through |

Total: **~13 call sites**, 8 files for core wiring + 2 binaries. Scope
is tighter than the parent plan's ~15 estimate.

## Tests

### Positive

1. **`test_hiding_pedersen_basic`**: commit the same value twice with
   different `r`; commitments differ. Verify a well-formed
   `(commitment, r)` pair opens correctly.
2. **`test_hiding_pedersen_zero_r`**: `r = 0` reproduces the
   non-hiding commitment (bit-identical). Regression guard that the
   addition of `r · H` is the only change.
3. **`test_hiding_pp_roundtrip`**: write pp with new format, re-read,
   check magic / version / H / G_i match.
4. **`test_hiding_legacy_read`**: loading an old-format pp emits the
   warning path and returns a non-hiding commitment.
5. **Existing regression**: all of
   `test_zkargmax`, `test_zklog`, `test_zknormalcdf`, `test_zkentropy`
   still pass after the migration.

### Negative

6. **`test_hiding_wrong_r`**: opening with `r' ≠ r` fails
   `verifyWeightClaim` (the `opening != c.claim` assertion trips).
7. **`test_hiding_missing_r_rejects`**: loading a hiding-pp and then
   calling `create_weight` without an r-file raises (once we've
   dropped the compat shim; initially this is a warning).
8. **`test_hiding_H_not_identity`**: sanity check that `H` is not the
   G1 identity after `ppgen`.

### Hiding property (statistical)

9. **`test_hiding_distinguisher`**: sample 10 000 commitments of
   value `0` and 10 000 of value `1`, both with fresh `r`. Measure
   coordinate-wise χ² between the two distributions of `C` projected
   to `F_r` (via a fixed pairing-based map). Expect χ² statistic below
   the 99th percentile of the null. This is the "red-team
   distinguisher" test — if someone later backdoors `r` sampling, this
   should catch it.

The distinguisher test is the robust-against-red-teaming piece — it's
not a soundness proof, but a concrete statistical gate that a
malicious/flawed implementation would need to pass.

## Simulator argument (informal)

For each committed tensor `t`, the commitment `C = Σ tᵢ Gᵢ + r · H` has
the following property: given `H` with unknown discrete log and `r`
uniform in `F_r`, the distribution of `C` over `r` is independent of
`t` — this is exactly Hyrax's "perfect hiding" property (§3.1 quoted
above).

The simulator for Phase 1 (standalone, no opening) is trivial:

```
Sim(statement):
    sample r' ← F_r
    output C' = r' · H
```

The simulator's output is *perfectly* indistinguishable from a real
commitment: for any target `t`, since `r ∈ F_r` is uniform, so is
`r − Σ tᵢ · dlog_H(Gᵢ)` (ill-defined as an algorithm but well-defined
as a distribution), meaning `C = Σ tᵢ Gᵢ + r H` is uniform over `G1`.
This matches Hyrax §3.1's definition of perfect hiding.

The hiding property proved by this simulator is a *commitment-level*
property only. Phase 2 extends it to the opening transcript using the
blinded recursive-halving construction of Bulletproofs §4.2; Phase 3
extends it to the sumcheck transcript using Hyrax §4.

## Risks

1. **Reading `r` inadvertently leaks it.** Tests and debug prints must
   not print `r`. Mitigation: add a `[[nodiscard]]` wrapper type
   `Blinding` that disallows ostream operations.
2. **`H` equal to zero or to an existing `Gᵢ` by accident.** Nominally
   impossible at cryptographic probability, but an RNG bug could make
   it happen. Mitigation: `ppgen` checks `H ≠ 0` after sampling and
   loudly aborts otherwise. No protection against `H = α · G_i` for
   small `α` because that's the setup trust assumption by design.
3. **Pp file format migration breaks other branches.** The
   `pq-goldilocks` branch uses its own pp format, so no cross-branch
   impact. `zk-masking-implementation` is reference-only.

## Acceptance criteria

- [ ] All positive tests pass.
- [ ] All negative tests pass (reject malformed inputs).
- [ ] χ² distinguisher test passes.
- [ ] `test_zkargmax`, `test_zklog`, `test_zknormalcdf`, `test_zkentropy`
      all still pass.
- [ ] `zkllm_entropy.cu` end-to-end run produces a proof file
      identical in structure to pre-migration (the `r` lives in
      separate `.r.bin` files, not in the proof).
- [ ] Independent audit (A1) finds no gaps in: H independence, every
      commit site uses non-zero `r`, `r` never logged, simulator
      argument written down in this doc.

## Audit checklist for A1 (fresh opus agent, no context)

The Phase 1 audit should check:

1. Walk every call to any `commit*` method and confirm a fresh `r`
   is sampled — never reused, never zero (unless intentional
   backwards-compat path).
2. Walk every logging / serialization site for `r`: ensure `r` is
   never written to any file that ends up in the proof transcript.
3. Confirm `H` derivation: one fresh `Fr` scalar sampled in `ppgen`,
   `H = s_H · G`, `H` saved with the pp, non-zero check present.
4. Confirm `create_weight` fails loudly when pp is hiding but
   `r_file` is missing (once compat shim is removed).
5. Confirm the simulator argument in this doc is consistent with the
   implementation — i.e. the real commitment really is
   `rowwise_sum(...) + r · H`, not something algebraically different.

## What unblocks after Phase 1

- Phase 2 (blinded `me_open`): needs `Weight.r` populated.
- Phase 3 (Hyrax ZK sumcheck): needs `(commitment, r)` primitive to
  build round-polynomial commitments.
- Later verifier work (Phase 5+): needs the pp file format finalized.
