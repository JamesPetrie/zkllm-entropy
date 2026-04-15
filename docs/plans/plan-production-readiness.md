# Plan: Production-Ready BLS12-381 ZK Entropy System

**Date:** 2026-04-14 (revised 2026-04-15)
**Status:** Approved, execution in progress
**Target:** `main` branch (BLS12-381 / Pedersen)
**ZK masking approach:** Hyrax-faithful (Wahby et al. §4 + §A.2), **not** Plonky2 / XZZ+19

## Revision note (2026-04-15)

An earlier draft described Phase 3 as a port of the `zk-masking-implementation`
branch. That branch implements Plonky2-style masking (vanishing polynomial
over the sumcheck transcript, plus XZZ+19 full-transcript blinding). Plonky2's
technique is designed for FRI, not for Pedersen, and the Hyrax paper uses a
different construction — proof-of-opening / proof-of-equality / proof-of-product
over Pedersen commitments (Hyrax §4), finalized with the ZK dot-product
protocol of Hyrax §A.2 (Figure 6).

We are going Hyrax-faithful to minimize deviation from the cited paper.
Consequently, the `zk-masking-implementation` branch is a **reference**, not a
port source, and Phase 3's reuse estimate drops accordingly.

## Goal

Bring the BLS12-381 entropy pipeline to a state where we can honestly say:

1. The code produces zero-knowledge proofs of the conditional-entropy bound,
   verifiable by an independent verifier binary, with no known soundness gaps
   for one supported model configuration (Llama-2-7B).
2. We have a parametric cost model that accurately predicts prover/verifier
   cost for other decoder-transformer architectures and context lengths,
   validated against at least one unseen configuration.

## Non-goals

- **Fiat-Shamir.** Interactive is fine. A production deployment would add
  Fiat-Shamir but the security argument is standard and can be documented
  as future work.
- **Arbitrary architectures.** We pick Llama-2-7B end-to-end. Other
  architectures get *predicted* costs via the cost model, not implementations.
- **Reproducing the full zkLLM paper.** We are not proving attention layers
  1–31 in our pipeline — the zkLLM fork already handles those. We cover the
  final-layer entropy proof plus whatever upstream stages we need.

---

## Existing work inventory

A survey of commit history surfaced significant prior work on sibling
branches. Most of the crypto engineering we would otherwise write from
scratch already exists, but was built for Goldilocks + FRI. Porting to
BLS12-381 + Pedersen saves meaningful effort.

### On `zk-masking-implementation`

| Artifact | Description | Reusability |
|---|---|---|
| `src/proof/zk_mask.cu/cuh` (273 lines) | Plonky2-style vanishing-polynomial + XZZ+19 transcript masking | **Reference only.** Different masking family from Hyrax §4; would mix with a Pedersen commitment scheme in non-standard ways. |
| `src/proof/zk_sumcheck.cu/cuh` | Degree-4 ZK sumcheck kernels | **Reference only.** Same reason — degree inflation assumes FRI-side tricks we won't use. |
| `verifier/sumcheck_verifier.h` (259 lines) | CPU verifier for standard and ZK sumcheck proofs | Partial: the *standard* sumcheck verifier structure is reusable; the ZK variant is not a direct match for our Hyrax-§4 transcript. |
| `test_zk_verifier` (12 tests passing) | CPU-only regression suite | Test *scaffolding* is reusable; individual test cases must be rewritten against Hyrax-faithful transcript structure. |
| `docs/plans/plan-zk-blinding.md` (470 lines) | Design doc — Plonky2 family | Use as reference for comparison in security writeup; not as the design we're following. |

### On `pq-goldilocks` (was `goldilocks-fri`)

| Artifact | Description | Reusability |
|---|---|---|
| `verifier/verifier.cpp` (1118 lines) | Full cryptographic verifier | Port: sumcheck + tLookup logic reusable; Merkle path verification must be replaced with Pedersen opening verification |
| `verifier/tlookup_verifier.h` (204 lines) | tLookup verifier | Port: field-swap |
| `verifier/verifier_utils.h` (487 lines) | Proof parsing, host field arithmetic | Port: field-swap |
| `test_verifier.cpp` (685 lines) + `test_verifier_negative.cpp` (490 lines) | Verifier regression + negative tests | Port: ~mechanical; negative tests need to target BLS-specific failure modes too |
| Interactive verifier w/ pluggable challenge source (commit `0eb1dbc`) | Protocol framework | Reuse design |
| Weight-binding proof serialization (commit `f193ed7`) | All weight-commitment opening proofs serialized into proof file | Port: format adapts, Pedersen opening replaces FRI query |
| `docs/plans/plan-full-verifier.md` | Verifier architecture and phase plan | Use as reference |

### On current `main`

- BLS12-381 / Pedersen prover for entropy pipeline (works)
- Binding-only Pedersen commit (`src/commit/commitment.cu`)
- Plain sumcheck (`src/proof/proof.cu`)
- `verify_entropy.py` arithmetic-only verifier
- `zkfc`, `rescaling`, `zksoftmax`, `zkargmax`, `zklog`, `zknormalcdf`, `zkentropy`, RMSNorm (Llama-style)

### Honest reuse estimate

Porting Goldilocks → BLS12-381 is not free. Two pieces of friction:

1. **Field size.** BLS12-381 scalar is 255-bit vs Goldilocks 64-bit. All
   hash inputs, polynomial coefficient buffers, serialization formats grow.
2. **Commitment layer is structurally different.** Merkle path verification
   (log-depth hash chain) is replaced by Pedersen opening verification
   (Bulletproofs-style recursive MSM check). Sumcheck and tLookup verification
   are field-generic and port cleanly; the commitment-opening path is a rewrite.

Realistic effective savings from reuse: **30–40%** of verifier effort from
the `pq-goldilocks` branch (non-ZK sumcheck + tLookup verifier structure,
proof-parsing scaffolding, test harness). **Negligible** ZK-sumcheck savings
from `zk-masking-implementation` because we are not using Plonky2-style
masking — the Hyrax §4 construction (per-round Pedersen commitments with
proof-of-opening / -equality / -product, finalized by the §A.2 ZK dot-product)
must be written fresh.

---

## Gap analysis

| Requirement | Current main | Exists elsewhere? | Action |
|---|---|---|---|
| Hiding Pedersen (`rH` blinding) | No | No | Write from scratch |
| Hiding opening proof (blinded recursive halving) | No | No | Write from scratch |
| ZK sumcheck, Hyrax §4 style | No | No (branch uses different family) | Write from scratch against Hyrax §4 + §A.2 |
| Weight-binding proof serialization | No | Yes (pq-goldilocks) | Port |
| Cryptographic verifier — entropy pipeline | No | Partial (pq-goldilocks, Goldilocks) | Port sumcheck/tLookup verifier, rewrite commitment verifier |
| Verifier coverage for Llama layers (attention, FFN, RMSNorm, residuals) | No | Partial | Extend ported verifier |
| SwiGLU zkModule | No | No | Write from scratch |
| Parametric cost model | No | Partial (scaling-analysis doc) | Derive, calibrate, validate |
| Independent audit | Periodic | Periodic | Repeat pre-publication |

---

## Phased plan

Each phase lists the concrete work, the reuse source if any, and an estimate
for one experienced engineer working focused.

### Phase 1 — Hiding Pedersen commitment (2 weeks)

**Work:**
- Add independent generator `H` to `ppgen.cu` and `commit-param.cu`; bump
  public-parameter file format version.
- Modify `Commitment::commit` (`src/commit/commitment.cu:22-29`) to sample `r`
  and return `(commitment, r)`. Update `commit_int`, `commit_int_multi`.
- Extend `Weight` struct (`src/commit/commitment.cuh:28-34`) to carry `r`.
- Thread `r` through every call site (~15, in `bin/`, `src/llm/`, `src/zknn/`).
- Add unit tests covering hiding property (randomness is used, commitments
  of the same value with different `r` differ).

**Reuse:** none — new work.
**Risk:** medium — large number of call sites. Mechanical but easy to miss one.

### Phase 2 — Hiding opening proof (1.5 weeks)

**Work:**
- Rewrite `Commitment::me_open` (`src/commit/commitment.cu:109-129`) to
  thread blinding through each recursive-halving round (Bulletproofs §4.2
  construction).
- Update `verifyWeightClaim` and all upstream openers to handle the blinded
  return value.
- Soundness tests: verifier rejects when blinding is replayed incorrectly.

**Reuse:** structure of existing `me_open`. Algorithm change is the
Bulletproofs blinding pattern.
**Risk:** medium — easy to introduce a subtle bug that still verifies for
correct inputs but silently weakens hiding.

### Phase 3 — ZK sumcheck, Hyrax §4 / §A.2 (3 weeks, revised up)

**Approach:** Follow Hyrax Protocol 3 (§4, p. 11–13) and Figure 6 (§A.2).
Per sumcheck round `j`, the prover sends a Pedersen *commitment* to the
round polynomial `g_j(X)` rather than the polynomial in the clear. The
verifier's round challenge `r_j` is matched to a commitment to `g_j(r_j)`
via a proof-of-equality (Hyrax §A.1, Figure 5 proof-of-opening +
proof-of-equality). The final round's claimed sumcheck value is reduced
to a dot-product claim between committed vectors and discharged with the
ZK dot-product protocol of Hyrax §A.2, Figure 6.

**Subcomponents to build (all new code, all Hyrax-faithful):**

1. **Pedersen commitment-to-polynomial primitive.** Commit to a
   degree-`d` univariate (`d = 1` for linear sumcheck rounds, `d = 2`
   for hadamard, `d = K` for multi-hadamard) by committing each
   coefficient with its own blinding. `src/proof/zk_round_commit.cu/cuh`.
2. **Hyrax proof-of-opening / -equality / -product** (§A.1 Figure 5).
   `src/proof/hyrax_sigma.cu/cuh`.
3. **Hyrax ZK dot-product, §A.2 Figure 6.** `src/proof/hyrax_dotprod.cu/cuh`.
4. **ZK sumcheck driver.** Replaces the four plain-sumcheck variants in
   `src/proof/proof.cu` (`inner_product_sumcheck`,
   `hadamard_product_sumcheck`, `binary_sumcheck`,
   `multi_hadamard_sumchecks`). Emits per-round commitments, opening
   challenges, and the final dot-product proof. New file
   `src/proof/zk_sumcheck.cu/cuh` — name matches the branch for
   continuity but implementation is Hyrax-faithful.
5. **Test suite.** Positive tests per Σ-protocol and per sumcheck
   variant; negative tests for blinding replay, wrong commitment,
   wrong dot-product final value, and distinguishability tests
   (real-vs-simulated transcript χ² / KS statistic).

**Wire into:** `zkentropy.cu`, `zkfc.cu`, `zksoftmax.cu`, `rescaling.cu`,
`tlookup.cu`, Llama layer proofs.

**Reuse:** negligible for the ZK-specific work. Some test scaffolding
from `test_zk_verifier` (branch) carries over. The Σ-protocol
implementations are not large — roughly ~600–800 LOC in C++/CUDA — but
they need careful review.

**Risk:** medium. This is the largest deviation-from-existing-code
phase. Mitigation: subcomponents 2 and 3 are published protocols with
explicit transcripts; we can audit them against the Hyrax figures
directly.

### Phase 4 — Weight-binding proof serialization (1 week)

**Work:**
- Port the `f193ed7` design: serialize `verifyWeightClaim`, `zkFC.prove`,
  `Rescaling.prove` proof elements into the proof file.
- Adapt format to Pedersen: opening proofs are `O(log D)` group elements
  (not Merkle paths).
- Update `zkllm_entropy.cu` and `zkllm_entropy_timed.cu` to emit these.

**Reuse:** ~50%. Structure is known; serialization format changes.
**Risk:** low.

### Phase 5 — Cryptographic verifier, entropy pipeline only (3 weeks)

**Work:**
- Port `verifier/sumcheck_verifier.h` and `verifier/tlookup_verifier.h`
  from `pq-goldilocks` (field swap).
- Port `verifier/verifier_utils.h` (field swap, proof parsing adapts to
  Pedersen format).
- **Rewrite** the commitment-verification path: replace Merkle-path
  verification with Pedersen opening verification (checks the recursive MSM
  identity across `O(log D)` rounds).
- Cover: zkArgmax, zkNormalCDF, zkLog, zkEntropy (batched), row-sum
  sumcheck, indicator extraction, quotient-remainder, surprise accumulation.

**Reuse:** ~40%. Sumcheck+tLookup verification logic ports well; commitment
layer is a rewrite.
**Risk:** medium — this is where the biggest unknown is. Budget slack.

### Phase 6 — Verifier coverage for Llama layers (4 weeks)

**Work:** extend Phase 5 verifier to check:
- RMSNorm (`src/llm/rmsnorm.cu` — Hadamard product sumcheck + opening)
- Self-attention (matmul × 2, softmax via `zksoftmax` — K-digit decomposition)
- FFN (matmul × 2, SwiGLU)
- Residual connections (no new proof; linear combinations)
- Embedding / unembedding

Each layer type needs: proof-format specification, verifier kernel,
positive test, at least one negative test per check.

**Reuse:** sumcheck verifier ports directly. Layer-specific orchestration is new.
**Risk:** medium — scope creep here is the most likely schedule risk.

### Phase 7 — SwiGLU zkModule (1 week)

**Work:**
- Specialized tLookup table for Swish(x) × x (single-variable lookup).
- `src/zknn/zkswiglu.cu/cuh` parallel to `zkrelu.cu/cuh`.
- Generator script (`python/generate_swiglu_table.py` already exists) +
  integration tests.

**Reuse:** `zkrelu` as template.
**Risk:** low.

### Phase 8 — Parametric cost model (2 weeks)

**Work:**
- Derive analytical cost formulas for each primitive:
  - Sumcheck round cost as a function of tensor size and field-op cost
  - Hiding Pedersen commit cost as a function of element count
  - Pedersen opening cost as a function of `log D`
  - tLookup cost as a function of query count and table size
  - MSM cost as a function of vector length
- Express each layer in those primitives:
  - Matmul (m, k, n)
  - Softmax (n, d, sequence length)
  - RMSNorm (n, d)
  - SwiGLU (n, d)
- Build a benchmark harness (`bench/bench_cost_model.cu`) that measures each
  primitive at several sizes and fits coefficients.
- Document the model in `docs/analysis/cost-model.md`.

**Reuse:** partial. `docs/analysis/zkllm-entropy-scaling-analysis.md`
contains some relevant measurements; `bench/bench_*.cu` stubs exist.
**Risk:** low on the modeling, higher on presenting it convincingly.

### Phase 9 — Validation (1 week)

**Work:**
- Calibrate cost model on one configuration (Llama-2-7B, seq 1024).
- Predict a held-out configuration (e.g., Llama-2-7B, seq 4096, or a smaller
  model with the same architecture).
- Measure the held-out configuration.
- Report prediction error. Target: within ±15% on total prover time.
- If error is larger, diagnose which primitive's model is wrong and fix.

**Reuse:** none.
**Risk:** medium — the first time a cost model is validated, it's often
wrong in ways that need investigation. Budget slack.

### Phase 10 — Independent audit (1.5 weeks)

**Work:**
- Launch a fresh agent with no context to audit:
  - Transcript completeness: does every prover emission have a verifier check?
  - Verifier coverage: does every proof element get used?
  - Negative test coverage: is there a targeted negative test per check?
  - Hiding property: is `r` actually used throughout?
- Address findings. Distinguish real gaps from documented future work.

**Reuse:** follow `CLAUDE.md` "Independent Audits" procedure.
**Risk:** reveals unknown unknowns; that's the point.

---

## Timeline summary

| Phase | Weeks | Cumulative |
|---|---|---|
| 1. Hiding Pedersen | 2 | 2 |
| 2. Hiding opening proof | 1.5 | 3.5 |
| 3. ZK sumcheck, Hyrax §4 / §A.2 | 3 | 6.5 |
| 4. Weight-binding serialization (port) | 1 | 7.5 |
| 5. Crypto verifier — entropy only | 3 | 10.5 |
| 6. Verifier — Llama layers | 4 | 14.5 |
| 7. SwiGLU zkModule | 1 | 15.5 |
| 8. Cost model | 2 | 17.5 |
| 9. Validation | 1 | 18.5 |
| 10. Audit | 1.5 | 20 |
| **Total, one engineer** | **~20 weeks (≈5 months)** | |

Two engineers splitting prover vs. verifier work: **~12–14 weeks**.

Compared to writing from scratch the savings are modest (~2–3 weeks),
concentrated in the non-ZK verifier structure from `pq-goldilocks`.
Hyrax-faithful ZK sumcheck is fresh work; field porting is not free; and
Pedersen opening verification has to be written from scratch.

---

## Milestones

- **M1 (week 3.5):** Hiding commits + hiding openings land on `main`. All
  existing tests still pass. `main` branch is now "hiding Pedersen ready"
  but sumcheck still leaks.
- **M2 (week 9.5):** Entropy pipeline end-to-end ZK, with interactive
  crypto verifier for the entropy layer only. We can demo: prover emits
  proof file, verifier binary reads and returns pass/fail.
- **M3 (week 14.5):** Full Llama-2-7B pipeline verifiable end-to-end with
  a supported activation set (ReLU + SwiGLU + softmax).
- **M4 (week 17.5):** Cost model calibrated and validated on ≥1 held-out
  configuration. Paper-ready scaling claims.
- **M5 (week 19):** Audit complete, residual gaps documented.

## Risks and mitigations

**Scope creep in Phase 6.** Covering all Llama layers means touching a lot
of existing prover code that was written without a verifier in mind. Each
layer may reveal missing proof elements. Mitigation: do Phase 6 strictly
*after* the entropy verifier is working (Phase 5), so the verifier
framework is debugged before extending it.

**Pedersen opening verification is where the crypto is new.** This isn't
a port — it's a rewrite of the verification side of Bulletproofs-style
recursive halving with blinding. Easy to get subtly wrong. Mitigation:
peer-review this specific piece; write explicit negative tests targeting
replay of blinding values, wrong opening scalar, wrong final-round group
element.

**Cost model validation error.** If the first validation run disagrees with
prediction by more than ±15%, we need to diagnose. Most likely culprits:
memory-bandwidth-bound primitives not captured by flop counts, NTT-like
structure in tLookup that isn't linear in table size, or field-op costs
varying with data patterns. Mitigation: validate early (week 17.5) so
there's time for one or two calibration revisions before paper submission.

**Goldilocks branch drift.** The `zk-masking-implementation` branch is
dated. If it has diverged from `main`'s structure, porting gets harder.
Mitigation: port early (Phase 3); defer decisions on whether to delete
that branch until after port is validated.

## What this plan does not do

- No Fiat-Shamir. We remain interactive; the paper should say so.
- No support for architectures that use LayerNorm, GELU, Sigmoid, or that
  differ structurally from Llama. Cost predictions are extrapolations, not
  executions.
- No production hardening: no constant-time field ops, no side-channel
  resistance, no third-party cryptographic audit. These are post-research
  concerns.

## Decision points

Before starting, confirm:

1. **Target model.** Llama-2-7B is assumed. If a different decoder
   transformer (e.g., Llama-2-13B, Llama-3, Mistral) is preferred, adjust
   commitment generator sizes and benchmark targets accordingly.
2. **Masking scheme.** **Decided 2026-04-15: Hyrax-faithful.** Phase 3
   implements the per-round Pedersen commitment construction of
   Hyrax §4, finalized with the ZK dot-product protocol of §A.2
   (Figure 6). Rationale: minimizes deviation from the cryptographic
   protocol our soundness+ZK argument cites. The Plonky2-style
   vanishing-polynomial alternative on `zk-masking-implementation`
   is cheaper per round but introduces a transcript structure that
   isn't what Hyrax proves ZK for.
3. **Verifier language.** **Decided 2026-04-15: C++ preferred, Rust
   acceptable.** User's stated preference is code reuse from
   `pq-goldilocks` (C++) unless the implementing agent prefers Rust.
   Default: C++. Revisit if Phase 5 uncovers structural reasons a
   Rust rewrite would materially help audit-friendliness.

## References

- `docs/plans/plan-zk-blinding.md` — ZK masking design on the
  `zk-masking-implementation` branch (reference only; Plonky2 family,
  not what we implement)
- `docs/plans/plan-full-verifier.md` — detailed verifier architecture
  (from `pq-goldilocks` branch)
- `docs/plans/phase-1-hiding-pedersen.md` — kickoff doc for Phase 1
- `docs/analysis/prover-determinism.md` — sources of prover freedom and
  the sandwich fix
- `docs/analysis/security-review.md` — identified soundness gaps
- `docs/analysis/zkllm-entropy-scaling-analysis.md` — existing scaling
  measurements (will feed Phase 8)
- **Wahby et al. 2018, "Doubly-efficient zkSNARKs without trusted setup"
  (Hyrax) — eprint 2017/1132.** Primary cryptographic reference for
  Phases 2–5. §4 = ZK sumcheck compilation; §A.1 Fig 5 = Σ-protocols
  (proof-of-opening, -equality, -product); §A.2 Fig 6 = ZK dot-product;
  §6.1 = matrix-layout multilinear commitment.
- Bünz et al. 2018, "Bulletproofs" — §4.2 recursive halving opening
  (blinded form used in Phase 2).
- Xie et al. 2019 / XZZ+19 — transcript masking technique (comparison
  reference, not implemented).
- Plonky2 whitepaper §3.6 — comparison reference for the alternative
  masking family.
