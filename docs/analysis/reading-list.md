# Reading List: Post-Quantum ZK for Matmul

Organized by priority for our work: proving `y = Wx` (and multi-layer inference) in post-quantum zero knowledge using sumcheck-based protocols.

## Tier 1: Must Read

These directly define the system we'd build.

| # | Paper | Why it matters |
|---|---|---|
| 1 | **Spartan** — Setty, CRYPTO 2020 ([ePrint 2019/550](https://eprint.iacr.org/2019/550)) | Core protocol: R1CS satisfiability via two sumchecks. Only z̃(r_y) revealed. Introduces Spark for sparse MLE evaluation. |
| 2 | **BaseFold** — ePrint 2023/1705 ([paper](https://eprint.iacr.org/2023/1705)) | Post-quantum multilinear PCS based on FRI. The commitment scheme we'd pair with Spartan. |
| 3 | **Lasso** — Setty & Thaler, EUROCRYPT 2024 ([ePrint 2023/1216](https://eprint.iacr.org/2023/1216)) | Successor to Spark. Lookup arguments for sparse polynomial evaluation. O(m+n) prover cost for m lookups into table of size n. Subsumes Spark entirely. |
| 4 | **Thaler, Proofs, Arguments, and Zero-Knowledge** (textbook) | Chapters on sumcheck, GKR, MLE. Essential background. Key quote: "even a single evaluation of W̃ leaks information about w." |

## Tier 2: Important Context

These inform design decisions and alternatives.

| # | Paper | Why it matters |
|---|---|---|
| 5 | **SuperSpartan / CCS** — ePrint 2023/552 ([paper](https://eprint.iacr.org/2023/552)) | Customizable Constraint Systems generalize R1CS. May represent matmul more compactly. Prover crypto cost independent of constraint degree. |
| 6 | **Libra** — Xie, Zhang, Song, CCS 2019 | ZK masking for GKR via cross-layer R_i polynomials. Doesn't directly fit our architecture (needs two eval points per layer from GKR wiring) but is the main competing ZK approach. |
| 7 | **Habock & Al Kindi — ZK for STARK** | Identifies three masking layers needed for full ZK: (1) witness polynomial randomization, (2) sumcheck masking g+ρ·p, (3) BSCR+19 R(X) for FRI entropy loss. |
| 8 | **XZZ+19** (Xie, Zhang, Zhang) | The g + ρ·p sumcheck masking optimization where p = sum of univariates. Used in Libra and applicable to any sumcheck-based system. |
| 9 | **DeepProve** | Closest competitor: BaseFold + GKR for neural net inference. But has NO actual ZK implementation despite claims. Useful as a performance reference point. |

## Tier 3: Deeper Dives

For when we're implementing specific components.

| # | Paper | Why it matters |
|---|---|---|
| 10 | **BabySpartan** — ePrint 2023/1799 ([paper](https://eprint.iacr.org/2023/1799)) | Replaces Spark with Lasso internally. Most modern Spartan variant. |
| 11 | **Blum et al. — Offline memory checking** | Foundation for Spark/Lasso. Proves sparse lookups were done correctly via timestamps/counters and grand product arguments. |
| 12 | **BSCR+19** | FRI folding entropy loss: each FRI fold leaks information. The R(X) mask polynomial compensates. Needed if/when we move to non-interactive proofs. |

## Code References

| Repo | Status | Notes |
|---|---|---|
| [microsoft/Spartan](https://github.com/microsoft/Spartan) | Has Spark implemented | Locked to one PCS (not post-quantum). Best reference for Spark implementation. |
| [microsoft/Spartan2](https://github.com/microsoft/Spartan2) | PCS-agnostic design | No Spark, no BaseFold yet. Only IPA + Hyrax working. README aspirations exceed implementation. |
| DeepProve / zkml | BaseFold + GKR | No ZK masking. Performance reference only. |

## Explainers

- [Alin Tomescu's Spartan explainer](https://alinush.github.io/spartan) — visual walkthrough
- [Spartan SumCheck Trick (HackMD)](https://hackmd.io/@mprashker12/rJkM_JPQh) — detailed sumcheck mechanics
- [Setty's Spartan overview (HackMD)](https://hackmd.io/@srinathsetty/spartan) — author's own summary

## Suggested Reading Order

1. Thaler textbook (sumcheck + MLE chapters) — if not already familiar
2. Spartan paper (sections 1-5) — the core protocol
3. BaseFold paper (sections 1-3) — the PCS we'd use
4. Lasso paper (sections 1-4) — how to make the prover fast
5. Habock/Al Kindi — the ZK masking layers we need on top
6. SuperSpartan/CCS — whether we can avoid R1CS overhead
