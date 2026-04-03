# ZK Approach Comparison: Three Ways to Prove Matmul in Zero Knowledge

## Background

Our goal is to prove `y = Wx` (matrix-vector multiplication) in **post-quantum zero knowledge** using sumcheck-based protocols with FRI/BaseFold commitments.

The core challenge: a single sumcheck for matmul reveals intermediate polynomial evaluations (e.g., `W̃(r,s)`, `x̃(s)`) that leak information about the witness. Achieving ZK requires hiding these evaluations while preserving soundness.

We prototyped three approaches on a 3×3 matmul (padded to 4×4) in F_257 to compare their structure, cost, and ZK properties. Code: `zk_approach_experiments.py`.

---

## Approach 1: Vanishing Polynomial Masking on Direct Sumcheck

**Idea:** Keep the direct sumcheck `T = Σ_s W̃(r,s) · x̃(s)` but mask each witness polynomial using vanishing polynomial terms:

```
Z_W(X) = W̃(X) + Σ_i c_W_i · X_i(1 - X_i)
Z_x(X) = x̃(X) + Σ_i c_x_i · X_i(1 - X_i)
```

The correction terms `X_i(1 - X_i)` vanish on the Boolean hypercube `{0,1}^k`, so the sum is preserved: `Σ_s Z_W(r,s) · Z_x(s) = Σ_s W̃(r,s) · x̃(s) = T`.

Combine with `g + ρ·p` sumcheck masking (XZZ+19) to hide the round polynomials.

### Results

| Metric | Without masking | With masking |
|---|---|---|
| Degree per round | 2 | 4 |
| Evaluations per round | 3 | 5 |
| Sumcheck rounds | 2 | 2 |
| Total evals | 6 | 10 |

**What verifier learns:**
- `Z_W(r,s)` and `Z_x(s)` — masked values, not the raw `W̃(r,s)` or `x̃(s)`
- `p(s)` — the zero-sum masking polynomial evaluation (from commitment opening)
- Round polynomials are masked by `ρ·p`

**What verifier does NOT learn:**
- `W̃(r,s)` or `x̃(s)` (hidden behind vanishing corrections)
- The masking coefficients `c_W`, `c_x`

### Cross-Layer Junction Problem

This is the key limitation. At the junction between two layers (e.g., layer 0 proves `y = h·x`, layer 1 proves `h = Wv`), the verifier needs `h̃(s)` to set up the next layer's claim. With masking, they get `Z_h(s)` instead. The prover must reveal `correction_h(s) = Σ_i c_h_i · s_i(1-s_i)` so the verifier can compute `h̃(s) = Z_h(s) - correction_h(s)`.

**This reveals `h̃(s)` — exactly what we were trying to hide.**

The vanishing polynomial masking protects within a single sumcheck but does not help at cross-layer junctions. For multi-layer inference, we'd need a way to avoid revealing intermediate evaluations entirely.

### Verdict

**Good for single-layer proofs. Breaks down at multi-layer junctions.** The 1.7× overhead in evaluations per round is modest, but the cross-layer information leak is a fundamental structural problem, not a tuning issue.

---

## Approach 2: GKR Circuit Representation

**Idea:** Represent the matmul as a layered arithmetic circuit and use the GKR protocol. Circuit structure for 4×4 matmul:

- Layer 0 (output): 4 gates = `y` values
- Layer 1 (add): 8 gates = partial sums from binary addition tree
- Layer 2 (mult): 16 gates = `W[i][j] * x[j]`

GKR starts at the output and reduces layer by layer via sumchecks.

### Results

| Metric | Value |
|---|---|
| Total circuit gates | 28 |
| Circuit depth | 3 (2 add + 1 mult) |
| Addition layer sumchecks | 2 rounds (1 per layer, trivial binary tree) |
| Mult layer sumcheck | 4 rounds (over wiring predicate, degree 3) |
| **Total sumcheck rounds** | **6** |
| For comparison: direct sumcheck | 2 rounds |

**What verifier learns (without ZK masking):**
- `V_0(g)` — output layer evaluation (public, fine)
- `V_1(u1)` — intermediate layer evaluation (leaks info about partial sums)
- `V_2(u2)` — mult layer evaluation (leaks info)
- `W̃(r)` and `x̃(r_col)` — weight and input evaluations (leaks info)

**Libra's R_i masking:** Would hide the intermediate `V_1`, `V_2` claims using random polynomials `R_i` that cancel across layers. Libra's technique requires **two evaluation points** per layer (arising from GKR's `add(g,x,y)` and `mult(g,x,y)` wiring predicates, which produce claims at both `u` and `v`). Our simplified binary-tree GKR produces only one point per layer, but a full GKR implementation would naturally produce two.

### Cross-Layer Junction

GKR's cross-layer structure is inherently different from chained sumchecks. Each layer transition is a sumcheck where the verifier doesn't learn individual gate values — just the MLE evaluation at a random point. Libra adds ZK by ensuring these evaluations are masked via `R_i` polynomials with correlated randomness across layers.

**Key insight:** The cross-layer junction problem from Approach 1 doesn't arise in GKR+Libra because the masking is designed to cancel across layers. But this requires restructuring our entire computation as a layered circuit.

### Scaling Analysis

For M×N matmul (padded to pM × pN):
- Mult layer: pM·pN gates, log2(pM·pN) variables → log2(pM·pN) round wiring sumcheck
- Addition tree: log2(pN) layers, each 1 round → log2(pN) rounds
- **Total: log2(pM·pN) + log2(pN) rounds** (vs. log2(pN) for direct sumcheck)
- For 128×128: 14 + 7 = 21 rounds (vs. 7 for direct)

### Verdict

**Correct ZK is achievable via Libra, but ~3× more sumcheck rounds than direct approach.** The circuit representation is natural for matmul but adds overhead from the wiring predicate sumcheck. The main advantage is that Libra's ZK technique is well-studied and proven secure.

---

## Approach 3: R1CS + Spartan-Style Two Sumchecks

**Idea:** Flatten the entire computation into an R1CS constraint system `(A·z) ∘ (B·z) = C·z`, then prove satisfaction using Spartan's two-sumcheck protocol:

1. **Sumcheck 1 (satisfiability):** `Σ_x eq(τ,x) · [Ã(x)·B̃(x) - C̃(x)] = 0` where `Ã(x) = Σ_y A_mle(x,y)·z̃(y)`
2. **Sumcheck 2 (batched inner products):** Verify `Ã(r_x)`, `B̃(r_x)`, `C̃(r_x)` via batched sumcheck over `y`

### R1CS Construction for 3×3 Matmul

- 9 multiplication constraints: `p[i][j] = W[i][j] · x[j]`
- 3 addition constraints: `y[i] = Σ_j p[i][j]`
- Total: 12 constraints, padded to 16
- Variables: `z = [1, x_0..x_2, y_0..y_2, p_00..p_22]`, 16 entries

### Results

| Metric | Value |
|---|---|
| R1CS constraints | 16 |
| R1CS variables | 16 |
| Sumcheck 1 | 4 rounds, degree 3 |
| Sumcheck 2 | 4 rounds, degree 2 |
| **Total sumcheck rounds** | **8** |
| For comparison: direct sumcheck | 2 rounds |

**What verifier learns:**
- `A_mle(r_x, r_y)`, `B_mle(r_x, r_y)`, `C_mle(r_x, r_y)` — public matrix evaluations (no privacy concern, since A, B, C encode the circuit structure)
- `z̃(r_y)` — **a single evaluation of the entire witness vector**

**What verifier does NOT learn:**
- Any individual intermediate value (products, partial sums)
- Any layer-specific evaluation

### Cross-Layer Junction

**There is no cross-layer junction.** All computation layers are encoded in a single constraint system. The intermediate values (products `p[i][j]`, partial sums) are entries in `z`, but only `z̃(r_y)` — a single random evaluation of the entire witness — is ever revealed to the verifier. This is the critical structural advantage.

The single `z̃(r_y)` evaluation is hidden by the polynomial commitment scheme's opening protocol. With BaseFold + vanishing polynomial masking on just this one polynomial, we get full ZK.

### Scaling Analysis

For M×N matmul:
- Constraints: M·N (multiplications) + M (additions) = M·(N+1)
- Variables: 1 + N + M + M·N = (M+1)·(N+1)
- Pad both to next power of 2: let C = next_pow2(M·(N+1)), V = next_pow2((M+1)·(N+1))
- Sumcheck 1: log2(C) rounds, degree 3
- Sumcheck 2: log2(V) rounds, degree 2
- **Total: log2(C) + log2(V) rounds**
- For 128×128: ~14 + ~14 = 28 rounds (vs. 7 for direct, 21 for GKR)
- But prover cost per round is lower (sparse matrices)

### Verdict

**Best ZK properties: only one private scalar revealed, no cross-layer problem.** More sumcheck rounds than the other approaches, but the ZK story is clean — only `z̃(r_y)` needs masking, and existing techniques (vanishing polynomial or commitment blinding) handle this directly. The R1CS structure is also compatible with existing Spartan implementations.

---

## Comparison Table

| | Direct + Vanishing | GKR + Libra | R1CS + Spartan |
|---|---|---|---|
| **Sumcheck rounds (3×3)** | 2 | 6 | 8 |
| **Sumcheck rounds (128×128)** | 7 | ~21 | ~28 |
| **Degree per round** | 4 | 1–3 | 2–3 |
| **Private values revealed** | 2 per layer junction | 0 (with Libra) | 1 total (z̃(r_y)) |
| **Cross-layer junction** | Leaks h̃(s) | Handled by Libra R_i | No junctions |
| **Multi-layer ZK** | Unsolved | Solved (proven) | Solved (proven) |
| **Restructuring needed** | Minimal | Full circuit rewrite | R1CS encoding |
| **Existing implementations** | None | Libra (not post-quantum) | Spartan (not post-quantum) |
| **PCS compatibility** | BaseFold natural fit | BaseFold works | BaseFold works |
| **Prover cost dominant** | Polynomial evals | Circuit wiring evals | Sparse matrix MLE evals |

---

## Recommendations

### For single-layer proof (one matmul, no chaining)
**Approach 1 (vanishing polynomial masking)** is simplest and most efficient. Only 1.7× overhead, no restructuring, and the cross-layer problem doesn't arise.

### For multi-layer inference (chained matmuls + activations)
Two viable options:

1. **GKR + Libra-style masking (Approach 2):** More natural for layered computation, ~3× rounds overhead. Libra's ZK technique is proven but was designed for pre-quantum (Pedersen) commitments. Adapting to BaseFold/FRI needs careful analysis of how R_i polynomials interact with the commitment scheme.

2. **R1CS + Spartan (Approach 3):** Cleanest ZK story — only one private value ever revealed. ~4× rounds overhead but the prover exploits sparsity. Spartan's ZK is simpler to reason about (just mask z̃). The main cost is encoding everything as R1CS constraints, which grows with circuit size but is structurally straightforward.

### What we recommend investigating next

1. **Spartan + BaseFold integration:** Spartan's two-sumcheck protocol is the most promising path to multi-layer ZK. The key question is prover performance: for 128×128 matmul, the R1CS has ~16K constraints and ~16K variables. The Spartan prover needs to evaluate sparse matrix MLEs, which may be optimizable on GPU.

2. **CCS/HyperNova as an alternative to R1CS:** CCS (Customizable Constraint Systems) generalize R1CS and can represent computations more compactly. HyperNova uses CCS with a single sumcheck + folding, potentially reducing the round count. This is worth investigating if the R1CS round overhead proves prohibitive.

3. **Hybrid approach:** Use direct sumcheck within each layer (efficient, Approach 1) and Spartan-style batching across layers. The cross-layer junction issue might be solvable by batching all layer outputs into a single witness polynomial, then doing one Spartan-style proof that all layers are consistent. This could combine the best of both worlds.

---

## Bugs Found and Fixed During Prototyping

1. **MLE bit ordering in GKR layer transitions:** Binary tree addition layers prepend the new variable as bit 0 (LSB), not append as MSB. Fix: `pt = [j] + list(g)` instead of `list(g) + [j]`.

2. **MLE of products ≠ product of MLEs:** The GKR multiplication layer cannot be verified as `W̃(u) · x̃(u_col) = V_mult(u)` at random points. A proper wiring-predicate sumcheck is needed: `V_mult(u) = Σ_b eq(u,b) · W̃(b) · x_ext(b)`.

3. **eq() evaluation for non-Boolean inputs:** The `eq_eval` helper originally used `if bi == 1` branching, which only works for Boolean inputs. After sumcheck, the evaluation point contains random field elements. Fix: general formula `eq(a,b) = Π_i (2·a_i·b_i - a_i - b_i + 1)`.

4. **Row/column MLE variable ordering in Spartan:** Flat array `A[row*n + col]` puts column bits in low positions (MLE variables 0..n-1) and row bits in high positions. The MLE evaluation point must be `(col_vars, row_vars)`, not `(row_vars, col_vars)`.
