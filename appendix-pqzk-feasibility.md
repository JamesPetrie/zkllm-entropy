# Appendix: Feasibility of Post-Quantum Zero-Knowledge Inference Verification

## Motivation

The current proof system (per-operation sumchecks + Pedersen commitments on BLS12-381) achieves soundness and weight confidentiality but not full zero knowledge. Intermediate polynomial evaluations are revealed at each sumcheck step — for example, when proving $y = Wx$, the verifier learns $\tilde{W}(r,s)$ and $\tilde{x}(s)$ at the sumcheck's random evaluation point. These evaluations leak information about model weights and activations. Additionally, Pedersen commitments rely on the discrete logarithm assumption, which is vulnerable to quantum attack.

This appendix describes a concrete design for achieving both post-quantum security and zero knowledge with minimal overhead over the current proof system.

## Design Overview

The proposed system preserves the current per-operation sumcheck architecture and adds two components:

1. **Vanishing polynomial masking** — each private polynomial is masked with random corrections that vanish on the Boolean hypercube, hiding evaluations at random points without changing the sumcheck's claimed sum.
2. **BaseFold** (Zeilberger et al., ePrint 2023/1705) — a post-quantum polynomial commitment scheme based on FRI, generalized to multilinear polynomials. Replaces Pedersen commitments. Provides binding (cross-layer consistency) and hiding (masked values stay hidden).

Non-arithmetic operations (CDF, logarithm) continue to use **tLookup** (Sun et al., 2024), which is already sumcheck-based and composes naturally alongside the arithmetic sumchecks.

The key insight is that the current proof system's per-operation sumcheck structure is already efficient. The only missing pieces are (1) hiding the polynomial evaluations revealed at each sumcheck step and (2) replacing the quantum-vulnerable commitment scheme. Both can be added without restructuring the proof pipeline.

## Proof Structure

### Step 1: Forward Pass and Intermediate Collection

The prover runs the full forward pass of the model, collecting all intermediate values: layer activations, attention scores, FFN intermediates, and the final entropy bound $H$. This is identical to what the current system does.

### Step 2: Vanishing Polynomial Masking

For each private polynomial $\tilde{f}$ (weights, activations, intermediate products), the prover constructs a masked version:

$$Z_f(X_1, \ldots, X_k) = \tilde{f}(X_1, \ldots, X_k) + \sum_{i=1}^{k} c_i \cdot X_i(1 - X_i)$$

where $c_1, \ldots, c_k$ are random field elements chosen by the prover.

**Why this works:** Each correction term $X_i(1 - X_i)$ evaluates to zero on the Boolean hypercube $\{0,1\}^k$, so $Z_f$ agrees with $\tilde{f}$ at every Boolean input. Since sumchecks sum over the Boolean hypercube, the claimed sum is unchanged: $\sum_{b \in \{0,1\}^k} Z_f(b) = \sum_{b \in \{0,1\}^k} \tilde{f}(b)$. But at the random evaluation point $r$ produced by the sumcheck, $Z_f(r) \neq \tilde{f}(r)$ — the verifier sees only the masked value.

**Degree increase:** Each $\tilde{f}$ is multilinear (degree 1 per variable). After masking, $Z_f$ is degree 2 per variable. For a sumcheck proving $y = Wx$ with both $W$ and $x$ masked, the round polynomial $g(X_j) = Z_W(r, X_j, \ldots) \cdot Z_x(X_j, \ldots)$ has degree $2 + 2 = 4$ per round variable, up from degree $1 + 1 = 2$ without masking.

**Which variables to mask.** For a weight matrix $W$ with MLE variables $(c_1, \ldots, c_b, r_1, \ldots, r_a)$ where $c$ indexes columns and $r$ indexes rows, only the column (sumcheck) variables receive masking terms. The row variables are set to a verifier-chosen random point $r$ — they are not summed over, so masking them would add a non-vanishing constant to the sum and break the sumcheck relation. The column-only masking still hides $\tilde{W}(r, s^*)$ at the final evaluation point, because the correction $\sum_i c_i \cdot s_i^*(1-s_i^*)$ depends on the secret coefficients $c_i$ and the random sumcheck point $s^*$.

**Coefficient count requirement.** Each opening of a masked polynomial at a random point gives the verifier one linear equation in the masking coefficients. If a polynomial is opened at $m$ points, the prover needs strictly more than $m$ masking coefficients to keep the system underdetermined. With $k$ variables, there are $k$ degree-2 vanishing terms $X_i(1-X_i)$, providing $k$ coefficients. For intermediate values opened at 2 points (once as the output of one layer, once as the input of the next), this requires $k > 2$. For realistic vector sizes ($k = \log_2 4096 = 12$), this is trivially satisfied. For small $k$ (e.g., $k = 2$), the polynomial can be extended with random blinding entries to add variables, keeping the masking at degree 2 while adding one extra sumcheck round per variable.

### Step 3: BaseFold Commitment

The prover commits to each masked polynomial $Z_f$ via BaseFold:

1. Evaluate $Z_f$ on a domain (extending the multilinear evaluations to a Reed-Solomon codeword)
2. Build a Merkle tree over the evaluations (using SHA-256)
3. Send the Merkle root to the verifier

This replaces the current Pedersen commitment (BLS12-381 multi-scalar multiplication) with a hash-based construction. The Merkle root is the commitment; it is binding under SHA-256 collision resistance and post-quantum secure.

### Step 4: Per-Operation Sumchecks

Each operation in the forward pass is proven via an independent sumcheck, exactly as in the current system. For a matmul $y = Wx$:

1. **Claim:** $\tilde{y}(r) = \sum_{s \in \{0,1\}^k} Z_W(r, s) \cdot Z_x(s)$ for a random $r$ chosen by the verifier
2. **Sumcheck:** $k$ rounds, degree 4 per round (5 evaluations per round instead of 3)
3. **Final check:** The verifier needs $Z_W(r, s^*)$ and $Z_x(s^*)$ at the sumcheck's terminal point $s^*$. The prover opens these via BaseFold.

The verifier sees $Z_W(r, s^*)$ and $Z_x(s^*)$ — the masked values — not the raw $\tilde{W}(r, s^*)$ or $\tilde{x}(s^*)$.

**Cross-layer binding:** If layer $\ell$ proves $y^{(\ell)} = W^{(\ell)} x^{(\ell)}$ and layer $\ell+1$ proves $y^{(\ell+1)} = W^{(\ell+1)} x^{(\ell+1)}$ where $x^{(\ell+1)} = y^{(\ell)}$, the verifier checks that the committed polynomial for $x^{(\ell+1)}$ is the same commitment as $y^{(\ell)}$. This is a Merkle root equality check — no information is revealed beyond what the sumchecks already expose.

**Claim derivation.** For each layer's sumcheck, the prover sends the claim $T = \tilde{y}(r)$ (the output polynomial evaluated at the verifier's random row point). The sumcheck proves that $T$ equals the correct output of the matmul by reducing to committed polynomial openings. For the final layer, the verifier independently computes $T$ from the public output. For intermediate layers, the verifier does not need to independently verify $T$ — the end-to-end chain enforces consistency: if the committed intermediate $y^{(\ell)}$ is incorrect, the final layer's sumcheck will fail because the claimed public output $y_{\text{final}}$ won't match the sum computed from incorrect intermediates. This avoids the need for any explicit binding between intermediate claims and their commitments, and avoids revealing intermediate MLE evaluations to the verifier.

**Why independent sumchecks.** An earlier analysis (documented in `docs/analysis/zk-approach-comparison.md`) considered three alternatives for multi-layer ZK proofs: (1) vanishing polynomial masking on chained sumchecks, which breaks at cross-layer junctions because the verifier must learn $\tilde{h}(s)$ to set up the next layer's claim; (2) GKR with Libra-style cross-layer masking, which requires restructuring as a layered circuit (~3$\times$ round overhead); and (3) R1CS + Spartan, which eliminates junctions by encoding everything in one constraint system (~86$\times$ overhead). The per-operation design with committed intermediates avoids all three drawbacks: no junction problem (sumchecks are independent), no circuit restructuring, and no R1CS encoding overhead.

### Step 5: Sumcheck Transcript Masking ($g + \rho \cdot p$)

Vanishing polynomial masking (Step 2) hides the final opened values, but the round polynomials sent during the sumcheck are deterministic functions of the witness and would leak information without additional masking. The XZZ+19 technique addresses this.

The prover generates a random masking polynomial $p(X_1, \ldots, X_b) = a_0 + \sum_{i=1}^{b} p_i(X_i)$, where each $p_i$ is a univariate of degree $d$ (matching the sumcheck round degree, $d = 4$). The prover computes $P = \sum_{c \in \{0,1\}^b} p(c)$ and sends it to the verifier. The verifier picks a random challenge $\rho$. The combined claim becomes $T + \rho \cdot P$.

At each sumcheck round, the prover sends evaluations of $g(X) + \rho \cdot p(X)$ instead of $g(X)$ alone, where $g$ is the honest round polynomial. Since $p$ is random and independent of the witness, the combined round polynomial $g + \rho \cdot p$ is statistically indistinguishable from random.

At the final check, the prover reveals $p(s^*)$ (the masking polynomial evaluated at the sumcheck's terminal point). The verifier checks:

$$Z_W(r, s^*) \cdot Z_x(s^*) + \rho \cdot p(s^*) = \text{current\_claim}$$

Revealing $p(s^*)$ is safe because $p$ is purely random and carries no information about the witness. The degree of the combined polynomial is $\max(d_g, d_p) = 4$, so the number of evaluations per round is unchanged.

### Step 6: tLookup (Non-Arithmetic Operations)

Non-arithmetic operations are proven via tLookup's LogUp protocol, unchanged from the current system:

| Lookup | Table | Witness | Purpose |
|---|---|---|---|
| CDF | $\Phi(d/\sigma)$ for $d \in [0, 2^p)$ | $n \times V$ diffs | Win probabilities |
| Log | $-\log_2(q)$ for $q \in [1, 2^p]$ | $n$ quotients | Surprisal values |

tLookup is already sumcheck-based and runs independently alongside the arithmetic sumchecks. Its witness polynomials are masked and committed via BaseFold in the same way.

### Step 7: BaseFold Openings

All sumcheck final checks (arithmetic + tLookup) require polynomial evaluations at random points. The prover opens BaseFold at these points. BaseFold verification involves:

1. Checking Merkle paths (SHA-256 collision resistance)
2. FRI proximity testing (Reed-Solomon low-degree check)

## Security Analysis

### Soundness

Each sumcheck provides independent soundness via the Schwartz-Zippel lemma. Over $R$ total sumcheck rounds with a 64-bit Goldilocks field ($|\mathbb{F}| = 2^{64} - 2^{32} + 1$):

| Component | Soundness source | Error bound per round |
|---|---|---|
| Arithmetic sumchecks | Schwartz-Zippel lemma | $\leq 4/|\mathbb{F}|$ (degree 4) |
| tLookup (LogUp) | Schwartz-Zippel + grand product | $\leq 2/|\mathbb{F}|$ |
| BaseFold binding | SHA-256 collision resistance | $2^{-128}$ |
| FRI proximity | Reed-Solomon distance | $\leq 2^{-\lambda}$ configurable |

For $R = 100$ sumcheck rounds, total soundness error is at most $4R / 2^{64} \approx 2^{-54}$.

In the interactive setting (verifier sends fresh challenges), the prover cannot grind: the output is committed before challenges are revealed. A single failed proof is a detection event.

### Zero Knowledge

The ZK property rests on three mechanisms:

**1. Vanishing polynomial masking (witness hiding).** Each private polynomial $\tilde{f}$ is masked as $Z_f = \tilde{f} + \sum c_i \cdot X_i(1-X_i)$. At the sumcheck's random evaluation point $s^*$, the verifier sees $Z_f(s^*) = \tilde{f}(s^*) + \sum c_i \cdot s_i^*(1-s_i^*)$. The correction is a random linear combination of the $c_i$ values (which are unknown to the verifier), so $Z_f(s^*)$ is uniformly distributed and reveals nothing about $\tilde{f}(s^*)$. The number of masking coefficients must exceed the number of openings for this to hold (see Step 2).

**2. Sumcheck transcript masking ($g + \rho \cdot p$).** Without masking, the round polynomials sent during each sumcheck are deterministic functions of the witness and leak information. Adding $\rho \cdot p(X)$ (where $p$ is a random sum-of-univariates polynomial and $\rho$ is a verifier challenge) makes the round polynomials statistically indistinguishable from random (XZZ+19). The prover reveals $p(s^*)$ at the final check, which is safe because $p$ is independent of the witness.

**3. BaseFold commitment hiding.** BaseFold commits via Merkle trees over salted evaluations. Openings reveal only the requested evaluation, not surrounding data. The Merkle root itself reveals nothing about the polynomial's values.

**What the verifier sees (complete list per sumcheck):**

| Value | Source | Leaks info? |
|---|---|---|
| $T$ (claim) | Prover sends; proved correct by sumcheck | No — for final layer, derived from public output; for intermediate layers, proved by sumcheck and bound by end-to-end chain |
| $P$ (masking poly sum) | Prover sends | No — random, independent of witness |
| $\rho$ | Verifier's own challenge | No |
| Round polynomials ($b$ rounds $\times$ 5 evals) | Prover sends $g + \rho \cdot p$ | No — masked by $\rho \cdot p$, statistically random |
| Round challenges $\alpha_j$ | Verifier's own challenges | No |
| $Z_W(r, s^*)$, $Z_x(s^*)$ | BaseFold openings | No — masked by vanishing polynomial corrections |
| $p(s^*)$ | Prover reveals | No — random, independent of witness |
| Merkle roots | BaseFold commitments | No — hash outputs, computationally hiding |

**No cross-layer junction problem.** Unlike approaches that chain sumchecks (where the output of one sumcheck feeds as input to the next, forcing the verifier to learn intermediate evaluations), this design uses independent sumchecks bound by commitments. The verifier checks cross-layer consistency by comparing Merkle roots, not by inspecting polynomial evaluations.

## Back-of-the-Envelope Performance Estimates

### Overhead from Masking

The only computational change relative to the current proof system is the degree increase per sumcheck round:

| Metric | Current (unmasked) | Masked |
|---|---|---|
| Degree per round | 2 | 4 |
| Evaluations per round | 3 | 5 |
| Overhead factor | — | $5/3 \approx 1.67\times$ |

The evaluation overhead is $5/3$ per round. However, the correction terms $c_i \cdot X_i(1 - X_i)$ are computationally cheap (one multiply and one add per variable per evaluation point), so the actual wall-clock overhead is closer to $1.5\times$ than $1.67\times$. The number of sumcheck rounds is unchanged ($\log_2 N$ per operation).

### Wall-Clock Estimates (8B Model, 1024-Token Context)

Using the reference point of $\sim$8 minutes for the current proof on BLS12-381, and the measured 9.8$\times$ speedup from switching to Goldilocks arithmetic:

| Component | Current (BLS12-381) | Goldilocks (projected) | Goldilocks + ZK masking |
|---|---|---|---|
| Sumcheck arithmetic | ~8 min | ~49 sec | ~74–82 sec |
| Commitment | Pedersen MSM | BaseFold Merkle | BaseFold Merkle |
| Commitment cost | Expensive (MSM) | Cheap (SHA-256 tree) | Cheap (SHA-256 tree) |
| **Total proof time** | **~8 min** | **~49 sec** | **~1.2–1.4 min** |
| Overhead vs. inference | ~32$\times$ | ~3$\times$ | ~5$\times$ |

The Pedersen MSM that dominates commitment cost in the current system is replaced by Merkle tree construction, which is substantially cheaper. The net effect is that the post-quantum ZK proof may be *faster* than the current non-ZK proof, because the savings from dropping Pedersen MSM outweigh the $1.5\times$ sumcheck overhead.

### Scaling to Larger Models

The overhead factors are multiplicative constants independent of model size:

| Model | Inference | Current proof (est.) | PQ-ZK proof (est.) |
|---|---|---|---|
| 8B, 1K context | ~15 sec | ~8 min | ~1.2 min |
| 70B, 4K context | ~minutes | ~hours | ~hours $\times$ 0.15 |
| 1T, 1M context | ~hours | ~years | ~years $\times$ 0.15 |

The dominant scaling bottleneck remains the $O(n^2)$ self-attention proof, which is inherent to the transformer architecture and unaffected by the ZK/PQ additions.

### Comparison with Spartan-Based Alternative

An earlier version of this analysis considered encoding the entire inference as an R1CS constraint system and proving it via Spartan's two-sumcheck protocol. That approach achieves the same security properties but with substantially higher overhead:

| Metric | Per-operation + masking (this design) | R1CS + Spartan |
|---|---|---|
| Post-quantum | Yes | Yes |
| Zero knowledge | Yes | Yes |
| Sumcheck overhead vs. current | $\sim 1.5\times$ | $\sim 86\times$ |
| Requires R1CS encoding | No | Yes |
| Requires Lasso (sparse evaluation) | No | Yes (critical dependency) |
| Implementation complexity | Incremental | Major rewrite |

The per-operation design avoids Spartan's global sumcheck overhead ($\log_2(\text{total constraints}) \approx 43$ additional rounds over the entire circuit) by keeping sumchecks local to each operation and using commitments for cross-layer binding.

## Key Caveats

1. **BaseFold proof size.** Each BaseFold opening requires a Merkle path and FRI query responses. With many openings (one per sumcheck final check), the total proof size grows. Batching techniques (opening multiple polynomials at a single random point) can mitigate this.

2. **Commitment overhead.** While Merkle tree construction is cheaper than Pedersen MSM, the number of committed polynomials is larger (every intermediate value, not just model weights). The total hashing cost depends on the sum of polynomial sizes across all intermediates.

3. **FRI folding and ZK.** In the non-interactive (Fiat-Shamir) setting, FRI folding can leak information about the committed polynomial (BSCR+19). This requires an additional masking polynomial $R(X)$ added before commitment. In the interactive setting (our design), this is not needed since the verifier provides fresh challenges.

4. **Masking coefficient storage.** The prover must store $k$ random masking coefficients per polynomial, where $k = \log_2 N$ for a polynomial over $N$ elements. For an 8B model with thousands of intermediates, this is negligible compared to the model itself.

5. **The $1.5\times$ estimate assumes the current sumcheck is purely degree-2.** Operations that already have degree 3 (e.g., tLookup's LogUp sumcheck) would increase to degree 5 with masking, a $6/4 = 1.5\times$ overhead — the same ratio. The overhead is consistent across operation types.

## Conclusion

Post-quantum zero-knowledge inference verification is feasible with minimal overhead over the current proof system. By adding vanishing polynomial masking to hide intermediate evaluations and replacing Pedersen commitments with BaseFold (hash-based, post-quantum), the proof achieves both properties without restructuring the existing per-operation sumcheck pipeline. The sumcheck overhead is approximately $1.5\times$ from the degree increase, and the commitment cost decreases (Merkle trees vs. MSM). The main implementation work is integrating BaseFold as the polynomial commitment scheme and adding masking to each sumcheck invocation — both are modular changes to the existing codebase.
