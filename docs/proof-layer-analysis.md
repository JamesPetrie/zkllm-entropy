# Proof Layer Analysis

A reference manual for the proof system in zkllm-entropy (Hyrax/Pedersen variant).
For each proof layer: soundness, completeness, zero-knowledge, and computational cost.

**Scope.** This document analyses the *target state* of the interactive Hyrax/Pedersen
proof system, not the post-quantum BaseFold/Goldilocks path. Where the target state
is not yet fully defined or implemented, the gap is flagged with a ⚠ marker.

**Protocol model.** Interactive: the verifier sends fresh uniformly random challenges
each round. Fiat-Shamir is not applied. This simplifies the ZK argument (no
transcript circularity) but means proofs are not transferable.

**References.**

- **Hyrax:** Wahby, Tzialla, Shelat, Thaler, Walfish. "Doubly-Efficient zkSNARKs
  Without Trusted Setup." 2018 IEEE S&P. ePrint 2017/1132.
- **zkLLM:** Sun, Li, Xu, Zhang. "zkLLM: Zero Knowledge Proofs for Large Language
  Models." ACM CCS 2024. arXiv:2404.16109.
- **Libra:** Xie, Zhang, Song. "Libra: Succinct Zero-Knowledge Proofs with Optimal
  Prover Computation." CRYPTO 2019.
- **LogUp:** Haböck. "Multivariate lookups based on logarithmic derivatives."
  ePrint 2022/1530.

---

## 1. Notation and Definitions

| Symbol | Meaning |
|--------|---------|
| $\mathbb{F}$ | BLS12-381 scalar field, order $r \approx 2^{255}$ |
| $\mathbb{G}_1$ | BLS12-381 $G_1$ group (256-bit Jacobian coordinates) |
| $N$ | vector length (context-dependent) |
| $n$ | $= \lceil\log_2 N\rceil$, number of sumcheck rounds |
| $T$ | sequence length (number of token positions) |
| $V$ | vocabulary size |
| $d$ | degree of sumcheck round polynomial |
| $K$ | number of tensors in multi-Hadamard product |
| $\gamma$ | quantization scaling factor (Rescaling) |
| MLE | multilinear extension |
| $\tilde{f}$ | MLE of a function $f : \{0,1\}^n \to \mathbb{F}$ |

**Operation cost symbols.** Throughout this document:

| Symbol | Operation | Relative cost (approx.) |
|--------|-----------|------------------------|
| $\mathsf{F{\cdot}}$ | field multiplication in $\mathbb{F}$ | 1 |
| $\mathsf{F{+}}$ | field addition in $\mathbb{F}$ | ~0.05 |
| $\mathsf{F^{-1}}$ | field inversion | ~100 |
| $\mathsf{G_smul}$ | $\mathbb{G}_1$ scalar multiplication (255-bit) | ~4000 |
| $\mathsf{G_{add}}$ | $\mathbb{G}_1$ point addition | ~15 |
| $\mathsf{MSM}(k)$ | multi-scalar multiplication of $k$ points | $\sim k \cdot \mathsf{G_smul} / \log_2 k$ (Pippenger) |

All costs are normalised to $\mathsf{F{\cdot}} = 1$.  The relative costs are for
BLS12-381 on an H100 (GPU field arithmetic via the blst library).  $\mathsf{G_smul}$
dominates; a single scalar-mul is ~4000 field multiplications due to the 255-bit
double-and-add chain over 384-bit coordinates.

**Multilinear extension.**  For $f : \{0,1\}^n \to \mathbb{F}$, the unique
multilinear polynomial $\tilde{f} : \mathbb{F}^n \to \mathbb{F}$ satisfying
$\tilde{f}(x) = f(x)$ for all $x \in \{0,1\}^n$ is

$$\tilde{f}(x_1, \ldots, x_n) = \sum_{w \in \{0,1\}^n} f(w) \prod_{i=1}^{n} \big(w_i x_i + (1 - w_i)(1 - x_i)\big)$$

**Schwartz-Zippel lemma** (Schwartz 1980; Zippel 1979).  If
$p \in \mathbb{F}[x_1, \ldots, x_n]$ is a non-zero polynomial of total degree $\leq d$,
and $r_1, \ldots, r_n$ are chosen uniformly and independently from $\mathbb{F}$, then
$\Pr[p(r_1, \ldots, r_n) = 0] \leq d / |\mathbb{F}|$.

This lemma is the primary tool for converting algebraic identities into probabilistic
checks.  It is used throughout: the sumcheck protocol (§5), the LogUp identity (§6),
the division relation (§8), and all Schwartz-Zippel point-evaluation checks.

---

## 2. Pedersen Vector Commitment

**Reference:** Hyrax §3.1 (Wahby et al. 2018, ePrint 2017/1132, p. 4).

### 2.1 Construction

**Setup (hash-to-curve, no toxic waste).**  Given a domain-separation tag $\mathsf{DST}$
and vector length $N$, derive $N + 2$ independent $\mathbb{G}_1$ generators via
RFC 9380 hash-to-curve:

$$G_1, \ldots, G_N, H, U \leftarrow \mathsf{HashToCurve}(\mathsf{DST}, 1), \ldots, \mathsf{HashToCurve}(\mathsf{DST}, N{+}2)$$

No party knows any pairwise discrete logarithm.

**Commit.**  For message $\mathbf{t} = (t_1, \ldots, t_N) \in \mathbb{F}^N$ and
blinding $\rho \xleftarrow{\$} \mathbb{F}$:

$$C = \sum_{i=1}^{N} t_i \cdot G_i + \rho \cdot H$$

### 2.2 Properties

**Completeness.**  Trivial: honest evaluation of the formula always produces a valid
commitment.

**Binding (computational).**  Hyrax Theorem 7 (Wahby et al. 2018, Appendix A):

> The Pedersen commitment scheme is a non-interactive commitment scheme assuming
> the hardness of the discrete logarithm problem in $\mathcal{G}$.

More precisely, Hyrax Definition 4 requires **computational binding**: for every
(non-uniform) PPT $A$, the probability that $A$ produces two distinct openings of
the same commitment is negligible.  Hash-to-curve generators make this
unconditional modulo the random oracle model — no known dlog relations exist.

The zkLLM commitment scheme (Sun et al. 2024, §3.4) requires:

> **(Soundness)** if $y \neq \tilde{S}(v)$, then the output is False with
> $1 - \mathsf{negl}(\lambda)$ probability.

This binding property ensures that a committed tensor cannot be opened to a
different evaluation, which underpins the soundness of every proof layer that
relies on committed tensors.

**Hiding (perfect, unconditional).**  Hyrax Definition 4 requires **perfect hiding**:

> For any $\mathsf{pp} \in \{0,1\}^{*}$ and $m_0, m_1 \in \{0,1\}^{*}$ where
> $|m_0| = |m_1|$, the ensembles $\{\mathsf{Com}_{\mathsf{pp}}(m_0)\}_{n \in \mathbb{N}}$
> and $\{\mathsf{Com}_{\mathsf{pp}}(m_1)\}_{n \in \mathbb{N}}$ are identically distributed.

In our setting: for any two messages $\mathbf{t}_0 \neq \mathbf{t}_1$,
the distributions $\{C(\mathbf{t}_0; \rho)\}_{\rho \leftarrow \mathbb{F}}$ and
$\{C(\mathbf{t}_1; \rho)\}_{\rho \leftarrow \mathbb{F}}$ are identical (uniform
over $\mathbb{G}_1$).

The zkLLM commitment scheme additionally requires (Sun et al. 2024, §3.4):

> **(Zero-knowledge)** the verifier learns no information beyond $y = \tilde{S}(v)$.

**Homomorphism.**  Hyrax Definition 5 (additive homomorphism):

> Given $\mathsf{Com}(x; s_x)$ and $\mathsf{Com}(y; s_y)$, there is an operator
> $\odot$ such that $\mathsf{Com}(x; s_x) \odot \mathsf{Com}(y; s_y) = \mathsf{Com}(x + y; s_x + s_y)$.

In our additive $\mathbb{G}_1$ notation:
$C(\mathbf{t}_1; \rho_1) + C(\mathbf{t}_2; \rho_2) = C(\mathbf{t}_1 + \mathbf{t}_2; \rho_1 + \rho_2)$.
This is used extensively by the ZK sumcheck to fold commitments across rounds without
the prover revealing coefficients.

**Completeness** (zkLLM §3.4):

> **(Completeness)** if $(y, \pi) = \mathsf{ProveEval}(\ldots)$, then the output is True.

That is, an honest prover can always produce a valid opening proof for any evaluation
of a committed polynomial.

### 2.3 Cost

**Prover (commit):**

| Operation | Count | Notes |
|-----------|-------|-------|
| $\mathsf{MSM}(N{+}1)$ | 1 | $N$ message generators + 1 hiding generator |
| Random sampling | 1 | blinding $\rho$ |

Dominant cost: $\mathsf{MSM}(N{+}1)$.  With Pippenger's algorithm this is approximately
$(N{+}1) \cdot 255 / \log_2(N{+}1)$ $\mathsf{G_{add}}$ operations.

For the row-structured Hyrax layout (§6.1) with matrix of $R$ rows of length $L$
(where $N = R \cdot L$), each row is committed independently:

| Operation | Count |
|-----------|-------|
| $\mathsf{MSM}(L{+}1)$ | $R$ |
| Random sampling | $R$ |

**Verifier (recompute commitment for verification):** same cost as prover.
In practice the verifier stores the commitment and only recomputes when checking
an opening.

### 2.4 Implementation

`src/commit/commitment.cuh`: class `Commitment` with `hiding_random(size)` factory,
`commit_hiding()`, `save_hiding()`, `load_hiding()`.  Hash-to-curve in
`src/field/hash_to_curve.cu`.

---

## 3. Hyrax §A.1 Σ-Protocols

**Reference:** Hyrax §A.1 Figure 5 (Wahby et al. 2018, ePrint 2017/1132, p. 17).

Hyrax defines two Σ-protocol building blocks in Appendix A.1, Figure 5: a
proof-of-opening (proving knowledge of a committed scalar) and a proof-of-equality
(proving two commitments hide the same value).  These are standard Schnorr-style
protocols with special soundness and perfect honest-verifier zero-knowledge (HVZK).
They are composed into the ZK sumcheck (§5) and ZK opening (§4).

### 3.1 Proof-of-Opening

**Statement:** Given $C \in \mathbb{G}_1$, prover knows $(m, \rho)$ such that
$C = m \cdot U + \rho \cdot H$.

**Transcript:**

1. P → V: $A = s_m \cdot U + s_\rho \cdot H$ with fresh $s_m, s_\rho \xleftarrow{\$} \mathbb{F}$
2. V → P: challenge $e \xleftarrow{\$} \mathbb{F}$
3. P → V: $z_m = s_m + e \cdot m$, $z_\rho = s_\rho + e \cdot \rho$

**Verification:** $z_m \cdot U + z_\rho \cdot H \stackrel{?}{=} A + e \cdot C$

Hyrax Theorem 8 (Wahby et al. 2018, Appendix A):

> proof-of-opening *is complete, honest-verifier perfect ZK, and special sound
> under the discrete log assumption.*

**Completeness.**  Substituting honest responses:
$z_m \cdot U + z_\rho \cdot H = (s_m + em) \cdot U + (s_\rho + e\rho) \cdot H = A + e \cdot C$. ✓

**Soundness (special soundness).**  From two accepting transcripts $(A, e, z_m, z_\rho)$
and $(A, e', z_m', z_\rho')$ with $e \neq e'$, extract:

$$m = \frac{z_m - z_m'}{e - e'}, \qquad \rho = \frac{z_\rho - z_\rho'}{e - e'}$$

This is a valid opening of $C$.  Soundness error: $1/|\mathbb{F}| \approx 2^{-255}$.

**Zero-knowledge (perfect HVZK).**  Simulator: pick $z_m, z_\rho \xleftarrow{\$} \mathbb{F}$,
set $A = z_m \cdot U + z_\rho \cdot H - e \cdot C$.  The simulated transcript
$(A, e, z_m, z_\rho)$ is distributed identically to an honest transcript.

**Cost:**

| | Prover | Verifier |
|-|--------|----------|
| $\mathsf{G_smul}$ | 2 (compute $A$) | 3 ($z_m \cdot U + z_\rho \cdot H$, $e \cdot C$) |
| $\mathsf{G_{add}}$ | 1 | 2 |
| $\mathsf{F{\cdot}}$ | 2 ($e \cdot m$, $e \cdot \rho$) | 0 |
| $\mathsf{F{+}}$ | 2 | 0 |
| Communication | $1\ \mathbb{G}_1 + 2\ \mathbb{F}$ | — |

### 3.2 Proof-of-Equality

**Statement:** Given $C_1, C_2 \in \mathbb{G}_1$, prover knows $(m, \rho_1, \rho_2)$
such that $C_1 = m \cdot U + \rho_1 \cdot H$ and $C_2 = m \cdot U + \rho_2 \cdot H$.

Reduces to a Schnorr proof on $C_1 - C_2 = (\rho_1 - \rho_2) \cdot H$.

**Transcript:**

1. P → V: $A = s \cdot H$ with fresh $s \xleftarrow{\$} \mathbb{F}$
2. V → P: challenge $e \xleftarrow{\$} \mathbb{F}$
3. P → V: $z = s + e \cdot (\rho_1 - \rho_2)$

**Verification:** $z \cdot H \stackrel{?}{=} A + e \cdot (C_1 - C_2)$

Hyrax Theorem 9 (Wahby et al. 2018, Appendix A):

> proof-of-equality *is complete, honest-verifier perfect zero-knowledge, and
> special sound under the discrete log assumption.*

**Completeness, soundness, ZK:** Same structure as proof-of-opening but with a
single witness coordinate.  Soundness error $1/|\mathbb{F}|$.  Perfect HVZK.

**Cost:**

| | Prover | Verifier |
|-|--------|----------|
| $\mathsf{G_smul}$ | 1 | 2 ($z \cdot H$, $e \cdot (C_1 - C_2)$) |
| $\mathsf{G_{add}}$ | 0 | 2 ($C_1 - C_2$, $A + e \cdot \Delta$) |
| $\mathsf{F{\cdot}}$ | 1 | 0 |
| $\mathsf{F{+}}$ | 2 ($\rho_1 - \rho_2$, $s + e \cdot \Delta\rho$) | 0 |
| Communication | $1\ \mathbb{G}_1 + 1\ \mathbb{F}$ | — |

### 3.3 Implementation

`src/proof/hyrax_sigma.cu`: `prove_opening`, `verify_opening`, `prove_equality`,
`verify_equality`.

---

## 4. ZK Opening (Hyrax §A.2 Figure 6)

**Reference:** Hyrax §A.2 Figure 6 (Wahby et al. 2018, ePrint 2017/1132, p. 18),
composed with §6.1 row reduction.

This protocol lets the prover open a Pedersen-committed polynomial evaluation
$v = \tilde{f}(u)$ without revealing the witness $\mathbf{t}$ (the polynomial's
coefficient vector).  Hyrax §A.2 describes this as a zero-knowledge proof of a
dot-product relation between a committed vector and a public vector, yielding a
public inner-product value.  It is used to discharge weight claims produced by zkFC.

Hyrax §5 (Wahby et al. 2018, p. 7) describes the dot-product proof as the key
building block:

> Our starting point is an existing protocol for multi-commitments, which we call
> proof-of-dot-prod.  With this protocol, a prover that knows the openings of two
> commitments, one to a vector $\vec{x} = (x_1, \ldots, x_n) \in \mathbb{F}^n$
> and one to a scalar $y \in \mathbb{F}$, can prove in zero-knowledge that
> $y = \langle \vec{a}, \vec{x} \rangle$ for a public $\vec{a} \in \mathbb{F}^n$.
> The protocol is defined in Appendix A.2.

The §6.1 row reduction (Hyrax §6.1) reduces a commitment over $N = R \times L$
elements (stored as $R$ row commitments of length $L$) to a single row commitment
via a random linear combination using the multilinear eq-polynomial coefficients.
This converts an $N$-element opening into an $L$-element opening plus an
$R$-element MSM.

### 4.1 Protocol

**Setup.**  Commitment generators $\{G_1, \ldots, G_L\}$, hiding generator $H$,
inner-product target $U$.  Row-structured commitment: $R$ rows of length $L$, where
$C_i = \sum_j t_{i,j} \cdot G_j + \rho_i \cdot H$.

**Evaluation.**  The evaluation point $u \in \mathbb{F}^n$ splits into row indices
(last $\log_2 R$ coordinates) and within-row indices (first $\log_2 L$ coordinates).
The §6.1 row reduction folds $R$ row commitments into a single commitment
$\xi = \sum_i a_i \cdot C_i$ using coefficients $a_i = \tilde{eq}(u_{\text{row}}, i)$,
with corresponding blinding $r_\xi = \sum_i a_i \cdot \rho_i$ and witness
$\hat{x}_j = \sum_i a_i \cdot t_{i,j}$.

**Figure 6 protocol (after row reduction):**

1. P computes $\hat{x} \in \mathbb{F}^L$ (folded row), $r_\xi$ (folded blinding),
   $v = \langle \hat{a}, \hat{x} \rangle$ where $\hat{a}_j = \tilde{eq}(u_{\text{col}}, j)$.
2. P → V: $\delta = \sum_j d_j \cdot G_j + r_\delta \cdot H$ (masking commitment),
   $\beta = \langle \hat{a}, \mathbf{d} \rangle \cdot U + r_\beta \cdot H$ (masked inner product),
   $\tau = v \cdot U + r_\tau \cdot H$ (evaluation commitment).
3. V → P: challenge $c \xleftarrow{\$} \mathbb{F}$.
4. P → V: $\mathbf{z} = c \cdot \hat{x} + \mathbf{d}$, $z_\delta = c \cdot r_\xi + r_\delta$,
   $z_\beta = c \cdot r_\tau + r_\beta$, and $r_\tau$.

**Verification (4 checks):**

1. $\sum_j z_j \cdot G_j + z_\delta \cdot H \stackrel{?}{=} c \cdot \xi + \delta$
   (opening consistency)
2. $\langle \hat{a}, \mathbf{z} \rangle \cdot U + z_\beta \cdot H \stackrel{?}{=} c \cdot \tau + \beta$
   (inner product consistency)
3. $\tau \stackrel{?}{=} v \cdot U + r_\tau \cdot H$ (value binding)
4. Checks 1-3 all pass.

### 4.2 Properties

Hyrax Theorem 11 (Wahby et al. 2018, Appendix A.2):

> The protocol of Figure 6 *is complete, honest-verifier perfect zero-knowledge,
> and special sound under the discrete log assumption.*

**Completeness.**  Direct substitution shows all equations hold for an honest prover.

**Soundness.**  Special soundness: from two transcripts with challenges $c \neq c'$, the
extractor recovers $\hat{x}$, $r_\xi$, and the evaluation $v$.  The value binding check
(check 3) ensures $v$ is the claimed evaluation, not an arbitrary value.  Soundness
error: $1/|\mathbb{F}|$.

**Zero-knowledge (HVZK).**  The simulator picks $\mathbf{z}, z_\delta, z_\beta$ uniformly,
then computes $\delta, \beta$ from the verification equations.  The simulated
transcript is identically distributed because $\mathbf{d}$ is a uniform mask.  The
only value the verifier learns is $v$ (the evaluation), which is the intended
public output.

### 4.3 Cost

Let $L$ = row length, $R$ = number of rows.

**Prover:**

| Operation | Count | Notes |
|-----------|-------|-------|
| $\mathsf{MSM}(R)$ | 1 | Row reduction: $\xi = \sum a_i C_i$ |
| $\mathsf{MSM}(L{+}1)$ | 1 | Compute $\delta$ |
| $\mathsf{G_smul}$ | 3 | $\beta$ (2 terms), $\tau$ (2 terms — shared with 1 $\mathsf{G_smul}$) |
| $\mathsf{F{\cdot}}$ | $R + L + 4$ | Row reduction $a_i$, inner product, response computation |
| Communication | $3\ \mathbb{G}_1 + (L{+}3)\ \mathbb{F}$ | $\delta, \beta, \tau, \mathbf{z}, z_\delta, z_\beta, r_\tau$ |

**Verifier:**

| Operation | Count | Notes |
|-----------|-------|-------|
| $\mathsf{MSM}(R)$ | 1 | Row reduction: $\xi$ |
| $\mathsf{MSM}(L)$ | 1 | Check 1: $\sum z_j G_j$ |
| $\mathsf{G_smul}$ | 5 | $z_\delta H$, $c \xi$, $c \tau$, $v U$, $r_\tau H$ |
| Inner product | $L$ | $\langle \hat{a}, \mathbf{z} \rangle$ |

Dominant cost: $\mathsf{MSM}(L) + \mathsf{MSM}(R)$ for both prover and verifier.

The communication cost of proof-of-dot-prod is further reduced by the Bulletproofs
recursive halving technique (Bünz et al. 2018).  Hyrax Lemma 12 (Wahby et al. 2018,
Appendix A.3):

> The protocol of Figures 7–8 is complete, honest-verifier ZK, and has
> witness-extended emulation under the discrete log assumption.

This reduces communication from $O(L)$ to $O(\log L)$ elements, at the cost of
$O(\log L)$ rounds of interaction.

### 4.4 Implementation

`src/commit/commitment.cuh`: `Commitment::open_zk`, `Commitment::verify_zk`.
`src/proof/proof.cu`: `verifyWeightClaimZK` (wrapper that samples challenge, calls
`open_zk` then `verify_zk`).

---

## 5. ZK Sumcheck (Hyrax §4 Protocol 3)

**Reference:** Hyrax §4 Protocol 3 (Wahby et al. 2018, ePrint 2017/1132, pp. 11-13);
implementation in `src/proof/zk_sumcheck.cu`.

The ZK sumcheck wraps the standard sumcheck protocol by Pedersen-committing each
round polynomial instead of sending its coefficients in the clear.  Hyrax §4
describes this transformation (Wahby et al. 2018, p. 6):

> This section uses abstract commitments having a homomorphism property (§3.1).
> We also make black-box use of three sub-protocols, which operate on commitments:
> *proof-of-opening*($C$) convinces $\mathcal{V}$ that $\mathcal{P}$ can open $C$.
> *proof-of-equality*($C_0$, $C_1$) convinces $\mathcal{V}$ that $C_0$ and $C_1$
> commit to the same value, and that $\mathcal{P}$ can open both.

The key insight is that Pedersen's homomorphism allows the verifier to check the
sumcheck identity $g_j(0) + g_j(1) = g_{j-1}(r_{j-1})$ entirely in the commitment
space, without learning the polynomial coefficients.  Hyrax §4 describes the
concrete mechanism (p. 6):

> In round $j$ of the sum-check, $\mathcal{P}$ commits to
> $s_j(t) = c_{0,j} + c_{1,j}t + c_{2,j}t^2 + c_{3,j}t^3$, via
> $\delta_{c_{0,j}} \leftarrow \mathsf{Com}(c_{0,j})$,
> $\delta_{c_{1,j}} \leftarrow \mathsf{Com}(c_{1,j})$, [...] and $\mathcal{P}$
> and $\mathcal{V}$ execute proof-of-opening for each one. [...] $\mathcal{V}$ can
> use the homomorphism property to compute the required commitments: for
> $s_j(0) + s_j(1) = 2c_{0,j} + c_{1,j} + c_{2,j} + c_{3,j}$, $\mathcal{V}$
> computes $\delta_{c_{0,j}}^2 \odot \delta_{c_{1,j}} \odot \delta_{c_{2,j}} \odot \delta_{c_{3,j}}$.

(The codebase uses degree-2 round polynomials for inner-product sumcheck, so only
3 coefficients per round rather than 4.)

### 5.0 Plain Sumcheck (Background)

The standard sumcheck protocol for the claim $S = \sum_{x \in \{0,1\}^n} g(x)$
where $g$ is a degree-$d$ polynomial in each variable:

1. For $j = 1, \ldots, n$: prover sends univariate $g_j(X_j)$ (degree $\leq d$,
   specified by $d+1$ coefficients).
2. Verifier checks $g_j(0) + g_j(1) = g_{j-1}(r_{j-1})$ (where $g_0(r_0) := S$).
3. Verifier sends random $r_j \xleftarrow{\$} \mathbb{F}$.
4. After round $n$: verifier checks $g_n(r_n) = g(r_1, \ldots, r_n)$ using an oracle
   query (or a polynomial opening).

**Soundness:** A cheating prover must pass $g_j(0) + g_j(1) = g_{j-1}(r_{j-1})$ at
each round.  If the prover deviates, the check fails with probability $\geq 1 - d/|\mathbb{F}|$
per round.  Over $n$ rounds: soundness error $\leq n \cdot d / |\mathbb{F}|$.

**Completeness:** Honest evaluation makes all checks pass identically.

**ZK:** None — round polynomials leak information about $g$.

### 5.1 ZK Wrapper (Hyrax §4)

The wrapper replaces "send $g_j(X)$ in the clear" with:

**Per round $j$:**

1. **Commit:** Prover Pedersen-commits each coefficient $c_{k,j}$ of $g_j(X) = \sum_{k=0}^{d} c_{k,j} X^k$:

   $$T_{k,j} = c_{k,j} \cdot U + \rho_{k,j} \cdot H, \qquad k = 0, \ldots, d$$

   where $\rho_{k,j} \xleftarrow{\$} \mathbb{F}$ are fresh per round.

2. **Per-coefficient openings:** For each $k$, emit a §A.1 proof-of-opening for
   $T_{k,j}$ with challenge $e_{k,j}$.  This allows the extractor to lift the
   round polynomial (Hyrax §4, knowledge soundness requirement).

3. **Round-to-round equality:** Compute the LHS commitment
   $C_{\text{LHS}} = \sum_k \alpha_k \cdot T_{k,j}$
   where $\alpha$ encodes the sumcheck identity (see per-variant sections below).
   Emit a §A.1 proof-of-equality between $C_{\text{LHS}}$ and the previous-round
   evaluation commitment $T_{j-1}(r_{j-1})$.

4. **Next-round input:** $T_j(r_j) = \sum_k r_j^k \cdot T_{k,j}$ (homomorphic
   evaluation), with blinding $\rho_j(r_j) = \sum_k r_j^k \cdot \rho_{k,j}$.

**Top-level handoff (§4 Step 1):**  The verifier constructs the initial commitment
$C_0 = S \cdot U$ from the public claim $S$.  No prover commitment, no proof-of-opening.
Blinding is 0 because $S$ is public.  The first round's equality proof ties
$\sum_k \alpha_k \cdot T_{k,1}$ to $C_0$, strictly binding the chain to $S$.

**Final claim:**  After $n$ rounds, $T_{\text{final}}$ commits to $g_n(r_n)$.  The
caller discharges this via a §4 ZK opening (§4) or a weight claim.

### 5.2 Properties (All Variants)

Hyrax Lemma 4 (Wahby et al. 2018, §5):

> The protocol of Figure 1 is a complete, honest-verifier *perfect* ZK argument,
> with witness-extended emulation under the discrete log assumption, that its
> inputs constitute an accepting sum-check relation: on input a commitment $C_0$,
> commitments $\{a_j\}$ to polynomials $\{s_j\}$ in a sum-check invocation, rows
> $\{M_k\}$ of the matrix of Equation (5), and commitments $X = \mathsf{Com}(v_0)$,
> $Y = \mathsf{Com}(v_1)$, and $Z$, where $\{r_j\}$ are $\mathcal{V}$'s coins from
> the sum-check.

This is the formal security statement for the ZK sumcheck protocol as used in
Hyrax.  The "witness-extended emulation" property is stronger than standard
special soundness — it guarantees extraction of the witness from any convincing
prover.

**Completeness.**  The Pedersen homomorphism guarantees that honest commitments
satisfy the equality check: $\sum_k \alpha_k \cdot T_{k,j}$ commits to
$\sum_k \alpha_k \cdot c_{k,j}$ under blinding $\sum_k \alpha_k \cdot \rho_{k,j}$,
which equals $g_j(0) + g_j(1)$ by the sumcheck identity.  This matches the
previous-round evaluation commitment.

**Soundness.**  Composition of:

1. **Per-coefficient opening soundness:** Each proof-of-opening has soundness error
   $1/|\mathbb{F}|$.  Over $n(d{+}1)$ openings: error $\leq n(d{+}1)/|\mathbb{F}|$.
2. **Round-to-round equality soundness:** Each equality proof has error
   $1/|\mathbb{F}|$.  Over $n$ rounds: error $\leq n/|\mathbb{F}|$.
3. **Sumcheck soundness:** Given the extracted coefficients (from the openings), the
   standard sumcheck soundness applies: error $\leq nd/|\mathbb{F}|$.

Total soundness error: $\leq n(2d{+}2)/|\mathbb{F}|$.  For $n \leq 25$, $d \leq 4$:
error $\leq 250 / 2^{255} \approx 2^{-247}$.

**Zero-knowledge.**  The round polynomials are never revealed — only Pedersen
commitments to their coefficients.  Since Pedersen is perfectly hiding, the
commitments leak zero information.  The Σ-protocol responses are simulated via
the HVZK simulators (§3).  Therefore the transcript reveals nothing beyond what the
verifier could compute from the public claim $S$ and the verifier's own challenges.

⚠ **Formal simulation proof (Phase 5) is not yet written.**  The argument above
is informal.  A formal simulator must show that the full transcript (commitments,
Σ-protocol responses, challenges) is simulatable given only the public inputs.

The simulation structure would follow the pattern of zkLLM Theorem 7.4 (Sun et al.
2024): define a Real game (actual protocol execution) and an Ideal game (simulator
with access only to public inputs and the commitment oracle), then show
computational indistinguishability.  The simulator commits to random values (hiding
property ensures indistinguishability), then programs the Σ-protocol responses via
the HVZK simulators from §3.

### 5.3 Inner-Product Sumcheck

**Claim:** $S = \sum_{x \in \{0,1\}^n} a(x) \cdot b(x) = \langle \tilde{a}, \tilde{b} \rangle_{\{0,1\}^n}$

**Round polynomial:** degree 2.  Coefficients $(c_0, c_1, c_2)$ where
$g_j(X) = c_0 + c_1 X + c_2 X^2$.

**Round identity weights:** $\alpha = (2, 1, 1)$ (standard sumcheck:
$g_j(0) + g_j(1) = 2c_0 + c_1 + c_2$).

**Per-round kernel (`Fr_ip_sc_step`):**  For pairs $(a_{2i}, a_{2i+1})$, $(b_{2i}, b_{2i+1})$:

$$c_0[i] = a_{2i} \cdot b_{2i}, \quad c_1[i] = a_{2i}(b_{2i+1} - b_{2i}) + b_{2i}(a_{2i+1} - a_{2i}), \quad c_2[i] = (a_{2i+1} - a_{2i})(b_{2i+1} - b_{2i})$$

Then $c_k = \sum_i c_k[i]$.

**Final claim:** After $n$ rounds, tensors collapse to singletons $a(r)$, $b(r)$.
These are discharged via ZK opening (§4) against the corresponding weight
commitments.

**Cost per round (prover):**

| Operation | Count | Notes |
|-----------|-------|-------|
| $\mathsf{F{\cdot}}$ (kernel) | $3 \cdot N_j/2 + 3 \cdot N_j/2$ | Coefficient computation + .mont() correction |
| $\mathsf{F{+}}$ (kernel) | $5 \cdot N_j/2$ | Subtractions and additions |
| $\mathsf{F{+}}$ (sum) | $3 \cdot (N_j/2 - 1)$ | Sum 3 coefficient tensors |
| $\mathsf{F{\cdot}}$ (fold) | $2 \cdot N_j/2$ | Fold $a$ and $b$ with $r_j$ |
| $\mathsf{G_smul}$ | $3 \times 2 + 3 \times 2 + 1 + 3 + 1 = 19$ | 3 commits (6), 3 openings (6), equality (1), weights (3), fold (3) |
| $\mathsf{G_{add}}$ | 11 | Commitment and fold additions |
| Communication | $3\ \mathbb{G}_1 + 3 \times (1\ \mathbb{G}_1 + 2\ \mathbb{F}) + (1\ \mathbb{G}_1 + 1\ \mathbb{F})$ | 3 commitments, 3 opening proofs, 1 equality proof |

where $N_j = N / 2^j$ is the tensor size at round $j$.

**Total cost (prover, $n$ rounds):**

| Operation | Total |
|-----------|-------|
| $\mathsf{F{\cdot}}$ (field) | $\sum_{j=0}^{n-1} (6 \cdot N/2^{j+1} + 2 \cdot N/2^{j+1}) = 8(N-1) \approx 8N$ |
| $\mathsf{G_smul}$ | $19n$ |
| $\mathsf{G_{add}}$ | $11n$ |
| Communication | $n \cdot (7\ \mathbb{G}_1 + 7\ \mathbb{F})$ per round |

At $n = 15$ (for $N = 32768$): ~120K $\mathsf{F{\cdot}}$, 285 $\mathsf{G_smul}$.
The group operations dominate: $285 \times 4000 \approx 1.1\text{M}$ field-mul
equivalent.

**Total cost (verifier, $n$ rounds):**

| Operation | Per round | Total |
|-----------|-----------|-------|
| $\mathsf{G_smul}$ | 17 (3×3 openings + 3 weights + 2 equality + 3 fold) | $17n$ |
| $\mathsf{G_{add}}$ | 12 | $12n$ |

### 5.4 Hadamard-Product Sumcheck (Eq-Factored)

**Claim:** $S = \sum_{x \in \{0,1\}^n} \tilde{eq}(x, u) \cdot a(x) \cdot b(x) = (\tilde{a} \circ \tilde{b})(u)$

This is the eq-factored variant from Libra (Xie, Zhang, Song, CRYPTO 2019,
Appendix A).  The Libra optimization factors $\tilde{eq}$ out of the round
polynomial: since $\tilde{eq}(x, u) = \prod_i eq(x_i, u_i)$ and each round fixes
one variable, the eq-factor peels off one term per round, reducing the degree of the
polynomial the prover must commit.  The round polynomial
$g_j(X) = eq(X, u_j) \cdot h_j(X)$ factors into an eq-weight and a degree-2
polynomial $h_j$.  The prover sends only $h_j$'s 3 coefficients.

**Round identity weights:** $\alpha = (1, u_j, u_j)$ where $u_j$ is the eq-factor
challenge for this round.  The identity is:
$(1 - u_j) \cdot h_j(0) + u_j \cdot h_j(1) = h_{j-1}(v_{j-1})$, which in coefficient
form is $c_0 + u_j \cdot c_1 + u_j \cdot c_2 = h_{j-1}(v_{j-1})$.

**Two challenge sequences:**

- $u = (u_1, \ldots, u_n)$: eq-factor point (where the claim is evaluated)
- $v = (v_1, \ldots, v_n)$: sumcheck fold challenges

**Cost:** Same structure as inner-product; the only difference is the per-round
$\alpha$ weights.  Field cost is slightly higher due to the partial MLE evaluations
at $u_{[j+1:]}$ per round:

| Operation | Total |
|-----------|-------|
| $\mathsf{F{\cdot}}$ (field) | $\approx 8N + 3N$ (extra partial MLE) $= 11N$ |
| $\mathsf{G_smul}$ | $19n$ |

### 5.5 Binary Sumcheck

**Claim:** $S = \sum_{x \in \{0,1\}^n} a(x) \cdot (a(x) - 1) = 0$ (proves $a$ is binary).

Round polynomial degree 2.  Uses a corrected kernel (`Fr_bin_sc_step_zk`) that
applies Montgomery normalization inline.

**Round identity weights:** $\alpha = (2, 1, 1)$ (standard).

**Per-round kernel:**

$$c_0[i] = a_{2i}(a_{2i} - 1), \quad c_1[i] = (a_{2i+1} - a_{2i})(2a_{2i} - 1), \quad c_2[i] = (a_{2i+1} - a_{2i})^2$$

**Cost:** Same group operation count as inner-product (19 $\mathsf{G_smul}$ per round).
Field cost is lower: only one tensor to fold.

| Operation | Total |
|-----------|-------|
| $\mathsf{F{\cdot}}$ (field) | $\approx 5N$ |
| $\mathsf{G_smul}$ | $19n$ |

### 5.6 Multi-Hadamard Sumcheck (Degree $K$)

**Claim:** $S = \sum_{x \in \{0,1\}^n} \tilde{eq}(x, u) \cdot \prod_{k=1}^{K} X_k(x)$

Round polynomial factors as $g_j(X) = eq(X, u_{\text{last}}) \cdot h_j(X)$ where
$h_j$ has degree $K$ (not 2).  The prover commits $K{+}1$ coefficients per round.

**Round identity weights:** $\alpha = (1, u_{\text{last}}, u_{\text{last}}, \ldots, u_{\text{last}})$
of length $K{+}1$.

**Sigma challenges per round:** $K{+}2$ ($K{+}1$ openings + 1 equality).

**Per-round kernel:** Running-product construction over $K$ tensors, same as
`multi_hadamard_sumchecks` in `proof.cu`.  Cost per round:

| Operation | Count |
|-----------|-------|
| $\mathsf{F{\cdot}}$ (coefficient building) | $O(K^2 \cdot N_j/2)$ |
| $\mathsf{F{\cdot}}$ (partial MLE) | $K \cdot N_j/2$ |
| $\mathsf{G_smul}$ (ZK wrapper) | $2(K{+}1) + 2(K{+}1) + 1 + (K{+}1) + (K{+}1)$ |

Simplifying the group ops: $(K{+}1)$ commits at 2 $\mathsf{G_smul}$ each = $2(K{+}1)$;
$(K{+}1)$ opening proofs at 2 $\mathsf{G_smul}$ each = $2(K{+}1)$; 1 equality at
1 $\mathsf{G_smul}$; weighted combination at $(K{+}1)$ $\mathsf{G_smul}$; fold at
$(K{+}1)$ $\mathsf{G_smul}$.  Total per round: $6(K{+}1) + 1$ $\mathsf{G_smul}$.

**Total ($n$ rounds, $K$ tensors):**

| Operation | Total |
|-----------|-------|
| $\mathsf{F{\cdot}}$ | $O(K^2 N + Kn \cdot N/2)$ |
| $\mathsf{G_smul}$ | $(6K + 7) \cdot n$ |

For softmax with $K = 3$, $n = 10$: $25 \cdot 10 = 250$ $\mathsf{G_smul}$ per sumcheck.

### 5.7 Implementation

`src/proof/zk_sumcheck.cu`: `prove_zk_inner_product`, `prove_zk_hadamard_product`,
`prove_zk_binary`, `prove_zk_multi_hadamard`, and corresponding `verify_*` functions.
`src/proof/zk_round_commit.cu`: `commit_round_poly`, `fold_commitments_at`,
`sumcheck_identity_lhs`, `combine_commitments_weighted`.

---

## 6. tLookup (LogUp-Based Lookup Argument)

**Reference:** zkLLM (Sun et al. 2024, §4.2); LogUp (Haböck, ePrint 2022/1530).

The tLookup protocol proves that every element of a data vector $S \in \mathbb{F}^D$
is contained in a table $T \in \mathbb{F}^N$.  It is used for CDF lookups
(zkNormalCDF), log lookups (zkLog), remainder range checks (Rescaling), and softmax
segment mappings.

The core identity is stated as zkLLM Lemma 4.1 (Sun et al. 2024):

> Given tensors $\mathbf{S} \in \mathbb{F}^D$ and $\mathbf{T} \in \mathbb{F}^N$,
> $\mathbf{S} \subset \mathbf{T}$ as sets if and only if there exists
> $\mathbf{m} \in \mathbb{F}^N$ such that
> $\sum_{i \in [D]} \frac{1}{X + \mathbf{S}_i} = \sum_{i \in [N]} \frac{\mathbf{m}_i}{X + \mathbf{T}_i}$

This is the LogUp identity (Haböck 2022): the lookup relation reduces to an equality
of rational functions that can be checked at a random point $\beta$ via sumcheck.

### 6.1 Protocol

**Inputs:**

- Data $S \in \mathbb{F}^D$ (the looked-up values)
- Table $T \in \mathbb{F}^N$ (public)
- Multiplicity vector $m \in \mathbb{F}^N$ where $m_j = |\{i : S_i = T_j\}|$
- $D$ must be a power of 2, $N$ must be a power of 2, and $D \equiv 0 \pmod{N}$.

**LogUp identity (Lemma 4.1 instantiation).**  From the identity above, at a random
$\beta \xleftarrow{\$} \mathbb{F}$:

$$\sum_{i=0}^{D-1} \frac{1}{S_i + \beta} = \sum_{j=0}^{N-1} \frac{m_j}{T_j + \beta}$$

Soundness: if the lookup relation fails, the LHS and RHS are distinct degree-$(D{+}N)$
rational functions of $\beta$, so they agree on at most $D + N$ values.  Probability
of false acceptance: $\leq (D + N) / |\mathbb{F}|$.

**Protocol structure:**

Define $A_i = 1 / (S_i + \beta)$ and $B_j = 1 / (T_j + \beta)$.

1. **Random challenges:** Verifier sends $\alpha, \beta \xleftarrow{\$} \mathbb{F}$
   and challenge vectors $u \in \mathbb{F}^{\log_2 D}$, $v \in \mathbb{F}^{\log_2 D}$.

2. **Phase 1** (over $D$ elements, $\log_2(D/N)$ rounds):
   Custom sumcheck reducing $A$ and $S$ from size $D$ to size $N$, proving a combined
   identity involving $\alpha \cdot A \cdot (S + \beta)$ and $\sum A$.

3. **Phase 2** (over $N$ elements, $\log_2 N$ rounds):
   Sumcheck reducing $A, S, B, T, m$ to singletons, proving the full LogUp identity
   plus a dot-product equality between multiplicities and inverses.

4. **Final evaluations:** $A(u), S(u), B(v_2), T(v_2), m(v_2)$ are emitted for the
   verifier to check consistency.

**tLookupRangeMapping extension.**  For lookups with input-output mapping
($S_{\text{in}} \to S_{\text{out}}$ via table), a random linear combination
$S_{\text{com}} = S_{\text{in}} + r \cdot S_{\text{out}}$ and
$T_{\text{com}} = T + r \cdot T_{\text{mapped}}$ reduces to the base tLookup.

### 6.2 Properties

**Completeness.**  If $S \subseteq T$ with the stated multiplicities, the LogUp
identity holds for all $\beta$, and both phases of the sumcheck pass.

zkLLM Theorem 7.2 (Sun et al. 2024) states:

> Assuming the verifier $\mathcal{V}$ is semi-honest, Protocol 1 incurs a
> completeness error of $O(N / |\mathbb{F}|)$.

The $O(N/|\mathbb{F}|)$ term arises from the Schwartz-Zippel checks at random
evaluation points during the sumcheck phases, not from the LogUp identity itself
(which holds exactly for honest provers).

**Soundness.**  zkLLM Theorem 7.3 (Sun et al. 2024) states:

> For any probabilistic polynomial-time (p.p.t.) prover $\mathcal{P}$, if in Line 6,
> the message $\mathcal{P}$ sends to $\mathcal{V}$ is $\llbracket \mathbf{S} \rrbracket
> \leftarrow \mathsf{Commit}(\mathbf{S})$ such that $\mathbf{S} \not\subset \mathbf{T}$,
> then except with probability $\mathsf{negl}(\lambda)$, the execution of Protocol 1
> is unsuccessful.

Composition of:

1. LogUp identity: error $\leq (D + N) / |\mathbb{F}|$
2. Phase 1 sumcheck ($\log_2(D/N)$ rounds, degree 3): error $\leq 3 \log_2(D/N) / |\mathbb{F}|$
3. Phase 2 sumcheck ($\log_2 N$ rounds, degree 4): error $\leq 4 \log_2 N / |\mathbb{F}|$

Total: $\leq (D + N + 3\log_2(D/N) + 4\log_2 N) / |\mathbb{F}|$.
For $D = 2^{25}$, $N = 2^{15}$: error $\leq 2^{25} / 2^{255} \approx 2^{-230}$.

This matches the zkLLM Theorem 7.3 guarantee: except with negligible probability,
a cheating prover (one that commits $S \not\subset T$) cannot make Protocol 1
succeed.  The dominant term $D + N$ comes from the Schwartz-Zippel bound on the
LogUp rational identity (Lemma 4.1), not from the sumcheck rounds.

**Zero-knowledge:**

zkLLM Theorem 7.4 (Sun et al. 2024) establishes zero-knowledge via a simulation
argument: there exists a simulator $\mathsf{Sim}$ such that for any semi-honest
verifier, the distribution of the real protocol transcript is computationally
indistinguishable from the simulated transcript.  The Real game runs the actual
protocol; the Ideal game runs $\mathsf{Sim}$ which has access only to the public
inputs and the commitment oracle.  The proof relies on the hiding property of the
commitment scheme and the simulatability of the underlying sumcheck.

**tLookup round polynomials are now ZK-wrapped** (Phase 3 Step 5).  Both
`tLookup_phase1` and `tLookup_phase2` call `emit_zk_round` to Pedersen-commit each
degree-3 round polynomial via the Hyrax §4 Protocol 3 framework.  The round identity
uses standard-sumcheck weights $\alpha = (2, 1, 1, 1)$ with 5 sigma challenges per
round (4 proof-of-opening + 1 proof-of-equality).  The commitment chain starts from
$C_0 = S \cdot U$ (public claim, blinding 0) and threads through all rounds.

⚠ **Final evaluations still leak.**  After all sumcheck rounds, the protocol still
emits $A(u), S(u), B(v), T(v), m(v)$ as raw scalars.  To complete the ZK wrapping:

- $T(v)$ can remain in the clear (public table).
- $S(u)$, $m(v)$, and $A(u) = 1/(S(u) + \beta)$ carry witness information and must
  be replaced with ZK openings against committed tensors.

This is a Phase 4 task (committed final evaluations).

### 6.3 Cost

**Phase 1** ($n_1 = \log_2(D/N)$ rounds):

Per round $j$ (tensor size $D_j = D / 2^j$):

| Operation | Count |
|-----------|-------|
| $\mathsf{F{\cdot}}$ | $5 \cdot D_j/2$ (kernel: 3 products + partial MLE) |
| $\mathsf{F{+}}$ | $4 \cdot D_j/2$ |
| $\mathsf{F{\cdot}}$ (fold) | $2 \cdot D_j/2$ |

Total Phase 1 field: $\sum_{j=0}^{n_1-1} 7 D/2^{j+1} = 7(D - D/2^{n_1}) = 7(D - N)$ $\mathsf{F{\cdot}}$.

**Phase 2** ($n_2 = \log_2 N$ rounds):

Per round (5 tensors of size $N_j$):

| Operation | Count |
|-----------|-------|
| $\mathsf{F{\cdot}}$ | $\sim 15 \cdot N_j/2$ (eval kernel + dotprod kernel + sum kernel) |
| $\mathsf{F{\cdot}}$ (fold) | $5 \cdot N_j/2$ |

Total Phase 2 field: $\sum_{j=0}^{n_2-1} 20 N/2^{j+1} = 20(N - 1) \approx 20N$ $\mathsf{F{\cdot}}$.

**Precomputation:**

| Operation | Count |
|-----------|-------|
| $\mathsf{F^{-1}}$ | $D + N$ (computing $A$ and $B$) |

**Total field cost:** $7D + 20N + (D + N) \cdot 100$ (inversions) $\approx 107D + 120N$
$\mathsf{F{\cdot}}$ equivalent.

**Communication:** $(n_1 + n_2)$ polynomials (degree 3-4) + 5 final evaluations.

**ZK wrapper cost (Phase 3 Step 5).**  The tLookup ZK wrapper uses degree-3 round
polynomials with 4 coefficients per round.  Per round: 4 commitments (8 $\mathsf{G_smul}$),
4 proof-of-opening (8 $\mathsf{G_smul}$), 1 proof-of-equality (1 $\mathsf{G_smul}$),
weighted combination (4 $\mathsf{G_smul}$), fold (4 $\mathsf{G_smul}$) = **25 $\mathsf{G_smul}$
per round**.

Total: $25(n_1 + n_2)$ $\mathsf{G_smul}$.  For $D = 2^{25}$, $N = 2^{15}$:
$25 \times 25 = 625$ $\mathsf{G_smul}$ $\approx 2.5 \times 10^6$ $\mathsf{F{\cdot}}$
equivalent (0.07% of the $3.6 \times 10^9$ field cost).

### 6.4 Implementation

`src/zknn/tlookup.cu`: `tLookup::prove`, `tLookup_phase1`, `tLookup_phase2`,
`tLookupRange`, `tLookupRangeMapping::prove`.

---

## 7. zkFC (Matrix Multiplication)

**Reference:** zkLLM (Sun et al. 2024, §6.1); Thaler (2013) GKR matrix multiply.

The matrix multiplication proof reduces to an inner-product sumcheck via the
multilinear extension identity (zkLLM §6.1, Eq. 28):

$$\tilde{C}(\mathbf{u}, \mathbf{v}) = \sum_{s \in \{0,1\}^k} \tilde{A}(\mathbf{u}, s) \cdot \tilde{B}(s, \mathbf{v})$$

where $C = A \cdot B$, and $\mathbf{u}, \mathbf{v}$ are random evaluation points.
This identity holds because $C_{i,j} = \sum_s A_{i,s} B_{s,j}$ and the multilinear
extension preserves the sum over the Boolean hypercube.

### 7.1 Protocol

**Claim:** Given weight matrix $W \in \mathbb{F}^{I \times O}$, input $X \in \mathbb{F}^{B \times I}$,
output $Y \in \mathbb{F}^{B \times O}$, prove $Y = X \cdot W$.

1. Verifier sends random $u_B \in \mathbb{F}^{\log_2 B}$, $u_O \in \mathbb{F}^{\log_2 O}$,
   $u_I \in \mathbb{F}^{\log_2 I}$.

2. Compute claim: $c = \tilde{Y}(u_B, u_O)$.

3. Partial MLE reductions:
   $X_r = X.\mathsf{partial\_me}(u_B)$ (length $I$);
   $W_r = W.\mathsf{partial\_me}(u_O)$ (length $I$).

4. Inner-product sumcheck: prove $c = \langle X_r, W_r \rangle$ over the input dimension.

5. After sumcheck: terminal values $X_r(u_I)$ and $W_r(u_I)$.  Product should equal
   the final sumcheck claim.

6. Return weight claim: $\tilde{W}(u_I, u_O) = W_r(u_I)$, to be discharged via ZK
   opening (§4) against the committed weight matrix.

### 7.2 Properties

**Completeness.**  $\tilde{Y}(u_B, u_O) = \sum_s \tilde{X}(u_B, s) \cdot \tilde{W}(s, u_O)$
is a multilinear identity that holds when $Y = XW$.  The inner-product sumcheck
verifies this.

**Soundness.**  From the inner-product sumcheck (§5.3): error $\leq 2n/|\mathbb{F}|$
where $n = \log_2 I$.  Plus the Schwartz-Zippel check on the claim evaluation:
error $\leq (\log_2 B + \log_2 O + \log_2 I)/|\mathbb{F}|$ for the random point
evaluations.

**Zero-knowledge.**

When the inner-product sumcheck uses the ZK wrapper (§5): the sumcheck round
polynomials are hidden.  The final values $X_r(u_I)$ and $W_r(u_I)$ are discharged
via ZK opening (§4), which reveals only the evaluation value (not the underlying
vectors).

⚠ **The partial MLE reductions** $X_r$, $W_r$ are computed by the prover on plaintext
tensors.  The verifier never sees these tensors — only the committed sumcheck
transcripts and the final ZK opening.  However, the final evaluation values
$\tilde{X}(u_B, u_I)$ and $\tilde{W}(u_I, u_O)$ are revealed to the verifier as
part of the ZK opening.  These are random-point evaluations of the multilinear
extensions, which leak one bit of information each about the underlying functions.
In the interactive model this is inherent to the sumcheck (the verifier needs to
check the final claim).

⚠ **zkFC still uses the *plain* sumcheck** (`zkip` in `zkfc.cu`), not the ZK-wrapped
version.  The entropy pipeline's inner-product sumchecks were migrated to
`prove_ip_zk` in Phase 3 Step 2, and the tLookup rounds were ZK-wrapped in Phase 3
Step 5, but `zkip` in zkFC has not yet been migrated.  The migration is
straightforward (the API matches).

### 7.3 Cost

Let $B$ = batch size, $I$ = input dimension, $O$ = output dimension.

**Partial MLE reductions:**

| Operation | Count |
|-----------|-------|
| $\mathsf{F{\cdot}}$ ($X$ partial MLE) | $\sim 2 \cdot B \cdot I$ |
| $\mathsf{F{\cdot}}$ ($W$ partial MLE) | $\sim 2 \cdot O \cdot I$ |

**Inner-product sumcheck** ($n = \log_2 I$ rounds, vectors of length $I$):
Per §5.3: $8I$ $\mathsf{F{\cdot}}$ + $19n$ $\mathsf{G_smul}$.

**Weight claim ZK opening** (§4):
$\mathsf{MSM}(L) + \mathsf{MSM}(R)$ where the weight matrix has $R = I$ rows of
length $L = O$ (or vice versa depending on layout).

**Total prover cost:**

$$\underbrace{2(B + O) I}_{\text{partial MLE}} + \underbrace{8I}_{\text{sumcheck field}} + \underbrace{19 \log_2 I}_{\text{sumcheck group}} \cdot \mathsf{G_smul} + \underbrace{\mathsf{MSM}(O) + \mathsf{MSM}(I)}_{\text{ZK opening}}$$

For LLaMA-2-7B lm_head ($I = 4096$, $O = 32000$, $B = 1$):
- Partial MLE: $\sim 2 \times 33000 \times 4096 \approx 2.7 \times 10^8$ $\mathsf{F{\cdot}}$
- Sumcheck: $\sim 33000$ $\mathsf{F{\cdot}}$ + $19 \times 12 = 228$ $\mathsf{G_smul}$
- ZK opening: $\mathsf{MSM}(32000) + \mathsf{MSM}(4096)$

The partial MLE and ZK opening dominate.

### 7.4 Implementation

`src/zknn/zkfc.cu`: `zkFC::prove`, `zkip`, `zkip_stacked`.

---

## 8. Rescaling

**Reference:** zkLLM (Sun et al. 2024, §4.1).

zkLLM §4.1 describes the Rescaling protocol for quantized arithmetic: after each
multiplication of quantized values, the product must be divided by the scaling factor
$\gamma$ to prevent bit-width blowup.  The prover demonstrates the correctness of
this integer division by exhibiting the quotient and remainder and proving the
remainder is in range via a lookup argument.

### 8.1 Protocol

For scaling factor $\gamma$, given input $X$ and output $X' = \lfloor (X + \gamma/2) / \gamma \rfloor$,
the relation is $X = X' \cdot \gamma + r$ where $r \in [-\gamma/2, \gamma/2)$.

1. Prover computes quotient $X'$ and remainder $r$ on GPU.
2. Verifier sends random $u \in \mathbb{F}^n$.
3. **Division relation check:** $\tilde{X}(u) \stackrel{?}{=} \tilde{X'}(u) \cdot \gamma + \tilde{r}(u)$.
   (Schwartz-Zippel over multilinear extensions.)
4. **Range proof on $r$:** tLookupRange proves $r \in [-\gamma/2, \gamma/2)$ for every
   element, using a table of size $\gamma$.

### 8.2 Properties

**Completeness.**  The division relation holds identically when $X = X' \gamma + r$.
The range lookup succeeds when all remainders are in range.

**Soundness.**
1. Division relation (Schwartz-Zippel): error $\leq 1/|\mathbb{F}|$.
2. tLookup soundness on remainder: error per §6.2.

**Zero-knowledge.**

⚠ **Currently not ZK.**  The division relation check sends raw MLE evaluations
$\tilde{X}(u)$, $\tilde{X'}(u)$, $\tilde{r}(u)$ in the clear.  The tLookup is also
not ZK-wrapped (§6.2).  To achieve ZK:
- Replace the point-evaluation check with a committed relation (e.g., commit to
  $X - X' \gamma - r$ and prove it evaluates to 0 via ZK opening).
- Wrap the tLookup in ZK (§6.2).

### 8.3 Cost

**Division computation (GPU):** $N$ divisions + $N$ modular reductions = $O(N)$ integer ops.

**tLookup on remainder** ($D = N$, table size = $\gamma$):
Per §6.3: $\sim 107N + 120\gamma$ $\mathsf{F{\cdot}}$ equivalent.

**Division relation check:** $3N$ $\mathsf{F{\cdot}}$ (three MLE evaluations of size-$N$ tensors).

**Total:** $\sim 110N + 120\gamma$ $\mathsf{F{\cdot}}$.

### 8.4 Implementation

`src/zknn/rescaling.cu`: `Rescaling::operator()`, `Rescaling::prove`.

⚠ **`Rescaling::prove` returns an empty Claims vector** — the weight claim linkage
is not wired up in the current code.

---

## 9. zkArgmax

### 9.1 Protocol

**Claim:** Given logits $\ell \in \mathbb{F}^V$, the prover claims $v^* = \ell_{t^*} = \max_j \ell_j$.

1. Prover computes $t^* = \arg\max_j \ell_j$ and $v^*$ on CPU (host-side scan).
2. Compute diffs: $d_i = v^* - \ell_i$ for all $i$ (GPU kernel).
3. **Bit decomposition:** For each $d_i$, extract $B$ bit planes:
   $\text{bits}_b[i] = (d_i \gg b) \mathbin{\&} 1$ for $b = 0, \ldots, B{-}1$.

4. **Reconstruction check (Schwartz-Zippel):** At random $u \in \mathbb{F}^n$:
   $$\tilde{d}(u) \stackrel{?}{=} \sum_{b=0}^{B-1} 2^b \cdot \widetilde{\text{bits}_b}(u)$$

5. **Indicator binding:** Construct indicator $\text{ind} \in \{0,1\}^V$ with
   $\text{ind}_{t^*} = 1$, all others 0.  Check:
   - $\sum \text{ind} = 1$ (exactly one position selected)
   - $\langle \text{ind}, d \rangle = 0$ (selected position has diff = 0, so $v^* = \ell_{t^*}$)

6. **Batched binary check:** Random linear combination over $B + 1$ tensors
   (bit planes + indicator):
   $$\text{combined}[i] = \sum_{k} r_k \cdot a_k[i] \cdot (a_k[i] - 1) = 0$$
   Check $\widetilde{\text{combined}}(u) = 0$.

7. **Return:** $\tilde{\ell}(u) = v^* - \tilde{d}(u)$ as a claim on the logits MLE.

### 9.2 Properties

**Completeness.**  When $v^*$ is the true argmax:
- All diffs are non-negative → bit decomposition exists in $[0, 2^B)$.
- Reconstruction holds identically.
- Indicator is binary with sum 1 and dot-product 0.
- Batched binary check passes because all tensors are binary.

**Soundness.**

1. **Reconstruction (Schwartz-Zippel):** error $\leq n/|\mathbb{F}|$.
2. **Binary check:** The batched random linear combination reduces $B{+}1$ binary
   checks to a single random-point evaluation.  A non-binary tensor $a_k$ produces
   a non-zero polynomial $a_k(x)(a_k(x) - 1)$; the random coefficient $r_k$ prevents
   cancellation with other terms.

   Formally: if any $a_k$ is not binary on $\{0,1\}^n$, then
   $\sum_k r_k \cdot a_k(x)(a_k(x) - 1)$ is a non-zero polynomial of degree $\leq 2n$
   in $x$ (the $r_k$ are chosen after the $a_k$ are fixed).  Error:
   $\leq 2n/|\mathbb{F}|$ (Schwartz-Zippel in $x$) $+ (B{+}1)/|\mathbb{F}|$
   (the random $r_k$ could cause spurious cancellation).

   Total binary check error: $\leq (2n + B + 1)/|\mathbb{F}|$.

3. **Range soundness (non-negativity):** If any $d_i < 0$ (i.e., $\ell_i > v^*$), then
   $d_i = p - k$ for some $k > 0$.  This has $\geq 255$ non-zero bits, exceeding
   $B$, so the reconstruction check fails (the bit-extracted planes only capture $B$
   bits).  Error: 0 (exact, assuming $B < 255$).

4. **Indicator constraints:** $\sum \text{ind} = 1$ is checked exactly.
   $\langle \text{ind}, d \rangle = 0$ is checked exactly.  Combined with the binary
   check on ind: the indicator selects exactly one position with diff 0.

**Zero-knowledge.**

⚠ **The current zkArgmax is NOT zero-knowledge.** The following values are revealed:

- $v^*$ (the argmax logit value) — used directly in the diff computation
- $\tilde{d}(u)$ and each $\widetilde{\text{bits}_b}(u)$ — MLE evaluations at the challenge point
- $\text{ind}_{\text{sum}} = 1$ and $\langle \text{ind}, d \rangle = 0$ — these are always
  the same for honest provers, so no information leak
- $v^* - \tilde{d}(u)$ — the returned claim on logits

To achieve ZK: commit to all intermediate tensors (diffs, bit planes, indicator) and
replace point evaluations with ZK openings.  The reconstruction check becomes a
committed relation.  The binary checks are already over the ZK sumcheck (in the
entropy pipeline's `prove_nonneg`), but the batched random-combination approach in
standalone zkArgmax is not.

⚠ **In the entropy pipeline, zkArgmax is NOT used directly.**  The batched entropy
pipeline computes diffs via `batched_diffs_kernel` and relies on the CDF tLookup's
implicit argmax verification (negative diffs can't match CDF table entries) rather
than the bit-decomposition approach.  The bit decomposition is used in `prove_nonneg`
for the quotient-remainder range proofs, where it IS wrapped with ZK IP sumchecks.

### 9.3 Cost

**Compute (GPU):**

| Operation | Count |
|-----------|-------|
| $\mathsf{F{+}}$ (diffs kernel) | $V$ |
| Integer bit extract | $B \cdot V$ |

**Prove:**

| Operation | Count | Notes |
|-----------|-------|-------|
| MLE evaluations | $(B + 2) \cdot O(N)$ | $\tilde{d}(u)$, each $\widetilde{\text{bits}_b}(u)$, combined(u) |
| $\mathsf{F{\cdot}}$ (batched binary) | $(B+1) \cdot 3V$ | $a_k \cdot a_k$, $a_k \cdot (a_k - 1)$, $r_k \cdot \ldots$ |
| Communication | $(B + 4)\ \mathbb{F}$ | evaluations + indicator constraints |

When used inside the entropy pipeline with ZK binary proofs (via `prove_nonneg`):
add $B$ ZK inner-product sumchecks on length-$N$ tensors, each costing $19n$
$\mathsf{G_smul}$ + $8N$ $\mathsf{F{\cdot}}$.

### 9.4 Implementation

`src/zknn/zkargmax.cu`: `zkArgmax::compute`, `zkArgmax::prove`.

---

## 10. zkNormalCDF

### 10.1 Protocol

Wrapper around `tLookupRangeMapping` for the standard normal CDF.

**Table:** $2^p$ entries where entry $d$ stores
$\lfloor \Phi(d / \sigma_{\text{eff}}) \cdot s_{\text{out}} + 0.5 \rfloor$
for $d \in [0, 2^p)$.

**Input:** Non-negative integer diffs $d_{t,j} = v^*_t - \ell_{t,j} \in [0, 2^p)$.

**Lookup:** tLookupRangeMapping with $S_{\text{in}} = $ diffs, $S_{\text{out}} = $ CDF values.

### 10.2 Properties

Inherits completeness and soundness from tLookupRangeMapping (§6).

**Zero-knowledge:** ⚠ Partially ZK after Phase 3 Step 5 — round polynomials are
Pedersen-committed, but final evaluations ($S(u)$, $m(v)$, $A(u)$) are still
sent in the clear.  See §6.2.

### 10.3 Cost

Same as tLookupRangeMapping (§6.3) with $D = T \cdot V$ (padded to power of 2) and
$N = 2^p$.

For $T = 1024$, $V = 32000$, $p = 15$: $D \approx 2^{25}$, $N = 2^{15}$.
Field cost: $\sim 107 \cdot 2^{25} + 120 \cdot 2^{15} \approx 3.6 \times 10^9$
$\mathsf{F{\cdot}}$ equivalent.

This is the **dominant cost** in the entire entropy proof pipeline.

### 10.4 Implementation

`src/zknn/zknormalcdf.cu`: `zkNormalCDF::compute`, `zkNormalCDF::prove`.

---

## 11. zkLog

### 11.1 Protocol

Wrapper around `tLookupRangeMapping` for $-\log_2$.

**Table:** $2^p$ entries where entry $i$ (for input index $i + 1$) stores
$\lfloor (\text{precision} - \log_2(i + 1)) \cdot s_{\text{out}} + 0.5 \rfloor$.

**Input:** Quantized probability indices $q_t \in [1, 2^p]$.

### 11.2 Properties

Same as §10.  **Zero-knowledge:** ⚠ Partially ZK (Phase 3 Step 5); inherits
tLookup partial status — round polys ZK, final evals in clear.

### 11.3 Cost

$D$ = number of positions (padded), $N = 2^p$.  Much smaller than CDF lookup since
$D = O(T)$ rather than $O(TV)$.

For $T = 1024$, $p = 15$: $D = 2^{16}$ (padded for tLookup $D > N$ constraint),
$N = 2^{15}$.  Field cost: $\sim 107 \cdot 2^{16} + 120 \cdot 2^{15} \approx 1.1 \times 10^7$
$\mathsf{F{\cdot}}$ equivalent.

### 11.4 Implementation

`src/zknn/zklog.cu`: `zkLog::compute`, `zkLog::prove`.

---

## 12. zkConditionalEntropy (Composition)

The full entropy pipeline proves:

> Given committed weights $W$ and public tokens $(o_1, \ldots, o_T)$, the total
> conditional surprisal satisfies $H = \sum_t -\log_2 q_t(o_t)$ where $q_t$ is
> derived from logits $\ell = f_W(\text{input})$ via the Gaussian noise model.

### 12.1 Sub-proof Pipeline

The pipeline consists of 7 sub-proofs executed in sequence.  Each consumes and
produces claims that chain to the next.

**Stage 2a: CDF tLookup** (diffs → win probabilities)

- Proves $w_{t,j} = s_{\text{out}} - \Phi(d_{t,j} / \sigma)$ for all $T \times V$ entries.
- Implicit argmax correctness: negative diffs (from a false $v^*$) wrap to near $p$,
  cannot match any CDF table entry, so the LogUp identity would fail.
- **Input:** diffs tensor of size $T \cdot V$ (padded to $D_{\text{cdf}}$).
- **Cost:** one tLookupRangeMapping prove on $(D_{\text{cdf}}, 2^{p_{\text{cdf}}})$.

**Stage 2b: Diffs-to-logits linking**

- Proves the algebraic relation: $\tilde{d}(u) + \tilde{\ell}(u) = \tilde{v^*}(u_T) \cdot \widetilde{\mathbf{1}_V}(u_V)$
  at a random point $u = (u_T, u_V) \in \mathbb{F}^{\log_2(TV)}$.
- Emits MLE evaluations: diffs$(u)$, logits$(u)$, $v^*(u_T)$, $\mathbf{1}_V(u_V)$.
- Returns a **Claim** on logits for upstream chaining (to zkFC/lm_head proof).

**Soundness:** Schwartz-Zippel over the multilinear identity; error $\leq \log_2(TV)/|\mathbb{F}|$.

⚠ **ZK gap:** Sends $\tilde{d}(u)$, $\tilde{\ell}(u)$, $\tilde{v^*}(u_T)$ in the clear.
The logits evaluation $\tilde{\ell}(u)$ leaks one scalar about the weight matrix.
To fix: commit to diffs and $v^*$ tensors, use ZK openings.  The relation becomes a
committed check: $\tilde{d}(u) + \tilde{\ell}(u) - \tilde{v^*}(u_T) \cdot \widetilde{\mathbf{1}_V}(u_V) = 0$
verified via a Σ-protocol on the committed sum.

**Stage 2c: Total-win row-sum**

- Proves $\text{tw}_t = \sum_j w_{t,j}$ for each position $t$.
- Method: fix the $T$-dimension via partial MLE at random $u_T$, yielding a
  $V$-length tensor `wp_partial`.  Then inner-product sumcheck proves
  $\text{tw}(u_T) = \langle \text{wp\_partial}, \mathbf{1}_V \rangle$.
- **ZK:** Uses `prove_ip_zk` (§5.3) — round polynomials are Pedersen-committed. ✓
- **Cost:** One ZK IP sumcheck on length-$V$ tensors.

**Stage 2d: Actual-token extraction**

- Proves $w_{\text{sel},t} = w_{t,o_t}$ using a public indicator tensor
  $\mathbf{1}_{o}[t \cdot V + o_t] = 1$.
- Inner-product sumcheck: $\sum_t w_{\text{sel},t} = \langle w, \mathbf{1}_o \rangle$.
- **ZK:** Uses `prove_ip_zk`. ✓
- **Cost:** One ZK IP sumcheck on length-$TV$ tensors.

⚠ **The indicator $\mathbf{1}_o$ is constructed from public tokens.** If tokens are
private, this construction leaks them.  The current design assumes output tokens are
public (see the main paper's threat model).  If tokens must be hidden, the indicator
must be committed and the extraction proof restructured.

**Stage 2e: Quotient-remainder division**

- Proves $q_t = \lfloor w_{\text{sel},t} \cdot 2^p / \text{tw}_t \rfloor$ via the
  division relation $q_t \cdot \text{tw}_t + r_t = w_{\text{sel},t} \cdot 2^p$ and
  three range proofs:
  - $q \in [0, 2^{p+1})$ via bit decomposition ($p + 1$ bits)
  - $r \in [0, 2^B)$ where $B = \lceil \log_2(V \cdot s_{\text{cdf}}) \rceil$
  - $\text{tw} - r - 1 \in [0, 2^B)$ (proves $r < \text{tw}$)

- **Division relation check:** $\tilde{q \cdot \text{tw}}(u) + \tilde{r}(u) = \widetilde{w_{\text{sel}} \cdot 2^p}(u)$
  at random $u$.

- **Range proofs via `prove_nonneg`:** For each of the 3 tensors, extract bit planes
  and prove each bit plane is binary via ZK IP sumcheck.

- **ZK:** The ZK IP sumchecks for binary proofs are Pedersen-committed. ✓
  The reconstruction checks and the division relation evaluations are sent in the clear.

  ⚠ **ZK gap:** $\tilde{q \cdot \text{tw}}(u)$, $\tilde{r}(u)$, $\widetilde{w_p}(u)$
  are sent as raw evaluations.  The bit-plane evaluations $\widetilde{\text{bits}_b}(u)$
  are also sent in the clear.  To fix: commit to $q$, $r$, $\text{tw}$, and bit planes;
  replace raw evaluations with ZK openings.

- **Cost (prover):** Let $n_T = \lceil \log_2 T \rceil$.
  - Division relation: $3$ MLE evaluations of size-$T$ tensors = $3T$ $\mathsf{F{\cdot}}$
  - `prove_nonneg` for $q$: $(p+1)$ bit planes, each needs: bit extraction ($T$ ops),
    MLE evaluation ($T$ ops), one ZK IP sumcheck ($19 n_T$ $\mathsf{G_smul}$ + $8T$ $\mathsf{F{\cdot}}$)
  - `prove_nonneg` for $r$: $B$ bit planes × same
  - `prove_nonneg` for gap: $B$ bit planes × same
  - Total binary sumchecks: $(p + 1 + 2B)$ ZK IP sumchecks on length-$T$ tensors.

  For $p = 15$, $B = 25$, $T = 1024$, $n_T = 10$:
  - Binary sumchecks: $16 + 50 = 66$ ZK IP sumchecks
  - Group ops: $66 \times 19 \times 10 = 12{,}540$ $\mathsf{G_smul}$
  - Field ops: $66 \times 8 \times 1024 \approx 5.4 \times 10^5$ $\mathsf{F{\cdot}}$

**Stage 2f: Surprise log lookup**

- tLookupRangeMapping proves surprise$_t = -\log_2(q_t / 2^p)$.
- **Input:** $q$ tensor (padded to $D_{\text{log}}$), table size $2^{p_{\text{log}}}$.
- **Cost:** One tLookupRangeMapping prove. See §11.3.
- **ZK:** ⚠ Inherits tLookup ZK gap.

**Stage 2g: Entropy summation**

- ZK IP sumcheck proves $H = \langle \text{surprise}, \mathbf{1}_T \rangle$.
- **ZK:** Uses `prove_ip_zk`. ✓
- **Cost:** One ZK IP sumcheck on length-$T$ tensor.

### 12.2 Soundness Summary

The overall soundness error is the sum of errors across all sub-proofs.  For
$T = 1024$, $V = 32000$:

| Stage | Error bound |
|-------|-------------|
| 2a: CDF tLookup | $\leq 2^{25} / |\mathbb{F}|$ |
| 2b: Diffs-to-logits | $\leq 25 / |\mathbb{F}|$ |
| 2c: Row-sum IP | $\leq 2 \cdot 15 / |\mathbb{F}|$ |
| 2d: Extraction IP | $\leq 2 \cdot 25 / |\mathbb{F}|$ |
| 2e: Division + 66 binary IPs | $\leq (1 + 66 \cdot 2 \cdot 10) / |\mathbb{F}|$ |
| 2f: Log tLookup | $\leq 2^{16} / |\mathbb{F}|$ |
| 2g: Summation IP | $\leq 2 \cdot 10 / |\mathbb{F}|$ |
| **Total** | $\leq 2^{26} / |\mathbb{F}| \approx 2^{-229}$ |

Soundness is overwhelmingly dominated by the CDF tLookup (via the LogUp identity's
$D + N$ term), not by the sumcheck rounds.

### 12.3 Completeness Summary

All sub-proofs have straightforward completeness: honest computation of diffs, CDF
values, row sums, token extraction, division, log lookup, and summation satisfies
every algebraic identity checked by the protocol.  The only non-trivial completeness
consideration is the **clamping approximation**: win probabilities are clamped to
$\geq 1$ to avoid $\log(0)$.  This is a deliberate overestimate (not a completeness
failure) that marginally inflates the entropy bound.

### 12.4 Zero-Knowledge Summary

| Stage | ZK Status | Gap |
|-------|-----------|-----|
| 2a: CDF tLookup | ⚠ Partial | Round polys ZK (Phase 3 Step 5); final evals still in clear |
| 2b: Diffs-to-logits | ⚠ Not ZK | Raw MLE evaluations sent |
| 2c: Row-sum IP | ✓ ZK | Via `prove_ip_zk` (Phase 3 Step 2) |
| 2d: Extraction IP | ✓ ZK | Via `prove_ip_zk` (Phase 3 Step 2) |
| 2e: Division | ⚠ Partial | Binary checks ZK; division relation + bit evals not ZK |
| 2f: Log tLookup | ⚠ Partial | Round polys ZK (Phase 3 Step 5); final evals still in clear |
| 2g: Summation IP | ✓ ZK | Via `prove_ip_zk` (Phase 3 Step 2) |

**What the verifier currently learns (beyond the public $H$):**

1. MLE evaluations of diffs, logits, $v^*$ at the linking challenge point (stage 2b)
2. MLE evaluations of $q \cdot \text{tw}$, $r$, $w_p$ at the QR challenge (stage 2e)
3. Bit-plane evaluations at the QR challenge (stage 2e)
4. tLookup final evaluations $A(u), S(u), B(v), m(v)$ (stages 2a, 2f) — round
   polynomials are now hidden via Pedersen commitments

**To achieve full ZK, the following must be completed:**

1. **Committed tLookup finals** (blocking: stages 2a and 2f) — replace final
   evaluations $S(u), m(v), A(u)$ with ZK openings against committed tensors.
   ($T(v)$ is public and can remain in the clear.)
2. **Committed linking** (stage 2b) — commit to diffs and $v^*$; replace raw
   evaluations with a Σ-protocol on the committed sum.
3. **Committed division** (stage 2e) — commit to $q$, $r$, bit planes; replace raw
   evaluations with ZK openings.  The reconstruction check becomes a committed
   relation verified via proof-of-equality.
4. **Formal simulation proof** — construct a simulator for the full composed protocol.

### 12.5 Cost Summary

**Prover cost breakdown** for $T = 1024$, $V = 32000$, $p_{\text{cdf}} = 15$,
$p_{\text{log}} = 15$, $s_{\text{cdf}} = 65536$:

| Stage | $\mathsf{F{\cdot}}$ equiv. | $\mathsf{G_smul}$ | Notes |
|-------|---------------------|---------------------|-------|
| 2a: CDF tLookup | $\sim 3.6 \times 10^9$ | 625 (Phase 3) | $D \approx 2^{25}$, $N = 2^{15}$ |
| 2b: Linking | $\sim 10^8$ | 0 | 4 MLE evals on $TV$-tensors |
| 2c: Row-sum IP | $\sim 2.6 \times 10^5$ | 285 | $V = 32000$, $n = 15$ |
| 2d: Extraction IP | $\sim 2.6 \times 10^8$ | 475 | $TV \approx 3.3 \times 10^7$, $n = 25$ |
| 2e: Division + range | $\sim 5.4 \times 10^5$ | 12,540 | 66 binary IP sumchecks |
| 2f: Log tLookup | $\sim 1.1 \times 10^7$ | 400 (Phase 3) | $D = 2^{16}$, $N = 2^{15}$ |
| 2g: Summation IP | $\sim 8.2 \times 10^3$ | 190 | $T = 1024$, $n = 10$ |
| **Total** | $\sim 4.0 \times 10^9$ | 14,515 | |

**Group operations in field-mul equivalents:** $14{,}515 \times 4000 \approx 5.8 \times 10^7$.

**Overall:** The CDF tLookup dominates at $\sim 3.6 \times 10^9$ $\mathsf{F{\cdot}}$
(~90% of total field cost).  Group operations from ZK sumchecks (including the
tLookup ZK wrapper from Phase 3 Step 5) contribute $\sim 5.8 \times 10^7$ field-mul
equivalent (~1.4% of total).

**Verifier cost:** The verifier's work is dominated by MSM operations for ZK opening
checks.  Per sub-proof, the verifier performs $O(n)$ $\mathsf{G_smul}$ operations
(same counts as the verifier column in §5.3).  For the full pipeline:
$\sim 17 \times (15 + 25 + 66 \times 10 + 16 + 10) = 17 \times 726 \approx 12{,}340$
$\mathsf{G_smul}$ $\approx 4.9 \times 10^7$ $\mathsf{F{\cdot}}$ equivalent.

**Communication (proof size):**

| Component | Size (approx.) |
|-----------|---------------|
| tLookup ZK rounds (2a + 2f) | 25 rounds × (4 $\mathbb{G}_1$ + 4 openings + 1 equality) $\approx$ 44 KB per tLookup |
| ZK sumcheck transcripts (2c, 2d, 2e, 2g) | $n \times (7\ \mathbb{G}_1 + 7\ \mathbb{F})$ per sumcheck |
| Per ZK IP sumcheck at $n = 10$: | $10 \times (7 \times 96 + 7 \times 32) = 8{,}960$ B $\approx 9$ KB |
| 66 binary sumchecks | $66 \times 9 \approx 594$ KB |
| Final evaluations | $< 1$ KB |
| **Total** | $\sim 620$ KB |

The tLookup ZK round overhead (Phase 3 Step 5) is included above.  ⚠ The committed
final evaluations (Phase 4) would add additional ZK opening proofs per tLookup
invocation.

---

## 13. Upstream Layers (zkFC + Rescaling for lm_head)

The entropy pipeline receives logits from a zkFC proof of the lm_head layer
($\ell = \text{RMSNorm}(\text{hidden}) \cdot W_{\text{lm}}$).  The full chain is:

1. **Committed weights:** $W_{\text{norm}}$ and $W_{\text{lm}}$ are committed via
   Pedersen (§2) with per-row blindings.  Weight commitments are reusable across
   inferences (Pedersen is perfectly hiding).

2. **RMSNorm:** Rescaling proof (§8) on the normalized hidden state.

3. **lm_head zkFC:** Matrix multiply proof (§7) reducing to a weight claim.

4. **Weight claim discharge:** `verifyWeightClaimZK` (§4) opens the weight commitment
   at the sumcheck terminal point.

5. **Entropy pipeline:** §12, consuming the logits tensor.

**End-to-end soundness:** The overall security is captured by Hyrax Theorem 2
(Wahby et al. 2018, §4):

> Let $C(\cdot, \cdot)$ be a layered arithmetic circuit of fan-in two, consisting of
> $N$ identical sub-computations, each of depth $d$, with all layers of each
> sub-computation having width at most $G$.  Assuming the existence of
> computationally binding, perfectly hiding homomorphic commitment schemes that
> support proof-of-opening, proof-of-equality, and proof-of-product (Appx. A) [...]
> there exists a PZK argument for the NP relation "$\exists w$ such that
> $C(x, w) = y$."

In our setting, soundness error is the union bound over per-layer errors.  Each
layer contributes $\leq 2^{-229}$; with ~5 layers, total error $\leq 5 \times 2^{-229} \approx 2^{-227}$.

**End-to-end ZK (target state):**

⚠ The full ZK chain requires all layers to be ZK.  After Phase 3:
- Weight commitment: ✓ (Pedersen, perfectly hiding)
- ZK opening: ✓ (Hyrax §A.2 Figure 6)
- zkFC: ⚠ still uses plain sumcheck (`zkip`) — migration to ZK sumcheck needed
- Rescaling: ⚠ partial — tLookup rounds ZK (Phase 3 Step 5), division evals not ZK
- RMSNorm HP: ✓ uses `prove_zk_hadamard_product` (Phase 3 Step 2)
- Entropy pipeline: ⚠ partial (see §12.4) — IP/HP sumchecks and tLookup rounds ZK;
  linking evals, division evals, and tLookup finals still in clear

---

## 14. Summary Tables

### 14.1 Soundness Error by Layer

| Layer | Soundness Error | Dominant Term |
|-------|-----------------|---------------|
| Pedersen commitment | computational (DLOG) | — |
| Σ-protocol (opening) | $1/|\mathbb{F}|$ | — |
| Σ-protocol (equality) | $1/|\mathbb{F}|$ | — |
| ZK opening (§A.2) | $1/|\mathbb{F}|$ | — |
| ZK IP sumcheck ($n$ rounds) | $\leq 6n/|\mathbb{F}|$ | $2n$ (sumcheck) + $4n$ (Σ-protocols) |
| tLookup ($D$ data, $N$ table) | $\leq (D + N + 7\log_2 D)/|\mathbb{F}|$ | $D + N$ (LogUp identity) |
| zkFC ($I$ input dim) | $\leq 6\log_2 I / |\mathbb{F}|$ | sumcheck |
| Rescaling ($N$ elements, $\gamma$ table) | $\leq (N + \gamma + 7\log_2 N) / |\mathbb{F}|$ | tLookup |
| zkArgmax ($V$ vocab, $B$ bits) | $\leq (3n + B + 1)/|\mathbb{F}|$ | binary + reconstruction |
| Entropy pipeline (full) | $\leq 2^{26}/|\mathbb{F}| \approx 2^{-229}$ | CDF tLookup |

### 14.2 ZK Status by Layer

| Layer | ZK Status | Blocker |
|-------|-----------|---------|
| Pedersen commitment | ✓ Perfectly hiding | — |
| Σ-protocols | ✓ Perfect HVZK | — |
| ZK opening | ✓ HVZK | — |
| ZK sumcheck (all variants) | ✓ ZK (informal) | Formal simulation (Phase 5) |
| tLookup | ⚠ **Partial** (Phase 3 Step 5) | Round polys ZK; final evals still in clear |
| zkFC | ⚠ **Plain sumcheck** | Migration to ZK sumcheck |
| Rescaling | ⚠ **Partial** | tLookup rounds ZK; raw division evals |
| zkArgmax (standalone) | ⚠ **Not ZK** | Raw evaluations |
| zkNormalCDF | ⚠ **Partial** | Inherits tLookup partial status |
| zkLog | ⚠ **Partial** | Inherits tLookup partial status |
| Entropy pipeline (composed) | ⚠ **Partial ZK** | 2a/2f finals, 2b linking, 2e division |

### 14.3 Prover Cost by Layer (Dominant Terms)

All costs normalized to $\mathsf{F{\cdot}} = 1$.

| Layer | Field Cost | Group Cost ($\mathsf{G_smul}$) | Dominant |
|-------|------------|--------------------------------|----------|
| Pedersen commit ($N$ elements) | 0 | $\mathsf{MSM}(N)$ | Group |
| ZK IP sumcheck ($N$ elements) | $8N$ | $19n$ | Field for large $N$; group for small $N$ |
| ZK HP sumcheck ($N$ elements) | $11N$ | $19n$ | Field |
| ZK multi-HP ($K$ tensors, $N$) | $O(K^2 N)$ | $(6K+7)n$ | Field |
| tLookup ($D$ data, $N$ table) | $107D + 120N$ | $25 \lceil\log_2 D\rceil$ (Phase 3) | Field |
| zkFC ($B \times I \times O$) | $2(B+O)I + 8I$ | $19\log_2 I$ + MSM | MSM |
| Entropy pipeline (full) | $4.0 \times 10^9$ | 13,490 | tLookup (90%) |
