# Prover Determinism in zkLLM-Style Protocols

## Why This Matters

The proof produces $q_{i}$ at each step (computed from committed intermediate values) and reports $\hat{H} = -\sum_{i} \log_{2} q_{i}(o_{i})$. Any prover freedom that yields different valid $q_{i}$ values for the same inputs lets a dishonest prover pick one with higher $q_{i}(o_{i})$, depressing $\hat{H}$ below $H(O \mid D)$ and breaking the bound.

## Where the Slack Lives in zkLLM

zkLLM (Sun et al. 2024) handles non-arithmetic operations via lookup tables and shift constructions. The relevant locations:

**Softmax shift (per row, per attention head, per layer).** Protocol 2, line 15: `Z' = Z − ⌊ẑ⌉ · 1^T`. The prover commits the integer shift $\lfloor \hat{z} \rceil$. The verifier does not directly check $\hat{z}$ against eq. 17 (the log-sum-exp); it only checks the resulting row sum $\hat{y} \in T_{R} = [\theta - E, \theta + E]$ via tlookup (Protocol 2, lines 11 and 34–35). Multiple integer shifts can satisfy the band, so $\lfloor \hat{z} \rceil$ is not pinned by the data alone.

**Setup parameters.** Protocol 2's Require block takes $\gamma$, $\theta$, $(b^{(k)})$, $(\theta^{(k)})$, $K$, $M$, $L$, and the tolerance $E$ as inputs. These shape the lookup tables and the normalization band. If setup is controlled by the prover rather than fixed by the protocol/regulator, every per-proof value of these parameters is a freedom.

**Lookup table contents.** zkLLM's tlookup setup commits a public table deterministically (`Commit(T; 0)`, no hiding randomness), so both parties can in principle reconstruct the same commitment from the spec. The freedom is upstream of the commitment: the softmax digit tables $T^{(k)}_{Y}[j] = \theta^{(k)} \cdot \exp(-B^{(k)} j / \gamma\sqrt{d})$ and the elementwise activation tables (ReLU, GELU, SiLU, sigmoid) are real-valued functions that must be rounded to integers before committing, and zkLLM does not specify the rounding rule. Two honest parties following the paper with different rounding conventions (floor, round-half-to-even, truncation) compute different integer tables and therefore different commitments. A dishonest prover can exploit this by committing a table rounded in a direction that shifts $q_{i}$ favorably. This freedom exists even though the lookup mechanism itself (decomposition via Example 4.2, bijection, LogUp identity) has no slack.

**Layer/RMS norm (proposed extension).** zkLLM eq. 30 shows how to evaluate layer norm inside the proof system (two sequential tlookups: downscale, then quantized normalize), but it does not specify how the verifier checks that the normalization was performed with the right scale factor. For softmax the analogous check is the row-sum tolerance band — and that band is exactly where softmax freedom comes from. Any concrete layer-norm check must do something analogous, and the natural "output norm is within $E$ of the target" construction reproduces the same band freedom. A protocol that extends zkLLM to cover layer/RMS norm must pick this check explicitly and use a unique-witness construction (e.g. the sandwich below) rather than a band.

Operations that have **no slack** in zkLLM as written: matrix multiplications and residual additions. Elementwise activations (ReLU, GELU, SiLU) have no slack in their decomposition mechanism — Example 4.2 (page 6) defines the quantization remainder range as $T_{R} := [-r/2, r/2 - 1]$ in integer semantics (half-open and unique), so the integer pair $(q, r)$ in $x = \gamma q + r$ is fully determined — but they still depend on the table-contents freedom above.

## The Sandwich Fix

The slack sources above (softmax shift, layer/RMS norm scale) share a structure: the prover chooses an integer $s$ such that an aggregate quantity $g(s)$ lands in a tolerance band, where $g$ is non-increasing in $s$. The band can be replaced with a uniqueness condition.

**Construction.** Instead of accepting $g(s) \in [\text{target} - E, \text{target} + E]$, the verifier requires the prover to commit $s$ and prove the two-sided sandwich

$$g(s) \geq \text{target} \quad \text{and} \quad g(s+1) \lt \text{target}.$$

Both checks are integer comparisons. When $g$ is non-increasing, exactly one integer satisfies the sandwich — $s$ is pinned down uniquely by the data. Applied to softmax: $g(s) = \sum_{i} y_{i}(s)$ and target $= \theta$.

**Cost.** Doubles the lookup work for the constrained operation. Matmul dominates proving cost in zkLLM-style protocols, so the relative overhead is small; quantifying it precisely requires a measurement we haven't done.

**Completeness.** The sandwich is defined on the quantized $g$ — the same table-evaluated row sum the verifier checks. So the honest prover does not compute real-valued $\hat{z}$ and round it; they find the integer $s$ directly by locating where $g_{\text{quantized}}$ crosses $\theta$, and that $s$ satisfies the sandwich by construction. Uniqueness follows from the asymmetric inequalities: if $s_1 \lt s_2$ both satisfied $g(s) \geq \theta$ and $g(s+1) \lt \theta$, monotonicity would give $g(s_2) \leq g(s_1 + 1) \lt \theta$, contradicting $g(s_2) \geq \theta$. Plateaus at $\theta$ do not break uniqueness — the witness is the rightmost index of the plateau.

The only setup-time condition is existence: the input range and table size must guarantee that at least one $s$ in the committed domain has $g(s) \geq \theta$ and $g(s+1) \lt \theta$ — i.e. $g_{\text{quantized}}$ actually crosses $\theta$ in range. Verify this at setup alongside the monotonicity check.

## Monotonicity Is a Setup-Time Check, Not a Theorem

The aggregate $g(s)$ is non-increasing for the real-valued operation, but the protocol uses rounded integer lookup tables decomposed across digits. Carry boundaries between digits can introduce small local non-monotonicities even with deterministic rounding — a constructed example with truncated tables can produce $y(x+1) \gt y(x)$ at a carry transition.

The fix: at protocol setup, enumerate $x \in [0, B)$ and verify $y(x) \geq y(x+1)$ for the per-row aggregate. The check is $O(B)$ at setup, run once. If it fails, retune the table parameters $(b^{(k)}, \theta^{(k)})$ until it succeeds. Once verified, the sandwich is sound for every proof.

## Spec Requirements

For a zkLLM-style protocol to support the entropy bound without residual prover freedom, the protocol specification must:

1. **Use the sandwich construction (not a tolerance band) for the softmax shift and the layer/RMS norm normalizer.** Verify per-row aggregate monotonicity at setup and reject parameter choices that fail. Under the sandwich, the prover's external method for picking the integer witness is a prover-side implementation detail — the two integer inequalities on the quantized $g$ pin the witness regardless, so no separate rounding rule for $\lfloor \hat{z} \rceil$ is needed.
2. **Pin all setup parameters** — $(\gamma, \theta, b^{(k)}, \theta^{(k)}, K, M, L, E)$ — at the protocol level, not per-proof. The prover does not choose them.
3. **Specify the rounding rule so both parties compute the same integer table.** zkLLM's tlookup setup deterministically commits a public table, but the real-valued table entries — $T_{Y}[i] = \gamma \cdot f(i/\gamma)$ for elementwise $f$, and $T^{(k)}_{Y}[j] = \theta^{(k)} \cdot \exp(-B^{(k)} j / \gamma\sqrt{d})$ for the softmax digit tables — must be rounded to integers first. The spec must fix a deterministic rounding rule (e.g. round-half-to-even or truncation) so the honest table commitment is reproducible. The verifier then reconstructs the table from the spec and checks the prover's commitment against it (or the spec-defined commitment is included in the verifying key at protocol setup).
4. **Keep the half-open quantization remainder convention** $r \in [-\gamma/2, \gamma/2)$ for every elementwise activation lookup. (zkLLM Example 4.2 already specifies this; the spec must not weaken it.)
5. **Make all sampling randomness an explicit input commitment.** zkLLM proves the next-token distribution but not the sampling step itself; the calling protocol must commit the sampling seed and prove the realized $o_{i}$ is consistent with $(q_{i}, s)$.

With these in place, every quantity the verifier accepts is pinned down by $D$, and $q_{i}$ is a deterministic function of the committed inputs.

## References

Sun, H. et al. (2024). zkLLM: Zero Knowledge Proofs for Large Language Models. CCS '24.
