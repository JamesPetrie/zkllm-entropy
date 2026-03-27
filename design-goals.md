# Design Goals for the Zero-Knowledge Proof of Inference

**Core purpose.** The ZK proof replaces a physically isolated recomputation cluster. Instead of requiring a split-trust enclosure where both parties contribute hardware, the prover generates a cryptographic proof on their own hardware that their output is consistent with the declared model — with no trusted hardware needed on either side.

**What the proof attests to.** Given committed weights W, logged input x, seed s, and output sequence O = (o₁, …, oₙ), the prover correctly computed the conditional entropy estimate Ĉ(O) = −Σ log₂ qᵢ(oᵢ), where each qᵢ was derived from W applied to (x, o_{<i}) with seed s, following the declared sampling method. The verifier receives a proved upper bound on covert capacity — regardless of what steganographic scheme an adversary might use, they cannot extract more than Ĉ(O) bits from observing O.

**Causality enforcement.** The critical security property is that qᵢ (the token probability distribution) must be fully determined *before* oᵢ (the output token) is known. If the prover could choose qᵢ after seeing oᵢ, they could assign qᵢ(oᵢ) = 1 to every token and claim zero entropy while encoding arbitrary covert information. The zkLLM commitment structure enforces this naturally: weights are committed before inference, logits are committed before any claim about the output token, and softmax is deterministic given committed logits.

**Confidentiality.** The verifier learns nothing beyond whether the prover is compliant. Model weights remain hidden behind a cryptographic commitment (e.g., Pedersen). The model architecture must be revealed, though it may be possible to substitute an equivalent architecture that hides proprietary details.

**Interactive proofs preferred.** The design favors interactive proofs (verifier provides fresh random challenges) over non-interactive (Fiat-Shamir). Because the output is committed before the proof is requested, a failed proof is a detection event, not an invisible failure. This eliminates the need for proof repetitions even at small field sizes — at 64-bit fields, the adversary is caught with probability 1 − 10⁻¹³ per steganographic attempt. Non-interactive proofs lose this advantage because the prover can grind offline, trying many outputs and only committing the one whose proof passes.

**Minimizing total time.** The objective is to minimize the sum of proof construction and verification time. This is the primary performance metric — tradeoffs between prover and verifier cost are acceptable as long as the total is reduced.

**Post-quantum security.** The proof system should be post-quantum secure, given that the threat model involves nation-state adversaries and that agreements built on this verification infrastructure are intended to be durable.

**Proof size.** Smaller proofs are preferred, but proofs up to the size of the input (logged tokens plus context) are acceptable. Proof size is a secondary concern relative to total proving and verification time.

**Scaling as a constant factor.** Proving overhead mirrors inference complexity — it's a fixed multiplicative constant, not an asymptotic blowup. Self-attention dominates both inference and proving at long contexts. Multi-GPU scaling is near-linear (only ~4.5 MB of proof communication across the entire proof; the dominant cost is the same tensor-parallel all-reduces as regular inference).

**Performance path.** Current overhead is ~17,000× on BLS12-381 (255-bit field). Switching to Goldilocks (64-bit) or Mersenne31 (31-bit) fields could yield 10–25× speedup. Target: proving a 1T-parameter, 1M-context model in ~40 minutes on 1,000 B200 GPUs.
