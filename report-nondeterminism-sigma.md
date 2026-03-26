# Calibrating σ: Nondeterminism in LLM Inference

This report estimates the effective logit noise parameter σ from empirical measurements of nondeterminism in production LLM inference, and quantifies its impact on per-token entropy.

---

## Background

When verifying LLM inference, the verifier recomputes the model's output and checks it against the prover's transcript. If inference is perfectly deterministic, any discrepancy proves cheating. In practice, floating-point nondeterminism can cause legitimate disagreements: the same model, given the same input, may occasionally produce a different token. The parameter σ quantifies this noise — it is the standard deviation of an effective Gaussian perturbation to the logits that would produce the observed rate of token disagreements.

---

## Experimental Setup

We ran two complementary experiments:

**Experiment 1 — Single-GPU determinism (vLLM + Llama-2-7B on H100).**
Llama-2-7B was served via vLLM (v0.18.0, bfloat16, FlashAttention v3, `--enforce-eager`) on a single NVIDIA H100 PCIe GPU. Six prompts were each submitted 50 times at temperature 0 with `max_tokens=64`. Top-5 log-probabilities were collected at every token position, yielding 384 token positions with full logprob data.

**Experiment 2 — Production-scale nondeterminism (Claude API).**
The same six prompts were each submitted 10 times at temperature 0 to Claude Haiku (claude-haiku-4-5-20251001) via the Anthropic API with `max_tokens=50`. Since the API does not return log-probabilities, only text-level agreement was measured.

---

## Results

### Single-GPU inference is bitwise deterministic

Across all 470,400 pairwise token comparisons (6 prompts × C(50,2) pairs × 64 tokens), vLLM produced **zero disagreements**. Log-probability differences across runs were exactly 0.000 at every position. This confirms prior findings (Cankaya 2024) that single-GPU floating-point inference is deterministic when hardware, software, and batch composition are held fixed.

### Production inference shows measurable nondeterminism

Across 270 pairwise comparisons of Claude API outputs, 85 pairs (31.5%) diverged at some point during generation. However, divergence is concentrated at specific token positions — most tokens are generated identically across all runs. Factual prompts ("The capital of France is", "The solution to x² − 5x + 6 = 0 is") were fully deterministic. Open-ended prompts diverged at consistent character positions, suggesting a small number of "decision points" where the model's top candidates are nearly tied.

The **per-token first-divergence rate** — counting only the first token at which a pair of runs disagrees, conditional on all prior tokens matching — was **0.74%** (85 first-divergence events out of ~11,500 token comparisons up to divergence).

---

## Estimating σ from the Logit Gap Distribution

The key insight is that the vLLM experiment gives us the **logit gap distribution** (the difference between the top-1 and top-2 log-probabilities at each position), while the Claude API experiment gives us the **observed flip rate**. Combining these allows us to solve for σ.

### Model

At each token position, let Δ be the gap between the highest and second-highest logit. If independent Gaussian noise N(0, σ²) is added to each logit, the probability that the argmax changes is:

$$P(\text{flip}) = \Phi\!\left(\frac{-\Delta}{\sigma\sqrt{2}}\right)$$

where Φ is the standard normal CDF.

### Complication: bfloat16 quantization

The vLLM experiment uses bfloat16 inference. Bfloat16 has a 7-bit mantissa, giving a unit in the last place (ULP) of ~0.016 for logit magnitudes in the range 1–4. At 12 of 384 token positions (3.1%), the top-1 and top-2 log-probabilities were bitwise identical — not because the logits are truly equal, but because their difference is below the bfloat16 ULP.

For these tied positions, we model the true gap as uniformly distributed on [0, 0.016) and average the flip probability over this interval.

### Result

Solving for the σ that produces a 0.74% per-token flip rate against the measured gap distribution (using numerical root-finding):

| Parameter | Value |
|-----------|-------|
| **σ** | **0.007** (in natural-log logit space) |
| **σ_eff (×65536)** | **456** |

For reference, the σ_eff value hardcoded in the prototype is 5,223 — roughly 11× larger than this empirically calibrated estimate.

### Sensitivity

| σ | σ_eff | Per-token flip rate |
|---|-------|---------------------|
| 0.001 | 65 | 0.12% |
| 0.005 | 327 | 0.56% |
| **0.007** | **456** | **0.74%** |
| 0.010 | 655 | 0.94% |
| 0.050 | 3,276 | 1.87% |
| 0.080 | 5,223 | 2.68% |
| 0.100 | 6,553 | 3.10% |
| 0.500 | 32,768 | 12.1% |

---

## Entropy Implications

### Simplified model: close-call positions

The nondeterminism is concentrated at "close-call" token positions where the top two candidates are nearly tied. At these positions, the outcome is effectively a coin flip (~1 bit of entropy); at all other positions, the nondeterminism entropy is negligible. This leads to a simple estimate:

- A pairwise first-divergence occurs when a close-call position exists **and** the two runs land on opposite sides of the coin flip (probability ≈ 50%).
- Therefore: **close-call rate ≈ 2 × first-divergence rate = 2 × 0.74% = 1.48%**.
- Each close-call contributes ~1 bit of nondeterminism entropy.
- **Nondeterminism entropy per token ≈ 0.0148 × 1 = 0.0148 bits**.

### Gaussian noise model (from logit gap distribution)

Fitting σ to the Llama-2-7B logit gap distribution (Experiment 1) to match the 0.74% flip rate gives a slightly higher estimate, because some positions contribute fractional entropy rather than a clean 0-or-1. However, the gap distribution is sharply bimodal — positions are either bf16-tied (gap < 0.016, flip ≈ 50%) or well-separated (gap > 0.1, flip ≈ 0) — so the fractional contributions are negligible:

| Metric | Simplified model | Gaussian fit (Llama-2-7B) |
|--------|-----------------|---------------------------|
| Nondeterminism entropy per token | **0.015 bits** | **0.031 bits** |
| Close-call / tied positions | 1.48% (from API) | 3.1% (12/384, from logprobs) |

The difference between the two estimates (0.015 vs 0.031 bits/token) reflects the different close-call rates: 1.48% derived from the Claude API first-divergence rate vs 3.1% observed in Llama-2-7B logprob gaps. These measure different models under different serving conditions, so some discrepancy is expected. Both estimates are small.

### Ratio to token representation size

Llama-2-7B has a vocabulary of 32,000 tokens, requiring log₂(32,000) ≈ **15 bits** to represent a token.

| Estimate | Entropy / token size |
|----------|---------------------|
| Simplified (from API disagreement rate) | 0.015 / 15 = **0.10%** |
| Gaussian fit (from Llama-2-7B logprobs) | 0.031 / 15 = **0.21%** |

The nondeterminism entropy is **0.1–0.2% of the information needed to specify a token**. Equivalently, 99.8–99.9% of each token's identity is fully determined by the model's computation; only 0.1–0.2% is attributable to noise.

---

## Interpretation and Caveats

1. **Source of production nondeterminism.** Single-GPU inference is deterministic. The 0.74% flip rate observed in the Claude API likely arises from multi-replica routing and dynamic batching — different request groupings change padding and attention masks, altering floating-point accumulation order across replicas. This is not Gaussian noise on logits; it is better described as an occasional tie-breaking mechanism at positions where the model's top candidates are nearly equal.

2. **The Gaussian model is a convenient fiction.** The actual nondeterminism is binary at each position: either the gap is large enough to be stable (>99% of tokens) or it is near-degenerate and the outcome depends on implementation details (~1% of tokens). Modeling this as continuous Gaussian noise with σ = 0.007 reproduces the correct aggregate flip rate but overstates the noise at most positions and understates it at the vulnerable ones. The simplified close-call model better reflects the actual mechanism.

3. **Model and prompt dependence.** The logit gap distribution depends on the model, vocabulary size, prompt content, and position in the sequence. The 12/384 near-tied positions observed for Llama-2-7B on these six prompts may not be representative of other models or workloads. A larger-vocabulary model may have more near-ties (more candidates to be close), while a more confident model may have fewer.

4. **bfloat16 vs float32.** If inference were run in float32, the bfloat16-tied positions would likely resolve (float32 ULP ≈ 2.4×10⁻⁷ vs bfloat16 ULP ≈ 0.016), and a larger σ would be needed to produce the same flip rate. The estimate σ = 0.007 is specific to bfloat16 inference.

5. **Conservative upper bound.** For the purpose of the verification protocol, σ = 0.007 calibrated from a production API represents a reasonable empirical estimate. The verifier can set σ to a moderately larger value (e.g., 2–3× this estimate) to account for variation across models and serving configurations, while still maintaining a tight entropy bound: even at σ = 0.02 (σ_eff ≈ 1,310), the nondeterminism entropy would be ~0.08 bits/token, or 0.5% of the token representation size.

---

## Summary

| Quantity | Value |
|----------|-------|
| Observed per-token first-divergence rate (Claude API) | 0.74% |
| Implied close-call rate | 1.48% |
| Calibrated σ (logit space) | 0.007 |
| Calibrated σ_eff (×65536) | 456 |
| Nondeterminism entropy per token (simplified) | 0.015 bits |
| Nondeterminism entropy per token (Gaussian fit) | 0.031 bits |
| Bits per token (vocab 32K) | 15 bits |
| **Nondeterminism / token size** | **0.1–0.2%** |
