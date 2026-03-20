# Comparative Analysis: zkML/zkLLM Efficiency Claims

An analysis of zero-knowledge proof systems for LLM inference, comparing their benchmarks, credibility, and practical viability against the zkLLM baseline (Sun et al., CCS 2024).

---

## Summary Comparison Table

| System | Venue | Largest Model Proved | Seq Len | Prover Time | Proof Size | Verifier Time | Hardware | Code? |
|--------|-------|---------------------|---------|-------------|------------|---------------|----------|-------|
| **zkLLM** | CCS 2024 | LLaMA-2-13B (13B) | 2048 | 803s (13.4 min) | 188 kB | 3.95s | A100 GPU | Yes (CUDA, archived) |
| **zkGPT** | USENIX Sec 2025 | GPT-2 (117M) | 32 | 21.8s | 101 kB | 0.35s | 16-core Xeon CPU | Yes (C++, code dump) |
| **ZKTorch** | arXiv 2025 | LLaMA-2-7B (7B) | 1 token | 2645s (44 min) | 22.85 MB | 100s | 64-thread Xeon, 4TB RAM | Yes (Rust, active) |
| **zkPyTorch** | ePrint 2025 | Llama-3-8B (8B) | Unspecified | 150s/token | Not reported | Not reported | Single CPU core | No (Expander prover only) |
| **ZK-DeepSeek** | arXiv 2025 | Single matmul from DeepSeek-V3 | N/A | 56.7 hrs (one op) | 32-36 kB | 342ms | RTX 5090 | Yes (JavaScript) |
| **zkCNN** | CCS 2021 | VGG-16 (15M) | N/A (vision) | 88.3s | 341 kB | 59ms | Not specified | Partial (no ZK) |
| **Folding zkLLM** | ePrint 2024 | None (theory only) | N/A | N/A | N/A | N/A | N/A | No |

---

## Detailed Analysis

### 1. zkLLM (Sun, Li, Zhang — CCS 2024) — The Baseline

**Status: The most credible and complete system. Published at a top venue with working code.**

zkLLM is the benchmark everything else should be measured against. Key facts:

- **Models:** OPT-125M through OPT-13B, LLaMA-2-7B, LLaMA-2-13B
- **Sequence length: 2048 tokens** — the only system tested at a realistic context length
- **Prover time:** 620s for LLaMA-2-7B, 803s for LLaMA-2-13B (full inference, all layers)
- **Proof size:** 183 kB (7B), 188 kB (13B) — remarkably small and near-constant
- **Verifier time:** 2.36s (7B), 3.95s (13B)
- **Memory:** 15.5 GB (7B), 23.1 GB (13B) — fits on a single A100-40GB
- **Hardware:** NVIDIA A100-40GB GPU, 12-core AMD EPYC, 124.5 GB RAM

**Architecture:** Sumcheck protocol over multilinear extensions + Pedersen commitments on BLS12-381. Novel contributions include tlookup (parallelized GPU lookup argument) and zkAttn (dedicated attention protocol exploiting shift-invariance of softmax).

**Quantization:** 16-bit fixed-point (scaling factor 2^16). Perplexity impact is negligible (<0.1 for most models).

**Proves full inference:** Yes — embedding through all transformer layers through final logits projection.

**Code:** [GitHub](https://github.com/jvhs0706/zkllm-ccs2024) — 58 stars, 31 forks. Archived July 2025. Functionally complete for demonstration but:
- Prover/verifier not separated (same process)
- Fiat-Shamir transform NOT implemented (proofs are interactive, not non-interactive)
- Author has stated he no longer maintains the project
- Intermediate values leaked to files (security issue)

**Credibility: High.** Peer-reviewed at ACM CCS, the top systems security venue. Benchmarks are thorough with per-model breakdowns and perplexity validation. The main caveat is that the published code is a demo, not production-ready.

---

### 2. zkGPT (Qu, Sun et al. — USENIX Security 2025)

**Status: Solid engineering on a tiny model at a toy sequence length. Claims of superiority over zkLLM are not supported.**

**What they actually proved:** GPT-2 (117M params) at sequence length 32. That's it. No larger models, no longer contexts.

**Performance on GPT-2 (seq len 32):**
- Prover time: 21.8s (32 threads, all optimizations)
- Proof size: 101 kB
- Verifier time: 0.35s
- Hardware: 16-core Intel Xeon 6126, 200 GB RAM (CPU only, no GPU)

**Key innovations:**
- Constraint fusion: merges adjacent rounding constraints (~30-50% reduction)
- Circuit squeeze: better multi-thread parallelism than sequential block-by-block proving
- Optimized matrix multiplication bookkeeping (6.5x faster)
- Lasso lookup protocol for non-linear ops (advice paradigm)

**The zkLLM comparison is misleading:**
- zkGPT prover time: 21.8s on CPU. zkLLM prover time: 15.8s on A100 GPU.
- **zkLLM is 30% faster**, even though zkGPT is CPU-only.
- The authors claim they "would very likely outperform [zkLLM] under the same hardware" — this is speculation, not a benchmark.
- The 279x and 185x speedup headlines compare against much older/slower systems (Hao et al. at 6096s, ZKML at 4026s), not zkLLM.

**What's NOT proved:** The final linear projection (768 x 50257 vocabulary projection) is explicitly omitted, following prior work.

**Credibility assessment:**
- Peer-reviewed at USENIX Security — a top venue. The paper itself is legitimate.
- But the scope is limited: only GPT-2, only 32 tokens. Whether the techniques scale to 7B+ models at 2048+ tokens is unknown.
- The 200 GB RAM requirement for just GPT-2 raises scalability concerns.
- Code is a single-commit dump with no tests or CI.
- Attention is O(s^2) — scaling from 32 to 2048 tokens means ~4000x more attention computation. The paper provides no evidence this is tractable.

**Bottom line:** A genuine contribution to ZK proof engineering, but it operates in a fundamentally different regime than zkLLM. Comparing 117M params at 32 tokens to 13B params at 2048 tokens is not meaningful. The paper does not demonstrate that its techniques work at scale.

---

### 3. ZKTorch (Chen, Tang, Kang — arXiv 2025)

**Status: The most ambitious open-source effort. Real code, real models, but extreme hardware requirements and minimal sequence lengths.**

**What they actually proved:**
- GPT-J (6B): 1397s (~23 min), 6.54 MB proof, 2 tokens
- LLaMA-2-7B: 2645s (~44 min), 22.85 MB proof, 1 token
- BERT (110M): 880s, 4.88 MB proof, 1 token
- ResNet-50: 6270s, 85 kB proof
- Stable Diffusion XL: 68,950s (~19 hours)
- Also every model in MLPerf Edge Inference Suite v4.1

**Architecture:** Mira accumulation scheme (pairing-based recursive SNARK) with KZG commitments on BN254. Key contribution: extends Mira from sequential to parallel accumulation via Merkle tree reduction.

**Hardware:** 64-thread Intel Xeon Platinum 8358, **4 TB RAM**. CPU-only, no GPU.

**Comparison with zkLLM:**
- ZKTorch explicitly declines to compare with zkLLM because "zkLLM is GPU-based ... whereas ZKTorch serves for high-performance CPU-intensive environments."
- On GPT-2: ZKTorch 599s vs. ZKML (same group's prior work) 3601s → 6x speedup. But zkLLM does GPT-2 in ~16s on GPU.
- On LLaMA-2-7B at 1 token: ZKTorch takes 44 minutes on CPU. zkLLM proves 7B at 2048 tokens in ~10 minutes on GPU.

**Credibility assessment:**
- Code is real and substantial: 146 commits, 36 stars, Rust implementation
- The team (Daniel Kang at UIUC) has a strong track record in zkML
- But: 4 TB RAM is not commodity hardware
- Sequence lengths of 1-2 tokens make the LLM benchmarks nearly meaningless for practical applications
- Proof sizes (6-23 MB) are much larger than zkLLM (183 kB)
- Verification times (62-100s) are much slower than zkLLM (2-4s)

**Bottom line:** Impressive breadth (proves everything from ResNet to Stable Diffusion to LLaMA), but LLM results are at trivial sequence lengths on extreme hardware. The CPU-only approach means it cannot compete with GPU-accelerated systems like zkLLM for throughput. Best understood as a universal compiler rather than a competitive LLM prover.

---

### 4. zkPyTorch (Polyhedra Network — ePrint 2025)

**Status: Marketing document with insufficient benchmarks. Key metrics missing.**

**What they claim:** Llama-3-8B at 150 seconds per token on a single CPU core.

**What they actually report (the entire benchmark):**

| Model | Params | Proving Time |
|-------|--------|-------------|
| VGG-16 | 15.2M | 2.2s / image |
| Llama-3 | 8B | 150s / token |

That's it. The paper provides:
- No verification time
- No proof size
- No memory usage
- **No sequence length** (making the "150s/token" figure uninterpretable)
- No baseline comparisons
- No multi-core/GPU numbers

**Architecture:** GKR protocol via Polyhedra's Expander prover, operating over the M61 Mersenne prime field (2^61 - 1).

**Discrepancy:** The paper says 150s/token single-core. The blog says 2.7 hours single-core for Llama-3. These are consistent only if ~65 tokens were proved (65 × 150s ≈ 2.7 hrs), but neither source states this.

**Code:** zkPyTorch itself is NOT open source. The underlying Expander prover is open source, but the model compilation pipeline is proprietary.

**Credibility assessment:**
- The team is legitimate (Tiancheng Xie PhD from Berkeley under Dawn Song, strong prior work including Libra, Orion, zkBridge)
- Polyhedra has shipped production systems (zkBridge, 40M+ proofs)
- BUT: this is a 9-page paper with 2 benchmark numbers, posted on ePrint (not peer-reviewed)
- Missing verification time and proof size is a major red flag — these are the most basic metrics for any ZKP system
- "99.32% cosine similarity" for Llama-3 quantization is not a standard LLM eval metric and may hide quality degradation
- The paper reads as a product announcement, not a rigorous systems evaluation
- Polyhedra is a crypto company with a token (ZKJ) — commercial incentives to overstate performance

**Bottom line:** The team could plausibly build a competitive system, but the published evidence is far too thin to evaluate. Without sequence length, verification time, proof size, or memory usage, the "150s/token" number is essentially meaningless. Cannot be compared to zkLLM in any rigorous way.

---

### 5. ZK-DeepSeek (Wang — arXiv 2511.19902)

**Status: Misleading headline claims. Proved individual operations, not a model.**

**What they claim:** "Scalability to DeepSeek-V3 (671B parameters) with constant proof size and verification time."

**What they actually proved:** Individual layer components of DeepSeek-V3, tested in isolation:

| Component | Proving Time |
|-----------|-------------|
| Embedding (24×7168) | 4,823s |
| RMSNorm (24×7168) | 9,941s |
| RoPE (24×8192) | 19,275s |
| Softmax (3072 heads) | 39,456s |
| MatMul (24×7168×512) | **204,138s (~56.7 hours)** |

A single matrix multiplication took 56.7 hours. DeepSeek-V3 has thousands of such operations per forward pass. No end-to-end inference was attempted or demonstrated.

**Additional problems:**
- The model had to be requantized from 680 GB (BF16/BF8) to **2.5 TB** in integer format
- Backend uses **o1js** (a JavaScript SNARK library) — an unusual and likely performance-limiting choice
- Single-author paper, not peer-reviewed
- Proof sizes are 32-36 kB per component, but composing all components is not demonstrated

**Credibility: Low.** The "671B" headline is not supported by the benchmarks. This is a proof-of-concept for individual operations, not a viable inference verification system. Extrapolating naively, full DeepSeek-V3 inference would take years to prove.

---

### 6. Other Systems (Brief Notes)

**zkCNN (CCS 2021):** Proved VGG-16 (15M params) in 88.3s. Legitimate peer-reviewed work, but limited to CNNs. The open-source code explicitly states it "doesn't add complete zero-knowledge property." Historical significance as an early zkML system.

**EZKL:** Well-maintained open-source framework (ONNX → Halo2). Excels at small models (regression, random forests, SVMs). Has not published benchmarks for anything approaching LLM scale. Lagrange Labs' DeepProve claims 54-158x faster than EZKL for GPT-2.

**Folding-based zkLLM (ePrint 2024/480):** Purely theoretical proposal. No implementation, no benchmarks, no code. Single author, not peer-reviewed. Should be cited only as a theoretical exploration.

---

## Key Observations

### 1. zkLLM remains the only system with credible large-model benchmarks at realistic context lengths

No other system has demonstrated proving a 7B+ parameter model at 2048 tokens. The closest competitors either test at trivial sequence lengths (ZKTorch: 1-2 tokens, zkGPT: 32 tokens) or don't report sequence length at all (zkPyTorch).

### 2. The GPU advantage is decisive

zkLLM's GPU-based approach (A100) dramatically outperforms CPU-based systems. zkLLM proves LLaMA-2-7B in ~10 minutes on GPU. ZKTorch proves the same model in ~44 minutes on a 64-thread server with 4TB RAM — at only 1 token vs. 2048.

### 3. Proof sizes vary enormously

| System | Proof Size |
|--------|-----------|
| zkLLM | ~188 kB (near constant across models) |
| zkGPT | ~101 kB (GPT-2 only) |
| ZKTorch | 6-23 MB (varies with model) |
| zkPyTorch | Not reported |

zkLLM's near-constant ~188 kB proof size is a remarkable property that none of the competitors match at scale.

### 4. "Sequence length 1" benchmarks for LLMs are nearly meaningless

An LLM proving system that only works at 1-2 tokens has not demonstrated practical viability. Attention is O(n^2) in sequence length, so scaling from 1 token to 2048 tokens fundamentally changes the computational profile. zkLLM is the only system that has confronted this challenge.

### 5. Several claims in the literature are not credible

- **ZK-DeepSeek's "671B" claim:** One matmul took 56 hours. Full inference would take years.
- **zkPyTorch's "150s/token":** Without knowing the sequence length, this number is uninterpretable. Missing verification time and proof size makes evaluation impossible.
- **zkGPT's "279x speedup":** Compares against obsolete baselines, not zkLLM. Against zkLLM, zkGPT is actually 30% slower.
- **Folding-based zkLLM:** Pure theory with no implementation.

### 6. No system proves floating-point inference

All systems require quantization to finite-field arithmetic. This is the gap that zkllm-entropy addresses: rather than requiring exact reproduction of finite-field inference, it bounds the conditional entropy of floating-point outputs.

### 7. Code availability and quality varies widely

| System | Code Status |
|--------|------------|
| zkLLM | Complete demo, archived, not production-ready |
| zkGPT | Single-commit dump, no tests |
| ZKTorch | Active development, 146 commits, substantial Rust codebase |
| zkPyTorch | Proprietary (only Expander prover is open source) |
| ZK-DeepSeek | Available but JavaScript-based |
| zkCNN | Available but incomplete (no actual ZK property) |

---

## Recommendations for zkllm-entropy Paper

When positioning zkllm-entropy relative to this landscape:

1. **zkLLM is the primary comparison point** — it's the only peer-reviewed system with credible large-model benchmarks at realistic context lengths, and zkllm-entropy directly extends its codebase.

2. **The fundamental gap all these systems share** is the requirement for finite-field arithmetic. None can verify actual GPU floating-point inference. This is zkllm-entropy's key contribution.

3. **Be cautious citing performance numbers from zkPyTorch and ZK-DeepSeek** — their benchmarks are incomplete or misleading.

4. **zkGPT and ZKTorch are legitimate but limited** — cite their techniques (constraint fusion, parallel accumulation) as complementary engineering advances, but note they haven't demonstrated scaling beyond GPT-2/tiny-context.

5. **The ~1000x proving overhead** (zkLLM takes ~10 min for inference that runs in <1s) is shared across all systems. zkllm-entropy's random sampling approach to make this practical is a distinct contribution regardless of which underlying ZK system is used.
