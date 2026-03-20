# Improvement Opportunities from Other zkML Projects

An analysis of techniques from zkGPT, ZKTorch, zkPyTorch, and other systems that could be incorporated into zkLLM/zkllm-entropy, cross-referenced with the bottlenecks identified in [zkllm-entropy-scaling-analysis.md](zkllm-entropy-scaling-analysis.md).

---

## Bottleneck Recap

From the scaling analysis, the dominant bottlenecks at scale are:

1. **Self-attention softmax proof: O(n^2 x d)** — accounts for ~99% of proving time at 1M context
2. **FFN matrix multiply proofs: O(L x n x d x d_ff)** — dominates at short contexts
3. **Weight commitment MSM: O(P x log P)** — one-time cost, significant for large models
4. **Single-GPU limitation** — no multi-GPU support exists

The entropy proof itself is <1% of cost and is not the bottleneck.

---

## High-Value Techniques to Incorporate

### 1. Constraint Fusion (from zkGPT) — Medium Effort, 30-50% Reduction in Non-Linear Proof Cost

**What it is:** zkGPT observes that adjacent quantization/rounding steps in the proof pipeline produce range constraints that can be merged. For example, if step A proves `x in [0, 2^16)` and step B immediately proves `y = f(x)` where `y in [0, 2^16)`, the intermediate constraint can be fused into a single Lasso lookup.

**Where it helps in zkLLM:** Every non-linear operation (RMSNorm, SwiGLU, softmax) involves quantization → lookup → requantization chains. The current implementation proves each step independently. Constraint fusion would merge adjacent range proofs.

**Estimated impact:**
- zkGPT reports ~30% reduction in attention rounding constraints, ~50% in GELU/normalization
- In zkLLM, non-linear proofs (tlookup) are a minority of total cost for large models (~10-20%), so the overall impact is **~5-10% total proving time reduction**
- More significant at small model sizes or short contexts where attention doesn't dominate

**Implementation difficulty:** Medium. Requires refactoring the tlookup call sites to pass constraint metadata forward, but doesn't require changing the underlying proof system.

**Recommendation: Worth doing.** Modest improvement, but relatively straightforward to implement and it compounds with other optimizations.

---

### 2. Parallel Proof Accumulation (from ZKTorch) — High Effort, Potentially Transformative

**What it is:** ZKTorch extends the Mira accumulation scheme from sequential to parallel accumulation. Instead of proving layers one by one, it builds a Merkle tree of proof accumulations, enabling parallel proving of independent circuit blocks. This gives 6x speedup on GPT-J.

**Where it helps in zkLLM:** The scaling analysis identifies **layer parallelism** as potentially offering up to 128x speedup (for 128-layer models), but notes it requires proof composition. ZKTorch's parallel accumulation is exactly this — it's the concrete mechanism for composing independent layer proofs.

**Estimated impact:**
- At 8B/32 layers: up to ~32x speedup with enough GPUs (one layer per GPU)
- At 1T/128 layers: up to ~128x speedup
- ZKTorch achieved 6.2x on GPT-J with 64 threads — the gap from 6.2x to theoretical 32x is due to sequential dependencies and accumulation overhead

**Implementation difficulty:** High. This requires:
1. Adopting or implementing a recursive/accumulation scheme (Mira, Nova, or similar)
2. Replacing the current monolithic proof assembly with composable per-layer proofs
3. Significant refactoring of the proof pipeline
4. ZKTorch uses KZG on BN254, while zkLLM uses Pedersen on BLS12-381 — the accumulation scheme would need to work with BLS12-381 or the curve would need to change

**Key concern:** ZKTorch's proof sizes (6-23 MB) are much larger than zkLLM's (~188 kB). The Mira accumulation approach trades proof size for parallelism. Whether this tradeoff is acceptable depends on the use case.

**Recommendation: Study carefully, don't adopt wholesale.** The principle of parallel proof composition is exactly what the scaling analysis calls for in Phase 3. But ZKTorch's specific approach (Mira + KZG + BN254) may not be the best fit for zkLLM's architecture. A Nova/HyperNova folding approach over BLS12-381 might integrate more naturally. The key takeaway is that **someone has actually built working parallel proof composition for LLMs** — study their Rust code to understand the engineering challenges.

---

### 3. GKR Protocol for Matrix Multiplications (from zkGPT/zkPyTorch) — Medium Effort, 2-5x Speedup on Linear Layers

**What it is:** Both zkGPT and zkPyTorch use the GKR (Goldwasser-Kalai-Rothblum) protocol for proving matrix multiplications, rather than the sumcheck-over-multilinear-extensions approach used by zkLLM. GKR exploits the layered circuit structure of matrix multiplication, reducing the prover's bookkeeping overhead.

**Where it helps in zkLLM:** The scaling analysis identifies FFN matrix multiply proofs as the second-largest bottleneck. zkGPT specifically reports a **6.5x speedup** on their optimized matrix multiplication bookkeeping table computation vs. the classical approach (which is close to what zkLLM uses).

**Estimated impact:**
- 2-5x speedup on all linear layer proofs (QKV projections, FFN, lm_head, output projection)
- Linear layers are ~60% of proving cost at short contexts (n=1024), ~30% at medium contexts (n=16K), negligible at n=1M (where attention dominates)
- Net impact: **~1.5-3x overall speedup at n=1024-4096**, diminishing at longer contexts

**Implementation difficulty:** Medium. The GKR protocol is well-understood and has open-source implementations (including in zkGPT's code). The main work is:
1. Implementing a GKR prover for dense matrix multiplication on GPU
2. Replacing the current sumcheck-based linear layer proofs
3. Ensuring the GKR outputs (evaluation claims) are compatible with the rest of the proof pipeline

**Recommendation: Worth doing for short-to-medium context workloads.** This is probably the best effort-to-impact ratio among all techniques. The GKR protocol for matmul is well-understood, and zkGPT has demonstrated it works. It won't help with the n^2 attention bottleneck, but it significantly improves the constant factor for everything else.

---

### 4. Lasso Lookup Protocol (from zkGPT) — Medium Effort, Potential Improvement Over tlookup

**What it is:** zkGPT uses the Lasso lookup protocol (Setty, Thaler, Wahby) instead of the plookup-derived tlookup used in zkLLM. Lasso is a "lookup singularity" approach that avoids the sorting step required by plookup, using instead a decomposition into smaller sub-tables.

**Where it helps in zkLLM:** tlookup is used for all non-linear operations (ReLU, SwiGLU, softmax exponentiation, RMSNorm). The scaling analysis notes that non-linear proofs scale as O(n x d_ff x log(d_ff)) for FFN SwiGLU.

**Estimated impact:** Unclear without benchmarking. Lasso's theoretical advantage is sublinear prover time in table size (the prover only touches table entries that are actually looked up). For zkLLM's use case:
- Softmax exponentiation: table size 2^16 with many repeated lookups — Lasso's advantage may be minimal
- SwiGLU: table size 2^16 — similar situation
- The real win would be if the table size needs to increase (e.g., for higher-precision quantization)

**Implementation difficulty:** Medium-high. Lasso is structurally different from plookup/tlookup and would require replacing the tlookup infrastructure. However, Lasso has open-source implementations.

**Recommendation: Low priority.** The theoretical advantages of Lasso over tlookup are marginal for zkLLM's current table sizes (2^16). If higher precision is needed in the future (e.g., 24-bit or 32-bit quantization requiring 2^24+ table entries), Lasso becomes much more attractive. Monitor but don't implement now.

---

### 5. Expander/GKR Prover Backend (from zkPyTorch) — High Effort, Potentially Significant

**What it is:** Polyhedra's Expander is an open-source GKR-based prover written in Rust with CUDA support. It operates over the M61 Mersenne prime field (2^61 - 1), which has much faster arithmetic than BLS12-381's ~254-bit field.

**Where it helps in zkLLM:** The scaling analysis identifies optimized field arithmetic as a 2-3x improvement opportunity. The M61 field is approximately:
- 4x faster for multiplication (64-bit vs 256-bit)
- 4x less memory per field element (8 bytes vs 32 bytes)

**Estimated impact:**
- 2-4x speedup on all field arithmetic operations
- 4x memory reduction for all intermediate tensors
- The memory reduction is particularly important — it would make 1M context more feasible

**Key tradeoff:** BLS12-381 supports pairing-based polynomial commitments (KZG), which give constant-size proofs. M61 does not support pairings — you'd need to use hash-based commitments (FRI/Orion) or inner-product commitments (Hyrax/Bulletproofs), which have larger proofs (polylogarithmic rather than constant).

**Implementation difficulty:** Very high. This is essentially a proof-system migration, not an incremental improvement. Would require:
1. Rewriting all field arithmetic from BLS12-381 to M61
2. Replacing Pedersen commitments with an M61-compatible scheme
3. Potentially changing proof sizes and verification costs

**Recommendation: Not now, but keep on radar.** The Mersenne prime field approach is genuinely faster for proving, and Expander is well-engineered open-source code. But migrating from BLS12-381 is a massive undertaking. A more practical approach: use M61/Goldilocks for "inner" proofs (the heavy computation) and wrap with a BLS12-381 "outer" proof for the final commitment/verification. This is the approach Plonky2/Plonky3 take.

---

### 6. ONNX/PyTorch Model Import Pipeline (from ZKTorch/zkPyTorch) — Medium Effort, Usability Win

**What it is:** Both ZKTorch and zkPyTorch accept standard ONNX model files and automatically compile them to proof circuits. zkLLM currently requires manual model decomposition and weight export via custom Python scripts.

**Where it helps:** Not a performance improvement, but a significant usability improvement. Currently, supporting a new model architecture requires writing custom Python scripts and potentially new CUDA kernels. An ONNX frontend would make the system model-agnostic.

**Implementation difficulty:** Medium. The ONNX specification for transformer models is well-defined. The main work is:
1. ONNX graph parsing and operator mapping
2. Automatic weight quantization and commitment generation
3. Circuit generation from the operator graph

**Recommendation: Lower priority than performance work, but valuable for adoption.** If the goal is for others to use the system, an ONNX frontend is important. If the goal is research on a specific model, the current manual approach is fine.

---

## Techniques That Are NOT Worth Incorporating

### Circuit Squeeze (from zkGPT)
Circuit squeeze improves multi-threaded CPU parallelism by breaking the sequential dependency between transformer blocks in the GKR circuit. Since zkLLM runs on GPU with very different parallelism patterns, this technique doesn't transfer.

### Mira Accumulation Specifically (from ZKTorch)
The specific Mira scheme requires KZG commitments on BN254, which is incompatible with zkLLM's BLS12-381 Pedersen commitments. The general principle of proof accumulation is valuable (see item 2 above), but not this specific scheme.

### o1js Backend (from ZK-DeepSeek)
A JavaScript SNARK library is obviously not competitive with CUDA. Nothing to learn here.

### VOLE-based Proofs (from Lu et al.)
VOLE-based ZK (as in the protocol compared against in zkGPT) produces 2+ GB proofs. The proof size regression is unacceptable for any practical deployment.

---

## Prioritized Roadmap

Given the scaling analysis bottlenecks, here is the recommended order of implementation:

| Priority | Technique | Source | Effort | Impact | Bottleneck Addressed |
|----------|-----------|--------|--------|--------|---------------------|
| **1** | GKR for matrix multiplications | zkGPT | Medium | 2-5x on linear layers | FFN proofs (#2 bottleneck) |
| **2** | Constraint fusion | zkGPT | Medium | 5-10% total | Non-linear proof overhead |
| **3** | Multi-GPU tensor parallelism | Novel (informed by ZKTorch) | High | Near-linear with GPUs | Single-GPU limitation (#4) |
| **4** | Parallel proof composition | ZKTorch (principle, not specifics) | High | Up to Lx (layers) | Single-GPU limitation (#4) |
| **5** | Smaller field for inner proofs | zkPyTorch/Expander | Very High | 2-4x all operations | Field arithmetic overhead |
| **6** | Lasso lookups | zkGPT | Medium | Marginal now | Future: higher precision |
| **7** | ONNX model import | ZKTorch/zkPyTorch | Medium | Usability only | Developer experience |

Items 1-2 are concrete, well-understood improvements that can be implemented incrementally. Items 3-4 are the transformative changes needed for scaling but require significant engineering. Item 5 is a research-level change. Items 6-7 are nice-to-haves.
