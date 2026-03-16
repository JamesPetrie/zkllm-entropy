# Zero Knowledge Conditional Entropy Bounds for LLM Inference Verification
## With Probabilistic Tolerance for Hardware Non-Determinism

This repository forks [zkLLM](https://github.com/jvhs0706/zkllm-ccs2024) and adds a zero knowledge proof of conditional entropy for LLM inference outputs.

---

## Summary

We developed a zero-knowledge proof system that verifies LLM inference without requiring exact floating-point reproducibility. Instead of proving "the model produced exactly these tokens," we prove "at most X bits of hidden information could be encoded in this output." This handles real-world GPU non-determinism while still bounding covert channels like model weight exfiltration. Proof generation is ~1000× slower than inference, but random sampling makes it practical. Inference verification could be used to catch bugs in inference stacks, detect model weight exfiltration through API outputs, verify compute usage for international agreements, or confirm that a purchased model matches what's actually being served.

---

## What Is Proven

**Inputs (public or committed):**
- Model weights W (committed via Pedersen commitment — hidden from verifier)
- Input token sequence x (public)
- Output token sequence o (public)
- Hardware noise parameter σ (public, calibrated empirically)

**Output:**
- Conditional entropy bound H (public)
- Zero-knowledge proof π

**The proof guarantees:**

Given the committed model weights W and public input x, the claimed output sequence o has conditional entropy at most H bits. This means at most H bits of covert information could be hidden in the output.

---

## The Conditional Entropy Argument

### Why conditional entropy bounds covert channel capacity

Any covert message encoded in output o must survive compression. By Shannon's source coding theorem, o can be compressed to H(o | x, W) bits given the input and model. Therefore, covert information in o is bounded by H(o | x, W).

### Why approximations give valid upper bounds

By Gibbs' inequality, for any approximation q of the true token distribution p:

$$H(o | x, W) = -\sum_i \log_2 p(o_i) \leq -\sum_i \log_2 q(o_i)$$

This means we don't need exact token probabilities. Any reasonable approximation yields a valid (though possibly loose) upper bound on covert capacity.

### Handling hardware non-determinism

Real GPU inference is non-deterministic: the same model on the same input can produce slightly different logits due to floating-point accumulation order. We model this as Gaussian noise on logits:

$$q(o_i) = \frac{\Phi\left(\frac{v^* - \ell_i}{\sigma\sqrt{2}}\right)}{\sum_j \Phi\left(\frac{v^* - \ell_j}{\sigma\sqrt{2}}\right)}$$

Where:
- v* is the maximum logit (greedy decoding)
- ℓ_i is the logit for token i
- σ is calibrated so the noise model predicts ~97% exact token matches
- Φ is the standard normal CDF

This gives the probability that token i would be selected under hardware noise, even if it wasn't the exact argmax in the verifier's computation.

---

## What Is Inference Verification?

Verifying that a token stream is consistent with inference by a claimed model on a claimed input — like model fingerprinting, but with a slightly different goal: we are not trying to rule out other ways of generating the text.

### Why verify inference?

- Find bugs in an inference stack
- Verify that the model purchased is the same one that is served
- Verify that regulator evals run on the correct model
- Verify that model weights aren't being secretly exfiltrated in inference results
- Verify that compute is being used as claimed (potentially for international agreements about AI)

### Why do this in zero knowledge?

- Enables verification of closed-weight models where the verifier may not have access to model weights
- Protects against model weight theft even if the verifier has model access
- Side benefit: formal security guarantees that don't rely on a complex software and hardware stack

---

## What Was Missing Before This Project

Existing work (e.g. [zkLLM](https://github.com/jvhs0706/zkllm-ccs2024), [zkML](https://github.com/zkml), [Rinberg et al.](https://arxiv.org/pdf/2511.02620), and [Karvonen et al.](https://arxiv.org/abs/2511.20621)) either verifies exact inference on finite fields (not using floating point tensor cores), or classifies whether a token stream is consistent with a claimed model without a zero knowledge guarantee. There was no zero knowledge verification of efficient floating point inference — to use prior ZK approaches you would have had to fully rewrite the inference stack to use finite fields instead of floating point.



## Quickstart

Requires a CUDA-capable GPU and conda.

### 1. Build

```bash
conda activate zkllm-env
cd zkllm-ccs2024
make -j16 all
```

### 2. Generate public parameters for the lm_head logit vectors (once)

```bash
srun --gpus=1 --pty bash
./ppgen 32768 ./zkllm-workdir/Llama-2-7b/lm_head-pp.bin
```

### 3. Run the existing zkLLM layer proofs, then generate logit tensors

```bash
python run_proofs.py --model_size 7 --seq_len 1024 --num_layers 32
python gen_logits.py --model_size 7 --seq_len 1024 \
    --generators ./zkllm-workdir/Llama-2-7b/lm_head-pp.bin \
    --output_dir ./zkllm-workdir/Llama-2-7b/logits
```

### 4. Run the entropy prover

```bash
./zkllm_entropy \
    ./zkllm-workdir/Llama-2-7b/logits \
    ./zkllm-workdir/Llama-2-7b/logits/tokens.txt \
    proof.bin \
    3277 \
    --generators ./zkllm-workdir/Llama-2-7b/lm_head-pp.bin \
    --commits    ./zkllm-workdir/Llama-2-7b/logits
```

**Parameters:**
- `logits_dir`: Directory containing logit tensors (logits_0.bin, logits_1.bin, ...)
- `tokens.txt`: File containing the actual output token IDs
- `proof.bin`: Output proof file
- `3277`: σ_eff = σ × 65536 (calibrate by running inference twice and fitting match rate)

**Output:**
- Conditional entropy bound in scaled fixed-point units
- Zero-knowledge proof file (proof.bin)

### 5. Run tests

```bash
./test_zkargmax && ./test_zklog && ./test_zknormalcdf && ./test_zkentropy
```

---

## Calibrating σ

The noise parameter σ should be calibrated empirically:

1. Run the same model on the same input twice with temperature 0
2. Count what fraction of output tokens match exactly
3. Find σ such that the noise model predicts this match rate

Target: ~97% exact match rate. For Llama-2-7B, σ ≈ 0.05 (σ_eff = 3277 with 65536 scaling).

---

## Architecture

```
zkllm_entropy.cu
  └─ zkentropy.cuh/cu (zkConditionalEntropy)
       ├─ zkargmax.cuh/cu   — proves token is argmax via bit decomposition
       ├─ zknormalcdf.cuh/cu — proves Φ(d/σ) via lookup table
       └─ zklog.cuh/cu       — proves −log₂(q) via lookup table
```

**Per-position pipeline:**
1. Prove argmax: find max logit token t* and prove it's the maximum
2. Compute differences: d[i] = v* − logits[i]
3. Prove CDF: win_probs[i] = Φ(d[i] / σ√2)
4. Normalize: q[i] = win_probs[i] / Σ_j win_probs[j]
5. Prove log: surprise = −log₂(q[actual_token])
6. Sum surprises across all positions for total entropy bound
