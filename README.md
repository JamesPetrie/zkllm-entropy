# Zero Knowledge Conditional Entropy Bounds for LLM Inference Verification
## With Probabilistic Tolerance for Hardware Non-Determinism

This repository forks [zkLLM](https://arxiv.org/abs/2404.16109) and adds a zero knowledge proof of conditional entropy for LLM inference outputs.

## What is inference verification?

Verifying that a token stream is consistent with inference by a claimed model on a claimed input — like model fingerprinting, but with a slightly different goal: we are not trying to rule out other ways of generating the text.

## Why verify inference?

- Find bugs in an inference stack
- Verify that the model purchased is the same one that is served
- Verify that regulator evals run on the correct model
- Verify that model weights aren't being secretly exfiltrated in inference results
- Verify that compute is being used as claimed (potentially for international agreements about AI)

## Why do this in zero knowledge?

- Enables verification of closed-weight models where the verifier may not have access to model weights
- Protects against model weight theft even if the verifier has model access
- Side benefit: formal security guarantees that don't rely on a complex software and hardware stack

## What was missing before this project

Existing work (e.g. zkLLM, zkML, [this paper](https://arxiv.org/pdf/2511.02620), and [this paper](https://arxiv.org/abs/2511.20621)) either verifies exact inference on finite fields (not using floating point tensor cores), or classifies whether a token stream is consistent with a claimed model without a zero knowledge guarantee. There was no zero knowledge verification of efficient floating point inference — to use prior ZK approaches you would have had to fully rewrite the inference stack to use finite fields instead of floating point.

## Contributions

- Reframed the problem as bounding conditional entropy in the output (a quantity, not a classification)
- Showed that arbitrary approximations of token probability provide a strict upper bound on conditional entropy (Gibbs' inequality)
- Developed a simple formula for token probabilities that accounts for hardware noise: normal distribution with difference in logits between the token and the max token
- Prototyped these modifications by forking zkLLM and adding a zero knowledge proof of conditional entropy

## Quickstart

Requires a CUDA-capable GPU and conda.

**1. Build**
```bash
conda activate zkllm-env
cd zkllm-ccs2024
make -j16 all
```

**2. Generate public parameters for the lm_head logit vectors (once)**
```bash
srun --gpus=1 --pty bash
./ppgen 32768 ./zkllm-workdir/Llama-2-7b/lm_head-pp.bin
```

**3. Run the existing zkLLM layer proofs, then generate logit tensors**
```bash
python run_proofs.py --model_size 7 --seq_len 1024 --num_layers 32
python gen_logits.py --model_size 7 --seq_len 1024 \
    --generators ./zkllm-workdir/Llama-2-7b/lm_head-pp.bin \
    --output_dir ./zkllm-workdir/Llama-2-7b/logits
```

**4. Run the entropy prover**
```bash
./zkllm_entropy \
    ./zkllm-workdir/Llama-2-7b/logits \
    ./zkllm-workdir/Llama-2-7b/logits/tokens.txt \
    proof.bin \
    3277 \
    --generators ./zkllm-workdir/Llama-2-7b/lm_head-pp.bin \
    --commits    ./zkllm-workdir/Llama-2-7b/logits
```

The output is a conditional entropy bound in scaled fixed-point units and a proof file. The `sigma_eff` parameter (3277 ≈ 0.05 × 65536) should be calibrated empirically by running inference twice on the same input and fitting the token match rate.

**5. Run tests**
```bash
./test_zkargmax && ./test_zklog && ./test_zknormalcdf && ./test_zkentropy
```

## Conclusion

- Prototyped a zero knowledge proof of conditional entropy
- Demonstrated feasibility of zero knowledge inference verification without reducing floating point inference efficiency
- (~1000x slower to generate the proof than to do inference, but this can be done on a small random sample to amortize the cost)
