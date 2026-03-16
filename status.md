# Implementation Status

## Build / Test Status

| Binary | Status |
|---|---|
| `test_zkargmax` | **All 6 tests PASS** (job 1898) |
| `test_zklog` | **All 5 tests PASS** (job 1898) |
| `test_zknormalcdf` | **All 5 tests PASS** (job 1898) |
| `test_zkentropy` | **All 6 tests PASS** (job 1898) |
| `zkllm_entropy` | **Built** |
| `commit_logits` | **Built** |

## New files added

| File | Description |
|---|---|
| `zkargmax.cuh/cu` | ZK argmax via bit-decomposition range proofs |
| `zklog.cuh/cu` | ZK −log₂ via tLookupRangeMapping |
| `zknormalcdf.cuh/cu` | ZK normal CDF via tLookupRangeMapping |
| `zkentropy.cuh/cu` | Full conditional entropy pipeline (compute + prove) |
| `zkllm_entropy.cu` | Standalone binary: loads saved logit tensors, runs prover |
| `commit_logits.cu` | Utility: commits a logit FrTensor using ppgen generators |
| `test_zkargmax.cu` | Unit tests for zkArgmax |
| `test_zklog.cu` | Unit tests for zkLog |
| `test_zknormalcdf.cu` | Unit tests for zkNormalCDF |
| `test_zkentropy.cu` | Integration tests for zkConditionalEntropy |
| `gen_logits.py` | Python script: applies lm_head to final hidden state, saves logit tensors + commitments |

## Architecture summary

```
zkllm_entropy.cu
  └─ zkentropy.cuh/cu (zkConditionalEntropy)
       ├─ zkargmax.cuh/cu   — argmax + bit-decomp range proof
       ├─ zknormalcdf.cuh/cu — Phi(d/sigma) lookup (single element, no D%N constraint)
       └─ zklog.cuh/cu       — −log2(q) lookup (single element)
```

### Per-position pipeline

CDF is computed for all vocab tokens; the actual sum of win probabilities
is used as the normalisation denominator (tighter bound than the earlier
`vocab_size × cdf_scale` approximation).

1. `zkArgmax::compute` + `prove` — find greedy token t_star, produce range proof
2. `Commitment::me_open(logits, generators, e_actual)` — prove `logits[actual_token]`
   is bound to the committed logit tensor; produces G1 proof elements
3. GPU: `diffs[i] = v_star − logits[i]` for all i; `win_probs[i] = (1 − Φ(diff_i/σ)) × cdf_scale`
4. CPU: `total_win = sum(win_probs)`; `q_idx = floor(win_prob[actual] × 2^log_precision / total_win)`
5. `zkLog::compute` on 1 element — `surprise = −log₂(q_idx / 2^log_precision) × log_scale`
6. Return surprise; accumulate over sequence

### Prove path output

- `vector<Polynomial>& proof` — argmax range proof polynomials + claimed scalar values
- `vector<G1Jacobian_t>& g1_proof` — G1 curve points from `Commitment::me_open`
  (one set of log2(vocab_size) triplets per position)

## Remaining TODOs

1. **Logit generators**: Run `./ppgen 32768 ./zkllm-workdir/Llama-2-7b/lm_head-pp.bin`
   before calling `zkllm_entropy --generators`.

2. **Logit tensor generation**: Run `gen_logits.py` to apply lm_head to the final
   hidden state and save per-position logit tensors + commitments to disk.

3. **Verifier not yet written**: A verifier would load the G1 proof elements and
   check each `me_open` proof against the stored logit commitment.

4. **Calibration not done**: `sigma_eff` is user-provided; empirical calibration
   (two runs, measure token match rate) has not been performed.

## How to run (end-to-end)

### 1. Generate public parameters for logit vectors (once)
```bash
srun --gpus=1 --pty bash
cd /mnt/sharefs/user50/zk/zkllm-ccs2024
./ppgen 32768 ./zkllm-workdir/Llama-2-7b/lm_head-pp.bin
```

### 2. Run zkLLM layer proofs + generate logit tensors
```bash
# Run existing layer proofs
python run_proofs.py --model_size 7 --seq_len 1024 --num_layers 32
# Apply lm_head, save logit tensors + commitments
python gen_logits.py --model_size 7 --seq_len 1024 \
    --generators ./zkllm-workdir/Llama-2-7b/lm_head-pp.bin \
    --output_dir ./zkllm-workdir/Llama-2-7b/logits
```

### 3. Run entropy prover
```bash
./zkllm_entropy \
    ./zkllm-workdir/Llama-2-7b/logits \   # logits_N.bin files
    tokens.txt \                            # one token id per line
    proof.bin \                             # output
    3277 \                                  # sigma_eff (0.05 * 65536)
    --generators ./zkllm-workdir/Llama-2-7b/lm_head-pp.bin \
    --commits    ./zkllm-workdir/Llama-2-7b/logits  # logits_N-commitment.bin
```

### 4. Run tests
```bash
srun --gpus=1 --pty bash
cd /mnt/sharefs/user50/zk/zkllm-ccs2024
./test_zkargmax && ./test_zklog && ./test_zknormalcdf && ./test_zkentropy
```
