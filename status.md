# Implementation Status

## Build / Test Status

| Binary | Status |
|---|---|
| `test_zkargmax` | **All 6 tests PASS** (job 1898) |
| `test_zklog` | **All 5 tests PASS** (job 1898) |
| `test_zknormalcdf` | **All 5 tests PASS** (job 1898) |
| `test_zkentropy` | **All 6 tests PASS** (job 1898) |
| `zkllm_entropy` | Built (needs rebuild after architecture change) |

## Files

| File | Description |
|---|---|
| `zkargmax.cuh/cu` | ZK argmax via bit-decomposition range proofs |
| `zklog.cuh/cu` | ZK −log₂ via tLookupRangeMapping |
| `zknormalcdf.cuh/cu` | ZK normal CDF via tLookupRangeMapping |
| `zkentropy.cuh/cu` | Conditional entropy pipeline (compute + prove); logit linkage delegated to caller |
| `zkllm_entropy.cu` | Main prover: proves final RMSNorm + lm_head via committed weights, then entropy |
| `commit_final_layers.py` | Quantise and commit lm_head + final norm weights (run once) |
| `gen_entropy_inputs.py` | Generate final_norm-rms_inv.bin + tokens.txt from layer-31-output.bin |
| `verify_entropy.py` | CPU-only verifier: checks all arithmetic claims in a proof file |
| `calibrate_sigma.py` | Find sigma_eff from empirical token match rate or a target rate |
| `test_zkargmax.cu` | Unit tests for zkArgmax |
| `test_zklog.cu` | Unit tests for zkLog |
| `test_zknormalcdf.cu` | Unit tests for zkNormalCDF |
| `test_zkentropy.cu` | Integration tests for zkConditionalEntropy |

## Architecture

```
committed W_norm  committed W_lm           public sigma_eff
     |                  |                        |
zkRMSNorm(hidden, W_norm) → normed_hidden        |
     |                                           |
zkFC(normed_hidden, W_lm) → logits              |
     |                                           |
zkConditionalEntropy(logits, tokens, sigma_eff) → entropy bound H
```

The hidden state (`layer-31-output.bin`) comes from the existing zkLLM layer
proofs.  No separate logit commitment is needed: the logits are proven to
derive from committed weights via `zkFC + verifyWeightClaim`.

### Per-position entropy pipeline (inside zkConditionalEntropy)

1. `zkArgmax::prove` — bit-decomposition range proof that t_star is the argmax
2. GPU: `diffs[i] = v_star − logits[i]`; `win_probs[i] = (1 − Φ(diff_i/σ)) × cdf_scale`
3. CPU: `total_win = sum(win_probs)`; `q_idx = floor(win_prob[actual] × 2^log_precision / total_win)`
4. `zkLog::compute` — `surprise = −log₂(q_idx / 2^log_precision) × log_scale`
5. Accumulate over sequence → total entropy bound H

## Recent Debugging

- **Job 2063 / 2066 (BIT_WIDTH=16/25)**: `zkArgmax::prove: bit reconstruction mismatch` — root cause: Llama-2 logit range spans >512 float units (some tokens have logits < -512); after int scaling (×65536), max diff = v_star_int + |min_logit_int| > 2^25. Fixed by increasing BIT_WIDTH to 32 (2^32/65536 = 65536 float units, safe upper bound). Added max_diff diagnostic to zkargmax.cu to print the actual range.

## Remaining TODOs

1. **Rebuild and re-run** (job 2066 with BIT_WIDTH=32): rebuild zkllm_entropy, run prover, verify.

2. **Commit final weights** (once per model):
   ```bash
   srun --gpus=1 --pty bash
   cd /mnt/sharefs/user50/zk/zkllm-ccs2024
   source /mnt/sharefs/user50/miniconda3/etc/profile.d/conda.sh && conda activate zkllm-env
   # lm_head-pp.bin already exists from ppgen (job 2046)
   python commit_final_layers.py --model-size 7
   ```

3. **Generate entropy inputs** (once per inference run):
   ```bash
   python gen_entropy_inputs.py --model-size 7 --seq-len 1024
   ```

4. **Run entropy prover**:
   ```bash
   ./zkllm_entropy ./zkllm-workdir/Llama-2-7b tokens.txt proof.bin 3277
   ```

5. **Verify**:
   ```bash
   python verify_entropy.py proof.bin
   ```

6. **Calibrate sigma** (requires two inference runs):
   ```bash
   python calibrate_sigma.py \
       --logits-dir ./zkllm-workdir/Llama-2-7b \
       --tokens1 tokens_run1.txt --tokens2 tokens_run2.txt
   ```
   Or with a target match rate:
   ```bash
   python calibrate_sigma.py \
       --logits-dir ./zkllm-workdir/Llama-2-7b \
       --target-match-rate 0.97
   ```
   Note: `calibrate_sigma.py` reads logit gap distributions from `logits_N.bin` files.
   These should be generated separately (e.g. by `gen_logits.py` without commitment).
