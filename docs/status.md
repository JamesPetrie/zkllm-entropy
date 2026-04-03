# Implementation Status

## Build / Test Status

| Binary | Status |
|---|---|
| `test_zkargmax` | **All 6 tests PASS** (job 1898) |
| `test_zklog` | **All 5 tests PASS** (job 1898) |
| `test_zknormalcdf` | **All 5 tests PASS** (job 1898) |
| `test_zkentropy` | **All 6 tests PASS** (job 1898) |
| `zkllm_entropy` | Built (needs rebuild after architecture change) |
| `gold_test_zk_mask` | 5/10 PASS, remaining 5 pending GPU reset (branch: zk-masking-implementation) |
| `gold_test_zk_sumcheck` | Pending GPU reset (branch: zk-masking-implementation) |
| `test_zk_verifier` | **All 12 tests PASS** — CPU-only, no GPU needed (branch: zk-masking-implementation) |

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
| `zk_mask.cuh/cu` | ZK masking: vanishing polynomial + XZZ+19 transcript masking (CPU-side) |
| `zk_sumcheck.cuh/cu` | Degree-4 ZK sumcheck kernels: `zkip_zk`, `zkip_stacked_zk`, `inner_product_sumcheck_zk` |
| `sumcheck_verifier.h` | CPU verifier for standard and ZK sumcheck proofs |
| `verifier_utils.h` | Host-side Goldilocks arithmetic and proof parsing utilities |
| `test_zk_verifier.cpp` | CPU-only tests for ZK sumcheck verifier |
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

- **Job 2063 / 2066 (BIT_WIDTH=16/25)**: `zkArgmax::prove: bit reconstruction mismatch` — root cause traced to two bugs:

### Bug 1: `fr_gt` wrong comparison for negative-negative field elements (`zkargmax.cu`)
For two "negative" field elements (val[2..7]≠0, representing p−k), the original code used `av < bv`, but the correct comparison is `av > bv`: a larger raw value = p − (smaller k) = less negative = greater signed integer.
**Fix**: Changed to `return av > bv;` (correct for both positive and negative cases).

### Bug 2: Out-of-bounds GPU write in CDF lookup (`tlookup.cu`)
`lookuprange_tensor_prep_kernel` used `vals[tid].val[0]` as a lookup table index without bounds-checking. With wrong argmax (Bug 1), some diffs were negative field elements (val[2..7]≠0), giving huge `val[0]` values (≈2^32). With correct argmax, valid positive diffs can still be in the millions (logit range × 65536), far beyond the 4096-entry CDF table. This caused `tlookup_kernel`'s `atomicAdd(&counts[huge_index], 1)` to write to arbitrary GPU memory, corrupting the logit tensors for all subsequent positions.
**Fix**: Added `table_bound` parameter to kernel; indices are clamped to `[0, table_bound-1]`. *(See Bug 4 for a correction to the negative-element handling.)*

### Bug 3: CDF table too small (`zkllm_entropy.cu`)
Default `cdf_precision=12` gives a 4096-entry table covering diffs `[0, 4095]` = `[0, 0.78×sigma_eff]`. With sigma_eff=5223, the table must cover at least 6×sigma=31338 to get `Phi≈1.0` at the boundary (ensuring `win_prob≈0` for large-diff tokens). Increased default to `cdf_precision=15` (32768 entries, covers 6.28×sigma).

**Bugs 1–3 fixed → job 2102 passed entropy proof (34.29 bits, 1024 positions, n_negative_diffs=0 for all).** But then crashed at `Rescaling::prove` → see Bug 4.

### Bug 4: `lookuprange_tensor_prep_kernel` incorrectly maps valid negative remainders to index 0 (`tlookup.cu`)
The Bug 2 fix added `is_neg` check that maps all negative field elements (val[2..7]≠0) to index 0. This is **correct for CDF lookups** (after the argmax fix, all diffs are non-negative). But `Rescaling::prove` also calls `tl_rem.prep(rem)` via the same kernel, and remainder values in `[-32768, -1]` are valid negative integers stored as p−k. Mapping these to index 0 corrupts the multiplicity vector `m`, breaking the LogUp constraint in `tLookup_phase1` (`claim != p(0) + p(1)`).

Key insight: `Scalar_sub(p−k, p−(-32768)) = 32768 − k` (standard field arithmetic), which is the correct table index. The upper-bound clamp `(raw < table_bound) ? raw : (table_bound−1)` handles out-of-range large values without an `is_neg` check.

**Fix**: Removed the `is_neg` branch; kept only the upper-bound clamp.

**Job 2142 SUCCESS: full end-to-end PASSED.**
- Entropy bound: 34.2947 bits total, 0.0335 bits/token over 1024 tokens
- 6144 polynomials written to entropy-proof.bin
- All 1024 positions verified: 1024 OK, 0 FAIL
- `VERIFICATION PASSED`

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
