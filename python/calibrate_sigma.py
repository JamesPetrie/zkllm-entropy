#!/usr/bin/env python3
"""
calibrate_sigma.py — calibrate sigma_eff for the zkEntropy noise model.

Runs Llama-2 greedy inference twice on the same prompt and measures how
often the output tokens agree between runs.  Any disagreement is due to
GPU floating-point non-determinism — the same noise source that
zkConditionalEntropy models as Gaussian on logit differences.

Given the empirical match rate r and the per-position logit gap
  gap[t] = logits[t_star] - logits[t_second]
the noise model predicts:
  P_match(t) = Phi(gap[t] / (sigma * sqrt(2)))
where sigma is the per-logit noise std dev.  We binary-search for the
sigma that satisfies mean(P_match) == r.

The output is sigma_eff = sigma * logit_scale (integer units), which is
passed directly to ./zkllm_entropy.

Usage:
    # GPU required (noise is GPU hardware non-determinism)
    python calibrate_sigma.py [--model-size 7] [--seq-len 256]
        [--prompt "Once upon a time"]
        [--logit-scale 65536] [--n-runs 2] [--verbose]
"""

import argparse, math, sys
import torch
import numpy as np

def phi(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))

def expected_match_rate(sigma: float, gaps: np.ndarray) -> float:
    return float(np.mean([phi(g / (sigma * math.sqrt(2.0))) for g in gaps]))

def find_sigma(target_rate: float, gaps: np.ndarray,
               lo: float = 1e-6, hi: float = 1e6) -> float:
    """Binary search: match rate is decreasing in sigma."""
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if expected_match_rate(mid, gaps) > target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def run_greedy(model, input_ids: torch.Tensor, max_new_tokens: int,
               device) -> tuple[list[int], list[np.ndarray]]:
    """
    Run greedy decoding, returning:
      tokens    : list of generated token ids
      logit_rows: list of np.ndarray (vocab_size,) raw logits per step
    """
    tokens = []
    logit_rows = []
    past = None
    cur = input_ids.to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(cur, past_key_values=past, use_cache=True)
            logits = out.logits[:, -1, :]   # (1, vocab_size)
            past   = out.past_key_values
            tok    = int(logits.argmax(dim=-1).item())
            tokens.append(tok)
            logit_rows.append(logits[0].cpu().float().numpy())
            cur = torch.tensor([[tok]], device=device)

    return tokens, logit_rows


def main():
    parser = argparse.ArgumentParser(
        description='Calibrate sigma_eff by measuring GPU non-determinism')
    parser.add_argument('--model-size', type=int, default=7, choices=[7, 13])
    parser.add_argument('--seq-len', type=int, default=256,
                        help='Number of tokens to generate per run (default 256)')
    parser.add_argument('--prompt', type=str,
                        default='The quick brown fox jumps over the lazy dog. '
                                'In a world where science and magic coexist,',
                        help='Prompt text for calibration')
    parser.add_argument('--logit-scale', type=int, default=65536,
                        help='zkLLM fixed-point scale (default 65536 = 1<<16)')
    parser.add_argument('--n-runs', type=int, default=2,
                        help='Number of independent inference runs (default 2)')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print('WARNING: running on CPU — noise may not reflect GPU non-determinism',
              file=sys.stderr)

    MODEL_CARD = f'meta-llama/Llama-2-{args.model_size}b-hf'
    print(f'Loading {MODEL_CARD} on {device}...', flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CARD, local_files_only=True, cache_dir='./model-storage')
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CARD, local_files_only=True, cache_dir='./model-storage',
        torch_dtype=torch.float32).to(device)
    model.eval()

    input_ids = tokenizer(args.prompt, return_tensors='pt').input_ids

    # ── Run inference n_runs times ────────────────────────────────────────────
    all_tokens = []
    all_logits = []   # logits from run 0 only (used for gap computation)
    print(f'Running {args.n_runs} greedy passes of {args.seq_len} tokens...', flush=True)

    for run in range(args.n_runs):
        print(f'  Run {run + 1}/{args.n_runs}...', flush=True)
        toks, logit_rows = run_greedy(model, input_ids, args.seq_len, device)
        all_tokens.append(toks)
        if run == 0:
            all_logits = logit_rows

    # ── Compute pairwise match rates ──────────────────────────────────────────
    T = args.seq_len
    match_counts = []
    for i in range(args.n_runs):
        for j in range(i + 1, args.n_runs):
            n_match = sum(a == b for a, b in zip(all_tokens[i], all_tokens[j]))
            rate = n_match / T
            match_counts.append(rate)
            if args.verbose or args.n_runs == 2:
                print(f'  Run {i+1} vs Run {j+1}: {n_match}/{T} tokens match  '
                      f'(rate={rate:.4f})')

    empirical_rate = float(np.mean(match_counts))
    print(f'Mean empirical match rate: {empirical_rate:.4f}')

    if empirical_rate >= 1.0:
        print('All runs produced identical output — '
              'GPU appears fully deterministic for this prompt.')
        print('Try a longer sequence or different prompt, or use --target-match-rate.')
        # Fall back to a plausible default
        empirical_rate = 0.999

    # ── Compute logit gaps from run 0 ─────────────────────────────────────────
    gaps = []
    for row in all_logits:
        top2 = np.partition(row, -2)[-2:]
        gap  = float(top2[1] - top2[0])  # max - second_max (in raw fp units)
        gaps.append(max(gap, 1e-9))

    gaps_arr = np.array(gaps)
    print(f'\nLogit gap stats (raw fp32 logit units):')
    print(f'  mean={gaps_arr.mean():.4f}  std={gaps_arr.std():.4f}  '
          f'min={gaps_arr.min():.4f}  max={gaps_arr.max():.4f}')

    if args.verbose:
        print('\nGreedy tokens (run 0):',
              tokenizer.decode(all_tokens[0], skip_special_tokens=True)[:200])

    # ── Binary-search for sigma in raw fp units ───────────────────────────────
    sigma_fp = find_sigma(empirical_rate, gaps_arr)
    achieved = expected_match_rate(sigma_fp, gaps_arr)

    # Convert to zkLLM integer units: sigma_eff = sigma_fp * logit_scale
    sigma_eff = sigma_fp * args.logit_scale

    print(f'\nCalibrated sigma (fp32 units)  : {sigma_fp:.6f}')
    print(f'Calibrated sigma_eff (integer) : {sigma_eff:.1f}  '
          f'(= sigma_fp × {args.logit_scale})')
    print(f'Achieved match rate            : {achieved:.4f}  '
          f'(target {empirical_rate:.4f})')

    print(f'\nUse this sigma_eff in zkllm_entropy:')
    print(f'  ./zkllm_entropy <workdir> tokens.txt proof.bin {sigma_eff:.0f}')

    # ── Sensitivity table ─────────────────────────────────────────────────────
    print(f'\nSensitivity (sigma_eff → expected match rate):')
    for s in [sigma_eff * 0.5, sigma_eff * 0.75, sigma_eff,
              sigma_eff * 1.25, sigma_eff * 2.0]:
        r = expected_match_rate(s / args.logit_scale, gaps_arr)
        marker = ' <--' if abs(s - sigma_eff) < 1 else ''
        print(f'  sigma_eff={s:10.1f}  match_rate={r:.4f}{marker}')


if __name__ == '__main__':
    main()
