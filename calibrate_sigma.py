#!/usr/bin/env python3
"""
calibrate_sigma.py — calibrate the Gaussian noise model sigma_eff for zkEntropy.

Method:
  1. Run gen_logits.py twice (or any two identical-input inference passes) to get
     two token sequences.  Compare them to measure the empirical token match rate.
  2. Given the logit tensors, compute the argmax gap (v_star - v_second) per position.
  3. Under the Gaussian noise model with per-logit std sigma, the probability that the
     greedy winner matches between two runs is:
         P_match(pos) = Phi(gap_pos / (sigma * sqrt(2)))
     (Two independent noise draws on each logit; difference of two is N(0, sigma*sqrt(2)).)
  4. Find the sigma_eff = sigma * logit_scale that minimises:
         (mean(P_match) - empirical_match_rate)^2

Usage:
    # Compare two token files and known logit dir:
    python calibrate_sigma.py \\
        --tokens1 ./zkllm-workdir/Llama-2-7b/logits/tokens.txt \\
        --tokens2 ./zkllm-workdir/Llama-2-7b/logits/tokens_run2.txt \\
        --logits-dir ./zkllm-workdir/Llama-2-7b/logits \\
        --logit-scale 65536

    # If you only have one run, estimate from logit gap distribution:
    python calibrate_sigma.py \\
        --logits-dir ./zkllm-workdir/Llama-2-7b/logits \\
        --target-match-rate 0.97 \\
        --logit-scale 65536
"""

import argparse, math, struct, os, sys
import numpy as np

FR_T_SIZE = 32  # bytes per Fr_t


def phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def load_tokens(path):
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]


def load_logit_row(path):
    """Load a logits_N.bin file as a numpy array of float64 (from Fr_t int32 representation)."""
    data = np.fromfile(path, dtype=np.uint32)
    n_fr = len(data) // 8
    data = data.reshape(n_fr, 8)
    # word0 holds the int32 value (bit-reinterpreted as uint32)
    word0 = data[:, 0].view(np.int32).astype(np.float64)
    return word0


def match_rate(tokens1, tokens2):
    assert len(tokens1) == len(tokens2), "Token sequences differ in length"
    matches = sum(a == b for a, b in zip(tokens1, tokens2))
    return matches / len(tokens1)


def argmax_gap(logit_row):
    """Return (v_star - v_second) for a single position's logit array."""
    sorted_vals = np.sort(logit_row)[::-1]
    return float(sorted_vals[0] - sorted_vals[1])


def expected_match_rate(sigma_eff, gaps):
    """E[Phi(gap / (sigma_eff * sqrt(2)))] over all positions."""
    return np.mean([phi(g / (sigma_eff * math.sqrt(2))) for g in gaps])


def find_sigma(target_rate, gaps, lo=1.0, hi=1e6, tol=1.0):
    """Binary-search for sigma_eff in integer units that achieves target_rate."""
    # At small sigma_eff, most tokens match (high probability). At large sigma, fewer match.
    # So match_rate is decreasing in sigma_eff.
    for _ in range(60):
        mid = (lo + hi) / 2
        r = expected_match_rate(mid, gaps)
        if r > target_rate:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def main():
    parser = argparse.ArgumentParser(description='Calibrate sigma_eff for zkEntropy')
    parser.add_argument('--tokens1', default='',
                        help='First token sequence file (tokens.txt)')
    parser.add_argument('--tokens2', default='',
                        help='Second token sequence file (from a second inference run)')
    parser.add_argument('--logits-dir', required=True,
                        help='Directory containing logits_N.bin files')
    parser.add_argument('--target-match-rate', type=float, default=0.97,
                        help='Target empirical match rate (default 0.97); used when '
                             '--tokens2 not given')
    parser.add_argument('--logit-scale', type=int, default=65536,
                        help='Fixed-point scale used when quantising logits (default 65536)')
    parser.add_argument('--max-pos', type=int, default=None,
                        help='Maximum number of positions to use (default: all)')
    args = parser.parse_args()

    logits_dir = args.logits_dir

    # ── Determine empirical match rate ────────────────────────────────────────
    if args.tokens1 and args.tokens2:
        tok1 = load_tokens(args.tokens1)
        tok2 = load_tokens(args.tokens2)
        rate = match_rate(tok1, tok2)
        T = min(len(tok1), len(tok2))
        print(f"Empirical match rate: {rate:.4f} ({int(rate*T)}/{T} positions match)")
    else:
        rate = args.target_match_rate
        print(f"No second token file given; targeting match rate = {rate:.4f}")

    # ── Load logit tensors and compute argmax gaps ────────────────────────────
    print(f"Loading logit tensors from {logits_dir} ...")
    gaps = []
    t = 0
    while True:
        path = os.path.join(logits_dir, f"logits_{t}.bin")
        if not os.path.isfile(path):
            break
        if args.max_pos is not None and t >= args.max_pos:
            break
        row = load_logit_row(path)
        gaps.append(argmax_gap(row))
        t += 1

    if not gaps:
        print(f"ERROR: no logits_N.bin found in {logits_dir}", file=sys.stderr)
        sys.exit(1)

    T_used = len(gaps)
    print(f"Loaded {T_used} positions.")
    gaps_arr = np.array(gaps)
    print(f"Argmax gap stats (in logit-scale integers):")
    print(f"  mean={gaps_arr.mean():.1f}  std={gaps_arr.std():.1f}  "
          f"min={gaps_arr.min():.0f}  max={gaps_arr.max():.0f}")

    # ── Binary-search for sigma_eff ───────────────────────────────────────────
    sigma_eff = find_sigma(rate, gaps_arr)
    sigma_real = sigma_eff / args.logit_scale
    achieved_rate = expected_match_rate(sigma_eff, gaps_arr)

    print()
    print(f"Calibrated sigma_eff : {sigma_eff:.1f}  (sigma_real = {sigma_real:.5f})")
    print(f"Achieved match rate  : {achieved_rate:.4f}  (target {rate:.4f})")
    print()
    print("Use this sigma_eff value in zkllm_entropy:")
    print(f"  ./zkllm_entropy <logits_dir> <tokens.txt> <proof.bin> {sigma_eff:.0f}")
    print()

    # ── Sensitivity table ─────────────────────────────────────────────────────
    print("Sensitivity (sigma_eff → expected match rate):")
    for s in [sigma_eff * 0.5, sigma_eff * 0.75, sigma_eff,
              sigma_eff * 1.25, sigma_eff * 1.5, sigma_eff * 2.0]:
        r = expected_match_rate(s, gaps_arr)
        marker = " <--" if abs(s - sigma_eff) < 1 else ""
        print(f"  sigma_eff={s:8.1f}  match_rate={r:.4f}{marker}")


if __name__ == '__main__':
    main()
