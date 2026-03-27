#!/usr/bin/env python3
"""
verify_entropy.py — verify proof files produced by ./zkllm_entropy.

Checks all arithmetic claims in the proof without needing a GPU:
  1. win_prob == (1 - Phi(diff_actual / sigma_eff)) * cdf_scale
  2. q_fr    == clamp(floor(win_prob * 2^log_precision / total_win), 1, 2^log_precision)
  3. surprise == round((log_precision - log2(q_fr)) * log_scale)
  4. entropy_sum == claimed_entropy (over all positions)

Usage:
    python verify_entropy.py <proof_file>
        [--cdf-precision 12] [--cdf-scale 65536] [--log-precision 15]
        [--verbose]
"""

import struct, math, sys, argparse

MAGIC     = 0x5A4B454E54524F50

# Field element size depends on the field.  Set USE_GOLDILOCKS=True when
# verifying proofs produced by the Goldilocks build.
import os
USE_GOLDILOCKS = os.environ.get("USE_GOLDILOCKS", "0") not in ("0", "")
FR_T_SIZE = 8 if USE_GOLDILOCKS else 32  # bytes


def phi(x):
    """Standard normal CDF using erfc."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def cdf_table_value(d, sigma_eff, cdf_scale):
    """Expected CDF table entry: round(Phi(d / sigma_eff) * cdf_scale)."""
    if sigma_eff <= 0:
        return cdf_scale if d >= 0 else 0
    return round(phi(d / sigma_eff) * cdf_scale)


def log_table_value(q_fr, log_precision, log_scale):
    """Expected log table entry: round((log_precision - log2(q_fr)) * log_scale)."""
    if q_fr <= 0:
        return round(log_precision * log_scale)  # clamp
    return round((log_precision - math.log2(q_fr)) * log_scale)


def read_fr(f):
    """Read one field element (Goldilocks: 8 bytes / uint64; BLS12-381: 32 bytes / 8 x uint32)."""
    data = f.read(FR_T_SIZE)
    if len(data) < FR_T_SIZE:
        raise EOFError("Unexpected end of proof file reading Fr_t")
    if USE_GOLDILOCKS:
        return struct.unpack('<Q', data)   # 1-tuple of uint64
    return struct.unpack('<8I', data)


def fr_to_ull(words):
    """Extract uint64 value from an Fr_t word tuple."""
    if USE_GOLDILOCKS:
        return words[0]
    return words[0] | (words[1] << 32)


def fr_nonzero_high(words):
    """True if the field element represents a 'negative' (large) value."""
    if USE_GOLDILOCKS:
        GOLDILOCKS_P = (1 << 64) - (1 << 32) + 1
        return words[0] > (GOLDILOCKS_P >> 1)
    return any(words[2:])


def main():
    parser = argparse.ArgumentParser(description='Verify zkllm_entropy proof file')
    parser.add_argument('proof_file')
    parser.add_argument('--cdf-precision', type=int, default=12,
                        help='CDF table bits (default 12, same as prover default)')
    parser.add_argument('--cdf-scale', type=int, default=65536,
                        help='CDF output scale (default 65536)')
    parser.add_argument('--log-precision', type=int, default=15,
                        help='Log table bits (default 15, same as prover default)')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    with open(args.proof_file, 'rb') as f:

        # ── Header ────────────────────────────────────────────────────────────
        magic,       = struct.unpack('<Q', f.read(8))
        entropy_val, = struct.unpack('<Q', f.read(8))
        T,           = struct.unpack('<I', f.read(4))
        vocab_size,  = struct.unpack('<I', f.read(4))
        sigma_eff,   = struct.unpack('<d', f.read(8))
        log_scale,   = struct.unpack('<I', f.read(4))
        n_polys,     = struct.unpack('<I', f.read(4))

        if magic != MAGIC:
            print(f"ERROR: bad magic {magic:#x}, expected {MAGIC:#x}")
            sys.exit(1)

        print(f"Proof: T={T}, vocab_size={vocab_size}, sigma_eff={sigma_eff:.4f}")
        print(f"       log_scale={log_scale}, n_polys={n_polys}")
        print(f"Params: cdf_precision={args.cdf_precision}, cdf_scale={args.cdf_scale}, "
              f"log_precision={args.log_precision}")
        print()

        cdf_len = 1 << args.cdf_precision
        log_len = 1 << args.log_precision

        # ── Read all polynomial coefficients ──────────────────────────────────
        # Each polynomial: [n_coeffs: uint32] [n_coeffs * Fr_T_SIZE bytes]
        poly_values = []
        for _ in range(n_polys):
            n_coeffs, = struct.unpack('<I', f.read(4))
            coeffs = [read_fr(f) for _ in range(n_coeffs)]
            # Constant polynomial (degree 0): one coefficient = the value.
            # The prover evaluates at x=0..n_coeffs-1; for a constant poly value==coeffs[0].
            poly_values.append(coeffs[0] if coeffs else (0,)*8)

        # ── Expected polynomials per position: 6 constants ────────────────────
        # 0: logit_act   — MLE of logits at one-hot(actual_token)
        # 1: diff_actual — v_star - logit_act
        # 2: win_prob    — (1 - Phi(diff/sigma)) * cdf_scale
        # 3: total_win   — sum of win_probs over all vocab
        # 4: q_fr        — quantized probability index
        # 5: surprise    — -log2(q_fr/2^log_precision)*log_scale

        POLYS_PER_POS = 6
        if n_polys != T * POLYS_PER_POS:
            print(f"WARNING: expected {T * POLYS_PER_POS} polynomials for {T} positions "
                  f"(6 each), got {n_polys}")

        # ── Verify each position ───────────────────────────────────────────────
        entropy_sum = 0
        n_ok = 0
        n_fail = 0
        log_precision = args.log_precision

        for pos in range(T):
            base = pos * POLYS_PER_POS
            if base + 5 >= len(poly_values):
                print(f"  pos {pos}: not enough polynomials in proof")
                n_fail += 1
                continue

            logit_act_w  = poly_values[base + 0]
            diff_act_w   = poly_values[base + 1]
            win_prob_w   = poly_values[base + 2]
            total_win_w  = poly_values[base + 3]
            q_fr_w       = poly_values[base + 4]
            surprise_w   = poly_values[base + 5]

            # Extract scalar values (lower 64 bits, all should be non-negative small integers)
            diff_actual = fr_to_ull(diff_act_w)
            win_prob    = fr_to_ull(win_prob_w)
            total_win   = fr_to_ull(total_win_w)
            q_fr        = fr_to_ull(q_fr_w)
            surprise    = fr_to_ull(surprise_w)

            pos_ok = True
            errors = []

            # Check 1: win_prob == cdf_scale - cdf_table[diff_actual]
            d_clamped = min(diff_actual, cdf_len - 1)
            cdf_val = cdf_table_value(d_clamped, sigma_eff, args.cdf_scale)
            expected_win = args.cdf_scale - cdf_val
            if expected_win < 0:
                expected_win = 0
            if win_prob != expected_win:
                errors.append(f"win_prob mismatch: got {win_prob}, expected {expected_win} "
                               f"(diff={diff_actual}, cdf_val={cdf_val})")
                pos_ok = False

            # Check 2: q_fr == clamp(floor(win_prob * 2^log_precision / total_win), 1, 2^log_precision)
            if total_win == 0:
                expected_q = 1
            else:
                expected_q = (win_prob * log_len) // total_win
                expected_q = max(1, min(log_len, expected_q))
            if q_fr != expected_q:
                errors.append(f"q_fr mismatch: got {q_fr}, expected {expected_q} "
                               f"(win={win_prob}, total={total_win})")
                pos_ok = False

            # Check 3: surprise == log_table[q_fr]
            if q_fr < 1 or q_fr > log_len:
                errors.append(f"q_fr={q_fr} out of range [1, {log_len}]")
                pos_ok = False
            else:
                expected_surprise = log_table_value(q_fr, log_precision, log_scale)
                if surprise != expected_surprise:
                    errors.append(f"surprise mismatch: got {surprise}, expected {expected_surprise} "
                                   f"(q_fr={q_fr})")
                    pos_ok = False

            # Check 4: consistency — win_prob <= total_win
            if win_prob > total_win:
                errors.append(f"win_prob ({win_prob}) > total_win ({total_win}): inconsistent")
                pos_ok = False

            entropy_sum += surprise

            if pos_ok:
                n_ok += 1
                if args.verbose:
                    bits = surprise / log_scale
                    print(f"  pos {pos:4d}: OK  surprise={bits:.3f} bits  "
                          f"diff={diff_actual}  q={q_fr}/{log_len}")
            else:
                n_fail += 1
                print(f"  pos {pos:4d}: FAIL")
                for e in errors:
                    print(f"    {e}")

        # ── Check entropy sum ─────────────────────────────────────────────────
        print()
        claimed = entropy_val
        if entropy_sum != claimed:
            print(f"FAIL: entropy_sum mismatch: computed {entropy_sum}, claimed {claimed}")
            n_fail += 1
        else:
            print(f"OK: entropy_sum matches claimed value ({claimed})")

        # ── Summary ───────────────────────────────────────────────────────────
        print()
        total_entropy_bits = entropy_sum / log_scale
        print(f"Conditional entropy bound : {total_entropy_bits:.4f} bits total")
        print(f"Average per token         : {total_entropy_bits / T:.4f} bits/token")
        print()
        print(f"Positions checked: {T}  OK: {n_ok}  FAIL: {n_fail}")
        if n_fail == 0:
            print("VERIFICATION PASSED")
            sys.exit(0)
        else:
            print("VERIFICATION FAILED")
            sys.exit(1)


if __name__ == '__main__':
    main()
