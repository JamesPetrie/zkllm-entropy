#!/usr/bin/env python3
"""Generate swiglu-table.bin without torch. Pure Python/math implementation.
SwiGLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
"""
import struct, math

def generate(out_file='swiglu-table.bin', in_range_bits=10, in_prec_bits=12, out_prec_bits=16):
    half_range = 1 << (in_range_bits - 1)  # 512
    step_inv = 1 << in_prec_bits            # 4096
    scale = 1 << out_prec_bits              # 65536
    total = (1 << (in_range_bits + in_prec_bits))  # 4194304

    data = bytearray(total * 4)
    for i in range(total):
        x = (i - total // 2) / step_inv
        if x > 20:
            val = x
        elif x < -20:
            val = 0.0
        else:
            val = x / (1.0 + math.exp(-x))
        v = int(round(val * scale))
        struct.pack_into('<i', data, i * 4, v)

    with open(out_file, 'wb') as f:
        f.write(data)
    print(f'Generated {out_file}: {total} entries ({len(data)} bytes)')

if __name__ == '__main__':
    generate()
