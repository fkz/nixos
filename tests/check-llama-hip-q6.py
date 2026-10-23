#!/usr/bin/env python3
"""Compare original/optimized HIP outputs from llama-hip-q6.cpp."""
import array
import math
import sys


def read(path):
    values = array.array('f')
    with open(path, 'rb') as source:
        values.frombytes(source.read())
    return values


baseline, optimized = map(read, sys.argv[1:])
assert len(baseline) == len(optimized) == 205068, 'wrong output size'
shapes = [(6144, 2560), (2560, 10240), (2560, 512), (2560, 640),
          (2560, 6144), (2560, 12288), (2560, 1), (2560, 639),
          (2560, 641), (1280, 513)]
offset = 0
max_abs = max_scaled = 0.0
for weight_type in ('q6_K', 'q5_K'):
    for tokens in (1, 2):
        for cols, rows in shapes:
            count = rows * tokens
            a = baseline[offset:offset + count]
            b = optimized[offset:offset + count]
            assert all(math.isfinite(x) for x in a + b), 'non-finite output'
            error = max(abs(x - y) for x, y in zip(a, b))
            scaled = max(abs(x - y) / (1 + abs(x)) for x, y in zip(a, b))
            assert scaled < 1e-4, (weight_type, cols, rows, tokens, scaled)
            if weight_type != 'q6_K' or tokens != 1 or cols == 1280:
                assert a == b, ('fallback changed', weight_type, cols, rows, tokens)
            max_abs = max(max_abs, error)
            max_scaled = max(max_scaled, scaled)
            offset += count
assert offset == len(baseline)
print(f'PASS: 40 cases, {offset} outputs; max abs={max_abs:.8g}, scaled={max_scaled:.8g}')
