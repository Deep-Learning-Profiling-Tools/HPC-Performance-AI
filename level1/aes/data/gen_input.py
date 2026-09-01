#!/usr/bin/env python3
"""Deterministic input generator for the AES benchmark.

The upstream Hetero-Mark data server is offline, so inputs are generated
locally. Content does not affect correctness semantics: the GPU ciphertext is
verified against the upstream CPU AES-256 reference implementation.

Usage: gen_input.py <output_dir>
Writes: input_1MB.hex (1 MiB plaintext as hex), key.hex (32-byte key as hex)
"""
import random, sys, os

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out_dir, exist_ok=True)
rng = random.Random(20260831)  # fixed seed -> reproducible input

n_bytes = 1 << 20  # 1 MiB plaintext (within the upstream 1KB..32MB sweep)
with open(os.path.join(out_dir, "input_1MB.hex"), "w") as f:
    f.write("".join(f"{rng.randrange(256):02x}" for _ in range(n_bytes)))
with open(os.path.join(out_dir, "key.hex"), "w") as f:
    f.write("".join(f"{rng.randrange(256):02x}" for _ in range(32)))
print("aes input generated in", out_dir)
