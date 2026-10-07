#!/usr/bin/env python3
"""Deterministic input generator for the PageRank benchmark.

The upstream Hetero-Mark data server is offline, so the CSR link matrix is
generated locally: a seeded random directed graph with 16384 nodes and average
out-degree 16 (matching the upstream N.data naming/scale, largest sweep size
16384). M[i][j] = 1/outdegree(j) for each edge j->i, stored in the CSR text
format read by PrBenchmark::LoadInputFile:
  num_connections num_nodes
  row_offsets (num_nodes+1 ints)
  column_numbers (num_connections ints)
  values (num_connections floats)

Correctness semantics are unchanged: the GPU PageRank result is verified
against the upstream CPU reference on the same matrix.

Usage: gen_input.py <output_dir>   (writes 16384.data, 4096.data, 1024.data)
"""
import random, sys, os

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out_dir, exist_ok=True)
def generate(n, seed, name):
    """Seeded random directed graph with n nodes, average out-degree 16, written as CSR."""
    rng = random.Random(seed)
    avg_out = 16
    out_edges = []
    for j in range(n):
        deg = rng.randint(max(1, avg_out // 2), avg_out * 2 - avg_out // 2)
        targets = set()
        while len(targets) < deg:
            targets.add(rng.randrange(n))
        out_edges.append(sorted(targets))
    incoming = [[] for _ in range(n)]
    for j, ts in enumerate(out_edges):
        v = 1.0 / len(ts)
        for t in ts:
            incoming[t].append((j, v))
    row_offsets = [0]
    cols, vals = [], []
    for i in range(n):
        for j, v in sorted(incoming[i]):
            cols.append(j)
            vals.append(v)
        row_offsets.append(len(cols))
    with open(os.path.join(out_dir, name), "w") as f:
        f.write(f"{len(cols)} {n}\n")
        f.write(" ".join(map(str, row_offsets)) + "\n")
        f.write(" ".join(map(str, cols)) + "\n")
        f.write(" ".join(f"{v:.9g}" for v in vals) + "\n")

generate(16384, 20260831, "16384.data")       # the ctest default (unchanged: same seed, same algorithm)
# Registered-input sizes (inputs.yaml): smaller points of the upstream N.data sweep, own seeds
generate(4096, 20260831 + 4096, "4096.data")
generate(1024, 20260831 + 1024, "1024.data")
print("pagerank input generated in", out_dir)
