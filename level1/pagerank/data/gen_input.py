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

Usage: gen_input.py <output_dir>
"""
import random, sys, os

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out_dir, exist_ok=True)
rng = random.Random(20260831)

n = 16384
avg_out = 16
# out_edges[j] = set of targets of node j
out_edges = []
for j in range(n):
    deg = rng.randint(max(1, avg_out // 2), avg_out * 2 - avg_out // 2)
    targets = set()
    while len(targets) < deg:
        targets.add(rng.randrange(n))
    out_edges.append(sorted(targets))

# incoming CSR: row i lists edges j->i with value 1/outdeg(j)
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

with open(os.path.join(out_dir, "16384.data"), "w") as f:
    f.write(f"{len(cols)} {n}\n")
    f.write(" ".join(map(str, row_offsets)) + "\n")
    f.write(" ".join(map(str, cols)) + "\n")
    f.write(" ".join(f"{v:.9g}" for v in vals) + "\n")
print("pagerank input generated in", out_dir)
