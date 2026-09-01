#!/usr/bin/env python3
"""Standalone correctness check for Rodinia bfs (upstream provides none).

Runs the benchmark, then recomputes BFS costs (hop counts from the source
node) on the CPU from the input graph and compares against result.txt.

Usage: verify.py <bfs_executable> <graph_file>
"""
import subprocess, sys
from collections import deque

exe, graph = sys.argv[1], sys.argv[2]
r = subprocess.run([exe, graph], capture_output=True, text=True)
if r.returncode != 0:
    print(f"FAIL: benchmark exited with {r.returncode}\n{r.stdout}{r.stderr}")
    sys.exit(1)

# parse graph (Rodinia format)
tok = open(graph).read().split()
p = 0
n = int(tok[p]); p += 1
starts, counts = [0]*n, [0]*n
for i in range(n):
    starts[i] = int(tok[p]); counts[i] = int(tok[p+1]); p += 2
source = int(tok[p]); p += 1
ne = int(tok[p]); p += 1
edges = [0]*ne
for e in range(ne):
    edges[e] = int(tok[p]); p += 2  # skip edge cost (unused by bfs.cu)

# CPU BFS
cost = [-1]*n
cost[source] = 0
q = deque([source])
while q:
    u = q.popleft()
    for e in range(starts[u], starts[u] + counts[u]):
        v = edges[e]
        if cost[v] == -1:
            cost[v] = cost[u] + 1
            q.append(v)

# compare with result.txt written by the benchmark (in cwd)
bad = 0
with open("result.txt") as f:
    for line in f:
        node, c = line.strip().rstrip(")").split(")")[0], line.split("cost:")[1]
        i, c = int(node), int(c)
        if c != cost[i]:
            bad += 1
            if bad <= 5:
                print(f"  mismatch node {i}: gpu {c} expected {cost[i]}")
if bad:
    print(f"FAIL: {bad} mismatching nodes out of {n}")
    sys.exit(1)
print(f"PASS: all {n} BFS costs match the CPU reference")
