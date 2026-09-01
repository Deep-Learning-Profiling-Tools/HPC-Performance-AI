# Graph Coloring VBBIT (Kokkos Kernels port)

Distance-1 greedy graph coloring; faithful standalone CUDA/HIP port of the Kokkos Kernels COLORING_VBBIT algorithm (the COLORING_DEFAULT choice on GPUs).

## Source

Source suite: Kokkos Kernels
Upstream repository: https://github.com/kokkos/kokkos-kernels
Upstream commit: 23a699f3c5bd662bf2ed52116e56533ecc3ddae0

## Port provenance (Option 2: manual faithful port)

Ported verbatim from upstream
(`graph/impl/KokkosGraph_Distance1Color_impl.hpp`,
`graph/src/KokkosGraph_Distance1ColorHandle.hpp`):

- `functorGreedyColor_IMPLOG`: speculative greedy coloring with a 64-bit
  forbidden-color window (VBBIT_COLORING_FORBIDDEN_SIZE=64), chunked work
  items (vb_chunk_size=8, dropping to 1 for short worklists).
- `functorFindConflicts_Atomic`: conflict detection (i < neighbor rule),
  uncoloring + atomic append to the next worklist (COLORING_ATOMIC scheme,
  the VBBIT default).
- `GraphColor_VB::color_graph` main loop: worklist swap per iteration,
  max_number_of_iterations=200, then the serial host `resolveConflicts`
  fallback for any remaining vertices.

Input (upstream perf_test requires an .mtx file; none ships with the repo):
deterministic random graph from `kk_sparseMatrix_generate` (srand 13721),
200000 vertices, ~30 entries/row, symmetrized (G = A union A^T) and sorted.

## Verified Environment

GCC/G++ 13.3.0 (conda, pinned) | C++20 | CMake 3.28.4 | Ninja 1.13.2 | Python 3.12.3
CUDA Toolkit 13.2 (nvcc 13.2.78, /usr/local/cuda) | NVIDIA B200 (sm_100), driver 595.58.03
HIP/ROCm: source + build config present where noted, unverified (no AMD GPU available)

Reproduce the toolchain from the repository root: `./setup_env.sh` then
`source hpcperf_env.sh` (all user-space versions are pinned in environment.yml).

## Backends

CUDA: working (configure + build + run + validation verified on B200)
HIP: build configuration present (same source via a thin compatibility layer), untested (no AMD GPU / ROCm on the development machine)

## Build

```bash
source hpcperf_env.sh
cmake -S level1/graph_coloring -B build/graph_coloring/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/graph_coloring/cuda
```

## Run

```bash
./build/graph_coloring/cuda/graph_coloring_cuda   # optional: --nv --nnz --repeat
```

## Validation

Added check (coloring has a complete correctness spec): host verifies every vertex is colored and no edge joins two vertices of the same color; reports color count and phases. Prints PASS/FAIL.

```bash
ctest --test-dir build/graph_coloring/cuda --output-on-failure
```

## LOC

CUDA: 244 (cloc, cuda/); HIP: 244 (identical mirrored source)
