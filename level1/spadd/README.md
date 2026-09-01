# SpAdd (Kokkos Kernels port)

Sparse matrix addition C = A + B (sorted inputs); faithful standalone CUDA/HIP port of the Kokkos Kernels sorted SpAdd used by perf_test.

## Source

Source suite: Kokkos Kernels
Upstream repository: https://github.com/kokkos/kokkos-kernels
Upstream commit: 23a699f3c5bd662bf2ed52116e56533ecc3ddae0

## Port provenance (Option 2: manual faithful port)

Ported verbatim from upstream:

- Symbolic count kernel: `SortedCountEntriesTeam` (+ `SortedCountEntriesRange`
  long-row fallback) from `sparse/impl/KokkosSparse_spadd_symbolic_impl.hpp`,
  including the `runSortedCountEntries` launch heuristics (c_est_nnz,
  pot_est_nnz, vector-length selection; one row per warp, bitonic merge of
  the two sorted rows in shared memory).
- Numeric kernel: `SortedNumericSumFunctor` (one thread per row, two-way
  sorted merge, alpha=beta=1) from
  `sparse/impl/KokkosSparse_spadd_numeric_impl.hpp`.
- Driver semantics (`perf_test/sparse/KokkosSparse_spadd.cpp`): m=n=10000,
  nnzPerRow=30, both inputs from `kk_sparseMatrix_generate` (srand(13721) is
  reset per call, so A and B share the same sparsity structure -- upstream
  behavior), sorted algorithm, repeat=1.

Documented deviations (packaging only): matrix values use a fixed-seed
xorshift64* uniform(-50,50) instead of the Kokkos XorShift64 pool; the rowmap
prefix sum replaces Kokkos parallel_scan with an equivalent exclusive scan;
Kokkos' occupancy-recommended team size is fixed at 8 warps; input rows are
sorted on the host during untimed setup (upstream sorts on device, untimed).

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
cmake -S level1/spadd -B build/spadd/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/spadd/cuda
```

## Run

```bash
./build/spadd/cuda/spadd_cuda   # optional: --m --n --nnz --repeat
```

## Validation

Added check (upstream perf_test has none): C recomputed on the CPU with the identical merge order; structure compared exactly and values at rel tol 1e-13. Prints PASS/FAIL.

```bash
ctest --test-dir build/spadd/cuda --output-on-failure
```

## LOC

CUDA: 312 (cloc, cuda/); HIP: 312 (identical mirrored source)
