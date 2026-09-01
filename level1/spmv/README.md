# SpMV (Kokkos Kernels port)

CSR sparse matrix-vector multiply; faithful standalone CUDA/HIP port of the Kokkos Kernels perf_test SpMV (test=kk).

## Source

Source suite: Kokkos Kernels
Upstream repository: https://github.com/kokkos/kokkos-kernels
Upstream commit: 23a699f3c5bd662bf2ed52116e56533ecc3ddae0

## Port provenance (Option 2: manual faithful port)

Ported verbatim from upstream (semantics, launch heuristics, and input
generation preserved):

- GPU kernel + launch heuristics: `SPMV_Functor` / `launch_parameters` from
  `perf_test/sparse/spmv/Kokkos_SPMV.hpp` (team-based CSR SpMV; team ->
  thread block, TeamThreadRange -> threadIdx.y, ThreadVectorRange ->
  threadIdx.x lanes with a sub-warp shuffle reduction). Resulting config for
  the default workload: vector_length=2, team_size=128, rows_per_team=128.
- Matrix structure: `kk_sparseMatrix_generate` (glibc srand/rand, seed 13721)
  with the driver defaults: numRows=numCols=110503, nnz=10*numRows,
  bandwidth=(int)(0.01*numRows). x/y init continues the same rand() stream
  (`rand()%40-20`), byte-identical to upstream on glibc.
- Gold standard + relative-error check: `generate_gold_standard` /
  `check_errors` from `KokkosSparse_spmv_test.{hpp,cpp}`. Upstream overwrites
  the matrix values with `colidx + row` before running, so the XorShift64
  random value fill never reaches the benchmark and is omitted.

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
cmake -S level1/spmv -B build/spmv/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/spmv/cuda
```

## Run

```bash
./build/spmv/cuda/spmv_cuda   # optional: -s <numRows> -l <loop>
```

## Validation

Upstream built-in check: sequential CPU gold-standard SpMV; relative squared error of y must be <= 1e-5 (check_errors). Prints numErrors and PASS/FAIL.

```bash
ctest --test-dir build/spmv/cuda --output-on-failure
```

## LOC

CUDA: 200 (cloc, cuda/); HIP: 200 (identical mirrored source)
