# SpGEMM KKMEM (Kokkos Kernels port)

Sparse matrix-matrix multiply C = A x A; standalone CUDA/HIP port of the Kokkos Kernels KKMEM hash-based numeric phase (upstream --algorithm KKMEM).

## Source

Source suite: Kokkos Kernels
Upstream repository: https://github.com/kokkos/kokkos-kernels
Upstream commit: 23a699f3c5bd662bf2ed52116e56533ecc3ddae0

## Port provenance (Option 2: manual faithful port, scoped)

Ported verbatim from upstream:

- Numeric phase: `PortableNumericCHASH::operator()(GPUTag)` from
  `sparse/impl/KokkosSparse_spgemm_impl_kkmem.hpp` -- one row per warp,
  two-level hash accumulator: per-thread shared-memory hashmap (linked-list
  chaining, bitwiseAnd hash) overflowing into a global-memory hashmap whose
  key/value storage is the C row itself.
- `HashmapAccumulator::vector_atomic_insert_into_hash_mergeAdd` (+
  `_TrackHashes`) from `common/src/KokkosKernels_HashmapAccumulator.hpp`.
- `UniformMemoryPool` (ManyThread2OneChunk: CAS chunk locks, linear probing)
  from `common/src/KokkosKernels_Uniform_Initialized_MemoryPool.hpp`.
- All launch/shmem heuristics with upstream defaults (shmem budget 16384 B,
  unit_memory 20 B, shared hash/key sizing formulas, vector size from avg
  B-row nnz, team 256/vector, min_hash_size = pow4 >= max C row nnz, pool
  chunk and count formulas). Resulting config for the default workload:
  vector_size=32, team_size=8, shmem hash 64 / keys 110, min_hash_size=1024.

Documented deviations:
- Upstream default `--algorithm KK` may select other variants (dense
  accumulators, cuckoo hashes) at runtime; this port pins KKMEM, an explicit
  upstream option.
- The symbolic phase (exact C row sizes) is computed on the host instead of
  porting the ~3.4k-line compression-based GPU symbolic; only the numeric
  phase is timed.
- Input: generated with upstream `kk_sparseMatrix_generate` (srand 13721),
  10000x10000, 30 nnz/row; values from fixed-seed xorshift64* uniform(-50,50).

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
cmake -S level1/spgemm -B build/spgemm/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/spgemm/cuda
```

## Run

```bash
./build/spgemm/cuda/spgemm_cuda   # optional: --m --nnz --repeat
```

## Validation

Added check: C = A*A recomputed on the CPU with a dense accumulator; per-row sorted entries compared (columns exact, values rel tol 1e-10). Prints PASS/FAIL.

```bash
ctest --test-dir build/spgemm/cuda --output-on-failure
```

## LOC

CUDA: 393 (cloc, cuda/); HIP: 393 (identical mirrored source)
