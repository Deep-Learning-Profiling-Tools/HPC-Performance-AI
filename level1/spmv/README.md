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

Correctness = y against the frozen B200 golden reference `reference/<input_id>.json` (sum (a - g)^2 / sum g^2 <= (sqrt 1e-5 - sqrt d0)^2; rule in `inputs.yaml` `golden:`). y is captured after the timed loop (the result of its last SpMV, copied back after the ROI). Between the single SpMV before the loop -- the one upstream's check compares -- and the loop, y is poisoned on the device and in the host copy (all-ones bytes, a NaN), outside the ROI and the timer, so a timed loop that does not compute y fails the check instead of passing on the earlier, identical result. The kernel writes every y[i] (beta = 0) and never reads y, so the poison cannot change a correct result: the post-loop y equals the frozen references, which were made from the pre-loop y, bit for bit on all three inputs (2026-10-27). The upstream check (sequential CPU gold-standard SpMV, relative squared error <= 1e-5, check_errors, prints PASS/FAIL) validated that reference once and runs only on request (below); by default the CPU gold y is not computed either (the matrix-value overwrite of generate_gold_standard still runs) and the numErrors column stays 0.

```bash
ctest --test-dir build/spmv/cuda --output-on-failure   # golden run of the registry's default input
```

**Correctness switch.** By default the host-side CPU check is skipped and the benchmark prints `SKIP_VERIFY` (tools/timing relies on it); a plain run, ctest, `tools/inputs` and `tools/timing` never run the CPU reference and nothing falls back to it. `HPCPERF_CPU_REFERENCE=1` runs the original CPU check (offline only: `tools/golden/hpcperf_golden.py make` uses it once to validate the golden reference); `HPCPERF_SKIP_VERIFY=1` still forces it off. `HPCPERF_GOLDEN_DUMP=<dir>` makes the run capture its outputs. See [tools/golden/README.md](../../tools/golden/README.md) and [tools/timing/README.md](../../tools/timing/README.md).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `cuda/main.cu` and the HIP port: the benchmark loop, not the warm-up call and check before it. Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 200 (cloc, cuda/); HIP: 200 (identical mirrored source)
