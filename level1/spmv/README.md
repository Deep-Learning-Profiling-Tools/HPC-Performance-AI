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

Correctness = y against the frozen B200 golden reference `reference/<input_id>.json` (sum (a - g)^2 / sum g^2 <= ((sqrt 1e-5 - sqrt d0) / (1 + sqrt d0))^2, the strict budget `tools/golden` applies; rule in `inputs.yaml` `golden:` -- whose frozen `basis` text still says `(sqrt 1e-5 - sqrt d0)^2`, a looser bound than the one applied, until the reference is re-made). y is captured after the timed loop (the result of its last SpMV, copied back after the ROI). Between the single SpMV before the loop -- the one upstream's check compares -- and the loop, y is poisoned on the device and in the host copy (all-ones bytes, a NaN), outside the ROI and the timer, so a timed loop that does not compute y fails the check instead of passing on the earlier, identical result. The kernel writes every y[i] (beta = 0) and never reads y, so the poison cannot change a correct result: the post-loop y equals the frozen references, which were made from the pre-loop y, bit for bit on all three inputs (2026-10-27). The upstream check (sequential CPU gold-standard SpMV, relative squared error <= 1e-5, check_errors, prints PASS/FAIL) validated that reference once and runs only on request (below); by default the CPU gold y is not computed either (the matrix-value overwrite of generate_gold_standard still runs) and the numErrors column stays 0.

```bash
ctest --test-dir build/spmv/cuda --output-on-failure   # golden run of the registry's default input
```

**Correctness probe (`probe-real`).** Upstream's data are integers (`values = colidx + row` from generate_gold_standard, `x = rand() % 40 - 20`): every product and partial sum of y is an integer far below 2^53, so every f64 evaluation order gives y bit for bit (18 legitimate variants measured 2026-10-28, cuSPARSE CSR ALG1/ALG2 included: all exact), values and x stay exact even when stored in f32 (x even in bf16), and f32 arithmetic changes y by at most 46 at |y| up to 5.3e8. The performance inputs' rule (upstream's l2_rel_sq 1e-5) then passes f32 arithmetic and accumulation (4.8e-18 / 1.3e-15 / 2.4e-15), TF32-rounded inputs (4.1e-8), a single zeroed row (<= 4.5e-6) and even 1.65% of the rows zeroed (rows [0.005 N, 0.0215 N), <= 9.7e-6). The registry's correctness-probe input `probe-real` (`spmv_cuda --probe`; `inputs.yaml` `purpose: correctness-probe`: never timed, never the default) runs the same kernel on the default matrix structure with real-valued set-up data selected in set-up only: values uniform in [-50, 50) (the range of upstream kk_sparseMatrix_generate's own random fill) and x uniform in [-20, 20), from fixed-seed splitmix64 streams; the CPU gold standard reads the probe values. Its own rule is per element, `max |a - g| <= 1e-8`: each row has 10 nonzeros and `sum_j |a_ij x_j| <= 6169`, so any IEEE f64 evaluation order (FMA or not, any reduction tree, atomics) lies within `gamma_10 * 6169 = 6.8e-12` of the exact y and two of them within 1.4e-11. Measured 2026-10-28 on the B200 (scratch reference): the 18 legitimate variants (fmad off, vector lengths 1-32, reverse, pairwise, sorted, Kahan, double-double, f64 atomics, merge-path splits of 7 and 3 nonzeros, cuSPARSE ALG1/ALG2) differ from the golden in 70-77% of the elements by at most 1.8e-12; the CPU gold standard by 1.4e-12; f32-stored values 1.5e-4, f32-stored x 1.8e-4, f32 accumulation 3.9e-4, f32 arithmetic 4.2e-4, f16-stored x 1.2, TF32 inputs 1.7, bf16-stored x 9.8, a zeroed row 954, a dropped nonzero per row 995, the last thread block skipped 3.1e3, a wrong column index 7.4e3, A^T x 7.0e3 -- all FAIL. ctest `spmv_probe` runs it against its frozen reference `reference/probe-real.json`. No output check can tell how many of the 100 timed SpMVs ran: each recomputes the same y from unchanged inputs.

**Correctness switch.** By default the host-side CPU check is skipped and the benchmark prints `SKIP_VERIFY` (tools/timing relies on it); a plain run, ctest, `tools/inputs` and `tools/timing` never run the CPU reference and nothing falls back to it. `HPCPERF_CPU_REFERENCE=1` runs the original CPU check (offline only: `tools/golden/hpcperf_golden.py make` uses it once to validate the golden reference); `HPCPERF_SKIP_VERIFY=1` still forces it off. `HPCPERF_GOLDEN_DUMP=<dir>` makes the run capture its outputs. See [tools/golden/README.md](../../tools/golden/README.md) and [tools/timing/README.md](../../tools/timing/README.md).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `cuda/main.cu` and the HIP port: the benchmark loop, not the warm-up call and check before it. Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 200 (cloc, cuda/); HIP: 200 (identical mirrored source)
