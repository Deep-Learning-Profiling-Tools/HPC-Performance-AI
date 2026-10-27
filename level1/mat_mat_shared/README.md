# MAT_MAT_SHARED

Shared-memory-tiled dense matrix-matrix multiply (upstream Base_CUDA variant).

## Source

Source suite: RAJAPerf (src/basic/MAT_MAT_SHARED.*)
Upstream repository: https://github.com/LLNL/RAJAPerf
Upstream commit: b0f1cd36afd7c2ac57fcb1dc676012be28fb1327

## Verified Environment

GCC/G++ 13.3.0 (conda, pinned) | C++20 | CMake 3.28.4 | Ninja 1.13.2 | Python 3.12.3
CUDA Toolkit 13.2 (nvcc 13.2.78, /usr/local/cuda) | NVIDIA B200 (sm_100), driver 595.58.03
HIP/ROCm: source + build config present where noted, unverified (no AMD GPU available)

Reproduce the toolchain from the repository root: `./setup_env.sh` then
`source hpcperf_env.sh` (all user-space versions are pinned in environment.yml).

## Backends

CUDA: working (configure + build + run + validation verified on B200)
HIP: extracted, untested (no AMD GPU / ROCm on the development machine)

## Build

```bash
source hpcperf_env.sh
cmake -S level1/mat_mat_shared -B build/mat_mat_shared/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/mat_mat_shared/cuda
```

HIP (build configuration present, unverified without ROCm):

```bash
cmake -S level1/mat_mat_shared -B build/mat_mat_shared/hip -DBACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build/mat_mat_shared/hip
```

## Run

```bash
./build/mat_mat_shared/cuda/mat_mat_shared_cuda
```

## Validation

Frozen golden reference ([tools/golden](../../tools/golden/hpcperf_golden.py)): every run captures the checked output `C` (f64[len]) and `hpcperf_golden.py check` compares it with the frozen B200 golden reference `reference/<input_id>.json` by the driver's own rule, max |g - r| / (1 + |r|) <= 1e-10 minus the B200 output's own deviation from the CPU reference (measured when the golden was frozen). The added driver's CPU reference (upstream Base_Seq variant with identical data initialization, compared element-wise, rel tol 1e-10, PASS/FAIL) runs only on the explicit offline request `HPCPERF_CPU_REFERENCE=1`, which `hpcperf_golden.py make` uses once to validate the golden.

Run it via (one plain run of the registry's default input, then the golden check):

```bash
ctest --test-dir build/mat_mat_shared/cuda --output-on-failure
```

**Correctness probe (`probe-rand`).** Upstream's set-up data `A = B = 1` makes every element of `C` equal to `N`, so a constant fill (`C = N`) -- or any tiling, indexing or operand-order error that still sums `N` ones -- passes every performance input with error 0. The registry's correctness-probe input `probe-rand` (`mat_mat_shared_cuda 1000 5 --probe`; `inputs.yaml` `purpose: correctness-probe`: never timed, never the default) runs the same kernel on probe set-up data selected in `main` before the device copy (and identically for the CPU reference): `A` from upstream's own `initDataRandValue` (values in [0, 1]) and `B` from `initDataRandSign` (+-0.1 with random signs), so `A != B` and `B != A^T`. `C` then spans -8.5 .. 8.0; measured 2026-10-27 on the B200: the constant fill fails it (max rel err 1.0e3), swapped operands (`C = B A`) fail it (8.3), `--fmad=false` passes (3.9e-15); the CPU reference's deviation is 3.9e-15. ctest `mat_mat_shared_probe` runs it against its frozen reference `reference/probe-rand.json`. No output-based check can tell how many of the `reps` repetitions ran: each repetition recomputes the same `C`.

**Correctness switch.** By default the CPU reference is skipped: the benchmark prints `SKIP_VERIFY` (never PASS) and exits 0, so ctest, `tools/inputs` and `tools/timing` never run it and nothing falls back to it. `HPCPERF_CPU_REFERENCE=1` runs it (offline only; `HPCPERF_SKIP_VERIFY=1` still forces the skip); `HPCPERF_GOLDEN_DUMP=<dir>` captures the outputs ([tools/golden/hpcperf_golden.h](../../tools/golden/hpcperf_golden.h)).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `cuda/main.cu` and the HIP port: the timed repetition loop. Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 238 (2 source files, cloc, cuda/ + common/)
HIP: 238
