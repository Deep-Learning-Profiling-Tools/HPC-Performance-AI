# PRESSURE

Two-kernel hydrodynamics EOS pressure update (upstream Base_CUDA variant).

## Source

Source suite: RAJAPerf (src/apps/PRESSURE.*)
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
cmake -S level1/pressure -B build/pressure/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/pressure/cuda
```

HIP (build configuration present, unverified without ROCm):

```bash
cmake -S level1/pressure -B build/pressure/hip -DBACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build/pressure/hip
```

## Run

```bash
./build/pressure/cuda/pressure_cuda
```

## Validation

Frozen golden reference ([tools/golden](../../tools/golden/hpcperf_golden.py)): every run captures the checked outputs `bvc`, `p_new` (f64[size] each) and `hpcperf_golden.py check` compares them with the frozen B200 golden reference `reference/<input_id>.json` by the driver's own rule, max |g - r| / (1 + |r|) <= 1e-10 minus the B200 output's own deviation from the CPU reference (measured when the golden was frozen). The added driver's CPU reference (upstream Base_Seq variant with identical data initialization, compared element-wise, rel tol 1e-10, PASS/FAIL) runs only on the explicit offline request `HPCPERF_CPU_REFERENCE=1`, which `hpcperf_golden.py make` uses once to validate the golden.

Run it via (one plain run of the registry's default input, then the golden check):

```bash
ctest --test-dir build/pressure/cuda --output-on-failure
```

**Correctness probe (`probe-randvalue`).** Upstream's set-up data (initData) gives `p_new = bvc * e_old` ~ 0.0117 < `p_cut` and `vnewc >= eosvmax` for every element, so every `p_new` is clamped to `pmin`: a constant fill (`p_new = pmin`) passes every performance input with error 0. The registry's correctness-probe input `probe-randvalue` (`pressure_cuda 1000000 700 --probe`; `inputs.yaml` `purpose: correctness-probe`: never timed, never the default) runs the same two kernels on probe set-up data selected in `setUp` only: `vnewc` from upstream's own `initDataRandValue` (values in [0, 1], same init count) and `p_cut`, `pmin` multiplied by 1e-2. Then 19.6% of the elements (`vnewc < eosvmax`) keep `p_new = bvc * e_old` (0.0115-0.0117) and the rest are clamped; measured 2026-10-27 on the B200: the constant fill fails it (max rel err 1.1e-2), `--fmad=false` passes (0); the CPU reference's deviation is 0. ctest `pressure_probe` runs it against its frozen reference `reference/probe-randvalue.json`. No output-based check can tell how many of the `reps` repetitions ran: each repetition recomputes the same outputs from unchanged inputs.

**Correctness switch.** By default the CPU reference is skipped: the benchmark prints `SKIP_VERIFY` (never PASS) and exits 0, so ctest, `tools/inputs` and `tools/timing` never run it and nothing falls back to it. `HPCPERF_CPU_REFERENCE=1` runs it (offline only; `HPCPERF_SKIP_VERIFY=1` still forces the skip); `HPCPERF_GOLDEN_DUMP=<dir>` captures the outputs ([tools/golden/hpcperf_golden.h](../../tools/golden/hpcperf_golden.h)).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `cuda/main.cu` and the HIP port: the timed repetition loop. Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 216 (2 source files, cloc, cuda/ + common/)
HIP: 216
