# ENERGY

Six-kernel hydrodynamics EOS energy update sequence (upstream Base_CUDA variant).

## Source

Source suite: RAJAPerf (src/apps/ENERGY.*)
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
cmake -S level1/energy -B build/energy/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/energy/cuda
```

HIP (build configuration present, unverified without ROCm):

```bash
cmake -S level1/energy -B build/energy/hip -DBACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build/energy/hip
```

## Run

```bash
./build/energy/cuda/energy_cuda
```

## Validation

Frozen golden reference ([tools/golden](../../tools/golden/hpcperf_golden.py)): every run captures the checked outputs `e_new`, `q_new` (f64[size] each) and `hpcperf_golden.py check` compares them with the frozen B200 golden reference `reference/<input_id>.json` by the driver's own rule, max |g - r| / (1 + |r|) <= 1e-10 minus the B200 output's own deviation from the CPU reference (measured when the golden was frozen). The added driver's CPU reference (upstream Base_Seq variant with identical data initialization, compared element-wise, rel tol 1e-10, PASS/FAIL) runs only on the explicit offline request `HPCPERF_CPU_REFERENCE=1`, which `hpcperf_golden.py make` uses once to validate the golden.

Run it via (one plain run of the registry's default input, then the golden check):

```bash
ctest --test-dir build/energy/cuda --output-on-failure
```

**Correctness probe (`probe-randsign`).** Upstream's set-up data (initData) gives `delvc > 0` for every element, so the `delvc <= 0` branches of bodies 2, 5 and 6 never run, `q_new` stays 0 and every `e_new` (0.167-0.170) falls below `e_cut` and is clamped to `emin`: a constant fill (`e_new = emin`, `q_new = 0`) passes every performance input with error 0. The registry's correctness-probe input `probe-randsign` (`energy_cuda 1000000 130 --probe`; `inputs.yaml` `purpose: correctness-probe`: never timed, never the default) runs the same six kernels on probe set-up data selected in `setUp` only: `delvc` from upstream's own `initDataRandSign` (same factor and init count, about half the elements negative) and `e_cut`, `emin` multiplied by 1e-2. Then `q_new` is non-zero on 50.1% of the elements (0.199-0.206) and no `e_new` is clamped (0.168-0.266); measured 2026-10-27 on the B200: the constant fill fails it (max rel err 0.21 / 0.17), so does dropping the `delvc <= 0` branch (0.028 / 0.17), while `--fmad=false` passes (8.8e-17); the CPU reference's deviation is 8.8e-17 (`e_new`) / 4.6e-17 (`q_new`). ctest `energy_probe` runs it against its frozen reference `reference/probe-randsign.json`. No output-based check can tell how many of the `reps` repetitions ran: each repetition recomputes the same outputs from unchanged inputs.

**Correctness switch.** By default the CPU reference is skipped: the benchmark prints `SKIP_VERIFY` (never PASS) and exits 0, so ctest, `tools/inputs` and `tools/timing` never run it and nothing falls back to it. `HPCPERF_CPU_REFERENCE=1` runs it (offline only; `HPCPERF_SKIP_VERIFY=1` still forces the skip); `HPCPERF_GOLDEN_DUMP=<dir>` captures the outputs ([tools/golden/hpcperf_golden.h](../../tools/golden/hpcperf_golden.h)).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `cuda/main.cu` and the HIP port: the timed repetition loop. Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 340 (2 source files, cloc, cuda/ + common/)
HIP: 340
