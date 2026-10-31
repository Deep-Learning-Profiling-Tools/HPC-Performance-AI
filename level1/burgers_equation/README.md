# Burgers Equation

2D viscous Burgers equation solver (finite differences, explicit time stepping).

## Source

Source suite: HeCBench (src/burger-cuda, src/burger-hip)
Upstream repository: https://github.com/ORNL/HeCBench
Upstream commit: 23714d9980070bc5543e99c11fd93ee6f79c6947

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
cmake -S level1/burgers_equation -B build/burgers_equation/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/burgers_equation/cuda
```

HIP (build configuration present, unverified without ROCm):

```bash
cmake -S level1/burgers_equation -B build/burgers_equation/hip -DBACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build/burgers_equation/hip
```

## Run

```bash
./build/burgers_equation/cuda/burgers_equation_cuda 8200 8100 100
```

## Validation

Golden reference (tools/golden): a plain run captures the GPU solution fields `du` and `dv` (f64, dim_x * dim_y each; `HPCPERF_GOLDEN_DUMP`) and `hpcperf_golden.py check` compares them with the frozen B200 reference `reference/<input>.json`: max |a - g| <= 1e-6 - d0 per field, i.e. upstream's tolerance against the serial CPU solver minus the golden's own deviation d0 from that solver (measured once when the reference was frozen). The serial CPU solver itself (prints PASS/FAIL) runs only on the explicit offline request below. ctest runs `hpcperf_golden.py run` on the registry's default input (`x8200-y8100-t100`, the former ctest arguments); the `check:` block of `inputs.yaml` does the same for every registered input.

Run it via:

```bash
ctest --test-dir build/burgers_equation/cuda --output-on-failure
```


**Correctness probe (`probe-x820-y810-t100`).** Upstream's initial condition gives `u` and `v` the same hat and the two equations are identical when `u == v`, so `du == dv` bit for bit on every performance input (the frozen references carry the same sha256 for both): computing `u` and copying it to `v`, or swapping `u` and `v` in the advection terms of `v`, passes them. The registry's correctness-probe input `probe-x820-y810-t100` (`burgers_equation_cuda 820 810 100 --probe`; `inputs.yaml` `purpose: correctness-probe`: never timed, never the default) runs the same four kernels on a 1/10-per-axis grid of the upstream run with a non-symmetric initial condition, set in set-up only (`probe_initial_v`, after upstream's initialisation and identically before the serial CPU solver): `v`'s hat moved by +0.25 in x (2 on 0.75 < x < 1.25, 0.5 < y < 1), `u` unchanged. Measured 2026-10-27 on the B200: `u` copied to `v` fails it (max abs err `dv` 1.0, `du` 1.7e-2), the swapped advection velocities fail it (`dv` 1.7e-2), `--fmad=false` passes (2.2e-16); the CPU solver's deviation is 2.2e-16. ctest `burgers_equation_probe` runs it against its frozen reference `reference/probe-x820-y810-t100.json`. The HIP port carries the same set-up switch (untested: no ROCm).

**CPU reference switch.** By default the host-side CPU check is skipped: the benchmark prints `SKIP_VERIFY` (never PASS/FAIL) and its correctness is the golden check above. `HPCPERF_CPU_REFERENCE=1` runs the CPU reference (offline only: `hpcperf_golden.py make` uses it once to validate a new golden reference; no measurement, check or ctest path sets it, and nothing falls back to it); `HPCPERF_SKIP_VERIFY=1` (tools/timing) keeps it off as well. See [tools/golden](../../tools/golden/) and [tools/timing/README.md](../../tools/timing/README.md).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `cuda/main.cu` and the HIP port: each region the benchmark's own timer measures around its kernel launches. Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 231 (2 source files, cloc, cuda/ + common/)
HIP: 231
