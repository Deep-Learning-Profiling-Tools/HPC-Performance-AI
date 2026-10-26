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


**CPU reference switch.** By default the host-side CPU check is skipped: the benchmark prints `SKIP_VERIFY` (never PASS/FAIL) and its correctness is the golden check above. `HPCPERF_CPU_REFERENCE=1` runs the CPU reference (offline only: `hpcperf_golden.py make` uses it once to validate a new golden reference; no measurement, check or ctest path sets it, and nothing falls back to it); `HPCPERF_SKIP_VERIFY=1` (tools/timing) keeps it off as well. See [tools/golden](../../tools/golden/) and [tools/timing/README.md](../../tools/timing/README.md).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `cuda/main.cu` and the HIP port: each region the benchmark's own timer measures around its kernel launches. Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 231 (2 source files, cloc, cuda/ + common/)
HIP: 231
