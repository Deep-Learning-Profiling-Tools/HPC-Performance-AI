# Bezier Surface

Tessellation of a Bezier surface from a control-point grid.

## Source

Source suite: HeCBench (src/bezier-surface-cuda, src/bezier-surface-hip)
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
cmake -S level1/bezier_surface -B build/bezier_surface/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/bezier_surface/cuda
```

HIP (build configuration present, unverified without ROCm):

```bash
cmake -S level1/bezier_surface -B build/bezier_surface/hip -DBACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build/bezier_surface/hip
```

## Run

```bash
./build/bezier_surface/cuda/bezier_surface_cuda -f level1/bezier_surface/data/control.txt -n 8192
```

## Validation

Golden reference (tools/golden): a plain run captures the evaluated surface (`surface`, 3 floats per point; relative L1 norm <= 1e-6 minus the golden's own deviation from the CPU reference), and `tools/golden/hpcperf_golden.py check` compares the capture with the frozen B200 reference `reference/<input_id>.json` (inputs.yaml `golden:` / `check:`). ctest runs the registry's default input this way (`hpcperf_golden.py run`). The upstream CPU reference (`BezierCPU` plus `compare_output`, relative L1 norm < 1e-6) stays in the source and runs only with `HPCPERF_CPU_REFERENCE=1`, which `hpcperf_golden.py make` uses once to validate a new reference.

Run it via:

```bash
ctest --test-dir build/bezier_surface/cuda --output-on-failure
```


**Fixes (HPC-Performance-AI, 2026-10-27; CUDA and HIP).** (1) Kernel bound, class D: `BezierGPU` returned only for `i > RESOLUTIONI`, so thread `i == RESOLUTIONI` -- present whenever `-n` is not a multiple of the 256-thread block, e.g. the registered `-n 300` -- wrote a whole row past `outp` (compute-sanitizer memcheck: "Invalid __global__ write of size 4 ... after the allocation of size 1,080,000 bytes"; upstream HeCBench@23714d9 has the same line in its CUDA and HIP ports, while its SYCL port tests `i >= out_size_i` and its OpenMP port loops `i < out_size_i`). Now `i >= RESOLUTIONI`. The surface is unchanged: the extra thread only wrote outside the output; memcheck is clean at `-n 300` and `-n 8192`, the captured surfaces are byte-identical to the frozen references (sha256 5658f0cd... / 7c129b8a...), stdout is identical, the kernel ROI is unchanged (2.40 / 62.7 ms median, before and after). (2) Resolution guard: `run()` sizes the buffers with `int out_size = n * n * sizeof(XYZ)`, which wraps for `n >= 13378` (the point index `i * RESOLUTIONJ + j` wraps from `n >= 46341`), and `mu = i / (n - 1)` divides by zero at `n = 1`. Measured at `-n 16384`: `cudaMalloc` fails ("out of memory" for the wrapped size), the kernel writes through the unset `d_out`, and the upstream check prints FAIL -- with `out_size` as `size_t` (a scratch experiment, not applied) the same run PASSes, as does `-n 32768`. This, not the float evaluation, is why the unregistered x2 / x4 resolutions failed (inputs.yaml `coverage`). `main()` now refuses `-n` outside `2..13377` (`2..9459` with `DOUBLE_PRECISION`) with "error: unsupported output resolution" and exit 1; every accepted `n` runs unchanged.

**Measurement switch.** By default the host-side CPU reference is skipped and the benchmark prints `SKIP_VERIFY` (exit 0), so a plain run, ctest, `tools/inputs` and `tools/timing` never execute it; `HPCPERF_CPU_REFERENCE=1` (offline, golden make only) turns it on, `HPCPERF_SKIP_VERIFY=1` keeps it off regardless. Output capture (`HPCPERF_GOLDEN_DUMP`) is outside the timed region. See [tools/golden/hpcperf_golden.h](../../tools/golden/hpcperf_golden.h) and [tools/timing/README.md](../../tools/timing/README.md).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `cuda/main.cu` and the HIP port: each region the benchmark's own timer measures around its kernel launches. Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 223 (1 source files, cloc, cuda/ + common/)
HIP: 223
