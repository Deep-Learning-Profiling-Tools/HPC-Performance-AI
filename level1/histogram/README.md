# Histogram

256-bin RGBA histogram with global-memory-atomics and shared-memory-atomics kernel variants.

## Source

Source suite: HeCBench (src/histogram-cuda, src/histogram-hip)
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
cmake -S level1/histogram -B build/histogram/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/histogram/cuda
```

HIP (build configuration present, unverified without ROCm):

```bash
cmake -S level1/histogram -B build/histogram/hip -DBACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build/histogram/hip
```

## Run

```bash
./build/histogram/cuda/histogram_cuda --i=100
```

## Validation

Golden reference (tools/golden): a plain run captures the histogram of every variant after its timed iterations -- 6 outputs `hist.<uchar1|uchar4|float4>.<smem|gmem>` (u32, 256 or 3 x 256 bins), bit-identical to the verification launch's histogram that upstream's check compares and from which the frozen references were made (every launch recomputes `d_hist` from scratch; nothing accumulates across launches). Between the verification launch and the timed iterations `d_hist` -- shared by the smem and gmem variants of a genre -- is poisoned (all-ones bytes), outside the ROI and the timer, so timed iterations that do not run fail the check instead of passing on the verified histogram. `hpcperf_golden.py check` compares the six byte for byte with the frozen B200 reference `reference/<input>.json`, exactly as upstream compared them with the CPU histogram (`HistogramGold`). The CPU histogram itself (prints PASS/FAIL per variant) runs only on the explicit offline request below; a default run prints `SKIP_VERIFY` per variant and never PASS. ctest runs `hpcperf_golden.py run` on the registry's default input (`w1920-h1080`, the former ctest arguments `--i=100`); the `check:` block of `inputs.yaml` does the same for every registered input.

Run it via:

```bash
ctest --test-dir build/histogram/cuda --output-on-failure
```


**CPU reference switch.** By default the host-side CPU check is skipped: the benchmark prints `SKIP_VERIFY` (never PASS/FAIL) and its correctness is the golden check above. `HPCPERF_CPU_REFERENCE=1` runs the CPU reference (offline only: `hpcperf_golden.py make` uses it once to validate a new golden reference; no measurement, check or ctest path sets it, and nothing falls back to it); `HPCPERF_SKIP_VERIFY=1` (tools/timing) keeps it off as well. See [tools/golden](../../tools/golden/) and [tools/timing/README.md](../../tools/timing/README.md).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `cuda/histogram_compare.cu` and the HIP port: the timed iterations, not the verify/warm-up run before them. Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 1152 (5 source files, cloc, cuda/ + common/)
HIP: 1152
