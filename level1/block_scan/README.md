# Block Scan

Block-level inclusive prefix sum (scan) using shared memory.

## Source

Source suite: HeCBench (src/blockScan-cuda, src/blockScan-hip)
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
cmake -S level1/block_scan -B build/block_scan/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/block_scan/cuda
```

HIP (build configuration present, unverified without ROCm):

```bash
cmake -S level1/block_scan -B build/block_scan/hip -DBACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build/block_scan/hip
```

## Run

```bash
./build/block_scan/cuda/block_scan_cuda 2227104 100
```

## Validation

Golden reference (tools/golden): a plain run captures the scan output of each of the 18 variants after its timed launches (`scan.<algorithm>.t<threads>.i<items>`, 1024 prefix sums + the aggregate, int, compared exactly; bit-identical to the first warm-up launch's output that upstream's check compares and from which the frozen references were made). Between the warm-up launches and the timed region `d_out` (and its host copy) is poisoned (all-ones bytes, -1), outside the ROI and the timer, so timed launches that do not run fail the check instead of passing on the warm-up result. `tools/golden/hpcperf_golden.py check` compares the capture with the frozen B200 reference `reference/<input_id>.json` (inputs.yaml `golden:` / `check:`). ctest runs the registry's default input this way (`hpcperf_golden.py run`). The upstream CPU reference (`memcmp` with the host prefix sums from `Initialize`) stays in the source and runs only with `HPCPERF_CPU_REFERENCE=1`, which `hpcperf_golden.py make` uses once to validate a new reference.

Run it via:

```bash
ctest --test-dir build/block_scan/cuda --output-on-failure
```


**Measurement switch.** By default the host-side CPU reference is skipped and the benchmark prints `SKIP_VERIFY` (exit 0), so a plain run, ctest, `tools/inputs` and `tools/timing` never execute it; `HPCPERF_CPU_REFERENCE=1` (offline, golden make only) turns it on, `HPCPERF_SKIP_VERIFY=1` keeps it off regardless. Output capture (`HPCPERF_GOLDEN_DUMP`) is outside the timed region. See [tools/golden/hpcperf_golden.h](../../tools/golden/hpcperf_golden.h) and [tools/timing/README.md](../../tools/timing/README.md).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `cuda/main.cu` and the HIP port: each region the benchmark's own timer measures around its kernel launches. Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 162 (1 source files, cloc, cuda/ + common/)
HIP: 160
