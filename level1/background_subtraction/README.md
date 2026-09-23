# Background Subtraction

Video background-extraction kernel over synthetic frames.

## Source

Source suite: HeCBench (src/background-subtract-cuda, src/background-subtract-hip)
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
cmake -S level1/background_subtraction -B build/background_subtraction/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/background_subtraction/cuda
```

HIP (build configuration present, unverified without ROCm):

```bash
cmake -S level1/background_subtraction -B build/background_subtraction/hip -DBACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build/background_subtraction/hip
```

## Run

```bash
./build/background_subtraction/cuda/background_subtraction_cuda 4096 2048 0 102
```

## Validation

Built-in: GPU output compared against the upstream CPU reference (reference.h); prints PASS/FAIL.

Run it via:

```bash
ctest --test-dir build/background_subtraction/cuda --output-on-failure
```


**Measurement switch.** `HPCPERF_SKIP_VERIFY=1` skips the host-side check above so `tools/timing/measure_level1.sh` can time the GPU path alone; the benchmark then prints `SKIP_VERIFY` and exits 0. Default (unset) behaviour, and therefore ctest, is unchanged. See [tools/timing/README.md](../../tools/timing/README.md).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `cuda/main.cu` and the HIP port: the frame loop; frame generation and CPU reference excluded. The synthetic frame generation and the CPU reference (`merge_ref`) inside the frame loop are excluded. Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 180 (2 source files, cloc, cuda/ + common/)
HIP: 180
