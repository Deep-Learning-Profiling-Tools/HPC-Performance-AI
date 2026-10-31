# Burrows-Wheeler Transform

Burrows-Wheeler transform of a generated string via GPU suffix comparison.

## Source

Source suite: HeCBench (src/bwt-cuda, src/bwt-hip)
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
cmake -S level1/burrows_wheeler_transform -B build/burrows_wheeler_transform/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/burrows_wheeler_transform/cuda
```

HIP (build configuration present, unverified without ROCm):

```bash
cmake -S level1/burrows_wheeler_transform -B build/burrows_wheeler_transform/hip -DBACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build/burrows_wheeler_transform/hip
```

## Run

```bash
./build/burrows_wheeler_transform/cuda/burrows_wheeler_transform_cuda 10000000
```

## Validation

Golden reference (tools/golden): a plain run captures the GPU's transformed sequence `bwt` (u8, n characters; `HPCPERF_GOLDEN_DUMP`) and `hpcperf_golden.py check` compares it byte for byte (sha256) with the frozen B200 reference `reference/<input>.json`, exactly as upstream compared it with the host transform `bwt_cpu`. The host transform itself (prints PASS/FAIL) runs only on the explicit offline request below. ctest runs `hpcperf_golden.py run` on the registry's default input (`n10000000`, the former ctest argument); the `check:` block of `inputs.yaml` does the same for every registered input.

Run it via:

```bash
ctest --test-dir build/burrows_wheeler_transform/cuda --output-on-failure
```


**CPU reference switch.** By default the host-side CPU check is skipped: the benchmark prints `SKIP_VERIFY` (never PASS/FAIL) and its correctness is the golden check above. `HPCPERF_CPU_REFERENCE=1` runs the CPU reference (offline only: `hpcperf_golden.py make` uses it once to validate a new golden reference; no measurement, check or ctest path sets it, and nothing falls back to it); `HPCPERF_SKIP_VERIFY=1` (tools/timing) keeps it off as well. See [tools/golden](../../tools/golden/) and [tools/timing/README.md](../../tools/timing/README.md).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `cuda/bwt.cu` and the HIP port: bitonic sort of the rotations + reconstruct. Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 149 (3 source files, cloc, cuda/ + common/)
HIP: 149
