# AO Bench

Ambient-occlusion ray-tracing mini renderer (aobench) producing a PPM image.

## Source

Source suite: HeCBench (src/aobench-cuda, src/aobench-hip)
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
cmake -S level1/ao_bench -B build/ao_bench/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/ao_bench/cuda
```

HIP (build configuration present, unverified without ROCm):

```bash
cmake -S level1/ao_bench -B build/ao_bench/hip -DBACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build/ao_bench/hip
```

## Run

```bash
./build/ao_bench/cuda/ao_bench_cuda 100
```

## Validation

Added sanity check (verify.py): PPM structure, non-degenerate image, and byte-identical output across two runs (deterministic per-pixel RNG). Upstream ships no reference.

Run it via:

```bash
ctest --test-dir build/ao_bench/cuda --output-on-failure
```

## LOC

CUDA: 327 (1 source files, cloc, cuda/ + common/)
HIP: 327
