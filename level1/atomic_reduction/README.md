# Atomic Reduction

Sum reduction of an integer array implemented with atomic operations in several kernel variants.

## Source

Source suite: HeCBench (src/atomicReduction-cuda, src/atomicReduction-hip)
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
cmake -S level1/atomic_reduction -B build/atomic_reduction/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/atomic_reduction/cuda
```

HIP (build configuration present, unverified without ROCm):

```bash
cmake -S level1/atomic_reduction -B build/atomic_reduction/hip -DBACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build/atomic_reduction/hip
```

## Run

```bash
./build/atomic_reduction/cuda/atomic_reduction_cuda
```

## Validation

Built-in: reduction result checked against the host-computed checksum; prints VERIFICATION: PASS/FAIL.

Run it via:

```bash
ctest --test-dir build/atomic_reduction/cuda --output-on-failure
```

## LOC

CUDA: 112 (2 source files, cloc, cuda/ + common/)
HIP: 112
