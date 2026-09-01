# Adam

Adam optimizer update step over a parameter vector, repeated for multiple time steps.

## Source

Source suite: HeCBench (src/adam-cuda, src/adam-hip)
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
cmake -S level1/adam -B build/adam/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/adam/cuda
```

HIP (build configuration present, unverified without ROCm):

```bash
cmake -S level1/adam -B build/adam/hip -DBACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build/adam/hip
```

## Run

```bash
./build/adam/cuda/adam_cuda 10000 200 100
```

## Validation

Built-in: GPU result compared element-wise against the upstream CPU reference (reference.h); prints PASS/FAIL.

Run it via:

```bash
ctest --test-dir build/adam/cuda --output-on-failure
```

## LOC

CUDA: 177 (2 source files, cloc, cuda/ + common/)
HIP: 177
