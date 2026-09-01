# NPB IS

NAS Parallel Benchmarks Integer Sort: bucketed integer key ranking (CLASS=B).

## Source

Source suite: NPB-GPU (CUDA/IS/is.cu, HIP/IS/is.cpp)
Upstream repository: https://github.com/GMAP/NPB-GPU
Upstream commit: 3f12d84920ee315ab00ef283717c1e74b68f4d00

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
cmake -S level1/is -B build/is/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/is/cuda
```

HIP (build configuration present, unverified without ROCm):

```bash
cmake -S level1/is -B build/is/hip -DBACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build/is/hip
```

## Run

```bash
./build/is/cuda/is_cuda
```

## Validation

Built-in NPB verification (partial and full key ranking checks); prints 'Verification = SUCCESSFUL'.

Run it via:

```bash
ctest --test-dir build/is/cuda --output-on-failure
```

## LOC

CUDA: 1265 (8 source files, cloc, cuda/ + common/)
HIP: 1542
