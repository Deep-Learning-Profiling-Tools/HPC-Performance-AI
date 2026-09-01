# NPB MG

NAS Parallel Benchmarks MultiGrid: V-cycle Poisson solver on a 3D grid (CLASS=B).

## Source

Source suite: NPB-GPU (CUDA/MG/mg.cu, HIP/MG/mg.cpp)
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
cmake -S level1/mg -B build/mg/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/mg/cuda
```

HIP (build configuration present, unverified without ROCm):

```bash
cmake -S level1/mg -B build/mg/hip -DBACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build/mg/hip
```

## Run

```bash
./build/mg/cuda/mg_cuda
```

## Validation

Built-in NPB verification (reference L2 norm for the problem class); prints 'Verification = SUCCESSFUL'.

Run it via:

```bash
ctest --test-dir build/mg/cuda --output-on-failure
```

## LOC

CUDA: 2213 (8 source files, cloc, cuda/ + common/)
HIP: 2504

## Notes

Borderline Level 1: standalone CUDA source is ~2.2k LOC (> ~2000 guideline). Flagged for review, not removed.
