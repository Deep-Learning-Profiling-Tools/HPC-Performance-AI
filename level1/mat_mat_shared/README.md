# MAT_MAT_SHARED

Shared-memory-tiled dense matrix-matrix multiply (upstream Base_CUDA variant).

## Source

Source suite: RAJAPerf (src/basic/MAT_MAT_SHARED.*)
Upstream repository: https://github.com/LLNL/RAJAPerf
Upstream commit: b0f1cd36afd7c2ac57fcb1dc676012be28fb1327

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
cmake -S level1/mat_mat_shared -B build/mat_mat_shared/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/mat_mat_shared/cuda
```

HIP (build configuration present, unverified without ROCm):

```bash
cmake -S level1/mat_mat_shared -B build/mat_mat_shared/hip -DBACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build/mat_mat_shared/hip
```

## Run

```bash
./build/mat_mat_shared/cuda/mat_mat_shared_cuda
```

## Validation

Added driver: upstream Base_Seq CPU variant compared element-wise (rel tol 1e-10); prints PASS/FAIL.

Run it via:

```bash
ctest --test-dir build/mat_mat_shared/cuda --output-on-failure
```

## LOC

CUDA: 238 (2 source files, cloc, cuda/ + common/)
HIP: 238
