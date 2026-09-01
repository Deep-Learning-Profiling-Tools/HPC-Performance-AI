# MATVEC_3D_STENCIL

27-point stencil matrix-vector product over real zones of a 3D mesh (upstream Base_CUDA variant).

## Source

Source suite: RAJAPerf (src/apps/MATVEC_3D_STENCIL.*, src/apps/AppsData.*)
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
cmake -S level1/matvec_3d_stencil -B build/matvec_3d_stencil/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/matvec_3d_stencil/cuda
```

HIP (build configuration present, unverified without ROCm):

```bash
cmake -S level1/matvec_3d_stencil -B build/matvec_3d_stencil/hip -DBACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build/matvec_3d_stencil/hip
```

## Run

```bash
./build/matvec_3d_stencil/cuda/matvec_3d_stencil_cuda
```

## Validation

Added driver: upstream Base_Seq CPU variant compared element-wise (rel tol 1e-10); prints PASS/FAIL.

Run it via:

```bash
ctest --test-dir build/matvec_3d_stencil/cuda --output-on-failure
```

## LOC

CUDA: 319 (2 source files, cloc, cuda/ + common/)
HIP: 319
