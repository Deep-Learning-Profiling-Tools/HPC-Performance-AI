# LTIMES

Discrete-ordinates transport moment update phi(z,g,m) += ell(m,d) * psi(z,g,d) (upstream Base_CUDA variant; RAJA views expressed as raw layout-preserving indexing).

## Source

Source suite: RAJAPerf (src/apps/LTIMES.*)
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
cmake -S level1/ltimes -B build/ltimes/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/ltimes/cuda
```

HIP (build configuration present, unverified without ROCm):

```bash
cmake -S level1/ltimes -B build/ltimes/hip -DBACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build/ltimes/hip
```

## Run

```bash
./build/ltimes/cuda/ltimes_cuda
```

## Validation

Added driver: upstream Base_Seq CPU variant compared element-wise (rel tol 1e-10); prints PASS/FAIL.

Run it via:

```bash
ctest --test-dir build/ltimes/cuda --output-on-failure
```

## LOC

CUDA: 199 (2 source files, cloc, cuda/ + common/)
HIP: 199
