# Backprop

One training step of a fully-connected neural network (face recognition workload shape).

## Source

Source suite: Rodinia (cuda/backprop)
Upstream repository: https://github.com/HPC-FAIR/rodinia_3.1
Upstream commit: 366b283456506ef2fe2c2b7dc5e83e941ffc5524

## Verified Environment

GCC/G++ 13.3.0 (conda, pinned) | C++20 | CMake 3.28.4 | Ninja 1.13.2 | Python 3.12.3
CUDA Toolkit 13.2 (nvcc 13.2.78, /usr/local/cuda) | NVIDIA B200 (sm_100), driver 595.58.03
HIP/ROCm: source + build config present where noted, unverified (no AMD GPU available)

Reproduce the toolchain from the repository root: `./setup_env.sh` then
`source hpcperf_env.sh` (all user-space versions are pinned in environment.yml).

## Backends

CUDA: working (configure + build + run + validation verified on B200)
HIP: pending (upstream provides no HIP implementation)

## Build

```bash
source hpcperf_env.sh
cmake -S level1/backprop -B build/backprop/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/backprop/cuda
```

## Run

```bash
./build/backprop/cuda/backprop_cuda 65536
```

## Validation

Added check: an identically-initialized network is trained with the upstream CPU implementation (bpnn_train) and weights/errors are compared (rel tol 1e-3); prints PASS/FAIL.

Run it via:

```bash
ctest --test-dir build/backprop/cuda --output-on-failure
```

## LOC

CUDA: 694 (6 source files, cloc, cuda/ + common/)
HIP: - (pending)
