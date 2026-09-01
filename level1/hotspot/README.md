# HotSpot

2D thermal simulation of chip temperature (pyramid-blocked stencil).

## Source

Source suite: Rodinia (cuda/hotspot, data/hotspot)
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
cmake -S level1/hotspot -B build/hotspot/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/hotspot/cuda
```

## Run

```bash
python3 level1/hotspot/verify.py ./build/hotspot/cuda/hotspot_cuda 512 2 2 level1/hotspot/data/temp_512 level1/hotspot/data/power_512
```

## Validation

Added check (verify.py): numpy float32 re-simulation with the upstream constants, compared to output.out (rel tol 1e-3). Uses the upstream canonical 512x512 input.

Run it via:

```bash
ctest --test-dir build/hotspot/cuda --output-on-failure
```

## LOC

CUDA: 243 (1 source files, cloc, cuda/ + common/)
HIP: - (pending)
