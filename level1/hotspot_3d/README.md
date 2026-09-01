# HotSpot3D

3D thermal simulation of stacked chip layers.

## Source

Source suite: Rodinia (cuda/hotspot3D, data/hotspot3D)
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
cmake -S level1/hotspot_3d -B build/hotspot_3d/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/hotspot_3d/cuda
```

## Run

```bash
python3 level1/hotspot_3d/verify.py ./build/hotspot_3d/cuda/hotspot_3d_cuda 64 8 100 level1/hotspot_3d/data/power_64x8 level1/hotspot_3d/data/temp_64x8
```

## Validation

Built-in CPU reference (computeTempCPU) reports an RMSE ('Accuracy'); verify.py enforces Accuracy < 1e-2. Committed input is the upstream 64x8 grid; the canonical 512x8 files (~42 MB) are not committed for repository size reasons and can be taken from upstream data/hotspot3D.

Run it via:

```bash
ctest --test-dir build/hotspot_3d/cuda --output-on-failure
```

## LOC

CUDA: 246 (2 source files, cloc, cuda/ + common/)
HIP: - (pending)
