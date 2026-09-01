# SRAD v1

Speckle-reducing anisotropic diffusion on an ultrasound image.

## Source

Source suite: Rodinia (cuda/srad/srad_v1, data/srad)
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
cmake -S level1/srad_v1 -B build/srad_v1/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/srad_v1/cuda
```

## Run

```bash
python3 level1/srad_v1/verify.py ./build/srad_v1/cuda/srad_v1_cuda 100 0.5 502 458 level1/srad_v1/data/image.pgm
```

## Validation

Added check (verify.py): full numpy float32 re-implementation of the SRAD pipeline compared against image_out.pgm (max pixel diff <= 3, <0.1% pixels off by more than 1). The input image path was made a command-line argument (upstream hardcoded a relative path).

Run it via:

```bash
ctest --test-dir build/srad_v1/cuda --output-on-failure
```

## LOC

CUDA: 628 (12 source files, cloc, cuda/ + common/)
HIP: - (pending)
