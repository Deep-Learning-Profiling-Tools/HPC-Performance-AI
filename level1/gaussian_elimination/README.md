# Gaussian Elimination

Dense Gaussian elimination solver without pivoting.

## Source

Source suite: Rodinia (cuda/gaussian, data/gaussian)
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
cmake -S level1/gaussian_elimination -B build/gaussian_elimination/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/gaussian_elimination/cuda
```

## Run

```bash
python3 level1/gaussian_elimination/verify.py ./build/gaussian_elimination/cuda/gaussian_elimination_cuda level1/gaussian_elimination/data/matrix208.txt
```

## Validation

Added check (verify.py): parses the printed solution and verifies the residual ||Ax-b||_inf on the CPU (tol 1e-2 for the float32 solve). Solution print precision raised from %.2f to %.6f for validation.

Run it via:

```bash
ctest --test-dir build/gaussian_elimination/cuda --output-on-failure
```

## LOC

CUDA: 302 (1 source files, cloc, cuda/ + common/)
HIP: - (pending)
