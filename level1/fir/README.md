# FIR

Finite impulse response filter over blocked input data.

## Source

Source suite: Hetero-Mark (src/fir (benchmark + cuda + hip))
Upstream repository: https://github.com/NUCAR-DEV/Hetero-Mark
Upstream commit: aa685e02705f4be380cba215c6df10b42262a9cd

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
cmake -S level1/fir -B build/fir/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/fir/cuda
```

HIP (build configuration present, unverified without ROCm):

```bash
cmake -S level1/fir -B build/fir/hip -DBACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build/fir/hip
```

## Run

```bash
./build/fir/cuda/fir_cuda -y 1024 -x 4096
```

## Validation

Built-in (upstream Verify): GPU-filtered signal compared against CPU FIR; prints 'Passed!'.

Run it via:

```bash
ctest --test-dir build/fir/cuda --output-on-failure
```

## LOC

CUDA: 594 (20 source files, cloc, cuda/ + common/)
HIP: 522
