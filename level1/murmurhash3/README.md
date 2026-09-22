# MurmurHash3

MurmurHash3 128-bit (x64 variant) hashing of a batch of variable-length random
keys on the GPU.

Selection note: this benchmark replaced `triangle_counting` (Kokkos Kernels)
by project decision on 2026-08-31 -- upstream Kokkos Kernels has no GPU
triangle-counting implementation to extract (its GPU functor is an empty
stub; see the repository history / final reports for the evidence).

## Source

Source suite: HeCBench (src/murmurhash3-cuda, src/murmurhash3-hip)
Upstream repository: https://github.com/ORNL/HeCBench
Upstream commit: 23714d9980070bc5543e99c11fd93ee6f79c6947

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
cmake -S level1/murmurhash3 -B build/murmurhash3/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/murmurhash3/cuda
```

HIP (build configuration present, unverified without ROCm):

```bash
cmake -S level1/murmurhash3 -B build/murmurhash3/hip -DBACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build/murmurhash3/hip
```

## Run

```bash
./build/murmurhash3/cuda/murmurhash3_cuda 100000 100
```

(arguments: number of keys, repeat count -- the upstream canonical run)

## Validation

Built-in: every GPU 128-bit hash is compared against the upstream CPU
MurmurHash3 reference computed on the host; prints SUCCESS/FAIL.

```bash
ctest --test-dir build/murmurhash3/cuda --output-on-failure
```


**Measurement switch.** `HPCPERF_SKIP_VERIFY=1` skips the host-side check above so `tools/timing/measure_level1.sh` can time the GPU path alone; the benchmark then prints `SKIP_VERIFY` and exits 0. Default (unset) behaviour, and therefore ctest, is unchanged. See [tools/timing/README.md](../../tools/timing/README.md).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `cuda/murmurhash3.cu` and the HIP port: each region the benchmark's own timer measures around its kernel launches. Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 186 (1 source file, cloc, cuda/)
HIP: 186 (1 source file, cloc, hip/)
