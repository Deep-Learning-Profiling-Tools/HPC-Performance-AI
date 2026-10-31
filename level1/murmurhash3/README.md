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

Golden reference (tools/golden): a plain run captures the GPU hashes `hash` (u64, 2 x numKeys: both 64-bit halves of every key's 128-bit hash; `HPCPERF_GOLDEN_DUMP`) and `hpcperf_golden.py check` compares them byte for byte with the frozen B200 reference `reference/<input>.json`, exactly as upstream compared them with the host MurmurHash3 reference. The host reference itself (prints SUCCESS/FAIL) runs only on the explicit offline request below. ctest runs `hpcperf_golden.py run` on the registry's default input (`k100000`, the former ctest arguments `100000 100`); the `check:` block of `inputs.yaml` does the same for every registered input.

```bash
ctest --test-dir build/murmurhash3/cuda --output-on-failure
```


**CPU reference switch.** By default the host-side CPU check is skipped: the benchmark prints `SKIP_VERIFY` (never SUCCESS/FAIL) and its correctness is the golden check above. `HPCPERF_CPU_REFERENCE=1` runs the CPU reference (offline only: `hpcperf_golden.py make` uses it once to validate a new golden reference; no measurement, check or ctest path sets it, and nothing falls back to it); `HPCPERF_SKIP_VERIFY=1` (tools/timing) keeps it off as well. See [tools/golden](../../tools/golden/) and [tools/timing/README.md](../../tools/timing/README.md).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `cuda/murmurhash3.cu` and the HIP port: each region the benchmark's own timer measures around its kernel launches. Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 186 (1 source file, cloc, cuda/)
HIP: 186 (1 source file, cloc, hip/)
