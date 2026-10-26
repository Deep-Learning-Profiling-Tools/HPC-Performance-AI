# Channel Shuffle

NCHW/NHWC channel-shuffle operator (as used in ShuffleNet) across group counts.

## Source

Source suite: HeCBench (src/channelShuffle-cuda, src/channelShuffle-hip)
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
cmake -S level1/channel_shuffle -B build/channel_shuffle/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/channel_shuffle/cuda
```

HIP (build configuration present, unverified without ROCm):

```bash
cmake -S level1/channel_shuffle -B build/channel_shuffle/hip -DBACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build/channel_shuffle/hip
```

## Run

```bash
./build/channel_shuffle/cuda/channel_shuffle_cuda 2 224 224 100
```

## Validation

Golden reference (tools/golden): a plain run captures the shuffled output of every (N, C) configuration and both layouts -- 24 outputs `Y.<nhwc|nchw>.N<N>.C<C>` (f32, N * C * W * H each) -- and `hpcperf_golden.py check` compares them byte for byte with the frozen B200 reference `reference/<input>.json`, exactly as upstream memcmp-ed them against the CPU reference (reference.h). The outputs total 11-30 GB per run, so only their sha256 is recorded (`HPCPERF_GOLDEN_MODE=hash`). The CPU reference itself (prints `Average time ...` after each passed comparison, `Failed to pass ...` on a mismatch) runs only on the explicit offline request below; a default run prints neither, only `SKIP_VERIFY` once at the end. On dgx003 a plain run of the default input takes ~12 s, a captured (hash) run ~90-150 s (the sha256 of ~23 GB), and the offline CPU-reference run of `hpcperf_golden.py make` ~11 min. ctest runs `hpcperf_golden.py run` on the registry's default input (`g2-w224-h224`, the former ctest arguments); the `check:` block of `inputs.yaml` does the same for every registered input.

Run it via:

```bash
ctest --test-dir build/channel_shuffle/cuda --output-on-failure
```


**CPU reference switch.** By default the host-side CPU check is skipped: the benchmark prints `SKIP_VERIFY` (never PASS/FAIL) and its correctness is the golden check above. `HPCPERF_CPU_REFERENCE=1` runs the CPU reference (offline only: `hpcperf_golden.py make` uses it once to validate a new golden reference; no measurement, check or ctest path sets it, and nothing falls back to it); `HPCPERF_SKIP_VERIFY=1` (tools/timing) keeps it off as well. See [tools/golden](../../tools/golden/) and [tools/timing/README.md](../../tools/timing/README.md).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `cuda/main.cu` and the HIP port: each region the benchmark's own timer measures around its kernel launches (2 marked regions; their times add up). Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## Warnings

* **Feature maps of 256x256 and larger cannot be swept.** Upstream's `main.cu` computes the element count as
  `const int numel = N * C * W * H` ("assume no integer overflow", `cuda/main.cu:151`) and sweeps N in
  {1, 4, 16, 64} x C in {32, 128, 512}; at (N=16, C=512) a 512x512 map overflows, `cudaMalloc` fails, the
  program prints `Device memory allocation failed. Exit` and **exits 0** after 8 of the 12 configurations.
  The derived input `g2-w512-h512` ran that truncated sweep in every measurement and was deregistered on
  2026-09-28 (its records are INVALIDATED in the results directory); `g2-w255-h255` -- the largest square
  map for which the sweep completes (64 * 512 * 255 * 255 < 2^31) -- replaces it. HeCBench master still has
  the `int numel` (checked 2026-09-28); the kernels take `int numel` and index in `int`, so a `size_t`
  change would touch the kernels and was not applied. The correctness check (`check:` in `inputs.yaml`, the
  golden check) fails such a run: the outputs of the configurations it never reached are missing.

## LOC

CUDA: 208 (2 source files, cloc, cuda/ + common/)
HIP: 208
