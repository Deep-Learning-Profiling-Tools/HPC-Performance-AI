# Background Subtraction

Video background-extraction kernel over synthetic frames.

## Source

Source suite: HeCBench (src/background-subtract-cuda, src/background-subtract-hip)
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
cmake -S level1/background_subtraction -B build/background_subtraction/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/background_subtraction/cuda
```

HIP (build configuration present, unverified without ROCm):

```bash
cmake -S level1/background_subtraction -B build/background_subtraction/hip -DBACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build/background_subtraction/hip
```

## Run

```bash
./build/background_subtraction/cuda/background_subtraction_cuda 4096 2048 0 102
```

## Validation

Built-in: GPU output compared against the upstream CPU reference (reference.h); prints PASS/FAIL.

Run it via:

```bash
ctest --test-dir build/background_subtraction/cuda --output-on-failure
```

## Registered inputs (`inputs.yaml`, 2026-09-21)

Four inputs are registered for `tools/inputs/hpcperf_inputs.py` (Level 1 has no
run.sh; the tool runs the binary with the input's arguments, the ctest default
above is unchanged):

| id | what varies | source | args |
|---|---|---|---|
| `w4096-h2048-merged0-r102` (default) | -- (4096 x 2048 frames; repeat = 102 frames in total, 100 of them timed) | upstream `make run` line 1 | `4096 2048 0 102` |
| `w4096-h2048-merged1-r102` | implementation path (one fused kernel instead of three), same frames | upstream `make run` line 2 | `4096 2048 1 102` |
| `w8192-h4096-merged0-r102` | frame size 4x (8192 x 4096 = 33.5 M pixels); 102 frames, 100 timed | derived (upstream has one size) | `8192 4096 0 102` |
| `w4096-h2048-merged0-r1002` | repeat = 1002 frames in total, 1000 timed (instead of 102 / 100) | derived (repeat is the upstream parameter) | `4096 2048 0 1002` |

Timer: the benchmark's `Average kernel execution time` = kernel-only time per
frame (device sync included; host frame generation, H2D copies and the CPU
reference are outside), multiplied by the `repeat - 2` timed frames (cuda/main.cu:
the loop runs `repeat` frames, the timer accumulates for `i >= 2` only and the
average divides by `repeat - 2`; the first two frames are the benchmark's own
warm-up). Three different numbers therefore exist for one run and are reported
separately, never mixed: the per-frame average the benchmark prints, the
cumulative timed section (average x (repeat - 2) = `main_compute_s`) and the
process wall (E2E, which includes the host RNG frame generation and the CPU
reference pass -- neither is ever added to the main compute). Baseline:
`Max error is 0` + `PASS` (bit-exact against the in-process CPU reference). Note
that the binary prints `FAIL` but still exits 0 (main.cu line 185), so the exit
code alone says nothing about correctness; the tool checks the markers.

Pilot calibration on dgx003 (1x B200, 1 warm-up + 3 measured runs, medians):

| id | main compute | spread | process wall (E2E) | main >= 1 s |
|---|---|---|---|---|
| `w4096-h2048-merged0-r102` | 0.0115 s (115 us/frame) | 12.3 % over 3 runs (0.0115, 0.0129, 0.0115 s); re-measured with 5 runs: 0.0114-0.0115 s, 0.9 % | 6.18 s | no |
| `w4096-h2048-merged1-r102` | 0.0070 s | 0.5 % | 6.18 s | no |
| `w8192-h4096-merged0-r102` | 0.0495 s | < 0.1 % (below the 1e-6 s x 100 print resolution) | 23.6 s | no |
| `w4096-h2048-merged0-r1002` | 0.113 s | 0.6 % | 39.8 s | no |

Spread = (max - min) / median of the measured runs; "stable" = 3 or more runs
and spread <= 10 %. Within the measured range (frame sizes 4096x2048 and
8192x4096, 100 and 1000 timed frames) the timed kernel section stays between
0.007 s and 0.113 s per run and the GPU work is a few percent of the wall time
(host RNG + CPU reference dominate). This says nothing about sizes outside that
range; the inputs are kept as registered, the ~1 s value is a reference for the
timing protocol, not a deletion gate. Raw runs:
`HPC-Performance-AI-results/inputs-pilot-2026-09-21/measurements/level1-background_subtraction/`
(the 5-run re-measurement is `w4096-h2048-merged0-r102.remeasure-5reps/`).

## LOC

CUDA: 180 (2 source files, cloc, cuda/ + common/)
HIP: 180
