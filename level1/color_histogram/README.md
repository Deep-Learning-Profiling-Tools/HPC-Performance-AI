# Color Histogram

Histogram of random pixel colors (upstream 'hist' benchmark).

## Source

Source suite: Hetero-Mark (src/hist (benchmark + cuda + hip))
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
cmake -S level1/color_histogram -B build/color_histogram/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/color_histogram/cuda
```

HIP (build configuration present, unverified without ROCm):

```bash
cmake -S level1/color_histogram -B build/color_histogram/hip -DBACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build/color_histogram/hip
```

## Run

```bash
./build/color_histogram/cuda/color_histogram_cuda -x 1048576
```

**Input seed.** Upstream seeds the input generator from the clock (`seed = time(NULL)`), so every run drew another random input. Here the seed is fixed to **1** by default (Hetero-Mark's own fixed seed, `kSeedInitValue = 1` of its `ep` benchmark; upstream ships none for hist); the registered inputs (`inputs.yaml` `params.seed`) and their golden references are defined by it. `HPCPERF_RANDOM_SEED=time` restores upstream's clock seed and `HPCPERF_RANDOM_SEED=<n>` (unsigned 32-bit) picks another one; any other value is refused (exit 2). Every run prints the seed it used (`color_histogram: input seed N (...)`), a non-default one marked `NON-DEFAULT SEED`: such a run is not a registered input, and its golden check fails (its outputs differ). The generator itself (glibc `rand_r`) and its call order are upstream's; see `common/hpcperf_measure.h`.

## Validation

Correctness = the 256-bin histogram of the last timed run (captured by `HistBenchmark::CaptureGoldenOutputs`, called by `common/benchmark_runner.cc` after the timed repetitions) bin-for-bin identical to the frozen B200 golden reference `reference/<input_id>.json` (rule in `inputs.yaml` `golden:`). The upstream Verify (GPU histogram compared exactly against the CPU histogram, prints 'Passed.') validated that reference once and runs only on request (below).

```bash
ctest --test-dir build/color_histogram/cuda --output-on-failure   # golden run of the registry's default input
```

**Correctness switch.** By default the host-side CPU check is skipped and the benchmark prints `SKIP_VERIFY` (tools/timing relies on it); a plain run, ctest, `tools/inputs` and `tools/timing` never run the CPU reference and nothing falls back to it. `HPCPERF_CPU_REFERENCE=1` runs the original CPU check (offline only: `tools/golden/hpcperf_golden.py make` uses it once to validate the golden reference; it refuses a run with a non-default seed); `HPCPERF_SKIP_VERIFY=1` still forces it off. `HPCPERF_GOLDEN_DUMP=<dir>` makes the run capture its outputs. `HPCPERF_RANDOM_SEED=time HPCPERF_CPU_REFERENCE=1` reproduces upstream's behaviour (a new random input, validated against the CPU). See [tools/golden/README.md](../../tools/golden/README.md) and [tools/timing/README.md](../../tools/timing/README.md).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `common/benchmark_runner.cc`: the timed repetitions, not WarmUp/Verify. Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 541 (20 source files, cloc, cuda/ + common/)
HIP: 492
