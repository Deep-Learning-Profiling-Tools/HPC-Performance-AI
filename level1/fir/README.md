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

**Input seed.** Upstream seeds the input generator from the clock (`seed = time(NULL)`), so every run drew another random input. Here the seed is fixed to **1** by default (Hetero-Mark's own fixed seed, `kSeedInitValue = 1` of its `ep` benchmark; upstream ships none for fir); the registered inputs (`inputs.yaml` `params.seed`) and their golden references are defined by it. `HPCPERF_RANDOM_SEED=time` restores upstream's clock seed and `HPCPERF_RANDOM_SEED=<n>` (unsigned 32-bit) picks another one; any other value is refused (exit 2). Every run prints the seed it used (`fir: input seed N (...)`), a non-default one marked `NON-DEFAULT SEED`: such a run is not a registered input, and its golden check fails (its outputs differ). The generator itself (glibc `rand_r`) and its call order are upstream's; see `common/hpcperf_measure.h`.

## Validation

Correctness = the filtered signal of the last timed run (captured by `FirBenchmark::CaptureGoldenOutputs`, called by `common/benchmark_runner.cc` after the timed repetitions) within max |a - g| <= 1e-5 - d0 of the frozen B200 golden reference `reference/<input_id>.json` (d0 = the golden's own deviation from the CPU FIR, 1.4e-6 on the default input; rule in `inputs.yaml` `golden:`). The upstream Verify (GPU-filtered signal compared against the CPU FIR, abs 1e-5 per point, prints 'Passed!') validated that reference once and runs only on request (below).

```bash
ctest --test-dir build/fir/cuda --output-on-failure   # golden run of the registry's default input
```

**Correctness switch.** By default the host-side CPU check is skipped and the benchmark prints `SKIP_VERIFY` (tools/timing relies on it); a plain run, ctest, `tools/inputs` and `tools/timing` never run the CPU reference and nothing falls back to it. `HPCPERF_CPU_REFERENCE=1` runs the original CPU check (offline only: `tools/golden/hpcperf_golden.py make` uses it once to validate the golden reference; it refuses a run with a non-default seed); `HPCPERF_SKIP_VERIFY=1` still forces it off. `HPCPERF_GOLDEN_DUMP=<dir>` makes the run capture its outputs. `HPCPERF_RANDOM_SEED=time HPCPERF_CPU_REFERENCE=1` reproduces upstream's behaviour (a new random input, validated against the CPU). See [tools/golden/README.md](../../tools/golden/README.md) and [tools/timing/README.md](../../tools/timing/README.md).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `common/benchmark_runner.cc`: the timed repetitions, not WarmUp/Verify. Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 594 (20 source files, cloc, cuda/ + common/)
HIP: 522
