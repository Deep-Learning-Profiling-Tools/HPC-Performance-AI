# AES

AES-256 ECB encryption of a hex-encoded plaintext file.

## Source

Source suite: Hetero-Mark (src/aes (benchmark + cuda + hip))
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
cmake -S level1/aes -B build/aes/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/aes/cuda
```

HIP (build configuration present, unverified without ROCm):

```bash
cmake -S level1/aes -B build/aes/hip -DBACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build/aes/hip
```

## Run

```bash
./build/aes/cuda/aes_cuda -i build/aes/cuda/data/input_1MB.hex -k build/aes/cuda/data/key.hex
```

## Validation

Correctness = the ciphertext of the last timed run (captured by `AesBenchmark::CaptureGoldenOutputs`, called by `common/benchmark_runner.cc` after the timed repetitions) byte-identical to the frozen B200 golden reference `reference/<input_id>.json` (rule in `inputs.yaml` `golden:`). The upstream Verify (GPU ciphertext compared byte-wise against the CPU AES-256 reference, prints 'Passed.') validated that reference once and runs only on request (below). Input is generated deterministically at configure time (data/gen_input.py) because the upstream data server is offline.

```bash
ctest --test-dir build/aes/cuda --output-on-failure   # golden run of the registry's default input
```

**Correctness switch.** By default the host-side CPU check is skipped and the benchmark prints `SKIP_VERIFY` (tools/timing relies on it); a plain run, ctest, `tools/inputs` and `tools/timing` never run the CPU reference and nothing falls back to it. `HPCPERF_CPU_REFERENCE=1` runs the original CPU check (offline only: `tools/golden/hpcperf_golden.py make` uses it once to validate the golden reference); `HPCPERF_SKIP_VERIFY=1` still forces it off. `HPCPERF_GOLDEN_DUMP=<dir>` makes the run capture its outputs. See [tools/golden/README.md](../../tools/golden/README.md) and [tools/timing/README.md](../../tools/timing/README.md).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `common/benchmark_runner.cc`: the timed repetitions, not WarmUp/Verify. Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 881 (20 source files, cloc, cuda/ + common/)
HIP: 739
