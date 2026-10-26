# HotSpot

2D thermal simulation of chip temperature (pyramid-blocked stencil).

## Source

Source suite: Rodinia (cuda/hotspot, data/hotspot)
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
cmake -S level1/hotspot -B build/hotspot/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/hotspot/cuda
```

## Run

```bash
cd <run dir> && ./build/hotspot/cuda/hotspot_cuda 512 2 2 level1/hotspot/data/temp_512 level1/hotspot/data/power_512 output.out
```

## Validation

Correctness = the float32 temperature field (`MatrixOut` after the final copy-back, captured as `temperature`; output.out holds the same field as `%g` text) within max |a - g| / (1 + |g|) <= 1e-3 - d0 of the frozen B200 golden reference `reference/<input_id>.json` (d0 = the golden's own deviation from the CPU re-simulation; rule in `inputs.yaml` `golden:`). The added check `verify.py` (numpy float32 re-simulation with the upstream constants, compared to output.out, rel tol 1e-3) validated that reference once and runs only on request (below). Uses the upstream canonical 512x512 input.

```bash
ctest --test-dir build/hotspot/cuda --output-on-failure   # golden run of the registry's default input
python3 tools/golden/hpcperf_golden.py run level1/hotspot [<input_id>]   # the same for any registered input
```

**Correctness switch.** The binary has no built-in CPU check (nothing to skip, so it prints no `SKIP_VERIFY`); its CPU reference is `verify.py`, which a plain run, ctest, `tools/inputs` and `tools/timing` never run and nothing falls back to. `tools/golden/hpcperf_golden.py make` runs it once, offline (`HPCPERF_CPU_REFERENCE=1`, `HPCPERF_GOLDEN_DUMP`), to validate the golden reference; verify.py then also captures its CPU reference array (`temperature.cpu`, same order) so make can measure d0. `HPCPERF_GOLDEN_DUMP=<dir>` makes the binary capture its output. See [tools/golden/README.md](../../tools/golden/README.md).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `cuda/hotspot.cu`: compute_tran_temp (all simulation iterations). Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 243 (1 source files, cloc, cuda/ + common/)
HIP: - (pending)
