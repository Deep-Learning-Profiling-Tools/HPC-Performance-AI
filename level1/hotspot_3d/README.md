# HotSpot3D

3D thermal simulation of stacked chip layers.

## Source

Source suite: Rodinia (cuda/hotspot3D, data/hotspot3D)
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
cmake -S level1/hotspot_3d -B build/hotspot_3d/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/hotspot_3d/cuda
```

## Run

```bash
python3 level1/hotspot_3d/verify.py ./build/hotspot_3d/cuda/hotspot_3d_cuda 64 8 100 level1/hotspot_3d/data/power_64x8 level1/hotspot_3d/data/temp_64x8
```

## Validation

Golden reference (tools/golden): a plain run captures the GPU temperature field `tempOut` (f32, rows x cols x layers; `HPCPERF_GOLDEN_DUMP`) and `hpcperf_golden.py check` compares it with the frozen B200 reference `reference/<input>.json`: RMSE <= 1e-2 - d0, i.e. the built-in check's threshold (verify.py enforces the CPU-reference RMSE 'Accuracy' <= 1e-2) minus the golden's own RMSE d0 from `computeTempCPU` (measured once when the reference was frozen). The CPU reference runs only on the explicit offline request below; `hpcperf_golden.py make` runs it through verify.py (which then prints PASS/FAIL; without `HPCPERF_CPU_REFERENCE=1` verify.py finds no 'Accuracy' line and reports FAIL). Upstream quirk (kept): `hotspot_opt1` copies back the buffer of iterate numiter-1 (after the last pointer swap); for the registered even iteration counts the CPU reference's buffer holds the same iterate, so both sides compare iterate numiter-1. ctest runs `hpcperf_golden.py run` on the registry's default input (`g64-l8-i100`, the same 64x8 grid, 100 iterations as the former verify.py ctest); the `check:` block of `inputs.yaml` does the same for every registered input. Committed input is the upstream 64x8 grid; the canonical 512x8 files (~42 MB) are not committed for repository size reasons and can be taken from upstream data/hotspot3D.

Run it via:

```bash
ctest --test-dir build/hotspot_3d/cuda --output-on-failure
```


**CPU reference switch.** By default the host-side CPU check is skipped: the benchmark prints `SKIP_VERIFY` (no 'Accuracy' line) and its correctness is the golden check above. `HPCPERF_CPU_REFERENCE=1` runs the CPU reference (offline only: `hpcperf_golden.py make` uses it once to validate a new golden reference; no measurement, check or ctest path sets it, and nothing falls back to it); `HPCPERF_SKIP_VERIFY=1` (tools/timing) keeps it off as well. See [tools/golden](../../tools/golden/) and [tools/timing/README.md](../../tools/timing/README.md).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `cuda/opt1.cu`: the numiter stencil iterations. Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 246 (2 source files, cloc, cuda/ + common/)
HIP: - (pending)
