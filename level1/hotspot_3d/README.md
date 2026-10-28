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

Golden reference (tools/golden): a plain run captures the GPU temperature field `tempOut` (f32, rows x cols x layers; `HPCPERF_GOLDEN_DUMP`) and `hpcperf_golden.py check` compares it with the frozen B200 reference `reference/<input>.json` under two rules that must both hold: RMSE <= 1e-2 - d0, i.e. the built-in check's threshold (verify.py enforces the CPU-reference RMSE 'Accuracy' <= 1e-2) minus the golden's own RMSE d0 from `computeTempCPU` (measured once when the reference was frozen), and a per-element gate max |a - g| <= 5e-2 K - d0 (not upstream: the RMSE over 32768 elements alone lets one element be off by ~1.8 K; measured 2026-10-28, also on the corrected output: 23 legitimate implementations stay within 3.9e-3 / 9.3e-3 / 1.13e-2 K at i100 / i1000 / i10000, so for spread deviations the RMSE stays the tighter rule). The CPU reference runs only on the explicit offline request below; `hpcperf_golden.py make` runs it through verify.py (which then prints PASS/FAIL; without `HPCPERF_CPU_REFERENCE=1` verify.py finds no 'Accuracy' line and reports FAIL). `tempOut` is the temperature after `iterations` time steps on the GPU and in the CPU reference alike ("Fix: final iterate" below; upstream returned iterate numiter-1). ctest runs `hpcperf_golden.py run` on the registry's default input (`g64-l8-i100`, the same 64x8 grid, 100 iterations as the former verify.py ctest); the `check:` block of `inputs.yaml` does the same for every registered input. Committed input is the upstream 64x8 grid; the canonical 512x8 files (~42 MB) are not committed for repository size reasons and can be taken from upstream data/hotspot3D.

Run it via:

```bash
ctest --test-dir build/hotspot_3d/cuda --output-on-failure   # hotspot_3d_run (g64-l8-i100) and hotspot_3d_probe
```

**Correctness probe (`g64-l8-i50-amp10`).** At `g64-l8-i10000` the field is within ~1.2 K of its steady state, and the steady state of the explicit scheme does not depend on the time step: one skipped iteration changes `tempOut` by 6.3e-4 RMSE, 15 skipped iterations by 9.5e-3, and a time-consistent run with 2, 4 or 8 times larger steps (N/2, N/4, N/8 launches) by <= 5.4e-3 RMSE / 8.9e-3 K max -- all inside the legitimate envelope (6.5e-3 RMSE / 1.13e-2 K, essentially the golden's own f32 drift; measured 2026-10-28 on the B200, on the corrected output), so they pass; `g64-l8-i100` also passes the 2-step version (4.6e-3 RMSE, 1.4e-2 K) and only `g64-l8-i1000` fails it (2.1x). The registry's correctness-probe input (`inputs.yaml` `purpose: correctness-probe`: never timed, never the default) runs the same kernel for 50 steps on upstream's power map and upstream's initial temperature with its fluctuations about 330 K amplified tenfold (`data/temp_64x8_amp10`, written byte-reproducibly by `data/gen_temp_amp10.py`; `--check` verifies the committed file), i.e. far from steady state. Its own golden spec adds a per-element limit equal to the upstream RMSE budget, max |a - g| <= 1e-2 K - d0: 23 legitimate implementations stay <= 2.1e-3 K, while one skipped iteration -- also the final launch, or upstream's former copy-back of iterate N-1 -- deviates by 0.57 K, N/2 launches with 2 dt (time-consistent) by 0.063 K, N/4 with 4 dt by 0.18 K, the top or bottom layer skipped in the final launch by ~1.0 K, f16 / bf16 state by 5.9 / 32 K, an in-place (single-buffer) update by 1.5e-2 K, and a single element off by more than 9.4e-3 K fails; the closest wrong variant measured (x neighbours across a tile edge read from the previous iterate) fails narrowly (1.2e-2 K). ctest `hotspot_3d_probe` runs it against its frozen reference. A skipped final launch (= iterate N-1) also fails `g64-l8-i100` and `g64-l8-i1000` (RMSE 0.125 / 0.077, limit 9.9e-3); at `g64-l8-i10000` one step (6.3e-4 RMSE) is below the legitimate f32 envelope and passes, as above.

**Fix: final iterate (HPC-Performance-AI, 2026-10-28, requested by the owners once the semantics were confirmed; CUDA, the only backend).** `iterations` is the number of explicit time steps and the output is the temperature after them: the usage text says `<iteration> - number of iterations` (cuda/3D.cu:159), both solvers perform exactly `numiter` launches / sweeps, the 2D HotSpot of the same suite returns the buffer of its last step (Rodinia 3.1 `cuda/hotspot/hotspot.cu:241-250,331`: `return dst`, `cudaMemcpy(MatrixOut, MatrixTemp[ret], ...)`; `openmp/hotspot/hotspot_openmp.cpp:317`: `writeoutput((1&sim_time) ? result : temp, ...)`), and HeCBench (ORNL/HeCBench@23714d9: `d_sel = (iterations & 01) ? d_tIn : d_tOut` in `src/hotspot3D-{cuda,hip,sycl}`, the same selection in `-omp`) had to change this benchmark's copy-back so that its GPU result agrees with the CPU reference at odd N (even N still returns N-1 there). Upstream was wrong on both sides: `hotspot_opt1` swaps the device pointers after every launch (cuda/opt1.cu:86-92), so after the loop `tIn_d` holds iterate N and `tOut_d` iterate N-1, and it copied back `tOut_d` (for N = 1 the unchanged input); `computeTempCPU` swaps only its local pointer copies (cuda/3D.cu:127-129), so the caller's `answer` held iterate N for odd N and N-1 for even N. The two sides agreed only for even N; upstream's own check failed at every odd N (Accuracy 0.13 / 0.12 / 0.077 at N = 1 / 99 / 1001, limit 1e-2), and upstream's committed sample outputs (`{cuda,openmp,opencl}/hotspot3D/output.out`, `3D 512 8 100`) hold iterate 99. Now `hotspot_opt1` copies back `tIn_d` and `computeTempCPU` copies its final iterate into the caller's buffer when the last swap left it in the other one: both return iterate N for every N >= 1 (`iterations` < 1 is not a registered input; there the CPU reference's do-while still runs one sweep, as upstream). Nothing else changes: the device code is identical (cuobjdump SASS of the fixed and the unfixed build), the launches and the timed region (ROI 36.91 ms median at `g64-l8-i10000` before and after) are unchanged, and the copy-back stays one device-to-host copy of the same size after the timed region. Evidence (B200, 2026-10-28): a `--fmad=false` build's `tempOut` is bit-identical to a float32 NumPy emulation of N steps in the kernel's summation order at N = 1, 2, 3, 99, 100, 101, 999, 1000, 1001, 9999, 10000 and on the probe data at N = 1, 2, 49, 50, 51 (the unfixed build: iterate N-1 at every N), the CPU reference is bit-identical to the same emulation in its own summation order (unfixed: iterate N-1 at even N), and the default build's output at N is bit-identical to the unfixed build's output at N+1, i.e. exactly one time step later; against a float64 NumPy implementation of the same scheme (coefficients from the code) the fixed output deviates by the f32 drift only (RMSE 2.4e-5 / 8.2e-4 / 2.7e-3 / 5.8e-3 at N = 1 / 100 / 1000 / 10000, 4.3e-4 on the probe; one step is 0.13 / 0.12 / 0.077 / 6.3e-4 / 0.22). Upstream's check now passes at every N tested (Accuracy 1.4e-5 to 1.4e-4). Because the output changed, the frozen references of all four inputs are re-made from the fixed build (`make --replace`).

**CPU reference switch.** By default the host-side CPU check is skipped: the benchmark prints `SKIP_VERIFY` (no 'Accuracy' line) and its correctness is the golden check above. `HPCPERF_CPU_REFERENCE=1` runs the CPU reference (offline only: `hpcperf_golden.py make` uses it once to validate a new golden reference; no measurement, check or ctest path sets it, and nothing falls back to it); `HPCPERF_SKIP_VERIFY=1` (tools/timing) keeps it off as well. See [tools/golden](../../tools/golden/) and [tools/timing/README.md](../../tools/timing/README.md).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `cuda/opt1.cu`: the numiter stencil iterations. Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 246 (2 source files, cloc, cuda/ + common/)
HIP: - (pending)
