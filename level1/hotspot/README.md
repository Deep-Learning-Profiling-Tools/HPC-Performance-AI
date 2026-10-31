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

**Correctness probe (`g512-p2-t21-pnoise9`).** On the vendored inputs every cell changes by only 0.016-0.018 K per step, almost all of it the leakage term `(amb_temp - T) * Rz_1` (`cuda/hotspot.cu:197-200`); the power term is at most 0.00096 K per step because the time step is chosen so that the maximum power density heats a cell by `PRECISION` = 0.001 K per step (`cuda/hotspot.cu:237-238`), and the field is smooth (rms of its 5-point Laplacian 0.009 K, max 0.11 K). Against the denominator 1 + |T| ~ 324 K of `max_rel1p` a step is 5.7e-5 and the tolerance 1e-3 is 0.32 K, about 19 steps: a no-op passes `g512-p2-t2` (1.2e-4), one skipped iteration (5.4e-5..5.9e-5), a dropped power term (<= 5.7e-4) and an f16-rounded output (3.9e-4) pass all three performance inputs, a stale block in the last launch passes `g512-p2-t20`/`t200` (2.0e-4; `g512-p2-t2` catches it only because its single launch writes into the never-initialized second buffer: with both buffers initialized from the input it passes, 1.0e-4) (measured 2026-10-28). The registry's correctness-probe input (`inputs.yaml` `purpose: correctness-probe`: never timed, never the default) runs the unmodified binary for 21 steps (10 pyramid launches of 2 steps and a 1-step remainder) on the vendored `temp_512` with the power map `power_512` + 9 W x u(k), u(k) = lowbias32(k) / 2^32 of the cell index, which `data/gen_probe.py` writes at configure time to `build/hotspot/cuda/data/power_512_pnoise9`: every cell is heated by 0..3.1 K per step, differently in every cell, so the power, the four neighbours and the leakage move every cell of every block by far more than f32 rounding in every step. Its own rule is `max_rel1p` 3e-5, set from measurements on the probe (B200, 2026-10-28): legitimate variants (fmad off, fast math, blocks 8/32, pyramid heights 1-7 and 12, f32 or f64 arithmetic, f64 state, precomputed, centre-coefficient and delta forms, other summation orders, one step per launch) deviate by <= 2.7e-6; no-op 0.147, one iteration fewer or more 6.4e-3, one stale block in the last launch 1.65e-2..1.74e-2, stale pyramid halo 3.4e-3, frozen boundary 0.147, dropped power / leakage term 0.148 / 1.1e-3, wrong neighbour 2.9e-2, f16 state 3.6e-3, ambient 0 instead of 80 3.4e-4 and f16-rounded input / output field 3.5e-4 / 3.7e-4 fail, the last three with about 11x margin (3e-5 is the geometric mean of 2.7e-6 and 3.4e-4). An f16-stored power map (4.3e-5) fails by 1.4x only and f16-rounded increments on an f32 state (1.3e-5) pass. ctest `hotspot_probe` runs it against its frozen reference `reference/g512-p2-t21-pnoise9.json`. Precision: the benchmark's state is float32 (`MatrixTemp`, the shared tiles); within a step the `2.0` literals (`cuda/hotspot.cu:198-199`) promote the update to double before it is stored as float.

**Correctness switch.** The binary has no built-in CPU check (nothing to skip, so it prints no `SKIP_VERIFY`); its CPU reference is `verify.py`, which a plain run, ctest, `tools/inputs` and `tools/timing` never run and nothing falls back to. `tools/golden/hpcperf_golden.py make` runs it once, offline (`HPCPERF_CPU_REFERENCE=1`, `HPCPERF_GOLDEN_DUMP`), to validate the golden reference; verify.py then also captures its CPU reference array (`temperature.cpu`, same order) so make can measure d0. `HPCPERF_GOLDEN_DUMP=<dir>` makes the binary capture its output. See [tools/golden/README.md](../../tools/golden/README.md).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `cuda/hotspot.cu`: compute_tran_temp (all simulation iterations). Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 243 (1 source files, cloc, cuda/ + common/)
HIP: - (pending)
