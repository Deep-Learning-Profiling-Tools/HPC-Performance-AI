# ENERGY

Six-kernel hydrodynamics EOS energy update sequence (upstream Base_CUDA variant).

## Source

Source suite: RAJAPerf (src/apps/ENERGY.*)
Upstream repository: https://github.com/LLNL/RAJAPerf
Upstream commit: b0f1cd36afd7c2ac57fcb1dc676012be28fb1327

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
cmake -S level1/energy -B build/energy/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/energy/cuda
```

HIP (build configuration present, unverified without ROCm):

```bash
cmake -S level1/energy -B build/energy/hip -DBACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build/energy/hip
```

## Run

```bash
./build/energy/cuda/energy_cuda
```

## Validation

Frozen golden reference ([tools/golden](../../tools/golden/hpcperf_golden.py)): every run captures the checked outputs `e_new`, `q_new` (f64[size] each) and `hpcperf_golden.py check` compares them with the frozen B200 golden reference `reference/<input_id>.json` by the driver's own rule, max |g - r| / (1 + |r|) <= 1e-10 minus the B200 output's own deviation from the CPU reference (measured when the golden was frozen). The added driver's CPU reference (upstream Base_Seq variant with identical data initialization, compared element-wise, rel tol 1e-10, PASS/FAIL) runs only on the explicit offline request `HPCPERF_CPU_REFERENCE=1`, which `hpcperf_golden.py make` uses once to validate the golden.

Run it via (one plain run of the registry's default input, then the golden check):

```bash
ctest --test-dir build/energy/cuda --output-on-failure
```

**Correctness probe (`probe-randsign`).** Upstream's set-up data (initData) gives `delvc > 0` for every element, so the `delvc <= 0` branches of bodies 2, 5 and 6 never run, `q_new` stays 0 and every `e_new` (0.167-0.170) falls below `e_cut` and is clamped to `emin`: a constant fill (`e_new = emin`, `q_new = 0`) passes every performance input with error 0. The registry's correctness-probe input `probe-randsign` (`energy_cuda 1000000 130 --probe`; `inputs.yaml` `purpose: correctness-probe`: never timed, never the default) runs the same six kernels on probe set-up data selected in `setUp` only: `delvc` from upstream's own `initDataRandSign` (same factor and init count, about half the elements negative) and `e_cut`, `emin` multiplied by 1e-2. Then `q_new` is non-zero on 50.1% of the elements (0.199-0.206) and no `e_new` is clamped (0.168-0.266); measured 2026-10-27 on the B200: the constant fill fails it (max rel err 0.21 / 0.17), so does dropping the `delvc <= 0` branch (0.028 / 0.17), while `--fmad=false` passes (8.8e-17); the CPU reference's deviation is 8.8e-17 (`e_new`) / 4.6e-17 (`q_new`). ctest `energy_probe` runs it against its frozen reference `reference/probe-randsign.json`. No output-based check can tell how many of the `reps` repetitions ran: each repetition recomputes the same outputs from unchanged inputs.

**The probe's cut-off scaling is a project choice, not an upstream parameter.** RAJAPerf has no input with negative `delvc` or other cut-offs; the 1e-2 factor on `e_cut` and `emin` is this project's. Reason: with `initDataRandSign` alone, body 5 still zeroes and clamps every `delvc > 0` element (49.92%) to `emin`, so for half of the elements the `e_new` formula of bodies 1, 3, 4 and 5 would be hidden behind the clamp; with the scaling no element is clamped and every `e_new` carries the unclamped formula. The clamp itself is exercised by the performance inputs, where body 5 clamps every element. Branch census (2026-10-28; a host replica of the committed bodies and `setUp`, extracted verbatim from `cuda/main.cu`, its outputs bit-identical to the GPU capture of every input listed):

| input | `delvc <= 0` (bodies 2/5/6 formula branch) | body 5 `e_cut` -> 0 -> `emin` | final `e_new == emin` | `q_new != 0` |
|---|---|---|---|---|
| performance inputs (1e6, 4e6, 16e6) | 0% | 100% | 100% | 0% |
| `probe-randsign` | 50.08% | 0% | 0% | 50.08% (0.1989-0.2057) |
| `probe-cutoffs` (below) | 50.08% | 32.40% | 32.40% | 30.45% (0.1958-0.5) |
| not registered: `initDataRandSign` without the scaling | 50.08% | 49.92% (exactly the `delvc > 0` elements) | 49.92% | 50.08% |

**Second correctness probe (`probe-cutoffs`, 2026-10-28).** On the performance inputs and `probe-randsign`, body 4's `e_cut` / `emin` clamps, body 6's `q_cut` (the smallest non-zero `q_new` of `probe-randsign`, 0.1989, is just above `q_cut` = 0.1958) and its `delvc <= 0` guard (body 6 run on every element: `q_cut` zeroes the extra `q_new`), and the `ssc` floor of bodies 2, 5 and 6 never decide an output, so a candidate that drops any of them passes all four inputs with error 0. The registry's second correctness-probe input `probe-cutoffs` (`energy_cuda 1000000 130 --probe-cutoffs`; ctest `energy_probe_cutoffs`, reference `reference/probe-cutoffs.json`) changes set-up data only: `delvc` and `pbvc` from upstream's `initDataRandSign` (both helpers reseed `srand(4793)`, so `pbvc < 0` exactly where `delvc < 0` and the `ssc` numerator can be negative), `pHalfStep` and `qq_old` from `initDataRandValue` (values in [0, 1]); unlike `probe-randsign` the cut-offs keep their upstream values. Share of the elements whose `e_new` or `q_new` changes when one branch is dropped (the same host replica with one branch removed at a time; the unmodified replica is bit-identical to the GPU capture of all five inputs):

| branch dropped | performance inputs (1e6, 4e6, 16e6) | `probe-randsign` | `probe-cutoffs` |
|---|---|---|---|
| body 2 `delvc > 0` -> `q_new = 0` | 100% | 49.92% | 49.92% |
| body 2 `delvc <= 0` formula | 0% | 50.08% | 40.58% |
| body 2 `ssc` floor | 0% | 0% | 13.72% (NaN) |
| body 4 `e_cut` -> 0 | 0% | 0% | 9.50% |
| body 4 `emin` clamp | 0% | 0% | 27.02% |
| body 5 `delvc > 0` -> `q_tilde = 0` | 0% | 49.92% | 27.02% |
| body 5 `delvc <= 0` formula | 0% | 50.08% | 40.58% |
| body 5 `ssc` floor | 0% | 0% | 50.08% (NaN) |
| body 5 `e_cut` -> 0 | 100% | 0% | 32.40% |
| body 5 `emin` clamp | 100% | 0% | 32.40% |
| body 6 (whole `delvc <= 0` body) | 0% | 50.08% | 50.08% |
| body 6 `delvc <= 0` guard (body on every element) | 0% | 0% | 49.92% |
| body 6 `ssc` floor | 0% | 0% | 50.08% (NaN) |
| body 6 `q_cut` -> 0 | 0% | 0% | 19.63% |

Measured on the B200 against a scratch reference (2026-10-28): each candidate that drops body 4's `e_cut` test, its `emin` clamp, both, body 6's `q_cut`, body 6's `delvc <= 0` guard, or the `ssc` floor of body 2, 5 or 6 passes `size1000000` and `probe-randsign` with error 0 and fails `probe-cutoffs` (max rel err 0.134, 0.134, 0.134, 0.196, 1.11, non-finite x 3); `--fmad=false` passes (1.6e-16), the CPU reference's deviation is 1.6e-16 (`e_new`) / 0 (`q_new`). No element sits near a decision boundary: the smallest body-2 `|ssc|` is 9.9e-7 (no rounding amplification by `sqrt`) and `qq_old` stays at least 6.2e-8 (relative) away from `q_cut`. Replacing the floor by `sqrt(fmax(ssc, 0))` also passes, by construction: 0.3333333e-18 * `ql_old` is below the rounding of `q = ssc * ql_old + qq_old`, so the floor's constant is not observable in any output -- what is checked is that `ssc <= 0` gives `q = qq_old` (then zeroed by `q_cut` where `qq_old < q_cut`) instead of NaN. The kernels and the timed loop are unchanged by both probes (CUDA and HIP sources diffed against the commit before them; the captures of the three performance inputs and of `probe-randsign` are byte-identical to their committed references); the CPU reference calls the same `setUp(cd, size, probe)` and passes on both probes.

**Correctness switch.** By default the CPU reference is skipped: the benchmark prints `SKIP_VERIFY` (never PASS) and exits 0, so ctest, `tools/inputs` and `tools/timing` never run it and nothing falls back to it. `HPCPERF_CPU_REFERENCE=1` runs it (offline only; `HPCPERF_SKIP_VERIFY=1` still forces the skip); `HPCPERF_GOLDEN_DUMP=<dir>` captures the outputs ([tools/golden/hpcperf_golden.h](../../tools/golden/hpcperf_golden.h)).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `cuda/main.cu` and the HIP port: the timed repetition loop. Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 340 (2 source files, cloc, cuda/ + common/)
HIP: 340
