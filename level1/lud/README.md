# LUD

Blocked LU decomposition of a dense matrix.

## Source

Source suite: Rodinia (cuda/lud)
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
cmake -S level1/lud -B build/lud/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/lud/cuda
```

## Run

```bash
./build/lud/cuda/lud_cuda -s 256 -v
```

## Validation

The criterion is upstream's own (`common/common.c` `lud_verify`): the factors are recomposed, L with its unit
diagonal times U, and every one of the n*n elements with `fabs(A_ij - (L U)_ij) > 0.0001` prints `dismatch`.

**Default path (plain run, ctest, `tools/inputs check`).** `-v` stays in the registered arguments, but by default
it no longer runs upstream's host recomposition (a single-thread float32 O(n^3) loop: 7 ms / 14 s / 922-969 s at
s256 / s2048 / s8192, against 1 / 7 / 118-164 ms of GPU work). Instead the program captures the packed factor
matrix after the timed region and the final device-to-host copy (`hpcperf_golden_output("lu", ...)`, a no-op
unless `HPCPERF_GOLDEN_DUMP` is set; `tools/golden/README.md`) and prints `VERIFY_DEFERRED: ...`;
`lud_residual_check.py` then applies the same criterion to that capture:

* A is regenerated from `create_matrix` with the same float32 operations (coefficients
  `(float)(10 exp((double)(-0.001f * (float)i)))`, libm `exp`, `A_ij = coe[|i - j|]`): bit-identical to the
  program's matrix at 256, 2048 and 8192 (compared with the program's own copy). The check therefore judges the
  factors of the registered matrix, not a matrix the run supplies.
* `L U` (numpy matmul = OpenBLAS dgemm) and `A - L U` are formed in float64; PASS iff every factor is finite and
  `max_ij |A_ij - (L U)_ij| <= 1e-4` over all n*n elements (the same double constant). On FAIL the first ten
  offending elements are printed in upstream's `dismatch at (i, j): (o).. (n)..` format, row-major.
* The capture's manifest record (name `lu`, f32, n*n elements, sha256 of the bytes) is verified; a missing,
  partial, hash-only or wrongly sized capture is refused (exit 2, no PASS line). `-i <file>` is not supported (no
  registered input uses it). It needs no golden and no CPU reference.

```bash
ctest --test-dir build/lud/cuda --output-on-failure      # verify.py: one -s 256 -v run with the capture, then the check
python3 tools/inputs/hpcperf_inputs.py check level1/lud s8192 --out <dir>    # the registry check (any input)
python3 level1/lud/lud_residual_check.py --dump <capture dir> -- -s 8192 -v   # by hand, on a run's capture
python3 level1/lud/test_lud_residual_check.py            # CPU-only tests of the checker
```

**Offline: upstream's check, unchanged.** `HPCPERF_CPU_REFERENCE=1 ./build/lud/cuda/lud_cuda -s N -v` runs
`lud_verify` exactly as before (`Before LUD`, `>>>Verify<<<<`, one `dismatch` line per offending element);
`HPCPERF_SKIP_VERIFY=1` (tools/timing) still forces it off and prints `SKIP_VERIFY`. The capture works in every
mode, so one offline run yields both verdicts on the same factors.

**What differs in numerical meaning** (B200, 2026-10-27; upstream's float32 values from the real `lud_verify` and,
for the full s8192 sweep, from an OpenMP replica of its loop that reproduced the real check's `dismatch` lines
line for line in all 55 runs where both ran)

| input | exact max\|A - LU\| (this check) | upstream's float32 max\|A - tmp\| | upstream's own accumulation error max\|tmp - LU\| | float64 evaluation error bound |
|---|---|---|---|---|
| s256  | 3.82e-6 | 1.34e-5 | 1.33e-5 | 2.8e-13 |
| s2048 | 9.48e-6 | 2.10e-5 | 1.80e-5 | 2.3e-12 |
| s8192 | 2.78e-5 | 3.62e-5 | 1.93e-5 | 9.1e-12 |

* Accumulation: upstream sums L*U in float32 (no FMA: `-march=nocona`), so its verdict carries its own rounding
  (a-priori bound n u max(|L||U|) = 1.5e-4 / 1.2e-3 / 4.9e-3, measured at most 1.9e-5); this check's float64 error is
  below 1e-11. The float32 subtraction `m - tmp` is exact here (Sterbenz; A > 2.8e-3), so the accumulation is the
  only difference. Verdicts can therefore differ only when an element's exact residual lies within upstream's
  accumulation error (<= 1.9e-5) of 1e-4 -- and there this check judges the factors, upstream its own rounding.
  Constructed in-band single-element perturbations show both directions: exact residuals of 9.1e-5 .. 9.8e-5
  that upstream's float32 sum puts at 1.02e-4 .. 1.09e-4 (upstream FAIL, this check PASS; at every size), and at
  s8192 an exact 1.025e-4 that upstream computes as 9.73e-5 (upstream PASS, this check FAIL).
* NaN: `fabs(NaN) > 0.0001` is false, so upstream passes NaN factors (one NaN in U leaves 157 / 1250 / 4998 NaN elements
  of L*U and not one `dismatch` line, measured at s256 / s2048 / s8192); this check fails any non-finite factor.
* A comes from `create_matrix`, not from the program's memory: a run that changed its own matrix is judged against
  the registered one (upstream compared with the program's own copy).

**Why the exact residual is the intended criterion.** Upstream's criterion is a statement about the factors -- every
element of `A - L U` within 1e-4. The float32 sum that upstream uses to evaluate `L U` adds the rounding of the checking
loop itself (its summation order, and FMA contraction, which the current `-march=nocona` host build does not do but an
FMA-capable host target may); that rounding is not a property of the GPU result, differs between hosts and compilers,
and decides a verdict only inside the band |exact residual - 1e-4| <~ 1.9e-5 (about 8.1e-5 .. 1.19e-4). The float64
evaluation (error < 1e-11 whatever the BLAS, thread count or summation order) keeps the verdict the same on every host
and GPU architecture. No executed variant lies in that band: legitimate implementations reach at most 4.6e-5 (the
registered s8192 run 2.78e-5), the smallest deliberate fault 2.0e-4 and every algorithmic fault >= 0.31 (table below);
only constructed in-band perturbations (above) separate the two checks.

Re-checked 2026-10-28 on the committed tree (B200; capture of a default run, then `lud_residual_check.py`; perturbed
copies of the captured factors through `--lu`): baseline PASS at s256 / s2048 / s8192 (3.818e-6 / 9.485e-6 /
2.776e-5); one U element (n/2, n/2+3) + 2e-4 FAILs (max 2.003e-4 / 2.021e-4 / 2.090e-4; 128 / 694 / 690 elements above
1e-4: row i >= n/2 of column n/2+3 of `L U` shifts by `L[i][n/2] * 2e-4`, `L[n/2][n/2]` = 1), the same element + 5e-5 PASSes
(5.06e-5 / 5.22e-5 / 5.97e-5), one NaN in U (n-1, n-1) or in L (3, 1) FAILs (non-finite factor).

**Negative and legitimate variants (executed).** Fault-injection copies of these sources (scratch builds, not in the
tree) run with `HPCPERF_CPU_REFERENCE=1` and the capture, so both checks judge the same factors. Each verdict was the
expected one at s256 and s2048 (real upstream check, 24 variants each) and s8192 (real upstream check for 7
variants, the replica for all 24):

| variant | this check | upstream |
|---|---|---|
| baseline; BLOCK_SIZE 8 and 32 (`-DRD_WG_SIZE`); `--fmad=false`; float64 LU rounded to float32; numpy float32 blocked LU; LAPACK getrf (no row interchange occurs) | PASS (max 2.4e-7 .. 4.6e-5) | PASS |
| trailing update skipped at the first / middle / last step; a stale trailing tile (first / middle / last); last diagonal block not factored; no division by the pivot; Crout convention (unit U) | FAIL (0.31 .. 89.9; non-finite for no-division) | FAIL |
| one U element +2e-4 / one L element with max effect 2e-4 | FAIL (2.0e-4) | FAIL |
| the same with 5e-5 | PASS (5.0e-5 .. 5.1e-5) | PASS |
| one factor +inf | FAIL (non-finite) | FAIL |
| one factor NaN | FAIL (non-finite) | **PASS** (blind spot above) |
| in-band perturbations (above) | the exact verdict | the float32 verdict |

**Cost** (B200 node, 32 CPUs, OpenBLAS 0.3.34 with its default 32 threads; `OPENBLAS_NUM_THREADS` overrides)

| | s256 | s2048 | s8192 |
|---|---|---|---|
| upstream host check (offline only) | 7 ms | 9-14 s | 969 s alone, 1168-1436 s under load |
| `lud_residual_check.py` wall (interpreter + numpy import included) | 0.29-0.32 s | 0.32-0.38 s | 2.0-2.3 s (1 thread 15.5 s, 8 threads 3.5 s) |
| its peak RSS | 67 MB | 231 MB | 1.95 GB |
| capture in the run (write + sha256 of the factors) | within run-to-run noise | within run-to-run noise | ~0.3 s (256 MB) |
| the registered command `-s N -v`, wall, before -> after | 0.95-1.13 -> 0.86-1.06 s | 9.4-13.6 -> 0.95-1.44 s | 969 -> 1.1-1.3 s |
| registry `check` (run with capture + check; before: no check block) | 2.2-2.8 s | 2.3-3.4 s | 5.0-6.0 s |

ctest: 1.25-2.71 s before, 1.43-1.57 s after. The upstream stopwatch and the ROI are unchanged (`Time consumed`
0.85-0.97 / 9.4-10.2 / 156-164 ms with or without the capture, 0.87-0.97 / 9.4-10.1 / 156-163 ms for the previous
binary). `tools/inputs measure` and `tools/timing` never set `HPCPERF_GOLDEN_DUMP` and never run the check. The GPU
result is deterministic (factor sha256 identical across every run of the default build). No golden reference is
needed: the residual against the regenerated input decides correctness, and the factors themselves legitimately
differ between implementations (condition number 4.7e5 / 2.3e6).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `cuda/lud.cu`: lud_cuda (the blocked decomposition), not the copies inside the upstream stopwatch. Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 466 (4 source files, cloc, cuda/ + common/)
HIP: - (pending)
