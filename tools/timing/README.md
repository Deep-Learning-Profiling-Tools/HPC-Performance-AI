# tools/timing -- runtime measurement for Level 1

Measures how long each Level 1 benchmark takes and splits that time into GPU and
host work. Two scripts, no dependency on anything else in the repository:

```bash
tools/timing/measure_level1.sh --build-root build/all all   # measure (shell)
python3 tools/timing/summarize.py                           # one JSON per run + CSVs
```

Requires only `bash`, `python3` (standard library), `nsys` (ships with the CUDA
toolkit) and a built Level 1 tree. Results land in `results/timing/`, which is
git-ignored: **measurement output is never committed**.

## What is measured, and what it means

| field | meaning |
|---|---|
| `wall_s_median` (+ min/max/stddev, per-repeat array) | median of N un-instrumented runs -- no profiler attached, so no profiler overhead |
| `wall_s_profiled`, `profiling_overhead_ratio` | the one nsys run, and how much slower it was (measured: about 4x for sub-second benchmarks) |
| `gpu_kernel_time_s`, `gpu_kernel_launches`, `kernels[]` | per-kernel name, count, total/avg/min/max, from CUPTI device timestamps |
| `gpu_busy_s` | **union** of all GPU activity intervals (kernels + memcpy + memset). Concurrent operations are counted once |
| `gpu_op_time_sum_s`, `gpu_overlap_s` | the naive sum, and sum - union: how much GPU work overlapped |
| `gpu_active_span_s` | first GPU operation start to last GPU operation end |
| `gpu_idle_in_span_s` | span - busy: the GPU sat idle inside its own active window, i.e. the host was the bottleneck between launches |
| `host_outside_gpu_s` | wall - span: process start-up, CUDA context creation, allocation, input generation, teardown |
| `memops`, `cuda_api` | memcpy/memset time and bytes per direction; CUDA API time and the subset spent in synchronizing calls |

Everything GPU-side comes from `nsys stats` CSV reports (`cuda_gpu_kern_sum`,
`cuda_gpu_mem_time_sum`, `cuda_gpu_mem_size_sum`, `cuda_api_sum`,
`cuda_gpu_trace`). **The benchmarks' own timing printouts are never parsed**:
they disagree on units (s / ms / us / bandwidth-only), 17 of 50 print nothing,
and only one uses CUDA events, so they are neither comparable nor trustworthy.

A typical result shows why the decomposition matters: `daxpy` has a median wall
clock of 0.55 s but only 3.6 ms of GPU activity -- 99 % of the process is CUDA
initialization and input generation, not the kernel.

## Verification is skipped while measuring

Most Level 1 benchmarks validate by recomputing the entire GPU workload on one
CPU core afterwards. That is correctness machinery, not the workload being timed
(`channel_shuffle` spends 573 of its 574 ctest seconds in its CPU reference), so
`measure_level1.sh` sets `HPCPERF_SKIP_VERIFY=1` and each patched benchmark then
skips the host-side check and prints `SKIP_VERIFY`.

* **Default behaviour is unchanged.** No `add_test` command line was touched, so
  ctest never sets the variable and every correctness result stands. Verified per
  benchmark: `ctest` still passes with the variable unset.
* **The GPU workload is unchanged.** A/B on `daxpy`: wall clock 0.788 s -> 0.536 s
  while `gpu_busy_s` stayed 0.003640 s -> 0.003651 s and both modes ran 500
  kernels. That agreement is the evidence that skipping removes only host work.
  Use `--keep-verify` to measure the default path and compare for yourself.
* The switch elides only a host-side check: no kernel, data initialization,
  tolerance or algorithm is affected.

Cases where skipping is deliberately partial, and why:

| benchmark | what still runs | reason |
|---|---|---|
| `spmv` | the gold loop and one extra matvec | the gold loop also writes the matrix values that are uploaded to the GPU, and the extra matvec is GPU work -- removing either would change what is measured |
| `murmurhash3` | the key-generation loop | the reference hash sits inside the loop that fills the keys the GPU consumes; only the hash call is skipped |
| `block_scan` | `Initialize()` | it fills both the GPU input and the reference; only the two comparisons are skipped |
| `is` | its verification | `is` verifies inside the timed ranking kernels (`rank_gpu_kernel_7`) and in three further CUDA kernels; separating it would mean editing upstream kernels |
| `cg`, `ep`, `ft`, `mg` | their verification | an O(1) comparison against hardcoded reference constants -- it recomputes nothing, so skipping it would not change any measured time |
| `binary_search` | nothing to skip | its check is behind `#ifdef DEBUG`, which is never defined |
| 8 python-wrapped benchmarks | nothing to skip | verification lives in `verify.py`; the harness runs the binary directly, so it is never executed |

Every JSON therefore records what actually happened, not just what was requested:
`measurement.verify_kind` (`cpu-recompute` / `cpu-reference-cheap` /
`external-python` / `none`), `measurement.verify_skip_effect` (`skipped` /
`partially_skipped` / `not_skippable` / `not_executed` / `not_applicable` /
`not_requested`) and `measurement.verify_note`. Whenever the variable was set but
verification was only partly removed or not removable, a `caveats` entry says so
in words -- `cg`, `ep`, `ft`, `mg`, `is`, `spmv`, `murmurhash3` and `block_scan`
each carry one. Do not read their `host_outside_gpu_s` as verification-free. Both
fields are columns in `summary.csv`, so a model can filter on them.

## The case table

`cases.tsv` holds one row per benchmark: executable, arguments, working
directory, ctest timeout, wrapper kind and pass regex. It is generated from the
build tree, which is the source of truth:

```bash
python3 tools/timing/gen_cases.py --build-root build/all           # regenerate
python3 tools/timing/gen_cases.py --build-root build/all --check   # fail on drift
```

`gen_cases.py` does not trust a bare `ctest` from `PATH`. It resolves one in this
order and prints which it used: `CMAKE_CTEST_COMMAND` from the build tree's own
`CMakeCache.txt` (authoritative -- by construction the ctest that wrote the
`CTestTestfile.cmake` files being parsed, and independent of a conda/uv/system
choice), then `$HPCPERF_CTEST`, then `PATH` but only if `ctest --version` actually
succeeds; otherwise it fails naming every attempt. On the reference node
`~/.local/bin/ctest` is a pip shim whose `cmake` module is missing and dies with
`ModuleNotFoundError`, and a `PATH` search can reach an unrelated user's
virtualenv, so this is not hypothetical.

Nine benchmarks are wrapped by a repo-authored `verify.py` in their `add_test`
command; four of those rewrite the arguments (`gaussian_elimination` turns a
positional path into `-f <path>`, both `hotspot` variants append `output.out`,
`nearest_neighbor` turns `5 30 90` into `-r 5 -lat 30 -lng 90`) and `ao_bench`
runs the binary twice. `gen_cases.py` therefore obtains the inner argv by
importing each wrapper with `subprocess.run` monkey-patched, so the wrapper does
its own argument transformation and nothing is guessed. The wrappers are not
modified. Paths are stored as `{REPO}` / `{BUILD}` placeholders; `-` means an
empty field (bash's `IFS=$'\t' read` collapses runs of tabs, which would shift
every later column).

## Measurement protocol

1. one warm-up run, discarded (first touch pays CUDA context creation);
2. `--repeats` (default 5) un-instrumented runs, wall clock each;
3. one `nsys profile` run, then `nsys stats` CSV reports.

Absolutes come from step 2 and ratios from step 3 on purpose: 29 of 50
benchmarks finish in under 1.6 s while nsys attach/flush costs about a second,
so a single profiled run would distort the headline number. GPU-side durations
are timestamped on the device by CUPTI and are far less sensitive. Both wall
clocks are reported, so the perturbation is visible rather than assumed.

Raw artifacts (`.nsys-rep`, the CSVs, per-run stdout, `run_meta.txt`) stay in
`build/timing/<benchmark>/<run-id>/`, inside the git-ignored build tree.

### The benchmark runs under a minimal environment

`measure_level1.sh` launches everything through `env -i` with an explicit
allow-list (`PATH`, `HOME`, `TMPDIR`, `LD_LIBRARY_PATH`, `CUDA_*`,
`HPCPERF_SKIP_VERIFY`, ...). This is not cosmetic: **nsys stores the complete
process environment in the report.** Measured on this node, a default `nsys
profile` of `daxpy` recorded 368 variables and 16 366 characters in
`TARGET_INFO_SYSTEM_ENV` under the property name `DeviceEnvironment`, including
`CLAUDE_CODE_MESSAGING_TOKEN`, `SSH_CONNECTION` and `SSH_CLIENT`. The allow-list
keeps that out of the report and makes the measurement environment reproducible.

## Output

* `results/timing/level1/<benchmark>/<run-id>.json` -- schema `hpcperf-timing-1`:
  timing, GPU/host decomposition, per-kernel table, memcpy/API totals, plus the
  labels a model needs (GPU name/UUID/driver/compute capability, SM and memory
  clocks at start, power limit, temperature, CPU model, allowed cores, load
  average, nsys and nvcc versions, executable sha256, git commit and dirty flag,
  arguments, `skip_verify`, exit code).
* `results/timing/summary.csv` -- one row per run, fixed column list.
* `results/timing/kernels.csv` -- one row per (run, kernel).

Both CSVs are **regenerated, never appended**, so a rerun cannot produce
duplicate or reordered columns. `summarize.py` prints
`json_written= failed= rows= skipped=` and treats a schema mismatch as an error.

## Tests

```bash
bash tools/timing/tests/run_all.sh
```

CPU only, no GPU, no build required (groups that need a build tree skip
cleanly). Covers the case table's shape and completeness, argument validation
and the constructed command (including a planted-credential check asserting the
allow-list holds), the interval-union algorithm on eight interval shapes
(disjoint / overlapping / contained / adjacent / unsorted / identical / single /
empty), rejection of NaN and of missing CSV columns, and that nothing enables
skip-verify by default.

## Scope

CUDA only; `nsys` only. Not covered: HIP/ROCm (absent on this machine),
`ncu` hardware counters (occupancy, achieved bandwidth, cache hit rates -- they
replay every launch and need their own budget), Level 2 and Level 3, and
multi-GPU (every Level 1 benchmark is single-GPU, single-process).

## First recorded sweep

`build/gcc13` (GCC 13.3.0 from source, CUDA 13.2.78, nsys 2025.6.3), 1x NVIDIA
B200 sm_100 (driver 595.58.03), 16 allocated CPUs, `--repeats 5`, verification
skipped, 2026-09-21:

* 50 of 50 benchmarks measured, 0 failures; 50 JSON files, `summarize: failed=0`.
* The whole sweep (50 x [1 warm-up + 5 timed + 1 nsys]) takes **111 s**. The same
  50 cases under ctest with verification take about **815 s**, of which
  `channel_shuffle` alone is 573.9 s.
* Slowest by wall clock: `block_scan` 57.8 s (97.7 % GPU), `channel_shuffle`
  9.5 s (72.4 %), `bilateral_filter` 4.8 s (86.8 %), `cg` 3.4 s (5.5 %).
* About 30 benchmarks spend under 5 % of their wall clock on the GPU: a roughly
  constant 0.4-0.6 s of CUDA context creation and input generation dominates
  them. That is why `host_outside_gpu_s` and `gpu_busy_frac_of_wall` are
  first-class fields rather than derived afterthoughts.
* `gpu_overlap_s` is 0 for all 50: every benchmark is single-stream and serial,
  so no GPU operations overlap. The interval union therefore equals the naive sum
  here -- it is kept because a multi-stream benchmark would otherwise report more
  GPU time than wall clock.
* Largest host-bound gaps inside the GPU active window (`gpu_idle_in_span_s`):
  `channel_shuffle` 2.27 s, `background_subtraction` 1.28 s -- host-side loops
  between launches.

Numbers live in `results/timing/` and are not committed; rerun to regenerate.

## Cross-validation against the benchmarks' own instrumentation

The tool deliberately ignores what the benchmarks print, so its numbers were
checked against two independent, already-present instrumentations. Both are off
by default and enabled only for this check.

### NPB per-kernel table (`mg`, 2026-09-21)

`cmake -DHPCPERF_NPB_PROFILING=ON` compiles the per-kernel seconds+percentage
table upstream already ships behind `#if defined(PROFILING)`. Running the same
binary with and without nsys:

| kernel | launches | app (s) | nsys (s) | difference per timed region |
|---|---|---|---|---|
| comm3 | 1383 (461 regions x 3 kernels) | 0.006368 | 0.002684 | 7.99 us |
| interp | 140 | 0.002655 | 0.001631 | 7.32 us |
| psinv | 160 | 0.004807 | 0.003630 | 7.36 us |
| resid | 161 | 0.008159 | 0.006789 | 8.51 us |
| rprj3 | 140 | 0.002083 | 0.001078 | 7.18 us |
| zero3 | 140 | 0.001367 | 0.000341 | 7.33 us |
| norm2u3 | 2 | 0.001110 | 0.000142 | 484 us (see below) |

* **Names and counts agree completely.** All seven instrumented regions map onto
  the nine kernels nsys reports (`comm3` is three kernels per region, the rest are
  1:1) with nothing unmatched on either side. That is the check that the
  `cuda_gpu_kern_sum` parsing is right.
* **The systematic gap is explained by one constant.** Six of seven rows differ by
  7.2-8.5 us per timed region, which is the CUDA launch plus synchronization cost
  on this node: the app's `timer_start`/`timer_stop` bracket host code around the
  launch, nsys timestamps device execution. Short kernels are dominated by it
  (`zero3`: nsys is 25 % of the app number), long ones are barely affected
  (`resid`: 83 %).
* **The one outlier is real host work, not an artifact.** `norm2u3`'s timed region
  also contains `cudaMemcpy(..., cudaMemcpyDeviceToHost)` plus a host reduction
  loop and a `sqrt` (mg.cu:1288-1301), so its 484 us per call is host time that
  nsys correctly attributes outside the kernel.

### Hetero-Mark six-phase timer (`aes`, `black_scholes`, `color_histogram`, `fir`, `pagerank`)

`HPCPERF_L1_PHASE_TIMING=1` prints the `Initialize / WarmUp / Run / Verify /
Summarize / Cleanup` summary the upstream `BenchmarkRunner` already computes (to
stderr). These benchmarks allocate and upload in `Initialize()`, so the GPU active
span is not contained in `WarmUp + Run` and the two cannot be compared directly.
The checkable invariant is that kernel time must fit inside the two phases that
each run the workload once:

| benchmark | kernel total (s) | WarmUp + Run (s) | kernel share |
|---|---|---|---|
| aes | 0.000022 | 0.000623 | 3.5 % |
| black_scholes | 0.001837 | 0.013257 | 13.9 % |
| color_histogram | 0.000128 | 0.001077 | 11.9 % |
| fir | 0.005754 | 0.066230 | 8.7 % |
| pagerank | 0.004979 | 0.006059 | 82.2 % |

It holds for all five, and `pagerank` shows the bound is tight where the
benchmark is GPU-bound rather than vacuous.

These same runs also quantify why the benchmarks' own "GPU time" is not GPU time:
the `CPUGPUActivityLogger` in this family reports `GPU: 0.000198` for `aes` where
nsys measures `gpu_busy_s = 0.000117`, a factor of 4.4, because the logger
brackets host code around the launch and the blocking copy.
