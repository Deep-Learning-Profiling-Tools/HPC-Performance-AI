# tools/timing -- runtime measurement for Level 1 and Level 2

Measures how long each benchmark takes and splits that time into GPU and host
work. One shell script per level plus one python aggregator, with no dependency
on anything else in the repository:

```bash
tools/timing/measure_level1.sh --build-root build/all all   # Level 1 (shell)
tools/timing/measure_level2.sh all                          # Level 2 (shell)
python3 tools/timing/summarize.py                           # one JSON per run + CSVs
```

Requires only `bash`, `python3` (standard library), `nsys` (ships with the CUDA
toolkit) and a built tree. Results land in `results/timing/`, which is
git-ignored: **measurement output is never committed**.

The two levels are measured differently and their outputs are kept in separate
files so they cannot be averaged together by accident:

| | Level 1 | Level 2 |
|---|---|---|
| unit | 50 benchmark binaries | 24 `level2/<app>/run.sh` |
| runs per case | N clean (default 5) + 1 profiled | **1 profiled, no repeats** |
| wall clock | from the clean runs, profiler-free | from the profiled run, **profiler included** |
| verification | skipped in-binary (`HPCPERF_SKIP_VERIFY`, 41 source edits) | **not run at all** (it lives in `validate.sh`; zero source edits) |
| application's own metric | deliberately not collected | **FOM collected where it exists** (16 of 24) |

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
* `results/timing/summary.csv` -- one row per Level 1 run, fixed column list.
* `results/timing/kernels.csv` -- one row per (Level 1 run, kernel).
* `results/timing/level2/<app>/<run-id>.json` -- same schema, plus a `fom` block
  (`name`, `value`, `unit`, `better`, `status`, `from_profiled_run`) and a
  `launcher` block holding the GPU-binding audit line verbatim.
* `results/timing/summary_level2.csv`, `results/timing/kernels_level2.csv`.

Both CSVs are **regenerated, never appended**, so a rerun cannot produce
duplicate or reordered columns. `summarize.py` prints
`json_written= failed= rows= skipped=` and treats a schema mismatch as an error.

## Level 2

```bash
tools/timing/measure_level2.sh all            # 24 applications, one profiled run each
tools/timing/measure_level2.sh xsbench        # one application
tools/timing/measure_level2.sh --dry-run all  # print the commands, run nothing
```

`--env-script` (or `HPCPERF_TIMING_ENV_SCRIPT`) selects the environment loader
sourced inside the clean environment; it defaults to `hpcperf_env.sh`.

### No source change was needed

Level 1 keeps correctness checks inside the binary, which is why measuring it
required 41 source edits and an `HPCPERF_SKIP_VERIFY` switch. Level 2 separates
`run.sh` from `validate.sh`. All 24 `run.sh` were checked: every hit for
verification vocabulary is a comment or an error message, not a check, while
`validate.sh` carries 4-27 hits each. This tool never calls `validate.sh`, so
nothing is skipped inside the application and no patch exists.

### nsys wraps run.sh from the outside

```
nsys profile ... -- bash level2/<app>/run.sh CUDA
```

Two reasons it has to be outside rather than inside:

* 20 of the 24 `run.sh` end in `exec`, so there is no post-run hook to attach to.
* A profiler placed inside the launcher's wrapper breaks the GPU-binding audit:
  `mpi_gpu_bind.sh` logs `pid=$$` and `hpcperf_mpi_launch.sh` joins that pid
  against `nvidia-smi --query-compute-apps`. nsys would own that pid while the
  application got another, making every rank `unverified`.

Wrapping from the outside keeps the audit intact and still follows the process
tree. Verified on quicksilver (`bash` -> `mpirun` -> `mpi_gpu_bind.sh` -> exe):
audit `1 verified, 0 mismatch, 0 unverified` with nsys as an ancestor, and 410
`CycleTrackingKernel` launches captured. Each JSON stores the audit line verbatim
in `launcher.gpu_audit` and a parsed `audit_ok`; a run whose audit is not clean
gets a caveat. Applications that do not use the launcher (xsbench, minibude, ...)
leave `audit_ok` empty rather than claiming a pass.

### One profiled run, and what that costs

The protocol is a single profiled run per application: no warm-up, no repeats,
no clean baseline. The consequences are measured, and recorded in every JSON
rather than hidden. Measured on quicksilver at `HPCPERF_GPUS=1`:

| | wall clock | kernel total | launches | FOM | report |
|---|---|---|---|---|---|
| no profiler | 9.29 s | -- | -- | 5.631e6 | -- |
| `-s none` | 15.38 s | 3.6839 s | 411 | 5.327e6 | 3.2 MB |
| `-s process-tree` | 16.28 s | 3.6745 s | 411 | 5.271e6 | 7.2 MB |
| `+ --cpuctxsw --cuda-memory-usage` | 17.72 s | 3.6406 s | 411 | 5.329e6 | 7.2 MB |

Three things follow, and they are why the flags are what they are:

* **Wall clock is an upper bound, but the application is not slowed by 1.7x.**
  That ratio is the nsys *command's* wall clock. Read quicksilver's own timers for
  the same regions and the application barely moves -- most of the cost sits
  outside it, in nsys attach and post-run report writing (see below).
  `host_outside_gpu_s` is therefore NOT application host time.
* **GPU-side numbers are trustworthy.** CUPTI timestamps kernels on the device.
  Across the three flag sets the kernel total moved 1.2% with an identical launch
  count, so the full sampling set is taken: it is nearly free relative to the
  attach cost already being paid.
* **The application's own FOM is depressed by 5.4%-6.4%.** It is read from the
  same profiled run, so every record carries `fom.from_profiled_run = true` and a
  caveat. Do not compare these values against published unprofiled FOMs without
  that correction.

### Where the profiler overhead actually lands

The 1.66x-1.91x above is the wall clock of the `nsys` command, and reading it as
"the application ran 1.7x slower" is wrong. quicksilver's own timer table for the
same regions, unprofiled vs profiled:

| region | no profiler | `-s none` | `process-tree` | `+ctxsw` |
|---|---|---|---|---|
| `main` | 7.431 s | 7.960 | 8.091 | 8.007 |
| `cycleTracking` | 6.598 s | 6.975 | 7.049 | 6.972 |
| `cycleTracking_Kernel` | 4.302 s | 3.832 | 3.825 | 3.798 |
| `cycleTracking_MPI` | 2.295 s | 3.142 | 3.223 | 3.173 |

The application's `main` grew **8.9%**, not 70%. (The kernel region got shorter and
the MPI region longer -- profiling shifts where asynchronous waits are charged, so
the split between those two is not comparable across modes; their sum is.)
Subtracting `main` from the command's wall clock shows where the rest went:

| | command wall | app `main` | outside `main` |
|---|---|---|---|
| no profiler | 9.29 s | 7.431 | 1.86 s (start-up, `MPI_Init`, launcher) |
| `-s none` | 15.38 s | 7.960 | **7.42 s** |
| `process-tree` | 16.28 s | 8.091 | **8.19 s** |
| `+ctxsw --cuda-memory-usage` | 17.72 s | 8.007 | **9.71 s** |

That outside-`main` time is nsys attaching and then writing the report, and it
grows with how much was collected. Measured against an unprofiled reference run for
13 applications, the overhead separates into a fixed part and a per-API-call part:

| app | CUDA API calls | unprofiled s | profiled s | increment | us per API call |
|---|---|---|---|---|---|
| `minibude` | 756 | 5.60 | 8.46 | +2.86 | 3781.5 |
| `sw4lite` | 1,930 | 8.20 | 13.96 | +5.76 | 2985.3 |
| `kripke` | 2,511 | 3.20 | 9.43 | +6.23 | 2481.6 |
| `quicksilver` | 2,665 | 9.29 | 16.18 | +6.89 | 2584.3 |
| `p3_vlp4d` | 5,880 | 7.80 | 10.99 | +3.19 | 542.1 |
| `gamess_ri_mp2` | 10,043 | 4.20 | 9.45 | +5.25 | 523.1 |
| `exacmech` | 29,072 | 8.90 | 12.09 | +3.19 | 109.8 |
| `comb` | 138,438 | 2.10 | 8.85 | +6.75 | 48.8 |
| `exampm` | 249,721 | 11.70 | 17.52 | +5.82 | 23.3 |
| `miniweather` | 1,475,106 | 33.90 | 51.50 | +17.60 | 11.9 |
| `haccabanapm` | 1,534,132 | 20.60 | 40.97 | +20.37 | 13.3 |
| `cabanapic` | 4,512,104 | 62.00 | 91.26 | +29.26 | 6.5 |
| `miniem` | 15,165,911 | 110.09 | 235.10 | +125.01 | 8.2 |

Two regimes, and the *ratio* is the wrong thing to look at in either:

* **Under ~20k API calls the increment is a flat 2.9-6.9 s** regardless of anything
  -- attach plus report writing. `comb` looks like a 4.2x slowdown only because it
  runs for 2.1 s; its absolute cost is the same 6.75 s as everyone else's.
* **Above ~1M API calls the per-call interception dominates and converges to
  6.5-13 us per call.** This explains `miniem`, the worst case in the suite: 15.2
  million CUDA API calls (a Trilinos/Kokkos stack, not a high kernel count -- it has
  fewer launches than `cabanapic`, which has 17x more calls per launch) cost +125 s.

Rule of thumb: **~5 s + ~10 us per CUDA API call**, which fits these 13 to within
about 40%. `cuda_api_calls` is in the CSV, so the estimate is reproducible per run,
and each JSON's caveat states it. It is an order of magnitude, not a correction to
subtract: for an honest wall clock and an undistorted FOM, measure without the
profiler.

### The figure of merit, where there is one

A FOM is the application's own normalized throughput -- the number the upstream
proxy-app suites quote. It is a fourth quantity, not a substitute for anything
this tool measures. On the quicksilver run above: wall clock 9.29 s (with
start-up and `MPI_Init`), the FOM's own denominator `cycleTracking` 6.598 s (71%
of the wall clock, and it contains 2.295 s of MPI), and nsys kernel time ~3.67 s.

The wording, unit and position differ per application, so each pattern lives in
`cases_l2.tsv` with exactly one capture group and the last match wins. **16 of
24 applications report one; the other 8 do not and their `fom_value` is left
empty** -- never a fabricated or externally derived number. The `notes` column
records what those 8 do print, so the information is not lost.

| Metric | Applications |
|---|---|
| explicitly labelled a FOM (7) | `kripke` (a whole `Figures of Merit` section), `amg2023`, `quicksilver`, `miniem`, `branson`, `hipbone`, `remhos` |
| a rate, not labelled FOM (6) | `minibude` (GFLOP/s), `xsbench` (Lookups/s), `laghos` (megadofs x cg_iterations/s), `p3_heat3d` (GB/s), `examinimd` (Atomsteps/s), `shaw` (aveBandwidth GB/s) |
| grind time, lower is better (3) | `cloverleaf` and `tealeaf` (`time per cell`), `haccabanapm` (upstream's per-step median with a MAD) |
| **none -- left blank (8)** | `exacmech`, `sw4lite`, `miniweather`, `p3_vlp4d`, `gamess_ri_mp2`, `comb`, `cabanapic`, `exampm` |

Three traps found while building that table, all of which a keyword search gets
wrong:

* `shaw` prints `*** Compute FOM Jacobian matrices ***`. That "FOM" is a domain
  term, **not** a figure of merit; its real metric is `aveBandwidth(GB/s)`.
* `cloverleaf` and `tealeaf` do have a metric, worded `per cell` rather than
  `per second`, so a search for rate vocabulary misses them.
* `comb` prints no timing at all on stdout (74 lines, not one decimal): its
  per-phase table goes to `Comb_NN_summary.csv` in the run directory. It is left
  blank here because that table is times, not a rate.
* `remhos` prints per-stage `FOM RHS:` / `FOM LO:` lines as well as the final
  `FOM:`, so its pattern is anchored with `^FOM:`.

### Case table

`cases_l2.tsv`, ten tab-separated columns, `-` for an empty field (a bare empty
field would be swallowed -- tab is an IFS whitespace character, so runs of tabs
collapse in `IFS=$'\t' read`). Level 2 has no ctest to generate from, so the
table is written by hand: `app`, `backend`, `gpus`, `timeout_s`, `fom_name`,
`fom_unit`, `fom_better`, `fom_source`, `fom_regex`, `notes`. The tests assert
one row per `level2/<app>/run.sh` and no row without one, so an added application
cannot be forgotten.

Per-application timeouts, not repeats, are what the table tunes: the default
cases run from about 2 s (`amg2023`) to 16 min (`laghos`) and 12-30 min
(`miniem`) before profiler overhead.

### The clean environment, and why it is mandatory

nsys stores the complete process environment in the report
(`TARGET_INFO_SYSTEM_ENV`, row `DeviceEnvironment`; visible through
`nsys export --type sqlite`, not through `nsys stats`). Measured on this node:

| | variables | chars | `CLAUDE_CODE*` | `SSH_*` | `SLURM_*` |
|---|---|---|---|---|---|
| inheriting the login environment | 349 | 13187 | 9 | 2 | 37 |
| `env -i` allow-list | 101 | 8390 | **0** | **0** | 1 |

Level 2 applications need a full build environment, so a bare allow-list is not
enough: the script starts from `env -i` with the allow-list and then sources the
repository's environment script *inside* that clean environment to rebuild what
the applications need. Use `bash -c`, never `bash -lc`: a login shell re-reads
the site profile and puts back what was just dropped.

The allow-list includes eleven `SLURM_*` names because `hpcperf_mpi_launch.sh`
reads the allocation through them (and through `scontrol -d $SLURM_JOB_ID`); they
describe the allocation and cannot be rebuilt by sourcing anything. On top of the
allow-list there is a **deny rule that beats it** (`TOKEN|SECRET|PASSWD|
PASSWORD|CREDENTIAL|PRIVATE_KEY|API_KEY|SESSION`): a name matching it is dropped
even if it is allow-listed, and the dropped names are recorded in
`measurement.env_denied`. A planted-credential assertion in the tests covers it.

## Tests

```bash
bash tools/timing/tests/run_all.sh
```

33 assertions in six groups, CPU only: no GPU, no build tree and no `nsys`
required (groups that need something absent skip cleanly). Covers both case
tables' shape and completeness, argument validation and the constructed commands
(including a planted-credential check asserting the allow-list and the deny rule
hold), the interval-union algorithm on eight interval shapes (disjoint /
overlapping / contained / adjacent / unsorted / identical / single / empty),
rejection of NaN and of missing CSV columns, that nothing enables skip-verify by
default, that all 16 Level 2 FOM patterns compile with exactly one capture group
and declare a direction, and that FOM extraction handles thousands separators,
last-match-wins, a blank case and a pattern that misses.

## Scope

CUDA only; `nsys` only. Not covered:

* **HIP/ROCm** -- absent on this machine, so every HIP path is UNVERIFIED.
* **`ncu` hardware counters** (occupancy, achieved bandwidth, cache hit rates) --
  they replay every launch and need their own time budget.
* **Level 3.**
* **Multi-GPU / multi-rank.** Every Level 1 benchmark is single-GPU and
  single-process. Level 2 is measured at `HPCPERF_GPUS=1` because this
  allocation exposes one B200 (`CUDA_VISIBLE_DEVICES=0`); the multi-rank path is
  designed (see below) but **UNVERIFIED here**.

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

## First recorded Level 2 sweep

dgx003, 1x NVIDIA B200 (`CUDA_VISIBLE_DEVICES=0`), CUDA 13.2.78, nsys 2025.6.3,
`HPCPERF_GPUS=1`, default `run.sh` case per application, one profiled run each,
2026-09-22. **24/24 completed.** Wall clock includes profiler overhead; GPU-side
numbers do not.

| app | wall s | gpu_busy s | busy/wall | host s | launches | FOM | unit | audit |
|---|---|---|---|---|---|---|---|---|
| `amg2023` | 10.76 | 1.303 | 12.1% | 9.28 | 5,172 | 1.502e+09 | nnz_AP/s | clean |
| `branson` | 19.43 | 4.243 | 21.8% | 9.07 | 85,448 | 1.334e+06 | photons/s | clean |
| `cabanapic` | 91.26 | 57.366 | 62.9% | 29.62 | 2,016,006 | -- | -- | n/a |
| `cloverleaf` | 32.30 | 17.082 | 52.9% | 13.70 | 370,025 | 4.266e-10 | s/cell | clean |
| `comb` | 8.85 | 0.188 | 2.1% | 7.73 | 1,608 | -- | -- | clean |
| `exacmech` | 12.09 | 7.951 | 65.7% | 4.02 | 10,007 | -- | -- | n/a |
| `examinimd` | 20.43 | 10.068 | 49.3% | 7.68 | 10,388 | 1.602e+09 | atom-steps/s | clean |
| `exampm` | 17.52 | 8.569 | 48.9% | 7.98 | 43,155 | -- | -- | clean |
| `gamess_ri_mp2` | 9.45 | 1.467 | 15.5% | 7.91 | 1,208 | -- | -- | clean |
| `haccabanapm` | 40.97 | 7.680 | 18.7% | 17.32 | 132,582 | 0.0306 | s/step | clean |
| `hipbone` | 26.41 | 3.478 | 13.2% | 8.60 | 9,916 | 3458 | GFLOPs | clean |
| `kripke` | 9.43 | 0.823 | 8.7% | 8.58 | 936 | 3.173e+09 | unknowns/(s/iteration) | clean |
| `laghos` | 394.93 | 95.586 | 24.2% | 222.25 | 14,536,114 | 1387 | megadofs*cg_iterations/s | clean |
| `minibude` | 8.46 | 4.906 | 58.0% | 3.54 | 152 | 335.7 | GFLOP/s | n/a |
| `miniem` | 235.10 | 6.755 | 2.9% | 103.73 | 263,737 | 1.815e+04 | k-cell-steps/s | clean |
| `miniweather` | 51.50 | 32.134 | 62.4% | 19.18 | 737,293 | -- | -- | clean |
| `p3_heat3d` | 8.65 | 1.794 | 20.7% | 3.69 | 1,007 | 1347 | GB/s | n/a |
| `p3_vlp4d` | 10.99 | 2.505 | 22.8% | 4.15 | 2,457 | -- | -- | n/a |
| `quicksilver` | 16.18 | 3.990 | 24.7% | 8.06 | 411 | 5.256e+06 | segments/s | clean |
| `remhos` | 33.31 | 5.881 | 17.7% | 9.88 | 67,568 | 184.1 |  | clean |
| `shaw` | 178.53 | 14.439 | 8.1% | 153.03 | 120,002 | 4.782e+04 | GB/s | n/a |
| `sw4lite` | 13.96 | 0.128 | 0.9% | 13.70 | 576 | -- | -- | clean |
| `tealeaf` | 39.56 | 14.133 | 35.7% | 18.61 | 634,778 | 1.309e-06 | s/cell | clean |
| `xsbench` | 6.67 | 0.617 | 9.2% | 6.05 | 3 | 4.2e+08 | lookups/s | n/a |

`audit` = the launcher's nvidia-smi GPU-binding check: `clean` is
"N verified, 0 mismatch, 0 unverified", `n/a` means the application does not use
the launcher (no MPI), so nothing is claimed. 17 clean, 7 n/a, **0 not clean** --
nsys as an ancestor process does not break the audit.

What the sweep shows, and how it differs from Level 1:

* **GPU busy fraction spans 0.9% to 65.7%**, a far wider spread than Level 1
  (where about 30 of 50 benchmarks sat under 5% because a roughly constant
  0.4-0.6 s of CUDA context creation dominated a sub-second run). Here the host
  time is real work: `sw4lite` spends 13.70 s of 13.96 s outside GPU activity and
  `shaw` 153.03 s of 178.53 s, both set-up dominated; `miniem` is at 2.9% busy
  across 235 s. At the other end `exacmech` (65.7%), `cabanapic` (62.9%) and
  `miniweather` (62.4%) are genuinely GPU-resident.
* **Launch counts span 3 to 14.5 million** (`xsbench` 3, `laghos` 14,536,114).
  Anything reading per-launch overhead should start there.
* `gpu_overlap_s` is 0 everywhere, as at Level 1: these runs are single-stream.

### `--cuda-memory-usage` segfaults MiniEM (why the flag sets differ)

`measure_level1.sh` passes `--cuda-memory-usage=true`; `measure_level2.sh` does
not. MiniEM under that option dies with SIGSEGV inside
`cudaFreeAsync` -> `cuMemFreeAsync` (exit 139 at 203 s, before it prints its FOM).
Isolated by bisecting the flags on the same case:

| configuration | result |
|---|---|
| no profiler at all | rc 0, 110 s |
| `-s process-tree --cpuctxsw`, no `--cuda-memory-usage` | rc 0, 236 s, FOM 18277.2 |
| the same **plus** `--cuda-memory-usage=true` | **rc 139**, crash at 203 s, no FOM |

Dropping it costs nothing: with and without the option, all four reports this
tool reads (`cuda_gpu_kern_sum`, `cuda_gpu_mem_time_sum`, `cuda_gpu_mem_size_sum`,
`cuda_gpu_trace`) come back with identical shape and equivalent values (checked on
xsbench; memcpy totals differed by 1.8%, i.e. run-to-run noise). The option only
adds memory-pool allocation events that are never read. Level 1 keeps it because
its recorded sweep was taken with it and no Level 1 benchmark is affected.

### Values are per default case, and two labels need care

The FOM belongs to the case `run.sh` runs, not to the application in general.
`hipbone` reports 3458.3 GFLOPs here and 146.1 in an older `validate` log;
`remhos` 184.08 here and 10.29 there. Neither is wrong -- they are different
problem sizes. Always read a FOM together with the case.

`shaw`'s `aveBandwidth(GB/s)` is 4.782e4 on the default case, well above this
device's HBM peak, so it is an effective figure derived from an operation count
rather than achieved DRAM traffic. It is recorded as the application prints it
and flagged in `cases_l2.tsv`, not silently rescaled.

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
