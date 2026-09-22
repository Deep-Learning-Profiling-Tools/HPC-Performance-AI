# tools/timing -- runtime measurement for Level 1 and Level 2

Measures **how long the computation of each benchmark takes** and what the device
did during it, in a form that stays comparable when applications, inputs and
hardware platforms are added.

```bash
tools/timing/measure_level1.sh --build-root build/gcc13 all      # Level 1: 50 benchmarks
tools/timing/measure_level2.sh all                               # Level 2: 24 applications
python3 tools/timing/summarize.py                                # JSON per run + CSV per level
bash tools/timing/tests/run_all.sh                               # self-tests, CPU only
```

Requires `bash`, `python3` (standard library only) and a built tree; a profiler
is optional (NVIDIA: `nsys`, shipped with the CUDA toolkit). Results land in
`results/timing/` and raw evidence in `build/timing/`, both git-ignored:
**measurement output is never committed.** Formats: [SCHEMA.md](SCHEMA.md).

## What is measured: the region of interest

A process's wall clock is not the computation. It holds CUDA context creation,
input generation, warm-up iterations and verification -- measured on Level 1,
`daxpy` spends 99% of its process time outside its kernels and `channel_shuffle`
573 of 574 ctest seconds in its CPU reference. So every benchmark marks its
**region of interest (ROI)** in its source, and everything here measures between
those marks:

```c
HPCPERF_ROI_BEGIN_SYNC();            /* after set-up and warm-up */
... the time loop / the solve / the timed repetitions ...
    HPCPERF_ROI_EXCLUDE_BEGIN_SYNC(); write_snapshot(); HPCPERF_ROI_EXCLUDE_END();
HPCPERF_ROI_END_SYNC();              /* before verification and final output */
```

Inside the ROI: every step's work, including per-step copies, halo exchanges and
per-step scalar diagnostics. Outside or excluded: start-up, input set-up and its
one-time upload, warm-up, verification, bulk output. The markers are a no-op
unless measuring, so builds, ctest and `validate.sh` are unchanged. The API
(C/C++, Fortran, Python), the placement rule and how to onboard an application:
[roi/README.md](roi/README.md). All 50 Level 1 benchmarks (CUDA and HIP sources)
and all 24 Level 2 applications carry markers; where each ROI sits is recorded at
the markers, in each Level 2 README and in `cases/level*_apps.tsv`.

One set of markers serves two measurements:

| run | profiler | gives |
|---|---|---|
| **clean** | none | the markers log their own timestamps (`HPCPERF_ROI_LOG`) -> **`roi_wall_s`**, the headline; the application's FOM; the launcher audit |
| **profiled** | yes | device activity **clipped to the same markers** -> busy time, per-category time, ops, runtime API calls |

| | Level 1 | Level 2 |
|---|---|---|
| unit | 51 cases of 50 benchmark binaries | 28 cases of 24 `level2/<app>/run.sh` |
| runs per case | 1 warm-up + 5 clean + 1 profiled | 1 clean + 1 profiled |
| verification | outside the ROI; `HPCPERF_SKIP_VERIFY=1` also skips the CPU reference (minutes for some) | outside the ROI; `validate.sh` is never called |
| FOM | none (the benchmarks' own printouts are not comparable) | the application's own metric where it prints one (16 of 24) |

The headline therefore carries **no profiler overhead**: the profiler's cost shows
up only as `roi_profiler_inflation` (profiled ROI / clean ROI), and device-side
durations come from device timestamps, which it barely perturbs.

## Headline fields (`summary_level<N>.csv`)

| field | meaning |
|---|---|
| `roi_wall_s` (+ `_min`, `_max`, `_stddev`, `roi_runs`) | median ROI time of the clean runs |
| `roi_entries`, `roi_excluded_s` | how often the ROI was entered; time carved out by excludes |
| `roi_profiled_wall_s`, `roi_profiler_inflation` | the same region in the profiled run, and the ratio |
| `device_busy_s`, `device_busy_frac_of_roi` | union of all device activity inside the ROI -- concurrent operations count once |
| `host_gap_s`, `host_gap_frac_of_roi` | `roi_wall_s - device_busy_s`: time in the ROI when the device was idle, i.e. the host was the bottleneck |
| `device_<category>_s`, `device_*_ops`, `device_copy_*_bytes` | per category inside the ROI: `compute`, `copy_h2d`, `copy_d2h`, `copy_d2d`, `copy_other`, `fill`, `collective`, `other` |
| `device_op_time_sum_s`, `device_overlap_s` | naive sum of op durations, and sum - union (concurrency) |
| `runtime_api_calls`, `runtime_api_sync_calls`, ... | host runtime API calls inside the ROI |
| `top_op_*`, `ops_level<N>.csv` | the operations inside the ROI by total time |
| `process_wall_s`, `pre_roi_s`, `post_roi_s`, `whole_*` | context: the whole process -- never the headline |
| `fom_*` | the application's own metric, from the clean run |
| `device_*`, `driver_version`, `runtime_version`, `host_*` | the platform descriptor |
| `collector`, `conformance` | which profiler adapter produced the device columns, and whether the platform passed the conformance probe |

A device column is **empty when the platform cannot observe it** (no collector, or
a category outside the collector's capabilities) -- never 0. A 0 means observed
and absent. Every limitation of a record is spelled out in its JSON `caveats`.

## Cases: many inputs per application, new applications

A measurement's identity is `(level, app, case, platform)`. Cases live in
`cases/`, tab-separated, `-` for an empty field (bash `read` collapses runs of
tabs); `cases.py` resolves them into the rows the engine runs:

| table | what |
|---|---|
| `level1.tsv` | generated from ctest by `gen_cases.py` -- one case per ctest test (`default` when a benchmark has one) |
| `level1_extra.tsv` | Level 1 inputs ctest does not know (e.g. `all_pairs_distance/n20000`) |
| `level1_apps.tsv` | per benchmark: suite, backends, `roi_excludes`, `verify_vs_roi` |
| `level2_apps.tsv` | per application: backends, timeout, FOM pattern, `roi_excludes` |
| `level2_cases.tsv` | Level 2 cases: GPUs, input variables, arguments, timeout, FOM override |

* **Inputs are explicit.** A case sets only variables its `run.sh` reads
  (`cases.py allowed-env <app>` lists them). A variable the application reads that
  is set in *your* shell but not declared by the case is **refused**: every run
  starts from `env -i`, so it would be dropped and the default input measured
  under the wrong name.
* **Sweeps.** `HPCPERF_AMG_N=128|192` with case name `n{}` becomes `n128`, `n192`.
* **Selection.** `all`, `<app>`, or `<app>/<case>`.
* **A new application** needs markers (roi/README.md), an `*_apps.tsv` row and a
  case row; `cases.py check` and the tests fail until all three exist. A clean run
  that writes no ROI record is `roi_missing` -- a failure, never a silent fallback
  to the process wall clock.

`gen_cases.py` takes each benchmark's exact command, working directory and
timeout from `ctest --show-only=json-v1` of the build tree (`--check` fails on
drift). It trusts `CMAKE_CTEST_COMMAND` from the build tree's `CMakeCache.txt`
first, then `$HPCPERF_CTEST`, then a `PATH` ctest only if `ctest --version` works
(on the reference node `~/.local/bin/ctest` is a broken pip shim). Nine benchmarks
are wrapped by a repo-authored `verify.py`; their inner argv is obtained by
importing the wrapper with `subprocess.run` intercepted, so the binary is measured
directly and no wrapper is modified.

## Hardware neutrality

```
markers (roi/)  ->  collector adapter (collectors/<name>.py + lib/collectors.sh)
                ->  canonical activity model  ->  analysis.py (vendor-neutral)
device probe (probes/device.py) -> platform descriptor;  conformance probe -> platform record
```

* **Markers** know no vendor: NVTX by default, ROCTX on AMD (`dlopen`, no build
  dependency), Python annotators for XLA/TPU workloads.
* **A collector adapter** is the only vendor-specific code: it turns one profiled
  run into markers and device intervals in eight neutral categories, on one
  timeline, and declares the categories it can see (`CAPABILITIES`).
* **The analysis** clips intervals to "ROI minus excludes" with a streaming union
  per window; it never reads a vendor format.
* **The device probe** writes a neutral descriptor; `platform_id`
  (`nvidia-b200.cuda13.2`) joins measurements across hardware.
* **The conformance probe** (`probes/conformance/`) is the admission test for a
  platform + collector: a program with a known split (10 warm-up launches, an ROI
  of 20 launches and one device-to-device copy with 5 excluded check launches, one
  launch after) must come back exactly, clean and profiled ROI must agree, and a
  sentinel planted in the caller's environment must not reach anything the
  profiler wrote. Records of a platform without a pass are caveated.
* **`none`** works on any hardware from day one: ROI time and FOM, device columns
  null.

| platform | collector | status |
|---|---|---|
| NVIDIA B200, CUDA 13.2 | `nvidia_nsys` (Nsight Systems 2025.6.3) | conformance pass (`platforms/nvidia-b200.cuda13.2.json`) |
| any | `none` | works; device columns null |
| AMD (ROCm) | `amd_rocprofv3` | **interface only**: the adapter contract is written, `open()` refuses; the ROCTX marker backend and HIP env allow-list exist; UNVERIFIED (no ROCm here) |
| TPU (XLA) | `tpu_xprof` | **interface only** by decision: Python markers, adapter contract, device-probe stub; no Level 1/2/3 code runs on a TPU |

Onboarding a platform: implement the adapter to its contract, the measurement
wrapper in `lib/collectors.sh`, the device probe, and the environment allow-list
for its runtime (`backend_env_allow`); then pass the conformance probe.

## The clean environment

Profilers record the whole process environment: nsys stores it in
`TARGET_INFO_SYSTEM_ENV` (`DeviceEnvironment`). Measured on this node: 349
variables from a login shell, including session tokens and `SSH_*`. Every run
therefore starts from `env -i` plus an allow-list (`PATH`, `HOME`, `TMPDIR`,
`LD_LIBRARY_PATH`, eleven `SLURM_*` names the launcher reads, and the backend's
device variables); Level 2 then sources the repository environment script inside
that clean environment (`--env-script`, default `hpcperf_env.sh` or
`$HPCPERF_TIMING_ENV_SCRIPT`). A **deny rule beats the allow-list**
(`TOKEN|SECRET|PASSWD|PASSWORD|CREDENTIAL|PRIVATE_KEY|API_KEY|SESSION`). The
summarizer reads only the variable NAMES the profiler recorded and caveats any
that match the deny rule; values are never read. The tests plant credentials and
assert both rules.

## Profiler cost (why the headline comes from clean runs)

Measured with the previous whole-process protocol (2026-09-22, nsys 2025.6.3):

* The nsys command's wall clock grows by a fixed **2.9-6.9 s** (attach and report
  writing) plus **6.5-13 us per CUDA API call** -- rule of thumb 5 s + 10 us/call
  (13 Level 2 applications; `miniem`'s 15.2 M calls cost +125 s).
* That is mostly *outside* the application: quicksilver's own `main` timer grew
  8.9% (7.431 -> 8.09 s) while the command grew 1.7x.
* Device-side durations are insensitive: kernel totals moved 1.2% across sampling
  settings with identical launch counts.
* A FOM read from a profiled run is depressed 5.4-6.4%; FOMs are now read from the
  clean run.

Collection flags: `-t cuda,nvtx -s none --cpuctxsw=none --cuda-graph-trace=node`.
No CPU sampling (no metric uses it; it cost 0.9-2.3 s more on quicksilver). No
`--cuda-memory-usage`: it makes MiniEM crash with SIGSEGV in `cudaFreeAsync`
(exit 139) and feeds no metric.

## Launching Level 2

The collector wraps `run.sh` from the outside: 20 of 24 `run.sh` end in `exec`,
and a profiler inside the launcher's wrapper would own the pid the launcher's
nvidia-smi audit joins on, making every rank `unverified`. From the outside the
audit stays clean (`1 verified, 0 mismatch, 0 unverified`), the process tree is
followed, and `HPCPERF_ROI_LOG` reaches the ranks through the environment
(verified at one rank with quicksilver: `bash` -> `mpirun` -> `mpi_gpu_bind.sh` -> exe
wrote its ROI log, audit clean).

## Level 1 verification switch

Most Level 1 benchmarks validate by recomputing the whole workload on one CPU core.
That is outside the ROI now, but it can take minutes, so `measure_level1.sh` still
sets `HPCPERF_SKIP_VERIFY=1` (41 benchmarks honor it; `--keep-verify` turns it
off). ctest never sets it, and the tests assert that nothing enables it by
default. NPB `is` verifies inside its timed kernels and cannot be separated
without editing an upstream kernel: its ROI includes that check
(`verify_vs_roi=inside`, caveated). `background_subtraction` generates frames and
runs its CPU reference inside the frame loop; both are excluded.

## Tests

`bash tools/timing/tests/run_all.sh` -- CPU only, no GPU, profiler or build tree
needed (groups that need something absent skip). It covers the case tables and
every refusal rule, ROI log parsing, clipping/union/overlap and null-vs-0 on
hand-built traces, the nsys adapter on a synthetic sqlite export, summarize end to
end (deterministic CSVs, FOM, `roi_missing`, caveats), the marker header in C99 /
C11 / gnu11 / C++17 / ROCTX / no-annotation modes with nesting, cross-file state,
flushing past the buffer and default-off, the Fortran and Python APIs, that every
Level 1/2 source carries markers and every build sees the header, the front-ends,
the clean environment and deny rule with planted credentials, that AMD/TPU
interfaces refuse instead of guessing, `gen_cases.py`, and FOM extraction.

## Scope and what is UNVERIFIED

* **HIP/ROCm and AMD profiling**: marker backend and adapter contract only.
* **TPU**: interface only, by decision.
* **Multi-process ROI** (job ROI = slowest rank): implemented, but this allocation
  exposes one GPU, so Level 2 is measured at `HPCPERF_GPUS=1`.
* **Hardware counters** (`ncu`, occupancy, achieved bandwidth): not collected.
* **Level 3**: not covered; its run path is not yet wrapped in the clean
  environment.
