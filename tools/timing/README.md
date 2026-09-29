# tools/timing -- runtime measurement for Level 1, Level 2 and Level 3

Measures **how long the computation of each benchmark takes** and what the device
did during it, in a form that stays comparable when applications, inputs and
hardware platforms are added. What was changed, why, and the interfaces for adding
an application, an input or a platform: [DESIGN.md](DESIGN.md).

```bash
tools/timing/measure_level1.sh --build-root build/gcc13 all      # Level 1: 50 benchmarks
tools/timing/measure_level2.sh all                               # Level 2: 24 applications
tools/timing/measure_level3.sh all                               # Level 3: 10 applications, 2 GPUs each
python3 tools/timing/report.py --publish                         # copy the web page into docs/timing/
bash tools/timing/tests/run_all.sh                               # self-tests, CPU only
```

Each measurement ends by summarizing its own run (JSON per case, CSV per level) and
regenerating the web page `results/timing/report/index.html`; `summarize.py` does the
same by hand for all raw data.

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

| | Level 1 | Level 2 | Level 3 |
|---|---|---|---|
| unit | 51 cases of 50 benchmark binaries | 28 cases of 24 `level2/<app>/run.sh` | 10 cases of 10 `level3/<app>/run.sh`, 2 GPUs each |
| region | ROI markers | ROI markers | **the application's own loop timer** (no markers, see below) |
| runs per case | 1 warm-up + 5 clean + 1 profiled | 1 clean + 1 profiled | 1 clean + 1 profiled (QMCPACK: 1 clean, see below) |
| verification | outside the ROI; `HPCPERF_SKIP_VERIFY=1` also skips the CPU reference (minutes for some) | outside the ROI; `validate.sh` is never called | outside; `validate.sh` is never called, and only 2 of the 10 timing inputs are ones it checks |
| FOM | none (the benchmarks' own printouts are not comparable) | the application's own metric where it prints one (16 of 24) | LAMMPS and SPARTA print one |

The headline therefore carries **no profiler overhead**: the profiler's cost shows
up only as `roi_profiler_inflation` (profiled ROI / clean ROI), and device-side
durations come from device timestamps, which it barely perturbs.

## Level 3: the applications' own timers

Level 3 applications are full production codes (LAMMPS, CP2K, QMCPACK, ...): marking
their loops by hand is not practical, and every one of them already times its loop
itself. So the measured region of a Level 3 application is **its own timer**, chosen
once per application by the same rule as the markers -- the time-step / iteration
loop, without start-up, set-up, a warm-up step the application charges set-up to,
verification and final output -- and fixed with source citations in
[apptimers.py](apptimers.py) (`python3 tools/timing/apptimers.py describe` prints the
definitions; every record carries its own). The record keeps the Level 1/2 schema:
`roi.wall_s` is the timer (median of the clean runs), `roi.source = "app_timer"`.

| application | the region (application timer) | ranks combined | waits for the GPU |
|---|---|---|---|
| LAMMPS | `Loop time` of the last `run` (sections: Pair/Neigh/Comm/...) | average, barriers | total yes; **sections no** (asynchronous kernel time lands in Comm) |
| SPARTA | `Loop time` of the last `run` (the deck's first run is warm-up) | average, barriers | total yes |
| WarpX | final `Evolve time` (in-loop diagnostics included) | rank 0 | yes (TinyProfiler regions synchronize) |
| Nyx | sum of `[STEP n] Coarse TimeStep time`; checkpoint / plotfile writes excluded | max over ranks | step totals effectively |
| nekRS | runtime statistics `solve` minus its `checkpointing` row | max over ranks | yes (barriers, blocking event syncs) |
| SPECFEM3D | `Elapsed time` of the last stability report (`output_solver.txt`) | rank 0 after a max reduction | yes (the report synchronizes the stream) |
| ExaCA | `Time spent performing CA calculations` | rank 0, barriers | yes (fences) |
| CP2K | per-MD-step `UsedTime` (`.ener`) of steps 2..N; step 1 carries the initial force evaluation | rank 0 | back-ends synchronize their streams |
| QMCPACK | `DMCBatched::Production` of the stack timer profile (`--enable-timers=medium`) | rank 0, barrier at start | total yes; nested timers no |
| DFT-FE | per MD step `updateAtomPositionsAndMoveMesh` + the step's SCF iterations | barrier + max | implicit only; **force evaluation untimed** |

Two applications needed their `run.sh` to switch on a timer that already exists:
QMCPACK runs with `--enable-timers=medium` (`HPCPERF_QMCPACK_TIMERS`, default medium;
the default coarse level has no Production timer), DFT-FE with `VERBOSITY = 1` in its
deck copy (`HPCPERF_DFTFE_VERBOSITY`; the upstream decks ship 0, and `REPRODUCIBLE
OUTPUT = true` must stay because it also fixes numerical settings -- it is why DFT-FE
does not time its force evaluation). Both are output-only; `validate.sh` passes with
them at 1 and 2 GPUs.

Without markers the profiled run cannot be clipped to the region: its device picture
(`context.whole_process`, the ops table with `ops_scope = whole_process`) covers the
**whole process** -- set-up included -- and is context, not a breakdown of the region.
The exception is an application that emits an NVTX range for its loop itself: WarpX's
AMReX TinyProfiler pushes `WarpX::Evolve()`, and its device activity is clipped to that
range (`cases/level3_apps.tsv: nvtx_roi`).

**QMCPACK is not profiled by default** (`cases/level3_apps.tsv`, column `profile = no (<reason>)`;
the reason is repeated as a caveat in each of its records): its profiled run of the timing input
writes a 24 GB trace (57M kernels, 72M copies, 274M CUDA API calls) and cost about 55 of the 100
minutes of the first sweep for a 5-minute application, while its timed region grew only 9% under
nsys and the device picture would still be whole-process context, not the DMC loop.
`measure_level3.sh --profile-all` profiles it anyway; `--no-profile` skips every profiled run.

A clean run whose output does not contain the timer is `app_timer_missing` (FAIL); a
Level 3 run never falls back to the process wall clock. Each run writes its run
directory under `build/level3/<app>/<profile>/run.timing-<run id>-<c0|prof>/`
(`HPCPERF_L3_RUN_SUBDIR`), so no validated or historical run directory is touched, and
the files the timer is read from are copied into the raw evidence.

## Headline fields (`summary_level<N>.csv`)

| field | meaning |
|---|---|
| `roi_wall_s` (+ `_min`, `_max`, `_stddev`, `roi_runs`) | median ROI time of the clean runs |
| `roi_entries`, `roi_excluded_s` | how often the ROI was entered; time carved out by excludes |
| `roi_profiled_wall_s`, `roi_profiler_inflation` | the same region in the profiled run, and the ratio |
| `device_busy_s`, `device_busy_frac_of_roi` | union of all device activity inside the ROI -- concurrent operations count once; summed over processes, the fraction is per process (= per GPU) on average |
| `host_gap_s`, `host_gap_frac_of_roi` | `roi_wall_s - device_busy_s / processes`: time in the ROI when the device was idle, i.e. the host was the bottleneck |
| `device_<category>_s`, `device_*_ops`, `device_copy_*_bytes` | per category inside the ROI: `compute`, `copy_h2d`, `copy_d2h`, `copy_d2d`, `copy_other`, `fill`, `collective`, `other` |
| `device_op_time_sum_s`, `device_overlap_s` | naive sum of op durations, and sum - union (concurrency) |
| `runtime_api_calls`, `runtime_api_sync_calls`, ... | host runtime API calls inside the ROI |
| `top_op_*`, `ops_level<N>.csv` | the operations inside the ROI by total time |
| `process_wall_s`, `pre_roi_s`, `post_roi_s`, `whole_*` | context: the whole process -- never the headline |
| `fom_*` | the application's own metric, from the clean run |
| `device_*`, `driver_version`, `runtime_version`, `host_*` | the platform descriptor |
| `collector`, `conformance` | which profiler adapter produced the device columns, and whether the platform passed the conformance probe |
| `roi_source`, `roi_steps`, `roi_setup_s` | `markers` or `app_timer` (Level 3); steps in the region and the set-up the application reports (Level 3) |
| `ops_scope`, `whole_<category>_s`, `whole_runtime_api_calls` | `roi` or `whole_process`; the whole profiled process per category (Level 3 context) |

A device column is **empty when the platform cannot observe it** (no collector, or
a category outside the collector's capabilities) -- never 0. A 0 means observed
and absent. Every limitation of a record is spelled out in its JSON `caveats`.

## The web page

`report.py` renders `results/timing/` as one self-contained interactive page
(`index.html`: the data embedded as JSON, inline CSS and script, fonts from Google
Fonts with system fallbacks, light and dark theme) plus a Markdown twin (`README.md`)
that the repository browser displays. The page holds only the Level 1 and Level 2
timing results:

1. **Level tab**, then **an application** from the list (each shows its number of
   inputs and how many input x platform combinations were measured).
2. The application's **inputs x platforms** grid: inputs are its cases from
   `cases/` (with the variables and arguments that define them) plus anything
   measured; platforms are every platform with a measurement or a conformance record.
   A combination never measured shows `null`.
3. Choosing a measured combination shows **that measurement** -- and only then: ROI
   (median, min/max, every clean run, entries, excluded time), clean-run spread, device
   busy and host gap inside the ROI, ROI share of the process with the process
   breakdown bar, profiler inflation, device time / ops / bytes per category (`null`
   where the collector cannot observe it), the top operations, runtime API calls, the
   FOM, the application's own timer against the ROI, the launcher audit, the platform's
   conformance, the input and command as run, the caveats, and every run of the
   combination with its change against the previous one (the view shows the latest
   successful run; a later failed run is flagged).

The selection is kept in the URL hash (`index.html#L2/quicksilver/p200000/nvidia-b200.cuda13.2`),
so a view can be linked. The Markdown twin lists the latest successful run of every
measured combination as a Level 1 and a Level 2 table. Absolute paths of the checkout
are written as `{REPO}`; host names, the environment and GPU UUIDs are not in the page.
The output depends only on the records and the case tables, so the same data gives
the same bytes.

* **Automatic**: every `measure_level*.sh` run (unless `--no-summary`) and every
  `summarize.py` rewrite `results/timing/report/` (git-ignored).
* **In the repository**: `python3 tools/timing/report.py --publish` writes the same page
  to `docs/timing/`; committing it is the deliberate step that shows it. That snapshot is
  the only measurement output that enters git -- raw evidence, JSON and CSV never do.
  GitHub shows `docs/timing/README.md`; `index.html` needs a browser (or GitHub Pages
  serving `docs/`).

The own-timer check comes from `app_timer_regex` / `app_timer_unit` in
`cases/level2_apps.tsv`: the timer an application prints for exactly the region its
markers enclose (8 applications). It checks the marker placement; it is not a metric.

## Cases: many inputs per application, new applications

A measurement's identity is `(level, app, case, platform)`. Cases live in
`cases/`, tab-separated, `-` for an empty field (bash `read` collapses runs of
tabs); `cases.py` resolves them into the rows the engine runs:

| table | what |
|---|---|
| `level1.tsv` | generated from ctest by `gen_cases.py` -- one case per ctest test (`default` when a benchmark has one) |
| `level1_extra.tsv` | Level 1 inputs ctest does not know (e.g. `all_pairs_distance/n20000`) |
| `level1_apps.tsv` | per benchmark: suite, backends, `roi_excludes`, `verify_vs_roi` |
| `level2_apps.tsv` | per application: backends, timeout, FOM pattern, `roi_excludes`, own-timer pattern |
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

## Launching Level 3

The same as Level 2, at 2 GPUs: the collector wraps `run.sh` from the outside, the
launcher's audit stays clean, and nsys follows `mpirun` into both ranks. This is the
clean-environment wrapper CLAUDE.md asks for before profiling a Level 3 run: every
run starts from `env -i` + the allow-list + the credential deny rule. `--env-script`
must set up what `run.sh` needs (the toolchain, the MPI transport profile).

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
interfaces refuse instead of guessing, `gen_cases.py`, FOM extraction, and the web
page (byte-identical for the same records, inputs x platforms with `null` for
unmeasured combinations, latest-run selection and history, inert embedding of kernel
names, no absolute paths, the Markdown twin, the application timer, `--run-id`, no
summary on a dry run), and Level 3 (case resolution and refusal, every application's
timer extracted from synthetic evidence, a missing or incomplete timer failing loudly,
every cited source line existing, the dry run, the record, CSV and page, QMCPACK's
profile default and its override) -- 54 checks.

## Scope and what is UNVERIFIED

* **HIP/ROCm and AMD profiling**: marker backend and adapter contract only.
* **TPU**: interface only, by decision.
* **Multi-process ROI** (job ROI = slowest rank): implemented, but this allocation
  exposes one GPU, so Level 2 is measured at `HPCPERF_GPUS=1`.
* **Hardware counters** (`ncu`, occupancy, achieved bandwidth): not collected.
* **Level 3**: measured with the applications' own timers at 2 GPUs (one node,
  2 x B200). Only QMCPACK's and DFT-FE's timing inputs have numerical acceptance
  (`validate.sh`); the larger inputs of the other eight are completeness runs, and
  their records say so (`verify_vs_roi = none`). The device picture is whole-process
  except for WarpX.

## First ROI sweep (2026-09-22)

dgx003, 1x NVIDIA B200 (sm_100, driver 595.58.03), CUDA 13.2.78, Nsight Systems
2025.6.3, GCC 13.3 (uv toolchain), Level 1 `build/gcc13`, Level 2 at
`HPCPERF_GPUS=1`. Conformance `nvidia-b200.cuda13.2` passed before the sweep.

| | Level 1 | Level 2 |
|---|---|---|
| cases ok | **51 / 51** | **28 / 28** (24 default + `amg2023/n128,n192`, `quicksilver/p50000,p200000`) |
| sweep wall time | 15.7 min | 34 min (incl. summarize) |
| ROI as a share of the process (median) | **0.9%** (43 of 51 under 10%) | 66% (6 of 28 under 10%) |
| device busy inside the ROI (median) | 90.7% | 92.3% |
| clean-run spread (CV, median / max) | 0.18% / 6.4% | one clean run (see below) |
| profiler inflation of the ROI (median / max) | 1.012 / 1.44 | 1.023 / 1.35 |
| FOM captured | -- | 20 of 20 cases whose application prints one |
| launcher audit | -- | 21 clean, 7 without the launcher, **0 not clean** |

**The ROI agrees with the applications' own timers.** Ten Level 2 cases print a
timer for the same region; the ROI matches each to within 0.01%: exacmech 8.04129
vs 8.04127 s, miniweather 32.3133 s both, quicksilver `main` 7.481 vs 7.4809 s,
shaw `loopTime` 15.4616 s both, P3 heat3d/vlp4d `total` 1.59309/2.31051 vs
1.59306/2.31046 s, hipBone 0.2881 vs 0.28808 s (xsbench prints three digits:
0.040 vs 0.04008 s).

**The ROI changes the picture at Level 1.** Under the whole-process protocol about
30 of 50 benchmarks looked less than 5% GPU-busy: a constant 0.4-0.6 s of context
creation and input generation dominated. Inside the ROI the median device-busy
share is 90.7%; the 11 benchmarks under 50% are genuinely host-bound loops (bfs,
gaussian_elimination, pathfinder, spmv, fir, nearest_neighbor's host selection,
...) or ROIs of a few hundred microseconds (hotspot's single 90 us kernel). Those
tiny ROIs are also where the profiler inflation exceeds 1.2 (bfs, fir, hotspot,
nearest_neighbor, srad_v1) -- the headline comes from the clean runs, so it is
unaffected.

Findings that need a decision rather than a fix:

* **Several default Level 2 inputs are set-up dominated.** MiniEM: 106.7 s of a
  111.2 s process before its three time steps (mesh 24.6 s, DOF numbering 11.8 s,
  auxiliary operators 37.8 s, W operator 14.5 s, preconditioner 2.9 s), ROI 1.31 s.
  hipBone 0.29 s of 19.5 s, SW4lite 0.06 s of 8.3 s, XSBench 0.04 s of 3.8 s. The ROI
  is right by the rule (it matches upstream's own timed region), but as training data
  these cases carry little computation; larger cases (more time steps) would fix it.
* **Level 2 run-to-run spread is real.** quicksilver's ROI varied 4-7% over 5 clean
  runs (default: 6.84-7.70 s), and its own timers show the same spread (unified-memory
  migrations and the MPI phase). With one clean run per case the Level 2 protocol
  cannot show it; `--clean-runs 3` or `5` costs one extra run each.
* **Profiler inflation follows the API call count:** laghos 1.35 (14.5 M launches in
  the ROI), comb 1.23. Device durations are unaffected; this only matters when reading
  host-side quantities from the profiled run.

`device_overlap_s` is non-zero only for quicksilver (0.22-0.35 s: unified-memory
migrations overlap its kernel); everything else is single-stream.

## First Level 3 sweep (2026-09-29)

dgx003, **2 x NVIDIA B200** (one MPI rank per GPU, `--mca pml ob1 --mca btl self,sm,smcuda`),
CUDA 13.2.78, Nsight Systems 2025.6.3, uv toolchain (GCC 13.3.0 / system GCC 14.2.1 for CP2K and
DFT-FE, Open MPI 5.0.10), private LLVM 23.1.0 for QMCPACK. One clean + one profiled run per case,
run ids `20260929T045201Z-2209081` (ExaCA) and `20260929T045434Z-2212883` (the other nine);
the whole sweep took 99 min, 10 / 10 ok, launcher audit `2 verified, 0 mismatch` in every clean run.

| application | input (2 GPUs) | timed region | steps | per step | region share of process | profiler x | device busy per GPU | numerical acceptance |
|---|---|--:|--:|--:|--:|--:|--:|---|
| CP2K | H2O-128 MD (10 steps) | 54.9 s | 9 | 6.1 s | 52% | 1.02 | 7% (whole process) | none (validate.sh: H2O-64) |
| DFT-FE | al_md (32 Al, 4 MD steps) | 34.3 s | 3 | 11.4 s | 29% | 1.06 | 1% (whole process) | validate.sh |
| ExaCA | 512x256x1024 | 15.2 s | -- | -- | 78% | 1.16 | 25% (whole process) | none |
| LAMMPS | LJ 16.4M atoms, 2000 steps | 49.9 s | 2000 | 24.9 ms | 86% | 1.12 | 14% (whole process) | none |
| nekRS | ethier 32k elements N=7, 50 steps | 46.2 s | 50 | 924 ms | 14% | 1.00 | 17% (whole process) | none |
| Nyx | synthetic 256^3, 10 steps | 9.1 s | 10 | 914 ms | 51% | 1.11 | 10% (whole process) | none |
| QMCPACK | diamondC_2x1x1, 256 walkers | 286.9 s | 2500 | 115 ms | 93% | 1.09 | 6% (whole process) | validate.sh |
| SPARTA | collide 270M particles, 100 steps | 13.2 s | 100 | 132 ms | 12% | 1.04 | 61% (whole process) | none |
| SPECFEM3D | half-space 331,776 elements, 20000 steps | 18.1 s | 20000 | 0.90 ms | 6.5% | 1.00 | 6% (whole process) | none |
| WarpX | uniform plasma 256^3, 1000 steps | 20.5 s | 1000 | 20.5 ms | 82% | 1.07 | **70% inside `WarpX::Evolve()`** | none |

What the numbers say, and what they do not:

* **Set-up still dominates several processes even at these sizes**: SPECFEM3D spends 241 s
  generating its databases on the CPU (plus 2.8 s meshing) before an 18 s time loop; nekRS reports
  278 s of initialization (JIT compilation for this polynomial order, first run, and kernel
  autotuning) before a 46 s solve; SPARTA 62 s creating 270M particles plus its 8.7 s 30-step
  warm-up run before the 13.2 s timed run. The region excludes all of it by definition; "region
  share of process" makes it visible.
* **Whole-process device busy is not the region's device busy.** Only WarpX can be clipped
  (70% busy per GPU inside its loop vs 29% over the whole process); for the others the device
  column mixes set-up and loop. DFT-FE's 1% reflects a 32-atom problem whose MD step is dominated
  by host-side re-initialization (`updateAtomPositionsAndMoveMesh`, 17.8 s of each step at 1 GPU).
* **LAMMPS strong is communication-bound on this site at 2 GPUs**: 24.9 ms/step against
  ~11 ms/step at 1 GPU in its README (the 4-GPU run was also slower than 1 GPU there); its
  section table puts 97% in Comm, but those sections are not device-synchronized (caveat in
  the record), so that split is not a GPU breakdown. Recorded, not generalized.
* **The profiler costs QMCPACK a 24 GB trace**: 57M kernels, 72M copies and 274M CUDA API calls.
  The timed region grows only 9% under nsys (312 vs 287 s), but the profiled process took 35 min
  against 5 min clean (nsys writing its 1 GB report), the sqlite export 12 min and most of the
  8.6 min summary -- about 55 of the sweep's 100 minutes; `--no-profile` skips it.
* **Sweep cost**: the ten clean runs took 1367 s (23 min) -- that is what `--no-profile` costs.
  The profiled runs took 3301 s, their sqlite exports 790 s and the summary 8.6 min: with nsys the
  sweep took 100 min, 4.4x. Per case the profiled run plus export costs 1.1x (SPARTA, SPECFEM3D)
  to 3.6x (Nyx) a clean run, QMCPACK 9.1x; nekRS's profiled run was shorter than its clean run
  (234 vs 331 s) because the clean run, first at this polynomial order, paid the JIT compilation.
* One clean run per case: the spread column is null until `--clean-runs 3` is used.
