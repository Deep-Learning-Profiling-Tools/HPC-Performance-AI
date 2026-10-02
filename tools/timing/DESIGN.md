# tools/timing -- design, changes and interfaces

This is the entry point for anyone reviewing the runtime-measurement work (branch
`junyu-timing`) or extending it with an application, an input or a hardware platform.
It says **what the design is for, what was changed in the repository, and which
interfaces were left for extension**. The details live in the documents listed at the
end; this file does not repeat them.

## 1. What it is for

The repository is a GPU benchmark suite for an AI framework that predicts HPC
performance. Such a framework needs **timing labels that mean the same thing for every
application, every input and every piece of hardware**. Before this work the suite
proved correctness only; timing it naively gives numbers that are mostly not the
computation:

* A process's wall clock is dominated by things that are not the workload. Measured on
  Level 1: `daxpy` spends 99% of its process outside its kernels; `channel_shuffle`
  spends 573 of its 574 ctest seconds in a single-core CPU re-computation used for
  verification; across all 51 Level 1 cases the computation is a median **0.9%** of
  the process.
* The benchmarks' own timing printouts are not comparable (four units, 17 of 50 print
  nothing, most bracket host code around the launch).
* A profiler changes what it measures (Nsight Systems: +2.9-6.9 s fixed plus 6.5-13 us
  per CUDA API call on the command's wall clock).

The requirements that shaped the design:

1. Time **only the computation** -- no start-up, set-up, warm-up or verification.
2. **Many inputs per application**, and applications added later, without changing
   the method.
3. **Hardware neutrality**: other GPUs (AMD) and TPUs must fit without rewriting the
   analysis; what a platform cannot observe must say so instead of reading 0.
4. Machine-readable results (one JSON per run, CSV per level) that never enter git,
   plus a web page that can be shown in the repository.

## 2. Design decisions

| decision | why | consequence |
|---|---|---|
| The **region of interest (ROI)** is declared in each benchmark's source with markers (`HPCPERF_ROI_BEGIN/END`, `..._EXCLUDE_...`) | Only the source knows where the computation starts and ends; inferring it from a trace or from printed timers is guesswork | Every Level 1/2 source carries markers, inserted without changing any upstream line; ctest/validate.sh are unchanged because the markers are a no-op unless measuring |
| **One set of markers, two runs**: clean runs (markers log their own timestamps, no profiler) give the headline `roi_wall_s`; a profiled run gives the device picture clipped to the same markers | The profiler inflates host time but not device timestamps | The headline carries no profiler overhead; the inflation is reported (`roi_profiler_inflation`), not hidden |
| **Placement rule**: per-step work, copies and scalar diagnostics inside; one-time set-up, warm-up, final checks and output outside; bulk output and in-loop checks excluded; upstream's own timed region is the anchor | Makes the ROI of different applications comparable and reviewable | `tools/timing/roi/README.md`; each placement is described at the markers, in each Level 2 README and in `cases/*_apps.tsv` |
| **Vendor code only in collector adapters**; analysis works on a canonical model (markers + device intervals in 8 neutral categories) | A new platform must not touch the analysis | NVIDIA adapter done; AMD and TPU adapters are contracts that refuse; a `none` collector gives ROI time and FOM on any hardware |
| **null is never 0** | "not observable" and "observed, nothing happened" are different labels for a model | Each collector declares `CAPABILITIES`; missing categories are null in JSON, CSV and the page |
| **Admission test per platform** (conformance probe with a known 10 / 20+1 / 5 / 1 split) | A collector that merely runs is not a correct collector | `platforms/<id>.json`; runs on a platform without a pass are caveated |
| Measurement identity is **(level, app, case, platform)**; inputs are declared in case tables; an input variable set in the caller's shell but not declared is **refused** | Every run starts from `env -i`, so a stray variable would be dropped silently and the default input measured under another name | Adding an input is one table row; sweeps (`NAME=a\|b`) expand to one case per value |
| Every run starts from **`env -i` + allow-list + credential deny rule** | Profilers store the whole process environment (nsys: 349 variables incl. session tokens from a login shell) | Only variable names are ever read back from a trace; tests plant credentials |
| **Failures are loud**: a clean run that writes no ROI record is `roi_missing` (FAIL); every limitation of a record is a sentence in `caveats` | A silent fallback to the process wall clock would be a wrong label that looks right | Onboarding an application without markers fails the tests and the measurement |
| **Level 3: the application's own timer is the region** (no markers), defined once per application with source citations in `apptimers.py`; same record schema, `roi.source = "app_timer"` | Full production codes cannot reasonably be marked by hand, and each already times its loop; choosing WHICH timer by the marker placement rule keeps the label comparable | A missing timer is `app_timer_missing` (FAIL); the profiled run is whole-process context unless the application emits an NVTX range for its loop (WarpX) |
| Outputs are **deterministic** and raw data **never enters git** | Reviewable diffs; no measurement noise in the repository | The only committed measurement output is the rendered page `docs/timing/` (`report.py --publish`, deliberate) |

## 3. How the pieces fit

```
 source markers (roi/)                     case tables (cases/*.tsv) --- cases.py (resolve, refuse)
        |                                                   |
        v                                                   v
 measure_level1.sh / measure_level2.sh  -->  lib/engine.sh: per case, from env -i
        |                                      warm-up -> clean runs (HPCPERF_ROI_LOG) -> profiled run
        |                                      lib/collectors.sh wraps the run (nsys / none / ...)
        v
 build/timing/level<L>/<app>/<case>/<run_id>/     raw evidence (git-ignored)
        |
        v
 summarize.py  --  analysis.py (ROI logs, clip device activity to ROI minus excludes)
        |          collectors/<name>.py (vendor trace -> canonical model)
        |          probes/device.py (neutral device descriptor), platforms/ (conformance)
        v
 results/timing/  JSON per run (hpcperf-timing-2), summary_level<L>.csv, ops_level<L>.csv
        |
        v
 report.py  -->  results/timing/report/{index.html,README.md}   (after every measurement)
            -->  docs/timing/{index.html,README.md}             (--publish, committed on purpose)
```

## 4. What was changed in the repository

### 4.1 New: `tools/timing/` (self-contained; reads nothing from `level2/tools` or `level3/`)

| path | role |
|---|---|
| `roi/hpcperf_roi.h` | the C/C++ marker API, header-only; NVTX (vendored) or ROCTX (dlopen) annotation; clean-timing log |
| `roi/hpcperf_roi.f90`, `roi/hpcperf_roi_fortran.c` | Fortran module + C entry points (used by GAMESS RI-MP2) |
| `roi/hpcperf_roi.py` | Python twin with the same log format (the TPU / XLA path) |
| `roi/hpcperf_vendor/nvtx3/` | NVTX v3 headers copied verbatim from CUDA 13.2 (Apache-2.0, `SHA256SUMS`) |
| `measure_level1.sh`, `measure_level2.sh`, `measure_level3.sh` | front-ends: option parsing, then `lib/engine.sh` |
| `apptimers.py` | Level 3: per application, which printed timer is the region, its definition, source lines, rank reduction and device synchronization, the extractor, and the files to copy out of the run directory |
| `lib/engine.sh` | clean environment, warm-up / clean / profiled runs, raw layout, automatic summarize |
| `lib/collectors.sh` | measurement side of each collector, per-backend environment allow-list |
| `collectors/` | canonical model + adapters: `nvidia_nsys` (verified), `none`, `amd_rocprofv3` and `tpu_xprof` (contracts) |
| `analysis.py` | vendor-neutral: ROI log parsing, interval clipping and union, per-category totals |
| `probes/device.py` | neutral device descriptor and `platform_id`; AMD / TPU probes refuse |
| `probes/conformance/` | admission test: `probe.cu`, `expected.json`, `check.py`, `run_conformance.sh` |
| `platforms/nvidia-b200.cuda13.2.json` | the conformance record of the one platform measured so far |
| `cases.py`, `cases/*.tsv` | case tables and their resolver (identity, inputs, sweeps, refusal rules) |
| `gen_cases.py` | generates `cases/level1.tsv` from ctest (one case per ctest test) |
| `summarize.py` | raw data -> JSON per run, CSV per level, then the page |
| `report.py`, `report_assets/` | the interactive page and its Markdown twin |
| `tests/run_all.sh` | 54 CPU-only checks |
| `README.md`, `SCHEMA.md`, `roi/README.md`, this file | documentation |

Removed: the first-generation whole-process tables `cases.tsv` / `cases_l2.tsv` and
their schema (`hpcperf-timing-1`).

### 4.2 Level 1 (`level1/`, commit `e6f9a44` and earlier)

* All 50 benchmarks: `#include "hpcperf_roi.h"` and `HPCPERF_ROI_*` markers in the CUDA
  sources and in the 40 HIP ports (Hetero-Mark's shared `common/benchmark_runner.cc`
  covers both), a 3-line `target_include_directories` block in each `CMakeLists.txt`,
  and a "Measurement markers" paragraph in each benchmark README. Pure insertions;
  Rodinia's CRLF line endings preserved.
* Earlier on the branch: the `HPCPERF_SKIP_VERIFY=1` switch (41 benchmarks skip their
  CPU reference while measuring; default off, ctest unchanged; commit `8daa974`), and two
  off-by-default cross-check switches, `HPCPERF_L1_PHASE_TIMING=1` (Hetero-Mark phase
  table) and `-DHPCPERF_NPB_PROFILING=ON` (NPB per-kernel table) (commit `6589737`).
* `level1/README.md`: the runtime-measurement section.

### 4.3 Level 2 (`level2/`, one commit per application)

* All 24 applications: markers inserted in their sources (pure insertions, no upstream
  line changed or removed); `build.sh` exports `CPATH=$R/tools/timing/roi` so no upstream
  build file changes; GAMESS RI-MP2's `build.sh` compiles and links the Fortran module
  and C shim; each README's `## Changes from upstream` records the placement and the
  excluded parts, and the statements that previously claimed byte-identity were
  corrected (amg2023, cloverleaf, kripke, laghos, miniem, remhos, tealeaf; MiniEM's
  `build.sh` / `src/CMakeLists.txt` comments too -- `src/UPSTREAM_SHA256SUMS` still
  records upstream's checksums).
* `level2/README.md`: the runtime-measurement section.

### 4.4 Level 3 (`level3/`)

* No markers. `qmcpack/run.sh` passes `--enable-timers=$HPCPERF_QMCPACK_TIMERS` (default
  `medium`: the coarse default has no Production timer) and records it in the manifest;
  `dftfe/run.sh` sets `VERBOSITY = $HPCPERF_DFTFE_VERBOSITY` (default 1, the decks ship 0)
  in its copy of the deck and records the change in `deck.diff` and the manifest. Both
  are output-only; `validate.sh` passes with them at 1 and 2 GPUs.
* `cases/level3_apps.tsv`, `cases/level3_cases.tsv`: the applications and the hand-written timing
  cases (2 GPUs; `acceptance` says whether `validate.sh` checks that input). Since 2026-10-02 the
  registered inputs of `level3/<app>/inputs.yaml` are the measured Level 3 workloads:
  `gen_registry_cases.py` writes `cases/level3_registry.tsv` (one case per input, the
  application's selector variable = input id, `gpus` = the input's `runtime_config.gpus`),
  `measure_level3.sh --registry` runs them through the same engine (1 clean + 1 profiled run,
  the application's own timer, no markers), and `verify_registry_runs.py` judges every record
  from the run manifest, the launcher's audit lines and the placement records under the rules of
  `cases/registry_evidence.yaml` (section `level3`). The hand-written cases stay for the
  2-GPU sweep (history on the page).

### 4.5 Elsewhere

* `CLAUDE.md`: the `tools/timing` working notes and the rule on what may be committed.
* `docs/timing/`: the published page (a rendered snapshot; regenerate, do not edit).

## 5. Interfaces

### 5.1 Marking a region of interest

| language | API |
|---|---|
| C / C++ / CUDA / HIP | `#include "hpcperf_roi.h"`; `HPCPERF_ROI_BEGIN()` / `END()`, `..._BEGIN_SYNC()` / `..._END_SYNC()` (device-wide synchronize first, only while measuring), `HPCPERF_ROI_EXCLUDE_BEGIN()` / `..._BEGIN_SYNC()` / `..._END()` |
| Fortran | `use hpcperf_roi`; `call hpcperf_roi_begin()`, `..._end()`, `..._begin_sync()`, `..._end_sync()`, `..._exclude_begin()`, `..._exclude_begin_sync()`, `..._exclude_end()`; link `hpcperf_roi.f90` + `hpcperf_roi_fortran.c` |
| Python | `import hpcperf_roi as roi`; `roi.region(sync=True)`, `roi.excluded()`, `roi.begin/end/exclude_begin/exclude_end`, `roi.set_device_sync(fn)`, `roi.add_annotator(push, pop)` (`jax_annotator()` / `torch_annotator()` UNVERIFIED) |

Semantics: nesting allowed, only the outermost pair counts; an ROI may be entered many
times and the entries add up; excludes are summed per entry; the state is shared across
translation units. Activation: `HPCPERF_ROI_LOG=<prefix>` (clean timing, one log per
process) or an injected profiler (`NVTX_INJECTION64_PATH`, ROCTX/rocprofiler variables).
Placement rule and build integration: `roi/README.md`. Log format: `SCHEMA.md`.

### 5.2 Adding an application

1. Place the markers (rule in `roi/README.md`); describe the placement at the markers
   and in the application README.
2. Put `tools/timing/roi` on the include path: Level 1 `target_include_directories`,
   Level 2 `export CPATH=...` in `build.sh`.
3. Build and run its correctness check unchanged.
4. Add its rows: Level 1 -- `python3 tools/timing/gen_cases.py --build-root <build>`
   regenerates `cases/level1.tsv`, plus a row in `cases/level1_apps.tsv`; Level 2 -- a
   row in `cases/level2_apps.tsv` (backends, timeout, FOM pattern if it prints one,
   `roi_excludes`, and `app_timer_regex` / `app_timer_unit` if it prints a timer for
   exactly its ROI), a `default` row in `cases/level2_cases.tsv`, and its CPU-binding
   policy in `cases/level2_binding.tsv` (mpirun-launched: `cpus_per_rank` = its host
   threads, `omp_pin` when it has several; direct-exec: no binding).
5. `python3 tools/timing/cases.py check` and `bash tools/timing/tests/run_all.sh` must
   pass: the tests fail for a Level 1/2 source without BEGIN/END or a build that does
   not see the header, and for a `run.sh` without a case row.
6. Measure. A run without an ROI record is `roi_missing` (FAIL).

### 5.2b Adding a Level 3 application

1. Pick the application's own timer for its loop by the placement rule (loop in, set-up,
   warm-up step and final output out) and read its implementation: what it covers, how
   ranks are combined, whether it waits for the device.
2. Add a `Timer` to `apptimers.py`: the extractor (raise `TimerMissing`, never guess),
   the files to copy out of the run directory, the definition, the source lines, the
   reduction, the device synchronization, and a caveat if a sub-timer is misleading.
   If the timer has to be switched on, do it in `run.sh` as an output-only variable and
   re-run `validate.sh`.
3. A row in `cases/level3_apps.tsv` (FOM if it prints one, `nvtx_roi` if it emits an
   NVTX range around its loop, `profile = no (<reason>)` if its profiled run is not worth
   its cost -- QMCPACK) and a case in `cases/level3_cases.tsv` with its `acceptance`; the
   registered inputs come from the application's `inputs.yaml` (whose `timing.app_timer.source`
   names the timer) through `gen_registry_cases.py`, plus a `level3.<app>` rule block in
   `cases/registry_evidence.yaml` for the verifier.
4. Synthetic evidence for the extractor in `tests/run_all.sh` (13d), then
   `bash tools/timing/tests/run_all.sh` (13f checks the cited source lines exist).

### 5.3 Adding an input

* A benchmark with an `inputs.yaml`: add the input there (the registry is the source of truth) and
  run `python3 tools/timing/gen_registry_cases.py`; it is then measured with `--registry` (README,
  "Registered inputs"). The hand-written tables below remain for cases outside the registry.

* Level 2: a row in `cases/level2_cases.tsv` -- `app case gpus env args timeout_s
  fom_regex notes`; `env` may only set variables the application's `run.sh` reads
  (`python3 tools/timing/cases.py allowed-env <app>`) or its `extra_env`; one variable
  may sweep (`HPCPERF_AMG_N=128|192` with case name `n{}`).
* Level 1: a row in `cases/level1_extra.tsv` -- `app case args env timeout_s notes`
  (the default cases come from ctest).
* Reserved names, never settable by a case: `HPCPERF_ROI_LOG`, `HPCPERF_SKIP_VERIFY`,
  `HPCPERF_GPUS` (use the `gpus` column), `HPCPERF_NP`, `HPCPERF_DRY_RUN`; names
  matching the credential deny rule are refused.

### 5.4 Adding a hardware platform

| piece | where | contract |
|---|---|---|
| device probe | `probes/device.py`: `probe_<vendor>()` | fill the neutral fields (product, arch, count, memory, clocks, power, driver, runtime, `vendor_extras`) and `platform_id = <vendor>-<product>.<runtime><version>`; refuse rather than guess (`probe_amd`, `probe_tpu` are the templates) |
| measurement wrapper | `lib/collectors.sh` | `collector_available`, `collector_version`, `collector_wrap` (sets `COLLECTOR_ARGV`), `collector_export` (leaves what the adapter reads in `<raw>/prof/`) |
| environment | `lib/collectors.sh: backend_env_allow` | the variable NAMES the runtime needs (HIP and XLA lists exist) |
| adapter | `collectors/<name>.py`, registered in `collectors/__init__.py` | `NAME`, `RUNTIME`, `CAPABILITIES` (only what the trace really shows), `VERIFIED`, `open(raw_dir)` returning a Trace with `info()`, `markers()`, `intervals()` (sorted, one timeline), `op_names(keys)`, `runtime_calls(windows)`, `close()`; `amd_rocprofv3.py` and `tpu_xprof.py` document the mapping for their platforms |
| automatic choice | `lib/engine.sh: engine_setup` | map the `platform_id` prefix to the collector (today `nvidia-*` -> `nvidia_nsys`, anything else -> `none`) |
| admission | `probes/conformance/run_conformance.sh --collector <name> --backend <B>` | must reproduce `expected.json` exactly; writes `platforms/<platform_id>.json`; set `VERIFIED = True` only after it passes |

Until an adapter exists, `--collector none` measures the ROI time and the FOM on any
hardware; the page then shows the platform as a column and its device values as null.

### 5.5 Commands

| command | purpose |
|---|---|
| `tools/timing/measure_level1.sh --build-root build/gcc13 all\|<bm>\|<bm>/<case>` | Level 1: 1 warm-up + 5 clean + 1 profiled per case; options `--clean-runs`, `--warmup`, `--no-profile`, `--collector`, `--backend`, `--keep-verify`, `--raw-root`, `--results-root`, `--no-summary`, `--dry-run` |
| `tools/timing/measure_level2.sh all\|<app>\|<app>/<case>` | Level 2: 1 clean + 1 profiled per case; `--env-script` (or `HPCPERF_TIMING_ENV_SCRIPT`) and the same options |
| `tools/timing/measure_level3.sh all\|<app>\|<app>/<case>` | Level 3: the same protocol and options; the region is the application's own timer; QMCPACK is not profiled unless `--profile-all` |
| `python3 tools/timing/apptimers.py describe\|extract <app> <dir>` | Level 3 timer definitions / extraction from one run's evidence |
| `python3 tools/timing/summarize.py [--run-id ID] [--csv-only] [--no-report]` | raw -> JSON / CSV / page (the front-ends run it on their own run) |
| `python3 tools/timing/report.py [--publish \| --out DIR]` | the page; `--publish` writes `docs/timing/` |
| `python3 tools/timing/cases.py check \| resolve --level N ... \| allowed-env <app>` | validate / resolve the case tables |
| `python3 tools/timing/gen_cases.py --build-root DIR [--check]` | regenerate / check the Level 1 case table against ctest |
| `tools/timing/probes/conformance/run_conformance.sh` | the platform admission test |
| `bash tools/timing/tests/run_all.sh` | 44 CPU-only checks |

### 5.6 Data formats (all versioned; details in `SCHEMA.md`)

| format | version | where |
|---|---|---|
| ROI log | `hpcperf-roi-log 2` | `<raw>/clean.<i>/roi.<pid>` |
| raw run directory | `hpcperf-timing-raw-2` | `build/timing/level<L>/<app>/<case>/<run_id>/` |
| run record | `hpcperf-timing-2` | `results/timing/level<L>/<app>/<case>/<run_id>.json` |
| CSV | fixed column list, new columns only appended | `results/timing/summary_level<L>.csv`, `ops_level<L>.csv` |
| device descriptor | `hpcperf-device-1` | inside every record |
| platform record | -- | `tools/timing/platforms/<platform_id>.json` |
| page | embedded JSON | `results/timing/report/`, `docs/timing/` |

## 6. Verified state (dgx003, 1x NVIDIA B200, CUDA 13.2, 2026-09-22)

* Correctness unchanged by the markers: Level 1 ctest 51/51 test directories, Level 2
  `validate.sh` 24/24 PASS, Level 2 infrastructure tests 69/69.
* Conformance `nvidia-b200.cuda13.2` with Nsight Systems 2025.6.3: 16/16 checks.
* Sweep: 51/51 Level 1 and 28/28 Level 2 cases measured. The ROI matches the
  applications' own timers for the same region within 0.01% in all 10 Level 2 cases
  that print one (xsbench 0.19%: it prints 3 digits). Level 1 clean-run spread median
  0.18%.
* `tools/timing/tests/run_all.sh`: 44/44.

## 7. Not done, and decisions still open

* **UNVERIFIED**: HIP/ROCm builds and the AMD collector (no ROCm on the node); TPU
  (interface only, by decision); the multi-process ROI (implemented; the allocation
  exposes one GPU, so Level 2 ran at one rank); hardware counters (`ncu`) are not
  collected. Level 3 is measured with the applications' own timers (the 43 registered inputs at
  1 GPU, 2026-10-02; the earlier hand-written 2-GPU sweep is history); its device picture is
  whole-process except for WarpX, one clean run per input (no spread), and the scientific
  correctness of each input is the registry's own check (PASS / INCOMPLETE / none per input, no
  invented tolerance -- see the page).
* **Decisions for the maintainers**:
  - Level 2 uses one clean run per case; quicksilver's ROI varies 4-7% run to run (its
    own timers agree), so `--clean-runs 3` or `5` may be worth the cost.
  - Several default Level 2 inputs are set-up dominated (MiniEM: 106.7 s before a
    1.31 s ROI; hipBone, SW4lite, XSBench similar); larger inputs would give more useful
    training points.
  - GitHub Pages: `docs/` is prepared (`.nojekyll`, a root `index.html` redirecting to
    `timing/`); enabling it (Settings -> Pages -> `main`, `/docs`) is a repository-admin
    decision, and it makes the measurements public unless the plan restricts Pages
    visibility. Until then the repository browser shows `docs/timing/README.md`.

## 8. Where to read further

| document | content |
|---|---|
| `tools/timing/README.md` | the method, headline fields, cases, hardware neutrality, clean environment, profiler cost, the web page, the recorded sweep |
| `tools/timing/roi/README.md` | marker API, placement rule, build integration, onboarding an application |
| `tools/timing/SCHEMA.md` | every data format |
| `tools/timing/collectors/*.py` docstrings | the adapter contract and the AMD / TPU mappings |
| `level2/<app>/README.md` `## Changes from upstream`, `level1/<bm>/README.md` | where each ROI sits and what it excludes |
| `docs/timing/` | the published results page |
