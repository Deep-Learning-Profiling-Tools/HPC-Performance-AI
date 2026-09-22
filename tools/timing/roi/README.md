# tools/timing/roi -- region-of-interest markers

The measured time of a benchmark is its **region of interest (ROI)**: the
computation from its first step to its last, without process start-up, set-up,
warm-up, verification and final output. The application marks it in its source;
everything in `tools/timing` measures between those marks.

| file | for |
|---|---|
| `hpcperf_roi.h` | C and C++ (CUDA, HIP, Kokkos, RAJA, ...), header-only |
| `hpcperf_roi.f90` + `hpcperf_roi_fortran.c` | Fortran: a `bind(C)` module and its C entry points |
| `hpcperf_roi.py` | Python workloads (JAX, PyTorch, ...) -- the path a TPU workload would take |
| `hpcperf_vendor/nvtx3/` | NVTX v3 headers copied from CUDA 13.2 (Apache-2.0; `SHA256SUMS`) |

## API

```c
#include "hpcperf_roi.h"          /* before other includes; see "Strict ISO C" */

set_up();  warm_up();
HPCPERF_ROI_BEGIN_SYNC();         /* drain queued device work, then open the ROI */
for (step = 0; step < n; step++) {
    advance(step);
    print_energy(step);           /* a per-step scalar diagnostic stays inside */
    if (step % dump_every == 0) {
        HPCPERF_ROI_EXCLUDE_BEGIN_SYNC();   /* bulk output is carved out */
        write_snapshot(step);
        HPCPERF_ROI_EXCLUDE_END();
    }
}
HPCPERF_ROI_END_SYNC();
verify();  write_output();
```

| macro | effect |
|---|---|
| `HPCPERF_ROI_BEGIN()` / `HPCPERF_ROI_END()` | open / close the ROI. Nesting is allowed; only the outermost pair counts. The ROI may be entered many times; the entries sum |
| `HPCPERF_ROI_BEGIN_SYNC()` / `HPCPERF_ROI_END_SYNC()` | the same after a device-wide synchronize, so the ROI holds exactly the device work issued inside it |
| `HPCPERF_ROI_EXCLUDE_BEGIN()` / `HPCPERF_ROI_EXCLUDE_END()` | carve a stretch out of the ROI (ignored outside an ROI) |
| `HPCPERF_ROI_EXCLUDE_BEGIN_SYNC()` | an exclude that follows asynchronous device work: without the synchronize the tail of that work would be excluded with it |

Fortran: `use hpcperf_roi`, then `call hpcperf_roi_begin()`, `..._end()`,
`..._begin_sync()`, `..._end_sync()`, `..._exclude_begin()`,
`..._exclude_begin_sync()`, `..._exclude_end()`. Compile `hpcperf_roi.f90` with the
Fortran compiler and `hpcperf_roi_fortran.c` with any C compiler
(`-I tools/timing/roi`), link both (`level2/gamess_ri_mp2/build.sh` does).

Python: `import hpcperf_roi as roi`; `roi.begin(sync=)`, `roi.end(sync=)`,
`roi.exclude_begin(sync=)`, `roi.exclude_end()`, the context managers
`roi.region(sync=True)` and `roi.excluded(sync=False)`, `roi.set_device_sync(fn)`
(how to drain the device: `jax.effects_barrier`, `torch.cuda.synchronize`, ...) and
`roi.add_annotator(push, pop)` for profiler ranges. `jax_annotator()` and
`torch_annotator()` adapt the frameworks' trace annotations and are **UNVERIFIED**
(no JAX, PyTorch or TPU on the node this was written on).

## Off by default

Unless `HPCPERF_ROI_LOG` is set or a profiler is injected into the process
(`NVTX_INJECTION64_PATH`, `ROCP_TOOL_LIBRARIES`, `ROCPROFILER_REGISTER_LIBRARY`),
every marker is a few-nanosecond no-op: no synchronization, no file, no output.
ctest and `validate.sh` set neither, so correctness runs are unaffected; the
`_SYNC` variants synchronize only while measuring.

The synchronize needs no GPU header or link dependency: it calls
`cuCtxSynchronize` (works with a static cudart) or `hipDeviceSynchronize` from the
runtime the process has **already** loaded (`dlopen(RTLD_NOLOAD)`).

## Two consumers of one set of markers

* **Clean timing, no profiler.** `HPCPERF_ROI_LOG=<prefix>` makes each process
  append its record to `<prefix>.<pid>`: a header (pid, MPI rank, host, exe, cwd,
  argv) and one `B`/`E` line per ROI entry with `CLOCK_MONOTONIC` and
  `CLOCK_REALTIME` nanoseconds. Excludes are **summed per entry** into one `x` line
  (so an exclude inside a 100 000-step loop costs no buffer). Events are buffered
  in memory and written at exit, or after an ROI ends once the buffer is half full
  -- never inside an ROI. Format: `tools/timing/SCHEMA.md`.
* **A profiler.** The markers are also named ranges, `hpcperf:roi` and
  `hpcperf:exclude`, through the backend's annotation API, so the analysis clips
  device activity to exactly the same region.

Annotation backend, chosen at compile time:

| backend | when | status |
|---|---|---|
| NVTX v3 (vendored, header-only) | default | verified with Nsight Systems 2025.6.3 (conformance probe) |
| ROCTX via `dlopen` (`librocprofiler-sdk-roctx.so.1`, `libroctx64.so`) | `__HIP_PLATFORM_AMD__` or `-DHPCPERF_ROI_ROCTX` | **UNVERIFIED** (no ROCm here); without the library the clean timing still works |
| none | `-DHPCPERF_ROI_NO_ANNOTATION` | clean timing only |

## Placement rule

1. **The ROI is the computation the application performs for its result**, from
   its first step to its last: the time loop, the solve, the timed repetitions.
   Algorithmic set-up that is part of the method stays inside (AMG hierarchy
   construction, per-step assembly); program set-up stays outside (reading or
   generating the input, allocation, the one-time upload of the input to the
   device).
2. **Warm-up is outside.** Where the application warms up in the same loop,
   `BEGIN` sits at the first timed iteration (`minibude`:
   `if (i == p.warmupIterations) HPCPERF_ROI_BEGIN_SYNC();`).
3. **Per-step work stays inside**, including copies made on every step, halo
   exchanges, reductions and per-step scalar diagnostics (energies, norms,
   residuals, progress lines): they are part of how the application runs.
4. **Excluded inside the ROI:** bulk file output (snapshots, dumps,
   visualization, checkpoints) and verification (checks against a reference or
   an analytic solution). Final verification and final output simply come after
   `END`.
5. **Where upstream already times the computation, its region is the anchor**
   (XSBench's "section that should be profiled", the P3 `Total` timer, ExaCMech's
   `run_time`, miniWeather's `t1`/`t2`). Where upstream's region also holds set-up,
   the ROI is narrower (SW4lite: the time-stepping loop, not `time_start_solve`).
6. **`_SYNC` at the boundaries** whenever device work may be queued there, and
   `EXCLUDE_BEGIN_SYNC` before an exclude that follows asynchronous work.
7. **Pure insertions.** No upstream line is changed or removed. A conditional
   exclude repeats the condition rather than adding braces:
   ```c++
   if(input->dumpbinaryflag) HPCPERF_ROI_EXCLUDE_BEGIN_SYNC();
   if(input->dumpbinaryflag)
     dump_binary(step);
   if(input->dumpbinaryflag) HPCPERF_ROI_EXCLUDE_END();
   ```
8. **Threads and ranks.** `BEGIN`/`END` on the thread that drives the device (NVTX
   ranges are per thread). Every MPI rank marks the same region; the job's ROI is
   the slowest rank's (multi-rank is UNVERIFIED on the node this was built on: one
   GPU is visible).
9. **Inseparable checks are declared, not hidden.** NPB `is` verifies inside its
   timed kernels; that is recorded as `verify_vs_roi=inside` in
   `cases/level1_apps.tsv` and every record carries a caveat.

Each application's placement is described where it is made: a comment at the
markers, the `## Changes from upstream` section of each Level 2 README, and the
`roi_excludes` / `verify_vs_roi` columns of `tools/timing/cases/level*_apps.tsv`.

## Build integration

* Level 1: every `level1/<bm>/CMakeLists.txt` adds
  `target_include_directories(<target> PRIVATE .../tools/timing/roi)`.
* Level 2: every `level2/<app>/build.sh` exports
  `CPATH="$R/tools/timing/roi${CPATH:+:$CPATH}"` (GCC, Clang and nvcc honor it),
  so no upstream build file changes.
* Linking: glibc >= 2.34 has `dlopen` in libc; older systems need `-ldl`.

### Strict ISO C

`-std=c99`/`-std=c11` hide POSIX `clock_gettime`. The header requests
`_POSIX_C_SOURCE` only in that mode and only if no system header came first; in
the gnu modes and in C++ nothing is defined, so the application's own code keeps
`M_PI` and friends. Included after a system header in strict mode it stops with
its own `#error` instead of silently losing the clock.

## Onboarding a new application

1. Place the markers following the rule above; describe the placement next to
   them and in the application README.
2. Put `tools/timing/roi` on the include path (Level 1 CMake, Level 2 `CPATH`).
3. Build, then run its correctness check unchanged (ctest / `validate.sh`).
4. `bash tools/timing/tests/run_all.sh` -- group 6 fails on an application whose
   sources lack a BEGIN or an END, or whose build does not see the header.
5. Add its case rows (`tools/timing/cases/`) and measure. A clean run that writes
   no ROI record is `roi_missing`, a **failure**, never a silent fallback to the
   process wall clock.
