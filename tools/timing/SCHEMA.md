# tools/timing -- data formats

Every format carries a version. A reader refuses a version it does not know rather
than guessing; a change of meaning is a new version.

| format | version | written by | read by |
|---|---|---|---|
| ROI log | `hpcperf-roi-log 2` | the markers (`roi/`) | `analysis.py` |
| raw run directory | `hpcperf-timing-raw-2` | `lib/engine.sh` | `summarize.py` |
| canonical activity model | (code: `collectors/__init__.py`) | collector adapters | `analysis.py` |
| device descriptor | `hpcperf-device-1` | `probes/device.py` | the engine, `summarize.py` |
| platform record | (`platforms/<platform>.json`) | `probes/conformance/check.py` | `summarize.py` |
| run record | `hpcperf-timing-2` | `summarize.py` | you, `report.py` |
| web page | (`index.html` + `README.md`) | `report.py` | a browser / the repository browser |

## ROI log, version 2

One file per process, `<HPCPERF_ROI_LOG>.<pid>`, appended to in blocks:

```
# hpcperf-roi-log 2
pid 1572585
rank 0                                   OMPI_COMM_WORLD_RANK / PMIX_RANK / PMI_RANK / SLURM_PROCID, or -
clock CLOCK_MONOTONIC CLOCK_REALTIME
host dgx003
exe /abs/path/of/the/executable
cwd /abs/run/directory
argv ["./daxpy_cuda", "..."]             JSON array
B <monotonic ns> <realtime ns>           outermost ROI entry begins
E <monotonic ns> <realtime ns>           ... ends
x <excluded ns> <count>                  after the E of an entry that had excludes: their sum
U <monotonic ns> <realtime ns>           at exit: an ROI that was begun and never ended
overflow <n>                             events dropped (buffer full; the markers sit too deep)
unmatched_end <n>                        END without BEGIN
```

The ROI wall time of a process is `sum(E - B) - sum(x.excluded)` over its entries,
on the monotonic clock. The realtime stamps place the ROI inside the process
(`pre_roi_s`, `post_roi_s`). An unterminated entry is not counted and is caveated.
A job's ROI is the slowest process's; the spread is `imbalance_s`.

## Raw run directory (`hpcperf-timing-raw-2`)

`build/timing/level<L>/<app>/<case>/<run_id>/` (git-ignored):

```
run_meta.txt        key=value: schema, run_id, utc, level, app, case, backend, gpus, cwd,
                    timeout_s, case_env, argv, fom_*, roi_excludes, verify_vs_roi, notes,
                    roi_where, warmup_runs, clean_runs, profiled_runs, skip_verify, collector,
                    collector_version, env_script, env_allow, env_deny_regex, platform_id,
                    device_json, git_commit, git_dirty, exe_sha256 (Level 1), status
warmup.<i>/run.log  discarded
clean.<i>/run.log   stdout+stderr (the FOM and the launcher audit are read from clean.0)
clean.<i>/run.txt   start_ns= end_ns= rc=
clean.<i>/roi.<pid> ROI logs, one per process
prof/               the same for the profiled run, plus the collector's files
                    (nvidia_nsys: trace.nsys-rep, trace.sqlite, export.log)
```

`status`: `ok`, `clean_failed`, `clean_timeout`, `roi_missing` (a clean run wrote no
ROI record), `prof_failed`, `export_failed`; `summarize.py` adds
`roi_missing_in_trace` (the profiler did not see the markers). Only `ok` is a result.

## Canonical activity model

What a collector adapter delivers, whatever the vendor (`collectors/__init__.py`):

* `Marker(proc, kind, start, end)` -- `kind` is `roi` or `exclude`.
* `Interval(start, end, category, key, proc, nbytes)` -- one device operation.
* Categories, exhaustive: `compute`, `copy_h2d`, `copy_d2h`, `copy_d2d`,
  `copy_other`, `fill`, `collective`, `other`.
* `CAPABILITIES`: the categories the collector can observe. A category outside it
  is **null** in every output (not observable), never 0; 0 means observed and
  absent.
* All timestamps of one trace are nanoseconds on one timeline, and `proc` is the
  same key in markers and intervals of one process (the conformance probe checks
  both).
* `runtime_calls(windows)` -> `{calls, time_ns, sync_calls, sync_ns}` of the host
  runtime API (`cuda`, `hip`, `xla`), or None when the platform has no such notion.

| collector | platform | capabilities | status |
|---|---|---|---|
| `nvidia_nsys` | NVIDIA, Nsight Systems | compute, copy_h2d/d2h/d2d/other, fill | verified (conformance 16/16) |
| `none` | any | -- | verified: ROI time and FOM only |
| `amd_rocprofv3` | AMD, rocprofv3 | (contract in its docstring) | interface only, refuses |
| `tpu_xprof` | TPU, XLA profiler | (contract in its docstring) | interface only, refuses |

## Device descriptor (`hpcperf-device-1`)

Neutral names; what a vendor calls differently goes into `vendor_extras`.

```json
{"schema": "hpcperf-device-1",
 "device": {"vendor": "nvidia", "product": "B200", "arch": "sm_100", "uuid": "...",
            "count_visible": 1, "memory_total_mib": 183359,
            "core_clock_mhz": 120, "core_clock_max_mhz": 1965,
            "mem_clock_mhz": 3996, "mem_clock_max_mhz": 3996,
            "power_limit_w": 1000, "temperature_c": 24, "driver_version": "595.58.03",
            "runtime": {"name": "cuda", "version": "13.2"},
            "vendor_extras": {"compute_capability": "10.0", ...},
            "platform_id": "nvidia-b200.cuda13.2"},
 "host": {"hostname": "...", "cpu_model": "...", "cpus_allowed": 16, "kernel": "...",
          "arch": "x86_64", "loadavg_1m": 2.03, "mem_available_kb": ...}}
```

`platform_id` = `<vendor>-<product>.<runtime><version>` and is part of a
measurement's identity `(level, app, case, platform)`. The AMD and TPU probes are
interface-only: on such hardware they raise instead of describing it wrongly; on a
host where no probe applies the vendor is `unknown`.

## Platform record

`platforms/<platform_id>.json`, written by `probes/conformance/run_conformance.sh`:
the collector, its version, the probe's raw directory and every check with its
detail. A run on a platform without a passing record is caveated. The probe
(`probe.cu`) has a known split -- 10 warm-up launches, an ROI of 20 launches and one
device-to-device copy with 5 excluded check launches, then one more launch -- and
the collector must reproduce it exactly (`expected.json`).

## Run record (`hpcperf-timing-2`)

`results/timing/level<L>/<app>/<case>/<run_id>.json` (git-ignored):

| block | content |
|---|---|
| identity | `level`, `app`, `case`, `platform`, `run_id`, `utc`, `status` |
| `measurement` | tool, protocol (warm-up / clean / profiled runs), collector (name, version, verified, capabilities), `skip_verify`, `verify_vs_roi`, `roi_where` (marker source lines), `roi_excludes`, backend, gpus, timeout, env script, env allow-list and deny rule, notes |
| `inputs` | `declared_env`, `declared_argv`, and per process the argv / exe / cwd / host / rank the ROI log recorded, `exe_sha256` |
| `roi` | `wall_s` (median of the clean runs), `runs_s`, min / max / stddev, `entries`, `excluded_s`, `processes`, `imbalance_s`, `profiled_wall_s` (the same region in the trace), `profiled_marker_wall_s` (the profiled run's own ROI log), `profiler_inflation` |
| `device` | inside the ROI: `busy_s` (union of all device activity), `busy_frac_of_roi`, `host_gap_s` = `roi.wall_s - busy_s`, `op_time_sum_s`, `overlap_s`, per category `<cat>_s` and `<cat>_ops`, copy/fill `<cat>_bytes` (pro rata when clipped). Null without a collector |
| `runtime_api` | `name`, `roi` and `whole` call counts and time, synchronizing calls separately |
| `ops` | per device operation inside the ROI: name, category, count, total / avg / min / max, share |
| `context` | `process_wall_s`, `pre_roi_s`, `post_roi_s`, `whole_process` (device activity of the whole run) -- context only, never the headline |
| `fom` | the application's own metric from the clean run: name, value, unit, better, source, regex, status (`ok`, `none`, `not_matched`, `log_missing`) |
| `app_timer` | Level 2 only, null when the application prints no timer for its ROI region: `regex` (from `cases/level2_apps.tsv` at summarize time), `value_s` (clean run, last match), `roi_diff_frac` = (ROI - timer) / timer, `status`; a difference above 2% is caveated |
| `launcher` | the common launcher's GPU-binding audit line and whether it is clean |
| `platform_info` | device descriptor, host, conformance record |
| `profiler` | collector metadata; `recorded_env_name_count` (names only -- values are never read) |
| `provenance` | git commit, dirty flag, raw directory |
| `caveats` | every condition that limits how a number may be read, in words |

`summary_level<L>.csv` flattens one record per row with a fixed column list
(`summarize.py: COLUMNS`, new columns are only ever appended: `app_timer_s`,
`roi_vs_app_timer`); `ops_level<L>.csv` has one row per (run, operation).
Levels are separate files so runs under different protocols are never averaged
together. Both are regenerated from the JSONs, never appended.
