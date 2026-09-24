# Registered benchmark inputs (`inputs.yaml`, schema `hpcperf-inputs-1`)

A benchmark that offers more than its historical default problem carries an
`inputs.yaml` next to its `run.sh` (Level 2/3) or `CMakeLists.txt` (Level 1).
`tools/inputs/hpcperf_inputs.py` reads it; nothing else in the repository
interprets the file. The default command of every benchmark is unchanged: an
input is selected only through the benchmark's documented selector variable
(`HPCPERF_HIPBONE_INPUT`, `HPCPERF_QUICKSILVER_INPUT_ID`, `HPCPERF_LAMMPS_INPUT`, ...),
and `run.sh` refuses an input id together with the knobs or extra arguments it
would override (nothing is silently replaced).

Registered so far (branch `inputs/upstream-pilot`; every other benchmark has not
been wired into this registry yet -- that says nothing about how many inputs it
could have):

| Level | Benchmark | Selector | Inputs | Covers | batch |
|---|---|---|---|---|---|
| 1 | `background_subtraction` | none (the tool runs the binary) | 4 | implementation path, frame size, frame count | 1 |
| 2 | `hipbone` | `HPCPERF_HIPBONE_INPUT` | 5 | 3 of the 15 upstream polynomial-degree sweep points, README size, larger derived size | 1 |
| 2 | `quicksilver` | `HPCPERF_QUICKSILVER_INPUT_ID` | 4 | official CORAL-2 P1/P2 and CTS-2 decks + the derived default | 1 |
| 3 | `lammps` | `HPCPERF_LAMMPS_INPUT` | 5 | same case at 3 sizes (32k/2M/16M atoms) + two other bench cases (EAM, rhodopsin) | 1 |
| 2 | `tealeaf` | `HPCPERF_TEALEAF_INPUT` | 5 | upstream `Benchmarks/` decks: 1000^2 / 2000^2 / 4000^2 / 8000^2 cells at 10 steps, 4000^2 at 2 steps (each with its own `tea.problems` reference) | 2 |
| 3 | `sparta` | `HPCPERF_SPARTA_INPUT` | 6 | the three upstream bench decks (collide / free / sphere) at the sizes upstream ships reference logs for (10K, 100K, 1M, 10M particles) | 2 |
| 2 | `cloverleaf` | `HPCPERF_CLOVERLEAF_INPUT` | 5 | upstream `InputDecks/` at 960^2 / 1920^2 / 3840^2 cells for 87 steps (built-in references), the 3840^2 x 2955-step default and the 15360x7680 87-step deck | 3 |
| 2 | `laghos` | `HPCPERF_LAGHOS_INPUT` | 6 | upstream README verification rows (Taylor-Green 3D, Sedov 2D/3D, triple point 3D, Rayleigh-Taylor 2D) + the single-GPU FOM deck | 3 |
| 3 | `lammps` (+1) | `HPCPERF_LAMMPS_INPUT` with `HPCPERF_LAMMPS_VARIANT=reaxff` | 1 | CORAL-2 ReaxFF HNS crystal (examples/reaxff/HNS) on the separate `reaxff.cuda` build profile | 3 |

## What an entry records

```yaml
schema: hpcperf-inputs-1
benchmark: hipbone            # level: 1|2|3
selector: HPCPERF_HIPBONE_INPUT
default_input: coral2-nx24-p14           # the input the default command corresponds to
entry: {kind: run.sh, path: level2/hipbone/run.sh, backend_arg: CUDA}   # or {kind: binary, path: build/...}
timing:                        # how the benchmark's OWN timer is read
  scope: <where the timer starts/ends, what it includes and excludes>
  kind: total | per_iteration | per_step
  unit: s | ms | us | ns
  regex: '^hipBone: \d+, \d+, (?P<value>[0-9.eE+-]+), \d+,'
  select: only | first | last  # which matching line when a log has several
  section_start: '^Timer\s+Cumulative'   # optional: ignore lines before this one
  work: {key: repeat, offset: -2}        # per_iteration only: main_compute_s = value * (params[key] + offset)
  secondary:                   # optional sub-interval timers, reported NEXT TO main, never added or substituted
    - {name: cycleTracking, scope: <text>, kind: total, unit: us, section_start: ..., regex: ..., select: only}
baseline:                      # scientific quantities kept per input and the comparison rule
  method: <text>; reference: <text>
  quantities:
    - {name: r_norm_final, regex: '^CG: it 100, r norm (?P<value>[0-9.eE+-]+)', select: only, compare: {rule: abs_lt, value: 1.0e-8}}
    # rules: exact | rel (tol) | abs (tol) | abs_lt/ge/le (value) | present | absent | record (kept, tolerance TBD)
inputs:
  - id: sweep-nx16-p8          # stable id defined by workload content, never by a measured time
    case: upstream-sweep-2m-dofs
    variant: size | case | parameter | implementation-path | steps | build-config | default
    source:
      kind: upstream-file | upstream-parameterized | derived | custom
      upstream: {repo, version, path, sha256}     # for the two upstream kinds
      derivation: <how it differs from upstream>   # for derived/custom
    params: {...}              # the actual problem parameters (run.sh reads them with `param`)
    args: [...]                # binary entries / hipBone: the argument list (`args`)
    files: [...]               # repository files the input needs (checked at validate time)
    runtime_config: {gpus: 1, ranks: 1}
    build_config: null         # or the build configuration a compile-time input needs
    backends_validated: [cuda]
    baseline: {quantities: [...]}   # optional per-input override (another output format)
```

Source kinds: `upstream-file` = a file upstream ships, used verbatim (the sha256 is
recorded); `upstream-parameterized` = upstream's own generator / parameter set /
documented run line; `derived` = built from upstream parameters at a point upstream
does not publish (the `derivation` says what changed); `custom` = new data or workload
(last resort, needs a stated reason).

## Coverage status, compile-time inputs, timing support (round 5, all 84 benchmarks)

Every active benchmark directory (`level1/*`, `level2/*`, `level3/*` except the retired
GEOS) carries an `inputs.yaml`. Two optional top-level/per-input additions, all
backward compatible:

```yaml
coverage:                      # the benchmark's multiple-input status (validated)
  status: MULTI_INPUT | SINGLE_INPUT | BLOCKED
  reason: <why SINGLE/BLOCKED, or what the MULTI set is>
  blocker: <what is missing to add the remaining upstream inputs>   # required for BLOCKED
  upstream_inputs_not_added: [<upstream input/case that exists but is not registered>, ...]
timing: {kind: none, status: NEEDS_TIMING_SUPPORT, reason: <what the benchmark prints instead>}
inputs:
  - id: class-a
    input_form: runtime | file | compile-time   # default runtime
    build_config: {CLASS: A, ...}               # required for compile-time inputs; part of the workload identity
    materialized: false                          # compile-time only: registered, NOT runnable (build_command refuses it)
    binary: build/cg/cuda-classA/cg_cuda         # a materialized compile-time input's own binary
```

`coverage.status` is checked against the number of *runnable* (materialized)
inputs: `MULTI_INPUT` needs at least two, `SINGLE_INPUT` means exactly one,
`BLOCKED` at most one (upstream offers more, the `blocker` says what is missing).
A benchmark with two or more runnable inputs and further upstream inputs that were
not added is `MULTI_INPUT` with `upstream_inputs_not_added`.

An input may carry its own `timing:` block (same fields, validated the same way) when its output
differs from the benchmark's usual one: a sweep input that prints one timer block per configuration
(miniBUDE's all-PPWI default: its `best:` summary line is timed), or an upstream deck too short for the
benchmark's timed window (HACCabanaPM's 5-step demo indat: `kind: none`). `measure` records
`timing_override: true` for such inputs.

`timing.kind: none` (with `status: NEEDS_TIMING_SUPPORT` and a `reason`) records a
benchmark whose stdout carries no usable native timer; `parse-timing` fails with
`NEEDS_TIMING_SUPPORT: ...`, `measure` still records the runs (exit 0 when they
complete) with `timing_status: NEEDS_TIMING_SUPPORT` and no `main_compute_s`; the
E2E wall is auxiliary only.

Compile-time inputs: the NPB ports (`cg ep ft is mg`) are built for CLASS=B in the
default build directory; classes A and C are materialized by
`tools/inputs/npb_materialize_class.sh <bench> <class>` (the vendored sources with
`level1/<bench>/inputs/npbparams.<class>.hpp` -- the class-B header with upstream's
setparams table values for the class -- built into `build/<bench>/cuda-class<X>`).
miniWeather's grid / initial condition / simulated time are build variables:
`HPCPERF_MINIWEATHER_BUILD_TAG=<tag> level2/miniweather/build.sh CUDA` builds a
tagged directory and the registered input sets the same tag for run.sh.

Selecting a registered input in a `run.sh`: every Level 2/3 script applies
`HPCPERF_<APP>_INPUT=<id>` (`HPCPERF_SHAW_INPUT_ID`, `HPCPERF_SW4LITE_INPUT_ID`,
`HPCPERF_QUICKSILVER_INPUT_ID` where the plain name is an existing knob) through
`tools/inputs/hpcperf_input_selector.sh`: the input's `env` knobs are exported and
its `args` appended (`shell-env` prints them as `E<TAB>key<TAB>value` /
`A<TAB>arg` lines); a knob already set to another value is refused (exit 2, nothing
is overridden silently), an unknown id is refused (exit 2), and without the variable
the script behaves exactly as before. Arguments of a *binary* entry that name an
existing repository-relative path (`level1/...`, `build/...`) are passed as absolute
paths because `measure` runs every repetition in its own run directory; the registry
keeps the relative spelling (workload identity).

`tools/inputs/hpcperf_inputs_audit.py [--measurements DIR] [--blockers FILE]
[--md FILE] [--json FILE]` generates the coverage audit from the registries: one row
per benchmark (level, status, inputs, runnable, cases, size variants,
upstream/derived/custom, input forms, selector, native timing status, measured
inputs, correctness status, blocker) plus the SINGLE_INPUT / BLOCKED /
derived-custom / not-measured lists and the totals.

## Runtime measurement (tools/timing)

Runtime is measured by `tools/timing` over each benchmark's marked region of interest; the registered
inputs are measured with `tools/timing/measure_level<N>.sh --registry` (see `tools/timing/README.md`,
"Registered inputs"). `hpcperf_inputs.py identity <bench_dir> <input_id>` prints the identity that
links such a measurement to its registry entry (exit 3 when it cannot be established: a named file is
missing or a compile-time input is not materialized). The `measure` command below is the earlier
native-timer protocol (benchmark's own timer) and stays available for correctness baselines.

## Invalidated measurements (`INVALIDATED.json`)

A measurement found to have run another workload than the input it is filed under (for example the
registry arguments never reached the program) is not deleted or relabelled: its directory gets an
`INVALIDATED.json` (schema `hpcperf-invalidation-1`: reason, actual vs expected workload, fix commit,
replacement record, manifest). The marker covers everything below the directory -- measurement.json,
the baseline.json written from it, the run logs. `compare` refuses an invalidated baseline or candidate
log (exit 2), `migrate-baseline` refuses invalidated baseline or evidence (an invalidated record never
becomes a baseline, and no identity migration can relabel it), `status` prints `INVALIDATED` (exit 3),
and `hpcperf_inputs_audit.py` loads it as `INVALIDATED` -- not completed, no timing, no verdict -- and
lists it under `invalidated_inputs`. A corrected input needs a new measurement and its own correctness
evidence.

## Commands

```bash
python3 tools/inputs/hpcperf_inputs.py validate level2/hipbone
python3 tools/inputs/hpcperf_inputs.py list     level2/hipbone
python3 tools/inputs/hpcperf_inputs.py show     level2/hipbone sweep-nx16-p8
python3 tools/inputs/hpcperf_inputs.py args     level2/hipbone sweep-nx16-p8     # what run.sh uses
python3 tools/inputs/hpcperf_inputs.py param    level3/lammps  lj-2m x
python3 tools/inputs/hpcperf_inputs.py parse-timing level2/quicksilver <stdout.log> [--rc N] [--input ID]
python3 tools/inputs/hpcperf_inputs.py extract  level3/lammps  <log> --input rhodo-32k
python3 tools/inputs/hpcperf_inputs.py compare  level3/lammps  <baseline.json> <log> [--input ID] [--rc N]
python3 tools/inputs/hpcperf_inputs.py status   level2/hipbone <measurement.json>
python3 tools/inputs/hpcperf_inputs.py migrate-baseline level2/hipbone <old baseline.json> --input coral2-nx24-p14 --evidence <its measurement.json> --note "..."
python3 tools/inputs/hpcperf_inputs.py measure  level2/hipbone sweep-nx16-p8 --out <dir> [--warmup 1] [--reps 3] [--timeout 900] [--gpus 1]
```

`measure` is the pilot calibration run: one complete warm-up run (kept, not
counted) followed by N measured runs, one directory per run (`stdout.log`,
`result.json`; a run directory that already exists is removed first, so a stale
log is never read); `measurement.json` holds the command, host/GPU, git identity,
the timing scope, every raw value and the summary (median, min, max, MAD,
spread = (max - min) / median, whether the spread is at or below the timer's
print resolution). `baseline.json` stores the quantities of the first measured
run that exited 0 and passed the benchmark's own baseline-free checks, plus the
rules to compare a later run against them. Level 3 runs get
`HPCPERF_L3_RUN_SUBDIR=run.inputs.<id>.<label>` so application-written results
never overlap.

The summary keeps these statuses apart on purpose (a benchmark can complete
without anything having been verified):

| status | meaning |
|---|---|
| `run_completed` | every measured run exited 0 |
| `timing_ok` | every measured run yielded `main_compute_s` from the benchmark's own timer |
| `native_check` | PASS / FAIL / NONE -- the rules that need no baseline (`present`, `absent`, `abs_lt`, `ge`, `le`, i.e. the benchmark's own pass criteria) on every measured run; NONE when the input has none |
| `baseline_saved` | a working baseline was stored (never from a run whose native check failed) |
| `comparison_rules` | READY (every REQUIRED quantity has a verifying rule; diagnostic records allowed) / PARTIAL (a required quantity is still record) / NONE |
| `needs_validation` | the REQUIRED quantities whose rule is `record`: kept, not verified, tolerance still to be fixed; `diagnostic_recorded` lists the diagnostic ones |
| `baseline_verdict` | PASS / INCOMPLETE / FAIL of the measured runs against the working baseline (PASS only when every required quantity was verified) |
| `compute_ge_1s`, `stable` | median main compute >= 1 s (a reference value, not a gate); n >= 3 and spread <= 10 % |
| `baseline_self_consistent` | every measured run compares OK against the working baseline (`ok` = nothing failed; it is not a verdict -- see `baseline_verdict`) |

Quantity roles (`role: required | diagnostic`, default required): only required
quantities decide acceptance; a diagnostic quantity is reported and may stay
`record` without a tolerance (its absence is noted, never a failure).

`compare` acceptance (round 3): the JSON carries `ok` (no rule failed),
`complete` (every required quantity has a verifying rule), `verified`
(`ok` and `complete` and at least one required quantity actually compared),
`verdict` PASS / INCOMPLETE / FAIL, `required_pending`, `diagnostic_recorded`,
`failed`. Exit codes: **0 only for verdict PASS**; 1 = a rule or the candidate
run failed (never downgraded to "incomplete"); 2 = refused -- the baseline belongs
to another input or benchmark, was recorded for a different workload (registry
params / args / env / input-file sha256 differ under the same id; the binary, git
revision and build fingerprint are deliberately NOT part of this identity, so an
optimized build compares normally), or baseline and candidate are the same output
file; 3 = INCOMPLETE -- nothing failed, but the comparison is not complete: a
required quantity is still `record` (hipBone's final residual, Quicksilver's scalar
flux, SPARTA sphere's particle count) **or the workload identity of the two sides
is not established**. Passing configuration checks (iterations, DOFs, step counts,
markers) never stand in for a pending required result. `measure` mirrors this as
`comparison_rules` READY / PARTIAL / NONE and `baseline_verdict` PASS / INCOMPLETE /
FAIL, computed over the independent runs only (`baseline_from_run`,
`independent_runs_compared`: the baseline run is never compared with itself);
`status` re-derives the vocabulary from a `measurement.json` (also for files written
by earlier rounds).

Workload identity (round 4). A comparison is formal only when baseline and
candidate are known to be the same workload: the baseline carries a complete
`workload` (input_id, params, args, env, input-file sha256 -- written by `measure`)
equal to the candidate input's registry identity, or an upstream reference carries
a `reference_binding` (`bound_input`, `source`, `evidence`, `adapted_by`) to this
input. A record without identity, with an incomplete identity, or with a binding
that lacks evidence is read and shown but returns 3 / INCOMPLETE with
`workload_status: not-established`; a contradicting identity (other params/args/
files under the same id, or a binding to another input) returns 2; a failing rule
returns 1 regardless. The candidate's identity is never copied into an old
record: `migrate-baseline <bench_dir> <baseline.json> --input ID --evidence
<measurement.json> [--note ..] [--out F]` attaches the identity only when the
evidence measurement names the same benchmark, input id, inputs.yaml sha256 and
command/selector, writes a NEW file and records `workload_migration` (source,
evidence, basis, note, time). The binary, git revision and build fingerprint are
recorded (`code_identity`) but never compared -- an optimized build is compared
against the baseline of the same workload by design.

What the tool never does: it never reports wall time as `main_compute_s` (a
failed run, a missing, ambiguous or non-finite timer line is an error), never
adds secondary timers to the main one, never treats a `record` quantity as
verified, never takes a baseline from a run that printed its own FAIL marker
even when it exited 0 (background_subtraction does exactly that), never renames
an input because a measurement changed, and never touches `level3/<app>/src`.

Tests: `tools/inputs/tests/run_all.sh` (no GPU; fixtures under `tests/fixtures/`,
run.sh guards through `HPCPERF_DRY_RUN=1`, and a fake benchmark under a temporary
repository root for the `measure` negative cases: nonzero exit, exit 0 with a FAIL
marker, NaN, missing timer line, stale log, record-only rules).
