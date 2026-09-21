# Registered benchmark inputs (`inputs.yaml`, schema `hpcperf-inputs-1`)

A benchmark that offers more than its historical default problem carries an
`inputs.yaml` next to its `run.sh` (Level 2/3) or `CMakeLists.txt` (Level 1).
`tools/inputs/hpcperf_inputs.py` reads it; nothing else in the repository
interprets the file. The default command of every benchmark is unchanged: an
input is selected only through the benchmark's documented selector variable
(`HPCPERF_HIPBONE_INPUT`, `HPCPERF_QUICKSILVER_INPUT_ID`, `HPCPERF_LAMMPS_INPUT`, ...),
and `run.sh` refuses an input id together with the knobs or extra arguments it
would override (nothing is silently replaced).

First batch (2026-09-21, branch `inputs/upstream-pilot`):

| Level | Benchmark | Selector | Inputs | Covers |
|---|---|---|---|---|
| 1 | `background_subtraction` | none (the tool runs the binary) | 4 | implementation path, frame size, frame count |
| 2 | `hipbone` | `HPCPERF_HIPBONE_INPUT` | 5 | upstream polynomial-degree sweep, README size, larger size |
| 2 | `quicksilver` | `HPCPERF_QUICKSILVER_INPUT_ID` | 4 | official CORAL-2 P1/P2 and CTS-2 decks + the derived default |
| 3 | `lammps` | `HPCPERF_LAMMPS_INPUT` | 5 | same case at 3 sizes (32k/2M/16M atoms) + two other bench cases (EAM, rhodopsin) |

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

## Commands

```bash
python3 tools/inputs/hpcperf_inputs.py validate level2/hipbone
python3 tools/inputs/hpcperf_inputs.py list     level2/hipbone
python3 tools/inputs/hpcperf_inputs.py show     level2/hipbone sweep-nx16-p8
python3 tools/inputs/hpcperf_inputs.py args     level2/hipbone sweep-nx16-p8     # what run.sh uses
python3 tools/inputs/hpcperf_inputs.py param    level3/lammps  lj-2m x
python3 tools/inputs/hpcperf_inputs.py parse-timing level2/quicksilver <stdout.log> [--rc N] [--input ID]
python3 tools/inputs/hpcperf_inputs.py extract  level3/lammps  <log> --input rhodo-32k
python3 tools/inputs/hpcperf_inputs.py compare  level3/lammps  <baseline.json> <log>
python3 tools/inputs/hpcperf_inputs.py measure  level2/hipbone sweep-nx16-p8 --out <dir> [--warmup 1] [--reps 3] [--timeout 900] [--gpus 1]
```

`measure` is the pilot calibration run: one complete warm-up run (kept, not
counted) followed by N measured runs, one directory per run (`stdout.log`,
`result.json`); `measurement.json` holds the command, host/GPU, git identity,
the timing scope, every raw value and the summary (median, min, max, MAD,
relative spread), plus the flags `run_ok`, `timing_ok`, `compute_ge_1s`,
`stable` (spread <= 10 %) and `baseline_self_consistent`. `baseline.json` stores
the quantities of the first measured run and the rules to compare a later run
against them. Level 3 runs get `HPCPERF_L3_RUN_SUBDIR=run.inputs.<id>.<label>` so
application-written results never overlap.

What the tool never does: it never reports wall time as `main_compute_s` (a
failed run, a missing, ambiguous or non-finite timer line is an error), never
adds nested timers, never renames an input because a measurement changed, and
never touches `level3/<app>/src`.

Tests: `tools/inputs/tests/run_all.sh` (no GPU; fixtures under `tests/fixtures/`,
run.sh guards through `HPCPERF_DRY_RUN=1`).
