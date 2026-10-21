# VPIC (vpic-kokkos, VPIC 2.0) -- relativistic electromagnetic particle-in-cell

VPIC is LANL's 3D3V fully relativistic kinetic particle-in-cell code (Bowers et al. 2008; VPIC 2.0, Bird et al.
2021): explicit leapfrog particle push (Boris), charge-conserving Esirkepov current deposition, a Yee-grid
FDTD field solve with Marder divergence cleaning, and per-rank domain decomposition over MPI. The Kokkos
version runs particles and fields on the device through Kokkos (CUDA here). An "input" is a deck: a C++
file with the problem set-up and diagnostics, compiled against `libvpic` into its own executable.

## Provenance

- Upstream: https://github.com/lanl/vpic-kokkos, branch `devel`, commit `7eb2947e12741df4931088a3ae6175adb21a6535`
  ("Merge pull request #113 from lanl/energies_plotter", 2026-10-13). The repository publishes **no release
  tags** (only `archive/*` tags of old feature branches), so the identity is this devel commit; reason for a
  devel snapshot: it is the only line of development, and it is the one that pins Kokkos 5 (upstream PR #127).
- Kokkos: the submodule upstream pins, `6d4434baacc86ec428b5300ebe853a085760d9af` (Kokkos 5.2.2), built by
  VPIC's own CMake (`BUILD_INTERNAL_KOKKOS=ON`, upstream's `arch/CUDA-Release` recipe).
- License: BSD-3-Clause (the "Copyright" section of `README.md`, Triad National Security, LLC, 2022; the
  repository ships no separate LICENSE file); Kokkos Apache-2.0 WITH LLVM-exception (`provenance/LICENSES.md`).
- Frozen source bundle (scheme 3, `provenance/source.lock.yaml`): `src/` = the upstream tree with its
  `kokkos/` submodule, no patches (`patch_series.txt` empty, `original_vs_baseline.diff` empty), 1786 entries,
  `source_tree_sha256 464de895...`, artifact `vpic-hpcperf-l3-v1.tar.zst` (1.8 MB, unpublished until the
  maintainer publishes it). `tools/prepare_benchmark.sh level3 vpic [--artifact FILE]` materializes it.
- `fetch.sh` (freeze time only) clones the pinned commit with the Kokkos submodule into `_upstream/level3/vpic-kokkos`.

## Build strategy: NATIVE (upstream CMake + the Kokkos submodule)

`build.sh [CUDA]`: `cmake` on `src/` with `-DCMAKE_CXX_COMPILER=src/kokkos/bin/nvcc_wrapper`
(`NVCC_WRAPPER_DEFAULT_COMPILER` = conda GCC 13.3.0), `BUILD_INTERNAL_KOKKOS=ON`, `ENABLE_KOKKOS_CUDA=ON`,
`Kokkos_ARCH_BLACKWELL100` (sm_100 instead of upstream's ADA89), `ENABLE_KOKKOS_OPENMP=OFF` (one host thread
per rank; the device does the work), `ENABLE_INTEGRATED_TESTS=ON` (the analytic test decks validate.sh runs),
`ENABLE_UNIT_TESTS=OFF` (Catch2 download), `VPIC_ENABLE_GPU_AWARE_MPI=OFF` (upstream default; host-staged
exchanges -- irrelevant at one rank), `-rdynamic` (upstream), Release. Modification class A: no upstream file
changed; cache entries only. Then every registered deck (`inputs.yaml`) is compiled with the build tree's own
deck compiler `bin/vpic` (the installed `bin/vpic` of upstream's `make install` points at a `deck/` directory
that the install does not create -- an upstream inconsistency; the build-tree script is what upstream's
documentation uses) into `build/level3/vpic/<profile>/decks/<id>.Linux`, next to a `<id>.deck.sha256` that
`run.sh` checks against the deck source before every run. Build time on dgx003: ~6 min (Kokkos + libvpic +
5 decks + the integrated tests) at `HPCPERF_BUILD_JOBS=24`.

Conda's `CFLAGS/CXXFLAGS/LDFLAGS` are cleared (`l3_clean_conda_build_env`): nvcc_wrapper would otherwise
inherit `-march=nocona` and the conda rpaths through CMake.

## Changes from upstream

None in the frozen source. The registered decks under `decks/` are this benchmark's files, derived from
upstream's `sample/harris`, `test/integrated/energy_comparison/weibel.deck` and `sample/short_pulse.cxx` as
`inputs.yaml` states for each (resolution / cell count, dump intervals, for the Weibel deck the box length
scaled with the cell count so that the cell size and the physical duration of its 700 steps stay upstream's,
and for the short pulse the rank topology); the physics, units, boundary conditions, particle loading and
step-count rules are upstream's.

## Execution model

One MPI rank per GPU through the common launcher (`--gpus 1 --bind wrapper`: the process sees one GPU, the
mapping is audited). Every registered deck fixes a `1 x 1 x 1` rank topology (compile-time), so `run.sh`
refuses `HPCPERF_GPUS` other than 1 and any scale mode other than `smoke`: a rank count the deck cannot
partition is never silently changed, and a multi-GPU deck would be another registered input. Runs live under
`build/level3/vpic/<profile>/<run subdir>/<id>.np1/`; the deck's energies diagnostic (`energies`, or
`rundata/energies` for the short pulse) is written there, `run_manifest.txt` records the deck and its sha256,
the executable and its sha256, num_step, grid and particle counts.

## Registered inputs (`inputs.yaml`, `HPCPERF_VPIC_INPUT`)

| id | problem | grid | particles | steps | timed loop (B200, campaign median 2026-10-21) | correctness reference |
|---|---|---|---|---|---|---|
| harris-256sq | Harris-sheet magnetic reconnection (sample/harris), nx = ny = 256 | 256 x 256 x 1 | 4.2 M | 1939 | 5.36 s | CPU (Kokkos OpenMP) run of the same deck |
| harris-512sq (default) | same, 512 | 512 x 512 x 1 | 16.8 M | 3878 | 28.2 s | CPU run of the same deck |
| harris-1024sq | same, 1024 | 1024 x 1024 x 1 | 67.1 M | 7757 | 190.8 s | this build's own frozen run (no CPU run of this size) |
| weibel-1d-131072 | counter-streaming (Weibel / two-stream) instability of the energy_comparison test deck, 131072 cells, box scaled with the cell count (upstream's cell size and physical duration) | 131072 x 1 x 1 | 52.4 M | 700 | 10.4 s | CPU run of the same deck |
| shortpulse-2d-500sq | 2-D laser-plasma short pulse (sample/short_pulse.cxx, RUN3D = 0), one rank | 500 x 1 x 500 | 6.7 M (slab) | 8652 | 60.9 s | statistical group rule against this build's frozen run (five realisations differ by ~10 %) |

The upstream decks as shipped (32^2 Harris with dumps every status interval, the 16-cell Weibel test) run in
seconds dominated by I/O and are not registered; `validate.sh` runs upstream's own analytic and completion
decks (below). The three Harris sizes are the scaling series (dt follows the Courant limit, so the step count
doubles with the resolution); `harris-1024sq` is the large input of the series.

## Timed region (`tools/timing/apptimers.py: vpic`)

`*** Done (<s>s / ... elapsed)`: `deck/main.cc` takes `wallclock()` immediately before and after
`while( simulation->advance() )` (lines 123-127, printed at line 137). `advance()` returns 0 without work once
`step() >= num_step` (`src/vpic/advance.cc:17`), so the loop holds exactly num_step steps -- sort, push,
boundaries, current accumulation, field advance and the deck's `user_diagnostics` (the energies row every
energies_interval steps); initialization, particle loading, the Run Details banner and clean-up are outside.
Every phase of `advance()` ends in a `KOKKOS_TOC` with a `Kokkos::fence` (`src/util/profile/profile.h`),
so the host clock sees the device work. No NVTX range: the profiled run's device picture is whole-process context.

## Validation (`validate.sh`, upstream mechanism)

Upstream's integrated test decks (`test/integrated/legacy`, built with `ENABLE_INTEGRATED_TESTS=ON`), each
with its analytic expectation checked inside the deck and `pass` / `FAIL` + abort as the outcome:
`accel` (a particle in a uniform E field: ux after each step must be the exact multiple of the impulse),
`cyclo` (gyration in a uniform B field: |u - u_exact| <= 0.6e-6 after every step), `inbndj` (current
deposition of a boundary-entering particle: position and divergence tests within `tol * eps`), `interpe`
(field interpolation against the exact linear interpolation), plus `to_completion/simple` (a small Harris
run that must reach `*** Done`). The harness adds exit code, timeout, the `ENABLE_KOKKOS_CUDA: ON` build
banner (the GPU backend) and the launcher audit. `pcomm` (8 ranks) and `outbndj` are not built by upstream's
CMake and are not run. Result (2026-10-21, 1 GPU): PASS -- accel, cyclo, inbndj, interpe print their `pass`,
`to_completion/simple` reaches `*** Done`; 13 s in all (the launcher audit reports the ranks as `unverified`: the
runs are too short for its nvidia-smi sampling).

### Correctness of the registered inputs (`check_energies.py`, frozen references)

The scientific record of a run is its energies history (`src/vpic/dump.cc dump_energies`: ex ey ez bx by bz
field energies and the kinetic energy of every species, every energies_interval steps, 6 significant digits).
`check_energies.py` compares it with a frozen reference (`reference/<id>.json`, provenance and the rule inside)
and checks total-energy conservation:

- at every sampled step, every column within `max(1e-5 relative, 1e-6 of the row's total energy)` of the
  reference; `|E_total(step) - E_total(0)| / E_total(0) <= 1e-3`.
- Basis (measured 2026-10-21 on harris-256sq, before any optimized candidate existed): the Kokkos OpenMP
  (CPU) build and this CUDA build, same deck, same particle loading (host-side RNG), agree at every sampled
  step to 5.4e-7 relative in the dominant columns (bz = 73 % of the total, the two species at 14 % each: the
  6 printed digits), i.e. 7e-8 of the total; the perturbation fields (ex, ey, ez, bx, by: 1e-7 .. 1.1e-4 of
  the total, growing from the reconnection perturbation and from noise) deviate by up to 4.2e-3 relative to
  themselves but by at most 4e-8 of the total; both realisations drift by 3.4e-5 in total energy over the
  1939 steps. So: 1e-5 relative is 20 x the print-precision agreement of the dominant columns, 1e-6 of the
  total is 25 x the largest small-column deviation and still 100 x below the end-state ey energy (1.1e-4 of
  the total), so a field evolution that is wrong or missing is caught; 1e-3 drift is 30 x the measured drift
  of the two correct realisations. The worst tolerance usage of the CUDA run against the CPU reference is
  5.4 % (species columns). The same rule is applied to every input whose two realisations satisfy it with a
  comparable margin; the reference of an input records its own observed usage (the check usages are in
  the results section below).
- References: harris-256sq, harris-512sq, weibel-1d-131072, shortpulse-2d-500sq from the Kokkos OpenMP
  (CPU) build of the same frozen source (independent realisation; `provenance.binary`); harris-1024sq from
  this unmodified CUDA build's own run (a CPU run of 67 M particles x 7757 steps is ~10 h; self-baseline of
  the unoptimized implementation, stated as such).
- The short pulse is reproducible only statistically: the laser-plasma interaction amplifies the
  non-deterministic particle reductions, and five realisations of the unmodified source (the CPU build and
  four runs of this CUDA build, same deck and loading) differ per column by up to 15 % and in the energy
  sums by up to 10 % (total), 11 % (particles), 12 % (fields) over the run (1e-3 at step 500, growing with
  the interaction). A per-column rule has no basis there, so its reference is frozen in the checker's
  `groups` mode: at every sampled step the sums total / particles / fields within 0.35 relative (3 x the
  largest observed sum deviation), no per-column and no conservation bound (open system: the laser injects
  energy). This is a coarse check by construction -- it distinguishes a wrong field solve, push or
  injection (order-1 changes) from a correct realisation, not two correct roundings.
- Three separate results per input: timing SUCCESS (the `*** Done` record of a clean run), workload
  verification PASS (`verify_registry_runs.py`: the manifest's deck sha256 = the registered deck, the
  executable's hash, the `Run Details` num_step / grid / particle counts and the launcher audit) and
  scientific correctness PASS (`hpcperf_inputs.py check` + `verdict`: the energies check of a fresh run of
  the same deck plus the stdout quantities).

## Results on dgx003 (B200, CUDA 13.2.78)

Campaign `l3-ext-2026-10-20` (2026-10-21 05:02-05:43 UTC, Slurm job 2732030, GPU `0000:9D:00.0` by UUID, final
Level 3 protocol: 0 warm-up + 3 clean + 1 nsys-profiled run per input, headline = median of the application
timer, spread = (max - min) / median with 10 % as the UNSTABLE threshold, no clock locking, strictly serial;
`docs/timing/index.html` has the current rows). The day's first campaign (01:27-01:55 UTC) measured the previous deck
build; `build.sh` recompiles every registered deck, so the Weibel deck change (03:17 UTC) left those records without
their executables on disk (`verify_registry_runs.py`: binary hash mismatch); they are set aside in the results tree
(`raw-superseded/`, README there) and every input was re-measured on the current build:

| input | timer median (s) | min / max (s) | spread | stable | profiled run (s) | run verification | correctness |
|---|---|---|---|---|---|---|---|
| harris-256sq | 5.357 | 5.333 / 6.129 | 14.9 % | no (UNSTABLE) | 6.366 | PASS | PASS (10.8 % of the rule) |
| harris-512sq (default) | 28.200 | 28.180 / 28.410 | 0.8 % | yes | 30.414 | PASS | PASS (5.4 %) |
| harris-1024sq | 190.822 | 190.714 / 190.953 | 0.1 % | yes | 195.286 | PASS | PASS (5.4 %) |
| weibel-1d-131072 | 10.404 | 10.403 / 10.440 | 0.4 % | yes | 10.765 | PASS | PASS (0.2 %) |
| shortpulse-2d-500sq | 60.895 | 60.879 / 60.902 | 0.0 % | yes | 66.536 | PASS | PASS (5.8 % of the group rule) |

- Timing SUCCESS for all five (`*** Done` of every clean run; `verify_registry_runs.py` PASS 5/5: manifest deck
  sha256 = the registered deck, the executable's hash, `Run Details` num_step / grid / particle counts, the
  launcher audit `1 verified, 0 mismatch`). harris-256sq is UNSTABLE by the 10 % rule: its first clean run took
  6.129 s, the next two 5.357 / 5.333 s (0.5 % apart); the set-aside record of the previous build shows the same
  pattern (5.918, then 5.346 / 5.331) and harris-512sq a faint one (28.410, then 28.180 / 28.200). The first run after
  an idle GPU is the slow one in every record; the cause is not established (UNKNOWN: the protocol has no warm-up run
  and no clock locking, and nothing else ran on the GPU). The median is unaffected (5.357 vs 5.346 s).
- The profiled run (nsys, CUDA trace) is 2-19 % slower than the clean median (harris-256sq 19 %: the shortest
  run, launch-bound); its device picture is the whole-process one (no NVTX range in VPIC).
- Correctness (`check_energies.py` against the frozen references, `run_checks.sh vpic` = one fresh run of each
  registered deck in `run.check.<id>`, then `hpcperf_inputs.py verdict`): all five PASS (2026-10-21 05:08-05:17 UTC,
  the executables of the campaign records). Worst tolerance usage: harris-256sq 10.8 % (the ion column), harris-512sq
  5.4 %, harris-1024sq 5.4 %, weibel-1d-131072 0.2 %, shortpulse-2d-500sq 5.8 % of the 0.35 group rule (the fields
  sum; total 2.6 %, particles 2.2 %); the short pulse has no conservation bound (open system: its total energy grows
  1.9e3-fold with the laser injection, in the run as in the reference).
- References: harris-256sq from the Kokkos OpenMP build (`vpic-cpu`, 563 s), harris-1024sq from this CUDA
  build's own `run.reference` run (660.7 s under a CPU-loaded node: the loop is host-latency sensitive; a
  reference's value does not depend on its speed); harris-512sq and weibel-1d-131072 from the Kokkos OpenMP build
  as well (3490 s and 1683 s, single-threaded), shortpulse-2d-500sq from this build's `run.reference` run with the
  statistical group rule (five realisations). Every reference's rule and basis are recorded in its JSON.


## Limitations

- One rank per deck: multi-GPU runs would need their own decks (topology is compile-time); multi-node
  BLOCKED/UNVERIFIED on this site; HIP path present in build.sh, untested (no ROCm).
- The upstream repository has no release tags: the frozen commit is a devel snapshot (stated above).
- `harris-1024sq` has no independent (CPU) reference; its check is a self-baseline of the unoptimized build.
- The deck-compile script upstream installs (`make install` -> `bin/vpic`) references a `deck/` directory the
  install does not create; the build tree's `bin/vpic` is used (recorded, not patched).

## Source distribution

Scheme 3 (`level3/EXTERNAL_ARTIFACT_DESIGN.md`): `provenance/source.lock.yaml` (schema hpcperf-source-lock-2),
artifact `vpic-hpcperf-l3-v1.tar.zst` in the maintainer's local staging, `publish_status: unpublished` until
published by the user. LOC: `provenance/LOC.md`.
