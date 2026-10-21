# Cholla (v3.0.0) -- GPU finite-volume hydrodynamics

Cholla (Schneider & Robertson 2015) is a 3-D Eulerian hydrodynamics code written for GPUs: the whole grid lives on the
device and every stage of the time step (PLMC reconstruction, the HLLC Riemann solver, the Van Leer predictor-corrector
update, the boundary conditions) is a CUDA kernel; MPI decomposes the grid over GPUs. This directory integrates the
`hydro` build type on one B200 as a Level 3 application: five registered inputs derived from upstream's `examples/3D`
(the Sod shock tube at two grid sizes, a linear sound wave over one period, the Noh implosion, a spherical blast), the
application's own per-step timer as the timed region, analytic / conservation criteria plus frozen self-baseline digests
as the correctness checks.

## Provenance

- Upstream: https://github.com/cholla-hydro/cholla, release tag `v3.0.0` = `8478c9ede8a5e2cb72d6481a50023666d489a763`
  (2024-02-02, "Merge pull request #370 from cholla-hydro/dev"); the latest tag -- the `main` branch holds four CI /
  docker commits after it and no code change. MIT license (`LICENSE.txt`).
- Frozen source bundle (scheme 3, `provenance/`): `cholla-hpcperf-l3-v1.tar.zst` (356 KB), source tree
  `44537e4f47e9a1c7f005561de3064b400560508f30bcd477db875b75d4527450`; `fetch.sh` re-fetches the tag for a re-freeze
  (freeze-time only). Excluded from the bundle: `docs/`, `docker/`, `python_scripts/`, `scale_output_files/` (not built,
  not run) and the `cholla-tests-data` submodule (the fiducial outputs of upstream's gtest system tests; the registered
  inputs are checked against analytic solutions and this build's own frozen references instead).
- One class-D patch (`patches/0001-makefile-git-hash-from-make-variable.patch`, `provenance/patch_series.txt`): the
  Makefile takes `GIT_HASH` as a make variable (`GIT_HASH ?= $(shell git rev-parse --verify HEAD)`), because upstream
  `static_assert`s a 40-character hash at compile time (`src/utils/error_handling.cpp:41`) and the frozen bundle is not a
  git checkout; `build.sh` passes the frozen upstream commit, so the hash Cholla writes into every output file is the real
  one. No source code is changed (`provenance/original_vs_baseline.diff`).
- LOC (`provenance/LOC.md`, cloc on the materialized tree): application-owned 49,984 lines in 205 files (C++ 17,509,
  CUDA 15,900, headers 4,965, build system 558; the rest text data of the source tree); 53,062 lines materialized in all.

## Build strategy: NATIVE (upstream GNU Makefile, `make TYPE=hydro MACHINE=hpcperf`)

`build.sh [CUDA]` copies `src/` to `build/level3/cholla/<profile>/src`, writes the host file
`builds/make.host.hpcperf` there in the style of upstream's `builds/make.host.*` (class A, a build-side configuration file:
`CXX = mpicxx` = conda GCC 13.3 + Open MPI 5.0.10, nvcc with `-ccbin=mpicxx`, `CUDA_ARCH = sm_100`, `HDF5_ROOT` and
`MPI_ROOT` = the conda environment, `CUDA_ROOT` = the toolkit of the nvcc on PATH (13.2.78)), and runs upstream's
`make TYPE=hydro MACHINE=hpcperf GIT_HASH=<frozen commit> -j16`. `builds/make.type.hydro` is upstream's, unchanged:
`-DMPI_CHOLLA -DPRECISION=2 -DPLMC -DHLLC -DVL -DDENSITY_FLOOR -DTEMPERATURE_FLOOR -DOUTPUT -DHDF5`. Upstream's optimize
flags are kept (`-Ofast -std=c++17` host, `-O3 -std=c++17` device, `--expt-extended-lambda -fmad=false` from the Makefile)
plus `-fPIE` / `-Xcompiler -fPIE`: the conda GCC links position-independent executables, so nvcc's host objects must be
compiled that way too (without it: "relocation R_X86_64_32 against .bss can not be used when making a PIE object"). The
conda build variables (`CFLAGS`, `LDFLAGS`, ...) are cleared as for every Level 3 build (they inject `-march=nocona`,
`-O2` and `--gc-sections` into upstream's recipe). Result: `.deps/level3/cholla/<profile>/install/bin/cholla`
(2.5 MB, 11 s to build; the only executable -- Cholla's inputs are run-time parameter files).

## Execution model

One MPI rank on one GPU per registered input (`HPCPERF_GPUS=1` only; the problem sizes were chosen for one B200 and a
rank count the registry did not validate is refused, not adjusted). `run.sh [CUDA]` runs
`cholla <registered parameter file> outdir=<run dir>/out/` through the common launcher with the per-rank GPU wrapper
(`hpcperf-launch` audit: the rank's GPU observed by UUID); the registered file is read in place from `inputs/` and
`outdir` is the only command-line override (upstream's own `key=value` mechanism). Cholla writes `out/0/0.h5.0` (the
initial state) and `out/1/1.h5.0` (the final state at `tout`; every registered input has `outstep = tout`), `run_output.log`,
and one `n_step: ... timestep time = <ms> ms   total time = <s> s` line per time step on stdout. The manifest records the
parameter file's hash, the problem (`init`), grid and `tout`, the executable's hash and the final output's hash.

## Registered inputs (`inputs.yaml`, `HPCPERF_CHOLLA_INPUT`)

| id | problem (upstream file) | grid | cells | tout | steps | timed loop (B200, campaign median) | correctness |
|---|---|---|---|---|---|---|---|
| sod-256 (default) | Sod shock tube, planar along x (`examples/3D/sod256.txt` as shipped) | 256^3 | 16.8 M | 0.2 | 402 | 7.32 s | exact Riemann solution (L1 rule) + frozen digest |
| sod-512 | the same at 512^3 (derived: nx = ny = nz = 512) | 512^3 | 134 M | 0.2 | 806 | 111.9 s | exact Riemann solution (L1 rule) + frozen digest |
| sound-wave-512 | linear sound wave, A = 1e-4, one period (derived from `examples/3D/sound_wave.txt`: 512^3, tout 1.0) | 512^3 | 134 M | 1.0 | 427 | 46.1 s | analytic sine, 0.1 A (upstream's one-period linear-wave rule) + frozen digest |
| noh-200 | Noh implosion (`examples/3D/Noh_3D.txt`; outstep 0.01 -> 2.0: only the final snapshot) | 200^3 | 8.0 M | 2.0 | 1709 | 11.0 s | analytic post-shock density 64 within 10 % + frozen digest |
| blast-256 | spherical over-pressure blast, periodic box (`examples/3D/Spherical_Overpressure.txt`; outstep 0.01 -> 0.1) | 256^3 | 16.8 M | 0.1 | 570 | 9.68 s | conservation of mass / momentum / energy (1e-9) + frozen digest |

`sod-256` is upstream's `examples/3D/sod256.txt` as shipped (its `outdir=./` line removed: `run.sh` supplies `outdir`);
the other four are derived from upstream's examples as stated per input in `inputs.yaml` (`source.derivation`): the grid
size (sod-512, sound-wave-512), the integration length (sound-wave-512: exactly one wave period instead of 1/20 of one),
and the snapshot cadence (noh-200, blast-256: the initial and the final output only instead of 10-200 intermediate
snapshots of 0.3-0.7 GB each; the computation is unchanged and the snapshot writes are outside the timed region in any
case). The step counts are the CFL sequence of each problem (deterministic) and are registered as workload evidence.
Pilot durations of the candidates (one run each, the sum of the per-step timer): the upstream 256^3 Sod 7.3 s, Noh 11 s,
the 256^3 sound wave over one period 3.1 s (too short: registered at 512^3, 46 s), the blast 9.7 s, the 512^3 Sod 112 s,
the 128^3 Kelvin-Helmholtz example 7.0 s in 3445 steps of 2 ms (not registered: no analytic criterion, latency-sized steps).

## Timed region (`tools/timing/apptimers.py: cholla`)

Cholla's own per-step clock: `src/main.cpp:247-325` takes `Get_Time()` before (line 252) and after (line 316) every
iteration of `while (G.H.t < P.tout)` -- `set_dt`, `Update_Hydro_Grid` (reconstruction, Riemann solver, Van Leer update),
`Update_Time`, `Set_Boundary_Conditions_Grid` -- and prints it as `timestep time = <ms> ms`. The timed region is the sum of
those values over all steps = the whole hydrodynamics loop; initialization (grid set-up, initial conditions, the initial
snapshot, everything before `Starting calculations.`) and the HDF5 output writes (`Write_Data`, line 344, after the
print) are outside. The per-step `total time` (wall since program start) minus the first step gives the set-up estimate
(5-22 s: MPI/CUDA initialization and the initial conditions on the host). No warm-up run: the first step of each run is
in the sum (it carries the device warm-up, a few ms of a 7-112 s region).

## Validation (`validate.sh`, upstream mechanism)

Upstream validates Cholla with gtest system tests (`src/system_tests/`, fiducial HDF5 data in the `cholla-tests-data`
submodule, fixed epsilon 5e-12) and a few analytic tests. `validate.sh` reproduces upstream's analytic
`tHYDROtMHDSYSTEMSoundWave3D` test (`src/system_tests/hydro_system_tests.cpp:99-140`) without the gtest harness: its
input file (`src/system_tests/input_files/tHYDROtMHDSYSTEMSoundWave3D_CorrectInputExpectCorrectOutput.txt`, a 64^3 linear
sound wave of amplitude 1e-5 run to t = 0.05) on the GPU, then `check_cholla.py soundwave --sine-tol 0.01`: density and
momentum_x against the analytic sine `A sin(2 pi (x - c_s t))` within upstream's tolerance 1e-7 = 1 % of the amplitude
(the L1 and the maximum error, which upstream's cell-by-cell assertion implies). Result (2026-10-21, 1 GPU): PASS,
maximum error 4.8e-8 (usage 0.48 of upstream's tolerance), 11 steps.

### Correctness of the registered inputs (`check_cholla.py`, frozen references)

Three separate results per input: timing SUCCESS (the per-step lines of a clean run), workload verification PASS
(`verify_registry_runs.py`: the manifest's parameter-file hash, problem, grid and tout, the parameter banner, the step
count and the final snapshot on stdout, the launcher audit), scientific correctness PASS (`hpcperf_inputs.py check` = one
fresh run in `run.check.<id>` + `check_cholla.py` + `verdict`). The check reads the final HDF5 output (`h5dump` binary
dumps + numpy; no h5py) and asserts first that its time equals `tout`, then:

- `sod-256`, `sod-512`: the exact solution of the Riemann problem (Toro 1999; iterative star state, the left/right states,
  `diaph` and `gamma` of the parameter file) at t = 0.2 against the y,z-averaged profiles of density, velocity and
  pressure (the run is planar to 4e-16). The L1 errors must not exceed 1.5 x the unmodified build's errors frozen in
  `reference/<id>.json` (sod-256: density 2.52e-3, velocity 4.73e-3, pressure 1.49e-3; sod-512: 1.38e-3 / 1.89e-3 /
  6.94e-4 -- second-order convergence). Basis: a defect in the Riemann solver or the reconstruction raises the L1 error
  by an order of magnitude (first order: ~4 x; a wrong wave speed: ~10 x), far beyond the 1.5 x margin.
- `sound-wave-512`: at t = 1 (one period) density = 1 + A sin(2 pi (x - c_s t)) and momentum_x = A sin(2 pi (x - c_s t)),
  c_s = 1; the L1 and the maximum error of both fields within 0.1 x A -- upstream's own criterion for its one-period
  linear-wave tests (`tHYDROtMHDSYSTEMLinearWavesParameterizedMpi`: `allowedL1Error = allowedError = 1e-7` for A = 1e-6
  with PLMC). Measured: maximum 1.01e-6 = 1 % of A (the amplitude decays by 1 % over one period at 512 cells per
  wavelength), L1 lower. The transverse momenta are not checked (not part of the eigenvector; upstream does not either).
- `noh-200`: the analytic Noh solution at t = 2 (gamma = 5/3, 3-D): the mean density of the cells inside 0.8 x the shock
  radius t/3 within 10 % of the post-shock value 64; measured 62.77 (deviation 1.9 %; the wall-heating dip at the centre
  of second-order Godunov codes is ~5 % of the plateau, Stone et al. 2008 test 11; the Noh tests of other
  hydrodynamics codes accept 15 %). The pre-shock profile (1 + t/r)^2 in the shell 0.80 <= r <= 0.95 is reported (measured mean relative error
  1.0e-3).
- `blast-256`: no analytic solution for a finite over-pressure region; the conservative scheme in the periodic box must
  conserve the total mass, momentum and energy between the initial and the final output to 1e-9 relative (the momenta
  relative to M sqrt(E/M)); measured 6e-20 (double-precision roundoff).
- all inputs: a frozen self-baseline digest of the final fields (`reference/<id>.json`: per field the sum, the extrema
  and the 8 x 8 x 8 block means) within 5e-12 relative of the unmodified build's `run.reference` run -- upstream's fixed
  epsilon for its fiducial-data system tests; Cholla's hydro step is deterministic (two runs of the same input agree
  bitwise in every field, digest deviation 0). The references were frozen once from this build (provenance, rule and
  basis inside each JSON) and are never regenerated by a check.

## Results on dgx003 (B200, CUDA 13.2.78)

Campaign `l3-ext-2026-10-20` (2026-10-21 14:23-14:48 UTC, Slurm job 2732030, GPU `0000:9D:00.0` by UUID, final
Level 3 protocol: 0 warm-up + 3 clean + 1 nsys-profiled run per input, headline = median of the application timer,
spread = (max - min) / median with 10 % as the UNSTABLE threshold, no clock locking, strictly serial;
`docs/timing/index.html` has the current rows):

| input | timer median (s) | min / max (s) | spread | stable | profiled run (s) | device busy (nsys, whole process) | run verification | correctness |
|---|---|---|---|---|---|---|---|---|
| sod-256 (default) | 7.317 | 7.317 / 7.320 | 0.04 % | yes | 7.322 | 36 % | PASS | PASS (L1 = the frozen values; digest 0) |
| sod-512 | 111.916 | 111.914 / 111.916 | 0.00 % | yes | 111.919 | 72 % | PASS | PASS (L1 = the frozen values; digest 0) |
| sound-wave-512 | 46.120 | 46.119 / 46.121 | 0.00 % | yes | 46.124 | 50 % | PASS | PASS (max error 0.101 of the 0.1 A rule; digest 0) |
| noh-200 | 10.963 | 10.962 / 10.963 | 0.01 % | yes | 11.025 | 49 % | PASS | PASS (62.77 vs 64, 1.9 % of the 10 % rule; digest 0) |
| blast-256 | 9.684 | 9.683 / 9.684 | 0.01 % | yes | 9.692 | 42 % | PASS | PASS (conservation 6e-20 of 1e-9; digest 0) |

- Timing SUCCESS for all five (the per-step lines of every clean run; `verify_registry_runs.py` PASS 5/5: the manifest's
  parameter-file hash, problem, grid and tout, the parameter banner, the registered step count, the final snapshot, the
  launcher audit `1 verified, 0 mismatch`). The runs are reproducible to 0.04 % or better: the loop is kernel-bound and
  the step sequence deterministic.
- The nsys-profiled run (CUDA trace) is within 0.6 % of the clean median for every input: the kernels are large and few
  per step (Cholla launches a handful of grid-wide kernels per step), so tracing costs nothing measurable. The
  "device busy" fraction is nsys's kernel-active share of the whole process (initialization and the two HDF5 writes
  included); within the loop the device is busier.
- Correctness (`run_checks.sh cholla` = one fresh run of every input in `run.check.<id>`, then `check_cholla.py` and
  `hpcperf_inputs.py verdict`; 2026-10-21 14:48-14:54 UTC): all five PASS. The digests of the check runs equal the
  frozen references exactly (deviation 0: deterministic), the Sod L1 errors equal the frozen values, the sound wave is
  at 0.101 of its rule (1.0 % of the amplitude after one period, the discretization error of PLMC at 512 cells per
  wavelength), the Noh plateau 62.77 (1.9 % below 64), the blast's conserved totals change by 6e-20.

## Limitations

- One rank per registered input; multi-GPU runs (`HPCPERF_GPUS=N`) are refused by `run.sh` (not validated here);
  multi-node BLOCKED (site). No HIP build here (upstream's Frontier recipe exists, untested without ROCm).
- The `hydro` build type only: upstream's MHD, gravity, particle, cooling and cosmology build types (and their examples)
  are not registered.
- The `Git Commit Hash` Cholla writes into its outputs is the frozen upstream commit passed by `build.sh` (the bundle is
  not a git checkout); `MACRO_FLAGS` in the outputs is upstream's string.
- The timed region sums per-step host-clock readings printed by the application (1 line per step, 400-1700 lines per
  run); the nsys-profiled run's device picture is the whole process (no NVTX range in Cholla).

## Source distribution

Scheme 3: the frozen bundle published 2026-10-21 as an asset of the GitHub prerelease `level3-source-hpcperf-l3-v1-ext1` of this repository (https://github.com/Deep-Learning-Profiling-Tools/HPC-Performance-AI/releases/download/level3-source-hpcperf-l3-v1-ext1/cholla-hpcperf-l3-v1.tar.zst; anonymous download verified against the lock, `provenance/remote_artifact_verification.yaml`); `provenance/source.lock.yaml`: `publish_status: published`. A fresh clone runs
`tools/prepare_benchmark.sh level3 cholla` (the artifact from the recorded URL, verified by size, sha256 and
`source_tree_sha256`; or from the cache / `--artifact FILE`), then `./build.sh` -- nothing is fetched at build time. `fetch.sh` is for a
re-freeze only.
