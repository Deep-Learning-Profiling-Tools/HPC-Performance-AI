# HACCabanaPM (pmkokkos)

HACCabanaPM ("pmkokkos") is a cosmological N-body particle-mesh (PM) mini-app
from the HACC team (Argonne), written on Cabana + Kokkos. `pm_ic` generates
Zel'dovich initial conditions from a CAMB transfer function (counter-based
Random123/Threefry white noise, sigma_8-normalised power spectrum, FFT-based
displacement field); `pm_run` then evolves the particles with a
drift-kick-drift integrator: cloud-in-cell (CIC) mass deposit onto the mesh
(Cabana::Grid p2g with GPU atomics), a Poisson solve in Fourier space
(heFFTe, cuFFT backend, MPI pencil decomposition), gradient interpolation back
to the particles (g2p), and strict-ownership particle migration between ranks
(Cabana::Distributor). Both programs write parallel-HDF5 snapshots. The motifs
are therefore scattered atomic particle-to-grid deposits, 3-D distributed
FFTs, gather-type grid-to-particle interpolation and all-to-all particle
exchange; one source tree covers CUDA and HIP through Kokkos.

## Provenance

Upstream repository: https://github.com/ECP-copa/HACCabanaPM
Upstream commit: 32fad91c1602077f22e89c92173714f401ede242 (main, "Paper title change.", 2026-08-03)
Cloned: 2026-09-01 (reference clone in `_upstream/level2/HACCabanaPM`)
License: BSD 3-clause (UChicago Argonne, LLC) -- copied verbatim as `LICENSE`.
The vendored `external/Random123` (D. E. Shaw Research, BSD) and
`external/cbrng` (HACC, BSD) carry their own license files.

`external/` is vendored upstream (plain files, no git submodules).

Copied from upstream (byte-identical unless listed under "Changes"):

- `CMakeLists.txt` -- upstream build (two additions, see below)
- `src/` -- library `pmkokkos_core` (47 files, 190 KB): `cosmo/` (cosmology,
  linear growth, CAMB transfer parser), `ic/` (white noise, Zel'dovich IC),
  `pm/` (CIC deposit/interpolation, FFT Poisson solve), `comm/` (Migrator),
  `io/` (indat parser, HDF5 snapshot writer/reader), `Grid.cpp`, `Particles.cpp`;
  `src/pm/PoissonSolve.hpp` is unused upstream too but kept for fidelity
- `apps/` -- `pm_ic.cpp`, `pm_run.cpp`, `pm_run_lib.{cpp,hpp}`,
  `snapshot_inspect.cpp`, `TopologyOverride.hpp`, `diag/pm_single_kick.cpp`
- `apps/demo/indat.params` -- upstream demo indat (128^3, z=200 -> 50, 5 steps)
- `external/cbrng/` (20 KB, 5 files) and `external/Random123/` (212 KB,
  22 headers) -- vendored CBRNG white-noise generator and its Random123 dependency
- `docs/` -- upstream design/runtime notes (7 Markdown files, 68 KB)
- `LICENSE`

Left out: `tests/` (257 KB, GoogleTest unit + end-to-end suite; GTest is not in
the environment, so `-DPMKOKKOS_ENABLE_TESTS=OFF`), `external/swfft/`
(137 KB, not referenced by the build), `apps/demo/{README.md,run_hacc.sh,
run_pmkokkos.sh,compare_ic.py}` (Aurora/HACC-side comparison scripts),
upstream `README.md`, `.gitignore`.

Added here: `apps/demo/cmbM000.tf` (generated, see below),
`tools/make_camb_transfer.py`, `apps/demo/indat_bench_256.params`,
`apps/snapshot_check.cpp`, `build.sh`, `run.sh`, `validate.sh`, this `README.md`.

## Changes from upstream

Source (one line, Cabana 0.8 API drift):

- `src/Grid.cpp`: `params.setAllToAll(true)` -> `params.setAlltoAll(true)`.
  Cabana 0.8.0 renamed `FastFourierTransformParams::setAllToAll(bool)`;
  same meaning (heFFTe alltoallv). Upstream targets Cabana 0.7.0. Commented
  in place.

`CMakeLists.txt` (two additions, both marked `HPC-Performance-AI`):

- `set_source_files_properties(external/cbrng/CBRNG_Random.cxx PROPERTIES
  COMPILE_DEFINITIONS "R123_CUDA_DEVICE=")`. With a CUDA-enabled Kokkos every
  CXX source goes through `nvcc_wrapper`, and Random123's
  `features/nvccfeatures.h` then declares its functions `__device__` only, so
  the host-only `CBRNG_Random.cxx` failed with 15x "calling a `__device__`
  function from a `__host__` function". `R123_CUDA_DEVICE` is Random123's
  documented override hook; defining it empty for that one translation unit
  yields plain host code (exactly what gcc/icpx produce upstream). The
  vendored files are untouched.
- `add_executable(snapshot_check apps/snapshot_check.cpp)` linked against
  `HDF5::HDF5` only -- the correctness checker used by `validate.sh`
  (see "Validation").

Data / configuration:

- `apps/demo/cmbM000.tf` (63 KB) is **not in the upstream repository**: the
  demo indat and upstream's demo README refer to it, but it was never
  committed and no public copy exists. It was regenerated with CAMB 2.0.4 for
  the HACC "M000" cosmology in `indat.params` (Omega_cdm 0.22, omega_b
  0.02258, h 0.71, n_s 0.963, T_CMB 2.726 K, flat, w=-1, massless neutrinos
  only) via `tools/make_camb_transfer.py` (`pip install camb`, ~15 s). The
  file has the 7-column CAMB `transfer_data` layout (k[h/Mpc], T_cdm, T_b,
  T_gamma, T_nu, T_nu_massive, T_tot; 564 rows, k = 5.06e-6 .. 715 h/Mpc) that
  `src/cosmo/TransferFunction.cpp` expects; only k, T_cdm, T_b are read and
  the amplitude is renormalised to sigma_8 = SS8 by `pm_ic`, so only the shape
  matters. It is therefore *a* valid M000 transfer function, not
  bit-identical to the file HACC uses internally.
- `apps/demo/indat_bench_256.params`: upstream's demo indat with exactly five
  value changes -- `NG 256`, `NP 256`, `RL 256.0` (same 1 Mpc/h spacing),
  `Z_FIN 0.0`, `N_STEPS 500` (upstream's own validated z=200 -> 0 / 500-step
  reference window, `docs/RUNTIME_CONTROL.md`). Cosmology, seed and transfer
  function unchanged. The header comment documents this.

Runtime (in `run.sh`, not in the code):

- `OMPI_MCA_opal_cuda_support=true` is exported for CUDA runs (overridable).
  Cabana's halo exchange, Distributor and heFFTe hand **device** pointers to
  MPI even at one rank (periodic self-halo). The conda-forge OpenMPI 5.0.10 is
  built with CUDA support (`accelerator cuda`, `btl smcuda`), but its shipped
  `$CONDA_PREFIX/etc/openmpi-mca-params.conf` sets `opal_cuda_support = 0`,
  and in OpenMPI 5.0.10 `accelerator_cuda_init()` then returns NULL, the
  `null` accelerator is selected and `pm_run` segfaults in
  `opal_convertor_unpack` at the first halo exchange ("Invalid permissions").
  Setting the variable to `true` selects the CUDA accelerator and everything
  works. (Diagnosed with `OMPI_MCA_accelerator_base_verbose=100` and
  `ompi_info --param opal all --level 9`.) The sibling Cabana integrations
  (`level2/exampm`, `level2/examinimd`) need the same setting.

Upstream behaviours worth knowing (deliberately not changed):

- `CMAKE_CXX_STANDARD 17` in the upstream CMake, but Kokkos 5.2.1 was built
  with C++20 and propagates `cxx_std_20` to every consumer, so all
  Kokkos-linked translation units compile as `-std=c++20`; only
  `snapshot_check` is C++17.
- Both apps count positional arguments strictly (`pm_ic <indat> <ic.h5>`,
  `pm_run <indat> <ic.h5> <out.h5>`); Kokkos strips `--kokkos-*` options
  first, so `run.sh` can forward extra Kokkos options.
- The multi-rank topology is chosen with `MPI_Dims_create` (or `TOPOLOGY` in
  the indat / `PMKOKKOS_TOPOLOGY`); `pm_run` reads the topology stored in the
  IC file, so `pm_ic` and `pm_run` must use the same rank count.
- `PMKOKKOS_TIMING=1` makes `pm_run` print a per-step CSV and a
  `pm_run_timing: summary ... median_s=... total_s=...` line (first
  `PMKOKKOS_TIMING_WARMUP`=5 steps excluded). `run.sh` enables it by default.

## Dependencies

All from `hpcperf_env.sh` / `.deps/install` (none rebuilt for this app):

- Kokkos 5.2.1 -- `.deps/install/kokkos` (CUDA + OpenMP + Serial, C++20,
  `Kokkos_ARCH BLACKWELL100`); the build uses its `bin/nvcc_wrapper` as the
  C++ compiler (host compiler = conda GCC 13.3.0 via
  `NVCC_WRAPPER_DEFAULT_COMPILER`).
- Cabana 0.8.0 -- `.deps/install/cabana` (header-only; Core + Grid with MPI,
  HDF5 and heFFTe enabled). Its CMake package lives in `share/cmake/Cabana`,
  not `lib*/cmake` (build.sh knows).
- heFFTe 2.4.1 -- `.deps/install/heffte` (FFTW + CUDA/cuFFT backends, static,
  GPU-aware).
- HDF5 2.2.0 -- conda, parallel (MPI-IO) build, C + HL components.
- FFTW 3.3.11, CUDA Toolkit 13.2 (cuFFT), OpenMPI 5.0.10 -- conda/system,
  found through `CMAKE_PREFIX_PATH`. OpenMPI needs
  `OMPI_MCA_opal_cuda_support=true` at run time (see above).
- GoogleTest -- **absent**, so the upstream test suite is not built
  (`-DPMKOKKOS_ENABLE_TESTS=OFF`; `HPCPERF_HACCABANAPM_TESTS=ON ./build.sh`
  would need `gtest` added to `environment.yml`).
- CAMB (Python) -- only needed to regenerate `apps/demo/cmbM000.tf`; not
  needed to build or run.
- HIP variant: `hipcc` plus HIP builds of Kokkos/Cabana/heFFTe
  (`.deps/install/{kokkos,cabana,heffte}-hip`) -- not available on this machine.

## Build / Run / Validate

```bash
source hpcperf_env.sh                        # optional -- the scripts source it themselves
./level2/haccabanapm/build.sh                # CUDA (default); ./build.sh HIP for the HIP build
./level2/haccabanapm/run.sh                  # pm_ic + pm_run, 256^3, z=200 -> 0 in 500 steps (~27 s)
./level2/haccabanapm/validate.sh             # demo 128^3 / 5 steps + snapshot_check, PASS/FAIL + exit code
```

`build.sh` reads the GPU's compute capability (`nvidia-smi
--query-gpu=compute_cap`, override `HPCPERF_CUDA_ARCH=100`) and refuses to
configure if it disagrees with the architecture baked into the Kokkos install;
`HPCPERF_KOKKOS_ROOT` / `HPCPERF_CABANA_ROOT` / `HPCPERF_HEFFTE_ROOT` override
the dependency locations, `HPCPERF_BUILD_JOBS` the parallelism (default 4),
extra `-D...` options are forwarded to CMake. Build tree:
`build/level2/haccabanapm/cuda` (binaries `pm_ic`, `pm_run`,
`snapshot_inspect`, `snapshot_check`, `pm_single_kick`). A clean build takes
~2.3 min at `-j4`.

`run.sh` takes a different indat via `HPCPERF_HACCABANAPM_INDAT` (absolute or
relative to `level2/haccabanapm`; the transfer function named by
`INPUT_BASE_NAME` must sit next to it), a rank count via `HPCPERF_NP` (adds
`--oversubscribe` for >1; all ranks share the one GPU here), an output tag via
`HPCPERF_HACCABANAPM_TAG`, and forwards extra arguments as Kokkos options.
Outputs go to `build/level2/haccabanapm/<model>/run/{ic,evolved}_<tag>.h5`
(0.77 GB each for 256^3), never into the repository.

Equivalent raw commands (CUDA):

```bash
source hpcperf_env.sh
R=$PWD
cmake -S level2/haccabanapm -B build/level2/haccabanapm/cuda -DCMAKE_BUILD_TYPE=Release \
      -DPMKOKKOS_ENABLE_TESTS=OFF \
      -DCMAKE_CXX_COMPILER=$R/.deps/install/kokkos/bin/nvcc_wrapper \
      -DCMAKE_PREFIX_PATH="$R/.deps/install/kokkos;$R/.deps/install/cabana;$R/.deps/install/heffte"
cmake --build build/level2/haccabanapm/cuda -j4
export OMPI_MCA_opal_cuda_support=true PMKOKKOS_TIMING=1
cd level2/haccabanapm/apps/demo                     # INPUT_BASE_NAME is resolved relative to the cwd
mpirun -np 1 $R/build/level2/haccabanapm/cuda/pm_ic  indat_bench_256.params $R/build/level2/haccabanapm/cuda/run/ic.h5
mpirun -np 1 $R/build/level2/haccabanapm/cuda/pm_run indat_bench_256.params $R/build/level2/haccabanapm/cuda/run/ic.h5 \
                                                     $R/build/level2/haccabanapm/cuda/run/evolved.h5
$R/build/level2/haccabanapm/cuda/snapshot_check $R/build/level2/haccabanapm/cuda/run/ic.h5 \
      $R/build/level2/haccabanapm/cuda/run/evolved.h5 --nsteps 500 --zfin 0.0
```

HIP (script logic present, unverified without ROCm):

```bash
cmake -S level2/haccabanapm -B build/level2/haccabanapm/hip -DCMAKE_BUILD_TYPE=Release \
      -DPMKOKKOS_ENABLE_TESTS=OFF -DCMAKE_CXX_COMPILER=hipcc \
      -DCMAKE_PREFIX_PATH="$R/.deps/install/kokkos-hip;$R/.deps/install/cabana-hip;$R/.deps/install/heffte-hip"
cmake --build build/level2/haccabanapm/hip -j4
```

Problem-size guidance: cost scales with NG^3 (FFT, deposit) and NP^3
(particles); snapshot size is 46 bytes per particle. Measured on one B200
(1 rank, `PMKOKKOS_TIMING` on):

| indat | size | steps | pm_ic | pm_run | of which stepping | snapshot |
|---|---|---|---|---|---|---|
| `apps/demo/indat.params` (validate.sh) | 128^3 | 5 (z 200 -> 50) | 2.6 s | 2.7 s | ~0.1 s | 96 MB |
| `apps/demo/indat_bench_256.params` (run.sh) | 256^3 | 500 (z 200 -> 0) | 5.2 s | 18.4 s | 13.9 s (median 27.7 ms/step, MAD 0.05 ms) | 772 MB |

The remaining `pm_run` time is Kokkos/MPI/CUDA start-up plus reading and
writing the two snapshots. 512^3 was not chosen as default: ~8x the time and
~12 GB of HDF5 I/O per run. To scale, edit `NG`/`NP`/`RL` together (keep the
1 Mpc/h spacing) in a copy of the bench indat. A 2-rank run
(`HPCPERF_NP=2`, automatic topology 2x1x1, both ranks on the one GPU) works
and validates.

## Validation

Upstream ships no reference output; its end-to-end GoogleTest
(`tests/test_pm_run_end_to_end.cpp`, not built here) checks structural invariants of
the evolved snapshot (particle count, step count, final scale factor,
positions inside the box). The runs are **not bit-reproducible** -- two
identical demo runs differ in 35,074 position and 4,277,537 velocity values
(`h5diff`), as expected from GPU atomic CIC deposits -- so no stored reference
snapshot is used. `validate.sh` runs `run.sh` on the upstream demo indat
(128^3, z=200 -> 50, 5 steps) and then `snapshot_check <ic.h5> <evolved.h5>
--nsteps 5 --zfin 50`, which reads both HDF5 files (serial HDF5 C API) and
requires all of:

1. particle count: `N(evolved) == N(ic) == attribute np` (strict-ownership
   migration must lose or duplicate nothing);
2. step count: `step == N_STEPS`; scale factor: `|a_out - 1/(1+Z_FIN)| <= 1e-6`
   relative and `a_out > a_in`;
3. positions finite and in `[0, rL)` on every axis;
4. velocities finite; masses finite, positive and all equal;
5. particle ids are a permutation of `[0, np)` (bitmap check);
6. total momentum: `max_d |sum_i v_d,i| / (N v_rms) <= 1e-6` -- the CIC
   deposit/interpolation force pair is antisymmetric, so sum(v) must stay at
   float round-off (measured 2.8e-11 evolved, 3.1e-11 in the IC);
7. linear growth: the rms density contrast of the particles CIC-deposited onto
   a coarse `(max(4, NG/8))^3` mesh must grow by `D(a_out)/D(a_in)` within a
   factor `[0.5, 2.0]`, where `D` is the flat-LambdaCDM linear growth factor
   (Heath integral with Omega_m from the file's attributes, radiation
   neglected). Measured: 0.005916 -> 0.02277, ratio 3.85 vs linear 3.94
   (0.977 of linear); for the 256^3 z=0 run 0.005843 -> 1.046, ratio 179 vs
   linear 152 (1.18 -- mildly non-linear by z=0, still inside the window).

Each check prints `[ok]` or `[FAIL]` with the measured numbers; the checker
prints `snapshot_check: PASS` (exit 0) or `snapshot_check: FAIL (n check(s)
failed)` (exit 1) and `validate.sh` forwards that as
`HACCabanaPM CUDA validation (apps/demo/indat.params): PASS|FAIL`. A negative
test (`--nsteps 6` against a 5-step run) yields FAIL / exit 1.

Observed here: `validate.sh` PASS in ~10 s wall (all 8 lines `[ok]`); the
benchmark indat also passes
(`HPCPERF_HACCABANAPM_INDAT=apps/demo/indat_bench_256.params ./validate.sh`,
~27 s); 2 ranks (`HPCPERF_NP=2 ./validate.sh`) PASS.

## Warnings

Compiler (clean CUDA build, 29 translation units + 5 links, `-O3`):

- 34x `nvcc_wrapper has been given multiple optimization flags (-O*)` --
  conda's `CXXFLAGS` carry `-O2` and CMake Release adds `-O3`; nvcc_wrapper
  notes it and keeps the last (`-O3`). Environment-wide, harmless.
- 1x nvcc `Warning #20014-D: calling a __host__ function from a __host__
  __device__ function is not allowed` -- inside
  `Kokkos_ViewLegacy.hpp:1033` via `Cabana_Migrate_Mpi.hpp:120`, instantiated
  from `src/comm/Migrator.cpp:160` (Kokkos/Cabana headers, not app code).
- 1x nvcc `src/ic/InitialConditions.cpp(86): warning #177-D: variable
  "nx_local" was declared but never referenced` -- upstream, harmless.
- 0 g++ (host compiler) warnings; 0 CMake warnings.

## LOC

cloc v2.06, code lines only (blank/comment excluded); `apps/demo` (indat
files, transfer table), `docs/`, scripts and this README excluded. CUDA and
HIP share the same source tree, so the count is given once.

Application (`src/` + `apps/`): **4321** (55 files) = 28 .cpp 3738 + 27 .hpp 583
  = upstream 4072 (`src/` 2481: 22 .cpp 1952 + 25 .hpp 529; `apps/` 1591:
  5 .cpp 1537 + 2 .hpp 54) + HPC-Performance-AI `apps/snapshot_check.cpp` 249.
Vendored (`external/`, counted separately): `cbrng` 140 (1 .cxx 54 + 2 .h 86),
  `Random123` 2851 (22 headers).
Tooling: `tools/make_camb_transfer.py` 34 (Python).

```bash
cloc --quiet level2/haccabanapm/src level2/haccabanapm/apps --exclude-dir=demo
cloc --quiet level2/haccabanapm/external/cbrng level2/haccabanapm/external/Random123
```

## Verified Environment

GCC/G++ 13.3.0 (conda, pinned) | C++20 (via Kokkos; upstream asks for 17) | CMake 3.28.4 | Ninja 1.13.2 | Python 3.12.3
CUDA Toolkit 13.2 (nvcc 13.2.78, /usr/local/cuda) | NVIDIA B200 (sm_100), driver 595.58.03
Kokkos 5.2.1 | Cabana 0.8.0 | heFFTe 2.4.1 (FFTW 3.3.11 + cuFFT) | HDF5 2.2.0 (parallel)
OpenMPI 5.0.10 (conda, CUDA-aware build; `opal_cuda_support` defaults to 0 in its mca-params.conf, re-enabled per run by run.sh)
HIP/ROCm: source is backend-agnostic Kokkos; build script logic present, unverified (no AMD GPU available)
OS: RHEL 10 (Linux 6.12)

Reproduce the toolchain from the repository root: `./setup_env.sh` then
`source hpcperf_env.sh` (all user-space versions are pinned in environment.yml).

## Status

CUDA: **Working** -- clean configure + build (`build.sh CUDA` from a foreign
cwd, 0 host-compiler warnings, 2 nvcc warnings listed above) + `run.sh`
(256^3, 500 steps, 26.6 s wall) + `validate.sh` (demo indat, PASS) all
succeeded on this machine, with 1 and 2 ranks; the benchmark indat also
passes `snapshot_check`.
HIP: untested (no ROCm/hipcc on the development machine; `./build.sh HIP`
exits 1 with "HIP requested but hipcc was not found (no ROCm on this
machine); set HIPCXX or install ROCm"). It additionally needs HIP builds of
Kokkos, Cabana and heFFTe under `.deps/install/*-hip`.
Open items for the maintainer: the CAMB-generated `cmbM000.tf` (upstream
never published theirs); GTest absent so the upstream test suite is not
built; `OMPI_MCA_opal_cuda_support=true` is set per run by `run.sh` -- it
could be set once in `hpcperf_env.sh` since every Cabana app needs it.
