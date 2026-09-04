# ExaCMech (orientation_evolution miniapp)

ExaCMech (ECMech) is LLNL's GPU-friendly library of crystal-plasticity constitutive models; the
`orientation_evolution` miniapp drives it the way a finite-element host code would. A set of N single
crystals (random orientations, here 1,000,000) is loaded by a prescribed macroscopic velocity gradient
(uniaxial tension, `diag(-0.5,-0.5,1.0)`), and every time step each crystal independently integrates
the elasto-viscoplastic FCC model `evptn_FCC_A` (Voce hardening, power-law slip kinetics on 12 slip
systems): an implicit Newton solve of the coupled stress / slip / lattice-rotation update using the
bundled SNLS non-linear solver, followed by a rotation (texture) update of the orientation quaternion.
The motif is an embarrassingly parallel, per-point, register/L1-heavy small dense nonlinear solve
(RAJA `forall` over points with CHAI/Umpire-managed state arrays), plus a RAJA reduction per step for
the volume-averaged Cauchy stress that is printed as `Step# n Stress: s11 s22 s33 s23 s13 s12`.

## Provenance

Upstream repository: https://github.com/LLNL/ExaCMech (`develop` branch)
Upstream commit: 9ca5913fe95bdc83085cdde36f06c671dd674dd5 (tag `v0.4.3`, 2025-10-31; `ECMech_VERSION` 0.4.3)
Submodules: `snls` = https://github.com/LLNL/SNLS f72e98d99e6de0977631fd6cf2a1e45964de4171 (2025-10-24),
`cmake/blt` = https://github.com/LLNL/blt e783e30f2823ee1a208f7f90741b41c1f5a08063 (BLT 0.7.1, 2025-09-04)
Cloned: 2026-09-01 (into `_upstream/level2/ExaCMech`, with submodules)
License: BSD 3-Clause (copied as `LICENSE`, from upstream `LICENSE`; `NOTICE` kept alongside). SNLS
and BLT are also BSD 3-Clause (`snls/LICENSE`, `cmake/blt/LICENSE`).

Copied from upstream (paths relative to the upstream root):

- `CMakeLists.txt`, `LICENSE`, `NOTICE`, `ecmech.bib`
- `src/ecmech/**` (the library: `ECMech_*.h/.cxx`, `cases/`, `evptn/`, `kinetics/`, `slipgeom/`, `util/`, `CMakeLists.txt`)
- `miniapp/` (`orientation_evolution.cxx`, `setup_kernels.*`, `material_kernels.*`, `retrieve_kernels.*`,
  `miniapp_util.h`, `CMakeLists.txt`, `miniapp_script.bash`, `cases/*` -- option decks, property
  files and quaternion files for FCC and BCC models)
- `cmake/CMakeBasics.cmake`, `cmake/ECMechOptions.cmake`, `cmake/thirdpartylibraries/*`
- `cmake/blt/` (BLT 0.7.1, trimmed to what `SetupBLT.cmake` reads: `SetupBLT.cmake`, `cmake/`,
  `share/`, `scripts/`, `thirdparty_builtin/CMakeLists.txt` + `patches/`, `LICENSE`, `NOTICE`, `README.md`)
- `snls/` (`CMakeLists.txt`, `LICENSE`, `NOTICE`, `README.md`, `src/**`, `cmake/CMakeBasics.cmake`,
  `cmake/SNLSOptions.cmake`, `cmake/thirdpartylibraries/*`)

Left out: `README.md`, `DeveloperGuide.md`, `uncrustify.cfg`, `.github/`, `.gitignore`, `.gitmodules`,
`pybind11/` + `pyecmech/` (Python bindings), `test/` (gtest unit tests), `snls/test`, `snls/Win32`,
`snls/cmake/blt` (second BLT copy), and from `cmake/blt`: `tests/`, `docs/`, `host-configs/`,
`thirdparty_builtin/{googletest-1.16.0,benchmark-1.9.1,fruit-3.4.1}` and CI/meta files. Consequence:
`-DENABLE_TESTS=ON` cannot work with this tree (no test sources, no googletest); `build.sh` passes
`-DENABLE_TESTS=OFF`.

## Changes from upstream

- `snls/src/SNLS_device_forall.cxx` lines 98 and 107 (`Device::WaitFor`):
  `wait_for(event)` -> `wait_for(*event)` (two lines). Reason: camp v2026.07 changed
  `camp::resources::Resource::wait_for` to take `Event const&` instead of `Event*`; with the
  installed RAJA suite the pointer form is a hard nvcc error ("no instance of overloaded function
  ... wait_for matches the argument list (snls::rrese *)"). The function is not called by ECMech or
  the miniapp; the change only lets the file compile.
- `miniapp/CMakeLists.txt` line 15 (BLT >= 0.6 / `ENABLE_CUDA` branch):
  `list(APPEND ECMECH_DEPENDS blt::cuda_runtime blt::cuda)` ->
  `list(APPEND ECMECH_MINIAPP_DEPENDS blt::cuda_runtime blt::cuda)`. Reason: upstream appends to the
  wrong list (`ECMECH_DEPENDS` is the library's list, already consumed), so `orientation_evolution`
  is not marked as depending on `blt::cuda` and BLT 0.7 compiles its sources as plain C++ (host
  compiler): `RAJA::cuda_reduce does not name a type`, `'strat' was not declared` in
  `miniapp_util.h`. With the fix the miniapp sources are compiled by nvcc, as intended.
- Build-configuration workarounds in `build.sh` (no source change): `-DBLT_CXX_STD=c++20` -- upstream
  sets C++17, but every RAJA/camp/Umpire/CHAI v2026.07 header requires C++20 (`camp/concepts.hpp` uses
  `concept`/`requires`); BLT applies `BLT_CXX_STD` to both `CMAKE_CXX_STANDARD` and
  `CMAKE_CUDA_STANDARD` after the project's own `set(CMAKE_CXX_STANDARD 17)`.
  `-DUSE_BATCH_SOLVERS=ON -DSNLS_USE_RAJA_PORT_SUITE=ON` -- upstream derives these inside the
  `snls/` subdirectory (`ENABLE_CUDA` -> batch solvers -> RAJA Portability Suite), i.e. after the
  top-level `cmake/thirdpartylibraries/SetupThirdPartyLibraries.cmake` has already run, so a fresh
  configure never imports the `chai`/`umpire`/`fmt::fmt` targets that `src/ecmech` and `miniapp`
  link against ("orientation_evolution links to fmt::fmt but the target was not found"). Passing
  both options explicitly makes the first configure see the complete dependency set.

Everything else (library, solver, miniapp kernels, decks, CMake logic) is byte-identical to upstream
(all 147 copied files checked with `cmp`). Added wrapper/documentation files: `build.sh`, `run.sh`,
`validate.sh`, `README.md`.

## Dependencies

ExaCMech is a BLT/CMake project. On a GPU back end it requires the RAJA Portability Suite (all
v2026.07.0, all built with the conda GCC 13.3.0 + CUDA 13.2 toolchain, static libraries):

- RAJA + camp: `$R/.deps/install/raja` (built by `$R/setup_level2_deps.sh`, tag `v2026.07.0`;
  `lib/cmake/raja`, `lib/cmake/camp`).
- Umpire (`v2026.07.0`, bundles fmt 12.1.0) at `$R/.deps/install/umpire` and CHAI (`v2026.07.0`) at
  `$R/.deps/install/chai`, both built by `$R/setup_level2_deps.sh` (targets `umpire`, `chai`; the
  equivalent raw commands are listed below). They are needed because ExaCMech's GPU path
  (`USE_BATCH_SOLVERS`) uses CHAI `ManagedArray`/`managed_ptr` for its state and model objects and
  Umpire for the device allocations; the RAJA-only mode cannot run on the GPU.
- CUDA: nvcc (CUDA 13.2, `/usr/local/cuda`), conda GCC 13.3.0 as host compiler (`$CXX`), CMake >= 3.14
  (3.28.4), Ninja, OpenMP (GCC's libgomp; the `OpenMP` device option of the miniapp).
- HIP: `hipcc` (ROCm) plus HIP builds of RAJA/camp, Umpire and CHAI -- none available on the
  development machine.
- No MPI, no Python (bindings disabled), no other external libraries.

Umpire and CHAI as built here -- what `setup_level2_deps.sh umpire chai` does (run from `$R` after
`source hpcperf_env.sh`; sources are shallow clones of the release tags with the listed submodules;
~2 min each with `-j4`):

```bash
INST=$R/.deps/install; SRC=$R/.deps/src; BLD=$R/.deps/build
git clone --depth 1 --branch v2026.07.0 https://github.com/LLNL/Umpire.git $SRC/umpire
git -C $SRC/umpire submodule update --init --depth 1 blt src/tpl/umpire/fmt
cmake -S $SRC/umpire -B $BLD/umpire -DCMAKE_BUILD_TYPE=Release -DBLT_CXX_STD=c++20 \
      -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_CUDA_COMPILER=$(which nvcc) \
      -DCMAKE_CUDA_HOST_COMPILER=$CXX -DCMAKE_CUDA_ARCHITECTURES=100 \
      -DENABLE_CUDA=ON -DENABLE_OPENMP=ON -DUMPIRE_ENABLE_C=OFF -DBUILD_SHARED_LIBS=OFF \
      -DENABLE_TESTS=OFF -DENABLE_EXAMPLES=OFF -DENABLE_DOCS=OFF -DENABLE_BENCHMARKS=OFF \
      -Dcamp_DIR=$INST/raja/lib/cmake/camp -DCMAKE_INSTALL_PREFIX=$INST/umpire
cmake --build $BLD/umpire -j4 && cmake --install $BLD/umpire        # installs into lib64/
git clone --depth 1 --branch v2026.07.0 https://github.com/LLNL/CHAI.git $SRC/chai
git -C $SRC/chai submodule update --init --depth 1 blt
cmake -S $SRC/chai -B $BLD/chai -DCMAKE_BUILD_TYPE=Release -DBLT_CXX_STD=c++20 \
      -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_CUDA_COMPILER=$(which nvcc) \
      -DCMAKE_CUDA_HOST_COMPILER=$CXX -DCMAKE_CUDA_ARCHITECTURES=100 \
      -DENABLE_CUDA=ON -DENABLE_OPENMP=ON -DBUILD_SHARED_LIBS=OFF \
      -DCHAI_ENABLE_RAJA_PLUGIN=ON -DCHAI_ENABLE_MANAGED_PTR=ON \
      -DENABLE_TESTS=OFF -DENABLE_EXAMPLES=OFF -DENABLE_DOCS=OFF -DENABLE_BENCHMARKS=OFF \
      -Dcamp_DIR=$INST/raja/lib/cmake/camp -Draja_DIR=$INST/raja/lib/cmake/raja \
      -Dumpire_DIR=$INST/umpire/lib64/cmake/umpire -Dfmt_DIR=$INST/umpire/lib64/cmake/fmt \
      -DCMAKE_INSTALL_PREFIX=$INST/chai
cmake --build $BLD/chai -j4 && cmake --install $BLD/chai
```

`build.sh` locates the packages under `lib/cmake` or `lib64/cmake` of `$R/.deps/install/{raja,umpire,chai}`
(override with `HPCPERF_RAJA_DIR`, `HPCPERF_UMPIRE_DIR`, `HPCPERF_CHAI_DIR`) and fails with a pointer
to this section if one is missing. Upstream does not pin a RAJA-suite version anywhere (no CI
workflows in the repository; the README example is an old CORAL `-DCUDA_ARCH=sm_70` build); the code
as tagged targets C++17 and camp's former `wait_for(Event*)` API, i.e. RAJA Portability Suite
releases up to roughly v2025.09.

## Build / Run / Validate

```bash
cd /projects/kzhou6/bcui2/research/HPC-Performance-AI && source hpcperf_env.sh
level2/exacmech/build.sh            # CUDA (default); arch detected via nvidia-smi, override HPCPERF_CUDA_ARCH=100
level2/exacmech/run.sh              # 1,000,000 random orientations x 1000 steps, evptn_FCC_A, device GPU (~9.5 s)
level2/exacmech/validate.sh         # GPU vs CPU-device run of upstream's cases/option_cpu.txt (50k x 60), PASS/FAIL
level2/exacmech/build.sh HIP        # HIP (needs hipcc + HIP builds of RAJA/Umpire/CHAI; HPCPERF_HIP_ARCH=gfx942)
```

Build tree: `build/level2/exacmech/cuda` (binary `bin/orientation_evolution`; BLT puts executables in
`bin/`) and `build/level2/exacmech/hip`. The scripts run from any cwd; `run.sh`/`validate.sh` `cd` into
`level2/exacmech/miniapp` because the upstream decks reference `./cases/...`. `HPCPERF_BUILD_JOBS`
(default 4) sets the build parallelism; a fresh CUDA build takes ~1m50s with `-j4` (~5 min CPU).

Equivalent raw commands (CUDA):

```bash
unset CUDAARCHS                                                                        # hpcperf_env.sh exports "native"
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')   # 100 on B200
D=$R/.deps/install
cmake -S level2/exacmech -B build/level2/exacmech/cuda -DCMAKE_BUILD_TYPE=Release -DBLT_CXX_STD=c++20 \
      -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_CUDA_COMPILER=$(which nvcc) -DCMAKE_CUDA_HOST_COMPILER=$CXX \
      -DCMAKE_CUDA_ARCHITECTURES=$ARCH -DENABLE_CUDA=ON -DENABLE_OPENMP=ON \
      -DENABLE_MINIAPPS=ON -DENABLE_TESTS=OFF -DENABLE_PYTHON=OFF \
      -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DUSE_BATCH_SOLVERS=ON -DSNLS_USE_RAJA_PORT_SUITE=ON \
      -DRAJA_DIR=$D/raja/lib/cmake/raja -DCAMP_DIR=$D/raja/lib/cmake/camp \
      -DUMPIRE_DIR=$D/umpire/lib64/cmake/umpire -DFMT_DIR=$D/umpire/lib64/cmake/fmt -DCHAI_DIR=$D/chai/lib/cmake/chai
cmake --build build/level2/exacmech/cuda -j4
cd level2/exacmech/miniapp && ../../../build/level2/exacmech/cuda/bin/orientation_evolution cases/option_gpu.txt
```

HIP (build configuration present, unverified without ROCm; the same RAJA source is compiled with
`ENABLE_HIP` and needs HIP builds of the dependencies):

```bash
cmake -S level2/exacmech -B build/level2/exacmech/hip -DCMAKE_BUILD_TYPE=Release -DBLT_CXX_STD=c++20 \
      -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_HIP_COMPILER=hipcc -DROCM_PATH=$ROCM_PATH \
      -DCMAKE_HIP_ARCHITECTURES=gfx942 -DGPU_TARGETS=gfx942 -DAMD_GPU_TARGETS=gfx942 -DENABLE_HIP=ON \
      <same ENABLE_*/USE_BATCH_SOLVERS/*_DIR options as above, pointing at HIP builds of RAJA, Umpire, CHAI>
cmake --build build/level2/exacmech/hip -j4
```

`run.sh` options: `./run.sh [CUDA|HIP] [option_file]`. Without an option file it writes
`build/level2/exacmech/run/option_cuda.txt` (and a `#random N` quaternion file) for the standard
problem; the deck's lines are quaternion file, model, property file, device, dt, number of steps,
velocity gradient. Environment overrides: `EXACMECH_NQPTS=1000000`, `EXACMECH_NSTEPS=1000`,
`EXACMECH_DT=0.00025`, `EXACMECH_MODEL=evptn_FCC_A`, `EXACMECH_PROPS=miniapp/cases/props.txt`,
`EXACMECH_DEVICE=GPU|OpenMP|CPU`. An explicit deck is run unchanged with cwd `miniapp/`, so the
upstream decks work as-is, e.g. `./run.sh CUDA cases/option_gpu.txt` (350,000 points x 60 steps,
0.25 s on the B200) or `cases/option_gpu_md.txt` (BCC `evptn_BCC_MD` model).

Benchmark problem (`run.sh` defaults): upstream's `miniapp_script.bash` / `cases/option_gpu.txt`
problem -- `evptn_FCC_A` with `cases/props.txt`, velocity gradient `diag(-0.5,-0.5,1.0)`,
dt = 0.00025 -- scaled from 350,000 points x 60 steps to 1,000,000 random orientations (seed 42) x
1000 steps, i.e. 25 % axial strain. Observed on the B200: `Run time of set-up, material, and
retrieve kernels over 1000 time steps is: 8.09(s)` (8.1 ms per step for 10^6 crystals), 9.5 s wall
including the host-side quaternion generation and setup; last line
`Step# 1000 Stress: -117.564 -117.51 235.075 0.0617772 -0.104468 0.0514891`. The GPU was shared with
other jobs, so times vary somewhat between runs.

## Validation

Upstream ships no reference outputs (the miniapp's own comment says the per-step stress print is
there "to make sure things are correct between the different runs" of the CPU, OpenMP and GPU
paths), so `validate.sh` implements exactly that cross-check with upstream's own deck
`miniapp/cases/option_cpu.txt`: 50,000 random orientations (seed 42; the quaternions are generated
on the host with `std::minstd_rand0`, identically for every device), `evptn_FCC_A`, `cases/props.txt`,
dt = 0.00025, 60 steps, `diag(-0.5,-0.5,1.0)`. The same binary runs the deck once with device `GPU`
and once with device `CPU` (RAJA sequential; only line 4 of the deck differs, the copies are written
to `build/level2/exacmech/validate/`). Both runs must exit 0 and print all 60 `Step#` lines, and all
60 x 6 volume-averaged Cauchy stress components must satisfy
`|gpu - cpu| <= 1e-5 * |cpu| + 1e-6` (MPa; override with `EXACMECH_RTOL`/`EXACMECH_ATOL`). The values
are printed with 6 significant digits, so the tolerance means "agree to print precision"; the
full-precision results differ only by summation order in the reduction. `EXACMECH_REF_DEVICE=OpenMP`
uses the OpenMP path as the reference instead. It prints `ExaCMech CUDA validation: PASS|FAIL` and
exits 0/1. A deliberately perturbed value is caught (checked once; exit 1).

Result observed here (CUDA, B200): PASS -- `compared 60 steps; max |gpu-ref| = 0`, i.e. all 360
printed values are identical between GPU and CPU (`Step# 1 Stress: -12.9029 -12.9012 25.8041
0.0216062 0.027763 0.00338614` ... `Step# 60 Stress: -32.7423 -32.6921 65.4344 -0.0320422 0.0451726
0.0012924`), and also identical run-to-run. GPU run 0.067 s of kernel time, CPU reference 18.5-26 s
(single core), ~20-30 s total. With `EXACMECH_REF_DEVICE=OpenMP OMP_NUM_THREADS=8` also PASS
(max diff 0; the OpenMP reference took 20.9 s on the shared node).

## Warnings

After the fixes above, a fresh `build.sh CUDA` (configure + build) shows:

- 3 deprecation sites in upstream source, reported once per translation unit: `[=]` lambdas that
  implicitly capture `this` (deprecated in C++20) in `src/ecmech/kinetics/ECMech_kinetics_BCCMD.h`
  lines 458, 479 and 492 -- GCC `-Wdeprecated` "implicit capture of 'this' via '[=]' is deprecated
  in C++20" (21 messages, host passes) and nvcc `#2908-D` (24 messages, device passes). Harmless;
  a consequence of building with C++20. Left untouched to keep the source identical.
- 24 CMake developer warnings at configure time: 23x "Target 'umpire' is deprecated. Please use
  'umpire::umpire' instead" (SNLS/ECMech CMake links the legacy Umpire target name, which Umpire
  v2026.07 still provides) and 1x policy CMP0146 (`find_package(CUDA)` in BLT 0.7.1's
  `BLTSetupCUDA.cmake`). Both informational.
- No other GCC/nvcc warnings; no link warnings.

## LOC

cloc v2.06, code lines only (CMake files, text/deck files, scripts and READMEs excluded):

ExaCMech own code (`src/` + `miniapp/`): 7266 (49 files: 38 C/C++ headers 6452 + 11 C++ sources 814)
-- `src/ecmech/**` 6569 (34 headers 6353 + 7 C++ 216) and `miniapp/*.cxx,*.h` 697 (4 C++ 598 + 4 headers 99).
Bundled dependency SNLS (`snls/src/`): 3376 (19 files: 17 headers 3221 + 2 C++ 155).
CUDA and HIP variants are the same RAJA/CHAI source (back end selected by `ENABLE_CUDA`/`ENABLE_HIP`
at configure time), so the counts are identical for both. Not counted: `cmake/blt` (build system),
`cmake/`, `snls/cmake/`, `miniapp/cases` (data), `miniapp/miniapp_script.bash`.

## Verified Environment

GCC/G++ 13.3.0 (conda, pinned) | C++20 (`-DBLT_CXX_STD=c++20`, see above) | CMake 3.28.4 | Ninja 1.13.2 | Python 3.12.3
CUDA Toolkit 13.2 (nvcc 13.2.78, /usr/local/cuda) | NVIDIA B200 (sm_100), driver 595.58.03
RAJA Portability Suite v2026.07.0: RAJA + camp (`.deps/install/raja`), Umpire (+fmt 12.1.0, `.deps/install/umpire`), CHAI (`.deps/install/chai`)
RHEL 10 (Linux 6.12), Slurm allocation, 1 GPU
HIP/ROCm: source + build config present where noted, unverified (no AMD GPU available)

Reproduce the toolchain from the repository root: `./setup_env.sh` then
`source hpcperf_env.sh` (all user-space versions are pinned in environment.yml); RAJA via
`./setup_level2_deps.sh`; Umpire and CHAI as in "Dependencies".

## Status

CUDA: Working (configure + build + run + validate all passed on the B200 via `build.sh`, `run.sh`,
`validate.sh` from a fresh shell; build 1m48s, run 9.5 s, validate PASS with max |gpu-cpu| = 0).
HIP: source and build configuration present (`build.sh HIP`, `-DENABLE_HIP=ON`), untested -- no
ROCm/hipcc on this machine, and HIP builds of RAJA/camp, Umpire and CHAI would be needed as well.
