# Branson

Branson is LANL's proxy application for Implicit Monte Carlo (IMC) thermal
radiative transfer: photons are emitted from a hot material and source
boundaries, tracked through a 3D Cartesian mesh (absorption, effective
scattering, census at the end of the time step), and the absorbed energy
feeds back into the material temperature. It is a header-only C++17 code:
`main.cc` pulls in everything and is compiled as a CUDA (or HIP) source, so
the CUDA and HIP variants share one source tree. The GPU path is the
event-based transport loop (`event_based_transport.h`): batches of photons are
advanced on the device with `thrust::remove_if` / sort-based compaction on raw
device pointers, Random123 counter-based RNG, warp-level reductions, and the
tallies are copied back after each step. Main motif: irregular,
branch-heavy particle tracking with data-dependent memory access.

## Provenance

Upstream repository: https://github.com/lanl/branson
Upstream commit: eed7f71699e9e2d7afc96a849c9c9b2a215cb1dd ("Merge pull request #82 from lanl/fix_weight_function", 2026-08-31)
Cloned: 2026-09-01 (`$R/_upstream/level2/branson`)
License: BSD-3 (Triad National Security, LLC) -- copied from upstream `LICENSE.md` to `LICENSE`

Copied into this directory (byte-identical except for the changes listed below):

- `src/` -- all Branson headers, `main.cc`, `CMakeLists.txt`, `config.h.in`,
  `config/` (find_tpls.cmake, FindMETIS.cmake, ...), `test/` (upstream unit
  tests + `simple_input.xml`), `random123/` (bundled, complete).
  Dropped: `src/doxygen.config` (documentation only).
- `src/pugixml/` -- bundled XML parser, trimmed to what the build needs:
  `CMakeLists.txt`, `LICENSE.md`, `README.md`, `readme.txt`,
  `src/{pugiconfig.hpp,pugixml.cpp,pugixml.hpp}`,
  `scripts/{pugixml-config.cmake.in,pugixml.pc.in,pugixml_dll.rc}`.
  Dropped: `docs/`, `tests/`, `.github/`, appveyor/codecov configs, `Makefile`,
  IDE/NuGet/CocoaPods project files under `scripts/`, `SECURITY.md`.
- `inputs/` -- all 8 upstream input decks (`3D_hohlraum_single_node.xml`,
  `3D_hohlraum_multi_node.xml`, `3D_lb_hohlraum.xml`, `big_cube.xml`,
  `cube_decomp_test.xml`, `hot_zone_input.xml`, `marshak_wave_dd.xml`,
  `marshak_wave_replicated.xml`) plus the two helper scripts `block_it.py`,
  `cubanova.py`. No deck was modified; no extra deck was needed.
- Not copied: upstream `README.md`, `scripts/` (job scripts), `CCS-memo-2020-11-18.pdf`.

## Changes from upstream

1. `src/CMakeLists.txt`, CUDA branch: added
   `set_target_properties(BRANSON PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}")`.
   Upstream sets `CMAKE_CUDA_ARCHITECTURES` to `CUDA_ARCH` only *after*
   `add_executable(BRANSON ...)`, so the target kept the hard-coded default list
   `70 75 80 86 90` and nvcc 13.2 aborted with
   `nvcc fatal : Unsupported gpu architecture 'compute_70'` (CUDA 13 dropped
   Volta/Turing). Now the detected architecture (sm_100 here) is the only one built.
2. `src/CMakeLists.txt`, CUDA branch: added the missing space in
   `string(APPEND CMAKE_CUDA_FLAGS " -expt-extended-lambda")`; upstream
   concatenated it onto the previous flag (`-lineinfo-expt-extended-lambda`).
3. `src/CMakeLists.txt`, HIP branch: same ordering fix as (1) for
   `HIP_ARCHITECTURES` (`set_target_properties(BRANSON PROPERTIES HIP_ARCHITECTURES "${HIP_ARCH}")`).
   Untested -- no ROCm on this machine.
4. `src/event_based_transport.h`: `thrust::distance(` -> `std::distance(`
   (8 host-side pointer differences on `int*`) plus `#include <iterator>`.
   CCCL 3.x shipped with CUDA 13.2 marks `thrust::distance` deprecated
   (32 warnings "Use cuda::std::distance instead"); `std::distance` is the
   portable equivalent and also compiles for HIP.
5. `src/test/CMakeLists.txt`: `MPIEXEC_POSTFLAGS` `-bind-to none` ->
   `--bind-to none`. OpenMPI 5's `prterun` rejects the single-dash spelling
   ("Executable: none"), which made all 12 ctest tests fail.
6. `src/test/CMakeLists.txt`, `add_branson_gpu_test`: the GPU unit test target
   was hard-coded to `CUDA_ARCHITECTURES "80"` (ran on the B200 only through
   PTX JIT); it now uses `${CMAKE_CUDA_ARCHITECTURES}`, i.e. the same
   `CUDA_ARCH` as BRANSON.

Every change is marked with an `HPC-Performance-AI:` comment in the source.
No kernel, algorithm, or input-deck change.

## Dependencies

- MPI: conda Open MPI 5.0.10 from `$CONDA_PREFIX` (`hpcperf_env.sh`); found by
  CMake as MPI 3.1. Required.
- CUDA 13.2 (`nvcc` 13.2.78, `/usr/local/cuda`) for the CUDA variant; ROCm
  (`$ROCM_PATH/llvm/bin/clang++`, default `/opt/rocm`) for the HIP variant.
- METIS 5.1.0: conda, found via `-DCMAKE_PREFIX_PATH=$CONDA_PREFIX`
  (`$CONDA_PREFIX/lib/libmetis.so`). Only used by the domain-decomposed
  (`particle_pass`) mode; the benchmark and validation decks run replicated.
- HDF5: conda parallel HDF5, found; Silo: not available, so `VIZ_LIBRARIES_FOUND`
  is off and the optional Silo output is compiled out (no effect on the runs).
- OpenMP: detected but "disabled for this build" -- upstream's CMake
  `option(USE_OPENMP ... FALSE)` defaults to OFF (its README says ON). It only
  threads the CPU transport loop; the GPU transport is unaffected.
- Umpire (`USE_UMPIRE`), Caliper (`USE_CALIPER`): OFF, not built.
- Bundled in `src/`: pugixml (XML decks), Random123 (RNG).
- Gray (1-group) build, upstream default (`N_GROUPS` not set).

## Build

```bash
level2/branson/build.sh            # CUDA (default): $R/build/level2/branson/cuda/BRANSON
level2/branson/build.sh HIP        # untested here (no ROCm): $R/build/level2/branson/hip/BRANSON
```

`build.sh` sources `hpcperf_env.sh` if needed, detects the GPU architecture
(`nvidia-smi --query-gpu=compute_cap`, override with `HPCPERF_CUDA_ARCH=100`),
unsets `CUDAARCHS` (upstream sets the architecture explicitly) and builds with
`-j4` (`MAKE_JOBS`). Equivalent raw commands:

```bash
source hpcperf_env.sh; unset CUDAARCHS
cmake -S level2/branson/src -B build/level2/branson/cuda \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX \
      -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
      -DUSE_CUDA=ON -DUSE_GPU=ON -DCUDA_ARCH=100 -DCMAKE_CUDA_COMPILER=$(which nvcc) \
      -DUSE_UMPIRE=OFF -DUSE_CALIPER=OFF -DBUILD_TESTING=ON
cmake --build build/level2/branson/cuda -j4
```

HIP (form only, unverified):

```bash
cmake -S level2/branson/src -B build/level2/branson/hip \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX \
      -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
      -DUSE_HIP=ON -DUSE_GPU=ON -DHIP_ARCH=gfx942 -DROCM_PATH=/opt/rocm \
      -DUSE_UMPIRE=OFF -DUSE_CALIPER=OFF -DBUILD_TESTING=ON
cmake --build build/level2/branson/hip -j4
```

Clean CUDA configure + build: 46 s. Configure output worth knowing:
`Found MPI ... (found version "3.1")`, `Looking for METIS.....found`,
`Looking for HDF5..found hdf5-shared`, `Looking for Silo..not found`,
`Looking for OpenMP... found, but disabled for this build`,
`Energy groups : Gray (1-group)`, `Making GPU(CUDA) BRANSON with 100 architecture`.

## Run

```bash
level2/branson/run.sh                      # = mpirun -np 1 build/level2/branson/cuda/BRANSON level2/branson/inputs/3D_hohlraum_single_node.xml
level2/branson/run.sh CUDA --photons 1000000 --t-stop 0.02   # extra args override the deck's <common> block
```

The standard problem is upstream's documented single-node performance
problem `inputs/3D_hohlraum_single_node.xml`: simplified 3D hohlraum,
65 x 65 x 140 = 591 500 cells, 10 M photons per step, 5 steps of 0.01 sh,
replicated mesh, event-based transport, SoA particle storage, one MPI rank
driving the GPU (the CPU-only parts -- source sampling, census combing -- run
serially on that rank). Upstream suggests a 30-group build for this deck; it is
run here with the default gray build. Observed on the B200 (shared node,
another job on the GPU): wall 28-45 s, `Total transport: 7.5 .. 36 s`,
`Photons Per Second (FOM): 2.8e5 .. 1.3e6`; per-step
`Radiation conservation ~1e-15..1e-17`, `Material conservation ~1e-13..1e-15`.
`HPCPERF_NP=<n>` runs more ranks (`--oversubscribe` is added automatically).

Note: with the GPU build, `--use-gpu-transporter FALSE` (CPU transport inside
the GPU binary) aborts in `GPU_Setup` (`Insist ... CUDA/HIP error synchronizing
after source kernel`) -- an upstream limitation of that mixed mode, left as is;
use a CPU-only build instead (validate.sh builds one, see below).

## Validate

```bash
level2/branson/validate.sh          # PASS: branson CUDA (...) / FAIL: ..., exit 0/1
```

Logs go to `build/level2/branson/cuda/validate_{ctest,marshak_gpu,marshak_cpu}.log`.

## Validation

Three checks; all must pass.

(A) Upstream unit tests: `ctest --test-dir build/level2/branson/cuda -E test_input_1pe`
    -- 11 tests: test_buffer, test_cell, test_proto_cell, test_counter_rng,
    test_mesh, test_mpi_types, test_sampling_functions, test_imc_parameters,
    test_warp_reductions (CUDA kernel, sm_100) on 1 rank, test_imc_state and
    test_photon on 2 ranks (`mpiexec -n 2 --bind-to none`). Result:
    `100% tests passed, 0 tests failed out of 11` (7.4 s).
    `test_input_1pe` is excluded: upstream's `test_input.cc` expects
    `dd_batch_size == 10000` and `event_batch_size == 10000` while its own
    `simple_input.xml` contains 1000 and 777 -- a test/data mismatch in the
    upstream commit, unrelated to the GPU port (verified: the test passes with
    an XML that carries 10000/10000). With it included ctest reports 11/12.

(B) Physics run on the GPU: `mpirun -np 1 BRANSON inputs/marshak_wave_replicated.xml --t-stop 0.05 --seed 1234`
    (1D Marshak wave, 25 cells, 50k photons/step, 5 steps, ~6 s). Checked:
    exit code 0; exactly 5 steps and a final `Photons Per Second (FOM)` line;
    GPU transport used every step (`Transferring 25 cell(s) to the GPU`, no
    "GPU kernel not available" fallback); per step Branson's own energy balances
    `|Radiation conservation| <= 1e-9 * (Emission E + Source E + Pre census E)`
    and `|Material conservation| <= 1e-9 * Pre mat E`.
    Observed: radiation 2e-16 .. 8e-14 relative, material 6e-16 .. 2e-15 relative.

(C) GPU vs CPU cross-check: the same deck and seed run with a CPU-only Branson
    built from the same sources (`-DUSE_GPU=OFF`, `build/level2/branson/cpu_ref`,
    built on first use in ~10 s, run ~6 s). Final-step `Post mat E`,
    `Absorption E`, `Exit E` must agree to 5 % relative and all 25 cell
    temperatures `T_e` to 0.02 absolute. The two binaries process photons in
    different order (device compaction vs serial loop), so agreement is
    statistical; the seed-to-seed scatter measured for these quantities is
    0.2-0.8 % (energies) and <= 0.006 (front-cell T_e), so the tolerances are
    > 6 sigma yet far below any transport error.
    Observed: Post mat E 2.00794e-04 vs 2.01081e-04 (1.4e-3 rel), Absorption E
    2.0e-3 rel, Exit E 1.7e-3 rel, max |dT_e| = 0.0019 (cell 2, wave front:
    0.0800 vs 0.0818); T_e profile GPU [0.9216, 0.7864, 0.0800, 0.01, ...] vs
    CPU [0.9220, 0.7870, 0.0818, 0.01, ...].

Result on this machine (fresh shell, cwd `/tmp`):
`PASS: branson CUDA (ctest 11/11 excl. test_input_1pe; Marshak 5 steps: rad/mat conservation <= 1e-9 rel; GPU vs CPU final Post-mat/Absorption/Exit E within 5%, T_e within 0.02)`
-- 42 s including the one-time CPU reference build.

## Warnings

After the fixes above, the clean CUDA build prints 2 warnings:

- 2 x `config.h:55: warning: "USE_GPU" redefined` -- upstream defines
  `USE_GPU ON` inside `#ifdef __NVCC__` in `config.h.in` while its CMake also
  passes `-DUSE_GPU` to the compiler (`target_compile_definitions(BRANSON PRIVATE USE_GPU)`);
  harmless (both spellings are only tested with `#ifdef`). Not an incompatibility
  with CUDA 13.2 / GCC 13, so left untouched.

Fixed (no longer emitted): 32 x `thrust::distance ... is deprecated` (change 4).

## LOC

`cloc` code lines (CUDA and HIP variants are the same source tree -- `main.cc`
is compiled as CUDA or HIP; there are no per-backend directories):

- Branson (CUDA = HIP): 7384 -- `src/*.h` (34 headers, 7151) + `src/main.cc` (233)
- Bundled third party, counted separately: `src/pugixml/src` 10619 (3 files),
  `src/random123` 3288 (28 headers)
- Unit tests, counted separately: `src/test` 1622 (13 `.cc` + 1 header)

Excluded: CMake files, `config.h.in`, XML decks, Python helpers, READMEs, scripts.

## Verified Environment

GCC/G++ 13.3.0 (conda, pinned) | C++17 | CMake 3.28.4 | Ninja 1.13.2 | Python 3.12.3
CUDA Toolkit 13.2 (nvcc 13.2.78, /usr/local/cuda) | NVIDIA B200 (sm_100), driver 595.58.03
Open MPI 5.0.10 (conda, `mpirun`/`mpiexec` from `$CONDA_PREFIX`) | METIS 5.1.0 (conda)
RHEL 10.0, Slurm allocation (1 GPU visible)
HIP/ROCm: source + build config present where noted, unverified (no AMD GPU available)

Reproduce the toolchain from the repository root: `./setup_env.sh` then
`source hpcperf_env.sh` (all user-space versions are pinned in environment.yml).

## Status

CUDA: Working -- configure + build (46 s) + run (`run.sh`, 3D hohlraum, ~30-45 s)
+ validate (`validate.sh`, PASS) all verified on the B200 from a fresh shell.
HIP: source and `build.sh HIP` present, untested (no ROCm on this machine).

Notes for the maintainer:

- Upstream `test_input_1pe` fails on its own data (see Validation (A)); excluded
  rather than editing upstream's test or XML.
- `-DUSE_OPENMP` defaults to OFF in upstream's CMake although its README says ON;
  left at upstream's CMake default. Passing `-DUSE_OPENMP=ON` would only affect CPU transport.
- The hard-coded architecture lists in upstream's *automatic* CUDA/HIP detection
  path (`CMAKE_CUDA_ARCHITECTURES "70 75 80 86 90"` when `USE_CUDA` is not given)
  still contain compute_70/75; not used by `build.sh` (which always passes
  `-DUSE_CUDA=ON -DCUDA_ARCH=<detected>`), so not changed.
