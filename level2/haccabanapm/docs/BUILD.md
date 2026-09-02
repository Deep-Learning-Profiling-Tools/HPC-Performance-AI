# Build guide

## Dependencies

| Dependency | Version | How it is found | Notes |
|---|---|---|---|
| CMake | ≥ 3.22 | — | `cmake_minimum_required(VERSION 3.22)` |
| C++ compiler | C++17 | — | no compiler extensions (`CMAKE_CXX_EXTENSIONS OFF`) |
| Kokkos | 4.6.x validated | `find_package(Kokkos REQUIRED)` | the resolved install fixes the backend |
| Cabana | 0.7+ | `find_package(Cabana REQUIRED COMPONENTS Core Grid)` | must be built with `Cabana_ENABLE_GRID=ON` and `Cabana_REQUIRE_HEFFTE=ON` |
| heFFTe | 2.4.x validated | transitive via `Cabana::Grid` | FFT backend follows the memory space |
| MPI | — | transitive via `Cabana::Grid` | must be **GPU-aware** for multi-rank runs |
| Parallel HDF5 | 1.14.x validated | `find_package(HDF5 COMPONENTS C HL REQUIRED)` | **parallel (MPI-IO) build required** |
| GoogleTest | 1.15.x validated | `find_package(GTest REQUIRED)` | only when `PMKOKKOS_ENABLE_TESTS=ON` (the default) |

Requesting `Cabana COMPONENTS Core Grid` is enough to cover the whole
stack: `Cabana::Grid` pulls in `Kokkos::kokkos`, `MPI::MPI_CXX`, and
`Heffte::Heffte` transitively.

**HDF5 must be a parallel build.** The snapshot writer and reader call
`H5Pset_fapl_mpio` and use `H5FD_MPIO_COLLECTIVE` transfers unconditionally
— a serial HDF5 will fail to link or fail at runtime. Verify with
`h5pcc -showconfig | grep -i parallel` (expect `Parallel HDF5: yes/ON`).

Vendored third-party code under `external/` needs no separate install:
`external/cbrng/` and `external/Random123/` (HACC's counter-based RNG,
mirrored verbatim — do not modify in-tree) and `external/swfft/` (unused by
the default build; see [below](#swfft)).

## Basic build

```sh
cmake -B build -DCMAKE_PREFIX_PATH=/path/to/cabana/install \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

`CMAKE_PREFIX_PATH` should point at the Cabana install; Kokkos and heFFTe
are found from Cabana's package config. For a machine-specific recipe
including modules, see [SYSTEMS.md](SYSTEMS.md).

## CMake options

The build defines exactly three project options. There are **no** backend
selection options — see [Backend selection](#backend-selection).

| Option | Type | Default | Effect |
|---|---|---|---|
| `PMKOKKOS_VECTOR_LENGTH` | STRING | `16` | Inner SIMD width of the particle AoSoA. Propagated as a compile definition and read in `src/Types.hpp`. 16 matches the Cabana SYCL/PVC default. |
| `PMKOKKOS_ENABLE_TESTS` | BOOL | `ON` | Build the GoogleTest suite and register CTest entries. Requires GoogleTest. |
| `PMKOKKOS_FP_PRECISE` | BOOL | `OFF` | Adds `-fp-model=precise -ffp-contract=off` globally. |

### `PMKOKKOS_FP_PRECISE`

Suppresses FMA fusion and floating-point reassociation across the whole
code base — a single-flag change that touches no other build setting. Turn
it **on** for bit-level cross-code comparison against HACC, where fused
multiply-add in the CIC gather otherwise changes results at the ULP level.
Leave it **off** to measure FMA-fused performance.

Note it is Intel-compiler syntax (`-fp-model=`). On GCC or Clang you would
want `-ffp-contract=off` alone; the option as written is not portable and
has only been used with `icpx`.

> The paper's floating-point manifest also lists `-fimf-precision=high` as
> part of the matched-flag configuration used on **both** codes. That flag
> is **not** added by this CMake option — supply it yourself via
> `CMAKE_CXX_FLAGS` if you are reproducing that configuration.

### `PMKOKKOS_VECTOR_LENGTH`

```sh
cmake -B build -DPMKOKKOS_VECTOR_LENGTH=8 ...
```

Changes the AoSoA inner vector length. It is a performance knob, not a
correctness one, but it does change memory layout — so it is worth
recording alongside any timing result. `pm_run`'s timing provenance line
reports the compiled-in value.

## Build targets

| Target | Kind | Sources |
|---|---|---|
| `pmkokkos_core` | library | everything under `src/` that is compiled, plus `external/cbrng/CBRNG_Random.cxx` |
| `pm_run_lib` | static library | `apps/pm_run_lib.cpp` — the `pm_run` pipeline, factored out so tests can call `pm_run_main()` directly instead of spawning a nested `mpiexec` |
| `pm_ic` | executable | IC generator |
| `pm_run` | executable | thin `main()` over `pm_run_lib` |
| `snapshot_inspect` | executable | single-rank snapshot dumper |
| `pm_single_kick` | executable | diagnostic driver (`apps/diag/`) |

## GPU vs CPU builds

**There is no CMake switch for this.** The execution space is
`Kokkos::DefaultExecutionSpace`, so whichever Kokkos install
`find_package(Kokkos)` resolves determines whether you get a GPU or CPU
build. To build both, install Kokkos twice and point `CMAKE_PREFIX_PATH` at
one or the other:

```sh
# GPU build (Kokkos built with SYCL/HIP/CUDA enabled)
cmake -B build_gpu -DCMAKE_PREFIX_PATH=/path/to/cabana-sycl  -DCMAKE_BUILD_TYPE=Release

# CPU build (Kokkos built with OpenMP or Serial)
cmake -B build_cpu -DCMAKE_PREFIX_PATH=/path/to/cabana-openmp -DCMAKE_BUILD_TYPE=Release
```

Everything downstream follows automatically — `MemorySpace` is
`DefaultExecutionSpace::memory_space`, and heFFTe picks its backend from
that.

For a CPU build, set `OMP_NUM_THREADS` at runtime and keep GPU-specific
environment variables (`ZE_AFFINITY_MASK`, `ONEAPI_DEVICE_SELECTOR`,
`MPIR_CVAR_ENABLE_GPU`) unset.

## Backend selection

Neither backend is chosen by pmkokkos. Both are inherited:

| Layer | Chosen by | Result |
|---|---|---|
| Kokkos execution backend | the resolved `find_package(Kokkos)` install | SYCL / HIP / CUDA / OpenMP / Serial |
| heFFTe FFT backend | the Kokkos memory space | device → vendor FFT (oneMKL on SYCL, rocFFT on HIP, cuFFT on CUDA); host → FFTW3 |

So on Aurora, a Kokkos-SYCL install yields `heffte::backend::onemkl`
automatically; a Kokkos-OpenMP install on the same machine yields FFTW3.
Nothing in `CMakeLists.txt` names an FFT backend.

### SWFFT

`src/pm/SwfftBackend.{hpp,cpp}` wraps HACC's SWFFT so the exact HACC FFT
pipeline can be run for cross-code comparison. **It is not wired into the
build.** Specifically:

- there is no `PMKOKKOS_FFT_BACKEND_SWFFT` option in `CMakeLists.txt`
  (despite what the file's own header comment claims);
- `SwfftBackend.cpp` is not in any target's source list;
- `src/pm/PMStep.cpp` does not reference it.

It is also single-rank only — multi-rank would require reworking the
Green's-function k-space indexing for SWFFT's 2D-z-pencil layout. Treat it
as an experiment-branch artifact, not a supported configuration.

## Tests

```sh
cd build && ctest --output-on-failure
```

Two classes of test are registered:

- **Host-only** (`test_types`, `test_units`, `test_cosmology`,
  `test_transfer_function`, `test_power_spectrum_normalization`,
  `test_growth_factor_*`, `test_white_noise_against_hacc`,
  `test_indat_parser`) — pure arithmetic, no `Kokkos::initialize`, no
  device. These run anywhere, including a login node.
- **Device tests**, labeled `gpu` and launched under `mpiexec -n 1`. Select
  them with `ctest -L gpu`. On a machine whose Kokkos default space is a
  GPU, these need a compute node — `Kokkos::initialize()` will try to
  acquire a device unconditionally and fail on a login node.

Several tests are additionally registered at 2, 4, and 8 ranks to exercise
the distributed paths: `test_mass_assign` (halo scatter),
`test_migrator` and `test_migration_protects_scatter` (the
`Cabana::Distributor` path and the pre-kick migration guard),
`test_white_noise_reproducibility` (rank-independence of the IC field), and
`test_snapshot_roundtrip` / `test_snapshot_partial_particles` (collective
HDF5, including empty-rank cases).

**Multi-rank tests need GPU-aware MPI.** On Aurora, export
`MPIR_CVAR_ENABLE_GPU=1` before running `ctest`, or the
`Cabana::Distributor` tests will fail — see
[RUNTIME_CONTROL.md](RUNTIME_CONTROL.md#gpu-aware-mpi).
