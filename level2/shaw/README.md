# SHAW

SHAW ("SHeAr Waves", Pressio project) simulates elastic seismic shear-wave
propagation through the Earth's mantle in an axisymmetric (r, theta) domain
between the core-mantle boundary and the surface. The velocity/stress
formulation is discretised on a staggered polar grid and advanced with a
leapfrog scheme: each time step applies the sparse velocity Jacobian to the
stress field (`KokkosSparse::spmv`), scales by the inverse density
(`KokkosBlas::mult`) and adds the point source, then applies the stress
Jacobian to the velocity field. The main motif is therefore
memory-bandwidth-bound sparse matrix-vector products on fixed CSR matrices
(4 and 2 non-zeros per row) with a few vector updates per step; there are no
reductions or halo exchanges. It is a single-process Kokkos application
(no MPI); the whole state lives on the device and the host drives the time
loop. Kokkos Kernels and yaml-cpp are the only libraries it needs.

## Provenance

Upstream repository: https://github.com/Pressio/SHAW
Upstream commit: 66f7ac5d0ffccfbf772c88e0d8387024c69424bc (main, 2021-12-22 "fix CI")
Cloned: 2026-09-01 (reference clone in `_upstream/level2/SHAW`)
License: BSD-3-Clause (copied as `LICENSE`)

Copied from upstream:

- `CMakeLists.txt` (modified, see below), `LICENSE`, `input_template.yaml`
- `src/kokkos/` -- the Kokkos application: `main.cc`, FOM problem/run/kernels,
  forcing, seismogram and snapshot observer, `shwavepp.hpp` (Jacobian
  assembly), ROM problem headers (16 files; 3 modified, see below).
  `src/kokkos/torevise/` (unfinished rank-2 ROM, not included in any build) left out.
- `src/shared/` -- header-only host code: yaml parser, mesh reader
  (`mesh_helpers`), PREM/unilayer/bilayer material models, source signals,
  CFL / numerical-dispersion checkers, output writers, complexity model
  (44 files, byte-identical). `src/shared/material_models/not_complete_yet/` left out.
- `meshing/` -- the python mesh generator: `create_single_mesh.py`,
  `fullMeshWithTwoVarGroups.py`, `fullMeshWithThreeVarGroups.py`,
  `helpers/*.py` (12 files; 6 modified, see below). Not copied:
  `sampleMeshWithTwoVarGroups.py`, `grid_schematic.py`, `plotMeshPaper` (ROM
  sample-mesh and paper-figure scripts).
- `tests/CMakeLists.txt` (modified), `tests/compare.py`,
  `tests/fullMesh21x51/` (the 21x51 test mesh, 4 `.dat` files) and the
  end-to-end FOM regression test `tests/fomInnerDomain/` (`CMakeLists.txt`
  modified, `input.yaml` and `test.cmake` byte-identical, gold data
  xz-compressed).

Left out: `src/tools/` (post-processing / ROM tools, need the bundled
`eigen/`, 12 MB), `eigen/`, `demos/` (only yaml inputs; demo1's input is
reproduced by `run.sh`), `docs/`, `bash_scripts/`, `helper_scripts/`,
`python_scripts_revise/`, `various/`, `.github/`, upstream `README.md`, and
all other tests: the unit tests (`meshInfo`, `parser`, `seismogram`,
`forcing_rank1`, `graphs`, `coords`, `jacobian_vp`, `jacobian_sp`,
`stress_labels`), the ROM/multi-forcing tests (`multi*ForcingRank{1,2}`), and
the four other FOM regression tests (`fomNearCmb`, `fomNearEarthSurface`,
`fomSymmetryAxisThetaZero`, `fomSymmetryAxisThetaPi`) whose gold data is
9-16 MB each (54 MB in total, ~20 MB compressed). They can be run from the
reference clone if ever needed (same `test.cmake` mechanism).

Added here: `build.sh`, `run.sh`, `validate.sh`, this `README.md`.

## Changes from upstream

Source (needed to compile against Kokkos / Kokkos Kernels 5.2.1 with CUDA;
upstream targeted Kokkos 3.x):

- `src/kokkos/types.hpp`: `jacobian_ord_type` is `std::int64_t` instead of
  `mesh_info_type::ordinal_type` (= `std::size_t`). Kokkos Kernels >= 4.0
  `static_assert`s that `KokkosSparse::CrsMatrix`'s ordinal type is signed
  (`KokkosSparse_CrsMatrix.hpp:306`). `int64_t` keeps the 8-byte index width,
  so the memory traffic of the SpMVs (and upstream's complexity model, which
  uses `sizeof(ord_t)`) is unchanged. The mesh reader's `std::size_t` ordinal
  is untouched.
- `src/kokkos/shwavepp.hpp` (`fillVpJacobian`, `fillSpJacobian`): the host
  row-pointer / column-index arrays `ptr_h`, `ind_h` are
  `View<jacobian_ord_type*>` instead of `View<mesh_ord_type*>` (4 lines), because
  the raw-pointer `CrsMatrix` constructor requires `OrdinalType*` exactly.
- `src/kokkos/rom_run_rank_one.hpp`: the ROM time loop passed *device* views
  to `observerObj.observe(...)`, which `static_assert`s host accessibility
  (`state_observer.hpp:179`, "View not accessible on host"); with a
  CUDA-enabled Kokkos the ROM path therefore does not compile upstream, and
  `main.cc` instantiates the ROM problem unconditionally (`main.cc:71`), so the
  FOM executable cannot be built without it. Fixed the same way the FOM loop
  (`fom_run.hpp`) does it: host mirrors + `deep_copy` when snapshot collection
  is enabled. The ROM path is not exercised by anything in this repo.

Build system:

- `CMakeLists.txt`: the four tool executables (`extractStateFromSnaps`,
  `reconstructFomState`, `reconstructSeismogram`, `computeThinSVD` with its
  `find_package(OpenMP)` and bundled-Eigen include) are removed; only
  `shawExe` is built. Upstream's `set(CMAKE_CXX_STANDARD 14)` is left as is:
  the Kokkos CMake config raises the standard to C++20 itself
  (`Kokkos_CXX_STANDARD 20` in `KokkosConfigCommon.cmake`), and `build.sh`
  passes `-DCMAKE_CXX_STANDARD=20` explicitly.
- `tests/CMakeLists.txt`: only `add_subdirectory(fomInnerDomain)` (the other
  test directories are not copied, see Provenance).
- `tests/fomInnerDomain/CMakeLists.txt`: the gold files `snaps_vp_0_gold`
  (3.1 MB) and `snaps_sp_0_gold` (6.3 MB) are stored as `.xz` (3.5 MB total)
  and decompressed into the build tree at configure time (`xz -dkc`) instead
  of `configure_file(... COPYONLY)`. `test.cmake` and `input.yaml` are unchanged.

Mesh generator (`meshing/`):

- `matplotlib` is imported inside `try/except` (set to `None` when missing) in
  `create_single_mesh.py`, `fullMeshWithTwoVarGroups.py`,
  `fullMeshWithThreeVarGroups.py`, `helpers/unit_cell.py`,
  `helpers/plot_utils.py`, `helpers/convert_graph_dic_to_sparse_matrix.py`,
  and the unconditional `plt.show()` at the end of
  `mainFullMeshWithTwoVarGroups` is moved under the existing
  `if plotting != "none":` block. matplotlib is only used by the optional
  plotting modes and is not in the project conda environment; meshes are
  generated with `-plotting none` (the default).
- `sys.path.insert(0, './helpers')` in the three top-level scripts is
  replaced by the path relative to the script file
  (`os.path.dirname(os.path.abspath(__file__))`), so `create_single_mesh.py`
  can be run from any directory (upstream had this commented out). The
  generator still writes its `.dat` files into the cwd before moving them
  into `<working-dir>/mesh<nr>x<nth>`, so `run.sh` runs it from a scratch
  directory under the mesh cache.

Upstream behaviours worth knowing (deliberately *not* changed):

- The per-step kernel timers in `fom_run.hpp` (`ct2`, `ct3`) stop *before*
  the `Kokkos::fence()`/`deep_copy` (upstream comment: "need to fix timers for
  async launch"), so on a GPU `totTime`, `aveTime`, `aveBandwidth(GB/s)` and
  `aveGFlop` printed by `print_perf.hpp` measure kernel-launch time only and
  are meaningless (e.g. "aveBandwidth = 25206 GB/s"). `loopTime` (wall time of
  the whole time loop) is the number to use; the GPU time ends up inside
  `dataCollectionTime`, which is measured across the fence.
- When a seismogram or snapshots are requested, the *entire* velocity state
  (and stress state for snapshots) is copied device->host at every time step
  (`fom_run.hpp:113-121`); the seismogram then samples a handful of entries.
  On the GPU this copy dominates the run (see "Run").
- `tests/fomInnerDomain/test.cmake` checks `if(RES)` after the FOM run but the
  exit code is stored in `CMD_RESULT`, so a crashing run is only caught
  indirectly (missing snapshot files make `compare.py` fail). The gold
  comparison itself works as intended.
- Kokkos is initialised with the OpenMP host backend enabled, so Kokkos
  prints its usual `OMP_PROC_BIND`/`OMP_PLACES` hint at start-up.

## Dependencies

- Kokkos 5.2.1 (CUDA + OpenMP + Serial, `Kokkos_ARCH_BLACKWELL100`, C++20) --
  `$R/.deps/install/kokkos` (built by `setup_level2_deps.sh`); its
  `bin/nvcc_wrapper` is the C++ compiler for the CUDA build and forwards host
  code to `$CXX` (conda g++) via `NVCC_WRAPPER_DEFAULT_COMPILER`.
- Kokkos Kernels 5.2.1 -- `$R/.deps/install/kokkos-kernels` (ETI: double,
  `int` ordinal, `int`/`size_t` offset, LayoutLeft, CudaSpace/HostSpace).
  SHAW's `CrsMatrix<double, int64_t, Cuda>` and its `KokkosBlas` calls are not
  in the ETI set, so they are compiled inline (`KOKKOSKERNELS_ETI_ONLY` is not
  defined), which is why `main.cc` takes ~80 s to compile.
- yaml-cpp 0.8.0 -- conda (`$CONDA_PREFIX`, `-DYAMLCPP_DIR`).
- CUDA 13.2 (nvcc) and conda GCC 13.3 via `hpcperf_env.sh`.
- Mesh generation (`run.sh`): python3 with numpy and scipy (conda: numpy
  2.5.2, scipy 1.18.0). `validate.sh` needs `xz` (configure time) and python3 +
  numpy (`compare.py`).
- HIP variant: ROCm `hipcc` plus a HIP-enabled Kokkos / Kokkos Kernels
  (expected at `$R/.deps/install/kokkos-hip`, `kokkos-kernels-hip`) -- not
  available on this machine.

## Build / Run / Validate

```bash
source hpcperf_env.sh                 # optional -- the scripts source it themselves
./level2/shaw/build.sh                # CUDA (default); ./build.sh HIP for the HIP variant
./level2/shaw/run.sh                  # demo1 physics on a 1000x5000 grid, 40000 steps (~15-30 s loop)
./level2/shaw/validate.sh             # upstream ctest fomInnerDomain, PASS/FAIL + exit code
```

SHAW is a Kokkos code: one source tree is both variants. `build.sh CUDA`
compiles with `nvcc_wrapper` against the CUDA Kokkos install; `build.sh HIP`
uses `hipcc` and `HPCPERF_KOKKOS_HIP_ROOT` / `HPCPERF_KOKKOSKERNELS_HIP_ROOT`
(default `$R/.deps/install/kokkos-hip`, `kokkos-kernels-hip`) and exits with a
clear message when `hipcc` is missing. `build.sh` detects the GPU
(`nvidia-smi --query-gpu=compute_cap`, override `HPCPERF_CUDA_ARCH`) only to
report it -- the architecture is baked into the Kokkos install. Extra
`-D...` options are forwarded to CMake; `HPCPERF_BUILD_JOBS` (default 4) sets
the parallelism (there is a single translation unit anyway).

Equivalent raw commands (CUDA):

```bash
source hpcperf_env.sh
export NVCC_WRAPPER_DEFAULT_COMPILER=$CXX
cmake -S level2/shaw -B build/level2/shaw/cuda -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20 \
      -DCMAKE_CXX_COMPILER=$PWD/.deps/install/kokkos/bin/nvcc_wrapper \
      -DKOKKOSKERNELS_DIR=$PWD/.deps/install/kokkos-kernels -DYAMLCPP_DIR=$CONDA_PREFIX \
      -DCMAKE_PREFIX_PATH="$PWD/.deps/install/kokkos-kernels;$PWD/.deps/install/kokkos"
cmake --build build/level2/shaw/cuda -j4
# mesh (once) + run
mkdir -p build/level2/shaw/mesh/.work && cd build/level2/shaw/mesh/.work && \
  python3 ../../../../../level2/shaw/meshing/create_single_mesh.py -nr 1000 -nth 5000 -working-dir .. && cd -
build/level2/shaw/cuda/shawExe build/level2/shaw/cuda/run/input.yaml   # yaml as written by run.sh
ctest --test-dir build/level2/shaw/cuda -R fomInnerDomain --output-on-failure
```

HIP (form only, unverified without ROCm):

```bash
cmake -S level2/shaw -B build/level2/shaw/hip -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20 \
      -DCMAKE_CXX_COMPILER=hipcc -DKOKKOSKERNELS_DIR=$PWD/.deps/install/kokkos-kernels-hip \
      -DYAMLCPP_DIR=$CONDA_PREFIX -DCMAKE_PREFIX_PATH="$PWD/.deps/install/kokkos-kernels-hip;$PWD/.deps/install/kokkos-hip"
cmake --build build/level2/shaw/hip -j4
```

### Run

`run.sh` reproduces upstream demo1 (`demos/demo1/input.yaml`, documented in
`docs/`): PREM material, Ricker-wavelet source at 640 km depth (period 65 s,
delay 180 s), 2000 s of simulated propagation, CFL and numerical-dispersion
checks on. demo1 uses a 200x1000 grid (2.0e5 velocity + 4.0e5 stress dofs,
dt 0.25 s, 8000 steps), which takes 0.2 s of GPU time; the default here is a
1000x5000 grid (5.0e6 velocity + 1.0e7 stress dofs, ~1.4 GB of state + matrix
traffic per step) with dt scaled to keep demo1's CFL number (0.05 s, 40000
steps). The mesh is generated once by the upstream python mesher (137 s,
1.3 GB of ASCII files) into `build/level2/shaw/mesh/mesh1000x5000/` and then
cached; reading it takes ~10 s per run. The generated yaml and the outputs go
to `build/level2/shaw/<model>/run/`.

The default run writes no `io:` section (upstream's template documents it as
optional), so the time loop is kernels + one fence per step. demo1 itself also
collects a seismogram at 3 surface receivers and binary snapshots of the full
state every 100 steps; SHAW implements both with a full device->host copy of
the state at every step, which on the GPU costs 5-10x more than the kernels.
`HPCPERF_SHAW_SEISMOGRAM=1` adds demo1's seismogram section (writes
`seismogram_0`). Snapshots are not offered (up to GBs of output).

Observed on the B200 (`loopTime` = wall time of the time loop; `run.sh` wall
time adds ~3 s Kokkos/CUDA start-up plus mesh reading):

| grid | dofs (vp + sp) | dt / steps | io | loopTime | run.sh wall |
|---|---|---|---|---|---|
| 1000x5000 (default) | 5.0e6 + 1.0e7 | 0.05 / 40000 | none | **15.5 s** quiet node; 18.3 s and 29.8 s with 4 concurrent compilations on the node | 28-42 s |
| 500x2500 | 1.25e6 + 2.5e6 | 0.1 / 20000 | none | 2.5 s | 6 s |
| 500x2500 | 1.25e6 + 2.5e6 | 0.1 / 20000 | seismogram | 42.6 s (42.1 s in the per-step D2H copy) | 46 s |
| 200x1000 (demo1 grid) | 2.0e5 + 4.0e5 | 0.25 / 8000 | seismogram | 13.3 s (13.1 s in the D2H copy) | 14 s |

For the default case 15.5 s / 40000 steps = 390 us per step for ~1.41 GB of
modelled traffic, i.e. ~3.6 TB/s effective; the loop issues 4 small kernels
plus a fence per step, so it is sensitive to host-side latency (hence the
spread under CPU load). The printed `aveBandwidth`/`aveGFlop` are
launch-time artefacts, see "Changes from upstream".

Environment overrides for `run.sh`: `HPCPERF_SHAW_NR`, `HPCPERF_SHAW_NTH`
(grid; the mesh is generated and cached per size), `HPCPERF_SHAW_DT` (default
derived from the grid so that the CFL number stays at demo1's 0.235 and the
step count is a multiple of the seismogram frequency), `HPCPERF_SHAW_FINAL_TIME`
(2000), `HPCPERF_SHAW_SEISMOGRAM=1`, `HPCPERF_SHAW_INPUT=<yaml>` (run an
arbitrary input; no mesh generation), `HPCPERF_SHAW_MESH_DIR`,
`HPCPERF_PYTHON`. Extra arguments are passed to `shawExe` after the yaml
(e.g. `--kokkos-num-threads=1`). Note that SHAW aborts by design if the grid is
too coarse for the source period ("Numerical dispersion criterion violated",
e.g. 100x500) or the CFL number exceeds 0.28.

## Validation

`validate.sh` runs upstream's own end-to-end regression test through ctest
(`ctest --test-dir build/level2/shaw/<model> -R fomInnerDomain`). The test
(`tests/fomInnerDomain/test.cmake`) runs `shawExe` on the 21x51 mesh
(`tests/fullMesh21x51`) with `tests/fomInnerDomain/input.yaml`: dt 1 s, 150
steps, sinusoidal source at 1100 km depth (period 40 s, delay 10 s),
single-layer material (rho 2000 kg/m^3, vs 5000 m/s), ASCII snapshots of the
full velocity and stress fields at every step and a seismogram at 4 receivers.
It then compares the complete snapshot matrices with `numpy.allclose` against
the gold files generated upstream (CPU):

- `snaps_vp_0` (1071 velocity dofs x 150 steps) vs `snaps_vp_0_gold`, atol 1e-13
- `snaps_sp_0` (2070 stress dofs x 150 steps) vs `snaps_sp_0_gold`, atol 1e-10

(the seismogram is written but not compared upstream). `validate.sh` forwards
ctest's verdict as the final line `SHAW CUDA validation (fomInnerDomain): PASS|FAIL`
with exit code 0/1 and prints the observed differences.

Observed here (CUDA, B200):

```
1/1 Test #1: fomInnerDomain ...................   Passed    3.87 sec
snaps_vp_0 shape (1071, 150)  max|gold| 2.844e-05  max|diff| 2.499e-20  atol 1e-13  ok
snaps_sp_0 shape (2070, 150)  max|gold| 2.777e+02  max|diff| 1.137e-13  atol 1e-10  ok
SHAW CUDA validation (fomInnerDomain): PASS
```

i.e. the GPU results agree with the CPU gold data to rounding (relative
1e-15). A deliberately perturbed gold file was verified to produce `FAIL` and
exit code 1.

## Warnings

Clean build (`build/level2/shaw_build.log`, nvcc 13.2 + GCC 13.3, one
translation unit `main.cc`, 88 s):

- 6x nvcc `Warning #20014-D: calling a __host__ function from a __host__
  __device__ function is not allowed`. All originate inside Kokkos / Kokkos
  Kernels headers (`Kokkos::Impl::ViewMapping::assign` subview construction
  inside `KokkosBlas::Impl::MV_Scal_Invoke_Left` <- `KokkosBlas::scal` <-
  `KokkosBlas::gemm` on the LayoutRight ROM Jacobians), instantiated from the
  ROM problem (`rom_compute_jacobians.hpp:92`, 4x) and the rank-2 FOM problem
  (`fom_problem_rank_two.hpp:173-174`, 2x). Host-only code paths; harmless.
  None come from SHAW's own code.
- 2x `nvcc_wrapper - *warning* you have set multiple optimization flags
  (-O*)`: conda's `CXXFLAGS` carries `-O2` and CMake Release adds `-O3`; nvcc
  uses the last one (`-O3`).
- CMake: none.

## LOC

cloc v2.06, code lines only (blank/comment excluded). SHAW is a Kokkos code:
the CUDA and the HIP variant are the same source tree, `src/` (60 files).
CMake files, yaml, test data, the python mesher (749 lines), scripts and this
README are excluded.

CUDA variant: **4120** (60 files) = `src/kokkos/` 1745 (15 .hpp + main.cc) + `src/shared/` 2375 (44 .hpp)
HIP variant:  **4120** (same files)

```bash
cloc --quiet level2/shaw/src
```

## Verified Environment

GCC/G++ 13.3.0 (conda, pinned) | C++20 | CMake 3.28.4 | Ninja 1.13.2 | Python 3.12.3
CUDA Toolkit 13.2 (nvcc 13.2.78, /usr/local/cuda) | NVIDIA B200 (sm_100), driver 595.58.03
Kokkos 5.2.1 + Kokkos Kernels 5.2.1 (`$R/.deps/install`, CUDA/OpenMP/Serial, BLACKWELL100) | yaml-cpp 0.8.0 (conda)
HIP/ROCm: source + build config present where noted, unverified (no AMD GPU available)
OS: RHEL 10 (Linux 6.12)

Reproduce the toolchain from the repository root: `./setup_env.sh` then
`source hpcperf_env.sh` (all user-space versions are pinned in environment.yml);
framework libraries with `./setup_level2_deps.sh`.

## Status

CUDA: **Working** -- configure + build (88 s clean, warnings as listed) + run
(`run.sh`, 1000x5000, 40000 steps, loopTime 15.5-29.8 s depending on node
load) + validate (`fomInnerDomain` PASS) all succeeded on this machine, from
a foreign cwd. The ROM code path was made to compile (required by `main.cc`)
but is not exercised.
HIP: same source tree, untested (no ROCm/hipcc and no HIP-enabled Kokkos on
the development machine; `./build.sh HIP` exits 1 with an explanatory message).
