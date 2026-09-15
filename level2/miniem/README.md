# MiniEM

Electromagnetics mini-app from Trilinos (Panzer `mini-em`, Sandia). It time-steps the first-order Maxwell
system for the electric field `E` (edge elements) and the magnetic flux `B` (face elements) on an inline-
generated hexahedral or tetrahedral mesh, assembled with Panzer/Phalanx/Intrepid2 and solved implicitly every
step with a block preconditioner: Teko drives the 2x2 block system and MueLu's **RefMaxwell** algebraic
multigrid solves the curl-curl edge-element block, inside a Belos Krylov solver. Motif: implicit
finite-element electromagnetics -- unstructured sparse matrix assembly, block-preconditioned Krylov iterations
and algebraic multigrid setup/solve (SpMV, sparse products, smoothers) on Tpetra/Kokkos, all memory-access
bound and communication heavy. Upstream uses it as its own GPU performance test
(`MiniEM-BlockPrec_RefMaxwell_Performance`, 4 and 16 ranks). Figure of merit printed by the driver: cells per
second of solve time (`--print-fom`, default on).

## Source

Upstream repository: https://github.com/trilinos/Trilinos (`packages/panzer/mini-em`)
Upstream commit: efbab1057fdf0dba7d97a8a417057a817d07ada1 (develop, 2026-09-01; the same commit
`setup_level2_deps.sh` pins for the Trilinos dependency, see below)
License: Trilinos, BSD-3-Clause style per package (Panzer: BSD 3-Clause) -- the top-level Trilinos `LICENSE`
is copied verbatim.

MiniEM is not a stand-alone code. Its physics -- closure models, equation sets, responses and the RefMaxwell
solver setup -- is the Trilinos library **PanzerMiniEM** (`mini-em/src`, 6,093 cloc code lines, 70 files) and
is built inside Trilinos. What runs is the **BlockPrec driver** (`mini-em/example/BlockPrec`): `main.cpp`,
`MiniEM_helpers.cpp/.hpp` (1,343 code lines) and 41 input decks (`*.xml`, 5,030 lines). Those files are copied
here byte-identically into `src/` and `src/decks/` (`src/UPSTREAM_SHA256SUMS`, sha256 of every copied file at the
pinned commit). Not copied: the TriBITS `CMakeLists.txt` of the example (see "Changes from upstream").

## Changes from upstream

- No source change. `main.cpp`, `MiniEM_helpers.cpp`, `MiniEM_helpers.hpp` and the decks compile and run
  unmodified with CUDA 13.2 / GCC 13.3 / Kokkos 5.2.1 against this repo's Trilinos.
- `src/CMakeLists.txt` is new: upstream builds the driver only through TriBITS as part of a Trilinos source
  build (`TRIBITS_ADD_EXECUTABLE(BlockPrec ...)`, not installed). Here the driver is compiled as a stand-alone
  CMake project against the exported targets of the installed Trilinos (`find_package(Trilinos ...)`,
  `Trilinos::all_selected_libs`), with the same compiler recipe the dependency used (mpicxx over Trilinos'
  `nvcc_wrapper`, host compiler = the project GCC). The decks are copied next to the binary so the relative
  solver-file names inside them (`solverMueLu.xml`, ...) resolve exactly as in upstream's test directory.
- `build.sh`, `run.sh`, `validate.sh`, `LICENSE` and this `README.md` are new files added by this integration.

## Dependencies

- **Trilinos at commit efbab105 (develop)**, CUDA + Serial Kokkos back ends, `$R/.deps/install/trilinos`,
  built once by `$R/setup_level2_deps.sh trilinos` (multi-hour; ~20 GB build tree). Package set: PanzerMiniEM
  and what it requires -- Panzer (Core, DofMgr, DiscFE, AdaptersSTK), Phalanx, Intrepid2, Sacado, Tpetra,
  Kokkos/KokkosKernels (Trilinos' bundled 5.2.1, the same version as `.deps/install/kokkos`), Belos, Ifpack2,
  Amesos2, MueLu, Teko, Thyra, Stratimikos, Piro, NOX, Zoltan, **Zoltan2** (MueLu RefMaxwell repartitions the
  coarse problems whenever more than one rank runs -- upstream's solver decks enable it -- and throws
  "Zoltan2 interface is not available" without it; found the hard way at 2 ranks), STK (Util, Topology, Mesh,
  IO, Tools), SEACAS Ioss/Exodus. Off: tests, examples, Fortran, Epetra, ML, OpenMP, complex/float
  instantiations.
  Why a develop commit and not the newest release: 16.2.2 (2026-08-17) still bundles Kokkos 4.7 while every
  other Level 2 dependency is built against Kokkos 5.2.1; this commit bundles 5.2.1, so the Level 2 stack keeps
  one Kokkos version.
- TPLs from the project conda environment (`environment.yml`): Open MPI 5.0.10 (CUDA-aware build), OpenBLAS
  0.3.34 (BLAS/LAPACK), parallel HDF5 2.2.0, **netCDF 4.10.1 (`libnetcdf`, MPI build, added for MiniEM)** and
  PnetCDF 1.15.0 -- Exodus/Ioss need netCDF even though every deck here uses the inline mesh factory
  (STK I/O is a required dependency of the Panzer STK adapters); **GoogleTest 1.17.0 (`gtest`, added for MiniEM)**
  -- the STK packages declare the `gtest` TPL as required and TriBITS enables it even with tests off, so it is
  provided as a pinned conda package instead of Trilinos' configure-time download. CUDA 13.2 with cuBLAS/cuSPARSE
  from the system toolkit.

## Verified Environment

dgx003: 4 x NVIDIA B200 (sm_100), CUDA 13.2.78 (driver 595.58.03), conda GCC 13.3.0 host compiler through
Trilinos' nvcc_wrapper, Open MPI 5.0.10 (conda, CUDA-aware build; `--mca pml ob1 --mca btl self,sm,smcuda`), CMake
3.28.4 / Ninja 1.13.2, Trilinos develop efbab105 (reports itself as 17.3.0) with bundled Kokkos 5.2.1.

## Backends

CUDA (Kokkos::Cuda execution space through Tpetra; `--linAlgebra=Tpetra --solver=MueLu`). HIP: not available
-- the Trilinos dependency is built for CUDA only on this site; `build.sh HIP` exits 1 with an explanation.

## Build

```bash
source hpcperf_env.sh
./setup_level2_deps.sh trilinos            # once; hours; .deps/install/trilinos
level2/miniem/build.sh CUDA                # minutes; build/level2/miniem/cuda/PanzerMiniEM_BlockPrec (+ decks/)
```

`HPCPERF_TRILINOS_PREFIX` overrides the dependency prefix, `HPCPERF_BUILD_JOBS` the parallelism (default 4).

## Run

```bash
HPCPERF_GPUS=1 HPCPERF_SCALE_MODE=smoke  level2/miniem/run.sh CUDA   # maxwell.xml, 15^3 hex, 1 step (upstream's Maxwell_MueLu order1)
HPCPERF_GPUS=4 HPCPERF_SCALE_MODE=strong level2/miniem/run.sh CUDA   # maxwell-large.xml (tet), fixed global HPCPERF_MINIEM_GLOBAL^3 (64^3), 3 steps
HPCPERF_GPUS=4 HPCPERF_SCALE_MODE=weak   level2/miniem/run.sh CUDA   # same deck, fixed HPCPERF_MINIEM_N^3 (48^3) elements per rank
```

One MPI rank per GPU through `level2/tools/hpcperf_mpi_launch.sh` (GPU binding wrapper + audit). The inline
mesh factory decomposes the global element grid over the ranks itself (`X/Y/Z Procs = -1`); the weak deck
scales the global grid with the topology from `hpcperf_topology.py`. Mesh sizes and the deck can only be set
through `HPCPERF_MINIEM_N`, `HPCPERF_MINIEM_GLOBAL`, `HPCPERF_MINIEM_STEPS`, `HPCPERF_MINIEM_DECK`; passing
`--x-elements`/`--inputFile` directly is refused so a run is always described by the checked variables. Extra
arguments (e.g. `--basis-order=2`, `--matrixFree`) are appended and win.

## Validation

`validate.sh` reproduces upstream's own analytic-solution test (`Maxwell_MueLu_order1_analytic`):
`maxwell-analyticSolution.xml` sets a Maxwell problem with an analytic forcing term and the exact `E` field as a
closure model, and the driver asserts that the L2 error of the computed `E` field is below **0.065**
(`main.cpp`, `TEUCHOS_ASSERT_INEQUALITY` on `L2 Error E maxwell - analyticSolution`). The script (1) runs that
case at 1 rank and requires exit 0, (2) parses the printed L2 error and re-checks the same 0.065 bound so the
number is visible in the log, and (3) with `HPCPERF_GPUS=N > 1` repeats the case at N ranks and requires the
L2 error to agree with the 1-rank value to a relative 1e-4 (rank-count consistency). GPU execution is
recorded through the launcher's binding audit; a run shorter than the nvidia-smi sampling window reports
"unverified" (an observation gap, not a mismatch).

Results (2026-09-14, this node): L2 error E = **0.0566793** at 1 rank, **0.0566793** at 2 ranks, **0.0566793** at 4 ranks, all below
0.065 and consistent across rank counts to the 1e-4 relative tolerance; the deck runs to its `final time` 5e-9 in 6
implicit steps (the driver's `--numTimeSteps` is superseded by the deck's final time). GPU binding audit: N verified,
0 mismatch at every rank count.

## Warnings

- **GPU-aware MPI is off for MiniEM** (`TPETRA_ASSUME_GPU_AWARE_MPI=0`, exported by `run.sh`): Tpetra asks Open MPI
  whether it is CUDA-aware, this conda build answers yes through `smcuda`, and Tpetra then passes device pointers
  to MPI. On this node that path **hung at 4 ranks** inside the RefMaxwell setup (all ranks spinning at 100 % CPU
  in a Tpetra import, no progress for 12 minutes) -- the same `smcuda` transport the Level 2/3 notes already flag
  as slow or unreliable. Every other Level 2 dependency is built with GPU-aware MPI off, so MiniEM stages its
  communication buffers through the host too. Set `TPETRA_ASSUME_GPU_AWARE_MPI=1` to test the device path.
- Wall clock is dominated by start-up: the driver is a ~480 MB statically linked executable read from the network
  file system; the analytic validation case itself takes about one second of Mini-EM time.
- The Trilinos dependency is a **develop** commit, pinned by hash and fetched by commit; it is not a Trilinos
  release. Rebuilding it needs hours and about 20 GB.
- Static Trilinos libraries (as every Level 2 dependency); the driver links ~60 of them.
- Exodus output is off by default (`--exodus-output` writes mesh files through Ioss/netCDF); the validation
  does not exercise it.

## LOC

CUDA: 1,343 (cloc code lines: `main.cpp` + `MiniEM_helpers.cpp` 1,253 C++, `MiniEM_helpers.hpp` 90 header).
The physics library (PanzerMiniEM, 6,093 lines) and everything below it (Panzer, MueLu, Tpetra, Kokkos, ...)
are the Trilinos dependency and are not counted. The 41 decks (5,030 XML lines), `CMakeLists.txt`, the
license and the three scripts are excluded.

## Status

Working (CUDA): Trilinos dependency built (2026-09-14, ~30 min at 48 jobs), driver built, smoke run
(`maxwell.xml`, 15^3 hex, 1 step: FOM 40.5 k-cell-steps/s, Belos solve 0.083 s), validation PASS at 1, 2 and 4 GPUs
(analytic L2 error 0.0566793 / 0.0566793 / 0.0566793 < 0.065), strong and weak decks completed at 1 and 4 GPUs -- completion and the
driver's own figure of merit only, **no scaling claim**:

| mode | GPUs | global elements (tets) | cells | Belos solve (3 steps) | FOM k-cell-steps/s | Mini-EM total |
|---|---|---|---|---|---|---|
| strong | 1 | 64^3 | 3,145,728 | 0.36 s | 26,024 | 285 s |
| strong | 4 | 64^3 | 3,145,728 | 1.17 s | 8,067 | 84 s |
| weak | 1 | 48^3 per rank (1x1x1) | 1,327,104 | 0.20 s | 19,604 | 110 s |
| weak | 4 | 48^3 per rank (2x2x1) | 5,308,416 | 2.24 s | 7,113 | 147 s |

Read these with care. The FOM counts only the Belos solve, a few tenths of a second here, so at 4 ranks it is
dominated by communication that goes through the host (`TPETRA_ASSUME_GPU_AWARE_MPI=0`, see Warnings) and comes
out **lower** than at 1 rank; the total Mini-EM time is dominated by mesh, assembly and the RefMaxwell setup and
does shrink from 285 s to 84 s on the strong deck. Whether the device-buffer MPI path can be made to work on this
site (it hangs today) decides how MiniEM's solve phase scales; that is an open item, not a result. Single node only;
multi-node is BLOCKED site-wide. HIP: not available (Trilinos dependency is CUDA-only here).
