# nekRS (Level 3)

Spectral-element incompressible Navier-Stokes (pressure Poisson with p-multigrid
+ HYPRE coarse grid, velocity Helmholtz solves, subcycled advection, passive
scalars, gather-scatter halo exchange) -- the complete solver driven by its own
case files, with all GPU kernels JIT-compiled by OCCA at run time.

## Provenance

- Official repository: https://github.com/Nek5000/nekRS (`master` = latest
  stable release); docs https://nekrs.readthedocs.io/
- Release policy: tagged releases on `master` (v26.0 2026-01-27; previous
  v23.0 2023-05); `next` is the preview branch.
- Selected: **v26.0**, commit `96b3cf9e5bacede16568826c04a21bc0fe50dc7d`,
  fetched by `fetch.sh` into `_upstream/level3/nekRS` (shallow, read-only).
- License: BSD-3-Clause.
- Application-owned LOC (cloc 2.06, code lines): `src/` **53,131** (C++
  35,784; headers 12,322; C 3,782; Fortran 1,104). Vendored third-party
  libraries in `3rd_party/` are counted separately (~2.27 M: LAPACK 829,795,
  ADIOS2 618,827, HYPRE 445,492, CVODE 169,316, OCCA 89,888, Nek5000 87,143,
  gslib 13,038, parRSB 6,450).

## Build strategy: NATIVE (upstream CMake, vendored libraries)

`build.sh CUDA` = upstream's CMake route (`build.sh` upstream is only a
wrapper with interactive prompts): `CC=mpicc CXX=mpicxx FC=mpif90 cmake -G
"Unix Makefiles" -DOCCA_ENABLE_CUDA=ON -DOCCA_ENABLE_HIP=OFF
-DOCCA_ENABLE_DPCPP=OFF -DENABLE_HYPRE_GPU=ON -DENABLE_ADIOS=OFF
-DENABLE_CVODE=OFF -DNEKRS_BUILD_FLOAT=OFF`, install = `NEKRS_HOME` =
`.deps/level3/nekrs/install` (with `nekrs.conf` recording the JIT toolchain:
`OCCA_CXX` = conda g++ 13.3.0, `OCCA_CUDA_COMPILER_FLAGS = -w -O3 -lineinfo
--use_fast_math`, `NEKRS_GPU_MPI = 0`). Toolchain: conda GCC 13.3.0 through the
conda Open MPI 5.0.10 wrappers for C/C++, system gfortran 14.2.1 through
`mpif90` (`OMPI_FC`) for the Nek5000 interface and the vendored LAPACK, CUDA
13.2.78 for HYPRE's device build and for the run-time JIT. OKL kernels are
compiled for the device found at run time (sm_100 here); HYPRE's device
kernels are compiled for sm_80/90/100 (patch below). Build time: the single
full compile of all vendored libraries + nekRS took ~30 min at `-j32` (its
exact wall time was not captured because the first complete pass failed at the
install step; the final incremental install took 27 s); **0 compiler warning
lines** in the final pass. Fingerprint: upstream commit, vendored library
versions, compilers, CUDA 13.2.78, MPI, CMake options, patch list.

Why not the others: the Spack `nekrs` recipe is stale (23.0, option names that
no longer exist); all dependencies are vendored, so Spack would add nothing;
no Apptainer on the node and no upstream image; JIT couples nekRS to the host
compiler/nvcc anyway; site modules broken. HIP: `build.sh HIP` selects
`OCCA_ENABLE_HIP` and exits with a clear message here (no ROCm). **Untested.**

## Changes from upstream (all recorded in `patches/` and `build.sh`)

| Class | Change | Size / reason |
|---|---|---|
| B | `0001-hypre-cuda-sm100.patch`: `cmake/hypre.cmake` `HYPRE_CUDA_SM=80 90` -> `80 90 100` for CUDA >= 13 | 1 line; upstream lists no Blackwell SASS (SASS only, no PTX) |
| D | `0002-hypre-cuda13-thrust-pair.patch`: vendored HYPRE 2.32.0 declares `thrust::reduce_by_key` results as `thrust::pair<...>`, a name the Thrust 3.2 of CUDA 13 no longer exposes there -> `auto` | 2 lines |
| D | `0003-hypre-cuda13-thrust3-compat.patch`: explicit `<thrust/iterator/reverse_iterator.h>` / `<thrust/pair.h>` includes (no longer transitive) in HYPRE's `device_utils.h` **and** in the pre-generated concatenated `_hypre_utilities.hpp` the sources include; `thrust::not1` (removed) -> its documented replacement `thrust::not_fn` (16 uses) | 36 lines, mechanical |
| B | CMake generator pinned to upstream's Unix Makefiles (`build.sh`): the conda environment exports `CMAKE_GENERATOR=Ninja`, under which HYPRE's ExternalProject install rule (`$(MAKE) install`) is invalid | build.sh only |
| C | `OMPI_FC=/usr/bin/gfortran` (no conda gfortran); `LDFLAGS += -fno-lto` (CMake's FortranCInterface probe compiles with `-flto -ffat-lto-objects`; GCC 13 bytecode vs the gfortran 14 link driver); `FFLAGS += -fPIC` (conda GCC links PIE); `unset AR` (HYPRE's configure takes `$AR` as the full archive command; conda exports the bare tool); run time: `unset CMAKE_GENERATOR` (UDF build), `OMPI_MCA_osc=^ucx` (see execution model), `ulimit -s unlimited` (as upstream's `nrsqsub_utils`: the Nek5000 side keeps lx1^3*lelt work arrays on the stack; the h-refined cases segfault in `useric` at the default 8 MB) | environment |

No numerics changed. HYPRE 2.32.0 + CUDA 13 is not an upstream-validated
combination; the patches only restore names the CCCL removed.

## Execution model

One MPI rank per GPU (upstream: "NekRS binds 1 GPU to 1 MPI rank"). nekRS'
default is `--device-id LOCAL-RANK`; under the common launcher's per-rank
wrapper each rank sees exactly one GPU, so `run.sh` passes `--device-id 0` and
the launcher audits the mapping (4/4 verified). GPU-aware MPI is upstream's
default OFF (`NEKRS_GPU_MPI=0`; RELEASE.md warns enabling it "may cause a
performance regression"); `HPCPERF_NEKRS_GPU_MPI=1` turns it on. nekRS uses
MPI one-sided operations (`MPI_Win_lock`): with the default Open MPI selection
these went through `osc ucx` -> UCX/InfiniBand even on one node and aborted
in `uct_ib` with 4 ranks ("'abort' is not implemented for protocol
amo64/fetch"); `run.sh` sets `OMPI_MCA_osc=^ucx`, the one-sided counterpart of
the site profile's `pml ob1 / btl self,sm,smcuda`, and 4 ranks then pass
(candidate for the gmu-hopper site profile). Rank count is unconstrained
(parRSB graph partitioning). The first run of a new kernel set JIT-compiles
OKL kernels with nvcc into `build/level3/nekrs/cuda/cache` (minutes; shared by
later runs), and the case's `.usr` file is compiled with `mpif90` at run time.

## Inputs (`HPCPERF_SCALE_MODE`, case `examples/ethier`)

| Mode | Elements | Order N | Grid points | Per rank @4 GPU | Steps | GPU memory (est.) | Validation quantity |
|---|---|---|---|---|---|---|---|
| smoke (default) | 32 (upstream `ethier.par`) | 9 | 32,000 | 8 elements | 100 (CI mode: 30) | < 0.1 GB | upstream `--cimode 2` CI checks (analytic solution) |
| strong | 32 x H^3, H=`HPCPERF_NEKRS_HREFINE` (10): 32,000 (`ethierRefine.par`, `hrefine`) | 7 | 16.4 M | 8,000 | 100 | ~2 GB | run completes; L2 errors vs exact solution printed every step |
| weak | 32 x H^3 with H = round(cbrt(250 N)) -> ~8,000 elements/rank (upstream's reference load, kershaw README "E/GPU=8000") | 7 | 4.1 M x N | 6.9k-8.5k (integer H) | 100 | ~2 GB | run completes |

Memory estimate ~60 KB per element at N=7 (velocity, pressure, two scalars,
multistep history, preconditioner). Derived `ethier.par` files change only
`hrefine` and `numSteps` (class A).

## Coarse-solver location: what runs on the GPU (read `COMPATIBILITY.md`)

The ethier case's HYPRE BoomerAMG coarse solve runs where the cimode selects:

- `--cimode 2` (and the default `.par`): `FLUID PRESSURE MULTIGRID COARSE SOLVER
  LOCATION = CPU` -- the coarse solve is on the **host**. The nekRS main
  application (advection, Helmholtz, pressure pMG smoother, gather-scatter) runs
  on the **GPU** via OCCA/CUDA. So this is not a CPU-only application, but a
  cimode-2 PASS does **not** exercise GPU HYPRE.
- `--cimode 3`: `FLUID PRESSURE PRECONDITIONER = MULTIGRID+SEMFEM`, `COARSE
  SOLVER LOCATION = DEVICE` -- the coarse solve runs on the **GPU** (this is the
  mode that exercises the GPU HYPRE built by `ENABLE_HYPRE_GPU=ON` and the three
  patches).

Two build variants exist (isolated src/build/install/JIT-cache; select with
`HPCPERF_NEKRS_HYPRE_GPU`/`HPCPERF_NEKRS_VARIANT`):

| variant | `ENABLE_HYPRE_GPU` | patches | GPU HYPRE coarse (cimode 3) | CPU coarse (cimode 2) |
|---|---|---|---|---|
| `hypregpu` (default) | ON | 0001+0002+0003 | built + **verified** (cimode 3, 1/4 GPU, 9/9, coarse=DEVICE) | verified 1/2/4 |
| `cpucoarse` (candidate) | OFF | none | not built; a DEVICE request is **explicitly rejected** by nekRS (`HYPRE+DEVICE not enabled!`, exit 1), no silent fallback | verified 1/2/4 (candidate, 0 patches, 113 s build) |

Build/run isolation per variant: `hypregpu` keeps the legacy paths
(`.deps/level3/nekrs/{src,install}`, `build/level3/nekrs/cuda`); other variants
use `.deps/level3/nekrs/<variant>/{src,install}` and `build/level3/nekrs/<variant>.<backend>`
with their own OCCA/nekRS JIT cache. The source-copy cache key is the upstream
SHA plus the ordered patch-content hash, so the two variants never share a
patched/unpatched tree.

Which to make default is a decision for review: `cpucoarse` is minimal (no
patches, fast build) and covers the current CPU-coarse workload; `hypregpu` is
required if a case selects GPU (DEVICE) coarse. CPU-coarse scalability at 40/80
GPUs is UNVERIFIED and is not claimed to be optimal at all scales.

## Validation (`validate.sh`, upstream mechanism)

`nekrs --setup ethier --cimode 2` is one of the modes upstream's CI runs on this
case (`.github/workflows/ci.yml`): it fixes the solver settings (velocity
solver +BLOCK, subcycling 1, tolerances 1e-12/1e-10, 30 steps) and at the last
step checks the L2 errors of velocity, pressure and both scalars against the
exact Ethier-Steinman solution (references in `examples/ethier/ci.inc`,
relative tolerance EPS = 0.3) plus the iteration counts of the pressure,
velocity and scalar solves. nekRS prints `CI test <...> passed|failed` per
check and exits non-zero on failure; `validate.sh` uses that verdict unchanged
(upstream runs it on CPUs with 2 ranks; here the CUDA backend on 1, 2 and 4
ranks). The validator now (a) captures the run's real exit code (nonzero/timeout ->
FAIL), (b) requires the COMPLETE set of CI checks (9 for cimode 2/3, not merely
"some passed"), (c) asserts the coarse-solver LOCATION recorded in the log
matches the cimode (CPU for 2, DEVICE for 3 -- no silent fallback), and (d)
rejects NaN/Inf. `HPCPERF_NEKRS_CIMODE` selects the mode.

Observed on dgx003 (2026-09-05), `hypregpu` variant:
- `--cimode 2` (CPU coarse): **PASS at 1, 2, 4 GPUs, 9/9 checks, coarse=CPU**.
- `--cimode 3` (DEVICE / GPU HYPRE coarse): **PASS at 1 and 4 GPUs, 9/9 checks,
  coarse=DEVICE** -- this is the run that actually exercises the GPU HYPRE coarse
  solve the three patches enable, verified against the analytic solution.
Earlier CI L2 errors (cimode 2): velocity 2.78e-10, pressure 6.98e-10, scalars
6.67e-12 / 7.49e-12 (CI references 2.77e-10 / 7.14e-10 / 7.49e-12 / 7.22e-12).

## Results on dgx003 (4x B200, CUDA 13.2.78, Slurm job 9552083)

| Run | Ranks x GPUs | rank->GPU | CPU binding | Topology | Problem | Time | Validation |
|---|---|---|---|---|---|---|---|
| smoke/CI | 1 x 1 | wrapper, `--device-id 0`; audit 1/1 verified | runtime default | parRSB | 32 el., N=9, 30 steps | 1.18 s for 30 steps (8.9 ms/step) | PASS 9/9 |
| smoke/CI | 2 x 2 | wrapper; 2/2 verified | runtime default | parRSB | same | 1.66 s (24.7 ms/step) | PASS 9/9 |
| smoke/CI | 4 x 4 | wrapper; 4/4 verified | runtime default | parRSB | same | 1.40 s | PASS 9/9 |
| strong | 1 x 1 | wrapper; 1/1 verified | runtime default | parRSB | 32,000 el., N=7 (16.4 M points), 100 steps | 177.2 s (1.77 s/step) | completes; L2 err vs exact at step 100: u 2.88e-11, p 1.69e-10, s00 1.38e-10, s01 1.39e-10 |
| strong | 4 x 4 | wrapper; 4/4 verified | runtime default | parRSB | 32,000 el. (8,000/rank) | 62.5 s (0.63 s/step) | completes; L2 err u 2.87e-11, p 1.71e-10, s00 1.38e-10, s01 1.39e-10 |
| weak | 4 x 4 | wrapper; 4/4 verified | runtime default | parRSB | H=10 -> 32,000 el. (8,000/rank; coincides with strong at N=4) | 65.6 s (0.66 s/step) | completes; same L2 errors |

The 32-element CI case is launch/communication bound (2 and 4 GPUs are slower
per step than 1); it is a correctness case, not a scaling result. The
h-refined runs keep upstream's CI-oriented solver tolerances
(`residualTol` 1e-12 velocity/scalars, 1e-8 pressure), so their step times
are not performance figures either; the 1 -> 4 GPU ratio (2.8x) is recorded
as observed.

Dry-runs (`HPCPERF_DRY_RUN=1`, hypothetical allocations) -- **DRY-RUN /
UNVALIDATED**, nothing executed:

| GPUs | Nodes x GPUs/node | Mode | hrefine | Elements | Per rank | Launch |
|---|---|---|---|---|---|---|
| 8 | 1 x 8 | strong | 10 | 32,000 | 4,000 | `mpirun -np 8 --host dgx003:8 --map-by ppr:8:node ...` (single node) |
| 40 | 5 x 8 | weak | 22 | 340,736 | 8,518 | 5 nodes x 8 -- multi-node BLOCKED on this site |
| 80 | 10 x 8 | weak | 27 | 629,856 | 7,873 | 10 nodes x 8 -- multi-node BLOCKED on this site |

## Limitations

- Multi-node: BLOCKED/UNVERIFIED on this site; 40/80-GPU shapes are plans
  (multi-node also needs the JIT cache strategy `NEKRS_CACHE_LOCAL/BCAST`).
- HIP: untested; no gfx950 statement upstream.
- Vendored HYPRE 2.32.0 needed CUDA 13 compatibility patches (above); the
  combination is not upstream-validated.
- Only the ethier family is wrapped (CI case + h-refined strong/weak); turbPipe,
  kershaw (needs `genbox`, not shipped), tgv, pb146 build with this install
  but have no wrappers yet. ADIOS2 checkpointing and CVODE are not built.
