# Level 2: Proxy Applications / Mini-Apps

20 selected GPU mini-apps / proxy applications, each copied from its upstream
repository into `level2/<name>/` together with the input decks it needs and
made to build and run with the pinned Level 1 toolchain (conda GCC 13.3.0,
CMake 3.28.4, system CUDA 13.2) plus the Level 2 additions below. Unlike
Level 1, the upstream build systems are kept (CMake or GNU make); each
directory adds three thin wrappers and a README:

```
level2/<name>/
├── README.md        provenance (repo, commit, license), changes from upstream, deps, validation, warnings, LOC, status
├── LICENSE          upstream license (copied verbatim)
├── build.sh         ./build.sh [CUDA|HIP]   -> configures + builds into build/level2/<name>/<cuda|hip>
├── run.sh           ./run.sh   [CUDA|HIP]   -> the standard GPU-sized benchmark problem
├── validate.sh      ./validate.sh [CUDA|HIP] -> correctness check, prints PASS/FAIL, exit 0/1
└── <upstream sources, decks, data>
```

Build and validate one mini-app (after `source hpcperf_env.sh`, and after
`./setup_level2_deps.sh` if the mini-app needs one of the framework libraries):

```bash
level2/<name>/build.sh CUDA
level2/<name>/run.sh CUDA
level2/<name>/validate.sh CUDA
```

Nothing is hardcoded to the B200: `build.sh` detects the GPU compute
capability (`HPCPERF_CUDA_ARCH` overrides it), and all scripts work from any
cwd and any clone path.

## Additional environment for Level 2

Everything below is installed inside the clone, next to the Level 1 toolchain
(see the root [README](../README.md) for the base environment and
[environment.yml](../environment.yml) for the pins).

| Component | Version | Provided by |
|---|---|---|
| Open MPI (CUDA-aware) | 5.0.10 | conda (`setup_env.sh`) |
| METIS | 5.1.0 | conda |
| yaml-cpp | 0.8.0 | conda |
| OpenBLAS | 0.3.34 | conda |
| FFTW (openmpi) | 3.3.11 | conda |
| HDF5 (parallel, openmpi) | 2.2.0 | conda |
| PnetCDF (openmpi) | 1.15.0 | conda |
| Kokkos / Kokkos Kernels | 5.2.1 | `setup_level2_deps.sh` -> `.deps/install/kokkos{,-kernels}` |
| Cabana | 0.8.0 | `setup_level2_deps.sh` -> `.deps/install/cabana` |
| heFFTe | 2.4.1 | `setup_level2_deps.sh` -> `.deps/install/heffte` |
| RAJA / Umpire / CHAI | 2026.07.0 | `setup_level2_deps.sh` -> `.deps/install/{raja,umpire,chai}` |
| hypre (MPI + CUDA) | 3.2.0 | `setup_level2_deps.sh` -> `.deps/install/hypre` |
| MFEM (MPI + CUDA + METIS) | 4.10 | `setup_level2_deps.sh` -> `.deps/install/mfem` (with `patches/mfem-v4.10-blackwell-cicc-O1.patch`, see [patches/README.md](../patches/README.md)) |

`setup_level2_deps.sh` clones each framework at the pinned tag into
`.deps/src/`, applies the build-system patches from `patches/`, builds it for
the detected GPU architecture, and installs it into `.deps/install/<name>`
(idempotent; a marker records the built tag). `hpcperf_env.sh` puts
`.deps/install/*` on `CMAKE_PREFIX_PATH`. `check_env.sh` lists which framework
libraries are present.

MPI defaults set by `hpcperf_env.sh` (all Level 2 runs are single-node):
`OMPI_MCA_pml=ob1`, `OMPI_MCA_btl=self,sm,smcuda` (skips the InfiniBand /
libfabric probing that cost ~5 s per `MPI_Init` on the dev machine; set
`HPCPERF_MPI_SINGLE_NODE=0` for multi-node), `OMPI_MCA_opal_cuda_support=true`
(the conda Open MPI ships with CUDA buffer support switched off in its
`openmpi-mca-params.conf`; ExaMiniMD, ExaMPM and the Cabana halo exchanges pass
device buffers to MPI and segfault without it), and oversubscription inside a
Slurm allocation so that `mpirun -np N` works with fewer allocated tasks.

## MPI and multi-GPU

This is a **correctness / communication** validation, not a scaling study. The
standard decks are sized for one GPU; running them across four GPUs is
communication-bound (e.g. ExaMiniMD 160^3: ~49 s of a ~53 s run is halo
exchange), which is expected strong-scaling behaviour, not a failure. Real
performance scaling needs a larger fixed workload or a weak-scaling setup and
is deferred.

**CUDA-aware MPI capability.** The conda Open MPI is a CUDA-aware build, but
its `openmpi-mca-params.conf` ships `opal_cuda_support = 0`; `hpcperf_env.sh`
sets `OMPI_MCA_opal_cuda_support=true` so device buffers may be passed to MPI
("requested"). Confirm the capability actually works at runtime with

```bash
./check_env.sh --mpi-cuda            # builds+runs level2/tools/mpi_cuda_check
```

which reports `MPIX_Query_cuda_support()` and runs a numerically-checked
device-buffer ring + Allreduce over 2 (or N) ranks. Verified 2026-09-03 on a
4x B200 allocation: `MPIX_Query_cuda_support()=1` and the device-buffer
exchange passes at 2 and 4 ranks.

**Transport, and a node-specific caveat.** By default `hpcperf_env.sh` forces
no PML/BTL -- Open MPI / the site stack chooses. On this node the default
choice is UCX, and **host-side MPI over UCX works but CUDA device-buffer MPI
over UCX hangs**. The opt-in single-node profile `HPCPERF_MPI_SINGLE_NODE=1`
(pml `ob1`, btl `self,sm,smcuda`) makes device-buffer MPI work and also skips
~5 s of transport probing per `MPI_Init`. **So multi-rank CUDA runs on this
node need `HPCPERF_MPI_SINGLE_NODE=1`** (the `mpi_cuda_check` tool sets the
shared-memory transport itself for this reason). The profile is cleanly
reversible: re-sourcing with it unset clears exactly what it set.

**GPU-aware MPI (hypre / MFEM).** The bundled hypre and MFEM are built with
`HYPRE_ENABLE_GPU_AWARE_MPI=OFF` -- the validated default. A GPU-aware **ON**
variant was also built (`.deps/install/{hypre,mfem}-gpuaware`, not disturbing
the default tree) and checked at 4 rank x 4 B200 under the SM transport:
AMG2023 (19 iterations, residual `7.582795e-13`), Laghos (`-gam`, energy
`3.3909635545e+03`) and Remhos (`-gam`, mass `0.1163086683`) all reproduce the
OFF results to the last printed digit; the FOM difference is within run-to-run
noise on a loaded node. It is **not** made the Level 2 default, deliberately:
(1) with GPU-aware ON, hypre passes device buffers to MPI internally, which
would hang on this node's default UCX transport unless the SM profile is used,
whereas OFF is safe under either transport; (2) single-node `smcuda` already
moves data by CUDA IPC, so the GPU-aware (RDMA) benefit is a multi-node / at-
scale property not measurable here; (3) the validated OFF dependency tree
should not be disturbed. Revisit ON with a controlled A/B on an idle or
multi-node allocation.

**Rank-to-GPU binding differs by framework:**

- *Automatic* -- no wrapper needed: hypre (AMG2023) via `hypre_bind_device`;
  Kokkos/Cabana apps (ExaMiniMD, ExaMPM, HACCabanaPM) by local rank; Branson
  via `set_device_ID(rank, n_ranks)`.
- *Needs an explicit wrapper* -- every rank otherwise lands on GPU 0: the MFEM
  apps (Laghos, Remhos) and the flag-selected backends (CloverLeaf, TeaLeaf).
  Use the scheduler-safe wrapper `level2/tools/mpi_gpu_bind.sh`:

  ```bash
  mpirun --oversubscribe -np 4 level2/tools/mpi_gpu_bind.sh <exe> <args>
  ```

  It picks one GPU per rank from the local rank (`OMPI_COMM_WORLD_LOCAL_RANK`,
  falling back to `SLURM_LOCALID`) *within* the scheduler's own
  `CUDA_VISIBLE_DEVICES` list -- so a Slurm allocation of, say,
  `CUDA_VISIBLE_DEVICES=2,3,6,7` is honoured, not overwritten with `0,1,2,3`.
  Only if the scheduler set no `CUDA_VISIBLE_DEVICES` does it fall back to
  physical index = local rank.

**4 rank x 4 B200 correctness verified (2026-09-03):** bare CUDA-aware MPI
(`mpi_cuda_check`), Laghos and Remhos (final energies / masses match the
1-rank run to all printed digits), AMG2023 (BoomerAMG converged, 2x2x1),
ExaMiniMD, ExaMPM (FreeFall analytic PASS), HACCabanaPM (full validation,
2x2x1), CloverLeaf and TeaLeaf (`tea.problems` PASSED), Kripke (spatial
`--procs 2,2,1` sweep; final particle count identical to 1 rank) and Branson
(material conservation ~8e-15). See the `CUDA 4-GPU / MPI` column below.

## Status legend and columns

Instead of a single "Working" flag, each mini-app is scored on independent
axes:

- **CUDA 1-GPU** -- configure + build + run + correctness validation on one
  B200 (CUDA 13.2, GCC 13.3.0). `PASS` for all 19 integrated apps.
- **CUDA 4-GPU / MPI** -- ran correctly across 4 B200s with one GPU per rank,
  MPI communication path exercised, a conserved/reference quantity checked
  (2026-09-03). `PASS` = verified; `single-GPU` = the app has no MPI
  communication path in its frozen benchmark (runs one process), so 4-GPU is
  not applicable; `not yet` = has an MPI path but a 4-GPU run is not verified
  yet. This axis is **correctness/communication only, never a scaling claim**.
- **HIP** -- `untested` for every app: a HIP build configuration exists but
  there is no AMD GPU / ROCm on the development machine.
- **Source mod** -- `yes` if any upstream source file was changed (always for
  a documented compatibility reason, listed in that app's README "Changes from
  upstream"); `no` if only build glue / wrappers were added.
- **Env/dep workaround** -- a build-system or environment workaround the app
  depends on (patched dependency, CUDA-aware MPI flag, transport profile,
  binding wrapper); `-` if none beyond the standard toolchain.

LOC (in each app's README, not repeated here) is `cloc` code lines of the
mini-app's own source, excluding bundled third-party libraries, build files,
decks, scripts and READMEs.

## Catalog

| Mini-app | Upstream | Model | CUDA 1-GPU | CUDA 4-GPU / MPI | HIP | Source mod | Env/dep workaround |
|----------|----------|-------|:---------:|-----------------|:---:|:----------:|--------------------|
| [amg2023](amg2023/) | LLNL/AMG2023 | C + hypre | PASS | PASS (2x2x1) | untested | no | hypre built for CUDA (dep) |
| [branson](branson/) | lanl/branson | CUDA/HIP C++ | PASS | PASS (4 ranks) | untested | yes | CUDA_ARCHITECTURES ordering fix |
| [cabanapic](cabanapic/) | ECP-copa/CabanaPIC | Kokkos + Cabana | PASS | single-GPU (no MPI comm) | untested | yes | REAL_TYPE=double; NFS run dir |
| [cloverleaf](cloverleaf/) | UoB-HPC/CloverLeaf | C++ + cuda/hip | PASS | PASS (4 ranks) | untested | no | binding wrapper (flag-selected device) |
| [exacmech](exacmech/) | LLNL/ExaCMech | RAJA + CHAI | PASS | single-GPU (no MPI) | untested | yes | RAJA/Umpire/CHAI 2026.07 (dep); C++20 |
| [examinimd](examinimd/) | ECP-copa/ExaMiniMD | Kokkos + K.Kernels | PASS | PASS (4 ranks) | untested | yes | Kokkos 5 / K.Kernels; CUDA-aware MPI |
| [exampm](exampm/) | ECP-copa/ExaMPM | Kokkos + Cabana | PASS | PASS (4 ranks) | untested | yes | Cabana 0.8 API; CUDA-aware MPI |
| [haccabanapm](haccabanapm/) | ECP-copa/HACCabana | Kokkos+Cabana+heFFTe | PASS | PASS (2x2x1) | untested | yes | Random123 nvcc guard; CUDA-aware MPI |
| [hipbone](hipbone/) | paranumal/hipBone | OCCA (run-time) | PASS | not yet (single-GPU deck) | untested | yes | OCCA CUDA-13 API patch; OMP threads |
| [kripke](kripke/) | LLNL/Kripke | RAJA | PASS | PASS (2,2,1) | untested | no | bundled RAJA/CHAI CUDA-13 guard (dep) |
| [laghos](laghos/) | CEED/Laghos | MFEM (device PA) | PASS | PASS (4 ranks) | untested | yes | MFEM_UNROLL(1) Blackwell workaround; MFEM+patch (dep) |
| [minibude](minibude/) | UoB-HPC/miniBUDE | CUDA / HIP | PASS | single-GPU (no MPI) | untested | yes | CMP0104 / arch flag fix |
| [miniweather](miniweather/) | mrnorman/miniWeather | YAKL (C++) | PASS | not yet (MPI halo) | untested | no | gfortran detection; PnetCDF |
| [p3_heat3d](p3_heat3d/) | yasahi-hpc/P3-miniapps | Thrust + mdspan | PASS | single-GPU (no MPI) | untested | yes | mdspan bundled |
| [p3_vlp4d](p3_vlp4d/) | yasahi-hpc/P3-miniapps | Thrust+mdspan+cuFFT | PASS | single-GPU (no MPI) | untested | yes | cuFFT; mdspan bundled |
| [remhos](remhos/) | CEED/Remhos | MFEM (device PA) | PASS | PASS (4 ranks) | untested | no | makefile route; MFEM+patch (dep); binding wrapper |
| [shaw](shaw/) | Pressio/SHAW | Kokkos + K.Kernels | PASS | single-GPU (no MPI) | untested | yes | K.Kernels signed ordinal; ROM host path |
| [tealeaf](tealeaf/) | UoB-HPC/TeaLeaf | C++ + cuda/hip | PASS | PASS (4 ranks) | untested | yes | no upstream LICENSE; binding wrapper |
| [xsbench](xsbench/) | ANL-CESAR/XSBench | CUDA / HIP | PASS | single-GPU (no MPI) | untested | yes | -std=c++17 (CCCL) |
| miniem | Trilinos (Panzer) | Kokkos / Tpetra | -- | -- | -- | -- | pending: full Trilinos build |

Brief descriptions (motif) per app: **amg2023** boomeramg algebraic-multigrid solve of a 3d 27-point laplace system; device kernels all from hypre; **branson** implicit monte carlo thermal radiative transfer (photon transport); **cabanapic** relativistic em particle-in-cell (weibel / two-stream); **cloverleaf** compressible euler hydrodynamics, structured staggered grid; **exacmech** crystal-plasticity constitutive update at 1e6 material points; **examinimd** lennard-jones md with neighbour lists and mpi halos; **exampm** material point method (dam break / free fall); **haccabanapm** hacc cosmological particle-mesh n-body (cic, fft poisson, kick/drift); **hipbone** nekbone-style high-order spectral-element poisson cg; **kripke** deterministic sn neutron-transport sweeps; **laghos** high-order lagrangian shock hydrodynamics (sedov, triple point); **minibude** molecular-docking energy evaluation (bude); **miniweather** 2d compressible atmospheric dynamics, finite volume; **p3_heat3d** 3d heat-equation stencil; **p3_vlp4d** 4d vlasov-poisson semi-lagrangian kinetic solver; **remhos** high-order dg remap / advection with flux-corrected transport; **shaw** elastic shear-wave propagation (seismic, sparse jacobians); **tealeaf** implicit linear heat conduction, sparse iterative solvers; **xsbench** monte carlo neutron cross-section lookup. MiniEM is a Trilinos/Panzer electromagnetics (Maxwell) FE mini-app.

Notes:
- Kokkos, RAJA, OCCA and YAKL codes have one source tree for both variants, so
  the CUDA and HIP counts are the same; only CloverLeaf, TeaLeaf, miniBUDE,
  XSBench and the P3 mini-apps have separate CUDA and HIP backend directories.
- AMG2023 has no device code of its own (`amg.c` is plain C); the GPU work is
  entirely inside hypre.
- TeaLeaf ships no license file upstream (its README says it replicates the
  UK-MAC TeaLeaf code); nothing was copied as `LICENSE` there.
- MiniEM is a Trilinos (Panzer) mini-app and cannot be built stand-alone: it
  needs a Trilinos build with Tpetra, Panzer, MueLu, Intrepid2 and Kokkos
  enabled (a multi-hour source build that adds Trilinos to
  `setup_level2_deps.sh`). It is held until that dependency is approved.
- Blackwell (sm_100) + CUDA 13.2 compile pathologies found while integrating
  the MFEM codes: the nvcc device front end (cicc) unrolls thread-strided
  loops into enormous PTX for some high-order kernels -- MFEM v4.10's
  element-assembly / batched-LOR files (hours per file; handled by the
  build-system patch in [patches/](../patches/), which lowers only the device
  optimisation level of three files that Laghos and Remhos never execute) and
  Laghos' `QKernel` (107 MB of PTX, `ptxas` killed at 32 GB; handled by a
  13-line `MFEM_UNROLL(1)` source hint, see [laghos/README.md](laghos/README.md)).
- The two MFEM/Laghos compiler workarounds have different status. The MFEM
  `-Xcicc=-O1` patch is a **dependency build workaround** that lowers the
  device optimisation level of three files Laghos/Remhos never execute, so it
  cannot affect their kernels. The Laghos `MFEM_UNROLL(1)` hint touches a
  kernel that *is* executed; 4-GPU correctness passes with it, but that does
  **not** prove zero performance impact -- it is documented as a
  `Blackwell/CUDA 13.2 source-level compiler workaround, performance impact
  pending A/B validation`, to be compared against the vanilla build on a
  Hopper/sm_90 (or fixed-CUDA) machine later.

Summary: 19/20 CUDA single-GPU Working (validated on one B200); 1 pending
decision (MiniEM, needs Trilinos). MPI communication paths: 10 apps carry one
(amg2023, branson, cloverleaf, examinimd, exampm, haccabanapm, kripke, laghos,
remhos, tealeaf) and all 10 pass a 4 rank x 4 B200 correctness check;
miniweather and hipbone have an MPI path but their frozen decks are
single-GPU-sized and 4-GPU is not yet verified; the remaining apps have no MPI
communication path. Every HIP variant is untested (no AMD GPU here).
