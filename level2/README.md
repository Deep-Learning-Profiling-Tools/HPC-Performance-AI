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

## Status legend

- `Working` -- configure + build + run + correctness validation all verified on
  the B200 reference machine (CUDA 13.2, GCC 13.3.0).
- `HIP untested` -- the HIP variant (upstream HIP backend, or the same
  Kokkos/RAJA/OCCA/YAKL source built for HIP) has a build configuration, but
  there is no AMD GPU / ROCm on the development machine, so it is not
  compile- or run-verified. This applies to every entry.
- `Pending decision` -- not integrated yet; see the notes below the table.

LOC is `cloc` code lines of the mini-app's own source for the variant (driver
+ backend, or the single portable source tree), excluding bundled third-party
libraries, build files, decks, scripts and READMEs; each README lists the
exact paths counted and the size of any bundled dependency.

## Catalog

| Mini-app | Upstream | Programming model | Brief summary | CUDA LOC | HIP LOC | Status |
|----------|----------|-------------------|---------------|----------|---------|--------|
| [amg2023](amg2023/) | LLNL/AMG2023 | C + hypre | BoomerAMG algebraic-multigrid solve of a 3D 27-point Laplace system; all device kernels come from hypre | 2712 | 2712 | Working (CUDA), HIP untested |
| [branson](branson/) | lanl/branson | CUDA / HIP C++ (single source) | Implicit Monte Carlo thermal radiative transfer (photon transport) | 7384 | 7384 | Working (CUDA), HIP untested |
| [cabanapic](cabanapic/) | ECP-copa/CabanaPIC | Kokkos + Cabana | Relativistic electromagnetic particle-in-cell (Weibel instability / two-stream) | 2038 | 2038 | Working (CUDA), HIP untested |
| [cloverleaf](cloverleaf/) | UoB-HPC/CloverLeaf | C++ driver + cuda/hip backends | Compressible Euler hydrodynamics on a structured staggered grid (Lagrangian-Eulerian, explicit) | 5220 | 5223 | Working (CUDA), HIP untested |
| [exacmech](exacmech/) | LLNL/ExaCMech | RAJA + CHAI | Crystal-plasticity constitutive-model update at 10^6 material points | 7266 | 7266 | Working (CUDA), HIP untested |
| [examinimd](examinimd/) | ECP-copa/ExaMiniMD | Kokkos (+ Kokkos Kernels) | Lennard-Jones molecular dynamics with neighbour lists and MPI halos | 6164 | 6164 | Working (CUDA), HIP untested |
| [exampm](exampm/) | ECP-copa/ExaMPM | Kokkos + Cabana | Material point method (dam break, particle-grid transfers) | 1501 | 1501 | Working (CUDA), HIP untested |
| [haccabanapm](haccabanapm/) | ECP-copa/HACCabana | Kokkos + Cabana (+ heFFTe) | HACC cosmological particle-mesh N-body proxy (CIC deposit, FFT Poisson solve, kick/drift) | 4321 | 4321 | Working (CUDA), HIP untested |
| [hipbone](hipbone/) | paranumal/hipBone | OCCA (CUDA/HIP selected at run time) | Nekbone-style high-order spectral-element Poisson CG solve | 9515 | 9515 | Working (CUDA), HIP untested |
| [kripke](kripke/) | LLNL/Kripke | RAJA | Deterministic Sn neutron-transport sweeps | 6497 | 6497 | Working (CUDA), HIP untested |
| [laghos](laghos/) | CEED/Laghos | MFEM (partial assembly, device) | High-order Lagrangian shock hydrodynamics (Sedov blast, triple point) | 3702 | 3702 | Working (CUDA), HIP untested |
| [minibude](minibude/) | UoB-HPC/miniBUDE | CUDA / HIP | Molecular-docking energy evaluation (BUDE) | 885 | 851 | Working (CUDA), HIP untested |
| [miniweather](miniweather/) | mrnorman/miniWeather | YAKL (C++) | 2D compressible atmospheric dynamics, finite volume | 659 | 659 | Working (CUDA), HIP untested |
| [p3_heat3d](p3_heat3d/) | yasahi-hpc/P3-miniapps | Thrust + mdspan | 3D heat-equation stencil | 1160 | 1157 | Working (CUDA), HIP untested |
| [p3_vlp4d](p3_vlp4d/) | yasahi-hpc/P3-miniapps | Thrust + mdspan + cuFFT/hipFFT | 4D Vlasov-Poisson semi-Lagrangian kinetic solver | 1965 | 2002 | Working (CUDA), HIP untested |
| [remhos](remhos/) | CEED/Remhos | MFEM (partial assembly, device) | High-order DG remap / advection with flux-corrected transport (monotone, 3D remap) | 7128 | 7128 | Working (CUDA), HIP untested |
| [shaw](shaw/) | Pressio/SHAW | Kokkos (+ Kokkos Kernels) | Elastic shear-wave propagation (seismic, staggered grid, sparse Jacobians) | 4120 | 4120 | Working (CUDA), HIP untested |
| [tealeaf](tealeaf/) | UoB-HPC/TeaLeaf | C++ driver + cuda/hip backends | Implicit linear heat conduction, sparse iterative solvers (CG / Chebyshev / PPCG) | 2863 | 2870 | Working (CUDA), HIP untested |
| [xsbench](xsbench/) | ANL-CESAR/XSBench | CUDA / HIP | Monte Carlo neutron cross-section lookup | 1514 | 1061 | Working (CUDA), HIP untested |
| miniem | Trilinos (Panzer MiniEM) | Kokkos / Tpetra (Trilinos) | Electromagnetics (Maxwell) finite-element mini-app | -- | -- | Pending decision (needs a Trilinos build) |

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
  13-line `MFEM_UNROLL(1)` hint that is numerically a no-op, see
  [laghos/README.md](laghos/README.md)).

Summary: 19/20 Working (CUDA validated); 1 pending decision (MiniEM). Every
HIP variant is untested (no AMD GPU on the development machine).
