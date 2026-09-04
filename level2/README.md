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

MPI environment set by `hpcperf_env.sh`: `OMPI_MCA_opal_cuda_support=true`
only (the conda Open MPI ships with CUDA buffer support switched off in its
`openmpi-mca-params.conf`; ExaMiniMD, ExaMPM and the Cabana halo exchanges
pass device buffers to MPI and segfault without it). **No PML/BTL is forced
and no oversubscription policy is set by default**; transport comes from the
site profile through the common launcher (next section), and the single-node
shared-memory profile is opt-in (`HPCPERF_MPI_SINGLE_NODE=1`, cleanly
reversible).

## MPI and multi-GPU

The 4-GPU results below are **single-node multi-GPU correctness /
communication validation**, not scaling results (and NOT multi-node
scale-out). The historical decks are sized for one GPU; strong-scaling them
across four GPUs is communication-bound (e.g. ExaMiniMD 160^3: ~49 s of a
~53 s run is halo exchange), which is expected behaviour, not a failure.
Real scaling runs use the `HPCPERF_SCALE_MODE=strong|weak` decks now provided
by the refactored benchmarks (amg2023, laghos, examinimd so far).

**Common launcher.** Multi-rank runs go through
`level2/tools/hpcperf_mpi_launch.sh` (see [tools/README.md](tools/README.md)):

```bash
HPCPERF_GPUS=4 level2/amg2023/run.sh CUDA            # 4 ranks, one per GPU
HPCPERF_GPUS=all HPCPERF_SCALE_MODE=weak level2/examinimd/run.sh CUDA
HPCPERF_DRY_RUN=1 HPCPERF_NODES=5 HPCPERF_GPUS_PER_NODE=8 HPCPERF_GPUS=40 \
    level2/laghos/run.sh CUDA                        # plan a 40-GPU run
```

The execution model is **one MPI rank per GPU**: `--gpus N` starts exactly N
ranks even if the allocation has more GPUs, `--gpus all` uses every allocated
GPU, ranks > allocated GPUs fails fast, and GPU sharing between ranks is
forbidden unless `HPCPERF_ALLOW_OVERSUBSCRIBE=1` (debug only, loud warning).
The launcher keeps three resource views apart: the **actual** allocation
(Slurm; heterogeneous node groups are refused, nothing is assumed uniform), the
**requested** subset (`HPCPERF_NODES` / `HPCPERF_GPUS_PER_NODE` may only shrink
it, and a node subset is enforced in the placement via `mpirun --host` /
`srun --nodelist`), and a **hypothetical** shape accepted only in dry-run and
labelled as such. `HPCPERF_CPUS_PER_RANK=C` binds C cores per rank
(mpirun `--map-by ppr:R:node:PE=C --bind-to core`, srun `--cpus-per-task`)
after a CPU-capacity check; Slurm task-slot relaxation happens only after both
the GPU and the CPU checks pass. The GPU binding audit reports, per rank, the
**expected** GPU (arranged from the local rank, UUID resolved through
nvidia-smi's own index table under `CUDA_DEVICE_ORDER=PCI_BUS_ID`) and the
**observed** GPU (sampled from `nvidia-smi --query-compute-apps` while the
job runs): `verified`, `MISMATCH`, or `unverified` -- never "verified" without
an observation. GPU mode with no usable GPU fails (exit 13).
`HPCPERF_NP` survives as a compatibility alias and must agree with
`HPCPERF_GPUS` when both are set. Process grids for topology-parameterised
apps come from `level2/tools/hpcperf_topology.py`, which fails with nearby
feasible rank counts rather than silently changing N. The full scale-out
audit of all 20 candidates lives in [SCALEOUT_AUDIT.md](SCALEOUT_AUDIT.md).

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
node need the shared-memory transport** (the launcher's gmu-hopper site
profile applies it for single-node launches; `mpi_cuda_check` sets it
itself). The profile is cleanly reversible and is **single-node only** --
the launcher never applies it to a multi-node launch. **Multi-node MPI is
BLOCKED/UNVERIFIED on this site**: no cross-node GPU-aware transport has been
validated, and real multi-node runs need a site-provided (or purpose-built)
GPU-aware UCX/Open MPI plus rebuilds of hypre/MFEM/Cabana/heFFTe on top of
it. The conda Open MPI remains the single-node bring-up MPI.

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

## Status

Column meanings -- Build/Smoke-1GPU: configure+build / single-GPU run+validate
on the B200. Distributed model: what a multi-GPU run IS (native-mpi = one
coupled decomposed workload; replica = independent copies, never counted as
multi-GPU; "sharded-EP" = one global embarrassingly-parallel task divisible
over ranks with a final reduction -- distinct from full-input replicas, and
counted only once implemented). Selectable N: `HPCPERF_GPUS` through the
common launcher ("no scale modes" = rank selection only; `HPCPERF_SCALE_MODE`
is rejected with an error there, never silently ignored). 1-node
multi-GPU: 4 rank x 4 B200 correctness (2026-09-03/04) -- correctness only,
not scaling. Multi-node: BLOCKED site-wide (transport, see above). Strong /
Weak deck: a size policy exists that scales the problem with N. Rank
constraints: legal rank counts.

| App | Build | Smoke 1-GPU | Distributed model | Selectable N | 1-node multi-GPU | Multi-node | Strong deck | Weak deck | CUDA | HIP | Rank constraints | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| amg2023 | PASS | PASS | native-mpi | yes (launcher) | validated (2x2x1) | BLOCKED (site) | yes (512^3 global) | yes (256^3/rank) | PASS | untested | product=N; global<2^31 (<=127 @256^3/rank) | hypre self-binds |
| laghos | PASS | PASS | native-mpi | yes (launcher) | validated (4) | BLOCKED (site) | yes (-rs 4 deck) | yes (-epm/rank) | PASS | untested | N<=elements (strong); any N (weak) | MFEM_UNROLL workaround (perf A/B pending) |
| examinimd | PASS | PASS | native-mpi | yes (launcher) | validated (4) | BLOCKED (site) | yes (160^3 box) | yes (100^3 cells/rank) | PASS | untested | any N; <=20M atoms/rank (int32) | CUDA-aware MPI required |
| remhos | PASS | PASS | native-mpi | yes (launcher; no scale modes) | validated (4) | BLOCKED (site) | manual | manual (-epm exists) | PASS | untested | N<=elements; GPU combo rank-indep | launcher integration next phase |
| exampm | PASS | PASS | native-mpi | yes (launcher; no scale modes) | validated (4) | BLOCKED (site) | manual (refine cell) | no clean knob | PASS | untested | 1xNx1 Y-slabs; N<~35 @cell 0.01 (halo>=3 unchecked) | fix 1D slab before 80 ranks |
| haccabanapm | PASS | PASS | native-mpi | yes (launcher; no scale modes) | validated (2x2x1) | BLOCKED (site) | manual | manual (NG=NP=RL scale) | PASS | untested | any N; pm_ic/pm_run same N | heFFTe pencils; big IC files at scale |
| cloverleaf | PASS | PASS | native-mpi | yes (launcher; no scale modes) | validated (4) | BLOCKED (site) | yes (same deck) | manual (bm family) | PASS | untested | any N (auto chunks) | needs binding wrapper |
| tealeaf | PASS | PASS | native-mpi | yes (launcher; no scale modes) | validated (4) | BLOCKED (site) | yes (same deck) | manual (+tea.problems rows) | PASS | untested | any N (auto chunks) | needs binding wrapper |
| kripke | PASS | PASS | native-mpi | yes (launcher; no scale modes) | validated (2,2,1) | BLOCKED (site) | manual (--zones div) | manual (32^3/rank) | PASS | untested | product=N; zones divisible; groups%gset | needs binding wrapper |
| branson | PASS | PASS | native-mpi (replicated split) | yes (launcher; no scale modes) | validated (4) | BLOCKED (site) | yes (photons/N auto) | manual (--photons*N) | PASS | untested | any N | PARTICLE_PASS mode = later extension |
| hipbone | PASS | PASS | native-mpi | yes (launcher; no scale modes) | not yet | BLOCKED (site) | manual (divide global) | native (-nx per rank) | PASS | untested | cube N unless -px/-py/-pz given | add -px/py/pz decks |
| miniweather | PASS | PASS | native-mpi (1D x-split; compile-time size = limitations, not a disqualifier) | yes (launcher; no scale modes) | not yet | BLOCKED (site) | rebuild per size | rebuild per size | PASS | untested | N<=nx_glob | multi-GPU capable; limitations: compile-time nx, x-only split |
| p3_heat3d | PASS | PASS | shardable (upstream heat3d_mpi ready) | n/a today | n/a | n/a | (sibling) | (sibling: nx/rank) | PASS | untested | sibling: product=N | adopt heat3d_mpi (extension) |
| p3_vlp4d | PASS | PASS | shardable (upstream vlp4d_mpi ready) | n/a today | n/a | n/a | (sibling: same deck) | (sibling: scale grid) | PASS | untested | sibling: >=10 pts/cut | vlp4d_mpi changes interpolation scheme |
| cabanapic | PASS | PASS | replica-only as shipped (shardable at rewrite cost) | n/a | n/a | n/a | n/a | n/a | PASS | untested | n/a | supplemental |
| shaw | PASS | PASS | shardable-not-implemented | n/a | n/a | n/a | n/a | n/a | PASS | untested | n/a | supplemental |
| exacmech | PASS | PASS | sharded-EP possible (split points + Allreduce), not implemented; replica-only as shipped | n/a | n/a | n/a | n/a | n/a | PASS | untested | n/a | supplemental |
| minibude | PASS | PASS | sharded-EP possible (split poses + reduce), not implemented; replica-only as shipped | n/a | n/a | n/a | n/a | n/a | PASS | untested | n/a | supplemental |
| xsbench | PASS | PASS | sharded-EP possible (split lookups + reduce), not implemented; upstream MPI mode = full-input replicas | n/a | n/a | n/a | n/a | n/a | PASS | untested | n/a | supplemental |
| miniem | -- | -- | native-mpi (upstream) | -- | -- | -- | -- | -- | -- | -- | any N | pending Trilinos decision |

### Per-app interface support

Every `run.sh` resolves its rank count with `level2/tools/hpcperf_launch_common.sh`
and launches through `hpcperf_mpi_launch.sh`; a request the script cannot
honour is an error (exit 2), never a silent 1-rank run.

| App | `HPCPERF_GPUS` | `HPCPERF_SCALE_MODE` | GPU binding | Rejected extra args (validated params) | App-specific guard |
|---|---|---|---|---|---|
| amg2023 | yes | smoke/strong/weak | app (hypre) | `-P`, `-n` | `HPCPERF_AMG_P` product must equal ranks; global unknowns < 2^31 |
| laghos | yes | smoke/strong/weak | wrapper | `-m -rs -rp -epm -nx -ny -nz -dev -d` | serial elements >= ranks (strong/smoke) |
| examinimd | yes | smoke/strong/weak | app (Kokkos) | `-il` | <= 20M atoms/rank |
| remhos | yes | rejected | wrapper | `-m -rs -o -dt -tf -d -epm` | -- |
| exampm | yes | rejected | app (Kokkos) | -- | Y-cells >= 3 x ranks (1xNx1 slab halo) |
| haccabanapm | yes | rejected | app (Kokkos) | -- | pm_ic and pm_run launched with the same N |
| cloverleaf | yes | rejected | wrapper | `--device` | -- |
| tealeaf | yes | rejected | wrapper | `--device`, `-d` | MPI build required for N > 1 |
| kripke | yes (also `KRIPKE_NP`) | rejected | wrapper | -- | N > 1 needs `--procs` with product N; zones divisible |
| branson | yes | rejected | wrapper | -- | -- |
| hipbone | yes (also `HIPBONE_NP`) | rejected | app (own local rank) | -- | N > 1 needs `-px -py -pz` (product N) or a cube N |
| miniweather | yes | rejected | wrapper | any argument | -- |
| cabanapic, exacmech, minibude, p3_heat3d, p3_vlp4d, shaw, xsbench | n/a (single process) | n/a | -- | -- | -- |

## Catalog (provenance and modifications)

| Mini-app | Upstream | Model | Source mod | Env/dep workaround |
|----------|----------|-------|:----------:|--------------------|
| [amg2023](amg2023/) | LLNL/AMG2023 | C + hypre | no | hypre built for CUDA (dep) |
| [branson](branson/) | lanl/branson | CUDA/HIP C++ | yes | CUDA_ARCHITECTURES ordering fix |
| [cabanapic](cabanapic/) | ECP-copa/CabanaPIC | Kokkos + Cabana | yes | REAL_TYPE=double; NFS run dir |
| [cloverleaf](cloverleaf/) | UoB-HPC/CloverLeaf | C++ + cuda/hip | no | binding wrapper (flag-selected device) |
| [exacmech](exacmech/) | LLNL/ExaCMech | RAJA + CHAI | yes | RAJA/Umpire/CHAI 2026.07 (dep); C++20 |
| [examinimd](examinimd/) | ECP-copa/ExaMiniMD | Kokkos + K.Kernels | yes | Kokkos 5 / K.Kernels; CUDA-aware MPI |
| [exampm](exampm/) | ECP-copa/ExaMPM | Kokkos + Cabana | yes | Cabana 0.8 API; CUDA-aware MPI |
| [haccabanapm](haccabanapm/) | ECP-copa/HACCabana | Kokkos+Cabana+heFFTe | yes | Random123 nvcc guard; CUDA-aware MPI |
| [hipbone](hipbone/) | paranumal/hipBone | OCCA (run-time) | yes | OCCA CUDA-13 API patch; OMP threads |
| [kripke](kripke/) | LLNL/Kripke | RAJA | no | bundled RAJA/CHAI CUDA-13 guard (dep) |
| [laghos](laghos/) | CEED/Laghos | MFEM (device PA) | yes | MFEM_UNROLL(1) Blackwell workaround; MFEM+patch (dep) |
| [minibude](minibude/) | UoB-HPC/miniBUDE | CUDA / HIP | yes | CMP0104 / arch flag fix |
| [miniweather](miniweather/) | mrnorman/miniWeather | YAKL (C++) | no | gfortran detection; PnetCDF |
| [p3_heat3d](p3_heat3d/) | yasahi-hpc/P3-miniapps | Thrust + mdspan | yes | mdspan bundled |
| [p3_vlp4d](p3_vlp4d/) | yasahi-hpc/P3-miniapps | Thrust+mdspan+cuFFT | yes | cuFFT; mdspan bundled |
| [remhos](remhos/) | CEED/Remhos | MFEM (device PA) | no | makefile route; MFEM+patch (dep); binding wrapper |
| [shaw](shaw/) | Pressio/SHAW | Kokkos + K.Kernels | yes | K.Kernels signed ordinal; ROM host path |
| [tealeaf](tealeaf/) | UoB-HPC/TeaLeaf | C++ + cuda/hip | yes | no upstream LICENSE; binding wrapper |
| [xsbench](xsbench/) | ANL-CESAR/XSBench | CUDA / HIP | yes | -std=c++17 (CCCL) |
| miniem | Trilinos (Panzer) | Kokkos / Tpetra | -- | pending: full Trilinos build |

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
decision (MiniEM, needs Trilinos). 10 MPI apps (amg2023, branson, cloverleaf,
examinimd, exampm, haccabanapm, kripke, laghos, remhos, tealeaf) are
**single-node multi-GPU correctness validated** at 4 ranks x 4 B200 -- NOT
multi-node scale-out validated; multi-node is BLOCKED site-wide (transport).
miniweather and hipbone have an MPI path but no verified multi-GPU deck yet;
the remaining apps have no MPI communication path (see SCALEOUT_AUDIT.md for
the honest classification). Every HIP variant is untested (no AMD GPU here).
