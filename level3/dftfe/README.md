# DFT-FE -- Level 3 second batch

Real-space finite-element Kohn-Sham DFT (University of Michigan). GPU build
(`WITH_GPU=ON GPU_LANG=cuda GPU_VENDOR=nvidia`): Chebyshev-filtered subspace
iteration, dense linear algebra (cuBLAS), Poisson/Helmholtz solves and the
ELPA eigensolver with NVIDIA GPU kernels.

## Provenance / versions

| Item | Value |
|---|---|
| DFT-FE | release 1.2.0, `7147faa51f7c9f3075fffaa5e48ba989bcd329c1` (branch `release1.2`), LGPL-2.1-or-later |
| Recipe | upstream `install_DFTFE` (branch `frontierDevelop`: `dftfe2.sh` + `setupUser.sh`) transcribed to this node's toolchain in `build.sh`; every dependency tarball is pinned by sha256 in `fetch.sh` |
| Dependencies (private, same profile) | OpenBLAS 0.3.30 (`TARGET=SAPPHIRERAPIDS`, single-threaded) -> ScaLAPACK 2.2.2 -> libxc 7.0.0 -> spglib `02159eef` -> ALGLIB 4.06.0 -> p4est 2.8.7 (dftfe's `p4est-setup.sh`, FAST+DEBUG) -> Kokkos 4.6.00 (Serial, deal.II only) -> deal.II **9.6.2** (see decision below) (MPI, p4est, 64-bit indices, LAPACK=OpenBLAS, bundled Boost, no TBB/taskflow) -> ELPA 2026.02.001 (`--enable-nvidia-gpu-kernels --with-NVIDIA-GPU-compute-capability=sm_100`) -> DFT-FE real (`CMAKE_CUDA_ARCHITECTURES=100 WITH_DCCL=OFF WITH_GPU_AWARE_MPI=OFF USE_64BIT_INT=ON`); libxml2 from the system |
| Compilers | system GCC 14.2.1 (C/C++/Fortran) through the conda Open MPI 5.0.10 wrappers (`OMPI_CC/CXX/FC`), nvcc 13.2.78 with g++ 14.2.1 host |
| GPU target | sm_100 for DFT-FE and for the ELPA kernels (ELPA's generic `--with-NVIDIA-GPU-compute-capability=sm_XX` path; sm_100 never used upstream) |
| NCCL | not used (`WITH_DCCL=OFF`; DFT-FE's default all-reduce path) |
| Build strategy | NATIVE (no Spack: the `dftfe` recipe is 0.6/2019 without GPU variants; no container) |
| Profile | `cuda132-gcc142-ompi5010` (`HPCPERF_DFTFE_ELPA_GPU=OFF` would build `-elpacpu`, a named CPU-ELPA diagnostic profile, not used for the results) |

### Node adaptations (class C, recorded in build.sh)

- dftfe's `p4est-setup.sh` is written for Cray systems: it hardcodes
  `CC=cc CXX=CC FC=ftn F77=ftn` (no MPI on this node's `cc`), relies on the Cray
  wrappers' implicit `-lm`, and checks `<build>/src/p4est_config.h` while p4est
  2.8.7 writes `<build>/config/p4est_config.h`. build.sh passes
  `CC=mpicc CXX=mpicxx FC=mpifort F77=mpifort LIBS=-lm` through the script's
  trailing configure arguments and seds the header path in its private copy.
- The conda environment's `CFLAGS/LDFLAGS/AR/...` are cleared for the
  system-GCC builds (`l3_clean_conda_build_env`); OpenBLAS 0.3.30 misdetects the
  Xeon Platinum 8570 and is built with `TARGET=SAPPHIRERAPIDS`.
- ELPA's configure expects the host SIMD flags in `CFLAGS` (it probes AVX-512
  intrinsics with the given flags and aborts otherwise; EL10's GCC 14 defaults to
  x86-64-v3 = AVX2): `-march=native` for C/C++/Fortran -- a node-specific
  binary like OpenBLAS. Its later CUDA link checks (`cublasDgemm`) keep
  `LIBS=-lscalapack` but replace `LDFLAGS` by the CUDA path, so the
  ScaLAPACK/OpenBLAS `-L`/rpath entries are given in `LDFLAGS` as well.

### deal.II version decision (no DFT-FE source change)

Release 1.2.0 pins deal.II **9.5.2** in its own `setupUser*.sh`; the current
`install_DFTFE` recipe (frontierDevelop/master) builds deal.II **9.7.1** -- but
together with the *develop* branch (`publicGithubDevelop`), not with the release.
Attempt 1 here (1.2.0 + 9.7.1) failed to compile: 9.7 removed three APIs 1.2.0
still uses (`Utilities::MPI::create_group`, `parallel::distributed::Triangulation::load(name, autopartition)`,
`DataOutBase::VtkFlags::ZlibCompressionLevel`; upstream develop already carries the
replacements). Rather than back-porting a growing set of develop changes into the
release, the profile uses **deal.II 9.6.2**, the newest deal.II that still provides
those (deprecated) APIs, so 1.2.0 builds unpatched; recorded as attempt 2. The
9.7.1 build and its install were removed from the profile
(`.deps/level3/dftfe/<profile>/ATTEMPT-1-dealii-9.7.1.txt`, logs `attempt1-*`).

## ELPA GPU kernels verified independently (`elpa_probe.sh`)

ELPA's own test programs (`validate_real_double_eigenvectors_{1stage,2stage_default_kernel}_gpu_analytic`,
which call `e%set("nvidia-gpu", 1)`) on 1, 2 and 4 GPUs (one rank per GPU through the
common launcher) and their CPU counterparts on 1 rank: analytic test matrix, ELPA's own
limits residual `max ||A z - lambda z|| <= 9e-10` and orthogonality `max |Z^T Z - I| <= 9e-10`
re-parsed and re-applied; record `<install>/elpa/ELPA_GPU_PROBE.txt`, required PASS by
`validate.sh`.

## Cases and validation

`run.sh`: `al_md` (upstream GPU regression deck `testsGPU/pseudopotential/real/Input_MD_0.prm`:
32-atom fcc Al, order-3 FE, 85 states, BOMD 1400 K, 4 steps, `USE GPU = true`,
`REPRODUCIBLE OUTPUT = true`) -- smoke/strong verbatim, weak = derived Al supercell
series (32 atoms per GPU; SYNTHETIC, same construction as upstream's
dftfe-benchmarks Mo series); `llzo` (`parameterFile_LLZO.prm`, 192 atoms, 720
states, ELPA) as the heavier fixed-size strong case. One MPI rank per GPU, 1
thread per rank (upstream's GPU job scripts), decks used verbatim except the
documented weak derivation.

`validate.sh`: [0] ELPA GPU probe PASS; [1] `al_md` on N GPUs vs upstream's own GPU
reference `accuracyBenchmarks/output_MD_0` through `dftfe_check.py` (pre-fixed:
ground-state energy 1e-5 Ha, per-step MD energies 2e-5 Ha, temperatures 0.1 K,
forces 2e-5 Ha/Bohr; SCF converged, MD completed, finite), launcher audit "N
verified, 0 mismatch"; N > 1 additionally vs the 1-GPU run under the same
tolerances. Results: `SECOND_BATCH_STATUS.md`.
