# CP2K -- Level 3 second batch

Quickstep GPW density-functional theory (SCF/OT) and Born-Oppenheimer MD,
psmp binary (MPI + OpenMP + CUDA), GPU-accelerated DBCSR (sparse matrix
multiply, JIT libsmm_acc kernels), DBM, GRID (collocate/integrate) and PW (FFT).

## Provenance / versions

| Item | Value |
|---|---|
| CP2K | tag `v2026.2`, `67b5da876dd6a76b8b021d5a04d1c81ba79a4c50` (2026-07-15), GPL-2.0-or-later; CMake-only build |
| Dependencies | upstream's own bootstrap `tools/toolchain/install_cp2k_toolchain.sh` (from the same tag): OpenBLAS 0.3.33 (SAPPHIRERAPIDS detected), ScaLAPACK 2.2.3, FFTW 3.3.11, libint 2.13.1 (lmax 5), libxc 7.0.0, LIBXSMM 2.0.0 / LIBXS 1.0.0, spglib 2.7.0, **DBCSR 2.10.0** (sha256 `3d897220...`), Eigen 5.0.1; ELPA/COSMA/SIRIUS/tblite/libvori/HDF5/PLUMED/... **off** (minimal GPW stack) |
| Compilers | **system GCC 14.2.1** for C/C++/Fortran (the conda GCC 13.3 has no gfortran; one GCC for all three languages avoids the mixed-LTO/PIE issues met by SPECFEM/nekRS); nvcc 13.2.78 with g++ 14.2.1 host (probe OK on B200) |
| MPI | conda Open MPI 5.0.10 (site-validated transport), wrappers redirected via `OMPI_CC/CXX/FC` (probe: C and `mpi_f08` programs build and run with 2 ranks) |
| GPU target | sm_100 (`-DCMAKE_CUDA_ARCHITECTURES=100`, accepted by v2026.2's CMake); toolchain `--gpu-ver=B200` |
| Build strategy | NATIVE + upstream toolchain (Spack rejected: recipes cap `cuda_arch` at 90 and the local Spack is 2025-05) |
| Profile | `cuda132-gcc142-ompi5010` |

### B200 support: upstream backport, not an unsourced sed

v2026.2 knows H100/GB10 but not B200; upstream master added B200 in commit
`378b2fab2f20224e5468d3279fa7f051403977c1` ("Add NVIDIA B200 GPU support
(#5788)", 2026-08-20). `patches/0001-toolchain-b200-backport-cp2k-378b2fab.patch`
back-ports that commit's two toolchain hunks (`--gpu-ver=B200 -> ARCH_NUM 100`;
`install_dbcsr.sh`: add `B200 -> GPU_ARCH_NUMBER 100` to DBCSR 2.10.0's
CMakeLists and copy `parameters_H100.json -> parameters_B200.json`) to the
v2026.2 toolchain (class B). The CMakeLists.txt hunk of that commit is not needed
because CP2K itself is configured with `CMAKE_CUDA_ARCHITECTURES=100`. So:
**DBCSR device code is native sm_100; the libsmm_acc kernel *parameters* are the
H100 set reused, not B200-tuned** (upstream's own approach for GB10 and B200).

### Node adaptations (class C, recorded)

- DBCSR 2.10.0's `cmake/GetGitRevisionDescription.cmake` aborts when its source
  lives inside a git *worktree* (it treats the absolute gitdir in the `.git` file
  as relative): the toolchain's private copy therefore lives on local scratch
  (`/tmp/hpcperf-l3-b2-scratch/cp2k-toolchain/<profile>`, symlinked from
  `.deps/level3/cp2k/<profile>/src/toolchain`); the install prefix, `setup`,
  `toolchain.conf` and all logs stay under the profile tree.
- DBCSR's own ctest suite launches `mpiexec -n 4` itself; under this 1-task-slot
  Slurm allocation PRRTE's slot accounting is relaxed for those launches
  (`PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe`), the same bookkeeping
  relaxation the common launcher applies (4 ranks on 4 GPUs, no GPU sharing).

## DBCSR verified before CP2K (build.sh stage B)

DBCSR 2.10.0 test build (`USE_ACCEL=cuda WITH_GPU=B200`, MPI + OpenMP,
`BUILD_TESTING=ON`, libdbcsr.a device code `sm_100`), its official ctest suite
with 4 MPI ranks x 4 OpenMP threads on the 4 B200s: **19/19 passed** (10
dbcsr_perf inputs, dbcsr_unittest1-4, tensor/tas unit tests, csr conversions,
dbcsr_test, dbcsr_tensor_test). Log: `.deps/level3/cp2k/<profile>/logs/dbcsr-ctest.log`.

## Cases and validation

`run.sh`: `h2o` (benchmarks/QS/H2O-<S>.inp, upstream's GPW-DFT MD benchmark, 10 NVE
steps; smoke S=64, strong S=128 fixed over rank counts, weak = **size sweep**
S=32*N -- not a strict weak-scaling series, see run.sh header) and `regtest`
(a single upstream regression input). One MPI rank per GPU, OpenMP threads per
rank (`HPCPERF_CPUS_PER_RANK`, default 8), input used verbatim, `CP2K_DATA_DIR`
= the checkout's data/.

`validate.sh`: [1] an adapted subset of upstream's regression tests
(`tests/QS/regtest-gpw-1`: Ar, H2O-geoopt, pyridine; `regtest-dm-ls-scf-1`:
H2-big-1, H2-big-5) run on N GPUs and compared with the upstream reference
values and tolerances from `TEST_FILES.toml` through the same matcher
definitions (`tests/matchers.py`: `E_total` = last "Total energy:" col 3, `M011`
= last "ENERGY| Total FORCE_EVAL" col 9); [2] H2O-64 MD: 10 steps reached, every
SCF converged, finite energies, DBCSR reports >= 1 accelerator device and the
GRID/DBM/PW GPU backends are active in CP2K's banner; for N > 1 the MD potential
energy at steps 1 and 10 must agree with the 1-GPU run within **1e-8 Ha**
(pre-fixed; upstream's `check-release-comparison.py` demands 1e-10 across CPU
MPIxOMP layouts -- printed as well).

Results are recorded in `SECOND_BATCH_STATUS.md`.
