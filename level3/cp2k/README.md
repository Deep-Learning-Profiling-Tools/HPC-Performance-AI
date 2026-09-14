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

### BLAS/LAPACK actually linked: three attempts (recorded under `install/ATTEMPT-*`)

1. First build: CP2K's CMake found the toolchain OpenBLAS (`-L.../openblas-0.3.33/lib
   -lopenblas`), but the conda `LDFLAGS` leaked into the link (`-Wl,--disable-new-dtags`
   + the MPI wrapper's `-Wl,-rpath <conda lib>`), so `libopenblas.so.0` resolved at
   run time to the **conda pthreads OpenBLAS** ("OpenBLAS Warning : Detect OpenMP
   Loop" in every run). All validations of that build passed (numerics are
   BLAS-implementation independent within the tolerances), but the configuration
   was not the recorded one and the OpenMP x pthreads oversubscription made its
   timings meaningless -- found 2026-09-06 while preparing the strong-scaling runs.
2. `l3_clean_conda_build_env` + `CMAKE_INSTALL_RPATH` with the toolchain dirs: the
   wrapper's rpath still came first in the link line -> same resolution; `run.sh`'s
   new guard (`ldd libcp2k.so` must resolve BLAS under the toolchain) refused every
   run of this build.
3. `CP2K_BLAS_VENDOR=CUSTOM` with the toolchain `libopenblas.a` and the toolchain
   rpath in the linker flags: CMake still records a dynamic `libopenblas.so.0`
   dependency, but the RPATH now lists the toolchain directories first and `ldd`
   resolves it to `toolchain/openblas-0.3.33/lib/libopenblas.so.0` (OpenBLAS 0.3.33,
   `USE_OPENMP=1`, the toolchain's own build). This is the validated configuration;
   `run_manifest.txt` records `blas_resolved=`.

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
= last "ENERGY| Total FORCE_EVAL" col 9); [2] H2O-64 MD checked by
`cp2k_md_summary.py --check`: 10 steps reached, **every MD-step SCF cycle
converged**, finite energies, GPU evidence from CP2K's own output (cp2kflags
`offload_cuda dbcsr_acc`, `DBCSR| ACC: Number of devices/node >= 1`, GRID task
statistics with tasks executed on the GPU, `pw_gpu_*` timers when they reach the
timing report); for N > 1 the per-step `ENERGY| Total FORCE_EVAL` energies of MD
steps 1..10 must agree with the 1-GPU run within **1e-8 Ha** (pre-fixed;
upstream's `check-release-comparison.py` demands 1e-10 across CPU MPIxOMP
layouts -- printed as well).

**The initial SCF of upstream's H2O-64 deck does not converge -- by design.**
`benchmarks/QS/H2O-*.inp` start from `SCF_GUESS ATOMIC` with the default
`MAX_SCF 50`, no outer SCF, and declare `IGNORE_CONVERGENCE_FAILURE`; the first
cycle stops after 50 OT/DIIS iterations ("Leaving inner SCF loop after reaching
50 steps", gradient 4e-5) and MD starts from that state. The checker reports this
explicitly (`initial_scf_converged=False initial_scf_iterations=50
deck_ignore_convergence_failure=True`) and would FAIL if the deck did not declare
`IGNORE_CONVERGENCE_FAILURE` or if any of the 10 MD-step SCF cycles were not
converged (negative-tested in `level3/tools/tests/test_l3_validators.sh`).
The deck is used verbatim; loosening/repairing it was not attempted.

### Results (2026-09-06, profile `cuda132-gcc142-ompi5010`, 8 OpenMP threads per rank)

| GPUs | regtests (5) vs upstream refs | H2O-64 MD | FORCE_EVAL energies vs 1 GPU (steps 1..10) | launcher audit |
|---|---|---|---|---|
| 1 | all within tolerance (max \|diff\| 6.0e-14, tol 8e-14..3e-13) | 10/10 MD SCFs converged, GRID GPU tasks 7.5e7, DBCSR ACC 1 device | -- | 1 verified |
| 2 | all within tolerance | 10/10, GRID GPU 3.7e7, `pw_gpu_c1dr3d_3d_ps/pw_gpu_r3dc1d_3d_ps` in timing report | max 5.7e-12 Ha (1e-10 also met) | 2 verified, 0 mismatch |
| 4 | all within tolerance | 10/10, GRID GPU 1.9e7, pw_gpu timers present | max 8.6e-12 Ha (1e-10 also met) | 4 verified, 0 mismatch |

VALIDATED_PASS at 1/2/4 GPUs -- first with the attempt-1 binary (conda BLAS, see above) and
again with the attempt-3 binary (toolchain OpenBLAS; 2026-09-06 04:09-04:12 UTC: regtests within
upstream tolerances, H2O-64 10/10 MD-step SCFs converged, FORCE_EVAL energies vs 1 GPU: max 1.1e-11
Ha on 2 GPUs, 8.6e-12 Ha on 4 GPUs, `blas_resolved=` recorded in every manifest). Runs:
`build/level3/cp2k/cuda132-gcc142-ompi5010/run/` (`validate.*.stdout`, per-run `cp2k.out`,
`md_summary.txt`, `run_manifest.txt`). Strong (H2O-128) / size-sweep timings:
`SECOND_BATCH_STATUS.md` (H2O-128 on 1 GPU: 119.8 s with the toolchain OpenBLAS vs 204.9 s
with the conda pthreads OpenBLAS of attempt 1 -- the BLAS mix-up was also a 1.7x slowdown).

<!-- hpcperf:source-section:begin -->
## Source distribution (frozen source artifact, scheme 3, 2026-09-10)

The application source is not in git and not read from `_upstream/`: `tools/prepare_benchmark.sh level3 cp2k` materializes the frozen source artifact (`<app>[-<variant>]-<source_version>.tar.zst`, found in the local content-addressed cache `.artifacts/sha256/` or downloaded from the immutable URL recorded in `provenance/source.lock*.yaml` once published; `--artifact FILE` for a local copy) into `src/` (+ `deps/`), the only source `build.sh`/`run.sh`/`validate.sh` use. Archive size + sha256 and `source_tree_sha256` are verified before anything is placed. Identity, patch series, licenses, redistribution status and the equivalence proof against the tree the results above were validated from are under `provenance/` (`source.lock*.yaml`, `patch_series*.txt`, `original_vs_baseline*.diff`, `LICENSES*.md`, `equivalence*.md`, `LOC*.md`); `benchmark.yaml` is the machine-readable contract (entries, inputs, references, identity). The benchmark does not prescribe which part of the source an optimization agent may modify; the integrity layer only protects the harness and the validation assets. Remote status: see `level3/SOURCE_ARTIFACTS.md`.

| variant | artifact | source version | compressed / uncompressed | entries | source_tree_sha256 | archive sha256 | upstream | patches (pre-applied) | redistribution | equivalence | remote | LOC app-owned / bundled deps / benchmark deps / tests / total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| - | `cp2k-hpcperf-l3-v1.tar.zst` | hpcperf-l3-v1 | 201.2 MB / 495.3 MB | 9009 | `d877d2d4de4643ce90c1ced68b0cc5d18e389f5055efbf3309cbeba059ef6fa2` | `d6776dd4fa3107d4323d6f75144d047f28776bb9e87adc03597bd69a5dcc02b2` | v2026.2 `67b5da876dd6` | 0001-toolchain-b200-backport-cp2k-378b2fab.patch | cleared | src: EQUIVALENT, src/tools/toolchain: EQUIVALENT | REMOTE_FETCH_VERIFIED | 1085842 / 0 / 20350978 / 38960 / 21479099 |

LOC = cloc 2.06 code lines of the materialized tree (no blank/comment lines, documentation and data excluded); source-ownership categories from `provenance/source.lock*.yaml` (`source_scope`, descriptive metadata written at freeze time). Dependencies are counted per benchmark, so totals overlap across benchmarks that ship the same dependency. The validated results recorded above were produced from trees proven content-equivalent to this artifact (`provenance/equivalence*.md`); they are not re-run by the migration.
<!-- hpcperf:source-section:end -->
