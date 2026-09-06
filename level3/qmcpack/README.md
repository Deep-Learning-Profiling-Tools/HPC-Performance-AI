# QMCPACK -- Level 3 second batch

Real-space quantum Monte Carlo (VMC/DMC) with upstream's recommended NVIDIA GPU
configuration: OpenMP target offload for the batched drivers plus CUDA
(cuBLAS/cuSOLVER) -- `QMC_GPU="openmp;cuda"` -- which requires an LLVM/Clang
compiler with NVPTX offload ("For NVIDIA GPUs, LLVM clang", upstream docs).

## Provenance / versions

| Item | Value |
|---|---|
| QMCPACK | tag `v4.4.0`, `2601d62e353934f1526cab1f67f30b6672b7c76f` (2026-08-31), NCSA/Illinois licence |
| Compiler (private) | **LLVM 23.1.0** (llvmorg-23.1.0, 2026-08-25) built from `llvm-project-23.1.0.src.tar.xz` (sha256 `ab1f0e3e...479ff`): clang;lld + runtimes openmp;offload + GPU runtimes target `nvptx64-nvidia-cuda` (`libc;openmp` -> libompdevice/libomptarget-nvptx.bc); host GCC 14.2.1; `.deps/level3/qmcpack/clang231-cuda132-offload/install/llvm` |
| CUDA | 13.2.78 (clang 23 lists CUDA 13.2 as FULLY_SUPPORTED), sm_100; nvcc for the CUDA parts with clang as host compiler (QMCPACK adds `--allow-unsupported-compiler`) |
| MPI | conda Open MPI 5.0.10 wrappers redirected to clang (`OMPI_CC/OMPI_CXX`) |
| Dependencies (private, same profile) | OpenBLAS 0.3.30 (TARGET=SAPPHIRERAPIDS, single-threaded), HDF5 1.14.5 (parallel; upstream-tested version), Boost 1.90.0 headers; FFTW3/libxml2 from the conda env / system |
| Build strategy | NATIVE + private toolchain (Spack not used: the local Spack is 2025-05 and its `qmcpack` recipe is inert for 4.x GPU options -- audit) |
| Profile | `clang231-cuda132-offload` |

### Why a private LLVM, and why built from source

- The node has no offload-capable clang: `/home/bcui2/clang+llvm` is an old
  install (nvptx bitcode up to sm_60) that does not even start (`libtinfo.so.5`).
- The official binary release `LLVM-23.1.0-Linux-X64.tar.xz` (2.0 GB, sha256
  `18da30f7...fec8c`) contains `libomp.so` and the clang-offload tools but **no
  libomptarget / device runtime** (recorded in
  `.deps/level3/qmcpack/downloads/LLVM-binary-release-check.txt`) -- upstream's
  FAQ says the same ("pre-packaged LLVM releases: most likely no").
- First source build (LLVM_ENABLE_RUNTIMES=openmp;offload only) produced
  `libomptarget.so` but no NVPTX device runtime: since LLVM 21 the device runtime
  is built through `LLVM_RUNTIME_TARGETS=...;nvptx64-nvidia-cuda` and
  `LIBOMPTARGET_DEVICE_ARCHITECTURES` is unused (openmp/docs/ReleaseNotes);
  clang failed with "no library 'libomptarget-nvptx.bc' found". Reconfigured with
  the GPU runtimes target: `lib/nvptx64-nvidia-cuda/libompdevice.a` +
  `libomptarget-nvptx.bc` (62 s incremental). Build time host part: 573 s at -j24
  on local scratch (`/tmp/hpcperf-l3-b2-scratch/qmcpack-llvm`; extracting the
  150k-file tree on the project NFS alone took >1 h and was abandoned).

### Offload probe (toolchain/probe_offload.sh) -- PASS before any QMCPACK build

`OMP_TARGET_OFFLOAD=MANDATORY` (no silent host fallback), `--offload-arch=sm_100`:
`num_devices=1`, default device != initial device, `omp_is_initial_device()==0`
inside the target region, axpy/reduction vs host `max|d|=4.4e-16`,
`rel(sum)=6.4e-13`; MPI 2 and 4 ranks through the common launcher: every rank
exactly one visible GPU, target region on the device; launcher audit 4 ranks
verified (2-rank run too short to be sampled: unverified, expected mapping
logged). Record: `install/llvm/OFFLOAD_PROBE.txt`.

## Build / run / validate

`build.sh` (deps + QMCPACK real; `HPCPERF_QMCPACK_COMPLEX=1` adds the complex
build) refuses to run unless the probe PASSED. Stages: OpenBLAS 0.3.30
(`TARGET=SAPPHIRERAPIDS`, single-threaded) -> HDF5 1.14.5 (parallel, clang MPI
wrappers; `-DCMAKE_IGNORE_PATH=/usr/lib64/cmake/ZLIB;/lib64/cmake/ZLIB` because
the node's zlib-ng CMake package references a `libz.a` that is not installed) ->
Boost 1.90.0 headers -> QMCPACK (`QMC_GPU="openmp;cuda" QMC_GPU_ARCHS=sm_100
QMC_MPI=ON QMC_COMPLEX=OFF QMC_MIXED_PRECISION=OFF ENABLE_PHDF5=ON
BUILD_UNIT_TESTS=ON QMC_GPU_VISIBILITY_VARIABLE=CUDA_VISIBLE_DEVICES`). Build
2026-09-06: deps 310 s, QMCPACK 672 s at -j12; `cuobjdump` archs of the
executable: sm_100 only; `libomptarget.so` resolved from the private LLVM
(checked by run.sh before every run).

`run.sh`: `tests/solids/diamondC_2x1x1_pp/qmc_short_vmcbatch_dmcbatch.in.xml`
(upstream's batched VMC+DMC test deck, 16 electrons, B-spline orbitals from
`pwscf.pwscf.h5`, BFD pseudopotential, J1+J2) -- smoke verbatim (256 walkers),
strong = `total_walkers 4096` fixed over N, weak = `walkers_per_rank 1024`; one
rank per GPU, `OMP_NUM_THREADS` crowds, `OMP_TARGET_OFFLOAD=MANDATORY` (no silent
host fallback). NiO performance decks are not used (orbital files only behind an
anl.box.com link).

`validate.sh` (criteria in its header): [1] upstream ctests on the built tree
(1 GPU): **unit 64/64 passed** (219 s), **deterministic diamond subset 526/526
passed** (260 s; `deterministic-diamondC_{1x1x1,2x1x1}_pp-{vmcbatch,dmcbatch,
sdbatch,vmc_sdj,vmc_dmc}*`, upstream's exact-scalar references with upstream
tolerances); [2] the science case on N GPUs through `qmc_check.py`:
completeness, offload/CUDA banners, "Running OpenMP offload code path" /
"Running on a GPU via CUDA/HIP acceleration" reported by the wavefunction and
driver components, device memory allocated through the offload runtime, 25 DMC
blocks finite, upstream's `check_scalars.py --ns 3 --series 1 -e 2 --le
"-21.844975 0.02"` (DIAMOND2_DMC_SCALARS), and for N > 1 statistical consistency
with the 1-GPU run (3 sqrt(sigma_1^2 + sigma_N^2)). First 1-GPU run: DMC
LocalEnergy -21.84924 +- 0.01517 Ha vs -21.844975 +- 0.02 (deviation -0.21 sigma),
534 s wall, launcher audit 1 verified. Full 1/2/4-GPU results and the
strong/weak timings: `SECOND_BATCH_STATUS.md`.
