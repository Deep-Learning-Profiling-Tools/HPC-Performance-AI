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
strong = `total_walkers` fixed over N (default 256 = the verbatim population, see
the population limit below), weak = `walkers_per_rank` (default 256); one rank per
GPU, `OMP_NUM_THREADS` crowds, `OMP_TARGET_OFFLOAD=MANDATORY` (no silent
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
534 s wall, launcher audit 1 verified.

### Results (2026-09-06, profile `clang231-cuda132-offload`, 8 crowds per rank)

| Run | GPUs | Walkers (total / per GPU) | DMC LocalEnergy [Ha] (vs ref -21.844975 +- 0.02) | Wall [s] | Audit |
|---|---|---|---|---|---|
| validate / smoke = strong np1 | 1 | 256 / 256 | -21.8492 +- 0.0152 (-0.21 sigma) | 534 | 1 verified |
| validate | 2 | 256 / 128 | -21.8566 +- 0.0126 (-0.58 sigma; 0.37 sigma vs 1 GPU) | 316 | 2 verified, 0 mismatch |
| validate | 4 | 256 / 64 | -21.8313 +- 0.0105 (+0.68 sigma; 0.97 sigma vs 1 GPU) | 254 | 4 verified, 0 mismatch |
| strong | 2 | 256 / 128 | -21.8365 +- 0.0088 | 332 | 2 verified |
| strong | 4 | 256 / 64 | -21.8511 +- 0.0100 | 211 | 4 verified |
| weak | 2 | 512 / 256 | -21.8466 +- 0.0077 | 632 | 2 verified |
| weak | 4 | 1024 / 256 | -21.8433 +- 0.0034 | 656 | 4 verified |

Unit ctests 64/64 and deterministic diamond ctests 526/526 (1 GPU) precede the
science case. Strong scaling 1.61x / 2.53x on 2 / 4 GPUs (64 walkers per B200 is
far below saturation), weak efficiency 84 % / 81 %; the error bars of the weak
series shrink as 1/sqrt(walkers), as they should. Every run passes upstream's
`check_scalars.py` window (`qmc_summary.txt` in each run directory). First
attempts at 4096 total / 1024 per-rank walkers FAILED (next section).

### Walker-population limit found on this build (open issue, documented, not worked around)

Device memory grows by ~320 MB per walker although QMCPACK's own allocators report
~27 MiB: `Free memory on the default device` drops from 181.8 GB to 99.5 GB with 256
walkers and to 17.1 GB with 512 (VMC-only probes, 8 crowds); with 1024 walkers on
one B200 (128 per crowd) every crowd aborts in `cuSolverInverter.hpp:55`
(`CUSOLVER_STATUS_INTERNAL_ERROR`, i.e. cuSOLVER cannot get workspace), and so
did the first strong/weak attempts (4096 total / 1024 per rank), recorded as
FAILED in `SECOND_BATCH_STATUS.md`. Ruled out: allocation granularity (4096 x 1
KiB `cudaMalloc`/`omp_target_alloc` cost 1.0/0.5 KiB each, microbenchmark), the
libomptarget memory manager (`LIBOMPTARGET_MEMORY_MANAGER_THRESHOLD=0` changes
nothing). Not identified: which component of the LLVM 23.1 offload runtime + CUDA
13.2 + QMCPACK 4.4.0 stack holds the memory (a `LIBOMPTARGET_INFO=-1` trace of the
256-walker probe produced 3.5e8 lines and was abandoned). Consequences: run.sh
refuses more than `HPCPERF_QMCPACK_MAX_WALKERS_PER_GPU` (default 300) walkers per
GPU unless forced; strong scaling = the verbatim 256-walker deck over 1/2/4 GPUs,
weak scaling = 256 walkers per GPU. The 256-walker DMC runs use ~120 GB of the
183 GB per GPU.

<!-- hpcperf:source-section:begin -->
## Source distribution (frozen source artifact, scheme 3, 2026-09-10)

The application source is not in git and not read from `_upstream/`: `tools/prepare_benchmark.sh level3 qmcpack` materializes the frozen source artifact (`<app>[-<variant>]-<source_version>.tar.zst`, found in the local content-addressed cache `.artifacts/sha256/` or downloaded from the immutable URL recorded in `provenance/source.lock*.yaml` once published; `--artifact FILE` for a local copy) into `src/` (+ `deps/`), the only source `build.sh`/`run.sh`/`validate.sh` use. Archive size + sha256 and `source_tree_sha256` are verified before anything is placed. Identity, patch series, licenses, redistribution status and the equivalence proof against the tree the results above were validated from are under `provenance/` (`source.lock*.yaml`, `patch_series*.txt`, `original_vs_baseline*.diff`, `LICENSES*.md`, `equivalence*.md`, `LOC*.md`); `benchmark.yaml` is the machine-readable contract (entries, inputs, references, identity). The benchmark does not prescribe which part of the source an optimization agent may modify; the integrity layer only protects the harness and the validation assets. Remote status: see `level3/SOURCE_ARTIFACTS.md`.

| variant | artifact | source version | compressed / uncompressed | entries | source_tree_sha256 | archive sha256 | upstream | patches (pre-applied) | redistribution | equivalence | remote | LOC app-owned / bundled deps / benchmark deps / tests / total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| - | `qmcpack-hpcperf-l3-v1.tar.zst` | hpcperf-l3-v1 | 215.0 MB / 533.3 MB | 8344 | `f0dbdc83677ff4a69b5623c2d24c20b172099ddc53e49dc0581da09b1f5c5b8e` | `6c1585bd027f44f4d6c19324114a2c19112652d87bd37797f17c81b54959f3e3` | v4.4.0 `2601d62e3539` | none | cleared | src: EQUIVALENT | REMOTE_FETCH_VERIFIED | 337840 / 294128 / 10949914 / 538165 / 12122677 |

LOC = cloc 2.06 code lines of the materialized tree (no blank/comment lines, documentation and data excluded); source-ownership categories from `provenance/source.lock*.yaml` (`source_scope`, descriptive metadata written at freeze time). Dependencies are counted per benchmark, so totals overlap across benchmarks that ship the same dependency. The validated results recorded above were produced from trees proven content-equivalent to this artifact (`provenance/equivalence*.md`); they are not re-run by the migration.
<!-- hpcperf:source-section:end -->
