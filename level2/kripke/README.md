# Kripke

3D Sn (discrete-ordinates) deterministic particle-transport mini-app from LLNL
(version 1.2.8 line). Kripke solves the steady-state Boltzmann transport equation
for the Kobayashi "3i" test problem with a source-iteration scheme: each iteration
is a full up-wind **wavefront sweep** of the angular flux over zones, directions and
energy groups (`SweepSubdomain`), followed by moment/scattering kernels (`LTimes`,
`Scattering`, `LPlusTimes`, `Source`, `Population`). All kernels are written once in
RAJA and dispatched at run time to a Sequential, OpenMP, CUDA or HIP back end
(`--arch`) and to one of six data layouts / loop nestings of Direction, Group and
Zone (`--layout DGZ|DZG|GDZ|GZD|ZDG|ZGD`). GPU data motion is managed by CHAI
`ManagedArray`s on top of an Umpire device memory pool; multi-rank runs use MPI
with an asynchronous parallel sweep.

## Provenance

Upstream repository: https://github.com/LLNL/Kripke
Upstream commit: `1b35d13af69ba443b59948a2ab7dfcdfeafc612e` (merge of PR #78, 2026-07-21; after the
v1.2.8 tag -- the bundled TPLs are RAJA/camp/CHAI/Umpire 2025.12.0 and BLT 0.7.1)
Cloned: 2026-09-01 into `$R/_upstream/level2/Kripke`
License: BSD-3-Clause (LLNL-CODE-775068); see `LICENSE`, `COPYRIGHT`, `NOTICE`.
Bundled TPLs keep their own licenses (RAJA/camp/CHAI/Umpire: BSD-3-Clause; fmt: MIT;
nlohmann json: MIT; CLI11: BSD-3-Clause; judy: LGPL-2.1; BLT: BSD-3-Clause).

Copied from upstream (byte-identical except for the changes listed below):

| Path | Content |
|------|---------|
| `src/` | Kripke sources (`kripke.cpp`, `Kripke/**`), complete |
| `CMakeLists.txt`, `cmake/` | upstream superbuild + inner project `cmake/kripke/CMakeLists.txt` |
| `blt/` | BLT 0.7.1 CMake macros (without `thirdparty_builtin/{googletest,benchmark,fruit}` and `docs/`) |
| `host-configs/` | all 16 upstream LLNL host-configs (reference for the flags used here) |
| `tpl/raja` (+ `tpl/raja/tpl/camp`) | RAJA 2025.12.0 with camp 2025.12.0 |
| `tpl/chai` (+ `tpl/chai/src/tpl/raja/{,tpl/camp}`) | CHAI 2025.12.0 with its own nested RAJA/camp copy (identical to `tpl/raja`; this nested copy is the one CHAI actually builds, see Dependencies) |
| `tpl/umpire` (+ `src/tpl/umpire/{fmt,json,CLI11,judy}`) | Umpire 2025.12.0 with its header-only helpers |
| `LICENSE`, `COPYRIGHT`, `NOTICE`, `RELEASE`, `CHANGELOG`, `tpl/README.txt` | license / release notes |

Not copied (not needed to build or run the CUDA/HIP benchmark): `tpl/googletest`,
`tpl/adiak`, `tpl/caliper` (tests and Caliper instrumentation are off), the second
nested copies `tpl/chai/src/tpl/umpire` and `tpl/umpire/src/tpl/umpire/camp` (never
entered by CMake because the `umpire`/`camp` targets already exist), all nested
`blt/` copies inside the TPLs (BLT is loaded once from the top level), every TPL's
`test/ tests/ docs/ examples/ exercises/ benchmark/ scripts/ reproducers/ tools/
host-configs/` directory, upstream CI/packaging files (`.gitlab*, .github, .travis.yml,
azure-pipelines.yml, Dockerfile, .uberenv_config.json, .clang-format, ...`), Kripke's own
`scripts/` (release/CI helpers), upstream `README.md` (replaced by this file) and all
`.git*` metadata. Result: `du -sh level2/kripke` = **12 MB** (upstream working tree
without `.git`: 157 MB), of which `tpl/` is 11 MB (chai 4.0 MB, raja 3.8 MB,
umpire 3.3 MB), `blt/` 349 KB, `src/` 259 KB.

## Changes from upstream

Copied code is byte-identical except for the following.

1. **`tpl/raja/include/RAJA/policy/cuda/MemUtils_CUDA.hpp`** and the identical
   **`tpl/chai/src/tpl/raja/include/RAJA/policy/cuda/MemUtils_CUDA.hpp`**
   (both copies kept identical) -- CUDA 13 compatibility, guarded by
   `#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000` so older toolkits compile the
   original code:
   * `DevicePinnedAllocator::malloc`: CUDA 13 removed the `cudaMemAdvise(ptr, n, advice, int device)`
     overload; the function now takes a `cudaMemLocation`. The two calls build
     `cudaMemLocation{cudaMemLocationTypeDevice, device}` / `{cudaMemLocationTypeHost, cudaCpuDeviceId}`
     (same fix the bundled Umpire already carries in `src/umpire/op/CudaAdviseOperation.cpp`).
     Without it: `error: no suitable constructor exists to convert from "int" to "cudaMemLocation"`.
   * camp's `CAMP_CUDA_API_INVOKE_AND_CHECK` macro streams every argument into the error
     message, which requires `operator<<` for `cudaMemLocation`
     (`camp/helpers.hpp(269): error: no operator "<<" matches these operands`). Added
     `camp::experimental::StreamInsertHelper<cudaMemLocation&>` / `<cudaMemLocation const&>`
     specializations (camp's documented extension point for printing foreign types) to the
     same header, printing `{type=<int>, id=<int>}`.
2. **TPL trimming** (see Provenance). Three files that the generic excludes would have
   dropped but that CMake needs were restored verbatim:
   `tpl/raja/tpl/camp/scripts/gen-header-list.sh` and
   `tpl/chai/src/tpl/raja/tpl/camp/scripts/gen-header-list.sh` (camp's `CMakeLists.txt`
   executes it at configure time) and `tpl/umpire/src/tpl/umpire/fmt/ChangeLog.md`
   (fmt's `CMakeLists.txt` lists it as a target source; without it CMake generation fails).
3. **`build.sh` configures the inner project `cmake/kripke` directly** with
   `-DKRIPKE_SOURCE_ROOT=<level2/kripke>` instead of the top-level superbuild
   `CMakeLists.txt`. The superbuild is an `ExternalProject` wrapper whose only purpose is
   to optionally build Caliper/Adiak first; configuring the inner project is exactly what
   it does internally, but gives a single build tree that honours `-j4` and places
   `kripke.exe` at the root of the build directory. The superbuild command is still
   available (see Build).
4. **CUB location**: the bundled RAJA requires the CUDA toolkit's CUB for CUDA >= 11
   (`RAJA_ENABLE_EXTERNAL_CUB=VersionDependent` -> ON) and its `cmake/thirdparty/FindCUB.cmake`
   only searches `${CUB_DIR}` and `${CUDA_TOOLKIT_ROOT_DIR}/include`. CUDA 13.2 ships
   CUB/Thrust as CCCL under `/usr/local/cuda/include/cccl`, so `build.sh` passes
   `-DCUB_DIR=$CUDA_HOME/include/cccl` (override: `HPCPERF_CUB_DIR`; falls back to
   `$CUDA_HOME/include` for older layouts). No download was added; no RAJA files changed.
5. Added `build.sh`, `run.sh`, `validate.sh` and this `README.md`.

Not changed (documented as known upstream quirks): `src/Kripke/Core/MemoryManager.cpp`
uses `#ifdef KRIPKE_USE_CHAI && (...)` (3 x "extra tokens" warnings, see Warnings; the
extra condition is ignored by the preprocessor, which is harmless in a CUDA build);
`ENABLE_GTEST`/`ENABLE_TESTS` are simply turned off instead of removing the
`add_subdirectory(tpl/googletest)`-style references.

## Dependencies

* **MPI**: conda OpenMPI 5.0.10 from `$R/.conda_env` (found by CMake `FindMPI` through
  `$CONDA_PREFIX`, `Found MPI_CXX: .../.conda_env/lib/libmpi.so (version 3.1)`). `ENABLE_MPI=On`
  as in upstream's GPU host-configs; single-rank runs are the benchmark default.
* **CUDA 13.2 toolkit** (`nvcc`, cudart, CUB/Thrust from `include/cccl`), host compiler
  conda GCC 13.3.0 (`$CXX`, passed as `CMAKE_CUDA_HOST_COMPILER`).
* **Bundled (in `tpl/`, built as part of this project, no `$R/.deps` library used)**:
  RAJA 2025.12.0 + camp 2025.12.0 (kernel abstraction, CUDA back end), CHAI 2025.12.0
  (`ManagedArray` host/device coherence), Umpire 2025.12.0 (`QuickPool` device allocator;
  `--dev_pool_size <GB>` sets the initial pool block, default 4 GB, the pool grows on demand),
  fmt 10.2.1 / nlohmann-json 3.10.4 / CLI11 1.8.0 / judy (Umpire helpers), BLT 0.7.1
  (CMake macros). With `ENABLE_CHAI=On` Kripke's CMake adds `tpl/raja/tpl/camp`,
  `tpl/umpire` and `tpl/chai`; CHAI then adds its nested `src/tpl/raja` (no `RAJA` target
  exists yet), so that copy is the RAJA that is compiled, while `tpl/raja` contributes camp.
  Both RAJA copies are identical upstream and carry the identical CUDA 13 patch.
* Not used: OpenMP (`ENABLE_OPENMP=Off`, as in upstream's GPU host-configs), Caliper/Adiak,
  googletest, the `$R/.deps/install/raja` framework build (Kripke pins its own RAJA/CHAI/Umpire
  versions and its CMake always uses the bundled copies).

## Build

```bash
cd $R && source hpcperf_env.sh
./level2/kripke/build.sh CUDA          # -> build/level2/kripke/cuda/kripke.exe   (default)
./level2/kripke/build.sh HIP           # -> build/level2/kripke/hip/kripke.exe    (untested, no ROCm here)
```

`build.sh` detects the GPU (`nvidia-smi --query-gpu=compute_cap`, override
`HPCPERF_CUDA_ARCH=100`), unsets `CUDAARCHS`, uses `-j${HPCPERF_BUILD_JOBS:-4}`, and mirrors
`host-configs/llnl-toss4-H100-cuda12-gcc-vector.cmake` (CUDA) /
`llnl-toss4-MI300A-rocm6-adams.cmake` (HIP): Release, `-O3 -ffast-math` host flags,
`-restrict --expt-relaxed-constexpr` + `-O3 --expt-extended-lambda` device flags, C++17
(upstream `BLT_CXX_STD`), `ENABLE_CHAI=On ENABLE_MPI=On ENABLE_OPENMP=Off`, tests/examples/
benchmarks/docs off. Clean CUDA build on this machine: 103 targets, about 3 min (2m58s measured) with `-j4`.

Equivalent raw commands (CUDA):

```bash
cd $R && source hpcperf_env.sh && unset CUDAARCHS
cmake -S level2/kripke/cmake/kripke -B build/level2/kripke/cuda \
  -DKRIPKE_SOURCE_ROOT=$PWD/level2/kripke -DCMAKE_BUILD_TYPE=Release -DBLT_CXX_STD=c++17 \
  -DENABLE_CHAI=On -DENABLE_MPI=On -DENABLE_OPENMP=Off \
  -DENABLE_TESTS=Off -DENABLE_GTEST=Off -DENABLE_EXAMPLES=Off -DENABLE_BENCHMARKS=Off -DENABLE_DOCS=Off \
  -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_CXX_FLAGS_RELEASE="-O3 -ffast-math" \
  -DENABLE_CUDA=On -DCMAKE_CUDA_COMPILER=$(command -v nvcc) -DCMAKE_CUDA_HOST_COMPILER=$CXX \
  -DCMAKE_CUDA_ARCHITECTURES=100 -DCUB_DIR=$CUDA_HOME/include/cccl \
  -DCMAKE_CUDA_FLAGS="-restrict --expt-relaxed-constexpr" \
  -DCMAKE_CUDA_FLAGS_RELEASE="-O3 --expt-extended-lambda --expt-relaxed-constexpr"
cmake --build build/level2/kripke/cuda -j4
```

Upstream's superbuild form (same result, binary ends up in
`build/level2/kripke/cuda-super/kripke-build/kripke.exe`; set `CMAKE_BUILD_PARALLEL_LEVEL=4`
because the ExternalProject step does not inherit `-j`):
`cmake -S level2/kripke -B build/level2/kripke/cuda-super <same -D options minus KRIPKE_SOURCE_ROOT> && cmake --build build/level2/kripke/cuda-super`.

HIP (form only, cannot be compiled here): `build.sh HIP` uses `${HIPCXX:-hipcc}` as
`CMAKE_CXX_COMPILER`/`CMAKE_HIP_COMPILER`, `-DENABLE_HIP=On -DROCM_PATH=<rocm>
-DCMAKE_HIP_ARCHITECTURES=${HPCPERF_HIP_ARCH:-gfx942} -DGPU_TARGETS=... -DAMD_GPU_TARGETS=...`
with the same common options.

## Run

```bash
./level2/kripke/run.sh [CUDA|HIP] [extra kripke.exe args]
```

Default problem (single rank, one GPU), the upstream-style GPU configuration:

```
mpirun -np 1 kripke.exe --arch CUDA --layout GDZ --groups 64 --legendre 4 --quad 128 \
       --zones 32,32,32 --gset 1 --dset 8 --zset 1,1,1 --niter 10
```

32^3 zones x 64 groups x 128 directions = 268,435,456 unknowns, P4 scattering, 10 source
iterations; ~5 GB device memory. Kripke prints the input summary, the memory table, the
per-iteration `particle count`/`change`, a timer table and the Figures of Merit. Observed on
the B200: `Solve 0.86-0.94 s` (10 iterations), `Throughput 2.8-3.1e9 unknowns/(s/iter)`,
`Grind time 3.2e-10 s`, `Device high water mark 4.99 GB`, wall time 2.5 s including MPI
start-up. Overrides: `KRIPKE_LAYOUT`, `KRIPKE_ZONES`, `KRIPKE_GROUPS`, `KRIPKE_QUAD`,
`KRIPKE_NITER`, `KRIPKE_NP`/`KRIPKE_MPIRUN`; e.g. `KRIPKE_ZONES=64,64,64 ./run.sh` runs the
8x larger problem (2.1e9 unknowns, 39 GB device memory, `Solve 6.5 s`, 13 s wall), and
`KRIPKE_MPIRUN="mpirun --oversubscribe -np 2" ./run.sh CUDA --procs 2,1,1` runs two ranks
sharing the GPU (verified: identical particle counts).

## Validate

```bash
./level2/kripke/validate.sh [CUDA|HIP]     # prints "Kripke CUDA validation: PASS|FAIL", exit 0/1
```

## Validation

Kripke 1.2.8 has no built-in reference check or `--test` mode (upstream CI only runs the
default problem). Every `kripke.exe` always contains the Sequential (host) RAJA architecture
next to the GPU one, so `validate.sh` runs the same small problem
`--layout GDZ --groups 32 --legendre 4 --quad 32 --zones 16,16,16 --gset 1 --dset 8 --zset 1,1,1 --niter 10`
(4,194,304 unknowns) twice, `--arch CUDA` and `--arch Sequential`, and compares the
`iter N: particle count=<P>` value of every one of the 10 iterations. The particle count is
the global zone-volume-weighted sum of the scalar flux, i.e. the end-to-end result of the
sweep, moment, scattering and population kernels. Requirements: both runs exit 0 and print
`Solver terminated` and `END`; exactly 10 positive finite counts each; per-iteration relative
difference <= `KRIPKE_VALIDATE_RTOL` (default `1e-6`, chosen because Kripke prints the count
with `%e`, i.e. 7 significant digits). A wrong GPU result at any iteration fails the check
(verified by perturbing one iteration of the GPU output through a wrapper launcher:
`iter 5: CUDA=9.382710e+07 Sequential=7.382710e+07 rel=2.709e-01 > 1e-6 ... FAIL`, exit 1).

Result observed here (B200, CUDA 13.2 vs. GCC 13.3 `-O3 -ffast-math` host code):

```
== final particle count: CUDA=7.486672e+07 Sequential=7.486672e+07
== 10 iterations compared, max relative difference = 0.000e+00 (tolerance 1e-6)
Kripke CUDA validation: PASS
```

(12-25 s wall depending on host load; the Sequential run of the 4.2M-unknown problem dominates.)

## Warnings

After the changes above, the CUDA build emits 3 distinct compiler warnings (6 lines, each
reported twice by nvcc's host pass), all in upstream Kripke code, left as is:

* `src/Kripke/Core/MemoryManager.cpp:23:24`, `:39:24`, `:47:24`: `warning: extra tokens at end of
  #ifdef directive` -- upstream writes `#ifdef KRIPKE_USE_CHAI && (defined(KRIPKE_USE_CUDA) || defined(KRIPKE_USE_HIP))`;
  the preprocessor evaluates only `KRIPKE_USE_CHAI`, which is the intended result for this build.

Configure-time CMake notices (not compiler warnings): CMP0146 developer warning from BLT's
`FindCUDA` usage and a CMP0096 "OLD" deprecation notice from the Umpire and CHAI
`CMakeLists.txt` -- both from the bundled upstream build files, harmless with CMake 3.28.

## LOC

Kripke's own code, `cloc level2/kripke/src` (the CUDA and HIP variants are the same RAJA
source; kernels are dispatched by `--arch` at run time, so this is counted once):

| Variant | Paths counted | Files | Code lines |
|---------|---------------|-------|-----------|
| CUDA | `src/` (C++ 2702 + headers 3795) | 51 | **6497** |
| HIP  | same `src/` | 51 | **6497** |

Bundled third-party code, `cloc level2/kripke/tpl level2/kripke/blt` (C/C++ headers 114,432,
C++ 13,772, CUDA 62, CMake 7,181, Python 1,515): **150,328** code lines in 869 files
(RAJA counted twice because of CHAI's nested copy). Excluded from all counts: READMEs,
data, scripts, this file.

## Verified Environment

GCC/G++ 13.3.0 (conda, pinned) | C++17 (upstream `BLT_CXX_STD`) | CMake 3.28.4 | Ninja 1.13.2 | Python 3.12.3
CUDA Toolkit 13.2 (nvcc 13.2.78, /usr/local/cuda) | NVIDIA B200 (sm_100), driver 595.58.03
OpenMPI 5.0.10 (conda, `$R/.conda_env`), single node, `mpirun -np 1`
OS: RHEL 10 (kernel 6.12)
HIP/ROCm: source + build config present where noted, unverified (no AMD GPU available)

Reproduce the toolchain from the repository root: `./setup_env.sh` then
`source hpcperf_env.sh` (all user-space versions are pinned in environment.yml).

## Status

CUDA: **Working** -- configure + build (`build.sh CUDA`, clean tree) + run (`run.sh`) +
validate (`validate.sh` PASS) all verified on the B200 on 2026-09-01.
HIP: extracted with build configuration (`build.sh HIP`, mirrors upstream's MI300A
host-config), **untested** (no ROCm/hipcc on this machine).
