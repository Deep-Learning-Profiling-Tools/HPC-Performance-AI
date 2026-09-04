# hipBone

hipBone is the paranumal group's (Chalmers, Karakus, Austin, Swirydowicz,
Warburton, McDougall) GPU port of the Nekbone CORAL-2 proxy application. It
solves the screened Poisson (Helmholtz-type) equation on a box of hexahedral
spectral elements with an unpreconditioned conjugate gradient solver: every
CG iteration applies the matrix-free high-order spectral-element operator
(tensor-product derivative contractions per element in `okl/hipBoneAx.okl`),
a gather-scatter that sums the element-local results at shared nodes
(`libs/ogs`), and a few vector updates/reductions. The kernels are written
once in the OCCA kernel language (`.okl`) and JIT-compiled at run time for the
selected backend (`-m CUDA`, `-m HIP`, `-m Serial`, ...). The benchmark runs
1000 warm-up CG iterations and then 100 timed iterations and reports the
NekBone figure of merit in GFLOPs.

## Provenance

Upstream repository: https://github.com/paranumal/hipBone
Upstream commit: e9f6908ae2dbf03307ab74798a8c5a3cca929e86 (2026-05-05)
OCCA submodule (`occa/`, https://github.com/libocca/occa): 7a364ea3aee23201d76f74600f367f0994f6822d
Cloned: 2026-09-01 (`_upstream/level2/hipBone`)
License: MIT (`LICENSE`; OCCA is MIT as well, `occa/LICENSE`)

Copied byte-identical from upstream: `hipBone.cpp`, `hipBone.hpp`, `makefile`,
`make.top`, `LICENSE`, `AUTHORS`, `src/`, `include/`, `libs/` (core, mesh,
ogs, primitives, each with its `okl/` kernels), `okl/`, `json/` (kernel tuning
tables), and the OCCA submodule `occa/` reduced to what its Makefile build
needs: `occa/Makefile`, `occa/bin/occa.cpp`, `occa/include/`, `occa/src/`,
`occa/scripts/{build,codegen,compiler}`, `occa/config.default.json`,
`occa/LICENSE`, `occa/README.md`, `occa/INSTALL.md`.

Not copied: upstream `README.md`, `.gitignore`, `.gitmodules`, `run.sh`
(a 15-problem-size sweep for a 4-GPU node, replaced by the `run.sh` here),
`hipBone_poly` (polynomial-degree sweep driver script), and from `occa/`:
`tests/`, `examples/`, `docs/`, `.github/`, `cmake/`, `CMakeLists.txt`,
`configure-cmake.sh`, `configure-sycl-nv-cmake.sh`, `modulefiles/`,
`scripts/docs`, `scripts/editor`, `scripts/valgrind.supp`, `.doxygen`,
`.codecov.yml`, `.gitignore` (unused by the Makefile build, ~30 MB).

## Changes from upstream

- `occa/src/occa/internal/modes/cuda/utils.cpp` (`advise()`, `prefetch()`):
  CUDA 13 removed the `CUdevice` overloads of `cuMemAdvise` and
  `cuMemPrefetchAsync` (the headers now map them to the `_v2` functions that
  take a `CUmemLocation`), which is a hard compile error with CUDA 13.2
  (`could not convert 'cuDevice' from 'CUdevice' {aka 'int'} to
  'CUmemLocation'`). Added `#if CUDA_VERSION >= 13000` branches that build the
  equivalent `CUmemLocation` (type DEVICE + device id for the CUDA device,
  type HOST otherwise); the pre-13 code is kept unchanged in the `#else`.
  Upstream OCCA `main`/`development` still carry the old calls. hipBone does
  not use unified memory, so this only affects compilation.
- Trimmed the OCCA submodule as listed under Provenance. OCCA's Makefile
  globs `tests/src` at parse time, so `build.sh` creates an empty
  `occa/tests/src` directory in the build tree (otherwise `find` prints a
  warning on every make invocation). No makefile was edited.
- Build wrappers only, no source or makefile changes for the toolchain:
  - OCCA is built with `CXX=$CXX CC=$CC` (conda GCC 13.3) and with the conda
    `CXXFLAGS/CFLAGS/LDFLAGS/CPPFLAGS` unset (`env -u`): OCCA replaces its own
    release flags (`-O3 -march=native ...`) with `$CXXFLAGS` whenever that
    variable is set, and conda exports a generic `-O2 -march=nocona` set.
  - OCCA auto-detects every backend whose headers it finds and would also
    enable OpenCL through the CUDA toolkit's `CL/cl.h` (picking
    `/usr/local/cuda-13.1/include` from the default search list). `build.sh`
    selects backends explicitly: `OCCA_CUDA_ENABLED=1 OCCA_HIP_ENABLED=0
    OCCA_OPENCL_ENABLED=0 OCCA_DPCPP_ENABLED=0 OCCA_METAL_ENABLED=0` (mirrored
    for HIP), `OCCA_INCLUDE_PATH=$CUDA_HOME/include`,
    `OCCA_LIBRARY_PATH=$CUDA_HOME/lib64/stubs` (link against the `libcuda.so`
    stub; the real driver library is picked up at run time). OpenMP stays
    auto-detected as upstream intends.
  - hipBone is built with `OPENBLAS_DIR=$CONDA_PREFIX/lib` (its `make.top`
    reads `OPENBLAS_DIR` from the environment and links `-lopenblas`).
- Run-time environment set by `run.sh` / `validate.sh` (upstream leaves all
  of this to the user):
  - `OCCA_CACHE_DIR` and `HIPBONE_CACHE_DIR` point to
    `build/level2/hipbone/occa-cache` so the JIT cache never lands in
    `~/.occa` (OCCA's default, created as soon as libocca is loaded) or next
    to the executable.
  - `OCCA_CXX=$CXX` (host launcher / Serial kernels) and
    `OCCA_CUDA_COMPILER="nvcc -ccbin $CUDAHOSTCXX"` so the JIT uses the same
    conda GCC 13.3 + CUDA 13.2 toolchain as the build instead of whatever
    `g++`/`nvcc` is first on `PATH` (system GCC 14.2 here).
  - `OMP_NUM_THREADS` defaults to 4 (override by exporting it). hipBone's host
    code (`libs/ogs` halo exchange, `libp::memory` operations) has OpenMP
    regions inside the CG loop; with the OpenMP default (one thread per
    allocated core = 16, all busy with other jobs on this shared node) every
    CG iteration cost ~13 ms of thread-synchronisation overhead (GPU kernel
    time per iteration is ~0.07 ms), i.e. the CORAL-2 FOM read 0.7 TFLOPs
    instead of 3.5 TFLOPs. With 1-4 threads the overhead disappears.
- `run.sh`'s default problem is the upstream README's single-GPU CORAL-2
  example (`-nx 24 -ny 24 -nz 24 -p 14`); extra arguments replace that
  problem specification (hipBone aborts when a setting is given twice), `-m`
  is always added by the script.

## Dependencies

- MPI: conda OpenMPI 5.0.10 (`mpic++` wrapper, `mpirun`); hipBone's
  `make.top` hard-codes `HIPBONE_CXX = mpic++`.
- OpenBLAS: conda `libopenblas` 0.3.34 (`$CONDA_PREFIX/lib`), used for the
  LAPACK calls in `libs/core/matrixInverse.cpp` / `matrixRightSolve.cpp`
  (small dense matrices of the 1-D reference-element basis setup,
  `libs/mesh/meshBasis1D.cpp`).
- OCCA: bundled in `occa/`, built by `build.sh` (needs the CUDA driver API
  headers/stub at build time and `nvcc` + `libcuda.so` at run time for the
  JIT). Backends compiled in here: Serial, OpenMP, CUDA.
- CUDA Toolkit 13.2 (system, `/usr/local/cuda`), GCC 13.3 (conda) with
  OpenMP; HIP variant would need ROCm (`$HIP_PATH/bin/hipconfig`, `hipcc`).
- All from `hpcperf_env.sh`; no Level 2 framework library is required.

## Build

```bash
level2/hipbone/build.sh CUDA        # -> build/level2/hipbone/cuda/hipBone
level2/hipbone/build.sh HIP         # form only: fails here, no ROCm
```

`build.sh` mirrors `level2/hipbone/` into `build/level2/hipbone/<cuda|hip>/`
with rsync (the upstream makefiles build in place), then runs the two upstream
makes (`-j4`, `MAKE_JOBS` to change). Equivalent raw commands:

```bash
source hpcperf_env.sh
B=build/level2/hipbone/cuda
rsync -a level2/hipbone/ $B/ && mkdir -p $B/occa/tests/src
# 1. OCCA (occa/Makefile)
export OCCA_CUDA_ENABLED=1 OCCA_HIP_ENABLED=0 OCCA_OPENCL_ENABLED=0 OCCA_DPCPP_ENABLED=0 OCCA_METAL_ENABLED=0
export OCCA_INCLUDE_PATH=$CUDA_HOME/include OCCA_LIBRARY_PATH=$CUDA_HOME/lib64/stubs
env -u CXXFLAGS -u CFLAGS -u LDFLAGS -u CPPFLAGS make -C $B/occa -j4 CXX=$CXX CC=$CC
# 2. hipBone (makefile / make.top: mpic++ -O3 -march=native -std=c++17 -fopenmp)
OPENBLAS_DIR=$CONDA_PREFIX/lib make -C $B -j4 hipBone
```

The build does not bake in a GPU architecture: OCCA queries the device at run
time and passes `-arch=sm_<cc>` to nvcc when it JIT-compiles a kernel
(`HPCPERF_CUDA_ARCH` is only echoed for information). Clean build time on the
shared node: 145 s (OCCA ~2 min, hipBone ~20 s); the kernels are compiled on
the first run (~10 s for the CORAL-2 problem) and cached in
`build/level2/hipbone/occa-cache`.

## Run

```bash
level2/hipbone/run.sh CUDA                              # CORAL-2 single-GPU problem
level2/hipbone/run.sh CUDA -nx 8 -ny 8 -nz 8 -p 8 -v    # other problem, per-iteration residuals
HIPBONE_NP=2 level2/hipbone/run.sh CUDA                 # more ranks (adds --oversubscribe)
```

Equivalent raw command (cache/toolchain variables as described above):

```bash
source hpcperf_env.sh
export OCCA_CACHE_DIR=$PWD/build/level2/hipbone/occa-cache HIPBONE_CACHE_DIR=$OCCA_CACHE_DIR
export OCCA_CXX=$CXX OCCA_CUDA_COMPILER="nvcc -ccbin $CUDAHOSTCXX" OMP_NUM_THREADS=4
cd build/level2/hipbone/cuda && mpirun -np 1 ./hipBone -m CUDA -nx 24 -ny 24 -nz 24 -p 14
```

Options: `-nx/-ny/-nz` elements per rank per direction, `-p` polynomial
degree (1-15), `-m` OCCA mode, `-d` device number, `-ga` GPU-aware MPI, `-v`
print the CG residual every iteration. Output observed here (B200, run.sh
wall time 11 s with cached kernels, 21 s including the first JIT compile):

```
hipBone: 14, 37595375, 0.2870, 100, 7.63e-09, 2732.2, 3412.4, 1.31e+10; N, DOFs, elapsed, iterations, time per DOF, avg BW (GB/s), avg GFLOPs, DOFs*iterations/ranks*time
hipBone: NekBone FOM = 3479.4 GFLOPs.
```

(37.6 M DOFs, 100 timed CG iterations in 0.287 s, 2.7 TB/s effective
bandwidth; results are bit-reproducible run to run on the GPU.)

## Validation

```bash
level2/hipbone/validate.sh CUDA
```

Upstream has no reference solution or built-in check; its "Verifying
correctness" section says to run with `-v` and confirm that the CG residual
norm after the fixed 100 iterations is small, quoting the Nekbone CORAL-2
rule "results are considered correct if the reported r norm is small,
generally less than 1e-8, after 100 conjugate gradient iterations".
`validate.sh` runs `-nx 3 -ny 3 -nz 3 -p 14 -v` (27 elements at the CORAL-2
polynomial degree 14, 68,921 DOFs) once with `-m CUDA` and once with OCCA's
`-m Serial` CPU backend as reference (~12 s in total) and requires:

1. both runs exit 0 and report exactly 100 CG iterations (hipBone runs CG
   with tolerance 0 for a fixed 100 iterations; fewer means the residual
   became NaN/zero);
2. the CUDA final residual `||r_100||` is a finite number and `< 1e-8`
   (CORAL-2 rule), likewise for the Serial reference;
3. the CUDA relative reduction `||r_100|| / ||r_0|| <= 1e-9`;
4. CUDA vs Serial: initial residual norms agree to 1 % and final residual
   norms agree within a factor of 10 (`|log10(ratio)| <= 1`).

Check 4 is deliberately loose because hipBone's right-hand side is the
pseudo-random sequence `rhs[n] = sin(1e9*cos(1e9*n*n))`
(`okl/hipBoneRhs.okl`), whose values depend on the last bits of the
transcendental functions and therefore differ between glibc's libm and the
CUDA math library: the two backends solve the same operator with different
random realisations of the same RHS distribution, so residual norms agree
statistically (both `||r_0||` are ~sqrt(N/2)) but not bit-wise. A broken
operator or gather-scatter shows up as a stagnating or NaN residual (checks
1-3), which is what the script is designed to catch. Result on this machine:

```
CUDA:   initial res norm 185.690385677832   it 100: r norm 9.338101836041e-10   (||r_100||/||r_0|| = 5.0e-12)
Serial: initial res norm 186.105935862556   it 100: r norm 1.079268644721e-09
PASS: hipbone CUDA (-nx 3 -ny 3 -nz 3 -p 14: 100 CG iterations, final r norm 9.338101836041e-10 < 1e-8, Serial reference 1.079268644721e-09)
```

Note for the default `run.sh` problem: at the CORAL-2 size the residual after
100 iterations is 3.58e-08 on this GPU (initial 4335.8, reduction 8e-12),
i.e. above the "generally less than 1e-8" guidance; the residual decreases
monotonically by ~0.77 per iteration there, so this is convergence rate at
37.6 M DOFs, not a defect (Serial gives the same order of magnitude on the
smaller problems tested: `-nx 8 -ny 8 -nz 8 -p 8` 4.40e-08 Serial vs
4.43e-08 CUDA).

## Warnings

None: 0 compiler warnings in the clean `build.sh CUDA` log (hipBone is
compiled with `-O3 -Wall -Wshadow -Wno-unused-function`; OCCA with its
upstream release flags `-O3 -march=native`, which enable no `-W` diagnostics,
so its 269 translation units are only checked for hard errors). The OCCA Makefile prints
one harmless `diff: .../occa/include/occa/defines/compiledDefines.hpp: No
such file or directory` on a clean build (it diffs the freshly generated
defines against the not-yet-existing previous file; upstream behaviour).

## LOC

cloc 2.06, `.okl` counted as C++, makefiles excluded.

hipBone (same source for the CUDA and HIP variants; the backend is chosen at
run time): 9,515 lines in 84 files -- `hipBone.cpp hipBone.hpp src/ include/
libs/ okl/` (C++ 7,271 in 64 files, of which 884 lines are the 8 `.okl`
kernel files in `okl/`, `libs/core/okl/`, `libs/ogs/okl/`; C/C++ headers
2,244 in 20 files).

Bundled OCCA runtime (`occa/src occa/include occa/bin/occa.cpp`): 57,179
lines in 635 files (C++ 38,360; headers 16,779; Fortran interface 1,494;
Objective-C++ (Metal) 546). Of this, the CUDA backend
`occa/src/occa/internal/modes/cuda/` is 1,767 lines (19 files) and the HIP
backend `occa/src/occa/internal/modes/hip/` 1,590 lines (19 files).

CUDA variant: 9,515 (hipBone) + OCCA runtime with the 1,767-line CUDA backend
HIP variant: 9,515 (hipBone) + OCCA runtime with the 1,590-line HIP backend

## Verified Environment

GCC/G++ 13.3.0 (conda, pinned) | C++17 (hipBone) | CMake 3.28.4 | Ninja 1.13.2 | Python 3.12.3
CUDA Toolkit 13.2 (nvcc 13.2.78, /usr/local/cuda) | NVIDIA B200 (sm_100), driver 595.58.03
OpenMPI 5.0.10 (conda, CUDA-aware) | OpenBLAS 0.3.34 (conda) | RHEL 10 (Linux 6.12)
HIP/ROCm: source + build config present where noted, unverified (no AMD GPU available)

Reproduce the toolchain from the repository root: `./setup_env.sh` then
`source hpcperf_env.sh` (all user-space versions are pinned in environment.yml).

## Status

CUDA: Working -- `build.sh CUDA` (clean, 145 s, 0 warnings), `run.sh CUDA`
(CORAL-2 problem, FOM 3479 GFLOPs) and `validate.sh CUDA` (PASS) all verified
on the B200 from a fresh shell on 2026-09-02.
HIP: untested -- `build.sh HIP` stops with "hipconfig not found" (no ROCm on
this machine). The HIP variant is the same source built with
`OCCA_HIP_ENABLED=1` and run with `-m HIP`; the OCCA HIP backend
(`occa/src/occa/internal/modes/hip/`) is unmodified upstream code.
