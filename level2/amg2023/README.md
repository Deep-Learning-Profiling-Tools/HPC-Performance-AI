# AMG2023

Algebraic multigrid benchmark (LLNL). A single C driver on top of hypre that
builds a 3D diffusion problem on a cuboid through hypre's linear-algebraic IJ
interface and solves it with a Krylov method preconditioned by BoomerAMG,
hypre's parallel algebraic multigrid solver: Problem 1 is a 27-point stencil
solved with AMG-GMRES(100) to a relative residual of 1e-12, Problem 2 a 7-point
Laplacian solved with AMG-PCG (one level of aggressive coarsening) to 1e-8.
Motif: unstructured sparse linear algebra -- SpMV, sparse triple matrix
products (RAP) and interpolation in the multigrid setup phase, plus Krylov
vector operations, all irregular, memory-access bound and communication heavy
(amg-doc: "about 1-2 computations per memory access"). Figure of merit: total
matrix + interpolation nonzeros divided by setup and solve wall clock time,
printed by the driver as `Figure of Merit (FOM)`. Every GPU kernel lives in
hypre; the mini-app itself is host code.

## Source

Upstream repository: https://github.com/LLNL/AMG2023
Upstream commit: fc4d32089c7c8c72e1dfa48101aaa5a29247effe (cloned 2026-09-01)
License: dual Apache-2.0 OR MIT (LLNL, LLNL-CODE-846758) -- both upstream texts
copied as `LICENSE.APACHE` and `LICENSE.MIT`, with `LICENSE` a short pointer
naming the dual license; `NOTICE.md` copied verbatim.

Copied from upstream (byte-identical): `amg.c` (the whole benchmark),
`amg-config.h.in`, `CMakeLists.txt`, `LICENSE.APACHE`, `LICENSE.MIT`,
`NOTICE.md`. Not copied: the legacy `Makefile` (hard-coded LLNL paths for
hypre/CUDA/Umpire; the upstream CMake build is used instead), `amg-doc.pdf`
(the benchmark documentation -- read it at
https://github.com/LLNL/AMG2023/blob/master/amg-doc.pdf; its reference numbers
are quoted below and in `validate.sh`), and upstream's `README.md`. AMG2023 has
no input decks: the problem is generated at run time from `-P`/`-n`/`-problem`.

## Changes from upstream

- None. `amg.c`, `amg-config.h.in` and `CMakeLists.txt` are byte-identical to
  upstream and compile, link and run unmodified with CUDA 13.2 / GCC 13.3 /
  CMake 3.28 against this repo's hypre.
- Two things that looked like they would need a fix, but did not:
  - `find_library(HYPRE_LIBRARIES NAMES HYPRE HINTS ${HYPRE_PREFIX}/lib)` --
    our hypre installs into `lib64`. CMake's `find_library` rewrites a hint
    ending in `/lib` to `/lib64` on 64-bit Linux
    (`FIND_LIBRARY_USE_LIB64_PATHS`), so it resolves
    `.deps/install/hypre/lib64/libHYPRE.a` with no edit.
  - hypre's CUDA objects are C++, so the executable needs `libstdc++` and the
    CUDA runtime. Upstream's `CMakeLists.txt` already does
    `enable_language(CXX)` + `set_target_properties(amg PROPERTIES
    LINKER_LANGUAGE CXX)` and links `CUDA::cudart/cusolver/cublas/cusparse/
    curand` via `find_package(CUDAToolkit)`, so the link works as-is. No
    `-lstdc++` or extra library had to be added.
- `build.sh`, `run.sh`, `validate.sh`, `LICENSE` and this `README.md` are new
  files added by this integration; they do not modify upstream code.

## Dependencies

- **hypre 3.2.0 with CUDA** -- `$R/.deps/install/hypre` (built by
  `$R/setup_level2_deps.sh`; do not rebuild). Static `lib64/libHYPRE.a`, device
  code for sm_100, `HYPRE_USING_CUDA` + `HYPRE_USING_CUSPARSE`, MPI on, unified
  memory off, Umpire off, 32-bit `HYPRE_Int` (so keep the global unknown count
  below 2^31). The driver uses hypre internals (`_hypre_utilities.h`,
  `_hypre_seq_mv.h`), which is why the hypre version matters; upstream requires
  >= 2.27.0 and `amg.c` selects the `_hypre_seq_mv.h` include path for
  `HYPRE_RELEASE_NUMBER > 23300`. Override with `HPCPERF_HYPRE_PREFIX`.
- **MPI** -- conda OpenMPI 5.0.10 from `hpcperf_env.sh`, found by CMake's
  `find_package(MPI)` (`$CONDA_PREFIX/lib/libmpi.so`).
- **CUDA toolkit 13.2** -- `find_package(CUDAToolkit)`; only the runtime and
  the math libraries are linked. No `.cu` file is compiled here, so there is no
  GPU architecture flag in this build: the target architecture is fixed by
  hypre (sm_100). `build.sh` detects the GPU's compute capability and prints it
  next to hypre's for a sanity check only.
- **C/C++ compiler** -- conda GCC 13.3.0 (`$CC`/`$CXX`), the same compiler
  hypre and its nvcc host pass were built with; the final link is C++.
- HIP variant: a hypre built with `--with-hip` plus ROCm (hipcc, rocsparse,
  rocrand). Neither is available on this machine.

## Verified Environment

GCC/G++ 13.3.0 (conda, pinned) | CMake 3.28.4 | Ninja 1.13.2 | Python 3.12.3
CUDA Toolkit 13.2 (nvcc 13.2.78, /usr/local/cuda) | NVIDIA B200 (sm_100), driver 595.58.03
OpenMPI 5.0.10 (conda, CUDA-aware) | hypre 3.2.0 (CUDA, sm_100)
HIP/ROCm: build config present, unverified (no AMD GPU available)
OS: RHEL 10 (Linux 6.12)

Reproduce the toolchain from the repository root: `./setup_env.sh` then
`source hpcperf_env.sh` (all user-space versions are pinned in environment.yml).

## Backends

CUDA: working (configure + build + run + validation verified on B200)
HIP: build configuration present, untested (no ROCm, and no HIP hypre here)

## Build

```bash
source hpcperf_env.sh            # optional: the scripts source it themselves
level2/amg2023/build.sh          # CUDA (default)
level2/amg2023/build.sh HIP      # exits 1 with a message: no hipcc / no HIP hypre
```

Equivalent raw commands (CUDA):

```bash
R=/projects/kzhou6/bcui2/research/HPC-Performance-AI
cmake -S $R/level2/amg2023 -B $R/build/level2/amg2023/cuda \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX \
      -DAMG_WITH_MPI=ON -DAMG_WITH_CUDA=ON \
      -DHYPRE_PREFIX=$R/.deps/install/hypre
cmake --build $R/build/level2/amg2023/cuda -j4
```

HIP (unverified; needs ROCm and a hypre configured `--with-hip`):

```bash
cmake -S $R/level2/amg2023 -B $R/build/level2/amg2023/hip \
      -DCMAKE_BUILD_TYPE=Release -DAMG_WITH_MPI=ON -DAMG_WITH_HIP=ON \
      -DHYPRE_PREFIX=<hip-hypre> -DCMAKE_PREFIX_PATH=$ROCM_PATH
cmake --build $R/build/level2/amg2023/hip -j4
```

## Run

Standard problem: Problem 1 (upstream's default) on one MPI rank / one GPU with
a 256 x 256 x 256 local grid -- 16,777,216 unknowns, 27-point stencil,
AMG-GMRES(100), tolerance 1e-12. The amg-doc single-GPU reference study sweeps
exactly this problem from n=50 to n=200 on a 16 GB V100; 256 is a B200-sized
point on the same curve (~30 GB of device memory).

```bash
level2/amg2023/run.sh                        # == mpirun -np 1 amg -P 1 1 1 -n 256 256 256 -problem 1
level2/amg2023/run.sh CUDA -n 320 320 320    # extra args win (amg's parser takes the last occurrence)
level2/amg2023/run.sh CUDA -printstats       # hypre AMG setup/convergence tables
HPCPERF_AMG_PROBLEM=2 level2/amg2023/run.sh  # 7-point Laplacian with AMG-PCG
HPCPERF_AMG_P="2 1 1" HPCPERF_AMG_N=100 level2/amg2023/run.sh   # 2 ranks, adds --oversubscribe
```

`-n` is the local size per rank, so the global grid is
`(px*nx) x (py*ny) x (pz*nz)`.

Observed here (1 rank, B200), Problem 1 at n=256:

```
Running on "NVIDIA B200", Comp. Capability: 10.0, Total VRAM: 178.34 GiB
Spatial Operator:   wall clock time = 2.76 s     (host-side matrix generation)
GMRES Setup:        wall clock time = 0.72 s     FOM_Setup 1.28e+09
GMRES Solve:        wall clock time = 0.27 s     FOM_Solve 3.38e+09
Iterations = 19     Final Relative Residual Norm = 6.763764e-13
Figure of Merit (FOM): nnz_AP / (Setup + Solve) 9.29e+08
```

~9-14 s total wall clock, of which ~5 s is MPI/CUDA start-up and ~2.8 s is the
single-threaded host loop that assembles the IJ matrix before it is copied to
the device; the AMG setup + solve being measured take ~1 s. Peak device memory
~30 GB (n=100 uses ~2.4 GB). Runs are bit-reproducible here: repeated runs give
the identical final residual.

## Validation

AMG2023 ships no self-check, so `validate.sh` compares against the reference
runs tabulated in upstream's `amg-doc.pdf`, section K ("Suggested Test Runs"),
for 1 NVIDIA V100 with hypre configured `--with-cuda`. Those tables sweep the
single-rank local grid size and list the iteration count for each n: Problem 1
needs 19 iterations for every n from 50 to 200; Problem 2 needs 30 at n=80-100,
rising slowly to 35 at n=320.

`validate.sh` runs both problems at n = 100 x 100 x 100 on one rank and, for
each, requires:

1. exit code 0;
2. `Final Relative Residual Norm` below the tolerance `amg.c` sets for that
   problem -- 1e-12 for Problem 1, 1e-8 for Problem 2 (`amg.c:122`, `amg.c:457`),
   i.e. the solver actually converged rather than hitting `mg_max_iter = 100`;
3. `Iterations` within +/-2 of the amg-doc reference count (19 for Problem 1,
   30 for Problem 2) -- this is the part that checks the multigrid hierarchy
   itself, since a broken coarsening or interpolation still converges, only
   much more slowly;
4. hypre's device banner (`Running on "<device>", Comp. Capability: ...`) in the
   output. `HYPRE_PrintDeviceInfo()` exists only in a GPU build of hypre, and
   `amg.c` runs with `HYPRE_MEMORY_DEVICE` / `HYPRE_EXEC_DEVICE` by default
   (`amg.c:154-155`), so its presence is the run-time proof that setup and
   solve executed on the GPU rather than on the host.

```bash
level2/amg2023/validate.sh          # CUDA
```

Observed here:

```
=== Problem 1: n = 100x100x100, 1 rank (reference: 19 iterations, residual < 1e-12)
  Iterations = 19   Final Relative Residual Norm = 5.022698e-13
  Running on "NVIDIA B200", Comp. Capability: 10.0, ...
  -> ok (converged below 1e-12 in 19 iterations, reference 19)
=== Problem 2: n = 100x100x100, 1 rank (reference: 30 iterations, residual < 1e-8)
  Iterations = 30   Final Relative Residual Norm = 8.881986e-09
  Running on "NVIDIA B200", Comp. Capability: 10.0, ...
  -> ok (converged below 1e-8 in 30 iterations, reference 30)
PASS: amg2023 CUDA (Problem 1 and 2 at n=100: converged below tolerance, iteration counts match the amg-doc single-GPU reference)
```

Other points of the reference tables reproduce exactly as well: Problem 1 at
n=128/200/256 -> 19 iterations (doc: 19), Problem 2 at n=128 -> 31 (doc: 31 at
n=120-130), Problem 2 at n=320 -> 36 (doc: 35 -- one more, inside the +/-2
band). Two ranks (`-P 2 1 1 -n 100 100 100`, both bound to the one visible GPU)
also converge in 19 iterations.

## Warnings

None. The build (`Release`, upstream's flags: no `-Wall`) compiles `amg.c` and
links with zero diagnostics.

For information, compiling the same file with `-Wall -Wextra` produces 3
warnings, all pre-existing upstream and harmless -- not fixed, to keep the
source byte-identical:

- `amg.c:151` `dev_pool_size` set but not used, and
- `amg.c:151` `um_pool_size` set but not used -- both are only read inside
  `#if defined(HYPRE_USING_UMPIRE)`, and this hypre has Umpire off;
- `amg.c:116` unused variable `coarsen_type` -- the matching
  `HYPRE_BoomerAMGSetCoarsenType` call is commented out upstream.

## LOC

CUDA: 2712 (cloc, 1 file: `amg.c`)
HIP: 2712 (same file)

AMG2023 has exactly one source file and no per-backend code: the CUDA and HIP
variants are the same `amg.c`, compiled with `-DAMG_WITH_CUDA=ON` or
`-DAMG_WITH_HIP=ON` only to select which vendor libraries are linked. All
device kernels come from hypre, which is an external dependency and is not
counted here. `CMakeLists.txt`, `amg-config.h.in`, the licenses and the three
scripts are excluded.

## Status

Working (CUDA): configure + build + run + validate all passed on this machine.
HIP: untested -- no hipcc and no HIP build of hypre available; `build.sh HIP`
exits 1 with an explanatory message.
