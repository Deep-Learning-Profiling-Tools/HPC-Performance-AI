# Laghos

High-order Lagrangian shock-hydrodynamics mini-app (CEED / LLNL). Laghos
(LAGrangian High-Order Solver) solves the time-dependent compressible Euler
equations in a moving Lagrangian frame: kinematic quantities (mesh position,
velocity) live in a continuous H1 space of order `-ok` (default Q2), the
specific internal energy in a discontinuous L2 space of order `-ot` (default
Q1), and the semi-discrete system is advanced with explicit high-order
Runge-Kutta (RK4 by default). Each RK stage assembles the corner-force operator
from the stress at the quadrature points (`UpdateQuadratureData`: velocity
gradient, artificial viscosity, eigen-decomposition of the symmetrized velocity
gradient, time-step estimate), applies it (`Forces`) and inverts the velocity
mass matrix with a CG solve (`CG (H1)`) plus a local L2 mass inverse. With
partial assembly (`-pa`, the default) all of this is matrix-free,
sum-factorized on tensor-product elements and written as `mfem::forall` /
`MFEM_FORALL_2D/3D` kernels on top of MFEM, so the same source runs on the CPU
or on the GPU chosen at run time with `-d cuda` (or `-d hip`). Standard test
problems: Taylor-Green vortex (`-p 0`), Sedov blast (`-p 1`, the main problem
of interest), 1D Sod tube (`-p 2`), triple point (`-p 3`), Gresho vortex
(`-p 4`), Rayleigh-Taylor (`-p 7`), .... Figures of merit printed by the
driver: the `CG (H1)`, `CG (L2)`, `Forces` and `UpdateQuadData` rates
(megadofs or megaquads x iterations or time steps per second) and the overall
`Major kernels total rate`; `-f` adds a one-line FOM table.

## Source

Upstream repository: https://github.com/CEED/Laghos
Upstream commit: e1d8787917be6260559fa5375298dcb0ce7dc238 ("Merge pull request
#215 from scheibelp/update-cmake", 2026-08-05; cloned 2026-09-01 into
`$R/_upstream/level2/Laghos`)
License: BSD 2-Clause (Copyright (c) 2017 Lawrence Livermore National Security,
LLC) with the LLNL "Additional BSD Notice" -- copied verbatim as `LICENSE` and
`NOTICE`.

Copied from upstream (byte-identical except for `laghos_solver.cpp`, see
"Changes from upstream"): `laghos.cpp`, `laghos_assembly.cpp/.hpp`,
`laghos_solver.cpp/.hpp`, `CMakeLists.txt`, `cmake/FindMETIS.cmake`,
`cmake/MfemCmakeUtilities.cmake`, `makefile`, `LICENSE`, `NOTICE`, the five
files of `sedov/` (`sedov_sol.cpp/.hpp`, `adaptive_quad.hpp`, `bisect.hpp`,
`sedov.cpp`) and the 16 meshes in `data/` (`box01_hex`, `cube01_hex`,
`cube_12_hex`, `cube_211_hex`, `cube_24_hex`, `cube_311_hex`, `cube_522_hex`,
`cube_922_hex`, `rectangle01_quad`, `rt2D`, `segment01`, `square01_quad`,
`square01_quad_unstr`, `square01_tri`, `square_10x9_quad`, `square_gresho`;
27 KB in total).

`sedov/` could not be left out: `laghos.cpp` includes `sedov/sedov_sol.hpp`
(exact Sedov solution, used by `--check-exact-sedov`) and upstream's
`CMakeLists.txt` compiles `sedov/sedov_sol.cpp` into the `laghos` executable
and additionally builds the small stand-alone `sedov` exact-solution tool from
`sedov/sedov.cpp` (175 LOC, built alongside `laghos` into the same build
directory, not used by the wrappers).

Not copied: `serial/` (the old serial/CPU-only version of Laghos), `amr/`
(the AMR variant; both have their own makefiles and are separate programs),
`data/*.png` (six images, 420 KB), `.github/`, `.travis.yml`, `.gitignore`,
`CHANGELOG`, `INSTALL_cmake.md`, `INSTALL_makefile.md` and upstream's
`README.md`.

## Changes from upstream

Copied code is byte-identical (`cmp` against `$R/_upstream/level2/Laghos`)
except for one file:

- **`laghos_solver.cpp`** (function `QKernel<DIM,Q1D>`, the
  `UpdateQuadratureData` device kernels): `MFEM_UNROLL(1)` was inserted in
  front of the five `MFEM_FOREACH_THREAD(q*,*,Q1D)` loops (two in the 2D
  branch, three in the 3D branch), together with a four-line comment; 13 lines
  added, nothing removed or changed. Full diff:

  ```diff
  @@ -1301,8 +1301,14 @@   (2D branch of QKernel)
            double stressJiT[DIM2];
  +         // HPC-Performance-AI: keep nvcc's device front end (cicc) from unrolling
  +         // these per-thread loops Q1D times per nesting level (each executes once
  +         // per thread since the block is Q1D^dim); on sm_100 with CUDA 13.2 the
  +         // unrolled 3D kernels produced 107 MB of PTX and ptxas ran out of memory.
  +         MFEM_UNROLL(1)
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
  +            MFEM_UNROLL(1)
               MFEM_FOREACH_THREAD(qy,y,Q1D)
  @@ -1330,10 +1336,17 @@  (3D branch of QKernel)
            double stressJiT[DIM2];
  +         // HPC-Performance-AI: (same comment)
  +         MFEM_UNROLL(1)
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
  +            MFEM_UNROLL(1)
               MFEM_FOREACH_THREAD(qy,y,Q1D)
               {
  +               MFEM_UNROLL(1)
                  MFEM_FOREACH_THREAD(qz,z,Q1D)
  ```

  **Why.** With CUDA 13.2 and `CMAKE_CUDA_ARCHITECTURES=100` the unmodified
  file could not be compiled on this machine: nvcc's device front end `cicc`
  ran for 14-16 minutes and emitted **107 MB of PTX** (2.59 M lines), after
  which `ptxas` grew past **32 GB** of RSS within a minute and was killed by
  the kernel when the Slurm job's cgroup reached its 64 GB limit
  (`nvcc error : 'ptxas' died due to signal 9 (Kill signal)`, twice). The
  PTX shows the cause: the eight `QKernel<3,Q1D>` kernels (Q1D = 2, 4, 6, 8;
  the `MFEM_FORALL_3D` launch and, for each, its never-executed
  `MFEM_FORALL_2D` twin) account for 2.58 M of the 2.59 M lines --
  `QKernel<3,8>` alone is 1.43 M lines -- and their size grows like Q1D^3:
  cicc's loop unroller for sm_100 unrolls every
  `for (q = threadIdx.k; q < Q1D; q += blockDim.k)` loop Q1D times per nesting
  level, i.e. it replicates the inlined `QUpdateBody<3>` (3x3 eigen-solver,
  matrix inverses, ...) up to 512 times. `MFEM_UNROLL(1)` is MFEM's own macro
  for this purpose (`general/forall.hpp`: `#pragma unroll(1)` under nvcc
  device compilation, empty otherwise, so the host pass, a CPU-only MFEM and a
  HIP build are untouched). The loops execute exactly once per thread anyway
  (MFEM launches `MFEM_FORALL_3D` with a `Q1D x Q1D x Q1D` block), so the
  pragma removes only dead replicas: same instructions executed, same
  numerical results (all 16 built-in `--checks` at 1e-13 relative and all
  eight README reference runs reproduce to the last printed digit, see
  Validation).

  **Measured effect** (same nvcc flags, `-O3`, sm_100): `laghos_solver.cpp`
  now compiles in 16 s (cicc 5 s, 1.1 MB of PTX) instead of failing after
  16 min; the whole `build.sh CUDA` takes 35 s. The alternative without a
  source change, compiling this one file with `-Xcicc -O1` (what
  `patches/mfem-v4.10-blackwell-cicc-O1.patch` does for three MFEM files),
  also builds in 16 s but halves the speed of the `UpdateQuadratureData` FOM
  kernel: 3D Sedov `-rs 4`, 300 steps, `UpdateQuadData rate` 915 vs 1745
  megaquads x steps/s, overall `Major kernels total rate` 1729 vs 1872
  megadofs x steps/s (CG and Forces rates unchanged). The source fix was
  therefore preferred. The same cicc pathology is reported for MFEM on
  sm_120 / CUDA 12.8 in [mfem/mfem#5363](https://github.com/mfem/mfem/issues/5363).
- Nothing else. In particular upstream's `CMakeLists.txt` is unmodified: its
  `target_include_directories(laghos PUBLIC "${MFEM_DIR}/../../../include/mfem")`
  is correct for this repository's MFEM layout (`find_package(MFEM)` resolves
  `MFEM_DIR` to `$R/.deps/install/mfem/lib/cmake/mfem`, three levels below the
  prefix, and MFEM installs into `lib/`, not `lib64/`), so the extra include
  directory becomes `$R/.deps/install/mfem/include/mfem` (which is what
  `#include "general/forall.hpp"` in `laghos_solver.hpp` needs). The same
  derivation also holds for a `lib64/cmake/mfem` install (same depth); only a
  non-standard package location would break it.
- `build.sh`, `run.sh`, `validate.sh` and this `README.md` are new files added
  by this integration.

## Dependencies

- **MFEM v4.10 (MPI + CUDA sm_100 + METIS 5)** -- `$R/.deps/install/mfem`,
  built by `$R/setup_level2_deps.sh` with the conda GCC 13.3.0 as CUDA host
  compiler (not rebuilt here). Static `lib/libmfem.a`; the CMake package
  `lib/cmake/mfem/MFEMConfig.cmake` (imported target `mfem` carrying the
  hypre, METIS, MPI, cuBLAS/cuSPARSE/cuRAND/cuSOLVER and cudart link
  interface, `MFEM_USE_CUDA=ON`) is what upstream's `find_package(MFEM)`
  consumes; the makefile route uses `share/mfem/config.mk` instead. MFEM's
  `Device` class selects the backend at run time (`-d cuda`; `-d gpu` is
  accepted too and maps to `cuda` on this build; the default is `cpu`).
  MFEM patch note: on this machine MFEM v4.10 is built with the repository
  patch `patches/mfem-v4.10-blackwell-cicc-O1.patch` (see
  `$R/patches/README.md`): with CUDA 13.2 targeting sm_100 the nvcc device
  front end takes hours on three MFEM translation units (element-assembly and
  batched-LOR kernels), so those three files are compiled with `-Xcicc=-O1`.
  Laghos does not execute those kernels (it uses partial assembly, or legacy
  full assembly with `-fa`; no `AssemblyLevel::ELEMENT`, no LOR), so its
  performance is unaffected. Laghos' own `laghos_solver.cpp` hit the same
  class of problem and is handled at the source level as described above.
- **hypre 3.2.0 (MPI + CUDA)** -- `$R/.deps/install/hypre` (`lib64/libHYPRE.a`,
  `lib64/cmake/HYPRE`), found by upstream's `find_package(HYPRE REQUIRED)` and
  also linked through MFEM. Laghos uses hypre only for the parallel FE space /
  `HypreParVector` plumbing; the CG solves are MFEM's own `CGSolver` with
  Laghos' partial-assembly mass operators.
- **METIS 5.1.0** -- conda (`$R/.conda_env`), linked through MFEM (used for
  the mesh partitioning when running with more than one rank).
- **OpenMPI 5.0.10** -- conda (`mpirun`, `libmpi`); `hpcperf_env.sh` sets the
  single-node CUDA-aware defaults. Laghos is run with one rank here.
- **CUDA 13.2** -- `/usr/local/cuda` (nvcc, cudart and the math libraries
  MFEM/hypre link).
- No Caliper/Adiak (optional upstream, `LAGHOS_USE_CALIPER`; not enabled), no
  GLVis/VisIt output used.

## Verified Environment

GCC/G++ 13.3.0 (conda, pinned) | CMake 3.28.4 | Ninja 1.13.2 | Python 3.12.3
CUDA Toolkit 13.2 (nvcc 13.2.78, /usr/local/cuda) | NVIDIA B200 (sm_100), driver 595.58.03
OpenMPI 5.0.10 (conda, CUDA-aware) | MFEM 4.10 (CUDA, sm_100) | hypre 3.2.0 (CUDA, sm_100) | METIS 5.1.0 (conda)
HIP/ROCm: build config present, unverified (no AMD GPU available)
OS: RHEL 10 (Linux 6.12)

Reproduce the toolchain from the repository root: `./setup_env.sh` then
`source hpcperf_env.sh` (all user-space versions are pinned in environment.yml).

## Backends

CUDA: working (build + run + validation verified on B200).
HIP: `build.sh HIP` runs the same upstream `CMakeLists.txt`, which compiles the
sources as HIP (`CMAKE_HIP_ARCHITECTURES`, default `gfx942` as upstream) when
the MFEM it finds has `MFEM_USE_HIP=ON`; Laghos has no CUDA- or HIP-specific
source (`MFEM_UNROLL` is empty under HIP). Untested: no ROCm/hipcc and no HIP
build of MFEM/hypre on this machine (`build.sh HIP` exits 1 with an
explanatory message; point `HPCPERF_MFEM_PREFIX` / `HPCPERF_HYPRE_PREFIX` at
HIP builds to use it).

## Build

```bash
cd $R && source hpcperf_env.sh
level2/laghos/build.sh CUDA          # -> build/level2/laghos/cuda/laghos (+ sedov)
```

`build.sh` detects the GPU compute capability with
`nvidia-smi --query-gpu=compute_cap` (override: `HPCPERF_CUDA_ARCH=100`),
checks that MFEM was built with `MFEM_USE_CUDA=ON` and contains sm_100 device
code, and configures upstream's `CMakeLists.txt` unmodified, equivalent to

```bash
cmake -S level2/laghos -B build/level2/laghos/cuda -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_CUDA_HOST_COMPILER=$CXX \
      -DMFEM_ROOT=$R/.deps/install/mfem -DHYPRE_ROOT=$R/.deps/install/hypre \
      -DCMAKE_CUDA_ARCHITECTURES=100
cmake --build build/level2/laghos/cuda -j4
```

(`-DMFEM_DIR=$R/.deps/install/mfem` works as well; `find_package` then
descends into `lib/cmake/mfem`.) `CMAKE_CUDA_ARCHITECTURES` has to be given
explicitly: upstream's CMakeLists sets it to `70` before `enable_language(CUDA)`
if it is undefined, which also discards the `CUDAARCHS=native` environment
default of `hpcperf_env.sh`. Extra `-D` options can be appended to `build.sh`.
Resulting nvcc line (Ninja generator): `--expt-extended-lambda -O3 -DNDEBUG
-std=c++17 --generate-code=arch=compute_100,code=[compute_100,sm_100]
-ccbin <conda g++>`; all four `.cpp` files are compiled as CUDA. Configure
9 s, build 35 s at `-j4` (longest TU `laghos.cpp` 20 s); `HPCPERF_BUILD_JOBS`
changes the job count. Environment overrides are listed in the script header.

Upstream's reference makefile route also works against the installed MFEM
(verified in a scratch copy; it compiles in-source, so it is not what
`build.sh` uses):

```bash
cp -r level2/laghos build/level2/laghos/make-cuda && cd build/level2/laghos/make-cuda
make -j4 MFEM_DIR=$R/.deps/install/mfem laghos     # picks up share/mfem/config.mk; 24 s
```

## Run

```bash
level2/laghos/run.sh CUDA                       # default problem, 1 rank
level2/laghos/run.sh CUDA -rs 3                 # smaller (later options win)
HPCPERF_LAGHOS_ARGS="-p 3 -m data/box01_hex.mesh -rs 3 -tf 5.0 -pa -cgt 1e-12" level2/laghos/run.sh CUDA
```

Default problem: upstream's main problem of interest, the **3D Sedov blast
wave with partial assembly** -- README verification run 4
(`-p 1 -m data/cube01_hex.mesh -E0 2 -rs 2 -tf 0.6 -pa`) refined two more
levels -- executed as

```
mpirun -np 1 build/level2/laghos/cuda/laghos -p 1 -m data/cube01_hex.mesh -E0 2 -rs 4 -tf 0.6 -pa -f -d cuda
```

from `level2/laghos/` (the script `cd`s there so the relative mesh paths of
upstream's documentation resolve). Size: 32768 Q2-Q1 hexahedra (`-rs 4` of the
8-element `cube01_hex.mesh`), 823,875 kinematic (position/velocity) dofs,
262,144 specific-internal-energy dofs, 64 quadrature points per zone
(2.1 M quadrature points), 2408 RK4 time steps to `t = 0.6`, i.e. about 9.6 k
`UpdateQuadratureData` / `Forces` evaluations and 4 CG(H1) solves per step.
`HPCPERF_LAGHOS_RS` changes the refinement (`-rs 3`: 4096 zones, ~20 s;
`-rs 5`: 262 k zones, 6.4 M H1 dofs, several times longer), `-ms N` caps the
number of steps, `HPCPERF_NP` the rank count (`--oversubscribe` is added for
N > 1). Output (B200, one rank, `-d cuda`; 2 min 18 s wall clock):

```
Device configuration: cuda,cpu
Memory configuration: host-std,cuda
Number of zones in the serial mesh: 32768
Number of kinematic (position, velocity) dofs: 823875
Number of specific internal energy dofs: 262144
step  2408,	t = 0.6000,	dt = 0.000063,	|e| = 2.5655773158e+03
CG (H1) total time: 107.6567380870
CG (H1) rate (megadofs x cg_iterations / second): 1850.8132843388
CG (L2) total time: 5.1338166750
CG (L2) rate (megadofs x cg_iterations / second): 312.7560062320
Forces total time: 3.1691347020
Forces rate (megadofs x timesteps / second): 3332.2814424188
UpdateQuadData total time: 10.6354490500
UpdateQuadData rate (megaquads x timesteps / second): 1922.1602773792
Major kernels total time (seconds): 121.4613218390
Major kernels total rate (megadofs x time steps / second): 1895.7146519220
| Ranks | Zones   | H1 dofs | L2 dofs | QP | N dofs   | FOM0   | FOM1   | T1   | FOM2   | T2   | FOM3   | T3   | FOM    | TT   |
|      1|    32768|   823875|   262144|  64|   4007046|  86.945| 1850.813| 107.657| 3332.281| 3.169| 1922.160| 10.635| 1895.715| 121.461|
```

(`Device configuration: cuda,cpu` is MFEM's banner confirming the CUDA
backend; with `-d cpu` it prints `Device configuration: cpu`.) The velocity
mass-matrix CG dominates (`T1` = 108 of 121 s in the timed kernels) at this
order, as in upstream's own FOM discussions. A second run reproduced the rates
within 0.2 %. The `-rs 2` reference size (512 zones) runs in 20 s but is
latency-bound (`CG (H1) rate` 49); `-rs 4` is where the B200 is saturated.

## Validation

```bash
level2/laghos/validate.sh CUDA                                   # ~2.5 min, exit 0 = PASS
HPCPERF_LAGHOS_CASES="1 2 3 4 6 7 8 9" level2/laghos/validate.sh CUDA   # all 8 GPU rows, ~4 min
```

Two upstream verification mechanisms are used, both on the device
(`-d cuda`) with one MPI rank; logs go to `build/level2/laghos/cuda/run/`.

**Phase 1 -- upstream `make checks`, `-pa -d cuda` option.** Upstream's
makefile target `checks` runs, for every problem 0-7 in 2D
(`data/square01_quad.mesh`) and 3D (`data/cube01_hex.mesh`) and for each of
the option sets `-fa`, `-pa`, `-pa -d cuda` (with `MFEM_USE_CUDA`),
`./laghos -cgt 1.e-14 -rs 0 --checks -p <P> -m <mesh> <options>` on one rank
(`ranks=1`). With `--checks`, `laghos.cpp:Checks()` compares the L2 norm of
the internal-energy field at time step 5 and at the final step against values
hard-coded in the source with a **1e-13 relative** tolerance (`MFEM_VERIFY`
aborts on mismatch and requires that exactly two checks fired, which pins the
final step count). `validate.sh` runs the 16 `-pa -d cuda` combinations with
`mpirun -np 1` (the CMake build cannot use the makefile target directly) and
requires exit code 0 plus the `Device configuration: cuda` banner. Result:
16/16 OK (final steps 27, 15, 59, 16, 18, 36, 37, 25 in 2D; 188, 20, 59, 16,
18, 36, 37, 24 in 3D), ~30 s in total.

**Phase 2 -- upstream README "Verification of Results".** The README lists
final `step`, `dt` and `|e|` for nine `mpirun -np 8 ./laghos ... -pa` runs and
the equivalent `-d gpu` command lines for runs 1-4 and 6-9 (run 5 is 1D,
unsupported on the device; run 7's GPU line adds `-cgt 1e-12`). "An
implementation is considered valid if the final energy values are all within
round-off distance from the above reference values." `validate.sh` parses the
last `step N, t = ..., dt = ..., |e| = ...` line exactly as upstream's
`make tests` target does (`grep -E '^[[:space:]]*step[[:space:]]+[0-9]+' |
tail -n 1 | awk '{print $2, $8, $11}'`) and requires the step count and the
printed `dt` to match exactly and `|e|` to match within a relative tolerance
`HPCPERF_LAGHOS_E_RTOL` (default **1e-8**). Default cases 3, 2, 4, 7 (the 2D
Sedov run and the three 3D runs: Taylor-Green, Sedov, triple point);
`HPCPERF_LAGHOS_CASES` selects others. Results on the B200 (`-d cuda`,
`mpirun -np 1`, wall-clock time per run in parentheses):

| run | command (`-d cuda`, 1 rank) | ref step / dt / e | observed step / dt / e | rel. diff e | time |
|-----|-----------------------------|-------------------|------------------------|-------------|------|
| 1 | `-p 0 -m data/square01_quad.mesh -rs 3 -tf 0.75 -pa` | 339 / 0.000702 / 4.9695537349e+01 | 339 / 0.000702 / 4.9695537349e+01 | 0 | 7 s |
| 2 | `-p 0 -m data/cube01_hex.mesh -rs 1 -tf 0.75 -pa` | 1041 / 0.000121 / 3.3909635545e+03 | 1041 / 0.000121 / 3.3909635545e+03 | 0 | 24 s |
| 3 | `-p 1 -m data/square01_quad.mesh -rs 3 -tf 0.8 -pa` | 1154 / 0.001655 / 4.6303396053e+01 | 1154 / 0.001655 / 4.6303396053e+01 | 0 | 19 s |
| 4 | `-p 1 -m data/cube01_hex.mesh -E0 2 -rs 2 -tf 0.6 -pa` | 560 / 0.002449 / 1.3408616722e+02 | 560 / 0.002449 / 1.3408616722e+02 | 0 | 19 s |
| 5 | 1D `segment01.mesh -fa` -- not device-capable | 413 / 0.000470 / 3.2012077410e+01 | skipped | -- | -- |
| 6 | `-p 3 -m data/rectangle01_quad.mesh -rs 2 -tf 3.0 -pa` | 2872 / 0.000064 / 5.6547039096e+01 | 2872 / 0.000064 / 5.6547039096e+01 | 0 | 45 s |
| 7 | `-p 3 -m data/box01_hex.mesh -rs 1 -tf 5.0 -pa -cgt 1e-12` | 858 / 0.000474 / 5.6691500623e+01 | 858 / 0.000474 / 5.6691500623e+01 | 0 | 45 s |
| 8 | `-p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.62831853 -s 7 -pa` | 776 / 0.000045 / 4.0982431726e+02 | 776 / 0.000045 / 4.0982431726e+02 | 0 | 7 s |
| 9 | `-p 7 -m data/rt2D.mesh -tf 4 -rs 1 -ok 4 -ot 3 -pa` | 2462 / 0.000050 / 1.1792848680e+02 | 2462 / 0.000050 / 1.1792848680e+02 | 0 | 49 s |

All eight GPU-capable rows reproduce the reference `|e|` to all 11 printed
significant digits (relative difference below the 5e-11 print resolution) with
identical step counts and `dt`. Full run: `PASS: laghos CUDA (built-in --checks
for p0-7 in 2D/3D passed; README verification runs [1 2 3 4 5 6 7 8 9] match:
step and dt exact, |e| within 1e-8 relative)`, 4 min 08 s including phase 1.

*Rank independence.* The references were produced with `-np 8`; the
discretization is global and MPI changes only the order of the dot-product
and norm reductions, so results agree to round-off across rank counts. Checked
here: run 4 with `mpirun --oversubscribe -np 2` gives `step 560, dt 0.002449,
|e| = 1.3408616722e+02` (identical), run 1 with `-d gpu` and with `-d cpu`
(the CPU path of the same executable) both give `4.9695537349e+01` (identical).

*Tolerance.* The default `E_RTOL = 1e-8` is about 200x the print resolution of
`|e|` and leaves headroom for reduction-order / FMA differences on another GPU
or rank count, while a real defect is far coarser: a wrong kernel changes
`|e|` in the 3rd-6th significant digit or changes the number of steps (the
step count and `dt` are compared exactly).

## Warnings

- CMake build (`build.sh CUDA`, nvcc 13.2.78 + conda GCC 13.3.0 host, Ninja):
  **0 compiler or linker warnings**, no CMake warnings.
- Makefile route: 5x `nvcc warning : incompatible redefinition for option
  'optimize', the last value of this option was used` -- MFEM's installed
  `share/mfem/config.mk` carries both `-O3 -DNDEBUG` and the conda
  `CXXFLAGS` (`... -O2 -ffunction-sections -pipe ...`) in `MFEM_CXXFLAGS`,
  and upstream's makefile reuses that string; harmless (the CMake route does
  not use `config.mk`).
- Run time: none. MFEM prints `Device configuration: cuda,cpu` /
  `Memory configuration: host-std,cuda` as its device banner.
- Environment: `source hpcperf_env.sh` prints the site lmod `lua ... module
  'posix' not found` traceback on this machine (broken system lmod; unrelated
  to the toolchain, ignored by the wrappers with `2>/dev/null`).
- Build environment: the first two build attempts failed with
  `nvcc error : 'ptxas' died due to signal 9 (Kill signal)` after 16 min --
  the ptxas out-of-memory described under "Changes from upstream" (the Slurm
  job cgroup limits the whole allocation to 64 GB; `memory.events` recorded
  `oom_kill 2`). Fixed by the `MFEM_UNROLL(1)` change; with it ptxas needs a
  few hundred MB.

## LOC

`cloc` code lines (blank and comment lines excluded) of the mini-app's own
source; same for CUDA and HIP (single portable source tree, no backend
directories):

| Files | Code lines |
|-------|-----------|
| `laghos.cpp`, `laghos_assembly.cpp`, `laghos_solver.cpp` (C++) | 3436 |
| `laghos_assembly.hpp`, `laghos_solver.hpp` (headers) | 266 |
| **Laghos mini-app total (`laghos*.cpp/hpp`)** | **3702** (upstream 3697 + 5 `MFEM_UNROLL(1)` lines) |
| `sedov/sedov_sol.cpp`, `sedov/sedov_sol.hpp`, `sedov/adaptive_quad.hpp`, `sedov/bisect.hpp` (exact-solution helper linked into `laghos`) | 417 |
| `sedov/sedov.cpp` (stand-alone `sedov` tool, not counted) | 175 |

Not counted: `CMakeLists.txt`, `cmake/*.cmake`, `makefile`, meshes, wrapper
scripts, README. MFEM/hypre/METIS are external dependencies (`.deps/`).

## Status

CUDA: **Working** -- configure + build (0 warnings, 35 s) + run (`run.sh`,
3D Sedov `-rs 4`, 2 min 18 s, FOM 1895.7 megadofs x steps/s) + validation
(`validate.sh`: 16/16 built-in checks, 8/8 README reference runs exact to the
printed digits) verified on the B200 (CUDA 13.2, GCC 13.3.0, OpenMPI 5.0.10,
MFEM 4.10, hypre 3.2.0).
HIP: build configuration present in `build.sh` (upstream CMake HIP path),
**untested** (no ROCm on the development machine, no HIP MFEM/hypre).
