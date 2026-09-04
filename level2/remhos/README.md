# Remhos

High-order remap mini-app (CEED / LLNL). Remhos (REMap High-Order Solver)
solves the pure advection equation with a discontinuous Galerkin (DG)
discretization on high-order (curved) meshes, either as *transport* (a fixed
mesh, a prescribed velocity field, e.g. solid-body rotation) or as *remap* (the
solution is fixed, the mesh moves along a prescribed displacement -- the
field-interpolation step of an arbitrary Lagrangian-Eulerian hydro code such
as Laghos/BLAST). Each Runge-Kutta stage computes a high-order (HO) DG update, a
bound-preserving low-order (LO) update, and combines them with flux-corrected
transport (FCT) into a high-order solution that stays within local bounds and
conserves mass. Motif: high-order finite elements on tensor-product elements --
matrix-free partial-assembly (sum-factorized) mass and convection operators,
element-local mass inverses, element-wise limiting and bound computations, all
expressed as `mfem::forall` kernels on top of MFEM. Figures of merit printed by
the driver: `FOM RHS / INV / LO / FCT` and total `FOM`, in
megadofs x time steps per second.

## Source

Upstream repository: https://github.com/CEED/Remhos
Upstream commit: c65e51dcd6884856d4225e612b2005b82cb2e64f ("baseline minor",
2026-08-04; cloned 2026-09-01 into `$R/_upstream/level2/Remhos`)
License: BSD 2-Clause (Copyright (c) 2018, CEED) with the LLNL "Additional BSD
Notice" -- copied verbatim as `LICENSE` and `NOTICE`.

Copied from upstream (all byte-identical, verified with `cmp`):
`remhos.cpp`, `remhos_fct.cpp/.hpp`, `remhos_ho.cpp/.hpp`,
`remhos_lo.cpp/.hpp`, `remhos_mono.cpp/.hpp`, `remhos_solvers.cpp/.hpp`,
`remhos_sync.cpp/.hpp`, `remhos_tools.cpp/.hpp`, `remhos_main.cpp`,
`remhos_tests.cpp`, `makefile`, `CMakeLists.txt`, `LICENSE`, `NOTICE`,
`autotest/test.sh`, `autotest/out_baseline.dat`, and the 12 meshes in `data/`
(`amr-quad`, `ball-nurbs`, `cube01_hex`, `disc-nurbs`, `inline-quad`,
`periodic-cube`, `periodic-hexagon`, `periodic-segment`, `periodic-square`,
`star-q2`, `star-q3`, `unstr`). Not copied: the six PNG images in `data/`,
`CHANGELOG`, `setup_cuda.sh` (a developer script that clones and builds hypre,
METIS and MFEM `make pcuda`), `.github/`, `.travis.yml` and upstream's
`README.md`.

README verification run 8 refers to `../mfem/data/ball-nurbs.mesh`; Remhos
ships its own `data/ball-nurbs.mesh`, which differs from
`$R/.deps/src/mfem/data/ball-nurbs.mesh` only in one comment line (line 4) and
is numerically identical, so that run works from `level2/remhos/data/` as well.

## Changes from upstream

- None to the upstream files: every copied file is byte-identical to the
  upstream commit and compiles, links and runs unmodified with CUDA 13.2 /
  GCC 13.3 / OpenMPI 5.0.10 against this repository's MFEM v4.10.
- **Build system: the upstream GNU `makefile` is used, not the upstream
  `CMakeLists.txt`.** The task preferred CMake with `find_package(MFEM)`, but
  Remhos' `CMakeLists.txt` is a developer-local file that cannot be made to
  work with a minimal fix: it hard-codes `ccache` as compiler launcher (not
  installed here: `/bin/sh: line 1: ccache: command not found`), an
  `${CMAKE_CURRENT_SOURCE_DIR}/../mfem` include/link directory, Ubuntu and
  Homebrew library paths, links `-lmpi -lmfem -lhypre -lmetis -lfmt` by name
  (`libfmt` is not available and MFEM's CMake config is never consulted -- no
  `find_package(MFEM)`, `MFEM_DIR` is unused), omits `remhos_solvers.cpp`
  from the `Remhos` library, and compiles nothing as CUDA (a CUDA MFEM
  consumer must compile its `mfem::forall` lambdas with nvcc). Making it work
  would have meant rewriting it, which the conventions forbid. The makefile is
  upstream's documented build (`README`: `make`; CI: `make -j` then
  `autotest/test.sh`; `setup_cuda.sh`: `make pcuda` for MFEM, `make` for
  Remhos); it takes `MFEM_CXX`/`MFEM_CXXFLAGS`/`MFEM_LIBS` from MFEM's
  `config.mk`, so Remhos' device kernels get exactly MFEM's nvcc flags and
  `-gencode arch=compute_100,code=sm_100`. It is pointed at the installed MFEM
  purely with command-line variables:
  `make remhos MFEM_DIR=$R/.deps/install/mfem CONFIG_MK=$R/.deps/install/mfem/share/mfem/config.mk TEST_MK=$R/.deps/install/mfem/share/mfem/test.mk`
  (the defaults are `../mfem` and `$(MFEM_DIR)/config/config.mk`, the layout of
  an MFEM *source* build; an installed MFEM has them in `share/mfem/`).
- The makefile compiles in-source (`cd $(<D); nvcc -c file.cpp`, executable in
  the current directory). `build.sh` therefore copies the sources and the
  makefile into `$R/build/level2/remhos/<cuda|hip>/` and builds there;
  `level2/remhos/` itself receives no objects.
- `build.sh`, `run.sh`, `validate.sh` and this `README.md` are new files added
  by this integration.

## Dependencies

- **MFEM v4.10 (MPI + CUDA sm_100 + METIS 5)** -- `$R/.deps/install/mfem`,
  built by `$R/setup_level2_deps.sh` with the conda GCC 13.3.0 as CUDA host
  compiler (do not rebuild). Static `lib/libmfem.a`; `share/mfem/config.mk`
  supplies the compiler (`/usr/local/cuda/bin/nvcc -x=cu -std=c++17 ...
  --expt-extended-lambda --expt-relaxed-constexpr -ccbin <conda g++>`), the
  include paths and the link line (hypre, METIS, cuRAND/cuSOLVER/cuSPARSE/
  cuBLAS, MPI). MFEM's `Device` class selects the backend at run time
  (`-d cuda`; `-d gpu` is accepted too and maps to `cuda` on this build).
  MFEM patch note: on this machine MFEM v4.10 is built with the repository
  patch `patches/mfem-v4.10-blackwell-cicc-O1.patch` (see
  `$R/patches/README.md`): with CUDA 13.2 targeting sm_100 the nvcc device
  front end takes hours on three MFEM translation units (element-assembly and
  batched-LOR kernels), so those three files are compiled with `-Xcicc=-O1`.
  Remhos does not execute those kernels (it uses partial assembly or legacy
  full assembly; no `AssemblyLevel::ELEMENT`, no LOR), so its performance is
  unaffected.
- **hypre 3.2.0 (MPI + CUDA)** -- `$R/.deps/install/hypre`, linked through
  MFEM (`libHYPRE.a`); 32-bit `HYPRE_Int`/`HYPRE_BigInt` (see the FOM overflow
  note under "Run").
- **METIS 5.1.0** -- conda (`$R/.conda_env`), linked through MFEM.
- **OpenMPI 5.0.10** -- conda (`mpirun`, `libmpi`); `hpcperf_env.sh` sets the
  single-node CUDA-aware defaults. Remhos itself is run with one rank here.
- **CUDA 13.2** -- `/usr/local/cuda` (nvcc, cudart and the math libraries
  MFEM/hypre link).
- No Caliper/Adiak/Umpire (all optional upstream; not enabled).

## Verified Environment

GCC/G++ 13.3.0 (conda, pinned) | CMake 3.28.4 | Ninja 1.13.2 | Python 3.12.3
CUDA Toolkit 13.2 (nvcc 13.2.78, /usr/local/cuda) | NVIDIA B200 (sm_100), driver 595.58.03
OpenMPI 5.0.10 (conda, CUDA-aware) | MFEM 4.10 (CUDA, sm_100) | hypre 3.2.0 (CUDA, sm_100) | METIS 5.1.0 (conda)
HIP/ROCm: build config present, unverified (no AMD GPU available)
OS: RHEL 10 (Linux 6.12)

Reproduce the toolchain from the repository root: `./setup_env.sh` then
`source hpcperf_env.sh` (all user-space versions are pinned in environment.yml).

## Backends

CUDA: working (build + run + validation verified on B200)
HIP: `build.sh HIP` is the same `make` invocation against an MFEM built with
`MFEM_USE_HIP=YES` (`HPCPERF_MFEM_PREFIX`); Remhos has no CUDA- or
HIP-specific source. Untested: no ROCm/hipcc and no HIP MFEM on this machine
(`build.sh HIP` exits 1 with an explanatory message).

### GPU execution model

Remhos accepts a device only for one solver combination. `remhos.cpp:398-404`:

```cpp
const bool gpu_setup = device.Allows(Backend::DEVICE_MASK);
if (gpu_setup)
{
   MFEM_VERIFY(ho_type  == HOSolverType::LocalInverse &&
               lo_type  == LOSolverType::MassBased &&
               fct_type == FCTSolverType::ClipScale, "Wrong GPU setup.");
}
```

i.e. `-ho 3 -lo 5 -fct 2` (local-inverse HO, mass-based LO, clip-and-scale
FCT) is the only configuration that runs with `-d cuda`; it is also the only
one whose FOM upstream tracks (README "Performance Timing and FOM": "This
configuration supports partial assembly and GPU execution"). Every other
`-ho/-lo/-fct/-mono` combination -- including all 13 runs of the README
verification table and all of `autotest/test.sh` -- aborts with
`Wrong GPU setup.` on a device and can only be run on the CPU (`-d cpu`, the
default); `validate.sh` runs those on the CPU and says so. With `-d cuda` the
GPU configuration should be combined with `-pa` (partial assembly): full
assembly on the device works but is far slower than PA on the CPU (periodic
hexagon row 1 problem: 24.1 s kernel time for FA on the GPU vs 0.52 s PA on
the GPU and 0.90 s PA on the CPU).

## Build

```bash
source hpcperf_env.sh
./level2/remhos/build.sh CUDA      # -> build/level2/remhos/cuda/remhos
./level2/remhos/build.sh HIP       # untested here (needs hipcc + HIP MFEM)
```

Equivalent raw commands (what `build.sh` runs):

```bash
source hpcperf_env.sh
B=$R/build/level2/remhos/cuda; mkdir -p $B
cp -p level2/remhos/remhos*.cpp level2/remhos/remhos*.hpp level2/remhos/makefile $B/
cd $B && make -j4 remhos \
   MFEM_DIR=$R/.deps/install/mfem \
   CONFIG_MK=$R/.deps/install/mfem/share/mfem/config.mk \
   TEST_MK=$R/.deps/install/mfem/share/mfem/test.mk
```

`build.sh` checks that `config.mk` has `MFEM_USE_MPI = YES` and
`MFEM_USE_CUDA = YES` (`MFEM_USE_HIP` for HIP), reports the GPU compute
capability (`nvidia-smi`, override `HPCPERF_CUDA_ARCH`) and compares it with
the `-gencode arch=compute_XXX` in `config.mk` -- the architecture is fixed by
the MFEM build, so a mismatch is only warned about. Environment overrides:
`HPCPERF_MFEM_PREFIX`, `HPCPERF_CUDA_ARCH`, `HPCPERF_BUILD_JOBS` (default 4).
Extra arguments are passed to `make` (e.g. `REMHOS_DEBUG=YES`). Eight
translation units (`remhos.cpp`, `remhos_tools.cpp`, `remhos_lo.cpp`,
`remhos_ho.cpp`, `remhos_fct.cpp`, `remhos_mono.cpp`, `remhos_sync.cpp`,
`remhos_solvers.cpp`) compile with nvcc in 36 s wall at `-j4`; the
executable is 181 MB (static MFEM + hypre + sm_100 device code). Output goes
to `build/level2/remhos/cuda/build.log` as well. `remhos_main.cpp` and
`remhos_tests.cpp` are copied but not compiled: they are the CMake-only
wrapper/test driver (`remhos.cpp` defines `main` itself under the makefile
build).

## Run

```bash
./level2/remhos/run.sh CUDA                  # default problem below, mpirun -np 1
./level2/remhos/run.sh CUDA -vs 10           # extra remhos args are appended
HPCPERF_REMHOS_RS=3 HPCPERF_REMHOS_DT=0.005 ./level2/remhos/run.sh CUDA
```

Default problem: upstream README sample / verification run 11, the 3D remap
on `data/cube01_hex.mesh` (problem 10: a discontinuous field is remapped while
the mesh follows a Taylor-Green deformation)

```
mpirun -np 8 remhos -m ./data/cube01_hex.mesh -p 10 -rs 1 -o 2 -dt 0.02 -tf 0.8 -ho 1 -lo 4 -fct 2
```

refined three more times (`-rs 4`: 32768 Q2 hexes, 884,736 unknowns) with the
time step scaled with the mesh size (`-dt 0.0025`, 400 RK3 steps), in the GPU
solver configuration with partial assembly, on one rank:

```bash
mpirun -np 1 build/level2/remhos/cuda/remhos -m level2/remhos/data/cube01_hex.mesh \
   -p 10 -rs 4 -o 2 -dt 0.0025 -tf 0.8 -ho 3 -lo 5 -fct 2 -pa -d cuda -no-vis
```

Observed on the B200 (28 s wall clock, ~5.6 GB device memory; measured while
another job kept the host CPUs at load ~13, so the untimed host parts may be
slower than on an idle node):

```
Device configuration: cuda,cpu
Memory configuration: host-std,cuda
Number of unknowns: 884736
time step: 400, time: 1, dt: 0.0025, residual: 0
RHS   kernel time: 3.3863757
L2inv kernel time: 1.3777311
LO    kernel time: 1.0396455
FCT   kernel time: 1.1747803
Total kernel time: 5.6008015
FOM RHS: 313.51607
FOM INV: 770.6026
FOM LO:  1021.1973
FOM FCT: 903.72918
FOM:     189.55915
(megadofs x time steps / second)
Final mass u:  0.1160157138
Max value u:   0.9999868034
Mass loss u:   2.46861e-08
```

`Device configuration: cuda,cpu` is MFEM's banner for an active CUDA backend
(`-d cpu` prints `Device configuration: cpu`). "Total kernel time" is
RHS + LO + FCT (upstream excludes the L2 inverse from the total,
`remhos.cpp:1921`). The same run on the CPU backend of the same binary
(`-d cpu -pa`) is ~6x slower in the kernels (periodic-cube row 7 problem:
1.41 s vs 0.23 s) and gives the same mass/max to all printed digits.

Notes for choosing other sizes (all learned the hard way):

- **`-tf` matters in remap mode** (`-p 10..19`). The pseudo-time always runs
  from 0 to 1 (`-tf` is overwritten with 1.0 for the time loop), but before
  that `apply_mesh_motion()` integrates the Taylor-Green velocity for `t_final`
  to construct the target mesh, so `-tf` sets how strongly the mesh is
  deformed. With the default `-tf 4` the refined `cube01_hex` meshes tangle
  and every solver combination blows up (final mass ~1e+100, or NaN), on the
  CPU as well as on the GPU and for any `-dt`; the README runs use `-tf 0.8`
  (or 0.7 in `autotest`). `run.sh` fixes `-tf 0.8` (`HPCPERF_REMHOS_TF`).
- **FOM integer overflow.** `dofs_steps = GlobalVSize() * steps` is a
  `HYPRE_BigInt`, a 32-bit `int` in this hypre build (no `--enable-bigint`),
  and `steps` already includes the RK stages (x3 for the default RK3). Runs with
  unknowns x steps x 3 > 2^31 print negative FOMs (observed with
  `-rs 4 -o 3 -dt 0.00175`: 2,097,152 x 572 x 3; kernel times are still
  correct). The default stays at 884,736 x 400 x 3 = 1.06e9.
- MFEM's `OptionsParser` rejects an option given twice ("Option
  --refine-serial provided multiple times"), so do not repeat
  `-m/-p/-rs/-o/-dt/-tf/-ho/-lo/-fct/-pa/-d` as extra arguments; use the
  environment overrides `HPCPERF_REMHOS_RS` (4), `HPCPERF_REMHOS_ORDER` (2),
  `HPCPERF_REMHOS_DT` (0.0025; scale with 2^-RS), `HPCPERF_REMHOS_TF` (0.8),
  `HPCPERF_NP` (1; `--oversubscribe` is added for more ranks).
- `run.sh` executes in `build/level2/remhos/<model>/run/` so that optional
  output files (`-save`, `-visit`, `errors.txt`, `si_init.gf` with `-si`) never
  land in `level2/remhos`.

## Validation

```bash
./level2/remhos/validate.sh CUDA     # ~1 min; PASS/FAIL line, exit 0/1
```

Upstream's verification is the README "Verification of Results" table (13
runs with final `mass` and `max`; "valid if the computed values are all within
round-off distance from the above reference values") and `autotest/test.sh`
with `autotest/out_baseline.dat`. None of those runs uses the GPU-capable
solver combination (see "GPU execution model"), so `validate.sh` checks two
things, all with `mpirun -np 1`:

**Part A -- the build reproduces the upstream reference table (CPU backend).**
README rows 1, 3, 5, 7 and 13 (2D transport on a periodic hexagon, transport
on a NURBS disc, 2D transport on a periodic square, 3D transport on a
periodic cube, and the monolithic solver on `inline-quad`) are run with
`-d cpu` and compared with the README values, relative tolerance 1e-8. The
references were produced with 8 ranks; 1 rank reproduces all five to the 10
printed digits, and rows 5 and 9 rerun with `mpirun --oversubscribe -np 2`
give the same digits as with 1 rank (as does `out_baseline.dat`, generated
upstream with 2 ranks), so the results are rank independent. Observed:

| row | command (mpirun -np 1 remhos ... -d cpu) | reference mass / max | computed |
| --- | --- | --- | --- |
| 1 | `-m periodic-hexagon.mesh -p 0 -rs 2 -dt 0.005 -tf 10 -ho 1 -lo 2 -fct 2` | 0.3888354875 / 0.9333315791 | identical |
| 3 | `-m disc-nurbs.mesh -p 1 -rs 3 -dt 0.005 -tf 3 -ho 1 -lo 2 -fct 2` | 3.5982222 / 0.9995717563 | identical |
| 5 | `-m periodic-square.mesh -p 5 -rs 3 -dt 0.005 -tf 0.8 -ho 1 -lo 2 -fct 2` | 0.1623263888 / 0.7676354393 | identical |
| 7 | `-m periodic-cube.mesh -p 0 -rs 1 -o 2 -dt 0.014 -tf 8 -ho 1 -lo 4 -fct 2` | 0.9607429525 / 0.7678305756 | identical |
| 13 | `-m inline-quad.mesh -p 6 -rs 2 -o 1 -dt 0.01 -tf 20 -mono 1 -si 1` | 0.3182739921 / 1 | identical |

**Part B -- the GPU configuration** `-ho 3 -lo 5 -fct 2 -pa -d cuda` on four
README problems (rows 1/2, 5/6, 7 and 11 with the GPU solver combination).
For each case the script requires (1) the MFEM banner
`Device configuration: cuda` (Remhos aborts if the combination cannot run on
the device, so a banner + a finished run means the `mfem::forall` kernels and
MFEM's PA operators executed on the GPU); (2) agreement with the *same*
configuration run on the CPU backend of the same binary (`-pa -d cpu`): mass
within 1e-8, max within 1e-6 relative; (3) for the transport problems, the
final mass equals the README reference mass within 1e-7 relative -- every
Remhos solver combination is conservative, so the table's mass applies to the
GPU combination as well (the PA path loses 1e-9..2e-8 relative mass to
quadrature, identically on CPU and GPU: `Mass loss u: 9.45e-10`, `2.78e-09`,
`1.68e-12`); (4) max <= 1 + 1e-12 (the FCT solution stays within the bounds
of the initial data, which is 0 <= u <= 1 for all four). Observed:

| case | mass (cuda / cpu) | max (cuda / cpu) | README mass | kernel s (cuda / cpu) |
| --- | --- | --- | --- | --- |
| periodic-hexagon `-p 0 -rs 2 -dt 0.005 -tf 10` | 0.3888354884 / 0.3888354884 | 0.866205952 / 0.8662060299 | 0.3888354875 | 0.52 / 0.90 |
| periodic-square `-p 5 -rs 3 -dt 0.005 -tf 0.8` | 0.1623263916 / 0.1623263916 | 0.5904983674 / 0.5904983674 | 0.1623263888 | 0.12 / 0.22 |
| periodic-cube `-p 0 -rs 1 -o 2 -dt 0.014 -tf 8` | 0.9607429525 / 0.9607429525 | 0.637056438 / 0.637056438 | 0.9607429525 | 0.23 / 1.41 |
| cube01_hex remap `-p 10 -rs 1 -o 2 -dt 0.02 -tf 0.8` | 0.1191358858 / 0.1191358858 | 0.9996101564 / 0.9996101564 | (remap: not conserved) | 0.07 / 0.09 |

Three cases agree to all 10 printed digits; the periodic hexagon differs by
9e-8 relative in `max` after 2000 nonlinear clip-and-scale steps (different
reduction order on the device), which is why the max tolerance is 1e-6 rather
than 1e-8. The periodic-cube GPU case rerun with 2 ranks (both on the one
B200) gives the same digits as with 1 rank. Final line observed:

```
PASS: remhos CUDA (README rows 1,3,5,7,13 reproduced on the CPU; 4 GPU runs of -ho 3 -lo 5 -fct 2 -pa -d cuda match the CPU backend and the README reference masses)
```

Logs: `build/level2/remhos/cuda/run/validate-*.log`. Total 57 s.

Reference material that was checked but is *not* used as a pass criterion:

- `autotest/out_baseline.dat` (generated upstream with 2 ranks, committed at
  this HEAD) is reproduced exactly with 1 rank for the runs tried (pacman /
  hexagon / square / cube with `-ho 1 -lo 2 -fct 2`, pacman with
  `-ho 3 -lo 4 -fct 2`), but the full `test.sh` takes many minutes on the CPU
  and its `cuda` mode (`lrun -n N ./remhos -d cuda` with those solver
  combinations) would abort with `Wrong GPU setup.` on the current code.
- README table rows 9 and 11 are stale at this commit: the current code gives
  0.08479546732 / 0.8186892061 (row 9: README 0.08479546709 / 0.8156091428)
  and 0.1197284592 / 0.9992468244 (row 11: README 0.1197294512 /
  0.9990312449), identically on 1 and 2 ranks, while `out_baseline.dat`,
  updated in the same commit series, matches the current code. Rows 2, 4, 6,
  8, 10 and 12 are simply omitted to keep the run short.
- `remhos_tests.cpp` (the CMake-only test driver) hard-codes final masses for
  `-ms 5` early-stopped runs of the GPU configuration (e.g.
  `0.09711395400387984` for `inline-quad -p 14 -rs 1 -o 2 -dt -1.0 -tf 0.5
  -ho 3 -lo 5 -fct 2 -ms 5`); this build gives 0.0971135253 on the CPU and the
  GPU alike (4e-6 relative), so those constants belong to a different MFEM /
  Remhos revision and cannot be used here.

## Warnings

17 compiler diagnostics from the unmodified upstream sources, none fixed (to
keep the code byte-identical); see `build/level2/remhos/cuda/build.log`:

- 9x `nvcc warning : incompatible redefinition for option 'optimize', the last
  value of this option was used` -- one per nvcc invocation (8 compiles + the
  link). MFEM's installed `config.mk` carries both its own `-O3 -DNDEBUG` and
  the conda toolchain's `-O2` in `MFEM_CXXFLAGS`, so host code is compiled at
  `-O2` (the last value). This is a property of the MFEM install, not of
  Remhos; MFEM itself was built the same way.
- 8x `warning #997-D: function "mfem::ODESolver::Init(mfem::TimeDependentOperator &)"
  is hidden by "mfem::ForwardEulerIDPSolver::Init" / "mfem::RKIDPSolver::Init"
  -- virtual function override intended?` at `remhos_solvers.hpp(90)` and
  `(122)`: two sites in a header included by four translation units.
  Pre-existing upstream: the IDP solvers deliberately overload `Init` for
  their own `LimitedTimeDependentOperator`. Harmless.

## LOC

CUDA: 7128 (cloc, 17 files: `remhos*.cpp` + `remhos*.hpp` in `level2/remhos/`;
6492 C++ + 636 header). Of these, 6976 lines (15 files) are compiled by the
makefile; `remhos_main.cpp` (14) and `remhos_tests.cpp` (138) are the
CMake-only wrapper and test driver.
HIP: 7128 (the same files)

Remhos has no per-backend source: the device kernels are `mfem::forall`
lambdas compiled by nvcc or hipcc according to MFEM's `config.mk`. MFEM and
hypre are external dependencies and are not counted. `makefile`,
`CMakeLists.txt`, `autotest/`, `data/`, `LICENSE`, `NOTICE`, the three scripts
and this README are excluded.

## Status

Working (CUDA): build + run + validate all passed on this machine (B200,
CUDA 13.2, MFEM 4.10 sm_100). Built with the upstream makefile, not the
upstream CMakeLists.txt (see "Changes from upstream").
HIP: untested -- no hipcc and no HIP build of MFEM available; `build.sh HIP`
exits 1 with an explanatory message.
