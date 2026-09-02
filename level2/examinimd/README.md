# ExaMiniMD

ExaMiniMD (ECP-copa) is a Kokkos-based proxy for classical molecular
dynamics in the style of LAMMPS/miniMD: a short-range pair potential
(Lennard-Jones by default, a SNAP machine-learning potential as an
alternative) integrated with velocity Verlet (NVE) on a periodic box of
atoms. Every step is dominated by the force kernel -- one thread (team) per
atom gathering positions of its neighbours from a Verlet neighbour list and
accumulating 12-6 LJ forces -- plus periodic neighbour-list rebuilds (binning
with `Kokkos::BinSort`, then per-atom neighbour search over the 27 adjacent
bins) and halo exchange of ghost atoms between MPI ranks. Atom data live in
Kokkos Views on the device; the host only drives the time loop. The same
source is the CUDA and the HIP variant: the backend is whatever the Kokkos
installation it is linked against was built for.

## Provenance

Upstream repository: https://github.com/ECP-copa/ExaMiniMD
Upstream commit: 3264e29e28e7a5a4695a959c12df6bdf03bada34 (master, 2023-10-12, "Merge pull request #38 from ECP-copa/iostream")
Cloned: 2026-09-01 (reference clone in `_upstream/level2/ExaMiniMD`)
License: 3-clause BSD (Copyright 2018 NTESS / Sandia) -- copied as `LICENSE`.

Copied from upstream (byte-identical unless listed under "Changes from
upstream"):

- `CMakeLists.txt` -- upstream CMake build (top level; modified, see below)
- `src/` -- all 58 C++ sources/headers and the 5 `CMakeLists.txt` of the
  application (`src/Makefile`, the GNU-make build against a Kokkos source
  tree, was not copied)
- `input/in.lj`, `input/CMakeLists.txt` -- the LJ melt deck (LAMMPS
  `bench/in.lj` at 40^3 unit cells, mass 2.0, T 1.4)
- `input/snap/` -- the two SNAP decks with their Ta/W coefficient and
  parameter files (7 KB in total; `run.sh HPCPERF_EXAMINIMD_DECK=...` can
  run them, see below)
- `LICENSE`

Left out: top-level `Makefile` and `src/Makefile` (GNU-make build),
`scripts/make_module_headers.bash` (developer script that regenerates the
`src/modules_*.h` headers), `.gitignore`, upstream `README.md`.

Added here: `build.sh`, `run.sh`, `validate.sh`, `validate_lj.py`, this
`README.md`.

## Changes from upstream

Upstream last built against Kokkos 3.x/4.0; the project's Kokkos is 5.2.1,
which removed three things the code relied on. The fixes are the smallest
edits that compile (no modernisation, no behaviour change):

1. `src/neighbor_types/neighbor_csr.h` -- `#include <Kokkos_StaticCrsGraph.hpp>`
   -> `#include <KokkosSparse_StaticCrsGraph.hpp>` and the four uses of
   `Kokkos::StaticCrsGraph<T_INT,Kokkos::LayoutLeft,MemorySpace,void,T_INT>`
   -> `KokkosSparse::StaticCrsGraph<...>` (same template signature).
   `Kokkos::StaticCrsGraph` was deprecated in Kokkos 4.6 and removed in 5.0;
   it now lives in Kokkos Kernels. Only the optional `--neigh-type CSR` /
   `CSR_MAPCONSTR` neighbour lists use it, but the header is compiled into
   every build.
2. `CMakeLists.txt` -- added `find_package(KokkosKernels REQUIRED)` after
   `find_package(Kokkos 3.0 REQUIRED)`; `src/CMakeLists.txt` -- added
   `Kokkos::kokkoskernels` to `target_link_libraries(ExaMiniMD ...)` (header-
   only use, but it brings the include path). Consequence of 1.
3. `src/binning_types/binning_kksort.h` / `.cpp` -- Kokkos 4.1 deprecated and
   5.0 deleted the `Kokkos::BinSort` default constructor, so the class member
   `t_sorter sorter;` (default-constructed with the `BinningKKSort` object)
   no longer compiles. The member declaration is replaced by a comment and
   `sorter = t_sorter(x,binop);` in `create_binning()` became
   `t_sorter sorter(x,binop);`. `sorter` was only ever used inside that
   function (the permutation/bin views it produces are separate members), so
   the semantics are unchanged.
4. `src/force_types/force_snap_neigh_impl.h` --
   `Kokkos::DefaultExecutionSpace::concurrency()` (4x) ->
   `Kokkos::DefaultExecutionSpace().concurrency()`: `concurrency()` is a
   non-static member since Kokkos 4.0. In the same block (HIP branch, not
   compiled here) `Kokkos::Experimental::HIP` -> `Kokkos::HIP`, the name
   Kokkos >= 4.0 uses.

The decks and everything else are byte-identical. `diff -ru
_upstream/level2/ExaMiniMD/src level2/examinimd/src` shows exactly the edits
above.

Upstream behaviours worth knowing (deliberately *not* changed):

- `T_INT` is `int` (`src/types.h`); the 2D neighbour list is indexed as
  `N * maxneighs`, which overflows for ~>20 M atoms: a 200^3 deck
  (32,000,000 atoms) dies with `cudaErrorIllegalAddress`. `run.sh` defaults
  to 160^3 (16,384,000 atoms), which works.
- The LJ potential energy always includes the cutoff-energy shift
  (`shift_flag = true` in `src/force_types/force_lj_neigh_impl.h`), so
  `PotE` is `0.44055606` per atom above LAMMPS' unshifted `E_pair` for the
  perfect lattice. Forces are unaffected; see Validation.
- `CommMPI` hands device (CudaSpace) buffers straight to `MPI_Send`/
  `MPI_Irecv`, i.e. it requires a CUDA-aware MPI when running more than one
  rank. The conda OpenMPI is built with CUDA support but ships an
  `etc/openmpi-mca-params.conf` with `opal_cuda_support = false`; `run.sh`
  exports `OMPI_MCA_opal_cuda_support=true` (without it a 2-rank run
  segfaults in `MPI_Send`). One rank never sends device buffers.
- SNAP: the coefficient/parameter files named in the deck are opened
  relative to the current directory with an unchecked `fopen`; run the SNAP
  decks from `level2/examinimd/input/snap/`, otherwise the program segfaults.
  The SNAP force does not compute a potential energy (`PotE` is printed as 0).
- The neighbour list is rebuilt every 20 steps without a displacement check
  (`neigh_modify delay 0 every 20 check no`, as in the LAMMPS benchmark).

## Dependencies

- Kokkos 5.2.1 -- `.deps/install/kokkos` (Serial + OpenMP + CUDA,
  `Kokkos_ARCH_BLACKWELL100`, C++20, `Kokkos_ENABLE_IMPL_VIEW_LEGACY=ON`,
  `Kokkos_ENABLE_DEPRECATED_CODE_4=ON`), built by `./setup_level2_deps.sh
  kokkos`. Applications compile through its `bin/nvcc_wrapper` (nvcc for
  device code, `$NVCC_WRAPPER_DEFAULT_COMPILER` = conda g++ 13.3 for host
  code). The GPU architecture is fixed by this installation, not by the app.
- Kokkos Kernels 5.2.1 -- `.deps/install/kokkos-kernels` (`./setup_level2_deps.sh
  kokkos-kernels`), for `KokkosSparse_StaticCrsGraph.hpp` (header only, see
  change 1).
- CUDA toolkit -- system CUDA 13.2 via `hpcperf_env.sh`.
- MPI (upstream default `USE_MPI=ON`, kept) -- conda OpenMPI 5.0.10 from
  `$CONDA_PREFIX`, found by `find_package(MPI)` without hints. Set
  `HPCPERF_EXAMINIMD_MPI=OFF` for an MPI-free build (upstream's
  `CommSerial`; `run.sh` then launches the binary directly with
  `--comm-type SERIAL`) -- also built and run once (`input/in.lj`, thermo
  rows identical to the MPI build).
- HIP variant: a HIP-enabled Kokkos 5.2.1 + Kokkos Kernels in
  `.deps/install/{kokkos-hip,kokkos-kernels-hip}` and `hipcc` -- none of
  this exists on the development machine.

## Build / Run / Validate

```bash
source hpcperf_env.sh                     # optional -- the scripts source it themselves
./level2/examinimd/build.sh               # CUDA (default); ./build.sh HIP for the HIP variant
./level2/examinimd/run.sh                 # in.lj scaled to 160^3 cells (16.4 M atoms), 1000 steps
./level2/examinimd/validate.sh            # in.lj (256 k atoms) + LAMMPS bench deck, PASS/FAIL + exit code
```

`build.sh` detects the GPU (`nvidia-smi --query-gpu=compute_cap`,
override `HPCPERF_CUDA_ARCH=100`) and compares it with the `Kokkos_ARCH` of
the Kokkos install (a mismatch prints a warning telling you to rebuild Kokkos;
the app itself carries no architecture flag). Extra `-D...` options are
forwarded to CMake; `HPCPERF_BUILD_JOBS` (default 4) sets `-j`.

`run.sh` generates its deck from `input/in.lj` with sed into
`build/level2/examinimd/cuda/run/` (`region box block 0 L 0 L 0 L`,
`thermo`, `run` replaced; `HPCPERF_EXAMINIMD_LATTICE=160`,
`HPCPERF_EXAMINIMD_STEPS=1000`, `HPCPERF_EXAMINIMD_THERMO=100`), or runs
`HPCPERF_EXAMINIMD_DECK=<file>` unchanged (absolute, or relative to
`level2/examinimd`, e.g. `input/in.lj`). `HPCPERF_NP` sets the rank count
(`mpirun --oversubscribe` for >1). Extra arguments go to the binary
(`--force-iteration NEIGH_FULL|NEIGH_HALF|CELL_FULL`, `--neigh-type
2D|CSR|CSR_MAPCONSTR`, `--kokkos-num-threads=N`, ...). It also sets
`OMP_NUM_THREADS=1 OMP_PROC_BIND=spread OMP_PLACES=threads` (Kokkos'
OpenMP host backend is initialised too) unless already set.

Equivalent raw commands (CUDA, MPI on):

```bash
source hpcperf_env.sh
cmake -S level2/examinimd -B build/level2/examinimd/cuda -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=$PWD/.deps/install/kokkos/bin/nvcc_wrapper -DUSE_MPI=ON \
      -DCMAKE_PREFIX_PATH="$PWD/.deps/install/kokkos;$PWD/.deps/install/kokkos-kernels"
cmake --build build/level2/examinimd/cuda -j4
cd build/level2/examinimd/cuda
sed -e 's/^region.*/region box block 0 160 0 160 0 160/' -e 's/^thermo.*/thermo 100/' \
    -e 's/^run.*/run 1000/' ../../../../level2/examinimd/input/in.lj > run/in.lj.L160.s1000
OMP_NUM_THREADS=1 OMP_PROC_BIND=spread OMP_PLACES=threads \
mpirun -np 1 ./src/ExaMiniMD -il run/in.lj.L160.s1000 --comm-type MPI
```

(`CMAKE_PREFIX_PATH` is redundant after `source hpcperf_env.sh`, which puts
every `.deps/install/*` on it; `-DKokkos_ROOT` would be ignored because
upstream's `cmake_minimum_required(VERSION 3.10)` predates policy CMP0074.)

HIP (same source; configuration present, unverified without ROCm):

```bash
cmake -S level2/examinimd -B build/level2/examinimd/hip -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=hipcc -DUSE_MPI=ON \
      -DCMAKE_PREFIX_PATH="$PWD/.deps/install/kokkos-hip;$PWD/.deps/install/kokkos-kernels-hip"
cmake --build build/level2/examinimd/hip -j4
```

Program output: one row per `thermo` interval, `step T PotE/atom ETot/atom
time atomsteps/s`, and a final `... PERFORMANCE` line
(`ranks atoms | total force neigh comm other | steps/s atomsteps/s
atomsteps/(rank*s)`).

## Validation

Upstream ships no reference outputs: its `--dumpbinary`/`--correctness`
options only compare a run against a binary dump of an earlier run of
itself, `scripts/` has none, and the README quotes no numbers. The check
(`validate.sh` + `validate_lj.py`) therefore uses two independent
references, both derived here and not taken from any ExaMiniMD run:

*Analytic lattice sums.* `create_atoms` places a perfect fcc lattice at
reduced density 0.8442 (a = 1.6795962), and `velocity create` scales the
velocities to exactly T0 with the zero-momentum, 3N-3 degrees-of-freedom
convention. Summing 4(r^-12 - r^-6) over the 54 lattice neighbours within
rc = 2.5 (shells of 12, 6, 24, 12 atoms at 1.1877, 1.6796, 2.0571, 2.3753)
gives, per atom, E_pair = -6.7733680533 unshifted, -6.3328119926 with
ExaMiniMD's cutoff shift (0.5 x 54 x 0.0163168911 = 0.4405560607), and
KE = 1.5 T0 (N-1)/N. `validate_lj.py` recomputes these for the deck's
density, cutoff, lattice size and T0.

*LAMMPS.* The LAMMPS repository publishes the log of the identical problem
(`bench/in.lj`, 20^3 cells = 32,000 atoms, mass 1.0, T0 = 1.44, LAMMPS
12 Jun 2025): https://raw.githubusercontent.com/lammps/lammps/develop/bench/log.15Jul25.lj.fixed.g++.1
-- step 0: `Temp 1.44  E_pair -6.7733681  TotEng -4.6134356`; step 100:
`Temp 0.7574531`. The analytic unshifted E_pair matches this LAMMPS value
to all 8 printed digits, which cross-checks both references.

Two decks are run (through `run.sh`, 1 rank):

| deck | check | tolerance | observed (B200) |
|---|---|---|---|
| A: `input/in.lj` unchanged, 40^3 = 256,000 atoms, T0 = 1.4, mass 2.0, 100 steps, thermo 10 | T(0) = 1.4 | abs 1e-5 | 1.400000 |
| | PotE(0)/atom = -6.3328120 | rel 1e-5 | -6.332812 (1.2e-9) |
| | ETot(0)/atom = -6.3328120 + 2.0999918 = -4.2328202 | rel 1e-5 | -4.232820 (4.6e-8) |
| | max_t \|ETot(t)-ETot(0)\|/\|ETot(0)\| over the 11 thermo rows | < 1e-3 | 9.4e-5 |
| B: `in.lj` with `region 0 20`, `mass 1 1.0`, `velocity ... 1.44` (= LAMMPS bench/in.lj), 32,000 atoms, 100 steps | the same four checks with T0 = 1.44, ETot(0) = -4.1728795 | as above | 1.440000 / -6.332812 / -4.172879 / drift 2.1e-4 |
| | PotE(0) - 0.4405561 vs LAMMPS E_pair(0) = -6.7733681 | rel 1e-5 | -6.7733681 (5.8e-9) |
| | ETot(0) - 0.4405561 vs LAMMPS TotEng(0) = -4.6134356 | rel 1e-5 | -4.6134351 (1.2e-7) |
| | T(100) vs LAMMPS Temp(100) = 0.7574531 | rel 1e-3 | 0.757453 (1.3e-7) |

Notes on what is and is not compared: PotE at later steps cannot be
compared with LAMMPS because the shift then depends on the instantaneous
number of pairs inside rc; T(100) can, because the shift does not change
the forces and ExaMiniMD reproduces LAMMPS' `velocity ... loop geom`
initial condition (the match to 7 digits confirms identical dynamics). The
energy-drift tolerance is loose on purpose: the drift is a property of the
deck (dt = 0.005, list rebuilt every 20 steps without a check), not of the
port, and LAMMPS' own unshifted TotEng moves by 1.9e-3 relative over the
same 100 steps. Runs are bit-reproducible here (two identical runs gave
identical thermo rows); a 2-rank run (`HPCPERF_NP=2`, both ranks on the one
GPU) reproduces deck A's step-100 row to 6 digits (0.733401 vs 0.733399).

`validate.sh` prints every check, then
`ExaMiniMD CUDA validation (in.lj energy + LAMMPS reference): PASS` and
exits 0 (or `FAIL` / 1). It takes 6-10 s wall (the two MD loops are 0.04 s
and 0.015 s; the rest is MPI/CUDA start-up and lattice creation on the host).

Benchmark (`run.sh` defaults, 160^3 cells = 16,384,000 atoms, 1000 steps):
17-21 s wall, 9.8-10.0 s in the MD loop (force 6.76 s, neighbour 2.68 s,
comm 0.10 s), **1.64-1.67e9 atom-steps/s** over two runs, energy -4.232812
-> -4.232823 over 1000 steps. For reference upstream's 256,000-atom deck runs at 6.5e8 atom-steps/s
(loop 0.04 s) and 4,000,000 atoms at 1.59e9. `--neigh-type CSR` and
`CSR_MAPCONSTR`, `--force-iteration NEIGH_HALF` and the SNAP deck
`input/snap/in.snap.Ta06A` (64 Ta atoms, 100 steps, run from its directory)
were also run once each; the LJ variants give identical thermo rows.

## Warnings

Compiler (nvcc 13.2.78 through nvcc_wrapper + GCC 13.3, 24 translation
units, clean build):

- 24x `nvcc_wrapper - *warning* you have set multiple optimization flags
  (-O*), only the last is used` -- one per TU; conda's `CXXFLAGS` carry `-O2`
  and CMake's Release adds `-O3`; harmless (`-O3` wins).
- 6x `warning #20011-D: calling a __host__ function
  ("NeighListCSR<...>::NeighListCSR(const NeighListCSR&)") from a
  __host__ __device__ function` at `neighbor_csr.h(368)`,
  `neighbor_csr_map_constr.h(234)`, `force_lj_neigh_impl.h(296)` (CudaSpace
  and HostSpace instantiations each): the functor classes' implicit
  host-device copy constructors copy a `NeighListCSR` member whose
  user-written copy constructor is host-only. Upstream code (present before
  the `KokkosSparse` rename); the copy only ever runs on the host.
- 4x `Warning #20014-D: calling a __host__ function from a __host__ __device__
  function is not allowed` from Kokkos' `View/Kokkos_ViewLegacy.hpp` (legacy
  View subview/range constructor), instantiated by the `Kokkos::subview` /
  `View(view, range)` calls in `binning_kksort.cpp` lines 128-136 (host
  code). Kokkos-internal, harmless.

CMake: 1 warning at configure time -- `The installed Kokkos configuration
does not support CXX extensions. Forcing -DCMAKE_CXX_EXTENSIONS=Off`
(Kokkos' config file, expected).

Runtime: none with `run.sh` (it sets `OMP_PROC_BIND`; without it Kokkos
prints its OpenMP "OMP_PROC_BIND environment variable not set" note).

## LOC

cloc v2.06, code lines only (blank/comment excluded); `CMakeLists.txt`
files, decks, scripts and this README excluded. ExaMiniMD is a Kokkos
application, so CUDA and HIP are the same `src/` -- counted once:

CUDA variant = HIP variant: **6164** (58 files) = 4344 in 35 headers + 1820 in 23 `.cpp`

```bash
cloc --quiet level2/examinimd/src --exclude-ext=cmake --not-match-f='CMakeLists.txt'
```

(The count includes the optional SNAP force, ~1400 lines in
`src/force_types/{force_snap_neigh*,sna*}`, and the alternative
CSR neighbour lists and serial comm, which the default LJ run does not
execute.)

## Verified Environment

GCC/G++ 13.3.0 (conda, pinned) | C++20 | CMake 3.28.4 | Ninja 1.13.2 | Python 3.12.3
CUDA Toolkit 13.2 (nvcc 13.2.78, /usr/local/cuda) | NVIDIA B200 (sm_100), driver 595.58.03
Kokkos 5.2.1 + Kokkos Kernels 5.2.1 (`.deps/install/{kokkos,kokkos-kernels}`, Serial+OpenMP+CUDA, BLACKWELL100)
OpenMPI 5.0.10 (conda, CUDA-aware build; CUDA support enabled at runtime by `OMPI_MCA_opal_cuda_support=true`, see above)
HIP/ROCm: source + build config present where noted, unverified (no AMD GPU available)
OS: RHEL 10 (Linux 6.12)

Reproduce the toolchain from the repository root: `./setup_env.sh` then
`source hpcperf_env.sh` (all user-space versions are pinned in environment.yml);
the Kokkos libraries with `./setup_level2_deps.sh kokkos kokkos-kernels`.

## Status

CUDA: **Working** -- configure + build (2 min 14 s clean, `-j4`) + run
(16.4 M atoms, 1000 steps, 1.64e9 atom-steps/s) + validate (both decks
PASS) all succeeded on this machine, from a clean shell and a foreign cwd,
with MPI on (1 rank; 2 ranks on the one GPU also run). Four small source
edits were needed for Kokkos 5.2.1 (see "Changes from upstream").
HIP: same source, untested (no ROCm/hipcc on the development machine;
`./build.sh HIP` exits with "HIP requested but hipcc was not found").
