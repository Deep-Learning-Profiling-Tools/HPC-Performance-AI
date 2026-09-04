# CabanaPIC

CabanaPIC (ECP CoPA) is a structured-grid, relativistic, electromagnetic
particle-in-cell (PIC) plasma mini-app written on Cabana (particle AoSoA data
structures) and Kokkos. Each time step interpolates the Yee-grid E/B fields to
the particles, does a Boris push of every particle (with periodic boundary
handling when a particle leaves its cell), scatters the particle currents back
onto the grid (`Kokkos::Experimental::ScatterView` atomics), and advances E and
B with a leapfrog FDTD field solve. The main motif is therefore a
particle-parallel gather/push/scatter (memory-bound streaming over the AoSoA
plus contended atomics into a small field array), with a few small
grid-stencil kernels and two energy reductions per step; the whole state lives
on the device and the host only drives the step loop. The problem (the "input
deck") is a C++ file compiled into the binary.

## Provenance

Upstream repository: https://github.com/ECP-copa/CabanaPIC
Upstream commit: 1ee2c84582b051d59653232abe86ac6da2c6b35e (master, 2025-02-19, "Merge pull request #51 from ECP-copa/ci_base_image")
Cloned: 2026-09-01 (reference clone in `_upstream/level2/CabanaPIC`)
License: BSD-3-Clause (Triad National Security, LLC, 2019) -- copied as `LICENSE`.

Copied from upstream (byte-identical unless listed under "Changes from upstream"):

- `CMakeLists.txt`, `src/CMakeLists.txt`, `example/CMakeLists.txt` -- upstream CMake build (modified, see below)
- `src/` -- the PIC library `libCabanaPIC` (14 files: `accumulator.{h,cpp}`, `interpolator.{h,cpp}`, `fields.h`, `push.h`, `move_p.h`, `uncenter_p.h`, `grid.h`, `helpers.h`, `types.h`, `logger.h`, `visualization.h`, `input/deck.h`); `uncenter_p.h` modified
- `example/example.cpp` -- the driver (`cbnpic`); modified
- `decks/custom_init.cxx` -- deck used by upstream's `tests/decks` smoke test
- `tests/CMakeLists.txt` (modified), `tests/decks/CMakeLists.txt`,
  `tests/energy_comparison/{CMakeLists.txt,2stream-em.cxx,compare_energies.h,energies_gold.2stream-em.double,energies_gold.2stream-em.float}` -- upstream's ctest suite (used by `validate.sh`)

Left out: `decks/2particle.cxx`, `decks/dioctron_3d.cxx` (uses `rand()` inside a
`KOKKOS_LAMBDA`, not device-safe), `decks/vpic/` (VPIC-format decks, unused),
`decks/2stream-short.cxx` (the deck shown in upstream's README: it sets nx=32,
ny=1 but relies on the default particle initialiser, which hard-codes cell
indices for nx=1 -- the particles land out of range and the energies go NaN
after ~800 steps; verified here), `tests/example.cpp`, `tests/include/catch.hpp`
(a Catch2 header that no built test uses), `tests/manual_tests/`, `scripts/`
(upstream's own Kokkos-profiling scripts), `.github/`, `summary.md`,
`.codecov.yml`, `.gitmodules`, upstream `README.md`.

Added here: `decks/hpcperf_weibel.cxx`, `build.sh`, `run.sh`, `validate.sh`, this `README.md`.

## Changes from upstream

Fixes needed to configure/build out of tree against Cabana 0.8.0 / Kokkos 5.2.1:

- `src/CMakeLists.txt`: link `Cabana::Core` instead of `Cabana::cabanacore`.
  Cabana renamed its exported core target in 0.6; the old name does not exist
  in 0.8.0 (configure error "target ... Cabana::cabanacore not found").
- `CMakeLists.txt` + `example/CMakeLists.txt`: the user deck (`-DINPUT_DECK=`)
  is now resolved to an absolute path (`INPUT_DECK_PATH`: source dir, then
  build dir, then as given) in the top-level file and that path is handed to
  `add_executable(cbnpic ...)`. Upstream passed the raw relative path into
  `example/`, where CMake resolves it against `example/` itself, so only an
  in-source build with `-DINPUT_DECK=../decks/x.cxx` ever worked; any
  out-of-tree build failed with "Cannot find source file: decks/x.cxx".

Changes made for the benchmark (all default to upstream behaviour):

- `example/example.cpp`: new compile-time macro `PARTICLE_DUMP_INTERVAL`
  (default 1 = upstream) guarding the per-step particle and field dumps
  (`dump_particles` -> `partloc`, `dump_fields` -> `ex1d`, plus the step-0
  particle dump). Upstream copies *every particle* to the host and prints it
  after every step; for the 1M-particle standard run that is ~1 GB of text
  per step. `build.sh` sets it to 0 (no dumps); 1 reproduces upstream
  byte-for-byte, N dumps every N steps. `energies.txt` (the validated
  quantity) is unaffected. Mirrors upstream's existing
  `ENERGY_DUMP_INTERVAL` macro.
- `CMakeLists.txt`: cache variable `PARTICLE_DUMP_INTERVAL` (default 1) that
  passes the macro above via `add_definitions`, like upstream does for
  `REAL_TYPE`.
- `decks/hpcperf_weibel.cxx` (new file): the upstream Weibel deck (identical
  parameters to the built-in default deck in `src/input/deck.h` and to the
  test deck `tests/energy_comparison/2stream-em.cxx`) with `ny`, `nppc` and
  `num_steps` overridable at run time via `HPCPERF_NY`, `HPCPERF_NPPC`,
  `HPCPERF_NUM_STEPS`. Without them it is exactly the upstream problem
  (32 / 100 / 6000). Decks are compiled in, so this is the only way to get a
  GPU-sized run and a smoke run from one build without touching upstream code.

Warning fixes (no semantic change):

- `src/uncenter_p.h`: the 18 `Cabana::slice<...>(f0)(ii)` calls inside the
  `KOKKOS_LAMBDA` were hoisted out of the lambda (`auto _ex = Cabana::slice<EX>(f0);`
  ... `_ex(ii)`), exactly the pattern `push.h` already uses and what upstream's
  own `// TODO: hoist slice call?` asks for. `Cabana::slice` is host-only;
  nvcc 13.2 emitted 18 "calling a __host__ function from a __host__ __device__
  function" warnings (#20011-D) and the code could not have run on the
  device. The function (`uncenter_particles`) is only called when a deck sets
  `perform_uncenter = true`, which none of the shipped decks do.
- `tests/CMakeLists.txt`: `remove_definitions(-DUSER_INPUT_DECK=...)` when
  `INPUT_DECK` is set. Each test compiles its own deck and sets
  `USER_INPUT_DECK` itself, so with `-DINPUT_DECK` *and* `-DENABLE_TESTS=ON`
  every test translation unit got the macro twice and the preprocessor
  printed `"USER_INPUT_DECK" redefined` (~10 times per build). Harmless, but
  noisy; the macro is only ever tested with `#ifdef`.

Upstream behaviours worth knowing (deliberately *not* changed):

- Upstream's `CMAKE_CXX_STANDARD 14` is overridden to C++20 by Kokkos'
  exported `cxx_std_20` compile feature (`-std=c++20` on every nvcc line);
  no `-DCMAKE_CXX_STANDARD` is needed and none is passed.
- `REAL_TYPE` (the `real_t` typedef) defaults to `float` in upstream's CMake,
  but upstream's CI validates with `-DREAL_TYPE=double` and only the `double`
  gold data passes on the GPU (see Validation). `build.sh` therefore builds
  `double` by default (`HPCPERF_REAL_TYPE=float` to override).
- `dump_energies` opens, appends to and closes `energies.txt` in the working
  directory on **every** step (`ENERGY_DUMP_INTERVAL`, default 1). On this
  machine's NFS home/project filesystem that costs 2-3 ms per step, i.e. the
  upstream 6000-step problem takes 18.7 s on NFS vs 1.7 s on local disk, and
  the standard run would take ~290 s instead of ~65 s. `run.sh`/`validate.sh`
  therefore run in a fresh directory under `${TMPDIR:-/tmp}` and copy the
  results back (override with `HPCPERF_CABANAPIC_RUNDIR`). The interval was
  left at 1 because the energy test needs one line per step. Note also that
  the file is never truncated (the truncating branch is for step 0, but the
  loop starts at step 1), so a re-run in the same directory appends to the
  old file; the scripts delete stale `energies.txt`/`partloc`/`ex1d` first.
  `partloc` and `ex1d` are still created (empty) with
  `PARTICLE_DUMP_INTERVAL=0`, because the `fopen` calls were left untouched.
- CabanaPIC makes no MPI calls (`MPI_Init` is never called); `libmpi` is only
  linked in because Cabana 0.8.0 was built with MPI and `Cabana::Core` carries
  `MPI::MPI_CXX`. The binaries are run directly, without `mpirun`.
- Kokkos also initialises its OpenMP host backend and prints its usual
  `OMP_PROC_BIND` advice plus "MPI detected ..." at start-up (the latter because
  the shell runs inside a SLURM allocation, `SLURM_*` variables are set).
  Everything executes on `Kokkos::Cuda`.
- When the energy test *fails*, its finaliser calls `std::exit(1)` while
  Kokkos is still initialised; static destruction then hits a
  `cudaErrorCudartUnloading` and the process aborts instead of exiting 1
  (ctest reports "Subprocess aborted"). Either way it is a failure; on a pass
  the shutdown is clean.

## Dependencies

- Kokkos 5.2.1 -- `$R/.deps/install/kokkos` (built by `setup_level2_deps.sh`:
  Serial + OpenMP + CUDA, `Kokkos_ARCH_BLACKWELL100`, C++20). The app is
  compiled with its `bin/nvcc_wrapper` (`NVCC_WRAPPER_DEFAULT_COMPILER` = conda
  g++ from `hpcperf_env.sh`).
- Cabana 0.8.0 -- `$R/.deps/install/cabana` (header-only; its CMake config
  pulls in Kokkos, MPI, HDF5 and heFFTe 2.4.1 from `$R/.deps/install/heffte`,
  all found through the `CMAKE_PREFIX_PATH` that `hpcperf_env.sh` exports).
- CUDA toolkit (nvcc) -- system CUDA 13.2 via `hpcperf_env.sh`.
- Host C++ compiler -- conda GCC 13.3.0.
- MPI -- conda OpenMPI 5.0.10, link-time only (see above).
- Tests: upstream's `tests/` need nothing beyond the above (no GTest; the
  shipped Catch2 header is unused and was not copied).
- HIP variant: `hipcc` (ROCm) plus HIP builds of Kokkos and Cabana in
  `$R/.deps/install/{kokkos-hip,cabana-hip}` -- none available on this machine.

## Build / Run / Validate

```bash
source hpcperf_env.sh                     # optional -- the scripts source it themselves
./level2/cabanapic/build.sh               # CUDA (default); ./build.sh HIP for the HIP variant
./level2/cabanapic/run.sh                 # Weibel, ny=512 nppc=2000 96000 steps (~65 s on B200)
./level2/cabanapic/validate.sh            # upstream 2stream-em energy test + custom_init smoke test
```

`build.sh [CUDA|HIP] [DECK] [-D...]` compiles the deck into the binary
(default `decks/hpcperf_weibel.cxx`; `HPCPERF_DECK` or the second argument
selects another: absolute path, `decks/custom_init.cxx`, or `custom_init`),
detects the GPU (`nvidia-smi --query-gpu=compute_cap`, override
`HPCPERF_CUDA_ARCH=100`) and checks it against the architecture baked into the
Kokkos install, and builds with `-DREAL_TYPE=double -DPARTICLE_DUMP_INTERVAL=0
-DENABLE_TESTS=ON` (`HPCPERF_REAL_TYPE`, `HPCPERF_PARTICLE_DUMP_INTERVAL`,
`HPCPERF_CABANAPIC_TESTS` override; extra `-D` options are forwarded).
Outputs: `build/level2/cabanapic/cuda/example/cbnpic`,
`.../tests/energy_comparison/2stream-em`, `.../tests/decks/custom_init`.

`run.sh` exports `HPCPERF_NY=512 HPCPERF_NPPC=2000 HPCPERF_NUM_STEPS=96000`
(unless already set), runs `cbnpic` in a temporary local directory, prints the
run header, the last energies line and the wall time, and copies
`energies.txt` and the full log (one energies line per step) to
`build/level2/cabanapic/cuda/run/`.

Equivalent raw commands (CUDA):

```bash
source hpcperf_env.sh
K=$PWD/.deps/install/kokkos; C=$PWD/.deps/install/cabana
cmake -S level2/cabanapic -B build/level2/cabanapic/cuda -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=$K/bin/nvcc_wrapper -DCMAKE_PREFIX_PATH="$K;$C" \
      -DINPUT_DECK=$PWD/level2/cabanapic/decks/hpcperf_weibel.cxx \
      -DREAL_TYPE=double -DPARTICLE_DUMP_INTERVAL=0 -DENABLE_TESTS=ON
cmake --build build/level2/cabanapic/cuda -j4
mkdir -p /tmp/cbnpic && cd /tmp/cbnpic && rm -f energies.txt
HPCPERF_NY=512 HPCPERF_NPPC=2000 HPCPERF_NUM_STEPS=96000 \
    $OLDPWD/build/level2/cabanapic/cuda/example/cbnpic           # standard run
rm -f energies.txt && $OLDPWD/build/level2/cabanapic/cuda/tests/energy_comparison/2stream-em   # energy test
# or: ctest --test-dir build/level2/cabanapic/cuda   (runs both tests inside the NFS build tree, ~16 s)
```

HIP (build configuration present, unverified without ROCm):

```bash
K=$PWD/.deps/install/kokkos-hip; C=$PWD/.deps/install/cabana-hip
cmake -S level2/cabanapic -B build/level2/cabanapic/hip -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_PREFIX_PATH="$K;$C" \
      -DINPUT_DECK=$PWD/level2/cabanapic/decks/hpcperf_weibel.cxx \
      -DREAL_TYPE=double -DPARTICLE_DUMP_INTERVAL=0 -DENABLE_TESTS=ON
cmake --build build/level2/cabanapic/hip -j4
```

### The standard problem

Upstream's benchmark problem is its built-in default deck: the Weibel /
filamentation instability of two counter-streaming electron beams
(v0 = 0.0866 c along +/-x, drift perturbed along y), EM solver, periodic
1 x 32 x 1 grid with Ly = 0.6283 gamma^1.5, n0 = 2, nppc = 100, 6000 steps of
dt = 0.99 x CFL -- 3200 particles, which is far too small for a GPU (the
whole run takes 1.7 s, dominated by launch latency). `run.sh` keeps every
physical parameter and changes exactly three numbers:

| | upstream default | `run.sh` standard |
|---|---|---|
| `ny` (cells) | 32 | 512 (16x finer grid) |
| `nppc` | 100 | 2000 |
| particles | 3,200 | 1,024,000 |
| `num_steps` | 6000 | 96000 |
| simulated time | 117.3 / omega_pe | 117.3 / omega_pe (dt shrinks 16x with the grid) |
| B200 wall time | ~2 s | 62-72 s (two runs) |

The scaled run reproduces the instability end to end (growth from B-energy
1e-14 to saturation at t ~ 90 / omega_pe, final `energies.txt` line
`96000 117.292 8.73e-04 4.05e-02`). Smaller/larger runs: e.g.
`HPCPERF_NY=256 HPCPERF_NPPC=4000 HPCPERF_NUM_STEPS=48000 ./run.sh` (~40 s);
the upstream problem itself is `HPCPERF_NY=32 HPCPERF_NPPC=100 HPCPERF_NUM_STEPS=6000`.
Only `ny` can be scaled -- the default particle initialiser hard-codes the
cell layout for nx = nz = 1. Measured throughput on the B200 (double) is
0.8-1.6 G particle-steps/s for 0.5-5 M particles (the standard run: 1.02 M
particles x 96000 steps in 62-72 s = 1.4-1.6 G/s).

## Validation

`validate.sh` runs the two programs of upstream's ctest suite (built with
`ENABLE_TESTS=ON`; `ctest --test-dir build/level2/cabanapic/cuda` runs the same
two, but inside the NFS build tree):

1. `tests/energy_comparison/2stream-em` -- upstream's regression test:
   `example.cpp` compiled with the Weibel deck (nx=1, ny=32, nz=1, nppc=100,
   6000 steps, exactly the upstream default problem) plus a `Custom_Finalizer`
   that reads back the `energies.txt` the run wrote (columns: step, time,
   E-field energy, B-field energy) and compares it with
   `energies_gold.2stream-em.double` (upstream's CPU reference for
   `REAL_TYPE=double`; a `.float` file exists too). It skips the first 3581
   lines and compares the next 1300 (steps 3581-4880, t = 70-95 / omega_pe,
   the growth-to-saturation phase) line by line: the E energy and,
   separately, the B energy must be within 10 % relative error
   (`|a-b|/min(a,b)`, absolute error when a value is ~0;
   `compare_energies.h`, `error_margin = 0.10`) on every line. It prints
   `E Test Pass: 1` / `B Test Pass: 1` and exits 1 if either fails.
2. `tests/decks/custom_init` -- upstream's 30-step smoke run of
   `decks/custom_init.cxx` (1-D two-stream along x with a custom initialiser);
   exit status only, as in upstream's ctest.

PASS requires both exit codes 0 and both `Test Pass: 1` lines. Observed here
(CUDA, B200, `REAL_TYPE=double`):

```
Max found err was 0.000499303% (1.00139e-06 vs 1.0014e-06) on line 3771 (Threshold: 10%)
E Test Pass: 1
Max found err was 0.000492334% (0.00101557 vs 0.00101557) on line 4033 (Threshold: 10%)
B Test Pass: 1
CabanaPIC CUDA validation (2stream-em energies vs upstream gold, REAL_TYPE=double; custom_init smoke): PASS
```

i.e. the GPU run matches upstream's CPU reference to 5e-4 % over the whole
window (6.5 s wall for `validate.sh`; ~16 s via `ctest` in the NFS build tree,
which also leaves `energies.txt`/`partloc`/`ex1d` behind in the test
directories).
With `REAL_TYPE=float` (upstream's CMake default) the same test **fails** on
the B200: `Max found err was 12.8158% (0.00010684 vs 0.000120532) on line 4778`,
`E Test Pass: 0` (B passes at 0.78 %) -- single precision plus a different
summation order in a chaotic instability drifts past the 10 % margin, which is
presumably why upstream's own CI validates with double.

## Warnings

Compiler (nvcc 13.2 + GCC 13.3, `-O3 -std=c++20 -arch=sm_100`, 6 translation
units incl. the two tests), after the fixes above: 20 warning lines of two kinds:

- 6x `src/helpers.h(53): warning #550-D: variable "iy"/"iz" was set but never used`
  -- `dump_particles` unpacks `ix,iy,iz` with `RANK_TO_INDEX` and only uses
  `ix` (1-D dump). Upstream code, harmless; left as is (2 warnings x 3 TUs that
  include `example.cpp`).
- 11x `nvcc_wrapper warning: you have set multiple optimization flags (-O*),
  only the last is used` -- conda's `CXXFLAGS` carries `-O2` and CMake's
  Release adds `-O3`; nvcc keeps `-O3`. Environment-wide, same for every
  Kokkos app in this repo.

Removed by the fixes: 18x `#20011-D calling a __host__ function from a
__host__ __device__ function` (`uncenter_p.h`) and ~10x
`<command-line>: warning: "USER_INPUT_DECK" redefined` (tests).

CMake: one warning from the Kokkos config (`The installed Kokkos configuration
does not support CXX extensions. Forcing -DCMAKE_CXX_EXTENSIONS=Off`),
environment-wide.

## LOC

cloc v2.06, code lines only (blank/comment excluded); CMake files, gold data,
tests, scripts and this README excluded. CabanaPIC is a Kokkos application:
`src/` + `example/` is both the CUDA and the HIP variant (the backend is chosen
by the Kokkos/Cabana it is linked against), so the count is the same for both.

CUDA variant = HIP variant: **2038** (16 files) = `src/` 1812 (2 .cpp + 12 .h) +
`example/example.cpp` 187 + `decks/hpcperf_weibel.cxx` 39 (the compiled-in
standard deck). Upstream's `src/` + `example/` count 1971; the +67 are the new
deck (+39), the hoisted slices in `uncenter_p.h` (76 -> 94) and the dump guard
in `example.cpp` (177 -> 187).

Not counted: `decks/custom_init.cxx` 82 and `tests/` (283 C++: test deck,
finaliser and comparison header), used only by `validate.sh`.

```bash
cloc --quiet level2/cabanapic/src level2/cabanapic/example level2/cabanapic/decks/hpcperf_weibel.cxx --exclude-lang=CMake
```

## Verified Environment

GCC/G++ 13.3.0 (conda, pinned) | C++20 | CMake 3.28.4 | Ninja 1.13.2 | Python 3.12.3
CUDA Toolkit 13.2 (nvcc 13.2.78, /usr/local/cuda) | NVIDIA B200 (sm_100), driver 595.58.03
OpenMPI 5.0.10 (conda) -- linked through Cabana, not called
Kokkos 5.2.1 + Cabana 0.8.0 (+ heFFTe 2.4.1) from `.deps/install` (`setup_level2_deps.sh`)
HIP/ROCm: source + build config present where noted, unverified (no AMD GPU available)
OS: RHEL 10 (Linux 6.12)

Reproduce the toolchain from the repository root: `./setup_env.sh` then
`source hpcperf_env.sh` (all user-space versions are pinned in environment.yml).

## Status

CUDA: **Working** -- configure + build (59 s, `-j4`, 20 benign warning lines) +
run (`run.sh`, ny=512/nppc=2000/96000 steps, 62-72 s, exit 0, instability
saturates as upstream's does) + validate (`validate.sh`: 2stream-em E and B
energies within 5e-4 % of upstream's CPU gold data, custom_init smoke test
exit 0 -> PASS) all succeeded on this machine from a clean shell and a foreign
cwd.
HIP: build form present, untested (no ROCm/hipcc and no HIP Kokkos/Cabana
installs on the development machine; `./build.sh HIP` exits with "hipcc was
not found").
Maintainer decisions to confirm: the three size parameters of the standard
run (`decks/hpcperf_weibel.cxx` defaults are upstream's; `run.sh` sets
512/2000/96000), the `PARTICLE_DUMP_INTERVAL=0` and `REAL_TYPE=double` build
defaults, and running from a local temporary directory instead of the build
tree.
