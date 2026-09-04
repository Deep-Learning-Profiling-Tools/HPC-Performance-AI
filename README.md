# HPC-Performance-AI

An End-to-End AI Framework for Performance Prediction and Optimization in HPC Applications

## Benchmark Levels

| Level | Content | Status |
|-------|---------|--------|
| [level1/](level1/) | 50 standalone GPU benchmarks (independently buildable / runnable / validatable) | 50/50 working, CUDA-validated |
| [level2/](level2/) | 20 proxy applications / mini-apps (upstream build systems kept, wrapped by `build.sh` / `run.sh` / `validate.sh`) | 19/20 working, CUDA-validated; MiniEM pending (Trilinos) |
| [level3/](level3/) | production / end-to-end HPC applications | planned |

Level 1 benchmarks are extracted (or faithfully ported) from six upstream
suites -- HeCBench, RAJAPerf, NPB-GPU, Hetero-Mark, Rodinia, Kokkos Kernels --
as correctness-preserving, standalone, reproducible programs. Each benchmark
records its upstream provenance (repository + commit), verified build/run
commands, validation method, and measured LOC in its own README. See
[level1/README.md](level1/README.md) for the full catalog.

Level 2 mini-apps (Kokkos, RAJA, MFEM, hypre, Cabana, OCCA, YAKL and plain
CUDA/HIP codes) are copied from their upstream repositories with their own
build systems and made to build against the same pinned toolchain plus a set
of framework libraries built inside the clone by `./setup_level2_deps.sh`.
See [level2/README.md](level2/README.md) for the catalog and the Level 2
environment additions.

## Quick Start (fresh clone)

```bash
git clone https://github.com/Deep-Learning-Profiling-Tools/HPC-Performance-AI.git
cd HPC-Performance-AI
./setup_env.sh          # one-time: installs the exact pinned toolchain (user-space only)
source hpcperf_env.sh   # every shell: loads the environment
./check_env.sh          # verify everything matches the validated configuration
```

Build, run, and validate a benchmark:

```bash
cmake -S level1/daxpy -B build/daxpy/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/daxpy/cuda
ctest --test-dir build/daxpy/cuda --output-on-failure
```

or everything at once from the repository root:

```bash
cmake -S . -B build/all -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/all -j
```

## Contributing

> **All collaborators must follow the contribution workflow: create a
> `<scope>/<short-description>` branch from the latest `main` and open a pull
> request -- do not push development work directly to `main`.**

See [CONTRIBUTING.md](CONTRIBUTING.md) for branch naming and pull request guidelines.

## Environment

The suite was validated with the following configuration. Everything in the
first table is **installed automatically by `./setup_env.sh`** -- versions are
pinned exactly in [environment.yml](environment.yml), so any clone reproduces
the same toolchain. Nothing is written outside the repository (no sudo, no
`conda init`, no changes to `~/.bashrc` or conda base); if no conda exists on
the machine, setup bootstraps a project-local Miniforge into `.tools/`.

| Tool (auto-installed, pinned) | Version |
|---|---|
| GCC / G++ (conda, NVCC host compiler) | 13.3.0 |
| C++ standard | C++20 |
| CMake | 3.28.4 |
| Ninja | 1.13.2 |
| Python | 3.12.3 |
| numpy / scipy (used by validation scripts) | 2.5.2 / 1.18.0 |
| cloc (downloaded into `.tools/bin`) | 2.06 |
| Open MPI (CUDA-aware; Level 2) | 5.0.10 |
| METIS / yaml-cpp / OpenBLAS (Level 2) | 5.1.0 / 0.8.0 / 0.3.34 |
| FFTW / parallel HDF5 / PnetCDF (openmpi builds; Level 2) | 3.3.11 / 2.2.0 / 1.15.0 |

Level 2 additionally needs the framework libraries built by
`./setup_level2_deps.sh` into `.deps/install/` (Kokkos + Kokkos Kernels 5.2.1,
Cabana 0.8.0, heFFTe 2.4.1, RAJA / Umpire / CHAI 2026.07.0, hypre 3.2.0,
MFEM 4.10 -- pinned tags, built for the detected GPU; see
[level2/README.md](level2/README.md)). Build-system patches applied to those
checkouts live in [patches/](patches/).

The GPU stack is a **system prerequisite** (deliberately not managed by conda);
install it once per machine before running `setup_env.sh`:

| System prerequisite | Validated with | Notes |
|---|---|---|
| NVIDIA driver | 595.58.03 | any driver supporting CUDA 13 should work |
| CUDA Toolkit | 13.2 (nvcc 13.2.78) | expected at `/usr/local/cuda`; otherwise `export CUDA_HOME=/path/to/cuda` before sourcing `hpcperf_env.sh`. Other CUDA 13.x are expected to work but are untested |
| GPU | NVIDIA B200 (sm_100) | no architecture is hardcoded -- CMake uses `native`, so any CUDA-13-supported GPU builds for its own arch |
| OS | RHEL 10.0 (kernel 6.12) | any recent Linux x86_64 should work |
| Nsight Compute (optional) | 2026.1.1 | only needed for profiling |
| ROCm / hipcc (optional) | none available | HIP sources and build configs exist for 40 benchmarks but are **unverified** (no AMD GPU on the development machine) |

`./check_env.sh` prints exactly how the current machine compares against this
validated configuration.

## Reproducibility Guarantee

A fresh `git clone` on any Linux x86_64 machine with the GPU prerequisites
above gets the **same configuration this suite was validated with**, via a
single command (`./setup_env.sh`). Concretely:

1. **Exact-version toolchain.** Every user-space tool is pinned to the exact
   validated version in [environment.yml](environment.yml) (GCC 13.3.0,
   CMake 3.28.4, Ninja 1.13.2, Python 3.12.3, numpy 2.5.2, scipy 1.18.0,
   pip 26.2.1, perl 5.32.1) plus cloc 2.06 pinned in `setup_env.sh` -- not
   "latest", not floating ranges.
2. **No assumptions about the machine.** If conda is absent, setup bootstraps
   a project-local Miniforge into `.tools/miniforge3`; cloc ships with its own
   pinned perl; nothing requires sudo, environment modules, or pre-installed
   build tools.
3. **Location-independent.** All scripts derive the repository root from their
   own location -- the clone can live at any path; `CUDA_HOME` is respected if
   the CUDA toolkit is not at `/usr/local/cuda`.
4. **Fully self-contained and removable.** Everything lands inside the clone
   (`.conda_env/`, `.tools/`); nothing touches `$HOME`, shell startup files,
   conda base, or the system. Deleting the clone removes every trace.
5. **Checkable.** `./check_env.sh` reports, item by item, how the current
   machine compares to the validated configuration.

This flow is verified end-to-end: a pristine copy of the repository in a fresh
directory ran `setup_env.sh` -> `hpcperf_env.sh` -> `check_env.sh` -> benchmark
build + ctest validation with no manual steps (2026-09-01, on the B200
development machine).

**Boundary:** the guarantee of identical versions covers the conda-managed
toolchain. The GPU stack (driver / CUDA toolkit / GPU model) is taken from the
host machine and is a prerequisite, not something this repository installs;
`check_env.sh` shows how the host's GPU stack differs from the validated one.

## What the scripts do

- `setup_env.sh` -- one-time, idempotent: finds or bootstraps conda, creates
  the project-local env `.conda_env/` from the pinned `environment.yml`,
  installs cloc, and compile+run-tests the GCC/NVCC pairing (recording the
  chosen host compiler in `.tools/compiler_source`).
- `hpcperf_env.sh` -- per-shell loader (must be `source`d): activates
  `.conda_env`, sets `CC`/`CXX`/`CUDAHOSTCXX`, puts the system CUDA on
  `PATH`/`LD_LIBRARY_PATH`, sets `CMAKE_GENERATOR=Ninja` and
  `CUDAARCHS=native`, puts `.deps/install/*` on `CMAKE_PREFIX_PATH`, and sets
  the single-node, CUDA-aware Open MPI defaults used by Level 2 (documented in
  [level2/README.md](level2/README.md)). Safe to source repeatedly; works from
  any clone path.
- `check_env.sh` -- read-only report of the active environment vs. the
  validated configuration above (including which Level 2 framework libraries
  are built).
- `setup_level2_deps.sh` -- Level 2 only, idempotent: clones the pinned
  framework libraries into `.deps/src/`, applies `patches/`, builds and
  installs them into `.deps/install/<name>` for the detected GPU architecture.
  Each install carries a `.hpcperf-fingerprint` (schema 2: profile, GPU
  arch, compiler, full CUDA toolkit version from `nvcc --version`, MPI, patch
  hashes); a recorded fingerprint that does not match the current toolchain
  fails fast instead of being reused silently. Installs predating fingerprints
  are accepted as legacy with a warning; `--stamp-existing` records a labelled
  post-hoc one, and `--migrate-fingerprints` re-records schema-1 files (which
  carried an empty `cuda=` field) with an explicit `migrated=` line -- both
  are records taken after the fact from the current environment, never a
  rebuild verification.
  `HPCPERF_DEPS_PROFILE=<name>` (default `level2`) selects an isolated
  `.deps/<name>/{build,install}` tree -- Level 3 uses its own profile and
  never modifies the validated Level 2 installs; `hpcperf_env.sh` exposes
  exactly one profile's prefixes and, when re-sourced with another profile,
  removes the dependency paths it added before (user/system entries untouched).
- `level2/tools/tests/run_all.sh` -- infrastructure regression tests that
  need no GPU execution (topology, dependency markers, launcher dry-run
  parsing, run.sh interface guards).

## Validation

Every Level 1 benchmark ships a correctness check, run via `ctest`: either the
upstream's built-in verification (e.g. NPB reference sums, HeCBench CPU
references) or an added independent reference (CPU re-computation, documented
per benchmark). "Working" in the catalog means configure + build + run +
validation all passed on the machine above.
