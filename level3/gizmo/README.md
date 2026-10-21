# GIZMO (gpu_kokkos_unified) -- meshless finite-mass hydrodynamics + gravity on the GPU

GIZMO (Hopkins 2015) is a Lagrangian multi-physics code: the meshless finite-mass (MFM) Riemann solver for the
gas, a Barnes-Hut / tree-PM gravity solver with adaptive softening, and the GADGET-style
domain decomposition, individual time steps and snapshot I/O (HDF5). The `gpu_kokkos_unified` branch is the
maintainer's active Kokkos port: the gravity tree walk, the neighbour-finding, density, gradient and hydro-force
loops and the cooling run as Kokkos kernels (CUDA here) on particle arrays held in a CUDA unified-memory arena;
the host keeps the time-stepping, domain decomposition and I/O.

## Provenance

- Upstream: https://github.com/pfhopkins/gizmo, branch `gpu_kokkos_unified`, commit
  `e9d2d0bea23b08872f747b780cdcedb5847a2bff` (2026-10-20; the branch has no release tags and is under active
  development -- the identity is this commit). License: GPL (README.md section 2; no separate license file).
- Frozen source bundle (scheme 3, `provenance/source.lock.yaml`): `src/` = the upstream tree without `docs/` and
  `analysis/`, with **two local patches** (`patches/`, `provenance/patch_series.txt`,
  `original_vs_baseline.diff`): `0001` `core/proto.h` -- `#include <cmath>` + `using std::isfinite/isnan/isinf`
  (glibc 2.39 with C++17/20 keeps them in `std::`) and the DEC-Alpha `pow()` prototype guarded against glibc
  (declared `noexcept` there; a second plain prototype is an error with GCC 13); `0002`
  `system/gpu_particles_arena.cc` -- the `cudaMemLocation` form of `cudaMemAdvise` under `CUDART_VERSION >= 13000`
  (CUDA 13 removed the int-device form; same meaning). Both class D (toolchain compatibility, no algorithm).
- `deps/ics/{evrard,sedov,noh,blob}_ics.hdf5` and `deps/exact/sedov_exact.txt`: the initial conditions and the
  Sedov reference solution upstream's test harness downloads (`test/harness/gizmo/test.py`,
  tapir.caltech.edu/~phopkins/sims), sha256-pinned `kind: file` components of the bundle (592 MB uncompressed,
  artifact `gizmo-hpcperf-l3-v1.tar.zst` 180 MB, unpublished until the maintainer publishes it);
  `tools/prepare_benchmark.sh level3 gizmo [--artifact FILE]` materializes everything.
- `fetch.sh` (freeze time only) clones the pinned commit into `_upstream/level3/gizmo`.

## Build strategy: NATIVE (upstream GNU Makefile, one build per registered Config.sh)

GIZMO's physics selection is the compile-time `Config.sh`, so every registered input is its own executable:
`build.sh [CUDA] [ID ...]` copies `src/src/` to `build/level3/gizmo/<profile>/<id>/src/` (upstream builds in place),
puts `inputs/<id>/Config.sh` there and runs `make` with upstream's "Vista" (NVIDIA GPU, `nvcc_wrapper`) recipe
expressed as command-line variables (`SYSTYPE=HPCPERF` matches no Makefile block, so the defaults are replaced):
conda GCC 13.3 through `mpicc` / `mpicxx -std=c++20` (Kokkos 5.2 needs C++20), `-O2 -fopenmp`, the repository's
Kokkos 5.2.1 install (`.deps/install/kokkos`, CUDA sm_100 + OpenMP backends; `nvcc_wrapper` with
`NVCC_WRAPPER_DEFAULT_COMPILER=mpicxx`, `--expt-relaxed-constexpr --expt-extended-lambda`), conda HDF5 and FFTW,
system zlib (`-idirafter /usr/include`: a plain `-I` shadows glibc's `math.h` for CCCL), `-L/usr/lib64`
(libcuda, libz), `PERL=/usr/bin/perl` (the conda environment ships none). Modification class A for the build.
Result: `.deps/level3/gizmo/<profile>/install/bin/GIZMO-<config>` per registered `inputs/<config>/Config.sh` (+ the Config.sh hash `run.sh` checks); inputs that register the same Config.sh with another parameter file share the executable.

## Execution model

One MPI rank per GPU through the common launcher (`--gpus N --bind wrapper`); the registered workloads are the
one-rank runs (`HPCPERF_GPUS=N` runs the same problem decomposed over N ranks: strong scaling, not validated
here). `run.sh` runs in `build/level3/gizmo/<profile>/<run subdir>/<id>.np<N>/`: it writes a copy of the
registered parameter file with only `InitCondFile` (the frozen IC, GIZMO appends `.hdf5`) and `OutputDir`
(`output/` of the run) rewritten and passes it as GIZMO's single argument; `run_manifest.txt` records the
Config.sh, parameter file, generated parameter file and IC with their sha256, TimeMax, the particle count,
the executable and its hash. GIZMO writes its snapshots, `cpu.txt` (the per-step wall-clock accounting the timer
reads) and logs under `output/`.

## Registered inputs (`inputs.yaml`, `HPCPERF_GIZMO_INPUT`)

| id | upstream test | particles | TimeMax (snapshots) | physics (Config.sh) | timed loop (B200) | correctness |
|---|---|---|---|---|---|---|
| evrard | test/evrard (Evrard 1988 cold-sphere collapse) | 27,001 | 0.8 (8) | MFM hydro + self-gravity, adaptive gas softening | RESULTS | radial density profile vs `evrard_exact.txt` (upstream criterion) |
| sedov (default) | test/sedov (Sedov-Taylor blast) | 2,097,152 | 0.03 (10) | MFM hydro, periodic box 6, no gravity | RESULTS | radial density profile vs `sedov_exact.txt` (upstream criterion) |
| sedov-t0.01 | derived: test/sedov to TimeMax 0.01 (the first third of the blast; same Config.sh, IC, parameters) | 2,097,152 | 0.01 (10) | MFM hydro, periodic box 6, no gravity | RESULTS | radial density profile vs `sedov_exact.txt` rescaled self-similarly from t = 0.03 to 0.01 (upstream criterion) |
| noh | test/noh (Noh implosion) | 2,097,152 | 1 (10) | MFM hydro, periodic box 6, no gravity | RESULTS | post-shock density vs the analytic 64 (upstream criterion) |
| blob | test/blob (cloud-crushing blob) | 9,641,651 | 2 (4) | MFM hydro, periodic 1 x 1 x 3 box, no gravity | RESULTS | mass conservation + blob survival (upstream criterion) |

Four inputs are upstream's test problems as shipped (`Config.sh`, `.params`, the IC its harness downloads); only
the two run-location lines of the parameter file are rewritten at run time. `sedov-t0.01` is the one derived
input: upstream's Sedov parameter file with `TimeMax` 0.03 -> 0.01 and `TimeBetSnapshot` 0.003 -> 0.001 (the
first 765 of the 1512 global steps of the full problem, ~13 min instead of ~52 min on a B200); the Sedov-Taylor
blast is self-similar, so its check is upstream's own criterion against `sedov_exact.txt` with the radii scaled
by (0.01/0.03)^(2/5). The remaining ~80 upstream tests are
CPU-sized (thousands of particles), use physics modules without a GPU path, or ship no acceptance criterion.

## Timed region (`tools/timing/apptimers.py: gizmo`)

GIZMO keeps cumulative wall-clock timers (`All.CPU_Sum[]`, `MPI_Wtime`; `core/main.cc:222-224` starts them
before `begrun`) and appends a block to `output/cpu.txt` after every step (`core/run.cc write_cpu_log`: `Step N,
Time: t, CPUs: n`, `total <seconds> 100.0%`, the per-phase lines). The timed region is `total` of the final step
block minus `total` of the `Step 0` block (written after the evaluation of the initial state): every
time-integration step -- domain decomposition, tree build and gravity, density, gradients, hydro forces,
kicks/drifts and the snapshots written in the loop -- without parameter and IC reading, the memory arenas, the
Step 0 evaluation and the shut-down. Steps = the step counters' difference. No NVTX range: the profiled run's
device picture is whole-process context.

## Validation (`validate.sh`, upstream mechanism)

Upstream's pytest harness (`src/test/harness`) builds, runs and judges every test problem; it needs h5py, scipy,
matplotlib and meshoid, absent here. `validate.sh` runs the same mechanism with the repository's tools on
upstream's Evrard problem (the registered `evrard` input) on one GPU and applies `test_evrard.py`'s criterion
through `check_profile.py` (h5dump + the Python standard library): final snapshot time = TimeMax, mean
|log10 rho_sim - log10 rho_exact| over 30 log-spaced radial bins (per-bin medians) against
`src/test/evrard/evrard_exact.txt` < 0.3; plus exit 0, the `Final time=... reached` line, the Kokkos CUDA device
line and a clean launcher audit. Result: RESULTS.

### Correctness of the registered inputs (`check_profile.py`)

Each input's `check` block runs the problem once more (`run.check.<id>`) and applies its upstream criterion:
`evrard` and `sedov` the radial density profile against the shipped reference solution (L1 in log density,
resp. relative L1 over the post-shock bins, both < 0.3 as upstream asserts), `noh` the median post-shock density
within 15 % of the analytic 64, `blob` total-mass conservation to 1e-3 and the blob's surviving mass fraction >
0.1; every check first asserts the final snapshot time. The stdout quantities of the registry (IC file with its
particle count, total particle count, TimeMax, the final-time line) pin the workload. These are upstream's
thresholds for its own problems (no tolerance invented here); they are coarse (30 %) by construction -- GIZMO
is a Lagrangian particle code with individual time steps, so two correct realisations differ at the particle
level -- and distinguish a correct run from a wrong hydro / gravity solution, not two correct roundings.
Three separate results per input: timing SUCCESS (the cpu.txt record of a clean run), workload verification PASS
(`verify_registry_runs.py`: manifest Config.sh / IC / parameter hashes, the generated parameter file, the stdout
templates, the launcher audit) and scientific correctness PASS (`hpcperf_inputs.py check` + `verdict`).

## Results on dgx003 (B200, CUDA 13.2.78)

RESULTS_PLACEHOLDER

## Limitations

- The Evrard problem (27 k particles) is latency-bound on a B200 (~0.7 s per step, 1240 steps): it is kept as the
  self-gravity workload and the problem the GPU path was validated with, not as a throughput measurement.
- One rank per registered input; multi-rank runs (`HPCPERF_GPUS=N`) are not validated here; multi-node
  BLOCKED/UNVERIFIED; no HIP recipe validated (upstream's Frontier block exists, no ROCm here).
- The branch is a moving development line (daily commits): the frozen commit is a snapshot; the two patches may
  become unnecessary upstream.
- GIZMO's memory arena lives in CUDA unified memory (upstream design); the host-side steps (domain decomposition,
  tree build on the host for part of the tree, snapshot I/O) are inside the timed region as upstream times them.

## Source distribution

Scheme 3 (`level3/EXTERNAL_ARTIFACT_DESIGN.md`): `provenance/source.lock.yaml`, artifact
`gizmo-hpcperf-l3-v1.tar.zst` in the maintainer's local staging, `publish_status: unpublished`. LOC:
`provenance/LOC.md`.
