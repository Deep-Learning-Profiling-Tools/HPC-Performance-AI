# SPARTA (Level 3)

Direct Simulation Monte Carlo (DSMC) for rarefied gas dynamics: particle
move/sort, collisions, grid-cell decomposition, MPI migration -- the full
application driven by its own input scripts, KOKKOS package on the GPU.

## Provenance

- Official repository: https://github.com/sparta/sparta (docs
  https://sparta.github.io/doc/Manual.html; Kokkos section
  https://sparta.github.io/doc/Section_accelerate.html)
- Release policy: one stream of dated tags (no stable/feature split).
- Selected: **`27Aug2026`** (2026-08-28), commit
  `95b9abaa8bd548991cc3c3f1c58b34722f7ade74`, fetched by `fetch.sh` into
  `_upstream/level3/sparta` (shallow, read-only). This release moved the
  bundled Kokkos to 5.0.2 and made KOKKOS builds CMake-only / C++20.
- License: GPL-2.0 (`LICENSE`).
- Application-owned LOC (cloc 2.06, code lines): `src/` **131,181** (C++
  104,683; headers 25,195; incl. `src/KOKKOS` 36,909 in 194 files). Bundled
  `lib/kokkos` (Kokkos 5.0.2, 223,495 lines) counted separately, not modified.

## Build strategy: NATIVE (upstream CMake preset + bundled Kokkos 5.0.2)

`build.sh CUDA` = the documented recipe: `cmake -S sparta/cmake -C
cmake/presets/kokkos_common.cmake` with `nvcc_wrapper` (host conda GCC 13.3.0)
as CXX, `Kokkos_ENABLE_CUDA`, `Kokkos_ARCH_BLACKWELL100` (the docs list "GB200
(Blackwell) -> BLACKWELL100" explicitly), `Kokkos_ENABLE_SERIAL=ON`,
`Kokkos_ENABLE_OPENMP=OFF`, `FFT_KOKKOS=CUFFT`, C++20, `BUILD_MPI` (conda Open
MPI 5.0.10, CUDA-aware), `SPARTA_MACHINE=kokkos_cuda`. Build time on dgx003:
**579 s** at `-j32` (278 targets); 108 warning lines (nvcc_wrapper multiple
`-O` flags, a few upstream notes), no errors. Executable
`build/level3/sparta/cuda/src/spa_kokkos_cuda` (302 MB, static Kokkos);
install prefix `.deps/level3/sparta/install` with fingerprint (upstream
commit, Kokkos 5.0.2, compiler, CUDA 13.2.78, MPI, CMake options).

Layout since 2026-09-15 (backend/profile isolation): profile `cuda` (or `hip`; override `HPCPERF_SPARTA_PROFILE`,
must name the backend); build tree `build/level3/sparta/<profile>/`, install/logs
`.deps/level3/sparta/<profile>/{install,logs}` with the fingerprint in the profile's install; results under
`build/level3/sparta/<profile>/run*/`. The results recorded above were produced with the pre-profile layout
(`.deps/level3/sparta/install`, `build/level3/sparta/cuda`), which is kept as historical state and is never read by
the current scripts.

Why not the others: **there is no Spack package for this SPARTA** -- the
`sparta` recipe in Spack (local and upstream) is the unrelated bioinformatics
tool sPARTA; upstream documents only CMake presets. No Apptainer on the node
and no upstream image. Site modules broken. Level 2's Kokkos 5.2.1 is not used
because the bundled 5.0.2 is the version the release was tested with (SPARTA
does not pin an external Kokkos, so `USE_EXTERNAL_KOKKOS=ON` remains a
documented fallback).

HIP: `build.sh HIP` carries the upstream `kokkos_hip` recipe (`hipcc`,
`Kokkos_ARCH_AMD_GFX950`, `FFT_KOKKOS=HIPFFT`) and exits with a clear message
here (no ROCm). Note the bundled Kokkos 5.0.2 has no `AMD_GFX950`
architecture (added in Kokkos 5.1); an MI355X build would need
`USE_EXTERNAL_KOKKOS`. **Untested.**

## Changes from upstream

Class **A -- none.** `bench/in.collide` is run unmodified from `bench/` with
its documented `-var x y z` size variables; the log goes to the build tree.

## Execution model

One MPI rank per GPU (upstream: "the -np setting ... should set the number of
MPI tasks/node to be equal to the # of physical GPUs on the node"), `-k on g 1
-sf kk -pk kokkos gpu/aware yes`. The common launcher's per-rank wrapper gives
each rank exactly one visible GPU and audits the mapping. GPU-aware MPI
defaults to `yes` -- unlike LAMMPS, SPARTA does **not** auto-detect
CUDA-awareness, so `HPCPERF_SPARTA_GPU_AWARE=no` must be used with a
non-CUDA-aware MPI. Any rank count is legal: the deck uses `balance_grid rcb
part` (recursive coordinate bisection), no processor-grid constraint.

## Inputs (`HPCPERF_SCALE_MODE`)

| Mode | Grid cells | Particles (10/cell) | Per rank @4 GPU | Topology | Steps | Memory/GPU (est.) | Runtime on B200 | Validation quantity |
|---|---|---|---|---|---|---|---|---|
| smoke (default) | 10x10x10 (upstream default) | 10,000 | 2,500 | RCB | 30 + 100 | < 0.1 GB | 0.03-0.05 s | Np, temp, Natt vs upstream reference log |
| strong | S^3, S=`HPCPERF_SPARTA_STRONG` (100) | 10,000,000 | 2,500,000 | RCB | 30 + 100 | ~2 GB | 0.87 s (1 GPU) / 0.38 s (4 GPU) | same stats |
| weak | (L*PX)x(L*PY)x(L*PZ), L=`HPCPERF_SPARTA_LOCAL` (50) | 1,250,000 x N | 1,250,000 | grid from `hpcperf_topology.py` (RCB inside) | 30 + 100 | ~0.3 GB | 0.23 s (4 GPU) | same stats |

The deck runs 30 equilibration steps followed by the 100-step benchmark
(`run 30` / `run 100`, upstream). Memory estimate ~100 B/particle plus
per-cell data; all sizes above are far below a B200's 180 GB. The strong
default is a *correctness* size: 10M particles per 100 steps take under a
second, so 4-GPU vs 1-GPU timings (0.38 s vs 0.87 s) indicate the run is
already partially communication/launch bound and are not a scaling result.

## Registered inputs (`inputs.yaml`, `HPCPERF_SPARTA_INPUT`, 2026-09-21)

`HPCPERF_SPARTA_INPUT=<id> level3/sparta/run.sh CUDA` runs one of the three
upstream bench decks at one of the sizes upstream ships reference logs for
(`src/bench/README`: 10K = 10x10x10 cells, 100K = 20x20x25, 1M = 40x50x50,
10M = 100x100x100; particles = 10 x cells for the box decks). The id is refused
together with `HPCPERF_SCALE_MODE=strong|weak` or `HPCPERF_SPARTA_STRONG/LOCAL`;
the deck must be `in.collide`, `in.free` or `in.sphere` from the frozen tree
(never modified: sizes enter through the decks' own `-var x/y/z`), the log is
`log.input.<id>.np<N>.sparta` and the manifest records `input_id`/`deck`. Without
the variable run.sh behaves exactly as above (smoke/strong/weak on `in.collide`).

| id | deck | grid | cells / particles | steps (equilibration + benchmark) | upstream reference log |
|---|---|---|---|---|---|
| `collide-10k` (default) | `in.collide` | 10x10x10 | 1,000 / 10,000 | 30 + 100 | `log.7Jul14.collide.icc.10K.1` |
| `collide-100k` | `in.collide` | 20x20x25 | 10,000 / 100,000 | 30 + 100 | `log.7Jul14.collide.icc.100K.1` |
| `collide-1m` | `in.collide` | 40x50x50 | 100,000 / 1,000,000 | 30 + 100 | `log.7Jul14.collide.icc.1M.1` |
| `collide-10m` | `in.collide` | 100x100x100 | 1,000,000 / 10,000,000 | 30 + 100 | `log.7Jul14.collide.icc.10M.1` |
| `free-1m` | `in.free` (no collisions) | 40x50x50 | 100,000 / 1,000,000 | 30 + 100 | `log.7Jul14.free.icc.1M.1` |
| `sphere-1m` | `in.sphere` + `data.sphere` (inflow, surface collisions, load balance) | 40x50x50 | 100,000 / ~1,000,000 at steady state | 1000 + 1000 | `log.7Jul14.sphere.icc.1M.1` |

Timer: SPARTA's own `Loop time` of the second `run` command (bench/README: "the
CPU time for the run is in the second Loop time line"), i.e. the benchmark
timestep loop only; the equilibration run, grid/particle creation, `balance_grid`,
surface read-in and the final statistics are outside it. Baseline (statistical,
DSMC is stochastic on the GPU): benchmark step count and final step exact,
particle count at the last stats row exact for the box decks (reflecting walls
conserve particles -- the validate.sh criterion), gas temperature within 2 % and
collision attempts within 15 % (both validate.sh tolerances, applied to the last
stats row); for `in.sphere` the particle count, surface-collision count and
max particles per cell are recorded only. The 15 % collision-attempt rule and the
2 % temperature rule are validate.sh's rules for `in.collide` (10,000 particles, mean
over the stats rows); they are applied as the registry rules of the box decks
(`in.collide`, `in.free`). For `in.sphere` no upstream or validate.sh basis exists, so
its collision-attempt count is recorded, not verified (it is not described as an
accepted rule). Roles: for the box decks every quantity is required and READY; for
`in.sphere` the steady-state particle count is the required result and still `record`,
the collision-attempt count is a required quantity without a rule basis (record), and
the surface-collision count and max particles per cell are diagnostics -> `compare`
exit 3 / INCOMPLETE for `sphere-1m` until rules with a basis exist. No seed, input,
boundary or threshold is changed to make the historical comparison pass. All styles of
the three decks have KOKKOS versions in this build.

Pilot calibration on dgx003 (1x B200, 1 warm-up + 3 measured runs; spread =
(max - min) / median; stable = spread <= 10 %):

| id | case (deck) | size | native timer field / scope | main compute median | per run | spread | run.sh wall (E2E) | main >= 1 s | stable |
|---|---|---|---|---|---|---|---|---|---|
| `collide-10k` | in.collide (default) | 10x10x10 cells, 10,000 particles | 2nd `Loop time` (100 benchmark steps after 30 equilibration steps) | 0.02669 s | 0.02669, 0.02663, 0.02673 | 0.39 % | 3.54 s | no | yes |
| `collide-100k` | in.collide | 20x20x25, 100,000 particles | 2nd `Loop time` (100 steps) | 0.03442 s | 0.03442, 0.03436, 0.03447 | 0.31 % | 3.48 s | no | yes |
| `collide-1m` | in.collide | 40x50x50, 1,000,000 particles | 2nd `Loop time` (100 steps) | 0.1098 s | 0.1097, 0.1098, 0.11 | 0.27 % | 4.29 s | no | yes |
| `collide-10m` | in.collide | 100x100x100, 10,000,000 particles | 2nd `Loop time` (100 steps) | 0.909 s | 0.9073, 0.9114, 0.909 | 0.45 % | 11.1 s | no | yes |
| `free-1m` | in.free | 40x50x50, 1,000,000 particles | 2nd `Loop time` (100 steps) | 0.08789 s | 0.08789, 0.08807, 0.08772 | 0.39 % | 4.68 s | no | yes |
| `sphere-1m` | in.sphere + data.sphere | 40x50x50, ~1,000,000 particles | 2nd `Loop time` (1000 benchmark steps after 1000 equilibration steps) | 1.594 s | 1.594, 1.591, 1.594 | 0.23 % | 7.89 s | yes | yes |

Comparison of every measured run with the upstream reference log of the same
size through the same rules (`measurements/level3-sparta/upstream-references/`;
the reference is a 1-process CPU run with icc, July 2014):

| id | particles at last step (exact) | temperature rel. diff (<= 2 %) | collision attempts rel. diff (<= 15 %, max of 3 runs) | verdict |
|---|---|---|---|---|
| `collide-10k` | identical | 3.7e-03 (275.42700 vs 274.40561) | 1.9e-02 | all rules pass in 3/3 runs |
| `collide-100k` | identical | 6.2e-03 (272.86255 vs 274.57169) | 6.8e-03 | all rules pass in 3/3 runs |
| `collide-1m` | identical | 1.2e-03 (272.96696 vs 273.28433) | 5.9e-03 | all rules pass in 3/3 runs |
| `collide-10m` | identical | 2.7e-05 (273.16399 vs 273.15661) | 9.7e-04 | all rules pass in 3/3 runs |
| `free-1m` | identical | 7.5e-04 (273.04732 vs 273.25108) | 0.0e+00 | all rules pass in 3/3 runs |
| `sphere-1m` | recorded: 990,933, 990,518, 990,092 vs upstream 999,920 (-0.9..-1.0 %) | n/a (not printed by in.sphere) | 9.9e-02 (recorded; no sphere rule basis) | step counts exact in 3/3 runs; np / natt / nscoll / c_max recorded only -> INCOMPLETE |

`sphere-1m` vs the 2014 reference log, facts checked (round 3): same seed
(12345), same grid (40x50x50), same `fnum` (7.33e+15), same timestep (1e-5),
same stats points (every 100 steps; last rows at step 1000 and 2000), same phase
structure (1000 equilibration + 1000 benchmark steps) and the same initial
population (`Created 955103 particles` in both). The 2014 deck (echoed in the
log) uses the commands of that version -- `fix in inflow air all`, `read_surf 1
data.sphere`, `surf_modify collide 1 1`, `compute g grid all n` -- where the
frozen 27Aug2026 deck uses `fix in emit/face air all`, `read_surf data.sphere`,
`surf_modify all collide 1`, `compute g grid all all n`; the current version
also prints `WARNING: One or more fix inflow faces oppose streaming velocity`
(fix_emit_face.cpp:210), absent in 2014. After the equilibration the particle
count is 990,824 here vs 1,000,977 in 2014 (-1.0 %), and 990,092-990,933 vs
999,920 at step 2000 (three runs here, mutually within 0.1 %; the 2014 log is a
single run). No statistical model has been validated for this comparison, so no
significance statement is made either way; the version/command differences above
are a candidate explanation, unconfirmed. It is a recorded quantity, not a
verified one, and no claim of correctness or of error is made for it. `collide-10m` (the grid of the
strong-scaling default) stays just under one second of loop time on a B200
(0.909 s); only `sphere-1m` exceeds it. Raw runs and baselines:
`HPC-Performance-AI-results/inputs-pilot-2026-09-21/measurements/level3-sparta/`;
`validate.sh CUDA` re-run after the change: PASS.

## Validation (`validate.sh`, upstream mechanism)

Upstream's own guidance (`examples/README`, `tools/testing/regression.py`)
is statistical: DSMC is stochastic and "should get statistically similar
answers ... on different numbers of processors, but not identical answers".
`validate.sh` therefore compares the stats table of the unmodified
`bench/in.collide` with the reference log SPARTA ships,
`bench/log.7Jul14.collide.icc.10K.1`, on three quantities:

1. particle count `Np` == 10,000 at every stats row (closed box, no
   chemistry: exact conservation);
2. gas temperature (`compute temp`, printed as `c_temp`; the 2014 log labels
   it `temp`): mean over the benchmark steps within 2 % of the reference.
   Elastic VSS collisions conserve energy exactly, so within a run the
   temperature is constant; its value is set by the Maxwellian sampling of the
   initial velocities, whose statistical scatter for 10^4 particles is
   sqrt(2/3N) ~ 0.8 %. 2 % is ~2.5 sigma of that noise and far below any
   unit/physics error;
3. mean collision attempts per step `Natt` within 15 % (fixed by density,
   temperature and cross-section; run-to-run scatter is a few %).

With N > 1 GPUs the same three criteria are applied between the N-rank run and
this build's 1-rank run. Observed on dgx003 (2026-09-04): **PASS at 1, 2 and 4
GPUs**:

| GPUs | Np | mean temp (ref 274.41 K) | rel | mean Natt (ref 943.7) | rel | vs 1-GPU temp / Natt |
|---|---|---|---|---|---|---|
| 1 | 10,000 at every row | 275.43 | 3.7e-3 | 946.1 | 2.5e-3 | -- |
| 2 | 10,000 | 271.65 | 1.0e-2 | 942.4 | 1.4e-3 | 1.4e-2 / 3.9e-3 |
| 4 | 10,000 | 274.64 | 8.5e-4 | 947.7 | 4.2e-3 | 2.9e-3 / 1.7e-3 |

(All temperature deviations are within ~1.7 sigma of the sampling noise; the
nominal gas temperature is 273.15 K.)

## Results on dgx003 (4x B200, CUDA 13.2.78, Slurm job 9552083)

| Run | Ranks x GPUs | rank->GPU | CPU binding | Topology | Problem | Loop time (100-step benchmark) | Validation |
|---|---|---|---|---|---|---|---|
| smoke | 1 x 1 | wrapper; audit unverified (0.03 s run, too short to sample) | runtime default | RCB | 10k particles | 0.026 s | PASS |
| smoke | 2 x 2 | wrapper; audit 1 verified / 1 unverified (short run) | runtime default | RCB | 10k particles | 0.041 s | PASS |
| smoke | 4 x 4 | wrapper; audit 4/4 verified | runtime default | RCB | 10k particles | 0.053 s | PASS |
| strong | 1 x 1 | wrapper; 1/1 verified | runtime default | RCB | 100^3 cells, 10M particles | 0.869 s | run completes, Np conserved |
| strong | 4 x 4 | wrapper; 4/4 verified | runtime default | RCB | 100^3 cells, 10M particles | 0.381 s | run completes |
| weak | 4 x 4 | wrapper; 4/4 verified | runtime default | 2x2x1 -> 100x100x50 | 5M particles (1.25M/rank) | 0.229 s | run completes |

Dry-runs (`HPCPERF_DRY_RUN=1`, hypothetical allocations) -- **DRY-RUN /
UNVALIDATED**, nothing executed:

| GPUs | Nodes x GPUs/node | Mode | Grid | Particles | Per rank | Launch |
|---|---|---|---|---|---|---|
| 8 | 1 x 8 | strong | 100^3 | 10M | 1.25M | `mpirun -np 8 --host dgx003:8 --map-by ppr:8:node ...` (single node) |
| 40 | 5 x 8 | weak | 250x200x100 | 50M | 1.25M | 5 nodes x 8 -- multi-node BLOCKED on this site |
| 80 | 10 x 8 | weak | 250x200x200 | 100M | 1.25M | 10 nodes x 8 -- multi-node BLOCKED on this site |

## Limitations

- Multi-node: BLOCKED/UNVERIFIED on this site; 40/80-GPU shapes are plans.
- HIP: recipe present, untested; bundled Kokkos lacks gfx950.
- Only `bench/in.collide` is wrapped; `in.free` and `in.sphere` (surface
  collisions, `fix balance`) build with this configuration but have no
  wrappers yet.
- Reference logs are from 2014 (Intel CPU) and only for 1 and 8 ranks; the
  comparison is statistical by design (upstream policy), not bitwise.

<!-- hpcperf:source-section:begin -->
## Source distribution (frozen source artifact, scheme 3, 2026-09-10)

The application source is not in git and not read from `_upstream/`: `tools/prepare_benchmark.sh level3 sparta` materializes the frozen source artifact (`<app>[-<variant>]-<source_version>.tar.zst`, found in the local content-addressed cache `.artifacts/sha256/` or downloaded from the immutable URL recorded in `provenance/source.lock*.yaml` once published; `--artifact FILE` for a local copy) into `src/` (+ `deps/`), the only source `build.sh`/`run.sh`/`validate.sh` use. Archive size + sha256 and `source_tree_sha256` are verified before anything is placed. Identity, patch series, licenses, redistribution status and the equivalence proof against the tree the results above were validated from are under `provenance/` (`source.lock*.yaml`, `patch_series*.txt`, `original_vs_baseline*.diff`, `LICENSES*.md`, `equivalence*.md`, `LOC*.md`); `benchmark.yaml` is the machine-readable contract (entries, inputs, references, identity). The benchmark does not prescribe which part of the source an optimization agent may modify; the integrity layer only protects the harness and the validation assets. Remote status: see `level3/SOURCE_ARTIFACTS.md`.

| variant | artifact | source version | compressed / uncompressed | entries | source_tree_sha256 | archive sha256 | upstream | patches (pre-applied) | redistribution | equivalence | remote | LOC app-owned / bundled deps / benchmark deps / tests / total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| - | `sparta-hpcperf-l3-v1.tar.zst` | hpcperf-l3-v1 | 20.3 MB / 72.5 MB | 3084 | `63519f0e3e9ac974e1f5aff64446b5fcb9ffc1f9c113901249efe7fa2666d36c` | `ea3be3032b3d6a41fa2130c08b0b3d49b1c7e223d87d45e93289d176130b3c09` | 27Aug2026 `95b9abaa8bd5` | none | cleared | src: EQUIVALENT | REMOTE_FETCH_VERIFIED | 131181 / 223529 / 0 / 0 / 354863 |

LOC = cloc 2.06 code lines of the materialized tree (no blank/comment lines, documentation and data excluded); source-ownership categories from `provenance/source.lock*.yaml` (`source_scope`, descriptive metadata written at freeze time). Dependencies are counted per benchmark, so totals overlap across benchmarks that ship the same dependency. The validated results recorded above were produced from trees proven content-equivalent to this artifact (`provenance/equivalence*.md`); they are not re-run by the migration.
<!-- hpcperf:source-section:end -->
