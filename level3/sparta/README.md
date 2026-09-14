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
