# LAMMPS (Level 3)

Full classical molecular dynamics (neighbor lists, short- and long-range
forces, spatial decomposition, MPI halo exchange) -- run as the complete
application through its own input scripts, KOKKOS package on the GPU.

## Provenance

- Official repository: https://github.com/lammps/lammps (docs
  https://docs.lammps.org/, Kokkos: https://docs.lammps.org/Speed_kokkos.html)
- Release policy: `stable_*` tags with `_updateN` bug-fix updates; `patch_*`
  are feature releases (GitHub marks them pre-release).
- Selected: **`stable_22Jul2025_update6`** (released 2026-09-03), commit
  `9c5ab448c78a14fd534619622162ba418d6a1fb1`, fetched by `fetch.sh` into
  `_upstream/level3/lammps` (shallow, read-only).
- License: GPL-2.0 (`LICENSE`).
- Application-owned LOC (cloc 2.06, code lines): `src/` **852,527** in
  3,865 files = 743,761 outside `src/KOKKOS` + 108,766 in `src/KOKKOS`
  (the GPU package). Bundled `lib/kokkos` (Kokkos 4.6.2) is counted
  separately and not modified.

## Build strategy: NATIVE (upstream CMake + bundled Kokkos 4.6.2)

`build.sh CUDA` = LAMMPS' documented Kokkos/CUDA recipe:
`nvcc_wrapper` (host compiler conda GCC 13.3.0) as CXX, `PKG_KOKKOS`,
`Kokkos_ENABLE_CUDA`, `Kokkos_ARCH_BLACKWELL100` (sm_100; the bundled
Kokkos 4.6.2 supports it), `Kokkos_ENABLE_OPENMP/SERIAL`, `FFT_KOKKOS=CUFFT`,
`FFT=KISS` (host), `BUILD_MPI` (conda Open MPI 5.0.10, CUDA-aware), C++17,
packages `MOLECULE KSPACE MANYBODY RIGID GRANULAR` (what `bench/` needs),
`WITH_JPEG=no WITH_PNG=no` (no `jpeglib.h` on the node; image dumps unused).
Build time on dgx003: **220 s** at `-j32` (708 targets). Warnings: 663
lines, essentially all `nvcc_wrapper: multiple optimization flags` (conda
`-O2` + Release `-O3`) plus two upstream unused-variable notes (`#550-D`,
`#177-D`); no errors. Install prefix `.deps/level3/lammps/install`
(fingerprinted: upstream commit, Kokkos 4.6.2, compiler, CUDA 13.2.78, MPI,
CMake options, GPU-aware setting).

Why not the others: upstream does not recommend Spack for GPU builds (the
Spack `lammps` package exists but the local Spack checkout is 2025-05 and
lacks `cuda_arch=100`); no Apptainer on the node and a container would not
provide the host MPI/transport; site modules are broken on dgx003. Level 2's
Kokkos 5.2.1 is **not** used: LAMMPS requires an external Kokkos
`>= 4.6.02` and pins 4.6.2 internally -- the bundled one is the supported
configuration.

HIP: `build.sh HIP` carries the upstream `Kokkos_ENABLE_HIP` +
`Kokkos_ARCH_AMD_GFX950` + `FFT_KOKKOS=HIPFFT` recipe and exits with a clear
message here (no ROCm). **Untested.**

## Changes from upstream

Class **A -- no source modification.** `run.sh` writes a *derived* copy of
`bench/in.lj` into the build tree with `run 100` -> `run ${steps}` and, in
weak mode, a `processors ${px} ${py} ${pz}` line; the upstream file is
untouched, and with the default 100 steps the derived deck is semantically
identical.

## Execution model

One MPI rank per GPU (upstream `Speed_kokkos`), `-k on g 1 t 1 -sf kk
-pk kokkos newton on neigh half gpu/aware on`. Ranks are launched by the
common launcher with the per-rank GPU wrapper, so each rank sees exactly one
GPU (`g 1`) and the launcher audits expected vs observed GPU. GPU-aware MPI
(`gpu/aware on`, LAMMPS default) is used with the CUDA-aware conda Open MPI;
`HPCPERF_LAMMPS_GPU_AWARE=off` selects host-staged communication. Any rank
count is legal: LAMMPS factors the box into a processor grid itself (weak mode
passes the grid explicitly). Threads per rank: 1 (`HPCPERF_CPUS_PER_RANK`
sets `t`), as upstream recommends for GPU runs.

## Inputs (`HPCPERF_SCALE_MODE`)

| Mode | Global box (fcc cells) | Atoms | Per rank @4 GPU | Topology | Steps | Memory/GPU (est.) | Runtime on B200 | Validation quantity |
|---|---|---|---|---|---|---|---|---|
| smoke (default) | 20^3 (upstream `in.lj`) | 32,000 | 8,000 | LAMMPS auto | 100 | < 0.1 GB | 0.02-0.09 s | thermo vs upstream reference log |
| strong | (20*S)^3, S=`HPCPERF_LAMMPS_STRONG` (8) = 160^3 | 16,384,000 | 4,096,000 | LAMMPS auto | 100 | ~2 GB | 1.1 s (1 GPU) / 2.3 s (4 GPU) | same thermo table |
| weak | (20*L*P)^3-shaped, L=`HPCPERF_LAMMPS_LOCAL` (4): 80^3 cells/rank | 2,048,000 x N | 2,048,000 | `hpcperf_topology.py` grid = `processors` | 100 | ~0.3 GB | 1.1 s (4 GPU) | same thermo table |

Memory estimate: ~120 B/atom for LJ with Kokkos neighbor lists (55
neighbors/atom, half list) -- well below the 180 GB of a B200 at every size
above; the strong default is deliberately a *correctness* size (100 steps run
in seconds), not a performance deck. The 4-GPU strong run (2.29 s) being
slower than 1 GPU (1.12 s) at 16.4M atoms/100 steps is the expected
communication-dominated behaviour of a fixed small workload and is **not** a
scaling result.

## Validation (`validate.sh`, upstream mechanism)

Thermo output (Temp, E_pair, TotEng, Press at steps 0 and 100) of the
unmodified `bench/in.lj` is compared with the reference log LAMMPS ships,
`bench/log.15Jul25.lj.fixed.g++.1` (CPU, 1 process; `velocity ... loop geom`
makes the initial state machine- and rank-count-independent). Tolerances:
1e-8 relative at step 0 (deterministic), 1e-5 at step 100 (reduction-order
divergence); with N > 1 GPUs the N-rank run is also compared with this build's
1-GPU run. Observed on dgx003 (2026-09-04): **all eight quantities identical
to the reference to every printed digit (rel 0.00e+00) at 1, 2 and 4 GPUs**.

## Results on dgx003 (4x B200, CUDA 13.2.78, Slurm job 9552083)

| Run | Ranks x GPUs | rank->GPU | CPU binding | Topology | Problem | Loop time | Validation |
|---|---|---|---|---|---|---|---|
| smoke | 1 x 1 | wrapper (1 visible GPU/rank); audit 1/1 verified | runtime default, `t 1` | 1x1x1 | 32k atoms, 100 steps | 0.020 s | PASS |
| smoke | 2 x 2 | wrapper; audit 1 verified / 1 unverified (0.07 s run, too short to sample) | runtime default | LAMMPS auto | 32k atoms | 0.070 s | PASS (vs ref and vs 1-GPU) |
| smoke | 4 x 4 | wrapper; audit 4/4 verified | runtime default | LAMMPS auto | 32k atoms | 0.085 s | PASS (vs ref and vs 1-GPU) |
| strong | 1 x 1 | wrapper | runtime default | 1x1x1 | 16.4M atoms | 1.118 s | run completes; thermo consistent |
| strong | 4 x 4 | wrapper; 4/4 verified | runtime default | LAMMPS auto | 16.4M atoms | 2.295 s | run completes |
| weak | 4 x 4 | wrapper; 4/4 verified | runtime default | 2x2x1 (`processors`) | 8.19M atoms (2.05M/rank) | 1.128 s | run completes |

Dry-runs (`HPCPERF_DRY_RUN=1`, hypothetical allocations) -- **DRY-RUN /
UNVALIDATED**, nothing executed:

| GPUs | Nodes x GPUs/node | Mode | Global box | Per rank | Ranks/node | Launch |
|---|---|---|---|---|---|---|
| 8 | 1 x 8 | strong | 160^3 cells = 16.4M atoms | 2.05M | 8 | `mpirun -np 8 --host dgx003:8 --map-by ppr:8:node ...` (single node) |
| 40 | 5 x 8 | weak | 400x320x160 = 81.9M atoms | 2.05M | 8 | `mpirun -np 40 --host <5 nodes>:8 --map-by ppr:8:node` -- multi-node BLOCKED on this site |
| 80 | 10 x 8 | weak | 400x320x320 = 163.8M atoms | 2.05M | 8 | `mpirun -np 80 ...` -- multi-node BLOCKED on this site |

## Limitations

- Multi-node: BLOCKED/UNVERIFIED on this site (transport); 40/80-GPU shapes
  are plans only.
- HIP: recipe present, untested (no AMD GPU).
- The benchmark family here is `bench/in.lj`; `in.eam`, `in.rhodo`
  (pppm/kk + cuFFT), `in.chain`, `in.chute` build with this package set but
  have no wrappers yet.
- Weak-mode `processors` grid comes from the generic balanced factorization;
  LAMMPS' own auto grid is used in smoke/strong.

## Agent-workspace closed loop (scheme 3 prototype, 2026-09-10)

Recorded in `provenance/agent_workspace_verification.yaml` (all step exit codes as expected): a workspace
created outside the repository from the materialized artifact (baseline check 17/17) -> a `#error` injected
into `src/src/KOKKOS/pair_lj_cut_kokkos.cpp` makes the workspace build FAIL (no binary) -> restore -> build
succeeds (90 s incremental) -> trusted validation on 1 GPU PASS -> a further agent edit is recompiled
(new binary sha256) and validated -> a tampered `validate.sh` or reference log is REFUSED by the trusted
harness before anything runs -> the canonical `level3/lammps` tree hash is unchanged. The workspace build
references neither `_upstream/` nor the canonical `src/`.

<!-- hpcperf:source-section:begin -->
## Source distribution (frozen source artifact, scheme 3, 2026-09-10)

The application source is not in git and not read from `_upstream/`: `tools/prepare_benchmark.sh level3 lammps` materializes the frozen source artifact (`<app>[-<variant>]-<source_version>.tar.zst`, found in the local content-addressed cache `.artifacts/sha256/` or downloaded from the immutable URL recorded in `provenance/source.lock*.yaml` once published; `--artifact FILE` for a local copy) into `src/` (+ `deps/`), the only source `build.sh`/`run.sh`/`validate.sh` use. Archive size + sha256 and `source_tree_sha256` are verified before anything is placed. Identity, patch series, licenses, redistribution status and the equivalence proof against the tree the results above were validated from are under `provenance/` (`source.lock*.yaml`, `patch_series*.txt`, `original_vs_baseline*.diff`, `LICENSES*.md`, `equivalence*.md`, `LOC*.md`); `optimization_scope.yaml` says what an agent may modify; `benchmark.yaml` is the machine-readable contract. Remote status: see `level3/SOURCE_ARTIFACTS.md`.

| variant | artifact | source version | compressed / uncompressed | entries | source_tree_sha256 | archive sha256 | upstream | patches (pre-applied) | redistribution | equivalence | remote | LOC app-owned / agent-modifiable / bundled deps / benchmark deps / tests / total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| - | `lammps-hpcperf-l3-v1.tar.zst` | hpcperf-l3-v1 | 110.0 MB / 435.9 MB | 13893 | `d7549c2f6d1b5b6c76575b7bf3b0f528cfa69b9b057aa0b3ef0f03921bdf8e48` | `4f6e1096de3a6671326c9aaabd40510ee4a610df961b605633f8cffd2c2cbca1` | stable_22Jul2025_update6 `9c5ab448c78a` | none | cleared | src: EQUIVALENT | REMOTE_ARTIFACT_UNPUBLISHED | 852527 / 852527 / 494133 / 0 / 124684 / 1471402 |

LOC = cloc 2.06 code lines of the materialized tree (no blank/comment lines, documentation and data excluded); categories from `optimization_scope.yaml` (`loc_categories`). The validated results recorded above were produced from trees proven content-equivalent to this artifact (`provenance/equivalence*.md`); they are not re-run by the migration.
<!-- hpcperf:source-section:end -->
