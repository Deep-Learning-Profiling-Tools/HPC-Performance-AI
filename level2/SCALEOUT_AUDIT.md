# Level 2 scale-out audit (CUDA)

Date: 2026-09-04. Scope: every Level 2 mini-app, audited from its `build.sh` /
`run.sh` / `validate.sh` / README, its sources under `level2/<app>/`, and the
upstream clone under `_upstream/level2/`. Goal: selectable GPU count
(2/4/8/40/80/`all`), one MPI rank per GPU, multi-node capability, and per-app
problem decomposition -- not the frozen single-GPU decks. HIP is out of scope
for this round (nothing HIP is claimed verified).

Definitions (execution type):
- **native-mpi** -- upstream implements a real distributed workload (domain
  decomposition / distributed solve with inter-rank communication).
- **naturally-shardable-but-not-implemented** -- one coupled global problem
  could be sharded, but the shipped code has no comm layer.
- **independent-replica-only** -- any multi-GPU form is N copies of the same
  problem (throughput mode). NEVER counted as distributed multi-GPU.
- **no-scaleout-path** -- not even replicas make sense.

## Executive matrix

| App | Execution type | rank->GPU binding | Multi-node capable | CUDA-aware MPI | Topology params | Rank-count constraints | Recommendation |
|---|---|---|---|---|---|---|---|
| amg2023 | native-mpi | in-app (hypre_bind_device, local rank) | yes | optional (hypre-gpuaware variant exists) | `-P px py pz` (product == NP) | any factorable N; global unknowns < 2^31 (<=127 ranks @256^3/rank) | **keep-in-core** |
| laghos | native-mpi | wrapper needed (`-dev` is one global ordinal) | yes | optional (`-gam`) | mesh path: internal (NC/METIS); `-epm` path: internal grid | any N <= elements (mesh path); any N (`-epm`) | **keep-in-core** |
| remhos | native-mpi | wrapper needed (no device-index option at all) | yes | optional (`-gam`) | like laghos (`-epm` supported) | any N <= elements; GPU solver combo rank-independent | **keep-in-core** |
| examinimd | native-mpi | in-app (Kokkos local-rank) | yes | **required** (device buffers to MPI_Send) | internal surface-minimizing grid | any N; <=~20M atoms/rank (int32 neigh list), <=2^31 global atoms | **keep-in-core** |
| exampm | native-mpi | in-app (Kokkos local-rank) | yes | **required** (Cabana halo/migration) | HARD-CODED 1xNx1 Y-slabs in examples | Y-cells/N >= halo(3): N <~ 35 @cell 0.01 (unchecked!) | **keep-in-core** (fix 1D slab before 80 ranks) |
| haccabanapm | native-mpi | in-app (Kokkos local-rank) | yes (parallel HDF5) | **required** (Cabana + heFFTe device buffers) | `PMKOKKOS_TOPOLOGY` / indat / MPI_Dims_create | any N (product check); ng%P warning only; pm_ic/pm_run same N | **keep-in-core** |
| cloverleaf | native-mpi | wrapper needed (`--device` one shared value) | yes | optional (`--staging-buffer auto`) | none (auto chunk factorization) | any N | **keep-in-core** |
| tealeaf | native-mpi | wrapper needed (same) | yes | optional (same) | none (auto chunk factorization) | any N; weak grids need new tea.problems rows | **keep-in-core** |
| kripke | native-mpi | wrapper needed (no binding code) | yes | optional (build flag exists) | `--procs x,y,z` (product == NP) | product == NP; zones divisible per dim; groups%gset, quad%dset | **keep-in-core** |
| branson | native-mpi (replicated work-split + Allreduce; PARTICLE_PASS = true domain decomposition, unvalidated) | in-app (`set_device_ID(rank%ndev)` -- GLOBAL rank: needs block mapping) | yes (caveat above) | no (host-buffer MPI only) | none (REPLICATED); deck `<mesh_decomposition>` for PARTICLE_PASS | any N | **keep-in-core** (replicated); PARTICLE_PASS = later extension |
| hipbone | native-mpi | in-app (hostname local-rank in OCCA props) | yes | optional (`-ga`) | `-px -py -pz` (product == NP; without them NP must be a cube) | any factorable N once -px/-py/-pz standardized in run.sh | **keep-in-core** (add -px/py/pz decks) |
| miniweather | native-mpi | wrapper needed (no binding code) | yes | optional (`-DGPU_AWARE_MPI` compile flag) | none (1D x-split only) | N <= nx_glob; problem size is COMPILE-TIME (`MINIWEATHER_NX`) | **keep-as-supplemental** (compile-time size, 1D-only split) |
| p3_heat3d | naturally-shardable; upstream sibling `heat3d_mpi` is native-mpi | sibling: in-app `cudaSetDevice(rank%n)` (global rank) | yes | **required** by sibling (device Views to MPI) | sibling: `--px --py --pz` (product == NP) | any factorable N; strong needs divisibility | **implement-distributed-extension** (adopt upstream heat3d_mpi) |
| p3_vlp4d | naturally-shardable; upstream sibling `vlp4d_mpi` is native-mpi | sibling: in-app (same) | yes | **required** by sibling (halo + device Allreduce) | none (internal 4D recursive bisection) | any N with >=10 points per cut direction | **implement-distributed-extension** (adopt vlp4d_mpi; NOTE: it switches Lagrange->spline interpolation -- a sibling benchmark, not the same numbers) |
| cabanapic | naturally-shardable-but-not-implemented (no comm layer at all; deck is 1D-in-y) | n/a (single process) | n/a | n/a | none | n/a | **keep-as-supplemental** (distributed PIC = rewrite; HACCabanaPM covers the Cabana-comm motif) |
| shaw | naturally-shardable-but-not-implemented (global CSR SpMV, no partitioned mesh) | n/a | n/a | n/a | none | n/a | **keep-as-supplemental** |
| exacmech | independent-replica-only (no inter-point coupling; real coupling lives in ExaConstit) | n/a | n/a | n/a | none | n/a | **keep-as-supplemental** |
| minibude | independent-replica-only (independent pose energies) | per-process `--device` flag | n/a | n/a | none | n/a | **keep-as-supplemental** |
| xsbench | independent-replica-only (upstream's own MPI mode is documented as "no decomposition ... all ranks accomplish the same work") | none (no cudaSetDevice) | n/a | no | none | n/a | **keep-as-supplemental** |
| miniem | native-mpi upstream (Trilinos/Panzer + Tpetra) | Tpetra/Kokkos local rank | yes | optional | Tpetra maps | any N | **pending** (blocked on the Trilinos dependency decision) |

Core set (distributed multi-GPU benchmarks): **amg2023, laghos, remhos,
examinimd, exampm, haccabanapm, cloverleaf, tealeaf, kripke, branson,
hipbone** (11). Distributed extensions to adopt: **p3_heat3d, p3_vlp4d** (from
their own upstream repo). Supplemental (single-GPU motifs / throughput-only):
**miniweather, cabanapic, shaw, exacmech, minibude, xsbench**. Pending:
**miniem**.

## Why the previous architecture could not scale to 40/80 GPUs

1. **Frozen 1-rank decks**: every run.sh defaulted to `mpirun -np 1`; rank
   counts came from ad-hoc `HPCPERF_NP` with no GPU accounting.
2. **Oversubscription by default**: `hpcperf_env.sh` globally exported
   `PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe` and each run.sh
   added `--oversubscribe`, so ranks > GPUs silently shared GPU 0 for every
   app without in-app binding (MFEM apps, CloverLeaf/TeaLeaf, Kripke,
   miniWeather).
3. **Single-GPU problem sizes**: the decks were sized for one B200; strong-
   scaling them to 4+ GPUs is communication-dominated by construction, and no
   weak-scaling policy existed.
4. **Single-node transport**: the only working CUDA device-buffer transport
   (`pml ob1, btl self,sm,smcuda`) is single-node-only; the site UCX hangs on
   CUDA device buffers, so there is currently NO verified multi-node path.
5. **No topology management**: apps with `-P`/`--procs`-style products had no
   helper to pick/validate grids, and infeasible N failed deep inside the app.

## Per-app fact sheets

The full per-field fact sheets (upstream MPI structure, size semantics with
file:line citations, strong/weak input generation, 40/80-GPU topologies,
blockers) were collected per app; the load-bearing facts are:

- **amg2023**: `-n` is PER-RANK (amg.c:332); NP must equal px*py*pz
  (amg.c:968); weak = fixed `-n`, strong = divide a global grid; 32-bit
  HYPRE_Int caps global unknowns at 2^31 (127 ranks @256^3/rank; 80 ranks =
  62.5% of the cap). 40/80: `-P 5 4 2` / `-P 5 4 4`.
- **laghos**: mesh path partitions any N <= elements (cube01_hex has 8
  elements, x8 per `-rs`; rs4 = 32768); `-epm <elems/rank>` gives exact
  per-rank weak scaling with an internal rank-grid factorization (2x4x5 @40,
  4x4x5 @80). `-gam` toggles GPU-aware MPI. FOM counters are 32-bit
  HYPRE_BigInt.
- **remhos**: same MFEM base; `-ho 3 -lo 5 -fct 2` (the GPU combo) is
  rank-count independent (remhos.cpp:398); FOM overflows 32-bit at
  unknowns*steps*3 > 2^31 (cosmetic).
- **examinimd**: the ~20M-atom "L~170" ceiling is PER-RANK: the 2D neighbor
  list view is indexed with `int` stride*index products
  (neighbor_2d.h:86-94), maxneighs~105 for the LJ melt => ~2.1e9/105 ~ 20M
  atoms/rank; scale-out raises the total ceiling by N. Global ids are also
  int => hard 2^31 global atom cap. Internal surface-minimizing rank grid
  (2x4x5 @40, 4x4x5 @80). Deck box is GLOBAL lattice cells.
- **exampm**: examples HARD-CODE a 1xNx1 Y-slab decomposition
  (dam_break.cpp:87-90); slabs thinner than the halo width (3 cells) are
  silently wrong -- cell 0.01 => N <~ 35, cell 0.005 => N <~ 70. Needs a
  ~5-line switch to `DimBlockPartitioner` before 80-rank runs; no clean
  weak-scaling knob (domain is a fixed unit cube).
- **haccabanapm**: topology via `PMKOKKOS_TOPOLOGY`/indat/`MPI_Dims_create`
  (TopologyOverride.hpp); any N; ng need not divide P (warning only); pm_ic
  and pm_run must use the same N; weak scaling = scale NG=NP=RL together
  (@40: NG=880 with 5,4,2; @80: NG=1120 with 5,4,4); benchmark indat (256^3)
  is too small for 40-80 GPUs.
- **cloverleaf / tealeaf**: GLOBAL deck cells, automatic any-N chunk
  factorization (auto 5x8 @40, 8x10 @80); CUDA-aware optional via
  `--staging-buffer auto` (elides host staging when MPIX query says aware);
  TeaLeaf weak-scaled grids need new `tea.problems` reference rows.
- **kripke**: `--procs` product must equal NP AND each `--zones` dim must be
  divisible by procs*zset in that dim; groups%gset==0, quad%dset==0. Weak:
  zones = 32*procs per dim (`--procs 5,4,2 --zones 160,128,64` @40;
  `5,4,4 / 160,128,128` @80).
- **branson**: REPLICATED mode = global photon count auto-split ~1/N per rank
  + per-step tally Allreduce (a real work-split, host-buffer MPI only), any
  N; mesh memory replicated (fine at 591k cells). PARTICLE_PASS (METIS
  domain decomposition, upstream `3D_hohlraum_multi_node.xml`) exists but is
  unvalidated here. Binding uses GLOBAL rank -> needs block rank mapping.
- **hipbone**: `-nx/-ny/-nz` are PER-RANK elements ("weak-scale" by design);
  without `-px/-py/-pz` the rank count must be a perfect cube -- the frozen
  run.sh lacks them, which is the only blocker. FOM is already per-rank
  normalized. @40: `-px 5 -py 4 -pz 2`; @80: `-px 5 -py 4 -pz 4`.
- **miniweather**: 1D x-only decomposition, problem size fixed at COMPILE
  time (`MINIWEATHER_NX`), no rank->GPU binding, PnetCDF output collective.
  Genuinely MPI but a poor selectable-N vehicle.
- **p3_heat3d / p3_vlp4d**: upstream P3-miniapps contains ready-made MPI
  variants (heat3d_mpi: `--px/--py/--pz` + per-rank `--nx/--ny/--nz`,
  CUDA-aware halo; vlp4d_mpi: internal 4D bisection, device-buffer halos +
  Allreduce). Adoption cost is build wiring, not code. vlp4d_mpi changes the
  interpolation scheme (Lagrange -> spline): document as its own benchmark
  configuration.
- **cabanapic / shaw**: shardable physics, but no comm layer exists and
  nothing upstream to adopt => rewrite-cost; keep as single-GPU motifs.
- **exacmech / minibude / xsbench**: physically replica-only (no coupling);
  XSBench upstream documents its MPI mode as replicated work. Any "multi-GPU"
  claim for these would be throughput mode -- excluded from core by
  definition.

## Infrastructure delivered this round

- `level2/tools/hpcperf_mpi_launch.sh` -- GPU-count-aware launcher (see
  tools/README.md): `--gpus N|all`, one rank per GPU, fail-fast on
  ranks > GPUs, site-profile transport, Slurm slot accounting, dry-run mode,
  per-rank binding audit lines.
- `level2/tools/hpcperf_topology.py` -- balanced process grids with
  constraint checking (divisibility, power-of-two, aspect); fails with nearby
  feasible counts instead of silently changing N.
- `level2/tools/mpi_gpu_bind.sh` -- scheduler-safe binding: respects
  scheduler per-task binding, picks the local-rank-th entry of the inherited
  `CUDA_VISIBLE_DEVICES`, FAILS (exit 12) when local rank >= visible GPUs
  unless `HPCPERF_ALLOW_OVERSUBSCRIBE=1`, audit line per rank.
- Site profiles `level2/tools/site/{gmu-hopper,generic}.sh`: launcher choice
  and transport per site; gmu-hopper marks multi-node **BLOCKED/UNVERIFIED**
  (validated `self,sm,smcuda` transport is single-node-only; site UCX hangs
  on CUDA device buffers).
- Refactored `run.sh` for **amg2023, laghos, examinimd**:
  `HPCPERF_GPUS` + `HPCPERF_SCALE_MODE=smoke|strong|weak`, topology/limits
  checked, launched through the common launcher. Verified 1/2/4 GPUs on this
  allocation; 8/40/80 validated as dry-run configuration only.

## Common resource parameters (defined this round)

| Variable | Meaning |
|---|---|
| `HPCPERF_GPUS=N\|all` | GPUs to use = MPI ranks (one rank per GPU) |
| `HPCPERF_NODES`, `HPCPERF_GPUS_PER_NODE` | allocation shape override (dry-run planning; normally auto-detected from Slurm) |
| `HPCPERF_CPUS_PER_RANK` | CPUs per rank (srun `--cpus-per-task`) |
| `HPCPERF_SCALE_MODE=smoke\|strong\|weak` | per-app size policy |
| `HPCPERF_LAUNCHER=auto\|mpirun\|srun` | launcher; default from site profile |
| `HPCPERF_ALLOW_OVERSUBSCRIBE=0\|1` | default 0; 1 = debug only, loud warning |
| `HPCPERF_SITE_PROFILE=<site>` | site profile (default auto: gmu-hopper on dgx003) |
| `HPCPERF_DRY_RUN=1` | print resource plan + command, do not execute |
| `HPCPERF_NP` | backwards-compat alias; must equal `HPCPERF_GPUS` if both set (error otherwise) |

## Open blockers for real 40/80-GPU runs

1. **Multi-node MPI transport (the hard one)**: the conda Open MPI's
   device-buffer traffic hangs over the site UCX even single-node; the
   working `self,sm,smcuda` transport cannot cross nodes. Real multi-node
   needs a site-provided (or purpose-built) GPU-aware UCX/Open MPI stack,
   plus rebuilds of hypre, MFEM, Cabana, heFFTe against it. Status:
   BLOCKED/UNVERIFIED (never claimed working).
2. Allocation shape: 40/80 GPUs = 5/10 nodes on this cluster (8 GPU/node);
   only single-node allocations have been available to this project so far.
3. Per-app work before 80 ranks: exampm 1D-slab fix; hipbone -px/py/pz decks;
   tealeaf reference rows for weak grids; kripke zones/procs deck family;
   haccabanapm large indat (NG=1120) + big IC files; branson block-mapping
   requirement documented.
4. Remaining 7 non-refactored MPI apps still use legacy `HPCPERF_NP` run.sh
   (gated, no launcher integration yet).

## Dependency profile / fingerprint design (item for the next phase)

Current state: `setup_level2_deps.sh` builds CUDA-only into flat
`.deps/install/<dep>` prefixes; `.hpcperf-built` records only the tag, so a
prefix built for a different GPU arch / compiler / MPI / GPU-aware flag is
indistinguishable; `hpcperf_env.sh` adds every *marked* prefix to
`CMAKE_PREFIX_PATH` (unmarked variants like the hand-built `*-gpuaware` are
already excluded, which is why they cannot leak into default builds).

Proposed layout (not yet implemented -- do not break the working tree):

```
.deps/install/<profile>/<dep>
  profile := <backend>-<arch>-<site>[-<feature>...]
  e.g.  cuda-sm100-gmu/hypre
        cuda-sm100-gmu-gpuaware/hypre
        hip-gfx950-lux/kokkos
```

- `HPCPERF_DEPS_PROFILE` selects the active profile; `hpcperf_env.sh` puts
  exactly ONE profile's prefixes on `CMAKE_PREFIX_PATH` (no cross-profile
  mixing by construction).
- Each install carries `.hpcperf-fingerprint` (implemented this round):
  dep, tag, backend, arch, compiler, CUDA version, MPI, GPU-aware flag, and
  the sha256 of every applied patch. `is_built` should graduate from
  "tag matches" to "fingerprint matches the requested profile" so that a
  CUDA upgrade, a patch edit, or an arch change triggers a rebuild instead of
  silently reusing a stale install.
- Migration: keep the flat layout as the implicit `cuda-sm100-gmu` profile;
  a symlink `.deps/install/cuda-sm100-gmu -> .` preserves both paths during
  the transition.

Minimal safe change made this round: `.hpcperf-fingerprint` is now written on
every `mark_built` (new builds only; existing installs keep working
unchanged), and the `*-gpuaware` experiment prefixes remain unmarked and
therefore outside `CMAKE_PREFIX_PATH`.
