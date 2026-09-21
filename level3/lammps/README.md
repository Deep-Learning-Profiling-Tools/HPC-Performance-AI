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

Layout since 2026-09-15 (backend/profile isolation): profile `cuda` (or `hip`; override `HPCPERF_LAMMPS_PROFILE`,
must name the backend); build tree `build/level3/lammps/<profile>/`, install/logs
`.deps/level3/lammps/<profile>/{install,logs}` with the fingerprint in the profile's install; results under
`build/level3/lammps/<profile>/run*/`. The results recorded above were produced with the pre-profile layout
(`.deps/level3/lammps/install`, `build/level3/lammps/cuda`), which is kept as historical state and is never read by
the current scripts.

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

## Registered inputs (`inputs.yaml`, `HPCPERF_LAMMPS_INPUT`, 2026-09-21)

`HPCPERF_LAMMPS_INPUT=<id> level3/lammps/run.sh CUDA` selects one of the frozen
`src/bench/` decks with its recorded replication and step count; the id is
refused together with `HPCPERF_SCALE_MODE=strong|weak` or the
`HPCPERF_LAMMPS_STEPS/STRONG/LOCAL` knobs. The derived deck written into the run
directory differs from upstream only in `run ${steps}` and absolute paths for
`data.*`/`*.eam` (the frozen tree is never touched; `benchmark.yaml` declares the
decks and reference logs as protected inputs/references).

| id | deck | atoms | source | reference |
|---|---|---|---|---|
| `lj-32k` (default) | `in.lj` x=y=z=1 | 32,000 | upstream file | `log.15Jul25.lj.fixed.g++.1` |
| `lj-2m` | `in.lj` x=y=z=4 | 2,048,000 | upstream scaled-size mechanism (bench/README) | none upstream (self baseline) |
| `lj-16m` | `in.lj` x=y=z=8 | 16,384,000 | upstream scaled-size mechanism | none upstream (self baseline) |
| `eam-32k` | `in.eam` + `Cu_u3.eam` | 32,000 | upstream file | `log.15Jul25.eam.fixed.g++.1` |
| `rhodo-32k` | `in.rhodo` + `data.rhodo` (CHARMM, PPPM/cuFFT, SHAKE, NPT) | 32,000 | upstream file | `log.15Jul25.rhodo.fixed.g++.1` |

All styles of these decks have Kokkos versions in this build (no host
fallback): lj/cut, eam, lj/charmm/coul/long, pppm, bond harmonic, angle/dihedral
charmm, improper harmonic, fix nve/shake/npt. `in.chain` and `in.chute` also
build with this package set but are not registered yet. Timer: LAMMPS' `Loop
time` (the 100-step run loop). Baseline: Temp/E_pair/TotEng/Press at steps 0 and
100 with the validate.sh tolerances (1e-8 / 1e-5 relative).

Pilot calibration on dgx003 (1x B200, 1 warm-up + 3 measured runs, medians):

| id | Loop time (main compute) | spread | run.sh wall (E2E) | vs upstream reference log |
|---|---|---|---|---|
| `lj-32k` | 0.0199 s | 0.6 % | 3.23 s | 7/7 selected fields identical at log print precision |
| `lj-2m` | 0.158 s | 0.5 % | 5.23 s | self baseline (no upstream log) |
| `lj-16m` | 1.086 s | 0.3 % | 13.5 s | self baseline (no upstream log) |
| `eam-32k` | 0.0765 s | 0.2 % | 3.12 s | 7/7 selected fields identical at log print precision |
| `rhodo-32k` | 0.387 s | 0.7 % | 6.33 s | 6/6 selected fields identical at log print precision |

"Identical at log print precision" means exactly this and no more: the selected
thermo fields -- `lj`/`eam`: atoms, Temp and E_pair at step 0, Temp, E_pair,
TotEng and Press at step 100 (`thermo_style one`, 8 significant digits as printed
by LAMMPS `%g`-style thermo output, `units lj` for `in.lj`, `units metal` for
`in.eam`); `rhodo`: TotEng at steps 0 and 100, Temp and Press at step 100
(`thermo_style multi`, `units real`, 4-6 decimals) -- have the same printed
string in this build's run and in the upstream reference logs
`bench/log.15Jul25.{lj,eam,rhodo}.fixed.g++.1` (a 1-process CPU run, July 2025).
The comparison rule applied is validate.sh's: 1e-8 relative at step 0, 1e-5
relative at step 100 (measured relative error 0 for every field). Other thermo
columns (E_mol, KinEng, E_bond, ...), other steps and the per-step timing
breakdown are not compared. Spread = (max - min) / median over 3 runs.

Only `lj-16m` reaches one second of loop time on a B200; the 32k-atom decks
are 0.02-0.4 s (the bench convention is 100 steps). Raw runs, baselines and the
upstream-reference comparisons: `HPC-Performance-AI-results/inputs-pilot-2026-09-21/measurements/level3-lammps/`.

### ReaxFF: the separate `reaxff.cuda` build profile (implemented 2026-09-21, round 3)

ReaxFF is the CORAL-2 LAMMPS tier-1 workload (HNS crystal). The frozen tree carries
`examples/reaxff/HNS/` (`in.reaxff.hns` with `x/y/z/t` free variables, `data.hns-equil`
(304 atoms), `ffield.reax.hns`, reference logs `log.30Nov23.reaxff.hns.g++.{1,4}`: CPU,
LAMMPS 21 Nov 2023) and the `REAXFF` package with its KOKKOS styles. The default `cuda`
profile does not enable `REAXFF`, and it is not changed; instead:

| item | as implemented |
|---|---|
| profile | `reaxff.cuda` = `HPCPERF_LAMMPS_VARIANT=reaxff` for `build.sh`, `run.sh` and `validate.sh` (`l3_backend_profile LAMMPS cuda reaxff`); `HPCPERF_LAMMPS_PROFILE` override still honoured |
| packages | `KOKKOS MOLECULE KSPACE MANYBODY RIGID GRANULAR` + `REAXFF` (`-DPKG_REAXFF=yes`); every other CMake option, the bundled Kokkos 4.6.2, `Kokkos_ARCH_BLACKWELL100`, the host compiler, CUDA 13.2.78 and Open MPI 5.0.10 identical to `cuda` |
| fingerprint | `.deps/level3/lammps/reaxff.cuda/install/.hpcperf-l3-fingerprint` with `PKGS=...,REAXFF` in `cmake_options`; the `cuda` fingerprint (built 2026-09-21T10:06Z) is untouched |
| paths | `.deps/level3/lammps/reaxff.cuda/{install,logs,cache}`, `build/level3/lammps/reaxff.cuda/` (+ `run.inputs.<id>.<label>/`); nothing under the `cuda` trees was written |
| build | measured: **292 s** at `HPCPERF_BUILD_JOBS=32` (2026-09-21T21:49:00Z -> 21:53:52Z; the round-2 estimate was 6-10 min); binary `build/level3/lammps/reaxff.cuda/lmp_kokkos_cuda` |
| frozen source | untouched (`prepare_benchmark.sh --status`: READY); the derived deck differs from upstream only in `run ${steps}` and absolute `read_data` / `pair_coeff * * ffield` paths |
| run.sh | registered inputs may name `deck_dir: examples/reaxff/HNS` and `variant: reaxff`; a ReaxFF id is refused on the `cuda` profile ("needs build variant 'reaxff'"); `bench/` decks and the default smoke command are unchanged, `validate.sh CUDA` on the default profile re-run: PASS |
| Kokkos options | the unchanged `-k on g 1 -sf kk -pk kokkos newton on neigh half gpu/aware on` (`neigh/qeq` stays at its default `full`) -- the CORAL-2 command line for this workload |

Registered inputs (`inputs.yaml`, `HPCPERF_LAMMPS_VARIANT=reaxff HPCPERF_LAMMPS_INPUT=<id>`):

| id | replication | atoms | steps | source | reference |
|---|---|---|---|---|---|
| `reaxff-hns-2k` | 2x2x2 (the README example syntax) | 2,432 | 100 | upstream deck + README `-v x 2 -v y 2 -v z 2 -v t 100` | upstream CPU log `log.30Nov23.reaxff.hns.g++.1` |
| `reaxff-hns-16k` | 4x4x4 (README size mechanism) | 19,456 | 100 | upstream deck, size variables | none upstream (self baseline) |

Executed GPU path (from the run logs' "Neighbor list info": `(1) pair reaxff/kk ...
kokkos_device`, `(2) fix qeq/reax/kk ... kokkos_device`; `KOKKOS mode with Kokkos version
4.6.2`): both the ReaxFF pair style and the QEq charge solver ran as Kokkos device styles.
Numerical check (rule: CORAL-2 LAMMPS acceptance, thermo Temp / PotEng / Press / E_vdwl /
E_coul within 0.1 % of the baseline, applied at steps 0 and 100; `thermo_modify norm yes`,
`units real`): `reaxff-hns-2k` vs the upstream CPU log -- Temp, PotEng, E_vdwl, E_coul
identical to all printed digits at both steps in 3/3 runs, Press within 4.8e-7 (step 0) and
3.6e-6 (step 100) relative; verdict PASS, exit 0 (`measurements/level3-lammps/upstream-references/`).
`reaxff-hns-16k` has no upstream log: self baseline, READY / PASS across the 3 runs.

Calibration on dgx003 (1x B200, 1 warm-up + 3 measured runs; `Loop time` of the 100-step run):

| id | Loop time median | per run | spread | run.sh wall (E2E) | main >= 1 s | stable |
|---|---|---|---|---|---|---|
| `reaxff-hns-2k` | 0.8612 s | 0.8549, 0.8612, 0.8687 | 1.61 % | 4.58 s | no | yes |
| `reaxff-hns-16k` | 0.9866 s | 0.9853, 0.9874, 0.9866 | 0.22 % | 4.63 s | no | yes |

Both stay just under one second of loop time on a B200 (a reference value, not a gate);
larger replications are the deck's own mechanism and can be registered later without any
build change. `validate.sh` on the `reaxff.cuda` profile (in.lj smoke on the reaxff binary):
PASS.

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
created outside the repository from the materialized artifact (baseline check PASS: 17 checks with the tool of 2026-09-10, 15 since 2026-09-13) -> a `#error` injected
into `src/src/KOKKOS/pair_lj_cut_kokkos.cpp` makes the workspace build FAIL (no binary) -> restore -> build
succeeds (90 s incremental) -> trusted validation on 1 GPU PASS -> a further agent edit is recompiled
(new binary sha256) and validated -> a tampered `validate.sh` or reference log is REFUSED by the trusted
harness before anything runs -> the canonical `level3/lammps` tree hash is unchanged. The workspace build
references neither `_upstream/` nor the canonical `src/`.

<!-- hpcperf:source-section:begin -->
## Source distribution (frozen source artifact, scheme 3, 2026-09-10)

The application source is not in git and not read from `_upstream/`: `tools/prepare_benchmark.sh level3 lammps` materializes the frozen source artifact (`<app>[-<variant>]-<source_version>.tar.zst`, found in the local content-addressed cache `.artifacts/sha256/` or downloaded from the immutable URL recorded in `provenance/source.lock*.yaml` once published; `--artifact FILE` for a local copy) into `src/` (+ `deps/`), the only source `build.sh`/`run.sh`/`validate.sh` use. Archive size + sha256 and `source_tree_sha256` are verified before anything is placed. Identity, patch series, licenses, redistribution status and the equivalence proof against the tree the results above were validated from are under `provenance/` (`source.lock*.yaml`, `patch_series*.txt`, `original_vs_baseline*.diff`, `LICENSES*.md`, `equivalence*.md`, `LOC*.md`); `benchmark.yaml` is the machine-readable contract (entries, inputs, references, identity). The benchmark does not prescribe which part of the source an optimization agent may modify; the integrity layer only protects the harness and the validation assets. Remote status: see `level3/SOURCE_ARTIFACTS.md`.

| variant | artifact | source version | compressed / uncompressed | entries | source_tree_sha256 | archive sha256 | upstream | patches (pre-applied) | redistribution | equivalence | remote | LOC app-owned / bundled deps / benchmark deps / tests / total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| - | `lammps-hpcperf-l3-v1.tar.zst` | hpcperf-l3-v1 | 110.0 MB / 435.9 MB | 13893 | `d7549c2f6d1b5b6c76575b7bf3b0f528cfa69b9b057aa0b3ef0f03921bdf8e48` | `4f6e1096de3a6671326c9aaabd40510ee4a610df961b605633f8cffd2c2cbca1` | stable_22Jul2025_update6 `9c5ab448c78a` | none | cleared | src: EQUIVALENT | REMOTE_FETCH_VERIFIED | 852527 / 494133 / 0 / 124684 / 1471402 |

LOC = cloc 2.06 code lines of the materialized tree (no blank/comment lines, documentation and data excluded); source-ownership categories from `provenance/source.lock*.yaml` (`source_scope`, descriptive metadata written at freeze time). Dependencies are counted per benchmark, so totals overlap across benchmarks that ship the same dependency. The validated results recorded above were produced from trees proven content-equivalent to this artifact (`provenance/equivalence*.md`); they are not re-run by the migration.
<!-- hpcperf:source-section:end -->
