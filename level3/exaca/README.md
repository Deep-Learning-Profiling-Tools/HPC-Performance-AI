# ExaCA (Level 3, replacement candidate for the tenth default-suite slot)

ExaCA -- "An exascale-capable cellular automaton for nucleation and grain growth" (LLNL / ExaAM, Exascale
Computing Project): a complete additive-manufacturing microstructure application (multilayer melt-pool
solidification from time-temperature histories, spot/directional/single-grain analytic problems, optional
Finch heat-transport coupling, grain-analysis post-processing), not a mini-app or proxy. Motifs: cellular
automata, grain nucleation, decentred-octahedron grain growth, neighbourhood updates over a steering vector of
active cells, MPI 1-D domain decomposition with halo exchange, Kokkos GPU execution.

Status (2026-09-10, node dgx003, 4x B200, CUDA 13.2.78): BUILD_PASS, VALIDATED_PASS at 1/2/4 GPUs
(smoke), strong/weak COMPLETED at 1/2/4 GPUs, 40/80 GPUs DRY-RUN only, HIP UNTESTED (no ROCm), multi-node
UNVERIFIED. **Admission: all twenty criteria of the replacement checklist (below) are met on this node;
the formal default suite stays "9 retained + ExaCA candidate" until the maintainer confirms the admission.**

## Provenance

| item | value |
|---|---|
| official repository | https://github.com/LLNL/ExaCA (README: `README.md`, license: `LICENSE`) |
| selected release | tag `2.1.0`, commit `d26e59cd51e241a327c5267d43fd70537e5425f7` (2025-10-22) -- latest release |
| license | MIT (`LICENSE` + `NOTICE`, LLNL); example material files and the 10,000-orientation grain files are repository content |
| Kokkos (benchmark-specific dependency, `deps/kokkos`) | tag `4.7.04`, commit `82799e4577568f9666bde36265ac15d78da3e6c8` (2026-04-09), Apache-2.0 WITH LLVM-exception; ExaCA requires Kokkos >= 4.0 |
| nlohmann_json (`deps/json/json-3.12.0.tar.xz`) | release tarball v3.12.0, sha256 `42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa`, MIT -- the exact file ExaCA's CMake would FetchContent-download; bundled so that the build needs no network |
| not bundled | Finch (optional coupled heat transport; not used, no coupled workflow is claimed), GoogleTest (unit tests not built), ExaCA-Data (external temperature-history data for `FromFile` problems; the benchmark uses the analytic `Directional` problem) |
| patches | none (modification class A: build flags only) |
| source artifact | `exaca-hpcperf-l3-v1.tar.zst` (see the source-distribution section at the end; identity in `provenance/source.lock.yaml`) |
| LOC (cloc code lines, `provenance/LOC.md`) | application-owned 6,512 (29 files: 5,976 header lines, 473 C++, 63 CMake), agent-modifiable 6,368, tests 2,760, benchmark-specific dependency source (Kokkos) 278,336, total materialized 287,697 |

Kokkos version choice: the first freeze used Kokkos 4.6.02; its `bin/nvcc_wrapper` still defaults to
`-arch=sm_70`, which CUDA 13.2 rejects in CMake's compiler test ("Unsupported gpu architecture 'sm_70'").
Kokkos 4.7.04 (default `sm_80`, "Support CUDA 13", "Fix compiling with Cuda 13.1" in its changelog) builds
unchanged; the artifact was re-frozen with it before any publication (same `source_version`,
`hpcperf-l3-v1`, because nothing had been released).

## Build (`build.sh [CUDA|HIP]`)

Three out-of-source stages from the materialized artifact only (`$HERE/src`, `$HERE/deps`; no fetch, no
patch): (1) Kokkos 4.7.04, CUDA backend, `Kokkos_ARCH_BLACKWELL100` (sm_100, detected), Serial host
backend, `nvcc_wrapper` with the conda GCC 13.3.0 host compiler -> `.deps/level3/exaca/install/kokkos-cuda`;
(2) nlohmann_json 3.12.0 from the bundled tarball (header-only, `JSON_BuildTests=OFF`) ->
`install/json`; (3) ExaCA with `ExaCA_REQUIRE_EXTERNAL_JSON=ON`, `ExaCA_ENABLE_TESTING=OFF`, no Finch ->
`build/level3/exaca/cuda` and `install/exaca-cuda` (`bin/ExaCA`, `bin/ExaCA-GrainAnalysis`, material and
orientation data under `share/ExaCA`, which is where ExaCA resolves the file names of the deck). MPI: conda
Open MPI 5.0.10 (CUDA-aware, not required: ExaCA stages its halos through host buffers). **Verified
build time 109 s** at `-j32` (all three stages). HIP: the same script selects `hipcc` + `Kokkos_ENABLE_HIP`
(ExaCA documents Serial/OpenMP/Threads/CUDA/HIP), **untested here (no ROCm)**. Fingerprint
(`.hpcperf-l3-fingerprint`): ExaCA commit, source tree sha256, Kokkos version/commit, json version, arch,
compiler, CUDA, MPI, CMake options.

## Execution model (`run.sh [CUDA|HIP]`)

One MPI rank per GPU (Kokkos CUDA, device 0 under the launcher's per-rank `CUDA_VISIBLE_DEVICES`
wrapper; the launcher audits the rank->GPU mapping: "N verified, 0 mismatch"). ExaCA decomposes the domain
in Y only (1-D; every rank count is legal as long as each rank gets >= 2 cells in Y); halos are exchanged
every time step. The run is ONE global simulation (the log records `NumberMPIRanks` and the subdomain
sizes/offsets of the single decomposition; validate.sh checks they cover the box exactly once).

Deck: `inputs/dirsolid.template.json` -- the physics of upstream's flagship example
`examples/Inp_DirSolidification.json` (Inconel 625 interfacial response, cell size 1 um, time step
0.0666667 us, thermal gradient G = 5e5 K/m, cooling rate R = 3e5 K/s, nucleation density 10 x 10^12 m^-3,
mean nucleation undercooling 5 K, sigma 0.5 K, RandomSeed 0), with ONE deliberate difference: the bottom
substrate uses `SurfaceSiteDensity: 0.25` (grains placed by a global RNG at physical locations, identical
for every decomposition) instead of `SurfaceSiteFraction` (upstream's mode (i), whose grain placement is
rank-local and therefore decomposition-dependent). Nuclei are drawn from one global RNG stream and
distributed to the owning ranks, so the initial condition is decomposition-independent. `run.sh` sets only
`Domain.Nx/Ny/Nz`, `RandomSeed`, the output path/name and the printing policy (the GrainID field is
written in smoke mode; in strong/weak mode only ExaCA's JSON log is written unless `HPCPERF_EXACA_PRINT=1`,
because the 4 B/cell field write is not the CA work being measured).

| mode (`HPCPERF_SCALE_MODE`) | box (cells) | per rank at N GPUs | steps | purpose |
|---|---|---|---|---|
| smoke (default) | 128 x 128 x 128 = 2.10 M | 2.10 M / N | 7,000 | correctness (validate.sh) |
| strong | 512 x 256 x 512 = 67.1 M (fixed) | 67.1 M / N | 17,000 | strong scaling; a single rank cannot hold 512^3: ExaCA indexes its 26-neighbour octahedron arrays with `int` (134 M x 26 overflows: "failed to allocate 1.678e+07 TiB"), an upstream limitation |
| weak | 512 x (128 N) x 512 | 33.5 M | 17,000 | weak scaling (Y extent grows with N) |
| 40 / 80 GPUs | strong box | 1.68 M / 0.84 M | -- | `HPCPERF_DRY_RUN=1 HPCPERF_NODES=10|20`: HYPOTHETICAL plan only, never executed (multi-node BLOCKED/UNVERIFIED on this site) |

Overrides: `HPCPERF_EXACA_NX/NY/NZ` (Ny per rank in weak mode), `HPCPERF_EXACA_SEED`. A second positional
deck on the command line is rejected. Results land in `build/level3/exaca/cuda/<run>/dirsolid.<mode>.np<N>/`
with `run_manifest.txt` (run id, binary sha256, deck sha256, sizes, exit code).

## Correctness criteria (`validate.sh`, exit 0 PASS / 1 FAIL)

ExaCA ships no reference output for this problem, and its final GrainID field is **not bitwise
reproducible**: liquid-cell captures are resolved with `Kokkos::atomic_compare_exchange`, so two identical
1-GPU runs differ bitwise (measured) while every physically meaningful statistic agrees. The criteria are
therefore statistical invariants of the final microstructure computed by `exaca_check.py` from the field
the run writes, with tolerances derived from the measured spread (below) times a safety factor >= 3:

1. completeness / single decomposition: run exits 0; field + log exist; `DIMENSIONS` = deck; every cell
   solidified (`GrainID != 0`); log `NumberMPIRanks` = N; the Y subdomain sizes sum to Ny + 2(N-1)
   (1-cell halos at the N-1 internal boundaries), i.e. one global run;
2. self-consistency: ExaCA's own `VolFractionNucleated` (log) equals the value recomputed from the field
   (|diff| <= 1e-3);
3. vs the frozen reference `references/dirsolid_smoke.reference.json` (statistics of the validated 1-GPU
   baseline run, with binary sha256 / source tree sha256 / run id recorded inside);
4. with N > 1: the N-GPU statistics vs this build's 1-GPU run (rank-count independence).

| statistic | meaning | tolerance | measured spread: identical 1-GPU runs | 4 GPU vs 1 GPU |
|---|---|---|---|---|
| `unsolidified_cells` | cells never captured | exactly 0 | 0 | 0 |
| `n_grains` | distinct grain IDs in the final field | 1 % rel | 3612 = 3612 | 3621 vs 3612 (+0.25 %) |
| `n_nucleated` | distinct nucleated (negative-ID) grains | 5 % rel (= +-1 of 20) | 20 = 20 | 20 = 20 |
| `vol_fraction_nucleated` | volume fraction of nucleated grains | 0.01 abs | 0.5495 vs 0.5483 | 0.5490 vs 0.5495 |
| `top_layer_grains` | grains reaching the top surface (growth selection) | 15 % rel | 28 = 28 | 30 vs 28 |
| `mean_misorientation_z_deg` | cell-weighted mean angle between the grain's closest <001> axis and the build direction (texture) | 0.25 deg abs | 26.972 vs 26.961 | 26.990 vs 26.972 |
| `mean_misorientation_z_top_deg` | the same over the top layer | 0.5 deg abs | 35.60 vs 35.51 | 35.50 vs 35.60 |
| `mean_grain_volume_cells` | cells / grains | 1 % rel | equal | 579.2 vs 580.6 |

The orientation of a grain is row (|GrainID| - 1) mod 10000 of `GrainOrientationVectors.csv` (ExaCA's
mapping); each row lists the three <001> unit vectors. No tolerance was loosened to obtain a PASS; the
tolerances were fixed before the 2- and 4-GPU validations were run.

## RESULTS (2026-09-10, `run/` tree; reference run id 20260910T212610Z-216184-29676, binary sha256 `f2a8875c6565a619...`)

| test | result |
|---|---|
| build (CUDA sm_100, Kokkos 4.7.04) | BUILD_PASS, 109 s |
| smoke 1 GPU | **PASS** (3612 grains, 20 nucleated, vf 0.5496, top 29, misorientation 26.99 / 35.6 deg; CA 0.80 s) |
| smoke 2 GPU | **PASS** (rank-count comparison ok; subdomains [65, 65]) |
| smoke 4 GPU | **PASS** (subdomains [33, 34, 34, 33]; CA 1.5 s) |
| strong 512x256x512, ExaCA "Time spent performing CA calculations" | 1 GPU 8.02 s, 2 GPU 6.26 s, 4 GPU 5.17 s (1.55x on 4 GPUs: the work per step is the active interface layer; per-step launch/halo overhead dominates at this size; COMPLETED, no correctness claim beyond the log statistics vf = 0.8833 / 0.8833 / 0.8833) |
| weak 512x128x512 per rank | 1 GPU 4.88 s, 2 GPU 6.16 s, 4 GPU 6.80 s (COMPLETED) |
| 40 / 80 GPUs | DRY-RUN (HYPOTHETICAL plan, `run/.dryrun/`), not executed |
| HIP | UNTESTED (no ROCm) |
| multi-node | UNVERIFIED (site transport limitation, see level3/README.md) |

Launcher audit of every executed run: "N verified, 0 mismatch, 0 unverified".

## Replacement-candidate checklist (requested criteria)

| # | criterion | status |
|---|---|---|
| 1 | complete production application, not a mini-app | yes (ExaAM application; feature set above); small code base (6.5 k application-owned LOC) -- stated plainly |
| 2 | exact upstream commit pinned | `d26e59cd51e2...` (tag 2.1.0) |
| 3 | license | MIT + NOTICE |
| 4 | dependency licenses (Kokkos, MPI, json, Finch) | Apache-2.0-LLVM, environment MPI, MIT; Finch not used |
| 5 | scientific input/data redistributable | material/orientation files are repository content (MIT); no external data |
| 6 | CUDA official path | Kokkos CUDA backend, documented; built and validated |
| 7 | HIP official path | Kokkos HIP backend, documented; untested here |
| 8 | MPI distributed path | 1-D Y decomposition, validated 1/2/4 ranks |
| 9 | real 1/2/4 GPU | validated (audit "N verified") |
| 10 | one global run, not N copies | log decomposition check (criterion 1) |
| 11 | correctness criterion | statistical invariants with measured tolerances (above) |
| 12 | strong / weak definition | fixed 67.1 M-cell box / 33.5 M cells per rank |
| 13 | 40/80 GPU dry-run only | yes |
| 14 | multi-node not claimed | UNVERIFIED |
| 15 | scheme-3 source artifact | `exaca-hpcperf-l3-v1.tar.zst`, LOCAL_ARTIFACT_VERIFIED, REMOTE_ARTIFACT_UNPUBLISHED |
| 16 | benchmark.yaml | yes |
| 17 | optimization_scope.yaml | yes (modifiable: `src/src/*`, `src/bin/*`, `src/analysis/src/*`) |
| 18 | provenance/source.lock.yaml | schema 2, redistribution cleared |
| 19 | application-owned LOC | 6,512 |
| 20 | agent-modifiable LOC | 6,368 |

Not claimed: Finch-ExaCA coupled workflow, `FromFile`/multilayer problem types, ExaCA unit tests (not
built), any performance optimization.

<!-- hpcperf:source-section:begin -->
## Source distribution (frozen source artifact, scheme 3, 2026-09-10)

**Replacement candidate** for the tenth slot of the default suite; admission depends on the bring-up criteria recorded in this README.

The application source is not in git and not read from `_upstream/`: `tools/prepare_benchmark.sh level3 exaca` materializes the frozen source artifact (`<app>[-<variant>]-<source_version>.tar.zst`, found in the local content-addressed cache `.artifacts/sha256/` or downloaded from the immutable URL recorded in `provenance/source.lock*.yaml` once published; `--artifact FILE` for a local copy) into `src/` (+ `deps/`), the only source `build.sh`/`run.sh`/`validate.sh` use. Archive size + sha256 and `source_tree_sha256` are verified before anything is placed. Identity, patch series, licenses, redistribution status and the equivalence proof against the tree the results above were validated from are under `provenance/` (`source.lock*.yaml`, `patch_series*.txt`, `original_vs_baseline*.diff`, `LICENSES*.md`, `equivalence*.md`, `LOC*.md`); `optimization_scope.yaml` says what an agent may modify; `benchmark.yaml` is the machine-readable contract. Remote status: see `level3/SOURCE_ARTIFACTS.md`.

| variant | artifact | source version | compressed / uncompressed | entries | source_tree_sha256 | archive sha256 | upstream | patches (pre-applied) | redistribution | equivalence | remote | LOC app-owned / agent-modifiable / bundled deps / benchmark deps / tests / total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| - | `exaca-hpcperf-l3-v1.tar.zst` | hpcperf-l3-v1 | 2.6 MB / 15.3 MB | 1579 | `b88e84b74db34b8adb58bc9924ec1e0f3c7d416c58577c5b16bcfbda11f81f44` | `3de419c92222c0a76e6afdb621bb46f6d7b530dc1d064fcee34393f88c9fe316` | 2.1.0 `d26e59cd51e2` | none | cleared | src: EQUIVALENT, deps/kokkos: EQUIVALENT | REMOTE_ARTIFACT_UNPUBLISHED | 6512 / 6368 / 0 / 278336 / 2760 / 287697 |

LOC = cloc 2.06 code lines of the materialized tree (no blank/comment lines, documentation and data excluded); categories from `optimization_scope.yaml` (`loc_categories`). The validated results recorded above were produced from trees proven content-equivalent to this artifact (`provenance/equivalence*.md`); they are not re-run by the migration.
<!-- hpcperf:source-section:end -->
