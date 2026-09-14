# SPECFEM3D Cartesian (Level 3)

Spectral-element seismic wave propagation: the complete workflow -- mesh
(CUBIT mesh + SCOTCH partitioner, or the in-house `xmeshfem3D`), per-slice
database generation, and the GPU time loop of `xspecfem3D` (stiffness,
MPI halo assembly, absorbing boundaries, seismogram output).

## Provenance

- Official repository: https://github.com/SPECFEM/specfem3d (moved from
  geodynamics/specfem3d); docs https://specfem3d.readthedocs.io/
- Release policy: tagged releases (v4.1.1 is the latest, 2024-03-15);
  development happens on `devel`, ~2.5 years ahead of the release.
- Selected: **v4.1.1**, commit `c67d3ae7d4bfc5ac75cb9e5601d93afa262d3d8d`,
  fetched by `fetch.sh` into `_upstream/level3/specfem3d` (shallow, read-only;
  the `m4`/`flexwin`/`pyCMT3D` submodules are not needed). Two source
  back-ports come from `devel` `cc2e9ffa7e7cb5338e05f5a7df81cfbe60e00683`
  (2026-07-24).
- License: GPL-3.0.
- Application-owned LOC (cloc 2.06, code lines): `src/` **142,398**
  (Fortran 90 119,057; CUDA 12,210; `src/gpu` 15,058 in 63 `.cu` files).
  `utils/` (163,847) and the bundled `external_libs/` (SCOTCH 5.1.12b,
  METIS, PaToH; 137,463) are counted separately.

## Build strategy: NATIVE (upstream autotools + bundled SCOTCH)

`build.sh CUDA` copies the sources into `.deps/level3/specfem3d/src`
(autotools builds in-tree), applies the two patches below and runs upstream's
recipe: `./configure --with-mpi --with-cuda=cuda12 FC=/usr/bin/gfortran
CC=<conda gcc 13.3> MPIFC=mpif90 MPI_INC=<conda mpi.h dir> CUDA_INC/CUDA_LIB
USE_BUNDLED_SCOTCH=1`, then `make -j all GENCODE="-gencode=arch=compute_100,
code=sm_100 -gencode=arch=compute_100,code=compute_100 -DGPU_DEVICE_Blackwell"`.
Toolchain: conda GCC 13.3.0 for C and as nvcc's host compiler, system
gfortran 14.2.1 for Fortran (the conda env has no gfortran), conda Open MPI
5.0.10 with `OMPI_FC=/usr/bin/gfortran`. Build time on dgx003: **21 s** at
`-j16` (428 objects; SCOTCH, 1,000+ Fortran units, 63 CUDA units), 7
warning lines. 24 executables installed under `.deps/level3/specfem3d/install/bin`
with the fingerprint (upstream commit, SCOTCH 5.1.12b, compilers, CUDA
13.2.78, MPI, configure/make options, patch list).

Why not the others: no Spack package exists for SPECFEM3D Cartesian (only
`specfem3d-globe`); no Apptainer on the node and no upstream image; site
modules broken. HIP: `build.sh HIP` carries `--with-hip` but v4.1.1 knows only
MI8..MI250 (gfx803..gfx90a); devel added MI300/MI350. **Untested.**

## Changes from upstream (all recorded in `patches/` and `build.sh`)

| Class | Change | Size / provenance |
|---|---|---|
| D | `0001-cuda13-deviceOverlap-guard.patch`: `src/gpu/initialize_gpu.cu` reads `cudaDeviceProp.deviceOverlap`, removed in CUDA 13 -> guarded, `asyncEngineCount` printed instead (diagnostic text only) | 10 lines, back-port of upstream devel |
| D | `0002-blackwell-device-block.patch`: `GPU_DEVICE_Blackwell` block in `src/gpu/mesh_constants_cuda.h` (`#undef USE_LAUNCH_BOUNDS`, identical to Hopper's) | 8 lines, back-port of upstream devel |
| B | make-time `GENCODE` override = devel's `--with-cuda=cuda13` value (sm_100 SASS + compute_100 PTX); v4.1.1's `configure` stops at `cuda12` and cannot be regenerated here (no autoreconf, empty `m4/`) | no file edited |
| B | bundled SCOTCH built without gzip support (generated `Makefile.inc`: `-DCOMMON_FILE_COMPRESS_GZ`/`-lz` removed; conda GCC has no `zlib.h`) | generated file only |
| C | `OMPI_FC=/usr/bin/gfortran`, `MPI_INC` (configure's `mpif90 -showme:incdirs` detection returns nothing with the conda wrapper) | environment |

No numerics, physics or algorithm changed; `flags.guess`'s gfortran flags
(`-std=f2008 -pedantic-errors -ffpe-trap=invalid,zero,overflow`) are used as
shipped and compile cleanly with gfortran 14.

## Execution model

One MPI rank per GPU (upstream: `device = myrank % device_count`). The common
launcher's per-rank wrapper gives each rank one visible GPU (device 0), and
audits the mapping (4/4 verified). `NPROC` in `Par_file` is the number of mesh
slices and must equal the rank count -- it is fixed when the mesh is
partitioned, so every rank count gets its own mesh + databases (SPECFEM3D's
normal workflow; `run.sh` does all three stages through the launcher with the
same N). Halo exchange in v4.1.1 is host-staged (no GPU-aware MPI). CPU
binding: runtime default.

## Inputs (`HPCPERF_SCALE_MODE`)

| Mode | Mesh | Elements | Per rank @4 GPU | Topology | NSTEP / DT | GPU memory (est.) | Time loop on B200 | Validation quantity |
|---|---|---|---|---|---|---|---|---|
| smoke (default) | upstream `homogeneous_halfspace` CUBIT mesh, SCOTCH partition | 20,736 | 5,184 | SCOTCH (any N) | 5000 / 0.05 s | ~0.2 GB | 0.8-0.9 s | seismograms vs upstream `REF_SEIS` |
| strong | `xmeshfem3D`, same 134x134x60 km domain refined G x (G=`HPCPERF_SPECFEM_STRONG`, 2): 72x72x32 | 165,888 | 41,472 | `NPROC_XI x NPROC_ETA` from `hpcperf_topology.py --dims 2` (NEX divisible) | 1000 / 0.025 s | ~1.7 GB | 0.975 s (1 GPU) / 0.412 s (4 GPU) | run completes; seismograms written |
| weak | per-rank block (36F)x(36F)x(16F), F=`HPCPERF_SPECFEM_LOCAL` (2), domain extended PX x PY at fixed resolution | 165,888 x N | 165,888 | 2-D grid | 1000 / 0.025 s | ~1.7 GB | 1.035 s (4 GPU, 663,552 elements) | run completes |

Memory estimate ~10 KB per element on the GPU (NGLL 5, single precision
fields). The weak deck keeps element shape, DT and per-rank work identical for
every N (the domain grows, source and stations stay in the first block).
Preprocessing cost is CPU-side and per slice: `xgenerate_databases` for
165,888 elements on one rank takes ~12 s; at G=4 (1.33M elements on one rank)
its serial neighbour search exceeded 20 minutes and was abandoned, which is
why the strong default is G=2.

## Validation (`validate.sh`, upstream mechanism)

Upstream ships reference seismograms for the homogeneous half-space
(`EXAMPLES/applications/homogeneous_halfspace/REF_SEIS`, 12 traces: 4 stations
x 3 components) and compares runs with them using
`utils/scripts/compare_seismogram_correlations.py` (per trace: correlation
coefficient, L2 misfit normalised by the reference energy, cross-correlation
time shift; upstream thresholds corr >= 0.8, misfit <= 1 %, shift <= 0.01 s).
The references are CPU/double-precision results; the GPU solver is single
precision, so upstream's tolerance-based comparison is the appropriate
criterion and is used unchanged. Observed on dgx003 (2026-09-04): **PASS at 1,
2 and 4 GPUs** -- correlation 1.00000 on all 12 traces, worst misfit 2.7e-4 /
2.3e-4 / 2.7e-4, worst time shift 2.5e-5 / 2.4e-5 / 2.7e-5 s.

## Results on dgx003 (4x B200, CUDA 13.2.78, Slurm job 9552083)

| Run | Ranks x GPUs | rank->GPU | CPU binding | Topology | Problem | Time loop (`output_solver.txt`) | Wall incl. mesh+databases | Validation |
|---|---|---|---|---|---|---|---|---|
| smoke | 1 x 1 | wrapper; audit 1/1 verified | runtime default | 1 slice | 20,736 el., 5000 steps | 0.860 s | 4 s | PASS |
| smoke | 2 x 2 | wrapper; 2/2 verified | runtime default | SCOTCH 2 | same | 0.779 s | 6 s | PASS |
| smoke | 4 x 4 | wrapper; 4/4 verified | runtime default | SCOTCH 4 | same | 0.806 s | 7 s | PASS |
| strong | 1 x 1 | wrapper; 1/1 verified | runtime default | 1x1 | 165,888 el., 1000 steps | 0.975 s | 15 s | completes |
| strong | 4 x 4 | wrapper; 4/4 verified | runtime default | 2x2 | 165,888 el. (41,472/rank) | 0.412 s | 17 s | completes |
| weak | 4 x 4 | wrapper; 4/4 verified | runtime default | 2x2 | 663,552 el. (165,888/rank) | 1.035 s | 32 s | completes |

Dry-runs (`HPCPERF_DRY_RUN=1`, hypothetical allocations) -- **DRY-RUN /
UNVALIDATED**, nothing executed (each of the three stages prints its plan):

| GPUs | Nodes x GPUs/node | Mode | Mesh | Elements | Per rank | NPROC_XI x NPROC_ETA | Launch |
|---|---|---|---|---|---|---|---|
| 8 | 1 x 8 | strong | 72x72x32 on 134x134x60 km | 165,888 | 20,736 | 4x2 | `mpirun -np 8 --host dgx003:8 --map-by ppr:8:node ...` (single node) |
| 40 | 5 x 8 | weak | 576x360x32 on 1072x670x60 km | 6,635,520 | 165,888 | 8x5 | 5 nodes x 8 -- multi-node BLOCKED on this site |
| 80 | 10 x 8 | weak | 720x576x32 on 1340x1072x60 km | 13,271,040 | 165,888 | 10x8 | 10 nodes x 8 -- multi-node BLOCKED on this site |

## Limitations

- Multi-node: BLOCKED/UNVERIFIED on this site; 40/80-GPU shapes are plans.
- HIP: untested; v4.1.1 has no MI300/MI350 configure option.
- v4.1.1 + CUDA 13.2 + sm_100 is not an upstream-validated combination; it
  needs the two back-ports and the make-time GENCODE above (all upstream
  devel content).
- Only the homogeneous half-space family is wrapped; layered_halfspace,
  Mount_StHelens, CPML and fault examples build with this configuration but
  have no wrappers yet.

<!-- hpcperf:source-section:begin -->
## Source distribution (frozen source artifact, scheme 3, 2026-09-10)

The application source is not in git and not read from `_upstream/`: `tools/prepare_benchmark.sh level3 specfem3d` materializes the frozen source artifact (`<app>[-<variant>]-<source_version>.tar.zst`, found in the local content-addressed cache `.artifacts/sha256/` or downloaded from the immutable URL recorded in `provenance/source.lock*.yaml` once published; `--artifact FILE` for a local copy) into `src/` (+ `deps/`), the only source `build.sh`/`run.sh`/`validate.sh` use. Archive size + sha256 and `source_tree_sha256` are verified before anything is placed. Identity, patch series, licenses, redistribution status and the equivalence proof against the tree the results above were validated from are under `provenance/` (`source.lock*.yaml`, `patch_series*.txt`, `original_vs_baseline*.diff`, `LICENSES*.md`, `equivalence*.md`, `LOC*.md`); `benchmark.yaml` is the machine-readable contract (entries, inputs, references, identity). The benchmark does not prescribe which part of the source an optimization agent may modify; the integrity layer only protects the harness and the validation assets. Remote status: see `level3/SOURCE_ARTIFACTS.md`.

| variant | artifact | source version | compressed / uncompressed | entries | source_tree_sha256 | archive sha256 | upstream | patches (pre-applied) | redistribution | equivalence | remote | LOC app-owned / bundled deps / benchmark deps / tests / total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| - | `specfem3d-hpcperf-l3-v1.tar.zst` | hpcperf-l3-v1 | 236.8 MB / 634.3 MB | 6205 | `9bdc4eed3a3593e4d87e57ed1802eff37d0a7b78a060e083cbcd3155592adc4f` | `62b9739b1440a1cb1d33c58d282b5f418f9f92f6c152cfde4697ee28c22eb316` | v4.1.1 `c67d3ae7d4bf` | 0001-cuda13-deviceOverlap-guard.patch, 0002-blackwell-device-block.patch | cleared | src: EQUIVALENT | REMOTE_FETCH_VERIFIED | 142516 / 147593 / 0 / 1027 / 307549 |

LOC = cloc 2.06 code lines of the materialized tree (no blank/comment lines, documentation and data excluded); source-ownership categories from `provenance/source.lock*.yaml` (`source_scope`, descriptive metadata written at freeze time). Dependencies are counted per benchmark, so totals overlap across benchmarks that ship the same dependency. The validated results recorded above were produced from trees proven content-equivalent to this artifact (`provenance/equivalence*.md`); they are not re-run by the migration.
<!-- hpcperf:source-section:end -->
