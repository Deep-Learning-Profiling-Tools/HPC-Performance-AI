# WarpX (Level 3)

Full 3D electromagnetic particle-in-cell: charge/current deposition,
Yee/FDTD Maxwell solve, Boris push, particle and guard-cell exchange -- the
complete application driven by its own inputs files, on the GPU through AMReX.

## Provenance

- Official repository: **https://github.com/BLAST-WarpX/warpx** (the former
  ECP-WarpX/WarpX URL redirects there); docs https://warpx.readthedocs.io/
- Release policy: monthly `YY.MM` tags; the pinned AMReX release is recorded
  in `dependencies.json`.
- Selected: **WarpX 26.09** (2026-09-03), commit
  `0c62c75e53a9ad08241535444bd7e53fd1deba88`, with **AMReX 26.09**
  `a52ca73324ac2c7b65ec04f131e6df99eec9c576` (the exact tag WarpX 26.09
  pins). Both fetched by `fetch.sh` into `_upstream/level3/{WarpX,amrex}`
  (shallow, read-only).
- License: BSD-3-Clause-LBNL (`LICENSE.txt`, `LEGAL.txt`).
- Application-owned LOC (cloc 2.06, code lines): `Source/` **112,459**
  (C++ 69,266 in 247 files; headers 37,146). AMReX `Src/` 273,313 counted
  separately (dependency, not modified). No third-party source is vendored
  in-tree; with the build options below nothing is downloaded at configure
  time.

## Build strategy: NATIVE (upstream CMake superbuild, local AMReX source)

`build.sh CUDA` = upstream's documented CMake route:
`-DWarpX_COMPUTE=CUDA -DCMAKE_CUDA_ARCHITECTURES=100 -DWarpX_DIMS=3
-DWarpX_MPI=ON -DWarpX_amrex_src=<AMReX 26.09 checkout> -DWarpX_OPENPMD=OFF
-DWarpX_QED=OFF -DWarpX_PYTHON=OFF -DWarpX_FFT=OFF -DWarpX_APP=ON
-DWarpX_LIB=OFF -DBUILD_TESTING=OFF`, host compiler conda GCC 13.3.0 (upstream
requires GCC 12+ / NVCC 12.4+; upstream's own Perlmutter profile pairs GCC 13
with NVCC 13.2.78), conda Open MPI 5.0.10 (CUDA-aware), CMake 3.28.4 >= 3.25,
Ninja. openPMD (needs HDF5/ADIOS2 for useful output; AMReX plotfiles are
produced without it), QED (PICSAR download) and Python are off for the
bring-up; they are documented options, not modifications. Build time on
dgx003: **1219 s (20.3 min)** at `-j32` (367 targets, AMReX built by the
superbuild), **0 compiler warning lines**; executable
`build/level3/warpx/cuda/bin/warpx.3d.MPI.CUDA.DP.PDP.EB` (726 MB), installed
under `.deps/level3/warpx/install` with the fingerprint (upstream commit,
AMReX commit, compiler, CUDA 13.2.78, MPI, CMake options).

Why not the others: the Spack `warpx` recipe stops at 26.08 and takes the
architecture only through the legacy `^amrex cuda_arch=` path; the local Spack
checkout is 2025-05 (warpx 25.04); no Apptainer on the node and the only
upstream container recipes are Perlmutter-specific (sm_80); site modules are
broken. HIP: `build.sh HIP` carries `-DWarpX_COMPUTE=HIP -DAMReX_AMD_ARCH=gfx950`
and exits with a clear message here (no ROCm). **Untested.**

## Changes from upstream

Class **A -- no source modification.** `run.sh` writes a derived inputs file
into the build tree: upstream's physics/numerics lines verbatim, plus
`warpx.numprocs` (one box per rank), the per-mode `amr.n_cell` /
`amr.max_grid_size` / `max_step`, `warpx.random_seed = 1`, and reduced
diagnostics instead of the plotfile/checkpoint diagnostics (I/O). For the
validation case only the openPMD diagnostic is dropped (openPMD not built).

## Execution model

One MPI rank per GPU (AMReX docs: "MPI ranks == Number of GPUs"). The common
launcher's per-rank wrapper gives each rank one visible GPU; AMReX then binds
device 0, and the launcher audits the mapping (4/4 verified at 4 GPUs).
`warpx.numprocs PX PY PZ` decomposes the domain into exactly one box per
rank, so the requested rank count is the decomposition (`PX*PY*PZ == ranks`,
each per-rank extent a multiple of the blocking factor; otherwise the run is
refused, nothing silently changed). GPU-aware MPI: AMReX auto-detects the
CUDA-aware Open MPI (`MPIX_Query_cuda_support`) and uses device buffers;
`HPCPERF_WARPX_GPU_AWARE=0` forces pinned-host staging (see the measurement
below). CPU binding: runtime default (no OpenMP threads used).

## Inputs (`HPCPERF_SCALE_MODE`, case `uniform_plasma`)

| Mode | Grid (cells) | Macroparticles (2/cell) | Per rank @4 GPU | Topology (numprocs) | Steps | Memory/GPU (est.) | Time/step on B200 | Validation quantity |
|---|---|---|---|---|---|---|---|---|
| smoke (default) | 64x32x32 (upstream) | 131,072 | 32,768 | 2x2x1 | 10 | < 0.1 GB | 0.02 s | particle number (exact) + energies recorded |
| strong | G^3, G=`HPCPERF_WARPX_GLOBAL` (256) | 33,554,432 | 8,388,608 | 2x2x1 (128x128x256 boxes) | 20 | ~4 GB | 0.029 s (1 GPU) / 0.081 s (4 GPU, GPU-aware MPI) / 0.026 s (4 GPU, host-staged) | run completes; reduced diagnostics |
| weak | (L*PX)x(L*PY)x(L*PZ), L=`HPCPERF_WARPX_LOCAL` (128) | 4,194,304 x N | 4,194,304 | `hpcperf_topology.py` grid | 20 | ~1 GB | 0.050 s (4 GPU) | run completes |

Memory estimate ~100 B per particle plus 6 field components (double) per cell
-- far below 180 GB at all sizes. The strong default is a correctness/bring-up
size (33.6M particles, 20 steps in under a second); per-step times come from
WarpX's own `Evolve time ... Avg. per step` output and exclude initialisation
(`warpx.serialize_initial_conditions = 1`, upstream default in this deck,
serialises particle initialisation across ranks, so `Total Time` grows with
the rank count and must not be read as a scaling number).

**Transport observation (single node, `pml ob1 / btl self,sm,smcuda`):** the
4-GPU strong run takes 0.081 s/step with AMReX's auto-enabled GPU-aware MPI
and 0.026 s/step with `amrex.use_gpu_aware_mpi = 0`; the 1-GPU run takes
0.029 s/step. Device-buffer MPI through `smcuda` is therefore the bottleneck
for WarpX's halo/particle exchange on this node (LAMMPS shows the opposite:
`gpu/aware on` 2.38 s vs `off` 5.87 s; SPARTA is indifferent). Recorded as a
site/transport finding; the default stays upstream's (auto-detect), correctness
is unaffected.

## Validation (`validate.sh`, upstream mechanism)

Upstream's regression mechanism is checksums with baselines that upstream
documents as architecture-dependent, so they cannot serve as a reference on a
B200. `validate.sh` therefore runs upstream's **analytic** regression test
`test_3d_langmuir_multi` (`Examples/Tests/langmuir/inputs_base_3d`: electron
+ positron Langmuir wave, 64^3 cells, 524,288 particles, 40 steps) and
re-implements `analysis_3d.py`'s checks on the final plotfile (read directly;
upstream's script needs yt/openPMD-viewer, absent here):

1. `max|E_sim - E_th| / max|E_th| < 5e-2` for Ex, Ey, Ez against the
   analytic solution `E = eps m_e c^2 k/e sin(kx) cos(ky) cos(kz) sin(wp t)`
   (and cyclic), evaluated at cell centres exactly as upstream does;
2. charge conservation `max|divE - rho/eps0| / max|rho/eps0| < 1e-11`
   (upstream's tolerance for Esirkepov deposition), with WarpX's own CODATA
   2022 constants (`Source/ablastr/constant.H`; using CODATA 2018 eps0 would
   show a spurious 6.8e-10 offset);
3. the `uniform_plasma` smoke run: macroparticle number constant at every
   step (exact); the particle+field energy series is recorded for information
   only -- the shipped 2-particles-per-cell thermal plasma with zero initial
   fields is not an energy-conservation test for the momentum-conserving Yee
   scheme (a few percent change in the first plasma periods is expected).

Observed on dgx003 (2026-09-04): **PASS at 1, 2 and 4 GPUs** --
`error_rel = 3.351e-02` for all three components and all three rank counts
(the decomposition does not change the result), charge-conservation residual
1.8e-12 / 1.9e-12 / 1.3e-12, particle number 131,072 at every step; recorded
energy change -3.45 % / -3.56 % / -3.47 % over 10 steps (initial energies
differ by the per-rank random sampling, as expected).

## Results on dgx003 (4x B200, CUDA 13.2.78, Slurm job 9552083)

| Run | Ranks x GPUs | rank->GPU | CPU binding | Topology | Problem | Time | Validation |
|---|---|---|---|---|---|---|---|
| langmuir | 1 x 1 | wrapper; audit 1/1 verified | runtime default | 1x1x1 | 64^3, 524k particles, 40 steps | Total 0.58 s | PASS |
| langmuir | 2 x 2 | wrapper; 2/2 verified | runtime default | 2x1x1 | same | Total 0.70 s | PASS |
| langmuir | 4 x 4 | wrapper; 4/4 verified | runtime default | 2x2x1 | same | Total 0.67 s | PASS |
| smoke | 1/2/4 | wrapper; verified where sampled | runtime default | 1x1x1 / 2x1x1 / 2x2x1 | 64x32x32, 131k particles, 10 steps | Total 0.21-0.35 s | particle number exact |
| strong | 1 x 1 | wrapper; 1/1 verified | runtime default | 1x1x1 | 256^3, 33.6M particles, 20 steps | 0.0289 s/step | completes |
| strong | 4 x 4 | wrapper; 4/4 verified | runtime default | 2x2x1 | 256^3, 33.6M particles | 0.0811 s/step (GPU-aware) / 0.0258 s/step (host-staged) | completes |
| weak | 4 x 4 | wrapper; 4/4 verified | runtime default | 2x2x1 | 256x256x128, 16.8M particles (4.2M/rank) | 0.0500 s/step | completes |

Dry-runs (`HPCPERF_DRY_RUN=1`, hypothetical allocations) -- **DRY-RUN /
UNVALIDATED**, nothing executed:

| GPUs | Nodes x GPUs/node | Mode | Grid | Particles | Per rank | numprocs | Launch |
|---|---|---|---|---|---|---|---|
| 8 | 1 x 8 | strong | 256^3 | 33.6M | 4.2M | 2x2x2 (128^3 boxes) | `mpirun -np 8 --host dgx003:8 --map-by ppr:8:node ...` (single node) |
| 40 | 5 x 8 | weak | 640x512x256 | 167.8M | 4.2M | 5x4x2 | 5 nodes x 8 -- multi-node BLOCKED on this site |
| 80 | 10 x 8 | weak | 640x512x512 | 335.5M | 4.2M | 5x4x4 | 10 nodes x 8 -- multi-node BLOCKED on this site |

## Limitations

- Multi-node: BLOCKED/UNVERIFIED on this site; 40/80-GPU shapes are plans.
- HIP: recipe present, untested (no AMD GPU); no gfx950 statement upstream.
- openPMD/HDF5 output, QED and Python bindings are not built (documented
  options, off for bring-up).
- Upstream's checksum baselines are not used (platform-dependent by
  upstream's own statement); validation is the analytic Langmuir test plus
  charge and particle conservation.
