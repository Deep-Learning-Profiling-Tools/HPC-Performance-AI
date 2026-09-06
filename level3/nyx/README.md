# Nyx (AMReX-Astro) -- Level 3 second batch

Cosmological N-body + baryon hydrodynamics (dark-matter particles, Poisson
gravity by AMReX MLMG multigrid, PPM hydro, comoving coordinates), full
application workflow: IC read, gravity solve, hydro/particle advance, particle
redistribution, plotfile/checkpoint I/O.

## Provenance / versions

| Item | Value |
|---|---|
| Nyx | tag `26.09`, `e06eabc1b9dbcad5612db9529aced682402daede` (2026-08-26), BSD-3-Clause-LBNL |
| AMReX (used) | tag `26.09`, `a52ca73324ac2c7b65ec04f131e6df99eec9c576` (2026-09-01) -- **external, private build** |
| AMReX (Nyx submodule pin) | `6e875b7cc1a4eec78e22ae4cdaa79f88acf5169e` (development 2026-08-12) -- **not used**, see below |
| SUNDIALS (heatcool variant only) | Nyx submodule pin `5c53be85c88f63c5201c130b8cb2c686615cfb03` = v7.2.1 |
| Build strategy | NATIVE (CMake): private AMReX 26.09 install + Nyx via `find_package(AMReX CONFIG)`; upstream's GPU CI options (`Nyx_HYDRO=YES Nyx_MPI=YES Nyx_OMP=NO`, C++17); double-precision particles as upstream's nightly regression builds |
| Compiler / Toolkit / MPI | conda GCC 13.3.0 (host), CUDA 13.2.78 sm_100, conda Open MPI 5.0.10 (site profile gmu-hopper: `pml ob1 / btl self,sm,smcuda`) |
| Profiles | `cuda132-gcc133-adiabatic` (Nyx_HEATCOOL=NO), `cpu-gcc133-adiabatic` (Nyx_GPU_BACKEND=NONE reference + AMReX plotfile tools + particle_compare), `cuda132-gcc133-heatcool` (planned: SUNDIALS 7.2.1 CVODE, see status) |
| Source changes | **none** (class A: build options; class A derived decks written at run time) |

Layout: `.deps/level3/nyx/<profile>/{src,build,install,logs,cache}`, application
build tree `build/level3/nyx/<profile>`, run dirs
`build/level3/nyx/<profile>/run/<case>.<mode>.np<N>` (dry-runs under `.dryrun/`).
Fingerprint `.deps/level3/nyx/<profile>/install/.hpcperf-l3-fingerprint` (schema
l3-2) + `BUILD_INFO.txt` (binary sha256, `AMREX_CUDA_ARCHS`, `cuobjdump` archs).

### Why AMReX 26.09 and not the submodule pin

The first build against the pin produced an **sm_86** binary: that AMReX's
`convert_cuda_archs()` (Tools/CMake/AMReXUtils.cmake) drops every SM >= 10.0
("CMake 3.30 does not support SM 10.0+ in cuda_select_nvcc_arch_flags"), the
list becomes empty, autodetection runs and CMake 3.28's table maps this B200
(compute capability 10.0, confirmed by the detection program itself) to
`8.6+PTX`. No option of that AMReX yields sm_100. AMReX 26.09 rewrote the
resolution (`AMReXCUDAArchs`, `nvcc --list-gpu-arch`) and produced
`AMREX_CUDA_ARCHS=100` / `cuobjdump: sm_100` -- the same path the first-batch
WarpX 26.09 build already used. 26.09 is 21 commits ahead of / 0 behind the
pin (GitHub compare), Nyx's minimum is AMReX 20.11, and Nyx consumes an
external AMReX through `find_package(AMReX CONFIG)` with the component set its
own superbuild would request (3D, DOUBLE, PARTICLES/PDOUBLE, MPI, CUDA,
LSOLVERS). The sm_86 attempt was removed
(`.deps/level3/nyx/ATTEMPT-1-sm86-removed.txt`). WarpX's AMReX *binary* is not
reused (different component set: EB, no linear solvers, FFT); only the same
read-only source checkout tag.

## Cases (all upstream decks, used from the read-only checkout)

| Case | Deck | What | Mode |
|---|---|---|---|
| `minisb` | `Exec/MiniSB/inputs.32` + `nyx.ppm_type=0` | Santa Barbara cluster, 32^3 cells, 32,686 DM particles (shipped ASCII IC), 10 steps; **exactly upstream's nightly GPU regression test "MiniSB"** (2 ranks there) | smoke |
| `lya_adiabatic` smoke | `Exec/LyA/inputs.rt.garuda` | upstream's GPU regression deck **"LyA-adiabatic"** (heat_cool_type=0, strang_split=1), 32^3, shipped `32.nyx` IC, z=100, 10 steps | smoke |
| `lya_adiabatic` strong | `Exec/LyA/inputs` with heating/cooling OFF | the flagship 64^3 Lyman-alpha science deck (shipped `64sssss_20mpc.nyx` IC, z=159) as a **named adiabatic derivative**: `nyx.heat_cool_type=0 sdc_split=0 strang_split=1`. Does **not** cover the heating/cooling LyA workload | strong |
| `scaling_synthetic` | `Exec/Scaling/inputs` (RandomPerCell) | upstream's scaling deck; `RandomPerCell` is documented as a testing-only initialisation -> **synthetic scaling/communication test, not a science IC** | strong (fixed G^3, default 256^3) / weak (64^3 cells per rank, box and total DM mass scaled with the tiles) |
| `lya_heatcool` | `Exec/LyA/inputs` as shipped (heat_cool_type=11, CVODE) | the heating/cooling LyA workload -- needs the `heatcool` profile (SUNDIALS) | smoke/strong -- **not built yet (status below)** |

Decomposition: `amr.max_grid_size` is fixed per case (16 for 32^3/64^3 decks,
64 for the synthetic decks) and `amr.refine_grid_layout=0` (as upstream's MiniSB
deck), so the BoxArray is identical for every rank count and only the
distribution changes; ranks > boxes is refused (never a silent idle rank),
boxes % ranks != 0 is reported as imbalanced. One MPI rank per GPU; AMReX binds
device 0 of the one GPU the launcher wrapper exposes; the launcher audits the
expected/observed mapping (all runs below: every rank `verified`).

Derived deck = upstream lines verbatim minus the I/O cadence / decomposition
lines listed in the file header, plus `amrex.the_arena_init_size=0` (as the
official test command) and checkpoints at step 0 and the final step
(`chk00000`, `chk<final>`: needed for particle identity across rank counts;
upstream's test command disables checkpoints -- I/O only).

## Validation (`validate.sh`, pre-fixed criteria)

Per case, for the N-GPU run:

1. **Completeness**: exit code 0 (timeout -> FAIL), `plt00000` + final plotfile,
   runlog reaches `max_step`, every plotfile variable finite (`amrex_fextrema`
   through `l3_check.require_finite`), DM particle count == IC count (exact).
2. **Official regression comparison at upstream's tolerance** (nightly GPU
   suite: `fcompare -n 0 --rel_tol 2e-10 --abort_if_not_all_found`): N=1 vs a
   second independent 1-GPU run (same-configuration reproducibility, what the
   nightly test measures); N>1 vs the 1-GPU plotfile of the same binary/deck.
   Particles: AMReX's `particle_compare` needs identical headers (incl.
   `next_id` and per-file layout, i.e. the same rank count) and returns 0 even
   when it prints "FAIL - Particle data headers do not agree" -> across rank
   counts `nyx_particle_compare.py` matches particles by their exact t=0
   position (chk00000 -> chk<final> by (id,cpu) within a run) and reports
   particle_compare's abs/rel norms per component at the same tolerance.
3. **Cross-backend reference**: CPU-profile binary (Nyx_GPU_BACKEND=NONE, same
   Nyx/AMReX/deck), rel_tol **1e-8** fixed before any run (FMA/libm/reduction
   differences host vs device over 10 steps).
4. **Conservation**: comoving baryon mass `sum(density*dV)` (`amrex_fvolumesum`)
   plt00000 -> final, |dM/M| <= 1e-9; DM count exact.

No tolerance was changed after seeing results.

## Results (dgx003, 2026-09-06; logs under `build/level3/nyx/cuda132-gcc133-adiabatic/run/`)

Build: AMReX 26.09 CUDA + Nyx = 121 s (nyx 115 s) at -j32; CPU profile 13 s
(after AMReX); `cuobjdump` sm_100 only; cudart static (`l3_binary_backend_check`
accepts embedded CUDA ELF).

| GPUs | Case | fcompare vs reference (max rel err, tol 2e-10) | DM particles (rel, tol 2e-10) | vs CPU (tol 1e-8) | mass, count | audit | Result |
|---|---|---|---|---|---|---|---|
| 1 | minisb | rerun: 3.0e-11 (Temp) | 6.5e-16 | 3.9e-11 / 7.0e-16 | exact / 32,686 | 1 verified | **VALIDATED_PASS** |
| 1 | lya_adiabatic | rerun: 6.5e-14 | 7.9e-16 | 9.6e-14 / 7.0e-16 | exact / 32,768 | 1 verified | **VALIDATED_PASS** |
| 2 | minisb | vs 1 GPU: 3.3e-12 | 6.5e-16 | 3.6e-11 / 6.9e-16 | exact | 2 verified | **VALIDATED_PASS** |
| 2 | lya_adiabatic | vs 1 GPU: 6.4e-14 | 8.8e-16 | 8.7e-14 / 7.0e-16 | exact | 2 verified | **VALIDATED_PASS** |
| 4 | minisb | vs 1 GPU: 1.38e-10 | 1.3e-15 | 1.77e-10 / 1.7e-15 | exact | 4 verified | **VALIDATED_PASS** |
| 4 | lya_adiabatic | vs 1 GPU: 6.7e-14 | 6.1e-16 | 9.3e-14 / 7.0e-16 | exact | 4 verified | **VALIDATED_PASS** |

(The `Temp` field carries the largest relative differences; MiniSB at 4 GPUs is
the closest to the official tolerance, 1.4e-10 vs 2e-10.)

Scaling runs (COMPLETED, 10 steps, not correctness-validated; timing = Nyx
"Run time", includes IC read and I/O; too short for a performance statement):

| Mode | Case | 1 GPU | 2 GPU | 4 GPU |
|---|---|---|---|---|
| strong | lya_adiabatic 64^3 (science IC, adiabatic) | 2.06 s | 2.35 s | 1.93 s |
| strong | scaling_synthetic 256^3 (synthetic) | 39.5 s | 41.5 s | 29.7 s |
| weak | scaling_synthetic 64^3 cells/rank (synthetic) | 1.63 s | 2.20 s | 3.52 s |

No science IC larger than 64^3 is shipped (256^3/1024^3 exist only at OLCF
paths); meaningful strong scaling of a science case needs such an IC
(**blocked by data availability**, documented, not worked around).

Dry-runs (`HPCPERF_DRY_RUN=1`, `HPCPERF_NODES` hypothetical): 8 GPUs (2 nodes)
planned for all three decks; 40/80 GPUs planned for `scaling_synthetic` weak
(40/80 boxes) and 40 GPUs for `lya_adiabatic` strong (64 boxes over 40 ranks =
imbalanced, reported); **refused** for `minisb` (8 boxes) and for 80 ranks on
the 64-box strong deck -- as designed. Multi-node remains BLOCKED/UNVERIFIED
on this site (launcher note); HIP untested (no ROCm).

## Status of the heating/cooling variant

`build.sh` with `HPCPERF_NYX_HEATCOOL=YES` builds SUNDIALS 7.2.1 (ENABLE_CUDA,
index 32, fused kernels -- Nyx's own `NyxSetupSUNDIALS.cmake` options) and AMReX
with `AMReX_SUNDIALS=ON`, then Nyx with `Nyx_HEATCOOL=YES`; `lya_heatcool` runs
`Exec/LyA/inputs` / `inputs.rt` as shipped. **Not executed in this round** (see
SECOND_BATCH_STATUS.md for the current state); the adiabatic results above make
no claim about heating/cooling.

## Files

`fetch.sh` (pinned clones + SHA checks), `build.sh` (staged, profile-aware,
fingerprinted), `run.sh` (cases/modes, decomposition guard, manifest),
`validate.sh` + `nyx_particle_compare.py` (criteria above), this README.
