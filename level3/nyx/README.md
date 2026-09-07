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
2. **Official regression comparison** through `nyx_fcompare_check.py`, a strict
   wrapper around AMReX's `fcompare -n 0 --rel_tol T --abort_if_not_all_found`
   (both Headers parsed and compared -- variable set, dimension, levels, time,
   domain, cell sizes, box arrays; `fextrema` on both plotfiles for raw finiteness;
   exactly one parsed table row per variable and level, no message rows, no
   duplicates, nothing dropped; a field that is identically zero in the reference
   must be identically zero in the test; parser and tool exit status must agree):
   N=1 vs a second independent 1-GPU run (same-configuration reproducibility, what
   the nightly test measures); N>1 vs the 1-GPU plotfile of the same binary/deck.
   Tolerance provenance (official GPU nightly reports at
   `ccse.lbl.gov/pub/GpuRegressionTesting/Nyx/`, read 2026-09-05/06): MiniSB compares
   with `--rel_tol 2e-10` (used as is); LyA-adiabatic compares with `--rel_tol 5e-09`
   -- this validator applies the MiniSB value 2e-10 to both adiabatic decks, a
   **project choice 25x stricter than upstream's** for LyA-adiabatic.
   History: the first validator version parsed fcompare's table with a loose awk
   filter that dropped `inf`/`nan` rows and message rows and accepted any `rc=1`
   as "over tolerance"; a reviewer's CPU-only injection (Ne `abs=1 rel=inf`, a
   missing-variable warning, a NaN row) passed it. Replaced on 2026-09-07 by the
   strict wrapper (negative tests in `level3/tools/tests/test_nyx_validator.sh`);
   every earlier verdict was re-derived offline from the saved plotfiles with the
   strict wrapper (see "Results").
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

For the two adiabatic decks no tolerance was changed after seeing results (2e-10 and
1e-8 were in `validate.sh` before its first execution, 2026-09-05 23:58 UTC vs 2026-09-06
00:02 UTC). For the heat/cool deck this is **not** true -- see the "I_R" section below.

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

## Heating/cooling variant (profile `cuda132-gcc133-heatcool`)

Staged build: SUNDIALS 7.2.1 (Nyx's pinned submodule commit; `ENABLE_CUDA`, index
size 32, `SUNDIALS_BUILD_PACKAGE_FUSED_KERNELS=ON` as Nyx's own
`NyxSetupSUNDIALS.cmake`; CVODE **and ARKODE** because AMReX 26.09's
`find_package(SUNDIALS)` requires the `arkode` component) -> AMReX 26.09 with
`AMReX_SUNDIALS=ON` -> Nyx `Nyx_HEATCOOL=YES` (CVODE vectorized, `heat_cool_type
11`). 310 s at -j16 (SUNDIALS 14 s, AMReX 134 s, Nyx 157 s); `cuobjdump` sm_100;
CPU reference profile `cpu-gcc133-heatcool` built the same way (CPU SUNDIALS).

**SUNDIALS independent probe first** (`sundials_probe.sh`: SUNDIALS' own example
regression tests, `SUNDIALS_TEST_ENABLE_DEV_TESTS`, CUDA sm_100, CUDA 13.2):
6 CUDA examples/tests, **5 pass** (cvAdvDiff_kry_cuda, _managed,
cvAdvDiff_diag_cuda x3); `cvRoberts_block_cusolversp_batchqr` **fails only in the
integrator statistics** printed with a 10 % integer allowance (nni 805 vs 823,
ncfn 5 vs 4, netf 33 vs 31; the solution values agree at the test's 4-digit
precision). That example uses the cuSolverSp batched-QR linear solver; **Nyx's
HEATCOOL path uses CVODE with `CVDiag`** (`Source/HeatCool/integrate_state_vec_3d.cpp`),
not cuSolverSp, so the failing example is off Nyx's path. Recorded as-is
(`.deps/level3/nyx/cuda132-gcc133-heatcool/logs/sundials-ctest.log`).

**Workload `lya_heatcool`**: `Exec/LyA/inputs.rt` exactly as shipped
(heat_cool_type 11, UVB table TREECOOL_middle, 32^3, 32.nyx IC), 10 steps; "Integrating
heating/cooling method ... Vectorized CVODE" in the log; 1.3 s on 1 GPU.

**Validation** (same script): same-config rerun (1 GPU), 1-GPU reference (2/4 GPU)
and CPU-heatcool reference all at **5e-5**. Provenance and history, stated plainly:
upstream's GPU nightly "LyA" test compares `plt00354` (354 steps of `inputs.rt`) with
`fcompare --rel_tol 5e-05` (its own maximum 2.8e-5 in Temp; it also compares `I_R`,
relative error 4.2e-8 at that point where ||I_R|| ~ 6e18). The project's 10-step
smoke deck (`plt00010`, z ~ 100 -> 90) is a different configuration. The first
heat/cool validation here (2026-09-06 01:20 UTC) was run with the adiabatic
tolerances 2e-10 / 1e-8 and FAILED; the official 5e-5 was looked up (01:25 UTC),
adopted (01:27 UTC) and the runs repeated -> PASS for every state field. The
`I_R` treatment (excluded from the gate) was introduced at the same time and is a
project decision with no upstream counterpart. **Every state variable (density,
momenta, rho_E, rho_e, Temp, Ne, phi_grav, grav_*, pressure, particle_*) agrees to
<= 1.6e-13 relative** (rerun, 2-GPU, 4-GPU and vs CPU alike; `Ne` is identically
zero in every run and is checked exactly), DM particles agree to ~1e-15, baryon
mass exact, counts exact, all ranks' GPUs verified. Verdict for the heating/cooling
LyA deck: **STATE_AND_PARTICLES_PASS at 1/2/4 GPUs, I_R_CHECK_PENDING** -- not a full
regression PASS (validate.sh exits 3 for this case). (Run dirs:
`build/level3/nyx/cuda132-gcc133-heatcool/run/`.) The adiabatic results above and
this section are separate claims; neither is presented as the other.

### I_R -- what it is, why it is not accepted by a tolerance yet

From the fixed source (Nyx `e06eabc1`, `Source/HeatCool/f_rhs_struct.H`,
`Source/Hydro/sdc_hydro.cpp`, `Source/Initialization/Nyx_setup.cpp`):

- `I_R` is the single component of the `SDC_IR_Type` state (`Nyx_setup.cpp:304`),
  initialised to 0 (`Nyx_initdata.cpp:335`), written to plotfiles like any state.
  It is computed once per cell per step in `ode_eos_finalize_struct` after the
  CVODE solve, as the reaction integral net of the hydro source:
  `I_R = [a_end^2 (rho e)_out - (a^2 (rho e)_orig + dt A_rhoe)] / (dt a_half)`
  (comoving energy density per unit time). It is used again: at the next step it
  is added to the external source for the hydro predictor (`sdc_hydro.cpp`, `IR_tmp`
  added to `ext_src_old` before `construct_hydro_source`, subtracted afterwards),
  and in the SDC branch it is the increment applied to (rho e)/(rho E). So it is a
  state-carried coupling term, recomputable from the state before/after the ODE and
  the hydro source -- not an independent physical observable.
- Magnitude in the 10-step deck (offline reading of the saved plotfiles, 2026-09-07):
  max|I_R| = 0.28, mean|I_R| = 0.044 at plt00010 (0 at plt00000); the terms it is
  built from are O(a^2 rho e / (dt a_half)) ~ 3.7e2, i.e. `I_R` is a ~1e-3 residual of
  nearly cancelling terms (z ~ 90: net heating/cooling negligible). Upstream's
  354-step comparison is in a regime where ||I_R|| ~ 6e18 (physical reaction term
  dominant) -- there a relative tolerance is meaningful; at step 10 it is not.
- Reproducibility: between the 1-GPU run and its identical rerun, and vs the 2-GPU,
  4-GPU and CPU runs, max|dI_R| = 0.27-0.33 (relative 0.97-1.17), spread over
  78-100 % of the cells; the differences are uncorrelated with the (1e-13 relative)
  state differences (corr(dI_R, d rho_e) = -0.006; dt a_half dI_R / a_end^2 would be
  a 8e6 change of rho_E, nothing of the kind is present: rho_E agrees to 1e-15).
  Whether the variability comes from the CVODE per-cell solution at its tolerance
  (`sundials_reltol/abstol` 1e-4 defaults), from the batched host-device buffers
  (`rho_init_vode`, `rhoe_src_vode`, `e_src_vode`) or from something else is
  **not identified** from the plotfiles; a race or an uninitialised read is
  neither shown nor excluded.
- Consequence: an energy-consistent absolute criterion of the form
  |dI_R| <= tol_state * max(a^2 rho e)/(dt a_half) (5e-5 * 3.7e2 = 1.9e-2) is
  **violated** by the observed 0.29, so no criterion derived from the state
  tolerance accepts the field, and "exclude it and PASS" is not a criterion. The
  validator therefore reports `I_R` (parsed, finite, not gated) and downgrades the
  case to I_R_CHECK_PENDING. Resolving it needs a run with per-step plotfiles (or
  the SDC_IR/hydro-source intermediates) to attribute the variation to a term, and
  a regime (more steps, lower z) where the reaction term is physical; neither was
  run in the 2026-09-07 round. Historical run directories are untouched.

Heat/cool strong scaling (`Exec/LyA/inputs` 64^3, `amr.max_grid_size=16` -> 64
boxes, 10 steps) and the 8/40/80-GPU dry-runs: 8 ranks planned (2 hypothetical
nodes, balanced); 40 ranks planned and flagged **IMBALANCED (64 boxes over 40
ranks)**; 80 ranks **refused** ("80 ranks requested but ... only 64 boxes -- a rank
without work is never launched silently"). Timings of the 1/2/4-GPU strong runs:
`SECOND_BATCH_STATUS.md`.

## Files

`fetch.sh` (pinned clones + SHA checks), `build.sh` (staged, profile-aware,
fingerprinted), `run.sh` (cases/modes, decomposition guard, manifest),
`validate.sh` + `nyx_particle_compare.py` (criteria above), this README.
