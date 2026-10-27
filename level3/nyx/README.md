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
| Profiles | `cuda132-gcc133-adiabatic` (Nyx_HEATCOOL=NO), `cpu-gcc133-adiabatic` (Nyx_GPU_BACKEND=NONE reference + AMReX plotfile tools + particle_compare), `cuda132-gcc133-heatcool` + `cpu-gcc133-heatcool` (SUNDIALS 7.2.1 CVODE, see "Heating/cooling variant") |
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
| `lya_heatcool` | `Exec/LyA/inputs.rt` as shipped (heat_cool_type=11, CVODE) | the heating/cooling LyA workload -- `heatcool` profile (SUNDIALS 7.2.1) | smoke (validation verdict STATE_AND_PARTICLES_PASS; I_R_CHECK_PENDING, see below) / strong (`Exec/LyA/inputs` 64^3, completeness record) |

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
   Box layouts (2026-09-08): the wrapper checks each plotfile's BoxArray geometrically
   (physical extents -> cell boxes through `prob_lo`/`dx`: inside the level domain, no
   overlap, level 0 covers the domain exactly). Identical box sets in any order are
   compared normally; a *legal re-blocking* (the same cells partitioned differently,
   e.g. another `amr.max_grid_size`) is reported as **UNSUPPORTED_LAYOUT** (comparator
   exit 4, `validate.sh` exit 4: the comparison is not performed, the case is neither
   PASS nor FAIL -- cross-BoxArray comparison would need `fcompare --allow_diff_grids`
   and is not enabled here); a missing box, an overlap, a coverage deficit or different
   covered cell sets at a level are STRUCTURAL failures (FAIL). All runs in this README
   use the run.sh-fixed `max_grid_size`, so every real comparison so far had identical
   box arrays (the 2026-09-08 offline re-check of the 2026-09-07 regression plotfiles
   reports "identical box array" for every pair).
   Particles: AMReX's `particle_compare` needs identical headers (incl.
   `next_id` and per-file layout, i.e. the same rank count) and returns 0 even
   when it prints "FAIL - Particle data headers do not agree" -> across rank
   counts `nyx_particle_compare.py` matches particles by their exact t=0
   position (chk00000 -> chk<final> by (id,cpu) within a run) and reports
   particle_compare's abs/rel norms per component at the same tolerance.
3. **Cross-backend reference**: CPU-profile binary (Nyx_GPU_BACKEND=NONE, same
   Nyx/AMReX/deck), rel_tol **1e-8** fixed before any run (FMA/libm/reduction
   differences host vs device over 10 steps). **Since 2026-10-27 the default path runs no
   CPU Nyx**: the run is compared with the frozen B200 GPU reference
   `reference/<case>.<mode>.json` at `(1e-8 - d0)/(1 + d0)` (heat/cool `(5e-5 - d0)/(1 + d0)`),
   see "Frozen B200 reference (check [3])" below; the live CPU reference runs only offline with
   `HPCPERF_CPU_REFERENCE=1` (unchanged code path).
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

**Joint-HEAD regression, 2026-09-07 (code state `fc4d2a1`; run trees
`build/level3/nyx/<profile>/run.regress-fc4d2a1/`, logs
`build/level3/nyx/regress-fc4d2a1/validate_full.hc{NO,YES}.np{1,2,4}.log` +
`queue_nyx.log`)**: the same decks re-run from scratch with the strict wrapper,
new run ids, the historical `run/` trees untouched. Six `validate.sh` calls:

| GPUs | Cases | vs rerun / 1-GPU (tol) | vs CPU (tol) | DM particles | audit | validate.sh exit / class |
|---|---|---|---|---|---|---|
| 1 | minisb, lya_adiabatic | 1.6e-11, 6.1e-14 (2e-10) | 5.2e-11, 8.7e-14 (1e-8) | <= 9.8e-16, counts exact | 1 verified | 0 **PASS** |
| 2 | minisb, lya_adiabatic | 2.2e-11, 6.5e-14 (2e-10) | 7.4e-11, 8.7e-14 (1e-8) | exact | 2 verified | 0 **PASS** |
| 4 | minisb, lya_adiabatic | 1.2e-11, 6.0e-14 (2e-10) | 6.5e-11, 9.6e-14 (1e-8) | exact | 4 verified | 0 **PASS** |
| 1 | lya_heatcool | state 1.4e-13 (5e-5); I_R rel 1.24 (reported) | state 1.9e-13; I_R rel 1.26 | 8.8e-16, exact | 1 verified | 3 **STATE_AND_PARTICLES_PASS; I_R_CHECK_PENDING** |
| 2 | lya_heatcool | state 1.4e-13; I_R rel 1.40 | state 1.9e-13; I_R rel 1.26 | exact | 2 verified | 3 **STATE_AND_PARTICLES_PASS; I_R_CHECK_PENDING** |
| 4 | lya_heatcool | state 1.2e-13; I_R rel 1.40 | state 1.9e-13; I_R rel 1.46 | exact | 4 verified | 3 **STATE_AND_PARTICLES_PASS; I_R_CHECK_PENDING** |

The three heat/cool calls are counted as PENDING, not as passes (exit-code classes:
`level3/tools/l3_verdict.py`).

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
  4-GPU and CPU runs, max|dI_R| = 0.27-0.33 (relative 0.97-1.46 over the two
  campaigns), spread over 78-100 % of the cells; within the 10 steps the differences
  are uncorrelated with the (1e-13 relative) state differences (corr(dI_R, d rho_e)
  = -0.006; the observed dI_R would correspond, through the predictor coupling
  dt a_half dI_R / a_end^2, to a change of rho_E of order 8e6 code units if it acted
  fully, and nothing of the kind is present in the final state: rho_E agrees to
  1e-15). This is an observation over 10 steps of a deck in which the net
  heating/cooling is negligible. **It does not show that the I_R variation has no
  effect on later steps**: I_R is consumed by the next step's hydro predictor and in
  the SDC branch it is the increment applied to (rho e)/(rho E), so it is a
  state-carried coupling term with an unresolved reproducibility question -- not a
  diagnostic field that has been shown to be irrelevant. Whether the variability
  comes from the CVODE per-cell solution at its tolerance (`sundials_reltol/abstol`
  1e-4 defaults), from the batched host-device buffers (`rho_init_vode`,
  `rhoe_src_vode`, `e_src_vode`) or from something else is **not identified** from
  the plotfiles; a race or an uninitialised read is neither shown nor excluded.
- Diagnostic estimate (not a verified error bound): if one asks how large a
  per-cell change of I_R could be before the next step's energy update moves out of
  the state tolerance, the linear one-step coupling gives
  `|dI_R|_max ~ tol_state * max_cells|a^2 rho e| / (dt * a_half)`
  with tol_state = 5e-5 (the heat/cool state tolerance), `max|a^2 rho e|/(dt a_half)`
  = 3.7e2 read from the saved plt00010 of the 1-GPU run (Nyx code units: rho e in
  comoving M_sun/Mpc^3 (km/s)^2, dt the runlog `dt` of step 10 = 2.99e5 code time
  units, a = 0.01094 at z = 90.4, a_half the mid-step value), i.e. **~1.9e-2 in the
  units of I_R (energy density per code time)**. Assumptions: linear one-step
  coupling only, no amplification over later steps, the maximum-|a^2 rho e| cell
  as the worst case, and a single tolerance for all cells. The observed
  |dI_R| = 0.28-0.33 exceeds this estimate ~15x, so no criterion derived from the
  state tolerance accepts the field; and "exclude it and PASS" is not a criterion.
  The estimate is recorded to show why no tolerance is set, not as a proven bound.
  **Correction (2026-10-27, see "I_R investigation" below): this estimate used the runlog's
  `dt` column (2.99e5, in years) instead of the code-unit step dt = 3.06e-7; with the code dt
  the scale `max|a^2 rho e|/(dt a_half)` is 5.7e14 and the estimate becomes ~2.9e10, i.e. the
  observed |dI_R| ~ 0.3 is ~11 orders of magnitude below it, not 15x above.** The judgement of
  I_R is unchanged (still reported, not gated, I_R_CHECK_PENDING).
- Consequence: the validator reports `I_R` (parsed, finite, not gated) and
  downgrades the case to I_R_CHECK_PENDING (exit 3). The heat/cool deck is
  therefore **not** in the count of full correctness passes and its timings do not
  enter a performance summary. Resolving it needs, at the original configuration,
  CPU-vs-CPU and GPU-vs-GPU repeat runs of I_R, a run with per-step plotfiles (or
  the SDC_IR/hydro-source intermediates) to attribute the variation to a term, an
  independent tolerance comparison, and a regime (more steps, lower z) where the
  reaction term is physical. None of this was run in the 2026-09-07/08 rounds;
  historical run directories are untouched. (The repeat runs, per-step plotfiles and
  the physical regime were run on 2026-10-27: next section.)

### I_R investigation (2026-10-27; no change to how I_R is judged)

Runs: this worktree's builds from the frozen source (profiles `cuda132-gcc133-heatcool`,
`cpu-gcc133-heatcool`), the registered deck through the unchanged `run.sh`, 1 rank, GPU 0;
run trees `build/level3/nyx/<profile>/run.irstudy-*` (per-step plotfiles: a copy of the derived
deck with `amr.plot_int = 1`, launched through the common launcher). Comparisons with fcompare's
own norm-0 definition (`||B - A||_inf / ||A||_inf`) on the plotfile data.

| Pair (10-step deck, plt00010) | I_R max abs / rel | cells with different I_R | worst state field | bitwise-identical fields |
|---|---|---|---|---|
| GPU vs GPU, 4 repeats (6 pairs), same binary/deck/ranks | 0.265-0.297 / 1.13-1.27 | 77-80 % | Temp 1.1-1.3e-13 | Ne (identically 0), particle_count |
| GPU vs GPU with `nyx.sundials_atomic_reductions=0` (2 runs) | 0.29 / 1.24 | 86 % | Temp 1.2e-13 | the same two |
| CPU vs CPU, 3 repeats | 0 / 0 | 0 | 0 | **all 16** |
| GPU vs CPU | 0.28 / 1.28 | 99.9 % | Temp 1.9e-13 | Ne, particle_count |
| GPU vs GPU, **plt00000** (before the first step) | 0 (I_R = 0) | 0 | rho_e 3.7e-14, Temp 4.6e-14, density 3.8e-16 | Ne, I_R, particle_count |

What I_R is in this deck: `I_R = [a_end^2 (rho e)_out - (a^2 (rho e)_orig + dt A_rhoe)] / (dt a_half) - a_end^2 reset / (dt a_half)`
(`f_rhs_struct.H`), with dt the code-unit step (3.06e-7 at step 10; the runlog's `dt` column is in
years). The cancelling terms are `a_end^2 |rho e| / (dt a_half)` = 3.0e14-5.7e14, so max|I_R| = 0.22-0.28
is 3.8-5.0e-16 of them (1.7-2.2 machine epsilon); per cell |I_R| is a median 0.5 and at most 3.1-4.1
units in the last place of the largest term (GPU and CPU alike); with per-step plotfiles max|I_R| is
1.6-2.8 eps x scale at **every** step 1-10, and the I_R difference between any two runs is 1.1-1.7
relative from step 1 on, while the state difference grows smoothly from 7e-14 to 1.5e-13. The
increment I_R adds to (rho e)/(rho E), `dt a_half I_R / a_end^2`, is at most 6.5e-6 absolute =
5.5e-16 of rho e. In this deck (z 100 -> 90, Ne identically 0, temperature following the adiabat)
the net heating/cooling is below the floating-point resolution of the energy bookkeeping: **I_R is
rounding residue, not a resolved physical quantity.**

Answers:
- Run-to-run variation: the CPU build is bitwise reproducible (all fields, also after 354 steps).
  The GPU build is not: its state already differs at plt00000 (initialisation, before any
  heating/cooling call -- consistent with atomic particle-to-mesh deposition; the adiabatic decks
  show the same, MiniSB rerun ~2e-11) and the CVODE atomic reductions are not needed for it (the
  `sundials_atomic_reductions=0` repeats differ the same way). Last-bit differences of the inputs
  re-draw a rounding residue completely, hence O(1) relative differences of I_R with ~1e-13 state
  differences. No race or uninitialised read is visible (CPU bitwise, GPU differences explained by
  the state nondeterminism); a GPU-side race is not excluded by a sanitizer run (none was made).
- A frozen-GPU-reference comparison of I_R in the 10-step deck does **not** make sense with any
  relative tolerance: the frozen value is one sample of rounding noise (a GPU run differs from its
  own rerun by 1.1-1.3 relative). A meaningful criterion would be reference-free and absolute, on the
  rounding scale (e.g. |I_R| <= k eps max a_end^2|rho e|/(dt a_half), measured k <= 2.8), or the
  acceptance would move to a regime where I_R is physical (below). Both are owner decisions; nothing
  was changed: I_R stays reported-not-gated and the case stays I_R_CHECK_PENDING (exit 3).
- The other heat/cool fields (10 steps): every state field agrees to <= 1.9e-13 (GPU-GPU <= 1.3e-13,
  GPU-CPU <= 1.9e-13, CPU-CPU bitwise); Ne is identically 0 in every run; DM particles ~1e-15.
- Physical regime, upstream's own test length (354 steps, plt00354, z = 2; `HPCPERF_NYX_STEPS=354`,
  GPU run time 15.5 s, CPU 213-227 s): max|I_R| = 5.7e18 (upstream reports ||I_R|| ~ 6e18 there).
  GPU vs GPU: I_R 1.4e-8, Temp 8.6e-7, rho_e 5.2e-8, Ne 7.3e-9 (upstream's nightly: I_R 4.2e-8,
  Temp 2.8e-5 vs its GPU benchmark) -- a frozen-GPU-reference comparison at 5e-5 is meaningful there.
  CPU vs CPU: bitwise. **GPU vs the CPU profile as built (default `nyx.sundials_use_tiling=1`):
  Temp 6.3e-3, rho_e 5.3e-4, I_R 4.5e-4, Ne 1.0e-5 -- above 5e-5.** The CPU build integrates the
  vectorised CVODE system per MFIter tile (16x8x8 = 1024 cells, default tile (1024000,8,8)), the GPU
  build per box (16^3 = 4096 cells); CVODE's step-size and error control (reltol = abstol = 1e-4) is
  shared over the batch, so the two solve the ODE to its tolerance along different step sequences.
  With `nyx.sundials_use_tiling=0` on the CPU (a run-time parameter, same source) the GPU-CPU
  difference at plt00354 falls to Temp 6.6e-6, rho_e 6.2e-7, I_R 5.5e-8 (inside 5e-5); at 10 steps
  the tiling makes no difference (bitwise identical CPU results). Consequence, recorded, not acted
  on: a CPU-validated frozen reference for the 354-step regime would need the CPU run with
  `nyx.sundials_use_tiling=0` (d0 = 6.6e-6 -> budget (5e-5 - d0)/(1 + d0) = 4.3e-5); with the CPU
  build's default the CPU reference itself is outside upstream's 5e-5.

Heat/cool strong scaling (`Exec/LyA/inputs` 64^3, `amr.max_grid_size=16` -> 64
boxes, 10 steps) and the 8/40/80-GPU dry-runs: 8 ranks planned (2 hypothetical
nodes, balanced); 40 ranks planned and flagged **IMBALANCED (64 boxes over 40
ranks)**; 80 ranks **refused** ("80 ranks requested but ... only 64 boxes -- a rank
without work is never launched silently"). Timings of the 1/2/4-GPU strong runs:
`SECOND_BATCH_STATUS.md`.

## Frozen B200 reference (check [3], 2026-10-27)

Project policy: no CPU recomputation of the scientific result on a default path. Check [3] used to run
the CPU-profile Nyx (1 rank) whenever no matching CPU run existed -- an automatic CPU fallback on every
`validate.sh` and registry check. By default it now compares the run with a frozen B200 GPU reference;
[1], [2], [4], the I_R handling and the exit codes 0/1/3/4 are unchanged.

| `HPCPERF_CPU_REFERENCE` | check [3] |
|---|---|
| unset / `0` (default: plain `validate.sh`, registry check, timing never calls it) | frozen B200 reference `reference/<case>.<mode>.json`; missing / corrupted / other workload -> **FAIL**, never a CPU run; no CPU run tree is created |
| `1` (explicit, offline) | the previous [3] unchanged (live CPU reference, reused while deck sha256 and steps match) |
| `1` + `HPCPERF_NYX_FREEZE_REFERENCE=1`, `HPCPERF_GPUS=1` | as `1`, then `nyx_reference.py freeze` of each case whose [3] agreed (refuses an uncommitted `level3/nyx`, an existing reference unless `HPCPERF_NYX_FREEZE_REPLACE_REASON` is set, a non-CUDA or multi-rank run, a report of another pair) |

What is frozen (`nyx_reference.py freeze`): the 1-GPU run's `plt<final>/` (whole plotfile) plus
`chk00000/DM` and `chk<final>/DM` (what [3] compares) as a deterministic tar in the content-addressed
golden store (`tools/golden/hpcperf_golden.py store-put`; `$HPCPERF_GOLDEN_STORE`, default
`<repo>/.golden-store`, never in git); the JSON (schema `hpcperf-nyx-frozen-reference-1`, committed) holds
the workload identity (case, mode, steps, grid, decomposition, upstream deck and IC sha256), every archive
member's size and sha256, d0 per variable / particle component, the budget, and the provenance (GPU
name/UUID/driver, CUDA, git HEAD, binary and fingerprint sha256 of the GPU and CPU runs, sha256 of the [3]
reports and the validate.sh log, tool hashes). `fetch` (the default [3]) checks the workload, gets the
archive with `store-get` (sha256-verified), checks every member against the JSON, refuses a JSON whose
budget is not the one derived from its own tol/d0 or whose d0 is not finite in [0, tol), refuses a JSON that is not the
committed, unmodified file (in an agent workspace: not byte-identical to the canonical repository's committed one;
`HPCPERF_GOLDEN_ALLOW_UNCOMMITTED_REFERENCE=1` is for test harnesses and is printed loudly), and extracts it next to
the run. Freeze mode freezes a case only when every other check of that case ([1], [2], [4]) passed in the same run.

Tolerance budget. fcompare `-n 0` reports per variable and level `||B - A||_inf / ||A||_inf`, A = the first
plotfile (the reference): the maximum difference normalised by the **reference field's maximum** over the
level, not a per-cell relative error (`Tools/Plotfile/fcompare.cpp`: `aerror = norm0(B - A)`,
`rerror = aerror / norm0(A)`; a field with `norm0(A) = 0` is the zero-reference case, judged exactly by
`nyx_fcompare_check.py`). nyx_particle_compare.py uses the same max-normalised form per component. With the
CPU reference C, the frozen GPU output G, `d0 = ||G - C|| / ||C||` and a candidate X with
`r = ||X - G|| / ||G||`:
`||X - C|| <= ||X - G|| + ||G - C|| = r ||G|| + d0 ||C||` and `||G|| <= (1 + d0) ||C||`, so
`||X - C|| / ||C|| <= r (1 + d0) + d0`, which is `<= tol` whenever `r <= (tol - d0) / (1 + d0)` -- a run
that passes against G passes the original criterion against C (for zero-reference fields both G and C are
identically 0, and the exact rule is unchanged). One bound per comparison: d0 = the largest d0 over the
gated variables (strictest), inflated by 1e-9 relative because fcompare prints 10 significant digits;
I_R (diagnostic) and identically-zero fields are not part of it.

The frozen references, their measured d0 and budgets: "Frozen references (check [3])" below, recorded with the reference files.

What can give another verdict than before: a run of another step count (`HPCPERF_NYX_STEPS`), deck or IC
than the frozen one is FAIL instead of a fresh CPU comparison; another `amr.max_grid_size`
(`HPCPERF_NYX_MGS`) makes the [3] pair a legal re-blocking -> UNSUPPORTED_LAYOUT (exit 4) where the CPU run
used the same layout before; a missing golden store (fresh clone, other machine) is FAIL until the archive
is provided; the budget is stricter by d0 (e.g. MiniSB 9.955e-9 instead of 1e-8). Multi-GPU runs compare
with the 1-GPU frozen reference (not exercised in this round: 1 GPU only). HIP would compare with the B200
reference (untested).

## Files

`fetch.sh` (pinned clones + SHA checks), `build.sh` (staged, profile-aware,
fingerprinted), `run.sh` (cases/modes, decomposition guard, manifest),
`validate.sh` + `nyx_fcompare_check.py` + `nyx_particle_compare.py` (criteria above),
`nyx_reference.py` + `reference/*.json` (frozen B200 reference of check [3]), this README.

<!-- hpcperf:source-section:begin -->
## Source distribution (frozen source artifact, scheme 3, 2026-09-10)

The application source is not in git and not read from `_upstream/`: `tools/prepare_benchmark.sh level3 nyx` materializes the frozen source artifact (`<app>[-<variant>]-<source_version>.tar.zst`, found in the local content-addressed cache `.artifacts/sha256/` or downloaded from the immutable URL recorded in `provenance/source.lock*.yaml` once published; `--artifact FILE` for a local copy) into `src/` (+ `deps/`), the only source `build.sh`/`run.sh`/`validate.sh` use. Archive size + sha256 and `source_tree_sha256` are verified before anything is placed. Identity, patch series, licenses, redistribution status and the equivalence proof against the tree the results above were validated from are under `provenance/` (`source.lock*.yaml`, `patch_series*.txt`, `original_vs_baseline*.diff`, `LICENSES*.md`, `equivalence*.md`, `LOC*.md`); `benchmark.yaml` is the machine-readable contract (entries, inputs, references, identity). The benchmark does not prescribe which part of the source an optimization agent may modify; the integrity layer only protects the harness and the validation assets. Remote status: see `level3/SOURCE_ARTIFACTS.md`.

| variant | artifact | source version | compressed / uncompressed | entries | source_tree_sha256 | archive sha256 | upstream | patches (pre-applied) | redistribution | equivalence | remote | LOC app-owned / bundled deps / benchmark deps / tests / total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| - | `nyx-hpcperf-l3-v1.tar.zst` | hpcperf-l3-v1 | 75.7 MB / 187.6 MB | 5933 | `71b64ddffc1e5c5fe9cd49173aba10d955f26082c6079a3a4548f2d9a6002120` | `218ba208e5d270abbc880a6f50fccd167f6f144361e3ef6e68e2677abd7dff36` | 26.09 `e06eabc1b9db` | none | cleared | src: EQUIVALENT, deps/amrex: EQUIVALENT | REMOTE_FETCH_VERIFIED | 32330 / 595317 / 366200 / 0 / 994298 |

LOC = cloc 2.06 code lines of the materialized tree (no blank/comment lines, documentation and data excluded); source-ownership categories from `provenance/source.lock*.yaml` (`source_scope`, descriptive metadata written at freeze time). Dependencies are counted per benchmark, so totals overlap across benchmarks that ship the same dependency. The validated results recorded above were produced from trees proven content-equivalent to this artifact (`provenance/equivalence*.md`); they are not re-run by the migration.
<!-- hpcperf:source-section:end -->
