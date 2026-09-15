# Level 3 backend/profile isolation of generated state (2026-09-15)

**Rule.** One frozen source tree per benchmark, `level3/<app>/src/` (+ `deps/`), materialized from the
source artifact and independent of the GPU backend -- never copied per backend. Everything a build or a run
generates belongs to exactly one backend/profile and is never shared between profiles:

```
.deps/level3/<app>/<profile>/
  src/        build-side source copy -- only for applications whose build writes into its source tree
  build/      dependency builds (e.g. ExaCA's Kokkos, Nyx's AMReX/SUNDIALS, the CP2K toolchain)
  install/    install prefix, .hpcperf-l3-fingerprint, BUILD_INFO.txt
  logs/       configure/build/install logs
  cache/      caches (nekRS' OCCA JIT cache)
build/level3/<app>/<profile>/                    application build tree + run directories (<L3_RUN_SUBDIR>/...)
```

CUDA and HIP never share a mutable source copy, a CMake/build tree, an install prefix, a fingerprint, logs, a
cache or run-time dependency state. **Backend separation applies to generated state, not to source
duplication.** This is not a new benchmark architecture: source artifacts, `source_tree_sha256`, the release,
the scientific inputs and the validation policy are untouched.

## Profile naming

A profile must uniquely identify a configuration whose binaries/installs are not interchangeable, and it
must name its backend. Enforced by `l3_paths_profile <app> <profile> <backend>` (`level3/tools/l3_common.sh`):
some `-`/`.`-separated component of the profile is the backend name (`cuda`, `hip`, `cpu`) or the backend name
followed by a version (`cuda132`), and no component names another backend. A profile that fails the rule is
refused before any directory is created.

| Application(s) | Profile | Derivation |
|---|---|---|
| LAMMPS, SPARTA, WarpX, SPECFEM3D, ExaCA | `cuda`, `hip` | `l3_backend_profile <APP> <backend>`; override `HPCPERF_<APP>_PROFILE` |
| nekRS | `<variant>.<backend>`: `hypregpu.cuda`, `cpucoarse.cuda`, `cpucoarse.hip` (untested); `hypregpu.hip` does not exist (refused) | `l3_backend_profile NEKRS <backend> <variant>`; override `HPCPERF_NEKRS_PROFILE` |
| Nyx | `cuda132-gcc133-<adiabatic\|heatcool>`, `hip-<arch>-<variant>`, `cpu-gcc133-<variant>` | unchanged (`HPCPERF_NYX_PROFILE`) |
| CP2K, DFT-FE (GEOS, retired) | `cuda132-gcc142-ompi5010[-elpacpu]` | unchanged (`HPCPERF_<APP>_PROFILE`) |
| QMCPACK | `clang231-cuda132-offload` | unchanged (`HPCPERF_QMCPACK_PROFILE`) |

Existing toolchain-identity profiles are kept as they are; they are not reduced to a bare backend name.
`build.sh`, `run.sh` and `validate.sh` of an application derive the profile through the same helper, so a run
can only use the binary, install and fingerprint of the profile the build wrote. `run.sh` additionally
requires the profile's fingerprint to exist and to record the requested backend
(`l3_fingerprint_expect_backend`); an unbuilt profile has no fallback. A pre-migration shared install
(`.deps/level3/<app>/install`) is reported by `l3_paths_profile` and never read.

## Audit and migration matrix (main `e30077f` -> this branch)

"Before" is the layout on `main` after PR #6; "after" is this branch. `<model>` = `cuda|hip` as before.

| Application | Source layout (unchanged) | Backends supported / validated | Build tree before -> after | Build-side source copy before -> after | Install (+ fingerprint) before -> after | Logs / caches before -> after | Run/result tree before -> after | Helper before -> after | Backend-isolated before? | Migrated |
|---|---|---|---|---|---|---|---|---|---|---|
| LAMMPS | `level3/lammps/src` (one tree) | cuda, hip / cuda | `build/level3/lammps/<model>` -> `build/level3/lammps/<profile>` (same path for `cuda`) | none | `.deps/level3/lammps/install` shared by all backends -> `.deps/level3/lammps/<profile>/install` | `.deps/level3/lammps/logs` shared -> `<profile>/logs`; `<profile>/cache` (unused) | `build/level3/lammps/<model>/$L3_RUN_SUBDIR` -> `.../<profile>/$L3_RUN_SUBDIR` | `l3_paths` -> `l3_paths_profile` | build tree only; install/logs/fingerprint shared | yes |
| SPARTA | `level3/sparta/src` | cuda, hip / cuda | as LAMMPS | none | `.deps/level3/sparta/install` shared -> `<profile>/install` | shared logs -> `<profile>/logs` | as LAMMPS | `l3_paths` -> `l3_paths_profile` | build tree only | yes |
| WarpX | `level3/warpx/{src,deps/amrex}` | cuda, hip / cuda | as LAMMPS (AMReX built by the superbuild inside the build tree) | none | `.deps/level3/warpx/install` shared -> `<profile>/install` | shared logs -> `<profile>/logs` | `build/level3/warpx/<model>/$L3_RUN_SUBDIR/<case>.<mode>.npN` -> `.../<profile>/...` | `l3_paths` -> `l3_paths_profile` | build tree only | yes |
| SPECFEM3D | `level3/specfem3d/src` | cuda, hip / cuda | `build/level3/specfem3d/<model>` (run dirs only) -> `.../<profile>` | **`.deps/level3/specfem3d/src` shared and rewritten by every build** -> `.deps/level3/specfem3d/<profile>/src` | `.deps/level3/specfem3d/install/bin` shared -> `<profile>/install/bin` | shared logs -> `<profile>/logs` | `.../<model>/$L3_RUN_SUBDIR/<mode>.npN` -> `.../<profile>/...` | `l3_paths` (build + run) -> `l3_paths_profile` | no (a HIP build would have overwritten the CUDA build-side tree and install) | yes |
| nekRS | `level3/nekrs/src` (one tree per variant, materialized one at a time) | cuda, hip / cuda; per variant: hypregpu cuda only, cpucoarse cuda validated + hip untested | hypregpu `build/level3/nekrs/<model>`, cpucoarse `build/level3/nekrs/cpucoarse.<model>` -> `build/level3/nekrs/<variant>.<backend>` | hypregpu `.deps/level3/nekrs/src`, cpucoarse `.deps/level3/nekrs/cpucoarse/src` (each shared across backends) -> `.deps/level3/nekrs/<variant>.<backend>/src` | hypregpu `.deps/level3/nekrs/install`, cpucoarse `.deps/level3/nekrs/cpucoarse/install` (each shared across backends) -> `<variant>.<backend>/install` | logs `.deps/level3/nekrs[/cpucoarse]/logs` -> `<profile>/logs`; OCCA JIT cache `build/level3/nekrs/<tree>/cache` -> `.deps/level3/nekrs/<profile>/cache` | `<build tree>/$L3_RUN_SUBDIR/<mode>.npN` -> `build/level3/nekrs/<profile>/...` | `l3_paths` + hand-written variant paths -> `l3_paths_profile` | by variant yes, by backend no | yes; `hypregpu x HIP` refused explicitly |
| ExaCA | `level3/exaca/{src,deps}` | cuda, hip / cuda | `build/level3/exaca/<model>` -> `.../<profile>` | none (out-of-source) | shared root `.deps/level3/exaca/install/{kokkos-<model>,json,exaca-<model>}` with ONE fingerprint at the shared root -> `.deps/level3/exaca/<profile>/install/{kokkos,json,exaca}` | logs `.deps/level3/exaca/logs` shared, dependency builds `.deps/level3/exaca/build/kokkos-<model>` -> `<profile>/logs`, `<profile>/build/kokkos` | `.../<model>/$L3_RUN_SUBDIR/dirsolid.<mode>.npN` -> `.../<profile>/...` | `l3_paths` + per-backend suffixes -> `l3_paths_profile` | partial (install suffixes; fingerprint, logs and json shared) | yes; suffix scheme removed |
| Nyx | `level3/nyx/{src,deps}` | cuda, hip, cpu / cuda | `build/level3/nyx/<profile>` | `.deps/level3/nyx/<profile>/src` | `<profile>/install` | `<profile>/{logs,cache}` | `build/level3/nyx/<profile>/$L3_RUN_SUBDIR/<case>.<mode>.npN` | `l3_paths_profile` (now passes the backend) | yes | no (paths unchanged) |
| CP2K | `level3/cp2k/{src,deps}` | cuda / cuda | `build/level3/cp2k/<profile>` | `<profile>/src/toolchain` | `<profile>/install/{toolchain,cp2k}` | `<profile>/logs` | `.../<profile>/$L3_RUN_SUBDIR/...` | `l3_paths_profile` (now passes `cuda`) | yes | no |
| QMCPACK | `level3/qmcpack/{src,deps}` | cuda / cuda | `build/level3/qmcpack/<profile>` | `<profile>/src` | `<profile>/install/{llvm,...,qmcpack-real}` | `<profile>/logs` | `.../<profile>/$L3_RUN_SUBDIR/...` | `l3_paths_profile` (now passes `cuda`) | yes | no |
| DFT-FE | `level3/dftfe/{src,deps}` | cuda / cuda | `build/level3/dftfe/<profile>` | `<profile>/src` | `<profile>/install` | `<profile>/logs` | `.../<profile>/$L3_RUN_SUBDIR/...` | `l3_paths_profile` (now passes `cuda`) | yes | no |
| GEOS (retired) | `level3/geos/{src,deps}` | cuda / cuda | `build/level3/geos/<profile>` | `<profile>/src` | `<profile>/install` | `<profile>/logs` | `.../<profile>/$L3_RUN_SUBDIR/...` | `l3_paths_profile` (untouched; still parses) | yes | no (retired; no new runs) |

Why the six were not isolated: `l3_paths <app>` (first batch, 2026-09-03) gave every application ONE
`.deps/level3/<app>/{src,build,install,logs}` regardless of the backend; only the application build tree
carried `<cuda|hip>`. A HIP build would have shared the install prefix and fingerprint with the CUDA build
(the fingerprint check would have failed fast, but by design of a shared location, not of isolation), and for
SPECFEM3D and nekRS the build-side source copy as well. The second batch (2026-09-05/06) introduced
`l3_paths_profile`; this change extends it to every application and removes `l3_paths`.

## What did not change

- `level3/<app>/src`, `level3/<app>/deps`, `provenance/source.lock*.yaml` (source identity, archive sha256,
  `source_tree_sha256`, `source_version`), the GitHub Release assets and the tag `level3-source-hpcperf-l3-v1-rc1`.
  Verified after the change: `git diff main -- 'level3/*/provenance/' 'tools/artifacts/release_manifest.json'
  'level3/RELEASE_PLAN.json'` is empty; `tools/prepare_benchmark.sh --status` reports READY with the recorded
  `source_tree_sha256` for the six materialized trees.
- Backend capability declarations: `supported_backends` / `validated_backends` in every `benchmark.yaml`
  are unchanged. HIP remains **untested / never built** (no ROCm on the validation node); CP2K, QMCPACK,
  DFT-FE remain CUDA-only (no HIP branch was added). nekRS gains `backend_support` per variant
  (`hypregpu: hip unavailable`, `cpucoarse: hip untested`) because a per-application "hip supported" was
  too coarse for a variant whose frozen tree exists only for CUDA.
- `install_root` in the six `benchmark.yaml` now names the validated CUDA profile root
  (`.deps/level3/<app>/cuda`, nekRS `.deps/level3/nekrs/hypregpu.cuda`), as the second-batch entries already
  did; `profile_scheme` documents the identity rule. `tools/create_agent_workspace.sh --link-prebuilt-deps`
  resolves `dependency_installs` relative to it (ExaCA's `install/kokkos`, `install/json` now match the real
  directory names).

## Historical state

The pre-migration installs, logs and run trees (`.deps/level3/<app>/{src,build,install,logs}`,
`.deps/level3/nekrs/cpucoarse/`, `build/level3/<app>/<cuda|...>/run*`) on the development worktrees are the
evidence behind the results recorded in the application READMEs and status documents. They were **not**
moved, renamed or re-labelled; the new profile roots were produced by new builds with their own fingerprints
(schema `l3-2`, including the patch-file hashes that the schema `l3-1` fingerprints of 2026-09-03 did not
carry). Historical run directories were not touched: the regression below used its own run tree
(`HPCPERF_L3_RUN_SUBDIR=run.backend-profile-<sha>`).

A defect surfaced by rebuilding from the materialized trees and fixed here: since the source-freeze round,
SPECFEM3D's and nekRS' `build.sh` took the patch series from the lock as basenames and passed them to the
fingerprint, which hashes the patch files -- the first rebuild aborted with "patch file not found". They now
resolve `level3/<app>/patches/<name>`; the fingerprint still records the basenames.

## CUDA smoke regression on the new layout (dgx003, 2026-09-15, code `c225562`)

Fresh worktree of this branch, sources materialized from the local artifact cache (`prepare_benchmark.sh`
READY with the recorded `source_tree_sha256` for every tree), **no legacy `.deps/level3/<app>/install` present**,
every application built into its profile root by the migrated `build.sh`, then validated with
`HPCPERF_L3_RUN_SUBDIR=run.backend-profile-c225562` (a new run tree; no historical run directory was touched).
The run manifests record `profile=` and the sha256 of the profile's own fingerprint; the binaries come from
`build/level3/<app>/<profile>/` or `.deps/level3/<app>/<profile>/install/`, never from another location.

| Application | Profile built (fingerprint: schema `l3-2`, `backend=cuda`, patches hashed) | Build | Validation (run tree `run.backend-profile-c225562`) | Binary recorded in `run_manifest.txt` |
|---|---|---|---|---|
| ExaCA | `cuda` (`install/{kokkos,json,exaca}`) | 99 s (8 jobs) | 1 GPU: **PASS** (dirsolid smoke 128^3 vs reference statistics) | `.deps/level3/exaca/cuda/install/exaca/bin/ExaCA` |
| LAMMPS | `cuda` | 415 s (16 jobs) | 1 GPU **PASS**, 2 GPU **PASS** (bench/in.lj vs upstream log; rank-count independence) | `build/level3/lammps/cuda/lmp_kokkos_cuda` |
| SPARTA | `cuda` | 626 s (16 jobs) | 1 GPU: **PASS** (bench/in.collide vs upstream log) | `build/level3/sparta/cuda/src/spa_kokkos_cuda` |
| SPECFEM3D | `cuda` (build-side copy `.deps/level3/specfem3d/cuda/src`; 2 patch files hashed) | 166 s (8 jobs) | 1 GPU: **PASS** (homogeneous_halfspace vs REF_SEIS, 12/12 traces) | `.deps/level3/specfem3d/cuda/install/bin/xspecfem3D` |
| WarpX | `cuda` | 1473 s (16 jobs) | 1 GPU: **PASS** (langmuir_multi analytic + charge conservation; uniform_plasma particle conservation) | `build/level3/warpx/cuda/bin/warpx.3d.MPI.CUDA.DP.PDP.EB` |
| nekRS `hypregpu` | `hypregpu.cuda` (3 HYPRE patch files hashed; JIT cache `.deps/level3/nekrs/hypregpu.cuda/cache`, 259 MB) | 834 s (16 jobs) | 1 GPU cimode 2: **PASS** (9/9, coarse=CPU); cimode 3: **PASS** (9/9, coarse=DEVICE -- the GPU-HYPRE install of this profile is the one running) | `.deps/level3/nekrs/hypregpu.cuda/install/bin/nekrs` |
| nekRS `cpucoarse` | `cpucoarse.cuda` (0 patches; own JIT cache, 257 MB) after re-materializing the cpucoarse tree (`6bde0318...`) | 402 s (32 jobs) | 1 GPU cimode 2: **PASS** (9/9, coarse=CPU) | `.deps/level3/nekrs/cpucoarse.cuda/install/bin/nekrs` |

Refusal paths exercised on the same worktree: `level3/nekrs/build.sh HIP` (variant hypregpu) exits 2 with
"variant hypregpu is CUDA-only" and creates no `hypregpu.hip` directory; `HPCPERF_LAMMPS_PROFILE=cuda
level3/lammps/build.sh HIP` exits 2 with the profile/BACKEND conflict and creates nothing; `run.sh CUDA` before
`build.sh` exits 1 naming the missing profile binary (no fallback); `level3/lammps/build.sh HIP` reaches the
hipcc check ("HIP build is UNTESTED on this machine") -- HIP was **not** built anywhere. Strong/weak modes and
the 1/2/4 matrix were not re-run (out of scope; unchanged physics and validation policy). Nyx, CP2K, QMCPACK
and DFT-FE keep their paths and were not rebuilt for this change.

## Tests (CPU-only, `level3/tools/tests/run_all.sh`)

`test_l3_infra.sh` section 9 (19 checks): CUDA and HIP profiles share no src-copy/dependency-build/install/
logs/cache/build-tree path; the profile's `src` is a build-side location under `.deps/`, never `level3/<app>/src`;
profile/BACKEND conflicts and backend-less profiles are refused and create nothing; nekRS `<variant>.<backend>`,
toolchain-identity and override profiles are accepted when they name their backend; one derivation helper
(override honoured); a legacy `.deps/level3/<app>/install` is reported and not used and does not satisfy the
run-side gate; fingerprints of two profiles live apart and differ; a CUDA fingerprint serves CUDA and is refused
for HIP (run side and build side); static checks that build/run/validate of the six migrated applications
(and build/run of Nyx/CP2K/QMCPACK/DFT-FE) call `l3_paths_profile` with the backend, that `l3_paths` has neither
definition nor caller, that no active wrapper hardcodes an unprofiled `.deps/level3/<app>/{install,logs,src,build}`,
that nekRS refuses `hypregpu x HIP`, and that the migrated build scripts still read the one materialized source
tree. All seven Level 3 test groups pass (infra 49, Nyx validator 33, verdict 18, ExaCA validator 19, source
tools 81, release mock 45).
