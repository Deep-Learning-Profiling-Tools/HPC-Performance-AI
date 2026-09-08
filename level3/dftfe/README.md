# DFT-FE -- Level 3 second batch

Real-space finite-element Kohn-Sham DFT (University of Michigan). GPU build
(`WITH_GPU=ON GPU_LANG=cuda GPU_VENDOR=nvidia`): Chebyshev-filtered subspace
iteration, dense linear algebra (cuBLAS), Poisson/Helmholtz solves and the
ELPA eigensolver with NVIDIA GPU kernels.

## Provenance / versions

| Item | Value |
|---|---|
| DFT-FE | release 1.2.0, `7147faa51f7c9f3075fffaa5e48ba989bcd329c1` (branch `release1.2`), LGPL-2.1-or-later |
| Recipe | upstream `install_DFTFE` (branch `frontierDevelop`: `dftfe2.sh` + `setupUser.sh`) transcribed to this node's toolchain in `build.sh`; every dependency tarball is pinned by sha256 in `fetch.sh` |
| Dependencies (private, same profile) | OpenBLAS 0.3.30 (`TARGET=SAPPHIRERAPIDS`, single-threaded) -> ScaLAPACK 2.2.2 -> libxc 7.0.0 -> spglib `02159eef` -> ALGLIB 4.06.0 -> p4est 2.8.7 (dftfe's `p4est-setup.sh`, FAST+DEBUG) -> Kokkos 4.6.00 (Serial, deal.II only) -> deal.II **9.6.2** (see decision below) (MPI, p4est, 64-bit indices, LAPACK=OpenBLAS, bundled Boost, no TBB/taskflow) -> ELPA 2026.02.001 (`--enable-nvidia-gpu-kernels --with-NVIDIA-GPU-compute-capability=sm_100`) -> DFT-FE real (`CMAKE_CUDA_ARCHITECTURES=100 WITH_DCCL=OFF WITH_GPU_AWARE_MPI=OFF USE_64BIT_INT=ON`); libxml2 from the system |
| Compilers | system GCC 14.2.1 (C/C++/Fortran) through the conda Open MPI 5.0.10 wrappers (`OMPI_CC/CXX/FC`), nvcc 13.2.78 with g++ 14.2.1 host |
| GPU target | sm_100 for DFT-FE and for the ELPA kernels (ELPA's generic `--with-NVIDIA-GPU-compute-capability=sm_XX` path; sm_100 never used upstream) |
| NCCL | not used (`WITH_DCCL=OFF`; DFT-FE's default all-reduce path) |
| Build strategy | NATIVE (no Spack: the `dftfe` recipe is 0.6/2019 without GPU variants; no container) |
| Profile | `cuda132-gcc142-ompi5010` (`HPCPERF_DFTFE_ELPA_GPU=OFF` would build `-elpacpu`, a named CPU-ELPA diagnostic profile, not used for the results) |

### Node adaptations (class C, recorded in build.sh)

- dftfe's `p4est-setup.sh` is written for Cray systems: it hardcodes
  `CC=cc CXX=CC FC=ftn F77=ftn` (no MPI on this node's `cc`), relies on the Cray
  wrappers' implicit `-lm`, and checks `<build>/src/p4est_config.h` while p4est
  2.8.7 writes `<build>/config/p4est_config.h`. build.sh passes
  `CC=mpicc CXX=mpicxx FC=mpifort F77=mpifort LIBS=-lm` through the script's
  trailing configure arguments and seds the header path in its private copy.
- The conda environment's `CFLAGS/LDFLAGS/AR/...` are cleared for the
  system-GCC builds (`l3_clean_conda_build_env`); OpenBLAS 0.3.30 misdetects the
  Xeon Platinum 8570 and is built with `TARGET=SAPPHIRERAPIDS`.
- ELPA's configure expects the host SIMD flags in `CFLAGS` (it probes AVX-512
  intrinsics with the given flags and aborts otherwise; EL10's GCC 14 defaults to
  x86-64-v3 = AVX2): `-march=native` for C/C++/Fortran -- a node-specific
  binary like OpenBLAS. Its later CUDA link checks (`cublasDgemm`) keep
  `LIBS=-lscalapack` but replace `LDFLAGS` by the CUDA path, so the
  ScaLAPACK/OpenBLAS `-L`/rpath entries are given in `LDFLAGS` as well.

### The one DFT-FE source patch (class D, `patches/0001-std-isnan.patch`)

Two lines in `src/atom/AtomicCenteredNonLocalOperator.cc` (785/791) call an
unqualified `isnan(` inside a template; GCC 14 (libstdc++ 14) no longer finds
it through `<cmath>`'s global fallback and stops with "there are no arguments to
'isnan' that depend on a template parameter". The patch qualifies both calls as
`std::isnan(` -- the same function, no numerical or algorithmic change (upstream
develop has the same spelling). Applied by build.sh with `git apply` (idempotent,
sha256 recorded in the fingerprint as `DFTFE_PATCH_SHA`).

### deal.II version decision (a dependency choice, not a DFT-FE change)

Release 1.2.0 pins deal.II **9.5.2** in its own `setupUser*.sh`; the current
`install_DFTFE` recipe (frontierDevelop/master) builds deal.II **9.7.1** -- but
together with the *develop* branch (`publicGithubDevelop`), not with the release.
Attempt 1 here (1.2.0 + 9.7.1) failed to compile: 9.7 removed three APIs 1.2.0
still uses (`Utilities::MPI::create_group`, `parallel::distributed::Triangulation::load(name, autopartition)`,
`DataOutBase::VtkFlags::ZlibCompressionLevel`; upstream develop already carries the
replacements). Rather than back-porting a growing set of develop changes into the
release, the profile uses **deal.II 9.6.2**, the newest deal.II that still provides
those (deprecated) APIs, so 1.2.0 builds unpatched; recorded as attempt 2. The
9.7.1 build and its install were removed from the profile
(`.deps/level3/dftfe/<profile>/ATTEMPT-1-dealii-9.7.1.txt`, logs `attempt1-*`).

## ELPA GPU kernels verified independently (`elpa_probe.sh`)

ELPA's own test programs (`validate_real_double_eigenvectors_{1stage,2stage_default_kernel}_gpu_analytic`,
which call `e%set("nvidia-gpu", 1)`; the real binaries in `build/elpa/.libs/`, the
top-level names being libtool relink wrappers) on 1, 2 and 4 GPUs (one rank per GPU
through the common launcher, na=2000 nev=1000 nblk=32) and their CPU counterparts on
1 rank when built (they are not, with the GPU configuration: SKIPPED, recorded).
The programs diagonalise the analytic test matrix of `test/shared/test_analytic_template.F90`
(known eigenpairs) and apply ELPA's own limits `max |lambda - lambda_exact| <= 5e-14`,
`max |z - z_exact| <= 6e-10` (real double; `stop 1` on violation); the probe re-parses the
printed "Maximum error in eigenvalues/eigenvectors", re-applies the limits, requires exit 0,
ELPA's GPU timers (`trans_ev_real_double_gpu`, `gpublas_*`) in the output and a launcher
audit with 0 mismatch; record `<install>/elpa/ELPA_GPU_PROBE.txt`, required PASS by
`validate.sh`. Two false starts, both recorded in the script header: the first pass
looked for ELPA's `*_default` wrapper-script names (MISSING), the second parsed the
`%Error Residual/Orthogonality` lines of the *random-matrix* programs, which the analytic
programs never print, and aborted under `pipefail`.

## Cases and validation

`run.sh`: `al_md` (upstream GPU regression deck `testsGPU/pseudopotential/real/Input_MD_0.prm`:
32-atom fcc Al, order-3 FE, 85 states, BOMD 1400 K, 4 steps, `USE GPU = true`,
`REPRODUCIBLE OUTPUT = true`) -- smoke/strong verbatim, weak = derived Al supercell
series (32 atoms per GPU; SYNTHETIC, same construction as upstream's
dftfe-benchmarks Mo series); `llzo` (`parameterFile_LLZO.prm`, 192 atoms, 720
states, ELPA) as the heavier fixed-size strong case. One MPI rank per GPU, 1
thread per rank (upstream's GPU job scripts), decks used verbatim except the
documented weak derivation.

`validate.sh`: [0] ELPA GPU probe PASS; [1] `al_md` on N GPUs vs upstream's own GPU
reference `accuracyBenchmarks/output_MD_0` through `dftfe_check.py` (pre-fixed:
ground-state energy 1e-5 Ha, per-step MD energies 2e-5 Ha, temperatures 0.1 K,
forces 2e-5 Ha/Bohr; SCF converged, MD completed, finite), launcher audit "N
verified, 0 mismatch"; N > 1 additionally vs the 1-GPU run under the same
tolerances.

### Results (2026-09-06, profile `cuda132-gcc142-ompi5010`, 1 rank per GPU)

| Run | GPUs | atoms / KS DOFs | wall (s) | check |
|---|---|---|---|---|
| `al_md` smoke (= strong, verbatim deck) | 1 | 32 / 50 653 | 216 | vs upstream GPU reference: e0, 4 MD energies, temperatures, forces all **identical at printed precision** (max diff 0.0; SCF iterations 8/9/9/9 as upstream) |
| `al_md` | 2 | 32 / 50 653 | 117 | identical to upstream reference and to the 1-GPU run |
| `al_md` | 4 | 32 / 50 653 | 70 | identical to upstream reference and to the 1-GPU run |
| weak x1 (2x2x2 cells) | 1 | 32 / 50 653 | 213 | complete (same deck as smoke) |
| weak x2 (4x2x2 cells, derived) | 2 | 64 / 108 151 | 284 | complete, 4 MD steps |
| weak x4 (4x4x2 cells, derived) | 4 | 128 / 230 917 | 378 | complete, 4 MD steps |
| `llzo` strong (192 atoms, 720 states, ELPA, verbatim deck) | 1 / 2 / 4 | 192 / 97 336 | 295 / 163 / 96 (1.81x, 3.07x) | ground state, 19 SCF iterations, E = -3579.26588980 Ha **identical on 1, 2 and 4 GPUs** |
| ELPA GPU probe, 1-stage (`validate_real_double_eigenvectors_1stage_gpu_analytic`, na=2000 nev=1000) | 1 / 2 / 4 | -- | -- | PASS: max eigenvalue error 2.7e-15 / 2.4e-15 / 2.0e-15 (limit 5e-14), eigenvector error 1.5e-12 / 1.7e-12 / 1.5e-12 (limit 6e-10), exit 0, GPU timers, audit 0 mismatch |
| ELPA GPU probe, 2-stage (`..._2stage_default_kernel_gpu_analytic`) | 1 / 2 / 4 | -- | -- | PASS: 5.8e-15 / 7.3e-15 / 6.9e-15; 1.0e-11 / 1.0e-11 / 8.1e-12; CPU counterparts SKIPPED (not built) |

**VALIDATED_PASS at 1, 2 and 4 GPUs** (validate.sh 2026-09-06 05:04-05:12 UTC: [0] probe
PASS, [1] al_md within the pre-fixed tolerances of upstream's reference and of the
1-GPU run, launcher audit N verified / 0 mismatch; the 05:01-05:03 UTC pass FAILED on
[0] only, with the aborted probe -- kept in `validate.*.stdout` history).
Strong scaling: 32-atom deck 1.85x / 3.1x on 2 / 4 GPUs (small problem, 50 k DOFs);
192-atom LLZO 1.81x / 3.07x. Dry-runs: 8/40/80 GPUs planned (HYPOTHETICAL) for the
strong decks; the weak series is defined for 1/2/4/8 GPUs only (40/80 refused by run.sh). The weak series is synthetic (32 atoms per GPU; the electronic-structure
cost grows faster than linearly with the cell, so constant wall time is not
expected -- reported, not "ideal"). Probe record: `<install>/elpa/ELPA_GPU_PROBE.txt`,
per-program logs `build/level3/dftfe/<profile>/elpa_probe/`.

<!-- hpcperf:source-section:begin -->
## Source distribution (frozen bundle, 2026-09-08)

The application source is no longer read from `_upstream/`: `tools/prepare_benchmark.sh level3 dftfe` materializes the frozen bundle into `src/` (+ `deps/`), the only source `build.sh`/`run.sh`/`validate.sh` use. Identity, patch series, licenses and the equivalence proof against the tree the results above were validated from are under `provenance/` (`source.lock*.yaml`, `patch_series*.txt`, `original_vs_baseline*.diff`, `LICENSES*.md`, `equivalence*.md`, `LOC*.md`); what an optimization agent may modify is in `optimization_scope.yaml`; `benchmark.yaml` is the machine-readable contract.

| variant | archive | compressed / uncompressed | files | source_tree_sha256 | archive sha256 | upstream | patches (pre-applied) | equivalence | LOC app-owned / agent-modifiable / bundled deps / benchmark deps / tests / total |
|---|---|---|---|---|---|---|---|---|---|
| - | `archives/source_bundle.tar.zst` | 174.5 MB / 262.3 MB | 3525 | `54bea98826f91a4ce298fd247da5f5168f595bef5b5ddac629b74bdc82ffea30` | `0259dfec2c9459f7ddd824a5eef468d04f77638b0a3cdd323a2cc342fe637670` | release 1.2.0 commit (2025-08-17) `7147faa51f7c` | 0001-std-isnan.patch | src: EQUIVALENT, deps/spglib: EQUIVALENT | 107718 / 107718 / 0 / 12784642 / 314 / 12896192 |

LOC = cloc 2.06 code lines of the materialized tree (no blank/comment lines, documentation and data excluded); categories from `optimization_scope.yaml` (`loc_categories`). The validated results recorded above were produced from trees proven content-equivalent to these bundles (`provenance/equivalence*.md`); they are not re-run by the migration.
<!-- hpcperf:source-section:end -->
