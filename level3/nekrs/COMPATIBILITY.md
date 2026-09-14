# nekRS -- CUDA / dependency compatibility (v26.0 on dgx003, B200, CUDA 13.2.78)

Scope: the exact software *combination* used for the Level 3 nekRS bring-up, and
what is and is not verified about it. This document separates four things the
review asked never to be conflated:

1. what upstream **documents**;
2. what upstream **CI / site testing** actually exercises;
3. the **local build** result for this exact combination;
4. the **local run** result for this exact case.

"Verified" below means *observed on this node*, never "endorsed by upstream".
Where no upstream evidence was found, the entry is `NOT_FOUND` / `UNVERIFIED` --
not "upstream forbids it".

Upstream references consulted (read at the pinned commit / current docs):
`github.com/Nek5000/nekRS` and `/releases`; the pinned
`CMakeLists.txt` and `cmake/hypre.cmake` at
`96b3cf9e5bacede16568826c04a21bc0fe50dc7d`; `nekrs.readthedocs.io/quickstart`;
`Nek5000/nekRS_HPCsupport`; `hypre-space/hypre`; the CCCL 3.0 migration guide
(`nvidia.github.io/cccl/.../3.0_migration_guide.html`); the Blackwell
compatibility guide (`docs.nvidia.com/cuda/blackwell-compatibility-guide`).

## The combination

| Component | Value |
|---|---|
| nekRS | v26.0, commit `96b3cf9e5bacede16568826c04a21bc0fe50dc7d` |
| vendored HYPRE | 2.32.0 (`3rd_party/hypre`, squashed subtree) |
| vendored OCCA | 2.0.0-dev (`OCCA_VERSION_STR`) |
| CUDA toolkit | 13.2.78 (`/usr/local/cuda` -> cuda-13.2); Thrust/CCCL 3.2 (`THRUST_VERSION 300200`) |
| GPU / arch | B200, sm_100 |
| host compiler | conda GCC 13.3.0 (C/C++), system gfortran 14.2.1 (Fortran) |
| MPI | conda Open MPI 5.0.10 (CUDA-aware) |
| default build option | `OCCA_ENABLE_CUDA=ON`, `ENABLE_HYPRE_GPU=ON` (variant `hypregpu`) |

## `cmake/hypre.cmake` at this commit already has a CUDA >= 13 branch

The pinned `cmake/hypre.cmake` contains:

```
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0.0")
    set(HYPRE_DEVICE_ARCH "HYPRE_CUDA_SM=80 90")
elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.0.0")
    set(HYPRE_DEVICE_ARCH "HYPRE_CUDA_SM=70 80")
endif()
```

So nekRS v26.0 **is aware of CUDA 13** (a dedicated branch exists) -- it is wrong
to say "nekRS does not support CUDA 13". Equally, the branch only sets the SASS
list to `80 90` (no sm_100, no PTX) and the presence of a branch is **not**
evidence that CUDA 13.2.78 + B200 + vendored HYPRE 2.32.0 passed any upstream CI:
upstream CI (`.github/workflows/ci.yml`) builds the serial/CPU backend with
MPICH + gfortran on ubuntu; the nekRS_HPCsupport machine files target
Frontier/Perlmutter/Polaris/etc., none of them a B200/CUDA-13 combination
(NOT_FOUND for this exact stack).

## Four problem classes, separated

### A. HYPRE GPU source vs CCCL/Thrust 3.2 API  (compile, GPU-only code)

`ENABLE_HYPRE_GPU=ON` compiles HYPRE's device sources with nvcc against CUDA 13's
Thrust 3.2. Three incompatibilities were hit and are addressed by the three
patches (all confined to HYPRE's *device* code / device headers):

| Symptom (build error) | Cause | Real API change? | Patch |
|---|---|---|---|
| `namespace "thrust" has no member "make_reverse_iterator"` / `"pair"` | `<thrust/iterator/reverse_iterator.h>` and `<thrust/pair.h>` are no longer pulled in transitively by other thrust headers in CCCL 3.x | **No** -- the names still exist (`thrust::make_reverse_iterator`/`reverse_iterator` are re-exported from `cuda::std` in `thrust/iterator/reverse_iterator.h`); they were merely not *visible* without an explicit include | 0003 (add the includes to `device_utils.h` and the pre-generated `_hypre_utilities.hpp`) |
| `thrust::pair<...>` as a declared result type of `reduce_by_key` | same non-transitive-include issue at the point of use | **No** (visibility) -- resolved either by the include or by letting `auto` take the library's real return type | 0002 (`auto`) + 0003 (include) |
| `thrust::not1` undefined | `thrust::not1` (C++17-deprecated) was **removed** in CCCL 3.x | **Yes -- genuinely removed**; documented replacement is `thrust::not_fn` (= `cuda::std::not_fn`) | 0003 (`not1` -> `not_fn`, 16 sites) |

So only `not1` is a true API removal; the `pair`/`reverse_iterator` errors are
missing explicit includes, not deleted APIs. This distinction is recorded per the
review's instruction.

Upstream fix status: **UNVERIFIED / NOT looked up in detail.** HYPRE's own master
is understood to build under CUDA 13, and the CCCL migration guide documents the
`not1 -> not_fn` change, but no specific HYPRE commit/PR or nekRS-vendored-HYPRE
bump was located and pinned. The patches are therefore labelled **project-local
compatibility patches**, not backports of a known upstream commit. Whether nekRS
`next` already vendors a CUDA-13-ready HYPRE was not checked (route A, below).

### B. HYPRE prebuilt device library arch list (sm_100 coverage)

Patch 0001 changes `HYPRE_CUDA_SM=80 90` to `80 90 100` so the device library
carries Blackwell SASS. Without it HYPRE's device kernels would have no sm_100
code (and the branch emits no PTX to JIT from). This matters **only when GPU
HYPRE is actually used** (see the coarse-solver finding below).

### C. OCCA CUDA / JIT

OCCA derives `-arch=sm_<device>` at run time and JIT-compiles the OKL kernels
with the host nvcc; no patch needed. Verified locally: `active occa mode: CUDA`,
kernels compiled, the main solver runs on the GPU (the launcher's per-rank GPU
audit reports the ranks on distinct GPUs).

### D. MPI one-sided (OSC) UCX runtime failure

Independent of the above: nekRS uses `MPI_Win_lock`, and Open MPI's default
`osc ucx` aborts in `uct_ib` at 4 ranks on this node. Worked around at run time
with `OMPI_MCA_osc=^ucx` in `run.sh` (nekRS-local; not a global default). This is
a transport/runtime issue, not a CUDA/HYPRE compatibility issue.

Fixing any one of A/B/C/D does not fix the others; they are tracked separately.

## The load-bearing finding: the benchmark case runs the HYPRE coarse solve on the CPU

For the ethier case (the only nekRS case wired up), the six recorded logs
(`smoke.np{1,2,4}.cimode2`, `strong.np{1,4}`, `weak.np4`) all show:

```
FLUID PRESSURE MULTIGRID COARSE SOLVER: BOOMERAMG
FLUID PRESSURE MULTIGRID COARSE SOLVER LOCATION: CPU
FLUID PRESSURE MULTIGRID COARSE SOLVER PRECISION: FP32
```

Cross-checked against the source, not just the logs:
- `src/platform/par/parsePreconditioner.hpp` sets the default
  `... MULTIGRID COARSE SOLVER LOCATION = CPU` and `... PRECISION = FP32`;
- `src/core/elliptic/elliptic.cpp` sets `... COARSE SOLVER LOCATION = CPU`;
- `examples/ethier/ethier.par` does **not** set a coarse-solver location, and
  `--cimode 2` (`examples/ethier/ci.inc`) does **not** set one either -> the
  default (CPU) applies;
- only `--cimode 3` sets `FLUID PRESSURE PRECONDITIONER = MULTIGRID+SEMFEM` and
  `... COARSE SOLVER LOCATION = DEVICE` (falling back to CPU only when the run is
  serial / CPU-backend).

Consequences (stated precisely):
- The nekRS **main application does run on the GPU** (OCCA CUDA) -- this is NOT a
  CPU-only application, and multi-GPU main-solve is real (per-rank GPU audit).
- The HYPRE **BoomerAMG coarse solve for the ethier cimode-2 case runs on the
  CPU**. The GPU HYPRE device library built by `ENABLE_HYPRE_GPU=ON` is compiled
  but **not exercised** by this case.
- Therefore the earlier "9/9 CI checks passed" for cimode 2 validated *the CUDA
  main application + a CPU coarse solve*. It did **not** validate GPU HYPRE. A
  case that selects `COARSE SOLVER LOCATION = DEVICE` (cimode 3) is required to
  exercise GPU HYPRE on the target GPU; that verification is tracked in the
  matrix below and must not be claimed from the cimode-2 result.

("Library compiled/loaded" is not "kernel executed" -- the judgement here is from
the solver-configuration lines the run prints and the source defaults, not from a
source grep alone.)

## Compatibility / verification matrix

| variant | ENABLE_HYPRE_GPU | patches | cimode | runtime coarse | build (dgx003, CUDA 13.2.78, sm_100) | main app CUDA multi-GPU | GPU HYPRE coarse | official CI for this stack |
|---|---|---|---|---|---|---|---|---|
| hypregpu (default) | ON | 0001+0002+0003 | 2 | CPU | ok (~30 min, prior round) | VERIFIED 1/2/4 GPU, 9/9 | not exercised (CPU coarse) | NOT_FOUND |
| hypregpu | ON | 0001+0002+0003 | 3 | DEVICE (GPU) | (same install) | VERIFIED 1/4 GPU, 9/9 | **VERIFIED** 1/4 GPU (9/9, coarse=DEVICE) | NOT_FOUND |
| cpucoarse (candidate) | OFF | **none** | 2 | CPU | ok, **113 s, 0 patches** | VERIFIED 1/2/4 GPU, 9/9 | not built | n/a |
| cpucoarse | OFF | none | 3 | (DEVICE requested) | (same install) | n/a | **explicitly rejected** -- nekRS aborts: `HYPRE+DEVICE not enabled! Recompile with -DENABLE_HYPRE_GPU=ON` (exit 1); NO silent CPU fallback | n/a |

Results (dgx003, 2026-09-05), all through the common launcher (one rank per GPU,
per-rank GPU wrapper, audit verified on distinct GPUs), analytic Ethier solution,
EPS 0.3, stricter validator (real exit code, complete 9/9 check set, coarse
location asserted, NaN/Inf rejected):

- hypregpu cimode 2: PASS 1/2/4 GPU, coarse=CPU.
- hypregpu cimode 3: PASS 1/4 GPU, coarse=DEVICE -> **the GPU HYPRE coarse solve
  enabled by the three patches is verified correct on B200/CUDA 13.2**, not just
  compiled.
- cpucoarse cimode 2: PASS 1/2/4 GPU, coarse=CPU, main app on distinct GPUs ->
  the entire current Ethier workload runs correctly with GPU HYPRE OFF and NO
  patches (build 113 s vs the ~30 min patched GPU-HYPRE build).
- cpucoarse cimode 3: nekRS's own `HYPRE+DEVICE not enabled!` check aborts the
  run (exit 1); the validator FAILs (coarse != DEVICE). No silent downgrade.

Finding: the three patches are needed ONLY to build/exercise GPU HYPRE. For the
current CPU-coarse Ethier workload the `cpucoarse` variant is a smaller,
patch-free, faster-to-build configuration that runs the same GPU main
application. The `hypregpu` variant remains the one that can run (and is verified
to run) the GPU coarse solve.

## What is and is not claimed for the default (hypregpu) variant

- CLAIMED (verified locally): CUDA main application, multi-GPU (1/2/4), on
  distinct GPUs; ethier cimode-2 correctness (analytic solution, 9/9 checks) with
  a **CPU** HYPRE coarse solve.
- NOT CLAIMED: that GPU HYPRE coarse solve is correct (cimode 2 does not run it);
  that the CUDA 13.2.78 + B200 + HYPRE 2.32.0 combination is upstream-CI-certified
  (NOT_FOUND); that CPU-coarse is the best configuration at 40/80 GPUs
  (UNVERIFIED -- CPU coarse scalability is a separate, unmeasured question).

## Routes if GPU HYPRE coarse is actually required (audited, not all built)

- **A -- upstream integration**: pin a newer nekRS release / a fixed
  master/next commit / an upstream HYPRE bump that resolves CCCL-3 + sm_100.
  STATUS: NOT_RUN (not audited to a specific SHA this round).
- **B -- keep CUDA 13.2 + HYPRE 2.32.0 + the local patches** as a
  project-maintained compatibility variant, and verify GPU coarse with cimode 3.
  STATUS: the cimode-3 verification is attempted this round (matrix).
- **C -- an earlier B200-capable CUDA (12.8/12.9) for nekRS only**: would still
  need HYPRE_CUDA_SM to include sm_100 and re-verification; does not
  automatically avoid the arch-target问题. Must not touch the system CUDA symlink
  or the other four apps. STATUS: NOT_RUN.
- **D -- replace the vendored HYPRE**: dependency-upgrade experiment; needs
  host/device wrapper, single/mixedint, precision, HYPRE_Int/BigInt and link
  checks. Not to be recorded as an LLM/source optimisation. STATUS: NOT_RUN.

Routes A/C/D are UNVERIFIED this round by design (no extra large builds started).
