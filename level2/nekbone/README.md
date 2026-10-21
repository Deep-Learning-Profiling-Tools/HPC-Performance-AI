# Nekbone (cuda-openacc branch) -- Nek5000's conjugate-gradient proxy on the GPU

Nekbone solves a 3-D Poisson problem on a brick of spectral elements (Gauss-Lobatto-Legendre nodes, order
`lx1 - 1`) with conjugate gradients: each iteration applies the matrix-free spectral-element Laplacian (tensor
products of the 1-D derivative matrices per element, geometric factors), assembles the result across element
faces with a gather-scatter (`dssum`, the Nek5000 `gs` library) and performs the inner products and vector
updates of CG. The `cuda-openacc` branch puts the solve on the device: `cg_acc` with OpenACC parallel loops for
the vector operations and the gather-scatter (`dssum_acc`), and a CUDA Fortran kernel (`ax_cuda.f`, `ax_cuf2`)
for the per-element matrix-vector product. Fortran 77 + C (the `jl` gather-scatter library), MPI (one rank per
GPU; the registered inputs run one rank).

## Source

- Upstream: https://github.com/Nek5000/Nekbone, branch `cuda-openacc`, commit
  `7fd43bcdc036a1092f03762e9cc5b21f04b7b3c2` ("Merge pull request #16 from person142/openacc-update-jl",
  2018-03-21; the branch's last commit, no release tag). License: the Nek5000 BSD-style license
  (`COPYRIGHT`, UCHICAGO ARGONNE, LLC).
- Vendored byte-identically (`UPSTREAM_SHA256SUMS` lists the upstream hash of every vendored file; the three
  patched files are listed with their upstream hash, the patches are described below): `src/` (the whole
  source tree: Fortran, the `jl` gather-scatter library, `makefile.template`, `makenek.inc`, `deprecated/`),
  `test/nek_gpu1/` (the GPU example the branch was developed with: `SIZE`, `data.rea`, the three `makenek.*`
  build scripts, `README`, `scripts/` with upstream's compare scripts and `master_branch.output`,
  `old_makefiles/`), `test/test_nekbone.py` (upstream's unittest harness), `COPYRIGHT`, `CHANGES`,
  upstream `README.md` as `README.upstream.md`. Not vendored: `bin/` (site job scripts), `USERGUIDE.pdf`
  (binary), the CPU-only examples `test/example1..3`, `nek_comm`, `nek_delay`, `nek_mgrid`.
- Toolchain: NVIDIA HPC SDK 25.7 `nvfortran` / `nvc` (site-provided, see the Level 2 README toolchain
  table: `NVLOCALRC` for the per-user localrc on dgx003), the repository's Open MPI 5.0.10 through
  `mpifort --showme` (Fortran 77 `include 'mpif.h'`; the `-lmpi_usempi*` module libraries of the MPI's own
  compiler are dropped from the link line as `level2/gamess_ri_mp2/build.sh` does), system CUDA 13.2 driver.
  The SDK's bundled CUDA 12.9 runtime is what the OpenACC / CUDA Fortran code links (rpath); the device code
  is compiled for the detected compute capability (`-gpu=cc100` on the B200).

## Changes from upstream (`HPC-Performance-AI` markers in the sources)

| file | change | why |
|---|---|---|
| `src/makenek.inc` | `*nvfortran*)` added to the compiler case list with pgfortran's flags `-r8 -Mpreprocess -Mfixed` | makenek knows pgf*/gfortran/ifort/...; nvfortran (pgfortran's successor, same flags) left `P` unset |
| `src/makenek.inc` | the debug test `echo $G \| grep '\-g'` became a whole-word test for `-g` | the substring test also matched the SDK's target option `-gpu=cc100` and silently rebuilt **every** object with `-O0` ("Activate DEBUG mode") |
| `src/jl/gs_defs.h` | `GS_DO_bpr` written as a structured `while` loop (same operation) | nvc 25.7 rejects the upstream `break`-out-of-`do{}while(0)` / `for(;;)` form inside the OpenACC regions of `gs_local.c` ("bad single-entry/single-exit region nesting"); the `bpr` operation is not used by the CG, the file must merely compile |
| `src/driver.f` | `use hpcperf_roi`; `call hpcperf_roi_begin_sync()` / `end_sync()` immediately around the timed `cg_acc` (and `cg`) call; after the timed solve: `!$ACC UPDATE HOST(x(1:n))` and a print of `sqrt(x^T C x)` and `x^T C f` (`HPCPERF solution:` line, host arithmetic, outside the timed region) | ROI markers (tools/timing/roi); a solution functional for the correctness check -- upstream prints only residual norms, which a consistently wrong operator still drives to zero |
| `src/driver.f` (`set_multiplicity`, `set_f`) | the `!$ACC UPDATE DEVICE` / `UPDATE HOST` pairs upstream puts around the two host-side gather-scatter calls of the set-up **only under `GPUDIRECT`** are enabled for every OpenACC build (`#if defined(GPUDIRECT) \|\| defined(_OPENACC)`) | an upstream defect of the non-GPUDIRECT path: the jl library executes a gather-scatter on the device copy of any array that is present on the device (`gs.c gs_aux`: `acc_is_present`), and the driver's `ACC DATA CREATE` makes `f` and `c` present for the whole run -- so `dssum(c)` and `dssum(f)` summed never-uploaded device copies and left the host `c` at 1 and `f` unassembled. The GPU then solved a different system from the CPU path (r0 = the raw 2-norm of `sin(cos(i*i))`, 13 % off, solution 3 x off, residual reduction 1e3 worse); with the updates both paths start from the same assembled right-hand side and weights, as upstream's own harness asserts (RESULTS). The timed solve is untouched |

The algorithms, the problem set-up (`set_f` right-hand side, mask, multiplicity, geometric factors) and the
100-iteration CG are upstream's. Build flags are upstream's `makenek.cuda` with the 2018 PGI spellings
replaced by their current ones (`-Mcuda` -> `-cuda`, `-ta=nvidia:ccNN` -> `-gpu=ccNN`) plus `-mcmodel=medium`
(the static arrays of the registered sizes exceed 2 GB) and upstream's per-file optimisation levels
(`-O2` / `-O3`, makefile.template).

## Build (`build.sh [CUDA|CPU] [ID ...]`)

`SIZE` fixes the polynomial order (`lx1`) and the maximum number of elements per rank (`lelt`) at compile
time, so every registered input is its own build: `build/level2/nekbone/cuda/<id>/` holds the copied `SIZE`,
the generated `makenek` (upstream's `test/nek_gpu1/makenek.cuda` with `SOURCE_ROOT`, `F77`, `CC`, `G`,
`USR_LFLAGS` replaced), the `makefile` upstream's `makenek.inc` generates from `src/makefile.template`,
`obj/` and `nekbone`. `src/` is never written. `build.sh CPU` builds upstream's `makenek.mpi` variant (no
`-acc`/`-cuda`: the host code path `cg` / `ax` / `dssum`) into `build/level2/nekbone/cpu/<id>/`; it is the
reference build of the correctness check and of `validate.sh`, never measured.

## Run (`run.sh [CUDA] [ID]`, `HPCPERF_NEKBONE_INPUT=<id>`)

The program reads `data.rea` from its working directory and writes nothing but stdout, so `run.sh` runs the
input's executable with `inputs/<id>/` as the working directory: the registered `data.rea` is read as is (no
copy, no generated file; the verifier's `cwd_files` evidence). One MPI rank per GPU through the common
launcher; `HPCPERF_GPUS=N` runs N ranks of the per-rank problem (weak scaling, `N <= lp` of `SIZE`) -- the
registered workloads are the one-rank runs. The single argument `data` is the Nek5000 session-name
convention; this program does not read it.

## Registered inputs (`inputs.yaml`)

| id | order `lx1` | elements / rank | points | timed solve | note |
|---|---|---|---|---|---|
| n16-e8192 | 16 | 8,192 | 33.6 M | RESULTS | element-count series at order 16 |
| n16-e16384 | 16 | 16,384 | 67.1 M | RESULTS | |
| n16-e32768 (default) | 16 | 32,768 | 134.2 M | RESULTS | the largest per-rank size upstream's gather-scatter set-up handles (Limitations) |
| n8-e262144 | 8 | 262,144 | 134.2 M | RESULTS | the default's point count (the same 512^3 global grid) at order 8 |
| n12-e78125 | 12 | 78,125 | 135.0 M | RESULTS | 135 M points at order 12 (5^7 elements, a 125 x 25 x 25 brick) |

Every pair `SIZE` + `data.rea` is upstream's `test/nek_gpu1` with `lx1` / `lelt` and the matching
`nx0..nxN` / `iel0..ielN` changed (one order, one element count per input; brick geometry, no multigrid,
automatic decomposition). `test/nek_gpu1` as shipped (256 elements of order 16, 1.0 M points) is the case
`validate.sh` runs; its timed solve takes RESULTS on the B200 -- too short to time.

## Timed region (ROI)

`driver.f` runs the CG twice per (nelt, nx1): a warm-up solve and the timed one between `set_timer_flop_cnt(0)`
and `(1)` (`dnekclock()` = `MPI_Wtime`), printed as `Solve Time = <s>`. The ROI markers sit immediately
around that second `cg_acc` call: 100 iterations of preconditioner copy, inner products (`glsc3_acc`,
device reductions + `MPI_Allreduce`), `add2s1_acc` / `add2s2_acc`, the CUDA Fortran `ax_cuf2` matrix-vector
product per element, the OpenACC gather-scatter `dssum_acc` and the mask. Set-up (mesh, gather-scatter
handles, geometric factors), the host-side right-hand side, the warm-up solve, the solution-functional print
and the flop summary are outside. The OpenACC constructs are synchronous (no `async`), so the host clock sees
the device work; the ROI's own synchronise is `cuCtxSynchronize` (tools/timing/roi). ROI = `Solve Time`.

## Validation (`validate.sh`)

Upstream's test harness (`test/test_nekbone.py`) builds the serial, OpenACC-device, OpenACC-host and
CUDA-device variants of `test/nek_gpu1` and asserts that every `cg:` line (residual norm, alpha, beta,
p^T A p, 5 significant digits) of the GPU runs equals the serial run's. `validate.sh` does the same with
this repository's two builds (`build.sh CUDA nek_gpu1`, `build.sh CPU nek_gpu1`): both run
`test/nek_gpu1/data.rea`, the `cg:` lines and the solution functionals are compared, the exact equality
upstream asserts is reported, and the verdict applies the rule of `check_cg.py` (below). Result (dgx003, 2026-10-21): **PASS** -- every `cg:` line of the CUDA run equals the CPU run's (upstream's exact assertion holds: `cg: 0 5.4586E+02`, `cg: 101 4.3425E-09 5.3233E-01 5.6261E-01 6.0577E-17`, second solve `5.6459E+02` / `4.8352E-09`), solution norm `9.33568182365457E+02` and functional `3.51509403984455E+05` to 1e-16 relative; the OpenACC code path run on the host (`-acc=multicore`) gives the same numbers.

### Correctness of the registered inputs (`check_cg.py`, frozen references)

For every registered input a reference run of the CPU build (same `SIZE` + `data.rea`, same source, host
code path) is frozen once in `reference/<id>.json` (`check_cg.py --make-reference`, provenance and rule
inside). The check compares the timed solve of a fresh CUDA run with it:

- `r0` (the weighted norm of the masked right-hand side, printed to 5 digits) within `r0_tol` relative:
  the right-hand side, the mask and the multiplicity;
- `sqrt(x^T C x)` and `x^T C f` of the solution (16 digits) within `x_tol` relative: the solution of the
  SPD system is unique, a wrong operator (geometric factors, gather-scatter, mask) converges to another one;
- the residual reduction `rN / r0` of the 100 iterations within `reduction_factor` times the reference's;
- `nelt`, `np`, `nx1`, the iteration count and the program's normal end exact.

Basis of the tolerances (measured 2026-10-21 on dgx003, before any optimized candidate existed, with the
set-up fix above): on `test/nek_gpu1` and on every registered input the CPU build and this CUDA build agree
**exactly** in every printed `cg:` quantity of both solves (r0, rN, alpha, beta, p^T A p to the 5 printed
digits) and in the solution functionals to the last printed digit:

| input | r0 (timed solve) | rN / r0 | sqrt(x^T C x) CPU vs CUDA (rel) | x^T C f (rel) |
|---|---|---|---|---|
| n16-e8192 | 3.13800E+03 | 7.79e-12 | 0 | 3.7e-16 |
| n16-e16384 | 4.43080E+03 | 7.72e-12 | 0 | 1.9e-16 |
| n16-e32768 | 6.25640E+03 | 7.65e-12 | 1.9e-16 | 5.6e-16 |
| n8-e262144 | 5.31410E+03 | 4.55e-10 | 1.9e-16 | 2.4e-16 |
| n12-e78125 | 5.96150E+03 | 2.33e-11 | 1.2e-16 | 2.6e-16 |

So `r0_tol` = 1e-4 (one unit of the 5th printed digit: the printed value cannot deviate by less than that
without being equal), `x_tol` = 1e-10 (1e5 x the observed deviation; still 10 x below the 1e-9 level at
which a lost or extra CG iteration moves the solution at these residual reductions -- the iteration count is
checked exactly anyway, and a wrong operator or gather-scatter moves the solution at order 1), the residual
reduction within 10 x the reference's (observed ratio 1.00 on every input; a broken reduction, lost
symmetry or lost iteration leaves it orders of magnitude above). The references are frozen in
`reference/<id>.json` with this rule and the CPU run's provenance; the checker applies the recorded rule.

Observations on the upstream solver that the check has to live with (not changed): the two solves of a
run do not start from the same masked right-hand side -- `maskit` (cg.f) records the Dirichlet entries on
its first call with an index local to the first element and reuses that list on the second call, so the
warm-up solve and the timed solve mask different entries (both `cg:   0` norms are printed; the reference
and the run are compared solve by solve). The right-hand side `sin(cos(i*i))` overflows a 32-bit integer
for i > 46340; nvfortran 25.7 evaluates it identically under every flag set used here (checked), and the
CPU and CUDA builds compile the same host routine. Upstream's 2018 PGI sample output
(`test/nek_gpu1/scripts/master_branch.output`) starts from `r0 = 6.1974E+02`, close to the value the
**unfixed** GPU build prints (`6.1811E+02`, the raw unassembled norm) and not to the assembled one
(`5.4586E+02` from both this repository's CPU path and its OpenACC path run on the host): whether the 2018
master-branch run was assembled is not established here.

## Results on dgx003 (B200, CUDA 13.2.78, NVIDIA HPC SDK 25.7)

RESULTS_PLACEHOLDER

## Limitations

- Single-GPU workloads registered; multi-GPU weak scaling (`HPCPERF_GPUS=N`) runs but is not validated here;
  multi-node BLOCKED (site). No HIP build (OpenACC + CUDA Fortran).
- The multigrid preconditioner (`hsmg.f`, `MGRID`) has no OpenACC path upstream; all inputs use the
  identity preconditioner as `test/nek_gpu1` does.
- 268 M points per rank (65,536 elements of order 16) crash in upstream's jl gather-scatter set-up
  (`gs_setup` -> `get_topology` -> `jl_sortp_ui`, 32-bit sort / index arithmetic) in both builds: 134 M
  points per rank is the largest registered size; larger problems need more ranks.
- The host-side set-up (gather-scatter handles for 134 M points, geometric factors, the right-hand side) is
  single-threaded upstream code and takes 4-9 minutes per run at the registered sizes -- outside the ROI,
  but it sets the wall time of a measurement.
- The NVIDIA HPC SDK is site-provided (not in the conda environment); the CUDA runtime the binary uses is
  the SDK's bundled 12.9 under the system 13.2 driver.

## LOC

`src/` (cloc code lines, `deprecated/` and `jl/ext` excluded): 56,635 in 140 files -- Nekbone proper
(driver, cg, comm_mpi, math, prox_dssum, prox_setup, semhat, speclib, hsmg_dum, ax_cuda, mxm_wrapper,
driver_comm, delay_dum: Fortran 77 + CUDA Fortran) 4,360; bundled numerical libraries (blas.f 16,627,
dsygv.f 3,724, mxm_std.f 3,576) 23,927; the `jl` gather-scatter library (C 16,889, headers 4,627) 21,516;
the unused Blue Gene assembly and CUDA kernel variants (`nek_kernel*.cuf`, not built) the rest.

## Status

RESULTS_STATUS
