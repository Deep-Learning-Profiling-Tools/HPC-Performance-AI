# pmkokkos profiling guide (Aurora PVC)

> **Accuracy note (added with the docs/ guide set).** Parts of this guide
> were written against experiment branches and do not apply to `master`:
>
> - **§9's env hooks** (`PMKOKKOS_KICK_DUMP_STEP`, `PMKOKKOS_KICK_DUMP_DIR`,
>   `PMKOKKOS_DUMP_KICK_F`) **do not exist on `master`.** The only
>   environment variables the code reads are `PMKOKKOS_TIMING`,
>   `PMKOKKOS_TIMING_WARMUP`, and `PMKOKKOS_TOPOLOGY`.
> - **§2's `PMKOKKOS_FFT_BACKEND_FFTW` / `_SWFFT` options do not exist.**
>   The FFT backend is not selectable at CMake time; it follows the Kokkos
>   memory space. See [BUILD.md](BUILD.md#backend-selection).
> - **§9's `pm_single_kick` invocation is wrong.** Its third positional
>   argument is `a_fin` (a number), not an indat path. See
>   [RUNNING.md](RUNNING.md#pm_single_kick-diagnostic).
> - **§11's `pmk::deposit_hacc`** is an experiment-branch CIC backend, not
>   the `master` deposit (which is `Cabana::Grid::p2g` with `Spline<1>()`).
>
> The profiler invocations, module setup, and reference timings below are
> accurate. See [RUNTIME_CONTROL.md](RUNTIME_CONTROL.md) for the verified
> environment-variable reference.

How to run `pm_run` for performance profiling on Aurora compute nodes.
Covers build configuration, run launch patterns, and the four profilers
available in the Aurora oneAPI module: **`unitrace`** (preferred for
kernel-level timeline + GPU metrics), **`onetrace`** (lighter-weight CLI
trace), **`vtune`** (CPU + GPU sampling profiler), and **`advisor`**
(roofline / vector advisor).

Reference reproduction config: NG=NP=128, RL=128 Mpc/h, 500 DKD steps,
cosmological IC (CMB transfer, seed 5126873).  Production wall on PVC:
**6.0 s at 1 rank**, **21.5 s at 8 ranks**.  See
`HACC/session_saves/attribution_20260619_174215/ATTRIBUTION_RESULT.md`
for the full reference run that produced these timings.

## 1. Module + env setup

```bash
# Aurora compute node, fresh shell:
module load gcc/13.4.0 oneapi/release/2025.3.1 \
            mpich/opt/5.0.0.aurora_test.3c70a61 \
            libfabric/1.22.0 cray-pals/1.8.0 cray-libpals/1.8.0
module load hdf5/1.14.6

# pmk's h5py path needs the spack-built libhdf5.so.310 explicitly:
export LD_LIBRARY_PATH=/opt/aurora/25.190.0/spack/unified/0.10.1/install/linux-sles15-x86_64/oneapi-2025.2.0/hdf5-1.14.6-kimgxz3/lib:${LD_LIBRARY_PATH:-}

# SYCL/Level Zero device selection:
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export MPIR_CVAR_ENABLE_GPU=1
export OMP_NUM_THREADS=8
```

## 2. Production build

pmk-master, FP-precise, heFFTe with oneMKL DFT backend, Cabana CIC,
SYCL execution space (Kokkos DefaultExecutionSpace = SYCL on Aurora).
Binary already at `build_fp_precise/pm_run` (5 MB).  In the commands below,
`$PMK_ROOT` is your pmkokkos checkout (set it to the absolute path).

```bash
cd $PMK_ROOT/build_fp_precise
make -j 16
```

Build option summary (from CMakeCache.txt):
- `PMKOKKOS_FP_PRECISE=ON` → `-fp-model=precise -ffp-contract=off`
- `PMKOKKOS_VECTOR_LENGTH=16` (PVC inner SIMD width)
- `CMAKE_BUILD_TYPE=Release` (`-O3`)
- Default FFT backend (no `PMKOKKOS_FFT_BACKEND_FFTW`/`_SWFFT`) → heFFTe → oneMKL DFT on SYCL.

For comparison runs:
- `build_fp_precise_fftw` — heFFTe with FFTW3 backend on host
- `build_fp_precise_cpu` — full Kokkos OpenMP exec space (no SYCL)
- `build_cic_hacc` — diagnostic CIC backend port; not for profiling production

Don't profile with `PMKOKKOS_FP_PRECISE=OFF` unless the goal is to measure
FMA-fused performance — production runs always have it ON.

## 3. Inputs needed at runtime

Two files per run plus the `gpu_tile_compact.sh` helper:

```
indat.params         # HACC indat (NG, NP, OL, N_STEPS, FULL_ALIVE_DUMP, ...)
shared_ic.h5         # pmk schema-v2 IC h5
gpu_tile_compact.sh  # PVC tile affinity wrapper (one copy per OUT dir)
```

The IC h5 is built once via:
```bash
python3 $ROOT/analysis/scripts/gio_ic_to_hdf5.py \
    <HACC GIO IC path> <out.h5> <indat.params>
```
or generated natively by pmk:
```bash
mpiexec -n 1 -ppn 1 ./gpu_tile_compact.sh \
    $ROOT/pmkokkos/build_fp_precise/pm_ic indat.params out.h5
```

Stable references on disk:
- 8-rank-OL=4 cosmological indat: `run_hacc/1-node-grav_full/indat.params`
- 1-rank-OL=0 cosmological indat: `session_saves/attribution_20260619_174215/tmp_attribution_8r/indat.params` (NG=128, OL=0, FULL_ALIVE_DUMP=499)
- 8-rank IC h5: `session_saves/attribution_20260619_174215/tmp_attribution_8r/shared_ic.h5`
- CMB transfer fn: `run_hacc/1-node-grav_full/cmbM000.tf`
- `gpu_tile_compact.sh`: `run_hacc/1-node-grav_full/gpu_tile_compact.sh`

## 4. Bare run (no profiler) — reference timing

```bash
OUT=/tmp/pmk_profile_baseline
mkdir -p $OUT
cp run_hacc/1-node-grav_full/gpu_tile_compact.sh $OUT/
chmod +x $OUT/gpu_tile_compact.sh
cd $OUT

NRANKS=8                                # or 1 for single-tile
time mpiexec -n $NRANKS -ppn $NRANKS --envall \
    ./gpu_tile_compact.sh \
    $PMK_ROOT/build_fp_precise/pm_run \
    <shared_ic.h5> <out_evolved.h5> <indat.params> 2>&1 | tee run.log
```

Expected wall (NG=128, 500 steps, cosmological IC):
- 1 rank: ~6 s
- 8 ranks: ~22 s (MPI/Alltoall overhead dominates at this problem size)

For meaningful kernel-timeline profiling, prefer **1 rank** at NG=128 —
the per-step kernel sequence is identical to multi-rank but unobscured by
inter-rank waits.  For MPI/scaling studies, use 8 ranks at production
NG=256+ where the FFT actually loads the network.

## 5. Profiler A: `unitrace` (preferred — PTI-GPU timeline + GPU metrics)

Aurora-recommended tool.  Captures Level Zero call timeline, SYCL kernel
durations, GPU EU occupancy, memory bandwidth.  Output is a JSON
chrome-trace + per-kernel CSV.

```bash
UNITRACE=/opt/aurora/26.26.0/support/tools/pti-gpu/0.16.0-rc1/bin/unitrace

# Sequential kernel summary, EU active, memory bandwidth:
mpiexec -n 1 -ppn 1 --envall \
    ./gpu_tile_compact.sh \
    $UNITRACE --chrome-device-timeline --chrome-call-logging \
              --device-timing --kernel-submission \
              --metric-query --output pmk_unitrace \
    $PMK_ROOT/build_fp_precise/pm_run \
    <ic.h5> <out.h5> <indat.params>
```

Outputs to inspect:
- `pmk_unitrace.<pid>.json` — Chrome trace; load in chrome://tracing or [perfetto.dev/viewer](https://perfetto.dev/viewer)
- `pmk_unitrace.<pid>.csv` — per-kernel time/count
- stderr — text summary of kernel hot list

Common flags:
- `--kernel-submission` — record sycl-queue submit/host-time gap (latency hiding indicator)
- `--metric-query` — sample GPU hardware counters per kernel
- `--call-logging` (CPU side) + `--device-timing` (GPU side) — minimal pair
- `--demangle` — readable Kokkos kernel names

NG=128, 500-step run produces ~10K kernel launches — JSON will be tens of MB.
For NG=256+ runs trim with `--max-kernel-count` or sample a single step
via the pmkokkos kernel-dump env hooks (see §9).

## 6. Profiler B: `onetrace` (lighter, CLI-friendly)

CSV-only, no GUI trace.  Good for quick hot-kernel lists.

```bash
ONETRACE=/opt/aurora/26.26.0/support/tools/pti-gpu/0.14.0/bin/onetrace

mpiexec -n 1 -ppn 1 --envall \
    ./gpu_tile_compact.sh \
    $ONETRACE --device-timing --kernel-submission \
              --output pmk_onetrace.csv \
    $PMK_ROOT/build_fp_precise/pm_run \
    <ic.h5> <out.h5> <indat.params>
```

## 7. Profiler C: `vtune`

Heavier setup; best for CPU-side bottlenecks and the host↔device boundary
(MPI + Cabana migration kernels).  GPU-side `gpu-offload` and
`gpu-hotspots` collection types work on PVC via Level Zero.

```bash
# GPU hotspots — kernel-level
mpiexec -n 1 -ppn 1 --envall \
    ./gpu_tile_compact.sh \
    vtune -collect gpu-hotspots -result-dir pmk_vtune_gpu \
          -knob characterization-mode=overview \
    $PMK_ROOT/build_fp_precise/pm_run \
    <ic.h5> <out.h5> <indat.params>

# Report:
vtune -report summary -result-dir pmk_vtune_gpu
vtune -report hotspots -result-dir pmk_vtune_gpu -group-by computing-task
```

GUI: copy result dir back to a workstation and open with the vtune client,
or run `vtune-gui` over X-forwarded SSH (slow).  CLI reports cover most
needs.

## 8. Profiler D: `advisor` (roofline)

Only relevant if you want a roofline plot for a specific kernel.  Heavy
overhead; recommend instrumenting a single representative step via the
pmkokkos `PMKOKKOS_KICK_DUMP_STEP` hook (§9) to bound the trace.

```bash
mpiexec -n 1 -ppn 1 --envall \
    ./gpu_tile_compact.sh \
    advisor --collect=roofline --target-gpu \
            --project-dir=pmk_advisor \
    $PMK_ROOT/build_fp_precise/pm_run \
    <ic.h5> <out.h5> <indat.params>
advisor --report=roofline --project-dir=pmk_advisor --report-output=pmk_roofline.html
```

## 9. pmkokkos kernel-instrumentation env hooks (use to bound trace scope)

pmk has built-in env-gated dump hooks that write FP32 rho, FP64 rho_k,
FP32 grad_phi, FP64 grad_phi_k, and per-particle (id, pos, vel) at
specific stages of a chosen step.  Useful for bounding profiler traces
to one step's worth of kernels.

```bash
export PMKOKKOS_KICK_DUMP_STEP=10           # 0-indexed; pick a step that's representative
export PMKOKKOS_KICK_DUMP_DIR=/tmp/pmk_step10_dump
mkdir -p $PMKOKKOS_KICK_DUMP_DIR

# Optionally also dump per-particle kickF (= gather force value pre-coeff):
export PMKOKKOS_DUMP_KICK_F=1
```

These hooks fire only on the matching step, so the per-kernel cost of
each fires once per run.  Pair with a shortened `N_STEPS 12` indat to
keep run time minimal while still exercising warm-up + one fully
instrumented step.

For an even smaller probe, use the pm_single_kick binary:
```
./pm_single_kick <ic.h5> <out.h5> <indat.params>
```
Performs one PM kick (deposit + FFT + Greens + iFFT × 3 + gather × 3) on
the IC and exits — useful for kernel microbenchmarking without DKD overhead.

## 10. Suggested first-time profiling sweep

For an initial pmk performance characterization, run these four configurations
and compare:

| Run | Ranks | N_STEPS | Profiler | Purpose |
|---|---|---|---|---|
| A | 1 | 500 | `unitrace --device-timing` | Kernel hot list, no MPI |
| B | 8 | 500 | `unitrace --device-timing --metric-query` | Multi-rank kernel + EU/BW |
| C | 1 | 12 (`PMKOKKOS_KICK_DUMP_STEP=10`) | `vtune -collect gpu-hotspots` | Single-step deep dive |
| D | 8 | 500 | `unitrace --kernel-submission` | Submission-latency / queue depth |

Reference walls for the baseline (no profiler):
- A: ~6 s     · profiled: ~10–20 s
- B: ~22 s    · profiled: ~30–60 s
- C: ~1–2 s
- D: ~22 s    · profiled: ~30–60 s

## 11. Things to check in the profiler output

Known hot kernels (from session experience, in rough cost order at NG=128):
1. `pmk::deposit_hacc` (CIC scatter with OMP locks) — host OMP, not on GPU
2. `heffte::*` forward + reverse FFT (oneMKL DFT) — GPU, many small batched 1D FFTs
3. `pmk::apply_greens_and_gradient` — GPU, k-space multiply
4. `pmk::gather_force_value` — GPU CIC gather, 3 axes
5. `Cabana::Distributor::migrate` — MPI Alltoall, pre-kick + post-step

Things to look at:
- **EU active %** in the FFT phase — should be 50–80%; if much lower, oneMKL
  is launching too many small FFTs serially (look at `--kernel-submission`
  for back-to-back kernel gaps)
- **Memory BW utilization** in deposit/gather — these are bandwidth-bound;
  expect ~70% of peak HBM BW (~1 TB/s on PVC)
- **MPI wait time** in `Cabana::Distributor` at 8-rank — should be a small
  fraction of total step at NG=128, dominates at NG=64
- **CPU↔GPU memcpy bursts** in the deposit lane — pmk runs CIC scatter
  on host OMP then deep-copies rho_array to device for the FFT.  These
  copies are visible as `zeMemCopy` blocks in the unitrace timeline.

## 12. Cleanup

Profiler outputs can be large (GB-scale chrome traces for 8-rank runs).
Don't leave them under `/tmp` between sessions on a shared node.  Move
artifacts to `$PMK_ROOT/profiling_runs/<date>/` on the parallel filesystem
for sharing.

## 13. References

- Aurora user guide on profiling: [docs.alcf.anl.gov/aurora/performance-tools](https://docs.alcf.anl.gov/aurora/performance-tools/)
- PTI-GPU (unitrace/onetrace) source: github.com/intel/pti-gpu
- Intel VTune docs: intel.com/content/www/us/en/docs/vtune-profiler
- Kokkos kernel naming guide: kokkos.org/kokkos-core-wiki/ProgrammingGuide/Profiling.html
- Reference cross-code residual closure: `analysis/WORKLOG.md` entry 2026-06-19
  (`skip_ngeom_attribution_closure`) and
  `HACC/session_saves/attribution_20260619_174215/ATTRIBUTION_RESULT.md`.
