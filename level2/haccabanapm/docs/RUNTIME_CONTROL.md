# Runtime control

How to steer a run without rebuilding: environment variables, device
selection, MPI topology, and timing instrumentation.

## Environment-variable reference

Three categories. Only the first group is read by pmkokkos itself; the rest
are consumed by the compiler, the MPI implementation, or the runtime, and
are listed because a correct run depends on them.

### Read by pmkokkos

Exhaustive — these are the only `getenv` calls in the code base.

| Variable | Read in | Values | Effect |
|---|---|---|---|
| `PMKOKKOS_TOPOLOGY` | `apps/TopologyOverride.hpp` | `"Px,Py,Pz"` or `"Px Py Pz"` | Forces the MPI Cartesian decomposition. Highest precedence — beats the indat and the snapshot. |
| `PMKOKKOS_TIMING` | `apps/pm_run_lib.cpp` | any value except empty or `0` | Enables per-step wall timing in `pm_run`. |
| `PMKOKKOS_TIMING_WARMUP` | `apps/pm_run_lib.cpp` | integer ≥ 0 | Steps excluded from the timing summary. Default 5. Ignored unless `PMKOKKOS_TIMING` is set. |

### Build-time (CMake, not environment)

These are `cmake -D` options, listed here so the full control surface is in
one place. Details in [BUILD.md](BUILD.md#cmake-options).

| Option | Default | Effect |
|---|---|---|
| `PMKOKKOS_VECTOR_LENGTH` | `16` | AoSoA inner SIMD width |
| `PMKOKKOS_ENABLE_TESTS` | `ON` | build the test suite |
| `PMKOKKOS_FP_PRECISE` | `OFF` | `-fp-model=precise -ffp-contract=off` |

### External — required for correct runs

Not read by pmkokkos, but a run is wrong or fails without them.

| Variable | Where | Required value | Why |
|---|---|---|---|
| `MPIR_CVAR_ENABLE_GPU` | Aurora / MPICH | `1` | **Mandatory for multi-rank.** `Cabana::Distributor` posts MPI with device pointers. See [below](#gpu-aware-mpi). |
| `ZE_AFFINITY_MASK` | Intel Level Zero | set per rank | Maps each rank to one PVC tile. Usually set by a wrapper script, not by hand. |
| `ONEAPI_DEVICE_SELECTOR` | Intel oneAPI | `level_zero:gpu` | Selects the Level Zero GPU backend for SYCL. |
| `OMP_NUM_THREADS` | OpenMP | site-dependent (8 on Aurora) | Host thread count. The only relevant knob for a CPU build. |
| `LD_LIBRARY_PATH` | loader | must include the parallel HDF5 lib dir | Aurora's module set needs the spack-built `libhdf5` made explicit. |

**Variables that do not exist.** `docs/PROFILING_GUIDE.md` §9 describes
kernel-dump hooks — `PMKOKKOS_KICK_DUMP_STEP`, `PMKOKKOS_KICK_DUMP_DIR`,
`PMKOKKOS_DUMP_KICK_F`. These are **not present on `master`**; they exist
only on experiment branches. Setting them on a `master` build does nothing.

---

## GPU vs CPU selection

There is no runtime switch. The execution space is fixed at build time by
whichever Kokkos install was resolved — see
[BUILD.md § GPU vs CPU builds](BUILD.md#gpu-vs-cpu-builds). To run on CPU,
build against a Kokkos with an OpenMP or Serial default space and use that
binary.

For a CPU build:

```sh
export OMP_NUM_THREADS=8
unset ZE_AFFINITY_MASK ONEAPI_DEVICE_SELECTOR MPIR_CVAR_ENABLE_GPU
mpiexec -n 8 ./pm_run ic.h5 out.h5 indat.params
```

Keeping `MPIR_CVAR_ENABLE_GPU=1` set for a host-only build is not
useful — the buffers are host pointers — and on some MPI builds it adds
overhead.

---

## MPI topology

pmkokkos decomposes the mesh into a 3D Cartesian block grid of
`Px × Py × Pz` ranks. **`Px·Py·Pz` must equal the rank count**; a mismatch
prints an error on rank 0 and calls `MPI_Abort`.

### Precedence

Highest first. Implemented in `apps/TopologyOverride.hpp`.

| # | Source | Applies to |
|---|---|---|
| 1 | `PMKOKKOS_TOPOLOGY` environment variable | `pm_ic`, `pm_run` |
| 2 | `TOPOLOGY` key in `indat.params` | `pm_ic`, `pm_run` |
| 3 | The snapshot's `topology_dims` attribute | `pm_run` only |
| 4 | `MPI_Dims_create` + descending sort | both, as fallback |

The environment variable outranking the snapshot is what lets you
**re-decompose an existing IC at run time** — generate once at 2×2×2, then
evolve the same file at 4×2×1 to study the decomposition's effect.

The default (rule 4) sorts factors descending so the largest goes on
dimension 0, matching HACC's `MY_Dims_create_3D`. For 1, 2, 4, and 8 ranks
both codes agree: `(1,1,1)`, `(2,1,1)`, `(2,2,1)`, `(2,2,2)`. The sort is
also why the default can never produce e.g. `2×2×3` for 12 ranks — you get
`3×2×2`. Override explicitly if you need a different ordering.

### Examples

```sh
# Explicit 4x2x1 on 8 ranks
PMKOKKOS_TOPOLOGY=4,2,1 mpiexec -n 8 ./pm_run ic.h5 out.h5 indat.params

# Same thing via the indat
echo "TOPOLOGY 4 2 1" >> indat.params
```

### Divisibility

If `ng` is not divisible by some axis's rank count, you get a **warning,
not an error**:

```
WARNING: ng=128 is not divisible by topology 3x2x2; blocks are uneven
```

heFFTe tolerates uneven pencils and the transform length is unchanged, but
load balance drifts and comparability against HACC weakens. For `pm_run`
this check runs after the snapshot is read, since `ng` comes from the file.

---

## GPU-aware MPI

**The single most common cause of multi-rank failure.**

`Cabana::Distributor` — used by every `migrate()` call, which happens twice
per DKD step — posts `MPI_Isend`/`MPI_Irecv` directly on device (SYCL/USM)
pointers. The MPI implementation must be able to read that memory.

On Aurora:

```sh
export MPIR_CVAR_ENABLE_GPU=1
```

Two things make this trap worse than it sounds:

1. **HACC's own jobscripts set `MPIR_CVAR_ENABLE_GPU=0`**, because HACC
   stages all MPI buffers through host memory. Copying a HACC jobscript is
   an easy way to get this exactly wrong.
2. **Small cases pass without it.** A 4-particle 4-rank test ran clean;
   the failure only appeared at thousand-particle scale, where buffers grow
   past whatever the MPI layer was quietly handling. A passing small test is
   not evidence that the setting is right.

The equivalent knob on other stacks: `MPICH_GPU_SUPPORT_ENABLED=1` on Cray
MPICH (Frontier), or a CUDA-aware OpenMPI/MPICH build on Perlmutter. Those
paths are not yet validated — see [SYSTEMS.md](SYSTEMS.md).

---

## Tile affinity

Aurora nodes carry 6 PVC GPUs × 2 tiles = 12 tiles. The reference
configuration runs **one rank per tile**, with `ZE_AFFINITY_MASK` assigning
the mapping.

Do not set `ZE_AFFINITY_MASK` globally in your shell — it must differ per
rank. The standard approach is a wrapper script that computes the mask from
the local rank ID and then `exec`s the binary:

```sh
mpiexec -n 8 -ppn 8 --envall ./gpu_tile_compact.sh ./pm_run ic.h5 out.h5 indat.params
```

ALCF distributes `gpu_tile_compact.sh`; copies live alongside the reference
run directories. Every rank sharing tile 0 — the symptom of a missing or
global mask — shows up as severe slowdown rather than an error.

---

## Per-step timing

`pm_run` has opt-in wall-clock instrumentation for scaling studies.

```sh
export PMKOKKOS_TIMING=1
export PMKOKKOS_TIMING_WARMUP=5     # optional, this is the default
mpiexec -n 8 ./pm_run ic.h5 out.h5 indat.params
```

### What it measures

Each step is bracketed by an `MPI_Barrier` before `MPI_Wtime()`, so the
delta reflects the whole rank set arriving together. The reported per-step
cost is the **maximum over ranks** — the slowest rank is the true scaling
wall. Rank 0 emits the results.

### Output

A provenance line, then CSV, then a summary:

```
pm_run_timing: provenance ranks=8 topology=2x2x2 ng=128 np=128 vector_length=16 nsteps=500 warmup=5
pm_run_timing: step,t_step_max_s,t_step_rank0_s
pm_run_timing: 0,0.0431,0.0428
pm_run_timing: 1,0.0402,0.0399
...
pm_run_timing: summary steps_timed=500 window=[5,500) median_s=0.0398 mad_s=0.0011 total_s=21.6
```

The provenance line records everything needed to interpret the numbers,
including the compiled-in `vector_length`.

The summary reports **median and median-absolute-deviation** over the
window `[warmup, nsteps)`, not mean and standard deviation — robust to the
occasional outlier step from system noise. `total_s` covers **all** steps
including warmup.

Warmup matters: the first few steps pay FFT plan setup, allocation, and
first-touch costs. Five is the default; raise it if the CSV shows the
transient lasting longer.

### Reference timings

`NG=128`, 500 steps, cosmological IC, Aurora PVC:

| Ranks | Wall |
|---|---|
| 1 | ~6 s |
| 8 | ~22 s |

The 8-rank run being *slower* is not a bug — at `ng=128` the per-rank work
is small enough that MPI `Alltoall` in the FFT dominates. Scaling studies
need `NG=256` or larger before the network is actually loaded.

For kernel-level attribution rather than per-step walls, see
[PROFILING_GUIDE.md](PROFILING_GUIDE.md).
