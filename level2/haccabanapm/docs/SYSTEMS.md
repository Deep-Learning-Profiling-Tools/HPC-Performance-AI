# Systems

Per-machine build and run recipes.

| System | Accelerator | Kokkos backend | FFT backend | Status |
|---|---|---|---|---|
| [Aurora (ALCF)](#aurora-alcf) | Intel PVC | SYCL | oneMKL | **Complete — validated** |
| [Frontier (OLCF)](#frontier-olcf) | AMD MI250X | HIP | rocFFT | *Stub — to be added* |
| [Perlmutter (NERSC)](#perlmutter-nersc) | NVIDIA A100 | CUDA | cuFFT | *Stub — to be added* |
| [CPU + FFTW](#cpu--fftw) | none | OpenMP / Serial | FFTW3 | *Stub — to be added* |

Aurora is the only platform on which this code has been built, run, and
validated. The other three sections record what is known and what still has
to be determined; they are placeholders, not instructions.

---

## Aurora (ALCF)

HPE Cray-EX. Intel Xeon CPU Max (Sapphire Rapids, HBM) hosts, Intel Data
Center GPU Max 1550 (Ponte Vecchio) accelerators, Slingshot-11
interconnect. All development and validation ran here.

### Validated toolchain

| Component | Version | Notes |
|---|---|---|
| Kokkos | 4.6.02 | SYCL backend for PVC |
| Cabana | 0.7.0 | `Cabana_ENABLE_GRID=ON`, `Cabana_REQUIRE_HEFFTE=ON` |
| heFFTe | 2.4.0 | oneMKL backend auto-selected on SYCL |
| FFTW | 3.3.10 | CPU comparison builds only |
| HDF5 | 1.14.6 | system module; parallel (MPI-IO) enabled |
| GoogleTest | 1.15.2 | |
| CMake | 3.31.11 | system module |
| oneAPI compilers | 2025.3 | |

Kokkos, Cabana, heFFTe, FFTW, and GoogleTest were built from source. HDF5
and CMake come from modules.

### Modules

For a build:

```sh
module load cmake/3.31.11 hdf5/1.14.6
```

HDF5 is needed even if you think you aren't using it directly — Cabana's
package config pulls it in through `find_dependency`.

For a run on a compute node:

```sh
module load gcc/13.4.0 oneapi/release/2025.3.1 \
            mpich/opt/5.0.0.aurora_test.3c70a61 \
            libfabric/1.22.0 cray-pals/1.8.0 cray-libpals/1.8.0
module load hdf5/1.14.6
```

### Build

```sh
module load cmake/3.31.11 hdf5/1.14.6

cmake -B build -DCMAKE_PREFIX_PATH=/path/to/cabana/install \
      -DCMAKE_BUILD_TYPE=Release \
      -DPMKOKKOS_FP_PRECISE=ON
cmake --build build -j 16
```

`PMKOKKOS_FP_PRECISE=ON` is the production setting here — every validated
run used it. It adds `-fp-model=precise -ffp-contract=off`.

For bit-level cross-code comparison against HACC, the full matched-flag
configuration applied to **both** codes was:

```
-fp-model=precise -ffp-contract=off -fimf-precision=high
```

The CMake option supplies the first two; add `-fimf-precision=high` via
`CMAKE_CXX_FLAGS` if you need the third. Bit-level results are sensitive to
both the compiler version and these flags — record them with any baseline.

### Runtime environment

```sh
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export MPIR_CVAR_ENABLE_GPU=1          # REQUIRED for multi-rank
export OMP_NUM_THREADS=8

# The parallel HDF5 runtime must be explicit on the loader path:
export LD_LIBRARY_PATH=/opt/aurora/25.190.0/spack/unified/0.10.1/install/linux-sles15-x86_64/oneapi-2025.2.0/hdf5-1.14.6-kimgxz3/lib:${LD_LIBRARY_PATH:-}
```

Adjust the HDF5 path to match the loaded module — it changes between
Aurora software stack releases.

### Job script pattern

Reference configuration: 1 node, 8 MPI ranks, 2×2×2 topology, one rank per
PVC tile.

```bash
#!/bin/bash
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -l filesystems=flare
#PBS -q debug
#PBS -A <project>

cd "$PBS_O_WORKDIR"

module load gcc/13.4.0 oneapi/release/2025.3.1 \
            mpich/opt/5.0.0.aurora_test.3c70a61 \
            libfabric/1.22.0 cray-pals/1.8.0 cray-libpals/1.8.0
module load hdf5/1.14.6

export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export MPIR_CVAR_ENABLE_GPU=1
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH=<parallel-hdf5-lib-dir>:${LD_LIBRARY_PATH:-}

BIN=/path/to/pmkokkos/build/
NRANKS=8

# gpu_tile_compact.sh assigns ZE_AFFINITY_MASK per rank (one rank per tile).
# ALCF distributes it; copy one into the run directory and chmod +x.

mpiexec -n $NRANKS -ppn $NRANKS --envall ./gpu_tile_compact.sh \
    $BIN/pm_ic indat.params ic.h5

mpiexec -n $NRANKS -ppn $NRANKS --envall ./gpu_tile_compact.sh \
    $BIN/pm_run ic.h5 evolved.h5 indat.params
```

Both the transfer-function file named by `INPUT_BASE_NAME` and
`gpu_tile_compact.sh` must be in the working directory.

### Reference run

- 1 node, 8 ranks, 2×2×2
- `NG = NP = 128`, `RL = 128` Mpc/h
- 500 DKD steps, z=200 → z=0
- Wall: ~6 s at 1 rank, ~22 s at 8 ranks

### Operational pitfalls

These are real failures that were hit during development, in rough order of
how much time they cost.

**`MPIR_CVAR_ENABLE_GPU=1` is mandatory and easy to get wrong.**
`Cabana::Distributor` posts MPI with device pointers. HACC's own jobscripts
set this to `0` (HACC stages through host memory), so copying a HACC script
gets it backwards. Small test cases pass without it — the crash appears at
scale. See
[RUNTIME_CONTROL.md § GPU-aware MPI](RUNTIME_CONTROL.md#gpu-aware-mpi).

**Stale `palsd` daemons wedge the launcher.** After a crashed `mpiexec`,
every subsequent `mpiexec` on that node — including `mpiexec -n 1 hostname`
— hangs until the 120 s timeout. Killing `palsd` does not help. **Only a
fresh PBS allocation recovers.** If the launcher starts timing out on
trivial commands, stop debugging your code and get a new allocation.

**SYCL's default Kokkos View layout is `LayoutLeft`.** Any View staged to
or from HDF5 must be declared explicit `LayoutRight`, because HDF5
hyperslabs for `(N,3)` datasets are row-major. A single-rank round-trip
will *not* catch a mistake here — the same scrambling happens on read and
cancels out. The multi-rank round-trip is the canary; the symptom is a
clean axis transpose in the recovered data.

**Don't profile with `PMKOKKOS_FP_PRECISE=OFF`** unless you specifically
want FMA-fused numbers — production runs always have it ON, so an OFF build
is not comparable to the reference timings.

**Single-rank OL=0 HACC comparison runs** need `HACC_SINGLE_RANK_OL0_WRAP=1`
for valid P(k). This is a HACC-side setting, relevant only when generating
comparison baselines.

---

## Frontier (OLCF)

> **Stub — to be added.** HACCabana has not been built or run on Frontier.
> Everything below is expected behavior derived from the architecture, not
> verified fact. Do not treat it as a recipe.

HPE Cray-EX, AMD EPYC hosts, AMD Instinct MI250X accelerators (2 GCDs per
package, conventionally one rank per GCD), Slingshot-11.

Expected configuration:

- **Kokkos backend:** HIP. Requires a Kokkos install built with
  `Kokkos_ENABLE_HIP=ON` and the correct `Kokkos_ARCH_VEGA90A`.
- **FFT backend:** heFFTe should select rocFFT automatically from the HIP
  memory space, by the same mechanism that selects oneMKL on SYCL.
- **GPU-aware MPI:** `MPICH_GPU_SUPPORT_ENABLED=1` is the Cray MPICH
  equivalent of Aurora's `MPIR_CVAR_ENABLE_GPU=1`. The
  `Cabana::Distributor` requirement is identical.
- **Affinity:** one rank per GCD via `ROCR_VISIBLE_DEVICES`, typically set
  by an ALCF/OLCF-provided wrapper analogous to `gpu_tile_compact.sh`.
- **`PMKOKKOS_FP_PRECISE` will not work as written.** It emits
  `-fp-model=precise`, which is Intel syntax. The `hipcc`/Clang equivalent
  is `-ffp-contract=off` alone. This option needs a compiler-conditional
  before it is usable here.

To be determined: parallel HDF5 module and whether `LD_LIBRARY_PATH` needs
the same explicit treatment; whether the `LayoutLeft`/`LayoutRight` staging
issue recurs (HIP's default layout should be checked); rocFFT scaling
behavior versus oneMKL at the reference problem size.

---

## Perlmutter (NERSC)

> **Stub — to be added.** Not built or run. Expected behavior only.

HPE Cray-EX, AMD EPYC hosts, NVIDIA A100 accelerators, Slingshot-11.

Expected configuration:

- **Kokkos backend:** CUDA, with `Kokkos_ENABLE_CUDA=ON` and
  `Kokkos_ARCH_AMPERE80`.
- **FFT backend:** cuFFT, selected automatically from the CUDA memory
  space.
- **GPU-aware MPI:** a CUDA-aware MPI build; on Cray MPICH,
  `MPICH_GPU_SUPPORT_ENABLED=1`. Same `Cabana::Distributor` requirement.
- **Affinity:** one rank per A100 via `CUDA_VISIBLE_DEVICES`, usually
  handled by the NERSC GPU-binding wrapper.
- **`PMKOKKOS_FP_PRECISE` will not work as written** — Intel flag syntax.
  The NVCC equivalent is `--fmad=false`, which is not what the option
  emits.

To be determined: whether Kokkos CUDA's default View layout introduces the
same HDF5 staging concern; cuFFT behavior at the reference size; whether
`Kokkos_ENABLE_CUDA_LAMBDA` needs to be explicit in the Kokkos build.

---

## CPU + FFTW

> **Stub — to be added.** CPU comparison builds were made during
> development, but no full validated CPU recipe was recorded.

Runs anywhere with a host-only Kokkos.

Expected configuration:

- **Kokkos backend:** OpenMP (or Serial). Build Kokkos with
  `Kokkos_ENABLE_OPENMP=ON` and no device backend, so
  `DefaultExecutionSpace` resolves to the host.
- **FFT backend:** heFFTe selects FFTW3 when the memory space is
  `HostSpace`. FFTW 3.3.10 was the validated version.
- **No GPU environment variables.** Leave `ZE_AFFINITY_MASK`,
  `ONEAPI_DEVICE_SELECTOR`, and `MPIR_CVAR_ENABLE_GPU` unset.
- **Threading:** `OMP_NUM_THREADS` is the only relevant knob. Watch total
  oversubscription — `ranks × OMP_NUM_THREADS` should not exceed the core
  count.
- Parallel HDF5 is still required; the collective-MPI-IO code path is
  unconditional.

Advantages worth noting: the host-only test subset runs here without a
device, and the `LayoutLeft` HDF5 staging pitfall does not arise because
the host default layout is already `LayoutRight`.

To be determined: a validated module/flag set on a named CPU machine;
whether `PMKOKKOS_FP_PRECISE` should become compiler-conditional as part of
this work; reference timings for comparison against the GPU path.
