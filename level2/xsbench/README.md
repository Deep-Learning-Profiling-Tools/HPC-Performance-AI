# XSBench

Monte Carlo neutron transport mini-app (ANL CESAR) that isolates the
continuous-energy macroscopic cross-section lookup kernel of codes such as
OpenMC. For each of N independent lookups it samples a random energy and
material, binary-searches a unionized energy grid, and accumulates the
microscopic cross sections of every nuclide in the material (up to 321 for
fuel in the Hoogenboom-Martin "large" model). Motif: data-dependent, memory-
latency-bound gather over a ~5.6 GB table with heavy thread divergence. Figure
of merit: lookups per second. Upstream's own verification checksum is checked
on every run.

## Source

Upstream repository: https://github.com/ANL-CESAR/XSBench
Upstream commit: ba08e5221af6106252b866e50ea123c69d31a4e2 (cloned 2026-09-01)
License: MIT (Argonne National Laboratory, 2012-2021) -- copied as `LICENSE`

Copied from upstream: `cuda/` (7 sources + Makefile), `hip/` (7 sources +
Makefile), `LICENSE`. Not copied: `openmp-threading/`, `openmp-offload/`,
`opencl/`, `sycl/`, `docs/`, `CHANGES.txt`, CI files, upstream README.
XSBench has no input decks; the problem is generated at run time from the
command line.

## Changes from upstream

- `cuda/Makefile`: `-std=c++14` -> `-std=c++17` (one token). CUDA 13.2 ships
  CCCL 3.x, whose Thrust/CUB headers (included by `XSbench_header.cuh`) refuse
  to compile below C++17 (`#error libcu++ requires at least C++ 17`,
  `#error CUB requires at least C++17`). No source change.
- Nothing else. The Makefile's `SM_VERSION` (default 80) and `CC` (`nvcc`)
  are plain assignments, so `build.sh` overrides them on the make command line
  (`make SM_VERSION=<cc> CC="nvcc -ccbin $CXX"`) without editing the file.
  Upstream's `-Xcompiler -Wall -Xcompiler -O3 -O3` flags are used unchanged.
- `hip/` is byte-identical to upstream (hipcc + `-std=c++14`; no Thrust).

## Dependencies

CUDA toolkit (nvcc, Thrust/CUB from CCCL) and a C++17 host compiler -- both
from `hpcperf_env.sh` (system CUDA 13.2, conda GCC 13.3.0). No MPI (upstream
`MPI` option is not enabled), no external libraries. HIP variant: ROCm hipcc
(not available on this machine).

## Verified Environment

GCC/G++ 13.3.0 (conda, pinned) | C++17 | CMake 3.28.4 | Ninja 1.13.2 | Python 3.12.3
CUDA Toolkit 13.2 (nvcc 13.2.78, /usr/local/cuda) | NVIDIA B200 (sm_100), driver 595.58.03
HIP/ROCm: source + build config present where noted, unverified (no AMD GPU available)
OS: RHEL 10 (Linux 6.12)

Reproduce the toolchain from the repository root: `./setup_env.sh` then
`source hpcperf_env.sh` (all user-space versions are pinned in environment.yml).

## Backends

CUDA: working (build + run + validation verified on B200)
HIP: extracted, untested (no AMD GPU / ROCm on the development machine)

## Build

The upstream Makefiles build in place, so `build.sh` mirrors the backend
directory into `build/level2/xsbench/<cuda|hip>` (rsync) and runs `make -j4`
there; `level2/xsbench/` stays clean. The CUDA arch is detected from
`nvidia-smi` (override: `HPCPERF_CUDA_ARCH=100` or `sm_100`).

```bash
source hpcperf_env.sh          # optional: the scripts source it themselves
level2/xsbench/build.sh        # CUDA (default)
level2/xsbench/build.sh HIP    # needs hipcc; fails with a clear message here
```

Equivalent raw commands (CUDA):

```bash
rsync -a level2/xsbench/cuda/ build/level2/xsbench/cuda/
make -C build/level2/xsbench/cuda -j4 SM_VERSION=100 CC="nvcc -ccbin $CXX"
```

HIP (build configuration present, unverified without ROCm):

```bash
rsync -a level2/xsbench/hip/ build/level2/xsbench/hip/
make -C build/level2/xsbench/hip -j4            # CC=hipcc from the Makefile
```

## Run

Standard problem = upstream's GPU default. The CUDA/HIP sources implement only
the event-based method (`-m history`, the CLI default, exits with a message),
so the run is the H-M "large" model with the event-based default lookup count:
355 nuclides x 11,303 gridpoints (4,012,565 unionized gridpoints, 5.6 GB),
17,000,000 lookups, unionized grid, baseline kernel (`-k 0`).

```bash
level2/xsbench/run.sh                       # == ./XSBench -m event -s large
level2/xsbench/run.sh CUDA -k 6             # extra args appended (sorted-lookup kernel)
level2/xsbench/run.sh CUDA -l 100000000     # more lookups
```

Observed on the B200: ~4.5 s wall clock, of which ~4 s is single-threaded host
grid initialization + 5.8 GB host-to-device copy (upstream marks it "DO NOT
PROFILE"); the lookup kernel itself reports 0.05-0.11 s (150-330 M lookups/s
by the host timer; `-k 6` ~1.15 G lookups/s). Use nsys/ncu for kernel timing,
as upstream recommends.

## Validation

Upstream's built-in mechanism: every lookup writes the index (1..5) of the
largest of its 5 macroscopic cross-section values into a device array; the
array is summed (`thrust::reduce`), reduced `% 999983`, and compared in
`io.cu:print_results()` against hard-coded reference checksums for the default
problem sizes. The program prints `Verification checksum: <n> (Valid)` and
exits 0 only on a match. Event-based references: large = **952131**,
small = **945990**.

`validate.sh` runs `-m event -s large` and `-m event -s small`, requires exit
code 0 and the expected checksum with `(Valid)` for both, and prints PASS/FAIL.

```bash
level2/xsbench/validate.sh          # CUDA
```

Observed here:

```
Verification checksum: 952131 (Valid)     (-s large, 17,000,000 lookups)
Verification checksum: 945990 (Valid)     (-s small, 17,000,000 lookups)
PASS: xsbench CUDA (event-based large=952131, small=945990 verification checksums match upstream reference)
```

`-k 6` (optimized sorted kernel) also reproduces 952131 / 945990.

## Warnings

None (nvcc 13.2 with `-Xcompiler -Wall`, all 6 translation units + link).

## LOC

CUDA: 1514 (cloc, 7 files: `cuda/*.cu cuda/*.cuh` -- Main, io, Simulation,
GridInit, XSutils, Materials, XSbench_header; Makefile excluded)
HIP: 1061 (cloc, 7 files: `hip/*.cpp hip/*.h`; Makefile excluded). The HIP
`Simulation.cpp` contains only the baseline kernel; the six optimized kernel
variants of the CUDA version are absent (commented out in `hip/Main.cpp`).

## Status

Working (CUDA): build + run + validate passed on this machine.
HIP: untested (no hipcc); `build.sh HIP` exits 1 with an explanatory message.
