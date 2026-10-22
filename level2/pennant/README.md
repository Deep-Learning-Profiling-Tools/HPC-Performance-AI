# PENNANT (CUDA branch) -- unstructured-mesh Lagrangian hydrodynamics mini-app

PENNANT is LANL's mini-app for the mesh data structures and a few physics
algorithms of the rad-hydro code FLAG (staggered-grid Lagrangian hydro on 2-D
unstructured polygonal meshes: points, zones, sides, corners; predictor-corrector
cycle with TTS and QCS artificial viscosity, gamma-law gas). The `cuda` branch is
the single-GPU port of version 0.6: the whole cycle runs in five CUDA kernels
(`gpuMain1..5`, `src/HydroGPU.cu`) on flat side/point/zone arrays; the host does
the mesh generation, the per-cycle dt bookkeeping and the final output.

## Source

- Upstream: https://github.com/lanl/PENNANT, branch `cuda`, commit
  `6c4e0d15a802349c454eeb772d9817630c47d175` ("Add clarification to CUDA README
  file", 2018-01-16; code base v0.6, February 2014). The same checkout is what
  LLNL distributes as the CORAL-2 `pennant-singlenode-cude.tgz`.
- Vendored byte-identically (`UPSTREAM_SHA256SUMS`: the upstream sha256 of every
  vendored file): `Makefile`, `README`, `LICENSE`, `src/` (all 31 files),
  `test/*/*.pnt` + `test/*/*.xy.std` (the nine decks and the five gold-standard
  profiles), `tools/gmvrect.py`, `doc/pennantdoc.tex`. Not vendored: the legacy
  `.gmv` meshes (26 MB; v0.6 generates every mesh internally and no shipped
  deck references them), `doc/*.pdf` and `doc/*.png`.
- `decks/sedovflatx4.pnt` is the CORAL-2 scaling deck `test/sedovflatx4/sedovflatx4.pnt`
  of upstream's `master` branch (commit `d7b770ecd5cc75bbb768554fcd9b2977e65b6e06`;
  the deck was added 2018-02-02 in `12c7f7ed`), copied verbatim -- it uses only
  keywords the cuda branch reads. `decks/nohpoly-6000.pnt` is the cuda branch's
  `test/nohpoly/nohpoly.pnt` with one line `cstop 6000` prepended (see "Inputs").
- License: `LICENSE` (Los Alamos National Security, LLC, 2012; BSD-style with the
  "modified software should be clearly marked" clause -- the two modified files
  below carry an `HPC-Performance-AI` comment at every changed line).

## Changes from upstream

Two files differ from the upstream sha256 list; both changes are marked in the
source. Nothing in the algorithm, precision, chunking or physics parameters was
touched; no Makefile line was edited (every compiler setting is overridden on the
`make` command line by `build.sh`).

- **`src/HydroGPU.cu` -- CUDA 13 compatibility (class D, 6 lines).** The six
  `cudaThreadSynchronize();` calls became `cudaDeviceSynchronize();`.
  `cudaThreadSynchronize` was deprecated in CUDA 4.0 and is no longer in the
  CUDA 13 runtime (the build fails with `identifier "cudaThreadSynchronize" is
  undefined`); its documented replacement has the same semantics (block the host
  until all preceding device work completed) and, as upstream did, the return
  value is ignored. Nothing else in the GPU code needed a change for nvcc 13.2 /
  sm_100.
- **`src/Driver.cc` -- tools/timing ROI markers (measurement only).** `#include
  "hpcperf_roi.h"`, `HPCPERF_ROI_BEGIN_SYNC()` just before PENNANT's own starting
  timestamp of the cycle loop and `HPCPERF_ROI_END_SYNC()` right after the loop
  (pure insertions). The ROI is therefore exactly the region of PENNANT's `hydro
  cycle run time`: every cycle (`calcGlobalDt`, the five kernels with their
  synchronisations, the dt read-back, the periodic `End cycle` progress line);
  mesh generation and `hydroInit` (upload) are before it, `getData` (read-back),
  the end message and the `.xy` / Ensight Gold output after it. The markers are a
  no-op unless `HPCPERF_ROI_LOG` is set or a profiler is attached; `build.sh`
  puts `tools/timing/roi` on `CPATH` (the upstream `.cu` rule clears `CPATH` for
  nvcc on purpose; the header is only needed by the host file).
- **Build (no file change).** `build.sh` mirrors `Makefile` + `src/` into
  `build/level2/pennant/cuda/` (the upstream Makefile builds in place) and runs
  `make CXX=<conda g++> CXXFLAGS_OPT=-O3 CUDAC="nvcc -ccbin <g++>"
  CUDACFLAGS="-arch=sm_<cc> --ptxas-options=-v -O3 -I." CUDA_INSTALL_PATH=<cuda>
  LDFLAGS="-L<cuda>/lib64 -lcudart"`: the Makefile's g++ flag set instead of the
  icpc one (`-O3 -fast -fno-alias`, which g++ does not accept), the detected GPU
  architecture instead of the hard-coded `-arch=sm_21` (unsupported since CUDA
  9), and the real toolkit path. Upstream's `-O3`, `--ptxas-options=-v` and `-I.`
  are kept.

## Dependencies

CUDA toolkit (nvcc; the program links only `libcudart`) and a C++ host compiler
-- both from `hpcperf_env.sh` (system CUDA 13.2, conda GCC 13.3.0). No MPI (the
cuda branch has none: the `Parallel` layer is the single-process stub), no
OpenMP, no external libraries. `check_xy.py` needs only the Python standard
library.

## Verified Environment

GCC/G++ 13.3.0 (conda, pinned) | CUDA Toolkit 13.2 (nvcc 13.2.78, /usr/local/cuda) |
NVIDIA B200 (sm_100, 183 GB) | OS: RHEL 10 (Linux 6.12) | dgx003, 2026-10-20.
Reproduce the toolchain from the repository root: `./setup_env.sh` then
`source hpcperf_env.sh`.

## Backends

CUDA only. Upstream keeps a separate `hip` branch (a hipified copy of the same
port); it is not vendored here, `build.sh HIP` / `run.sh HIP` exit 1 with a
message. No ROCm is available on the validation node anyway.

## Build

```bash
level2/pennant/build.sh CUDA          # -> build/level2/pennant/cuda/build/pennant
```

`HPCPERF_CUDA_ARCH=<cc>` overrides the detected compute capability, `MAKE_JOBS`
the parallelism (default 8). A build takes about 40 s (`HydroGPU.cu` dominates).

## Run

```bash
level2/pennant/run.sh CUDA                              # default: test/sedovbig/sedovbig.pnt
level2/pennant/run.sh CUDA test/nohsquare/nohsquare.pnt # any deck
HPCPERF_PENNANT_INPUT=sedovflatx4-1920x2160 level2/pennant/run.sh CUDA   # a registered input
```

PENNANT takes exactly one argument (the `.pnt` deck) and writes its outputs
next to the deck, named after it (`<name>.xy`: the per-zone density, specific
internal energy and pressure after the last cycle, 8 significant digits;
`<name>.case/.geo/.zr/.ze/.zp`: the same fields in Ensight Gold format). `run.sh`
therefore copies the deck byte-identically into
`build/level2/pennant/cuda/run/<name>/` and runs the binary there on the copy's
absolute path -- the vendored tree never receives output; one directory per
deck name, overwritten by a rerun. The program runs one process on one GPU
(`CUDA_VISIBLE_DEVICES` selects it); it prints `Running Hydro on device...`
when it takes the GPU path, the mesh statistics (`Zones:` ...), an `End cycle`
line every `dtreport` cycles, and at the end `cycle = ... cstop = ...`,
`time = ... tstop = ...` and `hydro cycle run time= <s>`.

`HPCPERF_PENNANT_INPUT=<id>` (the inputs registry, `inputs.yaml`) supplies the
deck; it is refused together with an explicit deck argument.

### Inputs (`inputs.yaml`, selector `HPCPERF_PENNANT_INPUT`)

| id | deck | problem / mesh | zones | cycles | timed loop (B200, one run 2026-10-20) | correctness reference |
|---|---|---|---|---|---|---|
| `sedovbig-540sq` (default) | `test/sedovbig/sedovbig.pnt` (cuda branch) | Sedov blast wave, 540 x 540 rect | 291,600 | 3882 (tstop 1.0) | 1.82 s | frozen CPU v0.6 run (`reference/sedovbig-540sq.json`) |
| `sedovflatx4-1920x2160` | `decks/sedovflatx4.pnt` (master, CORAL-2) | Sedov "flat" blast wave x4, 1920 x 2160 rect | 4,147,200 | 12511 (tstop 1.0; = upstream's sample output) | 83.5 s | frozen CPU v0.6 run (`reference/sedovflatx4-1920x2160.json`) |
| `leblancbig-160x1440` | `test/leblancbig/leblancbig.pnt` (cuda branch) | Leblanc shock tube (density ratio 1000), 160 x 1440 rect | 230,400 | 3775 (tstop 6.0) | 1.51 s | frozen CPU v0.6 run (`reference/leblancbig-160x1440.json`) |
| `nohsquare-360sq` | `test/nohsquare/nohsquare.pnt` (cuda branch) | Noh implosion, 360 x 360 square | 129,600 | 7677 (tstop 0.6) | 2.13 s | frozen CPU v0.6 run (`reference/nohsquare-360sq.json`) |
| `nohpoly-hex251-6000cycles` | `decks/nohpoly-6000.pnt` (derived: upstream nohpoly + `cstop 6000`) | Noh implosion, hex 251 polygonal mesh | 63,001 | 6000 of 9876 (t = 0.46 of 0.6) | about 1.4 s | frozen CPU v0.6 run (`reference/nohpoly-hex251-6000cycles.json`) |

Three physical problems on two mesh types; the timed loop is the whole cycle
loop. Not registered (recorded in `inputs.yaml`, `upstream_inputs_not_added`):
the five 10-cycle gold decks (validate.sh; 1-3 ms), master's `sedovflat` (run
once: 3136 cycles, the count of upstream's sample output; 1.38 s; the same
problem as `sedovbig` at a similar size), `sedovflatx16/x40/x120` (66 M .. 6 G
zones), the full `nohpoly` run and the `leblancx4` series (both below).

## Validation

```bash
level2/pennant/validate.sh CUDA
```

Runs the five upstream decks that ship a gold-standard profile -- `sedovsmall`
(81 zones), `nohsmall` (40), `sedov` (2025), `noh` (3000), `leblanc` (900), 10
cycles each -- through `run.sh` and compares every zone of every column of the
run's `<name>.xy` with `test/<name>/<name>.xy.std` (`check_xy.py`). Rule: zones
whose reference magnitude is at least 1e-8 of the column's maximum within 1e-6
relative (the profile's 8 printed digits); zones below that floor (roundoff
residues of the undisturbed near-vacuum region, 1e-10 .. 1e-50, compiler-
dependent) within 1e-8 of the column's maximum absolute. Each run must exit 0,
print `Run complete`, `hydro cycle run time` and the GPU banner. Result on
dgx003 (2026-10-20): **PASS** for all five; `noh`, `nohsmall` and `leblanc`
(density, energy) are identical to the gold files to all 8 digits in every
zone, and so is every populated zone of `sedovsmall` / `sedov` (the differences
are confined to the near-vacuum zones at 1e-18 of the column scale and below).
The CPU v0.6 build (below) reproduces the gold files the same way.

### Correctness of the registered inputs (frozen CPU references)

The large decks have no upstream gold file. Their reference is a run of the
same deck with upstream's own CPU implementation at the same code version --
`lanl/PENNANT` master `617414f` ("Update to version 0.6", 2014-02-28), the
version the cuda branch is a port of -- built with the conda `mpicxx -O3
-fopenmp` (1 MPI rank, 16-32 OpenMP threads, dgx003, 2026-10-20) and frozen
once with `check_xy.py --make-reference` into `reference/<id>.json`: per column
the zone count, scale (max |value|), sum, min, max and a fixed stride sample
of 2048 zones (the full profiles are 5-320 MB and are not kept in git).
Provenance (deck / profile / log sha256, binary, thread count, date) is inside
each file. The registry's `check` block runs `check_xy.py` on the run's `.xy`
against that file; the stdout quantities (final cycle count, final time, the
completion and GPU markers, no `negative side volumes`) are the registry's
baseline comparison.

Rule for the frozen references: populated zones (>= 1e-8 of the column maximum)
within **1e-5 relative**, floor zones within 1e-8 of the maximum, column sum /
min / max alike. Basis: the accumulated roundoff between two correct
realisations of the arithmetic (CPU v0.6 vs this unmodified-algorithm CUDA
build) was measured on every registered deck before any optimized candidate
existed (2026-10-20):

| input | cycles | max relative deviation in populated zones (zr / ze / zp) |
|---|---|---|
| sedovbig-540sq | 3882 | 8.9e-9 / 1.0e-8 / 3.3e-7 |
| sedovflatx4-1920x2160 | 12511 | 6.3e-9 / 7.8e-9 / 1.0e-7 (4.1 M zones; the CPU run took 1268 s on 32 threads) |
| leblancbig-160x1440 | 3775 | 2.8e-9 / 7.4e-9 / 3.7e-9 |
| nohsquare-360sq | 7677 | 1.9e-8 / 3.0e-7 / 3.0e-7 |
| nohpoly-hex251-6000cycles | 6000 | 1.2e-8 / 4.9e-7 / 4.8e-7 |

1e-5 is 30 x the largest of these and 100 x below the 1e-3 .. 1e-2 that a
changed method, a dropped term or a wrong mesh produce (the full `nohpoly` run,
below, shows what a genuine divergence looks like). Both CPU and GPU reach the
same cycle count on every deck (the dt sequence agrees to the last printed
digit), and the CORAL-2 deck's 12511 cycles equal the count in upstream's own
`test/sample_outputs/vulcan/sedovflatx4.out` (v0.9).

**`nohpoly` (hexagonal mesh) beyond cycle 6000.** With the upstream deck as
shipped (9876 cycles to tstop 0.6) the CPU v0.6 and CUDA profiles differ by up
to 3.6e-3 (zr) and 1.8e-2 (ze, zp) in almost half of the zones, while agreeing
to 1e-8 at cycles 100 / 1000 / 3000 and 5e-7 at 6000, then 1e-3 at 8000 and
4e-2 at 9000: the late phase of the implosion on the polygonal mesh amplifies
roundoff (the two runs are two correct realisations, not a GPU defect -- on a
21 x 21 hex mesh both agree bit for bit for 10 cycles, and on the square mesh
of `nohsquare` to 3e-7 after 7677 cycles). No per-zone criterion exists for the
full run, so the registered input stops at cycle 6000, where the agreement is
5e-7 and the 1e-5 rule applies with its margin.

**Leblanc meshes that fail on the GPU (upstream, not fixed here).** Master's
`leblancx4` (640 x 5760) and a derived 320 x 2880 Leblanc mesh abort at the
first cycle: `an illegal memory access was encountered` in the GPU kernels,
then `Error: 21916 negative side volumes`. 320 x 1440, 160 x 2880 and 240 x
2160 (518,400 zones) run; `sedovflatx4` with 4.1 M zones runs. Cause not
established (not pursued: the task was integration, not a rewrite of the
port); `leblancx16/x48/x64` were not attempted.

### Correctness vs timing vs workload verification

Three separate results per input (`tools/inputs`, `tools/timing`): timing
SUCCESS = the ROI record of a clean run; workload verification PASS =
`verify_registry_runs.py` confirmed the run read the registered deck
(`cases/registry_evidence.yaml`: the deck's byte-identical copy in the run
directory with the registered sha256, the `Zones:` line of the mesh it
generated); scientific correctness PASS = the registry verdict (the `check`
block's `check_xy.py` run plus the stdout quantities).

## Warnings

- `nvcc --ptxas-options=-v` (upstream's flag) prints register/stack usage for
  every kernel during the build; harmless.
- The Sedov decks run in a near-vacuum: PENNANT's `dt limiter: Hydro dV/V limit`
  messages and energies of 1e-20 .. 1e-50 in undisturbed zones are the normal
  behaviour of the gamma-law gas there.
- `sedovflatx4` writes a 320 MB `.xy` and 400 MB of Ensight files after the
  timed loop (`build/level2/pennant/cuda/run/sedovflatx4/`, overwritten per run).

## LOC

CUDA port (`src/`, cloc code lines, excluding `Makefile`, decks, docs): 3256 in
31 files -- C++ 1588 (14 files), CUDA 993 (`HydroGPU.cu`), headers 675 (16).

## Status

Working (CUDA): build + run + validate passed on dgx003 (B200, CUDA 13.2),
2026-10-20. Five registered inputs with frozen CPU v0.6 references. HIP: not
vendored (upstream `hip` branch), untested (no ROCm).

## RESULTS

Campaign of 2026-10-21 (results directory `l2-ext-2026-10-20`, GPU 0000:9D:00.0, Slurm job 2732030, commit
01caa24): final Level 2 protocol -- 1 discarded whole-process warm-up, 3 clean runs (headline = median,
spread = (max - min) / median, stable when <= 10 %), 1 separate nsys-profiled run; direct exec, unbound; one
input at a time on a quiet node. Timing SUCCESS, workload verification (`verify_registry_runs.py`) and
scientific correctness (`hpcperf_inputs.py check` + `verdict`) are three separate results:

| input | timed loop (median of 3) | min / max | spread | stable | device busy (profiled run) | run verification | correctness |
|---|---|---|---|---|---|---|---|
| sedovbig-540sq | 1.854 s | 1.849 / 1.857 s | 0.4 % | yes | 80 % | PASS | PASS |
| sedovflatx4-1920x2160 | 63.82 s | 63.78 / 63.91 s | 0.2 % | yes | 99 % | PASS | PASS |
| leblancbig-160x1440 | 1.529 s | 1.470 / 1.529 s | 3.9 % | yes | 77 % | PASS | PASS |
| nohsquare-360sq | 2.004 s | 2.002 / 2.119 s | 5.9 % | yes | 72 % | PASS | PASS |
| nohpoly-hex251-6000cycles | 1.466 s | 1.463 / 1.467 s | 0.3 % | yes | 62 % | PASS | PASS |

The three small decks are host-latency bound (0.45-0.5 ms per cycle: five kernel launches, five device
synchronisations and three symbol copies per cycle at 60-80 % device busy); `sedovflatx4` (4.1 M zones) is
device bound (99 % busy, 5.1 ms per cycle). Two earlier measurements of the first two inputs that overlapped
with a 32-thread CPU job of the operator in the same cpuset doubled their cycle time (sedovbig 3.27 s) and
were set aside, as was a complete first pass whose registry listed the frozen references as program files
(workload verification INSUFFICIENT); the results directory keeps both as evidence
(`raw-perturbed/`, `raw-superseded/`). Every correctness check: the run's `.xy` against
`reference/<id>.json` with the 1e-5 rule -- the largest observed deviation of these runs is 5.7e-8 relative
(sedovflatx4 zp), i.e. the same accumulated roundoff as the single runs the rule was set from.
