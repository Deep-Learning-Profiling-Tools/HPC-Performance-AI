# miniBUDE

Mini-app of the core kernel of the Bristol University Docking Engine (BUDE): a virtual screening
run that evaluates the binding energy of a ligand (small molecule) against a protein for a
generation of rigid-body poses. For each of the 65536 poses the kernel transforms every ligand atom
and accumulates steric, electrostatic and desolvation terms over all protein atoms (an
O(poses x ligand atoms x protein atoms) FP32 compute-bound N-body-style motif with heavy use of
branch-free selects). One kernel instantiation exists per compile-time "poses per work-item"
(PPWI) value; the driver can auto-tune over PPWI and work-group size and reports the best.

## Provenance

Upstream repository: https://github.com/UoB-HPC/miniBUDE (main branch, v2.0 driver)
Upstream commit: 570f66c1fd0c29b7f3ab6fa96fba87f4561aa44c
Cloned: 2026-09-01 (into `_upstream/level2/miniBUDE`)
License: Apache License 2.0 (copied as `LICENSE`, from upstream `LICENSE.txt`)

Copied from upstream (paths relative to the upstream root):

- `CMakeLists.txt`
- `cmake/register_models.cmake`, `cmake/git_watcher.cmake`, `cmake/extract_compile_command.cmake`
- `src/main.cpp`, `src/bude.h`, `src/meta_build.h.in`, `src/meta_vcs.h.in` (shared driver)
- `src/cuda/fasten.hpp`, `src/cuda/model.cmake` (CUDA model)
- `src/hip/fasten.hpp`, `src/hip/model.cmake` (HIP model)
- `data/bm1/*`, `data/bm2/*`, `data/README.md` (input decks incl. `ref_energies.out`; 2.3 MB + 2.4 MB)

Left out: the other programming models (`src/{acc,kokkos,ocl,omp,raja,serial,std-*,sycl,tbb,thrust,julia}`),
`cmake/Modules`, `cmake/toolchains`, `makedeck/` (deck generator + 100 MB of raw inputs), `heatmap.py`,
CI scripts/workflows, `data/bm2_long` (37 MB; bm2 with 1048576 poses -- fetch it from the upstream
repository's `data/bm2_long` directory if needed and pass it via `MINIBUDE_DECK=/path/to/bm2_long`).
The upstream `CMakeLists.txt` still lists all models in `register_model(...)`, but only the selected
model's directory is ever read, so `-DMODEL=cuda|hip` works with this reduced tree.

## Changes from upstream

- `src/cuda/model.cmake` (macro `setup`): `cmake_policy(SET CMP0104 OLD)` replaced by
  `cmake_policy(SET CMP0104 NEW)` plus `set(CMAKE_CUDA_ARCHITECTURES OFF CACHE STRING ...)`.
  Reason: with CMake 3.28 the OLD policy emits a "CMake Deprecation Warning" at configure time. The
  architecture is passed by upstream as `-arch=${CUDA_ARCH}` in `CMAKE_CUDA_FLAGS`; `OFF` tells CMake
  not to add its own architecture flags, which also stops the `CUDAARCHS=native` environment variable
  of `hpcperf_env.sh` from injecting a second `-arch=native` (nvcc then warned "incompatible
  redefinition for option 'gpu-architecture'"). The generated nvcc command line is unchanged apart
  from the removed duplicate flag (`-arch=sm_100` only).
- `src/meta_vcs.h.in`: `#define MINIBUDE_VCS_COMMIT_BODY @GIT_COMMIT_BODY@` -> `"@GIT_COMMIT_BODY@"`
  (added quotes). Reason: upstream's `git_watcher.cmake` escapes the commit body for use *inside* a
  string literal, but the template left it unquoted, so any quote/apostrophe in the HEAD commit body
  of the enclosing git repository produced a `missing terminating ' character` preprocessor warning
  (observed in the upstream tree, whose HEAD body contains quotes). The macro is not used by
  `main.cpp`; the change only makes the generated header well-formed for every commit message.
  Note that in this repository the `vcs:` block printed by the binary describes the HEAD of
  HPC-Performance-AI (the git repository that now contains the sources), not the miniBUDE commit.

Everything else (kernel, driver, input decks, CMake logic) is byte-identical to upstream.
Added wrapper/documentation files: `build.sh`, `run.sh`, `validate.sh`, `README.md`.

## Dependencies

- CUDA: nvcc (CUDA 13.2 from `/usr/local/cuda`), conda GCC 13.3.0 as host compiler (`$CXX`), CMake >= 3.14
  (3.28.4), Ninja, `git` (used by upstream's `git_watcher.cmake` build step to embed VCS information;
  the build does not fail without a repository).
- HIP: `hipcc` (ROCm) -- not available on the development machine.
- No MPI, no external libraries.

## Build / Run / Validate

```bash
cd /projects/kzhou6/bcui2/research/HPC-Performance-AI && source hpcperf_env.sh
level2/minibude/build.sh            # CUDA (default); arch detected via nvidia-smi, override HPCPERF_CUDA_ARCH=100
level2/minibude/run.sh              # bm1 deck, -i 16 -w 64 -p all (auto-tune over PPWI), prints upstream YAML output
level2/minibude/validate.sh         # bm1 all PPWI + bm2 PPWI 2, checks `valid: true`, prints PASS/FAIL
level2/minibude/build.sh HIP        # HIP (needs hipcc; HPCPERF_HIP_ARCH=gfx90a adds --offload-arch)
```

Build tree: `build/level2/minibude/cuda` (binary `cuda-bude`) and `build/level2/minibude/hip` (`hip-bude`).

Equivalent raw commands (CUDA):

```bash
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')   # 100 on B200
cmake -S level2/minibude -B build/level2/minibude/cuda -DMODEL=cuda -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_CUDA_COMPILER=$(which nvcc) -DCUDA_ARCH=sm_${ARCH}
cmake --build build/level2/minibude/cuda -j4
build/level2/minibude/cuda/cuda-bude --deck level2/minibude/data/bm1 -i 16 -w 64 -p all
```

HIP (build configuration present, unverified without ROCm):

```bash
cmake -S level2/minibude -B build/level2/minibude/hip -DMODEL=hip -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=hipcc [-DCXX_EXTRA_FLAGS=--offload-arch=gfx90a]
cmake --build build/level2/minibude/hip -j4
```

`run.sh` options: `./run.sh [CUDA|HIP] [extra bude args]`; extra arguments are appended and override
the defaults (e.g. `./run.sh CUDA -i 8 -p 2 --csv`). Environment: `MINIBUDE_DECK=bm1|bm2|/path`,
`MINIBUDE_ITERS=16`, `MINIBUDE_WGSIZE=64`, `MINIBUDE_PPWI=all`. `bm2` (2672-atom ligand, ~300x the
interactions of bm1) is the larger option: 1.5-2.1 s/iteration at PPWI=2 on the B200, e.g.
`MINIBUDE_DECK=bm2 MINIBUDE_PPWI=2 MINIBUDE_ITERS=8 ./run.sh`.

Benchmark problem (`run.sh` defaults): upstream `bm1` deck (65536 poses, 26 ligand atoms, 938 protein
atoms, 34 force-field types), 16 timed iterations + 2 warm-up per configuration, work-group size 64,
all eight compiled PPWI values. Observed on the B200 (GPU shared with other jobs at the time, so
absolute times vary ~2x between runs): best configuration PPWI=2, wgsize=64 at 5.2-13 ms/iteration
(~8-11 TFLOP/s as reported by miniBUDE's own FLOP model); PPWI >= 16 launch too few blocks
(65536/(PPWI*64)) to fill the 148 SMs and are progressively slower (PPWI=128: 8 blocks, 140-410 ms).
The whole sweep took 5.8 s here on a lightly loaded GPU (~25 s when contended).

## Validation

Upstream's built-in check is used unchanged: after each configuration `main.cpp` (`validate()`)
compares every one of the 65536 computed pose energies with `data/<deck>/ref_energies.out`
(reference energies produced by the full BUDE code). An entry passes if the relative difference
`|ref - computed| / |ref|` is below `DIFF_TOLERANCE_PCT = 0.025 %`; entries where both `|ref| < 1`
and `|computed| < 1` are skipped. The result block prints `outcome: { valid: true|false, max_diff_%: X }`
and the process exits non-zero if any configuration is invalid.

`validate.sh` runs (1) `bm1`, wgsize 64, PPWI = 1,2,4,8,16,32,64,128 (all eight kernel
instantiations, `-i 2`) and (2) `bm2`, wgsize 64, PPWI = 2 (`-i 1`; skip with
`MINIBUDE_VALIDATE_BM2=0`), and requires exit code 0, no `valid: false`, and one `valid: true` per
configuration. It prints `miniBUDE CUDA validation: PASS|FAIL` and exits 0/1. Takes ~8-9 s.

Result observed here (CUDA, B200): PASS -- bm1: 8/8 configurations `valid: true, max_diff_%: 0.014`;
bm2: `valid: true, max_diff_%: 0.000`.

## Warnings

None (CUDA configure + build with the two fixes above: 0 CMake warnings, 0 nvcc/GCC warnings).
Before the fixes: 1 CMake deprecation warning (CMP0104 OLD) and 1 nvcc warning ("incompatible
redefinition for option 'gpu-architecture'", from `CUDAARCHS=native` + upstream `-arch=`).

## LOC

cloc v2.06, code lines only (READMEs, CMake files, data and scripts excluded):

CUDA: 885 (3 files: `src/main.cpp` 564, `src/bude.h` 80, `src/cuda/fasten.hpp` 241)
HIP: 851 (3 files: `src/main.cpp` 564, `src/bude.h` 80, `src/hip/fasten.hpp` 207)

## Verified Environment

GCC/G++ 13.3.0 (conda, pinned) | C++17 | CMake 3.28.4 | Ninja 1.13.2 | Python 3.12.3
CUDA Toolkit 13.2 (nvcc 13.2.78, /usr/local/cuda) | NVIDIA B200 (sm_100), driver 595.58.03
RHEL 10 (Linux 6.12), Slurm allocation, 1 GPU
HIP/ROCm: source + build config present where noted, unverified (no AMD GPU available)

Reproduce the toolchain from the repository root: `./setup_env.sh` then
`source hpcperf_env.sh` (all user-space versions are pinned in environment.yml).

## Status

CUDA: Working (configure + build + run + validate all passed on the B200 via `build.sh`, `run.sh`,
`validate.sh` from a fresh shell).
HIP: extracted, build configuration present (`build.sh HIP`, `-DMODEL=hip`), untested -- no
ROCm/hipcc on this machine (configure of the hip model with the host compiler succeeds; compilation
needs `hip/hip_runtime.h`).
