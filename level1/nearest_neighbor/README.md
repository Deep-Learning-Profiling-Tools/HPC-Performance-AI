# Nearest Neighbor

k-nearest-neighbor search over hurricane track records.

## Source

Source suite: Rodinia (cuda/nn, data/nn)
Upstream repository: https://github.com/HPC-FAIR/rodinia_3.1
Upstream commit: 366b283456506ef2fe2c2b7dc5e83e941ffc5524

## Verified Environment

GCC/G++ 13.3.0 (conda, pinned) | C++20 | CMake 3.28.4 | Ninja 1.13.2 | Python 3.12.3
CUDA Toolkit 13.2 (nvcc 13.2.78, /usr/local/cuda) | NVIDIA B200 (sm_100), driver 595.58.03
HIP/ROCm: source + build config present where noted, unverified (no AMD GPU available)

Reproduce the toolchain from the repository root: `./setup_env.sh` then
`source hpcperf_env.sh` (all user-space versions are pinned in environment.yml).

## Backends

CUDA: working (configure + build + run + validation verified on B200)
HIP: pending (upstream provides no HIP implementation)

## Build

```bash
source hpcperf_env.sh
cmake -S level1/nearest_neighbor -B build/nearest_neighbor/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/nearest_neighbor/cuda
```

## Run

```bash
cd <run dir> && ./build/nearest_neighbor/cuda/nearest_neighbor_cuda build/nearest_neighbor/cuda/filelist_4 -r 5 -lat 30 -lng 90
```

## Validation

Correctness = the k reported distances (`records[i].distance` after the ROI, sorted ascending as verify.py compares them, captured as `topk_distances`, f32; printed with `%f`) within max |a - g| / (1 + |g|) <= 1e-4 - d0 of the frozen B200 golden reference `reference/<input_id>.json` (d0 = the golden's own deviation from the CPU distances; rule in `inputs.yaml` `golden:`). The added check `verify.py` (CPU recomputation of all distances from the db records; the sorted reported top-k must match within 1e-4 (1 + |e|)) validated that reference once and runs only on request (below). The file list is generated at configure time with absolute data paths.

**Fix (HPC-Performance-AI).** `loadData` read each .db path of the file list with `fscanf("%s")` into `char dbname[64]`; the configured absolute paths are longer (90 characters in this checkout), a stack overflow that the unmodified build survived by chance and the golden capture's insertion turned into a SIGBUS / SIGSEGV. The buffer is now 4096 bytes; nothing else changed (stdout byte-identical to the unmodified build). `char filename[100]` (`strncpy` of the file-list path, not NUL-terminated at >= 100 characters) is left as upstream has it.

```bash
ctest --test-dir build/nearest_neighbor/cuda --output-on-failure   # golden run of the registry's default input
python3 tools/golden/hpcperf_golden.py run level1/nearest_neighbor [<input_id>]   # the same for any registered input
```

**Correctness switch.** The binary has no built-in CPU check (nothing to skip, so it prints no `SKIP_VERIFY`); its CPU reference is `verify.py`, which a plain run, ctest, `tools/inputs` and `tools/timing` never run and nothing falls back to. `tools/golden/hpcperf_golden.py make` runs it once, offline (`HPCPERF_CPU_REFERENCE=1`, `HPCPERF_GOLDEN_DUMP`), to validate the golden reference; verify.py then also captures its CPU reference array (`topk_distances.cpu`, same order) so make can measure d0. `HPCPERF_GOLDEN_DUMP=<dir>` makes the binary capture its output. See [tools/golden/README.md](../../tools/golden/README.md).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `cuda/nn_cuda.cu`: distance kernel, copy back, host k-nearest selection. Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 236 (1 source files, cloc, cuda/ + common/)
HIP: - (pending)
