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

**Fixes (HPC-Performance-AI; host code only, stdout and captured output byte-identical for the registered input).** Paths are limited to `NN_PATH_CHARS` = 4095 characters (Linux `PATH_MAX` 4096 bytes including the NUL: the longest path `open(2)` accepts at all); a longer path is an error with a message and exit 1, never truncated or overflowed.

1. `.db` paths of the file list (`loadData`): upstream read each with an unbounded `fscanf("%s\n")` into `char dbname[64]`; the configured absolute paths are longer (90 characters in this checkout), a stack overflow that the unmodified build survived by chance and the golden capture's insertion turned into a SIGBUS / SIGSEGV. Now `char dbname[4096]` is read with the field width `%4095s`; a path that fills the field and continues fails ("a .db path in the filelist ... is longer than 4095 characters"); the whitespace skipping of upstream's `\n` directive is kept (blank lines, several paths per line, no final newline, CRLF: same behaviour).
2. The file-list path itself (`parseCommandline`): upstream `strncpy(filename, argv[1], 100)` into `char filename[100]` leaves it without a NUL terminator at >= 100 characters (an agent workspace's build directory, e.g. a 229-character path, crashed with SIGSEGV). Now `char filename[4096]`, filled only after a length check ("the filelist path is longer than 4095 characters").

Not changed (upstream behaviour): an empty file list prints "error reading filelist" and exits 0; a file list that cannot be opened is not checked (`fopen` NULL, then a crash in `feof`).

```bash
ctest --test-dir build/nearest_neighbor/cuda --output-on-failure   # golden run of the registry's default input
python3 tools/golden/hpcperf_golden.py run level1/nearest_neighbor [<input_id>]   # the same for any registered input
```

**Correctness switch.** The binary has no built-in CPU check (nothing to skip, so it prints no `SKIP_VERIFY`); its CPU reference is `verify.py`, which a plain run, ctest, `tools/inputs` and `tools/timing` never run and nothing falls back to. `tools/golden/hpcperf_golden.py make` runs it once, offline (`HPCPERF_CPU_REFERENCE=1`, `HPCPERF_GOLDEN_DUMP`), to validate the golden reference; verify.py then also captures its CPU reference array (`topk_distances.cpu`, same order) so make can measure d0. `HPCPERF_GOLDEN_DUMP=<dir>` makes the binary capture its output. See [tools/golden/README.md](../../tools/golden/README.md).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `cuda/nn_cuda.cu`: distance kernel, copy back, host k-nearest selection. Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 236 (1 source files, cloc, cuda/ + common/)
HIP: - (pending)
