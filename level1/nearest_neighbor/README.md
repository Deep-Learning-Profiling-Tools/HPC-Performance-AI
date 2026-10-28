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

3. Input handling (2026-10-27): every input error now ends the run with a message on stderr and exit 1 instead of a crash, a silent exit 0 or a silently different query. A valid command line and valid files run exactly as before (stdout, captured output and exit code byte-identical, checked on the registered input and 11 other valid command lines). Upstream behaviour that is now an error:
   * the file list: cannot be opened (upstream: `fopen` unchecked, SIGSEGV in `feof`), cannot be read (a directory), or names no `.db` file (empty / blank; upstream: "error reading filelist" and exit 0); a `.db` file that cannot be opened (upstream: "error opening a db" on stdout, without the name) or read (a directory: upstream looped forever);
   * `.db` records: each line must be the 48 characters + newline of the hurricane_gen format (`fgets(49)` + one `fgetc` assume exactly that; upstream shifted every later record on a shorter / longer / CRLF line and parsed garbage lat/lng) and the last one needs its newline (upstream dropped it silently); no records at all (upstream: a zero-sized grid, SIGFPE);
   * options: a missing value after `-r`, `-lat`, `-lng`, `-p`, `-d` (upstream: SIGSEGV); a value that is not a complete number (`abc`, `5x`, out of range; upstream: `atoi`/`atof` gave 0 or a truncated value), a non-finite `-lat`/`-lng`, `-r` < 1 (upstream: no output, exit 0); an unknown option or an extra positional argument (upstream: ignored, e.g. `filelist 5 30 90` ran k = 10 at 0, 0); an option before the file list; `-p`/`-d` of opposite sign (upstream: usage and exit 0). Options are matched exactly (`-r`, `-lat`, `-lng`, `-p`, `-d`, `-h`/`--help`, `-q`, `-t`, combinable as `-qt`); the values are parsed with `strtol`/`strtod`, which give the same value as `atoi`/`atof` for every well-formed number. `-h`/`--help` prints the usage and exits 0 (upstream: `--help` crashed); `lat`/`lng` default to the documented 0 (upstream: uninitialized);
   * CUDA: a failed device query, allocation or copy, and any error of the kernel launch / copy back (checked once after the ROI, nothing added inside it), and a failed host `malloc`.

   Tested 2026-10-27 (45 cases; release and ASan builds): every case above gives its error and exit 1; ASan (host code, `-fsanitize=address`): no report in any case, against 9 reports (SEGV / FPE) and one hang at HEAD.
4. The k-nearest selection (`findLowest`, inside the ROI; 2026-10-28): upstream swapped the records through a pointer to `records[i]` taken before `records[i]` was overwritten (`tempRec = &records[i]; records[i] = records[minLoc]; records[minLoc] = *tempRec;`), so `records[minLoc]` got its own value back and the record that sat at position i was lost while its distance moved to `minLoc`. Once a lost record belongs to the k nearest, the run prints a duplicate of an earlier record with the lost record's distance: on the registered data and query correct up to -r 363, one duplicate at -r 364, only 21447 distinct records at -r 42764; the distances were always right, so the distance-only check never saw it. Now a swap by value. The registered input (-r 5) prints and captures byte-identical output (and so does every -r <= 363); ROI 420.5 us vs 421.8 us before (median of 15 interleaved runs, B200).

```bash
ctest --test-dir build/nearest_neighbor/cuda --output-on-failure   # golden run of the registry's default input
python3 tools/golden/hpcperf_golden.py run level1/nearest_neighbor [<input_id>]   # the same for any registered input
```

**Correctness switch.** The binary has no built-in CPU check (nothing to skip, so it prints no `SKIP_VERIFY`); its CPU reference is `verify.py`, which a plain run, ctest, `tools/inputs` and `tools/timing` never run and nothing falls back to. `tools/golden/hpcperf_golden.py make` runs it once, offline (`HPCPERF_CPU_REFERENCE=1`, `HPCPERF_GOLDEN_DUMP`), to validate the golden reference; verify.py then also captures its CPU reference array (`topk_distances.cpu`, same order) so make can measure d0. `HPCPERF_GOLDEN_DUMP=<dir>` makes the binary capture its output. See [tools/golden/README.md](../../tools/golden/README.md).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `cuda/nn_cuda.cu`: distance kernel, copy back, host k-nearest selection. Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 236 (1 source files, cloc, cuda/ + common/)
HIP: - (pending)
