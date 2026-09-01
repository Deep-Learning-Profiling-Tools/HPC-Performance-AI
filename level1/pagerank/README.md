# PageRank

Power-iteration PageRank on a CSR link matrix.

## Source

Source suite: Hetero-Mark (src/pr (benchmark + cuda + hip))
Upstream repository: https://github.com/NUCAR-DEV/Hetero-Mark
Upstream commit: aa685e02705f4be380cba215c6df10b42262a9cd

## Verified Environment

GCC/G++ 13.3.0 (conda, pinned) | C++20 | CMake 3.28.4 | Ninja 1.13.2 | Python 3.12.3
CUDA Toolkit 13.2 (nvcc 13.2.78, /usr/local/cuda) | NVIDIA B200 (sm_100), driver 595.58.03
HIP/ROCm: source + build config present where noted, unverified (no AMD GPU available)

Reproduce the toolchain from the repository root: `./setup_env.sh` then
`source hpcperf_env.sh` (all user-space versions are pinned in environment.yml).

## Backends

CUDA: working (configure + build + run + validation verified on B200)
HIP: extracted, untested (no AMD GPU / ROCm on the development machine)

## Build

```bash
source hpcperf_env.sh
cmake -S level1/pagerank -B build/pagerank/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/pagerank/cuda
```

HIP (build configuration present, unverified without ROCm):

```bash
cmake -S level1/pagerank -B build/pagerank/hip -DBACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build/pagerank/hip
```

## Run

```bash
./build/pagerank/cuda/pagerank_cuda -i build/pagerank/cuda/data/16384.data
```

## Validation

Built-in (upstream Verify): GPU PageRank vector compared against the upstream CPU implementation (prints a diagnostic only on failure). The CSR matrix is generated deterministically at configure time (data/gen_input.py) because the upstream data server is offline.

Run it via:

```bash
ctest --test-dir build/pagerank/cuda --output-on-failure
```

## LOC

CUDA: 635 (20 source files, cloc, cuda/ + common/)
HIP: 562
