# BFS

Breadth-first search over a Rodinia graph file.

## Source

Source suite: Rodinia (cuda/bfs, data/bfs)
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
cmake -S level1/bfs -B build/bfs/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/bfs/cuda
```

## Run

```bash
python3 level1/bfs/verify.py ./build/bfs/cuda/bfs_cuda level1/bfs/data/graph65536.txt
```

## Validation

Added check (verify.py): CPU BFS recomputes all hop counts from the graph and compares result.txt. Test input: graph65536.txt (upstream file); the 64 MB canonical graph1MW_6.txt is not committed but works the same way.

Run it via:

```bash
ctest --test-dir build/bfs/cuda --output-on-failure
```

## LOC

CUDA: 176 (3 source files, cloc, cuda/ + common/)
HIP: - (pending)
