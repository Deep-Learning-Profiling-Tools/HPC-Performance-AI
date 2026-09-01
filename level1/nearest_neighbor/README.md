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
python3 level1/nearest_neighbor/verify.py ./build/nearest_neighbor/cuda/nearest_neighbor_cuda build/nearest_neighbor/cuda/filelist_4 5 30 90
```

## Validation

Added check (verify.py): CPU recomputation of all distances from the db records; the reported top-k matches within tolerance. The file list is generated at configure time with absolute data paths.

Run it via:

```bash
ctest --test-dir build/nearest_neighbor/cuda --output-on-failure
```

## LOC

CUDA: 236 (1 source files, cloc, cuda/ + common/)
HIP: - (pending)
