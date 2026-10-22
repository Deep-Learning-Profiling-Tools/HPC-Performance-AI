# Release plan: `level3-source-hpcperf-l3-v1-ext1` (source-artifact prerelease, NOT published)

Generated 2026-10-21T19:34:32Z at target commit `7e2a220b5682b4562e066d77e4feea51b80803ba`. Provider: GitHub Release assets of `Deep-Learning-Profiling-Tools/HPC-Performance-AI`. Immutable releases on the repository: `{'enabled': False, 'enforced_by_owner': False, 'probed': '2026-09-11 via GET /repos/{owner}/{repo}/immutable-releases (read-only); repository setting not changed'}` -- the project rule (never overwrite or delete an asset; consumers verify size, sha256 and source_tree_sha256) holds regardless. Nothing has been uploaded; every URL below is PLANNED and the locks stay `unpublished` until the asset exists and was re-downloaded and verified.

**3 assets, 155.0 MB (162526367 bytes)** + SHA256SUMS + one SOURCE_MANIFEST per artifact.

| artifact | file | bytes | archive sha256 | source_tree_sha256 | app license | bundle/data review | verify | planned URL |
|---|---|---|---|---|---|---|---|---|
| vpic | `vpic-hpcperf-l3-v1.tar.zst` | 1937373 | `ac0aec44f3245b6078021ba1cea0ba9b3e604b3d8ac2489f7807cb3c2d177950` | `464de895194405663afeff1d0bb76b6beae1f300f4a3d848507f73cc837d3e7c` | BSD-3-Clause (README.md "Copyright" section; Triad National Security, LLC, 2022) | cleared (2 components, `level3/vpic/provenance/LICENSES.md`) | PASS (full re-extraction) | `https://github.com/Deep-Learning-Profiling-Tools/HPC-Performance-AI/releases/download/level3-source-hpcperf-l3-v1-ext1/vpic-hpcperf-l3-v1.tar.zst` |
| picongpu | `picongpu-hpcperf-l3-v1.tar.zst` | 160232498 | `d243d831a44ec7c845464839594de73d6eeb55e7c5b63080a00ae0c74a03d91f` | `70d87f93d3a6fd7bf5625a3dbea3dd301e0a4572fcf2e8f5d4328575eab7454a` | GPL-3.0-or-later (LICENSE.md, COPYING) | cleared (5 components, `level3/picongpu/provenance/LICENSES.md`) | PASS (full re-extraction) | `https://github.com/Deep-Learning-Profiling-Tools/HPC-Performance-AI/releases/download/level3-source-hpcperf-l3-v1-ext1/picongpu-hpcperf-l3-v1.tar.zst` |
| cholla | `cholla-hpcperf-l3-v1.tar.zst` | 356496 | `bdc734e8674b9bc2764078e25128a9dc8aa06dcbd18e278bcb5145812d155506` | `44537e4f47e9a1c7f005561de3064b400560508f30bcd477db875b75d4527450` | MIT (LICENSE.txt, Copyright (c) 2018 Evan E. Schneider) | cleared (1 components, `level3/cholla/provenance/LICENSES.md`) | PASS (full re-extraction) | `https://github.com/Deep-Learning-Profiling-Tools/HPC-Performance-AI/releases/download/level3-source-hpcperf-l3-v1-ext1/cholla-hpcperf-l3-v1.tar.zst` |

## Excluded

- lammps: NOT_IN_THIS_ROUND (--app): not part of this publication round; its artifact keeps its own release
- sparta: NOT_IN_THIS_ROUND (--app): not part of this publication round; its artifact keeps its own release
- warpx: NOT_IN_THIS_ROUND (--app): not part of this publication round; its artifact keeps its own release
- specfem3d: NOT_IN_THIS_ROUND (--app): not part of this publication round; its artifact keeps its own release
- nekrs: NOT_IN_THIS_ROUND (--app): not part of this publication round; its artifact keeps its own release
- nyx: NOT_IN_THIS_ROUND (--app): not part of this publication round; its artifact keeps its own release
- cp2k: NOT_IN_THIS_ROUND (--app): not part of this publication round; its artifact keeps its own release
- qmcpack: NOT_IN_THIS_ROUND (--app): not part of this publication round; its artifact keeps its own release
- dftfe: NOT_IN_THIS_ROUND (--app): not part of this publication round; its artifact keeps its own release
- geos: NOT_IN_THIS_ROUND (--app): not part of this publication round; its artifact keeps its own release
- exaca: NOT_IN_THIS_ROUND (--app): not part of this publication round; its artifact keeps its own release

## SHA256SUMS (planned asset)

```
ac0aec44f3245b6078021ba1cea0ba9b3e604b3d8ac2489f7807cb3c2d177950  vpic-hpcperf-l3-v1.tar.zst
d243d831a44ec7c845464839594de73d6eeb55e7c5b63080a00ae0c74a03d91f  picongpu-hpcperf-l3-v1.tar.zst
bdc734e8674b9bc2764078e25128a9dc8aa06dcbd18e278bcb5145812d155506  cholla-hpcperf-l3-v1.tar.zst
```

## Scientific status and build prerequisites (per artifact)

- **vpic** (upstream devel `7eb2947e1274`, patches 0): validated backends ['cuda']; smoke: the registered deck executables (inputs.yaml); no strong/weak derivation -- VPIC decks are compiled programs, every size is its own registered deck; multi-node: BLOCKED/UNVERIFIED on this site; 40/80: not applicable (one-rank decks). Prerequisites: CUDA Toolkit 13.2.78; GCC (nvcc_wrapper host compiler) 13.3.0; Open MPI 5.0.10; CMake / Ninja.
- **picongpu** (upstream dev `93cba5b61b62`, patches 0): validated backends ['cuda']; smoke: the registered input sets (inputs.yaml); no strong/weak derivation; multi-node: BLOCKED/UNVERIFIED on this site; 40/80: not applicable (one-rank workloads). Prerequisites: CUDA Toolkit 13.2.78; GCC (CUDA host compiler) 13.3.0; Open MPI 5.0.10; CMake / Ninja.
- **cholla** (upstream v3.0.0 `8478c9ede8a5`, patches 1): validated backends ['cuda']; smoke: the registered parameter files (inputs.yaml); no strong/weak derivation -- every grid size is its own registered input; multi-node: BLOCKED/UNVERIFIED on this site; 40/80: not applicable (one-rank inputs). Prerequisites: CUDA Toolkit 13.2.78; GCC (nvcc host compiler through mpicxx) 13.3.0; Open MPI 5.0.10; HDF5 2.2.0 (mpi_openmpi build).

## Procedure after the maintainer's upload authorization

1. draft release on the target commit (maintainer authorization required)
2. upload assets + SHA256SUMS + SOURCE_MANIFEST.<artifact>.json, re-download each asset and compare sha256/size
3. publish as prerelease
4. clean clone + empty cache + anonymous (no token) download through prepare_benchmark.sh, tree hash verified (tools/artifacts/remote_fetch_check.sh)
5. only then: locks primary.url = asset URL, status published; provenance/remote_fetch_verification*.yaml written; REMOTE_FETCH_VERIFIED

