# Level 1: Standalone GPU Benchmarks

50 selected small GPU benchmarks extracted from six upstream suites and
repackaged so each one can be configured, built, run, and validated
independently of its original suite framework. Extraction preserves upstream
algorithms, workloads, and correctness semantics.

Build and validate (per benchmark, after `source hpcperf_env.sh`):

```bash
cmake -S level1/<benchmark> -B build/<benchmark>/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/<benchmark>/cuda
ctest --test-dir build/<benchmark>/cuda --output-on-failure
```

Status legend:
- `Working` -- configure + build + run + correctness validation all verified on
  the B200 reference machine (CUDA 13.2, GCC 13.3.0, C++20).
- `HIP untested` -- upstream HIP source extracted and a HIP build configuration
  exists, but there is no AMD GPU / ROCm stack on the development machine, so
  it is not compile- or run-verified.
- `HIP pending` -- upstream has no HIP implementation; porting is deferred.
- `Borderline Level 1` -- standalone source exceeds the ~2000 LOC guideline;
  flagged for review, not removed.
- `Working (ported, ...)` -- Kokkos Kernels entries rebuilt as faithful
  manual ports of the upstream algorithm to plain CUDA/HIP (Option 2); see
  the benchmark README for port provenance and documented deviations.
- `Blocked` -- cannot be delivered as specified; see the benchmark README.

LOC measured with cloc over the benchmark's `cuda/` + `common/` (CUDA) and
`hip/` + `common/` (HIP) source directories (shared `common/` code counted in
both; data files, build files, and validation scripts excluded).

## Catalog

| Benchmark | Source suite | Brief summary | CUDA | HIP | CUDA LOC | HIP LOC | Status |
|-----------|--------------|---------------|------|-----|----------|---------|--------|
| adam | HeCBench | Updates model parameters using Adam optimization. | working | untested | 177 | 177 | Working |
| all_pairs_distance | HeCBench | Computes pairwise distances between point sets. | working | untested | 187 | 187 | Working |
| ao_bench | HeCBench | Renders ambient occlusion through ray sampling. | working | untested | 327 | 327 | Working |
| atomic_reduction | HeCBench | Reduces values using GPU atomic operations. | working | untested | 112 | 112 | Working |
| background_subtraction | HeCBench | Extracts foreground objects from a background model. | working | untested | 180 | 180 | Working |
| bezier_surface | HeCBench | Evaluates points on a Bézier surface. | working | untested | 223 | 223 | Working |
| bilateral_filter | HeCBench | Performs edge-preserving image smoothing. | working | untested | 173 | 173 | Working |
| binary_search | HeCBench | Searches sorted arrays in parallel. | working | untested | 254 | 254 | Working |
| bitonic_sort | HeCBench | Sorts values using a bitonic sorting network. | working | untested | 116 | 116 | Working |
| block_scan | HeCBench | Computes block-level parallel prefix sums. | working | untested | 162 | 160 | Working |
| burgers_equation | HeCBench | Advances a finite-difference Burgers equation solver. | working | untested | 231 | 231 | Working |
| burrows_wheeler_transform | HeCBench | Applies the Burrows-Wheeler data transformation. | working | untested | 149 | 149 | Working |
| channel_shuffle | HeCBench | Rearranges channels in tensor data. | working | untested | 208 | 208 | Working |
| histogram | HeCBench | Counts input values into histogram bins. | working | untested | 1152 | 1152 | Working |
| murmurhash3 | HeCBench | Computes MurmurHash3 hashes for independent keys. | working | untested | 186 | 186 | Working |
| nbody | HeCBench | Simulates pairwise interactions among particles. | working | untested | 362 | 362 | Working |
| daxpy | RAJAPerf | Performs the vector update y = αx + y. | working | untested | 175 | 175 | Working |
| del_dot_vec_2d | RAJAPerf | Computes a 2D divergence-like mesh operator. | working | untested | 309 | 309 | Working |
| energy | RAJAPerf | Updates energy quantities in a hydrodynamics kernel. | working | untested | 340 | 340 | Working |
| fdtd_2d | RAJAPerf | Advances 2D electromagnetic fields using FDTD. | working | untested | 234 | 234 | Working |
| floyd_warshall | RAJAPerf | Computes all-pairs shortest paths. | working | untested | 182 | 182 | Working |
| jacobi_2d | RAJAPerf | Performs iterative 2D Jacobi relaxation. | working | untested | 194 | 194 | Working |
| ltimes | RAJAPerf | Performs a transport tensor contraction. | working | untested | 199 | 199 | Working |
| mat_mat_shared | RAJAPerf | Performs tiled matrix multiplication using shared memory. | working | untested | 238 | 238 | Working |
| matvec_3d_stencil | RAJAPerf | Applies a 3D stencil matrix-vector operator. | working | untested | 319 | 319 | Working |
| pressure | RAJAPerf | Computes cell-wise fluid pressure values. | working | untested | 216 | 216 | Working |
| cg | NPB-GPU | Runs a sparse conjugate-gradient computation. | working | untested | 1741 | 2018 | Working |
| ep | NPB-GPU | Generates Gaussian random pairs independently. | working | untested | 740 | 1017 | Working |
| ft | NPB-GPU | Performs a three-dimensional fast Fourier transform. | working | untested | 1824 | 2084 | Working |
| is | NPB-GPU | Sorts and ranks integer keys. | working | untested | 1265 | 1542 | Working |
| mg | NPB-GPU | Performs multigrid smoothing and grid transfers. | working | untested | 2213 | 2504 | Working, Borderline Level 1 |
| aes | Hetero-Mark | Encrypts data blocks using AES-256. | working | untested | 881 | 739 | Working |
| black_scholes | Hetero-Mark | Prices European options using Black-Scholes. | working | untested | 687 | 687 | Working |
| color_histogram | Hetero-Mark | Builds a histogram of image color values. | working | untested | 541 | 492 | Working |
| fir | Hetero-Mark | Applies a finite impulse response filter. | working | untested | 594 | 522 | Working |
| pagerank | Hetero-Mark | Iteratively computes PageRank scores on a graph. | working | untested | 635 | 562 | Working |
| backprop | Rodinia | Trains a neural network using backpropagation. | working | pending | 694 | - | Working |
| bfs | Rodinia | Traverses graph levels using breadth-first search. | working | pending | 176 | - | Working |
| gaussian_elimination | Rodinia | Solves linear systems using Gaussian elimination. | working | pending | 302 | - | Working |
| hotspot | Rodinia | Simulates two-dimensional chip heat diffusion. | working | pending | 243 | - | Working |
| hotspot_3d | Rodinia | Simulates three-dimensional chip heat diffusion. | working | pending | 246 | - | Working |
| lud | Rodinia | Performs LU decomposition of a dense matrix. | working | pending | 466 | - | Working |
| nearest_neighbor | Rodinia | Finds nearest points based on spatial distance. | working | pending | 236 | - | Working |
| needleman_wunsch | Rodinia | Performs global sequence alignment with dynamic programming. | working | pending | 305 | - | Working |
| pathfinder | Rodinia | Finds minimum-cost paths through a grid. | working | pending | 172 | - | Working |
| srad_v1 | Rodinia | Applies speckle-reducing anisotropic diffusion. | working | pending | 628 | - | Working |
| graph_coloring | Kokkos Kernels | Assigns colors to graph vertices without conflicts. | working | untested | 244 | 244 | Working (ported, VBBIT) |
| spadd | Kokkos Kernels | Adds two sparse matrices. | working | untested | 312 | 312 | Working (ported, sorted) |
| spgemm | Kokkos Kernels | Multiplies two sparse matrices. | working | untested | 393 | 393 | Working (ported, KKMEM numeric) |
| spmv | Kokkos Kernels | Multiplies a sparse matrix by a dense vector. | working | untested | 200 | 200 | Working (ported, perf_test kk) |

Summary: 50/50 Working (CUDA validated): 16 HeCBench, 10 RAJAPerf, 5 NPB-GPU,
5 Hetero-Mark, 10 Rodinia, 4 Kokkos Kernels. The four Kokkos Kernels entries
marked "ported" are faithful manual ports of the upstream algorithms to plain
CUDA/HIP (per-benchmark READMEs list exactly what was ported verbatim and any
documented deviations). Selection change (2026-08-31): triangle_counting
(Kokkos Kernels) was removed and replaced by murmurhash3 (HeCBench) because
upstream Kokkos Kernels has no GPU triangle-counting implementation to
extract (its GPU functor is an empty stub). One Working benchmark (mg) is
flagged Borderline Level 1 (~2.2k LOC).
