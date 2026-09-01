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

| Benchmark | Source suite | CUDA | HIP | CUDA LOC | HIP LOC | Status |
|-----------|--------------|------|-----|----------|---------|--------|
| adam | HeCBench | working | untested | 177 | 177 | Working |
| all_pairs_distance | HeCBench | working | untested | 187 | 187 | Working |
| ao_bench | HeCBench | working | untested | 327 | 327 | Working |
| atomic_reduction | HeCBench | working | untested | 112 | 112 | Working |
| background_subtraction | HeCBench | working | untested | 180 | 180 | Working |
| bezier_surface | HeCBench | working | untested | 223 | 223 | Working |
| bilateral_filter | HeCBench | working | untested | 173 | 173 | Working |
| binary_search | HeCBench | working | untested | 254 | 254 | Working |
| bitonic_sort | HeCBench | working | untested | 116 | 116 | Working |
| block_scan | HeCBench | working | untested | 162 | 160 | Working |
| burgers_equation | HeCBench | working | untested | 231 | 231 | Working |
| burrows_wheeler_transform | HeCBench | working | untested | 149 | 149 | Working |
| channel_shuffle | HeCBench | working | untested | 208 | 208 | Working |
| histogram | HeCBench | working | untested | 1152 | 1152 | Working |
| murmurhash3 | HeCBench | working | untested | 186 | 186 | Working |
| nbody | HeCBench | working | untested | 362 | 362 | Working |
| daxpy | RAJAPerf | working | untested | 175 | 175 | Working |
| del_dot_vec_2d | RAJAPerf | working | untested | 309 | 309 | Working |
| energy | RAJAPerf | working | untested | 340 | 340 | Working |
| fdtd_2d | RAJAPerf | working | untested | 234 | 234 | Working |
| floyd_warshall | RAJAPerf | working | untested | 182 | 182 | Working |
| jacobi_2d | RAJAPerf | working | untested | 194 | 194 | Working |
| ltimes | RAJAPerf | working | untested | 199 | 199 | Working |
| mat_mat_shared | RAJAPerf | working | untested | 238 | 238 | Working |
| matvec_3d_stencil | RAJAPerf | working | untested | 319 | 319 | Working |
| pressure | RAJAPerf | working | untested | 216 | 216 | Working |
| cg | NPB-GPU | working | untested | 1741 | 2018 | Working |
| ep | NPB-GPU | working | untested | 740 | 1017 | Working |
| ft | NPB-GPU | working | untested | 1824 | 2084 | Working |
| is | NPB-GPU | working | untested | 1265 | 1542 | Working |
| mg | NPB-GPU | working | untested | 2213 | 2504 | Working, Borderline Level 1 |
| aes | Hetero-Mark | working | untested | 881 | 739 | Working |
| black_scholes | Hetero-Mark | working | untested | 687 | 687 | Working |
| color_histogram | Hetero-Mark | working | untested | 541 | 492 | Working |
| fir | Hetero-Mark | working | untested | 594 | 522 | Working |
| pagerank | Hetero-Mark | working | untested | 635 | 562 | Working |
| backprop | Rodinia | working | pending | 694 | - | Working |
| bfs | Rodinia | working | pending | 176 | - | Working |
| gaussian_elimination | Rodinia | working | pending | 302 | - | Working |
| hotspot | Rodinia | working | pending | 243 | - | Working |
| hotspot_3d | Rodinia | working | pending | 246 | - | Working |
| lud | Rodinia | working | pending | 466 | - | Working |
| nearest_neighbor | Rodinia | working | pending | 236 | - | Working |
| needleman_wunsch | Rodinia | working | pending | 305 | - | Working |
| pathfinder | Rodinia | working | pending | 172 | - | Working |
| srad_v1 | Rodinia | working | pending | 628 | - | Working |
| graph_coloring | Kokkos Kernels | working | untested | 244 | 244 | Working (ported, VBBIT) |
| spadd | Kokkos Kernels | working | untested | 312 | 312 | Working (ported, sorted) |
| spgemm | Kokkos Kernels | working | untested | 393 | 393 | Working (ported, KKMEM numeric) |
| spmv | Kokkos Kernels | working | untested | 200 | 200 | Working (ported, perf_test kk) |

Summary: 50/50 Working (CUDA validated): 16 HeCBench, 10 RAJAPerf, 5 NPB-GPU,
5 Hetero-Mark, 10 Rodinia, 4 Kokkos Kernels. The four Kokkos Kernels entries
marked "ported" are faithful manual ports of the upstream algorithms to plain
CUDA/HIP (per-benchmark READMEs list exactly what was ported verbatim and any
documented deviations). Selection change (2026-08-31): triangle_counting
(Kokkos Kernels) was removed and replaced by murmurhash3 (HeCBench) because
upstream Kokkos Kernels has no GPU triangle-counting implementation to
extract (its GPU functor is an empty stub). One Working benchmark (mg) is
flagged Borderline Level 1 (~2.2k LOC).
