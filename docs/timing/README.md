# Timing results

Region-of-interest timing of the Level 1 benchmarks and Level 2 mini-applications (latest record 2026-09-22T22:28:58Z, 79 records). Open [index.html](index.html) to choose an application, an input and a platform; this file lists the latest successful run of every measured combination. `null`: not observable on that platform.

## Level 1

| application | input | platform | ROI | spread | device busy | host gap | kernels in ROI | ROI share of process | profiler x | runs | vs previous |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| adam | default | nvidia-b200.cuda13.2 | 7.37 ms | 0.1% | 98% | 115 us | 100 | 1.4% | 1.00 | 1 | first run |
| aes | default | nvidia-b200.cuda13.2 | 263 us | 1.4% | 22% | 204 us | 1 | 0.0% | 1.05 | 1 | first run |
| all_pairs_distance | default | nvidia-b200.cuda13.2 | 2.04 s | 0.0% | 99% | 15.6 ms | 30,000 | 78.0% | 1.00 | 1 | first run |
| all_pairs_distance | n20000 | nvidia-b200.cuda13.2 | 4.08 s | 0.0% | 99% | 32.8 ms | 60,000 | 88.0% | 1.00 | 1 | first run |
| ao_bench | default | nvidia-b200.cuda13.2 | 17.2 ms | 0.1% | 93% | 1.23 ms | 100 | 3.1% | 1.01 | 1 | first run |
| atomic_reduction | default | nvidia-b200.cuda13.2 | 169 ms | 0.0% | 97% | 4.54 ms | 2,000 | 11.8% | 1.02 | 1 | first run |
| background_subtraction | default | nvidia-b200.cuda13.2 | 64.6 ms | 3.5% | 81% | 12.3 ms | 300 | 3.4% | 0.95 | 1 | first run |
| backprop | default | nvidia-b200.cuda13.2 | 1.2 ms | 6.4% | 53% | 557 us | 2 | 0.2% | 1.13 | 1 | first run |
| bezier_surface | default | nvidia-b200.cuda13.2 | 62.7 ms | 0.0% | 100% | 0 us | 1 | 7.9% | 1.01 | 1 | first run |
| bfs | default | nvidia-b200.cuda13.2 | 392 us | 1.5% | 33% | 264 us | 20 | 0.1% | 1.22 | 1 | first run |
| bilateral_filter | default | nvidia-b200.cuda13.2 | 4.17 s | 0.0% | 100% | 922 us | 3,000 | 87.5% | 1.00 | 1 | first run |
| binary_search | default | nvidia-b200.cuda13.2 | 424 ms | 0.0% | 100% | 194 us | 400 | 35.1% | 1.00 | 1 | first run |
| bitonic_sort | default | nvidia-b200.cuda13.2 | 31.5 ms | 0.0% | 99% | 173 us | 325 | 3.0% | 1.00 | 1 | first run |
| black_scholes | default | nvidia-b200.cuda13.2 | 5.26 ms | 1.0% | 24% | 4 ms | 512 | 0.9% | 1.13 | 1 | first run |
| block_scan | default | nvidia-b200.cuda13.2 | 28.6 s | 0.0% | 99% | 356 ms | 1,800 | 49.5% | 0.99 | 1 | first run |
| burgers_equation | default | nvidia-b200.cuda13.2 | 86.6 ms | 0.0% | 100% | 248 us | 400 | 5.6% | 1.00 | 1 | first run |
| burrows_wheeler_transform | default | nvidia-b200.cuda13.2 | 33.2 ms | 0.0% | 100% | 110 us | 301 | 4.6% | 1.00 | 1 | first run |
| cg | default | nvidia-b200.cuda13.2 | 230 ms | 0.2% | 73% | 62.6 ms | 11,775 | 6.8% | 1.07 | 1 | first run |
| channel_shuffle | default | nvidia-b200.cuda13.2 | 2.21 s | 0.0% | 100% | 1.37 ms | 2,400 | 23.4% | 1.00 | 1 | first run |
| color_histogram | default | nvidia-b200.cuda13.2 | 462 us | 2.4% | 80% | 94 us | 1 | 0.1% | 1.11 | 1 | first run |
| daxpy | default | nvidia-b200.cuda13.2 | 2.03 ms | 0.2% | 90% | 208 us | 500 | 0.4% | 1.02 | 1 | first run |
| del_dot_vec_2d | default | nvidia-b200.cuda13.2 | 945 us | 0.5% | 87% | 119 us | 100 | 0.2% | 1.04 | 1 | first run |
| energy | default | nvidia-b200.cuda13.2 | 3.85 ms | 0.6% | 89% | 434 us | 780 | 0.7% | 1.01 | 1 | first run |
| ep | default | nvidia-b200.cuda13.2 | 43.4 ms | 0.1% | 99% | 550 us | 1 | 7.8% | 1.00 | 1 | first run |
| fdtd_2d | default | nvidia-b200.cuda13.2 | 4.65 ms | 0.1% | 91% | 435 us | 1,280 | 0.9% | 1.01 | 1 | first run |
| fir | default | nvidia-b200.cuda13.2 | 31 ms | 0.3% | 32% | 21 ms | 1,024 | 5.2% | 1.24 | 1 | first run |
| floyd_warshall | default | nvidia-b200.cuda13.2 | 31.9 ms | 0.0% | 93% | 2.12 ms | 8,000 | 5.6% | 1.00 | 1 | first run |
| ft | default | nvidia-b200.cuda13.2 | 182 ms | 0.0% | 99% | 1.65 ms | 229 | 23.3% | 1.00 | 1 | first run |
| gaussian_elimination | default | nvidia-b200.cuda13.2 | 3.9 ms | 0.7% | 20% | 3.11 ms | 414 | 0.7% | 1.13 | 1 | first run |
| graph_coloring | default | nvidia-b200.cuda13.2 | 6.41 ms | 0.4% | 98% | 129 us | 10 | 0.6% | 1.00 | 1 | first run |
| histogram | default | nvidia-b200.cuda13.2 | 52.7 ms | 0.4% | 32% | 36 ms | 1,200 | 8.8% | 1.05 | 1 | first run |
| hotspot | default | nvidia-b200.cuda13.2 | 90 us | 4.7% | 6% | 84 us | 1 | 0.0% | 1.44 | 1 | first run |
| hotspot_3d | default | nvidia-b200.cuda13.2 | 393 us | 0.3% | 88% | 48 us | 100 | 0.1% | 1.05 | 1 | first run |
| is | default | nvidia-b200.cuda13.2 | 1.93 ms | 0.1% | 99% | 20 us | 70 | 0.4% | 1.01 | 1 | first run |
| jacobi_2d | default | nvidia-b200.cuda13.2 | 18.6 ms | 0.0% | 94% | 1.11 ms | 4,000 | 3.4% | 1.00 | 1 | first run |
| ltimes | default | nvidia-b200.cuda13.2 | 2.12 ms | 0.3% | 95% | 101 us | 50 | 0.4% | 1.01 | 1 | first run |
| lud | default | nvidia-b200.cuda13.2 | 488 us | 2.1% | 73% | 133 us | 46 | 0.1% | 1.14 | 1 | first run |
| mat_mat_shared | default | nvidia-b200.cuda13.2 | 2.31 ms | 0.2% | 96% | 86 us | 5 | 0.4% | 1.01 | 1 | first run |
| matvec_3d_stencil | default | nvidia-b200.cuda13.2 | 2.93 ms | 0.2% | 96% | 114 us | 100 | 0.5% | 1.02 | 1 | first run |
| mg | default | nvidia-b200.cuda13.2 | 18 ms | 0.1% | 90% | 1.73 ms | 2,126 | 1.1% | 1.00 | 1 | first run |
| murmurhash3 | default | nvidia-b200.cuda13.2 | 97.7 ms | 0.0% | 100% | 0 us | 100 | 9.3% | 1.01 | 1 | first run |
| nbody | default | nvidia-b200.cuda13.2 | 20.6 ms | 0.0% | 98% | 319 us | 30 | 3.8% | 1.00 | 1 | first run |
| nearest_neighbor | default | nvidia-b200.cuda13.2 | 267 us | 3.4% | 3% | 259 us | 1 | 0.1% | 1.26 | 1 | first run |
| needleman_wunsch | default | nvidia-b200.cuda13.2 | 1.23 ms | 0.2% | 85% | 188 us | 255 | 0.2% | 1.04 | 1 | first run |
| pagerank | default | nvidia-b200.cuda13.2 | 2.96 ms | 0.4% | 86% | 408 us | 500 | 0.5% | 1.00 | 1 | first run |
| pathfinder | default | nvidia-b200.cuda13.2 | 138 us | 2.0% | 36% | 89 us | 5 | 0.0% | 1.18 | 1 | first run |
| pressure | default | nvidia-b200.cuda13.2 | 6.16 ms | 0.1% | 93% | 451 us | 1,400 | 1.1% | 1.00 | 1 | first run |
| spadd | default | nvidia-b200.cuda13.2 | 416 us | 1.9% | 13% | 362 us | 3 | 0.1% | 1.09 | 1 | first run |
| spgemm | default | nvidia-b200.cuda13.2 | 774 us | 1.2% | 88% | 94 us | 1 | 0.1% | 1.04 | 1 | first run |
| spmv | default | nvidia-b200.cuda13.2 | 1.26 ms | 1.5% | 40% | 764 us | 100 | 0.2% | 1.09 | 1 | first run |
| srad_v1 | default | nvidia-b200.cuda13.2 | 4.87 ms | 0.4% | 52% | 2.34 ms | 502 | 0.9% | 1.33 | 1 | first run |

## Level 2

| application | input | platform | ROI | spread | device busy | host gap | kernels in ROI | ROI share of process | profiler x | FOM | vs own timer | runs | vs previous |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| amg2023 | default | nvidia-b200.cuda13.2 | 598 ms | null | 84% | 96.1 ms | 5,069 | 11.4% | 1.03 | 1.549e+09 nnz_AP/s | - | 1 | first run |
| amg2023 | n128 | nvidia-b200.cuda13.2 | 149 ms | null | 50% | 74.2 ms | 4,508 | 6.8% | 1.15 | 7.694e+08 nnz_AP/s | - | 1 | first run |
| amg2023 | n192 | nvidia-b200.cuda13.2 | 306 ms | null | 72% | 85.9 ms | 5,039 | 9.5% | 1.06 | 1.275e+09 nnz_AP/s | - | 1 | first run |
| branson | default | nvidia-b200.cuda13.2 | 9.89 s | null | 43% | 5.64 s | 85,448 | 87.2% | 1.08 | 1.47e+06 photons/s | - | 1 | first run |
| cabanapic | default | nvidia-b200.cuda13.2 | 61.5 s | null | 93% | 4.16 s | 2,016,000 | 98.9% | 1.00 | none printed | - | 1 | first run |
| cloverleaf | default | nvidia-b200.cuda13.2 | 17.7 s | null | 96% | 652 ms | 369,965 | 91.1% | 1.04 | 4.07e-10 s/cell | - | 1 | first run |
| comb | default | nvidia-b200.cuda13.2 | 894 ms | null | 19% | 725 ms | 1,604 | 40.8% | 1.23 | none printed | - | 1 | first run |
| exacmech | default | nvidia-b200.cuda13.2 | 8.04 s | null | 99% | 115 ms | 10,000 | 91.8% | 1.00 | none printed | -0.000% | 1 | first run |
| examinimd | default | nvidia-b200.cuda13.2 | 9.85 s | null | 98% | 199 ms | 10,330 | 65.7% | 1.00 | 1.664e+09 atom-steps/s | - | 1 | first run |
| exampm | default | nvidia-b200.cuda13.2 | 9.03 s | null | 93% | 588 ms | 43,133 | 79.3% | 1.03 | none printed | - | 1 | first run |
| gamess_ri_mp2 | default | nvidia-b200.cuda13.2 | 741 ms | null | 99% | 8.82 ms | 604 | 23.2% | 1.01 | none printed | - | 1 | first run |
| haccabanapm | default | nvidia-b200.cuda13.2 | 13.6 s | null | 55% | 6.19 s | 132,500 | 66.0% | 1.19 | 0.0 s/step | - | 1 | first run |
| hipbone | default | nvidia-b200.cuda13.2 | 288 ms | null | 98% | 5.54 ms | 906 | 1.5% | 1.00 | 3,465.3 GFLOPs | -0.005% | 1 | first run |
| kripke | default | nvidia-b200.cuda13.2 | 843 ms | null | 97% | 21.8 ms | 928 | 38.8% | 1.00 | 3.183e+09 unknowns/(s/iteration) | - | 1 | first run |
| laghos | default | nvidia-b200.cuda13.2 | 126.3 s | null | 76% | 30.7 s | 14,536,044 | 97.7% | 1.35 | 1,900.1 megadofs*cg_iterations/s | - | 1 | first run |
| minibude | default | nvidia-b200.cuda13.2 | 4.37 s | null | 100% | 9.63 ms | 128 | 78.3% | 1.00 | 333.8 GFLOP/s | - | 1 | first run |
| miniem | default | nvidia-b200.cuda13.2 | 1.31 s | null | 72% | 368 ms | 67,081 | 1.2% | 1.07 | 19,511.1 k-cell-steps/s | - | 1 | first run |
| miniweather | default | nvidia-b200.cuda13.2 | 32.3 s | null | 99% | 176 ms | 737,280 | 95.9% | 1.00 | none printed | +0.000% | 1 | first run |
| p3_heat3d | default | nvidia-b200.cuda13.2 | 1.59 s | null | 100% | 7.86 ms | 1,000 | 28.2% | 1.00 | 1,348.0 GB/s | -0.002% | 1 | first run |
| p3_vlp4d | default | nvidia-b200.cuda13.2 | 2.31 s | null | 99% | 13.6 ms | 2,445 | 29.5% | 1.00 | none printed | -0.002% | 1 | first run |
| quicksilver | default | nvidia-b200.cuda13.2 | 7.48 s | null | 52% | 3.62 s | 410 | 80.3% | 1.06 | 5.585e+06 segments/s | -0.001% | 1 | first run |
| quicksilver | p200000 | nvidia-b200.cuda13.2 | 7.46 s | null | 46% | 4.03 s | 414 | 80.0% | 0.67 | 1.13e+07 segments/s | +0.004% | 1 | first run |
| quicksilver | p50000 | nvidia-b200.cuda13.2 | 6.2 s | null | 61% | 2.44 s | 403 | 74.4% | 1.10 | 3.354e+06 segments/s | -0.008% | 1 | first run |
| remhos | default | nvidia-b200.cuda13.2 | 19.6 s | null | 29% | 13.9 s | 67,199 | 79.7% | 1.08 | 193.1 | - | 1 | first run |
| shaw | default | nvidia-b200.cuda13.2 | 15.5 s | null | 93% | 1.1 s | 120,000 | 59.7% | 1.02 | 72,941.9 GB/s | -0.000% | 1 | first run |
| sw4lite | default | nvidia-b200.cuda13.2 | 59.7 ms | null | 92% | 4.88 ms | 575 | 0.7% | 1.01 | none printed | - | 1 | first run |
| tealeaf | default | nvidia-b200.cuda13.2 | 18.4 s | null | 77% | 4.26 s | 634,749 | 89.4% | 1.14 | 1.149e-06 s/cell | - | 1 | first run |
| xsbench | default | nvidia-b200.cuda13.2 | 40.1 ms | null | 97% | 1.22 ms | 3 | 1.1% | 1.01 | 4.239e+08 lookups/s | +0.186% | 1 | first run |
