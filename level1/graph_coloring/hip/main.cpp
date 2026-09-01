//
// Distance-1 graph coloring benchmark, faithful standalone port of the Kokkos
// Kernels COLORING_VBBIT algorithm (the COLORING_DEFAULT choice on GPU
// execution spaces), BSD-3-Clause; commit
// 23a699f3c5bd662bf2ed52116e56533ecc3ddae0.
//
// What is ported and from where (graph/impl/KokkosGraph_Distance1Color_impl.hpp
// and graph/src/KokkosGraph_Distance1ColorHandle.hpp):
//  - functorGreedyColor_IMPLOG: speculative greedy coloring with a 64-bit
//    forbidden-color window (VBBIT_COLORING_FORBIDDEN_SIZE = 64), chunked
//    work items (vb_chunk_size = 8, dropping to 1 for short worklists).
//  - functorFindConflicts_Atomic: conflict detection (i < neighbor rule),
//    uncoloring + atomic append to the next worklist (COLORING_ATOMIC
//    conflict scheme, the VBBIT default).
//  - GraphColor_VB::color_graph main loop: worklist swap per iteration,
//    max_number_of_iterations = 200, then the serial host fallback
//    resolveConflicts for any remaining vertices.
//  Defaults per Distance1ColorHandle::set_defaults for VBBIT (edge filtering
//  off, serial resolution off).
//
// Input graph (the upstream perf_test requires an .mtx file; none ships with
// the repo): a deterministic random graph built with the same generator used
// across kokkos-kernels tests, kk_sparseMatrix_generate (glibc srand(13721)),
// then symmetrized (G = A union A^T, as required for coloring) and sorted.
// Defaults: 200000 vertices, ~30 entries/row before symmetrization.
//
// Validation (added; coloring has a complete correctness spec): the host
// checks that every vertex is colored and no edge connects two vertices of
// the same color, and reports the number of colors used.
//
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>

using Ordinal = int;      // nnz_lno_t
using Offset  = size_t;   // size_type
using color_t = int;
using ban_type = long long int;
#define VBBIT_COLORING_FORBIDDEN_SIZE 64

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemset hipMemset
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaGetLastError hipGetLastError
#else
#include <cuda_runtime.h>
#endif

#define GPU_CHECK(call)                                                  \
  do {                                                                   \
    cudaError_t err_ = (call);                                           \
    if (err_ != cudaSuccess) {                                           \
      fprintf(stderr, "GPU error %s at %s:%d\n",                         \
              cudaGetErrorString(err_), __FILE__, __LINE__);             \
      exit(1);                                                           \
    }                                                                    \
  } while (0)

// ---------------------------------------------------------------------------
// Deterministic input graph (see header comment).
// ---------------------------------------------------------------------------
static void generate_symmetric_graph(Ordinal nv, int nnzPerRow,
                                     std::vector<Offset>& xadj, std::vector<Ordinal>& adj)
{
  // structure part of kk_sparseMatrix_generate (sparse/src/KokkosSparse_IOUtils.hpp)
  std::vector<Offset> rowPtr(nv + 1, 0);
  std::vector<Ordinal> colInd;
  Ordinal bandwidth = (nv + 3) / 3;
  srand(13721);
  for (int row = 0; row < nv; row++) {
    int varianz = (1.0 * rand() / RAND_MAX - 0.5) * 0;
    (void)varianz;
    rowPtr[row + 1] = rowPtr[row] + nnzPerRow;
  }
  colInd.resize(rowPtr[nv]);
  for (Ordinal row = 0; row < nv; row++) {
    for (Offset k = rowPtr[row]; k < rowPtr[row + 1]; ++k) {
      while (true) {
        Ordinal pos = (1.0 * rand() / RAND_MAX - 0.5) * bandwidth + row;
        while (pos < 0) pos += nv;
        while (pos >= nv) pos -= nv;
        bool dup = false;
        for (Offset j = rowPtr[row]; j < k; j++)
          if (colInd[j] == pos) { dup = true; break; }
        if (!dup) { colInd[k] = pos; break; }
      }
    }
  }
  // symmetrize: G = A union A^T (self-loops are skipped by the algorithm)
  std::vector<std::vector<Ordinal>> nbrs(nv);
  for (Ordinal i = 0; i < nv; i++)
    for (Offset k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
      Ordinal j = colInd[k];
      nbrs[i].push_back(j);
      nbrs[j].push_back(i);
    }
  xadj.assign(nv + 1, 0);
  for (Ordinal i = 0; i < nv; i++) {
    std::sort(nbrs[i].begin(), nbrs[i].end());
    nbrs[i].erase(std::unique(nbrs[i].begin(), nbrs[i].end()), nbrs[i].end());
    xadj[i + 1] = xadj[i] + nbrs[i].size();
  }
  adj.resize(xadj[nv]);
  for (Ordinal i = 0; i < nv; i++)
    std::copy(nbrs[i].begin(), nbrs[i].end(), adj.begin() + xadj[i]);
}

// ---------------------------------------------------------------------------
// Port of functorInitList
// ---------------------------------------------------------------------------
__global__ void init_list(Ordinal nv, Ordinal* vertexList)
{
  Ordinal i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nv) vertexList[i] = i;
}

// ---------------------------------------------------------------------------
// Port of functorGreedyColor_IMPLOG (VBBIT speculative greedy coloring)
// ---------------------------------------------------------------------------
__global__ void greedy_color_implog(Ordinal nv, const Offset* _idx, const Ordinal* _adj,
                                    color_t* _colors, const Ordinal* _vertexList,
                                    Ordinal _vertexListLength, Ordinal _chunkSize,
                                    Ordinal work_items)
{
  const Ordinal ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii >= work_items) return;
  Ordinal i = 0;
  for (Ordinal ichunk = 0; ichunk < _chunkSize; ichunk++) {
    if (ii * _chunkSize + ichunk < _vertexListLength)
      i = _vertexList[ii * _chunkSize + ichunk];
    else
      continue;
    if (_colors[i] > 0) continue;  // Already colored this vertex

    Offset my_xadj_end = _idx[i + 1];
    Offset xadjbegin   = _idx[i];
    color_t degree = my_xadj_end - xadjbegin;  // My degree
    color_t offset = 0;

    for (; (offset <= degree + VBBIT_COLORING_FORBIDDEN_SIZE); offset += VBBIT_COLORING_FORBIDDEN_SIZE) {
      ban_type forbidden = 0;  // Forbidden colors
      for (Offset j = xadjbegin; j < my_xadj_end; ++j) {
        Ordinal n = _adj[j];
        if (n == i || n >= nv) continue;  // Skip self-loops
        color_t c            = _colors[n];
        color_t color_offset = c - offset;
        if (color_offset <= VBBIT_COLORING_FORBIDDEN_SIZE && c > offset) {
          ban_type ban_color_bit = 1;
          ban_color_bit          = ban_color_bit << (color_offset - 1);
          forbidden = forbidden | ban_color_bit;
          if (~forbidden == 0) break;
        }
      }
      forbidden = (~forbidden);
      if (forbidden) {
        ban_type my_new_color = forbidden & (-forbidden);
        color_t val = 1;
        while ((my_new_color & 1) == 0) {
          ++val;
          my_new_color = my_new_color >> 1;
        }
        _colors[i] = val + offset;
        break;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Port of functorFindConflicts_Atomic (COLORING_ATOMIC conflict scheme)
// ---------------------------------------------------------------------------
__global__ void find_conflicts_atomic(Ordinal nv, const Offset* _idx, const Ordinal* _adj,
                                      color_t* _colors, const Ordinal* _vertexList,
                                      Ordinal _vertexListLength, Ordinal* _recolorList,
                                      Ordinal* _recolorListLength)
{
  const Ordinal ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii >= _vertexListLength) return;
  Ordinal i      = _vertexList[ii];
  color_t my_color = _colors[i];
  Offset xadjend = _idx[i + 1];
  Offset j       = _idx[i];
  for (; j < xadjend; j++) {
    Ordinal neighbor = _adj[j];
    if (i < neighbor && neighbor < nv && _colors[neighbor] == my_color) {
      _colors[i] = 0;  // Uncolor vertex i
      Ordinal k = atomicAdd(_recolorListLength, 1);
      _recolorList[k] = i;
      break;  // Once i is uncolored and marked conflict
    }
  }
}

int main(int argc, char** argv)
{
  Ordinal nv = 200000;   // graph size (see header; upstream reads an .mtx file)
  int nnzPerRow = 30;
  int repeat = 1;        // upstream --repeat default
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--nv") && i + 1 < argc) nv = atoi(argv[++i]);
    if (!strcmp(argv[i], "--nnz") && i + 1 < argc) nnzPerRow = atoi(argv[++i]);
    if (!strcmp(argv[i], "--repeat") && i + 1 < argc) repeat = atoi(argv[++i]);
  }

  std::vector<Offset> xadj; std::vector<Ordinal> adj;
  generate_symmetric_graph(nv, nnzPerRow, xadj, adj);
  const Offset ne = xadj[nv];
  printf("Graph: %d vertices, %zu directed edges (symmetrized)\n", nv, (size_t)ne);

  Offset* d_xadj; Ordinal* d_adj; color_t* d_colors;
  Ordinal *d_listA, *d_listB, *d_len;
  GPU_CHECK(cudaMalloc(&d_xadj, (nv + 1) * sizeof(Offset)));
  GPU_CHECK(cudaMalloc(&d_adj, ne * sizeof(Ordinal)));
  GPU_CHECK(cudaMalloc(&d_colors, nv * sizeof(color_t)));
  GPU_CHECK(cudaMalloc(&d_listA, nv * sizeof(Ordinal)));
  GPU_CHECK(cudaMalloc(&d_listB, nv * sizeof(Ordinal)));
  GPU_CHECK(cudaMalloc(&d_len, sizeof(Ordinal)));
  GPU_CHECK(cudaMemcpy(d_xadj, xadj.data(), (nv + 1) * sizeof(Offset), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_adj, adj.data(), ne * sizeof(Ordinal), cudaMemcpyHostToDevice));

  std::vector<color_t> h_colors(nv);
  double total_time = 0;
  int iters_used = 0, num_colors = 0;

  for (int rep = 0; rep < repeat; rep++) {
    GPU_CHECK(cudaMemset(d_colors, 0, nv * sizeof(color_t)));
    Ordinal* current = d_listA;
    Ordinal* next    = d_listB;
    Ordinal currentLength = nv;
    init_list<<<(nv + 255) / 256, 256>>>(nv, current);

    const int max_iterations = 200;   // upstream max_number_of_iterations
    Ordinal numUncolored = nv;
    auto t0 = std::chrono::steady_clock::now();
    int iter = 0;
    for (; (iter < max_iterations) && (numUncolored > 0); iter++) {
      // colorGreedy: chunk size 8, or 1 for short worklists (upstream rule)
      Ordinal chunk = 8;
      if (currentLength < 100 * chunk) chunk = 1;
      Ordinal work_items = currentLength / chunk + 1;
      greedy_color_implog<<<(work_items + 255) / 256, 256>>>(nv, d_xadj, d_adj, d_colors,
                                                             current, currentLength, chunk, work_items);
      GPU_CHECK(cudaMemset(d_len, 0, sizeof(Ordinal)));
      find_conflicts_atomic<<<(currentLength + 255) / 256, 256>>>(nv, d_xadj, d_adj, d_colors,
                                                                  current, currentLength, next, d_len);
      GPU_CHECK(cudaMemcpy(&numUncolored, d_len, sizeof(Ordinal), cudaMemcpyDeviceToHost));
      // worklist swap (upstream: skipped on the final allowed iteration)
      if (iter + 1 < max_iterations) {
        Ordinal* tmp = current; current = next; next = tmp;
        currentLength = numUncolored;
      }
    }
    GPU_CHECK(cudaGetLastError());
    GPU_CHECK(cudaDeviceSynchronize());

    // serial host fallback (upstream resolveConflicts), rarely triggered
    if (numUncolored > 0) {
      GPU_CHECK(cudaMemcpy(h_colors.data(), d_colors, nv * sizeof(color_t), cudaMemcpyDeviceToHost));
      std::vector<Ordinal> h_list(currentLength);
      GPU_CHECK(cudaMemcpy(h_list.data(), current, currentLength * sizeof(Ordinal), cudaMemcpyDeviceToHost));
      std::vector<color_t> forbidden(nv + 2, -1);
      std::vector<Ordinal> owner(nv + 2, -1);
      for (Ordinal k = 0; k < currentLength; k++) {
        Ordinal i = h_list[k];
        if (h_colors[i] > 0) continue;
        for (Offset j = xadj[i]; j < xadj[i + 1]; j++) {
          if (adj[j] == i) continue;
          owner[h_colors[adj[j]]] = i;
        }
        int c = 1;
        while (owner[c] == i) c++;
        h_colors[i] = c;
      }
      GPU_CHECK(cudaMemcpy(d_colors, h_colors.data(), nv * sizeof(color_t), cudaMemcpyHostToDevice));
    }
    total_time += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    iters_used = iter;
  }

  GPU_CHECK(cudaMemcpy(h_colors.data(), d_colors, nv * sizeof(color_t), cudaMemcpyDeviceToHost));
  num_colors = *std::max_element(h_colors.begin(), h_colors.end());
  printf("Time: %f Num colors: %d Num Phases: %d\n", total_time / repeat, num_colors, iters_used);

  // ---- validation (added): proper-coloring check on the host ----
  Ordinal uncolored = 0; long long conflicts = 0;
  for (Ordinal i = 0; i < nv; i++) {
    if (h_colors[i] <= 0) { uncolored++; continue; }
    for (Offset j = xadj[i]; j < xadj[i + 1]; j++) {
      Ordinal n = adj[j];
      if (n != i && n < nv && h_colors[n] == h_colors[i]) conflicts++;
    }
  }
  bool ok = (uncolored == 0) && (conflicts == 0);
  if (!ok)
    printf("uncolored=%d conflict-edge-endpoints=%lld\n", uncolored, conflicts);
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
