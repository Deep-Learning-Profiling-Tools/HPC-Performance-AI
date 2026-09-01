//
// SpAdd (C = A + B) benchmark, faithful standalone port of the Kokkos Kernels
// perf_test (perf_test/sparse/KokkosSparse_spadd.cpp, sorted path default,
// BSD-3-Clause; commit 23a699f3c5bd662bf2ed52116e56533ecc3ddae0).
//
// What is ported and from where:
//  - Symbolic count kernel: SortedCountEntriesTeam (+ sequential long-row
//    fallback SortedCountEntriesRange) from
//    sparse/impl/KokkosSparse_spadd_symbolic_impl.hpp, including the
//    launch heuristics of runSortedCountEntries (c_est_nnz, pot_est_nnz,
//    vector-length selection; one row per warp, bitonic merge in shared mem).
//  - Numeric kernel: SortedNumericSumFunctor (flat range policy, one thread
//    per row, two-way sorted merge, alpha = beta = 1) from
//    sparse/impl/KokkosSparse_spadd_numeric_impl.hpp.
//  - Driver semantics: m = n = 10000, nnzPerRow = 30, A and B generated with
//    kk_sparseMatrix_generate (glibc srand(13721) is reset per call, so A and
//    B have identical sparsity structure -- upstream behavior), rows sorted
//    before the run (sorted algorithm), repeat = 1 (symbolic + numeric).
//
// Documented deviations from upstream (packaging only, semantics preserved):
//  - Matrix values: upstream fills them with a Kokkos XorShift64 pool
//    (seed 13718, uniform(-50,50)); reproducing that pool bit-exactly requires
//    Kokkos internals, so a fixed-seed xorshift64* generator with the same
//    distribution is used instead.
//  - Kokkos parallel_scan (rowmap prefix sum) is replaced by an equivalent
//    hierarchical exclusive scan; Kokkos' occupancy-recommended team size is
//    fixed at 8 warps/block (256 threads).
//  - Input rows are sorted on the host during (untimed) setup; upstream sorts
//    on device with KokkosSparse::sort_crs_matrix, also untimed.
//
// Validation (added; upstream perf_test has no check): C is recomputed on the
// CPU with the same merge order and compared exactly (structure) and
// bitwise-order-identical accumulation (values, rel tol 1e-13).
//
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>

using Scalar  = double;
using Ordinal = int;
using Offset  = size_t;

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
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaGetLastError hipGetLastError
#define SYNCWARP()
#else
#include <cuda_runtime.h>
#define SYNCWARP() __syncwarp()
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

static const Ordinal ORDINAL_MAX = std::numeric_limits<Ordinal>::max();

// ---------------------------------------------------------------------------
// Port of kk_sparseMatrix_generate (sparse/src/KokkosSparse_IOUtils.hpp)
// ---------------------------------------------------------------------------
static void kk_sparseMatrix_generate(Ordinal nrows, Ordinal ncols, Offset& nnz,
                                     Ordinal row_size_variance, Ordinal bandwidth,
                                     std::vector<Scalar>& values, std::vector<Offset>& rowPtr,
                                     std::vector<Ordinal>& colInd)
{
  rowPtr.assign(nrows + 1, 0);
  Ordinal elements_per_row = nrows ? static_cast<Ordinal>(nnz / nrows) : 0;
  srand(13721);
  for (int row = 0; row < nrows; row++) {
    int varianz       = (1.0 * rand() / RAND_MAX - 0.5) * row_size_variance;
    int numRowEntries = elements_per_row + varianz;
    if (numRowEntries < 0) numRowEntries = 0;
    if (numRowEntries > 0.66 * ncols) numRowEntries = 0.66 * ncols;
    rowPtr[row + 1] = rowPtr[row] + numRowEntries;
  }
  nnz = rowPtr[nrows];
  values.resize(nnz);
  colInd.resize(nnz);
  for (Ordinal row = 0; row < nrows; row++) {
    for (Offset k = rowPtr[row]; k < rowPtr[row + 1]; ++k) {
      while (true) {
        Ordinal pos = (1.0 * rand() / RAND_MAX - 0.5) * bandwidth + row;
        while (pos < 0) pos += ncols;
        while (pos >= ncols) pos -= ncols;
        bool dup = false;
        for (Offset j = rowPtr[row]; j < k; j++)
          if (colInd[j] == pos) { dup = true; break; }
        if (!dup) { colInd[k] = pos; break; }
      }
    }
  }
  // values: uniform(-50,50), fixed-seed xorshift64* (see header comment)
  static uint64_t s = 0x9E3779B97F4A7C15ull;
  for (Offset k = 0; k < nnz; k++) {
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    uint64_t r = s * 0x2545F4914F6CDD1Dull;
    values[k] = -50.0 + 100.0 * ((r >> 11) * (1.0 / 9007199254740992.0));
  }
}

// ---------------------------------------------------------------------------
// GPU symbolic: port of SortedCountEntriesTeam (one row per warp, bitonic
// merge of the two sorted rows in shared memory, count distinct entries).
// blockDim = (vector_length, team_size). scratch: pot_est_nnz ints per warp.
// ---------------------------------------------------------------------------
extern __shared__ int spadd_scratch[];

__device__ void longRowFallback(Ordinal i, const Offset* Arowptrs, const Ordinal* Acolinds,
                                const Offset* Browptrs, const Ordinal* Bcolinds, Offset* Crowcounts)
{
  Offset numEntries = 0, ai = 0, bi = 0;
  Offset Arowstart = Arowptrs[i], Arowlen = Arowptrs[i + 1] - Arowstart;
  Offset Browstart = Browptrs[i], Browlen = Browptrs[i + 1] - Browstart;
  Ordinal Acol = (Arowlen == 0) ? ORDINAL_MAX : Acolinds[Arowstart];
  Ordinal Bcol = (Browlen == 0) ? ORDINAL_MAX : Bcolinds[Browstart];
  while (Acol != ORDINAL_MAX || Bcol != ORDINAL_MAX) {
    Ordinal Ccol = (Acol < Bcol) ? Acol : Bcol;
    numEntries++;
    while (Acol == Ccol) { ai++; Acol = (ai >= Arowlen) ? ORDINAL_MAX : Acolinds[Arowstart + ai]; }
    while (Bcol == Ccol) { bi++; Bcol = (bi >= Browlen) ? ORDINAL_MAX : Bcolinds[Browstart + bi]; }
  }
  Crowcounts[i] = numEntries;
}

__global__ void sorted_count_entries_team(Ordinal nrows, const Offset* Arowptrs,
                                          const Ordinal* Acolinds, const Offset* Browptrs,
                                          const Ordinal* Bcolinds, Offset* Crowcounts,
                                          int sharedPerThread)
{
  const Ordinal i = blockIdx.x * blockDim.y + threadIdx.y;
  if (i >= nrows) return;
  int* scratch = spadd_scratch + threadIdx.y * sharedPerThread;
  const Ordinal Arowstart = Arowptrs[i];
  const Ordinal Arowlen   = Arowptrs[i + 1] - Arowstart;
  const Ordinal Browstart = Browptrs[i];
  const Ordinal Browlen   = Browptrs[i + 1] - Browstart;
  const Ordinal n = Arowlen + Browlen;
  if (n > sharedPerThread) {
    if (threadIdx.x == 0)
      longRowFallback(i, Arowptrs, Acolinds, Browptrs, Bcolinds, Crowcounts);
    return;
  }
  if (n == 0) {
    if (threadIdx.x == 0) Crowcounts[i] = 0;
    return;
  }
  Ordinal npot = 1, levels = 0;
  while (npot < n) { levels++; npot <<= 1; }
  const int vlen = blockDim.x;
  // load A ascending, B descending (bitonic sequence), pad middle with MAX
  for (Ordinal j = threadIdx.x; j < Arowlen; j += vlen) scratch[j] = Acolinds[Arowstart + j];
  for (Ordinal j = threadIdx.x; j < Browlen; j += vlen) scratch[npot - 1 - j] = Bcolinds[Browstart + j];
  for (Ordinal j = threadIdx.x; j < npot - n; j += vlen) scratch[Arowlen + j] = ORDINAL_MAX;
  SYNCWARP();
  for (Ordinal level = 0; level < levels; level++) {
    for (Ordinal j = threadIdx.x; j < (npot >> 1); j += vlen) {
      Ordinal boxSize   = npot >> level;
      Ordinal boxID     = (j * 2) >> (levels - level);
      Ordinal boxStart  = boxID << (levels - level);
      Ordinal boxOffset = j - boxID * boxSize / 2;
      Ordinal elem1     = boxStart + boxOffset;
      Ordinal elem2     = elem1 + (boxSize >> 1);
      if (scratch[elem2] < scratch[elem1]) {
        Ordinal tmp = scratch[elem1]; scratch[elem1] = scratch[elem2]; scratch[elem2] = tmp;
      }
    }
    SYNCWARP();
  }
  // count rising edges (distinct entries) across vector lanes
  Ordinal local = 0;
  for (Ordinal j = threadIdx.x; j < n - 1; j += vlen)
    if (scratch[j] != scratch[j + 1]) local++;
#if defined(__HIPCC__)
  for (int off = vlen / 2; off > 0; off /= 2) local += __shfl_down(local, off, vlen);
#else
  for (int off = vlen / 2; off > 0; off /= 2) local += __shfl_down_sync(0xffffffffu, local, off, vlen);
#endif
  if (threadIdx.x == 0) Crowcounts[i] = local + 1;
}

// Hierarchical exclusive scan over nrows+1 entries (replaces Kokkos
// parallel_scan; single-block loop is sufficient at this problem size).
__global__ void exclusive_scan_single(Offset* data, Ordinal n)
{
  __shared__ Offset carry;
  __shared__ Offset buf[1024];
  if (threadIdx.x == 0) carry = 0;
  __syncthreads();
  for (Ordinal base = 0; base < n; base += 1024) {
    Ordinal idx = base + threadIdx.x;
    Offset v = (idx < n) ? data[idx] : 0;
    buf[threadIdx.x] = v;
    __syncthreads();
    // Hillis-Steele inclusive scan on the chunk
    for (int d = 1; d < 1024; d <<= 1) {
      Offset add = (threadIdx.x >= d) ? buf[threadIdx.x - d] : 0;
      __syncthreads();
      buf[threadIdx.x] += add;
      __syncthreads();
    }
    if (idx < n) data[idx] = carry + buf[threadIdx.x] - v;  // exclusive
    __syncthreads();
    if (threadIdx.x == 0) carry += buf[1023];
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// GPU numeric: port of SortedNumericSumFunctor (one thread per row).
// ---------------------------------------------------------------------------
__global__ void sorted_numeric_sum(Ordinal nrows, const Offset* Arowptrs, const Offset* Browptrs,
                                   const Offset* Crowptrs, const Ordinal* Acolinds,
                                   const Ordinal* Bcolinds, Ordinal* Ccolinds,
                                   const Scalar* Avalues, const Scalar* Bvalues, Scalar* Cvalues,
                                   const Scalar alpha, const Scalar beta)
{
  const Ordinal i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nrows) return;
  Offset ai = 0, bi = 0;
  Offset Arowstart = Arowptrs[i], Arowlen = Arowptrs[i + 1] - Arowstart;
  Offset Browstart = Browptrs[i], Browlen = Browptrs[i + 1] - Browstart;
  Ordinal Acol = (Arowlen == 0) ? ORDINAL_MAX : Acolinds[Arowstart];
  Ordinal Bcol = (Browlen == 0) ? ORDINAL_MAX : Bcolinds[Browstart];
  Offset Coffset = Crowptrs[i];
  while (Acol != ORDINAL_MAX || Bcol != ORDINAL_MAX) {
    Ordinal Ccol = (Acol < Bcol) ? Acol : Bcol;
    Scalar accum = 0;
    while (Acol == Ccol) {
      accum += alpha * Avalues[Arowstart + ai];
      ai++;
      Acol = (ai == Arowlen) ? ORDINAL_MAX : Acolinds[Arowstart + ai];
    }
    while (Bcol == Ccol) {
      accum += beta * Bvalues[Browstart + bi];
      bi++;
      Bcol = (bi == Browlen) ? ORDINAL_MAX : Bcolinds[Browstart + bi];
    }
    Ccolinds[Coffset] = Ccol;
    Cvalues[Coffset]  = accum;
    Coffset++;
  }
}

int main(int argc, char** argv)
{
  Ordinal m = 10000, n = 10000;   // upstream defaults
  int nnzPerRow = 30;
  int repeat = 1;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--m") && i + 1 < argc) m = atoi(argv[++i]);
    if (!strcmp(argv[i], "--n") && i + 1 < argc) n = atoi(argv[++i]);
    if (!strcmp(argv[i], "--nnz") && i + 1 < argc) nnzPerRow = atoi(argv[++i]);
    if (!strcmp(argv[i], "--repeat") && i + 1 < argc) repeat = atoi(argv[++i]);
  }

  // ----- generate A and B (upstream: srand reset per call -> same structure)
  std::vector<Scalar> Aval, Bval; std::vector<Offset> Arow, Brow; std::vector<Ordinal> Acol, Bcol;
  Offset nnzA = (Offset)m * nnzPerRow, nnzB = (Offset)m * nnzPerRow;
  kk_sparseMatrix_generate(m, n, nnzA, 0, (n + 3) / 3, Aval, Arow, Acol);
  kk_sparseMatrix_generate(m, n, nnzB, 0, (n + 3) / 3, Bval, Brow, Bcol);

  // ----- sorted algorithm: sort rows (untimed setup; upstream sort_crs_matrix)
  auto sort_rows = [](Ordinal rows, std::vector<Offset>& rp, std::vector<Ordinal>& ci, std::vector<Scalar>& v) {
    std::vector<std::pair<Ordinal, Scalar>> tmp;
    for (Ordinal i = 0; i < rows; i++) {
      tmp.clear();
      for (Offset k = rp[i]; k < rp[i + 1]; k++) tmp.push_back({ci[k], v[k]});
      std::sort(tmp.begin(), tmp.end());
      Offset k = rp[i];
      for (auto& p : tmp) { ci[k] = p.first; v[k] = p.second; k++; }
    }
  };
  sort_rows(m, Arow, Acol, Aval);
  sort_rows(m, Brow, Bcol, Bval);
  printf("A and B are %dx%d. A, B have %zu and %zu entries.\n", m, n, (size_t)nnzA, (size_t)nnzB);

  // ----- device data -----
  Offset *dArow, *dBrow, *dCrow; Ordinal *dAcol, *dBcol; Scalar *dAval, *dBval;
  GPU_CHECK(cudaMalloc(&dArow, (m + 1) * sizeof(Offset)));
  GPU_CHECK(cudaMalloc(&dBrow, (m + 1) * sizeof(Offset)));
  GPU_CHECK(cudaMalloc(&dCrow, (m + 1) * sizeof(Offset)));
  GPU_CHECK(cudaMalloc(&dAcol, nnzA * sizeof(Ordinal)));
  GPU_CHECK(cudaMalloc(&dBcol, nnzB * sizeof(Ordinal)));
  GPU_CHECK(cudaMalloc(&dAval, nnzA * sizeof(Scalar)));
  GPU_CHECK(cudaMalloc(&dBval, nnzB * sizeof(Scalar)));
  GPU_CHECK(cudaMemcpy(dArow, Arow.data(), (m + 1) * sizeof(Offset), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dBrow, Brow.data(), (m + 1) * sizeof(Offset), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dAcol, Acol.data(), nnzA * sizeof(Ordinal), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dBcol, Bcol.data(), nnzB * sizeof(Ordinal), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dAval, Aval.data(), nnzA * sizeof(Scalar), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dBval, Bval.data(), nnzB * sizeof(Scalar), cudaMemcpyHostToDevice));

  // ----- upstream runSortedCountEntries launch heuristics -----
  Offset c_est_nnz = 1.4 * (nnzA + nnzB) / m;
  int pot_est_nnz = 1;
  while ((Offset)pot_est_nnz < c_est_nnz) pot_est_nnz *= 2;
  int vector_length = 1;
  while (vector_length * 2 <= 32 && (Offset)vector_length * 2 <= (Offset)pot_est_nnz) vector_length *= 2;
  const int team_size = 256 / vector_length >= 8 ? 8 : 256 / vector_length;  // fixed 8 warps (see header)
  const bool use_team = c_est_nnz <= 512;

  Ordinal* dCcol = nullptr; Scalar* dCval = nullptr;
  Offset nnzC = 0;
  double symbolicTime = 0, numericTime = 0;

  for (int rep = 0; rep < repeat; rep++) {
    // -------- symbolic --------
    auto t0 = std::chrono::steady_clock::now();
    if (use_team) {
      dim3 block(vector_length, team_size);
      int league = (m + team_size - 1) / team_size;
      size_t shmem = (size_t)pot_est_nnz * team_size * sizeof(int);
      sorted_count_entries_team<<<league, block, shmem>>>(m, dArow, dAcol, dBrow, dBcol, dCrow, pot_est_nnz);
    }
    exclusive_scan_single<<<1, 1024>>>(dCrow, m + 1);
    GPU_CHECK(cudaGetLastError());
    GPU_CHECK(cudaDeviceSynchronize());
    symbolicTime += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    GPU_CHECK(cudaMemcpy(&nnzC, dCrow + m, sizeof(Offset), cudaMemcpyDeviceToHost));
    if (!dCcol) {
      GPU_CHECK(cudaMalloc(&dCcol, nnzC * sizeof(Ordinal)));
      GPU_CHECK(cudaMalloc(&dCval, nnzC * sizeof(Scalar)));
    }

    // -------- numeric --------
    t0 = std::chrono::steady_clock::now();
    sorted_numeric_sum<<<(m + 255) / 256, 256>>>(m, dArow, dBrow, dCrow, dAcol, dBcol, dCcol,
                                                 dAval, dBval, dCval, 1.0, 1.0);
    GPU_CHECK(cudaGetLastError());
    GPU_CHECK(cudaDeviceSynchronize());
    numericTime += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }

  printf("Mean total time:    %f\n", (symbolicTime + numericTime) / repeat);
  printf("Mean symbolic time: %f\n", symbolicTime / repeat);
  printf("Mean numeric time:  %f\n", numericTime / repeat);
  printf("C has %zu entries.\n", (size_t)nnzC);

  // ----- validation (added): CPU reference with identical merge order -----
  std::vector<Offset> Crow_ref(m + 1, 0);
  std::vector<Ordinal> Ccol_ref; std::vector<Scalar> Cval_ref;
  for (Ordinal i = 0; i < m; i++) {
    Offset ai = 0, bi = 0;
    Offset as = Arow[i], al = Arow[i + 1] - as, bs = Brow[i], bl = Brow[i + 1] - bs;
    Ordinal Ac = (al == 0) ? ORDINAL_MAX : Acol[as];
    Ordinal Bc = (bl == 0) ? ORDINAL_MAX : Bcol[bs];
    while (Ac != ORDINAL_MAX || Bc != ORDINAL_MAX) {
      Ordinal Cc = (Ac < Bc) ? Ac : Bc;
      Scalar accum = 0;
      while (Ac == Cc) { accum += Aval[as + ai]; ai++; Ac = (ai == al) ? ORDINAL_MAX : Acol[as + ai]; }
      while (Bc == Cc) { accum += Bval[bs + bi]; bi++; Bc = (bi == bl) ? ORDINAL_MAX : Bcol[bs + bi]; }
      Ccol_ref.push_back(Cc); Cval_ref.push_back(accum);
    }
    Crow_ref[i + 1] = Ccol_ref.size();
  }

  std::vector<Offset> Crow(m + 1); std::vector<Ordinal> Ccol(nnzC); std::vector<Scalar> Cval(nnzC);
  GPU_CHECK(cudaMemcpy(Crow.data(), dCrow, (m + 1) * sizeof(Offset), cudaMemcpyDeviceToHost));
  GPU_CHECK(cudaMemcpy(Ccol.data(), dCcol, nnzC * sizeof(Ordinal), cudaMemcpyDeviceToHost));
  GPU_CHECK(cudaMemcpy(Cval.data(), dCval, nnzC * sizeof(Scalar), cudaMemcpyDeviceToHost));

  bool ok = (Crow_ref == Crow) && ((Offset)Ccol_ref.size() == nnzC);
  if (ok) {
    for (Offset k = 0; k < nnzC && ok; k++) {
      if (Ccol[k] != Ccol_ref[k]) ok = false;
      else if (std::fabs(Cval[k] - Cval_ref[k]) > 1e-13 * (1.0 + std::fabs(Cval_ref[k]))) ok = false;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
