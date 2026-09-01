//
// SpMV benchmark, faithful standalone port of the Kokkos Kernels perf_test
// (perf_test/sparse/KokkosSparse_spmv.cpp, test=kk default, BSD-3-Clause;
// commit 23a699f3c5bd662bf2ed52116e56533ecc3ddae0).
//
// What is ported and from where:
//  - GPU kernel + launch heuristics: SPMV_Functor / launch_parameters from
//    perf_test/sparse/spmv/Kokkos_SPMV.hpp (team-based CSR SpMV; team ->
//    thread block, TeamThreadRange -> threadIdx.y, ThreadVectorRange ->
//    threadIdx.x lanes with a sub-warp shuffle reduction).
//  - Matrix generation: kk_sparseMatrix_generate from
//    sparse/src/KokkosSparse_IOUtils.hpp (glibc srand/rand, seed 13721),
//    with the driver defaults: numRows = numCols = 110503, nnz = 10*numRows,
//    row_size_variance = 0, bandwidth = (int)(0.01*numRows).
//  - x/y initialization, gold standard, and error check: setup_test /
//    generate_gold_standard / check_errors from KokkosSparse_spmv_test.{hpp,cpp}.
//    Note: upstream overwrites the matrix values with colidx + row before
//    running (generate_gold_standard), so the XorShift64 random fill of the
//    values never reaches the benchmark and is omitted here.
//
// Types are the upstream defaults: Scalar=double, Ordinal=int, Offset=size_t.
// Validation (upstream semantics): relative squared error of y vs the
// sequential CPU gold standard must be <= 1e-5.
//
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>

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
__device__ inline Scalar shfl_down_width(Scalar v, int delta, int width) {
  return __shfl_down(v, delta, width);
}
#else
#include <cuda_runtime.h>
__device__ inline Scalar shfl_down_width(Scalar v, int delta, int width) {
  return __shfl_down_sync(0xffffffffu, v, delta, width);
}
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
// Port of SPMV_Functor (Kokkos_SPMV.hpp), alpha=1, beta=0 (dobeta==0).
// blockDim.x = vector_length (vector lanes), blockDim.y = team_size.
// ---------------------------------------------------------------------------
__global__ void spmv_kernel(const Ordinal numRows, const Offset* __restrict__ row_map,
                            const Ordinal* __restrict__ entries,
                            const Scalar* __restrict__ values,
                            const Scalar* __restrict__ x, Scalar* __restrict__ y,
                            const Ordinal rows_per_team)
{
  const int team_size = blockDim.y;
  const int vlen      = blockDim.x;
  // Kokkos::TeamThreadRange(dev, 0, rows_per_team)
  for (Ordinal loop = threadIdx.y; loop < rows_per_team; loop += team_size) {
    const Ordinal iRow = static_cast<Ordinal>(blockIdx.x) * rows_per_team + loop;
    if (iRow >= numRows) continue;
    const Offset  row_start  = row_map[iRow];
    const Ordinal row_length = static_cast<Ordinal>(row_map[iRow + 1] - row_start);
    // Kokkos::ThreadVectorRange reduce over the row entries
    Scalar sum = 0;
    for (Ordinal iEntry = threadIdx.x; iEntry < row_length; iEntry += vlen) {
      sum += values[row_start + iEntry] * x[entries[row_start + iEntry]];
    }
    for (int offset = vlen / 2; offset > 0; offset /= 2) {
      sum += shfl_down_width(sum, offset, vlen);
    }
    // Kokkos::single(PerThread): vector lane 0 writes
    if (threadIdx.x == 0) {
      y[iRow] = sum;  // alpha = 1, beta = 0
    }
  }
}

// ---------------------------------------------------------------------------
// Port of launch_parameters<Kokkos::Cuda> (Kokkos_SPMV.hpp)
// ---------------------------------------------------------------------------
static int launch_parameters(Ordinal numRows, Offset nnz, int rows_per_thread,
                             int& team_size, int& vector_length)
{
  int nnz_per_row = static_cast<int>(nnz / numRows);
  if (nnz_per_row < 1) nnz_per_row = 1;
  if (vector_length < 1) {
    vector_length = 1;
    while (vector_length < 32 && vector_length * 6 < nnz_per_row) vector_length *= 2;
  }
  if (rows_per_thread < 1) rows_per_thread = 1;   // CUDA/GPU path
  if (team_size < 1) team_size = 256 / vector_length;
  return rows_per_thread * team_size;             // rows_per_team
}

// ---------------------------------------------------------------------------
// Port of kk_sparseMatrix_generate (sparse/src/KokkosSparse_IOUtils.hpp).
// Uses glibc srand/rand exactly like upstream, including the per-row rand()
// consumed by the (zero) row-size variance.
// ---------------------------------------------------------------------------
static void kk_sparseMatrix_generate(Ordinal nrows, Ordinal ncols, Offset& nnz,
                                     Ordinal row_size_variance, Ordinal bandwidth,
                                     Scalar*& values, Offset*& rowPtr, Ordinal*& colInd)
{
  rowPtr = new Offset[nrows + 1];
  Ordinal elements_per_row = nrows ? static_cast<Ordinal>(nnz / nrows) : 0;
  srand(13721);
  rowPtr[0] = 0;
  for (int row = 0; row < nrows; row++) {
    int varianz       = (1.0 * rand() / RAND_MAX - 0.5) * row_size_variance;
    int numRowEntries = elements_per_row + varianz;
    if (numRowEntries < 0) numRowEntries = 0;
    if (numRowEntries > 0.66 * ncols) numRowEntries = 0.66 * ncols;
    rowPtr[row + 1] = rowPtr[row] + numRowEntries;
  }
  nnz    = rowPtr[nrows];
  values = new Scalar[nnz];
  colInd = new Ordinal[nnz];
  for (Ordinal row = 0; row < nrows; row++) {
    for (Offset k = rowPtr[row]; k < rowPtr[row + 1]; ++k) {
      while (true) {
        Ordinal pos = (1.0 * rand() / RAND_MAX - 0.5) * bandwidth + row;
        while (pos < 0) pos += ncols;
        while (pos >= ncols) pos -= ncols;
        bool is_already_in_the_row = false;
        for (Offset j = rowPtr[row]; j < k; j++) {
          if (colInd[j] == pos) {
            is_already_in_the_row = true;
            break;
          }
        }
        if (!is_already_in_the_row) {
          colInd[k] = pos;
          break;
        }
      }
    }
  }
  // (upstream fills `values` with XorShift64 randoms here; they are
  // overwritten by generate_gold_standard before the benchmark runs)
}

int main(int argc, char** argv)
{
  long long int size = 110503;  // upstream default (a prime number)
  int loop           = 100;     // upstream default
  int vector_length  = -1;
  int team_size      = -1;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-s") && i + 1 < argc) size = atoll(argv[++i]);
    if (!strcmp(argv[i], "-l") && i + 1 < argc) loop = atoi(argv[++i]);
  }
  const Ordinal numRows = static_cast<Ordinal>(size);
  const Ordinal numCols = static_cast<Ordinal>(size);

  // ----- upstream driver: matrix generation -----
  srand(17312837);
  Offset nnz = 10 * static_cast<Offset>(numRows);
  Scalar* values; Offset* rowPtr; Ordinal* colInd;
  kk_sparseMatrix_generate(numRows, numCols, nnz, 0, 0.01 * numRows,
                           values, rowPtr, colInd);

  // ----- upstream setup_test: x/y init (continues the rand() stream) -----
  Scalar* h_x = new Scalar[numCols];
  Scalar* h_y = new Scalar[numRows];
  Scalar* h_y_compare = new Scalar[numRows];
  for (int i = 0; i < numCols; i++) h_x[i] = (Scalar)(1.0 * (rand() % 40) - 20.);
  for (int i = 0; i < numRows; i++) h_y[i] = (Scalar)(1.0 * (rand() % 40) - 20.);

  // ----- upstream generate_gold_standard: overwrite values, CPU gold -----
  for (int i = 0; i < numRows; i++) {
    Offset start = rowPtr[i], end = rowPtr[i + 1];
    for (Offset j = start; j < end; j++) values[j] = colInd[j] + i;
    h_y_compare[i] = 0;
    for (Offset j = start; j < end; j++) {
      Scalar tmp_val = colInd[j] + i;
      int idx = colInd[j];
      h_y_compare[i] += tmp_val * h_x[idx];
    }
  }

  // ----- device setup -----
  Offset* d_rowPtr; Ordinal* d_colInd; Scalar* d_values; Scalar* d_x; Scalar* d_y;
  GPU_CHECK(cudaMalloc(&d_rowPtr, (numRows + 1) * sizeof(Offset)));
  GPU_CHECK(cudaMalloc(&d_colInd, nnz * sizeof(Ordinal)));
  GPU_CHECK(cudaMalloc(&d_values, nnz * sizeof(Scalar)));
  GPU_CHECK(cudaMalloc(&d_x, numCols * sizeof(Scalar)));
  GPU_CHECK(cudaMalloc(&d_y, numRows * sizeof(Scalar)));
  GPU_CHECK(cudaMemcpy(d_rowPtr, rowPtr, (numRows + 1) * sizeof(Offset), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_colInd, colInd, nnz * sizeof(Ordinal), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_values, values, nnz * sizeof(Scalar), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_x, h_x, numCols * sizeof(Scalar), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_y, h_y, numRows * sizeof(Scalar), cudaMemcpyHostToDevice));

  const int rows_per_team = launch_parameters(numRows, nnz, -1, team_size, vector_length);
  const int worksets = (numRows + rows_per_team - 1) / rows_per_team;
  dim3 block(vector_length, team_size, 1);

  // ----- upstream setup_test: one matvec + error check -----
  spmv_kernel<<<worksets, block>>>(numRows, d_rowPtr, d_colInd, d_values, d_x, d_y, rows_per_team);
  GPU_CHECK(cudaGetLastError());
  GPU_CHECK(cudaDeviceSynchronize());
  GPU_CHECK(cudaMemcpy(h_y, d_y, numRows * sizeof(Scalar), cudaMemcpyDeviceToHost));

  int num_errors = 0;
  double total_error = 0.0;
  {  // upstream check_errors
    double error = 0.0, sum = 0.0;
    for (int i = 0; i < numRows; i++) {
      error += (h_y_compare[i] - h_y[i]) * (h_y_compare[i] - h_y[i]);
      sum += h_y_compare[i] * h_y_compare[i];
    }
    if (sum == 0.0) sum = 1.0;
    if ((error / sum) > 1e-5) ++num_errors;
    total_error += error;
  }

  // ----- upstream benchmark loop -----
  double ave_time = 0.0, max_time = 0.0, min_time = 1.0e32;
  for (int i = 0; i < loop; i++) {
    auto t0 = std::chrono::steady_clock::now();
    spmv_kernel<<<worksets, block>>>(numRows, d_rowPtr, d_colInd, d_values, d_x, d_y, rows_per_team);
    GPU_CHECK(cudaDeviceSynchronize());
    double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    ave_time += t;
    if (t > max_time) max_time = t;
    if (t < min_time) min_time = t;
  }
  GPU_CHECK(cudaGetLastError());

  // upstream-style performance summary
  double matrix_size = 1.0 * ((nnz * (sizeof(Scalar) + sizeof(Ordinal)) + numRows * sizeof(Offset))) / 1024 / 1024;
  double vector_readwrite = (numRows + numRows + numCols) * sizeof(Scalar) / 1024 / 1024;
  printf("NNZ NumRows NumCols ProblemSize(MB) AveBandwidth(GB/s) AveGFlop aveTime(ms) numErrors\n");
  printf("%zd %d %d %6.2lf %6.2lf %6.3lf %6.3lf %d\n",
         (size_t)nnz, numRows, numCols, matrix_size,
         (matrix_size + vector_readwrite) / ave_time * loop / 1024,
         2.0 * nnz * loop / ave_time / 1e9, ave_time / loop * 1000, num_errors);
  printf("launch config: vector_length=%d team_size=%d rows_per_team=%d worksets=%d\n",
         vector_length, team_size, rows_per_team, worksets);
  printf("%s\n", num_errors == 0 ? "PASS" : "FAIL");
  return num_errors == 0 ? 0 : 1;
}
