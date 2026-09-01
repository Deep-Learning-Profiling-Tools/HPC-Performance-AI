//
// MAT_MAT_SHARED benchmark, extracted standalone from the RAJA Performance
// Suite (src/basic/MAT_MAT_SHARED.*, BSD-3-Clause). Shared-memory-tiled
// matrix multiply. GPU kernel is the upstream Base_CUDA variant; the CPU
// reference is the upstream Base_Seq variant, with upstream default problem
// size, reps, and data initialization.
// Validation: element-wise comparison of C.
//
#include "rp_common.hpp"

constexpr Index_type TL_SZ = 16;  // upstream tile size

// upstream MAT_MAT_SHARED.hpp
#define MAT_MAT_SHARED_BODY_0_CPU(tile_size)                                   \
  Real_type As[tile_size][tile_size];                                          \
  Real_type Bs[tile_size][tile_size];                                          \
  Real_type Cs[tile_size][tile_size];

#define MAT_MAT_SHARED_BODY_0(tile_size)                                       \
  __shared__ Real_type As[tile_size][tile_size];                               \
  __shared__ Real_type Bs[tile_size][tile_size];                               \
  __shared__ Real_type Cs[tile_size][tile_size];

#define MAT_MAT_SHARED_BODY_1(tile_size)                                       \
  Cs[ty][tx] = 0;

#define MAT_MAT_SHARED_BODY_2(tile_size)                                       \
  const Index_type Row = by * tile_size + ty;                                  \
  const Index_type Col = bx * tile_size + tx;                                  \
  if (k * tile_size + tx < N && Row < N)                                       \
    As[ty][tx] = A[Row * N + k * tile_size + tx];                              \
  else                                                                         \
    As[ty][tx] = 0.0;                                                          \
  if (k * tile_size + ty < N && Col < N)                                       \
    Bs[ty][tx] = B[(k * tile_size + ty) * N + Col];                            \
  else                                                                         \
    Bs[ty][tx] = 0.0;

#define MAT_MAT_SHARED_BODY_3(tile_size)                                       \
  for (Index_type n = 0; n < tile_size; ++n)                                   \
    Cs[ty][tx] += As[ty][n] * Bs[n][tx];

#define MAT_MAT_SHARED_BODY_4(tile_size)                                       \
  const Index_type Row = by * tile_size + ty;                                  \
  const Index_type Col = bx * tile_size + tx;                                  \
  if (Row < N && Col < N)                                                      \
    C[Col + N * Row] = Cs[ty][tx];

// upstream MAT_MAT_SHARED-Cuda.cpp (Base_CUDA kernel)
__launch_bounds__(TL_SZ*TL_SZ)
__global__ void mat_mat_shared(Index_type N, Real_ptr C, Real_ptr A,
                               Real_ptr B) {
  Index_type tx = threadIdx.x;
  Index_type ty = threadIdx.y;
  Index_type bx = blockIdx.x;
  Index_type by = blockIdx.y;

  MAT_MAT_SHARED_BODY_0(TL_SZ)
  MAT_MAT_SHARED_BODY_1(TL_SZ)
  for (Index_type k = 0; k < (TL_SZ + N - 1) / TL_SZ; k++) {
    MAT_MAT_SHARED_BODY_2(TL_SZ)
    __syncthreads();
    MAT_MAT_SHARED_BODY_3(TL_SZ)
    __syncthreads();
  }
  MAT_MAT_SHARED_BODY_4(TL_SZ)
}

int main(int argc, char** argv)
{
  Index_type N = 1000;      // upstream m_N_default
  Index_type run_reps = 5;  // upstream default reps
  if (argc > 1) N = atol(argv[1]);
  if (argc > 2) run_reps = atol(argv[2]);
  const Index_type len = N * N;
  const size_t bytes = len * sizeof(Real_type);

  // ------------------------- GPU (upstream Base_CUDA) ----------------------
  resetDataInitCount();
  Real_ptr A; Real_ptr B; Real_ptr C;
  allocAndInitDataConst(A, len, 1.0);   // upstream setUp order
  allocAndInitDataConst(B, len, 1.0);
  allocAndInitDataConst(C, len, 0.0);

  Real_ptr dA, dB, dC;
  GPU_CHECK(cudaMalloc(&dA, bytes));
  GPU_CHECK(cudaMalloc(&dB, bytes));
  GPU_CHECK(cudaMalloc(&dC, bytes));
  GPU_CHECK(cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dC, C, bytes, cudaMemcpyHostToDevice));

  {
    dim3 blockDim(TL_SZ, TL_SZ, 1);
    dim3 gridDim((size_t)RP_DIVIDE_CEILING_INT(N, TL_SZ),
                 (size_t)RP_DIVIDE_CEILING_INT(N, TL_SZ), 1);
    for (Index_type irep = 0; irep < run_reps; ++irep) {
      mat_mat_shared<<<gridDim, blockDim>>>(N, dC, dA, dB);
    }
  }
  GPU_CHECK(cudaGetLastError());
  GPU_CHECK(cudaDeviceSynchronize());
  GPU_CHECK(cudaMemcpy(C, dC, bytes, cudaMemcpyDeviceToHost));
  cudaFree(dA); cudaFree(dB); cudaFree(dC);

  // ------------------------ CPU (upstream Base_Seq) ------------------------
  resetDataInitCount();
  Real_ptr A_r; Real_ptr B_r; Real_ptr C_r;
  allocAndInitDataConst(A_r, len, 1.0);
  allocAndInitDataConst(B_r, len, 1.0);
  allocAndInitDataConst(C_r, len, 0.0);
  {
    Real_ptr A = A_r; Real_ptr B = B_r; Real_ptr C = C_r;
    const Index_type Nx = RP_DIVIDE_CEILING_INT(N, TL_SZ);
    const Index_type Ny = RP_DIVIDE_CEILING_INT(N, TL_SZ);
    for (Index_type irep = 0; irep < run_reps; ++irep) {
      for (Index_type by = 0; by < Ny; ++by) {
        for (Index_type bx = 0; bx < Nx; ++bx) {
          MAT_MAT_SHARED_BODY_0_CPU(TL_SZ)
          for (Index_type ty = 0; ty < TL_SZ; ++ty) {
            for (Index_type tx = 0; tx < TL_SZ; ++tx) {
              MAT_MAT_SHARED_BODY_1(TL_SZ)
            }
          }
          for (Index_type k = 0; k < (TL_SZ + N - 1) / TL_SZ; ++k) {
            for (Index_type ty = 0; ty < TL_SZ; ++ty) {
              for (Index_type tx = 0; tx < TL_SZ; ++tx) {
                MAT_MAT_SHARED_BODY_2(TL_SZ)
              }
            }
            for (Index_type ty = 0; ty < TL_SZ; ++ty) {
              for (Index_type tx = 0; tx < TL_SZ; ++tx) {
                MAT_MAT_SHARED_BODY_3(TL_SZ)
              }
            }
          }
          for (Index_type ty = 0; ty < TL_SZ; ++ty) {
            for (Index_type tx = 0; tx < TL_SZ; ++tx) {
              MAT_MAT_SHARED_BODY_4(TL_SZ)
            }
          }
        }
      }
    }
  }

  // ------------------------------ validate ---------------------------------
  bool ok = compareArrays("C", C_r, C, len, 1.0e-10);
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
