//
// POLYBENCH_JACOBI_2D benchmark, extracted standalone from the RAJA
// Performance Suite (src/polybench/POLYBENCH_JACOBI_2D.*, BSD-3-Clause).
// GPU kernels are the upstream Base_CUDA variant; the CPU reference is the
// upstream Base_Seq variant, with upstream default problem size, reps, and
// data initialization. Validation: element-wise comparison of A and B.
//
#include "rp_common.hpp"

// upstream POLYBENCH_JACOBI_2D.hpp
#define POLYBENCH_JACOBI_2D_BODY1 \
  B[j + i*N] = 0.2 * (A[j + i*N] + A[j-1 + i*N] + A[j+1 + i*N] + A[j + (i+1)*N] + A[j + (i-1)*N]);

#define POLYBENCH_JACOBI_2D_BODY2 \
  A[j + i*N] = 0.2 * (B[j + i*N] + B[j-1 + i*N] + B[j+1 + i*N] + B[j + (i+1)*N] + B[j + (i-1)*N]);

// upstream POLYBENCH_JACOBI_2D-Cuda.cpp launch configuration
constexpr size_t block_size = 256;   // upstream default_gpu_block_size
constexpr size_t j_block_sz = 32;
constexpr size_t i_block_sz = block_size / j_block_sz;

__launch_bounds__(j_block_sz*i_block_sz)
__global__ void poly_jacobi_2D_1(Real_ptr A, Real_ptr B, Index_type N)
{
  Index_type i = 1 + blockIdx.y * i_block_sz + threadIdx.y;
  Index_type j = 1 + blockIdx.x * j_block_sz + threadIdx.x;
  if ( i < N-1 && j < N-1 ) {
    POLYBENCH_JACOBI_2D_BODY1;
  }
}

__launch_bounds__(j_block_sz*i_block_sz)
__global__ void poly_jacobi_2D_2(Real_ptr A, Real_ptr B, Index_type N)
{
  Index_type i = 1 + blockIdx.y * i_block_sz + threadIdx.y;
  Index_type j = 1 + blockIdx.x * j_block_sz + threadIdx.x;
  if ( i < N-1 && j < N-1 ) {
    POLYBENCH_JACOBI_2D_BODY2;
  }
}

int main(int argc, char** argv)
{
  Index_type N = 1002;          // upstream N_default
  Index_type run_reps = 2000;   // upstream default reps
  if (argc > 1) N = atol(argv[1]);
  if (argc > 2) run_reps = atol(argv[2]);
  const Index_type len = N * N;
  const size_t bytes = len * sizeof(Real_type);

  // ------------------------- GPU (upstream Base_CUDA) ----------------------
  resetDataInitCount();
  Real_ptr A; Real_ptr B;
  allocAndInitData(A, len);     // upstream setUp order
  allocAndInitData(B, len);

  Real_ptr dA, dB;
  GPU_CHECK(cudaMalloc(&dA, bytes));
  GPU_CHECK(cudaMalloc(&dB, bytes));
  GPU_CHECK(cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice));

  dim3 nthreads_per_block(j_block_sz, i_block_sz, 1);
  dim3 nblocks((size_t)RP_DIVIDE_CEILING_INT(N-2, (Index_type)j_block_sz),
               (size_t)RP_DIVIDE_CEILING_INT(N-2, (Index_type)i_block_sz), 1);
  for (Index_type irep = 0; irep < run_reps; ++irep) {
    poly_jacobi_2D_1<<<nblocks, nthreads_per_block>>>(dA, dB, N);
    poly_jacobi_2D_2<<<nblocks, nthreads_per_block>>>(dA, dB, N);
  }
  GPU_CHECK(cudaGetLastError());
  GPU_CHECK(cudaDeviceSynchronize());
  GPU_CHECK(cudaMemcpy(A, dA, bytes, cudaMemcpyDeviceToHost));
  GPU_CHECK(cudaMemcpy(B, dB, bytes, cudaMemcpyDeviceToHost));
  cudaFree(dA); cudaFree(dB);

  // ------------------------ CPU (upstream Base_Seq) ------------------------
  resetDataInitCount();
  Real_ptr A_ref; Real_ptr B_ref;
  allocAndInitData(A_ref, len);
  allocAndInitData(B_ref, len);
  {
    Real_ptr A = A_ref; Real_ptr B = B_ref;
    for (Index_type irep = 0; irep < run_reps; ++irep) {
      for (Index_type i = 1; i < N-1; ++i ) {
        for (Index_type j = 1; j < N-1; ++j ) {
          POLYBENCH_JACOBI_2D_BODY1;
        }
      }
      for (Index_type i = 1; i < N-1; ++i ) {
        for (Index_type j = 1; j < N-1; ++j ) {
          POLYBENCH_JACOBI_2D_BODY2;
        }
      }
    }
  }

  // ------------------------------ validate ---------------------------------
  bool ok = compareArrays("A", A_ref, A, len, 1.0e-10)
          & compareArrays("B", B_ref, B, len, 1.0e-10);
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
