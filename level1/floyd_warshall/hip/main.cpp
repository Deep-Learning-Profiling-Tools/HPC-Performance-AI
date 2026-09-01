//
// POLYBENCH_FLOYD_WARSHALL benchmark, extracted standalone from the RAJA
// Performance Suite (src/polybench/POLYBENCH_FLOYD_WARSHALL.*, BSD-3-Clause).
// GPU kernel is the upstream Base_CUDA variant; the CPU reference is the
// upstream Base_Seq variant, with upstream default problem size, reps, and
// data initialization. Validation: element-wise comparison of pout.
//
#include "rp_common.hpp"

// upstream POLYBENCH_FLOYD_WARSHALL.hpp
#define POLYBENCH_FLOYD_WARSHALL_BODY \
  pout[j + i*N] = pin[j + i*N] < pin[k + i*N] + pin[j + k*N] ? \
                  pin[j + i*N] : pin[k + i*N] + pin[j + k*N];

// upstream POLYBENCH_FLOYD_WARSHALL-Cuda.cpp launch configuration
constexpr size_t block_size = 256;   // upstream default_gpu_block_size
constexpr size_t j_block_sz = 32;
constexpr size_t i_block_sz = block_size / j_block_sz;

__launch_bounds__(j_block_sz*i_block_sz)
__global__ void poly_floyd_warshall(Real_ptr pout, Real_ptr pin,
                                    Index_type k,
                                    Index_type N)
{
  Index_type i = blockIdx.y * i_block_sz + threadIdx.y;
  Index_type j = blockIdx.x * j_block_sz + threadIdx.x;
  if ( i < N && j < N ) {
    POLYBENCH_FLOYD_WARSHALL_BODY;
  }
}

int main(int argc, char** argv)
{
  Index_type N = 1000;       // upstream N_default
  Index_type run_reps = 8;   // upstream default reps
  if (argc > 1) N = atol(argv[1]);
  if (argc > 2) run_reps = atol(argv[2]);
  const Index_type len = N * N;
  const size_t bytes = len * sizeof(Real_type);

  // ------------------------- GPU (upstream Base_CUDA) ----------------------
  resetDataInitCount();
  Real_ptr pin; Real_ptr pout;
  allocAndInitDataRandSign(pin, len);   // upstream setUp order
  allocAndInitDataConst(pout, len, 0.0);

  Real_ptr d_pin, d_pout;
  GPU_CHECK(cudaMalloc(&d_pin, bytes));
  GPU_CHECK(cudaMalloc(&d_pout, bytes));
  GPU_CHECK(cudaMemcpy(d_pin, pin, bytes, cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_pout, pout, bytes, cudaMemcpyHostToDevice));

  dim3 nthreads_per_block(j_block_sz, i_block_sz, 1);
  dim3 nblocks((size_t)RP_DIVIDE_CEILING_INT(N, (Index_type)j_block_sz),
               (size_t)RP_DIVIDE_CEILING_INT(N, (Index_type)i_block_sz), 1);
  for (Index_type irep = 0; irep < run_reps; ++irep) {
    for (Index_type k = 0; k < N; ++k) {
      poly_floyd_warshall<<<nblocks, nthreads_per_block>>>(d_pout, d_pin, k, N);
    }
  }
  GPU_CHECK(cudaGetLastError());
  GPU_CHECK(cudaDeviceSynchronize());
  GPU_CHECK(cudaMemcpy(pout, d_pout, bytes, cudaMemcpyDeviceToHost));
  cudaFree(d_pin); cudaFree(d_pout);

  // ------------------------ CPU (upstream Base_Seq) ------------------------
  resetDataInitCount();
  Real_ptr pin_ref; Real_ptr pout_ref;
  allocAndInitDataRandSign(pin_ref, len);
  allocAndInitDataConst(pout_ref, len, 0.0);
  {
    Real_ptr pin = pin_ref; Real_ptr pout = pout_ref;
    for (Index_type irep = 0; irep < run_reps; ++irep) {
      for (Index_type k = 0; k < N; ++k) {
        for (Index_type i = 0; i < N; ++i) {
          for (Index_type j = 0; j < N; ++j) {
            POLYBENCH_FLOYD_WARSHALL_BODY;
          }
        }
      }
    }
  }

  // ------------------------------ validate ---------------------------------
  bool ok = compareArrays("pout", pout_ref, pout, len, 1.0e-10);
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
