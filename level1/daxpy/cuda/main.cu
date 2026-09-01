//
// DAXPY benchmark, extracted standalone from the RAJA Performance Suite
// (src/basic/DAXPY.*, BSD-3-Clause). The GPU kernel is the upstream
// Base_CUDA variant and the CPU reference is the upstream Base_Seq variant,
// run with the upstream default problem size, rep count, and data
// initialization. Validation: element-wise comparison of GPU vs CPU result.
//
#include "rp_common.hpp"

// upstream DAXPY.hpp
#define DAXPY_BODY  \
  y[i] += a * x[i] ;

constexpr size_t block_size = 256;  // upstream default_gpu_block_size

// upstream DAXPY-Cuda.cpp (Base_CUDA kernel)
__launch_bounds__(block_size)
__global__ void daxpy(Real_ptr y, Real_ptr x,
                      Real_type a,
                      Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) {
     DAXPY_BODY;
   }
}

int main(int argc, char** argv)
{
  Index_type size = 1000000;    // upstream default problem size
  Index_type run_reps = 500;    // upstream default reps
  if (argc > 1) size = atol(argv[1]);
  if (argc > 2) run_reps = atol(argv[2]);
  const Index_type ibegin = 0;
  const Index_type iend = size;

  // ------------------------- GPU (upstream Base_CUDA) ----------------------
  resetDataInitCount();
  Real_ptr y; Real_ptr x; Real_type a;
  allocAndInitDataConst(y, size, 0.0);   // upstream setUp order
  allocAndInitData(x, size);
  initData(a);

  Real_ptr dx, dy;
  GPU_CHECK(cudaMalloc(&dy, size * sizeof(Real_type)));
  GPU_CHECK(cudaMalloc(&dx, size * sizeof(Real_type)));
  GPU_CHECK(cudaMemcpy(dy, y, size * sizeof(Real_type), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dx, x, size * sizeof(Real_type), cudaMemcpyHostToDevice));

  for (Index_type irep = 0; irep < run_reps; ++irep) {
    const size_t grid_size = RP_DIVIDE_CEILING_INT(iend, block_size);
    daxpy<<<grid_size, block_size>>>(dy, dx, a, iend);
  }
  GPU_CHECK(cudaGetLastError());
  GPU_CHECK(cudaDeviceSynchronize());
  GPU_CHECK(cudaMemcpy(y, dy, size * sizeof(Real_type), cudaMemcpyDeviceToHost));
  GPU_CHECK(cudaFree(dx));
  GPU_CHECK(cudaFree(dy));

  // ------------------------ CPU (upstream Base_Seq) ------------------------
  resetDataInitCount();
  Real_ptr y_ref; Real_ptr x_ref; Real_type a_ref;
  allocAndInitDataConst(y_ref, size, 0.0);
  allocAndInitData(x_ref, size);
  initData(a_ref);
  {
    Real_ptr y = y_ref; Real_ptr x = x_ref; Real_type a = a_ref;
    for (Index_type irep = 0; irep < run_reps; ++irep) {
      for (Index_type i = ibegin; i < iend; ++i ) {
        DAXPY_BODY;
      }
    }
  }

  // ------------------------------ validate ---------------------------------
  bool ok = compareArrays("y", y_ref, y, size, 1.0e-10);
  printf("%s\n", ok ? "PASS" : "FAIL");

  deallocData(y); deallocData(x);
  deallocData(y_ref); deallocData(x_ref);
  return ok ? 0 : 1;
}
