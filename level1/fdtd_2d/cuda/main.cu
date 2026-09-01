//
// POLYBENCH_FDTD_2D benchmark, extracted standalone from the RAJA Performance
// Suite (src/polybench/POLYBENCH_FDTD_2D.*, BSD-3-Clause). GPU kernels are the
// upstream Base_CUDA variant; the CPU reference is the upstream Base_Seq
// variant, with upstream default problem size, reps, and data initialization.
// Validation: element-wise comparison of hz (and ex, ey).
//
#include "rp_common.hpp"

// upstream POLYBENCH_FDTD_2D.hpp
#define POLYBENCH_FDTD_2D_BODY1 \
  ey[j + 0*ny] = fict[t];

#define POLYBENCH_FDTD_2D_BODY2 \
  ey[j + i*ny] = ey[j + i*ny] - 0.5*(hz[j + i*ny] - hz[j + (i-1)*ny]);

#define POLYBENCH_FDTD_2D_BODY3 \
  ex[j + i*ny] = ex[j + i*ny] - 0.5*(hz[j + i*ny] - hz[j-1 + i*ny]);

#define POLYBENCH_FDTD_2D_BODY4 \
  hz[j + i*ny] = hz[j + i*ny] - 0.7*(ex[j+1 + i*ny] - ex[j + i*ny] + \
                                     ey[j + (i+1)*ny] - ey[j + i*ny]);

// upstream POLYBENCH_FDTD_2D-Cuda.cpp launch configuration
constexpr size_t block_size = 256;   // upstream default_gpu_block_size
constexpr size_t j_block_sz = 32;
constexpr size_t i_block_sz = block_size / j_block_sz;

__launch_bounds__(block_size)
__global__ void poly_fdtd2d_1(Real_ptr ey, Real_ptr fict,
                              Index_type ny, Index_type t)
{
  Index_type j = blockIdx.x * block_size + threadIdx.x;
  if (j < ny) { POLYBENCH_FDTD_2D_BODY1; }
}

__launch_bounds__(j_block_sz*i_block_sz)
__global__ void poly_fdtd2d_2(Real_ptr ey, Real_ptr hz,
                              Index_type nx, Index_type ny)
{
  Index_type i = blockIdx.y * i_block_sz + threadIdx.y;
  Index_type j = blockIdx.x * j_block_sz + threadIdx.x;
  if (i > 0 && i < nx && j < ny) { POLYBENCH_FDTD_2D_BODY2; }
}

__launch_bounds__(j_block_sz*i_block_sz)
__global__ void poly_fdtd2d_3(Real_ptr ex, Real_ptr hz,
                              Index_type nx, Index_type ny)
{
  Index_type i = blockIdx.y * i_block_sz + threadIdx.y;
  Index_type j = blockIdx.x * j_block_sz + threadIdx.x;
  if (i < nx && j > 0 && j < ny) { POLYBENCH_FDTD_2D_BODY3; }
}

__launch_bounds__(j_block_sz*i_block_sz)
__global__ void poly_fdtd2d_4(Real_ptr hz, Real_ptr ex, Real_ptr ey,
                              Index_type nx, Index_type ny)
{
  Index_type i = blockIdx.y * i_block_sz + threadIdx.y;
  Index_type j = blockIdx.x * j_block_sz + threadIdx.x;
  if (i < nx-1 && j < ny-1) { POLYBENCH_FDTD_2D_BODY4; }
}

int main(int argc, char** argv)
{
  const Index_type m_tsteps = 40;         // upstream m_tsteps
  Index_type nx = 1000;                   // upstream nx_default
  Index_type ny = 1000;                   // upstream ny_default
  Index_type run_reps = 8 * m_tsteps;     // upstream default reps (320)
  if (argc > 1) { nx = atol(argv[1]); ny = nx; }
  if (argc > 2) run_reps = atol(argv[2]);
  const Index_type len = nx * ny;
  const size_t bytes = len * sizeof(Real_type);
  const size_t fict_bytes = m_tsteps * sizeof(Real_type);

  // ------------------------- GPU (upstream Base_CUDA) ----------------------
  resetDataInitCount();
  Real_ptr hz; Real_ptr ex; Real_ptr ey; Real_ptr fict;
  allocAndInitDataConst(hz, len, 0.0);   // upstream setUp order
  allocAndInitData(ex, len);
  allocAndInitData(ey, len);
  allocAndInitData(fict, m_tsteps);

  Real_ptr d_hz, d_ex, d_ey, d_fict;
  GPU_CHECK(cudaMalloc(&d_hz, bytes));
  GPU_CHECK(cudaMalloc(&d_ex, bytes));
  GPU_CHECK(cudaMalloc(&d_ey, bytes));
  GPU_CHECK(cudaMalloc(&d_fict, fict_bytes));
  GPU_CHECK(cudaMemcpy(d_hz, hz, bytes, cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_ex, ex, bytes, cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_ey, ey, bytes, cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_fict, fict, fict_bytes, cudaMemcpyHostToDevice));

  {
    Index_type t = 0;
    dim3 nthreads_per_block234(j_block_sz, i_block_sz, 1);
    dim3 nblocks234((size_t)RP_DIVIDE_CEILING_INT(ny, (Index_type)j_block_sz),
                    (size_t)RP_DIVIDE_CEILING_INT(nx, (Index_type)i_block_sz), 1);
    for (Index_type irep = 0; irep < run_reps; ++irep) {
      const size_t grid_size1 = RP_DIVIDE_CEILING_INT(ny, (Index_type)block_size);
      poly_fdtd2d_1<<<grid_size1, block_size>>>(d_ey, d_fict, ny, t);
      poly_fdtd2d_2<<<nblocks234, nthreads_per_block234>>>(d_ey, d_hz, nx, ny);
      poly_fdtd2d_3<<<nblocks234, nthreads_per_block234>>>(d_ex, d_hz, nx, ny);
      poly_fdtd2d_4<<<nblocks234, nthreads_per_block234>>>(d_hz, d_ex, d_ey, nx, ny);
      t = (t+1) % m_tsteps;
    }
  }
  GPU_CHECK(cudaGetLastError());
  GPU_CHECK(cudaDeviceSynchronize());
  GPU_CHECK(cudaMemcpy(hz, d_hz, bytes, cudaMemcpyDeviceToHost));
  GPU_CHECK(cudaMemcpy(ex, d_ex, bytes, cudaMemcpyDeviceToHost));
  GPU_CHECK(cudaMemcpy(ey, d_ey, bytes, cudaMemcpyDeviceToHost));
  cudaFree(d_hz); cudaFree(d_ex); cudaFree(d_ey); cudaFree(d_fict);

  // ------------------------ CPU (upstream Base_Seq) ------------------------
  resetDataInitCount();
  Real_ptr hz_r; Real_ptr ex_r; Real_ptr ey_r; Real_ptr fict_r;
  allocAndInitDataConst(hz_r, len, 0.0);
  allocAndInitData(ex_r, len);
  allocAndInitData(ey_r, len);
  allocAndInitData(fict_r, m_tsteps);
  {
    Real_ptr hz = hz_r; Real_ptr ex = ex_r; Real_ptr ey = ey_r; Real_ptr fict = fict_r;
    Index_type t = 0;
    for (Index_type irep = 0; irep < run_reps; ++irep) {
      for (Index_type j = 0; j < ny; j++) { POLYBENCH_FDTD_2D_BODY1; }
      for (Index_type i = 1; i < nx; i++) {
        for (Index_type j = 0; j < ny; j++) { POLYBENCH_FDTD_2D_BODY2; }
      }
      for (Index_type i = 0; i < nx; i++) {
        for (Index_type j = 1; j < ny; j++) { POLYBENCH_FDTD_2D_BODY3; }
      }
      for (Index_type i = 0; i < nx - 1; i++) {
        for (Index_type j = 0; j < ny - 1; j++) { POLYBENCH_FDTD_2D_BODY4; }
      }
      t = (t+1) % m_tsteps;
    }
  }

  // ------------------------------ validate ---------------------------------
  bool ok = compareArrays("hz", hz_r, hz, len, 1.0e-10)
          & compareArrays("ex", ex_r, ex, len, 1.0e-10)
          & compareArrays("ey", ey_r, ey, len, 1.0e-10);
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
