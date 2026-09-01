//
// PRESSURE benchmark, extracted standalone from the RAJA Performance Suite
// (src/apps/PRESSURE.*, BSD-3-Clause). GPU kernels are the upstream Base_CUDA
// variant; the CPU reference is the upstream Base_Seq variant, with upstream
// default problem size, reps, and data initialization.
// Validation: element-wise comparison of GPU vs CPU p_new (and bvc).
//
#include "rp_common.hpp"

// upstream PRESSURE.hpp
#define PRESSURE_BODY1 \
  bvc[i] = cls * (compression[i] + 1.0);

#define PRESSURE_BODY2 \
  p_new[i] = bvc[i] * e_old[i] ; \
  if ( fabs(p_new[i]) < p_cut ) p_new[i] = 0.0 ; \
  if ( vnewc[i] >= eosvmax ) p_new[i] = 0.0 ; \
  if ( p_new[i] < pmin ) p_new[i] = pmin ;

constexpr size_t block_size = 256;  // upstream default_gpu_block_size

// upstream PRESSURE-Cuda.cpp (Base_CUDA kernels)
__launch_bounds__(block_size)
__global__ void pressurecalc1(Real_ptr bvc, Real_ptr compression,
                              const Real_type cls,
                              Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) {
     PRESSURE_BODY1;
   }
}

__launch_bounds__(block_size)
__global__ void pressurecalc2(Real_ptr p_new, Real_ptr bvc, Real_ptr e_old,
                              Real_ptr vnewc,
                              const Real_type p_cut, const Real_type eosvmax,
                              const Real_type pmin,
                              Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) {
     PRESSURE_BODY2;
   }
}

struct Data {
  Real_ptr compression, bvc, p_new, e_old, vnewc;
  Real_type cls, p_cut, pmin, eosvmax;
};

// upstream PRESSURE.cpp setUp (init order preserved)
static void setUp(Data& d, Index_type size)
{
  resetDataInitCount();
  allocAndInitData(d.compression, size);
  allocAndInitData(d.bvc, size);
  allocAndInitDataConst(d.p_new, size, 0.0);
  allocAndInitData(d.e_old, size);
  allocAndInitData(d.vnewc, size);
  initData(d.cls);
  initData(d.p_cut);
  initData(d.pmin);
  initData(d.eosvmax);
}

int main(int argc, char** argv)
{
  Index_type size = 1000000;   // upstream default problem size
  Index_type run_reps = 700;   // upstream default reps
  if (argc > 1) size = atol(argv[1]);
  if (argc > 2) run_reps = atol(argv[2]);
  const Index_type ibegin = 0;
  const Index_type iend = size;
  const size_t bytes = size * sizeof(Real_type);

  // ------------------------- GPU (upstream Base_CUDA) ----------------------
  Data g; setUp(g, size);
  Real_ptr d_compression, d_bvc, d_p_new, d_e_old, d_vnewc;
  GPU_CHECK(cudaMalloc(&d_compression, bytes));
  GPU_CHECK(cudaMalloc(&d_bvc, bytes));
  GPU_CHECK(cudaMalloc(&d_p_new, bytes));
  GPU_CHECK(cudaMalloc(&d_e_old, bytes));
  GPU_CHECK(cudaMalloc(&d_vnewc, bytes));
  GPU_CHECK(cudaMemcpy(d_compression, g.compression, bytes, cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_bvc, g.bvc, bytes, cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_p_new, g.p_new, bytes, cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_e_old, g.e_old, bytes, cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_vnewc, g.vnewc, bytes, cudaMemcpyHostToDevice));

  for (Index_type irep = 0; irep < run_reps; ++irep) {
    const size_t grid_size = RP_DIVIDE_CEILING_INT(iend, block_size);
    pressurecalc1<<<grid_size, block_size>>>(d_bvc, d_compression, g.cls, iend);
    pressurecalc2<<<grid_size, block_size>>>(d_p_new, d_bvc, d_e_old, d_vnewc,
                                             g.p_cut, g.eosvmax, g.pmin, iend);
  }
  GPU_CHECK(cudaGetLastError());
  GPU_CHECK(cudaDeviceSynchronize());
  GPU_CHECK(cudaMemcpy(g.p_new, d_p_new, bytes, cudaMemcpyDeviceToHost));
  GPU_CHECK(cudaMemcpy(g.bvc, d_bvc, bytes, cudaMemcpyDeviceToHost));
  cudaFree(d_compression); cudaFree(d_bvc); cudaFree(d_p_new);
  cudaFree(d_e_old); cudaFree(d_vnewc);

  // ------------------------ CPU (upstream Base_Seq) ------------------------
  Data c; setUp(c, size);
  {
    Real_ptr compression = c.compression; Real_ptr bvc = c.bvc;
    Real_ptr p_new = c.p_new; Real_ptr e_old = c.e_old; Real_ptr vnewc = c.vnewc;
    const Real_type cls = c.cls; const Real_type p_cut = c.p_cut;
    const Real_type pmin = c.pmin; const Real_type eosvmax = c.eosvmax;
    for (Index_type irep = 0; irep < run_reps; ++irep) {
      for (Index_type i = ibegin; i < iend; ++i ) {
        PRESSURE_BODY1;
      }
      for (Index_type i = ibegin; i < iend; ++i ) {
        PRESSURE_BODY2;
      }
    }
  }

  // ------------------------------ validate ---------------------------------
  bool ok = compareArrays("bvc", c.bvc, g.bvc, size, 1.0e-10)
          & compareArrays("p_new", c.p_new, g.p_new, size, 1.0e-10);
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
