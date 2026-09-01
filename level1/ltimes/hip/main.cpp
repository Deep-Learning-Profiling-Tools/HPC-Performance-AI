//
// LTIMES benchmark, extracted standalone from the RAJA Performance Suite
// (src/apps/LTIMES.*, BSD-3-Clause). Discrete-ordinates moment update:
// phi(z,g,m) += ell(m,d) * psi(z,g,d).
//
// The upstream body is written with RAJA TypedViews (Perm<0,1,2> row-major
// layouts, stride-1 rightmost index); here the identical computation and
// memory layout are expressed with raw indexing so no RAJA dependency
// remains:
//   psi(z,g,d) -> psidat[d + num_d*(g + num_g*z)]
//   ell(m,d)   -> elldat[d + num_d*m]
//   phi(z,g,m) -> phidat[m + num_m*(g + num_g*z)]
// GPU kernel/launch shape are the upstream Base_CUDA variant; CPU reference
// is the upstream Base_Seq loop nest, with upstream default sizes, reps, and
// data initialization. Validation: element-wise comparison of phi.
//
#include "rp_common.hpp"

#define LTIMES_BODY \
  phidat[m + num_m*(g + num_g*z)] += \
      elldat[d + num_d*m] * psidat[d + num_d*(g + num_g*z)];

// upstream LTIMES-Cuda.cpp launch configuration (block_size 256)
constexpr size_t m_block_sz = 32;
constexpr size_t g_block_sz = 4;  // greater_of_squarest_factor_pair(256/32)
constexpr size_t z_block_sz = 2;  // lesser_of_squarest_factor_pair(256/32)

__launch_bounds__(m_block_sz*g_block_sz*z_block_sz)
__global__ void ltimes(Real_ptr phidat, Real_ptr elldat, Real_ptr psidat,
                       Index_type num_d, Index_type num_m,
                       Index_type num_g, Index_type num_z)
{
   Index_type m = blockIdx.x * m_block_sz + threadIdx.x;
   Index_type g = blockIdx.y * g_block_sz + threadIdx.y;
   Index_type z = blockIdx.z * z_block_sz + threadIdx.z;

   if (m < num_m && g < num_g && z < num_z) {
     for (Index_type d = 0; d < num_d; ++d ) {
       LTIMES_BODY;
     }
   }
}

int main(int argc, char** argv)
{
  // upstream defaults (RunParams ltimes_num_* and LTIMES ctor)
  const Index_type num_d = 6;
  const Index_type num_g = 32;
  const Index_type num_m = 25;
  Index_type target_size = 1000000;
  Index_type run_reps = 50;
  if (argc > 1) target_size = atol(argv[1]);
  if (argc > 2) run_reps = atol(argv[2]);
  const Index_type num_z =
      std::max((target_size + (num_d * num_g)/2) / (num_d * num_g), (Index_type)1);
  const Index_type philen = num_m * num_g * num_z;
  const Index_type elllen = num_d * num_m;
  const Index_type psilen = num_d * num_g * num_z;

  // ------------------------- GPU (upstream Base_CUDA) ----------------------
  resetDataInitCount();
  Real_ptr phidat; Real_ptr elldat; Real_ptr psidat;
  allocAndInitDataConst(phidat, philen, Real_type(0.0));  // upstream setUp order
  allocAndInitData(elldat, elllen);
  allocAndInitData(psidat, psilen);

  Real_ptr d_phi, d_ell, d_psi;
  GPU_CHECK(cudaMalloc(&d_phi, philen * sizeof(Real_type)));
  GPU_CHECK(cudaMalloc(&d_ell, elllen * sizeof(Real_type)));
  GPU_CHECK(cudaMalloc(&d_psi, psilen * sizeof(Real_type)));
  GPU_CHECK(cudaMemcpy(d_phi, phidat, philen * sizeof(Real_type), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_ell, elldat, elllen * sizeof(Real_type), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_psi, psidat, psilen * sizeof(Real_type), cudaMemcpyHostToDevice));

  {
    dim3 nthreads_per_block(m_block_sz, g_block_sz, z_block_sz);
    dim3 nblocks((size_t)RP_DIVIDE_CEILING_INT(num_m, (Index_type)m_block_sz),
                 (size_t)RP_DIVIDE_CEILING_INT(num_g, (Index_type)g_block_sz),
                 (size_t)RP_DIVIDE_CEILING_INT(num_z, (Index_type)z_block_sz));
    for (Index_type irep = 0; irep < run_reps; ++irep) {
      ltimes<<<nblocks, nthreads_per_block>>>(d_phi, d_ell, d_psi,
                                              num_d, num_m, num_g, num_z);
    }
  }
  GPU_CHECK(cudaGetLastError());
  GPU_CHECK(cudaDeviceSynchronize());
  GPU_CHECK(cudaMemcpy(phidat, d_phi, philen * sizeof(Real_type), cudaMemcpyDeviceToHost));
  cudaFree(d_phi); cudaFree(d_ell); cudaFree(d_psi);

  // ------------------------ CPU (upstream Base_Seq) ------------------------
  resetDataInitCount();
  Real_ptr phi_r; Real_ptr ell_r; Real_ptr psi_r;
  allocAndInitDataConst(phi_r, philen, Real_type(0.0));
  allocAndInitData(ell_r, elllen);
  allocAndInitData(psi_r, psilen);
  {
    Real_ptr phidat = phi_r; Real_ptr elldat = ell_r; Real_ptr psidat = psi_r;
    for (Index_type irep = 0; irep < run_reps; ++irep) {
      for (Index_type z = 0; z < num_z; ++z ) {
        for (Index_type g = 0; g < num_g; ++g ) {
          for (Index_type m = 0; m < num_m; ++m ) {
            for (Index_type d = 0; d < num_d; ++d ) {
              LTIMES_BODY;
            }
          }
        }
      }
    }
  }

  // ------------------------------ validate ---------------------------------
  bool ok = compareArrays("phi", phi_r, phidat, philen, 1.0e-10);
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
