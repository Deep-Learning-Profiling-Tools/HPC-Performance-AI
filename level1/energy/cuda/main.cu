//
// ENERGY benchmark, extracted standalone from the RAJA Performance Suite
// (src/apps/ENERGY.*, BSD-3-Clause). GPU kernels are the upstream Base_CUDA
// variant; the CPU reference is the upstream Base_Seq variant, with upstream
// default problem size, reps, and data initialization.
// Validation: element-wise comparison of GPU vs CPU e_new and q_new.
//
#include "rp_common.hpp"

// upstream ENERGY.hpp
#define ENERGY_BODY1 \
  e_new[i] = e_old[i] - 0.5 * delvc[i] * \
             (p_old[i] + q_old[i]) + 0.5 * work[i];

#define ENERGY_BODY2 \
  if ( delvc[i] > 0.0 ) { \
     q_new[i] = 0.0 ; \
  } \
  else { \
     Real_type vhalf = 1.0 / (1.0 + compHalfStep[i]) ; \
     Real_type ssc = ( pbvc[i] * e_new[i] \
        + vhalf * vhalf * bvc[i] * pHalfStep[i] ) / rho0 ; \
     if ( ssc <= 0.1111111e-36 ) { \
        ssc = 0.3333333e-18 ; \
     } else { \
        ssc = sqrt(ssc) ; \
     } \
     q_new[i] = (ssc*ql_old[i] + qq_old[i]) ; \
  }

#define ENERGY_BODY3 \
  e_new[i] = e_new[i] + 0.5 * delvc[i] \
             * ( 3.0*(p_old[i] + q_old[i]) \
                 - 4.0*(pHalfStep[i] + q_new[i])) ;

#define ENERGY_BODY4 \
  e_new[i] += 0.5 * work[i]; \
  if ( fabs(e_new[i]) < e_cut ) { e_new[i] = 0.0  ; } \
  if ( e_new[i]  < emin ) { e_new[i] = emin ; }

#define ENERGY_BODY5 \
  Real_type q_tilde ; \
  if (delvc[i] > 0.0) { \
     q_tilde = 0. ; \
  } \
  else { \
     Real_type ssc = ( pbvc[i] * e_new[i] \
         + vnewc[i] * vnewc[i] * bvc[i] * p_new[i] ) / rho0 ; \
     if ( ssc <= 0.1111111e-36 ) { \
        ssc = 0.3333333e-18 ; \
     } else { \
        ssc = sqrt(ssc) ; \
     } \
     q_tilde = (ssc*ql_old[i] + qq_old[i]) ; \
  } \
  e_new[i] = e_new[i] - ( 7.0*(p_old[i] + q_old[i]) \
                         - 8.0*(pHalfStep[i] + q_new[i]) \
                         + (p_new[i] + q_tilde)) * delvc[i] / 6.0 ; \
  if ( fabs(e_new[i]) < e_cut ) { \
     e_new[i] = 0.0  ; \
  } \
  if ( e_new[i]  < emin ) { \
     e_new[i] = emin ; \
  }

#define ENERGY_BODY6 \
  if ( delvc[i] <= 0.0 ) { \
     Real_type ssc = ( pbvc[i] * e_new[i] \
             + vnewc[i] * vnewc[i] * bvc[i] * p_new[i] ) / rho0 ; \
     if ( ssc <= 0.1111111e-36 ) { \
        ssc = 0.3333333e-18 ; \
     } else { \
        ssc = sqrt(ssc) ; \
     } \
     q_new[i] = (ssc*ql_old[i] + qq_old[i]) ; \
     if (fabs(q_new[i]) < q_cut) q_new[i] = 0.0 ; \
  }

constexpr size_t block_size = 256;  // upstream default_gpu_block_size

// upstream ENERGY-Cuda.cpp (Base_CUDA kernels)
__launch_bounds__(block_size)
__global__ void energycalc1(Real_ptr e_new, Real_ptr e_old, Real_ptr delvc,
                            Real_ptr p_old, Real_ptr q_old, Real_ptr work,
                            Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) { ENERGY_BODY1; }
}

__launch_bounds__(block_size)
__global__ void energycalc2(Real_ptr delvc, Real_ptr q_new,
                            Real_ptr compHalfStep, Real_ptr pHalfStep,
                            Real_ptr e_new, Real_ptr bvc, Real_ptr pbvc,
                            Real_ptr ql_old, Real_ptr qq_old,
                            Real_type rho0,
                            Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) { ENERGY_BODY2; }
}

__launch_bounds__(block_size)
__global__ void energycalc3(Real_ptr e_new, Real_ptr delvc,
                            Real_ptr p_old, Real_ptr q_old,
                            Real_ptr pHalfStep, Real_ptr q_new,
                            Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) { ENERGY_BODY3; }
}

__launch_bounds__(block_size)
__global__ void energycalc4(Real_ptr e_new, Real_ptr work,
                            Real_type e_cut, Real_type emin,
                            Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) { ENERGY_BODY4; }
}

__launch_bounds__(block_size)
__global__ void energycalc5(Real_ptr delvc,
                            Real_ptr pbvc, Real_ptr e_new, Real_ptr vnewc,
                            Real_ptr bvc, Real_ptr p_new,
                            Real_ptr ql_old, Real_ptr qq_old,
                            Real_ptr p_old, Real_ptr q_old,
                            Real_ptr pHalfStep, Real_ptr q_new,
                            Real_type rho0, Real_type e_cut, Real_type emin,
                            Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) { ENERGY_BODY5; }
}

__launch_bounds__(block_size)
__global__ void energycalc6(Real_ptr delvc,
                            Real_ptr pbvc, Real_ptr e_new, Real_ptr vnewc,
                            Real_ptr bvc, Real_ptr p_new,
                            Real_ptr q_new,
                            Real_ptr ql_old, Real_ptr qq_old,
                            Real_type rho0, Real_type q_cut,
                            Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) { ENERGY_BODY6; }
}

#define ENERGY_ARRAYS(OP) \
  OP(e_new) OP(e_old) OP(delvc) OP(p_new) OP(p_old) OP(q_new) OP(q_old) \
  OP(work) OP(compHalfStep) OP(pHalfStep) OP(bvc) OP(pbvc) OP(ql_old) \
  OP(qq_old) OP(vnewc)

struct Data {
#define DECL(n) Real_ptr n;
  ENERGY_ARRAYS(DECL)
#undef DECL
  Real_type rho0, e_cut, emin, q_cut;
};

// upstream ENERGY.cpp setUp (init order preserved)
static void setUp(Data& d, Index_type size)
{
  resetDataInitCount();
  allocAndInitDataConst(d.e_new, size, 0.0);
  allocAndInitData(d.e_old, size);
  allocAndInitData(d.delvc, size);
  allocAndInitData(d.p_new, size);
  allocAndInitData(d.p_old, size);
  allocAndInitDataConst(d.q_new, size, 0.0);
  allocAndInitData(d.q_old, size);
  allocAndInitData(d.work, size);
  allocAndInitData(d.compHalfStep, size);
  allocAndInitData(d.pHalfStep, size);
  allocAndInitData(d.bvc, size);
  allocAndInitData(d.pbvc, size);
  allocAndInitData(d.ql_old, size);
  allocAndInitData(d.qq_old, size);
  allocAndInitData(d.vnewc, size);
  initData(d.rho0);
  initData(d.e_cut);
  initData(d.emin);
  initData(d.q_cut);
}

int main(int argc, char** argv)
{
  Index_type size = 1000000;   // upstream default problem size
  Index_type run_reps = 130;   // upstream default reps
  if (argc > 1) size = atol(argv[1]);
  if (argc > 2) run_reps = atol(argv[2]);
  const Index_type ibegin = 0;
  const Index_type iend = size;
  const size_t bytes = size * sizeof(Real_type);

  // ------------------------- GPU (upstream Base_CUDA) ----------------------
  Data g; setUp(g, size);
#define DDECL(n) Real_ptr d_##n;
  ENERGY_ARRAYS(DDECL)
#undef DDECL
#define DALLOC(n) GPU_CHECK(cudaMalloc(&d_##n, bytes)); \
  GPU_CHECK(cudaMemcpy(d_##n, g.n, bytes, cudaMemcpyHostToDevice));
  ENERGY_ARRAYS(DALLOC)
#undef DALLOC

  for (Index_type irep = 0; irep < run_reps; ++irep) {
    const size_t grid_size = RP_DIVIDE_CEILING_INT(iend, block_size);
    energycalc1<<<grid_size, block_size>>>(d_e_new, d_e_old, d_delvc,
                                           d_p_old, d_q_old, d_work, iend);
    energycalc2<<<grid_size, block_size>>>(d_delvc, d_q_new,
                                           d_compHalfStep, d_pHalfStep,
                                           d_e_new, d_bvc, d_pbvc,
                                           d_ql_old, d_qq_old, g.rho0, iend);
    energycalc3<<<grid_size, block_size>>>(d_e_new, d_delvc, d_p_old, d_q_old,
                                           d_pHalfStep, d_q_new, iend);
    energycalc4<<<grid_size, block_size>>>(d_e_new, d_work, g.e_cut, g.emin, iend);
    energycalc5<<<grid_size, block_size>>>(d_delvc, d_pbvc, d_e_new, d_vnewc,
                                           d_bvc, d_p_new, d_ql_old, d_qq_old,
                                           d_p_old, d_q_old, d_pHalfStep, d_q_new,
                                           g.rho0, g.e_cut, g.emin, iend);
    energycalc6<<<grid_size, block_size>>>(d_delvc, d_pbvc, d_e_new, d_vnewc,
                                           d_bvc, d_p_new, d_q_new,
                                           d_ql_old, d_qq_old, g.rho0, g.q_cut, iend);
  }
  GPU_CHECK(cudaGetLastError());
  GPU_CHECK(cudaDeviceSynchronize());
  GPU_CHECK(cudaMemcpy(g.e_new, d_e_new, bytes, cudaMemcpyDeviceToHost));
  GPU_CHECK(cudaMemcpy(g.q_new, d_q_new, bytes, cudaMemcpyDeviceToHost));
#define DFREE(n) cudaFree(d_##n);
  ENERGY_ARRAYS(DFREE)
#undef DFREE

  // ------------------------ CPU (upstream Base_Seq) ------------------------
  Data cd; setUp(cd, size);
  {
    Real_ptr e_new = cd.e_new; Real_ptr e_old = cd.e_old; Real_ptr delvc = cd.delvc;
    Real_ptr p_new = cd.p_new; Real_ptr p_old = cd.p_old; Real_ptr q_new = cd.q_new;
    Real_ptr q_old = cd.q_old; Real_ptr work = cd.work;
    Real_ptr compHalfStep = cd.compHalfStep; Real_ptr pHalfStep = cd.pHalfStep;
    Real_ptr bvc = cd.bvc; Real_ptr pbvc = cd.pbvc; Real_ptr ql_old = cd.ql_old;
    Real_ptr qq_old = cd.qq_old; Real_ptr vnewc = cd.vnewc;
    const Real_type rho0 = cd.rho0; const Real_type e_cut = cd.e_cut;
    const Real_type emin = cd.emin; const Real_type q_cut = cd.q_cut;
    for (Index_type irep = 0; irep < run_reps; ++irep) {
      for (Index_type i = ibegin; i < iend; ++i ) { ENERGY_BODY1; }
      for (Index_type i = ibegin; i < iend; ++i ) { ENERGY_BODY2; }
      for (Index_type i = ibegin; i < iend; ++i ) { ENERGY_BODY3; }
      for (Index_type i = ibegin; i < iend; ++i ) { ENERGY_BODY4; }
      for (Index_type i = ibegin; i < iend; ++i ) { ENERGY_BODY5; }
      for (Index_type i = ibegin; i < iend; ++i ) { ENERGY_BODY6; }
    }
  }

  // ------------------------------ validate ---------------------------------
  bool ok = compareArrays("e_new", cd.e_new, g.e_new, size, 1.0e-10)
          & compareArrays("q_new", cd.q_new, g.q_new, size, 1.0e-10);
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
