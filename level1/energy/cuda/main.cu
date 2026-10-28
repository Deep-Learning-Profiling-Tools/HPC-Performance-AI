//
// ENERGY benchmark, extracted standalone from the RAJA Performance Suite
// (src/apps/ENERGY.*, BSD-3-Clause). GPU kernels are the upstream Base_CUDA
// variant; the CPU reference is the upstream Base_Seq variant, with upstream
// default problem size, reps, and data initialization.
// Validation: element-wise comparison of GPU vs CPU e_new and q_new.
//
#include "hpcperf_roi.h"
#include "hpcperf_golden.h"
#include "rp_common.hpp"
#include <cstring>

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
// probe 1 (correctness-probe input probe-randsign, `--probe`; set-up only, the kernels are unchanged): upstream's
// data gives delvc > 0 everywhere, so the delvc <= 0 branches of ENERGY_BODY2/5/6 never run, q_new stays 0 and
// every e_new (0.167-0.170) falls below e_cut and is clamped to emin -- a constant fill reproduces the output.
// The probe initializes delvc with upstream's own random-sign helper (DataUtils initDataRandSign, same
// factor, same init count) and scales the cut-offs e_cut and emin by 1e-2, so both branches run and no
// e_new is clamped.
// probe 2 (probe-cutoffs, `--probe-cutoffs`; set-up only): the branches no other input decides -- body 4's e_cut
// / emin clamps, body 6's q_cut and delvc <= 0 guard, the ssc floor of bodies 2, 5, 6. delvc and pbvc from
// initDataRandSign (both helpers reseed srand(4793): pbvc < 0 exactly where delvc < 0, so ssc can be <= 0),
// pHalfStep and qq_old from initDataRandValue (values in [0, 1]); upstream's cut-offs, unscaled.
static void setUp(Data& d, Index_type size, int probe)
{
  resetDataInitCount();
  allocAndInitDataConst(d.e_new, size, 0.0);
  allocAndInitData(d.e_old, size);
  if (probe) allocAndInitDataRandSign(d.delvc, size);
  else       allocAndInitData(d.delvc, size);
  allocAndInitData(d.p_new, size);
  allocAndInitData(d.p_old, size);
  allocAndInitDataConst(d.q_new, size, 0.0);
  allocAndInitData(d.q_old, size);
  allocAndInitData(d.work, size);
  allocAndInitData(d.compHalfStep, size);
  if (probe == 2) allocAndInitDataRandValue(d.pHalfStep, size);
  else            allocAndInitData(d.pHalfStep, size);
  allocAndInitData(d.bvc, size);
  if (probe == 2) allocAndInitDataRandSign(d.pbvc, size);
  else            allocAndInitData(d.pbvc, size);
  allocAndInitData(d.ql_old, size);
  if (probe == 2) allocAndInitDataRandValue(d.qq_old, size);
  else            allocAndInitData(d.qq_old, size);
  allocAndInitData(d.vnewc, size);
  initData(d.rho0);
  initData(d.e_cut);
  initData(d.emin);
  initData(d.q_cut);
  if (probe == 1) { d.e_cut *= 1.0e-2; d.emin *= 1.0e-2; }
}

int main(int argc, char** argv)
{
  Index_type size = 1000000;   // upstream default problem size
  Index_type run_reps = 130;   // upstream default reps
  if (argc > 1) size = atol(argv[1]);
  if (argc > 2) run_reps = atol(argv[2]);
  int probe = 0;               // correctness-probe set-up data (see setUp); never a performance input
  if (argc > 3) {
    if (argc == 4 && strcmp(argv[3], "--probe") == 0)              probe = 1;
    else if (argc == 4 && strcmp(argv[3], "--probe-cutoffs") == 0) probe = 2;
    else {
      fprintf(stderr, "usage: %s [size [reps [--probe|--probe-cutoffs]]]\n", argv[0]);
      return 2;
    }
  }
  const Index_type ibegin = 0;
  const Index_type iend = size;
  const size_t bytes = size * sizeof(Real_type);

  // ------------------------- GPU (upstream Base_CUDA) ----------------------
  Data g; setUp(g, size, probe);
#define DDECL(n) Real_ptr d_##n;
  ENERGY_ARRAYS(DDECL)
#undef DDECL
#define DALLOC(n) GPU_CHECK(cudaMalloc(&d_##n, bytes)); \
  GPU_CHECK(cudaMemcpy(d_##n, g.n, bytes, cudaMemcpyHostToDevice));
  ENERGY_ARRAYS(DALLOC)
#undef DALLOC

  HPCPERF_ROI_BEGIN();  // tools/timing ROI: the timed repetition loop
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
  HPCPERF_ROI_END();
  GPU_CHECK(cudaMemcpy(g.e_new, d_e_new, bytes, cudaMemcpyDeviceToHost));
  GPU_CHECK(cudaMemcpy(g.q_new, d_q_new, bytes, cudaMemcpyDeviceToHost));
  hpcperf_golden_output("e_new", g.e_new, size, "f64");   // tools/golden: the checked outputs
  hpcperf_golden_output("q_new", g.q_new, size, "f64");
#define DFREE(n) cudaFree(d_##n);
  ENERGY_ARRAYS(DFREE)
#undef DFREE

  bool ok = true;
  if (hpcperf_skip_verify()) {
    // default (hpcperf_skip_verify() = !hpcperf_cpu_reference_enabled(), rp_common.hpp): the
    // CPU reference below (upstream Base_Seq) is skipped; correctness = the captured outputs
    // against the frozen golden reference (tools/golden).
    printf("SKIP_VERIFY\n");
  } else {
    // ------------------------ CPU (upstream Base_Seq) ------------------------
    Data cd; setUp(cd, size, probe);
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
    hpcperf_golden_output("e_new.cpu", cd.e_new, size, "f64");   // the CPU reference (golden make only)
    hpcperf_golden_output("q_new.cpu", cd.q_new, size, "f64");

    // ------------------------------ validate ---------------------------------
    ok = compareArrays("e_new", cd.e_new, g.e_new, size, 1.0e-10)
            & compareArrays("q_new", cd.q_new, g.q_new, size, 1.0e-10);
    printf("%s\n", ok ? "PASS" : "FAIL");
  }
  return ok ? 0 : 1;
}
