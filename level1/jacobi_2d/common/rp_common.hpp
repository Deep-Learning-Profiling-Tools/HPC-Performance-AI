#ifndef HPCPERF_RP_COMMON_HPP
#define HPCPERF_RP_COMMON_HPP
//
// Minimal standalone support for benchmarks extracted from the RAJA
// Performance Suite (https://github.com/LLNL/RAJAPerf, BSD-3-Clause).
//
// The data initialization routines below are ported verbatim from
// RAJAPerf common/DataUtils.cpp so that input semantics (values and
// init-call ordering) match the upstream suite exactly.
//
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cmath>
#include <algorithm>

using Index_type = std::ptrdiff_t;
using Real_type = double;
using Real_ptr = Real_type*;
using Int_type = int;
using Int_ptr = Int_type*;

inline int& rp_data_init_count() { static int count = 0; return count; }
inline void resetDataInitCount() { rp_data_init_count() = 0; }
inline void incDataInitCount() { rp_data_init_count()++; }

template <typename T>
inline void allocData(T*& ptr, Index_type len) { ptr = new T[len]; }
template <typename T>
inline void deallocData(T*& ptr) { delete[] ptr; ptr = nullptr; }

// RAJAPerf DataUtils.cpp: initData(Real_ptr&, len)
inline void initData(Real_ptr& ptr, Index_type len)
{
  Real_type factor = ( rp_data_init_count() % 2 ? 0.1 : 0.2 );
  for (Index_type i = 0; i < len; ++i) {
    ptr[i] = factor*(i + 1.1)/(i + 1.12345);
  }
  incDataInitCount();
}

// RAJAPerf DataUtils.hpp: initDataConst
template <typename T, typename V>
inline void initDataConst(T*& ptr, Index_type len, V val)
{
  for (Index_type i = 0; i < len; ++i) {
    ptr[i] = val;
  }
  incDataInitCount();
}

// RAJAPerf DataUtils.cpp: initDataRandSign
inline void initDataRandSign(Real_ptr& ptr, Index_type len)
{
  Real_type factor = ( rp_data_init_count() % 2 ? 0.1 : 0.2 );
  srand(4793);
  for (Index_type i = 0; i < len; ++i) {
    Real_type signfact = Real_type(rand())/RAND_MAX;
    signfact = ( signfact < 0.5 ? -1.0 : 1.0 );
    ptr[i] = signfact*factor*(i + 1.1)/(i + 1.12345);
  }
  incDataInitCount();
}

// RAJAPerf DataUtils.cpp: initDataRandValue
inline void initDataRandValue(Real_ptr& ptr, Index_type len)
{
  srand(4793);
  for (Index_type i = 0; i < len; ++i) {
    ptr[i] = Real_type(rand())/RAND_MAX;
  }
  incDataInitCount();
}

// RAJAPerf DataUtils.cpp: initData(Real_type&)
inline void initData(Real_type& d)
{
  Real_type factor = ( rp_data_init_count() % 2 ? 0.1 : 0.2 );
  d = factor*1.1/1.12345;
  incDataInitCount();
}

template <typename T>
inline void allocAndInitData(T*& ptr, Index_type len)
{ allocData(ptr, len); initData(ptr, len); }
template <typename T, typename V>
inline void allocAndInitDataConst(T*& ptr, Index_type len, V val)
{ allocData(ptr, len); initDataConst(ptr, len, val); }
inline void allocAndInitDataRandSign(Real_ptr& ptr, Index_type len)
{ allocData(ptr, len); initDataRandSign(ptr, len); }
inline void allocAndInitDataRandValue(Real_ptr& ptr, Index_type len)
{ allocData(ptr, len); initDataRandValue(ptr, len); }

// Element-wise comparison of GPU result vs CPU reference.
inline bool compareArrays(const char* name, const Real_ptr ref, const Real_ptr test,
                          Index_type len, double tol)
{
  double max_err = 0.0;
  Index_type bad = -1;
  for (Index_type i = 0; i < len; ++i) {
    double err = std::fabs(test[i] - ref[i]) / (1.0 + std::fabs(ref[i]));
    if (err > max_err) { max_err = err; bad = i; }
  }
  if (max_err > tol) {
    printf("  %s: FAIL max rel err %.3e at %ld (ref %.17g vs gpu %.17g)\n",
           name, max_err, (long)bad, ref[bad], test[bad], tol);
    return false;
  }
  printf("  %s: max rel err %.3e (tol %.1e)\n", name, max_err, tol);
  return true;
}

#define RP_DIVIDE_CEILING_INT(a, b) (((a) + (b) - 1) / (b))

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#elif defined(__HIPCC__)
#include <hip/hip_runtime.h>
// Thin portability aliases so the shared driver source (written against the
// CUDA runtime API) also compiles as HIP. GPU kernel code is identical
// between the upstream Base_CUDA and Base_HIP variants.
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
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GPU_CHECK(call) \
  do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) { \
      fprintf(stderr, "GPU error %s at %s:%d\n", \
              cudaGetErrorString(err_), __FILE__, __LINE__); \
      exit(1); \
    } \
  } while (0)
#endif

#endif // HPCPERF_RP_COMMON_HPP
