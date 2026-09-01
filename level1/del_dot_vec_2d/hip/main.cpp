//
// DEL_DOT_VEC_2D benchmark, extracted standalone from the RAJA Performance
// Suite (src/apps/DEL_DOT_VEC_2D.*, BSD-3-Clause). Divergence of a vector
// field on a staggered 2D mesh over an indirection (real-zone) list.
// The ADomain mesh setup, setMeshPositions_2d, and setRealZones_2d are
// ported verbatim from upstream apps/AppsData.{hpp,cpp}. GPU kernel is the
// upstream Base_CUDA variant; CPU reference is the upstream Base_Seq variant,
// with upstream default problem size, reps, and data initialization.
// Validation: element-wise comparison of div.
//
#include "rp_common.hpp"
using Index_ptr = Index_type*;

// upstream apps/AppsData.hpp
#define NDSET2D(jp,v,v1,v2,v3,v4)  \
   v4 = v ;   \
   v1 = v4 + 1 ;  \
   v2 = v1 + jp ;  \
   v3 = v4 + jp ;

struct ADomain  // upstream apps/AppsData.hpp (2d path)
{
   ADomain( Index_type real_nodes_per_dim, Index_type ndims_ )
      : ndims(ndims_), NPNL(2), NPNR(1)
   {
      imin = NPNL;
      imax = NPNL + real_nodes_per_dim - 1;
      nnalls = (imax + 1 - imin + NPNL + NPNR);
      n_real_zones = (imax - imin);
      n_real_nodes = (imax + 1 - imin);
      jmin = NPNL;
      jmax = NPNL + real_nodes_per_dim - 1;
      jp = nnalls;
      nnalls *= (jmax + 1 - jmin + NPNL + NPNR);
      n_real_zones *= (jmax - jmin);
      n_real_nodes *= (jmax + 1 - jmin);
      kmin = 0; kmax = 0; kp = 0;
   }
   Index_type ndims, NPNL, NPNR;
   Index_type imin, jmin, kmin, imax, jmax, kmax;
   Index_type jp, kp, nnalls;
   Index_type n_real_zones, n_real_nodes;
};

// upstream apps/AppsData.cpp
static void setMeshPositions_2d(Real_ptr x, Real_type dx,
                                Real_ptr y, Real_type dy,
                                const ADomain& domain)
{
  Index_type imin = domain.imin, imax = domain.imax;
  Index_type jmin = domain.jmin, jmax = domain.jmax;
  Index_type jp = domain.jp;
  Index_type npnl = domain.NPNL, npnr = domain.NPNR;
  for (Index_type j = jmin - npnl; j < jmax+1 + npnr; j++) {
     for (Index_type i = imin - npnl; i < imax+1 + npnr; i++) {
        Index_type in = i + j*jp ;
        x[in] = i*dx;
        y[in] = j*dy;
     }
  }
}

static void setRealZones_2d(Index_type* real_zones, const ADomain& domain)
{
  Index_type imin = domain.imin, imax = domain.imax;
  Index_type jmin = domain.jmin, jmax = domain.jmax;
  Index_type jp = domain.jp;
  Index_type j_stride = (imax - imin);
  for (Index_type j = jmin; j < jmax; j++) {
     for (Index_type i = imin; i < imax; i++) {
        Index_type iz = i + j*jp ;
        Index_type il = (i-imin) + (j-jmin)*j_stride ;
        real_zones[il] = iz;
     }
  }
}

// upstream DEL_DOT_VEC_2D.hpp
#define DEL_DOT_VEC_2D_BODY_INDEX \
  Index_type i = real_zones[ii];

#define DEL_DOT_VEC_2D_BODY \
\
  Real_type xi  = half * ( x1[i]  + x2[i]  - x3[i]  - x4[i]  ) ; \
  Real_type xj  = half * ( x2[i]  + x3[i]  - x4[i]  - x1[i]  ) ; \
 \
  Real_type yi  = half * ( y1[i]  + y2[i]  - y3[i]  - y4[i]  ) ; \
  Real_type yj  = half * ( y2[i]  + y3[i]  - y4[i]  - y1[i]  ) ; \
 \
  Real_type fxi = half * ( fx1[i] + fx2[i] - fx3[i] - fx4[i] ) ; \
  Real_type fxj = half * ( fx2[i] + fx3[i] - fx4[i] - fx1[i] ) ; \
 \
  Real_type fyi = half * ( fy1[i] + fy2[i] - fy3[i] - fy4[i] ) ; \
  Real_type fyj = half * ( fy2[i] + fy3[i] - fy4[i] - fy1[i] ) ; \
 \
  Real_type rarea  = 1.0 / ( xi * yj - xj * yi + ptiny ) ; \
 \
  Real_type dfxdx  = rarea * ( fxi * yj - fxj * yi ) ; \
 \
  Real_type dfydy  = rarea * ( fyj * xi - fyi * xj ) ; \
 \
  Real_type affine = ( fy1[i] + fy2[i] + fy3[i] + fy4[i] ) / \
                     ( y1[i]  + y2[i]  + y3[i]  + y4[i]  ) ; \
 \
  div[i] = dfxdx + dfydy + affine ;

constexpr size_t block_size = 256;  // upstream default_gpu_block_size

// upstream DEL_DOT_VEC_2D-Cuda.cpp (Base_CUDA kernel)
__launch_bounds__(block_size)
__global__ void deldotvec2d(Real_ptr div,
                            const Real_ptr x1, const Real_ptr x2,
                            const Real_ptr x3, const Real_ptr x4,
                            const Real_ptr y1, const Real_ptr y2,
                            const Real_ptr y3, const Real_ptr y4,
                            const Real_ptr fx1, const Real_ptr fx2,
                            const Real_ptr fx3, const Real_ptr fx4,
                            const Real_ptr fy1, const Real_ptr fy2,
                            const Real_ptr fy3, const Real_ptr fy4,
                            const Index_ptr real_zones,
                            const Real_type half, const Real_type ptiny,
                            Index_type iend)
{
   Index_type ii = blockIdx.x * block_size + threadIdx.x;
   if (ii < iend) {
     DEL_DOT_VEC_2D_BODY_INDEX;
     DEL_DOT_VEC_2D_BODY;
   }
}

struct Data {
  Real_ptr x, y, xdot, ydot, div;
  Index_ptr real_zones;
  Real_type ptiny, half;
};

// upstream DEL_DOT_VEC_2D.cpp setUp (init order preserved)
static void setUp(Data& d, const ADomain& domain)
{
  resetDataInitCount();
  const Index_type array_length = domain.nnalls;
  allocAndInitDataConst(d.x, array_length, 0.0);
  allocAndInitDataConst(d.y, array_length, 0.0);
  allocAndInitDataConst(d.real_zones, domain.n_real_zones,
                        static_cast<Index_type>(-1));
  Real_type dx = 0.2;
  Real_type dy = 0.1;
  setMeshPositions_2d(d.x, dx, d.y, dy, domain);
  setRealZones_2d(d.real_zones, domain);
  allocAndInitData(d.xdot, array_length);
  allocAndInitData(d.ydot, array_length);
  allocAndInitDataConst(d.div, array_length, 0.0);
  d.ptiny = 1.0e-20;
  d.half = 0.5;
}

int main(int argc, char** argv)
{
  Index_type target_size = 1000*1000;  // upstream default problem size
  Index_type run_reps = 100;           // upstream default reps
  if (argc > 1) target_size = atol(argv[1]);
  if (argc > 2) run_reps = atol(argv[2]);
  // upstream setSize
  Index_type rzmax = std::sqrt((double)target_size) + 1 + std::sqrt(2.0) - 1;
  ADomain domain(rzmax, 2);
  const Index_type array_length = domain.nnalls;
  const Index_type iend = domain.n_real_zones;
  const size_t rbytes = array_length * sizeof(Real_type);
  const size_t zbytes = domain.n_real_zones * sizeof(Index_type);

  // ------------------------- GPU (upstream Base_CUDA) ----------------------
  Data g; setUp(g, domain);
  Real_ptr d_x, d_y, d_xdot, d_ydot, d_div; Index_ptr d_rz;
  GPU_CHECK(cudaMalloc(&d_x, rbytes));    GPU_CHECK(cudaMalloc(&d_y, rbytes));
  GPU_CHECK(cudaMalloc(&d_xdot, rbytes)); GPU_CHECK(cudaMalloc(&d_ydot, rbytes));
  GPU_CHECK(cudaMalloc(&d_div, rbytes));  GPU_CHECK(cudaMalloc(&d_rz, zbytes));
  GPU_CHECK(cudaMemcpy(d_x, g.x, rbytes, cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_y, g.y, rbytes, cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_xdot, g.xdot, rbytes, cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_ydot, g.ydot, rbytes, cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_div, g.div, rbytes, cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_rz, g.real_zones, zbytes, cudaMemcpyHostToDevice));

  {
    // upstream DEL_DOT_VEC_2D_DATA_SETUP (device pointers)
    Real_ptr x1,x2,x3,x4, y1,y2,y3,y4, fx1,fx2,fx3,fx4, fy1,fy2,fy3,fy4;
    NDSET2D(domain.jp, d_x, x1,x2,x3,x4) ;
    NDSET2D(domain.jp, d_y, y1,y2,y3,y4) ;
    NDSET2D(domain.jp, d_xdot, fx1,fx2,fx3,fx4) ;
    NDSET2D(domain.jp, d_ydot, fy1,fy2,fy3,fy4) ;
    for (Index_type irep = 0; irep < run_reps; ++irep) {
      const size_t grid_size = RP_DIVIDE_CEILING_INT(iend, (Index_type)block_size);
      deldotvec2d<<<grid_size, block_size>>>(d_div,
          x1, x2, x3, x4, y1, y2, y3, y4,
          fx1, fx2, fx3, fx4, fy1, fy2, fy3, fy4,
          d_rz, g.half, g.ptiny, iend);
    }
  }
  GPU_CHECK(cudaGetLastError());
  GPU_CHECK(cudaDeviceSynchronize());
  GPU_CHECK(cudaMemcpy(g.div, d_div, rbytes, cudaMemcpyDeviceToHost));
  cudaFree(d_x); cudaFree(d_y); cudaFree(d_xdot); cudaFree(d_ydot);
  cudaFree(d_div); cudaFree(d_rz);

  // ------------------------ CPU (upstream Base_Seq) ------------------------
  Data c; setUp(c, domain);
  {
    Real_ptr div = c.div;
    const Real_type ptiny = c.ptiny;
    const Real_type half = c.half;
    Real_ptr x1,x2,x3,x4, y1,y2,y3,y4, fx1,fx2,fx3,fx4, fy1,fy2,fy3,fy4;
    NDSET2D(domain.jp, c.x, x1,x2,x3,x4) ;
    NDSET2D(domain.jp, c.y, y1,y2,y3,y4) ;
    NDSET2D(domain.jp, c.xdot, fx1,fx2,fx3,fx4) ;
    NDSET2D(domain.jp, c.ydot, fy1,fy2,fy3,fy4) ;
    Index_ptr real_zones = c.real_zones;
    for (Index_type irep = 0; irep < run_reps; ++irep) {
      for (Index_type ii = 0; ii < iend; ++ii ) {
        DEL_DOT_VEC_2D_BODY_INDEX;
        DEL_DOT_VEC_2D_BODY;
      }
    }
  }

  // ------------------------------ validate ---------------------------------
  bool ok = compareArrays("div", c.div, g.div, array_length, 1.0e-10);
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
