//
// MATVEC_3D_STENCIL benchmark, extracted standalone from the RAJA Performance
// Suite (src/apps/MATVEC_3D_STENCIL.*, BSD-3-Clause). 27-point stencil
// matrix-vector product over an indirection (real-zone) list on a 3D mesh.
// ADomain and setRealZones_3d are ported verbatim from apps/AppsData.{hpp,cpp}.
// GPU kernel is the upstream Base_CUDA variant (pointer-alias structure
// preserved); CPU reference is the upstream Base_Seq variant, with upstream
// default problem size, reps, and data initialization.
// Validation: element-wise comparison of b.
//
#include "rp_common.hpp"
using Index_ptr = Index_type*;

struct ADomain  // upstream apps/AppsData.hpp (3d path)
{
   ADomain( Index_type real_nodes_per_dim, Index_type ndims_ )
      : ndims(ndims_), NPNL(2), NPNR(1)
   {
      int NPZL = NPNL - 1;
      int NPZR = NPNR+1 - 1;
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
      kmin = NPNL;
      kmax = NPNL + real_nodes_per_dim - 1;
      kp = nnalls;
      nnalls *= (kmax + 1 - kmin + NPNL + NPNR);
      n_real_zones *= (kmax - kmin);
      n_real_nodes *= (kmax + 1 - kmin);
      fpz = (kmin - NPZL)*kp + (jmin - NPZL)*jp + (imin - NPZL);
      lpz = (kmax-1 + NPZR)*kp + (jmax-1 + NPZR)*jp + (imax-1 + NPZR);
   }
   Index_type ndims, NPNL, NPNR;
   Index_type imin, jmin, kmin, imax, jmax, kmax;
   Index_type jp, kp, nnalls;
   Index_type fpz, lpz;
   Index_type n_real_zones, n_real_nodes;
};

// upstream apps/AppsData.cpp
static void setRealZones_3d(Index_type* real_zones, const ADomain& domain)
{
  Index_type imin = domain.imin, imax = domain.imax;
  Index_type jmin = domain.jmin, jmax = domain.jmax;
  Index_type kmin = domain.kmin, kmax = domain.kmax;
  Index_type jp = domain.jp, kp = domain.kp;
  Index_type j_stride = (imax - imin);
  Index_type k_stride = j_stride * (jmax - jmin);
  for (Index_type k = kmin; k < kmax; k++) {
     for (Index_type j = jmin; j < jmax; j++) {
        for (Index_type i = imin; i < imax; i++) {
           Index_type iz = i + j*jp + k*kp ;
           Index_type il = (i-imin) + (j-jmin)*j_stride + (k-kmin)*k_stride ;
           real_zones[il] = iz;
        }
     }
  }
}

// The 27 stencil position names (upstream naming), and the 14 base matrix
// arrays that upstream allocates (the other 13 are pointer aliases into them).
#define MV3D_POS(OP) \
  OP(dbl) OP(dbc) OP(dbr) OP(dcl) OP(dcc) OP(dcr) OP(dfl) OP(dfc) OP(dfr) \
  OP(cbl) OP(cbc) OP(cbr) OP(ccl) OP(ccc) OP(ccr) OP(cfl) OP(cfc) OP(cfr) \
  OP(ubl) OP(ubc) OP(ubr) OP(ucl) OP(ucc) OP(ucr) OP(ufl) OP(ufc) OP(ufr)
#define MV3D_ALLOC(OP) \
  OP(dbl) OP(dbc) OP(dbr) OP(dcl) OP(dcc) OP(dcr) OP(dfl) OP(dfc) OP(dfr) \
  OP(cbl) OP(cbc) OP(cbr) OP(ccl) OP(ccc)

// upstream MATVEC_3D_STENCIL.hpp
#define MATVEC_3D_STENCIL_BODY_INDEX \
  Index_type i = real_zones[ii];

#define MATVEC_3D_STENCIL_BODY \
  b[i] = dbl[i] * xdbl[i] + dbc[i] * xdbc[i] + dbr[i] * xdbr[i] + \
         dcl[i] * xdcl[i] + dcc[i] * xdcc[i] + dcr[i] * xdcr[i] + \
         dfl[i] * xdfl[i] + dfc[i] * xdfc[i] + dfr[i] * xdfr[i] + \
                                                                  \
         cbl[i] * xcbl[i] + cbc[i] * xcbc[i] + cbr[i] * xcbr[i] + \
         ccl[i] * xccl[i] + ccc[i] * xccc[i] + ccr[i] * xccr[i] + \
         cfl[i] * xcfl[i] + cfc[i] * xcfc[i] + cfr[i] * xcfr[i] + \
                                                                  \
         ubl[i] * xubl[i] + ubc[i] * xubc[i] + ubr[i] * xubr[i] + \
         ucl[i] * xucl[i] + ucc[i] * xucc[i] + ucr[i] * xucr[i] + \
         ufl[i] * xufl[i] + ufc[i] * xufc[i] + ufr[i] * xufr[i] ;

constexpr size_t block_size = 256;  // upstream default_gpu_block_size

// upstream MATVEC_3D_STENCIL-Cuda.cpp (Base_CUDA kernel; same 55-pointer
// argument structure, expressed via the stencil-name macro list)
#define ARG_X(n) Real_ptr x##n,
#define ARG_M(n) Real_ptr n,
__launch_bounds__(block_size)
__global__ void matvec_3d(Real_ptr b,
                          MV3D_POS(ARG_X)
                          MV3D_POS(ARG_M)
                          Index_ptr real_zones,
                          Index_type iend)
{
  Index_type ii = blockIdx.x * block_size + threadIdx.x;
  if (ii < iend) {
    MATVEC_3D_STENCIL_BODY_INDEX;
    MATVEC_3D_STENCIL_BODY;
  }
}
#undef ARG_X
#undef ARG_M

struct Ptrs {  // 27 stencil pointers for x and for the matrix
#define DECL(n) Real_ptr x##n; Real_ptr n;
  MV3D_POS(DECL)
#undef DECL
};

// upstream MATVEC_3D_STENCIL_DATA_SETUP pointer offsets
static void set_x_aliases(Ptrs& p, Real_ptr x, Index_type jp, Index_type kp)
{
  p.xdbl = x - kp - jp - 1; p.xdbc = x - kp - jp; p.xdbr = x - kp - jp + 1;
  p.xdcl = x - kp - 1;      p.xdcc = x - kp;      p.xdcr = x - kp + 1;
  p.xdfl = x - kp + jp - 1; p.xdfc = x - kp + jp; p.xdfr = x - kp + jp + 1;
  p.xcbl = x - jp - 1;      p.xcbc = x - jp;      p.xcbr = x - jp + 1;
  p.xccl = x - 1;           p.xccc = x;           p.xccr = x + 1;
  p.xcfl = x + jp - 1;      p.xcfc = x + jp;      p.xcfr = x + jp + 1;
  p.xubl = x + kp - jp - 1; p.xubc = x + kp - jp; p.xubr = x + kp - jp + 1;
  p.xucl = x + kp - 1;      p.xucc = x + kp;      p.xucr = x + kp + 1;
  p.xufl = x + kp + jp - 1; p.xufc = x + kp + jp; p.xufr = x + kp + jp + 1;
}

// upstream MATVEC_3D_STENCIL.cpp setUp matrix aliasing
static void set_matrix_aliases(Ptrs& p, Index_type jp, Index_type kp)
{
  p.ccr = p.ccl + 1;
  p.cfl = p.cbr - 1 + jp;
  p.cfc = p.cbc     + jp;
  p.cfr = p.cbl + 1 + jp;
  p.ubl = p.dfr - 1 - jp + kp;
  p.ubc = p.dfc     - jp + kp;
  p.ubr = p.dfl + 1 - jp + kp;
  p.ucl = p.dcr - 1      + kp;
  p.ucc = p.dcc          + kp;
  p.ucr = p.dcl + 1      + kp;
  p.ufl = p.dbr - 1 + jp + kp;
  p.ufc = p.dbc     + jp + kp;
  p.ufr = p.dbl + 1 + jp + kp;
}

int main(int argc, char** argv)
{
  Index_type target_size = 100*100*100;  // upstream default problem size
  Index_type run_reps = 100;             // upstream default reps
  if (argc > 1) target_size = atol(argv[1]);
  if (argc > 2) run_reps = atol(argv[2]);
  // upstream setSize
  Index_type rzmax = std::cbrt((double)target_size) + 1 + std::cbrt(3.0) - 1;
  ADomain domain(rzmax, 3);
  const Index_type len = domain.lpz + 1;      // m_zonal_array_length
  const Index_type iend = domain.n_real_zones;
  const size_t bytes = len * sizeof(Real_type);
  const size_t zbytes = iend * sizeof(Index_type);

  // ---------------- host data (upstream setUp; init order preserved) -------
  auto host_setup = [&](Real_ptr& b, Real_ptr& x, Ptrs& m, Index_ptr& rz) {
    resetDataInitCount();
    allocAndInitDataConst(b, len, 0.0);
    allocAndInitData(x, len);
#define AINIT(n) allocAndInitData(m.n, len);
    MV3D_ALLOC(AINIT)
#undef AINIT
    set_matrix_aliases(m, domain.jp, domain.kp);
    allocAndInitDataConst(rz, iend, static_cast<Index_type>(-1));
    setRealZones_3d(rz, domain);
  };

  // ------------------------- GPU (upstream Base_CUDA) ----------------------
  Real_ptr b, x; Ptrs m; Index_ptr rz;
  host_setup(b, x, m, rz);

  Real_ptr d_b, d_x; Ptrs dm; Index_ptr d_rz;
  GPU_CHECK(cudaMalloc(&d_b, bytes));
  GPU_CHECK(cudaMalloc(&d_x, bytes));
  GPU_CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice));
#define DALLOC(n) GPU_CHECK(cudaMalloc(&dm.n, bytes)); \
  GPU_CHECK(cudaMemcpy(dm.n, m.n, bytes, cudaMemcpyHostToDevice));
  MV3D_ALLOC(DALLOC)
#undef DALLOC
  set_matrix_aliases(dm, domain.jp, domain.kp);
  set_x_aliases(dm, d_x, domain.jp, domain.kp);
  GPU_CHECK(cudaMalloc(&d_rz, zbytes));
  GPU_CHECK(cudaMemcpy(d_rz, rz, zbytes, cudaMemcpyHostToDevice));

#define PASS_X(n) dm.x##n,
#define PASS_M(n) dm.n,
  for (Index_type irep = 0; irep < run_reps; ++irep) {
    const size_t grid_size = RP_DIVIDE_CEILING_INT(iend, (Index_type)block_size);
    matvec_3d<<<grid_size, block_size>>>(d_b,
        MV3D_POS(PASS_X)
        MV3D_POS(PASS_M)
        d_rz, iend);
  }
#undef PASS_X
#undef PASS_M
  GPU_CHECK(cudaGetLastError());
  GPU_CHECK(cudaDeviceSynchronize());
  GPU_CHECK(cudaMemcpy(b, d_b, bytes, cudaMemcpyDeviceToHost));
  cudaFree(d_b); cudaFree(d_x); cudaFree(d_rz);
#define DFREE(n) cudaFree(dm.n);
  MV3D_ALLOC(DFREE)
#undef DFREE

  // ------------------------ CPU (upstream Base_Seq) ------------------------
  Real_ptr b_r, x_r; Ptrs mr; Index_ptr rz_r;
  host_setup(b_r, x_r, mr, rz_r);
  set_x_aliases(mr, x_r, domain.jp, domain.kp);
  {
    Real_ptr b = b_r;
#define LOCAL_X(n) Real_ptr x##n = mr.x##n;
#define LOCAL_M(n) Real_ptr n = mr.n;
    MV3D_POS(LOCAL_X)
    MV3D_POS(LOCAL_M)
#undef LOCAL_X
#undef LOCAL_M
    Index_ptr real_zones = rz_r;
    for (Index_type irep = 0; irep < run_reps; ++irep) {
      for (Index_type ii = 0; ii < iend; ++ii ) {
        MATVEC_3D_STENCIL_BODY_INDEX;
        MATVEC_3D_STENCIL_BODY;
      }
    }
  }

  // ------------------------------ validate ---------------------------------
  bool ok = compareArrays("b", b_r, b, len, 1.0e-10);
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
