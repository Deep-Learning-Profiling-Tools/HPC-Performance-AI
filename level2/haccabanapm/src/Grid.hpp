#pragma once
// src/Grid.hpp
// PM mesh wrapper around Cabana::Grid: global/local grid, density (rho) and
// complex-potential (phi) Arrays, halo plans, and the heFFTe FFT plan.
//
// Per CABANA_VERIFICATION_v07_05c_local.md (sections 2, 4, 5):
//   - rho:  Cabana::Grid::Array<float,  Cell, ...>  with 1 dof  (FP32 mass mesh)
//   - phi:  Cabana::Grid::Array<double, Cell, ...>  with 2 dofs (FP64 complex)
//   - FFT is C2C only — phi is the FP64 complex buffer; rho must be packed in
//     before forward(), and the real part extracted after reverse().
//   - Aurora SYCL execution space auto-selects heffte::backend::onemkl.
//
// The cell-centered scalar field on a periodic uniform mesh maps to HACC's PM
// grid (HACC_PM_IC_PORTING_SPEC.md §4).
//
// On halo_width: the constructor default is 0, which is adequate only for
// single-rank use (the Green's function multiply is purely local in k-space,
// and Cabana::Grid::p2g/g2p carry their own halo exchange).  MULTI-RANK CIC
// REQUIRES halo_width >= 1 so the deposit's scatter and the gather's read
// have a ghost layer to work through.  Both production drivers
// (apps/pm_ic.cpp, apps/pm_run_lib.cpp) therefore construct with
// halo_width = 1 explicitly; do not rely on the default in new call sites.

#include "Types.hpp"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <array>
#include <memory>

namespace pmk {

class Grid
{
  public:
    using mesh_type    = Cabana::Grid::UniformMesh<double, 3>;
    using global_grid  = Cabana::Grid::GlobalGrid<mesh_type>;
    using local_grid   = Cabana::Grid::LocalGrid<mesh_type>;
    using rho_array_t  = Cabana::Grid::Array<float,  Cabana::Grid::Cell,
                                             mesh_type, MemorySpace>;
    using phi_array_t  = Cabana::Grid::Array<double, Cabana::Grid::Cell,
                                             mesh_type, MemorySpace>;
    using halo_t       = Cabana::Grid::Halo<MemorySpace>;
    // Concrete heFFTe FFT type — created via createHeffteFastFourierTransform
    // returns a shared_ptr<HeffteFastFourierTransform<...>>.  We hold it as a
    // shared_ptr<void> alias (auto-deduced concrete type) via the factory.
    using fft_t = decltype(
        Cabana::Grid::Experimental::createHeffteFastFourierTransform<
            double, MemorySpace>(
            std::declval<ExecSpace>(),
            std::declval<const Cabana::Grid::ArrayLayout<
                Cabana::Grid::Cell, mesh_type>&>(),
            std::declval<
                const Cabana::Grid::Experimental::FastFourierTransformParams&>()
        ));

    // Construct an ng^3 cubic periodic mesh on `comm`.
    //   ng         — global cells per dim
    //   halo_width — ghost-cell layers for rho/phi Arrays (0 is fine for the
    //                tests in this prompt; CIC P2G needs >=1 for multi-rank).
    Grid(MPI_Comm comm, int ng, int halo_width = 0);

    // Construct with an explicit (Px, Py, Pz) Cartesian block decomposition.
    // Required for HACC GIO comparability — HACC's MY_Dims_init_3D sets the
    // per-dim ranks from the indat TOPOLOGY string, and Cabana::Grid's default
    // DimBlockPartitioner runs MPI_Dims_create which can pick a different
    // ordering (it sorts factors descending, so 8 ranks → (2,2,2) which
    // happens to match, but 4 ranks → (2,2,1) where the unused-dim placement
    // is implementation-defined).  Passing the explicit dims uses
    // ManualBlockPartitioner and guarantees the topology matches.
    //
    // See ../analysis/references/hacc_ic_unit_conversions.md §5 for why the
    // GenericIO Coords[3] header tags rely on this Cartesian topology.
    Grid(MPI_Comm comm, int ng, int halo_width,
         const std::array<int, 3>& ranks_per_dim);

    // Accessors
    int ng() const { return m_ng; }
    int haloWidth() const { return m_halo_width; }
    MPI_Comm comm() const { return m_comm; }

    const std::shared_ptr<global_grid>& globalGrid() const { return m_global_grid; }
    const std::shared_ptr<local_grid>&  localGrid()  const { return m_local_grid;  }

    const std::shared_ptr<rho_array_t>& rho() const { return m_rho; }
    const std::shared_ptr<phi_array_t>& phi() const { return m_phi; }
    // FP32 cell-centered Array carrying the real component of phi after
    // the inverse FFT.  Populated by pm/PackUnpack::unpack_phi_real and read
    // by ForceInterp::g2p as the scalar field whose gradient is interpolated.
    const std::shared_ptr<rho_array_t>& phi_real() const { return m_phi_real; }

    const std::shared_ptr<halo_t>& rhoHalo()     const { return m_rho_halo; }
    const std::shared_ptr<halo_t>& phiHalo()     const { return m_phi_halo; }
    const std::shared_ptr<halo_t>& phiRealHalo() const { return m_phi_real_halo; }

    const fft_t& fft() const { return m_fft; }

  private:
    MPI_Comm m_comm;
    int m_ng;
    int m_halo_width;

    std::shared_ptr<global_grid>  m_global_grid;
    std::shared_ptr<local_grid>   m_local_grid;
    std::shared_ptr<rho_array_t>  m_rho;
    std::shared_ptr<phi_array_t>  m_phi;
    std::shared_ptr<rho_array_t>  m_phi_real;
    std::shared_ptr<halo_t>       m_rho_halo;
    std::shared_ptr<halo_t>       m_phi_halo;
    std::shared_ptr<halo_t>       m_phi_real_halo;
    fft_t                         m_fft;
};

} // namespace pmk
