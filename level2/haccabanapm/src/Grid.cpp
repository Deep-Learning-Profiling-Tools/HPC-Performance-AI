// src/Grid.cpp
// Grid wrapper construction.  Mirrors the example pattern from
//   Cabana/example/grid_tutorial/10_fft_heffte/
//     heffte_fast_fourier_transform_example.cpp
// and 15_interpolation/interpolation_example.cpp.
//
// The cell width is set to 1.0 — pmkokkos works in HACC code units where
// position is measured in grid cells (CodeUnits.hpp:phys2grid_pos = ng/rL),
// so the underlying mesh extents are [0, ng) per dim with cell_size=1.

#include "Grid.hpp"

namespace pmk {

namespace {

// Common construction body parameterized by the Cabana partitioner type.
// Templated so we can reuse it for both DimBlockPartitioner (auto pick) and
// ManualBlockPartitioner (HACC-matching explicit dims).
template <typename Partitioner>
void construct_grid_impl(
    MPI_Comm comm,
    int ng,
    int halo_width,
    const Partitioner& partitioner,
    std::shared_ptr<Cabana::Grid::GlobalGrid<Cabana::Grid::UniformMesh<double, 3>>>& global_grid_out,
    std::shared_ptr<Cabana::Grid::LocalGrid <Cabana::Grid::UniformMesh<double, 3>>>& local_grid_out,
    std::shared_ptr<Cabana::Grid::Array<float,  Cabana::Grid::Cell,
                                        Cabana::Grid::UniformMesh<double, 3>,
                                        MemorySpace>>& rho_out,
    std::shared_ptr<Cabana::Grid::Array<double, Cabana::Grid::Cell,
                                        Cabana::Grid::UniformMesh<double, 3>,
                                        MemorySpace>>& phi_out,
    std::shared_ptr<Cabana::Grid::Array<float,  Cabana::Grid::Cell,
                                        Cabana::Grid::UniformMesh<double, 3>,
                                        MemorySpace>>& phi_real_out,
    std::shared_ptr<Cabana::Grid::Halo<MemorySpace>>& rho_halo_out,
    std::shared_ptr<Cabana::Grid::Halo<MemorySpace>>& phi_halo_out,
    std::shared_ptr<Cabana::Grid::Halo<MemorySpace>>& phi_real_halo_out,
    Grid::fft_t& fft_out)
{
    const std::array<double, 3> low_corner  = {0.0, 0.0, 0.0};
    const std::array<double, 3> high_corner = {static_cast<double>(ng),
                                               static_cast<double>(ng),
                                               static_cast<double>(ng)};
    const double cell_size = 1.0;

    auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
        low_corner, high_corner, cell_size);

    const std::array<bool, 3> is_periodic = {true, true, true};
    global_grid_out = Cabana::Grid::createGlobalGrid(
        comm, global_mesh, is_periodic, partitioner);

    local_grid_out = Cabana::Grid::createLocalGrid(global_grid_out, halo_width);

    auto rho_layout = Cabana::Grid::createArrayLayout(
        local_grid_out, /*dofs=*/1, Cabana::Grid::Cell());
    rho_out = Cabana::Grid::createArray<float, MemorySpace>("rho", rho_layout);

    auto phi_layout = Cabana::Grid::createArrayLayout(
        local_grid_out, /*dofs=*/2, Cabana::Grid::Cell());
    phi_out = Cabana::Grid::createArray<double, MemorySpace>("phi", phi_layout);

    phi_real_out = Cabana::Grid::createArray<float, MemorySpace>(
        "phi_real", rho_layout);

    rho_halo_out = Cabana::Grid::createHalo(
        Cabana::Grid::NodeHaloPattern<3>(), halo_width, *rho_out);
    phi_halo_out = Cabana::Grid::createHalo(
        Cabana::Grid::NodeHaloPattern<3>(), halo_width, *phi_out);
    phi_real_halo_out = Cabana::Grid::createHalo(
        Cabana::Grid::NodeHaloPattern<3>(), halo_width, *phi_real_out);

    Cabana::Grid::Experimental::FastFourierTransformParams params;
    // HPC-Performance-AI: Cabana 0.8.0 renamed setAllToAll(bool) to
    // setAlltoAll(bool) (true -> heFFTe alltoallv, same as 0.7.0's
    // setAllToAll(true)); upstream targets Cabana 0.7.0.
    params.setAlltoAll(true);
    params.setPencils(true);
    params.setReorder(true);

    fft_out = Cabana::Grid::Experimental::createHeffteFastFourierTransform<
        double, MemorySpace>(ExecSpace{}, *phi_layout, params);
}

} // namespace

Grid::Grid(MPI_Comm comm, int ng, int halo_width)
    : m_comm(comm), m_ng(ng), m_halo_width(halo_width)
{
    Cabana::Grid::DimBlockPartitioner<3> partitioner;
    construct_grid_impl(comm, ng, halo_width, partitioner,
                        m_global_grid, m_local_grid,
                        m_rho, m_phi, m_phi_real,
                        m_rho_halo, m_phi_halo, m_phi_real_halo,
                        m_fft);
}

Grid::Grid(MPI_Comm comm, int ng, int halo_width,
           const std::array<int, 3>& ranks_per_dim)
    : m_comm(comm), m_ng(ng), m_halo_width(halo_width)
{
    Cabana::Grid::ManualBlockPartitioner<3> partitioner(ranks_per_dim);
    construct_grid_impl(comm, ng, halo_width, partitioner,
                        m_global_grid, m_local_grid,
                        m_rho, m_phi, m_phi_real,
                        m_rho_halo, m_phi_halo, m_phi_real_halo,
                        m_fft);
}

} // namespace pmk
