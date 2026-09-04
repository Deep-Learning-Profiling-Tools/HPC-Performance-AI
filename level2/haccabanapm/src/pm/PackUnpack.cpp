// src/pm/PackUnpack.cpp
// Owned-cell pack/unpack between rho (FP32, 1 dof) and phi (FP64 complex, 2 dofs).

#include "PackUnpack.hpp"

namespace pmk {

void pack_rho_to_complex(const Grid::rho_array_t& rho, Grid::phi_array_t& phi)
{
    auto rho_view = rho.view();
    auto phi_view = phi.view();
    auto owned = rho.layout()->localGrid()->indexSpace(
        Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local());

    Kokkos::parallel_for(
        "pmk::pack_rho_to_complex",
        Cabana::Grid::createExecutionPolicy(owned, ExecSpace{}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            phi_view(i, j, k, 0) = static_cast<double>(rho_view(i, j, k, 0));
            phi_view(i, j, k, 1) = 0.0;
        });
}

void unpack_phi_real(const Grid::phi_array_t& phi, Grid::rho_array_t& dst)
{
    auto phi_view = phi.view();
    auto dst_view = dst.view();
    auto owned = dst.layout()->localGrid()->indexSpace(
        Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local());

    Kokkos::parallel_for(
        "pmk::unpack_phi_real",
        Cabana::Grid::createExecutionPolicy(owned, ExecSpace{}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            dst_view(i, j, k, 0) = static_cast<float>(phi_view(i, j, k, 0));
        });
}

} // namespace pmk
