// src/pm/GreensFunction.cpp
// HACC-equivalent k-space Green's function multiply.
// See GreensFunction.hpp for the convention vs HACC's solver.hpp:355-414.

#include "GreensFunction.hpp"

#include <Kokkos_MathematicalConstants.hpp>

namespace pmk {

void apply_greens_function(Grid::phi_array_t& phi)
{
    auto layout    = phi.layout();
    auto local     = layout->localGrid();
    auto& gg       = local->globalGrid();
    auto phi_view  = phi.view();
    auto own_space = local->indexSpace(
        Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local());

    // Convert local (i,j,k) to global wavenumber indices via per-dim offsets.
    const int N    = gg.globalNumEntity(Cabana::Grid::Cell(), 0);
    const int gi0  = gg.globalOffset(0);
    const int gj0  = gg.globalOffset(1);
    const int gk0  = gg.globalOffset(2);
    // local (i,j,k) is in Local frame — subtract halo to get owned-local index,
    // then add the rank's global offset.
    const int hw   = local->haloCellWidth();
    const double inv_N = 1.0 / static_cast<double>(N);
    constexpr double two_pi = 2.0 * Kokkos::numbers::pi_v<double>;

    // coeff = 0.5 from HACC solver.hpp:396; the 1/N^3 normalization is
    // delivered separately by the reverse FFT's FFTScaleFull().
    const double coeff = 0.5;

    Kokkos::parallel_for(
        "pmk::apply_greens_function",
        Cabana::Grid::createExecutionPolicy(own_space, ExecSpace{}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            const int gx = gi0 + (i - hw);
            const int gy = gj0 + (j - hw);
            const int gz = gk0 + (k - hw);

            // cos(2π k/N) handles Nyquist wrap automatically (cos is even and
            // 2π-periodic), matching HACC's plain `kk * kstep` evaluation.
            const double cx = Kokkos::cos(two_pi * gx * inv_N);
            const double cy = Kokkos::cos(two_pi * gy * inv_N);
            const double cz = Kokkos::cos(two_pi * gz * inv_N);
            const double denom = cx + cy + cz - 3.0;

            // Pole at the (0,0,0) global mode — HACC solver.hpp:411-413.
            const bool is_dc = (gx == 0 && gy == 0 && gz == 0);
            const double G   = is_dc ? 0.0 : coeff / denom;

            const double re = phi_view(i, j, k, 0);
            const double im = phi_view(i, j, k, 1);
            phi_view(i, j, k, 0) = G * re;
            phi_view(i, j, k, 1) = G * im;
        });
}

void apply_greens_and_gradient(Grid::phi_array_t& phi, int axis)
{
    auto layout    = phi.layout();
    auto local     = layout->localGrid();
    auto& gg       = local->globalGrid();
    auto phi_view  = phi.view();
    auto own_space = local->indexSpace(
        Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local());

    const int N    = gg.globalNumEntity(Cabana::Grid::Cell(), 0);
    const int gi0  = gg.globalOffset(0);
    const int gj0  = gg.globalOffset(1);
    const int gk0  = gg.globalOffset(2);
    const int hw   = local->haloCellWidth();
    const double inv_N = 1.0 / static_cast<double>(N);
    constexpr double two_pi = 2.0 * Kokkos::numbers::pi_v<double>;

    const double coeff = 0.5;  // G(k) prefactor (see apply_greens_function).
    const int    ax    = axis;

    Kokkos::parallel_for(
        "pmk::apply_greens_and_gradient",
        Cabana::Grid::createExecutionPolicy(own_space, ExecSpace{}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            const int gx = gi0 + (i - hw);
            const int gy = gj0 + (j - hw);
            const int gz = gk0 + (k - hw);

            const double cx = Kokkos::cos(two_pi * gx * inv_N);
            const double cy = Kokkos::cos(two_pi * gy * inv_N);
            const double cz = Kokkos::cos(two_pi * gz * inv_N);
            const double denom = cx + cy + cz - 3.0;

            const bool is_dc = (gx == 0 && gy == 0 && gz == 0);
            const double G   = is_dc ? 0.0 : coeff / denom;

            // m_gradient[k_axis] from HACC solver.hpp:391 — sin(k·kstep).
            const int gk_axis = (ax == 0) ? gx : (ax == 1 ? gy : gz);
            const double s = Kokkos::sin(two_pi * gk_axis * inv_N);

            // HACC solver.hpp:305-308 expanded out:
            //   c   = -m_gradient[k_axis] · m_green[index]
            //   re' = -c · im(rho)
            //   im' =  c · re(rho)
            // i.e. multiply by (-i · sin(2πk/N)) · G(k).
            const double c   = -s * G;
            const double re  = phi_view(i, j, k, 0);
            const double im  = phi_view(i, j, k, 1);
            phi_view(i, j, k, 0) = -c * im;
            phi_view(i, j, k, 1) =  c * re;
        });
}

} // namespace pmk
