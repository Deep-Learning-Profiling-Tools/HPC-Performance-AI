// src/ic/ICGreensFunction.cpp
// Continuum Green's function with ik-gradient.  Per-axis multiply applied
// in k-space to the unit-variance Gaussian field (output of WhiteNoise).
//
// HACC reference:
//   - solve_gravity:                Initializer.cxx:553-622
//   - Nyquist wrap (k_int_signed):  Initializer.cxx:566-583
//   - phase, Green, k=0 pole:       Initializer.cxx:584-608
//
// Convention note (prompt-vs-HACC discrepancy, recorded in
// ../analysis/prompt_issues/10b_issues.md):
//   The 10b prompt writes the kernel as
//     factor = (i k_axis)·(-1/k²)·sqrt(P_cb(k_phys))·ng^(-1.5)
//   without specifying the units of k_axis vs k².  Read literally with
//   k_axis = k_axis_phys = (2π/rL)·k_int, this produces displacement in
//   physical Mpc/h, which is inconsistent with the rest of pmkokkos's
//   grid-unit position convention (positions are stored in cells, not Mpc/h).
//   HACC's solve_gravity uses k_grid = (2π/ng)·k_int for both gradient and
//   Laplacian, producing grid-unit displacement.  We follow HACC.
//
//   Equivalently: if interpreted as physical-k everywhere, the prompt formula
//   misses an ng/rL grid-units conversion that would otherwise need to live
//   in CoordConvert.  HACC's grid-unit choice is what the rL^(-1.5) post-FFT
//   compensation in InitialConditions assumes.

#include "ICGreensFunction.hpp"

#include "PowerSpectrum.hpp"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <cmath>

namespace pmk {

void apply_ic_kernel(Grid& grid, int axis,
                     const PowerSpectrum& pk,
                     double rL, double ng_d,
                     double d_z, double omega_cb,
                     bool is_potential)
{
    // d_z and omega_cb are accepted for HACC-signature parity; they are
    // applied to the real-space potential by the caller (InitialConditions),
    // not here.  Suppress unused-warning when not in potential mode.
    (void)d_z;
    (void)omega_cb;

    auto local = grid.phi()->layout()->localGrid();
    auto& gg   = local->globalGrid();
    auto own_space = local->indexSpace(
        Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local());
    auto phi_view = grid.phi()->view();

    const int N      = static_cast<int>(ng_d + 0.5);
    const int hw     = local->haloCellWidth();
    const int gi0    = gg.globalOffset(0);
    const int gj0    = gg.globalOffset(1);
    const int gk0    = gg.globalOffset(2);
    const int N_half = N / 2;

    constexpr double TWO_PI = 6.283185307179586476925286766559;
    const double tpi_over_ng = TWO_PI / ng_d;
    const double tpi_over_rL = TWO_PI / rL;
    const double inv_ng_pow15 = 1.0 / std::pow(ng_d, 1.5);

    // -------------------------------------------------------------------
    // Build a log-k lookup table for sqrt(P_cb) once per call.
    //
    // P_cb is host-only (binary search over std::vector inside
    // TransferFunction).  We pre-tabulate sqrt(P) on a log-k grid spanning
    // the transfer-function file's range, deep_copy to device, and the
    // kernel does a linear interpolation in log(k) per cell — saturated at
    // the table boundaries (the IC kernel only ever queries k in
    // [k_fund, k_Nyq] which lies well inside the cmb.tf range).
    // -------------------------------------------------------------------
    constexpr int TABLE_N = 4096;
    const double k_lo     = std::max(pk.k_min(), 1.0e-10);
    const double k_hi     = std::max(pk.k_max(), k_lo * 10.0);
    const double log_k_lo = std::log(k_lo);
    const double log_k_hi = std::log(k_hi);
    const double dlogk    = (log_k_hi - log_k_lo) / static_cast<double>(TABLE_N - 1);
    const double inv_dlogk = 1.0 / dlogk;

    Kokkos::View<double*, HostSpace> sqrtP_host(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "sqrtP_host"),
        TABLE_N);
    for (int i = 0; i < TABLE_N; ++i) {
        const double k_i = std::exp(log_k_lo + i * dlogk);
        const double P   = pk.P_cb(k_i);
        sqrtP_host(i) = (P > 0.0) ? std::sqrt(P) : 0.0;
    }
    Kokkos::View<double*, MemorySpace> sqrtP_dev(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "sqrtP_dev"),
        TABLE_N);
    Kokkos::deep_copy(sqrtP_dev, sqrtP_host);

    // Capture-by-value scalars for the lambda.
    const double tpi_over_ng_l = tpi_over_ng;
    const double tpi_over_rL_l = tpi_over_rL;
    const double inv_ng_pow15_l = inv_ng_pow15;
    const double log_k_lo_l = log_k_lo;
    const double inv_dlogk_l = inv_dlogk;
    const int    table_n_l   = TABLE_N;
    const int    hw_l        = hw;
    const int    gi0_l       = gi0;
    const int    gj0_l       = gj0;
    const int    gk0_l       = gk0;
    const int    N_l         = N;
    const int    N_half_l    = N_half;
    const int    axis_l      = axis;
    const bool   is_pot_l    = is_potential;

    Kokkos::parallel_for(
        "pmk::apply_ic_kernel",
        Cabana::Grid::createExecutionPolicy(own_space, ExecSpace{}),
        KOKKOS_LAMBDA(const int i, const int j, const int k_idx) {
            // Local (with halo) → owned-local → global (unsigned), then
            // wrap into signed wavenumber index in [-N/2, N/2).  HACC
            // Initializer.cxx:566-583.
            const int gxu = gi0_l + (i     - hw_l);
            const int gyu = gj0_l + (j     - hw_l);
            const int gzu = gk0_l + (k_idx - hw_l);
            const int kx_int = (gxu >= N_half_l) ? gxu - N_l : gxu;
            const int ky_int = (gyu >= N_half_l) ? gyu - N_l : gyu;
            const int kz_int = (gzu >= N_half_l) ? gzu - N_l : gzu;
            const double k2_int = static_cast<double>(
                kx_int*kx_int + ky_int*ky_int + kz_int*kz_int);

            const double re = phi_view(i, j, k_idx, 0);
            const double im = phi_view(i, j, k_idx, 1);

            // k = 0 mode: HACC Initializer.cxx:606-608.
            if (k2_int == 0.0) {
                phi_view(i, j, k_idx, 0) = 0.0;
                phi_view(i, j, k_idx, 1) = 0.0;
                return;
            }

            // sqrt(P_cb(k_phys)) via log-k linear interp.
            const double k_phys = tpi_over_rL_l * Kokkos::sqrt(k2_int);
            double sqrtP;
            const double u   = Kokkos::log(k_phys);
            const double idx = (u - log_k_lo_l) * inv_dlogk_l;
            if (idx <= 0.0) {
                sqrtP = sqrtP_dev(0);
            } else if (idx >= static_cast<double>(table_n_l - 1)) {
                sqrtP = sqrtP_dev(table_n_l - 1);
            } else {
                const int i0 = static_cast<int>(idx);
                const double t = idx - static_cast<double>(i0);
                sqrtP = sqrtP_dev(i0) * (1.0 - t) + sqrtP_dev(i0 + 1) * t;
            }

            // HACC convention: gradient and Green's in grid-unit wavenumbers
            //   k_grid = (2π/ng) · k_int   (HACC Initializer.cxx:584-591)
            //   Green  = -1 / k_grid²
            // For axis 0,1,2: combined operator is (i · k_grid_axis) · Green.
            // For axis 3 (potential):    operator is Green only.
            const double k_grid2 = (tpi_over_ng_l * tpi_over_ng_l) * k2_int;
            const double Green   = -1.0 / k_grid2;
            const double base    = sqrtP * inv_ng_pow15_l * Green;

            if (is_pot_l) {
                phi_view(i, j, k_idx, 0) = base * re;
                phi_view(i, j, k_idx, 1) = base * im;
            } else {
                int k_int_axis;
                if (axis_l == 0)      k_int_axis = kx_int;
                else if (axis_l == 1) k_int_axis = ky_int;
                else                  k_int_axis = kz_int;
                const double k_grid_axis =
                    tpi_over_ng_l * static_cast<double>(k_int_axis);
                // Multiply by i·k_grid_axis:  (re, im) → (-im, re) · k_grid_axis.
                const double s = base * k_grid_axis;
                phi_view(i, j, k_idx, 0) = -s * im;
                phi_view(i, j, k_idx, 1) =  s * re;
            }
        });
    Kokkos::fence();
}

} // namespace pmk
