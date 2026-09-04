// src/pm/PMStep.cpp
// End-to-end PM step.  Two pipelines coexist:
//
//   - LEGACY DIAGNOSTIC: poisson_solve_potential() — one forward + one
//     reverse FFT, computing the real-space potential phi(x), used by
//     compute_forces with createScalarGradientG2P.  Retained because the
//     prompt-08 acceptance test (test_pm_step_two_body) was written
//     against this exact pipeline and references it directly.  Documented
//     to be lower fidelity at intermediate k — see
//     analysis/references/pk_discrepancy_diagnosis.md §3c.
//
//   - PRODUCTION (post pm-gradient-fix): the three-FFT pipeline
//     implemented in pm_kick below.  After one forward FFT of rho, the
//     k-space density is saved into a shadow buffer; then for each axis
//     the saved rho_k is restored into the working phi array,
//     apply_greens_and_gradient multiplies in-place by G(k)·(-i sin(2π
//     k_axis/N)), the reverse FFT produces the real-space force component
//     along that axis, and gather_force_value (CIC value interpolation)
//     adds the kick to that velocity component.
//
//   Memory cost of the spectral pipeline:  one extra k-space buffer of
//   size ng^3 · 2 doubles per rank.  At ng=128 with 8 ranks per node this
//   is 128^3 · 16 / 8 = 4 MB per rank — negligible.  At ng=1024 it grows
//   to ~256 MB per rank, still well under typical PVC HBM.  This mirrors
//   HACC's approach (HACC does one forward FFT per kick, then 3 reverses).

#include "PMStep.hpp"

#include "MassAssign.hpp"
#include "PackUnpack.hpp"
#include "GreensFunction.hpp"
#include "ForceInterp.hpp"

#include <Cabana_Grid.hpp>

namespace pmk {

namespace {

// LEGACY: single-FFT pipeline used by compute_forces / kick_with_grid.
// FFT scaling convention:
//   forward(FFTScaleNone)  — no scaling on the forward transform
//   reverse(FFTScaleFull)  — 1/N^3 normalization, matching HACC end-to-end
//                            (see GreensFunction.hpp).
void poisson_solve_potential(Grid& grid)
{
    pack_rho_to_complex(*grid.rho(), *grid.phi());
    grid.fft()->forward(*grid.phi(),
                        Cabana::Grid::Experimental::FFTScaleNone());
    apply_greens_function(*grid.phi());
    grid.fft()->reverse(*grid.phi(),
                        Cabana::Grid::Experimental::FFTScaleFull());
    unpack_phi_real(*grid.phi(), *grid.phi_real());
}

} // namespace

void compute_forces(Particles& particles,
                    Grid& grid,
                    double uniform_mass,
                    Kokkos::View<double * [3], MemorySpace> force_out)
{
    // LEGACY DIAGNOSTIC PATH — exists for the prompt-08 two-body test, which
    // pinned the original CIC-gradient-of-CIC-basis scheme.  Production
    // gravity goes through pm_kick (the three-FFT spectral pipeline).
    deposit(particles, grid, uniform_mass);
    poisson_solve_potential(grid);
    gather_force(grid, particles, force_out);
}

void pm_kick(Particles& particles,
             Grid& grid,
             double uniform_mass,
             double dt)
{
    // ---- 1. Compute rho on the mesh, transform to k-space ----
    deposit(particles, grid, uniform_mass);
    pack_rho_to_complex(*grid.rho(), *grid.phi());
    grid.fft()->forward(*grid.phi(),
                        Cabana::Grid::Experimental::FFTScaleNone());

    // ---- 2. Save rho_k so each axis iteration can multiply by its own
    //          (-i sin(2π k_axis/N))·G(k) factor without recomputing the
    //          forward FFT (cf. HACC mc3.cxx:626-663) ----
    auto phi_view = grid.phi()->view();
    Kokkos::View<double****, MemorySpace> rho_k_save(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "pmk::pm_kick::rho_k_save"),
        phi_view.extent(0), phi_view.extent(1),
        phi_view.extent(2), phi_view.extent(3));
    Kokkos::deep_copy(rho_k_save, phi_view);

    // ---- 3. Per-axis: restore rho_k, multiply by G(k)·(-i sin(2π k/N)),
    //          inverse FFT, halo exchange, CIC value gather into vel[axis] ----
    for (int axis = 0; axis < 3; ++axis)
    {
        Kokkos::deep_copy(phi_view, rho_k_save);

        apply_greens_and_gradient(*grid.phi(), axis);

        grid.fft()->reverse(*grid.phi(),
                            Cabana::Grid::Experimental::FFTScaleFull());

        unpack_phi_real(*grid.phi(), *grid.phi_real());

        // Multi-rank: the gradient field needs its halo before the value
        // interpolation reads bracketing ghost cells.  Cabana::Grid::g2p
        // calls halo.gather() internally before the per-particle loop
        // (Cabana_Grid_Interpolation.hpp:855), so we don't repeat it here.
        gather_force_value(grid, particles, axis, dt);
    }
}

} // namespace pmk
