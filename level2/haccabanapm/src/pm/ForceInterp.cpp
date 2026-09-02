// src/pm/ForceInterp.cpp
// Implements gather_force (diagnostic) and kick_with_grid (production) on top
// of Cabana::Grid::g2p with createScalarGradientG2P.  They share an internal
// templated helper, gather_grad_into, that runs the g2p once into whatever
// rank-2 destination is provided (a Kokkos::View<double*[3]> for diagnostics,
// or a Cabana::Slice<float[3]> for the production velocity update).
//
// HACC-aligned CIC convention (cic_convention_fix, 2026-05-23)
// ============================================================
// Mirrors the +0.5 grid-unit position shift in MassAssign.cpp's
// promote_positions.  The shift is REQUIRED to be applied symmetrically in
// scatter (MassAssign) AND gather (this file) — otherwise CIC charge
// conservation and Newton's third law break.  See MassAssign.cpp header
// and bit_identical_kick_result.md for the convention details, and
// prompt-08 retrospective for why N3L on the two-body test is the guard.

#include "ForceInterp.hpp"

#include <Cabana_Grid.hpp>

namespace pmk {

namespace {

// MUST equal the shift in MassAssign.cpp.  If you change one, change the
// other — N3L is the guard (tests/test_pm_step_two_body{,_spectral}).
constexpr double HACC_NODE_CENTERED_SHIFT = 0.5;

// Promote the FP32 position slice to a contiguous Kokkos::View<double*[3]>
// for the g2p call.  Applies the +0.5 HACC-alignment shift in the
// promotion so the gather indexes the same cells the scatter wrote into.
Kokkos::View<double * [3], MemorySpace>
promote_positions(Particles& particles)
{
    const std::size_t n = particles.num_local();
    Kokkos::View<double * [3], MemorySpace> points(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "pmk::g2p::points"),
        n);
    auto pos = particles.pos();
    constexpr double shift = HACC_NODE_CENTERED_SHIFT;
    Kokkos::parallel_for(
        "pmk::g2p::promote_positions",
        Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(const int p) {
            points(p, 0) = static_cast<double>(pos(p, 0)) + shift;
            points(p, 1) = static_cast<double>(pos(p, 1)) + shift;
            points(p, 2) = static_cast<double>(pos(p, 2)) + shift;
        });
    return points;
}

// Internal helper: run g2p with createScalarGradientG2P on grid.phi_real(),
// summing the result into `dst` with the supplied multiplier.  The signature
// is:
//   dst(p, d) += multiplier * d phi_real / d x_d
// Passing multiplier=-1.0 produces -grad(phi), matching HACC's
// force = -grad(phi) convention (spec §4).  Both gather_force and
// kick_with_grid call this; gather_force first zeros dst, kick_with_grid
// supplies the velocity slice directly (existing values + dt * (-grad)).
template <class DstView>
void gather_grad_into(const Grid& grid,
                      Particles& particles,
                      DstView dst,
                      typename DstView::value_type multiplier)
{
    auto points = promote_positions(particles);
    auto g2p_op = Cabana::Grid::createScalarGradientG2P(dst, multiplier);
    Cabana::Grid::g2p(
        ExecSpace{},
        *grid.phi_real(),
        *grid.phiRealHalo(),
        points,
        particles.num_local(),
        Cabana::Grid::Spline<1>(),
        g2p_op);
}

} // namespace

void gather_force(const Grid& grid,
                  Particles& particles,
                  Kokkos::View<double * [3], MemorySpace> force_out)
{
    // Tests rely on overwrite semantics; the g2p functor accumulates so we
    // zero force_out first.
    Kokkos::deep_copy(force_out, 0.0);
    gather_grad_into(grid, particles, force_out, -1.0);
}

void kick_with_grid(const Grid& grid, Particles& particles, double dt)
{
    // Velocity slice is a Cabana::Slice<float[3], ...>; createScalarGradientG2P
    // deduces value_type=float, so multiplier and dt are cast to float for the
    // accumulating add  vel(p, d) += -dt * d phi / d x_d.
    auto vel = particles.vel();
    gather_grad_into(grid, particles, vel, static_cast<float>(-dt));
}

// Per-axis CIC value gather using Cabana::Grid::createScalarValueG2P.
// grid.phi_real() must hold the real-space force component along `axis`,
// produced by apply_greens_and_gradient + inverse FFT + unpack_phi_real.
//
// We need a rank-1 view into the velocity component along `axis` for the
// scalar value functor (which static_asserts rank==1).  Allocate a temporary
// FP32 view "v_axis" of size num_local(), zero it, run g2p with
// multiplier = dt (the integrator passes τ·fscal as dt); after the gather,
// add v_axis(p) into vel(p, axis) on device.
void gather_force_value(const Grid& grid,
                        Particles& particles,
                        int axis,
                        double dt)
{
    const std::size_t n = particles.num_local();
    if (n == 0) return;

    auto points = promote_positions(particles);

    Kokkos::View<float*, MemorySpace> v_axis(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "pmk::gather_force_value::v_axis"),
        n);
    Kokkos::deep_copy(v_axis, 0.0f);

    auto g2p_op = Cabana::Grid::createScalarValueG2P(
        v_axis, static_cast<float>(dt));

    Cabana::Grid::g2p(
        ExecSpace{},
        *grid.phi_real(),
        *grid.phiRealHalo(),
        points,
        n,
        Cabana::Grid::Spline<1>(),
        g2p_op);

    // Add v_axis(p) into vel(p, axis).
    auto vel = particles.vel();
    const int ax = axis;
    Kokkos::parallel_for(
        "pmk::gather_force_value::accumulate",
        Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(const int p) {
            vel(p, ax) += v_axis(p);
        });
}

} // namespace pmk
