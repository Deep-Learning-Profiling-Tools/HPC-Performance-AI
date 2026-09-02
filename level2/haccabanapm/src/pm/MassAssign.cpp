// src/pm/MassAssign.cpp
// CIC mass deposit using Cabana::Grid::p2g with the custom UniformMassP2G
// functor.  See MassAssign.hpp for the design choices.
//
// HACC-aligned CIC convention (cic_convention_fix, 2026-05-23)
// ============================================================
// Cabana::Grid::Spline<1> on a Cell entity internally subtracts 0.5 from
// the input position (mapToLogicalGrid: `(xp - low_x) * rdx` with
// low_x_cell0 = 0.5).  HACC's `ParticleActions::cic`
// (HACC/nbody/cpu/ParticleActions.cxx:1525-1540) does `ix = (int)FLOOR(x)`
// with NO half-cell subtraction — i.e. node-centered indexing.
//
// To match HACC's stencil indices and weights exactly, we shift the
// particle position by +0.5 grid units before handing it to p2g; the +0.5
// cancels Cabana's internal -0.5, producing HACC's node-centered behavior.
// E.g. particle at GRID 4.25: HACC ix=4, weights (0.75, 0.25) at (4, 5).
// Cabana with shift: logical = (4.25+0.5)-0.5 = 4.25, floor=4, frac=0.25,
// weights (1-frac, frac) = (0.75, 0.25) at cells (4, 5).  Match.
//
// THE SHIFT IS PAIRED WITH ForceInterp.cpp's gather (gather_force_value
// and gather_grad_into both apply the same +0.5 shift).  This pairing is
// REQUIRED for Newton's third law (CIC charge conservation under the
// scatter-then-gather composition): an asymmetric shift breaks momentum
// conservation and the two-body N3L tests fail.  See prompt-08
// retrospective and bit_identical_kick_result.md.

#include "MassAssign.hpp"

#include <Cabana_Grid.hpp>

namespace pmk {

namespace {

// Half-cell shift to align Cabana's Cell-entity CIC stencil with HACC's
// node-centered convention (see file header).  MUST equal the shift
// applied symmetrically in ForceInterp.cpp's promote_positions for N3L.
constexpr double HACC_NODE_CENTERED_SHIFT = 0.5;

// Promote the FP32 position slice (Cabana SoA layout) to a contiguous
// Kokkos::View<double*[3]> on device, suitable for Cabana::Grid::p2g.
// Applies the +0.5 HACC-alignment shift in the promotion so the shifted
// position is what Cabana's spline indexes from.
Kokkos::View<double * [3], MemorySpace>
promote_positions(Particles& particles)
{
    const std::size_t n = particles.num_local();
    Kokkos::View<double * [3], MemorySpace> points(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "pmk::deposit::points"),
        n);

    auto pos = particles.pos();
    constexpr double shift = HACC_NODE_CENTERED_SHIFT;
    Kokkos::parallel_for(
        "pmk::deposit::promote_positions",
        Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(const int p) {
            points(p, 0) = static_cast<double>(pos(p, 0)) + shift;
            points(p, 1) = static_cast<double>(pos(p, 1)) + shift;
            points(p, 2) = static_cast<double>(pos(p, 2)) + shift;
        });
    return points;
}

} // namespace

void deposit(Particles& particles, Grid& grid, double uniform_mass)
{
    // 1. Zero rho over the full (owned + ghost) extent so the scatter starts
    //    from a clean slate everywhere CIC weights might land.
    Cabana::Grid::ArrayOp::assign(
        *grid.rho(), 0.0f, Cabana::Grid::Ghost());

    // 2. Promote particle positions to FP64 (p2g requirement).
    auto points = promote_positions(particles);

    // 3. Custom uniform-mass functor — does not touch the AoSoA mass slice.
    UniformMassP2G p2g_op{ static_cast<float>(uniform_mass) };

    // 4. CIC scatter via the production Cabana::Grid::p2g entry point.
    //    halo.scatter() inside p2g reduces ghost-cell contributions back to
    //    their owning ranks (CABANA_VERIFICATION §1, p2g closing step).
    Cabana::Grid::p2g(
        ExecSpace{},
        p2g_op,
        points,
        particles.num_local(),
        Cabana::Grid::Spline<1>(),
        *grid.rhoHalo(),
        *grid.rho());
}

} // namespace pmk
