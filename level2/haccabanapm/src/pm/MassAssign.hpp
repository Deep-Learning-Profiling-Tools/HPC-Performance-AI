#pragma once
// src/pm/MassAssign.hpp
// CIC mass deposition to the PM density grid via Cabana::Grid::p2g with
// Spline<1>().  Wraps the production CIC scatter (CABANA_VERIFICATION §1).
//
// HACC reference:
//   - particles.cic(ts, los):  nbody/sycl/ParticleActions.h:241
//   - Called from map2_poisson_forward(): mc3.cxx:544-546
//
// Design choices (per prompt 08 §2):
//   (a) Custom UniformMassP2G functor.  FIELD_MASS exists in the AoSoA for
//       layout compatibility with HACC and for I/O round-trip, but during the
//       PM deposit every particle has the same mass.  Reading the slice would
//       waste memory bandwidth on a value that's constant — so the deposit
//       takes the uniform mass as a runtime constant and the functor returns
//       it for every particle without consulting the AoSoA mass slice.  This
//       matches the prompt-mandated API and the structural template in
//       Cabana/example/grid_tutorial/15_interpolation/interpolation_example.cpp.
//
//   (b) Position promotion.  Cabana::Grid::p2g requires
//       Kokkos::View<double*[3]>(p, d) for points; ParticleAoSoA stores
//       positions as POSVEL_T=float in a Cabana::Slice with cabana_layout.
//       Building a no-copy adaptor that satisfies p2g's static_assert on
//       MeshScalar=double would require reinterpreting the SoA strides — not
//       supported by Slice's view layout.  We promote to a temporary
//       Kokkos::View<double*[3], MemorySpace> once per deposit.  Cost is
//       O(N_particles) which is negligible compared to the FFT (O(N^3 log N)).

#include "../Particles.hpp"
#include "../Grid.hpp"

namespace pmk {

// Custom point-evaluation functor: returns the same uniform mass for every
// particle, bypassing the AoSoA mass slice.  Mirrors the structural template
// of P2GExampleFunctor in interpolation_example.cpp (lines 24-67).
struct UniformMassP2G {
    using value_type = float;
    float mass;

    template <class SplineDataType, class ScatterViewType>
    KOKKOS_INLINE_FUNCTION
    void operator()(const SplineDataType& sd,
                    const int /*p*/,
                    const ScatterViewType& view) const
    {
        Cabana::Grid::P2G::value(mass, sd, view);
    }
};

// Deposit a uniform mass per particle onto grid.rho() with CIC weighting.
// Steps:
//   1. ArrayOp::assign zeros rho over Ghost (owned + ghost cells).
//   2. Promote particle positions to a Kokkos::View<double*[3]>.
//   3. Cabana::Grid::p2g with Spline<1>() and UniformMassP2G — internally
//      a Kokkos::ScatterView, then halo.scatter() reduces ghost contributions
//      back to owning ranks.
void deposit(Particles& particles, Grid& grid, double uniform_mass);

} // namespace pmk
