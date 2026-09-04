#pragma once
// src/pm/PMStep.hpp
// Full PM gravity step: deposit → pack → forward FFT → Green's function →
// reverse FFT → unpack → gather force / kick.  Mirrors the cycle in HACC's
// map2_poisson_forward + map2_poisson_backward_gradient (mc3.cxx:516-663).
//
// Two flavors:
//   compute_forces — DIAGNOSTIC.  Writes -grad(phi) to force_out.  Used by
//                    tests and introspection.  No velocity update.
//   pm_kick        — PRODUCTION.  Same pipeline but the final step fuses
//                    -grad(phi) into a velocity kick on FIELD_VEL.  Used by
//                    the integrator in prompt 09.
//
// Both share the deposit + Poisson-solve stages; they differ only in whether
// the gather step writes to a separate force buffer or applies the kick.

#include "../Particles.hpp"
#include "../Grid.hpp"

namespace pmk {

// Diagnostic gravity step: deposit, solve, gather force into force_out.
// Pre: force_out.extent(0) >= particles.num_local()
void compute_forces(Particles& particles,
                    Grid& grid,
                    double uniform_mass,
                    Kokkos::View<double * [3], MemorySpace> force_out);

// Production gravity step: deposit, solve, kick.
// Pre: dt is the full kick step (caller folds in any cosmological factors).
void pm_kick(Particles& particles,
             Grid& grid,
             double uniform_mass,
             double dt);

} // namespace pmk
