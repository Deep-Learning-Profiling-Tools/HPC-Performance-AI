#pragma once
// src/pm/ForceInterp.hpp
// Force interpolation from grid.phi_real() to particles via Cabana::Grid::g2p
// with createScalarGradientG2P (CABANA_VERIFICATION §1).
//
// HACC reference:
//   - ParticleActions::inverse_cic(axis, ts, stepFraction): ParticleActions.h:242
//   - Called from map2_poisson_backward_gradient(): mc3.cxx:656
//   - Velocity update: vel[axis] += fscal * interpolated_grad_phi
//
// Sign convention (from spec §4 and prompt):
//   Force per unit mass = -grad(phi).
//   createScalarGradientG2P returns +grad with a runtime multiplier; passing
//   multiplier=-1 produces -grad directly, so both functions below set
//   multiplier=-1 and the result is force per unit mass.
//
// Two public functions, sharing the same internal g2p kernel:
//   - gather_force(...)    DIAGNOSTIC.  Writes -grad(phi) into a separate
//                          Kokkos::View<double*[3]> for tests / introspection.
//   - kick_with_grid(...)  PRODUCTION.  Fuses gradient + velocity update:
//                          vel += -grad(phi) * dt.

#include "../Particles.hpp"
#include "../Grid.hpp"

namespace pmk {

// DIAGNOSTIC interface used by tests.  force_out is overwritten with
// (-grad(phi)) at every alive particle, in particle index order matching
// particles.pos().  Cosmological factors are NOT applied — the caller wraps
// them in for production use via kick_with_grid (or, in prompt 09, the
// integrator).
//
// Pre: force_out.extent(0) >= particles.num_local()
void gather_force(const Grid& grid,
                  Particles& particles,
                  Kokkos::View<double * [3], MemorySpace> force_out);

// LEGACY DIAGNOSTIC interface used by the original prompt-08 production path.
// Fuses the same gradient interpolation as gather_force into a velocity
// update on FIELD_VEL:  vel(p, d) += -grad(phi)(p, d) * dt
//
// Retained for backward-compatible diagnostics; the production force gather
// now goes through gather_force_value, paired with apply_greens_and_gradient.
// See analysis/references/pk_discrepancy_diagnosis.md §3c for the rationale.
void kick_with_grid(const Grid& grid,
                    Particles& particles,
                    double dt);

// PRODUCTION interface used by the time integrator (post pm-gradient-fix).
// Trilinearly interpolates the real-space *gradient component* field
// (computed by apply_greens_and_gradient + inverse FFT + unpack into
// grid.phi_real()) to particle positions, multiplies by `dt`, and adds the
// result to the single velocity component `vel(p, axis)`.  The other two
// velocity components are untouched.
//
// Algorithm mirrors HACC's nbody/cpu/ParticleActions.cxx:1698-1727
// (inverse_cic):  vels[comp][nn] += f * fscal * tau  where f is the CIC
// value-interpolation of the per-axis gradient field.
//
// Sign convention.  apply_greens_and_gradient produces
//   phi_k <- (-i · sin(2π k_axis / N)) · G(k) · rho_k
// which after the inverse FFT is the real-space (-∂φ/∂x_axis) field — i.e.
// already the force component along axis (because G(k) and the gradient sign
// combine the same way HACC's `kspace_solve_gradient` does).  No further
// negation is needed inside this function; the integrator passes the
// fully-signed coefficient `dt = τ · fscal` (positive) and the gather just
// multiplies by it.
//
// The full PM kick is three calls to gather_force_value, one per axis,
// orchestrated by pm_kick (PMStep.cpp).
//
// axis: 0, 1, or 2 (which gradient component this field represents)
// dt:   time step coefficient (typically τ · fscal from the integrator)
void gather_force_value(const Grid& grid,
                        Particles& particles,
                        int axis,
                        double dt);

} // namespace pmk
