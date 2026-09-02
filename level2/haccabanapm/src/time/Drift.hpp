#pragma once
// src/time/Drift.hpp
// Drift kernel — equivalent to HACC's map1 / map1i in the symplectic
// stepping variable p = a^α.
//
// HACC reference:
//   - map1i loop:                ParticleActions.cxx:2564-2574
//   - map1 prefactor + tau:      ParticleActions.cxx:2576-2615
//   - p = a^α stepping variable: TimeStepper.cxx:61-65
//
// Habib 2011 HACC code-units note (analysis/references/hacc_units.pdf):
//   The dimensionless equation of motion is
//     dx̃/dy = P̃ / (α · ȧ · p^(1+1/α) )                       (Eq. 22)
//   where y = p is the stepping variable and P̃ is the canonical momentum
//   (HACC's velocity slot stores P̃ directly — see prompt 09 §RULES).
//
// The drift updates positions by integrating Eq. 22 over Δp = (stepFraction · τ):
//     Δx̃ = (stepFraction · τ) / (α · ȧ · p^(1+1/α)) · P̃
//
// For α=1 the prefactor reduces to 1 / (α · ȧ · a²)  (Eq. 25 / spec §3).
//
// The kernel is a one-line Kokkos::parallel_for over alive particles; the
// per-step prefactor is computed once on host (the integrator's responsibility)
// and folded into `coeff` so the device kernel does no division.

#include "../Particles.hpp"

namespace pmk {

// Apply a drift of Δp = tau_step * (1) — i.e., advance positions by
//     x_new = x_old + (tau_step / (α · adot · p^(1+1/α))) · P̃
//
// `tau_step` is the drift's Δp (e.g., τ/2 for the DKD half-drift).
// `alpha`, `pp`, `adot` are the current stepping-variable state captured from
// the Integrator at the moment of the drift call.  Positions are updated in
// place on FIELD_POS; the velocity slice (FIELD_VEL = P̃) is read but not
// modified.
//
// Periodic-box wrap is NOT applied here — the Migrator does it before
// recomputing destination ranks (see src/comm/Migrator.cpp).  This matches
// HACC's separation between map1 (drift only) and the subsequent particle
// migration step.
void drift(Particles& parts,
           double tau_step,
           double alpha,
           double pp,
           double adot);

} // namespace pmk
