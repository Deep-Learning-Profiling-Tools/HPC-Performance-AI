#pragma once
// src/ic/InitialConditions.hpp
// Top-level IC orchestration: white noise → IC kernel → Zel'dovich move,
// once per axis, producing a populated ParticleAoSoA.
//
// HACC reference:  initParticles per-axis loop, Initializer.cxx:838-876.
//
// Spec: ../analysis/prompts/10b_implement_zeldovich_ic.md (Deliverable 5)
//
// Output state per HACC convention (see CoordConvert.hpp for conversion to
// the simulation-loop convention):
//   POS:  q + D1(a_in) · F_a, in INIT units (= GRID units when np == ng)
//         (q is the GLOBAL cell position; periodic-wrap and global→local
//          shifts happen in CoordConvert)
//   VEL:  Ddot · F_a, NON-symplectic (CoordConvert applies a² scaling)
//   ID :  z_g + ng · (y_g + ng · x_g)
//   MASS: 1.0
//   MASK: 0
//   PHI : 0 (axis=3 deferred)

#include "../Particles.hpp"
#include "../Grid.hpp"
#include "../Cosmology.hpp"

namespace pmk {

class PowerSpectrum;

void generate_initial_conditions(Particles& parts,
                                 Grid& grid,
                                 const Cosmology& cosmo,
                                 const PowerSpectrum& pk,
                                 double rL,
                                 double a_in,
                                 unsigned long seed);

} // namespace pmk
