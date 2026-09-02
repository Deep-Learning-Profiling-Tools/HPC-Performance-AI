#pragma once
// src/ic/CoordConvert.hpp
// Single-shot conversion from IC convention (global coords, dx/dt velocity)
// to simulation convention (per-rank local coords, P̃ = a²·dx/dt velocity).
//
// HACC reference:  convertCoords pattern in Initializer.cxx (general form);
// pmkokkos's np == ng simplification removes HACC's gpscal multiplicative
// factor so this is just three local kernels: wrap, shift, scale.
//
// Spec: ../analysis/prompts/10b_implement_zeldovich_ic.md (Deliverable 6)
//
// Velocity convention (HACC code units, hacc_units.pdf §Eq. 18):
//     P̃ = a² · dx/dt
// The IC produced velocities in dx/dt directly (Zel'dovich vel = ddot · F);
// the integrator stores velocities in P̃.  Single multiplicative scale a_init²
// completes the conversion.
//
// Position convention (prompt-vs-pmkokkos discrepancy, recorded in
// ../analysis/prompt_issues/10b_issues.md):
//   The 10b prompt's deliverable 6 step (2) calls for a GLOBAL → LOCAL shift
//   so positions land in [0, local_ng) per dim.  The rest of pmkokkos —
//   src/comm/Migrator.cpp and src/pm/MassAssign.cpp specifically — keeps
//   particle positions in GLOBAL coordinates ([0, ng) per dim), because:
//     * Migrator looks up the destination rank by integer-floor of the global
//       cell index against a per-rank box partition.
//     * Cabana::Grid::p2g maps positions onto the local mesh implicitly using
//       the rank's mesh-frame offset, so global coordinates are the natural
//       input.
//   Performing the prompt's local shift would silently mis-route particles
//   on >1 rank and is therefore omitted; the periodic wrap into [0, ng)
//   preserves the global convention.

#include "../Particles.hpp"
#include "../Grid.hpp"

namespace pmk {

// In-place: wraps positions to [0, ng), shifts to local owned-cell coords,
// scales velocity by a_init².  Safe to call exactly once per Particles set.
void convert_ic_to_simulation_coords(Particles& parts,
                                     const Grid& grid,
                                     double a_init);

} // namespace pmk
