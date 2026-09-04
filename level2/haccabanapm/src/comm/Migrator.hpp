#pragma once
// src/comm/Migrator.hpp
// Particle migration after the integrator's two half-drifts.  Wraps positions
// into the periodic [0, ng)^3 box, computes the destination rank for each
// particle from the grid's domain decomposition, and calls the underlying
// Cabana::Distributor to redistribute the AoSoA in place.
//
// Reference:
//   - Cabana::Distributor:           Cabana_Distributor.hpp
//   - migration_example.cpp:         software/Cabana/example/core_tutorial/11_migration/
//   - GlobalGrid::blockRank/Offset:  software/install/cabana/include/Cabana_Grid_GlobalGrid.hpp:83-166
//
// Spec: HACC_PM_IC_PORTING_SPEC.md §6 (MPI domain decomposition).

#include "../Particles.hpp"
#include "../Grid.hpp"

namespace pmk {

// Migrate `parts` so that each particle ends up on the rank that owns its
// position under `grid`'s Cartesian block decomposition.  (Which partitioner
// produced that decomposition is irrelevant here: the destination lookup
// reads each rank's actual globalOffset/ownedNumCell box, so it is correct
// for both the DimBlockPartitioner default and the ManualBlockPartitioner
// path the drivers use when an explicit topology is given.)
//
// Called TWICE per DKD step by Integrator::full_step — once before the kick
// and once after the second half-drift.  The pre-kick call is load-bearing,
// not defensive: under this code's strict-ownership scheme a particle that
// crossed a rank boundary during the first half-drift would otherwise have
// its CIC stencil land on the wrong rank's local mesh and receive a wrong
// kick force.  Regression guard: tests/test_migration_protects_scatter.cpp
// (the 2-rank run is the decisive one).
//
// Multi-rank runs require GPU-aware MPI — Cabana::Distributor posts
// MPI_Isend/Irecv on device pointers.  On Aurora that means
// MPIR_CVAR_ENABLE_GPU=1; see docs/RUNTIME_CONTROL.md.
//
// Pre: positions may be slightly outside [0, ng) (from drift); they are
//      wrapped into the periodic box before destination ranks are computed.
//      The wrapped positions are written back to FIELD_POS.
// Post: parts.aosoa() is resized to the new local count (totalNumImport()),
//       parts.num_local() reflects the new count, and every particle is on
//       the rank that owns its [0, ng) cell range.
//
// `wrap_periodic` is provided for callers (notably pm_ic) where IC positions
// can land slightly outside [0, ng) due to Zel'dovich displacement crossing
// box boundaries.  When true, positions are wrapped into [0, ng) before the
// destination-rank lookup *and* the wrapped positions are written back.  In
// the post-drift case (the usual call site from Integrator::full_step) the
// behavior is functionally identical because the small-drift wrap is also
// done unconditionally — `wrap_periodic` is documentary at that call site
// and a no-op semantically.  Per prompt 12 deliverable 6 (option (b)).
void migrate(Particles& parts, const Grid& grid, bool wrap_periodic = false);

} // namespace pmk
