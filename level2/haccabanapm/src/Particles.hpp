#pragma once
// src/Particles.hpp
// ParticleAoSoA wrapper holding the on-device particle container, the MPI
// communicator, and slice accessors.
//
// Migration is NOT owned by this class: src/comm/Migrator.cpp drives
// Cabana::Distributor over parts.aosoa() and calls set_num_local() with the
// post-exchange owned count.  The wrapper stays a plain container so the
// migration policy lives in one place.
//
// Coordinate conventions carried by the slices (see docs/OVERVIEW.md):
//   FIELD_POS — GLOBAL grid units, [0, ng) per dim, NOT rank-local.  The
//               Migrator's destination lookup and Cabana::Grid::p2g/g2p both
//               assume this; see src/ic/CoordConvert.hpp for the rationale
//               and the divergence from HACC's LOCAL representation.
//   FIELD_VEL — the canonical momentum P~ = a^2 dx/dt, not a physical
//               velocity.  Conversion happens only at the I/O boundary
//               (apps/pm_ic.cpp, apps/pm_run_lib.cpp).
//
// HACC reference:
//   - Particles class:                nbody/common/Particles.h
//   - loanMemory/returnMemory pattern: Particles.h:139-158  (non-PM only)
//   - nAlive/nOverload split:         Domain.cxx convention
//
// Spec: HACC_PM_IC_PORTING_SPEC.md §5 (Data Structures), §8.5 (AoSoA layout).
//
// Per the prompt: FIELD_MASS lives on every particle for layout compatibility
// with HACC and for I/O round-trip, but the PM mass deposit treats mass as a
// uniform runtime constant so the scatter never reads the per-particle slice.
// See pm/MassAssign for the UniformMassP2G functor that codifies this.

#include "Types.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <cstddef>

namespace pmk {

class Particles {
  public:
    // Construct an empty container on `comm`.  Reserve `n_max` capacity if >0.
    explicit Particles(MPI_Comm comm = MPI_COMM_WORLD, std::size_t n_max = 0);

    // Resize to hold `n` owned particles (ghosts handled separately by Halo).
    void resize(std::size_t n);

    // Reserve capacity without changing the live size.
    void reserve(std::size_t n_max);

    // Number of owned particles on this rank.
    std::size_t num_local() const { return m_n_local; }

    // Total AoSoA size (owned + ghosts appended by Cabana::Halo, if any).
    std::size_t num_total() const { return m_particles.size(); }

    // Set the cached owned-particle count.  Used by Migrator after
    // Cabana::migrate has resized the AoSoA in place to totalNumImport(); the
    // resize is invisible to this wrapper otherwise.
    void set_num_local(std::size_t n) { m_n_local = n; }

    MPI_Comm comm() const { return m_comm; }

    // Slice accessors — call once per kernel and capture by value in the lambda.
    SlicePos  pos()  { return Cabana::slice<FIELD_POS> (m_particles); }
    SliceVel  vel()  { return Cabana::slice<FIELD_VEL> (m_particles); }
    SliceMass mass() { return Cabana::slice<FIELD_MASS>(m_particles); }
    SliceId   id()   { return Cabana::slice<FIELD_ID>  (m_particles); }
    SliceMask mask() { return Cabana::slice<FIELD_MASK>(m_particles); }
    SlicePhi  phi()  { return Cabana::slice<FIELD_PHI> (m_particles); }

    // Raw AoSoA access (for Cabana::Halo / Cabana::Distributor in prompt 09).
    ParticleAoSoA&       aosoa()       { return m_particles; }
    const ParticleAoSoA& aosoa() const { return m_particles; }

  private:
    MPI_Comm        m_comm;
    ParticleAoSoA   m_particles;
    std::size_t     m_n_local = 0;
};

} // namespace pmk
