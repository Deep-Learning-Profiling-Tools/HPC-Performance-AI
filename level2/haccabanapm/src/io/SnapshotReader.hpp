#pragma once
// src/io/SnapshotReader.hpp
//
// Parallel-HDF5 snapshot reader.  Mirror of SnapshotWriter.
//
// Note: this function does NOT assign particles to their spatially-correct
// ranks.  It uses a simple range partition of the global particle list:
// rank r gets N_global/N_ranks particles, plus one extra for the first
// (N_global mod N_ranks) ranks.  Call pmk::migrate(parts, grid) afterwards
// to redistribute according to position-based decomposition.
//
// All H5Dread calls use collective MPI-IO transfer mode.

#include "SnapshotSchema.hpp"
#include "../Particles.hpp"

#include <mpi.h>
#include <string>

namespace pmk {

// Read one snapshot file.  Resizes parts to fit this rank's share of
// the global particle list.  parts.comm() is used as the file-access
// communicator.  Asserts schema_version equals SCHEMA_VERSION.
void read_snapshot(const std::string& filename,
                   Particles&         parts,
                   SimulationState&   state);

} // namespace pmk
