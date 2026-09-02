#pragma once
// src/io/SnapshotWriter.hpp
//
// Parallel-HDF5 snapshot writer.  Each rank writes its owned particles
// as a hyperslab into a single global dataset per field; metadata is
// written as scalar attributes on the root group.
//
// All H5Dwrite calls use collective MPI-IO transfer mode.
//
// Schema is defined in SnapshotSchema.hpp (shared with the reader).

#include "SnapshotSchema.hpp"
#include "../Particles.hpp"

#include <mpi.h>
#include <string>

namespace pmk {

// Write one snapshot file.
//
// Collective on parts.comm() — every rank participates, even those
// with zero local particles.  The file at `filename` is created (or
// truncated if it already exists) with parallel access using
// parts.comm() as the file-access communicator.
//
// Per-rank ordering inside each global dataset is by MPI rank: rank 0's
// particles occupy [0, n_local_0), rank 1's occupy [n_local_0, n_local_0
// + n_local_1), and so on.  Round-trip through SnapshotReader is
// bit-identical for FP32/INT64/UINT16; ordering may differ across
// rank-count changes but the content is preserved.
void write_snapshot(const std::string&     filename,
                    Particles&             parts,
                    const SimulationState& state);

} // namespace pmk
