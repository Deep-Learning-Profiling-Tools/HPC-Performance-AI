#pragma once
// src/io/SnapshotSchema.hpp
//
// Snapshot file schema (HDF5).  Header-only contract shared by writer and
// reader.  Both must include this; do not redefine the dataset/attribute
// names elsewhere.
//
// Per prompt 11: GenericIO compatibility is intentionally out of scope.
// HDF5 is sufficient for pmkokkos and is the only on-disk format here.
//
// Schema v2 layout (prompt 12):
//   /particles/position : float32, shape (N_global, 3)   contiguous
//   /particles/velocity : float32, shape (N_global, 3)   contiguous
//   /particles/id       : int64,   shape (N_global,)
//   /particles/mask     : uint16,  shape (N_global,)
//   /particles/mass     : float32, shape (N_global,)
//   /particles/phi      : float32, shape (N_global,)     [v2; zeros at IC]
//   /particles/status   : int32,   shape (N_global,)     [v2; owning rank]
//
// Schema v1 had everything except /particles/phi and /particles/status, and
// no ATTR_TOPOLOGY_DIMS.  The reader transparently backfills phi=0 and
// status=local_rank for v1 files, logs a warning, and proceeds.
//
// Per-rank writes use HDF5 hyperslab selection in the global dataspace.
// All H5Dwrite/H5Dread calls use collective MPI-IO mode.
//
// Metadata lives as scalar attributes on the root group.  Bumping
// SCHEMA_VERSION requires a documented migration story for any old files.

#include <cstdint>

namespace pmk {

// ---------------------------------------------------------------------------
// Dataset paths (particle fields)
// ---------------------------------------------------------------------------
inline constexpr const char* PATH_POS    = "/particles/position";
inline constexpr const char* PATH_VEL    = "/particles/velocity";
inline constexpr const char* PATH_ID     = "/particles/id";
inline constexpr const char* PATH_MASK   = "/particles/mask";
inline constexpr const char* PATH_MASS   = "/particles/mass";
inline constexpr const char* PATH_PHI    = "/particles/phi";     // v2
inline constexpr const char* PATH_STATUS = "/particles/status";  // v2

// ---------------------------------------------------------------------------
// Attribute paths (metadata, on the root group "/")
// ---------------------------------------------------------------------------
inline constexpr const char* ATTR_SCHEMA_VERSION = "schema_version";
inline constexpr const char* ATTR_SCALE_FACTOR   = "a";
inline constexpr const char* ATTR_STEP           = "step";
inline constexpr const char* ATTR_NG             = "ng";
inline constexpr const char* ATTR_NP             = "np";
inline constexpr const char* ATTR_RL_MPC_H       = "rL";
inline constexpr const char* ATTR_Z_INIT         = "z_in";
inline constexpr const char* ATTR_OMEGA_M        = "omega_m";
inline constexpr const char* ATTR_OMEGA_CB       = "omega_cb";
inline constexpr const char* ATTR_OMEGA_B        = "omega_b";
inline constexpr const char* ATTR_H              = "h";
inline constexpr const char* ATTR_N_S            = "n_s";
inline constexpr const char* ATTR_SIGMA_8        = "sigma_8";
// v2: 3-element int array — (Px, Py, Pz) Cartesian topology used at write.
inline constexpr const char* ATTR_TOPOLOGY_DIMS  = "topology_dims";

// Schema version.  Bump on any incompatible change to dataset names,
// dtypes, shapes, or attribute set, and document the migration path.
inline constexpr int SCHEMA_VERSION = 2;

// ---------------------------------------------------------------------------
// SimulationState — POD bundle of metadata persisted alongside particles.
// ---------------------------------------------------------------------------
struct SimulationState {
    // Time / step
    double a        = 1.0;   // scale factor at output
    int    step     = 0;     // simulation step number

    // Box / particle counts
    int      ng     = 0;     // PM grid dim per side
    int64_t  np     = 0;     // global particle count (N^3 in IC, may evolve)

    // Cosmology
    double rL       = 0.0;   // box size [Mpc/h]
    double z_in     = 0.0;   // starting redshift
    double omega_m  = 0.0;
    double omega_cb = 0.0;
    double omega_b  = 0.0;
    double h        = 0.0;
    double n_s      = 0.0;
    double sigma_8  = 0.0;

    // v2: per-axis Cartesian topology dims (Px, Py, Pz).  Defaults to (0,0,0)
    // which the reader treats as "missing"; the writer asserts >0 and writes
    // the actual cart-comm dims.
    int topology_dims[3] = {0, 0, 0};
};

} // namespace pmk
