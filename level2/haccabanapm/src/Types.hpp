#pragma once
// src/Types.hpp
// Central type definitions and AoSoA layout for pmkokkos.
//
// HACC reference:
//   - Precision typedefs:  nbody/common/HaccTypes.h:22-35  (POSVEL_T=float, ID_T=int64_t)
//   - Particle SoA layout: nbody/common/Particles.h:92-110 (16 SoA arrays)
//   - MASK bit fields:     cosmotools/algorithms/halofinder/BasicDefinition.h:53-55
//
// Spec: HACC_PM_IC_PORTING_SPEC.md §1 (Code Units & Constants — precision typedefs),
//       §5 (Data Structures), §8.5 (recommended AoSoA layout).
//
// Design choices (vs HACC):
//   - POSVEL_T = float          matches HACC default (no POSVEL_64 macro in HACC).
//   - ID_T     = int64_t        matches HACC ID_64 build flag (default at scale).
//   - MASS_T   = float          HACC m_grid2phys_mass is float; per-particle mass is 1.
//   - For PM-only we drop HACC's apmx/apmy/apmz (force accumulator across substeps —
//     not needed for single-kick PM), the int32 status (halo finder, out of scope),
//     and perm/scratch (manage as temporary Views).  phi is kept as an optional
//     6th member; cost is negligible.
//   - VectorLength defaults to 16, matching the SYCL/PVC backend default
//     (Cabana::Impl::PerformanceTraits<SYCL>::vector_length = 16, marked FIXME_SYCL
//     in upstream — see CABANA_VERIFICATION_v07_05c_local.md A5/B5).  Override with
//     -DPMKOKKOS_VECTOR_LENGTH=N at cmake time.

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <cstdint>
#include <utility>  // std::declval

namespace pmk {

// ---------------------------------------------------------------------------
// Scalar types (mirror HACC HaccTypes.h)
// ---------------------------------------------------------------------------
using POSVEL_T = float;     // particle position and velocity precision
using MASS_T   = float;     // particle mass precision
using ID_T     = int64_t;   // particle ID (HACC ID_64)
using GRID_T   = float;     // PM grid scalar (HACC GRID_T, GRID_32)

// ---------------------------------------------------------------------------
// AoSoA member indices (use as template argument to Cabana::slice<IDX>)
// ---------------------------------------------------------------------------
enum ParticleMember : int {
    FIELD_POS    = 0,   // float[3] — position in grid units
    FIELD_VEL    = 1,   // float[3] — velocity in grid velocity units
    FIELD_MASS   = 2,   // float    — particle mass in code units (typically 1)
    FIELD_ID     = 3,   // int64_t  — unique particle ID
    FIELD_MASK   = 4,   // uint16_t — status/overload bitmask (HACC MASK_T)
    FIELD_PHI    = 5,   // float    — gravitational potential (code units)
    FIELD_COUNT  = 6
};

// Bit definitions for FIELD_MASK (mirrors HACC BasicDefinition.h:53-55)
inline constexpr uint16_t MASK_ACTIVE     = 0x0001;
inline constexpr uint16_t MASK_OVERLOAD   = 0x0002;
inline constexpr uint16_t MASK_DOWNSAMPLE = 0x0004;

// ---------------------------------------------------------------------------
// AoSoA member-types tuple
// ---------------------------------------------------------------------------
using ParticleTypes = Cabana::MemberTypes<
    POSVEL_T[3],   // FIELD_POS:  x, y, z
    POSVEL_T[3],   // FIELD_VEL:  vx, vy, vz
    MASS_T,        // FIELD_MASS
    ID_T,          // FIELD_ID
    uint16_t,      // FIELD_MASK
    POSVEL_T       // FIELD_PHI
>;

// VectorLength — inner SIMD dimension of the AoSoA.
#ifndef PMKOKKOS_VECTOR_LENGTH
#define PMKOKKOS_VECTOR_LENGTH 16
#endif
inline constexpr int VectorLength = PMKOKKOS_VECTOR_LENGTH;

// ---------------------------------------------------------------------------
// Kokkos execution and memory spaces
// ---------------------------------------------------------------------------
using ExecSpace   = Kokkos::DefaultExecutionSpace;
using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
using HostSpace   = Kokkos::HostSpace;

// Primary particle container
using ParticleAoSoA = Cabana::AoSoA<ParticleTypes, MemorySpace, VectorLength>;

// Slice convenience aliases (one per member)
using SlicePos  = decltype(Cabana::slice<FIELD_POS> (std::declval<ParticleAoSoA&>()));
using SliceVel  = decltype(Cabana::slice<FIELD_VEL> (std::declval<ParticleAoSoA&>()));
using SliceMass = decltype(Cabana::slice<FIELD_MASS>(std::declval<ParticleAoSoA&>()));
using SliceId   = decltype(Cabana::slice<FIELD_ID>  (std::declval<ParticleAoSoA&>()));
using SliceMask = decltype(Cabana::slice<FIELD_MASK>(std::declval<ParticleAoSoA&>()));
using SlicePhi  = decltype(Cabana::slice<FIELD_PHI> (std::declval<ParticleAoSoA&>()));

// 3D PM grid, stored on device as float[x][y][z].
using GridView     = Kokkos::View<GRID_T***, MemorySpace>;
using GridViewHost = GridView::HostMirror;

} // namespace pmk
