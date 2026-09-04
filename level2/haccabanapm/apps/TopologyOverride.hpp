#pragma once
// apps/TopologyOverride.hpp
// Explicit MPI Cartesian topology override for the pm_ic / pm_run drivers.
//
// The Grid layer already supports an arbitrary (Px,Py,Pz) block decomposition
// via ManualBlockPartitioner (src/Grid.cpp), but both drivers otherwise derive
// the topology from MPI_Dims_create + sort(greater), which forces the largest
// factor onto dim 0 (e.g. 12 ranks -> 3x2x2, never 2x2x3).  The P3HPC scaling
// experiments (E4, E4b CCS oversubscription) need to choose the decomposition
// explicitly, so this header adds an override source with precedence:
//
//   1. env  PMKOKKOS_TOPOLOGY="Px,Py,Pz"   (also accepts "Px Py Pz")
//   2. indat TOPOLOGY Px Py Pz             (when an IndatParser is available)
//   3. existing behavior (snapshot topology_dims, then MPI_Dims_create+sort)
//
// The caller is responsible for validating product == n_ranks and warning on
// uneven division (ng % P != 0); see topology_validate() below.
//
// Design note: notes/patch_topology_override.md.

#include "io/IndatParser.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

namespace pmk {

// Parse "a,b,c" or "a b c" into dims.  Returns true on success (three ints).
inline bool topology_parse3(std::string s, std::array<int, 3>& dims) {
    for (char& c : s)
        if (c == ',') c = ' ';
    std::istringstream is(s);
    return static_cast<bool>(is >> dims[0] >> dims[1] >> dims[2]);
}

// Fills `dims` and returns true if an explicit override was found.
//   - env PMKOKKOS_TOPOLOGY wins over everything (including a snapshot value,
//     so an existing IC can be re-decomposed in pm_run).
//   - indat TOPOLOGY is the second source (may be null; IndatParser has no
//     string-default overload, so guard with has()).
inline bool topology_override(const IndatParser* indat,
                              std::array<int, 3>& dims) {
    if (const char* e = std::getenv("PMKOKKOS_TOPOLOGY"))
        if (topology_parse3(e, dims))
            return true;
    if (indat && indat->has("TOPOLOGY"))
        if (topology_parse3(indat->get_string("TOPOLOGY"), dims))
            return true;
    return false;
}

// Validate a chosen topology against rank count and grid size.
//   - Aborts (returns false) if Px*Py*Pz != n_ranks, with a rank-0 message.
//   - Warns (rank 0) but does not fail if ng is not divisible by some axis:
//     heFFTe tolerates uneven pencils (transform length unchanged, <=1-plane
//     load drift), but load balance drifts and HACC GIO comparability weakens.
// Returns true if the topology is usable (i.e. product matches n_ranks).
inline bool topology_validate(const std::array<int, 3>& dims, int ng,
                              int n_ranks, int my_rank) {
    const long prod =
        static_cast<long>(dims[0]) * dims[1] * dims[2];
    if (dims[0] <= 0 || dims[1] <= 0 || dims[2] <= 0 || prod != n_ranks) {
        if (my_rank == 0)
            std::cerr << "ERROR: topology " << dims[0] << "x" << dims[1] << "x"
                      << dims[2] << " (product " << prod << ") does not match "
                      << n_ranks << " ranks\n";
        return false;
    }
    if (my_rank == 0 &&
        (ng % dims[0] != 0 || ng % dims[1] != 0 || ng % dims[2] != 0)) {
        std::cerr << "WARNING: ng=" << ng << " is not divisible by topology "
                  << dims[0] << "x" << dims[1] << "x" << dims[2]
                  << "; blocks are uneven (heFFTe tolerates it, but load "
                     "balance drifts and HACC comparability weakens)\n";
    }
    return true;
}

} // namespace pmk
