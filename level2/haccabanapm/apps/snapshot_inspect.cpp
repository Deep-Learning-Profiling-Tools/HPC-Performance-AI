// apps/snapshot_inspect.cpp
//
// Standalone HDF5 snapshot inspector.  Single-rank only — this is a
// human-readable debugging utility, not a parallel reader.  Prints
// schema version, metadata attributes, global particle count, and
// the first/last 5 particles' position/velocity/id.
//
//   mpiexec -n 1 ./build/snapshot_inspect path/to/snapshot.h5
//
// The implementation deliberately uses the SnapshotReader path so it
// also serves as a smoke check that the read side works against any
// file produced by SnapshotWriter.

#include "io/SnapshotReader.hpp"
#include "io/SnapshotSchema.hpp"
#include "Particles.hpp"
#include "Types.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace pmk;

namespace {

void dump_first_last(Particles& parts, std::size_t n_show)
{
    const std::size_t n = parts.num_local();
    if (n == 0) {
        std::cout << "  (no particles)\n";
        return;
    }
    Kokkos::View<float * [3], MemorySpace> pos_d("d_pos", n);
    Kokkos::View<float * [3], MemorySpace> vel_d("d_vel", n);
    Kokkos::View<int64_t*,    MemorySpace> id_d ("d_id",  n);
    auto pos_s = parts.pos();
    auto vel_s = parts.vel();
    auto id_s  = parts.id();
    Kokkos::parallel_for(
        "dump_gather", Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(const std::size_t i) {
            pos_d(i, 0) = pos_s(i, 0);
            pos_d(i, 1) = pos_s(i, 1);
            pos_d(i, 2) = pos_s(i, 2);
            vel_d(i, 0) = vel_s(i, 0);
            vel_d(i, 1) = vel_s(i, 1);
            vel_d(i, 2) = vel_s(i, 2);
            id_d(i)     = id_s(i);
        });
    Kokkos::fence();
    auto pos_h = Kokkos::create_mirror_view(pos_d);
    auto vel_h = Kokkos::create_mirror_view(vel_d);
    auto id_h  = Kokkos::create_mirror_view(id_d);
    Kokkos::deep_copy(pos_h, pos_d);
    Kokkos::deep_copy(vel_h, vel_d);
    Kokkos::deep_copy(id_h,  id_d);

    auto print_row = [&](std::size_t i) {
        std::printf("  i=%6zu  id=%-12lld  pos=(% .6e % .6e % .6e)  "
                    "vel=(% .6e % .6e % .6e)\n",
                    i, static_cast<long long>(id_h(i)),
                    pos_h(i, 0), pos_h(i, 1), pos_h(i, 2),
                    vel_h(i, 0), vel_h(i, 1), vel_h(i, 2));
    };

    const std::size_t k = std::min(n_show, n);
    std::cout << "First " << k << " particles:\n";
    for (std::size_t i = 0; i < k; ++i) print_row(i);
    if (n > 2 * k) std::cout << "  ...\n";
    if (n > k) {
        std::cout << "Last " << k << " particles:\n";
        for (std::size_t i = n - k; i < n; ++i) print_row(i);
    }
}

} // namespace

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rc = 0;
    {
        Kokkos::ScopeGuard guard(argc, argv);

        int n_ranks = 1, my_rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

        if (n_ranks != 1) {
            if (my_rank == 0)
                std::cerr << "snapshot_inspect: single-rank only (got "
                          << n_ranks << " ranks)\n";
            rc = 2;
        } else if (argc != 2) {
            std::cerr << "Usage: snapshot_inspect <snapshot.h5>\n";
            rc = 2;
        } else {
            try {
                const std::string path = argv[1];
                Particles parts(MPI_COMM_WORLD);
                SimulationState state;
                read_snapshot(path, parts, state);

                std::cout << "Snapshot:        " << path << "\n";
                std::cout << "Schema version:  " << SCHEMA_VERSION
                          << "  (in-tree)\n";
                std::cout << "Metadata:\n";
                std::cout << "  scale_factor a = " << state.a        << "\n";
                std::cout << "  step           = " << state.step     << "\n";
                std::cout << "  ng             = " << state.ng       << "\n";
                std::cout << "  np (global)    = " << state.np       << "\n";
                std::cout << "  rL [Mpc/h]     = " << state.rL       << "\n";
                std::cout << "  z_in           = " << state.z_in     << "\n";
                std::cout << "  omega_m        = " << state.omega_m  << "\n";
                std::cout << "  omega_cb       = " << state.omega_cb << "\n";
                std::cout << "  omega_b        = " << state.omega_b  << "\n";
                std::cout << "  h              = " << state.h        << "\n";
                std::cout << "  n_s            = " << state.n_s      << "\n";
                std::cout << "  sigma_8        = " << state.sigma_8  << "\n";
                std::cout << "Particles on this rank: "
                          << parts.num_local() << "\n";
                dump_first_last(parts, /*n_show=*/5);
            } catch (const std::exception& e) {
                std::cerr << "snapshot_inspect: error: " << e.what() << "\n";
                rc = 1;
            }
        }
    }
    MPI_Finalize();
    return rc;
}
