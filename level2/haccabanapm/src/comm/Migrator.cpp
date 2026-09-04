// src/comm/Migrator.cpp
// Cabana::Distributor wrapper for post-drift particle migration.
//
// Two-pass implementation:
//   1. Device kernel: for every alive particle, wrap pos into [0, ng)^3 and
//      compute the destination rank from the per-rank cell partition.
//   2. Cabana::Distributor + Cabana::migrate (in-place) redistributes the
//      AoSoA so each particle ends up on its destination rank.
//
// Domain layout: DimBlockPartitioner creates a 3-D Cartesian topology where
// rank R owns cells [globalOffset(d), globalOffset(d) + ownedNumCell(d)) in
// each dimension.  We replicate that arithmetic on device by gathering the
// per-rank offsets/counts on host into a single Kokkos::View.

#include "Migrator.hpp"

#include <Cabana_Core.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace pmk {

namespace {

// Per-rank Cartesian-block bounds, packed (low_i, low_j, low_k, count_i, count_j, count_k).
// Allgathered once per migrate call; cheap (6 ints * num_ranks).
struct RankBoxes {
    Kokkos::View<int * [6], MemorySpace> d_box;  // (rank, slot)
};

RankBoxes gather_rank_boxes(const Grid& grid)
{
    int comm_rank = 0, comm_size = 0;
    MPI_Comm_rank(grid.comm(), &comm_rank);
    MPI_Comm_size(grid.comm(), &comm_size);

    int my_box[6];
    my_box[0] = grid.globalGrid()->globalOffset(0);
    my_box[1] = grid.globalGrid()->globalOffset(1);
    my_box[2] = grid.globalGrid()->globalOffset(2);
    my_box[3] = grid.globalGrid()->ownedNumCell(0);
    my_box[4] = grid.globalGrid()->ownedNumCell(1);
    my_box[5] = grid.globalGrid()->ownedNumCell(2);

    std::vector<int> all_boxes(6 * comm_size, 0);
    MPI_Allgather(my_box, 6, MPI_INT,
                  all_boxes.data(), 6, MPI_INT,
                  grid.comm());

    RankBoxes rb;
    rb.d_box = Kokkos::View<int * [6], MemorySpace>("pmk::migrate::rank_boxes",
                                                    comm_size);
    auto h_box = Kokkos::create_mirror_view(rb.d_box);
    for (int r = 0; r < comm_size; ++r)
        for (int s = 0; s < 6; ++s)
            h_box(r, s) = all_boxes[6 * r + s];
    Kokkos::deep_copy(rb.d_box, h_box);
    return rb;
}

} // namespace

void migrate(Particles& parts, const Grid& grid, bool wrap_periodic)
{
    // wrap_periodic is documentary at this call site: the existing
    // implementation wraps unconditionally because the post-drift call from
    // Integrator::full_step always sees small over-shoots from the half-step
    // drift.  IC callers (pm_ic) pass wrap_periodic=true explicitly so the
    // contract is visible at the call site even though the kernel does not
    // need to branch on it.  Prompt 12 deliverable 6, option (b).
    (void)wrap_periodic;

    int comm_size = 0;
    MPI_Comm_size(grid.comm(), &comm_size);

    // Single-rank fast path: no migration needed, but still wrap positions
    // into [0, ng) so the next deposit doesn't see drifted-out coordinates.
    const std::size_t n = parts.num_local();
    const float ng_f = static_cast<float>(grid.ng());

    if (comm_size == 1) {
        auto pos = parts.pos();
        Kokkos::parallel_for(
            "pmk::migrate::wrap_pbc_serial",
            Kokkos::RangePolicy<ExecSpace>(0, n),
            KOKKOS_LAMBDA(const int p) {
                for (int d = 0; d < 3; ++d) {
                    float x = pos(p, d);
                    // Branch-free wrap: pos = pos - floor(pos/ng)*ng.
                    x -= std::floor(x / ng_f) * ng_f;
                    // Edge case: x exactly == ng_f after rounding.
                    if (x >= ng_f) x -= ng_f;
                    if (x < 0.0f) x += ng_f;
                    pos(p, d) = x;
                }
            });
        Kokkos::fence();
        return;
    }

    RankBoxes rb = gather_rank_boxes(grid);

    // Compute destination rank for each particle (and PBC-wrap pos in place).
    Kokkos::View<int *, MemorySpace> dest_ranks(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "pmk::migrate::dest_ranks"),
        n);

    auto pos = parts.pos();
    auto box = rb.d_box;
    const int cs = comm_size;

    Kokkos::parallel_for(
        "pmk::migrate::assign_dest",
        Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(const int p) {
            int cell[3];
            for (int d = 0; d < 3; ++d) {
                float x = pos(p, d);
                x -= std::floor(x / ng_f) * ng_f;
                if (x >= ng_f) x -= ng_f;
                if (x < 0.0f) x += ng_f;
                pos(p, d) = x;
                int c = static_cast<int>(std::floor(x));
                if (c < 0) c = 0;
                if (c >= static_cast<int>(ng_f)) c = static_cast<int>(ng_f) - 1;
                cell[d] = c;
            }
            // Linear scan of the comm_size rank boxes — comm_size is O(10s)
            // for the test scales here, and the kernel is bandwidth-bound on
            // the position view, not on this 6-compare hit test.
            int dest = -1;
            for (int r = 0; r < cs; ++r) {
                if (cell[0] >= box(r, 0) && cell[0] < box(r, 0) + box(r, 3) &&
                    cell[1] >= box(r, 1) && cell[1] < box(r, 1) + box(r, 4) &&
                    cell[2] >= box(r, 2) && cell[2] < box(r, 2) + box(r, 5)) {
                    dest = r;
                    break;
                }
            }
            dest_ranks(p) = dest;
        });
    Kokkos::fence();

    // Build the Distributor without a precomputed neighbor list: in the
    // general PM case any rank can talk to any other (long drifts at low z),
    // and the single-arg constructor performs the global discovery internally.
    //
    // Cabana::migrate uses MPI_Isend/MPI_Irecv with device pointers, so
    // GPU-aware MPI must be enabled at run time.  On Aurora that means
    // exporting MPIR_CVAR_ENABLE_GPU=1 (the apps/demo/run_pmkokkos.sh
    // wrapper sets this).  With MPIR_CVAR_ENABLE_GPU=0 the MPI_Waitall
    // inside Cabana::Impl::distributeData fails because the runtime
    // cannot read the SYCL/USM buffer.
    Cabana::Distributor<MemorySpace> distributor(grid.comm(), dest_ranks);

    // In-place migration resizes parts.aosoa() to totalNumImport().
    Cabana::migrate(distributor, parts.aosoa());

    // Refresh the cached owned count.  After in-place migrate the AoSoA size
    // equals the imported count.
    parts.set_num_local(parts.aosoa().size());
}

} // namespace pmk
