// src/ic/WhiteNoise.cpp
// Host-staged CBRNG white-noise generation, mirroring HACC's
// indens_whitenoise() (Initializer.cxx:260-299, CBRNG branch).
//
// Per the prompt's option (A): the vendored CBRNG_* / Random123 sources
// are pulled in by host-only #include here and called from a serial host
// loop nested in the order (i, j, k) that HACC uses.  The results are
// staged into a Kokkos::View<double*, HostSpace>, deep-copied to a
// device View, then a Kokkos::parallel_for writes them into phi(i,j,k,0)
// with phi(i,j,k,1) = 0.  The serial host loop is what gives us
// rank-count- AND backend-independent reproducibility: every cell is
// keyed on its GLOBAL (i, j*ng + k) coordinate, and the draw order
// inside a single rank does not interact with thread scheduling.

#include "WhiteNoise.hpp"

#include "../../external/cbrng/CBRNG_Random.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <vector>

namespace pmk {

namespace {

// Owned-cell extents pulled out of the localGrid index space.  Returns
// owned counts per dim and the (low) local-index offset that skips the
// halo (== haloCellWidth on every dim).
struct OwnedBox {
    int nx_local, ny_local, nz_local;
    int gx0, gy0, gz0;     // global indices of the first owned cell
    int hw;                // halo cell width (low-index offset in local coords)
};

OwnedBox owned_box(const Grid& grid)
{
    auto local = grid.phi()->layout()->localGrid();
    auto& gg   = local->globalGrid();
    auto own   = local->indexSpace(
        Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local());

    OwnedBox b;
    b.hw       = local->haloCellWidth();
    b.nx_local = static_cast<int>(own.extent(0));
    b.ny_local = static_cast<int>(own.extent(1));
    b.nz_local = static_cast<int>(own.extent(2));
    b.gx0      = gg.globalOffset(0);
    b.gy0      = gg.globalOffset(1);
    b.gz0      = gg.globalOffset(2);
    return b;
}

} // namespace

void generate_white_noise_real_space(Grid& grid,
                                     unsigned long seed,
                                     bool phase_shift)
{
    const OwnedBox b  = owned_box(grid);
    const int      ng = grid.ng();
    const double   phaseflip = phase_shift ? -1.0 : 1.0;

    // -------------------------------------------------------------------
    // Per-x-slab keys (HACC: GetSlabKeys(&keys[0], tnx1, tngx, seed)).
    // Each rank derives keys for its OWNED x-range only; the keys are a
    // function of (seed, global x-index), so reproducibility doesn't care
    // how the x-axis is partitioned.
    // -------------------------------------------------------------------
    std::vector<unsigned long> slab_keys(static_cast<size_t>(b.nx_local));
    GetSlabKeys(slab_keys.data(), b.gx0, b.nx_local, seed);

    // -------------------------------------------------------------------
    // Host staging buffer.  Layout matches grid.phi() owned-cell ordering:
    // index = (i*ny_local + j)*nz_local + k  for owned (i, j, k).
    // -------------------------------------------------------------------
    using HostBuf = Kokkos::View<double*, HostSpace>;
    HostBuf host_field("wn_host",
        static_cast<size_t>(b.nx_local) * b.ny_local * b.nz_local);

    // Mirror HACC's nested loop verbatim (Initializer.cxx:275-298).
    // Serial on host: per-mode independence makes parallelism trivially
    // safe, but the prompt explicitly opts for host staging; we keep it
    // serial here for backend-invariant determinism.
    for (int i = 0; i < b.nx_local; ++i) {
        const unsigned long ikey = slab_keys[static_cast<size_t>(i)];
        for (int j = 0; j < b.ny_local; ++j) {
            const long jg = b.gy0 + j;
            for (int k = 0; k < b.nz_local; ++k) {
                const long kg = b.gz0 + k;
                const unsigned long index =
                    static_cast<unsigned long>(jg) * static_cast<unsigned long>(ng)
                  + static_cast<unsigned long>(kg);

                double rn1, rn2;
                GetRandomDoublesWhiteNoise(rn1, rn2, ikey, index);

                const double rn = rn1 * rn1 + rn2 * rn2;
                const double g  = rn2 * std::sqrt(-2.0 * std::log(rn) / rn)
                                * phaseflip;

                const size_t flat =
                    (static_cast<size_t>(i) * b.ny_local + j) * b.nz_local + k;
                host_field(flat) = g;
            }
        }
    }

    // -------------------------------------------------------------------
    // Stage to device + scatter into phi.
    // -------------------------------------------------------------------
    using DevBuf = Kokkos::View<double*, MemorySpace>;
    DevBuf dev_field(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                        "wn_device"),
                     host_field.extent(0));
    Kokkos::deep_copy(dev_field, host_field);

    auto phi_view = grid.phi()->view();
    auto own = grid.localGrid()->indexSpace(
        Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local());

    const int hw       = b.hw;
    const int ny_local = b.ny_local;
    const int nz_local = b.nz_local;

    Kokkos::parallel_for(
        "pmk::scatter_whitenoise_into_phi",
        Cabana::Grid::createExecutionPolicy(own, ExecSpace{}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            const int oi = i - hw;
            const int oj = j - hw;
            const int ok = k - hw;
            const size_t flat =
                (static_cast<size_t>(oi) * ny_local + oj) * nz_local + ok;
            phi_view(i, j, k, 0) = dev_field(flat);
            phi_view(i, j, k, 1) = 0.0;
        });
    Kokkos::fence();
}

void white_noise_to_kspace(Grid& grid)
{
    // Forward with no scaling — matches HACC's convention where the
    // symmetric ng^(-1.5) scaling is delivered later by the √P(k)
    // multiply (deferred to prompt 10b).
    grid.fft()->forward(*grid.phi(),
                        Cabana::Grid::Experimental::FFTScaleNone());

    // Zero the k=0 (DC) mode.  Owned only on the rank whose global
    // offset is (0,0,0); on every other rank this is a no-op.
    auto local = grid.phi()->layout()->localGrid();
    auto& gg   = local->globalGrid();
    if (gg.globalOffset(0) == 0 &&
        gg.globalOffset(1) == 0 &&
        gg.globalOffset(2) == 0) {
        const int hw = local->haloCellWidth();
        auto phi_view = grid.phi()->view();
        Kokkos::parallel_for(
            "pmk::zero_kzero_mode",
            Kokkos::RangePolicy<ExecSpace>(0, 1),
            KOKKOS_LAMBDA(const int) {
                phi_view(hw, hw, hw, 0) = 0.0;
                phi_view(hw, hw, hw, 1) = 0.0;
            });
        Kokkos::fence();
    }
}

} // namespace pmk
