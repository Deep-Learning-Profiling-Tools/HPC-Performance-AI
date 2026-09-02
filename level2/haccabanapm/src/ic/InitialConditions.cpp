// src/ic/InitialConditions.cpp
// Per-axis IC orchestration: white_noise → forward FFT → IC kernel → reverse
// FFT → real-space rL^(-3/2) compensation → Zel'dovich move into the
// ParticleAoSoA.  Mirrors HACC initParticles loop (Initializer.cxx:838-876).
//
// Particle layout per the prompt: ng^3 particles per global grid, one per
// owned cell on each rank.  The local owned-cell ordering used here is
//     flat = (oi * ny_local + oj) * nz_local + ok
// matching WhiteNoise.cpp's host-staged layout — important for any caller
// that wants to associate particle index with cell index in tests.

#include "InitialConditions.hpp"

#include "ICGreensFunction.hpp"
#include "PowerSpectrum.hpp"
#include "WhiteNoise.hpp"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <cmath>

namespace pmk {

namespace {

// Convenience for owned-cell extents and global offsets.
struct OwnedBox {
    int nx_local, ny_local, nz_local;
    int gx0, gy0, gz0;
    int hw;
    long ng;  // global ng (long because of id arithmetic)
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
    b.ng       = static_cast<long>(grid.ng());
    return b;
}

} // namespace

void generate_initial_conditions(Particles& parts,
                                 Grid& grid,
                                 const Cosmology& cosmo,
                                 const PowerSpectrum& pk,
                                 double rL,
                                 double a_in,
                                 unsigned long seed)
{
    const OwnedBox b = owned_box(grid);
    const std::size_t n_particles =
        static_cast<std::size_t>(b.nx_local) * b.ny_local * b.nz_local;
    parts.resize(n_particles);

    // Growth at a_in.  Note HACC's "Ddot" returned by GrowthFactor is dD/dt
    // in code units (H0=1) — this is what the velocity multiplier needs per
    // HACC move_particles (Initializer.cxx:741): vel = ddot · F_a.
    double d_z = 0.0, ddot = 0.0;
    cosmo.growth_factor(a_in, d_z, ddot);

    const double rL_pow_neg15 = std::pow(rL, -1.5);
    const double ng_d = static_cast<double>(grid.ng());

    auto pos_slice  = parts.pos();
    auto vel_slice  = parts.vel();
    auto mass_slice = parts.mass();
    auto id_slice   = parts.id();
    auto mask_slice = parts.mask();
    auto phi_slice  = parts.phi();

    auto local_grid_ptr = grid.localGrid();
    const int hw       = b.hw;
    const int nx_local = b.nx_local;
    const int ny_local = b.ny_local;
    const int nz_local = b.nz_local;
    const int gx0 = b.gx0, gy0 = b.gy0, gz0 = b.gz0;
    const long ng_l = b.ng;

    for (int axis = 0; axis < 3; ++axis) {
        // (a) Regenerate the unit-variance white noise from the same seed.
        //     Per the prompt's procedure (a)–(c) we also re-zero phi(:,:,:,1),
        //     which generate_white_noise_real_space does as part of writing.
        generate_white_noise_real_space(grid, seed, /*phase_shift=*/false);

        // (b) Forward FFT and zero the DC mode.
        white_noise_to_kspace(grid);

        // (c) Multiply by sqrt(P_cb)·(i·k_axis)·(-1/k²)·ng^(-3/2).
        apply_ic_kernel(grid, axis, pk, rL, ng_d,
                        d_z, cosmo.params().omega_cb, /*is_potential=*/false);

        // Reverse FFT (FFTScaleNone — ng^(-3/2) was supplied by the kernel).
        grid.fft()->reverse(*grid.phi(),
                            Cabana::Grid::Experimental::FFTScaleNone());

        // Real-space rL^(-3/2) compensation matches HACC Initializer.cxx:617;
        // it converts the dimensionful FFT output into Zel'dovich-displacement
        // grid units when combined with the grid-unit Green's of the IC kernel.
        auto phi_view = grid.phi()->view();
        auto own_space = local_grid_ptr->indexSpace(
            Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local());

        // Move-particles loop: pos = q + d_z · F_a, vel = ddot · F_a, etc.
        // The q[axis] part is added only for the matching axis component.
        Kokkos::parallel_for(
            "pmk::ic_move_particles_axis",
            Cabana::Grid::createExecutionPolicy(own_space, ExecSpace{}),
            KOKKOS_LAMBDA(const int i, const int j, const int k_idx) {
                const int oi = i     - hw;
                const int oj = j     - hw;
                const int ok = k_idx - hw;
                const std::size_t flat =
                    (static_cast<std::size_t>(oi) * ny_local + oj) * nz_local + ok;

                const double F = phi_view(i, j, k_idx, 0) * rL_pow_neg15;

                const long qx_g = static_cast<long>(gx0) + oi;
                const long qy_g = static_cast<long>(gy0) + oj;
                const long qz_g = static_cast<long>(gz0) + ok;

                long q_axis;
                if (axis == 0)      q_axis = qx_g;
                else if (axis == 1) q_axis = qy_g;
                else                q_axis = qz_g;

                pos_slice(flat, axis) = static_cast<float>(
                    static_cast<double>(q_axis) + d_z * F);
                vel_slice(flat, axis) = static_cast<float>(ddot * F);

                if (axis == 0) {
                    // HACC Initializer.cxx:750: id = qz + ng·(qy + ng·qx).
                    id_slice(flat) =
                        qz_g + ng_l * (qy_g + ng_l * qx_g);
                    mass_slice(flat) = 1.0f;
                    mask_slice(flat) = 0;
                    phi_slice(flat)  = 0.0f;
                }
            });
        Kokkos::fence();
    }
}

} // namespace pmk
