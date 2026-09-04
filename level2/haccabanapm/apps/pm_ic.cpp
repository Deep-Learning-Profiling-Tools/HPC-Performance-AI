// apps/pm_ic.cpp
//
// IC generator binary.  Reads a HACC indat.params, generates Zel'dovich
// initial conditions on `np^3` particles via the prompt-10b pipeline,
// applies the INIT → PHYSICAL unit conversion documented in
// ../analysis/references/hacc_ic_unit_conversions.md §4 (revised), migrates
// particles onto the 3D Cartesian topology, sets status, and writes a
// pmkokkos schema-v2 HDF5 snapshot.
//
// Usage:
//   mpiexec -n N pm_ic <indat.params> <output_snapshot.h5>
//
// Arguments are positional and both are required (argc != 3 → usage, exit 2).
// Note the indat comes FIRST here and LAST in pm_run.
//
// Unlike pm_run, this driver takes its ENTIRE configuration from the indat:
// box/resolution (RL, NG, NP), cosmology (HUBBLE, OMEGA_CDM, DEUT, SS8, NS,
// and optionally OMEGA_NU / W_DE / WA_DE / T_CMB), the IC seed (I_SEED), the
// starting redshift (Z_IN), and the transfer-function path
// (INPUT_BASE_NAME, resolved relative to the working directory).
// User-facing reference: docs/RUNNING.md.
//
// Pipeline (mirrors HACC's loadParticles + writeAliveRestart sequence inside
// driver_pm.cxx, per hacc_ic_unit_conversions.md §5 corrected):
//   1.  MPI/Kokkos init.
//   2.  Parse indat.params (RL, NG, NP, OMEGA_*, HUBBLE, DEUT, SS8, NS, ZIN,
//       I_SEED, INPUT_BASE_NAME).
//   3.  Compute the Cartesian (Px, Py, Pz) topology dims using MPI_Dims_create
//       (MPI_Dims_create produces a sorted-descending tuple, matching HACC's
//       MY_Dims_create_3D after its March 2023 sort; for N=8 both pick
//       (2,2,2)).
//   4.  Construct Cosmology, TransferFunction, PowerSpectrum.
//   5.  Construct Grid (with explicit ranks_per_dim) and Particles.
//   6.  generate_initial_conditions() → particles in (INIT, GLOBAL, periodic,
//       non-symplectic) state per src/ic/InitialConditions.hpp.
//   7.  INIT → PHYSICAL unit conversion:
//         pos_phys = pos_init · (rL/np)
//         vel_phys = vel_init · (100·rL/np)
//         phi_phys = phi_init · (100·rL/np)²    (no-op since phi=0)
//   8.  Migrate particles to their post-conversion-position-owning rank in
//       the 3D Cartesian topology.  Migrator's wrap_periodic=true wraps any
//       Zel'dovich-displaced positions back into [0, ng) before computing
//       the destination rank.  After migration positions are in [0, rL)
//       (we apply the periodic-wrap kernel separately because Migrator
//       writes back grid-unit wrapped positions, which do not match
//       PHYSICAL-unit positions one-to-one).
//   9.  Write the HDF5 snapshot (schema v2): every rank writes its now-resident
//       particles as a hyperslab.
//
// HACC reference:
//   - INIT mode entry:  mc3.cxx:752 (dataType == "INIT")
//   - loadParticles:    mc3.cxx:735-832 + mc3.cxx:884 (status placeholder)
//   - convertCoords (INIT→PHYSICAL): nbody/common/Particles.cxx:1064-1073
//   - ParticleExchange:    nbody/common/ParticleExchange.cxx:527-579
//   - writeAliveRestart:   nbody/cpu/ParticleActions.cxx:1323-1344
//
// Spec: ../analysis/prompts/12_implement_drivers_and_demo.md (Deliverable 5)

#include "ic/InitialConditions.hpp"
#include "ic/PowerSpectrum.hpp"
#include "ic/TransferFunction.hpp"
#include "io/IndatParser.hpp"
#include "io/SnapshotSchema.hpp"
#include "io/SnapshotWriter.hpp"
#include "TopologyOverride.hpp"
#include "Cosmology.hpp"
#include "Grid.hpp"
#include "Particles.hpp"
#include "Types.hpp"
#include "comm/Migrator.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <string>

using namespace pmk;

namespace {

// Apply per-particle scale to position and velocity (and phi).  Runs once
// per IC build; the SoA layout makes this trivial.  See
// hacc_ic_unit_conversions.md §1 for the factors.
void scale_init_to_physical(Particles& parts,
                            float scale_pos,
                            float scale_vel,
                            float scale_phi)
{
    const std::size_t n = parts.num_local();
    if (n == 0) return;
    auto pos = parts.pos();
    auto vel = parts.vel();
    auto phi = parts.phi();
    Kokkos::parallel_for(
        "pm_ic::scale_init_to_physical",
        Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(const std::size_t i) {
            pos(i, 0) *= scale_pos;
            pos(i, 1) *= scale_pos;
            pos(i, 2) *= scale_pos;
            vel(i, 0) *= scale_vel;
            vel(i, 1) *= scale_vel;
            vel(i, 2) *= scale_vel;
            phi(i)    *= scale_phi;
        });
    Kokkos::fence();
}

// Wrap positions into [0, rL).  Performed *after* migration (so the
// destination-rank lookup uses the unwrapped global position) and
// *before* the snapshot write — matches HACC's writeRestartFormat
// convertCoords periodic flip at ParticleActions.cxx:1214.
void wrap_into_box(Particles& parts, double rL)
{
    const std::size_t n = parts.num_local();
    if (n == 0) return;
    auto pos = parts.pos();
    const float rL_f = static_cast<float>(rL);
    Kokkos::parallel_for(
        "pm_ic::wrap_into_box",
        Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(const std::size_t i) {
            for (int d = 0; d < 3; ++d) {
                float x = pos(i, d);
                x -= Kokkos::floor(x / rL_f) * rL_f;
                if (x >= rL_f) x -= rL_f;
                if (x <  0.0f) x += rL_f;
                pos(i, d) = x;
            }
        });
    Kokkos::fence();
}

// Convert PHYSICAL positions back to grid-units in place, run migrate(),
// then convert back to PHYSICAL.  This is what makes the position-based
// rank lookup work: pmk::Migrator computes the destination rank by
// integer-floor of the position against per-rank cell-block boxes (in
// units of the ng grid cells).
//
// Per hacc_ic_unit_conversions.md §5 corrected, HACC's
// ParticleExchange::identifyExchangeParticles uses
// `[layoutPos[d]·boxStep[d], (layoutPos[d]+1)·boxStep[d]]` with boxStep in
// physical units (Mpc/h); ours uses cell-unit boxes.  The end state — each
// particle on the rank that owns its physical sub-box — is identical.
//
// Conversion uses `rL/ng` (grid-unit scale) regardless of the np vs ng
// difference, because the destination-rank lookup partitions the ng grid.
// For ng == np this equals the INIT->PHYSICAL `rL/np` factor exactly.
void migrate_through_grid_units(Particles& parts, Grid& grid, double rL, int ng)
{
    const float to_grid = static_cast<float>(static_cast<double>(ng) / rL);
    const float to_phys = static_cast<float>(rL / static_cast<double>(ng));

    {
        const std::size_t n = parts.num_local();
        if (n > 0) {
            auto pos = parts.pos();
            Kokkos::parallel_for(
                "pm_ic::pos_to_grid",
                Kokkos::RangePolicy<ExecSpace>(0, n),
                KOKKOS_LAMBDA(const std::size_t i) {
                    pos(i, 0) *= to_grid;
                    pos(i, 1) *= to_grid;
                    pos(i, 2) *= to_grid;
                });
            Kokkos::fence();
        }
    }

    // Migrator wraps positions into [0, ng) internally.
    migrate(parts, grid, /*wrap_periodic=*/true);

    {
        const std::size_t n_after = parts.num_local();
        if (n_after > 0) {
            auto pos = parts.pos();
            Kokkos::parallel_for(
                "pm_ic::pos_to_physical",
                Kokkos::RangePolicy<ExecSpace>(0, n_after),
                KOKKOS_LAMBDA(const std::size_t i) {
                    pos(i, 0) *= to_phys;
                    pos(i, 1) *= to_phys;
                    pos(i, 2) *= to_phys;
                });
            Kokkos::fence();
        }
    }
}

} // namespace

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int exit_code = 0;
    {
        Kokkos::ScopeGuard guard(argc, argv);

        int my_rank = 0, n_ranks = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

        if (argc != 3) {
            if (my_rank == 0)
                std::cerr << "Usage: pm_ic <indat.params> <output_snapshot.h5>\n";
            MPI_Finalize();
            return 2;
        }

        try {
            const std::string indat_path = argv[1];
            const std::string out_path   = argv[2];

            IndatParser indat(indat_path);

            const double rL    = indat.get_double("RL");
            const int    ng    = indat.get_int   ("NG");
            const int    np    = indat.get_int   ("NP");
            const double hubble= indat.get_double("HUBBLE");
            const double Ocdm  = indat.get_double("OMEGA_CDM");
            const double deut  = indat.get_double("DEUT");
            const double Onu   = indat.get_double("OMEGA_NU", 0.0);
            const double sigma8= indat.get_double("SS8");
            const double n_s   = indat.get_double("NS");
            const double w_de  = indat.get_double("W_DE",  -1.0);
            const double wa_de = indat.get_double("WA_DE", 0.0);
            const double z_in  = indat.get_double("Z_IN");
            const unsigned long seed =
                static_cast<unsigned long>(indat.get_int("I_SEED"));
            const std::string tf_path = indat.get_string("INPUT_BASE_NAME");

            const double omega_baryon = deut / (hubble * hubble);
            const double omega_m      = Ocdm + omega_baryon + Onu;
            const double a_in         = 1.0 / (1.0 + z_in);

            // ------------------------------------------------------------
            // Cartesian topology.  MPI_Dims_create returns a sorted-descending
            // tuple (highest factor in dim 0); HACC's MY_Dims_create_3D since
            // March 2023 also sorts (its sort is ascending so dim 0 has the
            // *largest* volume per HACC's comment, equivalent up to factor
            // permutation).  For nnodes ∈ {1,2,4,8} the result is identical:
            //   1 → (1,1,1), 2 → (2,1,1), 4 → (2,2,1), 8 → (2,2,2).
            // ------------------------------------------------------------
            // Explicit override (env PMKOKKOS_TOPOLOGY or indat TOPOLOGY)
            // beats the auto-derivation; see apps/TopologyOverride.hpp.
            std::array<int, 3> ranks_per_dim = {0, 0, 0};
            if (topology_override(&indat, ranks_per_dim)) {
                if (!topology_validate(ranks_per_dim, ng, n_ranks, my_rank)) {
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            } else {
                MPI_Dims_create(n_ranks, 3, ranks_per_dim.data());
                // Sort descending so dim 0 has the highest factor (matches HACC).
                std::sort(ranks_per_dim.begin(), ranks_per_dim.end(),
                          std::greater<int>());
            }

            if (my_rank == 0) {
                std::cout << "pm_ic: " << n_ranks << " ranks; topology "
                          << ranks_per_dim[0] << "x"
                          << ranks_per_dim[1] << "x"
                          << ranks_per_dim[2] << "\n"
                          << "  rL=" << rL << " Mpc/h, ng=" << ng
                          << ", np=" << np << ", z_in=" << z_in
                          << ", a_in=" << a_in << "\n"
                          << "  cosmo: Omega_m=" << omega_m
                          << " Omega_cb=" << (Ocdm + omega_baryon)
                          << " Omega_b=" << omega_baryon
                          << " h=" << hubble
                          << " ns=" << n_s << " sigma8=" << sigma8 << "\n"
                          << "  transfer function: " << tf_path << "\n";
            }

            // ------------------------------------------------------------
            // Cosmology / transfer function / power spectrum.
            // ------------------------------------------------------------
            CosmoParams cp;
            cp.omega_matter    = omega_m;
            cp.omega_cdm       = Ocdm;
            cp.omega_baryon    = omega_baryon;
            cp.omega_cb        = Ocdm + omega_baryon;
            cp.omega_nu        = Onu;
            // omega_radiation per HACC convention:
            // HACC Basedata.h:237 computes Ω_r from CMB temperature and
            // hubble.  The 2.471e-5 prefactor comes from the canonical
            // expression for photon energy density (with neutrino terms
            // handled separately via f_nu_massless and Omega_nu).  Leaving
            // this at 0 caused the prompt-12 demo's velocity disagreement
            // with HACC at z=200 (~10%); see
            // ../analysis/references/velocity_discrepancy_diagnosis.md.
            const double t_cmb = indat.get_double("T_CMB", 2.725);
            cp.omega_radiation =
                2.471e-5 * std::pow(t_cmb / 2.725, 4.0) / (hubble * hubble);
            cp.f_nu_massless   = 0.0;
            cp.f_nu_massive    = 0.0;
            cp.w               = w_de;
            cp.wa              = wa_de;
            cp.hubble          = hubble;
            cp.ns              = n_s;
            cp.sigma8          = sigma8;
            Cosmology cosmo(cp);
            TransferFunction tf(tf_path, cosmo);
            PowerSpectrum    pk(tf, n_s, sigma8);

            // ------------------------------------------------------------
            // Grid and Particles on the explicit topology.  CIC P2G needs
            // halo_width >= 1 for multi-rank correctness (per WORKLOG entry
            // for prompt 08).
            // ------------------------------------------------------------
            Grid grid(MPI_COMM_WORLD, ng, /*halo_width=*/1, ranks_per_dim);
            Particles parts(grid.comm());

            // ------------------------------------------------------------
            // Generate IC: produces np^3 particles distributed across the
            // grid's owned cells (np == ng for the reference run).
            // Output state: (INIT, GLOBAL, periodic, non-symplectic).
            // ------------------------------------------------------------
            generate_initial_conditions(parts, grid, cosmo, pk,
                                        rL, a_in, seed);

            // ------------------------------------------------------------
            // INIT -> PHYSICAL unit conversion (hacc_ic_unit_conversions.md
            // §1).  For ng==np the prompt's note collapses to:
            //   pos *= rL/np,  vel *= 100·rL/np,  phi *= (100·rL/np)².
            // For ng != np the gpscal=ng/np factor is folded in via /np.
            // ------------------------------------------------------------
            const double scale_pos = rL / static_cast<double>(np);          // Mpc/h per INIT pos unit
            const double scale_vel = 100.0 * rL / static_cast<double>(np);  // km/s per INIT vel unit
            const double scale_phi = scale_vel * scale_vel;                 // (km/s)² per INIT phi unit
            scale_init_to_physical(parts,
                                   static_cast<float>(scale_pos),
                                   static_cast<float>(scale_vel),
                                   static_cast<float>(scale_phi));

            // ------------------------------------------------------------
            // Migration.  We need to land each particle on the rank that
            // owns its physical sub-box under the same Cartesian topology
            // the snapshot writer will use.  pmk::Migrator works in
            // grid-unit coordinates (it computes destination by integer-
            // floor of position against per-rank cell-block boxes), so we
            // round-trip through INIT/grid units.
            // ------------------------------------------------------------
            migrate_through_grid_units(parts, grid, rL, ng);

            // ------------------------------------------------------------
            // Periodic flip in PHYSICAL units (HACC's writeRestartFormat
            // convertCoords periodic flip, ParticleActions.cxx:1214).  The
            // grid-unit wrap inside Migrator already wrapped grid-unit
            // positions; we redo the wrap in PHYSICAL units to keep the
            // floats in [0, rL) exactly.
            // ------------------------------------------------------------
            wrap_into_box(parts, rL);

            // ------------------------------------------------------------
            // SimulationState — fill the full v2 schema.  topology_dims is
            // mandatory for v2; we report the writer's MPI Cartesian dims.
            // ------------------------------------------------------------
            SimulationState state;
            state.a        = a_in;
            state.step     = 0;
            state.ng       = ng;
            state.np       = static_cast<int64_t>(np)
                             * static_cast<int64_t>(np)
                             * static_cast<int64_t>(np);
            state.rL       = rL;
            state.z_in     = z_in;
            state.omega_m  = omega_m;
            state.omega_cb = cp.omega_cb;
            state.omega_b  = omega_baryon;
            state.h        = hubble;
            state.n_s      = n_s;
            state.sigma_8  = sigma8;
            state.topology_dims[0] = ranks_per_dim[0];
            state.topology_dims[1] = ranks_per_dim[1];
            state.topology_dims[2] = ranks_per_dim[2];

            write_snapshot(out_path, parts, state);

            if (my_rank == 0) {
                std::cout << "pm_ic: wrote " << out_path
                          << "  (np_global=" << state.np << ")\n";
                indat.report_unused(std::cout);
            }
        } catch (const std::exception& e) {
            if (my_rank == 0)
                std::cerr << "pm_ic: error: " << e.what() << "\n";
            exit_code = 1;
        }
    }
    MPI_Finalize();
    return exit_code;
}
