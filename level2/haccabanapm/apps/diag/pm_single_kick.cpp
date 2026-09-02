// apps/diag/pm_single_kick.cpp
//
// DIAGNOSTIC DRIVER — not a permanent test.  Loads an IC snapshot, applies
// exactly ONE PM kick at mid-step cosmological scalars (matching what HACC's
// DKD inner kick does between the two half-drifts), and writes a per-rank CSV
// of (id, x, y, z, dvx, dvy, dvz) for cross-code comparison with HACC.
//
// Companion HACC driver: HACC/nbody/simulation/driver_pm_singlekick.cxx.
// Companion comparison:  analysis/scripts/compare_single_kick.py.
//
// Pipeline (mirrors HACC driver_pm.cxx:449-484 with the half-drifts removed):
//
//   1. Read snapshot via SnapshotReader (positions in PHYSICAL Mpc/h, vel in
//      km/s — same state as a fresh IC).
//   2. Build the grid topology and migrate to position-owning rank
//      (same as pm_run_main).
//   3. PHYSICAL -> simulation-internal (pos: grid units; vel: v*a^2 — i.e. P̃).
//   4. Construct Integrator(alpha=1, a_in=snapshot.a, a_fin=a_in*1.001,
//      nsteps=1) and initialize.  This sets tau, pp, fscal at a=a_in.
//   5. advance_half_step() — promotes pp/aa/fscal to MID-step scalars, exactly
//      as HACC's DKD does at driver_pm.cxx:461.
//   6. Snapshot pre-kick (id, position, velocity) on each rank.
//   7. pm_kick(parts, grid, mass=1, dt = tau * fscal) — ONE kick only; no
//      drifts, no migration, no advance_half_step after.
//   8. Snapshot post-kick velocity.
//   9. Write per-rank CSV `<output>.<rank>.csv` and a small log file with the
//      (a, tau, fscal, pp, adot, phiscal) values actually used at the kick.
//
// CSV columns (whitespace-free, comma-separated):
//   id, x_grid, y_grid, z_grid, dvx, dvy, dvz
//   pos in simulation-internal grid units [0, ng);
//   dv in simulation-internal symplectic-momentum units (the P̃ slot pmkokkos
//   stores in FIELD_VEL).
//
// Usage:
//   mpiexec -n N pm_single_kick <ic_snapshot.h5> <output_basename> \
//        [a_fin] [nsteps] [t_cmb]
//   mpiexec -n 1 pm_single_kick --two-particle <output_basename>
//   mpiexec -n N pm_single_kick --full-step <ic_snapshot.h5> <output_basename> \
//        [a_fin] [nsteps] [t_cmb]
//
// FULL-STEP MODE (--full-step).  One full Integrator::full_step from the
// shared IC.  Same CLI surface as snapshot mode (plus the --full-step
// prefix flag); dumps per-rank CSV of (id, x_grid_post, y_grid_post,
// z_grid_post, dvx, dvy, dvz) where dv = vel_post - vel_pre and pos is the
// post-step (post-second-half-drift) GLOBAL grid position.  Used as the
// kernel-level gate for the DKD migration fix
// (analysis/prompts/fix_dkd_migration.md): boundary-band dv tightening
// after the fix is the direct test of the fix's mechanism.
//
// SNAPSHOT MODE.  Reads the snapshot's cosmology (omega_m / omega_cb /
// omega_b / h / n_s / sigma_8) for the kick scalars.  The integrator
// parameters a_fin and nsteps control tau = (a_fin^α - a_in^α)/nsteps
// directly, and so must MATCH the HACC driver's values for the cross-code
// comparison to be meaningful.
//
//   Defaults: a_fin = a_in * 1.001, nsteps = 1, t_cmb = 2.725.
//   For HACC parity at the 1-node-grav_full reference, pass
//   a_fin = 1.0, nsteps = 500, t_cmb = 2.726 so the (a, fscal, tau) scalars
//   at the mid-step kick exactly match HACC's first DKD step.
//
// TWO-PARTICLE MODE (--two-particle).  Bit-identical kernel test for Test A
// of microtest_bit_identical_and_shared_ic.md.  Single-rank only.  Hardcodes
// ng=16, two particles at (4.25, 4.25, 4.25) and (11.75, 11.75, 11.75) in
// GRID units (same layout as test_pm_step_two_body_spectral), uniform
// mass=1, dt=1.0 — no integrator, no cosmology.  Writes the same CSV layout
// as snapshot mode (id, x_grid, y_grid, z_grid, dvx, dvy, dvz) plus a
// sidecar info.txt listing the kick configuration.  Companion HACC driver:
// HACC/nbody/simulation/driver_pm_twoparticle.cxx.

#include "pm_run_lib.hpp"

#include "Cosmology.hpp"
#include "Grid.hpp"
#include "Particles.hpp"
#include "Types.hpp"
#include "comm/Migrator.hpp"
#include "io/SnapshotReader.hpp"
#include "io/SnapshotSchema.hpp"
#include "pm/PMStep.hpp"
#include "time/Integrator.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <hdf5.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Read topology_dims from rank 0 and broadcast.  Copy of the helper in
// pm_run_lib.cpp — kept local to avoid coupling this diag binary's surface.
void peek_topology_dims(const std::string& path, MPI_Comm comm, int dims[3])
{
    int my_rank = 0;
    MPI_Comm_rank(comm, &my_rank);
    if (my_rank == 0) {
        hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0) {
            dims[0] = dims[1] = dims[2] = 0;
        } else {
            htri_t has = H5Aexists(file, pmk::ATTR_TOPOLOGY_DIMS);
            if (has > 0) {
                hid_t a = H5Aopen(file, pmk::ATTR_TOPOLOGY_DIMS, H5P_DEFAULT);
                H5Aread(a, H5T_NATIVE_INT, dims);
                H5Aclose(a);
            } else {
                dims[0] = dims[1] = dims[2] = 0;
            }
            H5Fclose(file);
        }
    }
    MPI_Bcast(dims, 3, MPI_INT, 0, comm);
}

void physical_to_simulation(pmk::Particles& parts, double rL, int np, double a)
{
    const std::size_t n = parts.num_local();
    if (n == 0) return;
    auto pos = parts.pos();
    auto vel = parts.vel();
    const float inv_pos = static_cast<float>(static_cast<double>(np) / rL);
    const float inv_vel_to_init =
        static_cast<float>(static_cast<double>(np) / (100.0 * rL));
    const float a2 = static_cast<float>(a * a);
    Kokkos::parallel_for(
        "single_kick::physical_to_sim",
        Kokkos::RangePolicy<pmk::ExecSpace>(0, n),
        KOKKOS_LAMBDA(const std::size_t i) {
            pos(i, 0) *= inv_pos;
            pos(i, 1) *= inv_pos;
            pos(i, 2) *= inv_pos;
            vel(i, 0) = vel(i, 0) * inv_vel_to_init * a2;
            vel(i, 1) = vel(i, 1) * inv_vel_to_init * a2;
            vel(i, 2) = vel(i, 2) * inv_vel_to_init * a2;
        });
    Kokkos::fence();
}

void migrate_through_grid_units(pmk::Particles& parts, pmk::Grid& grid,
                                double rL, int ng)
{
    const float to_grid = static_cast<float>(static_cast<double>(ng) / rL);
    const float to_phys = static_cast<float>(rL / static_cast<double>(ng));
    {
        const std::size_t n = parts.num_local();
        if (n > 0) {
            auto pos = parts.pos();
            Kokkos::parallel_for(
                "single_kick::pos_to_grid",
                Kokkos::RangePolicy<pmk::ExecSpace>(0, n),
                KOKKOS_LAMBDA(const std::size_t i) {
                    pos(i, 0) *= to_grid;
                    pos(i, 1) *= to_grid;
                    pos(i, 2) *= to_grid;
                });
            Kokkos::fence();
        }
    }
    pmk::migrate(parts, grid, /*wrap_periodic=*/true);
    {
        const std::size_t n_after = parts.num_local();
        if (n_after > 0) {
            auto pos = parts.pos();
            Kokkos::parallel_for(
                "single_kick::pos_to_physical",
                Kokkos::RangePolicy<pmk::ExecSpace>(0, n_after),
                KOKKOS_LAMBDA(const std::size_t i) {
                    pos(i, 0) *= to_phys;
                    pos(i, 1) *= to_phys;
                    pos(i, 2) *= to_phys;
                });
            Kokkos::fence();
        }
    }
}

int run_two_particle(int argc, char* argv[]);
int run_full_step(int argc, char* argv[]);

int run(int argc, char* argv[])
{
    int my_rank = 0, n_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    if (argc >= 2 && std::string(argv[1]) == "--two-particle")
        return run_two_particle(argc, argv);
    if (argc >= 2 && std::string(argv[1]) == "--full-step")
        return run_full_step(argc, argv);

    if (argc < 3 || argc > 6) {
        if (my_rank == 0) {
            std::cerr
                << "Usage: pm_single_kick <ic_snapshot.h5> <output_basename>"
                   " [a_fin] [nsteps] [t_cmb]\n"
                << "       pm_single_kick --two-particle <output_basename>\n"
                << "\n"
                << "Loads the IC snapshot, applies ONE PM kick at mid-step\n"
                << "cosmological scalars (matching HACC's DKD inner kick),"
                   " and writes per-rank CSVs <output_basename>.<rank>.csv"
                   " with\n"
                << "  id, x_grid, y_grid, z_grid, dvx, dvy, dvz\n"
                << "plus a sidecar <output_basename>.info.txt listing the\n"
                << "(a, tau, fscal, pp, adot, phiscal, dt=tau*fscal) values.\n"
                << "\n"
                << "Defaults: a_fin = a_in*1.001, nsteps = 1, t_cmb = 2.725\n"
                << "For HACC parity at the 1-node-grav_full configuration,"
                   " pass a_fin=1.0 nsteps=500 t_cmb=2.726.\n"
                << "\n"
                << "--two-particle: ng=16, two particles at (4.25)^3 and"
                   " (11.75)^3, mass=1, dt=1, single-rank only.\n";
        }
        return 2;
    }

    const std::string input_path  = argv[1];
    const std::string output_base = argv[2];
    const bool have_a_fin  = (argc >= 4);
    const bool have_nsteps = (argc >= 5);
    const bool have_t_cmb  = (argc >= 6);
    const double a_fin_arg = have_a_fin  ? std::stod(argv[3]) : -1.0;
    const int    nsteps_arg = have_nsteps ? std::stoi(argv[4]) : 1;
    const double t_cmb_arg = have_t_cmb  ? std::stod(argv[5]) : 2.725;

    // -- Topology + read ----------------------------------------------------
    int file_dims[3] = {0, 0, 0};
    peek_topology_dims(input_path, MPI_COMM_WORLD, file_dims);
    int prod = file_dims[0] * file_dims[1] * file_dims[2];
    std::array<int, 3> ranks_per_dim = {0, 0, 0};
    if (prod == n_ranks && file_dims[0] > 0 && file_dims[1] > 0 &&
        file_dims[2] > 0) {
        ranks_per_dim = { file_dims[0], file_dims[1], file_dims[2] };
    } else {
        MPI_Dims_create(n_ranks, 3, ranks_per_dim.data());
        std::sort(ranks_per_dim.begin(), ranks_per_dim.end(),
                  std::greater<int>());
    }

    pmk::Particles parts(MPI_COMM_WORLD);
    pmk::SimulationState state;
    pmk::read_snapshot(input_path, parts, state);

    const double rL    = state.rL;
    const int    ng    = state.ng;
    const int    np    = static_cast<int>(std::round(
        std::cbrt(static_cast<double>(state.np))));
    if (static_cast<int64_t>(np)*np*np != state.np) {
        if (my_rank == 0) std::cerr << "state.np is not a perfect cube\n";
        return 1;
    }
    const double a_in = state.a;

    pmk::Grid grid(MPI_COMM_WORLD, ng, /*halo_width=*/1, ranks_per_dim);
    migrate_through_grid_units(parts, grid, rL, ng);
    physical_to_simulation(parts, rL, np, a_in);

    // Mirror pm_run_main: one extra migrate to position-owning rank in
    // grid-unit coordinates.  Without this, particles whose grid positions
    // are on a different rank than where the snapshot read placed them get
    // their kicks attributed to the wrong rank.
    pmk::migrate(parts, grid);

    // -- Cosmology ----------------------------------------------------------
    pmk::CosmoParams cp;
    cp.omega_matter = state.omega_m;
    cp.omega_cb     = state.omega_cb;
    cp.omega_baryon = state.omega_b;
    cp.omega_nu     = state.omega_m - state.omega_cb - state.omega_b;
    if (cp.omega_nu < 0.0) cp.omega_nu = 0.0;
    cp.hubble       = state.h;
    cp.ns           = state.n_s;
    cp.sigma8       = state.sigma_8;
    cp.w            = -1.0;
    cp.wa           = 0.0;
    // T_CMB not in v2 schema — default canonical 2.725, override via CLI.
    const double t_cmb = t_cmb_arg;
    cp.omega_radiation =
        2.471e-5 * std::pow(t_cmb / 2.725, 4.0) / (cp.hubble * cp.hubble);
    pmk::Cosmology cosmo(cp);

    // -- Integrator: forward step + nsteps controlling tau ------------------
    const double a_fin  = have_a_fin ? a_fin_arg : a_in * 1.001;
    const int    nsteps = have_nsteps ? nsteps_arg : 1;
    if (!(a_in < a_fin)) {
        if (my_rank == 0) std::cerr << "ERROR: a_in (" << a_in
            << ") must be < a_fin (" << a_fin << ")\n";
        return 1;
    }
    if (nsteps <= 0) {
        if (my_rank == 0) std::cerr << "ERROR: nsteps must be > 0\n";
        return 1;
    }
    pmk::Integrator integ(cosmo, /*alpha=*/1.0, a_in, a_fin, nsteps);
    integ.initialize();

    // HACC's DKD inner-kick scalars come AFTER advanceHalfStep
    // (driver_pm.cxx:461).  Mirror it here so the (a, fscal, tau) we apply at
    // the kick match HACC's mid-step values.
    integ.advance_half_step();

    const double a_kick     = integ.a();
    const double pp_kick    = integ.p();
    const double tau_kick   = integ.tau_full();
    const double fscal_kick = integ.f_scal();
    const double adot_kick  = integ.a_dot();
    const double phiscal_kick = integ.phi_scal();
    const double dt_kick    = tau_kick * fscal_kick;

    // -- Capture pre-kick state --------------------------------------------
    const std::size_t n_local = parts.num_local();

    using pmk::ExecSpace;
    using pmk::MemorySpace;

    Kokkos::View<std::int64_t*, MemorySpace> id_dev(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "single_kick::id"),
        n_local);
    Kokkos::View<float*[3], MemorySpace> pos_dev(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "single_kick::pos"),
        n_local);
    Kokkos::View<float*[3], MemorySpace> vel_pre_dev(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "single_kick::vel_pre"),
        n_local);
    {
        auto id  = parts.id();
        auto pos = parts.pos();
        auto vel = parts.vel();
        Kokkos::parallel_for(
            "single_kick::capture_pre",
            Kokkos::RangePolicy<ExecSpace>(0, n_local),
            KOKKOS_LAMBDA(const std::size_t i) {
                id_dev(i)        = static_cast<std::int64_t>(id(i));
                pos_dev(i, 0)    = pos(i, 0);
                pos_dev(i, 1)    = pos(i, 1);
                pos_dev(i, 2)    = pos(i, 2);
                vel_pre_dev(i, 0) = vel(i, 0);
                vel_pre_dev(i, 1) = vel(i, 1);
                vel_pre_dev(i, 2) = vel(i, 2);
            });
        Kokkos::fence();
    }

    // -- Apply ONE PM kick at the captured mid-step scalars ----------------
    const double uniform_mass = 1.0;
    pmk::pm_kick(parts, grid, uniform_mass, dt_kick);

    // -- Capture post-kick velocity (number of locals hasn't changed) ------
    Kokkos::View<float*[3], MemorySpace> vel_post_dev(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "single_kick::vel_post"),
        n_local);
    {
        auto vel = parts.vel();
        Kokkos::parallel_for(
            "single_kick::capture_post",
            Kokkos::RangePolicy<ExecSpace>(0, n_local),
            KOKKOS_LAMBDA(const std::size_t i) {
                vel_post_dev(i, 0) = vel(i, 0);
                vel_post_dev(i, 1) = vel(i, 1);
                vel_post_dev(i, 2) = vel(i, 2);
            });
        Kokkos::fence();
    }

    // -- Mirror to host, compute dv, write CSV -----------------------------
    auto id_h       = Kokkos::create_mirror_view(id_dev);
    auto pos_h      = Kokkos::create_mirror_view(pos_dev);
    auto vel_pre_h  = Kokkos::create_mirror_view(vel_pre_dev);
    auto vel_post_h = Kokkos::create_mirror_view(vel_post_dev);
    Kokkos::deep_copy(id_h, id_dev);
    Kokkos::deep_copy(pos_h, pos_dev);
    Kokkos::deep_copy(vel_pre_h, vel_pre_dev);
    Kokkos::deep_copy(vel_post_h, vel_post_dev);

    std::ostringstream rank_path;
    rank_path << output_base << '.' << my_rank << ".csv";
    std::ofstream out(rank_path.str());
    if (!out) {
        std::cerr << "rank " << my_rank << ": cannot open "
                  << rank_path.str() << "\n";
        return 1;
    }
    out << "id,x_grid,y_grid,z_grid,dvx,dvy,dvz\n";
    out << std::setprecision(9);
    for (std::size_t i = 0; i < n_local; ++i) {
        out << id_h(i) << ','
            << pos_h(i, 0) << ',' << pos_h(i, 1) << ',' << pos_h(i, 2) << ','
            << (vel_post_h(i, 0) - vel_pre_h(i, 0)) << ','
            << (vel_post_h(i, 1) - vel_pre_h(i, 1)) << ','
            << (vel_post_h(i, 2) - vel_pre_h(i, 2)) << '\n';
    }
    out.close();

    // -- Sidecar info file (rank 0 only) ------------------------------------
    if (my_rank == 0) {
        std::ostringstream info_path;
        info_path << output_base << ".info.txt";
        std::ofstream info(info_path.str());
        info << std::setprecision(17);
        info << "code        = pmkokkos\n";
        info << "input       = " << input_path << "\n";
        info << "n_ranks     = " << n_ranks << "\n";
        info << "topology    = "
             << ranks_per_dim[0] << "x" << ranks_per_dim[1] << "x"
             << ranks_per_dim[2] << "\n";
        info << "ng          = " << ng << "\n";
        info << "np_axis     = " << np << "\n";
        info << "rL          = " << rL << "\n";
        info << "a_in        = " << a_in << "\n";
        info << "a_fin       = " << a_fin << "\n";
        info << "nsteps      = " << nsteps << "\n";
        info << "alpha       = 1.0\n";
        info << "T_CMB       = " << t_cmb << "\n";
        info << "omega_m     = " << cp.omega_matter << "\n";
        info << "omega_cb    = " << cp.omega_cb << "\n";
        info << "omega_b     = " << cp.omega_baryon << "\n";
        info << "omega_nu    = " << cp.omega_nu << "\n";
        info << "omega_rad   = " << cp.omega_radiation << "\n";
        info << "h           = " << cp.hubble << "\n";
        info << "----- kick scalars (after advance_half_step) -----\n";
        info << "a_kick      = " << a_kick << "\n";
        info << "pp_kick     = " << pp_kick << "\n";
        info << "tau_kick    = " << tau_kick << "\n";
        info << "adot_kick   = " << adot_kick << "\n";
        info << "phiscal     = " << phiscal_kick << "\n";
        info << "fscal_kick  = " << fscal_kick << "\n";
        info << "dt_kick     = " << dt_kick << "\n";
        info.close();
        std::cout << "pm_single_kick: a=" << a_kick
                  << " tau=" << tau_kick
                  << " fscal=" << fscal_kick
                  << " dt=" << dt_kick
                  << "  wrote " << n_ranks << " CSVs + info.txt\n";
    }
    return 0;
}

// Two-particle bit-identical kernel test — Test A of
// microtest_bit_identical_and_shared_ic.md.  Single-rank only.  Hardcodes
// the same configuration as test_pm_step_two_body_spectral so positions are
// bit-identical between codes by construction.
int run_two_particle(int argc, char* argv[])
{
    int my_rank = 0, n_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    if (n_ranks != 1) {
        if (my_rank == 0)
            std::cerr << "--two-particle requires a single MPI rank "
                      << "(got " << n_ranks << ")\n";
        return 1;
    }
    if (argc != 3) {
        if (my_rank == 0)
            std::cerr << "Usage: pm_single_kick --two-particle <output_basename>\n";
        return 2;
    }
    const std::string output_base = argv[2];

    // Test config (mirrors tests/test_pm_step_two_body_spectral.cpp).
    constexpr int N = 16;
    const std::array<std::array<double, 3>, 2> xyz = {{
        { 4.25,  4.25,  4.25},
        {11.75, 11.75, 11.75}
    }};
    const std::array<std::int64_t, 2> ids = {0, 1};
    const double mass = 1.0;
    const double dt   = 1.0;

    pmk::Grid       grid(MPI_COMM_WORLD, N, /*halo_width=*/1);
    pmk::Particles  parts(MPI_COMM_WORLD);
    parts.resize(xyz.size());

    using pmk::ExecSpace;
    using pmk::MemorySpace;

    // Stage positions / ids / mass on the device.
    Kokkos::View<float*[3], MemorySpace>      pos_d("tp_pos_d", xyz.size());
    Kokkos::View<float*,    MemorySpace>      mass_d("tp_mass_d", xyz.size());
    Kokkos::View<std::int64_t*, MemorySpace>  id_d("tp_id_d", xyz.size());
    auto pos_h  = Kokkos::create_mirror_view(pos_d);
    auto mass_h = Kokkos::create_mirror_view(mass_d);
    auto id_h   = Kokkos::create_mirror_view(id_d);
    for (std::size_t i = 0; i < xyz.size(); ++i) {
        pos_h(i, 0) = static_cast<float>(xyz[i][0]);
        pos_h(i, 1) = static_cast<float>(xyz[i][1]);
        pos_h(i, 2) = static_cast<float>(xyz[i][2]);
        mass_h(i)   = static_cast<float>(mass);
        id_h(i)     = ids[i];
    }
    Kokkos::deep_copy(pos_d,  pos_h);
    Kokkos::deep_copy(mass_d, mass_h);
    Kokkos::deep_copy(id_d,   id_h);

    {
        auto pos  = parts.pos();
        auto vel  = parts.vel();
        auto mass_s = parts.mass();
        auto id_s = parts.id();
        Kokkos::parallel_for("tp::seed",
            Kokkos::RangePolicy<ExecSpace>(0, xyz.size()),
            KOKKOS_LAMBDA(const int i) {
                pos(i, 0) = pos_d(i, 0);
                pos(i, 1) = pos_d(i, 1);
                pos(i, 2) = pos_d(i, 2);
                vel(i, 0) = 0.0f;
                vel(i, 1) = 0.0f;
                vel(i, 2) = 0.0f;
                mass_s(i) = mass_d(i);
                id_s(i)   = id_d(i);
            });
        Kokkos::fence();
    }

    // ONE kick — production spectral pipeline (pm_kick).  Initial velocity
    // is zero, so post-kick vel == dv == dt · F.
    pmk::pm_kick(parts, grid, mass, dt);

    // Capture, write CSV.
    Kokkos::View<float*[3], MemorySpace> vel_post_d("tp_vel_post_d", xyz.size());
    {
        auto vel = parts.vel();
        Kokkos::parallel_for("tp::capture",
            Kokkos::RangePolicy<ExecSpace>(0, xyz.size()),
            KOKKOS_LAMBDA(const int i) {
                vel_post_d(i, 0) = vel(i, 0);
                vel_post_d(i, 1) = vel(i, 1);
                vel_post_d(i, 2) = vel(i, 2);
            });
        Kokkos::fence();
    }
    auto vel_post_h = Kokkos::create_mirror_view(vel_post_d);
    Kokkos::deep_copy(vel_post_h, vel_post_d);

    std::ostringstream csv_path;
    csv_path << output_base << ".0.csv";
    std::ofstream out(csv_path.str());
    if (!out) {
        std::cerr << "cannot open " << csv_path.str() << "\n";
        return 1;
    }
    out << "id,x_grid,y_grid,z_grid,dvx,dvy,dvz\n";
    out << std::setprecision(9);
    for (std::size_t i = 0; i < xyz.size(); ++i) {
        out << ids[i] << ','
            << pos_h(i, 0) << ',' << pos_h(i, 1) << ',' << pos_h(i, 2) << ','
            << vel_post_h(i, 0) << ',' << vel_post_h(i, 1) << ','
            << vel_post_h(i, 2) << '\n';
    }
    out.close();

    std::ostringstream info_path;
    info_path << output_base << ".info.txt";
    std::ofstream info(info_path.str());
    info << std::setprecision(17);
    info << "code        = pmkokkos\n";
    info << "mode        = two-particle\n";
    info << "n_ranks     = 1\n";
    info << "ng          = " << N << "\n";
    info << "np_axis     = 2\n";
    info << "rL          = " << N << "  (grid units; rL == ng)\n";
    info << "particles   = 2\n";
    info << "p0          = (" << xyz[0][0] << ", " << xyz[0][1]
                              << ", " << xyz[0][2] << ")\n";
    info << "p1          = (" << xyz[1][0] << ", " << xyz[1][1]
                              << ", " << xyz[1][2] << ")\n";
    info << "mass        = " << mass << "\n";
    info << "dt          = " << dt << "\n";
    info << "----- kick: dv = vel_post - vel_pre, vel_pre = 0 -----\n";
    for (std::size_t i = 0; i < xyz.size(); ++i) {
        info << "dv[" << i << "]    = ("
             << vel_post_h(i, 0) << ", "
             << vel_post_h(i, 1) << ", "
             << vel_post_h(i, 2) << ")\n";
    }
    info.close();

    std::cout << "pm_single_kick --two-particle: wrote " << csv_path.str()
              << " and " << info_path.str() << "\n";
    return 0;
}

// ---- FULL-STEP MODE ------------------------------------------------------
// Reuses the snapshot-load + physical_to_simulation + migrate setup from
// run().  Runs ONE Integrator::full_step (drift / advance / kick / advance /
// drift / migrate), then dumps per-rank CSV of (id, x_post, y_post, z_post,
// dvx, dvy, dvz) with post-step GLOBAL grid positions.
int run_full_step(int argc, char* argv[])
{
    int my_rank = 0, n_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    // argv: [0]=binary, [1]=--full-step, [2]=ic, [3]=output, [4]=a_fin,
    //       [5]=nsteps, [6]=t_cmb
    if (argc < 4 || argc > 7) {
        if (my_rank == 0)
            std::cerr << "Usage: pm_single_kick --full-step <ic.h5> <outbase>"
                      << " [a_fin] [nsteps] [t_cmb]\n";
        return 2;
    }
    const std::string input_path  = argv[2];
    const std::string output_base = argv[3];
    const bool have_a_fin  = (argc >= 5);
    const bool have_nsteps = (argc >= 6);
    const bool have_t_cmb  = (argc >= 7);
    const double a_fin_arg  = have_a_fin  ? std::stod(argv[4]) : -1.0;
    const int    nsteps_arg = have_nsteps ? std::stoi(argv[5]) : 1;
    const double t_cmb_arg  = have_t_cmb  ? std::stod(argv[6]) : 2.725;

    // -- Topology + read ----------------------------------------------------
    int file_dims[3] = {0, 0, 0};
    peek_topology_dims(input_path, MPI_COMM_WORLD, file_dims);
    int prod = file_dims[0] * file_dims[1] * file_dims[2];
    std::array<int, 3> ranks_per_dim = {0, 0, 0};
    if (prod == n_ranks && file_dims[0] > 0 && file_dims[1] > 0 &&
        file_dims[2] > 0) {
        ranks_per_dim = { file_dims[0], file_dims[1], file_dims[2] };
    } else {
        MPI_Dims_create(n_ranks, 3, ranks_per_dim.data());
        std::sort(ranks_per_dim.begin(), ranks_per_dim.end(),
                  std::greater<int>());
    }

    pmk::Particles parts(MPI_COMM_WORLD);
    pmk::SimulationState state;
    pmk::read_snapshot(input_path, parts, state);

    const double rL  = state.rL;
    const int    ng  = state.ng;
    const int    np  = static_cast<int>(std::round(
        std::cbrt(static_cast<double>(state.np))));
    if (static_cast<int64_t>(np)*np*np != state.np) {
        if (my_rank == 0) std::cerr << "state.np is not a perfect cube\n";
        return 1;
    }
    const double a_in = state.a;

    pmk::Grid grid(MPI_COMM_WORLD, ng, /*halo_width=*/1, ranks_per_dim);
    migrate_through_grid_units(parts, grid, rL, ng);
    physical_to_simulation(parts, rL, np, a_in);
    pmk::migrate(parts, grid);

    // -- Cosmology ----------------------------------------------------------
    pmk::CosmoParams cp;
    cp.omega_matter = state.omega_m;
    cp.omega_cb     = state.omega_cb;
    cp.omega_baryon = state.omega_b;
    cp.omega_nu     = state.omega_m - state.omega_cb - state.omega_b;
    if (cp.omega_nu < 0.0) cp.omega_nu = 0.0;
    cp.hubble       = state.h;
    cp.ns           = state.n_s;
    cp.sigma8       = state.sigma_8;
    cp.w            = -1.0;
    cp.wa           = 0.0;
    const double t_cmb = t_cmb_arg;
    cp.omega_radiation =
        2.471e-5 * std::pow(t_cmb / 2.725, 4.0) / (cp.hubble * cp.hubble);
    pmk::Cosmology cosmo(cp);

    // -- Integrator parity with HACC's first DKD step -----------------------
    const double a_fin  = have_a_fin ? a_fin_arg : a_in * 1.001;
    const int    nsteps = have_nsteps ? nsteps_arg : 1;
    if (!(a_in < a_fin)) {
        if (my_rank == 0) std::cerr << "ERROR: a_in < a_fin required\n";
        return 1;
    }
    if (nsteps <= 0) {
        if (my_rank == 0) std::cerr << "ERROR: nsteps must be > 0\n";
        return 1;
    }
    pmk::Integrator integ(cosmo, /*alpha=*/1.0, a_in, a_fin, nsteps);
    integ.initialize();

    // -- Capture pre-step state (id, pre-vel).  We use the IC id as the
    //    permanent id; positions snapshotted here are pre-step and discarded
    //    (post-step pos is what's written to CSV). ---------------------------
    const std::size_t n_local_pre = parts.num_local();

    using pmk::ExecSpace;
    using pmk::MemorySpace;

    Kokkos::View<std::int64_t*, MemorySpace> id_pre_dev(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "fs::id_pre"),
        n_local_pre);
    Kokkos::View<float*[3], MemorySpace> vel_pre_dev(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "fs::vel_pre"),
        n_local_pre);
    {
        auto id  = parts.id();
        auto vel = parts.vel();
        Kokkos::parallel_for("fs::capture_pre",
            Kokkos::RangePolicy<ExecSpace>(0, n_local_pre),
            KOKKOS_LAMBDA(const std::size_t i) {
                id_pre_dev(i)    = static_cast<std::int64_t>(id(i));
                vel_pre_dev(i, 0) = vel(i, 0);
                vel_pre_dev(i, 1) = vel(i, 1);
                vel_pre_dev(i, 2) = vel(i, 2);
            });
        Kokkos::fence();
    }

    // -- ONE full DKD step.  full_step internally drifts, migrates (post-fix),
    //    kicks, advances, drifts, and migrates again — particles may move to
    //    a different rank.  We id-match across the migration boundary below.
    const double uniform_mass = 1.0;
    integ.full_step(parts, grid, uniform_mass);

    const double a_post     = integ.a();
    const double pp_post    = integ.p();
    const double tau_kick   = integ.tau_full();
    const double fscal_kick = integ.f_scal();
    const double dt_kick    = tau_kick * fscal_kick;

    // -- Capture post-step state (id, pos, vel) ON THIS RANK ----------------
    const std::size_t n_local_post = parts.num_local();
    Kokkos::View<std::int64_t*, MemorySpace> id_post_dev(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "fs::id_post"),
        n_local_post);
    Kokkos::View<float*[3], MemorySpace> pos_post_dev(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "fs::pos_post"),
        n_local_post);
    Kokkos::View<float*[3], MemorySpace> vel_post_dev(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "fs::vel_post"),
        n_local_post);
    {
        auto id  = parts.id();
        auto pos = parts.pos();
        auto vel = parts.vel();
        Kokkos::parallel_for("fs::capture_post",
            Kokkos::RangePolicy<ExecSpace>(0, n_local_post),
            KOKKOS_LAMBDA(const std::size_t i) {
                id_post_dev(i)     = static_cast<std::int64_t>(id(i));
                pos_post_dev(i, 0) = pos(i, 0);
                pos_post_dev(i, 1) = pos(i, 1);
                pos_post_dev(i, 2) = pos(i, 2);
                vel_post_dev(i, 0) = vel(i, 0);
                vel_post_dev(i, 1) = vel(i, 1);
                vel_post_dev(i, 2) = vel(i, 2);
            });
        Kokkos::fence();
    }

    // -- Mirror to host -----------------------------------------------------
    auto id_pre_h   = Kokkos::create_mirror_view(id_pre_dev);
    auto vel_pre_h  = Kokkos::create_mirror_view(vel_pre_dev);
    auto id_post_h  = Kokkos::create_mirror_view(id_post_dev);
    auto pos_post_h = Kokkos::create_mirror_view(pos_post_dev);
    auto vel_post_h = Kokkos::create_mirror_view(vel_post_dev);
    Kokkos::deep_copy(id_pre_h,   id_pre_dev);
    Kokkos::deep_copy(vel_pre_h,  vel_pre_dev);
    Kokkos::deep_copy(id_post_h,  id_post_dev);
    Kokkos::deep_copy(pos_post_h, pos_post_dev);
    Kokkos::deep_copy(vel_post_h, vel_post_dev);

    // -- All-to-all exchange of pre-vel keyed by id so each rank gets its
    //    post-step particles' pre-step velocities ---------------------------
    // Strategy: every rank Allgathers all pre-step (id, vel_pre) records;
    // local ranks then look up by id.  N is ~2M, 7 floats per record ≈ 56MB
    // total per Allgather — acceptable for diagnostic.  For 8 ranks this is
    // 56 MB / 8 = 7 MB per rank message, well within MPI limits.
    const long long my_npre = static_cast<long long>(n_local_pre);
    long long total_npre = 0;
    MPI_Allreduce(&my_npre, &total_npre, 1, MPI_LONG_LONG, MPI_SUM,
                  MPI_COMM_WORLD);

    std::vector<int> counts(n_ranks, 0), displs(n_ranks, 0);
    int my_npre_i = static_cast<int>(my_npre);
    MPI_Allgather(&my_npre_i, 1, MPI_INT, counts.data(), 1, MPI_INT,
                  MPI_COMM_WORLD);
    int total_i = 0;
    for (int r = 0; r < n_ranks; ++r) {
        displs[r] = total_i;
        total_i  += counts[r];
    }

    std::vector<std::int64_t> all_ids(total_i, 0);
    {
        std::vector<std::int64_t> my_ids(n_local_pre);
        for (std::size_t i = 0; i < n_local_pre; ++i)
            my_ids[i] = id_pre_h(i);
        MPI_Allgatherv(my_ids.data(), my_npre_i, MPI_LONG_LONG,
                       all_ids.data(), counts.data(), displs.data(),
                       MPI_LONG_LONG, MPI_COMM_WORLD);
    }
    std::vector<int> counts3(n_ranks, 0), displs3(n_ranks, 0);
    for (int r = 0; r < n_ranks; ++r) {
        counts3[r] = counts[r] * 3;
        displs3[r] = displs[r] * 3;
    }
    std::vector<float> all_vels(total_i * 3, 0.0f);
    {
        std::vector<float> my_vels(n_local_pre * 3);
        for (std::size_t i = 0; i < n_local_pre; ++i) {
            my_vels[3*i + 0] = vel_pre_h(i, 0);
            my_vels[3*i + 1] = vel_pre_h(i, 1);
            my_vels[3*i + 2] = vel_pre_h(i, 2);
        }
        MPI_Allgatherv(my_vels.data(), my_npre_i * 3, MPI_FLOAT,
                       all_vels.data(), counts3.data(), displs3.data(),
                       MPI_FLOAT, MPI_COMM_WORLD);
    }
    // Build id → pre-vel lookup map (one per rank — cheap at ~2M entries).
    std::unordered_map<std::int64_t, std::array<float, 3>> prevel_by_id;
    prevel_by_id.reserve(total_i * 2);
    for (int k = 0; k < total_i; ++k) {
        std::array<float, 3> v = {
            all_vels[3*k + 0], all_vels[3*k + 1], all_vels[3*k + 2]};
        prevel_by_id.emplace(all_ids[k], v);
    }

    // -- Write per-rank CSV -------------------------------------------------
    std::ostringstream csv_path;
    csv_path << output_base << '.' << my_rank << ".csv";
    std::ofstream out(csv_path.str());
    if (!out) {
        std::cerr << "rank " << my_rank << ": cannot open " << csv_path.str()
                  << "\n";
        return 1;
    }
    out << "id,x_grid,y_grid,z_grid,dvx,dvy,dvz\n";
    out << std::setprecision(9);
    int n_missing = 0;
    for (std::size_t i = 0; i < n_local_post; ++i) {
        auto it = prevel_by_id.find(id_post_h(i));
        if (it == prevel_by_id.end()) { ++n_missing; continue; }
        const float dvx = vel_post_h(i, 0) - it->second[0];
        const float dvy = vel_post_h(i, 1) - it->second[1];
        const float dvz = vel_post_h(i, 2) - it->second[2];
        out << id_post_h(i) << ','
            << pos_post_h(i, 0) << ',' << pos_post_h(i, 1) << ','
            << pos_post_h(i, 2) << ','
            << dvx << ',' << dvy << ',' << dvz << '\n';
    }
    out.close();
    if (n_missing > 0) {
        std::cerr << "rank " << my_rank << ": " << n_missing
                  << " post-step ids missing from pre-vel allgather\n";
    }

    // -- Sidecar info file --------------------------------------------------
    if (my_rank == 0) {
        std::ostringstream info_path;
        info_path << output_base << ".info.txt";
        std::ofstream info(info_path.str());
        info << std::setprecision(17);
        info << "code        = pmkokkos\n";
        info << "mode        = full-step\n";
        info << "input       = " << input_path << "\n";
        info << "n_ranks     = " << n_ranks << "\n";
        info << "topology    = "
             << ranks_per_dim[0] << "x" << ranks_per_dim[1] << "x"
             << ranks_per_dim[2] << "\n";
        info << "ng          = " << ng << "\n";
        info << "np_axis     = " << np << "\n";
        info << "rL          = " << rL << "\n";
        info << "a_in        = " << a_in << "\n";
        info << "a_fin       = " << a_fin << "\n";
        info << "nsteps      = " << nsteps << "\n";
        info << "alpha       = 1.0\n";
        info << "T_CMB       = " << t_cmb << "\n";
        info << "omega_m     = " << cp.omega_matter << "\n";
        info << "omega_cb    = " << cp.omega_cb << "\n";
        info << "h           = " << cp.hubble << "\n";
        info << "----- end-of-step scalars -----\n";
        info << "a_post      = " << a_post << "\n";
        info << "pp_post     = " << pp_post << "\n";
        info << "tau_full    = " << tau_kick << "\n";
        info << "fscal_kick  = " << fscal_kick << "\n";
        info << "dt_kick     = " << dt_kick << "\n";
        info.close();
        std::cout << "pm_single_kick --full-step: wrote " << n_ranks
                  << " CSVs + info.txt to " << output_base << "*\n";
    }
    return 0;
}

} // namespace

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int exit_code = 0;
    {
        Kokkos::ScopeGuard guard(argc, argv);
        try {
            exit_code = run(argc, argv);
        } catch (const std::exception& e) {
            int my_rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
            if (my_rank == 0)
                std::cerr << "pm_single_kick: " << e.what() << "\n";
            exit_code = 1;
        }
    }
    MPI_Finalize();
    return exit_code;
}
