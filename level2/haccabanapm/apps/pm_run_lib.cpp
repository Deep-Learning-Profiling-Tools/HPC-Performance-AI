// apps/pm_run_lib.cpp
// Implementation of pm_run_main: the prompt-13 evolution pipeline.
//
// User-facing docs: docs/RUNNING.md, docs/RUNTIME_CONTROL.md.
//
// CONFIGURATION PRECEDENCE (the surprising part of this driver).  Unlike
// pm_ic, which takes its whole configuration from the indat, pm_run takes
// cosmology from the SNAPSHOT and reads only five keys from the indat:
//
//   from indat:     Z_FIN, N_STEPS  (required)
//                   T_CMB           (default 2.725 — the v2 schema does not
//                                    carry it, so it cannot come from the file)
//                   FULL_ALIVE_DUMP (optional dump schedule)
//                   TOPOLOGY        (optional, via apps/TopologyOverride.hpp)
//
//   from snapshot:  rL, ng, np, a_in, omega_m, omega_cb, omega_b, h, n_s,
//                   sigma_8, topology_dims
//
// Consequence worth stating explicitly: W_DE and WA_DE are NOT read here.
// This driver hardcodes cp.w = -1.0 and cp.wa = 0.0 (see the CosmoParams
// block below), so a non-LCDM dark-energy setting in the indat affects
// pm_ic only and is silently ignored during evolution.  omega_nu is
// likewise reconstructed from snapshot attributes rather than read from
// OMEGA_NU.  The validated parameter region is LCDM.
//
// Pipeline:
//   1.  Parse indat for Z_FIN, N_STEPS, T_CMB, FULL_ALIVE_DUMP.  Validate.
//   2.  Discover topology dims from snapshot, build Cartesian Grid.
//   3.  Read snapshot; migrate to position-owning rank.
//   4.  PHYSICAL → INIT/sym (vel *= a²).
//   5.  Build Cosmology (snapshot wins on cosmology; T_CMB from indat).
//   6.  Integrator(α=1, a_in, a_fin, N_STEPS); pre-loop migrate.
//   7.  Per DKD step: integ.full_step; if step ∈ FULL_ALIVE_DUMP, write
//       intermediate snapshot named `<output>.step.<i>.h5`.
//   8.  After loop: sym→non-sym + INIT→PHYSICAL, periodic-wrap, migrate,
//       write final snapshot.

#include "pm_run_lib.hpp"

#include "io/IndatParser.hpp"
#include "TopologyOverride.hpp"
#include "io/SnapshotReader.hpp"
#include "io/SnapshotSchema.hpp"
#include "io/SnapshotWriter.hpp"
#include "Cosmology.hpp"
#include "Grid.hpp"
#include "Particles.hpp"
#include "Types.hpp"
#include "comm/Migrator.hpp"
#include "time/Integrator.hpp"
#include "time/PMDriver.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <hdf5.h>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace pmk {

namespace {

// Read the topology_dims attribute from rank 0, broadcast to all.
void peek_topology_dims(const std::string& path,
                        MPI_Comm comm,
                        int dims_out[3])
{
    int my_rank = 0;
    MPI_Comm_rank(comm, &my_rank);
    if (my_rank == 0) {
        hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0) {
            dims_out[0] = dims_out[1] = dims_out[2] = 0;
        } else {
            htri_t has = H5Aexists(file, ATTR_TOPOLOGY_DIMS);
            if (has > 0) {
                hid_t a = H5Aopen(file, ATTR_TOPOLOGY_DIMS, H5P_DEFAULT);
                H5Aread(a, H5T_NATIVE_INT, dims_out);
                H5Aclose(a);
            } else {
                dims_out[0] = dims_out[1] = dims_out[2] = 0;
            }
            H5Fclose(file);
        }
    }
    MPI_Bcast(dims_out, 3, MPI_INT, 0, comm);
}

void physical_to_simulation(Particles& parts, double rL, int np, double a)
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
        "pm_run::physical_to_sim",
        Kokkos::RangePolicy<ExecSpace>(0, n),
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

void simulation_to_physical(Particles& parts, double rL, int np, double a)
{
    const std::size_t n = parts.num_local();
    if (n == 0) return;

    auto pos = parts.pos();
    auto vel = parts.vel();
    const float scale_pos = static_cast<float>(rL / static_cast<double>(np));
    const float scale_vel = static_cast<float>(100.0 * rL / static_cast<double>(np));
    const float inv_a2 = static_cast<float>(1.0 / (a * a));

    Kokkos::parallel_for(
        "pm_run::sim_to_physical",
        Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(const std::size_t i) {
            pos(i, 0) *= scale_pos;
            pos(i, 1) *= scale_pos;
            pos(i, 2) *= scale_pos;
            vel(i, 0) = vel(i, 0) * inv_a2 * scale_vel;
            vel(i, 1) = vel(i, 1) * inv_a2 * scale_vel;
            vel(i, 2) = vel(i, 2) * inv_a2 * scale_vel;
        });
    Kokkos::fence();
}

void wrap_into_box(Particles& parts, double rL)
{
    const std::size_t n = parts.num_local();
    if (n == 0) return;
    auto pos = parts.pos();
    const float rL_f = static_cast<float>(rL);
    Kokkos::parallel_for(
        "pm_run::wrap_into_box",
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

void migrate_through_grid_units(Particles& parts, Grid& grid, double rL, int ng)
{
    const float to_grid = static_cast<float>(static_cast<double>(ng) / rL);
    const float to_phys = static_cast<float>(rL / static_cast<double>(ng));
    {
        const std::size_t n = parts.num_local();
        if (n > 0) {
            auto pos = parts.pos();
            Kokkos::parallel_for(
                "pm_run::pos_to_grid",
                Kokkos::RangePolicy<ExecSpace>(0, n),
                KOKKOS_LAMBDA(const std::size_t i) {
                    pos(i, 0) *= to_grid;
                    pos(i, 1) *= to_grid;
                    pos(i, 2) *= to_grid;
                });
            Kokkos::fence();
        }
    }
    migrate(parts, grid, /*wrap_periodic=*/true);
    {
        const std::size_t n_after = parts.num_local();
        if (n_after > 0) {
            auto pos = parts.pos();
            Kokkos::parallel_for(
                "pm_run::pos_to_physical",
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

std::vector<int> parse_full_alive_dump(const std::string& raw, int nsteps,
                                       int my_rank)
{
    std::vector<int> out;
    if (raw.empty()) return out;
    std::istringstream iss(raw);
    int v;
    while (iss >> v) {
        if (v < 0 || v >= nsteps) {
            if (my_rank == 0) {
                std::cerr << "pm_run: ignoring FULL_ALIVE_DUMP step "
                          << v << " (outside [0, " << nsteps << "))\n";
            }
            continue;
        }
        out.push_back(v);
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

std::string intermediate_path(const std::string& base, int step)
{
    auto dot = base.find_last_of('.');
    auto slash = base.find_last_of('/');
    if (dot != std::string::npos &&
        (slash == std::string::npos || dot > slash)) {
        return base.substr(0, dot) + ".step." + std::to_string(step) +
               base.substr(dot);
    }
    return base + ".step." + std::to_string(step);
}

void write_intermediate_snapshot(const std::string& path,
                                 Particles& parts,
                                 Grid& grid,
                                 const Integrator& integ,
                                 SimulationState& state,
                                 double rL, int ng, int np_axis,
                                 int my_rank)
{
    const double a_now = integ.a();
    simulation_to_physical(parts, rL, np_axis, a_now);
    wrap_into_box(parts, rL);
    migrate_through_grid_units(parts, grid, rL, ng);

    SimulationState s = state;
    s.a    = a_now;
    s.step = integ.step();
    write_snapshot(path, parts, s);

    if (my_rank == 0) {
        std::cout << "pm_run: intermediate snapshot " << path
                  << "  (a=" << a_now << ", step=" << s.step << ")\n";
    }

    // Restore in-memory state for further evolution.
    physical_to_simulation(parts, rL, np_axis, a_now);
}

} // namespace

int pm_run_main(const std::string& input_path,
                const std::string& output_path,
                const std::string& indat_path,
                MPI_Comm           comm)
{
    int my_rank = 0, n_ranks = 1;
    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &n_ranks);

    try {
        // ----------------------------------------------------------------
        // Parse indat early so an indat TOPOLOGY override is available before
        // the topology is chosen (cosmology fields are read further below).
        // ----------------------------------------------------------------
        IndatParser indat(indat_path);

        // ----------------------------------------------------------------
        // Choose topology.  Precedence: explicit override (env
        // PMKOKKOS_TOPOLOGY, then indat TOPOLOGY) beats the snapshot's
        // topology_dims, which beats MPI_Dims_create+sort.  The env override
        // is what lets an existing IC be re-decomposed at run time.
        // ng-divisibility is warned about after the snapshot ng is known.
        // ----------------------------------------------------------------
        std::array<int, 3> ranks_per_dim = {0, 0, 0};
        bool topo_from_override = topology_override(&indat, ranks_per_dim);
        if (topo_from_override) {
            const long prod_ov = static_cast<long>(ranks_per_dim[0]) *
                                 ranks_per_dim[1] * ranks_per_dim[2];
            if (ranks_per_dim[0] <= 0 || ranks_per_dim[1] <= 0 ||
                ranks_per_dim[2] <= 0 || prod_ov != n_ranks) {
                if (my_rank == 0)
                    std::cerr << "ERROR: topology " << ranks_per_dim[0] << "x"
                              << ranks_per_dim[1] << "x" << ranks_per_dim[2]
                              << " (product " << prod_ov << ") does not match "
                              << n_ranks << " ranks\n";
                MPI_Abort(comm, 1);
            }
        } else {
            int file_dims[3] = {0, 0, 0};
            peek_topology_dims(input_path, comm, file_dims);
            int prod = file_dims[0] * file_dims[1] * file_dims[2];
            if (prod == n_ranks &&
                file_dims[0] > 0 && file_dims[1] > 0 && file_dims[2] > 0) {
                ranks_per_dim = { file_dims[0], file_dims[1], file_dims[2] };
            } else {
                if (my_rank == 0 && prod != 0)
                    std::cerr << "pm_run: file topology ("
                              << file_dims[0] << "," << file_dims[1] << ","
                              << file_dims[2] << ") doesn't match "
                              << n_ranks << " ranks; recomputing\n";
                MPI_Dims_create(n_ranks, 3, ranks_per_dim.data());
                std::sort(ranks_per_dim.begin(), ranks_per_dim.end(),
                          std::greater<int>());
            }
        }

        // ----------------------------------------------------------------
        // Read the snapshot.
        // ----------------------------------------------------------------
        Particles parts(comm);
        SimulationState state;
        read_snapshot(input_path, parts, state);

        const double rL    = state.rL;
        const int    ng    = state.ng;
        const int    np_axis = static_cast<int>(std::round(
            std::cbrt(static_cast<double>(state.np))));
        if (static_cast<int64_t>(np_axis) * np_axis * np_axis != state.np) {
            throw std::runtime_error(
                "pm_run: state.np is not a perfect cube");
        }

        // Warn on uneven blocks now that the snapshot ng is known (an explicit
        // override may not divide the grid; heFFTe tolerates it).
        if (topo_from_override && my_rank == 0 &&
            (ng % ranks_per_dim[0] != 0 || ng % ranks_per_dim[1] != 0 ||
             ng % ranks_per_dim[2] != 0)) {
            std::cerr << "WARNING: ng=" << ng << " is not divisible by topology "
                      << ranks_per_dim[0] << "x" << ranks_per_dim[1] << "x"
                      << ranks_per_dim[2]
                      << "; blocks are uneven (heFFTe tolerates it, but load "
                         "balance drifts and HACC comparability weakens)\n";
        }

        const double a_in = state.a;

        // ----------------------------------------------------------------
        // Parse indat.  Snapshot wins on cosmology except T_CMB (not in v2).
        // (IndatParser already constructed above for the TOPOLOGY override.)
        // ----------------------------------------------------------------
        const double z_fin   = indat.get_double("Z_FIN");
        const int    nsteps  = indat.get_int("N_STEPS");
        const double t_cmb   = indat.get_double("T_CMB", 2.725);
        const std::string full_alive_raw =
            indat.has("FULL_ALIVE_DUMP")
                ? indat.get_string("FULL_ALIVE_DUMP") : std::string();

        const double a_fin = 1.0 / (1.0 + z_fin);

        if (nsteps <= 0) {
            throw std::runtime_error(
                "pm_run: N_STEPS must be > 0 (got " +
                std::to_string(nsteps) + ")");
        }
        if (!(a_in < a_fin)) {
            std::ostringstream o;
            o << "pm_run: a_in (" << a_in
              << ") from snapshot must be < a_fin (" << a_fin
              << ") computed from indat Z_FIN=" << z_fin;
            throw std::runtime_error(o.str());
        }

        const std::vector<int> alive_steps =
            parse_full_alive_dump(full_alive_raw, nsteps, my_rank);

        // ----------------------------------------------------------------
        // Grid + migrate.  PHYSICAL → INIT/sym.
        // ----------------------------------------------------------------
        Grid grid(comm, ng, /*halo_width=*/1, ranks_per_dim);
        migrate_through_grid_units(parts, grid, rL, ng);
        physical_to_simulation(parts, rL, np_axis, a_in);

        // ----------------------------------------------------------------
        // Cosmology — from snapshot's metadata.
        // ----------------------------------------------------------------
        CosmoParams cp;
        cp.omega_matter = state.omega_m;
        cp.omega_cb     = state.omega_cb;
        cp.omega_baryon = state.omega_b;
        cp.omega_nu     = state.omega_m - state.omega_cb - state.omega_b;
        if (cp.omega_nu < 0.0) cp.omega_nu = 0.0;
        cp.hubble       = state.h;
        cp.ns           = state.n_s;
        cp.sigma8       = state.sigma_8;
        // Hardcoded LCDM: the indat's W_DE / WA_DE are deliberately NOT
        // consulted here (see the precedence block at the top of this file).
        // The v2 snapshot schema carries no dark-energy attributes, so an
        // evolution run cannot recover a non-LCDM equation of state from the
        // file either; changing this needs a schema bump, not just a read.
        cp.w            = -1.0;
        cp.wa           = 0.0;
        cp.omega_radiation =
            2.471e-5 * std::pow(t_cmb / 2.725, 4.0)
                     / (cp.hubble * cp.hubble);
        Cosmology cosmo(cp);

        // ----------------------------------------------------------------
        // Integrator + step loop with optional intermediate dumps.
        // ----------------------------------------------------------------
        Integrator integ(cosmo, /*alpha=*/1.0, a_in, a_fin, nsteps);
        integ.initialize();

        if (my_rank == 0) {
            std::cout << "pm_run: " << n_ranks << " ranks; topology "
                      << ranks_per_dim[0] << "x"
                      << ranks_per_dim[1] << "x"
                      << ranks_per_dim[2]
                      << ", ng=" << ng << ", np=" << np_axis
                      << ", z_in=" << (1.0/a_in - 1.0)
                      << " z_fin=" << z_fin
                      << " (a_in=" << a_in << " a_fin=" << a_fin << ")"
                      << ", nsteps=" << nsteps << "\n";
            if (!alive_steps.empty()) {
                std::cout << "pm_run: FULL_ALIVE_DUMP steps:";
                for (int s : alive_steps) std::cout << ' ' << s;
                std::cout << "\n";
            }
        }

        migrate(parts, grid);

        // Opt-in per-step wall timing for scaling studies (E1/E4/E4b).
        // Enabled by PMKOKKOS_TIMING=1.  Each step is bracketed by a barrier
        // so the MPI_Wtime delta reflects the whole rank set arriving together;
        // the per-step cost reported is the MAX over ranks (the slowest rank is
        // the true scaling wall).  Rank 0 emits a per-step CSV and a summary
        // (median + MAD of steps [warmup, nsteps), warmup defaults to 5).
        const bool timing = [] {
            const char* e = std::getenv("PMKOKKOS_TIMING");
            return e && e[0] && e[0] != '0';
        }();
        int warmup = 5;
        if (const char* w = std::getenv("PMKOKKOS_TIMING_WARMUP")) {
            const int wv = std::atoi(w);
            if (wv >= 0) warmup = wv;
        }
        std::vector<double> step_max;   // max-over-ranks wall per step (rank 0)
        if (timing) step_max.reserve(nsteps);

        if (timing && my_rank == 0) {
            std::cout << "pm_run_timing: provenance"
                      << " ranks=" << n_ranks
                      << " topology=" << ranks_per_dim[0] << "x"
                      << ranks_per_dim[1] << "x" << ranks_per_dim[2]
                      << " ng=" << ng << " np=" << np_axis
                      << " vector_length=" << VectorLength
                      << " nsteps=" << nsteps << " warmup=" << warmup
                      << "\npm_run_timing: step,t_step_max_s,t_step_rank0_s\n";
        }

        const double uniform_mass = 1.0;
        std::size_t alive_idx = 0;
        for (int s = 0; s < nsteps; ++s) {
            double t_local = 0.0;
            if (timing) {
                MPI_Barrier(comm);
                t_local = MPI_Wtime();
            }

            integ.full_step(parts, grid, uniform_mass);

            if (timing) {
                t_local = MPI_Wtime() - t_local;
                double t_max = t_local;
                MPI_Reduce(&t_local, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
                if (my_rank == 0) {
                    step_max.push_back(t_max);
                    std::cout << "pm_run_timing: " << s << ',' << t_max << ','
                              << t_local << '\n';
                }
            }

            while (alive_idx < alive_steps.size() &&
                   alive_steps[alive_idx] == s) {
                const std::string ip = intermediate_path(output_path, s);
                write_intermediate_snapshot(ip, parts, grid, integ, state,
                                            rL, ng, np_axis, my_rank);
                state.topology_dims[0] = ranks_per_dim[0];
                state.topology_dims[1] = ranks_per_dim[1];
                state.topology_dims[2] = ranks_per_dim[2];
                ++alive_idx;
            }
        }

        if (timing && my_rank == 0) {
            // Summary over the timed window [warmup, nsteps): median + MAD.
            std::vector<double> w(step_max.begin() +
                                      std::min<std::size_t>(warmup, step_max.size()),
                                  step_max.end());
            if (!w.empty()) {
                std::sort(w.begin(), w.end());
                const double median = w[w.size() / 2];
                std::vector<double> ad(w.size());
                for (std::size_t i = 0; i < w.size(); ++i)
                    ad[i] = std::fabs(w[i] - median);
                std::sort(ad.begin(), ad.end());
                const double mad = ad[ad.size() / 2];
                const double total = std::accumulate(step_max.begin(),
                                                     step_max.end(), 0.0);
                std::cout << "pm_run_timing: summary"
                          << " steps_timed=" << step_max.size()
                          << " window=[" << warmup << ',' << nsteps << ')'
                          << " median_s=" << median
                          << " mad_s=" << mad
                          << " total_s=" << total << '\n';
            }
        }

        const double a_out = integ.a();
        simulation_to_physical(parts, rL, np_axis, a_out);
        wrap_into_box(parts, rL);
        migrate_through_grid_units(parts, grid, rL, ng);

        state.a    = a_out;
        state.step = integ.step();
        state.topology_dims[0] = ranks_per_dim[0];
        state.topology_dims[1] = ranks_per_dim[1];
        state.topology_dims[2] = ranks_per_dim[2];

        write_snapshot(output_path, parts, state);

        if (my_rank == 0) {
            std::cout << "pm_run: wrote " << output_path
                      << "  (a_out=" << a_out
                      << ", step=" << state.step << ")\n";
            indat.report_unused(std::cout);
        }
    } catch (const std::exception& e) {
        if (my_rank == 0)
            std::cerr << "pm_run: error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

} // namespace pmk
