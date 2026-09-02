// HPC-Performance-AI addition (not part of upstream HACCabanaPM).
//
// snapshot_check: correctness checks on a pm_run output snapshot, used by
// level2/haccabanapm/validate.sh.  Upstream has no standalone reference
// check for pm_run output; its end-to-end test (tests/test_pm_run_end_to_end.cpp,
// needs GoogleTest) asserts that the particle count is conserved, that
// `step == N_STEPS`, that `a` advanced, and that every position lies in
// [0, rL).  This program applies the same criteria to the snapshots written
// by pm_ic / pm_run, plus a few cheap invariants of the scheme:
//
//   1. particle count:   N(evolved) == N(ic) == attribute np  (nothing lost or
//                        duplicated by the strict-ownership migration)
//   2. step == --nsteps, a_out == 1/(1+--zfin) to 1e-6 relative, a_out > a_in
//   3. positions finite and in [0, rL) in every component (periodic wrap)
//   4. velocities finite; masses finite, > 0, all equal (single species)
//   5. ids form a permutation of [0, np): each id appears exactly once
//   6. total momentum |sum_i v_i| / (N * v_rms) <= --pmom-tol per component
//      (default 1e-6; the CIC deposit / CIC interpolation force pair is
//      antisymmetric, so the kick conserves sum_i v_i up to float32 round-off
//      -- measured ~1e-10 -- and the drift and the a-dependent factors never
//      change it)
//   7. growth: rms density contrast of the particles CIC-deposited on a
//      (ng/8)^3 coarse mesh (8 Mpc/h cells at the 1 Mpc/h upstream spacing;
//      CIC because plain counts alias on a perturbed lattice), evolved / ic,
//      compared with the linear-theory growth ratio D(a_out)/D(a_in) for flat
//      LambdaCDM with the snapshot's omega_m (Heath 1977 integral, radiation
//      neglected). PASS if the ratio is within [--growth-lo, --growth-hi]
//      times the linear value (defaults 0.5 and 2.0: a missing, mis-signed or
//      grossly mis-scaled force fails; the ~1.3x nonlinear boost at z=0 and
//      the few-% integrator / radiation-era error pass).
//
// Serial HDF5 C API only (no MPI, no Kokkos); reads datasets in chunks.
// Exit 0 with a final "snapshot_check: PASS" line, 1 with "snapshot_check: FAIL".
//
//   snapshot_check <ic.h5> <evolved.h5> --nsteps N --zfin Z
//                  [--pmom-tol 1e-6] [--growth-lo 0.5] [--growth-hi 2.0]

#include <hdf5.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Meta {
    int schema = 0, step = 0, ng = 0;
    int64_t np = 0;
    double a = 0, rL = 0, z_in = 0, omega_m = 0;
};

[[noreturn]] void die(const std::string& msg)
{
    std::fprintf(stderr, "snapshot_check: ERROR: %s\n", msg.c_str());
    std::printf("snapshot_check: FAIL\n");
    std::exit(1);
}

template <typename T>
void read_attr(hid_t file, const char* name, hid_t mem_type, T* out)
{
    hid_t a = H5Aopen(file, name, H5P_DEFAULT);
    if (a < 0) die(std::string("missing attribute ") + name);
    if (H5Aread(a, mem_type, out) < 0) die(std::string("H5Aread ") + name);
    H5Aclose(a);
}

Meta read_meta(hid_t file)
{
    Meta m;
    read_attr(file, "schema_version", H5T_NATIVE_INT, &m.schema);
    read_attr(file, "step", H5T_NATIVE_INT, &m.step);
    read_attr(file, "ng", H5T_NATIVE_INT, &m.ng);
    read_attr(file, "np", H5T_NATIVE_INT64, &m.np);
    read_attr(file, "a", H5T_NATIVE_DOUBLE, &m.a);
    read_attr(file, "rL", H5T_NATIVE_DOUBLE, &m.rL);
    read_attr(file, "z_in", H5T_NATIVE_DOUBLE, &m.z_in);
    read_attr(file, "omega_m", H5T_NATIVE_DOUBLE, &m.omega_m);
    return m;
}

// Visit a dataset of shape (N) or (N, ncomp) in chunks of `chunk` rows.
template <typename T, typename F>
hsize_t for_each_chunk(hid_t file, const char* path, hid_t mem_type, int ncomp,
                       hsize_t chunk, F&& fn)
{
    hid_t ds = H5Dopen2(file, path, H5P_DEFAULT);
    if (ds < 0) die(std::string("missing dataset ") + path);
    hid_t fs = H5Dget_space(ds);
    int rank = H5Sget_simple_extent_ndims(fs);
    hsize_t dims[2] = {0, 1};
    H5Sget_simple_extent_dims(fs, dims, nullptr);
    if (rank != (ncomp > 1 ? 2 : 1) || (ncomp > 1 && dims[1] != (hsize_t) ncomp))
        die(std::string("unexpected shape for ") + path);
    const hsize_t n = dims[0];
    std::vector<T> buf((size_t) chunk * ncomp);
    for (hsize_t start = 0; start < n; start += chunk) {
        const hsize_t cnt = std::min(chunk, n - start);
        hsize_t off[2] = {start, 0}, len[2] = {cnt, (hsize_t) ncomp};
        H5Sselect_hyperslab(fs, H5S_SELECT_SET, off, nullptr, len, nullptr);
        hid_t ms = H5Screate_simple(rank, len, nullptr);
        if (H5Dread(ds, mem_type, ms, fs, H5P_DEFAULT, buf.data()) < 0)
            die(std::string("H5Dread ") + path);
        H5Sclose(ms);
        fn(buf.data(), (size_t) cnt);
    }
    H5Sclose(fs);
    H5Dclose(ds);
    return n;
}

struct ParticleStats {
    hsize_t n_pos = 0, n_vel = 0, n_id = 0, n_mass = 0;
    int64_t n_out_of_box = 0, n_nonfinite = 0;
    double psum[3] = {0, 0, 0}, v2sum = 0;
    float mass_min = 0, mass_max = 0;
    bool ids_ok = true;
    double delta_rms = 0;       // rms density contrast of the coarse CIC density
    int mcoarse = 0;
};

ParticleStats scan(hid_t file, const Meta& m, bool check_ids)
{
    ParticleStats s;
    const hsize_t chunk = 1 << 22;
    // Coarse cell mesh: 8 Mpc/h cells at the 1 Mpc/h upstream spacing, i.e.
    // ng/8 per side (at least 4).
    s.mcoarse = std::max(4, m.ng / 8);
    const int M = s.mcoarse;
    std::vector<double> cells((size_t) M * M * M, 0.0);
    const double inv_cell = M / m.rL;
    const float rL = (float) m.rL;

    s.n_pos = for_each_chunk<float>(file, "/particles/position", H5T_NATIVE_FLOAT, 3, chunk,
        [&](const float* p, size_t cnt) {
            for (size_t i = 0; i < cnt; ++i) {
                int i0[3];
                double w1[3];
                bool ok = true;
                for (int d = 0; d < 3; ++d) {
                    const float x = p[3 * i + d];
                    if (!std::isfinite(x)) { ++s.n_nonfinite; ok = false; break; }
                    if (!(x >= 0.0f && x < rL)) { ++s.n_out_of_box; ok = false; break; }
                    // CIC: cell centres at (j + 0.5) * cell; periodic wrap.
                    const double u = x * inv_cell - 0.5;
                    const double f = std::floor(u);
                    w1[d] = u - f;
                    i0[d] = ((int) f % M + M) % M;
                }
                if (!ok) continue;
                for (int c = 0; c < 8; ++c) {
                    const int dx = c & 1, dy = (c >> 1) & 1, dz = (c >> 2) & 1;
                    const int ix = (i0[0] + dx) % M, iy = (i0[1] + dy) % M, iz = (i0[2] + dz) % M;
                    const double w = (dx ? w1[0] : 1.0 - w1[0]) * (dy ? w1[1] : 1.0 - w1[1]) *
                                     (dz ? w1[2] : 1.0 - w1[2]);
                    cells[(size_t) (ix * M + iy) * M + iz] += w;
                }
            }
        });
    s.n_vel = for_each_chunk<float>(file, "/particles/velocity", H5T_NATIVE_FLOAT, 3, chunk,
        [&](const float* v, size_t cnt) {
            for (size_t i = 0; i < cnt; ++i)
                for (int d = 0; d < 3; ++d) {
                    const double x = v[3 * i + d];
                    if (!std::isfinite(x)) { ++s.n_nonfinite; continue; }
                    s.psum[d] += x;
                    s.v2sum += x * x;
                }
        });
    bool first = true;
    s.n_mass = for_each_chunk<float>(file, "/particles/mass", H5T_NATIVE_FLOAT, 1, chunk,
        [&](const float* w, size_t cnt) {
            for (size_t i = 0; i < cnt; ++i) {
                if (!std::isfinite(w[i])) { ++s.n_nonfinite; continue; }
                if (first) { s.mass_min = s.mass_max = w[i]; first = false; }
                s.mass_min = std::min(s.mass_min, w[i]);
                s.mass_max = std::max(s.mass_max, w[i]);
            }
        });
    if (check_ids) {
        std::vector<uint8_t> seen((size_t) m.np, 0);
        s.n_id = for_each_chunk<int64_t>(file, "/particles/id", H5T_NATIVE_INT64, 1, chunk,
            [&](const int64_t* id, size_t cnt) {
                for (size_t i = 0; i < cnt; ++i) {
                    if (id[i] < 0 || id[i] >= m.np || seen[(size_t) id[i]]) { s.ids_ok = false; continue; }
                    seen[(size_t) id[i]] = 1;
                }
            });
        if (s.n_id != (hsize_t) m.np) s.ids_ok = false;   // then some id is missing
    }
    // rms density contrast of the coarse CIC density
    double mean = 0;
    for (auto c : cells) mean += c;
    mean /= (double) cells.size();
    double var = 0;
    for (auto c : cells) { const double d = c / mean - 1.0; var += d * d; }
    s.delta_rms = std::sqrt(var / (double) cells.size());
    return s;
}

// Linear growth factor for flat LambdaCDM (matter + Lambda), unnormalised:
// D(a) = (5/2) Omega_m E(a) * int_0^a da' / (a' E(a'))^3   (Heath 1977).
double growth_D(double a, double om)
{
    const double ol = 1.0 - om;
    auto E = [&](double x) { return std::sqrt(om / (x * x * x) + ol); };
    const int n = 20000;
    double sum = 0;
    const double h = a / n;
    for (int i = 0; i < n; ++i) {            // midpoint rule; integrand ~ a^1.5 near 0
        const double x = (i + 0.5) * h;
        const double aE = x * E(x);
        sum += 1.0 / (aE * aE * aE);
    }
    return 2.5 * om * E(a) * sum * h;
}

double arg_d(int argc, char** argv, int& i, const char* name)
{
    if (i + 1 >= argc) die(std::string("missing value for ") + name);
    return std::atof(argv[++i]);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::fprintf(stderr, "usage: snapshot_check <ic.h5> <evolved.h5> --nsteps N --zfin Z "
                             "[--pmom-tol T] [--growth-lo L] [--growth-hi H]\n");
        return 2;
    }
    const char* ic_path = argv[1];
    const char* ev_path = argv[2];
    int nsteps = -1;
    double zfin = -1, pmom_tol = 1e-6, growth_lo = 0.5, growth_hi = 2.0;
    for (int i = 3; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--nsteps")) nsteps = (int) arg_d(argc, argv, i, "--nsteps");
        else if (!std::strcmp(argv[i], "--zfin")) zfin = arg_d(argc, argv, i, "--zfin");
        else if (!std::strcmp(argv[i], "--pmom-tol")) pmom_tol = arg_d(argc, argv, i, "--pmom-tol");
        else if (!std::strcmp(argv[i], "--growth-lo")) growth_lo = arg_d(argc, argv, i, "--growth-lo");
        else if (!std::strcmp(argv[i], "--growth-hi")) growth_hi = arg_d(argc, argv, i, "--growth-hi");
        else die(std::string("unknown option ") + argv[i]);
    }
    if (nsteps < 0 || zfin < 0) die("--nsteps and --zfin are required");

    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    hid_t fic = H5Fopen(ic_path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fic < 0) die(std::string("cannot open ") + ic_path);
    hid_t fev = H5Fopen(ev_path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fev < 0) die(std::string("cannot open ") + ev_path);

    const Meta mi = read_meta(fic);
    const Meta me = read_meta(fev);
    std::printf("snapshot_check: ic       %s\n", ic_path);
    std::printf("snapshot_check: evolved  %s\n", ev_path);
    std::printf("snapshot_check: ng=%d np=%lld rL=%g omega_m=%g z_in=%g a_in=%.8g a_out=%.8g step=%d\n",
                me.ng, (long long) me.np, me.rL, me.omega_m, me.z_in, mi.a, me.a, me.step);

    int nfail = 0;
    auto check = [&](bool ok, const char* what, const std::string& detail) {
        std::printf("  [%s] %-22s %s\n", ok ? "ok" : "FAIL", what, detail.c_str());
        if (!ok) ++nfail;
    };

    const ParticleStats si = scan(fic, mi, false);
    const ParticleStats se = scan(fev, me, true);
    H5Fclose(fic);
    H5Fclose(fev);

    char buf[512];

    // 1. particle count
    const bool count_ok = se.n_pos == (hsize_t) me.np && se.n_vel == (hsize_t) me.np &&
                          se.n_mass == (hsize_t) me.np && me.np == mi.np && si.n_pos == (hsize_t) mi.np;
    std::snprintf(buf, sizeof buf, "N_ic=%llu N_evolved=%llu (attr np=%lld)",
                  (unsigned long long) si.n_pos, (unsigned long long) se.n_pos, (long long) me.np);
    check(count_ok, "particle count", buf);

    // 2. step / scale factor
    const double a_expect = 1.0 / (1.0 + zfin);
    std::snprintf(buf, sizeof buf, "step=%d (expect %d)", me.step, nsteps);
    check(me.step == nsteps, "step count", buf);
    std::snprintf(buf, sizeof buf, "a_out=%.8g expect 1/(1+%g)=%.8g, a_in=%.8g", me.a, zfin, a_expect, mi.a);
    check(std::fabs(me.a - a_expect) <= 1e-6 * a_expect && me.a > mi.a, "scale factor", buf);

    // 3./4. finiteness, box, masses
    std::snprintf(buf, sizeof buf, "out of [0,rL): %lld, non-finite: %lld",
                  (long long) se.n_out_of_box, (long long) se.n_nonfinite);
    check(se.n_out_of_box == 0 && se.n_nonfinite == 0, "positions in box", buf);
    std::snprintf(buf, sizeof buf, "min=%g max=%g", se.mass_min, se.mass_max);
    check(se.mass_min > 0 && se.mass_min == se.mass_max, "masses equal, > 0", buf);

    // 5. ids
    check(se.ids_ok, "ids permutation", se.ids_ok ? "each id in [0,np) exactly once" : "duplicate/missing/out-of-range ids");

    // 6. momentum
    const double vrms_e = std::sqrt(se.v2sum / (3.0 * (double) me.np));
    const double vrms_i = std::sqrt(si.v2sum / (3.0 * (double) mi.np));
    double pmax = 0, pmax_ic = 0;
    for (int d = 0; d < 3; ++d) {
        pmax = std::max(pmax, std::fabs(se.psum[d]) / ((double) me.np * vrms_e));
        pmax_ic = std::max(pmax_ic, std::fabs(si.psum[d]) / ((double) mi.np * vrms_i));
    }
    std::snprintf(buf, sizeof buf, "max_d |sum v_d|/(N v_rms) = %.3e (ic %.3e), v_rms=%.4g km/s (ic %.4g), tol %g",
                  pmax, pmax_ic, vrms_e, vrms_i, pmom_tol);
    check(pmax <= pmom_tol, "total momentum ~ 0", buf);

    // 7. growth of the coarse density field
    const double D_ratio = growth_D(me.a, me.omega_m) / growth_D(mi.a, mi.omega_m);
    const double ratio = se.delta_rms / si.delta_rms;
    std::snprintf(buf, sizeof buf,
                  "delta_rms(%d^3 cells) %.4g -> %.4g, ratio %.3g; linear D(a_out)/D(a_in)=%.3g, ratio/linear=%.3f in [%g,%g]",
                  se.mcoarse, si.delta_rms, se.delta_rms, ratio, D_ratio, ratio / D_ratio, growth_lo, growth_hi);
    check(ratio / D_ratio >= growth_lo && ratio / D_ratio <= growth_hi, "density growth", buf);

    if (nfail == 0) { std::printf("snapshot_check: PASS\n"); return 0; }
    std::printf("snapshot_check: FAIL (%d check(s) failed)\n", nfail);
    return 1;
}
