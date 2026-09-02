#pragma once
// src/ic/PowerSpectrum.hpp
// CDM+baryon power spectrum  P_cb(k) = A · k^n_s · T_cb²(k),  σ₈-normalized.
//
// HACC reference:
//   - Pk_cb formula:        InitCosmology.cxx:349-351
//   - sigma2 integrand:     InitCosmology.cxx:365-368
//   - SetSigma8:            InitCosmology.cxx:385-391
//   - midpoint integrate:   InitCosmology.cxx:573-628
//
// Spec: ../analysis/prompts/10b_implement_zeldovich_ic.md (Deliverable 2)
//
// Differences from HACC:
//   - HACC uses linear-k midpoint integration over the full file range.
//   - We integrate in log(k) with Simpson's rule and a convergence check
//     (doubling points changes σ² by <0.1%).  The integrand σ² ∝ P(k)·k³ in
//     log k is sharply peaked near k ≈ 1/R = 0.125 h/Mpc for R = 8 Mpc/h, and
//     the linear-k rule needs ~10× more points to reach the same accuracy.
//   - The sigma2 integrand uses Pk_cb (CDM+baryon-only), not Pk_total — for
//     a no-neutrino cosmology these coincide exactly, and sigma8 normalization
//     of P_cb against itself is the simplest self-consistent setup.

#include "TransferFunction.hpp"

namespace pmk {

class PowerSpectrum {
public:
    // Construct + normalize.  Throws std::runtime_error if convergence fails
    // at the maximum point count (10000 log-spaced points).
    PowerSpectrum(const TransferFunction& tf, double n_s, double sigma8_target);

    // P_cb(k) at a=1 with the σ₈ normalization applied.  k in h/Mpc, output
    // in (Mpc/h)³.
    double P_cb(double k) const;

    // Recompute σ₈ from the current P(k); test hook to verify the
    // normalization integration did its job.
    double compute_sigma8() const;

    // Accessors used by the IC kernel and the recovery test.
    double n_s()           const { return n_s_; }
    double sigma8_target() const { return sigma8_target_; }
    double normalization() const { return normalization_; }

    // Forwarded transfer-function range — IC kernel needs k_min / k_max so it
    // can decide what to do for modes outside the table (the table goes from
    // k ~ 3e-6 to k ~ 50 h/Mpc, well outside the simulation's k range).
    double k_min() const { return tf_.k_min(); }
    double k_max() const { return tf_.k_max(); }

private:
    const TransferFunction& tf_;
    double n_s_;
    double sigma8_target_;
    double normalization_ = 1.0;   // A in P = A · k^n_s · T²(k)

    // Spherical top-hat in Fourier space: W(x) = 3(sin x − x cos x)/x³.
    // x → 0 limit handled with a Taylor series (W → 1).
    static double top_hat_window(double kR);

    // σ²(R = 8 Mpc/h) integrated in log(k) with `npts` log-spaced Simpson nodes.
    double sigma2_integral(int npts) const;

    // Find a point count where doubling changes σ² by <0.1%.  Throws if 10000
    // is not enough.
    int converge_npts() const;
};

} // namespace pmk
