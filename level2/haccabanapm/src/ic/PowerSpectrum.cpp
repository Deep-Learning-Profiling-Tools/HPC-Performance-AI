// src/ic/PowerSpectrum.cpp
// σ₈-normalized P_cb(k) via composite Simpson's rule in log(k).
//
// Source mapping:
//   P_cb formula      ↔ HACC InitCosmology.cxx:349-351
//   sigma2 integrand  ↔ HACC InitCosmology.cxx:365-368
//   normalization     ↔ HACC InitCosmology.cxx:385-391
//
// Algorithm:
//   σ²(R) = (1/2π²) ∫ P(k) W²(kR) k² dk
//         = (1/2π²) ∫ P(k(u)) W²(k(u) R) k(u)³ du,   u = log(k)
//   The change-of-variable absorbs dk = k du; the integrand peaks near u =
//   log(1/R) ≈ −2.08 for R = 8 Mpc/h with FWHM ~3 (decades).  A few thousand
//   log-spaced Simpson nodes give 4-decimal convergence.

#include "PowerSpectrum.hpp"

#include <cmath>
#include <stdexcept>

namespace pmk {

namespace {
constexpr double TWO_PI_SQ = 2.0 * M_PI * M_PI;
} // namespace

PowerSpectrum::PowerSpectrum(const TransferFunction& tf,
                             double n_s, double sigma8_target)
    : tf_(tf), n_s_(n_s), sigma8_target_(sigma8_target)
{
    // Normalize against itself: at A = 1, integrate σ²; rescale A so that
    // sigma8(A·P) = sqrt(A·σ²) == sigma8_target.
    normalization_ = 1.0;
    const int npts = converge_npts();
    const double s2_at_A1 = sigma2_integral(npts);
    if (!(s2_at_A1 > 0.0)) {
        throw std::runtime_error(
            "PowerSpectrum: sigma2(R=8) integration produced non-positive value");
    }
    normalization_ = (sigma8_target_ * sigma8_target_) / s2_at_A1;
}

double PowerSpectrum::P_cb(double k) const
{
    // Saturated tails on T_cb already handled inside the TransferFunction;
    // here we just guard k > 0.
    if (k <= 0.0) return 0.0;
    const double t = tf_.T_cb(k);
    return normalization_ * std::pow(k, n_s_) * t * t;
}

double PowerSpectrum::compute_sigma8() const
{
    const int npts = converge_npts();
    return std::sqrt(sigma2_integral(npts));
}

double PowerSpectrum::top_hat_window(double kR)
{
    // 3 (sin x − x cos x) / x³.  For very small x use the Taylor expansion to
    // avoid catastrophic cancellation: W = 1 − x²/10 + x⁴/280 − …
    const double x = kR;
    if (std::fabs(x) < 1.0e-3) {
        const double x2 = x * x;
        return 1.0 - x2 / 10.0 + (x2 * x2) / 280.0;
    }
    return 3.0 * (std::sin(x) - x * std::cos(x)) / (x * x * x);
}

double PowerSpectrum::sigma2_integral(int npts) const
{
    // Integration range — per the prompt, clamp the file's [k_min, k_max] to
    // a sane band so we don't waste samples on regions where the integrand is
    // numerically zero.
    constexpr double R = 8.0;  // Mpc/h
    const double k_lo = std::max(tf_.k_min(), 1.0e-5);
    const double k_hi = std::min(tf_.k_max(), 1.0e3);
    const double u_lo = std::log(k_lo);
    const double u_hi = std::log(k_hi);

    // Composite Simpson's rule needs an even number of intervals.
    int n = npts;
    if (n < 2) n = 2;
    if (n % 2 == 1) ++n;
    const double h = (u_hi - u_lo) / static_cast<double>(n);

    auto integrand = [&](double u) {
        const double k = std::exp(u);
        const double Wk = top_hat_window(k * R);
        return P_cb(k) * Wk * Wk * k * k * k;  // P · W² · k³ · du
    };

    double sum = integrand(u_lo) + integrand(u_hi);
    for (int i = 1; i < n; ++i) {
        const double u = u_lo + i * h;
        const double w = (i % 2 == 0) ? 2.0 : 4.0;
        sum += w * integrand(u);
    }
    return (h / 3.0) * sum / TWO_PI_SQ;
}

int PowerSpectrum::converge_npts() const
{
    // Doubling-trick convergence check.  Start at 2000 (the prompt's default);
    // accept when the relative change drops below 1e-3, give up at 10000.
    constexpr int N_MAX = 10000;
    int n = 2000;
    double s_prev = sigma2_integral(n);
    while (n < N_MAX) {
        const int n2 = std::min(n * 2, N_MAX);
        const double s_next = sigma2_integral(n2);
        const double rel = (s_prev > 0.0) ? std::fabs(s_next - s_prev) / s_prev
                                          : std::fabs(s_next - s_prev);
        if (rel < 1.0e-3) return n2;
        if (n2 == N_MAX) {
            throw std::runtime_error(
                "PowerSpectrum: σ² integral did not converge to 0.1% "
                "even at 10000 log-spaced Simpson nodes");
        }
        n      = n2;
        s_prev = s_next;
    }
    return n;
}

} // namespace pmk
