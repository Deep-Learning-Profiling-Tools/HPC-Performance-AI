// src/Cosmology.cpp
// Host-side cosmology: E(a), adot(a), and growth factors via adaptive Cash-Karp.
//
// HACC reference:
//   - TimeStepper.cxx:78-91 (omega_nu_massive, H_ratio)
//   - InitCosmology.cxx:70-113   (GrowthFactor, CCP::Omega_nu_massive, growths)
//   - InitCosmology.cxx:403-559  (odesolve, rkqs, rkck — Numerical Recipes)
//
// Spec: HACC_PM_IC_PORTING_SPEC.md §2 (Growth Factors), §3 (E(a))

#include "Cosmology.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pmk {

namespace {
// Numerical Recipes adaptive RK constants (InitCosmology.cxx:457-461)
constexpr double SAFETY = 0.9;
constexpr double PGROW  = -0.2;
constexpr double PSHRNK = -0.25;
constexpr double ERRCON = 1.89e-4;
constexpr double TINY   = 1.0e-30;

// Cash-Karp coefficients (InitCosmology.cxx:515-523)
constexpr double a2 = 0.2,        a3 = 0.3,        a4 = 0.6,        a5 = 1.0,        a6 = 0.875;
constexpr double b21 = 0.2;
constexpr double b31 = 3.0/40.0,  b32 = 9.0/40.0;
constexpr double b41 = 0.3,       b42 = -0.9,      b43 = 1.2;
constexpr double b51 = -11.0/54.0, b52 = 2.5,      b53 = -70.0/27.0, b54 = 35.0/27.0;
constexpr double b61 = 1631.0/55296.0,  b62 = 175.0/512.0,
                 b63 = 575.0/13824.0,   b64 = 44275.0/110592.0, b65 = 253.0/4096.0;
constexpr double c1 = 37.0/378.0, c3 = 250.0/621.0, c4 = 125.0/594.0, c6 = 512.0/1771.0;
constexpr double dc5 = -277.0/14336.0;
constexpr double dc1 = c1 - 2825.0/27648.0;
constexpr double dc3 = c3 - 18575.0/48384.0;
constexpr double dc4 = c4 - 13525.0/55296.0;
constexpr double dc6 = c6 - 0.25;

// High-redshift integration start for the growth ODE (InitCosmology.cxx:72).
constexpr double Z_INFINITY = 100000.0;

// Maximum number of steps in odesolve (InitCosmology.cxx:400)
constexpr int MAX_ODE_STEPS = 100000;
} // namespace

Cosmology::Cosmology(const CosmoParams& cp)
    : m_cp(cp)
{
    // Match HACC convention: omega_cb is CDM+baryon and is the source term in
    // the growth ODE (InitCosmology.cxx:106).  We do not enforce flatness here;
    // the dark-energy density is computed as 1 − Ω_m − (1+f_nu_ml)·Ω_r in E(a).
    //
    // Default-promotion: a CosmoParams left with the sentinel omega_cb==0.0
    // gets it set to omega_matter so callers in the no-neutrinos case need
    // only specify omega_matter.  An explicit nonzero omega_cb passes through.
    if (m_cp.omega_cb == 0.0)
        m_cp.omega_cb = m_cp.omega_matter;
}

double Cosmology::omega_nu_massive(double a) const
{
    // HACC: TimeStepper.cxx:78-82 — branch-free max() of matter-like and rad-like.
    const double mat = m_cp.omega_nu / (a * a * a);
    const double rad = m_cp.f_nu_massive * m_cp.omega_radiation / (a * a * a * a);
    return (mat >= rad) ? mat : rad;
}

double Cosmology::E(double a) const
{
    // HACC TimeStepper.cxx:84-91 — flat universe assumed (no curvature term).
    const double a2_ = a * a;
    const double a3_ = a2_ * a;
    const double a4_ = a2_ * a2_;
    const double de_exponent = -3.0 * (1.0 + m_cp.w + m_cp.wa);
    const double de_exp_corr = -3.0 * m_cp.wa * (1.0 - a);
    const double omega_de_today =
        1.0 - m_cp.omega_matter - (1.0 + m_cp.f_nu_massless) * m_cp.omega_radiation;
    return std::sqrt(
          m_cp.omega_cb / a3_
        + (1.0 + m_cp.f_nu_massless) * m_cp.omega_radiation / a4_
        + omega_nu_massive(a)
        + omega_de_today * std::pow(a, de_exponent) * std::exp(de_exp_corr)
    );
}

double Cosmology::omega_matter_a(double a) const
{
    const double Ea = E(a);
    return m_cp.omega_cb / (a * a * a * Ea * Ea);
}

// ---------------------------------------------------------------------------
// Growth ODE
// ---------------------------------------------------------------------------

void Cosmology::growth_rhs(double a, const double y[2], double dydx[2]) const
{
    // HACC InitCosmology.cxx:104-113.
    //   y[0] = D(a),  y[1] = D'(a) · a · H(a)
    //   dD/da     = y[1] / (a · H)
    //   d(y[1])/da = -2·y[1]/a + 1.5·omega_cb·y[0] / (H · a^4)
    const double H = E(a);  // H(a)/H0
    const double a4 = a * a * a * a;
    dydx[0] = y[1] / (a * H);
    dydx[1] = -2.0 * y[1] / a + 1.5 * m_cp.omega_cb * y[0] / (H * a4);
}

// ---------------------------------------------------------------------------
// Cash-Karp Runge-Kutta step (Numerical Recipes / HACC InitCosmology.cxx:510)
// ---------------------------------------------------------------------------

void Cosmology::rkck(const double y[2], const double dydx[2], double x, double h,
                     double yout[2], double yerr[2]) const
{
    double ytemp[2], ak2[2], ak3[2], ak4[2], ak5[2], ak6[2];

    for (int i = 0; i < 2; ++i)
        ytemp[i] = y[i] + b21 * h * dydx[i];
    growth_rhs(x + a2 * h, ytemp, ak2);

    for (int i = 0; i < 2; ++i)
        ytemp[i] = y[i] + h * (b31 * dydx[i] + b32 * ak2[i]);
    growth_rhs(x + a3 * h, ytemp, ak3);

    for (int i = 0; i < 2; ++i)
        ytemp[i] = y[i] + h * (b41 * dydx[i] + b42 * ak2[i] + b43 * ak3[i]);
    growth_rhs(x + a4 * h, ytemp, ak4);

    for (int i = 0; i < 2; ++i)
        ytemp[i] = y[i] + h * (b51 * dydx[i] + b52 * ak2[i] + b53 * ak3[i] + b54 * ak4[i]);
    growth_rhs(x + a5 * h, ytemp, ak5);

    for (int i = 0; i < 2; ++i)
        ytemp[i] = y[i] + h * (b61 * dydx[i] + b62 * ak2[i] + b63 * ak3[i]
                                + b64 * ak4[i] + b65 * ak5[i]);
    growth_rhs(x + a6 * h, ytemp, ak6);

    for (int i = 0; i < 2; ++i)
        yout[i] = y[i] + h * (c1 * dydx[i] + c3 * ak3[i] + c4 * ak4[i] + c6 * ak6[i]);

    for (int i = 0; i < 2; ++i)
        yerr[i] = h * (dc1 * dydx[i] + dc3 * ak3[i] + dc4 * ak4[i]
                       + dc5 * ak5[i] + dc6 * ak6[i]);
}

// ---------------------------------------------------------------------------
// Quality-controlled stepper (HACC InitCosmology.cxx:464)
// ---------------------------------------------------------------------------

void Cosmology::rkqs(double y[2], double dydx[2], double& x, double htry,
                     double eps, const double yscal[2],
                     double& hdid, double& hnext) const
{
    // HACC InitCosmology.cxx:464-497 (Numerical Recipes rkqs).
    double h = htry;
    double yerr[2], ytemp[2];
    double errmax = 0.0;

    for (;;) {
        rkck(y, dydx, x, h, ytemp, yerr);
        errmax = 0.0;
        for (int i = 0; i < 2; ++i)
            errmax = std::max(errmax, std::fabs(yerr[i] / yscal[i]));
        errmax /= eps;
        if (errmax <= 1.0) break;

        const double htemp = SAFETY * h * std::pow(errmax, PSHRNK);
        // Shrink, but not by more than a factor of 10.
        h = (h >= 0.0) ? std::max(htemp, 0.1 * h) : std::min(htemp, 0.1 * h);
        const double xnew = x + h;
        if (xnew == x)
            throw std::runtime_error("Cosmology::rkqs: stepsize underflow");
    }

    if (errmax > ERRCON) hnext = SAFETY * h * std::pow(errmax, PGROW);
    else                 hnext = 5.0 * h;

    hdid = h;
    x   += h;
    for (int i = 0; i < 2; ++i) y[i] = ytemp[i];
}

// ---------------------------------------------------------------------------
// odesolve driver  (HACC InitCosmology.cxx:403)
// ---------------------------------------------------------------------------

void Cosmology::odesolve(double y[2], double x1, double x2, double eps) const
{
    constexpr double H1 = 1.0e-6;       // initial trial step size (HACC default)
    const double dir = (x2 >= x1) ? 1.0 : -1.0;
    double h = dir * std::fabs(H1);
    double x = x1;

    double dydx[2], yscal[2];
    double hdid = 0.0, hnext = 0.0;

    for (int nstp = 0; nstp < MAX_ODE_STEPS; ++nstp) {
        growth_rhs(x, y, dydx);
        for (int i = 0; i < 2; ++i)
            yscal[i] = std::fabs(y[i]) + std::fabs(dydx[i] * h) + TINY;

        // Trim final step so we don't overshoot x2.
        if ((x + h - x2) * (x + h - x1) > 0.0) h = x2 - x;

        rkqs(y, dydx, x, h, eps, yscal, hdid, hnext);

        if ((x - x2) * (x2 - x1) >= 0.0) return;  // arrived

        if (std::fabs(hnext) <= 0.0)
            throw std::runtime_error("Cosmology::odesolve: step too small");

        h = hnext;
    }
    throw std::runtime_error("Cosmology::odesolve: too many steps");
}

// ---------------------------------------------------------------------------
// Normalization: solve from a_inf to a=1 once, cache the result.
// ---------------------------------------------------------------------------

void Cosmology::ensure_normalization() const
{
    if (m_norm_set) return;
    const double a_inf = 1.0 / (1.0 + Z_INFINITY);
    double y[2] = {a_inf, 0.0};   // matter-era growing-mode IC: D ∝ a, D'·a·H ≈ 0
    odesolve(y, a_inf, 1.0, 1.0e-6);
    m_D_at_1    = y[0];
    m_Ddot_at_1 = y[1];
    m_norm_set  = true;
}

// ---------------------------------------------------------------------------
// growth_factors(a)
// ---------------------------------------------------------------------------

GrowthFactors Cosmology::growth_factors(double a) const
{
    ensure_normalization();

    GrowthFactors gf{};
    if (a >= 1.0 - 1.0e-12 && a <= 1.0 + 1.0e-12) {
        // a = 1 exactly: D1 = 1 by normalization.
        gf.D1    = 1.0;
        gf.D1dot = m_Ddot_at_1 / m_D_at_1;
    } else {
        const double a_inf = 1.0 / (1.0 + Z_INFINITY);
        double y[2] = {a_inf, 0.0};
        odesolve(y, a_inf, a, 1.0e-6);
        gf.D1    = y[0] / m_D_at_1;        // HACC InitCosmology.cxx:93
        gf.D1dot = y[1] / m_D_at_1;        // HACC InitCosmology.cxx:94 (= D'·a·H)
    }

    // f1 = dlnD1/dlna = (dD1/da · a) / D1.
    // We have D1dot = dD1/da · a · H(a)/H0  (the HACC convention),
    // hence dlnD1/dlna = D1dot / (D1 · H(a)/H0) = D1dot / (D1 · E(a)).
    const double Ea = E(a);
    gf.f1 = (gf.D1 != 0.0) ? gf.D1dot / (gf.D1 * Ea) : 0.0;

    // ---- 2LPT growth (Scoccimarro 1998; Bouchet+ 1995 fit) ---------------
    // HACC has no 2LPT; we provide it for downstream IC use.
    //   D2(a) ≈ -3/7 · D1(a)^2 · Ω_m(a)^{-1/143}                  (Bouchet+ 1995)
    //   f2(a) ≈ 2 · Ω_m(a)^{6/11}                                 (Scoccimarro 1998 fit)
    //   D2dot = D2 · f2 · E(a)                                    (so dD2/da · a · H = f2·D2·H)
    const double om_a = omega_matter_a(a);
    gf.D2    = -3.0 / 7.0 * gf.D1 * gf.D1 * std::pow(om_a, -1.0 / 143.0);
    gf.f2    = 2.0 * std::pow(om_a, 6.0 / 11.0);
    gf.D2dot = gf.D2 * gf.f2 * Ea;

    return gf;
}

void Cosmology::growth_factor(double a, double& gf_out, double& gf_dot_out) const
{
    const GrowthFactors gf = growth_factors(a);
    gf_out     = gf.D1;
    // gf.D1dot = dD1/da · a · H(a)/H0   (the HACC convention; see header)
    // dD1/dt  = dD1/da · da/dt = dD1/da · a · H(a) = gf.D1dot in H0=1 units.
    gf_dot_out = gf.D1dot;
}

} // namespace pmk
