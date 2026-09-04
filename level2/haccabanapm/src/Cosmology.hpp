#pragma once
// src/Cosmology.hpp
// Cosmological functions: E(a) = H(a)/H0, growth factors D1/D2 and rates f1/f2.
//
// HACC reference:
//   - H(a)/H0:                  TimeStepper.cxx:84-91   (TimeStepper::H_ratio)
//   - omega_nu_massive(a):      TimeStepper.cxx:78-82
//   - adot(a) = a·H(a)/H0:      TimeStepper.cxx:93-103  (set_adot)
//   - Growth ODE (RHS):         InitCosmology.cxx:104-113  (CosmoClass::growths)
//   - Adaptive Cash-Karp RK:    InitCosmology.cxx:403-559  (odesolve, rkqs, rkck)
//   - Normalization D(z=0)=1:   InitCosmology.cxx:93
//
// Spec:  HACC_PM_IC_PORTING_SPEC.md §2 (Growth Factors), §3 (E(a))
//        PORTING_GAP_ANALYSIS.md §2 Gap 1 (no 2LPT in HACC — implement here)
//
// Design (vs HACC):
//   - Pure host-side, double precision (these are setup-time computations).
//     No Kokkos kernels, no device callability — keeps this module dependency-free
//     beyond <cmath>.
//   - 2LPT growth (D2/f2) uses the Scoccimarro 1998 / Bouchet+ 1995 fit.
//     HACC has no 2LPT; we provide it so the IC module can use it later.
//
// Sign / normalization conventions (matching HACC):
//   - E(a)  ≡ H(a)/H0   so E(a=1) == 1 for a flat ΛCDM with Ω_m+Ω_Λ=1.
//   - adot ≡ a·E(a)     in units of H0^{-1}  (TimeStepper comment line 93).
//   - D1   normalized so D1(a=1) == 1 exactly.
//   - D2   sign convention follows Scoccimarro: D2 < 0  (factor −3/7).
//   - f1, f2 = d ln D_i / d ln a  ∈ [0, 1]  for all reasonable cosmologies.

#include <cmath>

namespace pmk {

// All cosmological parameters — mirrors HACC TimeStepper.h state and CosmoClassParams.
struct CosmoParams {
    double omega_matter     = 0.3;     // total matter (CDM + bar + nu)
    double omega_cdm        = 0.2516;
    double omega_baryon     = 0.0484;
    // Sentinel 0.0 means "use omega_matter" — the Cosmology constructor
    // copies omega_matter into omega_cb in that case, so for the no-neutrinos
    // case the user only needs to set omega_matter.  Future neutrino extensions
    // can set omega_cb (CDM+baryon) explicitly without breaking the API.
    double omega_cb         = 0.0;     // CDM + baryon  (0 ⇒ default to omega_matter)
    double omega_nu         = 0.0;     // massive neutrinos
    // omega_radiation: photon + massless-neutrino density today.
    // For HACC parity, this MUST be computed from T_CMB and h per
    // HACC's Basedata.h:237 formula:
    //     omega_radiation = 2.471e-5 * (T_CMB/2.725)^4 / h^2
    // Leaving this at 0 causes H(a) at high z to underestimate
    // expansion, producing a ~2% growth-factor error and ~10%
    // velocity error at z=200 in the IC.  See
    // ../analysis/references/velocity_discrepancy_diagnosis.md.
    double omega_radiation  = 0.0;
    double f_nu_massless    = 0.0;     // fraction of radiation in massless neutrinos
    double f_nu_massive     = 0.0;     // (used inside omega_nu_massive transition)
    double w                = -1.0;    // CPL dark energy w0
    double wa               = 0.0;     // CPL dark energy wa
    double hubble           = 0.7;     // h
    double ns               = 0.96;
    double sigma8           = 0.8;
};

// Growth factor results from ODE integration (host-side, called once per a).
struct GrowthFactors {
    double D1;     // linear growth, normalized D1(a=1) = 1.
    double D1dot;  // dD1/da · a · H(a)/H0  (HACC's ddot convention)
    double D2;     // 2LPT growth (Scoccimarro 1998 fit; sign: negative).
    double D2dot;  // dD2/da · a · H(a)/H0
    double f1;     // dln D1 / dln a  ∈ [0, 1]
    double f2;     // dln D2 / dln a
};

class Cosmology {
public:
    explicit Cosmology(const CosmoParams& cp);

    const CosmoParams& params() const { return m_cp; }

    // omega_nu_massive(a) — branch-free max() of matter-like and radiation-like
    // contributions.  Source: HACC TimeStepper.cxx:78-82.
    double omega_nu_massive(double a) const;

    // E(a) = H(a)/H0.  Source: HACC TimeStepper.cxx:84-91.
    double E(double a) const;

    // adot = a · E(a)   (HACC TimeStepper.cxx:93-103, units of H0^{-1}).
    // In code units H0=1 so this is equivalent to ȧ at scale factor a.
    // Used by the time integrator (set_adot in src/time/Integrator.cpp).
    double adot(double a) const { return a * E(a); }

    // omega_cb = Ω_cdm + Ω_baryon — the Poisson source coefficient is
    // m_phiscal = 1.5·omega_cb (HACC TimeStepper.cxx:109).  Exposed separately
    // from omega_matter so future neutrino work can split the two without
    // breaking callers that need only the CDM+baryon density.
    double omega_cb() const { return m_cp.omega_cb; }

    // Effective matter density at scale factor a:  Ω_m(a) = Ω_cb / (a^3 · E^2).
    // Used for the Bouchet+ f1, f2 approximations.
    double omega_matter_a(double a) const;

    // Compute growth factors at scale factor a.  Integrates the linearized
    // growth ODE from a_inf=1/(1+z_infinity) to (a) and to (1.0) for normalization.
    // Source: HACC InitCosmology.cxx:70-96.
    GrowthFactors growth_factors(double a) const;

    // Linear growth D1 only (convenience).  Equal to growth_factors(a).D1.
    double D1(double a) const { return growth_factors(a).D1; }

    // Logarithmic growth rate f1(a) = dlnD1/dlna (convenience).
    double f1(double a) const { return growth_factors(a).f1; }

    // Convenience for the IC pipeline: D1(a) and dD1/dt|_a.
    //   gf_out     = D1(a)                   normalized to D1(a=1) = 1
    //   gf_dot_out = dD1/dt|_{a=a_in}        in code units (H0 = 1)
    // Mirrors HACC InitCosmology.cxx:70-96 (CosmoClass::GrowthFactor) which
    // returns dplus/ystart[0] and ddot/ystart[0]; HACC's "ddot" is what comes
    // out of the ODE state y[1] = D'·a·H = (dD/da · a · H), which equals
    // dD/dt because dD/dt = dD/da · da/dt = dD/da · a · H (with H ≡ H/H0
    // and time in H0^{-1}).
    void growth_factor(double a, double& gf_out, double& gf_dot_out) const;

private:
    CosmoParams m_cp;

    // Cached normalization integral: D(a=1) from the un-normalized ODE.
    // Computed lazily on first call to growth_factors().
    mutable double m_D_at_1   = -1.0;  // ystart[0] at a=1 from the high-z start
    mutable double m_Ddot_at_1 = -1.0; // ystart[1] at a=1 from the high-z start
    mutable bool   m_norm_set = false;

    // ODE RHS for the linearized growth equation.
    // y[0] = D(a),  y[1] = D'(a) · a · H(a)/H0   (HACC: InitCosmology.cxx:104-113)
    void growth_rhs(double a, const double y[2], double dydx[2]) const;

    // Adaptive Cash-Karp ODE solver (Numerical Recipes; Press et al.).
    // Integrates y[] from x1 to x2 with relative tolerance eps.
    // Source: HACC InitCosmology.cxx:403-559 (odesolve + rkqs + rkck).
    void odesolve(double y[2], double x1, double x2, double eps) const;
    void rkqs(double y[2], double dydx[2], double& x, double htry,
              double eps, const double yscal[2],
              double& hdid, double& hnext) const;
    void rkck(const double y[2], const double dydx[2], double x, double h,
              double yout[2], double yerr[2]) const;

    // Compute (and cache) ystart[0]/ystart[1] at a=1 from the high-z start.
    void ensure_normalization() const;
};

} // namespace pmk
