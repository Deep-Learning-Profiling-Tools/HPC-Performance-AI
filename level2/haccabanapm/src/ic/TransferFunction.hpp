#pragma once
// src/ic/TransferFunction.hpp
// CAMB transfer-function reader and T_cb(k) evaluator.
//
// HACC reference:
//   - TransferClass:                  nbody/initializer/InitCosmology.cxx:118-270
//   - camb_cb (T_cb evaluation):      InitCosmology.cxx:274-277
//   - table_tf_cb construction:       InitCosmology.cxx:165-168
//   - interpolate (linear in linear): InitCosmology.cxx:638-648
//
// Spec: ../analysis/prompts/10b_implement_zeldovich_ic.md (Deliverable 1)
//
// Format of the CAMB file (cmb.tf): 7 whitespace-separated columns, one per row,
// with no header.  Columns are:
//   k [h/Mpc],  T_cdm,  T_bar,  T_gamma,  T_nu_massless,  T_nu_massive,  T_total
// Only k, T_cdm, and T_bar are read here; the other columns are parsed and dropped.
//
// T_cb(k) per HACC InitCosmology.cxx ::camb_cb — divide by Omega_m (not Omega_cb).
// Equivalent for no-neutrino cosmology; differs when massive neutrinos are present.
// HACC is the canonical reference; the 10b prompt's suggestion to divide by
// Omega_cb is recorded as a prompt-vs-HACC discrepancy in
// ~/ClaudeCode/analysis/prompt_issues/10b_issues.md and we follow HACC.
//
//   T_cb(k) = (Omega_cdm * T_cdm(k) + Omega_baryon * T_bar(k)) / Omega_m
//
// Interpolation: linear in (k, T_cb), matching HACC InitCosmology.cxx:638-648.
// Outside [k_min, k_max] we saturate to the boundary value.

#include "../Cosmology.hpp"

#include <string>
#include <vector>

namespace pmk {

class TransferFunction {
public:
    // Read a CAMB-format file.  Throws std::runtime_error on open failure or a
    // malformed table (zero entries, non-monotone k, NaNs).
    TransferFunction(const std::string& camb_file, const Cosmology& cosmo);

    // T_cb(k), with k in h/Mpc.  Saturated to boundary values outside the
    // tabulated range (T(k_min) below k_min, T(k_max) above k_max).
    double T_cb(double k) const;

    double k_min() const { return k_table_.front(); }
    double k_max() const { return k_table_.back(); }

    // Number of tabulated rows (after parsing).  Exposed for tests.
    std::size_t size() const { return k_table_.size(); }

private:
    std::vector<double> k_table_;     // ascending, h/Mpc
    std::vector<double> T_cb_table_;  // CDM+baryon weighted, dimensionless
};

} // namespace pmk
