// src/ic/TransferFunction.cpp
// CAMB-table reader.  Single-file slurp on the constructing rank — these tables
// are <100KB and shared across all ranks anyway; broadcasting saves nothing.
//
// Source mapping:
//   parse loop  ↔ HACC InitCosmology.cxx:151-178
//   T_cb build  ↔ HACC InitCosmology.cxx:165-168 (we use Omega_m as denominator)
//   T_cb(k)     ↔ HACC InitCosmology.cxx:274-277

#include "TransferFunction.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace pmk {

namespace {

// Linear interpolation in (k, T) at a tabulated array (binary search).
// Mirrors HACC's interpolate() (InitCosmology.cxx:638-648), which is also
// linear-in-linear with no log transformation.
double interp_linear(const std::vector<double>& xs,
                     const std::vector<double>& ys,
                     double x)
{
    if (x <= xs.front()) return ys.front();
    if (x >= xs.back())  return ys.back();
    auto it = std::upper_bound(xs.begin(), xs.end(), x);
    const std::size_t hi = static_cast<std::size_t>(it - xs.begin());
    const std::size_t lo = hi - 1;
    const double t = (x - xs[lo]) / (xs[hi] - xs[lo]);
    return ys[lo] + t * (ys[hi] - ys[lo]);
}

} // namespace

TransferFunction::TransferFunction(const std::string& camb_file,
                                   const Cosmology& cosmo)
{
    std::ifstream in(camb_file);
    if (!in) {
        throw std::runtime_error(
            "TransferFunction: cannot open '" + camb_file + "'");
    }

    const auto& cp = cosmo.params();
    const double om_m = cp.omega_matter;
    const double om_cdm = cp.omega_cdm;
    const double om_bar = cp.omega_baryon;
    if (om_m <= 0.0) {
        throw std::runtime_error(
            "TransferFunction: omega_matter must be > 0 to weight T_cb");
    }

    k_table_.reserve(4096);
    T_cb_table_.reserve(4096);

    std::string line;
    std::size_t lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        // Skip blank / whitespace-only lines (common at end-of-file).
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;

        std::istringstream iss(line);
        double k, t_cdm, t_bar, t_gamma, t_nu_ml, t_nu_mass, t_total;
        if (!(iss >> k >> t_cdm >> t_bar >> t_gamma
                  >> t_nu_ml >> t_nu_mass >> t_total)) {
            std::ostringstream msg;
            msg << "TransferFunction: malformed CAMB line " << lineno
                << " in '" << camb_file << "'";
            throw std::runtime_error(msg.str());
        }
        if (!(std::isfinite(k) && std::isfinite(t_cdm) && std::isfinite(t_bar))) {
            std::ostringstream msg;
            msg << "TransferFunction: non-finite value at line " << lineno;
            throw std::runtime_error(msg.str());
        }
        if (k <= 0.0) {
            std::ostringstream msg;
            msg << "TransferFunction: non-positive k=" << k
                << " at line " << lineno;
            throw std::runtime_error(msg.str());
        }
        // T_cb per HACC InitCosmology.cxx:165-168 (denominator: Omega_m).
        const double t_cb = (om_cdm * t_cdm + om_bar * t_bar) / om_m;
        k_table_.push_back(k);
        T_cb_table_.push_back(t_cb);
    }

    if (k_table_.empty()) {
        throw std::runtime_error(
            "TransferFunction: no rows parsed from '" + camb_file + "'");
    }

    // Verify monotone-increasing k — CAMB always emits this, but a corrupt file
    // would silently produce nonsense interpolations.
    for (std::size_t i = 1; i < k_table_.size(); ++i) {
        if (k_table_[i] <= k_table_[i - 1]) {
            std::ostringstream msg;
            msg << "TransferFunction: k not strictly increasing at row "
                << i << " (k[" << i - 1 << "]=" << k_table_[i - 1]
                << ", k[" << i << "]=" << k_table_[i] << ")";
            throw std::runtime_error(msg.str());
        }
    }
}

double TransferFunction::T_cb(double k) const
{
    return interp_linear(k_table_, T_cb_table_, k);
}

} // namespace pmk
