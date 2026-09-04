#pragma once
// src/pm/PackUnpack.hpp
// FP32-real -> FP64-complex packing kernels for the Cabana::Grid heFFTe path.
//
// CABANA_VERIFICATION_v07_05c_local.md §B6: "real-to-complex FFT is not
// supported; must pack manually".  These two ~10-line kernels are the
// promotion/demotion pair around forward/reverse FFT.

#include "../Grid.hpp"

namespace pmk {

// Promote rho (1 dof, FP32) into phi (2 dofs, FP64): real <- rho, imag <- 0.
// Iterates owned cells only (matches Cabana::Grid::Own + Cell + Local).
void pack_rho_to_complex(const Grid::rho_array_t& rho, Grid::phi_array_t& phi);

// Extract the real part of phi (FP64 complex) into a 1-dof FP32 cell array.
// Used after the reverse FFT to write the potential / force component back
// to a real-valued field for downstream interpolation kernels.
//
// Precision note: this step performs an FP64 → FP32 narrowing cast and is
// therefore lossy.  Even though the FFT itself is FP64-accurate (the in-place
// forward+reverse round-trip on phi is good to ~1e-10; see
// tests/test_fft_roundtrip.cpp), any pipeline that ends in unpack_phi_real
// is bounded by single-precision epsilon (~1e-7).  test_greens_function uses
// a 1e-6 tolerance for that reason.  If FP64 fidelity is needed downstream,
// read phi(:,:,:,0) directly instead of unpacking.
void unpack_phi_real(const Grid::phi_array_t& phi, Grid::rho_array_t& dst);

} // namespace pmk
