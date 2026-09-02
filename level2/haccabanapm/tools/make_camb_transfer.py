#!/usr/bin/env python3
"""Generate the CAMB transfer-function table HACCabanaPM's pm_ic needs.

Upstream HACCabanaPM reads a CAMB transfer function (indat key
INPUT_BASE_NAME, e.g. ``cmbM000.tf``) but does not ship the file: it came
from the (non-public) HACC source tree, and the upstream ``data/`` directory
that the tests reference was never committed.  This script regenerates the
table with the public CAMB code for the cosmology in ``apps/demo/indat.params``
(HACC's "M000" model):

    OMEGA_CDM 0.22   DEUT 0.02258 (= Omega_b h^2)   HUBBLE 0.71
    NS 0.963         SS8 0.8                        T_CMB 2.726
    OMEGA_NU 0.0     W_DE -1.0                      WA_DE 0.0

File format (src/ic/TransferFunction.hpp): 7 whitespace-separated columns,
no header, k ascending:

    k [h/Mpc]  T_cdm  T_baryon  T_photon  T_nu_massless  T_nu_massive  T_total

which is exactly the classic CAMB ``*_transfer_out.dat`` layout, i.e. the
first seven rows of ``results.get_matter_transfer_data().transfer_data`` at
z = 0.  pm_ic uses only k, T_cdm and T_baryon, and renormalises P(k) to
sigma_8 itself, so only the *shape* of T(k) matters -- the overall CAMB
normalisation is irrelevant.

Usage:
    python3 tools/make_camb_transfer.py [output_path]   (default: apps/demo/cmbM000.tf)

Requires the ``camb`` Python package (pip install camb).  Deterministic for a
given CAMB version; the shipped file was made with CAMB 2.0.4.
"""
import os
import sys

import camb

# Cosmology from apps/demo/indat.params.
H0 = 71.0
OMEGA_CDM = 0.22
DEUT = 0.02258          # Omega_b h^2
NS = 0.963
TCMB = 2.726
h = H0 / 100.0

# Extend the table well past the Nyquist wavenumber of any grid we run
# (sqrt(3) * pi * ng / rL ~ 5.4 h/Mpc for 256 cells on 256 Mpc/h) so that the
# sigma_8 normalisation integral in src/ic/PowerSpectrum.cpp converges on
# tabulated values rather than on the saturated boundary value.
KMAX_MPC_INV = 500.0    # CAMB's kmax is in 1/Mpc (not h/Mpc)

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "apps", "demo", "cmbM000.tf")

pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=DEUT, omch2=OMEGA_CDM * h * h, omk=0.0,
                   mnu=0.0, num_massive_neutrinos=0, TCMB=TCMB)
pars.InitPower.set_params(ns=NS, As=2.0e-9)   # amplitude is renormalised by pm_ic
pars.set_dark_energy(w=-1.0, wa=0.0, dark_energy_model="fluid")
pars.WantCls = False
pars.WantScalars = False
pars.set_matter_power(redshifts=[0.0], kmax=KMAX_MPC_INV, k_per_logint=30,
                      nonlinear=False, silent=True)
pars.Transfer.high_precision = True
pars.set_accuracy(AccuracyBoost=2.0)

results = camb.get_results(pars)
tr = results.get_matter_transfer_data()
# transfer_data[col, ik, iz]; columns (0-based): 0 k/h, 1 CDM, 2 baryon,
# 3 photon, 4 massless nu, 5 massive nu, 6 total matter.
data = tr.transfer_data[:7, :, 0]

with open(out, "w") as f:
    for ik in range(data.shape[1]):
        f.write("  ".join("%.8e" % v for v in data[:, ik]) + "\n")

print("CAMB %s: wrote %d rows, k = [%.4e, %.4e] h/Mpc -> %s"
      % (camb.__version__, data.shape[1], data[0, 0], data[0, -1], out))
print("  sigma_8 of the CAMB (As=2e-9) spectrum: %.4f (pm_ic renormalises to SS8)"
      % results.get_sigma8_0())
