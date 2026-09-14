# Licenses of the CP2K source bundle

Declared components (path in the bundle, project, license, upstream, commit/version):

| path | project | license | upstream | commit / version |
|---|---|---|---|---|
| `src` | CP2K | GPL-2.0-or-later | https://github.com/cp2k/cp2k | 67b5da876dd6a76b8b021d5a04d1c81ba79a4c50 |
| `deps/cp2k-toolchain-dist/OpenBLAS-0.3.33.tar.gz` | OpenBLAS | BSD-3-Clause |  | 0.3.33 |
| `deps/cp2k-toolchain-dist/dbcsr-2.10.0.tar.gz` | DBCSR | GPL-2.0-or-later |  | 2.10.0 |
| `deps/cp2k-toolchain-dist/scalapack-2.2.3.tar.gz` | ScaLAPACK | BSD-3-Clause (modified) |  | 2.2.3 |
| `deps/cp2k-toolchain-dist/fftw-3.3.11.tar.gz` | FFTW | GPL-2.0-or-later |  | 3.3.11 |
| `deps/cp2k-toolchain-dist/libint-v2.13.1-cp2k-lmax-5.tar.xz` | Libint (CP2K build) | LGPL-3.0-only |  | 2.13.1 |
| `deps/cp2k-toolchain-dist/libxc-7.0.0.tar.bz2` | libxc | MPL-2.0 |  | 7.0.0 |
| `deps/cp2k-toolchain-dist/libxsmm-2.0.0.tar.gz` | LIBXSMM | BSD-3-Clause |  | 2.0.0 |
| `deps/cp2k-toolchain-dist/libxs-1.0.0.tar.gz` | LIBXS | BSD-3-Clause |  | 1.0.0 |
| `deps/cp2k-toolchain-dist/spglib-2.7.0.tar.gz` | spglib | BSD-3-Clause |  | 2.7.0 |
| `deps/cp2k-toolchain-dist/eigen-5.0.1.tar.gz` | Eigen | MPL-2.0 |  | 5.0.1 |

Redistribution notes:

- CP2K, DBCSR and FFTW are GPL-2.0-or-later: the bundle redistributes their complete, unmodified source (the toolchain scripts of CP2K carry the class-B B200 back-port, recorded in provenance/original_vs_baseline.diff).
- The toolchain tarballs are the exact files CP2K's installer downloads from cp2k.org (same sha256 as recorded in tools/toolchain/scripts/*/install_*.sh); they are redistributable under their own licenses (BSD/MPL/LGPL/GPL).
- data/ (basis sets, potentials) and benchmarks/ are upstream repository content under the CP2K license.

License/notice files present in the bundle (5):

- `src/LICENSE`
- `src/src/dbm/LICENSE`
- `src/src/grid/LICENSE`
- `src/src/grpp/LICENSE`
- `src/src/offload/LICENSE`
