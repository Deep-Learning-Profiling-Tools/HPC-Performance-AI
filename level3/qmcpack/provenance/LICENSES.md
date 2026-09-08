# Licenses of the QMCPACK source bundle

Declared components (path in the bundle, project, license, upstream, commit/version):

| path | project | license | upstream | commit / version |
|---|---|---|---|---|
| `src` | QMCPACK | NCSA (University of Illinois/NCSA Open Source) | https://github.com/QMCPACK/qmcpack | 2601d62e353934f1526cab1f67f30b6672b7c76f |
| `deps/hdf5` | HDF5 | BSD-3-Clause (HDF5) | https://github.com/HDFGroup/hdf5 | 1.14.5 |
| `deps/boost` | Boost | BSL-1.0 | https://www.boost.org | 1.90.0 |
| `deps/openblas` | OpenBLAS | BSD-3-Clause | https://github.com/OpenMathLib/OpenBLAS | 0.3.30 |

Redistribution notes:

- QMCPACK (NCSA), HDF5, Boost and OpenBLAS permit redistribution of the source with their license texts, which are inside the respective trees/tarballs.
- The in-repository tests/ data (diamondC pseudopotentials, wavefunction h5) are upstream repository content; the NiO performance datasets (external Box links) are NOT bundled.
- LLVM 23.1.0 (Apache-2.0 WITH LLVM-exception) is an environment-provided compiler and not part of the bundle.

License/notice files present in the bundle (5):

- `src/LICENSE`
- `src/external_codes/Catch2/LICENSE.txt`
- `src/external_codes/boost_multi/multi/LICENSE`
- `src/external_codes/mpi_wrapper/mpi3/LICENSE`
- `src/nexus/LICENSE`
