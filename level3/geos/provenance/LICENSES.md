# Licenses of the GEOS source bundle

Declared components (path in the bundle, project, license, upstream, commit/version):

| path | project | license | upstream | commit / version |
|---|---|---|---|---|
| `src` | GEOS | LGPL-2.1-only | https://github.com/GEOS-DEV/GEOS | b7a0f13305277c3d825ee93f34946ac4e8c94fee |
| `src/src/cmake/blt` | BLT | BSD-3-Clause | https://github.com/LLNL/blt | 9ff77344f0b2a6ee345e452bddd6bfd46cbbfa35 |
| `src/src/coreComponents/LvArray` | LvArray | BSD-3-Clause | https://github.com/GEOS-DEV/LvArray | b8bc3ef762bf0236d7f94965b3fbbd458e6d537d |
| `src/src/coreComponents/constitutive/HPCReact` | HPCReact | BSD-3-Clause | https://github.com/GEOS-DEV/HPCReact | 7423e3e2cd55e959d9a823ab8241351f8b1087df |
| `src/src/coreComponents/fileIO/coupling/hdf5_interface` | hdf5_interface | BSD-3-Clause | https://github.com/GEOS-DEV/hdf5_interface | efe82a8ed0f8ac4995b8859bff9f0bf5bf0b6ef4 |
| `deps/thirdPartyLibs` | GEOS thirdPartyLibs | LGPL-2.1-only (GEOS project) | https://github.com/GEOS-DEV/thirdPartyLibs | 9b55672f6f8a73d02fd632396eb0410e58c9b120 |
| `deps/tpl-dist/parmetis` | ParMETIS 4.0.3 (+METIS 5.1.0) | ParMETIS: University of Minnesota research license (redistribution restricted); METIS 5: Apache-2.0 | https://ftp.mcs.anl.gov/pub/pdetools/spack-pkgs/parmetis-4.0.3.tar.gz |  |
| `deps/tpl-dist/suitesparse` | SuiteSparse 5.10.1 | mixed LGPL/GPL/BSD per package | https://github.com/DrTimothySDavis/SuiteSparse |  |
| `deps/tpl-dist/uncrustify` | uncrustify | GPL-2.0-or-later | https://github.com/uncrustify/uncrustify |  |
| `deps/tpl-dist/scotch` | Scotch 7.0.8 | CeCILL-C | https://gitlab.inria.fr/scotch/scotch |  |
| `deps/tpl-dist/{chai,raja,conduit,silo,vtk,superlu_dist,hdf5,pugixml,fmt,hypre}` | CHAI/Umpire/camp, RAJA, Conduit, Silo, VTK, SuperLU_DIST, HDF5, pugixml, fmt, hypre | BSD-3-Clause / MIT / Apache-2.0 (permissive) | see the component list |  |
| `deps/openblas` | OpenBLAS | BSD-3-Clause |  | 0.3.30 |

Redistribution notes:

- LICENSE BLOCKER (decision needed): parmetis-4.0.3.tar.gz -- ParMETIS is distributed under the University of Minnesota license, which permits use for research/educational purposes but does not grant a general right to redistribute the source. GEOS' superbuild downloads it from an ANL mirror. Bundling it into a redistributed source artifact that is distributed with this repository may not be permitted; the freeze records it here and the user decides whether to (a) keep it in the bundle (private repository only), (b) move it to an environment-provided download resolved by build.sh from the recorded URL+sha256, or (c) drop ParMETIS. METIS 5.1.0 (inside the same tarball) is Apache-2.0.
- SuiteSparse 5.10.1 mixes LGPL/GPL modules (UMFPACK is GPL); the complete source is redistributed, which GPL permits; no derived binary is distributed.
- uncrustify (GPL-2.0-or-later) is only a build-time code formatter that the superbuild builds; its source is redistributed complete.
- The GEOS integrated-test baseline tarball (baseline_integratedTests-pr3994-17525-4ae3593.tar.gz, 1.57 GB, LGPL-2.1 GEOS project data) is a DATASET, not source: it stays outside the source bundle (.deps/level3/geos/downloads/, url+sha256 recorded in benchmark.yaml) -- see benchmark.yaml `datasets`.

License/notice files present in the bundle (26):

- `deps/thirdPartyLibs/cmake/blt/LICENSE`
- `deps/thirdPartyLibs/cmake/blt/NOTICE`
- `deps/thirdPartyLibs/cmake/blt/thirdparty_builtin/benchmark-1.9.1/LICENSE`
- `deps/thirdPartyLibs/cmake/blt/thirdparty_builtin/fruit-3.4.1/LICENSE.txt`
- `deps/thirdPartyLibs/cmake/blt/thirdparty_builtin/googletest-1.16.0/LICENSE`
- `src/COPYRIGHT`
- `src/LICENSE`
- `src/NOTICE`
- `src/scripts/copyrightPrepender.py`
- `src/src/cmake/blt/LICENSE`
- `src/src/cmake/blt/NOTICE`
- `src/src/cmake/blt/thirdparty_builtin/benchmark-1.8.0/LICENSE`
- `src/src/cmake/blt/thirdparty_builtin/fruit-3.4.1/LICENSE.txt`
- `src/src/cmake/blt/thirdparty_builtin/googletest/LICENSE`
- `src/src/coreComponents/LvArray/LICENSE`
- `src/src/coreComponents/LvArray/cmake/blt/LICENSE`
- `src/src/coreComponents/LvArray/cmake/blt/NOTICE`
- `src/src/coreComponents/LvArray/cmake/blt/thirdparty_builtin/benchmark-1.8.0/LICENSE`
- `src/src/coreComponents/LvArray/cmake/blt/thirdparty_builtin/fruit-3.4.1/LICENSE.txt`
- `src/src/coreComponents/LvArray/cmake/blt/thirdparty_builtin/googletest/LICENSE`
- `src/src/coreComponents/constitutive/HPCReact/LICENSE`
- `src/src/coreComponents/constitutive/HPCReact/cmake/blt/LICENSE`
- `src/src/coreComponents/constitutive/HPCReact/cmake/blt/NOTICE`
- `src/src/coreComponents/constitutive/HPCReact/cmake/blt/thirdparty_builtin/benchmark-1.9.1/LICENSE`
- `src/src/coreComponents/constitutive/HPCReact/cmake/blt/thirdparty_builtin/fruit-3.4.1/LICENSE.txt`
- `src/src/coreComponents/constitutive/HPCReact/cmake/blt/thirdparty_builtin/googletest-1.16.0/LICENSE`
