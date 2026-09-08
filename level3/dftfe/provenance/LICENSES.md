# Licenses of the DFT-FE source bundle

Declared components (path in the bundle, project, license, upstream, commit/version):

| path | project | license | upstream | commit / version |
|---|---|---|---|---|
| `src` | DFT-FE | LGPL-2.1-or-later | https://github.com/dftfeDevelopers/dftfe | 7147faa51f7c9f3075fffaa5e48ba989bcd329c1 |
| `deps/openblas` | OpenBLAS | BSD-3-Clause |  | 0.3.30 |
| `deps/scalapack` | ScaLAPACK | BSD-3-Clause (modified) |  | 2.2.2 |
| `deps/libxc` | libxc | MPL-2.0 |  | 7.0.0 |
| `deps/alglib` | ALGLIB free edition | GPL-2.0-or-later |  | 4.06.0 |
| `deps/p4est` | p4est | GPL-2.0-or-later |  | 2.8.7 |
| `deps/kokkos` | Kokkos | Apache-2.0 WITH LLVM-exception |  | 4.6.00 |
| `deps/dealii` | deal.II | LGPL-2.1-or-later |  | 9.6.2 |
| `deps/elpa` | ELPA | LGPL-3.0-only |  | 2026.02.001 |
| `deps/spglib` | spglib | BSD-3-Clause |  | 02159eef6e7349535049a43fe2272bb634c77945 |

Redistribution notes:

- All dependency sources are redistributable source tarballs/checkouts with their license files inside (LGPL/GPL/MPL/BSD/Apache); the bundle redistributes them unmodified.
- ALGLIB free edition and p4est are GPL: the bundle ships their complete source, which satisfies the source-distribution requirement; the resulting dftfe binary is not distributed.
- deal.II 9.7.1 (attempt 1, incompatible API) is NOT bundled; only the 9.6.2 tarball used by the validated build is.
- Pseudopotentials and the upstream GPU reference outputs (testsGPU/, accuracyBenchmarks) are upstream repository content.

License/notice files present in the bundle (2):

- `deps/spglib/COPYING`
- `src/LICENSE`
