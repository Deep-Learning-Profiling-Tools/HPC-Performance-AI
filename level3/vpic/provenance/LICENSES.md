# Licenses of the VPIC (vpic-kokkos) source bundle

Declared components (path in the bundle, project, license, upstream, commit/version):

| path | project | license | upstream | commit / version |
|---|---|---|---|---|
| `src` | vpic-kokkos (VPIC 2.0) | BSD-3-Clause (README.md "Copyright" section; Triad National Security, LLC, 2022) | https://github.com/lanl/vpic-kokkos | 7eb2947e12741df4931088a3ae6175adb21a6535 |
| `src/kokkos` | Kokkos (bundled submodule) | Apache-2.0 WITH LLVM-exception |  | 5.2.2 |

Redistribution notes:

- VPIC 2.0 is BSD-3-Clause (the license text is the "Copyright" section of README.md; the repository ships no separate LICENSE file); the unmodified source with that README is redistributed as a source bundle.
- The input decks under level3/vpic/decks/ are this benchmark's files, derived from upstream's sample and test decks as documented in inputs.yaml; they are not part of the frozen artifact.

License/notice files present in the bundle (4):

- `src/kokkos/COPYRIGHT.md`
- `src/kokkos/LICENSE`
- `src/kokkos/scripts/license_header.txt`
- `src/kokkos/tpls/gtest/gtest/LICENSE`
