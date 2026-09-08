# Licenses of the LAMMPS source bundle

Declared components (path in the bundle, project, license, upstream, commit/version):

| path | project | license | upstream | commit / version |
|---|---|---|---|---|
| `src` | LAMMPS | GPL-2.0-only | https://github.com/lammps/lammps | 9c5ab448c78a14fd534619622162ba418d6a1fb1 |
| `src/lib/kokkos` | Kokkos (bundled) | Apache-2.0 WITH LLVM-exception | https://github.com/kokkos/kokkos | 4.6.2 |
| `src/potentials + src/bench/POTENTIALS` | LAMMPS potential files (upstream repository content) | GPL-2.0-only (repository) | https://github.com/lammps/lammps |  |

Redistribution notes:

- LAMMPS is GPL-2.0-only: redistributing the (unmodified) source in a source bundle is permitted; the bundle carries the upstream LICENSE and the complete corresponding source.
- lib/kokkos (Apache-2.0 with LLVM exception) and the other lib/ components keep their own license files inside the tree.
- No dataset outside the upstream repository is bundled; the benchmark inputs (bench/in.lj, bench/POTENTIALS, potentials/) are upstream repository files.

License/notice files present in the bundle (15):

- `src/LICENSE`
- `src/doc/utils/converters/LICENSE`
- `src/doc/utils/sphinx-config/_themes/lammps_theme/LICENSE`
- `src/lib/awpmd/license/COPYING`
- `src/lib/gpu/cudpp_mini/license.txt`
- `src/lib/h5md/LICENSE`
- `src/lib/kokkos/Copyright.txt`
- `src/lib/kokkos/LICENSE`
- `src/lib/kokkos/LICENSE_FILE_HEADER`
- `src/lib/lepton/LICENSE`
- `src/lib/lepton/asmjit/LICENSE.md`
- `src/lib/poems/Copyright_Notice`
- `src/src/PTM/LICENSE`
- `src/src/fmt/LICENSE`
- `src/tools/phonon/tricubic/LICENSE`
