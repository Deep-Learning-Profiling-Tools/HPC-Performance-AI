# Licenses of the SPECFEM3D Cartesian source bundle

Declared components (path in the bundle, project, license, upstream, commit/version):

| path | project | license | upstream | commit / version |
|---|---|---|---|---|
| `src` | SPECFEM3D Cartesian | GPL-3.0-or-later | https://github.com/SPECFEM/specfem3d | c67d3ae7d4bfc5ac75cb9e5601d93afa262d3d8d |
| `src/external_libs/scotch_5.1.12b` | SCOTCH (bundled) | CeCILL-C | https://gitlab.inria.fr/scotch/scotch | 5.1.12b |

Redistribution notes:

- SPECFEM3D is GPL (see LICENSE); redistribution of the complete source with the two back-ported devel patches (also GPL) is permitted; the diff is recorded in provenance/original_vs_baseline.diff.
- SCOTCH 5.1.12b is bundled by upstream under CeCILL-C (LGPL-compatible free software license); its own license file stays in the tree.
- EXAMPLES/ (meshes, reference seismograms used by validate.sh) are upstream repository content; doc/ is excluded (documentation only).

License/notice files present in the bundle (6):

- `src/CUBIT_GEOCUBIT/LICENSE`
- `src/LICENSE`
- `src/external_libs/AxiSEM_for_SPECFEM3D/AxiSEM_modif_for_coupling_with_specfem/SOLVER/LICENSE_GPL.txt`
- `src/external_libs/scotch_5.1.12b/LICENSE_en.txt`
- `src/external_libs/scotch_5.1.12b/doc/Licence_CeCILL-C_V1-en.txt`
- `src/external_libs/scotch_5.1.12b/doc/Licence_CeCILL-C_V1-fr.txt`
