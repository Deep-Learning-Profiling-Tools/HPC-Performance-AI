# Source equivalence: frozen bundle vs validated trees (specfem3d)

## src vs `.deps/level3/specfem3d/src`: **EQUIVALENT**

frozen: `<freeze-scratch>/specfem3d/stage/src`
validated: `.deps/level3/specfem3d/src`

| class | count |
|---|---|
| identical | 6196 |
| expected_patch_difference | 0 |
| expected_generated_difference | 3 |
| expected_normalization_excluded | 0 |
| expected_build_artifact | 0 |
| expected_added | 6 |
| missing_source (UNEXPECTED) | 0 |
| extra_source (UNEXPECTED) | 0 |
| unexpected_content_difference (UNEXPECTED) | 0 |
| **unexpected total** | **0** |
| frozen entries / validated entries considered | 6205 / 6199 |

expected_generated_difference (3):
- `DATA/CMTSOLUTION`
- `DATA/Par_file`
- `DATA/STATIONS`

expected_added (6):
- `external_libs/scotch_5.1.12b/doc/Licence_CeCILL-C_V1-en.txt`
- `external_libs/scotch_5.1.12b/doc/Licence_CeCILL-C_V1-fr.txt`
- `external_libs/scotch_5.1.12b/doc/ptscotch_user5.1.pdf`
- `external_libs/scotch_5.1.12b/doc/ptscotch_user5.1.ps.gz`
- `external_libs/scotch_5.1.12b/doc/scotch_user5.1.pdf`
- `external_libs/scotch_5.1.12b/doc/scotch_user5.1.ps.gz`

