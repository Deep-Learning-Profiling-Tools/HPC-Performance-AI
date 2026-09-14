# Source equivalence: frozen bundle vs validated trees (cp2k)

## src vs `_upstream/level3/cp2k`: **EQUIVALENT**

frozen: `<freeze-scratch>/cp2k/stage/src`
validated: `_upstream/level3/cp2k`

| class | count |
|---|---|
| identical | 8997 |
| expected_patch_difference | 2 |
| expected_generated_difference | 0 |
| expected_normalization_excluded | 0 |
| expected_build_artifact | 0 |
| expected_added | 0 |
| missing_source (UNEXPECTED) | 0 |
| extra_source (UNEXPECTED) | 0 |
| unexpected_content_difference (UNEXPECTED) | 0 |
| **unexpected total** | **0** |
| frozen entries / validated entries considered | 8999 / 8999 |

expected_patch_difference (2):
- `tools/toolchain/install_cp2k_toolchain.sh`
- `tools/toolchain/scripts/stage9/install_dbcsr.sh`

## src/tools/toolchain vs `/tmp/hpcperf-l3-b2-scratch/cp2k-toolchain/cuda132-gcc142-ompi5010`: **EQUIVALENT**

frozen: `<freeze-scratch>/cp2k/stage/src/tools/toolchain`
validated: `/tmp/hpcperf-l3-b2-scratch/cp2k-toolchain/cuda132-gcc142-ompi5010`

| class | count |
|---|---|
| identical | 79 |
| expected_patch_difference | 0 |
| expected_generated_difference | 0 |
| expected_normalization_excluded | 0 |
| expected_build_artifact | 0 |
| expected_added | 0 |
| missing_source (UNEXPECTED) | 0 |
| extra_source (UNEXPECTED) | 0 |
| unexpected_content_difference (UNEXPECTED) | 0 |
| **unexpected total** | **0** |
| frozen entries / validated entries considered | 79 / 79 |

