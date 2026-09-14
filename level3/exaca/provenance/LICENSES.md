# Licenses of the ExaCA source bundle

Declared components (path in the bundle, project, license, upstream, commit/version):

| path | project | license | upstream | commit / version |
|---|---|---|---|---|
| `src` | ExaCA | MIT | https://github.com/LLNL/ExaCA | d26e59cd51e241a327c5267d43fd70537e5425f7 |
| `deps/kokkos` | Kokkos | Apache-2.0 WITH LLVM-exception | https://github.com/kokkos/kokkos | 82799e4577568f9666bde36265ac15d78da3e6c8 |
| `deps/json/json-3.12.0.tar.xz` | nlohmann_json (release tarball) | MIT | https://github.com/nlohmann/json | 3.12.0 |
| `src/examples/Materials + src/examples/Substrate` | ExaCA example material/orientation data (upstream repository content) | MIT | https://github.com/LLNL/ExaCA |  |

Redistribution notes:

- ExaCA (LLNL, MIT with NOTICE) and its example data (material interfacial-response files, 10,000-orientation grain files) are upstream repository content; redistribution in a source artifact is permitted.
- Kokkos 4.7.04 (Apache-2.0 WITH LLVM-exception) is a benchmark-specific source dependency (not bundled by ExaCA upstream); its LICENSE stays in the tree.
- nlohmann_json 3.12.0 (MIT) is the exact release tarball ExaCA's CMake would otherwise download at configure time (FetchContent); bundling it removes the build-time download.
- Finch (coupled heat transport) and GoogleTest are NOT bundled: the benchmark case does not use Finch and the unit tests are not built; no Finch-ExaCA coupled workflow is claimed.
- No external dataset (ExaCA-Data temperature files) is bundled: the benchmark uses the analytic Directional problem type.

License/notice files present in the bundle (6):

- `deps/kokkos/Copyright.txt`
- `deps/kokkos/LICENSE`
- `deps/kokkos/LICENSE_FILE_HEADER`
- `deps/kokkos/tpls/gtest/gtest/LICENSE`
- `src/LICENSE`
- `src/NOTICE`
