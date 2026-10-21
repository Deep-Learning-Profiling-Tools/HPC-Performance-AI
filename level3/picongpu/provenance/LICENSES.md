# Licenses of the PIConGPU source bundle

Declared components (path in the bundle, project, license, upstream, commit/version):

| path | project | license | upstream | commit / version |
|---|---|---|---|---|
| `src` | PIConGPU (main simulation, buildsystem, etc, include/picongpu, share, src/tools) | GPL-3.0-or-later (LICENSE.md, COPYING) | https://github.com/ComputationalRadiationPhysics/picongpu | 93cba5b61b628659a6ec8181f5e2dd28b8006f29 |
| `src/include/pmacc` | PMacc (PIConGPU's parallel library) | LGPL-3.0-or-later (LICENSE.md, COPYING.LESSER) |  |  |
| `src/thirdParty` | alpaka (MPL-2.0), mallocMC (MIT), nlohmann/json (MIT), toml11 (MIT), cmake-modules, cuda_memtest | per thirdParty/<name>/LICENSE |  |  |
| `deps/boost_1_86_0.tar.gz` | Boost | BSL-1.0 |  | 1.86.0 |
| `deps/gsl-2.8.tar.gz` | GNU Scientific Library | GPL-3.0-or-later |  | 2.8 |

Redistribution notes:

- PIConGPU is GPLv3+ (PMacc LGPLv3+); the unmodified source with its bundled third-party libraries is redistributed as a source bundle (no patches).
- Boost and GSL are redistributed as their unmodified upstream release tarballs (BSL-1.0 / GPLv3+) because the conda environment ships neither; build.sh builds them under .deps/level3/picongpu/<profile>/install.

License/notice files present in the bundle (11):

- `src/COPYING`
- `src/COPYING.LESSER`
- `src/LICENSE.md`
- `src/thirdParty/alpaka/LICENSE`
- `src/thirdParty/alpaka/benchmarks/babelstream/src/LICENSE`
- `src/thirdParty/alpaka/include/alpaka/rand/TinyMT/LICENSE.txt`
- `src/thirdParty/alpaka/thirdParty/catch2/LICENSE.txt`
- `src/thirdParty/cmake-modules/LICENSE`
- `src/thirdParty/mallocMC/LICENSE`
- `src/thirdParty/nlohmann_json/LICENSE.MIT`
- `src/thirdParty/toml11/LICENSE`
