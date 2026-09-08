# Licenses of the Nyx source bundle

Declared components (path in the bundle, project, license, upstream, commit/version):

| path | project | license | upstream | commit / version |
|---|---|---|---|---|
| `src` | Nyx | BSD-3-Clause-LBNL | https://github.com/AMReX-Astro/Nyx | e06eabc1b9dbcad5612db9529aced682402daede |
| `src/subprojects/sundials` | SUNDIALS | BSD-3-Clause | https://github.com/LLNL/sundials | 5c53be85c88f63c5201c130b8cb2c686615cfb03 |
| `deps/amrex` | AMReX | BSD-3-Clause-LBNL | https://github.com/AMReX-Codes/amrex | a52ca73324ac2c7b65ec04f131e6df99eec9c576 |

Redistribution notes:

- All three components are BSD-3-Clause (LBNL variant for Nyx/AMReX); license and notice files are kept in the bundle.
- Initial-condition files inside the Nyx repository (Exec/MiniSB/ic_sb_32.ascii, Exec/LyA/32.nyx, 64sssss_20mpc.nyx) are upstream repository content under the same license; no external dataset is bundled.

License/notice files present in the bundle (19):

- `deps/amrex/LICENSE`
- `deps/amrex/NOTICE`
- `src/copyright.txt`
- `src/license.txt`
- `src/subprojects/sundials/LICENSE`
- `src/subprojects/sundials/NOTICE`
- `src/subprojects/sundials/doc/shared/LicenseReleaseNumbers.rst`
- `src/subprojects/sundials/src/arkode/LICENSE`
- `src/subprojects/sundials/src/arkode/NOTICE`
- `src/subprojects/sundials/src/cvode/LICENSE`
- `src/subprojects/sundials/src/cvode/NOTICE`
- `src/subprojects/sundials/src/cvodes/LICENSE`
- `src/subprojects/sundials/src/cvodes/NOTICE`
- `src/subprojects/sundials/src/ida/LICENSE`
- `src/subprojects/sundials/src/ida/NOTICE`
- `src/subprojects/sundials/src/idas/LICENSE`
- `src/subprojects/sundials/src/idas/NOTICE`
- `src/subprojects/sundials/src/kinsol/LICENSE`
- `src/subprojects/sundials/src/kinsol/NOTICE`
