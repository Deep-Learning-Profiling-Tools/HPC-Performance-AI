# Licenses of the WarpX source bundle

Declared components (path in the bundle, project, license, upstream, commit/version):

| path | project | license | upstream | commit / version |
|---|---|---|---|---|
| `src` | WarpX | BSD-3-Clause-LBNL | https://github.com/BLAST-WarpX/warpx | 0c62c75e53a9ad08241535444bd7e53fd1deba88 |
| `deps/amrex` | AMReX | BSD-3-Clause-LBNL | https://github.com/AMReX-Codes/amrex | a52ca73324ac2c7b65ec04f131e6df99eec9c576 |

Redistribution notes:

- WarpX and AMReX carry the LBNL BSD-3-Clause license with NOTICE files; both files are kept in the bundle.
- PICSAR-QED and openPMD are not part of this build (WarpX_QED=OFF, WarpX_OPENPMD=OFF) and are not bundled.

License/notice files present in the bundle (4):

- `deps/amrex/LICENSE`
- `deps/amrex/NOTICE`
- `src/LICENSE.txt`
- `src/NOTICE.txt`
