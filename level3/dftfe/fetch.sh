#!/usr/bin/env bash
# Fetch DFT-FE 1.2.0 and the pinned sources of its dependency stack (read-only).
#
#   DFT-FE 1.2.0   7147faa51f7c9f3075fffaa5e48ba989bcd329c1 (github.com/dftfeDevelopers/dftfe, 2025-08-17)
#   install_DFTFE  reference recipe: branch frontierDevelop b27f89b (dftfe2.sh: deal.II 9.7.1, p4est 2.8.7,
#                  Kokkos 4.6.00, ELPA 2026.02.001, libxc 7.0.0, ScaLAPACK 2.2.2, ALGLIB 4.06.0, spglib 02159eef)
#   Tarballs (downloaded once into .deps/level3/dftfe/downloads, sha256 recorded in SHA256SUMS):
#     OpenBLAS 0.3.30, ScaLAPACK 2.2.2, libxc 7.0.0, ALGLIB 4.06.0 (GPL), p4est 2.8.7 (+ dftfe's p4est-setup.sh),
#     Kokkos 4.6.00, deal.II 9.7.1, ELPA 2026.02.001; spglib at commit 02159eef6e7349535049a43fe2272bb634c77945 (git).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
UP="$R/_upstream/level3"; DL="$R/.deps/level3/dftfe/downloads"
SHA=7147faa51f7c9f3075fffaa5e48ba989bcd329c1
mkdir -p "$UP" "$DL"
if [ ! -d "$UP/dftfe/.git" ]; then
    git clone -q https://github.com/dftfeDevelopers/dftfe.git "$UP/dftfe"
    git -C "$UP/dftfe" checkout -q "$SHA"
fi
got="$(git -C "$UP/dftfe" rev-parse HEAD)"; [ "$got" = "$SHA" ] || { echo "fetch.sh: dftfe at $got, expected $SHA" >&2; exit 1; }
declare -A URLS=(
  [OpenBLAS-0.3.30.tar.gz]=https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.30/OpenBLAS-0.3.30.tar.gz
  [v2.2.2.tar.gz]=https://github.com/Reference-ScaLAPACK/scalapack/archive/refs/tags/v2.2.2.tar.gz
  [libxc-7.0.0.tar.gz]=https://gitlab.com/libxc/libxc/-/archive/7.0.0/libxc-7.0.0.tar.gz
  [alglib-4.06.0.cpp.gpl.tgz]=https://www.alglib.net/translator/re/alglib-4.06.0.cpp.gpl.tgz
  [p4est-2.8.7.tar.gz]=https://p4est.github.io/release/p4est-2.8.7.tar.gz
  [p4est-setup.sh]=https://raw.githubusercontent.com/dftfeDevelopers/dftfe/manual/p4est-setup.sh
  [4.6.00.tar.gz]=https://github.com/kokkos/kokkos/archive/refs/tags/4.6.00.tar.gz
  [dealii-9.7.1.tar.gz]=https://github.com/dealii/dealii/releases/download/v9.7.1/dealii-9.7.1.tar.gz
  [dealii-9.6.2.tar.gz]=https://github.com/dealii/dealii/releases/download/v9.6.2/dealii-9.6.2.tar.gz
  [elpa-2026.02.001.tar.gz]=https://elpa.mpcdf.mpg.de/software/tarball-archive/Releases/2026.02.001/elpa-2026.02.001.tar.gz
)
# expected sha256 (recorded at first download, 2026-09-05)
declare -A SHAS=(
  [OpenBLAS-0.3.30.tar.gz]=27342cff518646afb4c2b976d809102e368957974c250a25ccc965e53063c95d
  [v2.2.2.tar.gz]=a2f0c9180a210bf7ffe126c9cb81099cf337da1a7120ddb4cbe4894eb7b7d022
  [libxc-7.0.0.tar.gz]=8d4e343041c9cd869833822f57744872076ae709a613c118d70605539fb13a77
  [alglib-4.06.0.cpp.gpl.tgz]=27fa4b0b8160cbb2ace4f69d1a8ef0990cf894c87dff3f46dc86ec8ac2d30c68
  [p4est-2.8.7.tar.gz]=0a1e912f3529999ca6d62fee335d51f24b5650b586e95a03ef39ebf73936d7f4
  [p4est-setup.sh]=86269c2ef751b33d06916299a3d15fa80f4c8c9de40eceedfb8b7f45bd90fb8b
  [4.6.00.tar.gz]=348b2d860046fc3ddef5ca3a128317be1a6f3fa35196f268338a180fcae52264
  [dealii-9.7.1.tar.gz]=0f2096ef83db54fdcebe9f3d148fa713f63f1c3f567941b53bcb4a1a8ea7de43
  [dealii-9.6.2.tar.gz]=1051e332de3822488e91c2b0460681052a3c4c5ac261cdd7a6af784869a25523
  [elpa-2026.02.001.tar.gz]=a379f27f4dbd27b2ee45017afec656d064301e97150c874649bdfd64957b75ed
)
for f in "${!URLS[@]}"; do
    [ -s "$DL/$f" ] || curl -sSL -o "$DL/$f" "${URLS[$f]}"
    got_sha="$(sha256sum "$DL/$f" | cut -d' ' -f1)"
    [ "$got_sha" = "${SHAS[$f]}" ] || { echo "fetch.sh: $f sha256 $got_sha != expected ${SHAS[$f]}" >&2; exit 1; }
done
if [ ! -d "$DL/spglib/.git" ]; then git clone -q https://github.com/spglib/spglib.git "$DL/spglib"; git -C "$DL/spglib" checkout -q 02159eef6e7349535049a43fe2272bb634c77945; fi
[ "$(git -C "$DL/spglib" rev-parse HEAD)" = 02159eef6e7349535049a43fe2272bb634c77945 ] || { echo "fetch.sh: spglib not at the pinned commit" >&2; exit 1; }
( cd "$DL" && for f in "${!SHAS[@]}"; do echo "${SHAS[$f]}  $f"; done | sort -k2 > SHA256SUMS )
echo "# dftfe $got; downloads under $DL (all sha256 verified):"; sed 's/^/#   /' "$DL/SHA256SUMS"; echo "#   spglib $(git -C "$DL/spglib" rev-parse HEAD)"
