#!/usr/bin/env bash
# Fetch QMCPACK v4.4.0 and the private dependency sources (read-only).
#
#   QMCPACK v4.4.0   2601d62e353934f1526cab1f67f30b6672b7c76f (2026-08-31), NCSA/Illinois licence
#   LLVM 23.1.0      llvm-project-23.1.0.src.tar.xz (toolchain/build_llvm.sh; sha256 in that script)
#   HDF5 1.14.5      hdf5-1.14.5.tar.gz   (QMCPACK's tested HDF5; the conda 2.x is outside upstream's test range)
#   Boost 1.90.0     boost-1.90.0-b2-nodocs.tar.xz (headers only are used; upstream nightlies use 1.90/1.84)
#   OpenBLAS 0.3.30  shared with the other second-batch profiles' downloads (own build per profile)
# Datasets: QMCPACK's in-repo tests/ (178 MB, deterministic + short statistical tests) are used;
# the NiO performance splines (external Box link, 43 MB-8.8 GB each) are NOT downloaded in this round.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
UP="$R/_upstream/level3"; DL="$R/.deps/level3/qmcpack/downloads"
SHA=2601d62e353934f1526cab1f67f30b6672b7c76f
mkdir -p "$UP" "$DL"
if [ ! -d "$UP/qmcpack/.git" ]; then
    git clone -q --branch v4.4.0 --depth 1 https://github.com/QMCPACK/qmcpack.git "$UP/qmcpack"
fi
got="$(git -C "$UP/qmcpack" rev-parse HEAD)"; [ "$got" = "$SHA" ] || { echo "fetch.sh: qmcpack at $got, expected $SHA (v4.4.0)" >&2; exit 1; }
declare -A URLS=(
  [hdf5-1.14.5.tar.gz]=https://github.com/HDFGroup/hdf5/releases/download/hdf5_1.14.5/hdf5-1.14.5.tar.gz
  [boost-1.90.0-b2-nodocs.tar.xz]=https://github.com/boostorg/boost/releases/download/boost-1.90.0/boost-1.90.0-b2-nodocs.tar.xz
  [OpenBLAS-0.3.30.tar.gz]=https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.30/OpenBLAS-0.3.30.tar.gz
)
# sha256 recorded at first download (2026-09-06); OpenBLAS matches the dftfe/geos downloads
declare -A SHAS=(
  [OpenBLAS-0.3.30.tar.gz]=27342cff518646afb4c2b976d809102e368957974c250a25ccc965e53063c95d
)
for f in "${!URLS[@]}"; do
    [ -s "$DL/$f" ] || curl -sSL -o "$DL/$f" "${URLS[$f]}"
    got_sha="$(sha256sum "$DL/$f" | cut -d' ' -f1)"
    if [ -n "${SHAS[$f]:-}" ]; then [ "$got_sha" = "${SHAS[$f]}" ] || { echo "fetch.sh: $f sha256 $got_sha != expected ${SHAS[$f]}" >&2; exit 1; }; fi
    echo "$got_sha  $f"
done | sort -k2 > "$DL/SHA256SUMS"
echo "# qmcpack $got; downloads under $DL:"; sed 's/^/#   /' "$DL/SHA256SUMS"
