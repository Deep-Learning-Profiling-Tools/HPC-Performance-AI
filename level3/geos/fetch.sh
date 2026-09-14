#!/usr/bin/env bash
# Fetch GEOS and its thirdPartyLibs superbuild at fixed commits (read-only checkouts).
#
#   GEOS develop     b7a0f13305277c3d825ee93f34946ac4e8c94fee (2026-09-04, "fix: SinglePhaseWell Thermal (#4126)")
#                    submodules: BLT, LvArray, HPCReact, hdf5_interface (as pinned by that commit)
#   thirdPartyLibs   9b55672f6f8a73d02fd632396eb0410e58c9b120 (2026-08-26, "Update TPLs (#361)" = the
#                    GEOS_TPL_TAG 361-1070 GEOS develop's .devcontainer/CI images are built from)
#
# Why a fixed develop snapshot and not release 1.2.0 (2024-10): 1.2.0 pins TPL tag 284-535
# (RAJA/CHAI/Umpire 2024.07, hypre 2.31, CUDA <= 12.5 heritage) and its CI never went
# beyond CUDA 12.5; the 2026 TPL set (RAJA/CHAI/Umpire 2026.07.0 -- the versions Level 2
# already builds with CUDA 13.2 on this node -- hypre f1374fb6, VTK 9.7) is the one with
# a plausible CUDA 13.2 / sm_100 path. The snapshot is pinned by SHA; nothing floats.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
UP="$R/_upstream/level3"
GEOS_SHA=b7a0f13305277c3d825ee93f34946ac4e8c94fee
TPL_SHA=9b55672f6f8a73d02fd632396eb0410e58c9b120
mkdir -p "$UP"
if [ ! -d "$UP/GEOS/.git" ]; then
    git clone -q --branch develop --depth 1 https://github.com/GEOS-DEV/GEOS.git "$UP/GEOS"
    git -C "$UP/GEOS" fetch -q --depth 1 origin "$GEOS_SHA"; git -C "$UP/GEOS" checkout -q "$GEOS_SHA"
    git -C "$UP/GEOS" submodule update -q --init --depth 1
fi
g="$(git -C "$UP/GEOS" rev-parse HEAD)"; [ "$g" = "$GEOS_SHA" ] || { echo "fetch.sh: GEOS at $g, expected $GEOS_SHA" >&2; exit 1; }
if [ ! -d "$UP/thirdPartyLibs/.git" ]; then
    git clone -q https://github.com/GEOS-DEV/thirdPartyLibs.git "$UP/thirdPartyLibs"
    git -C "$UP/thirdPartyLibs" checkout -q "$TPL_SHA"; git -C "$UP/thirdPartyLibs" submodule update -q --init --depth 1
fi
t="$(git -C "$UP/thirdPartyLibs" rev-parse HEAD)"; [ "$t" = "$TPL_SHA" ] || { echo "fetch.sh: thirdPartyLibs at $t, expected $TPL_SHA" >&2; exit 1; }
echo "# GEOS $g (develop snapshot); thirdPartyLibs $t (TPL tag 361-1070)"
git -C "$UP/GEOS" submodule status | sed 's/^/#   GEOS submodule /'
git -C "$UP/thirdPartyLibs" submodule status | sed 's/^/#   TPL submodule /'
