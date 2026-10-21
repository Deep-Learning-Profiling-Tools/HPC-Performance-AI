#!/usr/bin/env bash
# Fetch vpic-kokkos at the recorded commit, with its pinned Kokkos submodule, into
# _upstream/level3/vpic-kokkos (gitignored, read-only reference; FREEZE-TIME ONLY -- build.sh never
# calls this). Idempotent; a checkout at another commit is an error, never silently reused.
# vpic-kokkos has no release tags (only archive/* tags), so the identity is the devel-branch commit;
# Kokkos is the submodule commit upstream pins (lanl/vpic-kokkos PR #127 "Kokkos5": 5.2.2).
#
#   ./fetch.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
UPSTREAM_URL="https://github.com/lanl/vpic-kokkos.git"
UPSTREAM_REF="devel"
UPSTREAM_SHA="7eb2947e12741df4931088a3ae6175adb21a6535"
KOKKOS_SHA="6d4434baacc86ec428b5300ebe853a085760d9af"
DEST="$R/_upstream/level3/vpic-kokkos"
if [ -d "$DEST/.git" ]; then
    have="$(git -C "$DEST" rev-parse HEAD)"
    if [ "$have" = "$UPSTREAM_SHA" ]; then
        ksha="$(git -C "$DEST/kokkos" rev-parse HEAD 2>/dev/null || echo none)"
        [ "$ksha" = "$KOKKOS_SHA" ] || { echo "fetch.sh: $DEST/kokkos is at $ksha, not the recorded $KOKKOS_SHA" >&2; exit 1; }
        echo "fetch.sh: $DEST already at $UPSTREAM_REF ($UPSTREAM_SHA), kokkos $KOKKOS_SHA"; exit 0
    fi
    echo "fetch.sh: $DEST is at $have, not the recorded $UPSTREAM_SHA; remove it to re-fetch" >&2; exit 1
fi
mkdir -p "$(dirname "$DEST")"
echo "fetch.sh: cloning $UPSTREAM_URL @ $UPSTREAM_REF, checking out $UPSTREAM_SHA"
git clone --quiet --branch "$UPSTREAM_REF" "$UPSTREAM_URL" "$DEST"
git -C "$DEST" checkout --quiet "$UPSTREAM_SHA"
have="$(git -C "$DEST" rev-parse HEAD)"
[ "$have" = "$UPSTREAM_SHA" ] || { echo "fetch.sh: checkout resolved to $have, expected $UPSTREAM_SHA" >&2; exit 1; }
git -C "$DEST" submodule update --init --quiet kokkos
ksha="$(git -C "$DEST/kokkos" rev-parse HEAD)"
[ "$ksha" = "$KOKKOS_SHA" ] || { echo "fetch.sh: kokkos submodule resolved to $ksha, expected $KOKKOS_SHA" >&2; exit 1; }
echo "fetch.sh: ok -> $DEST ($UPSTREAM_SHA, kokkos $ksha)"
