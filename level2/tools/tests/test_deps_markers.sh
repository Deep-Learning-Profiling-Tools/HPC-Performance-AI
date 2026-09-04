#!/bin/bash
# Isolated tests for setup_level2_deps.sh mark_built / fingerprint / is_built.
# No dependency is built: the script is sourced in library mode against a
# temporary fake install root. Exit 0 = all cases pass.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../../.." && pwd)"
T="$(mktemp -d "${TMPDIR:-/tmp}/hpcperf-depstest.XXXXXX")"
trap 'chmod -R u+w "$T" 2>/dev/null; rm -rf "$T"' EXIT
pass=0; failn=0
ok()   { echo "ok   $*"; pass=$((pass+1)); }
bad()  { echo "FAIL $*"; failn=$((failn+1)); }

export HPCPERF_DEPS_LIBMODE=1
# shellcheck disable=SC1091
source "$R/setup_level2_deps.sh"
# redirect every root into the sandbox; fake toolchain facts
ROOT="$T/root"; INST="$T/inst"; SRC="$T/src"; PROFILE="test"
CUDA_ARCH=100; CXX=/bin/true
mkdir -p "$ROOT/patches" "$INST/kokkos" "$INST/mfem" "$INST/hypre" "$INST/legacy"

# ---- case A: no patch for the dep -> mark_built succeeds under set -e,
#      fingerprint written, marker written LAST and equal to the pinned tag
( set -e; mark_built kokkos ); rc=$?
if [ "$rc" -eq 0 ] && [ -f "$INST/kokkos/.hpcperf-fingerprint" ] \
   && [ "$(cat "$INST/kokkos/.hpcperf-built")" = "5.2.1" ] \
   && ! grep -q '^patch=' "$INST/kokkos/.hpcperf-fingerprint"; then
    ok "A: no patches -> mark_built rc=0, fingerprint without patch lines, marker=5.2.1"
else
    bad "A: no patches (rc=$rc)"; ls -la "$INST/kokkos"; cat "$INST/kokkos/.hpcperf-fingerprint" 2>/dev/null
fi

# ---- case B: a patch exists -> its name + sha256 recorded
printf 'diff --git a/x b/x\n' > "$ROOT/patches/mfem-test.patch"
sha="$(sha256sum "$ROOT/patches/mfem-test.patch" | cut -d' ' -f1)"
( set -e; mark_built mfem ); rc=$?
if [ "$rc" -eq 0 ] && grep -q "^patch=mfem-test.patch sha256=$sha\$" "$INST/mfem/.hpcperf-fingerprint" \
   && [ "$(cat "$INST/mfem/.hpcperf-built")" = "v4.10" ]; then
    ok "B: patch present -> patch line with sha256 recorded, marker written"
else
    bad "B: patch present (rc=$rc)"; cat "$INST/mfem/.hpcperf-fingerprint" 2>/dev/null
fi

# ---- case C: fingerprint cannot be written -> non-zero AND no marker
chmod 500 "$INST/hypre"
( set -e; mark_built hypre ) 2>/dev/null; rc=$?
if [ "$rc" -ne 0 ] && [ ! -f "$INST/hypre/.hpcperf-built" ]; then
    ok "C: unwritable install dir -> mark_built rc=$rc, no .hpcperf-built written"
else
    bad "C: write failure not isolated (rc=$rc, marker present: $([ -f "$INST/hypre/.hpcperf-built" ] && echo yes || echo no))"
fi
chmod 700 "$INST/hypre"

# ---- case D: recorded fingerprint differs from current env -> is_built FAILS FAST
sed -i 's/^profile=.*arch=sm_100/profile=test backend=cuda arch=sm_90/' "$INST/kokkos/.hpcperf-fingerprint"
( set -e; is_built kokkos ) >/dev/null 2>"$T/d.err"; rc=$?
if [ "$rc" -ne 0 ] && grep -q 'fingerprint mismatch' "$T/d.err" && grep -q 'refusing to reuse' "$T/d.err"; then
    ok "D: fingerprint mismatch (arch) -> is_built fails fast with explanation"
else
    bad "D: mismatch not detected (rc=$rc)"; cat "$T/d.err"
fi

# ---- case E: legacy install (marker, no fingerprint) -> accepted with a warning
echo "5.2.1" > "$INST/legacy/.hpcperf-built"
PINS+=("legacy|https://example.invalid/legacy.git|5.2.1")
( set -e; is_built legacy ) >/dev/null 2>"$T/e.err"; rc=$?
if [ "$rc" -eq 0 ] && grep -q 'no .hpcperf-fingerprint' "$T/e.err"; then
    ok "E: legacy install without fingerprint -> accepted, warning printed"
else
    bad "E: legacy handling (rc=$rc)"; cat "$T/e.err"
fi

# ---- case F: stamp_existing writes a post-hoc labelled fingerprint the check accepts
( set -e; stamp_existing ) >/dev/null 2>&1
if grep -q '^stamped=post-hoc' "$INST/legacy/.hpcperf-fingerprint" 2>/dev/null \
   && ( set -e; is_built legacy ) >/dev/null 2>&1; then
    ok "F: --stamp-existing labels the fingerprint post-hoc and is_built accepts it"
else
    bad "F: stamp_existing"; cat "$INST/legacy/.hpcperf-fingerprint" 2>/dev/null
fi

echo "test_deps_markers: $pass passed, $failn failed"
[ "$failn" -eq 0 ]
