#!/bin/bash
# Clean-clone completeness of the vendored Level 2 source copies: files that upstream keeps in a directory
# named build/ are swallowed by the repository's global `build/` ignore rule unless explicitly re-included
# (.gitignore negations). These checks fail in a fresh clone when such a file is missing -- the situation
# in which hipBone and miniWeather could not be built on 2026-09-15.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; R="$(cd "$HERE/../../.." && pwd)"
pass=0; failn=0
ok()  { echo "ok   $*"; pass=$((pass+1)); }
bad() { echo "FAIL $*"; failn=$((failn+1)); }
for f in level2/hipbone/occa/scripts/build/Makefile level2/hipbone/occa/scripts/build/compiledDefinesTemplate.hpp \
         level2/hipbone/occa/scripts/build/compiledDefinesTemplate.hpp.in level2/hipbone/occa/scripts/build/shellTools.sh \
         level2/hipbone/occa/scripts/build/Make.fortran level2/hipbone/occa/scripts/build/Make.fortran_rules \
         level2/miniweather/cpp/build/check_output.sh; do
    [ -f "$R/$f" ] && ok "present: $f" || bad "missing vendored upstream file: $f"
done
# the two directories must be re-included by .gitignore, so `git add` keeps tracking them
for d in level2/hipbone/occa/scripts/build/ level2/miniweather/cpp/build/; do
    /usr/bin/grep -qxF "!$d" "$R/.gitignore" && ok ".gitignore re-includes $d" || bad ".gitignore does not re-include $d"
done
# in a git checkout: no ignored file may exist under a vendored Level 2 tree (build products live under <repo>/build/)
if git -C "$R" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    stray="$(git -C "$R" ls-files --others --ignored --exclude-standard -- level2 2>/dev/null | /usr/bin/grep -v -E '__pycache__|\.pyc$|^level2/tools/mpi_cuda_check/mpi_cuda_check$' || true)"
    [ -z "$stray" ] && ok "no ignored file hides inside a vendored level2/ tree" || bad "ignored files inside level2/ (would be lost in a clean clone): $stray"
fi
echo; echo "test_vendored_completeness: $pass passed, $failn failed"; [ "$failn" -eq 0 ]
