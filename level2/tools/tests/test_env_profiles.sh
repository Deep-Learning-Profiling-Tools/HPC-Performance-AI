#!/bin/bash
# hpcperf_env.sh dependency-profile switching: the prefixes/lib dirs the script
# added for one profile must be removed when another profile is sourced, user
# entries must survive, re-sourcing must not duplicate, and a profile without
# installs must expose nothing. Runs in this shell (state carries between
# sources), no GPU, no build. (No `set -u`: conda's activation scripts, which
# hpcperf_env.sh sources, reference unset variables.)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../../.." && pwd)"
pass=0; failn=0
ok()  { echo "ok   $*"; pass=$((pass+1)); }
bad() { echo "FAIL $*"; failn=$((failn+1)); }
count_entries() { # $1 = var value, $2 = substring -> number of ':'-entries containing it
    tr ':' '\n' <<<"$1" | grep -c -- "$2" || true
}
dups() { tr ':' '\n' <<<"$1" | grep -v '^$' | sort | uniq -d | wc -l; }

L2="$R/.deps/install/"; L3="$R/.deps/level3/install/"
[ -d "$R/.deps/install" ] || { echo "test_env_profiles: SKIP (no .deps/install)"; exit 0; }
[ -d "$R/.deps/level3/install" ] && echo "note: a level3 install root exists; 'not installed' case uses profile 'zz-not-installed'"

# Start from a clean slate: user-owned entries only. (Do not inherit the
# caller's paths -- if the caller had already sourced hpcperf_env.sh, its
# dependency entries would arrive here WITHOUT the sentinel that records them
# as ours, and the test would then measure the harness, not the script.)
export CMAKE_PREFIX_PATH="/opt/userlib"
export LD_LIBRARY_PATH="/opt/userlib/lib"
unset HPCPERF_DEPS_PROFILE _HPCPERF_DEPS_ADDED_PREFIXES _HPCPERF_DEPS_ADDED_LIBS

# 1. level2 (default): level2 prefixes present, user entry present
source "$R/hpcperf_env.sh" 2>/dev/null
n2="$(count_entries "$CMAKE_PREFIX_PATH" "$L2")"
[ "$n2" -ge 1 ] && grep -q '/opt/userlib' <<<"$CMAKE_PREFIX_PATH" && ok "1: level2 exposes $n2 prefixes, user entry kept" || bad "1: CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH"
[ "$HPCPERF_DEPS_PROFILE_ACTIVE" = level2 ] && ok "1b: active profile recorded as level2" || bad "1b: active=$HPCPERF_DEPS_PROFILE_ACTIVE"
nlib2="$(count_entries "$LD_LIBRARY_PATH" "$L2")"
[ "$nlib2" -ge 1 ] && ok "1c: level2 lib dirs on LD_LIBRARY_PATH ($nlib2)" || bad "1c: LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# 2. switch to a profile that is not installed: level2 entries gone, user entries kept, nothing added
HPCPERF_DEPS_PROFILE=zz-not-installed source "$R/hpcperf_env.sh" 2>"$HERE/.env_test.err"
if [ "$(count_entries "$CMAKE_PREFIX_PATH" "$L2")" -eq 0 ] && [ "$(count_entries "$LD_LIBRARY_PATH" "$L2")" -eq 0 ] \
   && grep -q '/opt/userlib' <<<"$CMAKE_PREFIX_PATH" && grep -q '/opt/userlib/lib' <<<"$LD_LIBRARY_PATH"; then
    ok "2: switching to an uninstalled profile removed all level2 dep paths and kept user entries"
else bad "2: CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH LD=$LD_LIBRARY_PATH"; fi
grep -q 'has no installs yet' "$HERE/.env_test.err" && ok "2b: uninstalled profile is reported" || bad "2b: no report: $(cat "$HERE/.env_test.err")"
[ -z "${_HPCPERF_DEPS_ADDED_PREFIXES:-}" ] && ok "2c: nothing recorded as added" || bad "2c: added=$_HPCPERF_DEPS_ADDED_PREFIXES"
rm -f "$HERE/.env_test.err"

# 3. back to level2: entries restored exactly once
HPCPERF_DEPS_PROFILE=level2 source "$R/hpcperf_env.sh" 2>/dev/null
[ "$(count_entries "$CMAKE_PREFIX_PATH" "$L2")" -eq "$n2" ] && ok "3: level2 -> other -> level2 restores the $n2 prefixes" || bad "3: $(count_entries "$CMAKE_PREFIX_PATH" "$L2") vs $n2"
[ "$(dups "$CMAKE_PREFIX_PATH")" -eq 0 ] && [ "$(dups "$LD_LIBRARY_PATH")" -eq 0 ] && ok "3b: no duplicate entries" || bad "3b: duplicates present"

# 4. repeated source of the same profile: identical result, no duplicates
before="$CMAKE_PREFIX_PATH"; beforel="$LD_LIBRARY_PATH"
source "$R/hpcperf_env.sh" 2>/dev/null; source "$R/hpcperf_env.sh" 2>/dev/null
[ "$CMAKE_PREFIX_PATH" = "$before" ] && [ "$LD_LIBRARY_PATH" = "$beforel" ] && ok "4: re-sourcing twice leaves CMAKE_PREFIX_PATH / LD_LIBRARY_PATH unchanged" || bad "4: changed on re-source"

# 5. level3 explicitly (installed or not): no level2 paths remain
HPCPERF_DEPS_PROFILE=level3 source "$R/hpcperf_env.sh" 2>/dev/null
n3="$(count_entries "$CMAKE_PREFIX_PATH" "$L3")"
if [ "$(count_entries "$CMAKE_PREFIX_PATH" "$L2")" -eq 0 ] && [ "$HPCPERF_DEPS_PROFILE_ACTIVE" = level3 ]; then
    ok "5: level3 profile active, level2 prefixes removed ($n3 level3 prefixes exposed)"
else bad "5: level2 leftovers: $CMAKE_PREFIX_PATH"; fi
HPCPERF_DEPS_PROFILE=level2 source "$R/hpcperf_env.sh" 2>/dev/null
[ "$(count_entries "$CMAKE_PREFIX_PATH" "$L3")" -eq 0 ] && [ "$(count_entries "$CMAKE_PREFIX_PATH" "$L2")" -eq "$n2" ] \
    && ok "5b: level3 -> level2 removes level3 entries and restores level2" || bad "5b: $CMAKE_PREFIX_PATH"

echo "test_env_profiles: $pass passed, $failn failed"
[ "$failn" -eq 0 ]
