#!/usr/bin/env bash
# seed_scatter.sh -- the Monte Carlo scatter of a registered Quicksilver input's tallies: runs the
# reference build on the SAME deck with OTHER random seeds (the seed is a deck parameter), one run
# per seed, into <out>/seed-<seed>/stdout.log. seed_stats.py then reports mean and standard
# deviation of the final-cycle scalar flux (and census / segments) over the seeds and the rule it
# implies. These runs are reference-distribution runs, not measurements of the registered input
# (their workload identity differs in the seed).
#
#   seed_scatter.sh <input_id> <out dir> <seed> [<seed> ...]
#
# The deck copy has only the `seed:` line replaced; every other deck value and the run.sh size
# knobs of the input (p1-profile: cells / particles / steps from inputs.yaml params) are the same.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
TOOL="$R/tools/inputs/hpcperf_inputs.py"
ID="${1:?usage: seed_scatter.sh <input_id> <out dir> <seed>...}"; OUT="${2:?}"; shift 2
[ $# -gt 0 ] || { echo "seed_scatter.sh: give at least one seed" >&2; exit 2; }
DECK_REL="$(python3 "$TOOL" param "$HERE" "$ID" deck)" || exit 2
VERBATIM="$(python3 "$TOOL" param "$HERE" "$ID" verbatim)" || exit 2
DECK="$HERE/$DECK_REL"
[ -f "$DECK" ] || { echo "seed_scatter.sh: deck $DECK missing" >&2; exit 2; }
grep -qE '^\s*seed:' "$DECK" || { echo "seed_scatter.sh: $DECK has no seed line" >&2; exit 2; }
mkdir -p "$OUT"
for seed in "$@"; do
    d="$OUT/seed-$seed"; mkdir -p "$d"
    sed -E "s/^(\s*seed:\s*)[0-9]+/\1$seed/" "$DECK" > "$d/deck.inp"
    diff <(grep -vE '^\s*seed:' "$DECK") <(grep -vE '^\s*seed:' "$d/deck.inp") > /dev/null || { echo "seed_scatter.sh: deck copy differs beyond the seed line" >&2; exit 2; }
    echo "== seed $seed: $ID with deck $d/deck.inp"
    if [ "$VERBATIM" = "True" ] || [ "$VERBATIM" = "true" ]; then
        ( cd "$d" && HPCPERF_QUICKSILVER_INPUT="$d/deck.inp" bash "$HERE/run.sh" CUDA > stdout.log 2>&1 ); rc=$?
    else
        C="$(python3 "$TOOL" param "$HERE" "$ID" cells_per_rank)"; P="$(python3 "$TOOL" param "$HERE" "$ID" particles_per_rank)"; S="$(python3 "$TOOL" param "$HERE" "$ID" steps)"
        ( cd "$d" && HPCPERF_QUICKSILVER_INPUT="$d/deck.inp" HPCPERF_QUICKSILVER_CELLS_PER_RANK="$C" HPCPERF_QUICKSILVER_PARTICLES_PER_RANK="$P" HPCPERF_QUICKSILVER_STEPS="$S" bash "$HERE/run.sh" CUDA > stdout.log 2>&1 ); rc=$?
    fi
    echo "rc=$rc seed=$seed $(date -u +%FT%TZ)" > "$d/DONE"
    echo "   rc=$rc"
done
python3 "$HERE/seed_stats.py" "$OUT"
