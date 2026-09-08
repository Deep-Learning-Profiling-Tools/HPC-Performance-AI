#!/usr/bin/env bash
# create_agent_workspace.sh -- private, writable copy of a materialized Level 3 benchmark for one
# optimization run. The canonical level3/<app> is never handed to an agent.
#
#   tools/create_agent_workspace.sh level3 <app> <run-id> [--variant NAME] [--link-prebuilt-deps]
#
# Output: workspaces/<run-id>/ is a self-contained repository root for the harness scripts:
#   hpcperf_env.sh, check_env.sh, level2/tools/ (launcher), level3/tools/ (helpers)   copies
#   .conda_env, .tools, .deps/install                                                  symlinks to the
#        environment (compilers, MPI, CUDA activation; Level 2 prefixes) -- environment dependencies,
#        not application source
#   level3/<app>/  copy of the canonical benchmark directory: src/, deps/, build.sh, run.sh,
#        validate.sh, benchmark.yaml, optimization_scope.yaml, inputs/, references/, configs/,
#        provenance/, patches/, checkers -- archives/ are omitted (not needed), src/deps are REAL
#        copies (cp --reflink=auto; never symlinks back to the canonical tree)
#   build/, .deps/level3/<app>/  created by the workspace's own build.sh (run-id private)
# --link-prebuilt-deps: for every dependency install prefix listed in benchmark.yaml
#   (`dependency_installs`, relative to `install_root`) that exists and is stage-complete in the
#   canonical tree, a symlink is placed in the workspace so build.sh skips that dependency stage
#   (environment-provided prebuilt dependency; recorded in workspace.yaml). The application itself is
#   always rebuilt inside the workspace.
# Readonly ranges of optimization_scope.yaml are made non-writable (chmod a-w); the check
# (tools/check_workspace.py) must PASS before the workspace is used.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/.." && pwd)"
[ $# -ge 3 ] || { echo "usage: $0 level3 <app> <run-id> [--variant NAME] [--link-prebuilt-deps]" >&2; exit 2; }
LEVEL=$1; APP=$2; RUN_ID=$3; shift 3
VARIANT=""; LINK_DEPS=0
while [ $# -gt 0 ]; do case "$1" in --variant) VARIANT=$2; shift 2;; --link-prebuilt-deps) LINK_DEPS=1; shift;; *) echo "unknown option $1" >&2; exit 2;; esac; done
[ "$LEVEL" = level3 ] || { echo "create_agent_workspace: only level3 is supported" >&2; exit 2; }
case "$RUN_ID" in ""|*/*|.|..|.*) echo "create_agent_workspace: invalid run id '$RUN_ID'" >&2; exit 2;; esac
CAN="$R/level3/$APP"; WS="$R/workspaces/$RUN_ID"; WAPP="$WS/level3/$APP"
[ -d "$CAN" ] || { echo "create_agent_workspace: $CAN missing" >&2; exit 2; }
[ -d "$CAN/src" ] && [ -f "$CAN/.hpcperf-materialized.yaml" ] || { echo "create_agent_workspace: $CAN is not materialized -- run tools/prepare_benchmark.sh level3 $APP first" >&2; exit 3; }
[ ! -e "$WS" ] || { echo "create_agent_workspace: $WS already exists -- one workspace per run id, never reused" >&2; exit 3; }
VARG=(); [ -n "$VARIANT" ] && VARG=(--variant "$VARIANT")
# canonical must still be the frozen baseline
python3 "$HERE/check_workspace.py" "$CAN" "${VARG[@]}" --quick > "$R/.check_canonical.$$.log" 2>&1 || { cat "$R/.check_canonical.$$.log"; rm -f "$R/.check_canonical.$$.log"; echo "create_agent_workspace: canonical benchmark fails check_workspace -- not copying" >&2; exit 3; }
rm -f "$R/.check_canonical.$$.log"
CANON_TREE="$(python3 "$HERE/hpcperf_source.py" "$CAN" --subdir src $( [ -d "$CAN/deps" ] && echo --subdir deps ) | sed -n 's/^source_tree_sha256=\([0-9a-f]*\).*/\1/p')"

mkdir -p "$WS/level3" "$WS/level2"
cp -p "$R/hpcperf_env.sh" "$WS/"; [ -f "$R/check_env.sh" ] && cp -p "$R/check_env.sh" "$WS/"
cp -a "$R/level2/tools" "$WS/level2/tools"
cp -a "$R/level3/tools" "$WS/level3/tools"; rm -rf "$WS/level3/tools/__pycache__"
for e in .conda_env .tools; do [ -e "$R/$e" ] && ln -s "$R/$e" "$WS/$e"; done
mkdir -p "$WS/.deps"; [ -d "$R/.deps/install" ] && ln -s "$R/.deps/install" "$WS/.deps/install"
# the benchmark directory: everything except the archives and the canonical marker; src/deps as real copies
mkdir -p "$WAPP"
( cd "$CAN" && for e in * .[!.]*; do
    case "$e" in '*'|'.[!.]*'|archives|__pycache__|.materialize.tmp.*|.discarded-*) continue;; esac   # the marker (.hpcperf-materialized.yaml: variant + identity) is copied
    cp -a --reflink=auto "$e" "$WAPP/$e" 2>/dev/null || cp -a "$e" "$WAPP/$e"
  done )
find "$WAPP" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
# prebuilt dependency prefixes (environment-provided), optional
LINKED=()
if [ "$LINK_DEPS" -eq 1 ]; then
    ROOT_REL="$(python3 - "$CAN/benchmark.yaml" <<'PY'
import sys, yaml
by = yaml.safe_load(open(sys.argv[1])) or {}
print(by.get("install_root", ""))
for d in by.get("dependency_installs", []) or []: print("DEP " + d)
PY
)"
    IROOT="$(echo "$ROOT_REL" | head -1)"
    for d in $(echo "$ROOT_REL" | sed -n 's/^DEP //p'); do
        src="$R/$IROOT/$d"; dst="$WS/$IROOT/$d"
        if [ -d "$src" ] && [ -f "$src/.hpcperf-stage-done" ]; then mkdir -p "$(dirname "$dst")"; ln -s "$src" "$dst"; LINKED+=("$IROOT/$d"); fi
    done
fi
WS_TREE="$(python3 "$HERE/hpcperf_source.py" "$WAPP" --subdir src $( [ -d "$WAPP/deps" ] && echo --subdir deps ) | sed -n 's/^source_tree_sha256=\([0-9a-f]*\).*/\1/p')"
[ "$WS_TREE" = "$CANON_TREE" ] || { echo "create_agent_workspace: copied tree hash $WS_TREE != canonical $CANON_TREE" >&2; exit 4; }
# readonly ranges from the optimization scope (best effort on this filesystem; the check enforces the rest)
python3 - "$WAPP" <<'PY'
import glob, os, sys, yaml
W = sys.argv[1]; sc = yaml.safe_load(open(os.path.join(W, "optimization_scope.yaml"))) or {}
n = 0
for pat in (sc.get("readonly", []) or []) + (sc.get("excluded", []) or []):
    for p in glob.glob(os.path.join(W, pat), recursive=True):
        try:
            os.chmod(p, os.stat(p).st_mode & ~0o222); n += 1
        except OSError:
            pass
print(f"create_agent_workspace: {n} readonly entries protected (chmod a-w)")
PY
STAMP="$(date -u +%FT%TZ)"
cat > "$WAPP/workspace.yaml" <<EOF
run_id: $RUN_ID
benchmark: level3/$APP
variant: ${VARIANT:-$(sed -n 's/^variant: //p' "$CAN/.hpcperf-materialized.yaml" | head -1)}
canonical_dir: $CAN
canonical_source_tree_sha256: $CANON_TREE
workspace_initial_tree_sha256: $WS_TREE
created: $STAMP
prebuilt_dependency_prefixes: [$(IFS=,; echo "${LINKED[*]:-}")]
cwd_for_agent: $WAPP
EOF
cp "$WAPP/workspace.yaml" "$WS/workspace.yaml"
echo "create_agent_workspace: $WAPP (tree $WS_TREE, $STAMP)"
python3 "$HERE/check_workspace.py" "$WAPP" "${VARG[@]}" --quick
