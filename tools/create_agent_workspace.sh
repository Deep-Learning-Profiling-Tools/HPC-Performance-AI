#!/usr/bin/env bash
# create_agent_workspace.sh -- private, writable copy of a materialized Level 3 benchmark for one
# optimization run. The canonical level3/<app> is never handed to an agent.
#
#   tools/create_agent_workspace.sh level3 <app> <run-id> [--variant NAME] [--dest DIR] [--link-prebuilt-deps]
#
# Output: <dest> (default workspaces/<run-id>/) is a self-contained repository root for the harness scripts:
#   hpcperf_env.sh, check_env.sh, level2/tools/ (launcher), level3/tools/ (helpers)   copies
#   .conda_env, .tools, .deps/install                                                  symlinks to the
#        environment (compilers, MPI, CUDA activation; Level 2 prefixes) -- environment dependencies,
#        not application source
#   level3/<app>/  copy of the canonical benchmark directory: src/, deps/, build.sh, run.sh,
#        validate.sh, benchmark.yaml, optimization_scope.yaml, inputs/, references/, configs/,
#        provenance/, patches/, checkers -- src/deps are REAL copies (cp --reflink=auto; never
#        symlinks back to the canonical tree). The agent's cwd is this directory.
#   build/, .deps/level3/<app>/  created by the workspace's own build.sh (run-id private: no build,
#        install or result is shared between runs/models/iterations)
#   workspace.yaml               run_id, benchmark, source_version, canonical/initial tree hashes, timestamp
#   workspace_baseline.json      sha256 of EVERY file of the benchmark copy (trusted baseline for
#        check_workspace.py --agent-mode; a second copy is kept in <repo>/.hpcperf/workspace_baselines/,
#        outside the workspace)
# --link-prebuilt-deps: for every dependency install prefix listed in benchmark.yaml
#   (`dependency_installs`, relative to `install_root`) that exists and is stage-complete in the
#   canonical tree, a symlink is placed in the workspace so build.sh skips that dependency stage
#   (environment-provided prebuilt dependency; recorded in workspace.yaml). The application itself is
#   always rebuilt inside the workspace.
# Readonly ranges of optimization_scope.yaml and the harness copies are made non-writable (chmod a-w); the
# baseline check (tools/check_workspace.py) must PASS before the workspace is used. These are file-hash and
# permission controls checked by the trusted harness, NOT an operating-system sandbox.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/.." && pwd)"
[ $# -ge 3 ] || { echo "usage: $0 level3 <app> <run-id> [--variant NAME] [--dest DIR] [--link-prebuilt-deps]" >&2; exit 2; }
LEVEL=$1; APP=$2; RUN_ID=$3; shift 3
VARIANT=""; LINK_DEPS=0; DEST=""
while [ $# -gt 0 ]; do case "$1" in --variant) VARIANT=$2; shift 2;; --dest) DEST=$2; shift 2;; --link-prebuilt-deps) LINK_DEPS=1; shift;; *) echo "unknown option $1" >&2; exit 2;; esac; done
[ "$LEVEL" = level3 ] || { echo "create_agent_workspace: only level3 is supported" >&2; exit 2; }
case "$RUN_ID" in ""|*/*|.|..|.*) echo "create_agent_workspace: invalid run id '$RUN_ID'" >&2; exit 2;; esac
CAN="$R/level3/$APP"; WS="${DEST:-$R/workspaces/$RUN_ID}"; WS="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$WS")"; WAPP="$WS/level3/$APP"
[ -d "$CAN" ] || { echo "create_agent_workspace: $CAN missing" >&2; exit 2; }
[ -d "$CAN/src" ] && [ -f "$CAN/.hpcperf-materialized.yaml" ] || { echo "create_agent_workspace: $CAN is not materialized -- run tools/prepare_benchmark.sh level3 $APP first" >&2; exit 3; }
[ ! -e "$WS" ] || { echo "create_agent_workspace: $WS already exists -- one workspace per run id, never reused" >&2; exit 3; }
case "$WS" in "$CAN"|"$CAN"/*) echo "create_agent_workspace: the workspace cannot live inside the canonical benchmark directory" >&2; exit 2;; esac
VARG=(); [ -n "$VARIANT" ] && VARG=(--variant "$VARIANT")
# canonical must still be the frozen baseline
python3 "$HERE/check_workspace.py" "$CAN" "${VARG[@]}" --quick > "$R/.check_canonical.$$.log" 2>&1 || { cat "$R/.check_canonical.$$.log"; rm -f "$R/.check_canonical.$$.log"; echo "create_agent_workspace: canonical benchmark fails check_workspace -- not copying" >&2; exit 3; }
rm -f "$R/.check_canonical.$$.log"
CANON_TREE="$(python3 "$HERE/hpcperf_source.py" "$CAN" --subdir src $( [ -d "$CAN/deps" ] && echo --subdir deps ) | sed -n 's/^source_tree_sha256=\([0-9a-f]*\).*/\1/p')"
VAR_EFF="${VARIANT:-$(sed -n 's/^variant: //p' "$CAN/.hpcperf-materialized.yaml" | head -1 | sed "s/^'\(.*\)'$/\1/; s/^null$//")}"
SRC_VERSION="$(sed -n 's/^source_version: //p' "$CAN/.hpcperf-materialized.yaml" | head -1)"

mkdir -p "$WS/level3" "$WS/level2"
cp -p "$R/hpcperf_env.sh" "$WS/"; [ -f "$R/check_env.sh" ] && cp -p "$R/check_env.sh" "$WS/"
cp -a "$R/level2/tools" "$WS/level2/tools"
cp -a "$R/level3/tools" "$WS/level3/tools"; rm -rf "$WS/level3/tools/__pycache__"
for e in .conda_env .tools; do [ -e "$R/$e" ] && ln -s "$R/$e" "$WS/$e"; done
mkdir -p "$WS/.deps"; [ -d "$R/.deps/install" ] && ln -s "$R/.deps/install" "$WS/.deps/install"
# the benchmark directory: everything except transient entries; src/deps as real copies
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
schema: hpcperf-workspace-1
run_id: $RUN_ID
benchmark: level3/$APP
variant: ${VAR_EFF:-null}
source_version: ${SRC_VERSION:-unknown}
canonical_dir: $CAN
canonical_source_tree_sha256: $CANON_TREE
workspace_initial_tree_sha256: $WS_TREE
creation_timestamp: $STAMP
iteration: 0
prebuilt_dependency_prefixes: [$(IFS=,; echo "${LINKED[*]:-}")]
cwd_for_agent: $WAPP
baseline_file: ../../workspace_baseline.json
EOF
cp "$WAPP/workspace.yaml" "$WS/workspace.yaml"
# trusted baseline: sha256 of every file of the benchmark copy (harness + src + deps), kept OUTSIDE the agent's cwd
HPCPERF_TOOLS_DIR="$HERE" python3 - "$WAPP" "$WS/workspace_baseline.json" "$RUN_ID" "$APP" "${VAR_EFF:-}" "$CANON_TREE" "$STAMP" <<'PY'
import json, os, sys
sys.path.insert(0, os.environ["HPCPERF_TOOLS_DIR"])
import hpcperf_source as hs
W, out, run_id, app, variant, tree, stamp = sys.argv[1:8]
files = {e["path"]: {"sha256": e["sha256"], "type": e["type"]} for e in hs.manifest(W)}
wsr = os.path.abspath(os.path.join(W, "..", ".."))
harness = {}
for top in ("hpcperf_env.sh", "check_env.sh", "level2/tools", "level3/tools"):
    p = os.path.join(wsr, top)
    if os.path.isfile(p):
        harness[top] = {"sha256": hs.sha256_file(p)}
    elif os.path.isdir(p):
        for e in hs.manifest(p):
            if e["type"] == "F":
                harness[os.path.join(top, e["path"])] = {"sha256": e["sha256"]}
json.dump({"schema": "hpcperf-workspace-baseline-2", "run_id": run_id, "benchmark": f"level3/{app}", "variant": variant or None,
           "canonical_source_tree_sha256": tree, "created": stamp, "file_count": len(files), "files": files,
           "harness_file_count": len(harness), "harness_files": harness,
           "note": "file-hash baseline for check_workspace --agent-mode; trusted only from the repository copy or an external --baseline; not an OS sandbox"}, open(out, "w"), indent=0)
print(f"create_agent_workspace: baseline of {len(files)} benchmark files + {len(harness)} harness files -> {out}")
PY
chmod a-w "$WS/workspace_baseline.json"
find "$WS/level2/tools" "$WS/level3/tools" -type f -exec chmod a-w {} + 2>/dev/null || true; chmod a-w "$WS/hpcperf_env.sh" 2>/dev/null || true
mkdir -p "$R/.hpcperf/workspace_baselines" && cp -p "$WS/workspace_baseline.json" "$R/.hpcperf/workspace_baselines/$RUN_ID.json"
echo "create_agent_workspace: trusted baseline copy -> $R/.hpcperf/workspace_baselines/$RUN_ID.json (the copy inside the workspace is informational only)"
echo "create_agent_workspace: $WAPP (tree $WS_TREE, $STAMP)"
python3 "$HERE/check_workspace.py" "$WAPP" "${VARG[@]}" --quick
