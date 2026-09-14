#!/usr/bin/env python3
"""compare_source_trees -- content comparison of a frozen source tree against the tree that was actually
validated (the checkout or the patched private copy the recorded results were built from).

    compare_source_trees.py <frozen-tree> <validated-tree> [--ignore GLOB ...] [--generated GLOB ...]
                            [--excluded GLOB ...] [--patched PATH ...] [--added PATH ...] [--json OUT] [--md OUT]

Every regular file / symlink of both trees is classified (fnmatch globs; '*' also matches '/'):
    identical                        same path, same content
    expected_patch_difference        differs, path touched by the declared patch series (frozen = patched baseline)
    expected_generated_difference    differs, path matches --generated (build-time edit of a generated file)
    expected_normalization_excluded  only in validated tree, path matches --excluded (declared exclusion: docs,
                                     unused submodule, ...)
    expected_build_artifact          only in validated tree, path matches --ignore (.git, objects, logs, markers)
    expected_added                   only in frozen tree, path matches --added (files the freeze adds, e.g. LICENSES)
    missing_source                   only in validated tree, not explained            -> UNEXPECTED
    extra_source                     only in frozen tree, not explained               -> UNEXPECTED
    unexpected_content_difference    differs and not explained                        -> UNEXPECTED
Exit 0 when no UNEXPECTED class has members, 1 otherwise. Never modifies either tree.
"""
import argparse
import fnmatch
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hpcperf_source import manifest  # noqa: E402

UNEXPECTED = ("missing_source", "extra_source", "unexpected_content_difference")


def match_any(path, globs):
    return any(fnmatch.fnmatchcase(path, g) or fnmatch.fnmatchcase(path, g.rstrip("/") + "/*") for g in globs)


def compare(frozen, validated, ignore=(), generated=(), excluded=(), patched=(), added=()):
    fz = {e["path"]: e for e in manifest(frozen)}
    # --ignore describes build artifacts that only exist in the validated tree; a path present in BOTH trees
    # is always compared by content, whatever it matches (a tracked file the build modified in place must
    # be declared with --generated, never hidden by an ignore glob)
    vd = {e["path"]: e for e in manifest(validated) if e["path"] in fz or not match_any(e["path"], ignore)}
    classes = {k: [] for k in ("identical", "expected_patch_difference", "expected_generated_difference",
                               "expected_normalization_excluded", "expected_build_artifact", "expected_added",
                               "missing_source", "extra_source", "unexpected_content_difference")}
    patched = set(patched)
    for p in sorted(set(fz) | set(vd), key=lambda s: s.encode()):
        a, b = fz.get(p), vd.get(p)
        if a and b:
            if a["type"] == b["type"] and a["sha256"] == b["sha256"]:
                classes["identical"].append(p)
            elif p in patched:
                classes["expected_patch_difference"].append(p)
            elif match_any(p, generated):
                classes["expected_generated_difference"].append(p)
            else:
                classes["unexpected_content_difference"].append(p)
        elif b:
            if match_any(p, excluded):
                classes["expected_normalization_excluded"].append(p)
            elif match_any(p, ignore) or match_any(p, generated):
                classes["expected_build_artifact"].append(p)
            else:
                classes["missing_source"].append(p)
        else:
            if p in set(added) or match_any(p, added):
                classes["expected_added"].append(p)
            else:
                classes["extra_source"].append(p)
    summary = {k: len(v) for k, v in classes.items()}
    summary["unexpected_total"] = sum(summary[k] for k in UNEXPECTED)
    summary["frozen_entries"] = len(fz)
    summary["validated_entries_considered"] = len(vd)
    return {"frozen": portable_path(frozen), "validated": portable_path(validated), "summary": summary, "classes": classes}


def portable_path(p):
    """Record trees relative to the repository root or the freeze scratch (provenance files are committed;
    they must not carry this node's absolute directories)."""
    p = os.path.abspath(p)
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    scratch = os.path.abspath(os.environ.get("HPCPERF_FREEZE_SCRATCH", f"/tmp/hpcperf-freeze-{os.environ.get('USER', 'user')}"))
    if p == root or p.startswith(root + os.sep):
        return os.path.relpath(p, root)
    if p == scratch or p.startswith(scratch + os.sep):
        return "<freeze-scratch>/" + os.path.relpath(p, scratch)
    return p


def to_markdown(res, limit=40):
    s = res["summary"]
    lines = [f"frozen: `{res['frozen']}`", f"validated: `{res['validated']}`", "",
             "| class | count |", "|---|---|"]
    for k, v in s.items():
        if k in ("unexpected_total", "frozen_entries", "validated_entries_considered"):
            continue
        lines.append(f"| {k}{' (UNEXPECTED)' if k in UNEXPECTED else ''} | {v} |")
    lines.append(f"| **unexpected total** | **{s['unexpected_total']}** |")
    lines.append(f"| frozen entries / validated entries considered | {s['frozen_entries']} / {s['validated_entries_considered']} |")
    for k in ("expected_patch_difference", "expected_generated_difference", "expected_added", "missing_source", "extra_source", "unexpected_content_difference"):
        if res["classes"][k]:
            lines += ["", f"{k} ({len(res['classes'][k])}):"] + [f"- `{p}`" for p in res["classes"][k][:limit]]
            if len(res["classes"][k]) > limit:
                lines.append(f"- ... {len(res['classes'][k]) - limit} more (see the JSON report)")
    for k in ("expected_normalization_excluded", "expected_build_artifact"):
        if res["classes"][k]:
            tops = {}
            for p in res["classes"][k]:
                tops[p.split("/")[0]] = tops.get(p.split("/")[0], 0) + 1
            lines += ["", f"{k} ({len(res['classes'][k])}) by top-level directory: " + ", ".join(f"{t} ({n})" for t, n in sorted(tops.items()))]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frozen"); ap.add_argument("validated")
    ap.add_argument("--ignore", action="append", default=[]); ap.add_argument("--generated", action="append", default=[])
    ap.add_argument("--excluded", action="append", default=[]); ap.add_argument("--patched", action="append", default=[])
    ap.add_argument("--added", action="append", default=[])
    ap.add_argument("--json"); ap.add_argument("--md")
    a = ap.parse_args()
    res = compare(a.frozen, a.validated, a.ignore, a.generated, a.excluded, a.patched, a.added)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    md = to_markdown(res)
    if a.md:
        with open(a.md, "w") as f:
            f.write(md)
    print(md if not a.md else f"unexpected_total={res['summary']['unexpected_total']} identical={res['summary']['identical']}")
    return 0 if res["summary"]["unexpected_total"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
